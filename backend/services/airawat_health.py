"""AIRAWAT connectivity and model validation.

AIRAWAT is the platform's only AI provider (LLM and speech-to-text). This module answers, without ever
returning or logging the API key: is it configured, does the endpoint answer, are the configured model /
request format / response format accepted, and how fast?

Used by ``python -m backend.scripts.check_airawat`` (deployment check), ``GET /api/v1/system/health/airawat``
(cached) and a background line in the startup log.

What the checks establish about the gateway (verified against the live service):
  * LLM:     POST {base}/chat/completions, OpenAI-compatible; an unknown model answers 404, a bad key 401.
  * Whisper: POST {whisper_url} multipart ``file`` + ``language``. The language code is MANDATORY (there is no
             auto-detection: "400 need model or language"), a ``model`` field is REJECTED ("bad model"), and the
             reply is ``{"text", "language_code"}`` only (no timestamps, no detected language).
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import struct
import time
import wave
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

from backend.config.config import settings

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 20

# HTTP statuses worth another attempt: the gateway (or an upstream) was busy, not wrong. Everything else (401 bad
# key, 404 unknown model, 400 bad request) fails identically every time and is reported at once.
_TRANSIENT_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})


def diagnostic_attempts() -> int:
    """Attempts per check: ``1 + AIRAWAT_RETRY_COUNT`` (capped at 4).

    The NIC gateway intermittently accepts a request and then never answers it (measured: ~1 call in 4 hangs for the
    whole timeout while the rest answer in under a second). A single-shot probe therefore reported a healthy
    provider as "down" a quarter of the time; retrying the same request is what the platform's real calls do too.
    """
    try:
        retries = int(getattr(settings, "AIRAWAT_RETRY_COUNT", 2) or 0)
    except (TypeError, ValueError):
        retries = 2
    return max(1, min(retries + 1, 4))


def _with_retries(check, cfg: Dict[str, Any], timeout: float, attempts: int, **kwargs) -> Dict[str, Any]:
    """Run ``check`` up to ``attempts`` times, retrying only failures flagged ``transient`` (timeout, network, 5xx)."""
    out: Dict[str, Any] = {}
    for n in range(1, max(1, attempts) + 1):
        out = check(cfg, timeout, **kwargs)
        out["attempts"] = n
        if out.get("ok") or not out.get("transient") or n >= attempts:
            break
        time.sleep(min(2.0 * n, 5.0))
    out.pop("transient", None)
    return out


def key_fingerprint(key: str) -> str:
    """Short, non-reversible identifier of a key (first 8 hex of its SHA-256), for comparing keys across servers."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8] if key else ""


def _redact(text: Any, key: str) -> str:
    out = str(text)
    return out.replace(key, "<key>") if key else out


def _tone_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    """A short valid 16 kHz mono PCM WAV (440 Hz tone) - the same container the pipeline sends."""
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(int(seconds * rate))
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


def _resolve_config() -> Dict[str, Any]:
    from backend.services.model_router import model_router

    cfg = model_router.get_effective_config()
    return {
        "api_key": (cfg.get("api_key") or settings.AIRAWAT_API_KEY or "").strip(),
        "base_url": (cfg.get("base_url") or settings.AIRAWAT_BASE_URL or "").rstrip("/"),
        "llm_model": cfg.get("default_llm_model") or settings.AIRAWAT_LLM_MODEL,
        "llm_endpoint": cfg.get("llama_endpoint") or "/chat/completions",
        "whisper_url": _whisper_url(cfg),
        "language": (cfg.get("language") or "").strip().lower(),
    }


def _whisper_url(cfg: Dict[str, Any]) -> str:
    base_url = cfg.get("base_url") or settings.AIRAWAT_BASE_URL
    endpoint = cfg.get("whisper_endpoint") or ""
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    if endpoint and base_url:
        return base_url.rstrip("/") + "/" + endpoint.lstrip("/")
    return settings.AIRAWAT_WHISPER_URL


def check_llm(cfg: Dict[str, Any], timeout: float = _DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """One tiny chat completion: proves endpoint, authentication, model name and the response format."""
    key = cfg["api_key"]
    endpoint = cfg["llm_endpoint"]
    url = endpoint if endpoint.startswith(("http://", "https://")) else cfg["base_url"] + "/" + endpoint.lstrip("/")
    out: Dict[str, Any] = {"ok": False, "endpoint": urlparse(url).netloc + urlparse(url).path, "model": cfg["llm_model"]}
    t0 = time.monotonic()
    try:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": cfg["llm_model"], "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
                  "max_tokens": 8, "temperature": 0},
            verify=settings.SSL_VERIFY,
            timeout=timeout,
        )
        out["http_status"] = r.status_code
        out["latency_ms"] = int((time.monotonic() - t0) * 1000)
        if r.status_code == 200:
            body = r.json()
            content = (((body.get("choices") or [{}])[0]).get("message") or {}).get("content")
            out["response_format_ok"] = isinstance(content, str) and bool(content.strip())
            out["model_reported"] = body.get("model")
            out["ok"] = bool(out["response_format_ok"])
            if not out["ok"]:
                out["error"] = "HTTP 200 but the answer has no choices[0].message.content text"
        else:
            out["error"] = _redact(_http_error_text(r), key)
            out["transient"] = r.status_code in _TRANSIENT_HTTP
    except requests.exceptions.SSLError as e:
        out["error"] = f"TLS error (check ca-certificates / SSL_VERIFY): {_redact(e, key)[:200]}"
    except requests.exceptions.Timeout:
        out["error"] = f"no answer within {timeout:.0f}s (timeout)"
        out["transient"] = True
    except requests.exceptions.ConnectionError as e:
        out["error"] = f"connection failed (DNS / firewall / proxy): {_redact(e, key)[:200]}"
        out["transient"] = True
    except Exception as e:  # noqa: BLE001 - diagnostics must never raise
        out["error"] = f"{type(e).__name__}: {_redact(e, key)[:200]}"
    out.setdefault("latency_ms", int((time.monotonic() - t0) * 1000))
    return out


def check_whisper(cfg: Dict[str, Any], timeout: float = _DEFAULT_TIMEOUT, language: str = "en",
                  audio_seconds: float = 1.0) -> Dict[str, Any]:
    """Send a short WAV exactly as the pipeline does (file + language, NO model field) and check the reply shape."""
    key = cfg["api_key"]
    url = cfg["whisper_url"]
    out: Dict[str, Any] = {"ok": False, "endpoint": urlparse(url).netloc + urlparse(url).path, "language_sent": language,
                           "model_field_sent": False, "audio_bytes": 0}
    wav = _tone_wav(audio_seconds)
    out["audio_bytes"] = len(wav)
    t0 = time.monotonic()
    try:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            files={"file": ("healthcheck.wav", wav, "audio/wav")},
            data={"language": language, "language_code": language},
            verify=settings.SSL_VERIFY,
            timeout=timeout,
        )
        out["http_status"] = r.status_code
        out["latency_ms"] = int((time.monotonic() - t0) * 1000)
        if r.status_code == 200:
            body = r.json()
            out["response_format_ok"] = isinstance(body, dict) and "text" in body
            out["ok"] = bool(out["response_format_ok"])
            if not out["ok"]:
                out["error"] = "HTTP 200 but the answer has no 'text' field"
        else:
            out["error"] = _redact(_http_error_text(r), key)
            out["transient"] = r.status_code in _TRANSIENT_HTTP
    except requests.exceptions.SSLError as e:
        out["error"] = f"TLS error (check ca-certificates / SSL_VERIFY): {_redact(e, key)[:200]}"
    except requests.exceptions.Timeout:
        out["error"] = f"no answer within {timeout:.0f}s (timeout)"
        out["transient"] = True
    except requests.exceptions.ConnectionError as e:
        out["error"] = f"connection failed (DNS / firewall / proxy): {_redact(e, key)[:200]}"
        out["transient"] = True
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {_redact(e, key)[:200]}"
    out.setdefault("latency_ms", int((time.monotonic() - t0) * 1000))
    return out


def _http_error_text(resp: requests.Response) -> str:
    try:
        body = resp.json()
        detail = body.get("detail") or (body.get("error") or {}).get("message") or body.get("message") or ""
    except Exception:  # noqa: BLE001
        detail = (resp.text or "")[:160]
    return f"HTTP {resp.status_code}: {detail}".strip()


def run_diagnostics(deep: bool = True, timeout: Optional[float] = None, attempts: Optional[int] = None) -> Dict[str, Any]:
    """Full sanitized AIRAWAT report. ``deep=False`` checks only the LLM endpoint (what health polls need).

    Each endpoint is tried up to ``attempts`` times (default ``1 + AIRAWAT_RETRY_COUNT``) but only while the failure
    is transient (timeout / network / 5xx); ``report[...]["attempts"]`` says how many were needed.
    """
    timeout = timeout or float(getattr(settings, "AIRAWAT_HEALTH_CHECK_TIMEOUT_SEC", _DEFAULT_TIMEOUT) or _DEFAULT_TIMEOUT)
    report: Dict[str, Any] = {
        "provider": "airawat",
        "mode": settings.AI_MODE,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "ssl_verify": bool(settings.SSL_VERIFY),
        "retry_count": int(settings.AIRAWAT_RETRY_COUNT),
        "timeout_sec": int(settings.AIRAWAT_TIMEOUT_SEC),
    }
    try:
        cfg = _resolve_config()
    except Exception as e:  # noqa: BLE001
        report.update(configured=False, ok=False, error=f"configuration could not be resolved: {type(e).__name__}")
        return report

    report["configured"] = bool(cfg["api_key"])
    report["key_fingerprint"] = key_fingerprint(cfg["api_key"])
    report["default_language"] = cfg["language"] or None
    if not cfg["api_key"]:
        report.update(ok=False, error="no API key configured (LOCAL_AIRAWAT_API_KEY / SERVER_AIRAWAT_API_KEY / AIRAWAT_API_KEY)")
        return report

    tries = attempts if attempts is not None else diagnostic_attempts()
    report["llm"] = _with_retries(check_llm, cfg, timeout, tries)
    if deep:
        report["whisper"] = _with_retries(check_whisper, cfg, timeout, tries)
    report["ok"] = bool(report["llm"]["ok"]) and (not deep or bool(report["whisper"]["ok"]))
    if not report["ok"]:
        failing = [name for name in ("llm", "whisper") if name in report and not report[name]["ok"]]
        report["error"] = "; ".join(f"{n}: {report[n].get('error', 'failed')}" for n in failing)
    return report


# Result of the startup deep check in this process (secret-free), reported by /system/health.
LAST_STARTUP_REPORT: Optional[Dict[str, Any]] = None


def startup_status() -> Dict[str, Any]:
    """Compact readiness view of the startup check: configured / LLM reachable / Whisper reachable."""
    rep = LAST_STARTUP_REPORT
    if rep is None:
        return {"state": "checking"}
    return {
        "state": "ok" if rep.get("ok") else "failed",
        "configured": bool(rep.get("configured")),
        "llm_reachable": bool((rep.get("llm") or {}).get("ok")),
        "whisper_reachable": bool((rep.get("whisper") or {}).get("ok")),
        "default_language": rep.get("default_language"),
        "checked_at": rep.get("checked_at"),
        **({"error": rep["error"]} if rep.get("error") else {}),
    }


def log_startup_check() -> None:
    """Background startup line: result of the deep check (LLM + Whisper), secret-free. Never raises."""
    global LAST_STARTUP_REPORT
    try:
        rep = run_diagnostics(deep=True)
        LAST_STARTUP_REPORT = rep
        if rep.get("ok"):
            logger.info(
                "AIRAWAT check OK | configured=yes | llm %s reachable in %dms | whisper reachable in %dms | "
                "default language=%s | key fingerprint %s | mode=%s",
                rep["llm"].get("model"), rep["llm"].get("latency_ms", 0), rep["whisper"].get("latency_ms", 0),
                rep.get("default_language") or "none (uploads must name it)", rep.get("key_fingerprint"), rep.get("mode"),
            )
        else:
            logger.error(
                "AIRAWAT check FAILED | configured=%s | %s | mode=%s. AI calls are retried and then fail with a clear "
                "error; no other engine is substituted. Run `python -m backend.scripts.check_airawat` for details.",
                "yes" if rep.get("configured") else "NO", rep.get("error"), rep.get("mode"),
            )
    except Exception as e:  # noqa: BLE001
        LAST_STARTUP_REPORT = {"ok": False, "configured": False, "error": f"startup check crashed: {type(e).__name__}"}
        logger.error("AIRAWAT startup check crashed: %s", type(e).__name__)

import inspect
import json
import logging
import time
from typing import Any, Dict, Optional, Set
from urllib.parse import urlparse

import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.config.config import settings
from backend.models.models import AIProviderConfig
from backend.services.model_router import model_router
from backend.utils.auth import require_admin
from backend.utils.crypto import decrypt, encrypt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/settings", tags=["settings"])

_DIARIZATION_MODES = (
    "AUTO",
    "NEMO_ONLY",
    "PYANNOTE_ONLY",
    "NEMO_PRIMARY_PYANNOTE_FALLBACK",
    "PYANNOTE_PRIMARY_NEMO_FALLBACK",
)

# Numeric AIRAWAT tunables and their accepted (type, min, max). A bad value stored here
# would break EVERY later LLM call (ModelRouter.get_llm does int()/float() on them), so
# they are validated on write instead of trusted.
_AIRAWAT_NUMERIC_FIELDS = {
    "timeout": (int, 1, 3600),
    "retry_count": (int, 0, 10),
    "max_tokens": (int, 1, 32768),
    "temperature": (float, 0.0, 2.0),
}
_AIRAWAT_URL_FIELDS = ("base_url", "whisper_endpoint", "llama_endpoint")


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def _is_masked_or_empty(key: Any) -> bool:
    """True when the client did not supply a NEW key (blank, or the masked value we sent it)."""
    if key is None:
        return True
    key = str(key).strip()
    return not key or "..." in key or "*" in key


def _to_bool(value: Any) -> bool:
    """Parse a JSON/form boolean. ``bool("false")`` is True, which is never what a client means."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _bounded_int(value: Any, name: str, lo: int, hi: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"'{name}' must be an integer.") from None
    if not lo <= number <= hi:
        raise HTTPException(status_code=400, detail=f"'{name}' must be between {lo} and {hi}.")
    return number


def _endpoint_host(url: Any) -> str:
    """Lower-cased ``host[:port]`` of an absolute http(s) URL, else ''."""
    try:
        parsed = urlparse(str(url or "").strip())
    except ValueError:
        return ""
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return ""
    return parsed.netloc.rsplit("@", 1)[-1].lower()


def _validate_absolute_url(value: Any, name: str) -> None:
    """Reject anything that is not a plain http(s) URL (file://, gopher://, creds-in-URL, …).

    Private hosts are deliberately still allowed here: an on-prem AIRAWAT endpoint
    legitimately lives on an internal network. The key-exfiltration risk is handled by
    ``_trusted_endpoint_hosts`` instead.
    """
    text = str(value or "").strip()
    if not text:
        return
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(status_code=400, detail=f"'{name}' must be an http(s) URL.")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail=f"'{name}' must not embed credentials.")


def _trusted_endpoint_hosts(existing_data: Optional[dict] = None) -> Set[str]:
    """Hosts the stored API key is ALREADY allowed to be sent to (currently configured ones)."""
    candidates = [
        settings.AIRAWAT_BASE_URL,
        settings.AIRAWAT_WHISPER_URL,
    ]
    try:
        cfg = model_router.get_effective_config()
        candidates += [cfg.get("base_url"), cfg.get("whisper_endpoint"), cfg.get("llama_endpoint")]
    except Exception as exc:  # pragma: no cover - never let this break settings
        logger.debug("Could not read effective AI config: %s", exc)
    if existing_data:
        candidates += [existing_data.get(f) for f in _AIRAWAT_URL_FIELDS]
    return {h for h in (_endpoint_host(c) for c in candidates) if h}


def _load_existing_airawat(db: Session) -> tuple:
    """Return (row, decrypted dict) for the stored AIRAWAT config; ({} when unreadable)."""
    existing = (
        db.query(AIProviderConfig).filter(AIProviderConfig.provider == "airawat").first()
    )
    data: dict = {}
    if existing and existing.encrypted_config:
        try:
            parsed = json.loads(decrypt(existing.encrypted_config))
            if isinstance(parsed, dict):
                data = parsed
        except Exception as exc:
            logger.warning("Stored AIRAWAT config could not be read: %s", type(exc).__name__)
    return existing, data


# ─── AI provider settings (administrators only) ─────────────────────────────────────────────────────────────────
# Every route below needs the admin key: the answers name endpoints, model ids and (masked) keys, which ordinary users
# must not see, and a write changes where meeting content is sent.

_PROVIDER_IDS = ("ollama", "groq", "gemini")
# What an administrator may store per provider (anything else is rejected, so a typo is not silently kept).
_PROVIDER_FIELDS = {
    "ollama": {"base_url", "model", "timeout", "retry_count", "temperature", "max_tokens"},
    "groq": {"api_key", "base_url", "model", "whisper_model", "timeout", "retry_count", "temperature", "max_tokens"},
    "gemini": {"api_key", "base_url", "model", "timeout", "retry_count", "temperature", "max_tokens"},
}
_PROVIDER_NUMERIC_FIELDS = {
    "timeout": (float, 1, 3600),
    "retry_count": (int, 0, 10),
    "max_tokens": (int, 1, 128000),
    "temperature": (float, 0.0, 2.0),
}


def _validated_number(name: str, value: Any, kind, lo, hi):
    try:
        number = kind(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"'{name}' must be a number.") from None
    if not lo <= number <= hi:
        raise HTTPException(status_code=400, detail=f"'{name}' must be between {lo} and {hi}.")
    return number


def _load_provider_row(db: Session, provider: str) -> tuple:
    row = db.query(AIProviderConfig).filter(AIProviderConfig.provider == provider).first()
    data: dict = {}
    if row and row.encrypted_config:
        try:
            parsed = json.loads(decrypt(row.encrypted_config))
            data = parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            logger.warning("Stored %s settings could not be read: %s", provider, type(exc).__name__)
    return row, data


def _airawat_view(db: Session) -> dict:
    """AIRAWAT settings as the platform actually uses them, key masked."""
    _llama_endpoint = "/chat/completions"
    view = {
        "enabled": True,
        "api_key": "",
        "base_url": settings.AIRAWAT_BASE_URL,
        "whisper_endpoint": settings.AIRAWAT_WHISPER_URL,
        "llama_endpoint": settings.AIRAWAT_BASE_URL.rstrip("/") + "/" + _llama_endpoint.lstrip("/"),
        "default_llm_model": settings.AIRAWAT_LLM_MODEL,
        "timeout": settings.AIRAWAT_TIMEOUT_SEC,
        "retry_count": settings.AIRAWAT_RETRY_COUNT,
    }
    row, data = _load_existing_airawat(db)
    if row is not None and data:
        if "api_key" in data:
            data["api_key"] = mask_key(data["api_key"])
        view.update({"enabled": row.enabled, **data,
                     "last_updated": row.updated_at.isoformat() if getattr(row, "updated_at", None) else None})
        return view
    try:
        effective = model_router.get_effective_config()
    except Exception:
        effective = {}
    base_url = effective.get("base_url") or settings.AIRAWAT_BASE_URL
    view["api_key"] = mask_key(effective.get("api_key") or settings.AIRAWAT_API_KEY)
    view["base_url"] = base_url
    view["default_llm_model"] = effective.get("default_llm_model") or settings.AIRAWAT_LLM_MODEL
    if effective.get("whisper_endpoint"):
        view["whisper_endpoint"] = effective["whisper_endpoint"]
    view["llama_endpoint"] = base_url.rstrip("/") + "/" + _llama_endpoint.lstrip("/")
    return view


@router.get("/ai", dependencies=[Depends(require_admin)])
def get_ai_settings(db: Session = Depends(get_db)):
    """Which provider does what, and each provider's settings (keys masked)."""
    from backend.providers import provider_factory

    configs = provider_factory.get_effective_configs()
    result: Dict[str, Any] = {
        "active_provider": configs.get("active_provider"),
        "stt_provider": configs.get("stt_provider") or None,
        "supported_providers": provider_factory.get_supported_providers(),
        "airawat": _airawat_view(db),
    }
    try:
        result["effective_stt_provider"] = provider_factory.stt_provider_name()
    except Exception as exc:  # noqa: BLE001 - shown to the administrator as the reason
        result["effective_stt_provider"] = None
        result["stt_problem"] = str(exc)
    for prov_id in _PROVIDER_IDS:
        cfg = dict(configs.get(prov_id) or {})
        if "api_key" in cfg:
            cfg["api_key"] = mask_key(cfg["api_key"])
        result[prov_id] = cfg
    return result


def _save_airawat(db: Session, data: Any) -> None:
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="'airawat' settings must be an object.")
    enabled = _to_bool(data.get("enabled", False))
    existing, existing_data = _load_existing_airawat(db)
    config_to_save = {k: v for k, v in data.items() if k not in ("enabled", "last_updated")}

    for name in _AIRAWAT_URL_FIELDS:
        _validate_absolute_url(config_to_save.get(name), name)
    for name, (kind, lo, hi) in _AIRAWAT_NUMERIC_FIELDS.items():
        if config_to_save.get(name) in (None, ""):
            config_to_save.pop(name, None)
            continue
        config_to_save[name] = _validated_number(name, config_to_save[name], kind, lo, hi)

    new_key = data.get("api_key")
    if _is_masked_or_empty(new_key):
        # The client did not send a fresh key, so the STORED key would be reused. Never let that key follow a change
        # of endpoint host: otherwise anyone able to POST here could point base_url at their own server and harvest
        # the real key on the next LLM call. Changing the host requires typing the key again.
        trusted = _trusted_endpoint_hosts(existing_data)
        for name in _AIRAWAT_URL_FIELDS:
            host = _endpoint_host(config_to_save.get(name))
            if host and host not in trusted:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{name}' points to a new host ({host}). Re-enter the API key to confirm sending "
                           "credentials to it.",
                )
        config_to_save["api_key"] = existing_data.get("api_key", "")
    else:
        config_to_save["api_key"] = str(new_key).strip()

    encrypted_str = encrypt(json.dumps(config_to_save))
    if existing:
        existing.enabled = enabled
        existing.encrypted_config = encrypted_str
    else:
        db.add(AIProviderConfig(provider="airawat", enabled=enabled, encrypted_config=encrypted_str))


def _save_provider(db: Session, prov_id: str, data: Any) -> None:
    from backend.providers import provider_factory

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail=f"'{prov_id}' settings must be an object.")
    unknown = set(data) - _PROVIDER_FIELDS[prov_id] - {"enabled"}
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown {prov_id} setting(s): {', '.join(sorted(unknown))}.")
    row, existing = _load_provider_row(db, prov_id)
    config_to_save = dict(existing)
    for k, v in data.items():
        if k == "enabled":
            continue
        if k in _PROVIDER_NUMERIC_FIELDS:
            if v in (None, ""):
                config_to_save.pop(k, None)
                continue
            kind, lo, hi = _PROVIDER_NUMERIC_FIELDS[k]
            v = _validated_number(k, v, kind, lo, hi)
        elif k == "base_url":
            _validate_absolute_url(v, "base_url")
        elif k in ("model", "whisper_model") and v is not None and not str(v).strip():
            raise HTTPException(status_code=400, detail=f"'{k}' must not be empty.")
        if k != "api_key":
            config_to_save[k] = v

    if "api_key" in _PROVIDER_FIELDS[prov_id]:
        new_key = data.get("api_key")
        if _is_masked_or_empty(new_key):
            # Same rule as AIRAWAT: the stored key never follows a base_url change to a new host.
            new_host = _endpoint_host(data.get("base_url"))
            if new_host:
                current = provider_factory.get_effective_configs().get(prov_id, {}).get("base_url")
                trusted = {h for h in (_endpoint_host(current), _endpoint_host(existing.get("base_url"))) if h}
                if new_host not in trusted:
                    raise HTTPException(
                        status_code=400,
                        detail=f"'base_url' points to a new host ({new_host}). Re-enter the API key to confirm "
                               "sending credentials to it.",
                    )
            if "api_key" in existing:
                config_to_save["api_key"] = existing["api_key"]
        else:
            config_to_save["api_key"] = str(new_key).strip()

    encrypted = encrypt(json.dumps(config_to_save))
    enabled = _to_bool(data.get("enabled", True))
    if row:
        row.enabled = enabled
        row.encrypted_config = encrypted
    else:
        db.add(AIProviderConfig(provider=prov_id, enabled=enabled, encrypted_config=encrypted))


@router.post("/ai", dependencies=[Depends(require_admin)])
def save_ai_settings(payload: Dict[str, Any], db: Session = Depends(get_db)):
    """Save provider choices and per-provider settings; API keys are encrypted at rest. Validated before writing."""
    from backend.providers import provider_factory
    from backend.providers.factory import LLM_PROVIDERS, STT_PROVIDERS

    selection: Dict[str, Any] = {}
    if "active_provider" in payload:
        name = str(payload["active_provider"] or "").strip().lower()
        if name not in LLM_PROVIDERS:
            raise HTTPException(status_code=400, detail=f"'active_provider' must be one of: {', '.join(LLM_PROVIDERS)}.")
        selection["provider"] = name
    if "stt_provider" in payload:
        name = str(payload["stt_provider"] or "").strip().lower()
        if name and name not in STT_PROVIDERS:
            raise HTTPException(status_code=400, detail=f"'stt_provider' must be empty or one of: {', '.join(STT_PROVIDERS)}.")
        selection["stt_provider"] = name

    try:
        if "airawat" in payload:
            _save_airawat(db, payload["airawat"])
        for prov_id in _PROVIDER_IDS:
            if prov_id in payload:
                _save_provider(db, prov_id, payload[prov_id])
        if selection:
            row, current = _load_provider_row(db, "active_provider")
            merged = {**current, **selection}
            encrypted = encrypt(json.dumps(merged))
            if row:
                row.enabled = True
                row.encrypted_config = encrypted
            else:
                db.add(AIProviderConfig(provider="active_provider", enabled=True, encrypted_config=encrypted))
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        logger.error("Failed to save AI settings: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="The AI settings could not be saved.") from exc

    model_router.clear_cache()
    provider_factory.clear_cache()
    return {"status": "success", "message": "AI settings saved."}


def _test_airawat(config: dict, db: Session) -> dict:
    raw_key = config.get("api_key")
    api_key = "" if raw_key is None else str(raw_key).strip()
    using_stored_key = _is_masked_or_empty(api_key)

    existing_data: dict = {}
    if using_stored_key:
        _, existing_data = _load_existing_airawat(db)
        api_key = existing_data.get("api_key") or ""
        if not api_key:
            try:
                api_key = model_router.get_effective_config().get("api_key") or ""
            except Exception:
                api_key = ""
            api_key = api_key or settings.AIRAWAT_API_KEY

    start_time = time.time()
    base_url = config.get("base_url") or settings.AIRAWAT_BASE_URL
    llama_endpoint = config.get("llama_endpoint") or "/chat/completions"
    if llama_endpoint.startswith("http://") or llama_endpoint.startswith("https://"):
        full_url = llama_endpoint
    else:
        full_url = base_url.rstrip("/") + "/" + llama_endpoint.lstrip("/")
    _validate_absolute_url(full_url, "endpoint")

    # SSRF / credential-exfiltration guard: the server-held key may only be sent to a host it is already configured
    # for. A caller wanting to probe another host must supply that host's own key in the request.
    if using_stored_key:
        target_host = _endpoint_host(full_url)
        if target_host not in _trusted_endpoint_hosts(existing_data):
            raise HTTPException(
                status_code=400,
                detail=f"Testing a different host ({target_host}) requires entering its API key explicitly; the "
                       "stored key is only sent to the configured AIRAWAT endpoint.",
            )
    try:
        resp = requests.post(
            full_url,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json={
                "model": config.get("default_llm_model") or settings.AIRAWAT_LLM_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            verify=settings.SSL_VERIFY,
            timeout=8,
            allow_redirects=False,  # a redirect would carry the Authorization header elsewhere
        )
        latency = round((time.time() - start_time) * 1000)
        if 200 <= resp.status_code < 300:
            return {"connected": True, "latency_ms": latency, "message": "Connected successfully to AIRAWAT node.",
                    "details": {"api_version": resp.headers.get("x-api-version", "v1")}}
        return {"connected": False, "message": f"Validation failed with status code {resp.status_code}",
                "details": {"response_body": resp.text[:200]}}
    except Exception as e:
        return {"connected": False, "message": f"Connection timed out or failed: {type(e).__name__}"}


@router.post("/test", dependencies=[Depends(require_admin)])
def test_connection(payload: Dict[str, Any], db: Session = Depends(get_db)):
    """Live connection test of one provider, optionally with unsaved values from the form (``config``).

    ``provider`` defaults to ``airawat`` (the original contract). The configured model is checked against the
    provider's own model list, so a retired or missing model is reported instead of failing later in the pipeline.
    """
    from backend.providers import GeminiProvider, GroqProvider, OllamaProvider, provider_factory

    provider = str(payload.get("provider") or "airawat").strip().lower()
    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise HTTPException(status_code=400, detail="'config' must be an object.")
    if provider == "airawat":
        return _test_airawat(config, db)
    if provider not in _PROVIDER_IDS:
        raise HTTPException(status_code=400, detail="Unsupported provider. Use airawat, ollama, groq or gemini.")

    effective = dict(provider_factory.get_effective_configs().get(provider) or {})
    if config.get("base_url"):
        _validate_absolute_url(config["base_url"], "base_url")
    supplied_key = str(config.get("api_key") or "").strip()
    if provider != "ollama" and _is_masked_or_empty(supplied_key):
        # The stored key may only be sent to the host it is configured for.
        new_host = _endpoint_host(config.get("base_url"))
        if new_host and new_host != _endpoint_host(effective.get("base_url")):
            raise HTTPException(
                status_code=400,
                detail=f"Testing a different host ({new_host}) requires entering its API key explicitly.",
            )
    merged = {**effective, **{k: v for k, v in config.items() if k in _PROVIDER_FIELDS[provider] and v not in (None, "")}}
    if provider != "ollama":
        merged["api_key"] = effective.get("api_key", "") if _is_masked_or_empty(supplied_key) else supplied_key
    cls = {"ollama": OllamaProvider, "groq": GroqProvider, "gemini": GeminiProvider}[provider]
    accepted = inspect.signature(cls.__init__).parameters
    res = cls(**{k: v for k, v in merged.items() if k in accepted}).test_connection()
    return {
        "connected": bool(res.get("ok")),
        "latency_ms": res.get("latency_ms", 0),
        "message": res.get("message"),
        "models": res.get("models", []),
        "model_available": res.get("model_available"),
        "transcription_model_available": res.get("transcription_model_available"),
    }


@router.get("/models", dependencies=[Depends(require_admin)])
def get_provider_models(provider: str = "ollama"):
    """The provider's LIVE model list. When it cannot be read the answer says so (502) - it never invents a list."""
    from backend.providers import provider_factory

    name = (provider or "").strip().lower()
    if name not in _PROVIDER_IDS:
        raise HTTPException(status_code=400, detail="Unsupported provider. Use ollama, groq or gemini.")
    try:
        models = provider_factory.get_provider(name).get_available_models()
    except Exception as exc:  # noqa: BLE001 - reported to the administrator
        logger.warning("Model list for %s could not be read: %s", name, exc)
        raise HTTPException(status_code=502, detail=f"The {name} model list could not be read: {exc}") from exc
    return {"provider": name, "models": models}


@router.get("/report")
def get_report_settings(db: Session = Depends(get_db)):
    """Retrieves dynamic enterprise report settings."""
    defaults = {
        "logo_url": "",
        "theme_color_primary": "#1e3a8a",
        "theme_color_secondary": "#2563eb",
        "font_family": "Arial",
        "header_text": settings.REPORT_HEADER_TEXT,
        "footer_text": settings.REPORT_FOOTER_TEXT,
        "watermark": "CONFIDENTIAL",
        "company_name": settings.REPORT_COMPANY_NAME,
        "company_address": settings.REPORT_COMPANY_ADDRESS,
        "show_cover_page": True,
        "show_table_of_contents": True,
        "show_visual_charts": True,
    }
    try:
        config = (
            db.query(AIProviderConfig)
            .filter(AIProviderConfig.provider == "report_settings")
            .first()
        )
        if config and config.encrypted_config:
            decrypted = decrypt(config.encrypted_config)
            data = json.loads(decrypted)
            # Merge with defaults
            for k, v in defaults.items():
                if k not in data:
                    data[k] = v
            return {"enabled": config.enabled, **data}
    except Exception as e:
        logger.error(f"Error loading report settings: {e}")
    return {"enabled": True, **defaults}


@router.post("/report", dependencies=[Depends(require_admin)])
def save_report_settings(payload: Dict[str, Any], db: Session = Depends(get_db)):
    """Saves dynamic enterprise report settings."""
    try:
        enabled = _to_bool(payload.get("enabled", True))
        config_data = {
            k: v
            for k, v in payload.items()
            if k not in ("enabled", "provider", "updated_at", "last_updated")
        }

        serialized = json.dumps(config_data)
        encrypted = encrypt(serialized)

        config = (
            db.query(AIProviderConfig)
            .filter(AIProviderConfig.provider == "report_settings")
            .first()
        )
        if not config:
            config = AIProviderConfig(provider="report_settings")
            db.add(config)

        config.enabled = enabled
        config.encrypted_config = encrypted
        db.commit()
        return {"status": "success", "message": "Report settings saved successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving report settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to save report settings.") from e


@router.get("/diarization")
def get_diarization_settings():
    """Retrieves current universal diarization strategy settings."""
    from backend.diarization.manager import diarization_manager
    return {
        "mode": diarization_manager.get_configured_mode().value,
        "nemo_enabled": settings.NEMO_ENABLED,
        "pyannote_enabled": settings.PYANNOTE_ENABLED,
        "nemo_device": settings.NEMO_DEVICE,
        "pyannote_model": settings.PYANNOTE_MODEL,
        "supported_modes": list(_DIARIZATION_MODES),
    }


@router.post("/diarization", dependencies=[Depends(require_admin)])
def save_diarization_settings(payload: Dict[str, Any]):
    """Updates universal diarization strategy configuration."""
    if "mode" in payload:
        mode_val = str(payload["mode"]).upper()
        if mode_val not in _DIARIZATION_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported diarization mode. Choose one of: {', '.join(_DIARIZATION_MODES)}.",
            )
        settings.DIARIZATION_MODE = mode_val
    if "nemo_enabled" in payload:
        settings.NEMO_ENABLED = _to_bool(payload["nemo_enabled"])
    if "pyannote_enabled" in payload:
        settings.PYANNOTE_ENABLED = _to_bool(payload["pyannote_enabled"])
    return {
        "status": "success",
        "message": "Diarization strategy settings updated",
        "current_mode": settings.DIARIZATION_MODE,
    }


@router.get("/loop-engine")
def get_loop_engine_settings():
    """Retrieves current Loop Engineering quality control configuration."""
    return {
        "loop_engine_enabled": settings.LOOP_ENGINE_ENABLED,
        "max_stage_retries": settings.MAX_STAGE_RETRIES,
        "quality_validation_enabled": settings.QUALITY_VALIDATION_ENABLED,
    }


@router.post("/loop-engine", dependencies=[Depends(require_admin)])
def save_loop_engine_settings(payload: Dict[str, Any]):
    """Updates Loop Engineering quality control configuration."""
    # Validate everything first so a bad field cannot leave the settings half-applied.
    retries = (
        _bounded_int(payload["max_stage_retries"], "max_stage_retries", 0, 10)
        if "max_stage_retries" in payload
        else None
    )
    if "loop_engine_enabled" in payload:
        settings.LOOP_ENGINE_ENABLED = _to_bool(payload["loop_engine_enabled"])
    if retries is not None:
        settings.MAX_STAGE_RETRIES = retries
    if "quality_validation_enabled" in payload:
        settings.QUALITY_VALIDATION_ENABLED = _to_bool(payload["quality_validation_enabled"])
    return {
        "status": "success",
        "message": "Loop Engine settings updated",
        "loop_engine_enabled": settings.LOOP_ENGINE_ENABLED,
        "max_stage_retries": settings.MAX_STAGE_RETRIES,
    }

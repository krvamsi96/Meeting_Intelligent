"""Model Router - Centralized Enterprise AI Provider.

Credential resolution order
---------------------------
1. environment_selector.get_active_config()  ← server-first, local fallback
2. AIProviderConfig database table (admin-configured override)
3. settings.*  (fallback from .env)

The active environment (SERVER / LOCAL) is determined at startup by a real
health check in environment_selector.py. This module never hard-codes which
environment to use.
"""

import json
import logging
import os
import threading
import time
from typing import Callable

import httpx

from backend.config.config import (
    CLOUD_MAX_LLM_TOKENS,
    CLOUD_MAX_TRANSCRIPT_CHARS,
    settings,
)
from backend.database.connection import SessionLocal
from backend.models.models import AIProviderConfig
from backend.utils.crypto import decrypt
from backend.utils.exceptions import AIProviderNotConfiguredError, AirawatNotConfiguredError

logger = logging.getLogger(__name__)


class _AITimedStream(httpx.SyncByteStream):
    """Response body wrapper that reports once when the body has been fully read, closed or has failed."""

    def __init__(self, inner, on_done):
        self._inner, self._on_done, self._done = inner, on_done, False

    def __iter__(self):
        try:
            for chunk in self._inner:
                yield chunk
        except Exception as exc:
            self._finish(exc)
            raise

    def close(self):
        try:
            self._inner.close()
        finally:
            self._finish(None)

    def _finish(self, exc):
        if not self._done:
            self._done = True
            self._on_done(exc)


class _AITimedTransport(httpx.BaseTransport):
    """Logs every AI LLM HTTP attempt: endpoint, retry number, start/end, first byte, duration, outcome.

    The OpenAI client retries inside one ``invoke``; it numbers each attempt in the ``x-stainless-retry-count``
    header, so a hung attempt and the retry that rescued it appear as separate lines.
    """

    def __init__(self, inner: httpx.BaseTransport):
        self._inner = inner

    @staticmethod
    def _streamed(request: httpx.Request) -> bool:
        try:
            return bool(json.loads(request.content).get("stream"))
        except Exception:  # not a JSON body / body not available: treat as not streamed
            return False

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        endpoint = request.url.path
        retry = request.headers.get("x-stainless-retry-count", "0")
        start_wall = time.strftime("%H:%M:%S")
        t0 = time.monotonic()

        first_byte_limit = float(getattr(settings, "AIRAWAT_LLM_FIRST_BYTE_TIMEOUT_SEC", 0) or 0)
        timeouts = request.extensions.get("timeout")
        body_read_timeout = None
        if first_byte_limit > 0 and isinstance(timeouts, dict) and timeouts.get("read") and self._streamed(request):
            body_read_timeout = timeouts["read"]
            timeouts["read"] = min(first_byte_limit, body_read_timeout)
        try:
            response = self._inner.handle_request(request)
            if body_read_timeout is not None:
                timeouts["read"] = body_read_timeout
        except Exception as exc:
            logger.warning(
                "AI LLM request FAILED | endpoint=%s | retry=%s | start=%s | end=%s | duration=%.2fs | error=%s",
                endpoint, retry, start_wall, time.strftime("%H:%M:%S"), time.monotonic() - t0, type(exc).__name__,
            )
            raise
        first_byte = time.monotonic() - t0

        def _done(exc):
            ok = exc is None and response.status_code < 400
            (logger.info if ok else logger.warning)(
                "AI LLM request %s | endpoint=%s | retry=%s | status=%s | start=%s | end=%s | "
                "first_byte=%.2fs | duration=%.2fs%s",
                "OK" if ok else "FAILED", endpoint, retry, response.status_code, start_wall,
                time.strftime("%H:%M:%S"), first_byte, time.monotonic() - t0,
                f" | error={type(exc).__name__}" if exc else "",
            )

        response.stream = _AITimedStream(response.stream, _done)
        return response

    def close(self) -> None:
        self._inner.close()


# Backward compatibility aliases for test harnesses
_AirawatTimedStream = _AITimedStream
_AirawatTimedTransport = _AITimedTransport


class ModelRouter:
    """Singleton router standardized on universal Enterprise AI providers."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._cached_config = None
        self._cache_fetched_at: float = 0.0  # monotonic seconds
        # One shared httpx.Client per TLS-verify setting. get_llm() used to build a brand-new
        # client on every call and never close it, leaking a connection pool (and sockets)
        # per LLM invocation across long-running pipelines.
        self._http_clients: dict = {}
        self._http_clients_lock = threading.Lock()
        # Last AIRAWAT reachability verdict: (ok, monotonic time). See validate_airawat_config.
        self._validation: tuple = (False, 0.0)
        self._validation_lock = threading.Lock()

    def _get_http_client(self, verify: bool):
        with self._http_clients_lock:
            client = self._http_clients.get(verify)
            if client is None or client.is_closed:
                client = httpx.Client(transport=_AITimedTransport(httpx.HTTPTransport(verify=verify)))
                self._http_clients[verify] = client
            return client

    def _get_async_http_client(self, verify: bool):
        """Shared async client, only needed when TLS verification is OFF.

        ChatOpenAI takes separate sync (``http_client``) and async (``http_async_client``)
        clients. ``astream`` (the /chat/stream endpoint) uses the ASYNC one, which was never
        passed, so it silently fell back to a default client that ALWAYS verifies TLS — i.e.
        SSL_VERIFY=false worked for every call except streaming chat.
        """
        import httpx

        key = ("async", verify)
        with self._http_clients_lock:
            client = self._http_clients.get(key)
            if client is None or client.is_closed:
                client = httpx.AsyncClient(verify=verify)
                self._http_clients[key] = client
            return client

    def get_effective_config(self) -> dict:
        """Merged AIRAWAT config actually in use (env-selector → DB override → settings).

        Contains the real API key: for server-side use only, never return it to a client.
        """
        return dict(self._get_dynamic_config()["airawat"])

    def clear_cache(self):
        """Clears the cached dynamic database config (e.g. after admin settings update)."""
        self._cached_config = None
        self._cache_fetched_at = 0.0
        self._validation = (False, 0.0)
        logger.info("ModelRouter configuration cache cleared.")

    @property
    def llm_provider(self) -> str:
        """Name of the configured text-generation provider (raises when AI_PROVIDER is invalid)."""
        from backend.providers import provider_factory

        return provider_factory.llm_provider_name()

    @property
    def is_cloud(self) -> bool:
        return self.llm_provider != "ollama"

    @property
    def is_local(self) -> bool:
        return not self.is_cloud

    @property
    def provider_name(self) -> str:
        return self.llm_provider.title()

    @property
    def effective_mode(self) -> str:
        return self.llm_provider

    def _get_dynamic_config(self) -> dict:
        """Resolve AI credentials using server-first, local-fallback priority.

        Resolution order:
        1. environment_selector.get_active_config()  ← health-checked at startup
        2. AIProviderConfig database row (admin override via UI)
        3. settings.*  (raw .env fallback)

        Results are cached for AIRAWAT_CONFIG_CACHE_TTL_SEC (default 300 s).
        Call clear_cache() to force a refresh after admin key update.
        """
        import time as _time

        cache_ttl = int(os.getenv("AIRAWAT_CONFIG_CACHE_TTL_SEC", "300"))
        now = _time.monotonic()
        if self._cached_config is not None and (now - self._cache_fetched_at) < cache_ttl:
            return self._cached_config

        # ── Step 1: environment_selector (server-first, real health check) ───────
        env_api_key = ""
        env_base_url = ""
        env_whisper_url = ""
        env_llm_model = ""
        env_wh_model = ""
        try:
            from backend.config.environment_selector import get_active_config
            active = get_active_config()
            env_api_key     = active.airawat_api_key
            env_base_url    = active.airawat_base_url
            env_whisper_url = active.airawat_whisper_url
            env_llm_model   = active.airawat_llm_model
            env_wh_model    = active.airawat_whisper_model
            logger.debug("[ModelRouter] Using %s environment credentials.", active.environment.upper())
        except Exception as sel_err:
            logger.warning("[ModelRouter] EnvironmentSelector unavailable (%s); using settings fallback.", sel_err)
            env_api_key     = settings.AIRAWAT_API_KEY
            env_base_url    = settings.AIRAWAT_BASE_URL
            env_whisper_url = settings.AIRAWAT_WHISPER_URL
            env_llm_model   = settings.AIRAWAT_LLM_MODEL
            env_wh_model    = settings.AIRAWAT_WHISPER_MODEL

        # ── Step 2: Database admin override ────────────────────────────────
        config = {"airawat": {"enabled": True}}
        db = None
        try:
            db = SessionLocal()
            c = db.query(AIProviderConfig).filter(AIProviderConfig.provider == "airawat").first()
            if c and c.encrypted_config:
                try:
                    decrypted = decrypt(c.encrypted_config)
                    if decrypted:
                        provider_data = json.loads(decrypted)
                        config["airawat"] = {"enabled": c.enabled, **provider_data}
                    else:
                        logger.warning(
                            "Could not decrypt database config for 'airawat'; using environment fallback."
                        )
                except Exception as ex:
                    logger.error("Error parsing database config for airawat: %s", ex)
        except Exception as e:
            logger.error("Error querying database config: %s", e)
        finally:
            if db:
                db.close()

        # ── Step 3: Merge — DB override wins over env-selector values ─────────
        if not config["airawat"].get("api_key"):
            config["airawat"] = {
                "enabled": True,
                "api_key": env_api_key,
                "base_url": env_base_url,
                "whisper_endpoint": env_whisper_url,
                "llama_endpoint": "/chat/completions",
                "default_llm_model": env_llm_model or settings.AIRAWAT_LLM_MODEL,
                "default_whisper_model": env_wh_model or settings.AIRAWAT_WHISPER_MODEL,
                "temperature": settings.AIRAWAT_TEMPERATURE,
                "max_tokens": settings.AIRAWAT_MAX_TOKENS,
                "timeout": settings.AIRAWAT_TIMEOUT_SEC,
                "retry_count": settings.AIRAWAT_RETRY_COUNT,
            }
        # Default spoken language for AIRAWAT Whisper: an admin setting (database) wins over the environment.
        if not (config["airawat"].get("language") or "").strip():
            config["airawat"]["language"] = (
                getattr(settings, "AIRAWAT_WHISPER_LANGUAGE", "") or getattr(settings, "DEFAULT_WHISPER_LANGUAGE", "") or ""
            )

        self._cached_config = config
        self._cache_fetched_at = now
        return config

    # A failed verdict is re-checked sooner than a good one: a provider that has just recovered should be usable
    # again within seconds, while a healthy one need not be pinged before every pipeline step.
    _NEGATIVE_VERDICT_TTL_SEC = 30.0

    def validate_airawat_config(self, use_cache: bool = True) -> bool:
        """Whether the AIRAWAT LLM endpoint answers (reachable, key accepted, model known).

        * The verdict is cached (``AIRAWAT_HEALTH_TTL_SEC`` when good, 30 s when bad) so the pipeline steps and health
          polls that ask do not each send a live request. ``use_cache=False`` forces a fresh probe.
        * The NIC gateway sometimes accepts a request and never answers it (~1 call in 4 measured from a workstation).
          A timeout / network error / 5xx is therefore retried (``1 + AIRAWAT_RETRY_COUNT`` attempts, at most 3);
          a 401 / 404 / 400 fails at once because repeating it cannot help.
        * The timeout is AIRAWAT_HEALTH_CHECK_TIMEOUT_SEC (default 30 s: the TLS handshake to the GlobalSign
          intermediate on NIC / government networks adds 1-3 s, and the old hardcoded 10 s gave false negatives).
        """
        import time as _time

        import requests

        cfg = self._get_dynamic_config()["airawat"]
        api_key = cfg.get("api_key") or settings.AIRAWAT_API_KEY
        if not api_key or not api_key.strip():
            return False

        ttl = float(getattr(settings, "AIRAWAT_HEALTH_TTL_SEC", 300) or 300)
        if use_cache:
            ok, at = self._validation
            if at and (_time.monotonic() - at) < (ttl if ok else min(ttl, self._NEGATIVE_VERDICT_TTL_SEC)):
                return ok

        with self._validation_lock:              # one live probe at a time; the others reuse its verdict
            if use_cache:
                ok, at = self._validation
                if at and (_time.monotonic() - at) < (ttl if ok else min(ttl, self._NEGATIVE_VERDICT_TTL_SEC)):
                    return ok
            ok = self._probe_airawat(cfg, api_key, requests)
            self._validation = (ok, _time.monotonic())
            return ok

    def _probe_airawat(self, cfg: dict, api_key: str, requests) -> bool:
        """The live probe behind validate_airawat_config (uncached, with retries). Never logs the key."""
        import time as _time

        base_url = cfg.get("base_url") or settings.AIRAWAT_BASE_URL
        endpoint = cfg.get("llama_endpoint") or "/chat/completions"
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            full_url = endpoint
        else:
            full_url = base_url.rstrip("/") + "/" + endpoint.lstrip("/")
        health_timeout = getattr(settings, "AIRAWAT_HEALTH_CHECK_TIMEOUT_SEC", 30)
        try:
            attempts = max(1, min(int(getattr(settings, "AIRAWAT_RETRY_COUNT", 2) or 0) + 1, 3))
        except (TypeError, ValueError):
            attempts = 3
        transient_http = {408, 425, 429, 500, 502, 503, 504}

        for attempt in range(1, attempts + 1):
            retry = False
            try:
                resp = requests.post(
                    full_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    json={
                        "model": cfg.get("default_llm_model") or settings.AIRAWAT_LLM_MODEL,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1,
                    },
                    verify=settings.SSL_VERIFY,
                    timeout=health_timeout,
                )
                if 200 <= resp.status_code < 300:
                    return True
                retry = resp.status_code in transient_http
                logger.warning("AIRAWAT health-check answered HTTP %s (attempt %d/%d).", resp.status_code, attempt, attempts)
            except requests.exceptions.SSLError as e:
                logger.warning(f"AIRAWAT health-check SSL error (check ca-certificates on server): {type(e).__name__}")
                return False
            except requests.exceptions.ConnectTimeout:
                retry = True
                logger.warning("AIRAWAT health-check TCP connect timed out (attempt %d/%d).", attempt, attempts)
            except requests.exceptions.ReadTimeout:
                retry = True
                logger.warning(
                    "AIRAWAT health-check got no answer within %ss (attempt %d/%d).", health_timeout, attempt, attempts
                )
            except requests.exceptions.ConnectionError as e:
                retry = True
                logger.warning("AIRAWAT health-check connection error (DNS/network): %s (attempt %d/%d).",
                               type(e).__name__, attempt, attempts)
            except Exception as e:
                logger.warning(f"AIRAWAT health-check failed ({type(e).__name__}).")
                return False
            if not retry or attempt >= attempts:
                return False
            _time.sleep(min(2.0 * attempt, 5.0))
        return False

    def get_llm(self, temperature: float = 0.1, max_tokens: int = None):
        """The chat model of the configured AI_PROVIDER - and only that one.

        A provider that is not configured raises ``AIProviderNotConfiguredError``; it is never replaced by another
        provider (that would send meeting content somewhere the deployment did not choose).
        """
        name = self.llm_provider
        if name == "airawat":
            return self._get_airawat_llm(temperature=temperature, max_tokens=max_tokens)
        from backend.providers import provider_factory

        return provider_factory.get_provider(name).get_llm(temperature=temperature, max_tokens=max_tokens)

    def _get_airawat_llm(self, temperature: float = 0.1, max_tokens: int = None):
        """Returns the AIRAWAT Chat Model with configurable timeout and retry parameters.

        ``max_tokens`` overrides the configured default output cap for this
        client instance — used by callers that need a section's output length
        to scale with its own content (e.g. a short meeting's summary should
        not share the same token budget as a 3-hour meeting's).
        """
        from langchain_openai import ChatOpenAI

        cfg = self._get_dynamic_config()["airawat"]
        api_key = cfg.get("api_key") or settings.AIRAWAT_API_KEY
        if not api_key or not api_key.strip():
            raise AirawatNotConfiguredError(
                "AIRAWAT is not configured: no API key (LOCAL_AIRAWAT_API_KEY / SERVER_AIRAWAT_API_KEY / AIRAWAT_API_KEY)."
            )

        base_url = cfg.get("base_url") or settings.AIRAWAT_BASE_URL
        model_name = cfg.get("default_llm_model") or settings.AIRAWAT_LLM_MODEL
        timeout_sec = int(cfg.get("timeout") or settings.AIRAWAT_TIMEOUT_SEC)
        retries = int(cfg.get("retry_count") or settings.AIRAWAT_RETRY_COUNT)
        effective_max_tokens = int(max_tokens or cfg.get("max_tokens") or settings.AIRAWAT_MAX_TOKENS)

        # The NIC gateway accepts ~1 request in 5 and never answers it (measured: no response headers at all), while a
        # healthy streamed answer starts within 0.3-6 s even for a 100k-char prompt. Non-streamed, a hung call can
        # only be told apart from a long generation by the full AIRAWAT_TIMEOUT_SEC (180 s). Streamed, the read
        # timeout is the longest SILENCE: no first byte (hung) or a stalled stream within the idle timeout fails the
        # attempt and the client retries it, while a long answer keeps streaming and is never cut off.
        streaming = bool(getattr(settings, "AIRAWAT_LLM_STREAMING", True))
        if streaming:
            idle = float(settings.AIRAWAT_LLM_STREAM_IDLE_TIMEOUT_SEC)
            connect = float(settings.AIRAWAT_LLM_CONNECT_TIMEOUT_SEC)
            request_timeout = httpx.Timeout(connect=connect, read=idle, write=connect, pool=connect)
            timeout_desc = f"streaming, no-data timeout={idle:.0f}s"
        else:
            request_timeout = timeout_sec
            timeout_desc = f"timeout={timeout_sec}s"

        logger.info(
            f"Initializing AIRAWAT LLM client: model={model_name}, {timeout_desc}, "
            f"retries={retries}, max_tokens={effective_max_tokens}"
        )

        client_kwargs = {}
        if not settings.SSL_VERIFY:
            client_kwargs["http_async_client"] = self._get_async_http_client(False)

        return ChatOpenAI(
            temperature=temperature,
            model=model_name,
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            max_retries=retries,
            timeout=request_timeout,
            streaming=streaming,
            max_tokens=effective_max_tokens,
            top_p=0.1,
            http_client=self._get_http_client(settings.SSL_VERIFY),
            **client_kwargs,
        )

    def get_transcriber(self) -> Callable:
        """Speech-to-text of the configured STT provider. Not configured -> ``AIProviderNotConfiguredError``."""
        from backend.providers import provider_factory
        from backend.services.transcription_service import transcribe_audio_airawat, transcribe_audio_chunked

        name = provider_factory.stt_provider_name()
        if name == "airawat":
            cfg = self._get_dynamic_config()["airawat"]
            if not (cfg.get("api_key") or settings.AIRAWAT_API_KEY or "").strip():
                raise AirawatNotConfiguredError(
                    "AIRAWAT is not configured: no API key (LOCAL_AIRAWAT_API_KEY / SERVER_AIRAWAT_API_KEY / AIRAWAT_API_KEY)."
                )
            logger.info("Using AIRAWAT Whisper API for speech transcription.")
            return transcribe_audio_airawat

        provider = provider_factory.get_provider(name)
        if not provider.is_configured:
            raise AIProviderNotConfiguredError(f"Speech-to-text provider '{name}' is not configured (missing API key).")
        logger.info("Using %s Whisper API for speech transcription.", name)

        def _transcribe(audio_filepath: str, language: str = None, vocabulary_hint: str = None) -> dict:
            return transcribe_audio_chunked(
                audio_filepath, language=language, vocabulary_hint=vocabulary_hint,
                chunk_transcriber=provider.transcribe_chunk, provider_label=name, require_language=False,
            )

        return _transcribe

    def transcribe(self, audio_filepath: str, language: str = None, vocabulary_hint: str = None) -> dict:
        """Transcribe ``audio_filepath`` in its SOURCE language with AIRAWAT Whisper.

        ``language`` is the spoken language's ISO code (the upload's, else AIRAWAT_WHISPER_LANGUAGE). English
        translation is handled by translation_service further down the pipeline — never here — so word
        timestamps stay aligned.

        ``vocabulary_hint`` (optional): dynamically-built per-meeting proper
        noun/term hint forwarded to the ASR decoder — see
        ``transcript_correction_service.build_vocabulary_hint``.
        """
        return self.get_transcriber()(audio_filepath, language=language, vocabulary_hint=vocabulary_hint)

    def get_embedder(self):
        try:
            from backend.services.model_loader import get_embedding_encoder
            enc_type, model = get_embedding_encoder()
            if enc_type == "sentence_transformers":
                return model
            elif enc_type == "transformers":
                return model[1]
            return None
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            return None

    @property
    def max_transcript_chars(self) -> int:
        return CLOUD_MAX_TRANSCRIPT_CHARS

    @property
    def max_llm_tokens(self) -> int:
        return CLOUD_MAX_LLM_TOKENS

    def check_all_health(self) -> dict:
        """Reachability of the configured LLM provider (administrator diagnostics; never shown to end users)."""
        try:
            name = self.llm_provider
        except Exception as exc:  # noqa: BLE001 - an invalid AI_PROVIDER is reported, not raised
            return {"configured_mode": None, "effective_mode": None, "provider_name": None,
                    "provider_configured": False, "provider_reachable": False, "error": str(exc)}
        if name == "airawat":
            cfg = self._get_dynamic_config()["airawat"]
            api_key = cfg.get("api_key") or settings.AIRAWAT_API_KEY
            reachable = self.validate_airawat_config()
            return {"configured_mode": "airawat", "effective_mode": "airawat", "provider_name": "Airawat",
                    "airawat_key_configured": bool(api_key and len(api_key) > 5), "airawat_reachable": reachable,
                    "provider_configured": bool(api_key), "provider_reachable": reachable}
        from backend.providers import provider_factory

        provider = provider_factory.get_provider(name)
        conn = provider.test_connection(timeout=5.0)
        return {"configured_mode": name, "effective_mode": name, "provider_name": name.title(),
                "provider_configured": provider.is_configured, "provider_reachable": bool(conn.get("ok")),
                "message": conn.get("message")}


model_router = ModelRouter()

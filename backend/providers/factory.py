"""Which AI provider does what - resolved in ONE place.

* ``llm_provider_name()``  - the engine for every text-generation call (AI_PROVIDER).
* ``stt_provider_name()``  - the speech-to-text engine (STT_PROVIDER, else AI_PROVIDER when it can transcribe).
* ``get_provider(name)``   - an Ollama / Groq / Gemini client. AIRAWAT is served by ``model_router`` itself.

Precedence for every value: administrator setting stored in the database (encrypted ``ai_provider_configs`` row)
> environment (``settings``). An unknown provider name is a configuration ERROR - it used to fall back to Ollama
silently. Nothing here ever substitutes one provider for another.
"""

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from backend.config.config import settings
from backend.database.connection import SessionLocal
from backend.models.models import AIProviderConfig
from backend.providers.base import AIProvider, ProviderCapabilityError, ProviderConfigurationError
from backend.providers.gemini_provider import GeminiProvider
from backend.providers.groq_provider import GroqProvider
from backend.providers.ollama_provider import OllamaProvider
from backend.utils.crypto import decrypt

logger = logging.getLogger(__name__)

LLM_PROVIDERS = ("airawat", "ollama", "groq", "gemini")
STT_PROVIDERS = ("airawat", "groq")
_CONFIG_TTL_SEC = 60.0   # other worker processes pick up an administrator change within this


def _first(*values: Any) -> Any:
    """The first value that is not None / blank."""
    for v in values:
        if v is not None and not (isinstance(v, str) and not v.strip()):
            return v
    return None


class AIProviderFactory:
    """Process-wide registry of the configured providers."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._provider_cache: Dict[str, AIProvider] = {}
        self._config_cache: Optional[Dict[str, Any]] = None
        self._config_at = 0.0
        self._cache_lock = threading.Lock()

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._provider_cache.clear()
            self._config_cache = None
            self._config_at = 0.0

    @staticmethod
    def get_supported_providers() -> List[Dict[str, Any]]:
        """Capabilities as implemented in THIS application (not the vendors' marketing)."""
        return [
            {"id": "airawat", "type": "gateway", "is_cloud": True, "supports_transcription": True,
             "requires_language_for_transcription": True},
            {"id": "ollama", "type": "local", "is_cloud": False, "supports_transcription": False},
            {"id": "groq", "type": "cloud", "is_cloud": True, "supports_transcription": True,
             "requires_language_for_transcription": False},
            {"id": "gemini", "type": "cloud", "is_cloud": True, "supports_transcription": False},
        ]

    @staticmethod
    def _get_db_config() -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        db = None
        try:
            db = SessionLocal()
            for row in db.query(AIProviderConfig).all():
                if not row.provider:
                    continue
                data: Dict[str, Any] = {}
                if row.encrypted_config:
                    try:
                        parsed = json.loads(decrypt(row.encrypted_config) or "{}")
                        data = parsed if isinstance(parsed, dict) else {}
                    except Exception as exc:
                        logger.warning("Stored AI settings for '%s' could not be read: %s", row.provider, type(exc).__name__)
                result[row.provider] = {"enabled": row.enabled, **data}
        except Exception as exc:
            logger.warning("AI provider settings could not be read from the database (%s); using .env values.",
                           type(exc).__name__)
        finally:
            if db:
                db.close()
        return result

    def get_effective_configs(self) -> Dict[str, Any]:
        """Merged settings of every provider. Contains API keys: server-side use only."""
        with self._cache_lock:
            if self._config_cache is not None and (time.monotonic() - self._config_at) < _CONFIG_TTL_SEC:
                return self._config_cache

        db = self._get_db_config()
        o, g, m = db.get("ollama", {}), db.get("groq", {}), db.get("gemini", {})
        configs = {
            "active_provider": str(_first((db.get("active_provider") or {}).get("provider"),
                                          settings.AI_PROVIDER) or "").strip().lower(),
            "stt_provider": str(_first((db.get("active_provider") or {}).get("stt_provider"),
                                       settings.STT_PROVIDER) or "").strip().lower(),
            "ollama": {
                "base_url": _first(o.get("base_url"), settings.OLLAMA_BASE_URL),
                "model": _first(o.get("model"), settings.OLLAMA_MODEL),
                "timeout": float(_first(o.get("timeout"), settings.OLLAMA_TIMEOUT)),
                "retry_count": int(_first(o.get("retry_count"), settings.OLLAMA_RETRY_COUNT)),
                "temperature": float(_first(o.get("temperature"), 0.1)),
                "max_tokens": int(_first(o.get("max_tokens"), settings.OLLAMA_MAX_TOKENS)),
            },
            "groq": {
                "api_key": _first(g.get("api_key"), settings.GROQ_API_KEY) or "",
                "model": _first(g.get("model"), settings.GROQ_MODEL),
                "whisper_model": _first(g.get("whisper_model"), settings.GROQ_WHISPER_MODEL),
                "base_url": _first(g.get("base_url"), settings.GROQ_BASE_URL),
                "timeout": float(_first(g.get("timeout"), settings.GROQ_TIMEOUT)),
                "retry_count": int(_first(g.get("retry_count"), settings.GROQ_RETRY_COUNT)),
                "temperature": float(_first(g.get("temperature"), 0.1)),
                "max_tokens": int(_first(g.get("max_tokens"), settings.GROQ_MAX_TOKENS)),
            },
            "gemini": {
                "api_key": _first(m.get("api_key"), settings.GEMINI_API_KEY) or "",
                "model": _first(m.get("model"), settings.GEMINI_MODEL),
                "base_url": _first(m.get("base_url"), settings.GEMINI_BASE_URL),
                "timeout": float(_first(m.get("timeout"), settings.GEMINI_TIMEOUT)),
                "retry_count": int(_first(m.get("retry_count"), settings.GEMINI_RETRY_COUNT)),
                "temperature": float(_first(m.get("temperature"), 0.1)),
                "max_tokens": int(_first(m.get("max_tokens"), settings.GEMINI_MAX_TOKENS)),
            },
        }
        with self._cache_lock:
            self._config_cache, self._config_at = configs, time.monotonic()
        return configs

    def llm_provider_name(self) -> str:
        name = self.get_effective_configs()["active_provider"]
        if name not in LLM_PROVIDERS:
            raise ProviderConfigurationError(
                f"AI_PROVIDER='{name}' is not supported. Use one of: {', '.join(LLM_PROVIDERS)}."
            )
        return name

    def stt_provider_name(self) -> str:
        """The speech-to-text provider, or ``ProviderCapabilityError`` when none is configured."""
        cfg = self.get_effective_configs()
        name = cfg["stt_provider"]
        if name:
            if name not in STT_PROVIDERS:
                raise ProviderConfigurationError(
                    f"STT_PROVIDER='{name}' is not supported. Use one of: {', '.join(STT_PROVIDERS)}."
                )
            return name
        llm = self.llm_provider_name()
        if llm in STT_PROVIDERS:
            return llm
        raise ProviderCapabilityError(
            f"Speech-to-text is not configured: AI_PROVIDER='{llm}' does not transcribe audio in this application. "
            f"Set STT_PROVIDER to one of: {', '.join(STT_PROVIDERS)}."
        )

    def get_provider(self, name: Optional[str] = None) -> AIProvider:
        """The Ollama / Groq / Gemini client named (default: the LLM provider). AIRAWAT has no client here."""
        provider_name = (name or self.llm_provider_name()).lower().strip()
        with self._cache_lock:
            cached = self._provider_cache.get(provider_name)
        if cached is not None:
            return cached

        cfg: Dict[str, Any] = self.get_effective_configs().get(provider_name) or {}
        if provider_name == "ollama":
            provider: AIProvider = OllamaProvider(**cfg)
        elif provider_name == "groq":
            provider = GroqProvider(**cfg)
        elif provider_name == "gemini":
            provider = GeminiProvider(**cfg)
        else:
            raise ProviderConfigurationError(
                f"'{provider_name}' has no provider client (supported: ollama, groq, gemini; airawat is built in)."
            )
        with self._cache_lock:
            self._provider_cache[provider_name] = provider
        return provider

    def get_active_provider(self) -> AIProvider:
        """The LLM provider's client (raises for AIRAWAT, which is served by ``model_router``)."""
        return self.get_provider()


provider_factory = AIProviderFactory()

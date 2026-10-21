"""Google Gemini (cloud): text generation only. Meeting text is sent to Google's API.

Speech-to-text is deliberately NOT offered: Gemini returns transcript text without word timings, and the previous
integration invented them (0.4 s per word) and assumed English - which corrupts speaker alignment and every
timestamp shown to users. Use STT_PROVIDER=groq or airawat for transcription.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import requests
from langchain_openai import ChatOpenAI

from backend.providers.base import AIProvider, ProviderConfigurationError, ProviderConnectionError, model_check

logger = logging.getLogger(__name__)


class GeminiProvider(AIProvider):
    """Gemini through Google's OpenAI-compatible endpoint. The key travels in a header, never in a URL."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 90.0,
        retry_count: int = 2,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.api_key = (api_key or "").strip()
        self.default_model = (model or "").strip()
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = float(timeout)
        self.retry_count = int(retry_count)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def is_cloud(self) -> bool:
        return True

    @property
    def supports_transcription(self) -> bool:
        return False

    @property
    def model(self) -> str:
        return self.default_model

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.default_model)

    def _list_models(self, timeout: float) -> List[str]:
        # x-goog-api-key header: a ?key= query string ends up in exception text, proxies and access logs.
        resp = requests.get(f"{self.base_url}/models", headers={"x-goog-api-key": self.api_key}, timeout=timeout)
        if resp.status_code in (400, 401, 403):
            raise ProviderConfigurationError("Gemini rejected the API key (invalid, expired or not enabled).")
        if resp.status_code != 200:
            raise ProviderConnectionError(f"Gemini answered HTTP {resp.status_code} to the model list request.")
        return [m.get("name", "").replace("models/", "") for m in resp.json().get("models", []) if m.get("name")]

    def test_connection(self, timeout: float = 10.0) -> Dict[str, Any]:
        if not self.api_key:
            return {"ok": False, "latency_ms": 0.0, "models": [], "model_available": None,
                    "message": "Gemini API key is missing. Set GEMINI_API_KEY."}
        t0 = time.monotonic()
        try:
            models = self._list_models(timeout)
        except Exception as exc:  # noqa: BLE001 - reported to the administrator, never raised
            return {"ok": False, "latency_ms": round((time.monotonic() - t0) * 1000, 1), "models": [],
                    "model_available": None, "message": f"Gemini connection failed: {type(exc).__name__}: {str(exc)[:160]}"}
        available = model_check(models, self.default_model)
        return {
            "ok": available is not False,
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
            "models": models,
            "model_available": available,
            "message": (f"Connected to Gemini, but the model '{self.default_model}' is not offered (it may have been "
                        f"retired); choose one from the list." if available is False
                        else f"Connected to Gemini; model '{self.default_model}' is available."),
        }

    def get_available_models(self, timeout: float = 10.0) -> List[str]:
        if not self.api_key:
            raise ProviderConfigurationError("Gemini API key is not configured (GEMINI_API_KEY).")
        try:
            return self._list_models(timeout)
        except (ProviderConfigurationError, ProviderConnectionError):
            raise
        except Exception as exc:
            raise ProviderConnectionError(f"Could not read the Gemini model list: {type(exc).__name__}") from exc

    def get_llm(self, temperature: float = 0.1, max_tokens: Optional[int] = None) -> Any:
        if not self.api_key:
            raise ProviderConfigurationError("Gemini API key is not configured (GEMINI_API_KEY).")
        return ChatOpenAI(
            base_url=f"{self.base_url}/openai/",
            api_key=self.api_key,
            model=self.default_model,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.max_tokens,
            timeout=self.timeout,
            max_retries=self.retry_count,
            streaming=True,
        )

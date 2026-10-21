"""Ollama: local text generation through its OpenAI-compatible endpoint (no key, nothing leaves the host)."""

import logging
import time
from typing import Any, Dict, List, Optional

import requests
from langchain_openai import ChatOpenAI

from backend.providers.base import AIProvider, ProviderConnectionError

logger = logging.getLogger(__name__)


def _installed(models: List[str], configured: str) -> Optional[bool]:
    """Ollama lists ``name:tag``; ``llama3.2`` means ``llama3.2:latest``."""
    if not models:
        return None
    wanted = configured if ":" in configured else f"{configured}:latest"
    return configured in models or wanted in models


class OllamaProvider(AIProvider):
    """Local inference via Ollama. Speech-to-text is not offered (Ollama serves no Whisper-style endpoint)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        retry_count: int = 2,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.default_model = (model or "").strip()
        self.timeout = float(timeout)
        self.retry_count = int(retry_count)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def is_cloud(self) -> bool:
        return False

    @property
    def supports_transcription(self) -> bool:
        return False

    @property
    def model(self) -> str:
        return self.default_model

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.default_model)

    def _list_models(self, timeout: float) -> List[str]:
        resp = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            raise ProviderConnectionError(f"Ollama answered HTTP {resp.status_code} to the model list request.")
        return [m.get("name") for m in resp.json().get("models", []) if m.get("name")]

    def test_connection(self, timeout: float = 10.0) -> Dict[str, Any]:
        t0 = time.monotonic()
        try:
            models = self._list_models(timeout)
        except requests.exceptions.ConnectionError:
            return {"ok": False, "latency_ms": round((time.monotonic() - t0) * 1000, 1), "models": [],
                    "model_available": None,
                    "message": f"Could not connect to Ollama at {self.base_url}. Is `ollama serve` running?"}
        except Exception as exc:  # noqa: BLE001 - reported to the administrator, never raised
            return {"ok": False, "latency_ms": round((time.monotonic() - t0) * 1000, 1), "models": [],
                    "model_available": None, "message": f"Ollama connection error: {type(exc).__name__}: {str(exc)[:120]}"}
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        available = _installed(models, self.default_model)
        if available is False:
            return {"ok": False, "latency_ms": latency_ms, "models": models, "model_available": False,
                    "message": f"Ollama is running, but the model '{self.default_model}' is not installed "
                               f"(run: ollama pull {self.default_model})."}
        return {"ok": bool(models), "latency_ms": latency_ms, "models": models, "model_available": available,
                "message": (f"Connected to Ollama; model '{self.default_model}' is installed." if models
                            else "Ollama is running but has no models installed.")}

    def get_available_models(self, timeout: float = 10.0) -> List[str]:
        try:
            return self._list_models(timeout)
        except ProviderConnectionError:
            raise
        except Exception as exc:
            raise ProviderConnectionError(f"Could not read the Ollama model list: {type(exc).__name__}") from exc

    def get_llm(self, temperature: float = 0.1, max_tokens: Optional[int] = None) -> Any:
        return ChatOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="ollama",  # Ollama ignores it; the client requires a non-empty value
            model=self.default_model,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.max_tokens,
            timeout=self.timeout,
            max_retries=self.retry_count,
            streaming=True,
        )

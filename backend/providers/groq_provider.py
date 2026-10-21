"""Groq (cloud): text generation and Whisper speech-to-text. Meeting audio/text is sent to Groq's API."""

import logging
import time
from typing import Any, Dict, List, Optional

import requests
from langchain_openai import ChatOpenAI

from backend.providers.base import AIProvider, ProviderConfigurationError, ProviderConnectionError, model_check

logger = logging.getLogger(__name__)


class GroqProvider(AIProvider):
    """Groq's OpenAI-compatible API. Only the CONFIGURED model is ever called - no hidden fallback models."""

    def __init__(
        self,
        api_key: str,
        model: str,
        whisper_model: str,
        base_url: str,
        timeout: float = 60.0,
        retry_count: int = 2,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.api_key = (api_key or "").strip()
        self.default_model = (model or "").strip()
        self.whisper_model = (whisper_model or "").strip()
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = float(timeout)
        self.retry_count = int(retry_count)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    @property
    def name(self) -> str:
        return "groq"

    @property
    def is_cloud(self) -> bool:
        return True

    @property
    def supports_transcription(self) -> bool:
        return bool(self.whisper_model)

    @property
    def model(self) -> str:
        return self.default_model

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.default_model and self.base_url)

    def _require_key(self) -> None:
        if not self.api_key:
            raise ProviderConfigurationError("Groq API key is not configured (GROQ_API_KEY).")

    def _list_models(self, timeout: float) -> List[str]:
        resp = requests.get(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.api_key}"},
                            timeout=timeout)
        if resp.status_code in (401, 403):
            raise ProviderConfigurationError("Groq rejected the API key (invalid or expired).")
        if resp.status_code != 200:
            raise ProviderConnectionError(f"Groq answered HTTP {resp.status_code} to the model list request.")
        return [m.get("id") for m in resp.json().get("data", []) if m.get("id")]

    def test_connection(self, timeout: float = 10.0) -> Dict[str, Any]:
        if not self.api_key:
            return {"ok": False, "latency_ms": 0.0, "models": [], "model_available": None,
                    "message": "Groq API key is missing. Set GROQ_API_KEY."}
        t0 = time.monotonic()
        try:
            models = self._list_models(timeout)
        except Exception as exc:  # noqa: BLE001 - reported to the administrator, never raised
            return {"ok": False, "latency_ms": round((time.monotonic() - t0) * 1000, 1), "models": [],
                    "model_available": None, "message": f"Groq connection failed: {type(exc).__name__}: {str(exc)[:160]}"}
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        llm_ok = model_check(models, self.default_model)
        stt_ok = model_check(models, self.whisper_model) if self.whisper_model else None
        problems = []
        if llm_ok is False:
            problems.append(f"text model '{self.default_model}' is not offered by Groq")
        if stt_ok is False:
            problems.append(f"speech-to-text model '{self.whisper_model}' is not offered by Groq")
        return {
            "ok": not problems,
            "latency_ms": latency_ms,
            "models": models,
            "model_available": llm_ok,
            "transcription_model_available": stt_ok,
            "message": ("Connected to Groq; " + "; ".join(problems) + ".") if problems
            else f"Connected to Groq; model '{self.default_model}' is available.",
        }

    def get_available_models(self, timeout: float = 10.0) -> List[str]:
        self._require_key()
        try:
            return self._list_models(timeout)
        except (ProviderConfigurationError, ProviderConnectionError):
            raise
        except Exception as exc:
            raise ProviderConnectionError(f"Could not read the Groq model list: {type(exc).__name__}") from exc

    def get_llm(self, temperature: float = 0.1, max_tokens: Optional[int] = None) -> Any:
        self._require_key()
        return ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.default_model,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.max_tokens,
            timeout=self.timeout,
            max_retries=self.retry_count,   # the OpenAI client honours Groq's Retry-After on 429
            streaming=True,
        )

    def transcribe_chunk(
        self,
        chunk_filepath: str,
        chunk_index: int,
        total_chunks: int,
        language: Optional[str] = None,
        vocabulary_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One chunk through Groq Whisper (verbose_json with word timestamps), with the shared bounded retries."""
        self._require_key()
        from backend.services.transcription_service import post_whisper_chunk

        fields: List[tuple] = [
            ("model", self.whisper_model),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment"),
            ("temperature", "0"),
        ]
        if language:
            fields.append(("language", language))   # Groq detects the language itself when none is given
        if vocabulary_hint:
            fields.append(("prompt", vocabulary_hint))
        return post_whisper_chunk(
            chunk_filepath, chunk_index, total_chunks,
            url=f"{self.base_url}/audio/transcriptions",
            api_key=self.api_key,
            fields=fields,
            transport_retries=max(1, self.retry_count),
            provider_label="Groq",
        )

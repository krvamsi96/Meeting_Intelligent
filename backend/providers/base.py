"""Common contract of the text-generation / speech-to-text providers (Ollama, Groq, Gemini).

AIRAWAT, the deployment's original gateway, is served by ``model_router`` directly; the factory only names it.

The exceptions subclass the pipeline's ``AIProviderError`` family so a provider failure is reported (and turned into
a user-facing message) exactly like any other AI failure, instead of escaping as an unknown exception.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from backend.utils.exceptions import AIProviderError, AIProviderNotConfiguredError


class ProviderConfigurationError(AIProviderNotConfiguredError):
    """The provider is missing a key / URL / model, or the configured provider name is unknown."""


class ProviderCapabilityError(AIProviderNotConfiguredError):
    """The configured provider cannot do what was asked (e.g. speech-to-text on a text-only provider)."""


class ProviderConnectionError(AIProviderError):
    """The provider was unreachable, timed out, or rejected the request after the configured retries."""


class AIProvider(ABC):
    """One configured AI provider."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier ('ollama', 'groq', 'gemini')."""

    @property
    @abstractmethod
    def is_cloud(self) -> bool:
        """True when meeting content is sent to a third-party API."""

    @property
    @abstractmethod
    def supports_transcription(self) -> bool:
        """True only when a tested speech-to-text integration exists for this provider."""

    @property
    def is_configured(self) -> bool:
        """Whether the provider has the credentials / URL it needs."""
        return True

    @property
    @abstractmethod
    def model(self) -> str:
        """The configured text-generation model id."""

    @abstractmethod
    def test_connection(self, timeout: float = 10.0) -> Dict[str, Any]:
        """Live check. Returns ``ok``, ``latency_ms``, ``message``, ``models`` (the provider's real list; empty when
        it could not be read) and ``model_available`` (configured model in that list; None when unknown)."""

    @abstractmethod
    def get_available_models(self, timeout: float = 10.0) -> List[str]:
        """The provider's live model list. Raises ``ProviderConnectionError`` when it cannot be read - a fixed
        list would present models that may no longer exist as available."""

    @abstractmethod
    def get_llm(self, temperature: float = 0.1, max_tokens: Optional[int] = None) -> Any:
        """A LangChain chat model bound to the configured model (never to a substitute)."""

    def transcribe_chunk(
        self,
        chunk_filepath: str,
        chunk_index: int,
        total_chunks: int,
        language: Optional[str] = None,
        vocabulary_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe ONE audio chunk (Whisper-style response dict). Chunking, overlap trimming and merging are done
        by ``transcription_service.transcribe_audio_chunked`` for every provider alike."""
        raise ProviderCapabilityError(f"{self.name} does not provide speech-to-text in this application.")


def model_check(models: List[str], configured: str) -> Optional[bool]:
    """Whether ``configured`` is in a live ``models`` list (None when the list is empty / unknown)."""
    if not models:
        return None
    return configured in models

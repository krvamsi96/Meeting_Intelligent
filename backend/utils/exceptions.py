"""Centralized application exception hierarchy for consistent error handling and standard API responses."""

from typing import Any, Dict, Optional
from fastapi import status


class MeetingIntelligenceException(Exception):
    """Base exception for all application errors."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_SERVER_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}


class ResourceNotFoundException(MeetingIntelligenceException):
    """Raised when a requested database entity or resource is not found."""

    def __init__(self, resource: str, identifier: Any):
        super().__init__(
            message=f"{resource} with identifier '{identifier}' was not found.",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="RESOURCE_NOT_FOUND",
            details={"resource": resource, "identifier": identifier},
        )


class ValidationErrorException(MeetingIntelligenceException):
    """Raised when request payload or business validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class AIServiceException(MeetingIntelligenceException):
    """Raised when AI model generation, transcription, or RAG pipeline fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="AI_SERVICE_ERROR",
            details=details,
        )


class AIProviderError(AIServiceException):
    """AI provider (Ollama, Groq, Gemini) is not configured, unreachable, or rejected the request.

    Raised after the configured retries are exhausted.
    """

    error_code_default = "AI_PROVIDER_ERROR"

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details)
        self.error_code = self.error_code_default


class AIProviderNotConfiguredError(AIProviderError):
    error_code_default = "AI_PROVIDER_NOT_CONFIGURED"


# Aliases for backward compatibility
AirawatError = AIProviderError
AirawatNotConfiguredError = AIProviderNotConfiguredError


class LanguageIdentificationError(AIServiceException):
    """The spoken language could not be identified and none is configured.

    Speech-to-text endpoints require an explicit language code when auto-detection is not
    supported or configured default is needed.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details)
        self.error_code = "LANGUAGE_NOT_IDENTIFIED"


class DatabaseException(MeetingIntelligenceException):
    """Raised when database query or transaction operations fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="DATABASE_ERROR",
            details=details,
        )

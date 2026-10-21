"""Dynamic, config-driven file ingestion classification."""

import os

from backend.config.config import (
    ALL_SUPPORTED_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    MEDIA_EXTENSIONS,
)


def normalize_extension(filename: str) -> str:
    """Return the lower-cased extension with leading dot (e.g. .pdf)."""
    return os.path.splitext(filename)[1].lower()


def classify_file_type(filename: str) -> str:
    """Dynamically classify a file as 'document' or 'media' by extension."""
    ext = normalize_extension(filename)
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    return "media"


def is_supported_file(filename: str) -> bool:
    """Return True if the file extension is ingestible."""
    return normalize_extension(filename) in ALL_SUPPORTED_EXTENSIONS


def extension_category(filename: str) -> str:
    """Return 'document', 'media' or 'unsupported' for a filename."""
    ext = normalize_extension(filename)
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    if ext in MEDIA_EXTENSIONS:
        return "media"
    return "unsupported"

"""Security utilities for input validation and sanitization."""

import ipaddress
import logging
import math
import os
import re
import socket
from typing import Optional, Union
from urllib.parse import urlparse

from fastapi import HTTPException

logger = logging.getLogger(__name__)


# The one inline script the printable (PDF) report page needs. It lives here so that the report
# generator and the Content-Security-Policy hash computed from it can never drift apart.
REPORT_PRINT_SCRIPT = "window.onload = function() { window.print(); }"


def content_disposition(filename: str, disposition: str = "attachment") -> str:
    """RFC 6266 ``Content-Disposition`` value that is safe for ANY user-controlled filename.

    Titles are user data (a Hindi title, quotes, semicolons, CR/LF …). Interpolating them raw
    made Starlette fail to encode the header (non-Latin-1 → HTTP 500 on export) and allowed
    header/parameter injection. Emits an ASCII fallback ``filename=`` plus a UTF-8 ``filename*=``.
    """
    import unicodedata
    from urllib.parse import quote

    name = "".join(ch for ch in (filename or "") if ch.isprintable()).replace("\\", "_").replace("/", "_")
    name = name.strip().strip(".") or "download"
    stem, ext = os.path.splitext(name)
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    ascii_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_stem).strip("._") or "download"
    ascii_ext = ext if re.fullmatch(r"\.[A-Za-z0-9]{1,8}", ext) else ""
    ascii_name = ascii_stem + ascii_ext
    return f"{disposition}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name, safe='')}"


# ── SSRF Protection ───────────────────────────────────────────────────────────
_ALLOWED_SCHEMES = {"http", "https"}


def is_public_ip(ip: "ipaddress._BaseAddress") -> bool:
    """True only for globally routable unicast addresses.

    ``is_global`` is False for RFC-1918, loopback, link-local (incl. the 169.254.169.254
    cloud-metadata address), CGNAT 100.64/10, reserved/documentation ranges, ULA/fe80 IPv6,
    and the unspecified address. IPv4-mapped IPv6 (``::ffff:10.0.0.1``) is unwrapped first so
    it cannot be used to smuggle a private IPv4 target past the check.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(ip.is_global) and not ip.is_multicast


def assert_public_host(host: str, port: Optional[int] = None) -> None:
    """Raise HTTPException(400) unless EVERY address ``host`` resolves to is public.

    Uses getaddrinfo so IPv6 / dual-stack names are covered (the previous
    ``gethostbyname`` only ever looked at one IPv4 address).
    """
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        raise HTTPException(
            status_code=400, detail=f"URL host '{host}' could not be resolved."
        ) from None
    if not infos:
        raise HTTPException(status_code=400, detail=f"URL host '{host}' could not be resolved.")
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0].split("%", 1)[0])  # strip IPv6 zone id
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"URL host '{host}' resolved to an invalid address."
            ) from None
        if not is_public_ip(ip):
            raise HTTPException(
                status_code=400,
                detail="URL resolves to a private or reserved IP address, which is not permitted.",
            )


def validate_url(url: str) -> str:
    """Validate an external URL to prevent SSRF attacks.

    Raises HTTPException 400 for:
    - Non-HTTP(S) schemes
    - URLs with embedded credentials
    - Hosts that resolve to loopback / private / link-local / reserved addresses

    NOTE: this validates ONE hop. Anything that follows redirects must re-validate
    every hop (see ``url_service.safe_get``) — a public URL can 302 to an internal one.
    """
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise HTTPException(
            status_code=400,
            detail=f"URL scheme '{parsed.scheme}' is not permitted. Only http/https allowed.",
        )
    host = parsed.hostname or ""
    if not host:
        raise HTTPException(status_code=400, detail="URL must contain a valid host.")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="URLs with embedded credentials are not permitted.")
    try:
        port = parsed.port
    except ValueError:
        raise HTTPException(status_code=400, detail="URL contains an invalid port.") from None

    assert_public_host(host, port)
    return url


def validate_filename(filename: str) -> str:
    """Ensure filename doesn't contain path traversal."""
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Normalize slashes to allow Windows paths to be processed on Linux
    normalized = filename.replace("\\", "/")
    safe_name = os.path.basename(normalized)

    # Verify it's not empty after basename
    if not safe_name:
        raise ValueError("Invalid filename")

    return safe_name


def safe_file_path(directory: str, filename: str) -> str:
    """Securely join directory and filename, preventing path traversal."""
    safe_name = validate_filename(filename)
    full_path = os.path.join(directory, safe_name)

    # Canonicalize paths and verify the result is within the directory
    abs_directory = os.path.abspath(directory)
    abs_full_path = os.path.abspath(full_path)

    if (
        not abs_full_path.startswith(abs_directory + os.sep)
        and abs_full_path != abs_directory
    ):
        raise ValueError(f"Path traversal detected: {filename}")

    return abs_full_path


def safe_float(value: Union[str, float], name: str = "value") -> float:
    """Safely convert to float with validation."""
    try:
        val = float(value)
        # float("nan") / float("inf") parse fine and slip past `val < 0` (NaN compares False
        # to everything), so they must be rejected explicitly.
        if not math.isfinite(val):
            raise ValueError(f"{name} must be a finite number")
        if val < 0:
            raise ValueError(f"{name} must be non-negative")
        return val
    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid {name}: must be a non-negative number"
        ) from e


def validate_time_range(
    start: Union[str, float], end: Union[str, float]
) -> tuple[float, float]:
    """Validate and normalize time range for media operations."""
    start_val = safe_float(start, "start")
    end_val = safe_float(end, "end")

    if start_val >= end_val:
        raise HTTPException(
            status_code=400, detail="Start time must be less than end time"
        )

    return start_val, end_val


def sanitize_error_message(
    error: Exception, user_message: str = "An error occurred"
) -> str:
    """Create a safe error message without exposing internal details."""
    # Log the full error for debugging
    logger.error(f"Internal error: {str(error)}", exc_info=True)
    return user_message


# Allowed MIME types for file validation.
#
# The lists must cover every extension in config.ALL_SUPPORTED_EXTENSIONS, otherwise a
# supported upload (.mov, .mkv, .flv, .wma, legacy .doc/.xls/.ppt, .rtf, .log …) is rejected
# with a 400 by the extension-based fallback. Prefix match is used (see _mime_allowed).
ALLOWED_DOCUMENT_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/msword",
    "application/vnd.ms-",          # .xls / .ppt / .doc variants
    "application/rtf",
    "application/x-rtf",
    "application/json",
    "application/zip",              # libmagic reports OOXML containers as a bare zip
    "application/x-ole-storage",    # legacy OLE2 Office files
    "application/cdfv2",
    "application/octet-stream",     # libmagic could not classify (e.g. some .log/.csv)
    "inode/x-empty",
    "text/",                        # text/plain, text/csv, text/markdown, text/rtf, text/html …
}

ALLOWED_MEDIA_MIMES = {
    "audio/",                       # mpeg, wav, x-wav, ogg, aac, mp4, x-m4a, flac, x-ms-wma …
    "video/",                       # mp4, mpeg, webm, ogg, quicktime, x-matroska, x-flv, x-msvideo …
    "application/ogg",
    "application/x-matroska",
    "application/octet-stream",
}


def _mime_allowed(mime: str, allowed: set) -> bool:
    mime = (mime or "").lower()
    return any(mime.startswith(prefix) for prefix in allowed)


def _check_mime(mime: str, expected_type: str) -> None:
    """Raise HTTPException(400) if ``mime`` does not fit ``expected_type``."""
    if expected_type == "document" and not _mime_allowed(mime, ALLOWED_DOCUMENT_MIMES):
        raise HTTPException(status_code=400, detail=f"Invalid document type: {mime}")
    if expected_type == "media" and not _mime_allowed(mime, ALLOWED_MEDIA_MIMES):
        raise HTTPException(status_code=400, detail=f"Invalid media file type: {mime}")


def validate_file_type(file_path: str, expected_type: str) -> None:
    """Validate file type from content (python-magic) or, if unavailable, from the extension.

    The HTTPException raised for a genuine mismatch must NOT be swallowed: previously the
    content check ran inside a blanket ``except Exception`` that also caught the rejection
    itself, so a wrong-content file always fell through to the (weaker) extension check.
    """
    try:
        import magic  # type: ignore[import-not-found]

        actual_mime = magic.Magic(mime=True).from_file(file_path)
    except Exception as e:  # libmagic missing / unreadable — fall back to the extension
        logger.warning(f"python-magic validation unavailable ({e}), using standard mimetypes fallback")
        import mimetypes

        guessed_type, _ = mimetypes.guess_type(file_path)
        if guessed_type:
            _check_mime(guessed_type, expected_type)
        return

    _check_mime(actual_mime, expected_type)


def validate_media_file(file_path: str, timeout: int = 45) -> dict:
    """Reject, at upload time, a recording that cannot possibly be transcribed - with a reason the caller can act on.

    Content sniffing (python-magic) needs libmagic and falls back to the file EXTENSION without it, so a corrupt file
    or a video with no audio track used to be accepted, get a meeting row and a background job, and only then fail
    with a generic "FFmpeg returned non-zero exit code". ffprobe (the tool the pipeline itself relies on) is asked
    instead: is it a readable container, does it carry an audio stream, is its length sane?

    Returns ``{"duration": seconds | None, "has_video": bool, "audio_codec": str | None}`` (``{}`` when ffprobe is not
    installed - that is reported by the startup checks, and the pipeline still validates during extraction).
    ffprobe's own error text is logged, never returned: it can contain server paths.
    """
    import json
    import subprocess

    from backend.utils.media_tools import FFPROBE, ffprobe_available

    if not ffprobe_available():
        logger.warning("ffprobe is not available - media content validation skipped for %s", os.path.basename(file_path))
        return {}
    try:
        res = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name",
             "-of", "json", file_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("ffprobe timed out validating %s", os.path.basename(file_path))
        raise HTTPException(status_code=400, detail="The media file could not be analysed (it may be corrupt or truncated).")
    except OSError as exc:
        logger.warning("ffprobe could not run (%s) - media content validation skipped", exc)
        return {}

    if res.returncode != 0:
        logger.info("Rejected upload %s: ffprobe rc=%s: %s", os.path.basename(file_path), res.returncode,
                    (res.stderr or "").strip()[-300:])
        raise HTTPException(
            status_code=400,
            detail="The file is not a readable audio/video recording (it is corrupt, truncated or in an unsupported format).",
        )
    try:
        info = json.loads(res.stdout or "{}")
    except ValueError:
        info = {}
    streams = info.get("streams") or []
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        raise HTTPException(
            status_code=400,
            detail="The recording has no audio track, so there is nothing to transcribe. Upload a file that contains speech.",
        )
    duration = None
    try:
        duration = float((info.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        pass                                        # some containers do not declare a length; the audio stream is enough
    if duration is not None and duration <= 0:
        raise HTTPException(status_code=400, detail="The recording is empty (zero length).")
    return {
        "duration": duration,
        "has_video": any(s.get("codec_type") == "video" for s in streams),
        "audio_codec": audio[0].get("codec_name"),
    }


def validate_file_size(file_path: str, max_size_mb: int = 500) -> None:
    """Validate file size is within limits."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)

    if size_mb > max_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB (max {max_size_mb}MB)",
        )


def validate_chat_message(message: str, max_length: int = 0) -> str:
    """Ensure message is within bounds and not empty.
    
    max_length defaults to settings.CHAT_MAX_MESSAGE_LENGTH (or 5000 if not set).
    """
    from backend.config.config import settings as _settings
    effective_max = max_length or getattr(_settings, "CHAT_MAX_MESSAGE_LENGTH", 5000)
    
    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    cleaned = message.strip()
    if len(cleaned) > effective_max:
        raise HTTPException(
            status_code=400,
            detail=f"Message too long: {len(cleaned)} characters (max {effective_max} characters)",
        )
    return cleaned

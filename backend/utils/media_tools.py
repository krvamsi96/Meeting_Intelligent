"""Centralized, configurable resolution of the ffmpeg / ffprobe executables.

Every subprocess call in the codebase must use ``FFMPEG`` / ``FFPROBE`` from
this module instead of the bare string ``"ffmpeg"`` / ``"ffprobe"`` so that:

* the binary path is overridable per-deployment via env vars
  (``FFMPEG_BINARY`` / ``FFPROBE_BINARY``) — useful when ffmpeg is installed
  outside ``PATH`` (static build, conda, custom prefix);
* a missing binary produces one clear, actionable error at startup instead of
  opaque ``FileNotFoundError`` deep inside the pipeline.
"""

from __future__ import annotations

import logging
import os
import shutil

logger = logging.getLogger(__name__)


def _resolve(env_var: str, name: str) -> str:
    """Resolve an executable: explicit env override → PATH lookup → bare name."""
    override = (os.getenv(env_var) or "").strip()
    if override:
        # An explicit path (or name) was given — trust it, but warn if not found.
        if os.path.sep in override or override.endswith((".exe", ".bin")):
            if not os.path.isfile(override):
                logger.warning("%s=%s but that file does not exist.", env_var, override)
        return override
    found = shutil.which(name)
    return found or name


FFMPEG: str = _resolve("FFMPEG_BINARY", "ffmpeg")
FFPROBE: str = _resolve("FFPROBE_BINARY", "ffprobe")


def ffmpeg_available() -> bool:
    return shutil.which(FFMPEG) is not None or (os.path.sep in FFMPEG and os.path.isfile(FFMPEG))


def ffprobe_available() -> bool:
    return shutil.which(FFPROBE) is not None or (os.path.sep in FFPROBE and os.path.isfile(FFPROBE))


def ensure_ffmpeg_available(strict: bool = False) -> bool:
    """Log a clear message about ffmpeg/ffprobe availability.

    Returns True when both are usable. When ``strict`` is True and either is
    missing, raises RuntimeError (use at startup only if media processing is
    mandatory for this deployment).
    """
    ok_ffmpeg, ok_ffprobe = ffmpeg_available(), ffprobe_available()
    if ok_ffmpeg and ok_ffprobe:
        logger.info("Media tools OK | ffmpeg=%s | ffprobe=%s", FFMPEG, FFPROBE)
        return True

    missing = []
    if not ok_ffmpeg:
        missing.append(f"ffmpeg (looked for '{FFMPEG}')")
    if not ok_ffprobe:
        missing.append(f"ffprobe (looked for '{FFPROBE}')")
    msg = (
        "Media tools NOT found: " + ", ".join(missing) + ". "
        "Audio extraction, transcription chunking and media probing will fail. "
        "Install ffmpeg (Debian/Ubuntu: 'sudo apt-get install -y ffmpeg'; "
        "RHEL: 'sudo dnf install -y ffmpeg') or set FFMPEG_BINARY / FFPROBE_BINARY "
        "to absolute paths in .env."
    )
    if strict:
        raise RuntimeError(msg)
    logger.error(msg)
    return False

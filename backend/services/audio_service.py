import logging
import os
import subprocess

from backend.utils.media_tools import FFMPEG, FFPROBE

logger = logging.getLogger(__name__)


def _probe_duration(path: str) -> float:
    """Return media duration in seconds via ffprobe (0.0 on failure)."""
    try:
        res = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120,
        )
        return float((res.stdout or "0").strip() or 0.0)
    except Exception:
        return 0.0


def extract_audio(input_filepath: str, output_filepath: str) -> bool:
    """
    Extract audio from a video or audio file and convert it to a standard 16kHz mono WAV format
    (which is ideal for Whisper and NVIDIA NeMo).

    Audio filter rationale:
    - dynaudnorm: dynamics-only normalization that adjusts gain frame-by-frame without
      altering the inter-speaker amplitude ratio or spectral shape. Unlike loudnorm
      (EBU R128 two-pass), it does not distort speaker identity signals.
    - highpass=f=80: removes sub-bass rumble (< 80 Hz) that is microphone noise,
      while retaining low-frequency vocal resonance (fundamental frequency ~80-300 Hz)
      which contributes to speaker discrimination in MFCC features.

    Logs the source vs extracted duration so a truncated / partial extraction on a
    long recording is visible instead of silently transcribing only the first
    few seconds. `-drop_late_frames`/`-err_detect ignore_err` + a generous
    timeout make multi-hour inputs robust.
    """
    src_dur = _probe_duration(input_filepath)
    src_mb = os.path.getsize(input_filepath) / (1024 * 1024) if os.path.exists(input_filepath) else 0.0
    logger.info(
        "[AudioExtract] START | src=%s | src_size=%.1f MB | src_duration=%.1f s (%.1f min)",
        os.path.basename(input_filepath), src_mb, src_dur, src_dur / 60.0,
    )

    # No fixed timeout can suit every host, but an unbounded subprocess can hang a
    # worker forever on a corrupt stream. Scale the cap to the input length
    # (assume worst case ~2x realtime for filtering on a slow CPU) with a floor.
    timeout_s = max(1800, int(src_dur * 2.5) + 300) if src_dur > 0 else 3600
    _ff_threads = os.getenv("FFMPEG_THREADS", os.getenv("AI_MAX_THREADS", "2"))

    base_in = [
        FFMPEG, "-y", "-nostdin",
        "-hide_banner", "-loglevel", "error",   # stderr = the actual error, not a 1 KB build banner
        "-err_detect", "ignore_err",   # don't abort mid-stream on a bad packet
        "-threads", str(_ff_threads),
        "-i", input_filepath,
        "-vn",
    ]
    tail = ["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_filepath]

    def _run(cmd, label):
        logger.info("[AudioExtract] %s: %s", label, " ".join(cmd))
        try:
            p = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=timeout_s,
            )
            return p
        except subprocess.TimeoutExpired:
            logger.error("[AudioExtract] %s TIMED OUT after %ss", label, timeout_s)
            return None

    try:
        process = _run(
            base_in + ["-af", "dynaudnorm=f=150:g=15,highpass=f=80"] + tail,
            "normalized",
        )
        if process is None or process.returncode != 0:
            logger.warning(
                "[AudioExtract] normalized pass failed (%s); retrying plain PCM. stderr(tail): %s",
                None if process is None else process.returncode,
                "" if process is None else (process.stderr or "")[-800:],
            )
            process = _run(base_in + tail, "plain-pcm")
            if process is None or process.returncode != 0:
                logger.error(
                    "[AudioExtract] FAILED. stderr(tail): %s",
                    "" if process is None else (process.stderr or "")[-1200:],
                )
                return False

        if not os.path.exists(output_filepath) or os.path.getsize(output_filepath) == 0:
            logger.error("[AudioExtract] ffmpeg returned 0 but output is missing/empty: %s", output_filepath)
            return False

        out_dur = _probe_duration(output_filepath)
        out_mb = os.path.getsize(output_filepath) / (1024 * 1024)
        logger.info(
            "[AudioExtract] DONE | out=%s | out_size=%.1f MB | out_duration=%.1f s (%.1f min)",
            os.path.basename(output_filepath), out_mb, out_dur, out_dur / 60.0,
        )
        # Flag a materially short extraction (e.g. stream ended early / corruption)
        if src_dur > 5 and out_dur > 0 and out_dur < src_dur * 0.9:
            logger.warning(
                "[AudioExtract] Extracted audio is %.1f min but source is %.1f min "
                "(%.0f%% captured) — the recording may be truncated or partially corrupt. "
                "Transcription will still run on what was extracted.",
                out_dur / 60.0, src_dur / 60.0, (out_dur / src_dur) * 100.0,
            )
        return True
    except Exception as e:
        logger.error(f"[AudioExtract] Failed to extract audio: {e}", exc_info=True)
        return False

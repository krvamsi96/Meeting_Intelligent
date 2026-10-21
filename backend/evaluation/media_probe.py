"""Media signal telemetry — SNR, silence ratio, FPS, codec stats.

All values are measured directly from the media file via ffprobe/ffmpeg and
numpy signal analysis. No values are estimated; if a signal cannot be
measured the field is reported as None (→ ``not_measured`` upstream).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional

from backend.utils.media_tools import FFMPEG, FFPROBE

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False


def _run(cmd) -> Optional[str]:
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
        )
        return proc.stdout.decode("utf-8", errors="replace") if proc.returncode == 0 else None
    except Exception as e:
        logger.warning("Media probe command failed: %s", e)
        return None


def _probe_json(filepath: str) -> Optional[Dict[str, Any]]:
    out = _run(
        [
            FFPROBE, "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", filepath,
        ]
    )
    if not out:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


# ─── Audio ────────────────────────────────────────────────────────────────────

def probe_audio(filepath: str) -> Dict[str, Any]:
    """Measure audio telemetry: SNR (dB), silence ratio, bitrate, sample rate,
    codec, channels, peak/RMS levels. Returns ``None`` fields when unmeasurable.
    """
    result: Dict[str, Any] = {
        "codec": None, "sample_rate": None, "channels": None, "bitrate": None,
        "duration_sec": None, "snr_db": None, "silence_ratio": None,
        "peak_db": None, "rms_db": None, "clipping_ratio": None,
    }
    if not filepath or not os.path.isfile(filepath):
        return result

    probe = _probe_json(filepath)
    if probe:
        audio_streams = [
            s for s in probe.get("streams", []) if s.get("codec_type") == "audio"
        ]
        fmt = probe.get("format", {})
        duration = float(fmt.get("duration", 0.0) or 0.0)
        result["duration_sec"] = round(duration, 2)
        result["bitrate"] = int(fmt["bit_rate"]) if fmt.get("bit_rate") else None
        if audio_streams:
            s = audio_streams[0]
            result["codec"] = s.get("codec_name")
            result["sample_rate"] = int(s["sample_rate"]) if s.get("sample_rate") else None
            result["channels"] = int(s.get("channels", 0) or 0)
            result["bitrate"] = int(s["bit_rate"]) if s.get("bit_rate") else result["bitrate"]

    samples = _decode_pcm16_mono(filepath)
    if samples is not None and len(samples) > 0:
        audio = samples.astype(np.float64) / 32768.0
        frame_size = int(16000 * 0.05)  # 50 ms frames at 16 kHz
        n_frames = len(audio) // frame_size
        if n_frames > 0:
            frames = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))

            peak = float(np.max(np.abs(audio)))
            result["peak_db"] = round(20.0 * np.log10(peak + 1e-12), 2)
            result["rms_db"] = round(20.0 * np.log10(float(np.sqrt(np.mean(audio ** 2))) + 1e-12), 2)
            result["clipping_ratio"] = round(float(np.mean(np.abs(audio) >= 0.999)), 6)

            # Noise floor: 10th percentile frame RMS (robust estimator)
            floor = float(np.percentile(frame_rms, 10))
            signal = float(np.percentile(frame_rms, 90))
            if signal > 0 and floor > 0:
                result["snr_db"] = round(20.0 * np.log10(signal / floor), 2)

            silence_threshold = max(floor * 3.0, 1e-4)
            result["silence_ratio"] = round(
                float(np.mean(frame_rms < silence_threshold)), 4
            )
    return result


def _decode_pcm16_mono(filepath: str) -> Optional[Any]:
    """Decode file to raw 16 kHz mono PCM s16le samples via ffmpeg (numpy array)."""
    if not HAS_NUMPY or shutil.which(FFMPEG) is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
            tmp_path = tmp.name
        cmd = [
            FFMPEG, "-v", "quiet", "-i", filepath,
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "s16le", "-acodec", "pcm_s16le", "-y", tmp_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        if proc.returncode != 0:
            logger.warning("ffmpeg decode failed for %s: %s", filepath, proc.stderr[:200])
            return None
        with open(tmp_path, "rb") as f:
            raw = f.read()
        return np.frombuffer(raw, dtype=np.int16)
    except Exception as e:
        logger.warning("PCM decode error for %s: %s", filepath, e)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─── Video ────────────────────────────────────────────────────────────────────

def probe_video(filepath: str) -> Dict[str, Any]:
    """Measure video telemetry: FPS, resolution, bitrate, codec, duration,
    and frame-drop ratio (measured frames vs nominal frame count).
    """
    result: Dict[str, Any] = {
        "codec": None, "width": None, "height": None, "fps": None,
        "bitrate": None, "duration_sec": None, "frame_count": None,
        "frame_drop_ratio": None,
    }
    if not filepath or not os.path.isfile(filepath):
        return result

    probe = _probe_json(filepath)
    if not probe:
        return result

    fmt = probe.get("format", {})
    result["duration_sec"] = round(float(fmt.get("duration", 0.0) or 0.0), 2)
    result["bitrate"] = int(fmt["bit_rate"]) if fmt.get("bit_rate") else None

    for s in probe.get("streams", []):
        if s.get("codec_type") != "video":
            continue
        result["codec"] = s.get("codec_name")
        result["width"] = s.get("width")
        result["height"] = s.get("height")
        result["bitrate"] = int(s["bit_rate"]) if s.get("bit_rate") else result["bitrate"]
        if s.get("r_frame_rate"):
            num, _, den = s["r_frame_rate"].partition("/")
            fps = float(num) / float(den) if den else float(num)
            if fps > 0:
                result["fps"] = round(fps, 2)
        if s.get("nb_frames"):
            try:
                result["frame_count"] = int(s["nb_frames"])
            except (TypeError, ValueError):
                pass
        break

    duration = result["duration_sec"]
    if result["fps"] and duration and result["frame_count"]:
        nominal = duration * result["fps"]
        if nominal > 0:
            result["frame_drop_ratio"] = round(
                max(0.0, 1.0 - result["frame_count"] / nominal), 4
            )
    return result


def probe_media(filepath: str) -> Dict[str, Any]:
    """Probe both audio and video streams of a media file."""
    return {
        "audio": probe_audio(filepath),
        "video": probe_video(filepath),
    }

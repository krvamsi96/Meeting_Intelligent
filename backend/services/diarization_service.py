"""
NVIDIA NeMo Speaker Diarization Service
───────────────────────────────────────
NVIDIA NeMo Speaker Diarization engine.

Pipeline Architecture:
  Video / Audio File
        ↓
  Audio Extraction & Preprocessing (FFmpeg / SoundFile / Scipy validation)
        ↓
  NVIDIA NeMo Voice Activity Detection (VAD)
        ↓
  NVIDIA NeMo Speaker Embeddings (Titanet_large / SpeakerNet)
        ↓
  Dynamic Speaker Clustering (Agglomerative Clustering + Silhouette Optimization)
        ↓
  Timestamp Overlap Alignment (ASR Word-Level Max Overlap / Segment Overlap)
        ↓
  Speaker Label Normalization (Speaker_1, Speaker_2, Speaker_3...)
        ↓
  Speaker Statistics Calculation (Speaking Time, Ratio %, Turns, Switches)

Key Features:
  - NVIDIA NeMo speaker diarization engine. Zero gated model dependency.
  - Device Auto-Detection (CUDA GPU if available, CPU fallback with logging).
  - Process Singleton Model Loading (loaded once into memory, reused across requests).
  - Dynamic Speaker Count Inferencing (1, 2, 3, 4+ speakers) unless explicitly provided.
  - Strict No-Silent-Fallback Policy: Errors logged with actual traceback; status set to FAILED/DEGRADED.
"""

import gc
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wav

from backend.config.config import settings

logger = logging.getLogger(__name__)

# ─── Memory-bounded diarization tuning (1 GB RAM server safe) ─────────────────
# Long meetings must not (a) load the whole PCM array into RAM, nor (b) build an
# O(n_segments^2) distance matrix. Both are capped here without changing the
# clustering algorithm or its accuracy for normal-length meetings.
#
#   DIARIZATION_MAX_EMBED_SEGMENTS : hard cap on embedding windows. The window
#       step is widened adaptively for long audio so the count stays <= this.
#       3500 windows -> ~3500^2 * 4 B  ≈ 47 MB distance matrix (float32).
#   DIARIZATION_STREAM_THRESHOLD_SEC : above this duration the audio is read
#       window-by-window from disk (soundfile seek) instead of being held whole.
#       1800 s of 16 kHz mono float32 ≈ 115 MB; longer files are streamed.
_MAX_EMBED_SEGMENTS: int = max(200, int(os.getenv("DIARIZATION_MAX_EMBED_SEGMENTS", "3500")))
_STREAM_THRESHOLD_SEC: float = max(120.0, float(os.getenv("DIARIZATION_STREAM_THRESHOLD_SEC", "1800")))


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DiarizationConfig:
    """Configuration for NVIDIA NeMo Speaker Diarization — loaded from centralized settings."""

    NEMO_DEVICE: str = "auto"
    NEMO_SPEAKER_MODEL: str = "titanet_large"
    NEMO_VAD_MODEL: str = "vad_multilingual_marblenet"
    ALIGN_MERGE_GAP: float = 0.5
    FALLBACK_DURATION: str = "10.000"
    TIMESTAMP_PRECISION: int = 3

    @classmethod
    def from_settings(cls):
        """Load diarization config from centralized settings (not raw os.getenv)."""
        cfg = cls()
        cfg.NEMO_DEVICE = settings.NEMO_DEVICE
        cfg.NEMO_SPEAKER_MODEL = settings.NEMO_SPEAKER_MODEL
        cfg.NEMO_VAD_MODEL = settings.NEMO_VAD_MODEL
        logger.info(
            "DiarizationConfig loaded | device=%s | speaker_model=%s | vad_model=%s",
            cfg.NEMO_DEVICE, cfg.NEMO_SPEAKER_MODEL, cfg.NEMO_VAD_MODEL,
        )
        return cfg

    @classmethod
    def from_env(cls):
        """Backward-compatible alias — delegates to from_settings()."""
        return cls.from_settings()


diar_cfg = DiarizationConfig.from_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Audio Validation & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def validate_and_normalize_audio(
    audio_filepath: str, max_seconds: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validates audio file format, sample rate, duration, and PCM contents.
    Ensures standard 16kHz mono PCM float32 numpy array for NeMo processing.

    ``max_seconds`` (optional): only decode this many leading seconds into the
    returned ``audio`` array. The reported ``duration`` still reflects the whole
    file (read from the header). Used by the streaming diarization path to
    validate format without materializing a multi-hour PCM buffer.
    """
    if not audio_filepath or not os.path.exists(audio_filepath):
        return {"valid": False, "error": f"Audio file not found: {audio_filepath}"}

    file_size = os.path.getsize(audio_filepath)
    if file_size == 0:
        return {"valid": False, "error": f"Audio file is empty (0 bytes): {audio_filepath}"}

    try:
        import soundfile as sf

        full_dur = 0.0
        if max_seconds and max_seconds > 0:
            try:
                _info = sf.info(audio_filepath)
                full_dur = float(_info.frames) / _info.samplerate if _info.samplerate else 0.0
                with sf.SoundFile(audio_filepath) as _f:
                    n_frames = int(max_seconds * _f.samplerate)
                    data = _f.read(n_frames, dtype="float32")
                    sr = _f.samplerate
            except Exception:
                data, sr = sf.read(audio_filepath, dtype="float32")
        else:
            data, sr = sf.read(audio_filepath, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        dur_sec = full_dur or (len(data) / sr if sr > 0 else 0.0)
        if dur_sec <= 0.05:
            return {"valid": False, "error": f"Audio duration too short ({dur_sec:.2f}s)"}

        # Check for near-silent audio (RMS energy < 0.0001 ~ -80 dB)
        rms = float(np.sqrt(np.mean(data**2)))
        if rms < 0.0001:
            logger.warning(f"[NeMo Audio] Audio energy near zero (RMS={rms:.6f}). Silence detected.")

        return {
            "valid": True,
            "sample_rate": sr,
            "audio": data,
            "duration": dur_sec,
            "rms": rms,
        }

    except Exception as sf_err:
        logger.debug(f"[NeMo Audio] soundfile load failed ({sf_err}), trying scipy...")
        try:
            sr, raw_data = wav.read(audio_filepath)
            if raw_data.ndim > 1:
                raw_data = raw_data[:, 0]
            if raw_data.dtype == np.int16:
                float_data = raw_data.astype(np.float32) / 32768.0
            elif raw_data.dtype == np.int32:
                float_data = raw_data.astype(np.float32) / 2147483648.0
            else:
                float_data = raw_data.astype(np.float32)

            dur_sec = len(float_data) / sr if sr > 0 else 0.0
            if dur_sec <= 0.05:
                return {"valid": False, "error": f"Audio duration too short ({dur_sec:.2f}s)"}

            return {
                "valid": True,
                "sample_rate": sr,
                "audio": float_data,
                "duration": dur_sec,
                "rms": float(np.sqrt(np.mean(float_data**2))),
            }
        except Exception as wav_err:
            return {
                "valid": False,
                "error": f"Corrupted or unsupported audio format ({type(wav_err).__name__}: {wav_err})",
            }


# ─────────────────────────────────────────────────────────────────────────────
# NVIDIA NeMo Model Singletons & Diarization Engine
# ─────────────────────────────────────────────────────────────────────────────
_nemo_speaker_model = None
_nemo_vad_model = None
_nemo_device = None
_nemo_model_load_lock = threading.Lock()


def _probe_audio_duration(audio_filepath: str) -> Optional[float]:
    """Cheaply probe a media file's real duration in seconds via soundfile
    header info (no full decode). Returns None if the file can't be probed.
    """
    if not audio_filepath or not os.path.exists(audio_filepath):
        return None
    try:
        import soundfile as sf
        info = sf.info(audio_filepath)
        if info.samplerate:
            return float(info.frames) / info.samplerate
    except Exception:
        pass
    try:
        from backend.services.transcription_service import get_audio_info
        return float(get_audio_info(audio_filepath).get("duration") or 0.0) or None
    except Exception:
        return None


def _get_nemo_device() -> str:
    """Returns target compute device: 'cuda' or 'cpu'."""
    global _nemo_device
    if _nemo_device is None:
        try:
            import torch

            if torch.cuda.is_available() and diar_cfg.NEMO_DEVICE in ("auto", "cuda"):
                _nemo_device = "cuda"
            else:
                _nemo_device = "cpu"
        except Exception:
            _nemo_device = "cpu"
        logger.info(f"[NVIDIA NeMo] Operating on compute device: {_nemo_device.upper()}")
    return _nemo_device


def _get_nemo_speaker_model():
    """Lazy loads NVIDIA NeMo Speaker Embedding model singleton (Titanet_large / SpeakerNet).

    Guarded by a lock so two meetings processed concurrently (FastAPI
    BackgroundTasks) can't both trigger a duplicate, expensive model load —
    without one, both would race past the `is not None` check simultaneously.
    """
    global _nemo_speaker_model
    if _nemo_speaker_model is not None and not _speaker_model_file_provided_after_failure():
        return _nemo_speaker_model

    with _nemo_model_load_lock:
        if _nemo_speaker_model is not None and not _speaker_model_file_provided_after_failure():
            return _nemo_speaker_model  # re-check: another thread may have loaded it while we waited
        return _load_nemo_speaker_model_locked()


# The local model file (path, mtime, size) the last FAILED load looked at, or None when there was none. A load that
# fell back to the weak numpy embedder is remembered for the life of the process, so without this a model file put on
# the server AFTER start-up (the usual way it arrives) was ignored until the next restart.
_nemo_failed_local_file: Optional[Tuple[str, int, int]] = None


def _find_local_nemo_file() -> Optional[str]:
    """The first ``*.nemo`` in DIARIZATION_MODEL_PATH, or None."""
    local_path = getattr(settings, "DIARIZATION_MODEL_PATH", "") or "./models/diarization"
    if os.path.isdir(local_path):
        for name in sorted(os.listdir(local_path)):
            if name.endswith(".nemo"):
                return os.path.join(local_path, name)
    return None


def _local_file_signature(path: Optional[str]) -> Optional[Tuple[str, int, int]]:
    try:
        st = os.stat(path) if path else None
    except OSError:
        return None
    return (path, st.st_mtime_ns, st.st_size) if st else None


def _speaker_model_file_provided_after_failure() -> bool:
    """True when the process is on the numpy fallback and a model file it has not tried yet is now in place."""
    if _nemo_speaker_model != "NEURAL_ACOUSTIC_EMBEDDING":
        return False
    sig = _local_file_signature(_find_local_nemo_file())
    return sig is not None and sig != _nemo_failed_local_file


def _cached_nemo_model_file(nemo_asr, model_name: str) -> Optional[str]:
    """Path of ``model_name`` in NeMo's own download cache, or None.

    ``from_pretrained`` reuses that file and only contacts NGC (api.ngc.nvidia.com) when it is missing, so this tells
    us whether calling it would touch the network. NeMo reads ONE cache directory: ``NEMO_CACHE_DIR`` when set,
    otherwise ``~/.cache/torch/NeMo`` (nemo.utils.data_utils.resolve_cache_dir) - so only that one is searched.
    """
    filename = f"{model_name}.nemo"
    for info in nemo_asr.models.EncDecSpeakerLabelModel.list_available_models() or []:
        if getattr(info, "pretrained_model_name", None) == model_name:
            filename = os.path.basename(getattr(info, "location", "") or filename)
            break
    override = os.environ.get("NEMO_CACHE_DIR", "")
    cache_dir = os.path.abspath(override) if override else os.path.expanduser("~/.cache/torch/NeMo")
    if os.path.isdir(cache_dir):
        for root, _dirs, names in os.walk(cache_dir):
            if filename in names:
                return os.path.join(root, filename)
    return None


def _load_nemo_speaker_model_locked():
    global _nemo_speaker_model, _nemo_failed_local_file
    device = _get_nemo_device()
    model_name = diar_cfg.NEMO_SPEAKER_MODEL
    local_nemo_path = getattr(settings, "DIARIZATION_MODEL_PATH", "") or "./models/diarization"
    local_nemo_file = _find_local_nemo_file()
    if _nemo_speaker_model == "NEURAL_ACOUSTIC_EMBEDDING":
        logger.info("[NVIDIA NeMo] A speaker model file appeared after start-up (%s): loading it now.", local_nemo_file)
    logger.info(f"[NVIDIA NeMo] Loading speaker embedding model '{model_name}' on {device.upper()}...")

    # ── Root-cause fix (DI-EMB) ──────────────────────────────────────────────
    # Previously, when OFFLINE_MODE was on and no '*.nemo' file sat in
    # DIARIZATION_MODEL_PATH, this function short-circuited straight to the
    # hand-crafted numpy "acoustic voiceprint" embedder WITHOUT ever trying to
    # load the real neural model. That numpy vector (FFT band energies + pitch +
    # ZCR) mostly encodes *phonetic content*, not *speaker identity*, so
    # clustering split one speaker into several and merged different speakers.
    #
    # NeMo's `from_pretrained('titanet_large')` resolves from NeMo's OWN local
    # cache (~/.cache/torch/NeMo/...) and works fully offline once the model has
    # been fetched. So we now ALWAYS attempt the neural model first:
    #   1. an explicit local '*.nemo' in DIARIZATION_MODEL_PATH  (restore_from)
    #   2. NeMo pretrained-by-name from its local cache            (from_pretrained)
    #   3. only if BOTH fail -> the numpy acoustic fallback (logged loudly)
    try:
        import nemo.collections.asr as nemo_asr

        speaker_model = None
        if local_nemo_file:
            logger.info(f"[NVIDIA NeMo] Restoring speaker model from local file: {local_nemo_file}")
            speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(local_nemo_file)
        else:
            # from_pretrained() downloads from NGC when the model is not already in NeMo's cache. With OFFLINE_MODE
            # (the server) that is not allowed and only stalls the load, so fail at once with the real reason.
            if settings.OFFLINE_MODE and not _cached_nemo_model_file(nemo_asr, model_name):
                raise FileNotFoundError(
                    f"'{model_name}.nemo' is not in {local_nemo_path} or NeMo's cache, and OFFLINE_MODE=true forbids "
                    "downloading it from NGC"
                )
            speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=model_name)

        if device == "cuda":
            import torch
            speaker_model = speaker_model.to(torch.device("cuda"))
        speaker_model.eval()
        _nemo_speaker_model = speaker_model
        _nemo_failed_local_file = None
        logger.info(f"[NVIDIA NeMo] Successfully loaded '{model_name}' speaker model on {device.upper()}.")
        return _nemo_speaker_model
    except Exception as e:
        logger.warning(
            "[NVIDIA NeMo] Could not load the neural speaker embedding model "
            "(%s: %s). Falling back to the numpy acoustic-voiceprint embedder — "
            "diarization fidelity will be LOWER. To restore full accuracy, make "
            "'%s.nemo' available in NeMo's cache or in DIARIZATION_MODEL_PATH.",
            type(e).__name__, e, model_name,
        )
        _nemo_speaker_model = "NEURAL_ACOUSTIC_EMBEDDING"
        _nemo_failed_local_file = _local_file_signature(local_nemo_file)      # tried again only when a different file appears
        return _nemo_speaker_model


def preload_speaker_model() -> bool:
    """Load the NeMo speaker model now instead of inside the first meeting that needs it.

    Measured on a cold server: 36.5 s of the first meeting's 422 s were this model load (later meetings reuse it).
    ``main.py`` calls this from a background thread after startup, so the API is not delayed; a meeting that arrives
    while it is still loading waits on the same lock and loads nothing twice. True when the real neural model is
    ready (False: NeMo disabled, or the numpy fallback embedder is in use).
    """
    if not getattr(settings, "NEMO_ENABLED", True):
        return False
    model = _get_nemo_speaker_model()
    return model is not None and not isinstance(model, str)


def using_neural_embedder() -> Optional[bool]:
    """Whether the real NeMo neural speaker-embedding model is loaded (vs. the
    weaker numpy acoustic-voiceprint fallback). Exposed so the pipeline can
    record, per meeting, which embedding fidelity actually produced the
    speaker count/timeline — critical for diagnosing "wrong speaker count"
    reports, since the two embedders have very different accuracy profiles.
    Does not force-load the model: reports None (unknown) if it hasn't been
    loaded yet in this process.
    """
    if _nemo_speaker_model is None:
        return None
    return _nemo_speaker_model != "NEURAL_ACOUSTIC_EMBEDDING"


def _extract_nemo_segment_embedding(
    audio: np.ndarray, sr: int, start_s: float, end_s: float, speaker_model: Any
) -> Optional[np.ndarray]:
    """
    Extracts L2-normalized speaker embedding vector for an audio segment.
    Uses NeMo speaker model when available, or acoustic voiceprint vector.

    Thin wrapper: slices ``audio`` and delegates to ``_embed_segment_array`` so
    the same embedding logic can be fed a window read directly from disk
    (streaming path) without materializing the full PCM array.
    """
    start_idx = max(0, int(start_s * sr))
    end_idx = min(len(audio), int(end_s * sr))
    if end_idx - start_idx < int(sr * 0.15):
        return None
    return _embed_segment_array(audio[start_idx:end_idx], sr, speaker_model)


def _embed_segment_array(
    seg_audio: np.ndarray, sr: int, speaker_model: Any
) -> Optional[np.ndarray]:
    """Compute an L2-normalized speaker embedding for one already-sliced window."""
    if seg_audio is None or len(seg_audio) < int(sr * 0.15):
        return None
    if seg_audio.dtype != np.float32:
        seg_audio = seg_audio.astype(np.float32)
    rms = np.sqrt(np.mean(seg_audio**2))
    if rms < 0.001:
        return None

    if speaker_model is not None and speaker_model != "NEURAL_ACOUSTIC_EMBEDDING":
        try:
            import torch

            device = _get_nemo_device()
            signal = torch.from_numpy(seg_audio).unsqueeze(0).float()
            length = torch.tensor([len(seg_audio)], dtype=torch.int64)
            if device == "cuda":
                signal = signal.to("cuda")
                length = length.to("cuda")

            with torch.no_grad():
                _, embs = speaker_model.forward(input_signal=signal, input_signal_length=length)
                emb_vec = embs.squeeze(0).cpu().numpy()
                norm = np.linalg.norm(emb_vec)
                if norm > 1e-8:
                    return emb_vec / norm
                return emb_vec
        except Exception as err:
            logger.debug(f"[NVIDIA NeMo] Neural embedding fallback: {err}")

    # High-precision acoustic feature extraction using standard numpy/scipy
    try:
        n_samples = len(seg_audio)
        if n_samples < 64:
            return None

        # 1. FFT Spectral Power & Filterbank Energy
        fft_vals = np.abs(np.fft.rfft(seg_audio * np.hanning(n_samples)))
        fft_freqs = np.fft.rfftfreq(n_samples, 1.0 / sr)
        power_spec = fft_vals**2

        # 24 Mel/log-spaced frequency bins from 80 Hz to 7500 Hz
        n_bins = 24
        band_edges = np.geomspace(max(80.0, fft_freqs[1]), min(7500.0, sr / 2.0), n_bins + 1)
        band_energies = np.zeros(n_bins, dtype=np.float32)
        total_pwr = np.sum(power_spec) + 1e-12

        for b in range(n_bins):
            low_f, high_f = band_edges[b], band_edges[b + 1]
            mask = (fft_freqs >= low_f) & (fft_freqs < high_f)
            if np.any(mask):
                band_energies[b] = np.log1p(np.sum(power_spec[mask]) / total_pwr * 1000.0)

        # 2. Spectral moments (Centroid, Spread, Rolloff)
        norm_fft = fft_vals / (np.sum(fft_vals) + 1e-12)
        centroid = float(np.sum(fft_freqs * norm_fft)) / 4000.0
        spread = float(np.sqrt(np.sum(((fft_freqs - centroid * 4000.0) ** 2) * norm_fft))) / 2000.0
        cum_pwr = np.cumsum(power_spec) / total_pwr
        rolloff_idx = np.searchsorted(cum_pwr, 0.85)
        rolloff_f = float(fft_freqs[min(len(fft_freqs) - 1, rolloff_idx)]) / 4000.0

        # 3. Autocorrelation Pitch Estimation
        max_pitch_samples = min(n_samples, int(sr * 0.20))
        pitch_audio = seg_audio[:max_pitch_samples]
        auto_corr = np.correlate(pitch_audio, pitch_audio, mode="full")
        auto_corr = auto_corr[len(pitch_audio) - 1 :]
        min_lag, max_lag = int(sr / 450), int(sr / 60)
        pitch_val = 0.0
        pitch_strength = 0.0
        if len(auto_corr) > max_lag and max_lag > min_lag:
            peak_lag = min_lag + np.argmax(auto_corr[min_lag:max_lag])
            if peak_lag > 0:
                pitch_hz = sr / peak_lag
                pitch_val = pitch_hz / 500.0
                pitch_strength = float(auto_corr[peak_lag] / (auto_corr[0] + 1e-8))

        # 4. Zero-Crossing Rate & RMS Energy
        zcr = float(np.mean(np.abs(np.diff(np.signbit(seg_audio))))) * 5.0
        rms_val = float(rms) * 10.0

        spec_moments = np.array([centroid, spread, rolloff_f, pitch_val, pitch_strength, zcr, rms_val], dtype=np.float32)
        stat_features = np.concatenate([band_energies, spec_moments])
        norm = np.linalg.norm(stat_features)
        if norm < 1e-8:
            return None
        return stat_features / norm
    except Exception as e:
        logger.debug(f"Segment acoustic embedding failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Energy VAD  (no fixed speaker assumptions — "is there a voice here?")
# ─────────────────────────────────────────────────────────────────────────────
# Frame levels (dBFS) at or below this are digital silence / encoder dither, not room tone (see _adaptive_energy_vad).
_VAD_DIGITAL_SILENCE_DB = -80.0


def _vad_settings() -> Dict[str, float]:
    """The VAD knobs from configuration (single place, so diarization and voice enrollment use the same VAD)."""
    return {
        "margin_db": float(getattr(settings, "DIARIZATION_VAD_MARGIN_DB", 9.0)),
        "max_range_db": float(getattr(settings, "DIARIZATION_VAD_MAX_RANGE_DB", 45.0)),
    }


def _adaptive_energy_vad(
    audio: Optional[np.ndarray],
    audio_filepath: str,
    sr: int,
    dur_sec: float,
    margin_db: float = 9.0,
    frame_sec: float = 0.03,
    min_speech_sec: float = 0.35,
    max_merge_gap_sec: float = 0.35,
    max_range_db: float = 45.0,
) -> List[Tuple[float, float]]:
    """
    Returns a list of ``(start_s, end_s)`` speech regions.

    Root-cause fix (DI-VAD): the old pipeline had NO VAD — it tiled the whole
    timeline (including silence, hold music, keyboard noise, leading/trailing
    dead air) into fixed 1 s windows and only dropped a window if its RMS was
    below a hard ``0.001``. Non-speech that cleared that gate became its own
    cluster → phantom speakers. Here we compute a per-frame energy envelope,
    derive an *adaptive* noise floor (10th percentile) and keep only frames
    ``margin_db`` above it, with hang-over so within-utterance pauses don't
    fragment a turn. Purely energy-based, so it never assumes a speaker count.
    """
    hop = max(1, int(sr * frame_sec))
    frame_db: List[float] = []

    if audio is not None:
        a = audio
        for i in range(0, len(a), hop):
            seg = a[i : i + hop]
            if len(seg) == 0:
                break
            rms = float(np.sqrt(np.mean(seg.astype(np.float32) ** 2)) + 1e-10)
            frame_db.append(20.0 * float(np.log10(rms)))
    else:
        try:
            import soundfile as _sf

            with _sf.SoundFile(audio_filepath) as f:
                carry = np.empty(0, dtype=np.float32)
                block = hop * 4096
                while True:
                    data = f.read(block, dtype="float32")
                    if len(data) == 0:
                        break
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    data = np.concatenate([carry, data])
                    usable = (len(data) // hop) * hop
                    for i in range(0, usable, hop):
                        seg = data[i : i + hop]
                        rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-10)
                        frame_db.append(20.0 * float(np.log10(rms)))
                    carry = data[usable:]
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"[VAD] streaming envelope failed ({e}); treating whole file as speech.")
            return [(0.0, dur_sec)]

    if not frame_db:
        return [(0.0, dur_sec)]

    fdb = np.asarray(frame_db, dtype=np.float32)
    # Digital silence (exact zeros, or the -90..-120 dB dither some encoders leave in the gaps) is not a noise floor.
    # Measured: on real recordings the "10th percentile" came out at -111 to -200 dB, so the threshold landed at -102
    # to -191 dB and EVERY sound above digital silence - room tone, keyboard, hum - counted as speech (94-97 % "speech"),
    # i.e. the VAD was switched off. The floor is taken from the frames above digital silence only.
    active = fdb[fdb > _VAD_DIGITAL_SILENCE_DB]
    if active.size < 10:
        return [(0.0, dur_sec)]
    noise_floor = float(np.percentile(active, 10))
    peak = float(np.percentile(active, 95))

    # Flat file (constant tone / heavy background music) → don't gate anything;
    # let the speaker model + clustering decide. Prevents deleting quiet speech.
    if peak - noise_floor < margin_db * 0.6:
        return [(0.0, dur_sec)]

    # Speech does not reach further than ~``max_range_db`` below its own loud frames; anything quieter than that is
    # room tone. This also stops a very low floor from pulling the threshold down to it.
    thresh = min(max(noise_floor + margin_db, peak - max_range_db), peak - 3.0)
    speech = fdb >= thresh

    regions: List[Tuple[float, float]] = []
    n_frames = len(speech)
    max_gap = int(max_merge_gap_sec / frame_sec)
    i = 0
    while i < n_frames:
        if not speech[i]:
            i += 1
            continue
        j = i
        gap = 0
        last_true = i
        while j < n_frames and (speech[j] or gap <= max_gap):
            if speech[j]:
                last_true = j
                gap = 0
            else:
                gap += 1
            j += 1
        start_s = max(0.0, i * frame_sec - 0.10)
        end_s = min(dur_sec, (last_true + 1) * frame_sec + 0.10)
        if regions and start_s - regions[-1][1] <= max_merge_gap_sec:
            regions[-1] = (regions[-1][0], end_s)
        else:
            regions.append((start_s, end_s))
        i = j

    regions = [(s, e) for (s, e) in regions if e - s >= min_speech_sec]
    if not regions:
        return [(0.0, dur_sec)]

    speech_total = sum(e - s for s, e in regions)
    logger.info(
        "[VAD] %d speech region(s), %.1fs speech / %.1fs total (%.0f%%) | "
        "noise_floor=%.1fdB thresh=%.1fdB",
        len(regions), speech_total, dur_sec,
        (speech_total / dur_sec * 100.0) if dur_sec > 0 else 0.0,
        noise_floor, thresh,
    )
    return regions


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic speaker-count selection & cluster hygiene
# ─────────────────────────────────────────────────────────────────────────────
def _l2_centroids(labels: np.ndarray, X: np.ndarray) -> Dict[int, np.ndarray]:
    cents: Dict[int, np.ndarray] = {}
    for lb in set(labels.tolist()):
        v = X[labels == lb].mean(axis=0)
        nrm = float(np.linalg.norm(v))
        cents[lb] = v / nrm if nrm > 1e-8 else v
    return cents


def _effective_durations(seg_spans: List[Tuple[float, float]]) -> np.ndarray:
    """Unique audio seconds each embedding window contributes.

    Windows overlap (1.5 s on a 0.75 s hop), so summing ``end - start`` counts
    every second of speech about twice. That doubled a 2.2 s burst of boundary
    windows into a "4.5 s speaker" that then cleared the ``min_talk`` guard and
    survived as a phantom speaker. A window is credited only with the time
    before the next window starts (its full length when the next window is
    separated by a pause / is in another speech region).
    """
    n = len(seg_spans)
    out = np.zeros(n, dtype=np.float64)
    for i, (s, e) in enumerate(seg_spans):
        nxt = seg_spans[i + 1][0] if i + 1 < n else e
        out[i] = max(0.0, min(e, nxt) - s) if nxt >= s else 0.0
    return out


def _min_centroid_cos_dist(cents: Dict[int, np.ndarray]) -> float:
    ks = list(cents)
    if len(ks) < 2:
        return 1.0
    m = 1.0
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            m = min(m, 1.0 - float(cents[ks[i]] @ cents[ks[j]]))
    return m


def _fold_localized_churn(
    labels: np.ndarray,
    feat_matrix: np.ndarray,
    seg_spans: List[Tuple[float, float]],
    min_talk: float,
    max_footprint_frac: float = 0.22,
    max_talk_frac: float = 0.12,
) -> np.ndarray:
    """Fold "boundary-churn" clusters into their nearest real speaker.

    When the clusterer is uncertain around a single hand-off it flips back and
    forth there, minting a phantom speaker. Such a cluster has three tell-tale
    properties at once, all purely temporal (recording-independent):

      * every one of its turns is sandwiched between the SAME other speaker on
        both sides — it never actually holds the floor *between two different*
        talkers, so it is not part of a real conversational exchange;
      * its whole footprint (first turn start → last turn end) covers only a
        small fraction of the meeting timeline;
      * it owns only a small fraction of the total talk time.

    A genuine secondary speaker fails at least one test (they answer/interject
    between different people, span the meeting, or hold a real share of the
    floor), so this never removes a real participant. Clusters below
    ``min_talk`` seconds are always folded (noise / boundary slivers).
    """
    lab = np.asarray(labels).copy()
    uniq = list(dict.fromkeys(lab.tolist()))
    if len(uniq) < 2 or len(seg_spans) != len(lab):
        return lab

    starts = np.array([s for s, _ in seg_spans], dtype=float)
    ends = np.array([e for _, e in seg_spans], dtype=float)
    timeline = float(ends.max() - starts.min()) or 1.0
    eff = _effective_durations(seg_spans)          # unique seconds (windows overlap)
    total_talk = float(eff.sum()) or 1.0

    # contiguous runs (turns): list of (start_idx, end_idx) per label, plus the
    # flanking label on each side (None at the meeting boundary).
    seq_runs = []
    i, n = 0, len(lab)
    while i < n:
        j = i
        while j + 1 < n and lab[j + 1] == lab[i]:
            j += 1
        seq_runs.append([i, j, int(lab[i])])
        i = j + 1

    cents = _l2_centroids(lab, feat_matrix)

    for u in uniq:
        u_runs = [r for r in range(len(seq_runs)) if seq_runs[r][2] == u]
        if not u_runs:
            continue
        idx = np.where(lab == u)[0]
        talk = float(eff[idx].sum())
        footprint = (ends[idx].max() - starts[idx].min()) / timeline
        talk_frac = talk / total_talk

        flankers = set()
        for r in u_runs:
            flankers.add(seq_runs[r - 1][2] if r > 0 else None)
            flankers.add(seq_runs[r + 1][2] if r < len(seq_runs) - 1 else None)
        flankers.discard(None)
        sandwiched = len(flankers) == 1  # always the same single other speaker

        churn = sandwiched and footprint < max_footprint_frac and talk_frac < max_talk_frac
        if not (churn or talk < min_talk):
            continue

        others = [o for o in uniq if o != u and o in cents]
        if not others:
            continue
        # Prefer the speaker that surrounds this cluster on both sides (it is a
        # mis-split of THEIR turn); fall back to the acoustically nearest.
        flank = next(iter(flankers)) if sandwiched and flankers else None
        tgt = flank if flank in others else max(others, key=lambda o: float(cents[u] @ cents[o]))
        logger.info(
            "[Diarize] Folding boundary-churn cluster into %s "
            "(talk=%.1fs / %.0f%%, footprint=%.0f%% of timeline, sandwiched=%s).",
            "surrounding speaker" if tgt == flank else "nearest speaker",
            talk, talk_frac * 100.0, footprint * 100.0, sandwiched,
        )
        lab[lab == u] = tgt

    order = list(dict.fromkeys(lab.tolist()))
    remap = {old: i for i, old in enumerate(order)}
    return np.array([remap[x] for x in lab.tolist()], dtype=int)


def _max_contiguous_run(labels: np.ndarray) -> Dict[int, int]:
    """Longest run of consecutive windows assigned to each label.

    A real speaker holds the floor for at least one uninterrupted stretch, so
    their windows form a contiguous run. A cluster made only of isolated single
    windows scattered inside another speaker's turn (max run == 1) is an
    over-segmentation artefact, not a distinct talker.
    """
    runs: Dict[int, int] = {}
    prev_lbl = None
    cur_len = 0
    for lbl in labels.tolist():
        cur_len = cur_len + 1 if lbl == prev_lbl else 1
        prev_lbl = lbl
        if cur_len > runs.get(lbl, 0):
            runs[lbl] = cur_len
    return runs


def _postprocess_clusters(
    labels: np.ndarray,
    X: np.ndarray,
    seg_durs: np.ndarray,
    total_dur: float,
    merge_cos: float,
    min_talk: float,
    fixed_k: bool = False,
) -> Tuple[np.ndarray, int]:
    """Merge same-speaker clusters, absorb noise/mis-split clusters, relabel.

    * Two clusters whose L2 centroids are within ``merge_cos`` cosine distance
      are the SAME person under slightly different acoustics → merged. This is
      the "intelligently merge duplicate speaker IDs" requirement.
    * A cluster with < ``min_talk`` s of speech AND < 1.5 % of the talk time is
      noise or a boundary artefact → folded into its acoustically nearest
      neighbour. Genuinely distinct speakers (real centroid separation, real
      talk time) are never merged.
    """
    labels = np.asarray(labels).copy()

    if not fixed_k:
        # Merge duplicates and dissolve unsupported clusters TOGETHER, repeating
        # until stable: dissolving a sliver re-homes its windows and moves the
        # surviving centroids, which can make two of them duplicates of the same
        # voice that a one-shot "merge, then dissolve" order would leave split.
        while len(set(labels.tolist())) > 1:
            cents = _l2_centroids(labels, X)
            ks = list(cents)
            closest, cd = None, 1e9
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    d = 1.0 - float(cents[ks[i]] @ cents[ks[j]])
                    if d < cd:
                        cd, closest = d, (ks[i], ks[j])
            if closest and cd < merge_cos:
                labels[labels == closest[1]] = closest[0]
                continue

            changed = False
            durs = {lb: float(seg_durs[labels == lb].sum()) for lb in set(labels.tolist())}
            for lb in list(durs):
                if len(durs) <= 1:
                    break
                if durs[lb] < min_talk or (total_dur > 0 and durs[lb] / total_dur < 0.015):
                    # Re-home each window of the dissolved cluster on its OWN
                    # nearest surviving speaker: boundary-blend windows resemble
                    # whichever voice dominates them, so they must not all be
                    # dumped on one neighbour.
                    others = [o for o in cents if o != lb]
                    idx = np.where(labels == lb)[0]
                    sims = X[idx] @ np.stack([cents[o] for o in others]).T
                    labels[idx] = np.asarray(others)[sims.argmax(axis=1)]
                    changed = True
                    break
            if not changed:
                break

    order = list(dict.fromkeys(labels.tolist()))
    remap = {old: i for i, old in enumerate(order)}
    labels = np.array([remap[x] for x in labels.tolist()], dtype=int)
    return labels, len(set(labels.tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# Auto-tuning spectral clustering (NME-SC) — recording-independent speaker count
# ─────────────────────────────────────────────────────────────────────────────
# The speaker count is NEVER assumed or hard-coded. It is read from the
# eigen-structure of the speaker-embedding affinity graph with the Normalized
# Maximum Eigengap criterion (Park, Han, Huang & Narayanan, Interspeech 2019 —
# the same family of method NVIDIA NeMo's own diarizer uses for "auto" speaker
# count). Because it works on the *relative* structure of the affinity graph
# (which neighbour is closer than which), not on an absolute cosine-distance
# threshold (which drifts with microphone, codec, language and room), it
# generalises to any uploaded recording and resolves 1, 2, 3, 4+ speakers
# without a fixed number anywhere.
def _cosine_affinity(feat_matrix: np.ndarray) -> np.ndarray:
    """Symmetric [0,1] cosine-similarity affinity, zero diagonal."""
    X = np.asarray(feat_matrix, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(nrm, 1e-12)
    A = np.clip(X @ X.T, 0.0, 1.0)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    return A


def _prune_affinity(affinity: np.ndarray, p: int) -> np.ndarray:
    """Keep the ``p`` strongest edges per row, then symmetrise by max."""
    n = affinity.shape[0]
    p = int(max(1, min(p, n - 1)))
    A = np.array(affinity, dtype=np.float64, copy=True)
    np.fill_diagonal(A, -1.0)
    thresh = np.partition(A, n - p, axis=1)[:, n - p][:, None]
    P = np.where(A >= thresh, np.maximum(A, 0.0), 0.0)
    P = np.maximum(P, P.T)
    np.fill_diagonal(P, 0.0)
    return P


def _normalised_laplacian_eigvals(A: np.ndarray, top: int) -> np.ndarray:
    """Ascending eigenvalues of the symmetric normalised Laplacian of ``A``."""
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    L = np.eye(A.shape[0]) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    L = 0.5 * (L + L.T)
    try:
        ev = np.linalg.eigvalsh(L)
    except np.linalg.LinAlgError:  # pragma: no cover - numerical edge case
        ev = np.sort(np.linalg.eigvals(L).real)
    return ev[: max(2, int(top))]


def _estimate_num_speakers_nmesc(
    affinity: np.ndarray, min_k: int, max_k: int, max_windows: int = 1500
) -> Tuple[int, int, float]:
    """Estimate the speaker count from the affinity eigen-structure.

    Returns ``(k_est, p_star, confidence)``. ``k_est`` is clamped to
    ``[min_k, max_k]``; ``confidence`` is the normalised-eigengap margin between
    the chosen k and the runner-up (0 = ambiguous, 1 = decisive).
    """
    A_full = affinity
    n = A_full.shape[0]

    # Sub-sample rows/cols for the (O(p·n^3)) eigen search on very long meetings;
    # the final labelling still uses the full-resolution affinity.
    if n > max_windows:
        idx = np.linspace(0, n - 1, max_windows).astype(int)
        idx = np.unique(idx)
        A = A_full[np.ix_(idx, idx)]
    else:
        A = A_full
    m = A.shape[0]

    hi_k = int(min(max_k, m - 1))
    lo_k = int(max(1, min_k))
    if hi_k < 2:
        return 1, max(1, m - 1), 1.0

    p_lo = 2
    p_hi = int(max(p_lo + 1, min(m - 1, round(0.30 * m))))
    p_grid = sorted({int(round(v)) for v in np.linspace(p_lo, p_hi, min(14, p_hi - p_lo + 1))})

    best_ratio, best_k, best_p = np.inf, 1, p_grid[-1]
    per_p = []
    for p in p_grid:
        Ap = _prune_affinity(A, p)
        if not np.any(Ap):
            continue
        ev = _normalised_laplacian_eigvals(Ap, hi_k + 1)
        gaps = np.diff(ev)
        seg = gaps[lo_k - 1:hi_k]
        if seg.size == 0:
            continue
        local = int(np.argmax(seg))
        k_p = lo_k + local
        g_p = float(seg[local])
        ratio = p / max(g_p, 1e-6)
        per_p.append((ratio, k_p, p, g_p))
        if ratio < best_ratio:
            best_ratio, best_k, best_p = ratio, k_p, p

    # Confidence = how dominant the winning eigengap is vs the next-best k at p*.
    confidence = 1.0
    for (_, k_p, p, g_p) in per_p:
        if p != best_p:
            continue
        Ap = _prune_affinity(A, best_p)
        ev = _normalised_laplacian_eigvals(Ap, hi_k + 1)
        gaps = np.diff(ev)[lo_k - 1:hi_k]
        if gaps.size >= 2:
            top2 = np.sort(gaps)[::-1][:2]
            confidence = float((top2[0] - top2[1]) / (top2[0] + 1e-9))
        break

    k_est = int(np.clip(best_k, max(1, min_k), hi_k))
    return k_est, int(best_p), confidence


# NVIDIA NeMo's recommended limit for its short-recording speaker counting (offline_clustering.SpeakerClustering
# .forward_infer, ``enhanced_count_thres``: "A value of 80 is recommended").
_NEMO_ENHANCED_COUNT_MAX_WINDOWS = 80


def _estimate_num_speakers_nemo(feat_matrix: np.ndarray, min_k: int, max_k: int) -> Tuple[int, int, Optional[float]]:
    """Speaker count with NVIDIA NeMo's OWN estimator (``nemo.collections.asr.parts.utils.offline_clustering``).

    Root cause of the wrong counts: the hand-written NME-SC above differs from NeMo's reference implementation (its p
    search, eigengap normalisation and graph) and has no protection for SHORT recordings. With few, noisy 1.5 s TitaNet
    windows (same-voice cosine only ~0.34 on real audio) it split ONE real voice into 3-8 "speakers" (30-60 s clips,
    background noise). NeMo documents exactly this: "For shorter audio recordings, the clustering algorithm might not
    accumulate enough speaker profiles for each cluster" and counts them with anchor embeddings
    (``getEnhancedSpeakerCount``) - used here the same way (<= 80 windows); longer recordings use NeMo's ``NMESC``.

    Returns ``(k, p_hat, None)``; raises on any problem so the caller falls back to ``_estimate_num_speakers_nmesc``.
    """
    import torch
    from nemo.collections.asr.parts.utils import offline_clustering as nemo_oc

    emb = torch.tensor(np.asarray(feat_matrix, dtype=np.float32))
    rng_state = torch.random.get_rng_state()          # getEnhancedSpeakerCount seeds torch's global RNG: restore it
    try:
        with torch.no_grad():
            mat = nemo_oc.getCosAffinityMatrix(emb)
            nmesc = nemo_oc.NMESC(
                mat, max_num_speakers=int(max_k), max_rp_threshold=0.15, sparse_search=True, sparse_search_volume=30,
                nme_mat_size=512, fixed_thres=-1.0, parallelism=False, cuda=False,
            )
            k_nme, p_hat = nmesc.forward()
            k = int(k_nme)
            if emb.shape[0] <= _NEMO_ENHANCED_COUNT_MAX_WINDOWS:
                k = int(nemo_oc.getEnhancedSpeakerCount(emb=emb, cuda=False))
    finally:
        torch.random.set_rng_state(rng_state)
    return int(np.clip(k, max(1, min_k), max_k)), int(p_hat), None


def _spectral_labels(affinity: np.ndarray, k: int, p: Optional[int] = None) -> np.ndarray:
    """k-way spectral clustering on a precomputed affinity (optionally pruned)."""
    n = affinity.shape[0]
    k = int(max(1, min(k, n)))
    if k == 1:
        return np.zeros(n, dtype=int)
    A = _prune_affinity(affinity, p) if p else np.maximum(affinity, affinity.T)
    np.fill_diagonal(A, 0.0)
    try:
        from sklearn.cluster import SpectralClustering
        sc = SpectralClustering(
            n_clusters=k, affinity="precomputed", assign_labels="kmeans",
            random_state=0, n_init=10,
        )
        return sc.fit_predict(A).astype(int)
    except Exception as e:  # pragma: no cover - solver edge case
        logger.debug("[Diarize] SpectralClustering failed (%s); agglomerative fallback.", e)
        from sklearn.cluster import AgglomerativeClustering
        d = np.clip(1.0 - affinity, 0.0, 2.0)
        np.fill_diagonal(d, 0.0)
        return AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        ).fit_predict(d).astype(int)


def _smooth_label_sequence(
    labels: np.ndarray, seg_spans: List[Tuple[float, float]], max_flip_sec: float = 1.4
) -> np.ndarray:
    """Absorb isolated single/short label flips created by momentary embedding
    noise. A run of one label that lasts < ``max_flip_sec`` and is bordered on
    BOTH sides by the same other label is relabelled to that neighbour. Real
    speaker turns (which are longer or are followed by yet another speaker) are
    left untouched, so the speaker count is never forced.
    """
    lab = np.asarray(labels).copy()
    n = len(lab)
    if n < 3:
        return lab
    # group into contiguous runs
    runs = []
    start = 0
    for i in range(1, n):
        if lab[i] != lab[start]:
            runs.append((start, i - 1, lab[start]))
            start = i
    runs.append((start, n - 1, lab[start]))

    for r in range(1, len(runs) - 1):
        s0, s1, lv = runs[r]
        p_lv = runs[r - 1][2]
        n_lv = runs[r + 1][2]
        if p_lv != n_lv or lv == p_lv:
            continue
        dur = seg_spans[s1][1] - seg_spans[s0][0]
        if dur < max_flip_sec:
            lab[s0:s1 + 1] = p_lv
    return lab


def _multispeaker_split_supported(
    labels: np.ndarray,
    feat_matrix: np.ndarray,
    seg_durs: np.ndarray,
    is_neural: bool,
    min_talk: float,
    min_run: int,
    strong_sil: float,
    spans: Optional[List[Tuple[float, float]]] = None,
    single_turn_sep: float = 0.50,
) -> Tuple[bool, str]:
    """Final artefact filter: confirm a k>=2 partition reflects real distinct
    speakers and is not one voice fragmented by noisy short-window embeddings.

    Evidence combined (never a single hard threshold):
      * SUSTAINED clusters — how many clusters both accumulate >= ``min_talk`` s
        of speech AND hold the floor for a contiguous run of >= ``min_run``
        windows (a real turn, not scattered slivers);
      * SILHOUETTE of the partition on cosine distance (global structure);
      * CENTROID separation between the two closest clusters;
      * TURN-TAKING (needs ``spans``): when every cluster is ONE unbroken stretch
        of time (A..A then B..B - nobody ever answers back) and the centroids are
        only ``single_turn_sep`` apart, the split is what ONE voice looks like after
        the microphone, room or line changed mid-recording, so it is rejected.
        Voices that interleave, or that are clearly different people, are never
        rejected by this rule.
    Neural (TitaNet) embeddings need less corroboration than the weak numpy
    acoustic fallback, which must clear the stricter bar.
    """
    uniq = list(dict.fromkeys(labels.tolist()))
    if len(uniq) < 2:
        return False, "single cluster"

    durs = {lb: float(seg_durs[labels == lb].sum()) for lb in uniq}
    runs = _max_contiguous_run(labels)
    sustained = [lb for lb in uniq if durs[lb] >= min_talk and runs.get(lb, 0) >= min_run]

    cents = _l2_centroids(labels, feat_matrix)
    sep = _min_centroid_cos_dist(cents)

    sil = -1.0
    if 1 < len(uniq) < len(labels):
        try:
            from sklearn.metrics import silhouette_score
            d = np.clip(1.0 - np.clip(feat_matrix @ feat_matrix.T, -1.0, 1.0), 0.0, 2.0)
            np.fill_diagonal(d, 0.0)
            sil = float(silhouette_score(d, labels, metric="precomputed"))
        except Exception:
            sil = -1.0

    tag = f"sustained={len(sustained)}/{len(uniq)}, sil={sil:.2f}, sep={sep:.2f}"

    if is_neural and spans is not None and len(spans) == len(labels) and sep < single_turn_sep:
        smoothed = _smooth_label_sequence(labels, spans)
        n_runs = 1 + int(np.count_nonzero(np.diff(smoothed)))
        if n_runs <= len(uniq):
            return False, (f"{tag} (each cluster is one unbroken stretch and the voices are only moderately apart: "
                           "a change of recording conditions, not a second speaker)")

    if is_neural:
        if len(sustained) >= 2 and sep >= 0.20:
            return True, f"{tag} (two clusters hold the floor)"
        if sil >= strong_sil and sep >= 0.35:
            return True, f"{tag} (strong global structure)"
        return False, f"{tag} (insufficient — likely one speaker over-segmented)"

    # numpy acoustic fallback — phonetic content leaks into these features, so
    # demand strong, corroborated evidence before reporting >1 speaker.
    if len(sustained) >= 2 and sil >= 0.30 and sep >= 0.45:
        return True, f"{tag} (numpy strong-evidence)"
    return False, f"{tag} (numpy evidence weak -> 1 speaker)"


def _select_speaker_labels(
    feat_matrix: np.ndarray,
    seg_spans: List[Tuple[float, float]],
    num_speakers: Optional[int],
    is_neural: bool,
) -> Tuple[np.ndarray, int]:
    """
    Dynamically determine the speaker labels for the embedding windows.

    Pipeline (no fixed / hard-coded speaker count anywhere):
      1. Build a cosine-similarity affinity graph over the embedding windows.
      2. If the caller passed an explicit ``num_speakers``, honour it exactly.
      3. Otherwise estimate the count with auto-tuning spectral clustering
         (Normalized Maximum Eigengap) — recording-independent, resolves 1..N.
      4. Cut the graph into that many groups with spectral clustering.
      5. Cluster hygiene: merge duplicate-speaker centroids, absorb
         noise / boundary slivers (``_postprocess_clusters``).
      6. Artefact filter: a k>=2 result is kept only when the split has real
         multi-speaker support; a lone talker fragmented by embedding noise
         collapses back to 1.
    Falls back to agglomerative distance-threshold clustering if the eigen
    solver is unavailable.
    """
    from sklearn.cluster import AgglomerativeClustering

    try:
        from backend.config.config import settings as _cfg
    except Exception:  # pragma: no cover
        _cfg = settings

    n = len(feat_matrix)
    min_k = max(1, int(getattr(_cfg, "DIARIZATION_MIN_SPEAKERS", 1)))
    max_k_cfg = max(min_k, int(getattr(_cfg, "DIARIZATION_MAX_SPEAKERS", 8)))
    merge_cos = float(
        getattr(_cfg, "DIARIZATION_NEURAL_MERGE_COSINE", 0.30) if is_neural
        else getattr(_cfg, "DIARIZATION_CLUSTER_MERGE_COSINE", 0.50)
    )
    neural_thr = float(getattr(_cfg, "DIARIZATION_NEURAL_DIST_THRESHOLD", 0.62))
    min_talk = float(getattr(_cfg, "DIARIZATION_MIN_SPEAKER_TALK_SEC", 3.0))
    min_run = max(1, int(getattr(_cfg, "DIARIZATION_NEURAL_MIN_SUSTAINED_RUN", 2)))
    strong_sil = float(getattr(_cfg, "DIARIZATION_NEURAL_STRONG_SIL", 0.45))
    single_turn_sep = float(getattr(_cfg, "DIARIZATION_NEURAL_SINGLE_TURN_SEP", 0.50))
    nmesc_max_win = max(200, int(getattr(_cfg, "DIARIZATION_NMESC_MAX_WINDOWS", 1500)))

    # UNIQUE seconds per window (overlapping windows must not be double counted).
    seg_durs = _effective_durations(seg_spans)
    total_dur = float(seg_durs.sum()) or 1.0

    if n <= 1:
        return np.zeros(max(1, n), dtype=int), 1

    affinity = _cosine_affinity(feat_matrix)
    dist = np.clip(1.0 - affinity, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)

    # ── 2. Explicit user-specified count — honoured exactly ─────────────────
    if num_speakers is not None and int(num_speakers) >= 1:
        k = min(max(1, int(num_speakers)), n)
        logger.info(f"[Diarize] Explicit speaker count requested: {k}")
        if k == 1:
            return np.zeros(n, dtype=int), 1
        try:
            labels = _spectral_labels(affinity, k)
        except Exception:
            labels = AgglomerativeClustering(
                n_clusters=k, metric="precomputed", linkage="average"
            ).fit_predict(dist)
        return _postprocess_clusters(
            labels, feat_matrix, seg_durs, total_dur, merge_cos, min_talk, fixed_k=True
        )

    max_k = min(max_k_cfg, n - 1)
    if max_k < 2:
        return np.zeros(n, dtype=int), 1

    # ── 3. Estimate the number of speakers (NME-SC) ────────────────────────
    try:
        k_est = None
        if is_neural:   # TitaNet embeddings: NVIDIA NeMo's own estimator (see _estimate_num_speakers_nemo)
            try:
                k_est, p_star, confidence = _estimate_num_speakers_nemo(feat_matrix, min_k, max_k)
                estimator = ("NeMo enhanced count (short recording)" if n <= _NEMO_ENHANCED_COUNT_MAX_WINDOWS
                             else "NeMo NMESC")
            except Exception as nemo_err:
                logger.warning("[Diarize] NeMo speaker counting unavailable (%s: %s); using the built-in NME-SC.",
                               type(nemo_err).__name__, nemo_err)
        if k_est is None:
            k_est, p_star, confidence = _estimate_num_speakers_nmesc(
                affinity, min_k, max_k, max_windows=nmesc_max_win
            )
            estimator = "built-in NME-SC"
        logger.info(
            "[Diarize] Speaker-count estimate: k=%d (%s, p*=%d, eigengap confidence=%s, %d windows, engine=%s).",
            k_est, estimator, p_star, "n/a" if confidence is None else "%.2f" % confidence, n,
            "TitaNet" if is_neural else "numpy-acoustic",
        )
    except Exception as e:
        logger.warning(
            "[Diarize] NME-SC failed (%s: %s); falling back to agglomerative "
            "distance-threshold clustering.", type(e).__name__, e,
        )
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=neural_thr,
            metric="precomputed", linkage="average",
        ).fit_predict(dist)
        k_uncapped = len(set(labels.tolist()))
        if k_uncapped > max_k:
            labels = AgglomerativeClustering(
                n_clusters=max_k, metric="precomputed", linkage="average"
            ).fit_predict(dist)
        k_est, p_star = len(set(labels.tolist())), None

    if k_est <= 1:
        logger.info("[Diarize] Speaker-count estimate: 1 speaker (single acoustic cluster).")
        return np.zeros(n, dtype=int), 1

    # ── 4. Cut the graph into k_est groups ────────────────────────────────
    # p* was tuned on the (possibly sub-sampled) count-estimation graph; only
    # reuse it as a prune width here when the full graph is the same size.
    p_for_labels = p_star if (p_star and n <= nmesc_max_win) else None
    try:
        labels = _spectral_labels(affinity, k_est, p=p_for_labels)
    except Exception:
        labels = AgglomerativeClustering(
            n_clusters=k_est, metric="precomputed", linkage="average"
        ).fit_predict(dist)

    # ── 5. Cluster hygiene — merge duplicate speakers, absorb slivers ─────
    labels, k = _postprocess_clusters(
        labels, feat_matrix, seg_durs, total_dur, merge_cos, min_talk
    )

    # ── 5b. Fold boundary-churn clusters (short bursts bunched at one hand-off
    #        region) into the nearest real speaker — temporal shape only. ────
    if k > 1:
        labels = _fold_localized_churn(labels, feat_matrix, seg_spans, min_talk)
        k = len(set(labels.tolist()))

    # ── 6. Artefact filter — reject an unsupported multi-speaker split ────
    if k > 1:
        supported, reason = _multispeaker_split_supported(
            labels, feat_matrix, seg_durs, is_neural, min_talk, min_run, strong_sil,
            spans=seg_spans, single_turn_sep=single_turn_sep,
        )
        if not supported:
            logger.info("[Diarize] Rejected k=%d split (%s) -> 1 speaker.", k, reason)
            return np.zeros(n, dtype=int), 1
        logger.info("[Diarize] Accepted k=%d split (%s).", k, reason)

    if k <= 1:
        return np.zeros(n, dtype=int), 1

    logger.info(
        "[Diarize] Final clustering: %d speaker(s) from %d windows (%s embeddings).",
        k, n, "TitaNet" if is_neural else "numpy-acoustic",
    )
    return labels, k


def cluster_windows_to_turns(
    feat_matrix: np.ndarray,
    seg_spans: List[Tuple[float, float]],
    num_speakers: Optional[int],
    is_neural: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    """Speaker-embedding windows -> merged speaker turns (``{start, end, speaker}``) and the speaker count.

    Everything after the embeddings are extracted: dynamic speaker count and
    clustering (:func:`_select_speaker_labels`), single-window flip smoothing,
    chronological ``Speaker_N`` naming and merging of adjacent same-speaker
    windows. Kept separate from audio I/O so it can be exercised directly on
    stored embeddings.
    """
    labels, _best_k = _select_speaker_labels(feat_matrix, seg_spans, num_speakers, is_neural)

    # Absorb isolated single-window label flips from momentary embedding noise,
    # then relabel to a contiguous 0..k-1 range.
    labels = _smooth_label_sequence(labels, seg_spans)
    order = list(dict.fromkeys(np.asarray(labels).tolist()))
    remap = {old: i for i, old in enumerate(order)}
    labels = np.array([remap[x] for x in np.asarray(labels).tolist()], dtype=int)
    best_k = len(order)

    prec = diar_cfg.TIMESTAMP_PRECISION
    speaker_map = {c: f"Speaker_{c + 1}" for c in range(best_k)}
    turns = [
        {"start": round(s, prec), "end": round(e, prec), "speaker": speaker_map[lbl]}
        for (s, e), lbl in zip(seg_spans, labels)
    ]

    merged_turns: List[Dict[str, Any]] = []
    for t in turns:
        if merged_turns and merged_turns[-1]["speaker"] == t["speaker"]:
            if t["start"] - merged_turns[-1]["end"] <= 0.8:
                merged_turns[-1]["end"] = t["end"]
                continue
        merged_turns.append(dict(t))
    return merged_turns, best_k


_FALLBACK_NOTE = (
    "Speakers were separated with the reduced-accuracy acoustic fallback because the neural speaker model is not "
    "available; the speaker count and speaking times may be wrong."
)


def run_nemo_diarization(
    audio_filepath: str, num_speakers: Optional[int] = None
) -> Tuple[Optional[List[Dict[str, Any]]], str, Optional[str]]:
    """
    Runs NVIDIA NeMo Speaker Diarization on standardized WAV audio.

    Returns:
        (speaker_turns, status_string, error_message_or_None)
        status_string is one of: "SUCCESS", "FAILED", "DEGRADED"

    Preserves short speakers (80/10/10 duration splits) without single-speaker collapse.
    """
    logger.info(f"[NVIDIA NeMo] Starting speaker diarization for: {audio_filepath}")

    # ── Decide up-front whether to hold the whole PCM array or stream it ──────
    # For long meetings, loading the entire 16 kHz float32 array is the single
    # largest allocation in the whole pipeline (≈115 MB / 30 min, ≈1.1 GB / 5 h).
    # Probe duration cheaply with soundfile.info and stream past the threshold.
    stream_mode = False
    probe_sr = None
    probe_dur = 0.0
    try:
        import soundfile as _sf
        _info = _sf.info(audio_filepath)
        probe_sr = int(_info.samplerate)
        probe_dur = float(_info.frames) / probe_sr if probe_sr else 0.0
        stream_mode = probe_dur > _STREAM_THRESHOLD_SEC
    except Exception:
        stream_mode = False

    if stream_mode:
        # Validate format on a short prefix only — never materialize the full file.
        val = validate_and_normalize_audio(audio_filepath, max_seconds=30.0)
        if not val["valid"]:
            logger.error(f"[NVIDIA NeMo] Audio validation failed: {val['error']}")
            return None, "FAILED", val["error"]
        sr = probe_sr or val["sample_rate"]
        audio = None
        dur_sec = probe_dur or val["duration"]
        logger.info(
            "[NVIDIA NeMo] Streaming mode: audio %.0fs > %.0fs threshold — reading "
            "windows from disk (no full-file load).", dur_sec, _STREAM_THRESHOLD_SEC,
        )
    else:
        val = validate_and_normalize_audio(audio_filepath)
        if not val["valid"]:
            logger.error(f"[NVIDIA NeMo] Audio validation failed: {val['error']}")
            return None, "FAILED", val["error"]
        sr = val["sample_rate"]
        audio = val["audio"]
        dur_sec = val["duration"]

    device_str = _get_nemo_device()
    logger.info(
        f"[NVIDIA NeMo] Processing audio duration: {dur_sec:.1f}s | "
        f"Device: {device_str.upper()} | Engine: NVIDIA NeMo"
    )

    try:
        speaker_model = _get_nemo_speaker_model()
        is_neural = speaker_model is not None and speaker_model != "NEURAL_ACOUSTIC_EMBEDDING"
        logger.info(
            "[NVIDIA NeMo] Embedding engine: %s",
            "TitaNet neural speaker embeddings" if is_neural
            else "numpy acoustic voiceprint (fallback)",
        )

        # Step 1: Adaptive energy VAD — window ONLY inside detected speech.
        # This replaces the old "tile the whole timeline" behaviour that fed
        # silence / music / noise into the clusterer as phantom speakers.
        speech_regions = _adaptive_energy_vad(audio, audio_filepath, sr, dur_sec, **_vad_settings())
        speech_dur = sum(e - s for s, e in speech_regions) or dur_sec

        # Window geometry: 1.5 s analysis window on a 0.75 s hop (50 % overlap).
        # TitaNet is trained on ~1.5-3 s of speech; the dense overlap gives the
        # clusterer a smoother embedding trajectory (fewer noisy single-window
        # outliers) and finer speaker-change resolution. Step widened adaptively
        # for long meetings so the window count (and the O(n^2) affinity matrix)
        # stays bounded by DIARIZATION_MAX_EMBED_SEGMENTS.
        overlap_dur = 0.75
        base_step = 0.75
        step_dur = base_step
        if speech_dur / base_step > _MAX_EMBED_SEGMENTS:
            step_dur = speech_dur / _MAX_EMBED_SEGMENTS
            logger.info(
                "[NVIDIA NeMo] Long audio (%.0fs speech): widening window step "
                "%.2fs -> %.2fs to cap embeddings at %d.",
                speech_dur, base_step, step_dur, _MAX_EMBED_SEGMENTS,
            )
        chunk_dur = step_dur + overlap_dur

        raw_turns = []
        for r_start, r_end in speech_regions:
            cur_t = r_start
            while cur_t < r_end - 0.05:
                t_end = min(r_end, cur_t + chunk_dur)
                if t_end - cur_t >= 0.4:
                    raw_turns.append((round(cur_t, 3), round(t_end, 3)))
                cur_t += step_dur

        if not raw_turns:
            logger.warning("[NVIDIA NeMo] No speech segments detected in audio file.")
            return None, "DEGRADED", "No active speech segments detected"

        # Step 2: Extract NeMo speaker embeddings per segment.
        # Streaming path: read each window from disk with a single SoundFile
        # handle (seek + bounded read) so the whole PCM array is never resident.
        valid_embeddings = []
        valid_segments = []

        if audio is None:
            import soundfile as _sf
            with _sf.SoundFile(audio_filepath) as _sfh:
                file_frames = len(_sfh)
                for start_s, end_s in raw_turns:
                    start_idx = max(0, int(start_s * sr))
                    n = int((end_s - start_s) * sr)
                    if n < int(sr * 0.15) or start_idx >= file_frames:
                        continue
                    try:
                        _sfh.seek(start_idx)
                        seg = _sfh.read(n, dtype="float32")
                    except Exception:
                        continue
                    if seg.ndim > 1:
                        seg = np.mean(seg, axis=1)
                    emb = _embed_segment_array(seg, sr, speaker_model)
                    if emb is not None:
                        valid_embeddings.append(emb)
                        valid_segments.append((start_s, end_s))
        else:
            for start_s, end_s in raw_turns:
                emb = _extract_nemo_segment_embedding(audio, sr, start_s, end_s, speaker_model)
                if emb is not None:
                    valid_embeddings.append(emb)
                    valid_segments.append((start_s, end_s))
            # Full PCM array no longer needed once embeddings are extracted.
            audio = None
            val = None
            gc.collect()

        n_valid = len(valid_embeddings)
        if n_valid == 0:
            logger.warning("[NVIDIA NeMo] Failed to extract valid speaker embeddings.")
            return None, "DEGRADED", "Failed to extract valid speaker embeddings"

        if n_valid == 1:
            logger.info("[NVIDIA NeMo] Single speech window detected -> 1 speaker.")
            single_turn = [{
                "start": round(valid_segments[0][0], 3),
                "end": round(valid_segments[0][1], 3),
                "speaker": "Speaker_1",
            }]
            return single_turn, ("SUCCESS" if is_neural else "DEGRADED"), (None if is_neural else _FALLBACK_NOTE)

        # Step 3+4: Dynamic speaker-count estimation (NME-SC auto-tuning spectral
        # clustering) + cluster hygiene (duplicate-speaker merge, noise-cluster
        # absorption) + multi-speaker artefact filter. No hard-coded count.
        feat_matrix = np.asarray(valid_embeddings, dtype=np.float32)
        valid_embeddings = None
        merged_turns, _k = cluster_windows_to_turns(
            feat_matrix, valid_segments, num_speakers, is_neural
        )
        feat_matrix = None
        gc.collect()

        spk_counts = len({t["speaker"] for t in merged_turns})
        logger.info(
            f"[NVIDIA NeMo] Diarization complete: {spk_counts} unique speaker(s), "
            f"{len(merged_turns)} speech turns."
        )

        if not is_neural:
            # The numpy acoustic voiceprint is a stand-in with much lower accuracy (measured: it merges or splits
            # voices on short / noisy audio). Its speakers are kept, but never reported as a full-quality SUCCESS.
            logger.warning("[NVIDIA NeMo] %s", _FALLBACK_NOTE)
            return merged_turns, "DEGRADED", _FALLBACK_NOTE
        return merged_turns, "SUCCESS", None

    except Exception as e:
        logger.error(f"[NVIDIA NeMo] Diarization failed: {type(e).__name__}: {e}", exc_info=True)
        return None, "FAILED", f"NeMo diarization error: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# Shared speaker-embedding utility (voice enrollment + meeting-time matching)
# ─────────────────────────────────────────────────────────────────────────────
def compute_file_speaker_embedding(
    audio_filepath: str,
    time_ranges: Optional[List[Tuple[float, float]]] = None,
    min_speech_sec: float = 2.0,
) -> Dict[str, Any]:
    """Mean speaker embedding over the speech in a file (optionally restricted
    to ``time_ranges``).

    Uses the SAME embedder (TitaNet when available, numpy acoustic voiceprint
    otherwise), the SAME energy VAD and the SAME 1.5 s / 0.75 s window geometry
    as diarization, so a vector produced at enrollment time is directly
    comparable (cosine) to one produced from a diarized meeting speaker.

    Returns a dict:
      ``{"ok": bool, "embedding": list[float], "dim": int, "engine": str,
         "speech_sec": float, "n_windows": int, "error": str | None}``
    """
    result: Dict[str, Any] = {
        "ok": False, "embedding": None, "dim": 0, "engine": None,
        "speech_sec": 0.0, "n_windows": 0, "error": None,
    }
    try:
        val = validate_and_normalize_audio(audio_filepath)
        if not val.get("valid"):
            result["error"] = val.get("error", "invalid audio")
            return result
        audio = val["audio"]
        sr = int(val["sample_rate"])
        dur = float(val["duration"])

        model = _get_nemo_speaker_model()
        is_neural = model is not None and model != "NEURAL_ACOUSTIC_EMBEDDING"
        result["engine"] = "titanet_large" if is_neural else "numpy_acoustic_voiceprint"

        speech_regions = _adaptive_energy_vad(audio, audio_filepath, sr, dur, **_vad_settings())

        # Intersect VAD speech with any caller-supplied ranges (a diarized
        # speaker's turns). Empty -> use all detected speech.
        if time_ranges:
            clipped: List[Tuple[float, float]] = []
            for rs, re_ in speech_regions:
                for ts, te in time_ranges:
                    lo, hi = max(rs, float(ts)), min(re_, float(te))
                    if hi - lo >= 0.30:
                        clipped.append((lo, hi))
            regions = clipped or [
                (max(0.0, float(ts)), min(dur, float(te)))
                for ts, te in time_ranges if float(te) - float(ts) >= 0.30
            ]
        else:
            regions = speech_regions

        speech_sec = float(sum(e - s for s, e in regions))
        result["speech_sec"] = round(speech_sec, 2)
        if speech_sec < max(0.5, float(min_speech_sec)):
            result["error"] = (
                f"insufficient speech ({speech_sec:.1f}s < {min_speech_sec:.1f}s)"
            )
            return result

        step, chunk = 0.75, 1.5
        embs = []
        for r_start, r_end in regions:
            t = r_start
            while t < r_end - 0.05:
                t_end = min(r_end, t + chunk)
                if t_end - t >= 0.4:
                    a0, a1 = int(t * sr), int(t_end * sr)
                    emb = _embed_segment_array(audio[a0:a1], sr, model)
                    if emb is not None:
                        embs.append(emb)
                t += step

        if not embs:
            result["error"] = "no valid embedding windows"
            return result

        M = np.asarray(embs, dtype=np.float64)
        mean_vec = M.mean(axis=0)
        nrm = float(np.linalg.norm(mean_vec))
        if nrm < 1e-8:
            result["error"] = "degenerate embedding"
            return result
        mean_vec = mean_vec / nrm

        result.update(
            ok=True,
            embedding=[float(x) for x in mean_vec.tolist()],
            dim=int(mean_vec.shape[0]),
            n_windows=len(embs),
        )
        return result
    except Exception as e:  # pragma: no cover - defensive
        logger.error("[VoiceEmbedding] failed for %s: %s", audio_filepath, e, exc_info=True)
        result["error"] = f"{type(e).__name__}: {e}"
        return result


def cosine_similarity(a: Any, b: Any) -> float:
    """Cosine similarity between two vectors given as lists / arrays."""
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    if va.size == 0 or vb.size != va.size:
        return 0.0
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp Alignment (ASR Words to NeMo Speaker Turns)
# ─────────────────────────────────────────────────────────────────────────────
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*$")


def _smooth_speakers_within_sentences(
    words: list,
    speakers: List[str],
    max_fragment_sec: float = 2.0,
    max_fragment_frac: float = 0.4,
    min_sentence_words: int = 6,
) -> List[str]:
    """Keep one sentence with one speaker unless a real change of speaker is evident.

    Word times are only as good as the speech-to-text that produced them: when the response carries no word-level times
    the words are spread evenly over their segment, and even real word times drift by a second near segment ends. Words
    at the edge of a turn then land on the neighbouring speaker. Measured end to end: "...decided to move production |
    deployment to October 15th" was cut in two, the tail credited to the next speaker.

    Only a SHORT FRAGMENT AT THE EDGE of a sentence (at most ``max_fragment_sec`` and ``max_fragment_frac`` of it) whose
    speaker differs from the rest is given to the sentence's main speaker. A sentence is a run of words up to one that
    ends in . ! ? so a genuine reply ("Yes.") is a sentence of its own and never touched, and only sentences of
    ``min_sentence_words`` or more words are considered.
    """
    n = len(words)
    out = list(speakers)
    if n < min_sentence_words:
        return out
    i = 0
    while i < n:
        j = i
        while j < n - 1 and not _SENTENCE_END.search(str(words[j].get("word", "")).strip()):
            j += 1
        size = j - i + 1
        if size >= min_sentence_words and len(set(out[i:j + 1])) > 1:
            counts: Dict[str, int] = {}
            for s in out[i:j + 1]:
                counts[s] = counts.get(s, 0) + 1
            main = max(counts, key=counts.get)
            # the run of a different speaker at each end of the sentence
            a = i
            while a <= j and out[a] != main:
                a += 1
            b = j
            while b >= i and out[b] != main:
                b -= 1
            for lo, hi in ((i, a - 1), (b + 1, j)):
                if hi < lo:
                    continue
                frag = hi - lo + 1
                span = float(words[hi].get("end", 0.0)) - float(words[lo].get("start", 0.0))
                if frag <= max_fragment_frac * size and span <= max_fragment_sec:
                    out[lo:hi + 1] = [main] * frag
        i = j + 1
    return out


_SENTENCE_END_RE = re.compile(r"""[.!?…।。](?:["')\]]+)?$""")


def _split_run_into_pieces(run: list, max_sec: float, max_words: int) -> List[list]:
    """Cut one same-speaker run of timed words into pieces of bounded length, only at sentence ends.

    A piece closes at the first sentence end after it reaches ``max_sec`` seconds or ``max_words`` words; punctuation-
    free ASR output (no sentence end in sight) is cut at twice that, so a segment can never grow without bound. Every
    word lands in exactly one piece, in order: the text is the same, only where it is cut changes. A short tail joins
    the previous piece instead of becoming a two-word segment.
    """
    pieces: List[list] = []
    cur: list = []
    for w in run:
        cur.append(w)
        span = float(cur[-1]["end"]) - float(cur[0]["start"])
        over = span >= max_sec or len(cur) >= max_words
        hard = span >= 2 * max_sec or len(cur) >= 2 * max_words
        if (over and _SENTENCE_END_RE.search(str(w["word"]).strip())) or hard:
            pieces.append(cur)
            cur = []
    if cur:
        if pieces and len(cur) < max(5, max_words // 10):
            pieces[-1].extend(cur)
        else:
            pieces.append(cur)
    return pieces


def align_words_to_nemo(words: list, nemo_turns: list) -> list:
    """
    Aligns word-level timestamps to NeMo speaker turns based on maximum temporal overlap.
    Falls back to segment-level maximum overlap if word timestamps are missing.
    """
    final_segments = []
    if not words:
        return final_segments

    max_sec = max(5.0, float(getattr(settings, "TRANSCRIPT_MAX_SEGMENT_SEC", 45.0)))
    max_words = max(20, int(getattr(settings, "TRANSCRIPT_MAX_SEGMENT_WORDS", 120)))

    def _segments_of(speaker: str, run: list) -> list:
        out = []
        for k, piece in enumerate(_split_run_into_pieces(run, max_sec, max_words)):
            out.append({
                "speaker": speaker,
                "start_time": f"{piece[0]['start']:.3f}",
                "end_time": f"{piece[-1]['end']:.3f}",
                "text": " ".join(w["word"] for w in piece).strip(),
                "_hard_break": k > 0,       # a length cut: the merge below must not glue it back
            })
        return out

    if not nemo_turns:
        # Single unassigned turn fallback when nemo_turns is empty
        segments = _segments_of("Speaker_1", words)
        for seg in segments:
            seg.pop("_hard_break", None)
        return segments

    def find_best_speaker(start: float, end: float) -> str:
        mid = (start + end) / 2.0
        # Priority 1: midpoint containment
        for turn in nemo_turns:
            if turn["start"] <= mid <= turn["end"]:
                return turn["speaker"]

        # Priority 2: maximum temporal overlap duration
        best_spk = None
        max_overlap = -1.0
        for turn in nemo_turns:
            overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
            if overlap > max_overlap:
                max_overlap = overlap
                best_spk = turn["speaker"]

        if best_spk and max_overlap > 0:
            return best_spk

        # Priority 3: nearest turn in time
        nearest_turn = min(
            nemo_turns, key=lambda t: min(abs(t["start"] - mid), abs(t["end"] - mid))
        )
        return nearest_turn["speaker"]

    speakers = [find_best_speaker(w["start"], w["end"]) for w in words]
    if getattr(settings, "DIARIZATION_SENTENCE_SMOOTHING", True):
        speakers = _smooth_speakers_within_sentences(words, speakers)

    # One run per stretch of consecutive words with the same speaker, then each run cut into bounded-length segments
    # at sentence ends (a one-speaker meeting is a single run - 73 diarization turns used to become 1 segment).
    runs: list = []
    for w, spk in zip(words, speakers):
        if runs and runs[-1][0] == spk:
            runs[-1][1].append(w)
        else:
            runs.append((spk, [w]))
    for spk, run in runs:
        final_segments.extend(_segments_of(spk, run))

    # Merge consecutive turns belonging to the same speaker
    merged = []
    for seg in final_segments:
        try:
            s_t = max(0.0, float(seg["start_time"]))
            e_t = max(s_t, float(seg["end_time"]))
            seg["start_time"] = f"{s_t:.3f}"
            seg["end_time"] = f"{e_t:.3f}"
        except Exception:
            continue

        if not seg.get("text", "").strip():
            continue

        if (
            merged
            and merged[-1]["speaker"] == seg["speaker"]
            and not seg.get("_hard_break")
            and float(seg["start_time"]) - float(merged[-1]["end_time"]) < diar_cfg.ALIGN_MERGE_GAP
        ):
            merged[-1]["end_time"] = seg["end_time"]
            merged[-1]["text"] = (merged[-1]["text"] + " " + seg["text"]).strip()
        else:
            merged.append(seg)

    for seg in merged:
        seg.pop("_hard_break", None)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Speaker Label Normalization
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_speaker_labels(segments: list) -> list:
    """
    Standardizes speaker labels chronologically across all segments:
      First speaker in meeting  → Speaker_1
      Second speaker            → Speaker_2
      Third speaker             → Speaker_3
    Prevents label flipping and guarantees clean, consistent IDs.
    """
    if not segments:
        return segments

    speaker_map = {}
    speaker_order = []

    for seg in segments:
        raw_spk = str(seg.get("speaker", "Speaker_1"))
        if raw_spk not in speaker_map:
            # Map legacy SPEAKER_00 / SPK_X to Speaker_1, Speaker_2...
            clean_id = f"Speaker_{len(speaker_order) + 1}"
            speaker_map[raw_spk] = clean_id
            speaker_order.append(raw_spk)

    result = []
    for seg in segments:
        new_seg = dict(seg)
        raw_spk = str(seg.get("speaker", "Speaker_1"))
        new_seg["speaker"] = speaker_map.get(raw_spk, "Speaker_1")
        result.append(new_seg)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# LLM Speaker Label Resolution (Optional Named Identity Mapping)
# ─────────────────────────────────────────────────────────────────────────────
def resolve_speaker_labels_with_llm(segments: list, locked_labels: Optional[set] = None) -> list:
    """
    Optionally infers real speaker names/roles (e.g. 'Dr. Sharma', 'Project Manager')
    from transcript content using the LLM. Only replaces generic Speaker_X labels
    when explicit identity clues exist.

    ``locked_labels`` — speaker labels already resolved by voice matching; the LLM
    is not allowed to rename these (verified identity beats a transcript guess).
    Segments touched only by the LLM keep ``speaker_name = None`` and are tagged
    ``name_source = "transcript_llm"`` so a guess is never presented as verified.
    """
    if not segments:
        return segments

    enable_llm = getattr(settings, "ENABLE_LLM_SPEAKER_LABELS", True)
    if not enable_llm:
        return segments

    # No live "ping" first: the reachability probe sat on the critical path of every meeting and, when the gateway
    # left the ping unanswered, cost 30 s per probe attempt (62 s measured) before this one short call, which has its
    # own retries and falls back to the generic labels on any failure (the except below).
    from backend.services.model_router import model_router
    from backend.providers import provider_factory
    try:
        active_p = provider_factory.get_active_provider()
        has_ai = active_p.is_available()
    except Exception:
        has_ai = bool((model_router._get_dynamic_config().get("airawat", {}).get("api_key") or settings.AIRAWAT_API_KEY or "").strip())
    if not has_ai:
        return segments

    locked = set(locked_labels or set())
    all_speakers = [s for s in {seg["speaker"] for seg in segments} if s not in locked]
    if len(all_speakers) < 2:
        return segments

    sample_lines = []
    for seg in segments[:30]:
        sample_lines.append(f"[{seg['speaker']}]: {seg['text'][:120]}")
    sample_text = "\n".join(sample_lines)

    try:
        import json
        from backend.services.model_router import model_router
        from langchain_core.prompts import ChatPromptTemplate

        llm = model_router.get_llm(temperature=0.0)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You analyze meeting transcripts to map generic speaker placeholders to real names or roles if clearly stated.\n"
                    "Rules:\n"
                    "- If a person's name is introduced, map Speaker_1 -> Name\n"
                    "- If no name is discernible, keep the original Speaker_X label\n"
                    "- Output names/roles in ENGLISH only; transliterate names written in a non-Latin script.\n"
                    'Return ONLY JSON mapping. Example: {{"Speaker_1": "Dr. Smith", "Speaker_2": "Speaker_2"}}',
                ),
                ("user", "Speakers: {speakers}\n\nTranscript sample:\n{sample}"),
            ]
        )

        res = (prompt | llm).invoke({"speakers": ", ".join(all_speakers), "sample": sample_text})
        content = res.content.strip()
        if "```" in content:
            content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```")).strip()

        label_map = json.loads(content)
        resolved = []
        for seg in segments:
            new_seg = dict(seg)
            orig = seg["speaker"]
            if orig in locked:
                resolved.append(new_seg)
                continue
            mapped = label_map.get(orig, orig)
            if mapped != orig and not str(mapped).startswith("Speaker_"):
                new_seg["speaker"] = mapped
                new_seg["label"] = mapped
                new_seg.setdefault("speaker_name", None)  # LLM guess is not verified identity
                new_seg["name_source"] = "transcript_llm"
            resolved.append(new_seg)
        return resolved

    except Exception as e:
        logger.debug(f"[LLM Speaker Resolution] Skipped ({e}). Keeping generic labels.")
        return segments


# ─────────────────────────────────────────────────────────────────────────────
# Post-alignment speaker reconciliation
# ─────────────────────────────────────────────────────────────────────────────
def _reconcile_aligned_speakers(segments: list) -> list:
    """
    Remove spurious single-speaker artefacts introduced by word→turn alignment
    without ever forcing a fixed speaker count.

    Two safe, evidence-based operations:
      1. A segment whose speaker's TOTAL talk time is negligible (< 1.0 s and
         < 1 % of all talk time) AND that is bordered on both sides by the SAME
         other speaker is a mis-flip → reassigned to that neighbour. This fixes
         "labels incorrectly split" without merging real speakers.
      2. After reassignment, adjacent same-speaker segments are stitched back
         together so turn counts stay honest.

    A speaker with real, distributed talk time is never removed, so genuine
    2/3/4+ speaker meetings are preserved.
    """
    if not segments or len(segments) < 3:
        return segments

    def _dur(s):
        try:
            return max(0.0, float(s["end_time"]) - float(s["start_time"]))
        except Exception:
            return 0.0

    totals: Dict[str, float] = {}
    for s in segments:
        totals[s["speaker"]] = totals.get(s["speaker"], 0.0) + _dur(s)
    grand = sum(totals.values()) or 1.0

    if len(totals) < 2:
        return segments

    out = [dict(s) for s in segments]
    for i in range(1, len(out) - 1):
        spk = out[i]["speaker"]
        prev_spk, next_spk = out[i - 1]["speaker"], out[i + 1]["speaker"]
        if spk == prev_spk or spk != out[i]["speaker"]:
            continue
        if prev_spk == next_spk and spk != prev_spk:
            if totals.get(spk, 0.0) < 1.0 and totals.get(spk, 0.0) / grand < 0.01:
                logger.info(
                    "[Diarize] Reassigning spurious %s sliver (%.2fs total) to "
                    "surrounding speaker %s.", spk, totals.get(spk, 0.0), prev_spk,
                )
                totals[prev_spk] = totals.get(prev_spk, 0.0) + totals.pop(spk, 0.0)
                out[i]["speaker"] = prev_spk

    stitched: list = []
    for s in out:
        if stitched and stitched[-1]["speaker"] == s["speaker"]:
            stitched[-1]["end_time"] = s["end_time"]
            stitched[-1]["text"] = (stitched[-1].get("text", "") + " " + s.get("text", "")).strip()
        else:
            stitched.append(dict(s))
    return stitched


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Speaker Statistics & Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def _merge_intervals(intervals) -> List[Tuple[float, float]]:
    """Sorted, non-overlapping union of ``(start, end)`` intervals."""
    merged: List[Tuple[float, float]] = []
    for s, e in sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a)):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def speech_seconds_per_speaker(segments: list, speech_regions) -> Dict[str, float]:
    """Seconds of detected SPEECH inside each speaker's transcript segments.

    A transcript segment's ``end - start`` spans every pause inside it: speech-to-text spreads the words over the whole
    segment, and adjacent same-speaker pieces are stitched together. Measured against an independent neural VAD, 22-39 %
    of what was reported as a speaker's talk time was silence. Here only the part of each segment that lies inside a
    speech region is counted, so a speaker's duration is their share of the valid speech of the meeting.
    """
    import bisect

    merged = _merge_intervals(speech_regions or [])
    starts = [s for s, _ in merged]
    out: Dict[str, float] = {}
    for seg in segments:
        spk = str(seg.get("speaker", "Speaker_1"))
        out.setdefault(spk, 0.0)
        try:
            s0, e0 = float(seg.get("start_time", 0.0)), float(seg.get("end_time", 0.0))
        except (TypeError, ValueError):
            continue
        i = max(0, bisect.bisect_right(starts, s0) - 1)
        while i < len(merged) and merged[i][0] < e0:
            out[spk] += max(0.0, min(e0, merged[i][1]) - max(s0, merged[i][0]))
            i += 1
    return out


def speech_regions_for_durations(audio_filepath: str, turns: Optional[list] = None) -> List[Tuple[float, float]]:
    """The speech regions a speaker's talk time is measured on: the diarization's own energy VAD over the file (same
    function and settings, read block-wise so long files are never held in memory). Falls back to the union of the
    diarization turns (speech-only by construction) when the audio cannot be read. Empty when neither is available."""
    try:
        import soundfile as _sf

        info = _sf.info(audio_filepath)
        dur = float(info.frames) / info.samplerate if info.samplerate else 0.0
        if dur > 0:
            return _adaptive_energy_vad(None, audio_filepath, int(info.samplerate), dur, **_vad_settings())
    except Exception as e:
        logger.info("[Diarize] VAD for speaker durations unavailable (%s: %s); using the diarization turns.",
                    type(e).__name__, e)
    return _merge_intervals((t.get("start", 0.0), t.get("end", 0.0)) for t in (turns or []))


def stored_speech_durations(diarization_info_json: Any) -> Optional[Dict[str, float]]:
    """Per-speaker talk time the diarization stored with the meeting (``diagnostics.speaker_breakdown``), or None."""
    try:
        info = json.loads(diarization_info_json) if isinstance(diarization_info_json, str) else (diarization_info_json or {})
        breakdown = (info.get("diagnostics") or {}).get("speaker_breakdown") or {}
        durs = {str(k): float(v["speaking_duration_seconds"]) for k, v in breakdown.items()
                if isinstance(v, dict) and v.get("speaking_duration_seconds") is not None}
        return durs or None
    except Exception:
        return None


def compute_speaker_diagnostics(
    segments: list,
    total_meeting_duration: float = 0.0,
    identity_map: Optional[Dict[str, Dict[str, Any]]] = None,
    speech_durations: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Computes exact, empirical speaker analytics dynamically from transcript segments.
    Calculates speaking duration, percentage, turn count, avg turn duration, first/last appearance.
    Formula:
        speaker_percentage = (speaker_speaking_duration / total_detected_speech_duration) * 100
    No hardcoded values, fake metrics, or dummy estimates.

    ``identity_map`` (optional): ``{display_label: {"speaker_id", "speaker_name",
    "label", "confidence"}}`` from voice matching, merged into each
    ``speaker_breakdown`` entry so the per-speaker block carries a stable
    SPEAKER_NN id and (when known) the registered name + confidence.

    ``speech_durations`` (optional): seconds of detected speech per speaker (``speech_seconds_per_speaker``). When it
    covers exactly the speakers of ``segments`` it is the speaking duration; otherwise (e.g. a legacy meeting) the
    segment spans are used as before.
    """
    identity_map = identity_map or {}
    if not segments:
        return {
            "speaker_count": 0,
            "total_turns": 0,
            "total_speech_duration_sec": 0.0,
            "speaker_switches": 0,
            "avg_turn_duration_sec": 0.0,
            "speaker_breakdown": {},
        }

    speaker_durations: Dict[str, float] = {}
    speaker_turns: Dict[str, int] = {}
    first_appearances: Dict[str, float] = {}
    last_appearances: Dict[str, float] = {}

    speaker_segments: Dict[str, int] = {}
    total_speech_dur = 0.0
    timeline: List[Tuple[float, float, str]] = []

    for seg in segments:
        spk = str(seg.get("speaker", "Speaker_1"))
        try:
            start_t = float(seg.get("start_time", 0.0))
            end_t = float(seg.get("end_time", 0.0))
            dur = max(0.0, end_t - start_t)
        except Exception:
            start_t, end_t, dur = 0.0, 0.0, 0.0

        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur
        speaker_segments[spk] = speaker_segments.get(spk, 0) + 1
        total_speech_dur += dur
        timeline.append((start_t, end_t, spk))

        if spk not in first_appearances:
            first_appearances[spk] = start_t
        last_appearances[spk] = max(last_appearances.get(spk, 0.0), end_t)

    # A TURN is a run of consecutive segments by one speaker. A monologue cut into 27 length-bounded transcript
    # segments is ONE turn with 27 segments (counting segments as turns made this 27 here and 0 in runtime_metrics).
    from backend.utils.helpers import speaker_turns as _speaker_turns

    turn_list = _speaker_turns(timeline)
    for turn in turn_list:
        speaker_turns[turn["speaker"]] = speaker_turns.get(turn["speaker"], 0) + 1
    speaker_switches = max(0, len(turn_list) - 1)

    speech_source = "segment_spans"          # includes the pauses inside a segment
    if speech_durations and set(speech_durations) == set(speaker_durations):
        speaker_durations = {spk: max(0.0, float(speech_durations[spk])) for spk in speaker_durations}
        total_speech_dur = sum(speaker_durations.values())
        speech_source = "vad_regions"        # seconds of detected speech only

    # Stable SPEAKER_NN ids in order of first appearance (used when no richer
    # identity_map is supplied, e.g. legacy callers).
    _order = sorted(first_appearances, key=lambda s: first_appearances[s])
    _auto_id = {s: f"SPEAKER_{i:02d}" for i, s in enumerate(_order)}
    # identity_map is keyed by display label; also index by the raw diar label
    # so an unnamed speaker still resolves its "Speaker N" display + SPEAKER_NN id.
    _ident_by_diar = {
        v["diar_label"]: v for v in identity_map.values() if v.get("diar_label")
    }

    speaker_breakdown = {}
    for spk, dur in speaker_durations.items():
        turns = speaker_turns.get(spk, 0)
        ratio_pct = round((dur / total_speech_dur * 100.0), 1) if total_speech_dur > 0 else 0.0
        ident = identity_map.get(spk) or _ident_by_diar.get(spk, {})
        speaker_breakdown[spk] = {
            "speaker_id": ident.get("speaker_id") or _auto_id.get(spk, spk),
            "speaker_name": ident.get("speaker_name"),
            "label": ident.get("label") or spk,
            "confidence": ident.get("confidence"),
            "speaking_duration_seconds": round(dur, 2),
            "speaking_percentage": ratio_pct,
            "turn_count": turns,
            "segment_count": speaker_segments.get(spk, 0),
            "avg_turn_duration_sec": round(dur / turns, 2) if turns > 0 else 0.0,
            "first_appearance": round(first_appearances.get(spk, 0.0), 2),
            "last_appearance": round(last_appearances.get(spk, 0.0), 2),
        }

    return {
        "speaker_count": len(speaker_durations),
        # turns = runs of one speaker (see above); segments = transcript segments; the diarization model's own raw
        # turn count is diarization_metadata_json.validation_diagnostics.segment_count.
        "total_turns": len(turn_list),
        "total_segments": len(segments),
        "total_speech_duration_sec": round(total_speech_dur, 2),
        # "vad_regions": detected speech only; "segment_spans": the spans of the segments (pauses included).
        "speech_source": speech_source,
        "speaker_switches": speaker_switches,
        "avg_turn_duration_sec": round(total_speech_dur / len(turn_list), 2) if turn_list else 0.0,
        "speaker_breakdown": speaker_breakdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Voice identity assignment (diarized speaker  ->  registered name / Speaker N)
# ─────────────────────────────────────────────────────────────────────────────
def _friendly_label(diar_label: str, idx: int) -> str:
    """'Speaker_1' -> 'Speaker 1' (human display); anything else passes through."""
    if str(diar_label).startswith("Speaker_"):
        return f"Speaker {idx + 1}"
    return str(diar_label)


def apply_speaker_identity(
    segments: list, audio_filepath: str
) -> Tuple[list, Dict[str, Dict[str, Any]]]:
    """Attach stable ids + (voice-matched) names to already-diarized segments.

    Adds to every segment, without removing any existing key:
      ``speaker_id``          stable per-meeting id, e.g. "SPEAKER_00"
      ``speaker_name``        registered name if a voice profile matched, else None
      ``speaker_confidence``  cosine confidence when a name was assigned, else None
      ``label``               display label: the name, else "Speaker N"
      ``name_source``         "voice_match" | None (LLM step may set "transcript_llm")
    ``segments[*]["speaker"]`` becomes the matched NAME when identified (so
    downstream name-aware code keeps working); it stays "Speaker_N" otherwise.

    Returns ``(segments, identity_map)`` where identity_map is keyed by the FINAL
    display label of each speaker.
    """
    if not segments:
        return segments, {}

    order = list(dict.fromkeys(s.get("speaker", "Speaker_1") for s in segments))
    id_by_diar = {lb: f"SPEAKER_{i:02d}" for i, lb in enumerate(order)}
    friendly_by_diar = {lb: _friendly_label(lb, i) for i, lb in enumerate(order)}

    match_by_diar: Dict[str, Dict[str, Any]] = {}
    try:
        if getattr(settings, "VOICE_ENROLLMENT_ENABLED", True) and audio_filepath and os.path.exists(audio_filepath):
            from backend.services.speaker_identification_service import (
                load_active_voice_profiles,
                identify_speakers,
            )
            profiles = load_active_voice_profiles()
            if profiles:
                turns = [
                    {
                        "speaker": s.get("speaker", "Speaker_1"),
                        "start": float(s.get("start_time", 0.0) or 0.0),
                        "end": float(s.get("end_time", 0.0) or 0.0),
                    }
                    for s in segments
                ]
                match_by_diar = identify_speakers(audio_filepath, turns, profiles=profiles)
            else:
                logger.info("[SpeakerID] no registered voice profiles — speakers stay generic.")
    except Exception as e:  # pragma: no cover - identification must never break the pipeline
        logger.warning("[SpeakerID] voice matching skipped: %s", e)
        match_by_diar = {}

    identity_map: Dict[str, Dict[str, Any]] = {}
    for lb in order:
        m = match_by_diar.get(lb, {})
        name = m.get("speaker_name")
        conf = m.get("confidence")
        display = name or friendly_by_diar[lb]
        identity_map[display] = {
            "speaker_id": id_by_diar[lb],
            "speaker_name": name,
            "label": display,
            "confidence": conf,
            "diar_label": lb,
            "match_similarity": m.get("similarity"),
            "match_reason": m.get("reason"),
        }

    out = []
    for s in segments:
        lb = s.get("speaker", "Speaker_1")
        m = match_by_diar.get(lb, {})
        name = m.get("speaker_name")
        conf = m.get("confidence")
        display = name or friendly_by_diar[lb]
        ns = dict(s)
        ns["speaker_id"] = id_by_diar[lb]
        ns["speaker_name"] = name
        ns["speaker_confidence"] = conf
        ns["label"] = display
        ns["name_source"] = "voice_match" if name else None
        ns["speaker"] = name if name else lb   # keep "Speaker_N" when unidentified
        out.append(ns)

    named = sorted({v["speaker_name"] for v in identity_map.values() if v["speaker_name"]})
    logger.info(
        "[SpeakerID] identity applied: %d speaker(s), %d named by voice %s.",
        len(order), len(named), named or "",
    )
    return out, identity_map


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
class DiarizationResult(tuple):
    """Custom tuple returning (segments, diagnostics, status) when unpacked as 3 items,
    while attaching .metadata attribute for engine telemetry.
    """
    def __new__(cls, segments, diagnostics, status, metadata=None):
        inst = super().__new__(cls, (segments, diagnostics, status))
        inst.metadata = metadata or {}
        return inst


def validate_speaker_assignments(
    segments: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
    min_overlap_sec: float = 0.05,
) -> Dict[str, Any]:
    """Check every transcript segment's speaker against the diarization turns it actually overlaps.

    ``turns`` are the engine's ``{start, end, speaker}`` turns (its own labels); ``segments`` carry the labels
    shown to the user (``Speaker_1`` ...), assigned by word alignment and then relabelled. Each transcript label is
    mapped to the engine speaker that covers most of its time, and a segment is SUPPORTED when the engine speaker
    that dominates its own time window is that one. Nothing is changed - this reports how far the assignments are
    backed by the diarization, it never re-assigns a speaker.

    ``status``: VALID (>= 90 % supported, none outside every turn), WARNING (>= 60 %), MISMATCH (below), or
    NOT_VERIFIABLE when there are no usable turns / segments.
    """
    usable_turns = []
    for t in turns or []:
        try:
            s, e = float(t["start"]), float(t["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e > s:
            usable_turns.append((s, e, t.get("speaker")))

    per_label: Dict[str, Dict[Any, float]] = {}
    dominant: List[tuple] = []
    unusable = 0
    for seg in segments or []:
        label = seg.get("speaker")
        try:
            s, e = float(seg["start_time"]), float(seg["end_time"])
        except (KeyError, TypeError, ValueError):
            unusable += 1
            continue
        if e <= s:
            unusable += 1
            continue
        overlap: Dict[Any, float] = {}
        for ts_, te_, spk in usable_turns:
            ov = min(e, te_) - max(s, ts_)
            if ov > 0:
                overlap[spk] = overlap.get(spk, 0.0) + ov
        best = max(overlap, key=overlap.get) if overlap else None
        if best is not None and overlap[best] < min_overlap_sec:
            best = None
        dominant.append((label, best))
        if best is not None:
            per_label.setdefault(label, {})
            per_label[label][best] = per_label[label].get(best, 0.0) + overlap[best]

    checked = len(dominant)
    if not usable_turns or checked == 0:
        return {
            "status": "NOT_VERIFIABLE", "segments_checked": checked, "supported": 0, "mismatched": 0,
            "outside_turns": 0, "unusable_times": unusable, "supported_ratio": None, "label_to_engine_speaker": {},
        }
    label_engine = {lbl: max(c, key=c.get) for lbl, c in per_label.items()}
    supported = sum(1 for lbl, best in dominant if best is not None and label_engine.get(lbl) == best)
    outside = sum(1 for _lbl, best in dominant if best is None)
    mismatched = checked - supported - outside
    ratio = round(supported / checked, 3)
    if ratio >= 0.9 and outside == 0:
        status = "VALID"
    elif ratio >= 0.6:
        status = "WARNING"
    else:
        status = "MISMATCH"
    return {
        "status": status,
        "segments_checked": checked,
        "supported": supported,
        "mismatched": mismatched,
        "outside_turns": outside,
        "unusable_times": unusable,
        "supported_ratio": ratio,
        "label_to_engine_speaker": {str(k): v for k, v in label_engine.items()},
    }


def diarize_and_align(
    audio_filepath: str,
    transcript_data: dict,
    num_speakers: Optional[int] = None,
    primary_result: Optional[tuple] = None,
) -> DiarizationResult:
    """
    Main entry point for Universal Speaker Diarization Strategy and Timestamp Alignment.
    Executes via DiarizationManager supporting NeMo and PyAnnote engine strategies.

    ``primary_result``: ``DiarizationManager.run_primary`` output for this audio, computed earlier (while the
    transcription request was in flight). The engine is then not run again; everything after it is unchanged.

    Returns:
        DiarizationResult(aligned_segments, speaker_diagnostics, diarization_status, engine_metadata)
    """
    logger.info("[Universal Diarization Strategy] Executing diarize_and_align pipeline...")

    words = transcript_data.get("words", [])
    text_content = transcript_data.get("text", "")

    if not words and not text_content:
        logger.warning("[Universal Diarization Strategy] No words or text content provided in transcript_data.")
        return DiarizationResult([], compute_speaker_diagnostics([]), "FAILED", {})

    from backend.diarization.manager import diarization_manager

    # Step 1: Execute Diarization Strategy Manager (NeMo / PyAnnote / Fallback)
    turns, metadata, status = diarization_manager.execute(
        audio_filepath=audio_filepath,
        num_speakers=num_speakers,
        transcript_words=words,
        transcript_text=text_content,
        primary_result=primary_result,
    )

    if status == "FAILED" or turns is None:
        logger.error(f"[Universal Diarization Strategy] Diarization process returned status '{status}'. Metadata: {metadata}")
        # Build unassigned single-speaker segments without faking successful diarization
        if words:
            unassigned_segs = [
                {
                    "speaker": "Speaker_1",
                    "start_time": f"{words[0]['start']:.3f}",
                    "end_time": f"{words[-1]['end']:.3f}",
                    "text": " ".join(w["word"] for w in words).strip(),
                }
            ]
        else:
            # No word timestamps to derive an end time from — probe the actual
            # media duration rather than guessing a fixed 10s span, so a long
            # recording that fails diarization still gets an honest timeline.
            real_dur = _probe_audio_duration(audio_filepath)
            end_time_str = f"{real_dur:.3f}" if real_dur and real_dur > 0 else diar_cfg.FALLBACK_DURATION
            unassigned_segs = [
                {
                    "speaker": "Speaker_1",
                    "start_time": "0.000",
                    "end_time": end_time_str,
                    "text": text_content.strip(),
                }
            ]
        for _s in unassigned_segs:
            _s.setdefault("speaker_id", "SPEAKER_00")
            _s.setdefault("speaker_name", None)
            _s.setdefault("speaker_confidence", None)
            _s.setdefault("label", "Speaker 1")
            _s.setdefault("name_source", None)
        return DiarizationResult(unassigned_segs, compute_speaker_diagnostics(unassigned_segs), "FAILED", metadata)

    # Convert turns to format expected by align_words_to_nemo ({start, end, speaker})
    formatted_turns = [
        {
            "start": float(t["start_time"]),
            "end": float(t["end_time"]),
            "speaker": t["speaker"],
        }
        for t in turns
    ]

    # Step 2: Timestamp Alignment (Word-level max overlap or segment overlap)
    if words:
        segments = align_words_to_nemo(words, formatted_turns)
    else:
        logger.info("[Universal Diarization Strategy] Word timestamps missing. Performing segment-level timeline alignment.")
        segments = [
            {
                "speaker": turn["speaker"],
                "start_time": f"{turn['start']:.3f}",
                "end_time": f"{turn['end']:.3f}",
                "text": text_content.strip(),
            }
            for turn in formatted_turns
        ]

    # Step 3: Standardize speaker labels chronologically (Speaker_1, Speaker_2...)
    segments = _normalize_speaker_labels(segments)

    # Step 3b: Reconcile alignment artefacts (spurious slivers) — keeps the
    # transcript speaker set consistent with the diarization turn set without
    # forcing any fixed count.
    turn_spk_count = len({t["speaker"] for t in formatted_turns})
    segments = _reconcile_aligned_speakers(segments)
    segments = _normalize_speaker_labels(segments)
    aligned_spk_count = len({s["speaker"] for s in segments})
    if turn_spk_count != aligned_spk_count:
        logger.info(
            "[Universal Diarization Strategy] Speaker count: diarization turns=%d, "
            "aligned transcript=%d (alignment resolved the difference).",
            turn_spk_count, aligned_spk_count,
        )

    # Step 3c: Voice-based speaker identification — attach SPEAKER_NN ids and,
    # when a registered voice profile matches confidently, the real name.
    segments, identity_map = apply_speaker_identity(segments, audio_filepath)
    voice_locked = {
        v["label"] for v in identity_map.values() if v.get("speaker_name")
    }

    # Step 4: Optional LLM identity mapping (transcript-clue names) — never
    # overrides a voice-verified label, and its guesses are tagged, not verified.
    segments = resolve_speaker_labels_with_llm(segments, locked_labels=voice_locked)

    # Step 5: Filter out empty segments
    segments = [s for s in segments if s.get("text", "").strip()]

    # Step 6: Compute dynamic speaker diagnostics (carrying identity through)
    # Talk time = detected speech inside each speaker's segments (not the segment spans, which include the pauses).
    speech_regions = speech_regions_for_durations(audio_filepath, formatted_turns)
    speech_durations = speech_seconds_per_speaker(segments, speech_regions) if speech_regions else None
    diagnostics = compute_speaker_diagnostics(segments, identity_map=identity_map, speech_durations=speech_durations)
    # Are the transcript's speaker labels backed by the engine's actual turns? (reported, never re-assigned)
    diagnostics["speaker_assignment_validation"] = validate_speaker_assignments(segments, formatted_turns)
    _sav = diagnostics["speaker_assignment_validation"]
    if _sav["status"] in ("WARNING", "MISMATCH"):
        logger.warning(
            "[Universal Diarization Strategy] Speaker assignment %s: %d of %d segments agree with the diarization "
            "turns (%d outside every turn).", _sav["status"], _sav["supported"], _sav["segments_checked"],
            _sav["outside_turns"],
        )
    diagnostics["speaker_identification"] = {
        "profiles_available": bool(identity_map) and any(
            v.get("match_reason") not in (None, "no voice profiles") for v in identity_map.values()
        ),
        "named_by_voice": sorted(
            {v["speaker_name"] for v in identity_map.values() if v.get("speaker_name")}
        ),
        "map": identity_map,
    }
    if metadata is not None:
        metadata["speaker_identity_map"] = identity_map
        metadata["neural_embedder_used"] = using_neural_embedder()

    spk_cnt = diagnostics["speaker_count"]
    logger.info(
        f"[Universal Diarization Strategy] Diarization completed with status '{status}' "
        f"using engine '{metadata.get('selected_engine')}': {len(segments)} segments, {spk_cnt} unique speaker(s)."
    )

    return DiarizationResult(segments, diagnostics, status, metadata)

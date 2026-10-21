"""PyAnnote Speaker Diarization Engine Implementation.

Strategy 1: PyAnnote.audio pipeline (online, requires HF_TOKEN).
Strategy 2: High-resolution acoustic clustering fallback (offline capable).
  - Uses rich 31-dim acoustic embeddings (24 Mel + spectral moments + pitch + ZCR)
    matching the NeMo engine feature space, for better cross-speaker separation.
  - Dynamic speaker count via silhouette-score optimization (k=2..DIARIZATION_MAX_SPEAKERS).
    No hardcoded k=1 collapse.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.config.config import settings
from backend.diarization.base_engine import BaseDiarizationEngine, DiarizationSegment

logger = logging.getLogger(__name__)


class PyAnnoteDiarizationEngine(BaseDiarizationEngine):
    """PyAnnote.audio Speaker Diarization Engine."""

    @property
    def name(self) -> str:
        return "pyannote"

    def is_available(self) -> bool:
        if not getattr(settings, "PYANNOTE_ENABLED", True):
            return False
        return True

    def diarize(
        self, audio_filepath: str, num_speakers: Optional[int] = None
    ) -> Tuple[Optional[List[Dict[str, Any]]], str, Optional[str]]:
        logger.info(f"[PyAnnoteEngine] Executing diarization for: {audio_filepath}")

        if not os.path.exists(audio_filepath):
            return None, "FAILED", f"Audio file not found: {audio_filepath}"

        hf_token = getattr(settings, "HF_TOKEN", "") or os.getenv("HF_TOKEN", "")

        try:
            # Strategy 1: PyAnnote.audio Pipeline (online, accurate)
            import torch
            from pyannote.audio import Pipeline

            model_id = getattr(settings, "PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
            pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))

            kwargs = {}
            if num_speakers is not None and num_speakers >= 1:
                kwargs["num_speakers"] = num_speakers

            diarization_res = pipeline(audio_filepath, **kwargs)

            normalized: List[Dict[str, Any]] = []
            speaker_map: Dict[str, str] = {}

            for turn, _, speaker in diarization_res.itertracks(yield_label=True):
                if speaker not in speaker_map:
                    speaker_map[speaker] = f"Speaker_{len(speaker_map) + 1}"
                norm_spk = speaker_map[speaker]

                seg = DiarizationSegment(
                    speaker=norm_spk,
                    start_time=float(turn.start),
                    end_time=float(turn.end),
                    duration=float(turn.end - turn.start),
                )
                normalized.append(seg.to_dict())

            logger.info(f"[PyAnnoteEngine] PyAnnote pipeline: {len(normalized)} turns, {len(speaker_map)} speakers.")
            return normalized, "SUCCESS", None

        except Exception as p_err:
            logger.info(
                f"[PyAnnoteEngine] PyAnnote pipeline unavailable ({type(p_err).__name__}: {p_err}). "
                f"Falling back to high-resolution acoustic clustering engine..."
            )

            # Strategy 2: High-resolution acoustic clustering fallback
            return self._acoustic_cluster_fallback(audio_filepath, num_speakers)

    def _acoustic_cluster_fallback(
        self, audio_filepath: str, num_speakers: Optional[int]
    ) -> Tuple[Optional[List[Dict[str, Any]]], str, Optional[str]]:
        """
        Offline speaker diarization using rich 31-dim acoustic embeddings.

        Feature vector per window (31 dimensions):
          - 24 Mel filterbank energies (primary speaker discriminator)
          - 3 spectral moments: centroid, bandwidth, rolloff
          - 1 pitch estimate (fundamental frequency via autocorrelation)
          - 1 zero-crossing rate
          - 2 MFCC delta summary statistics (mean of delta1, delta2)

        Speaker count is selected dynamically using silhouette-score
        optimization over k=2..DIARIZATION_MAX_SPEAKERS (same algorithm as NeMo).
        num_speakers overrides k-selection when explicitly provided.
        """
        try:
            import soundfile as sf

            data, sr = sf.read(audio_filepath, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            dur_sec = len(data) / sr if sr > 0 else 0.0
            if dur_sec <= 0.1:
                return None, "FAILED", f"Audio duration too short ({dur_sec:.2f}s)"

            # ── Window segmentation ────────────────────────────────────────────
            chunk_dur = 1.5   # seconds per window
            step_dur  = 0.75  # hop size (50% overlap for better boundary detection)
            turns_raw = []
            cur = 0.0
            while cur < dur_sec:
                t_end = min(dur_sec, cur + chunk_dur)
                if t_end - cur >= 0.3:
                    turns_raw.append((cur, t_end))
                cur += step_dur

            if not turns_raw:
                return None, "DEGRADED", "No speech windows detected"

            # ── Rich 31-dim acoustic feature extraction per window ─────────────
            n_mel = 24
            embeddings = []
            valid_turns = []

            for s_t, e_t in turns_raw:
                s_idx = int(s_t * sr)
                e_idx = int(e_t * sr)
                seg_data = data[s_idx:e_idx]

                # Skip silence
                rms = np.sqrt(np.mean(seg_data ** 2))
                if rms < 0.0005:
                    continue

                n = len(seg_data)
                if n < 512:
                    seg_data = np.pad(seg_data, (0, 512 - n))

                # Feature 1-24: Mel filterbank energies
                fft_mag = np.abs(np.fft.rfft(seg_data, n=1024))
                freqs   = np.fft.rfftfreq(1024, d=1.0 / sr)
                mel_f   = self._mel_filterbank(freqs, n_mel, sr)
                mel_e   = np.maximum(1e-8, mel_f @ fft_mag)
                mel_e   = np.log(mel_e)
                mel_e   = (mel_e - mel_e.mean()) / (mel_e.std() + 1e-8)

                # Feature 25-27: Spectral moments
                spec_sum   = fft_mag.sum() + 1e-8
                spec_cent  = float(np.sum(freqs * fft_mag) / spec_sum)
                spec_bw    = float(np.sqrt(np.sum(((freqs - spec_cent) ** 2) * fft_mag) / spec_sum))
                cum_power  = np.cumsum(fft_mag)
                roll_idx   = np.searchsorted(cum_power, 0.85 * cum_power[-1])
                spec_roll  = float(freqs[min(roll_idx, len(freqs) - 1)])

                # Feature 28: Pitch estimate (normalized autocorrelation peak)
                max_lag = int(sr / 60)   # 60 Hz lowest fundamental
                min_lag = int(sr / 400)  # 400 Hz highest fundamental
                if max_lag > min_lag and max_lag < n:
                    autocorr = np.correlate(seg_data, seg_data, mode="full")[n - 1:]
                    autocorr = autocorr[min_lag:max_lag]
                    peak_idx = int(np.argmax(autocorr))
                    pitch_f0 = float(sr / (min_lag + peak_idx)) if (min_lag + peak_idx) > 0 else 0.0
                else:
                    pitch_f0 = 0.0

                # Feature 29: Zero-crossing rate
                zcr = float(np.mean(np.abs(np.diff(np.sign(seg_data)))) / 2.0)

                # Features 30-31: Delta MFCC summary (mean of 1st and 2nd derivative of mel energies)
                delta1_mean = float(np.mean(np.diff(mel_e)))
                delta2_mean = float(np.mean(np.diff(mel_e, n=2))) if len(mel_e) > 2 else 0.0

                # Normalize spectral scalars to [0,1] range
                feat = np.concatenate([
                    mel_e,                          # 24 dims
                    [spec_cent / (sr / 2 + 1e-8),   # normalized centroid
                     spec_bw   / (sr / 2 + 1e-8),   # normalized bandwidth
                     spec_roll / (sr / 2 + 1e-8),   # normalized rolloff
                     pitch_f0  / 400.0,             # normalized pitch (0–1 for 0–400 Hz)
                     zcr,                           # ZCR (already 0–1)
                     delta1_mean,                   # MFCC delta-1 mean
                     delta2_mean,                   # MFCC delta-2 mean
                     ],
                ], axis=0)

                norm = np.linalg.norm(feat)
                if norm > 1e-8:
                    embeddings.append(feat / norm)
                    valid_turns.append((s_t, e_t))

            if not valid_turns:
                single = [DiarizationSegment("Speaker_1", 0.0, dur_sec, dur_sec).to_dict()]
                return single, "SUCCESS", None

            n_valid = len(valid_turns)
            embed_matrix = np.array(embeddings, dtype=np.float32)

            # ── Dynamic speaker count — shared selector ───────────────────────
            # Routes through diarization_service._select_speaker_labels with
            # is_neural=False, so this offline acoustic fallback uses the SAME
            # strong-evidence gate as the NeMo path (require silhouette >= 0.30
            # AND real centroid separation before reporting >1 speaker) plus the
            # same duplicate-speaker merge / noise-cluster absorption. No
            # hardcoded or capped speaker count.
            from backend.services.diarization_service import _select_speaker_labels

            labels, best_k = _select_speaker_labels(
                embed_matrix, valid_turns, num_speakers, is_neural=False
            )
            logger.info(f"[PyAnnoteEngine] Acoustic selector: k={best_k} over {n_valid} windows.")

            # ── Build output segments ──────────────────────────────────────────
            spk_map    = {lbl: f"Speaker_{lbl + 1}" for lbl in set(labels)}
            normalized = []
            for (st, et), lbl in zip(valid_turns, labels):
                seg = DiarizationSegment(
                    speaker=spk_map[lbl],
                    start_time=st,
                    end_time=et,
                    duration=max(0.0, et - st),
                )
                normalized.append(seg.to_dict())

            # ── Merge adjacent same-speaker windows ────────────────────────────
            merged = []
            for s in normalized:
                if merged and merged[-1]["speaker"] == s["speaker"]:
                    gap = s["start_time"] - merged[-1]["end_time"]
                    if gap <= 0.5:  # tighter gap threshold for cleaner segments
                        merged[-1]["end_time"] = s["end_time"]
                        merged[-1]["duration"] = round(merged[-1]["end_time"] - merged[-1]["start_time"], 3)
                        continue
                merged.append(s)

            logger.info(
                f"[PyAnnoteEngine] Acoustic clustering completed: {len(merged)} turns, "
                f"{len(set(labels))} speaker(s)."
            )
            return merged, "SUCCESS", None

        except Exception as inner_err:
            logger.error(f"[PyAnnoteEngine] Diarization failed: {inner_err}", exc_info=True)
            return None, "FAILED", f"PyAnnote engine error: {str(inner_err)}"

    @staticmethod
    def _mel_filterbank(freqs: np.ndarray, n_mel: int, sr: int) -> np.ndarray:
        """Construct a triangular Mel filterbank matrix (n_mel × n_freq)."""
        f_min = 0.0
        f_max = float(sr) / 2.0

        def hz2mel(hz: float) -> float:
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel2hz(mel: float) -> float:
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_min = hz2mel(f_min)
        mel_max = hz2mel(f_max)
        mel_points = np.linspace(mel_min, mel_max, n_mel + 2)
        hz_points  = np.array([mel2hz(m) for m in mel_points])

        n_freq = len(freqs)
        fb = np.zeros((n_mel, n_freq), dtype=np.float32)
        for m in range(1, n_mel + 1):
            f_m_minus = hz_points[m - 1]
            f_m       = hz_points[m]
            f_m_plus  = hz_points[m + 1]
            for k, f in enumerate(freqs):
                if f_m_minus <= f <= f_m:
                    fb[m - 1, k] = (f - f_m_minus) / (f_m - f_m_minus + 1e-8)
                elif f_m < f <= f_m_plus:
                    fb[m - 1, k] = (f_m_plus - f) / (f_m_plus - f_m + 1e-8)
        return fb

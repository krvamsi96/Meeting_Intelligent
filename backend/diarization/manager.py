"""Universal Diarization Strategy Manager & Fallback Controller.

Supported Modes:
- AUTO
- NEMO_ONLY
- PYANNOTE_ONLY
- NEMO_PRIMARY_PYANNOTE_FALLBACK (Default production strategy)
- PYANNOTE_PRIMARY_NEMO_FALLBACK

Execution Rules:
- Primary engine executes first.
- If primary passes quality validation, secondary engine MUST NOT execute.
- If primary fails/degrades AND fallback is enabled in mode, secondary engine executes and validates.
- If explicit engine mode is configured (e.g. NEMO_ONLY), system will NOT silently switch to unauthorized engine.
- Complete metadata telemetry is recorded (configured_mode, primary_engine, selected_engine, fallback_used, fallback_reason, latency).
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.config.config import settings
from backend.diarization.base_engine import BaseDiarizationEngine
from backend.diarization.nemo_engine import NeMoDiarizationEngine
from backend.diarization.pyannote_engine import PyAnnoteDiarizationEngine
from backend.diarization.validator import (
    DiarizationQualityValidator,
    DiarizationValidationResult,
    diarization_validator,
)

logger = logging.getLogger(__name__)


class DiarizationMode(str, Enum):
    AUTO = "AUTO"
    NEMO_ONLY = "NEMO_ONLY"
    PYANNOTE_ONLY = "PYANNOTE_ONLY"
    NEMO_PRIMARY_PYANNOTE_FALLBACK = "NEMO_PRIMARY_PYANNOTE_FALLBACK"
    PYANNOTE_PRIMARY_NEMO_FALLBACK = "PYANNOTE_PRIMARY_NEMO_FALLBACK"


class DiarizationManager:
    """Strategy Manager orchestrating engine selection, validation, and fallback."""

    def __init__(self):
        self.nemo_engine = NeMoDiarizationEngine()
        self.pyannote_engine = PyAnnoteDiarizationEngine()
        self.validator = diarization_validator

    def get_configured_mode(self) -> DiarizationMode:
        raw = getattr(settings, "DIARIZATION_MODE", "NEMO_PRIMARY_PYANNOTE_FALLBACK")
        try:
            return DiarizationMode(raw.upper())
        except Exception:
            return DiarizationMode.NEMO_PRIMARY_PYANNOTE_FALLBACK

    def run_primary(
        self, audio_filepath: str, num_speakers: Optional[int] = None
    ) -> Tuple[Optional[List[Dict[str, Any]]], str, Optional[str]]:
        """The primary engine's pass over the AUDIO alone: ``(segments, status, error_message)``.

        It needs no transcript, so the pipeline runs it while the speech-to-text request is still waiting on AIRAWAT
        and hands the result to ``execute(primary_result=...)`` - the two used to run one after the other.
        """
        primary_engine, _ = self._resolve_engines(self.get_configured_mode())
        return primary_engine.diarize(audio_filepath, num_speakers=num_speakers)

    def execute(
        self,
        audio_filepath: str,
        num_speakers: Optional[int] = None,
        transcript_words: Optional[List[Dict[str, Any]]] = None,
        transcript_text: str = "",
        ref_timeline: Optional[List[Dict[str, Any]]] = None,
        primary_result: Optional[Tuple[Optional[List[Dict[str, Any]]], str, Optional[str]]] = None,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any], str]:
        """
        Execute diarization under configured strategy.

        ``primary_result``: the output of ``run_primary`` for this same audio, when it was already computed; the
        primary engine is then not run again (validation and any fallback proceed exactly as usual).

        Returns:
            (segments, metadata_dict, status_str)
            status_str: "SUCCESS", "FAILED", or "DEGRADED"
        """
        start_t = time.perf_counter()
        mode = self.get_configured_mode()
        logger.info(f"[DiarizationManager] Executing diarization strategy mode: {mode.value}")

        primary_engine, fallback_engine = self._resolve_engines(mode)

        # ── Step 1: Run Primary Engine ───────────────────────────────────────
        selected_engine = primary_engine.name
        fallback_used = False
        fallback_reason = None

        if primary_result is not None:
            segments, status, err_msg = primary_result
        else:
            segments, status, err_msg = primary_engine.diarize(
                audio_filepath, num_speakers=num_speakers
            )

        val_res: DiarizationValidationResult = self.validator.validate(
            segments, transcript_words, transcript_text, ref_timeline
        )

        # ── Step 2: Fallback Evaluation ───────────────────────────────────────
        if not val_res.passed and fallback_engine is not None:
            diag_reason = val_res.failure_reason or err_msg or "Primary engine failed quality validation"
            logger.warning(
                f"[DiarizationManager] Primary engine '{primary_engine.name}' failed quality gate ({diag_reason}). "
                f"Attempting fallback to secondary engine '{fallback_engine.name}'..."
            )

            fallback_start = time.perf_counter()
            fb_segments, fb_status, fb_err = fallback_engine.diarize(
                audio_filepath, num_speakers=num_speakers
            )
            fb_val_res = self.validator.validate(
                fb_segments, transcript_words, transcript_text, ref_timeline
            )

            # Only adopt the fallback when it actually PASSES validation, or when
            # the primary produced nothing at all. Previously any non-empty
            # fallback replaced a merely-"suspected" primary — which let a weak
            # secondary engine inflate a correct 1-speaker result to 2+.
            if fb_val_res.passed or not segments:
                segments = fb_segments
                selected_engine = fallback_engine.name
                fallback_used = True
                fallback_reason = diag_reason
                status, err_msg = fb_status, fb_err
                val_res = fb_val_res
                logger.info(f"[DiarizationManager] Fallback engine '{fallback_engine.name}' adopted.")
            else:
                logger.warning(
                    "[DiarizationManager] Fallback engine did not pass validation either; "
                    "keeping primary engine result (status DEGRADED)."
                )

        elapsed = round(time.perf_counter() - start_t, 3)

        metadata = {
            "configured_mode": mode.value,
            "primary_engine": primary_engine.name,
            "selected_engine": selected_engine,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "engine_status": status,
            "quality_passed": val_res.passed,
            "validation_status": val_res.status,
            "validation_diagnostics": val_res.diagnostics,
            "processing_latency_sec": elapsed,
        }

        # A result is SUCCESS only when it passed validation AND the engine itself reported full quality (it reports
        # DEGRADED e.g. when it had to use a reduced-accuracy embedder); the engine's verdict used to be discarded.
        engine_ok = str(status or "").upper() == "SUCCESS"
        final_status = "SUCCESS" if (val_res.passed and engine_ok) else ("DEGRADED" if segments else "FAILED")
        if segments and not engine_ok and err_msg:
            metadata["engine_note"] = err_msg

        return segments, metadata, final_status

    def _resolve_engines(
        self, mode: DiarizationMode
    ) -> Tuple[BaseDiarizationEngine, Optional[BaseDiarizationEngine]]:
        if mode == DiarizationMode.NEMO_ONLY:
            return self.nemo_engine, None
        elif mode == DiarizationMode.PYANNOTE_ONLY:
            return self.pyannote_engine, None
        elif mode == DiarizationMode.PYANNOTE_PRIMARY_NEMO_FALLBACK:
            return self.pyannote_engine, self.nemo_engine
        elif mode == DiarizationMode.AUTO:
            # Dynamic selection based on availability
            if self.nemo_engine.is_available():
                return self.nemo_engine, self.pyannote_engine
            return self.pyannote_engine, None
        else:
            # Default NEMO_PRIMARY_PYANNOTE_FALLBACK
            return self.nemo_engine, self.pyannote_engine


diarization_manager = DiarizationManager()

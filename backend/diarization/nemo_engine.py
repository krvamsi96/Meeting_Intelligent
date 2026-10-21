"""NVIDIA NeMo Speaker Diarization Engine Implementation."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.config.config import settings
from backend.diarization.base_engine import BaseDiarizationEngine, DiarizationSegment

logger = logging.getLogger(__name__)


class NeMoDiarizationEngine(BaseDiarizationEngine):
    """NVIDIA NeMo Speaker Diarization Engine."""

    @property
    def name(self) -> str:
        return "nvidia_nemo"

    def is_available(self) -> bool:
        if not getattr(settings, "NEMO_ENABLED", True):
            return False
        try:
            import numpy
            import scipy
            return True
        except ImportError:
            return False

    def diarize(
        self, audio_filepath: str, num_speakers: Optional[int] = None
    ) -> Tuple[Optional[List[Dict[str, Any]]], str, Optional[str]]:
        logger.info(f"[NeMoEngine] Executing diarization for: {audio_filepath}")

        # Delegate to underlying NeMo core implementation in diarization_service
        from backend.services import diarization_service as ds

        turns, status, error_msg = ds.run_nemo_diarization(
            audio_filepath, num_speakers=num_speakers
        )

        if status == "FAILED" or turns is None:
            return None, "FAILED", error_msg or "NeMo diarization failed"

        # Normalize into common internal output schema
        normalized: List[Dict[str, Any]] = []
        for t in turns:
            try:
                st = float(t.get("start", 0.0))
                et = float(t.get("end", 0.0))
                spk = str(t.get("speaker", "Speaker_1"))
                seg = DiarizationSegment(
                    speaker=spk, start_time=st, end_time=et, duration=max(0.0, et - st)
                )
                normalized.append(seg.to_dict())
            except Exception as e:
                logger.warning(f"[NeMoEngine] Failed to normalize turn {t}: {e}")

        return normalized, status, None

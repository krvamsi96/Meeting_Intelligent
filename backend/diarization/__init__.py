"""Universal Speaker Diarization Strategy Package (NVIDIA NeMo + PyAnnote)."""

from backend.diarization.base_engine import BaseDiarizationEngine, DiarizationSegment
from backend.diarization.nemo_engine import NeMoDiarizationEngine
from backend.diarization.pyannote_engine import PyAnnoteDiarizationEngine
from backend.diarization.validator import DiarizationQualityValidator
from backend.diarization.manager import DiarizationManager, DiarizationMode

__all__ = [
    "BaseDiarizationEngine",
    "DiarizationSegment",
    "NeMoDiarizationEngine",
    "PyAnnoteDiarizationEngine",
    "DiarizationQualityValidator",
    "DiarizationManager",
    "DiarizationMode",
]

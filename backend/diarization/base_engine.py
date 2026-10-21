"""Abstract Base Diarization Engine Interface & Common Schema.

Downstream systems interact with this normalized internal schema:
{
    "speaker": "Speaker_1",
    "start_time": 10.42,
    "end_time": 15.83,
    "duration": 5.41
}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DiarizationSegment:
    """Normalized internal speaker turn representation."""

    speaker: str
    start_time: float
    end_time: float
    duration: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(max(0.0, self.end_time - self.start_time), 3),
        }


class BaseDiarizationEngine(ABC):
    """Abstract interface for all Diarization Engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier string (e.g., 'nvidia_nemo', 'pyannote')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if required packages, hardware, and model weights are accessible."""
        pass

    @abstractmethod
    def diarize(
        self, audio_filepath: str, num_speakers: Optional[int] = None
    ) -> Tuple[Optional[List[Dict[str, Any]]], str, Optional[str]]:
        """
        Execute speaker diarization on audio file.

        Returns:
            (normalized_segments, status_str, error_message)
            status_str: "SUCCESS", "FAILED", or "DEGRADED"
        """
        pass

"""Diarization Quality Validator & Speaker Collapse Detector.

Validates output dynamically across both NeMo and PyAnnote engines:
- Speaker count
- Segment count & timestamps
- Speech coverage & duration distribution
- Speaker turn transitions & overlap
- Speaker collapse detection (distinguishes valid 1-speaker meeting from multi-speaker collapse)

Metrics Policy:
- If Ground Truth timeline exists: computes GROUND_TRUTH_METRIC (DER, JER).
- If Ground Truth timeline does NOT exist: computes RUNTIME_DIAGNOSTIC (never fabricate DER/JER).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DiarizationValidationResult:
    """Result of diarization output quality validation."""

    passed: bool
    status: str  # "SUCCESS", "DIARIZATION_SUSPECTED_FAILURE", "FAILED"
    failure_reason: Optional[str] = None
    metric_type: str = "RUNTIME_DIAGNOSTIC"  # "GROUND_TRUTH_METRIC" or "RUNTIME_DIAGNOSTIC"
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class DiarizationQualityValidator:
    """Universal quality validator for diarization engine outputs."""

    def validate(
        self,
        segments: Optional[List[Dict[str, Any]]],
        transcript_words: Optional[List[Dict[str, Any]]] = None,
        transcript_text: str = "",
        ref_timeline: Optional[List[Dict[str, Any]]] = None,
    ) -> DiarizationValidationResult:

        if not segments:
            return DiarizationValidationResult(
                passed=False,
                status="FAILED",
                failure_reason="No diarization segments generated",
                diagnostics={"segment_count": 0},
            )

        # Signal 1: Ground Truth evaluation if reference timeline exists
        if ref_timeline:
            from backend.evaluation import metrics as eval_metrics
            diar_res = eval_metrics.diarization_metrics(ref_timeline, segments)
            der = diar_res.get("der", 100.0)
            jer = diar_res.get("jer", 100.0)
            acc = diar_res.get("accuracy", 0.0)

            passed = der <= 35.0
            status = "SUCCESS" if passed else "FAILED"
            return DiarizationValidationResult(
                passed=passed,
                status=status,
                failure_reason=None if passed else f"DER ({der:.1f}%) exceeded maximum allowable threshold",
                metric_type="GROUND_TRUTH_METRIC",
                diagnostics={"der": der, "jer": jer, "accuracy": acc},
            )

        # Signal 2: Runtime Diagnostic Validation
        speakers = {s.get("speaker") for s in segments if s.get("speaker")}
        speaker_count = len(speakers)

        total_dur = sum(
            max(0.0, float(s.get("end_time", 0.0)) - float(s.get("start_time", 0.0)))
            for s in segments
        )

        # Speaker Collapse Detection logic
        # A 1-speaker meeting CAN be valid (e.g. solo lecture/presentation).
        # We check multiple signals to detect SUSPECTED collapse rather than automatically failing.
        #
        # Signal priority:
        #   1. Word-level long-gap analysis (PRIMARY) — works on raw Whisper output with no speaker labels
        #   2. Transcript turn-marker check (SECONDARY) — works only when text already has "Speaker:" labels
        suspected_collapse = False
        collapse_reason = ""

        if speaker_count == 1:
            # A 1-speaker result is only "suspected collapse" when there is
            # CORROBORATING evidence from two independent signals. Long pauses
            # ALONE are normal in a solo lecture / presentation / dictation and
            # must NOT trigger a fallback to an equally-weak secondary engine
            # (that only inflated 1-speaker meetings to 2). We require both:
            #   (a) several long acoustic pauses, AND
            #   (b) the transcript text already carries multiple "Speaker:" turn
            #       markers (i.e. an upstream stage genuinely saw >1 speaker).
            long_gaps = 0
            if transcript_words and len(transcript_words) > 20:
                for i in range(len(transcript_words) - 1):
                    gap = float(transcript_words[i + 1].get("start", 0.0)) - float(
                        transcript_words[i].get("end", 0.0)
                    )
                    if gap >= 2.5:
                        long_gaps += 1

            turn_markers = 0
            if transcript_text:
                turn_markers = sum(
                    1 for line in transcript_text.splitlines()
                    if line.strip() and ":" in line.strip()[:40]
                    and not line.strip()[:40].replace(":", "").replace(" ", "").isdigit()
                )

            if long_gaps >= 3 and turn_markers >= 4:
                suspected_collapse = True
                collapse_reason = (
                    f"{long_gaps} long speech pauses (>2.5s) AND {turn_markers} "
                    f"speaker-turn lines in the transcript, but diarization "
                    f"produced only 1 speaker."
                )

        if suspected_collapse:
            logger.warning(f"[DiarizationValidator] Speaker collapse suspected: {collapse_reason}")
            return DiarizationValidationResult(
                passed=False,
                status="DIARIZATION_SUSPECTED_FAILURE",
                failure_reason=collapse_reason,
                metric_type="RUNTIME_DIAGNOSTIC",
                diagnostics={
                    "speaker_count": speaker_count,
                    "total_speech_dur": round(total_dur, 2),
                    "suspected_collapse": True,
                },
            )

        # Calculate speaker duration breakdown
        spk_durations: Dict[str, float] = {}
        spk_turns: Dict[str, int] = {}
        prev = None
        for s in segments:
            spk = s.get("speaker", "Speaker_1")
            dur = max(0.0, float(s.get("end_time", 0.0)) - float(s.get("start_time", 0.0)))
            spk_durations[spk] = spk_durations.get(spk, 0.0) + dur
            if spk != prev:
                spk_turns[spk] = spk_turns.get(spk, 0) + 1
            prev = spk

        # Over-split diagnostic (informational — clustering owns the fix, this
        # only surfaces it in telemetry). A dominant speaker plus one or more
        # others that each have a trivial share AND only a single turn is the
        # signature of one voice fragmented into extra speakers.
        suspected_oversplit = False
        if speaker_count >= 2 and total_dur > 0:
            shares = sorted((d / total_dur for d in spk_durations.values()), reverse=True)
            minor = [
                spk for spk, d in spk_durations.items()
                if (d / total_dur) < 0.05 and spk_turns.get(spk, 0) <= 1
            ]
            if shares[0] >= 0.80 and minor:
                suspected_oversplit = True
                logger.warning(
                    "[DiarizationValidator] Possible speaker over-split: dominant "
                    "speaker holds %.0f%% of talk time; %d minor speaker(s) with a "
                    "single short turn — %s.",
                    shares[0] * 100.0, len(minor), ", ".join(sorted(minor)),
                )

        return DiarizationValidationResult(
            passed=True,
            status="SUCCESS",
            failure_reason=None,
            metric_type="RUNTIME_DIAGNOSTIC",
            diagnostics={
                "speaker_count": speaker_count,
                "segment_count": len(segments),
                "total_speech_duration_sec": round(total_dur, 2),
                "speaker_durations": {k: round(v, 2) for k, v in spk_durations.items()},
                "suspected_oversplit": suspected_oversplit,
            },
        )


diarization_validator = DiarizationQualityValidator()

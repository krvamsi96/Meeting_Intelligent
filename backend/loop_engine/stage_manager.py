"""Stage State & Retry Audit Management for Universal Loop Engineering.

Enforces strict lifecycle states across all quality-controlled pipeline stages:
- PENDING
- PROCESSING
- SUCCESS
- DEGRADED
- FAILED
- RETRYING
- REJECTED
- COMPLETED

Guarantees:
- A FAILED component NEVER becomes SUCCESS without passing explicit re-validation.
- Retries are bounded by MAX_STAGE_RETRIES (no infinite loops).
- Full audit history of attempts, failure reasons, and correction strategies is logged.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.config.config import settings

logger = logging.getLogger(__name__)


class StageState(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"


ALL_PIPELINE_STAGES = [
    "audio_extraction",
    "stt",
    "diarization",
    "alignment",
    "transcript_correction",
    "summary",
    "mom",
    "agenda",
    "decision",
    "action_items",
    "topics",
    "sentiment",
    "embeddings_vector",
    "rag_chatbot",
]


class StageTracker:
    """Manages stage state transitions, retry attempts, and state persistence for a meeting."""

    def __init__(self, meeting: Any, db: Any):
        self.meeting = meeting
        self.db = db
        self.max_retries = getattr(settings, "MAX_STAGE_RETRIES", 2)
        self.states: Dict[str, str] = self._load_states()
        self.retry_history: List[Dict[str, Any]] = self._load_retry_history()
        self.attempt_counts: Dict[str, int] = {}

    def _load_states(self) -> Dict[str, str]:
        raw = getattr(self.meeting, "pipeline_stage_states_json", None)
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception as e:
                logger.warning(f"[StageTracker] Failed to parse pipeline_stage_states_json: {e}")
        
        # Initial default states
        return {stage: StageState.PENDING.value for stage in ALL_PIPELINE_STAGES}

    def _load_retry_history(self) -> List[Dict[str, Any]]:
        raw = getattr(self.meeting, "retry_history_json", None)
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
            except Exception as e:
                logger.warning(f"[StageTracker] Failed to parse retry_history_json: {e}")
        return []

    def save(self, commit: bool = True):
        """Persist current stage states and retry history to the meeting record.

        ``commit=False`` only stages the change on the meeting row; the caller's own commit (which it needs anyway)
        writes it. A failed commit is rolled back and RAISED: the previous version logged it and carried on, so the
        pipeline went on believing the stage states were stored.
        """
        self.meeting.pipeline_stage_states_json = json.dumps(self.states)
        self.meeting.retry_history_json = json.dumps(self.retry_history)
        if commit and self.db:
            try:
                self.db.commit()
            except Exception as e:
                logger.error(f"[StageTracker] DB commit failed: {e}")
                try:
                    self.db.rollback()
                except Exception as rollback_err:
                    logger.error(f"[StageTracker] Rollback after a failed commit also failed: {rollback_err}")
                raise

    def transition(self, stage: str, new_state: StageState, reason: str = "", *, commit: bool = True):
        """Execute a safe state transition for a pipeline stage.

        Every transition used to commit on its own (15 commits per meeting, each expiring the meeting row so the next
        attribute read re-SELECTed it). Callers that make several transitions pass ``commit=False`` and let their own
        commit persist them.
        """
        if stage not in ALL_PIPELINE_STAGES:
            ALL_PIPELINE_STAGES.append(stage)
        
        old_state = self.states.get(stage, StageState.PENDING.value)

        # Protection rule: A FAILED stage CANNOT silently become SUCCESS without RETRYING pass
        if old_state == StageState.FAILED.value and new_state == StageState.SUCCESS.value:
            curr_attempts = self.attempt_counts.get(stage, 0)
            if curr_attempts == 0:
                logger.warning(
                    f"[StageTracker] Rejected invalid direct transition {stage}: FAILED -> SUCCESS without retry revalidation!"
                )
                new_state = StageState.DEGRADED

        self.states[stage] = new_state.value
        logger.info(f"[StageTracker] Meeting {self.meeting.id} | Stage '{stage}': {old_state} -> {new_state.value} ({reason})")
        self.save(commit=commit)

    def get_state(self, stage: str) -> StageState:
        val = self.states.get(stage, StageState.PENDING.value)
        return StageState(val)

    def can_retry(self, stage: str) -> bool:
        """Check if stage has remaining retries allowed under MAX_STAGE_RETRIES."""
        attempts = self.attempt_counts.get(stage, 0)
        return attempts < self.max_retries

    def record_retry_attempt(
        self,
        stage: str,
        failure_reason: str,
        failed_validation: str,
        correction_strategy: str,
        result: str,
    ) -> int:
        """Log structured retry metadata and increment attempt counter."""
        attempt = self.attempt_counts.get(stage, 0) + 1
        self.attempt_counts[stage] = attempt

        entry = {
            "stage": stage,
            "attempt": attempt,
            "max_retries": self.max_retries,
            "failure_reason": failure_reason,
            "failed_validation": failed_validation,
            "correction_strategy": correction_strategy,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.retry_history.append(entry)
        self.transition(stage, StageState.RETRYING, f"Attempt {attempt}/{self.max_retries}: {correction_strategy}")
        return attempt

    def evaluate_completion(self, overall_status: Optional[str] = None) -> Dict[str, Optional[str]]:
        """The meeting's final ``status`` / ``overall_status`` / ``error_message`` implied by the stage states.

        Pure decision, nothing is written: the pipeline applies it together with the final JSON in ONE commit, so the
        meeting is never reported ``completed`` before its final record exists (and never flips completed -> error
        afterwards, which the old "mark completed, then evaluate" order allowed).

        Any FAILED stage -> ``error``; otherwise ``completed`` (SUCCESS, or COMPLETED_WITH_WARNINGS when a stage is
        DEGRADED). Stages still PENDING/PROCESSING at that point are settled to SUCCESS.
        """
        failed_stages = [s for s, state in self.states.items() if state == StageState.FAILED.value]
        degraded_stages = [s for s, state in self.states.items() if state == StageState.DEGRADED.value]

        if failed_stages:
            return {
                "status": "error",
                "overall_status": "FAILED",
                "error_message": f"Critical pipeline failure in stages: {', '.join(failed_stages)}",
            }
        for s in ALL_PIPELINE_STAGES:
            if self.states.get(s) in (StageState.PENDING.value, StageState.PROCESSING.value):
                self.states[s] = StageState.SUCCESS.value
        return {
            "status": "completed",
            "overall_status": overall_status or ("COMPLETED_WITH_WARNINGS" if degraded_stages else "SUCCESS"),
            "error_message": None,
        }

    def mark_completed(self, overall_status: Optional[str] = None, *, commit: bool = True):
        """Finalize pipeline processing state (evaluate, apply to the meeting row, persist)."""
        outcome = self.evaluate_completion(overall_status)
        self.meeting.status = outcome["status"]
        self.meeting.overall_status = outcome["overall_status"]
        if outcome["error_message"]:
            self.meeting.error_message = outcome["error_message"]
        self.save(commit=commit)

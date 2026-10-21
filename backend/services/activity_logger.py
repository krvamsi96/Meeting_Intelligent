"""Activity Logger — records every AI pipeline step to AI_Meeting_Activty_Logs.

This module is the single source of truth for writing activity log records.
It is called from pipeline workers (sync context) and uses the same DB session
architecture as the rest of the pipeline (pipeline_db_session / passed-in db).

Key rules
---------
- NEVER store credentials, API keys, tokens, or secrets in any field.
- details_json is always sanitised through _safe_details() before write.
- bharatvc_meeting_id == meeting_id is enforced for every row.
- processing_id is scoped to one pipeline execution run; it is NOT a meeting ID.

Processing ID format
--------------------
    PROC-{meeting_pk}-{uuid4[:8].upper()}
    e.g. PROC-42-A3F7B1C2

Step order constants
--------------------
    See STEP_ORDER dict. Steps not listed default to order 99.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Centralized Feature & Step Order Catalogue (Single Source of Truth) ──────
# Required 18-feature logical order:
# 1. Video / Input Processing
# 2. Audio Processing
# 3. Transcription
# 4. Speaker Identification / Diarization
# 5. Speaker Duration
# 6. Transcript + Speaker Mapping
# 7. Agenda
# 8. Decision
# 9. Agenda → Decision Mapping
# 10. Action Items
# 11. Summary
# 12. MOM / Minutes of Meeting
# 13. Sentiment Analysis
# 14. Metrics
# 15. RAG
# 16. AI Chatbot
# 17. Database / Persistence
# 18. Final Validation

FEATURE_ORDER: Dict[str, int] = {
    # 1. Video / Input Processing
    "video_processing":             1,
    "meeting_upload":               1,
    "input_processing":             1,
    "document_text_extraction":     1,

    # 2. Audio Processing
    "audio_processing":             2,
    "audio_extraction":             2,
    "audio_preprocessing":          2,
    "voice_activity_detection":     2,
    "document_metadata_extraction": 2,
    "language_detection":           2,

    # 3. Transcription
    "transcription":                3,
    "speech_to_text":               3,

    # 4. Speaker Identification / Diarization
    "speaker_diarization":          4,
    "speaker_identification":       4,

    # 5. Speaker Duration
    "speaker_duration":             5,
    "speaker_timeline":             5,

    # 6. Transcript + Speaker Mapping
    "transcript_correction":        6,
    "speaker_mapping":              6,
    "transcript_save":              6,
    "document_segments_creation":   6,
    "translation":                  6,

    # 7. Agenda
    "agenda":                       7,
    "agenda_extraction":            7,

    # 8. Decision
    "decision":                     8,
    "decision_extraction":          8,

    # 9. Agenda → Decision Mapping
    "decision_mapping":             9,
    "agenda_decision_mapping":      9,

    # 10. Action Items
    "action_items":                 10,
    "action_item_extraction":       10,

    # 11. Summary
    "summary":                      11,
    "summary_generation":           11,

    # 12. MOM / Minutes of Meeting
    "mom":                          12,
    "mom_generation":               12,

    # 13. Sentiment Analysis
    "sentiment":                    13,
    "sentiment_analysis":           13,

    # 14. Metrics
    "metrics":                      14,
    "ai_metrics":                   14,
    "topic_detection":              14,

    # 15. RAG
    "rag":                          15,
    "embedding_generation":         15,
    "vector_indexing":              15,

    # 16. AI Chatbot
    "ai_chatbot":                   16,
    "chatbot_indexing":             16,
    "similar_meeting_search":       16,

    # 17. Database / Persistence
    "database":                     17,
    "data_persistence":             17,
    "db_save":                      17,

    # 18. Final Validation
    "final_validation":             18,
    "final_output":                 18,
}

# Alias for backward compatibility
STEP_ORDER: Dict[str, int] = FEATURE_ORDER

TOTAL_STEPS = 18

# Keys that must never appear in details_json (security)
_FORBIDDEN_KEYS = frozenset({
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "authorization", "auth", "credential", "private_key", "access_key",
    "db_url", "database_url", "connection_string", "dsn",
})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure dt is timezone-aware UTC, taking into account DB local time if naive."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        return dt.replace(tzinfo=local_tz).astimezone(timezone.utc)
    return dt.astimezone(timezone.utc)


def _get_duration(started: Optional[datetime], now: datetime) -> Optional[float]:
    """Safely calculate non-negative duration in seconds between started and now."""
    if not started:
        return None
    try:
        s = _as_aware(started)
        n = _as_aware(now)
        if s and n:
            dur = (n - s).total_seconds()
            return round(max(0.0, dur), 3)
    except Exception:
        pass
    return None


def _safe_details(raw: Optional[Dict[str, Any]]) -> Optional[str]:
    """Serialize details dict to JSON string, stripping any sensitive keys."""
    if not raw:
        return None
    safe = {
        k: v for k, v in raw.items()
        if k.lower() not in _FORBIDDEN_KEYS
    }
    try:
        return json.dumps(safe, default=str)
    except Exception:
        return None


def _new_processing_id(meeting_pk: int) -> str:
    """Generate a unique processing_id for one pipeline run."""
    suffix = uuid.uuid4().hex[:8].upper()
    return f"PROC-{meeting_pk}-{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# ActivityLogger
# ─────────────────────────────────────────────────────────────────────────────

class ActivityLogger:
    """Logs pipeline step activity to AI_Meeting_Activty_Logs.

    Designed to be instantiated once per pipeline execution, then passed down
    to sub-functions (_run_shared_pipeline etc.) or used via module-level
    helper functions.

    Parameters
    ----------
    meeting_pk       : meetings.id (integer PK)
    bharatvc_id      : meetings.bharatvc_meeting_id (string)
    meeting_id_str   : meetings.meeting_id (string; should equal bharatvc_id)
    processing_id    : Existing PROC-* string, or None to auto-generate one
    db               : Optional SQLAlchemy session — if None, each write opens
                       its own pipeline_db_session() context.
    """

    def __init__(
        self,
        meeting_pk: int,
        bharatvc_id: Optional[str],
        meeting_id_str: Optional[str],
        processing_id: Optional[str] = None,
        db=None,
    ) -> None:
        self.meeting_pk = meeting_pk
        self.bharatvc_meeting_id = bharatvc_id or str(meeting_pk)
        self.meeting_id_str = meeting_id_str or self.bharatvc_meeting_id
        self.processing_id = processing_id or _new_processing_id(meeting_pk)
        self._db = db  # may be None — each operation falls back to pipeline_db_session
        # (stage, seconds, outcome) for this run, printed as one table by log_timing_summary().
        self._timings: list = []

        logger.info(
            "[ActivityLogger] Initialized | meeting_pk=%s bvc_id=%s proc_id=%s",
            self.meeting_pk, self.bharatvc_meeting_id, self.processing_id,
        )

    # ── Internal DB helpers ────────────────────────────────────────────────────

    def _write(self, fn):
        """Execute fn(db) using the injected session or a fresh pipeline_db_session."""
        if self._db is not None:
            try:
                return fn(self._db)
            except Exception as e:
                logger.warning("[ActivityLogger] Write error (injected db): %s", e)
                return None
        else:
            from backend.database.connection import pipeline_db_session
            try:
                with pipeline_db_session() as db:
                    result = fn(db)
                    db.commit()
                    return result
            except Exception as e:
                logger.warning("[ActivityLogger] Write error (pipeline_db_session): %s", e)
                return None

    def _push_sse(self, log_record) -> None:
        """Push an SSE event to any active browser subscribers (fire-and-forget)."""
        try:
            from backend.services.sse_manager import sse_manager
            payload = {
                "event": "step_update",
                "meeting_pk": self.meeting_pk,
                "bharatvc_meeting_id": self.bharatvc_meeting_id,
                "meeting_id": self.meeting_id_str,
                "processing_id": self.processing_id,
                "step_name": log_record.step_name,
                "step_order": log_record.step_order,
                "status": log_record.status,
                "message": log_record.message,
                "started_at": log_record.started_at.isoformat() if log_record.started_at else None,
                "completed_at": log_record.completed_at.isoformat() if log_record.completed_at else None,
                "duration_seconds": log_record.duration_seconds,
                "progress_percent": log_record.progress_percent,
                "error_code": log_record.error_code,
                "error_message": log_record.error_message,
            }
            sse_manager.publish_sync(self.meeting_pk, payload)
        except Exception as sse_err:
            logger.debug("[ActivityLogger] SSE push skipped: %s", sse_err)

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_timing(self, stage: str, seconds: float, outcome: str = "ok") -> None:
        """Add a measured duration that has no activity-log step of its own (e.g. one parallel LLM section)."""
        self._timings.append((stage, float(seconds or 0.0), outcome))

    def log_timing_summary(self, total_sec: float, status: str = "") -> None:
        """Log every measured stage of this run, in order, as one table ending in the total."""
        width = max([len(name) for name, _, _ in self._timings] + [24])
        lines = [f"{name.upper():<{width}} {secs:>9.2f}s{'' if outcome == 'ok' else '  ' + outcome}"
                 for name, secs, outcome in self._timings]
        lines.append(f"{'TOTAL':<{width}} {float(total_sec or 0.0):>9.2f}s{('  ' + status) if status else ''}")
        logger.info("[PipelineTiming] meeting_pk=%s proc=%s\n%s", self.meeting_pk, self.processing_id, "\n".join(lines))

    def start_step(
        self,
        step_name: str,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        step_order: Optional[int] = None,
    ) -> Optional[int]:
        """Create a PROCESSING log entry. Returns the new log record id."""
        from backend.models.models import MeetingActivityLog

        order = step_order if step_order is not None else STEP_ORDER.get(step_name, 99)
        progress = round((order / TOTAL_STEPS) * 100.0, 1)
        now = _utcnow()

        def _create(db):
            record = MeetingActivityLog(
                meeting_pk=self.meeting_pk,
                bharatvc_meeting_id=self.bharatvc_meeting_id,
                meeting_id=self.meeting_id_str,
                processing_id=self.processing_id,
                step_name=step_name,
                step_order=order,
                status="PROCESSING",
                message=message or f"Starting {step_name.replace('_', ' ').title()}",
                started_at=now,
                completed_at=None,
                duration_seconds=None,
                progress_percent=progress,
                details_json=_safe_details(details),
                error_code=None,
                error_message=None,
                created_at=now,
                updated_at=now,
            )
            db.add(record)
            db.flush()
            record_id = record.id
            self._push_sse(record)
            logger.info(
                "[ActivityLogger] START step=%s proc=%s order=%d",
                step_name, self.processing_id, order,
            )
            return record_id

        return self._write(_create)

    def complete_step(
        self,
        log_id: int,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update an existing log record to COMPLETED, compute duration."""
        from backend.models.models import MeetingActivityLog

        now = _utcnow()

        def _update(db):
            record = db.query(MeetingActivityLog).filter(
                MeetingActivityLog.id == log_id
            ).first()
            if not record:
                return
            duration = _get_duration(record.started_at, now)
            record.status = "COMPLETED"
            record.completed_at = now
            record.duration_seconds = duration
            self._timings.append((record.step_name, float(duration or 0.0), "ok"))
            record.message = message or f"{record.step_name.replace('_', ' ').title()} completed"
            record.updated_at = now
            if details:
                merged = {}
                if record.details_json:
                    try:
                        merged = json.loads(record.details_json)
                    except Exception:
                        pass
                merged.update(details)
                record.details_json = _safe_details(merged)
            db.flush()
            self._push_sse(record)
            logger.info(
                "[ActivityLogger] COMPLETE step=%s dur=%.3fs proc=%s",
                record.step_name,
                duration or 0.0,
                self.processing_id,
            )

        self._write(_update)

    def fail_step(
        self,
        log_id: int,
        error_code: str = "PIPELINE_ERROR",
        error_message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update an existing log record to FAILED."""
        from backend.models.models import MeetingActivityLog

        now = _utcnow()
        # Strip any credential-like content from error message
        safe_error = _strip_credentials(error_message)

        def _update(db):
            record = db.query(MeetingActivityLog).filter(
                MeetingActivityLog.id == log_id
            ).first()
            if not record:
                return
            duration = _get_duration(record.started_at, now)
            record.status = "FAILED"
            record.completed_at = now
            record.duration_seconds = duration
            self._timings.append((record.step_name, float(duration or 0.0), "FAILED"))
            record.message = f"{record.step_name.replace('_', ' ').title()} failed"
            record.error_code = error_code
            record.error_message = safe_error[:2000]  # cap length
            record.updated_at = now
            if details:
                merged = {}
                if record.details_json:
                    try:
                        merged = json.loads(record.details_json)
                    except Exception:
                        pass
                merged.update(details)
                record.details_json = _safe_details(merged)
            db.flush()
            self._push_sse(record)
            logger.info(
                "[ActivityLogger] FAIL step=%s code=%s proc=%s",
                record.step_name, error_code, self.processing_id,
            )

        self._write(_update)

    def retry_step(
        self,
        step_name: str,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Create a new RETRYING record (does NOT overwrite the FAILED record)."""
        from backend.models.models import MeetingActivityLog

        order = STEP_ORDER.get(step_name, 99)
        progress = round((order / TOTAL_STEPS) * 100.0, 1)
        now = _utcnow()

        def _create(db):
            record = MeetingActivityLog(
                meeting_pk=self.meeting_pk,
                bharatvc_meeting_id=self.bharatvc_meeting_id,
                meeting_id=self.meeting_id_str,
                processing_id=self.processing_id,
                step_name=step_name,
                step_order=order,
                status="RETRYING",
                message=message or f"Retrying {step_name.replace('_', ' ').title()}",
                started_at=now,
                completed_at=None,
                duration_seconds=None,
                progress_percent=progress,
                details_json=_safe_details(details),
                created_at=now,
                updated_at=now,
            )
            db.add(record)
            db.flush()
            self._push_sse(record)
            logger.info("[ActivityLogger] RETRY step=%s proc=%s", step_name, self.processing_id)
            return record.id

        return self._write(_create)

    def skip_step(
        self,
        step_name: str,
        reason: str = "Not applicable for this meeting",
        step_order: Optional[int] = None,
    ) -> None:
        """Create a SKIPPED log entry (no start/complete cycle)."""
        from backend.models.models import MeetingActivityLog

        order = step_order if step_order is not None else STEP_ORDER.get(step_name, 99)
        progress = round((order / TOTAL_STEPS) * 100.0, 1)
        now = _utcnow()

        def _create(db):
            record = MeetingActivityLog(
                meeting_pk=self.meeting_pk,
                bharatvc_meeting_id=self.bharatvc_meeting_id,
                meeting_id=self.meeting_id_str,
                processing_id=self.processing_id,
                step_name=step_name,
                step_order=order,
                status="SKIPPED",
                message=reason,
                started_at=now,
                completed_at=now,
                duration_seconds=0.0,
                progress_percent=progress,
                created_at=now,
                updated_at=now,
            )
            db.add(record)
            db.flush()
            self._push_sse(record)
            logger.info("[ActivityLogger] SKIP step=%s proc=%s", step_name, self.processing_id)

        self._write(_create)

    def cancel_remaining(self, from_step_order: int = 0) -> None:
        """Mark all PENDING steps at or after from_step_order as CANCELLED.

        Called when the pipeline hits a fatal error mid-run to clean up
        any steps that were pre-created as PENDING.
        """
        from backend.models.models import MeetingActivityLog

        now = _utcnow()

        def _cancel(db):
            records = db.query(MeetingActivityLog).filter(
                MeetingActivityLog.processing_id == self.processing_id,
                MeetingActivityLog.status == "PENDING",
                MeetingActivityLog.step_order >= from_step_order,
            ).all()
            for r in records:
                r.status = "CANCELLED"
                r.completed_at = now
                r.updated_at = now
            db.flush()

        self._write(_cancel)

    def overall_progress(self) -> float:
        """Return aggregate progress % across all steps for this processing run."""
        from backend.models.models import MeetingActivityLog

        def _calc(db):
            records = db.query(MeetingActivityLog).filter(
                MeetingActivityLog.processing_id == self.processing_id,
            ).all()
            if not records:
                return 0.0
            completed = sum(1 for r in records if r.status == "COMPLETED")
            total = TOTAL_STEPS
            return round((completed / total) * 100.0, 1)

        result = self._write(_calc)
        return result if result is not None else 0.0

    # Aliases supporting feature-based naming
    start_feature = start_step
    complete_feature = complete_step
    fail_feature = fail_step
    skip_feature = skip_step


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_credentials(msg: str) -> str:
    """Remove obvious credential patterns from an error message string."""
    import re
    # Remove password= / password: / password:value patterns
    msg = re.sub(r'(?i)(password|passwd|secret|api[_-]?key|token)\s*[=:]\s*\S+', r'\1=***', msg)
    # Remove postgresql://user:pass@host style DSNs (capture user:pass@ part)
    msg = re.sub(r'(?i)(postgres(?:ql)?|mysql|sqlite)://[^/\s]*@', '[DB_URL_REDACTED]@', msg)
    return msg


def get_or_create_processing_id(meeting_pk: int, db) -> str:
    """Return the latest active processing_id for a meeting, or create a new one.

    "Active" means the most recent PROCESSING or RETRYING record in the last
    30 minutes. If none found, a new PROC-* ID is generated.
    """
    from datetime import timedelta
    from backend.models.models import MeetingActivityLog

    cutoff = _utcnow() - timedelta(minutes=30)
    active = (
        db.query(MeetingActivityLog.processing_id)
        .filter(
            MeetingActivityLog.meeting_pk == meeting_pk,
            MeetingActivityLog.status.in_(["PROCESSING", "RETRYING"]),
            MeetingActivityLog.created_at >= cutoff,
        )
        .order_by(MeetingActivityLog.created_at.desc())
        .first()
    )
    if active:
        return active[0]
    return _new_processing_id(meeting_pk)


def build_activity_logger_for_meeting(meeting, db=None) -> ActivityLogger:
    """Convenience factory: create an ActivityLogger from a Meeting ORM object."""
    return ActivityLogger(
        meeting_pk=meeting.id,
        bharatvc_id=meeting.bharatvc_meeting_id,
        meeting_id_str=meeting.meeting_id,
        db=db,
    )

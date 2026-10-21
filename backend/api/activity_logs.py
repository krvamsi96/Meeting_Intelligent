"""Activity Logs API — REST + SSE endpoints for pipeline processing visibility.

Routes
------
GET  /meetings/{meeting_id}/activity-logs
     Returns current/latest processing run logs for a meeting.
     Accepts both integer meetings.id and string bharatvc_meeting_id.

GET  /meetings/{meeting_id}/activity-logs/processing-runs
     Returns all distinct processing runs for a meeting (most recent first).

GET  /meetings/{meeting_id}/activity-logs/stream
     Server-Sent Events stream — pushes real-time step updates to the browser.
     On connect, replays the full latest run immediately, then streams live.

POST /meetings/{meeting_id}/activity-logs/new-run
     Force-create a new processing_id (for manually triggered reprocessing).

GET  /activity-logs/demo
     Serves the self-contained HTML demo page.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.models.models import Meeting, MeetingActivityLog
from backend.schemas.schemas import (
    ActivityLogEntry,
    ActivityLogsResponse,
    ProcessingRunSummary,
)
from backend.services.activity_logger import (
    _get_duration,
    _new_processing_id,
)
from backend.services.sse_manager import sse_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/meetings", tags=["Activity Logs"])

# meetings.status values after which no pipeline run is publishing for the meeting.
_TERMINAL_MEETING_STATUSES = frozenset({"completed", "error", "failed"})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_meeting(meeting_id_param: str, db: Session) -> Meeting:
    """Resolve meeting_id_param as integer PK or any public id form.

    Delegates to MeetingRepository.get_by_id so this endpoint accepts exactly the id forms
    every other meeting endpoint accepts (including legacy ``BVC-``/``MTG-`` prefixes).
    """
    from backend.repositories import MeetingRepository

    meeting: Optional[Meeting] = MeetingRepository(db).get_by_id(meeting_id_param)

    if meeting is None:
        raise HTTPException(status_code=404, detail=f"Meeting '{meeting_id_param}' not found")
    return meeting


def _log_to_schema(record: MeetingActivityLog) -> ActivityLogEntry:
    """Convert ORM record to Pydantic schema, parsing details_json."""
    details = None
    if record.details_json:
        try:
            details = json.loads(record.details_json)
        except Exception:
            details = {"raw": record.details_json}

    return ActivityLogEntry(
        id=record.id,
        meeting_pk=record.meeting_pk,
        bharatvc_meeting_id=record.bharatvc_meeting_id,
        meeting_id=record.meeting_id,
        processing_id=record.processing_id,
        step_name=record.step_name,
        step_order=record.step_order,
        feature=record.step_name,
        feature_order=record.step_order,
        status=record.status,
        message=record.message,
        started_at=record.started_at,
        completed_at=record.completed_at,
        duration_seconds=record.duration_seconds,
        duration=record.duration_seconds,
        progress_percent=record.progress_percent,
        details=details,
        metadata=details,
        error_code=record.error_code,
        error_message=record.error_message,
        error=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _build_run_summary(processing_id: str, records: List[MeetingActivityLog]) -> ProcessingRunSummary:
    """Build a ProcessingRunSummary from a list of records for one processing_id."""
    if not records:
        return ProcessingRunSummary(
            processing_id=processing_id,
            overall_status="UNKNOWN",
            step_count=0,
            completed_steps=0,
            failed_steps=0,
            progress_percent=0.0,
        )

    completed = sum(1 for r in records if r.status == "COMPLETED")
    failed = sum(1 for r in records if r.status == "FAILED")
    processing = [r for r in records if r.status == "PROCESSING"]
    retrying = [r for r in records if r.status == "RETRYING"]

    # Check if final_output step is completed
    has_final_output = any(r.step_name == "final_output" and r.status == "COMPLETED" for r in records)

    # Overall status: active step -> PROCESSING, failed -> FAILED/PARTIAL, completed/final -> COMPLETED
    if processing or retrying:
        overall = "PROCESSING"
    elif failed:
        overall = "PARTIAL" if completed > 0 else "FAILED"
    elif has_final_output or (completed > 0 and completed == len([r for r in records if r.status not in ("SKIPPED", "CANCELLED")])):
        overall = "COMPLETED"
    else:
        overall = "UNKNOWN"

    started_times = [r.started_at for r in records if r.started_at]
    started_at = min(started_times) if started_times else None

    completed_times = [r.completed_at for r in records if r.completed_at and r.status == "COMPLETED"]
    last_completed = max(completed_times) if completed_times else None

    total_duration = None
    if started_at and last_completed:
        total_duration = _get_duration(started_at, last_completed)

    current_step = None
    active = processing or retrying
    if active:
        current_step = active[-1].step_name

    # Calculate dynamic progress
    if overall == "COMPLETED":
        progress = 100.0
    elif len(records) > 0:
        progress = round(min(100.0, (completed / max(len(records), 1)) * 100.0), 1)
    else:
        progress = 0.0

    return ProcessingRunSummary(
        processing_id=processing_id,
        overall_status=overall,
        step_count=len(records),
        completed_steps=completed,
        failed_steps=failed,
        started_at=started_at,
        completed_at=last_completed,
        total_duration_seconds=total_duration,
        current_step=current_step,
        progress_percent=progress,
    )


def _latest_processing_id(meeting_pk: int, db: Session) -> Optional[str]:
    """Return the most recent processing_id for a meeting."""
    row = (
        db.query(MeetingActivityLog.processing_id)
        .filter(MeetingActivityLog.meeting_pk == meeting_pk)
        .order_by(MeetingActivityLog.created_at.desc())
        .first()
    )
    return row[0] if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{meeting_id}/activity-logs",
    response_model=ActivityLogsResponse,
    summary="Get pipeline activity logs for a meeting",
)
def get_activity_logs(
    meeting_id: str,
    processing_id: Optional[str] = Query(None, description="Filter to a specific processing run"),
    limit: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """
    Return all activity log entries for a meeting.

    Accepts meeting_id as either:
    - Integer meetings.id          e.g. 42
    - String bharatvc_meeting_id   e.g. "MTG-001"

    If processing_id is omitted, returns the most recent run.
    Includes all distinct processing runs in processing_runs for run selection.
    """
    meeting = _resolve_meeting(meeting_id, db)

    # Determine which processing_id to use
    target_proc_id = processing_id or _latest_processing_id(meeting.id, db)

    if target_proc_id is None:
        # No logs yet — return empty response
        return ActivityLogsResponse(
            meeting_pk=meeting.id,
            bharatvc_meeting_id=meeting.bharatvc_meeting_id,
            meeting_id=meeting.meeting_id,
            processing_id="",
            overall_status="NO_LOGS",
            current_step=None,
            progress_percent=0.0,
            logs=[],
            processing_runs=[],
        )

    # Fetch records for the target run in deterministic feature-order + timestamp order
    records = (
        db.query(MeetingActivityLog)
        .filter(
            MeetingActivityLog.meeting_pk == meeting.id,
            MeetingActivityLog.processing_id == target_proc_id,
        )
        .order_by(
            MeetingActivityLog.step_order.asc(),
            MeetingActivityLog.created_at.asc(),
            MeetingActivityLog.id.asc(),
        )
        .limit(limit)
        .all()
    )

    summary = _build_run_summary(target_proc_id, records)

    # Fetch summaries for all distinct runs (for the dropdown)
    all_proc_ids = (
        db.query(MeetingActivityLog.processing_id)
        .filter(MeetingActivityLog.meeting_pk == meeting.id)
        .distinct()
        .order_by(MeetingActivityLog.processing_id.desc())
        .all()
    )

    run_summaries: List[ProcessingRunSummary] = []
    for (proc_id,) in all_proc_ids:
        if proc_id == target_proc_id:
            run_summaries.append(summary)
        else:
            run_recs = (
                db.query(MeetingActivityLog)
                .filter(
                    MeetingActivityLog.meeting_pk == meeting.id,
                    MeetingActivityLog.processing_id == proc_id,
                )
                .all()
            )
            run_summaries.append(_build_run_summary(proc_id, run_recs))

    return ActivityLogsResponse(
        meeting_pk=meeting.id,
        bharatvc_meeting_id=meeting.bharatvc_meeting_id,
        meeting_id=meeting.meeting_id,
        processing_id=target_proc_id,
        overall_status=summary.overall_status,
        current_step=summary.current_step,
        progress_percent=summary.progress_percent,
        total_duration_seconds=summary.total_duration_seconds,
        started_at=summary.started_at,
        completed_at=summary.completed_at,
        logs=[_log_to_schema(r) for r in records],
        processing_runs=run_summaries,
    )


@router.get(
    "/{meeting_id}/activity-logs/processing-runs",
    summary="List all processing runs for a meeting",
)
def list_processing_runs(
    meeting_id: str,
    db: Session = Depends(get_db),
):
    """Return all distinct processing run summaries for a meeting (newest first)."""
    meeting = _resolve_meeting(meeting_id, db)

    all_proc_ids = (
        db.query(MeetingActivityLog.processing_id)
        .filter(MeetingActivityLog.meeting_pk == meeting.id)
        .distinct()
        .order_by(MeetingActivityLog.processing_id.desc())
        .all()
    )

    summaries = []
    for (proc_id,) in all_proc_ids:
        recs = (
            db.query(MeetingActivityLog)
            .filter(
                MeetingActivityLog.meeting_pk == meeting.id,
                MeetingActivityLog.processing_id == proc_id,
            )
            .all()
        )
        summaries.append(_build_run_summary(proc_id, recs).model_dump())

    return {
        "meeting_pk": meeting.id,
        "bharatvc_meeting_id": meeting.bharatvc_meeting_id,
        "meeting_id": meeting.meeting_id,
        "run_count": len(summaries),
        "processing_runs": summaries,
    }


@router.post(
    "/{meeting_id}/activity-logs/new-run",
    summary="Force-create a new processing_id for a meeting",
    status_code=201,
)
def create_new_processing_run(
    meeting_id: str,
    db: Session = Depends(get_db),
):
    """Creates and returns a new processing_id for this meeting.

    Used when manually triggering reprocessing to ensure logs from the new run
    are separated from previous runs.
    """
    meeting = _resolve_meeting(meeting_id, db)
    new_proc_id = _new_processing_id(meeting.id)
    return {
        "meeting_pk": meeting.id,
        "bharatvc_meeting_id": meeting.bharatvc_meeting_id,
        "meeting_id": meeting.meeting_id,
        "processing_id": new_proc_id,
        "message": "New processing_id reserved. Pass it to the pipeline worker.",
    }


@router.get(
    "/{meeting_id}/activity-logs/stream",
    summary="Server-Sent Events stream for real-time activity log updates",
)
async def stream_activity_logs(
    meeting_id: str,
    processing_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Open an SSE connection for real-time pipeline step updates.

    On connect:
    1. Immediately replays all existing logs for the latest (or specified) run.
    2. Then streams live events as pipeline steps complete.

    Client reconnect: browser EventSource automatically reconnects. On reconnect,
    existing logs are replayed again, ensuring no events are missed.

    Close connection: browser closes the request or the pipeline run completes
    (the server sends a 'done' event).
    """
    meeting = _resolve_meeting(meeting_id, db)
    meeting_pk = meeting.id
    final_status, final_error = meeting.status, meeting.error_message

    # Snapshot existing logs for this run (replayed on connect)
    target_proc_id = processing_id or _latest_processing_id(meeting_pk, db)

    snapshot: List[Dict] = []
    if target_proc_id:
        records = (
            db.query(MeetingActivityLog)
            .filter(
                MeetingActivityLog.meeting_pk == meeting_pk,
                MeetingActivityLog.processing_id == target_proc_id,
            )
            .order_by(
                MeetingActivityLog.step_order.asc(),
                MeetingActivityLog.created_at.asc(),
                MeetingActivityLog.id.asc(),
            )
            .all()
        )
        for r in records:
            details = None
            if r.details_json:
                try:
                    details = json.loads(r.details_json)
                except Exception:
                    pass
            snapshot.append({
                "event": "step_update",
                "meeting_pk": meeting_pk,
                "bharatvc_meeting_id": meeting.bharatvc_meeting_id,
                "meeting_id": meeting.meeting_id,
                "processing_id": r.processing_id,
                "step_name": r.step_name,
                "step_order": r.step_order,
                "feature": r.step_name,
                "feature_order": r.step_order,
                "status": r.status,
                "message": r.message,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "duration_seconds": r.duration_seconds,
                "duration": r.duration_seconds,
                "progress_percent": r.progress_percent,
                "details": details,
                "metadata": details,
                "error_code": r.error_code,
                "error_message": r.error_message,
                "error": r.error_message,
            })

    async def event_stream():
        import asyncio

        # 1. Send connection confirmation event
        connect_payload = json.dumps({
            "event": "connected",
            "meeting_pk": meeting_pk,
            "bharatvc_meeting_id": meeting.bharatvc_meeting_id,
            "meeting_id": meeting.meeting_id,
            "processing_id": target_proc_id or "",
            "snapshot_count": len(snapshot),
        }, default=str)
        yield f"data: {connect_payload}\n\n"

        # 2. Replay snapshot
        for snap in snapshot:
            yield f"data: {json.dumps(snap, default=str)}\n\n"
            await asyncio.sleep(0)  # yield control

        # A run that already finished will never publish again, and its end-of-stream signal was sent before this
        # connection existed: report the outcome and end, instead of holding the client open on keepalives.
        if final_status in _TERMINAL_MEETING_STATUSES:
            yield "data: " + json.dumps({"event": "done", "meeting_pk": meeting_pk, "status": final_status,
                                         "error_message": final_error}, default=str) + "\n\n"
            return

        # 3. Stream live updates
        try:
            async for event in sse_manager.subscribe(meeting_pk):
                yield event
        except Exception as e:
            logger.debug("SSE stream closed for meeting_pk=%s: %s", meeting_pk, e)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Demo HTML page
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_HTML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "templates", "activity_logs.html"
)

_FALLBACK_DEMO_ROUTER = APIRouter(prefix="/activity-logs-demo", tags=["Activity Logs Demo"])


@_FALLBACK_DEMO_ROUTER.get("", response_class=HTMLResponse, include_in_schema=False)
def activity_logs_demo_page():
    """Serve the self-contained Activity Logs demo page."""
    html_path = os.path.abspath(_DEMO_HTML_PATH)
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h2>Activity Logs Demo</h2><p>HTML page not found. "
                "Place activity_logs.html in backend/templates/.</p>",
        status_code=200,
    )


# Register the demo sub-router on our main router prefix so it's auto-discovered
router.include_router(_FALLBACK_DEMO_ROUTER)

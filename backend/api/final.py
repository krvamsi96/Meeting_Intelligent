"""Final Consolidated Meeting API — GET /api/v1/meetings/{meeting_id}/final.

This endpoint reads from AI_Meeting_Inteligent_final and returns the complete
consolidated meeting data in a single response.

Design
------
- The endpoint is READ-ONLY: it never triggers re-processing or LLM calls.
- meeting_id can be any of: integer meetings.id, string meeting_id, bharatvc_meeting_id.
- Returns HTTP 202 when no final record exists yet (meeting may still be processing).
- Returns HTTP 200 with full JSON when processing_status = COMPLETED.
- Returns HTTP 200 with error_info when processing_status = FAILED (not a 5xx —
  the failure is domain-level, not a server error).
- Auto-discovered by backend.api.discover_routers() via the `router` variable.

Security
--------
- No secrets, credentials, or API keys are exposed.
- JSON fields from AI_Meeting_Inteligent_final were sanitized during aggregation.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.models.models import AIMeetingIntelFinal, Meeting, TranscriptSegment
from backend.schemas.schemas import (
    FinalActionItem,
    FinalActivityLogEntry,
    FinalAgendaDecisionMapping,
    FinalAgendaItem,
    FinalDecision,
    FinalMeetingMetadata,
    FinalMeetingResponse,
    FinalMetrics,
    FinalSentiment,
    FinalSpeaker,
    FinalSpeakerDuration,
    FinalTranscriptSegment,
)

router = APIRouter(prefix="/meetings", tags=["Final Consolidated"])

logger = logging.getLogger(__name__)


# ── Helper: resolve meeting identity ──────────────────────────────────────────

def _resolve_meeting_pk(meeting_id_str: str, db: Session) -> Optional[int]:
    """Resolve a meeting identifier (int PK, string meeting_id, or bharatvc_meeting_id)
    to the integer meetings.id. Returns None if not found.
    """
    # Try integer PK first
    try:
        pk = int(meeting_id_str)
        row = db.query(Meeting.id).filter(Meeting.id == pk).first()
        if row:
            return pk
    except (ValueError, TypeError):
        pass

    # Try string meeting_id
    row = db.query(Meeting.id).filter(Meeting.meeting_id == meeting_id_str).first()
    if row:
        return row[0]

    # Try bharatvc_meeting_id
    row = db.query(Meeting.id).filter(Meeting.bharatvc_meeting_id == meeting_id_str).first()
    if row:
        return row[0]

    # Try display_id
    row = db.query(Meeting.id).filter(Meeting.display_id == meeting_id_str).first()
    if row:
        return row[0]

    return None


# ── Helper: safely parse JSON stored in Text columns ──────────────────────────

def _parse_json(value: Optional[str], fallback: Any = None) -> Any:
    """Safely parse a JSON string column; return fallback on failure."""
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


# ── Helper: build FinalMeetingResponse from ORM row ──────────────────────────

def _build_response(record: AIMeetingIntelFinal) -> FinalMeetingResponse:
    """Convert an AIMeetingIntelFinal ORM row to the FinalMeetingResponse schema.

    All JSON Text columns are parsed; invalid/missing JSON gracefully falls back
    to empty lists or None.
    """
    # Parse all JSON columns
    transcript_data: List[Dict] = _parse_json(record.transcript, [])
    speakers_data: List[Dict] = _parse_json(record.speakers, [])
    speaker_duration_data: List[Dict] = _parse_json(record.speaker_duration, [])
    agenda_data: List[Dict] = _parse_json(record.agenda, [])
    decisions_data: List[Dict] = _parse_json(record.decisions, [])
    mapping_data: List[Dict] = _parse_json(record.agenda_decision_mapping, [])
    action_items_data: List[Dict] = _parse_json(record.action_items, [])
    sentiment_data: Dict = _parse_json(record.sentiment, {})
    metrics_data: Dict = _parse_json(record.metrics, {})
    activity_logs_data: List[Dict] = _parse_json(record.activity_logs, [])
    error_info_data: Optional[Dict] = _parse_json(record.error_info, None)

    # Build meeting metadata
    meeting_meta = FinalMeetingMetadata(
        meeting_pk=record.meeting_pk,
        meeting_id=record.meeting_id,
        bharatvc_meeting_id=record.bharatvc_meeting_id,
        display_id=record.display_id,
        title=record.title,
        meeting_date=record.meeting_date,
        meeting_time=record.meeting_time,
        duration_seconds=record.duration_seconds,
        duration_formatted=record.duration_formatted,
        status=record.status,
        participant_count=record.participant_count,
        participants=record.participants,
        speaker_count=record.speaker_count,
    )

    # Build typed lists from parsed JSON
    transcript_list = [
        FinalTranscriptSegment(
            speaker=seg.get("speaker"),
            start_time=seg.get("start_time"),
            end_time=seg.get("end_time"),
            text=seg.get("text", ""),
            confidence=seg.get("confidence"),
        )
        for seg in transcript_data if isinstance(seg, dict)
    ]

    speakers_list = [
        FinalSpeaker(
            speaker_id=sp.get("speaker_id", "Unknown"),
            label=sp.get("label", sp.get("speaker_id", "Unknown")),
            segment_count=sp.get("segment_count", 0),
            total_words=sp.get("total_words", 0),
        )
        for sp in speakers_data if isinstance(sp, dict)
    ]

    speaker_duration_list = [
        FinalSpeakerDuration(
            speaker_id=sd.get("speaker_id", "Unknown"),
            duration_seconds=float(sd.get("duration_seconds", 0.0)),
            percentage=float(sd.get("percentage", 0.0)),
            segment_count=sd.get("segment_count", 0),
        )
        for sd in speaker_duration_data if isinstance(sd, dict)
    ]

    agenda_list = [
        FinalAgendaItem(
            agenda_id=ag.get("agenda_id", f"A{i+1}"),
            title=ag.get("title", ""),
            type=ag.get("type", "inferred"),
            order=ag.get("order", i + 1),
            source=ag.get("source", "llm_detected"),
        )
        for i, ag in enumerate(agenda_data) if isinstance(ag, dict)
    ]

    decisions_list = [
        FinalDecision(
            agenda_id=dec.get("agenda_id", "A1"),
            decision=dec.get("decision", ""),
            reason=dec.get("reason"),
            evidence=dec.get("evidence"),
            speaker=dec.get("speaker"),
            timestamp=dec.get("timestamp"),
            confidence=dec.get("confidence"),
            match_score=dec.get("match_score"),
        )
        for dec in decisions_data if isinstance(dec, dict)
    ]

    mapping_list = [
        FinalAgendaDecisionMapping(
            agenda_id=m.get("agenda_id", "A1"),
            agenda_title=m.get("agenda_title", ""),
            decision=m.get("decision", ""),
            has_decision=bool(m.get("has_decision", False)),
            match_score=m.get("match_score"),
            confidence_level=m.get("confidence_level"),
            is_grounded=bool(m.get("is_grounded", False)),
            evidence=m.get("evidence"),
            speaker=m.get("speaker"),
            timestamp=m.get("timestamp"),
        )
        for m in mapping_data if isinstance(m, dict)
    ]

    action_items_list = []
    for ai in action_items_data:
        if not isinstance(ai, dict):
            continue
        task = ai.get("task", "")
        if not task or not str(task).strip():
            continue  # skip invalid rows
        action_items_list.append(
            FinalActionItem(
                item_order=ai.get("item_order"),
                task=str(task).strip(),
                description=ai.get("description"),
                owner=ai.get("owner"),
                deadline=ai.get("deadline"),
                deadline_type=ai.get("deadline_type"),
                due_date=ai.get("due_date"),
                priority=ai.get("priority"),
                status=ai.get("status", "open"),
                evidence=ai.get("evidence"),
                speaker=ai.get("speaker"),
                timestamp_start=ai.get("timestamp_start"),
                timestamp_end=ai.get("timestamp_end"),
                confidence=ai.get("confidence"),
                agenda_ref=ai.get("agenda_ref"),
                decision_ref=ai.get("decision_ref"),
            )
        )

    # Sentiment
    sentiment_obj = None
    if sentiment_data and isinstance(sentiment_data, dict):
        from backend.schemas.schemas import FinalSpeakerSentiment
        spk_breakdown = [
            FinalSpeakerSentiment(
                speaker_id=sb.get("speaker_id", "Unknown"),
                sentiment=sb.get("sentiment"),
                confidence=sb.get("confidence"),
                emotions=sb.get("emotions"),
            )
            for sb in sentiment_data.get("speaker_breakdown", [])
            if isinstance(sb, dict)
        ]
        sentiment_obj = FinalSentiment(
            label=sentiment_data.get("label"),
            emotions=sentiment_data.get("emotions"),
            speaker_breakdown=spk_breakdown,
            methodology=sentiment_data.get("methodology"),
        )

    # Metrics
    metrics_obj = None
    if metrics_data and isinstance(metrics_data, dict):
        metrics_obj = FinalMetrics(
            reference_based=metrics_data.get("reference_based"),
            reference_free=metrics_data.get("reference_free"),
            quality_scores=metrics_data.get("quality_scores"),
            technical=metrics_data.get("technical"),
            evaluation_type=metrics_data.get("evaluation_type"),
            ground_truth_available=metrics_data.get("ground_truth_available"),
            transcript_quality=metrics_data.get("transcript_quality"),
            summary_quality=metrics_data.get("summary_quality"),
            rag_groundedness=metrics_data.get("rag_groundedness"),
            method_version=metrics_data.get("method_version"),
        )

    # Activity logs
    activity_logs_list = [
        FinalActivityLogEntry(
            step_name=log.get("step_name", ""),
            step_order=log.get("step_order", 0),
            status=log.get("status", "UNKNOWN"),
            message=log.get("message"),
            started_at=log.get("started_at"),
            completed_at=log.get("completed_at"),
            duration_seconds=log.get("duration_seconds"),
        )
        for log in activity_logs_data if isinstance(log, dict)
    ]

    return FinalMeetingResponse(
        processing_status=record.processing_status,
        aggregation_version=record.aggregation_version,
        error_info=error_info_data,
        created_at=record.created_at,
        updated_at=record.updated_at,
        meeting=meeting_meta,
        transcript=transcript_list,
        speakers=speakers_list,
        speaker_duration=speaker_duration_list,
        agenda=agenda_list,
        decisions=decisions_list,
        agenda_decision_mapping=mapping_list,
        action_items=action_items_list,
        summary=record.summary,
        mom=record.mom,
        sentiment=sentiment_obj,
        metrics=metrics_obj,
        activity_logs=activity_logs_list,
    )


# ── Helper: language + source-language transcript (read from the meeting itself) ─────────────

def _attach_language_info(response: FinalMeetingResponse, meeting: Optional[Meeting], db: Session) -> FinalMeetingResponse:
    """Add the detected spoken language and each utterance's original-language text to the consolidated response.

    Additive and read-only: the aggregated record stores the English transcript only, so a client of /final could not
    tell which language was spoken or see what was actually said. Segments are matched by position (the aggregation
    writes them in id order) and only when the counts agree, so text can never be attached to the wrong utterance.
    """
    if meeting is None:
        return response
    info = response.meeting
    info.source_language = getattr(meeting, "source_language", None)
    info.source_language_name = getattr(meeting, "source_language_name", None)
    info.source_language_confidence = getattr(meeting, "source_language_confidence", None)
    info.translation_applied = getattr(meeting, "translation_applied", None)
    info.output_language = getattr(meeting, "output_language", None) or "en"
    try:
        rows = (
            db.query(TranscriptSegment.original_text, TranscriptSegment.raw_text)
            .filter(TranscriptSegment.meeting_id == meeting.id)
            .order_by(TranscriptSegment.id.asc())
            .all()
        )
        if len(rows) == len(response.transcript):
            for segment, (original, raw) in zip(response.transcript, rows):
                segment.original_text = original or raw or None
    except Exception as exc:  # noqa: BLE001 - language info is an enrichment, never a reason to fail the response
        logger.warning(f"[FinalAPI] Could not attach the original transcript for meeting {meeting.id}: {exc}")
    return response


# ── GET /api/v1/meetings/{meeting_id}/final ────────────────────────────────────

@router.get(
    "/{meeting_id}/final",
    response_model=FinalMeetingResponse,
    summary="Get Consolidated Final Meeting Data",
    description="Returns the complete consolidated meeting output from AI_Meeting_Inteligent_final.",
    responses={
        200: {"description": "Final consolidated meeting data"},
        202: {"description": "Final record not yet available; meeting may still be processing"},
        404: {"description": "Meeting not found"},
    },
)
def get_final_meeting(
    meeting_id: str,
    db: Session = Depends(get_db),
):
    """Retrieve the consolidated final meeting record.

    meeting_id can be:
    - Integer meetings.id (e.g. "42")
    - String meetings.meeting_id
    - String meetings.bharatvc_meeting_id

    Returns the full consolidated response when processing_status = COMPLETED.
    Returns the record with processing_status = FAILED (and error_info) when aggregation failed.
    Returns HTTP 202 when no final record has been created yet.
    """
    # Resolve identity FIRST, sequentially (int PK -> meeting_id -> bharatvc_meeting_id
    # -> display_id), then look up AI_Meeting_Inteligent_final by meeting_pk ALONE.
    #
    # The previous implementation OR'd meeting_pk together with three denormalized,
    # non-unique string columns (meeting_id, bharatvc_meeting_id, display_id) in a
    # single query. Those columns carry only an index, not a uniqueness constraint,
    # so a caller-supplied external id that happened to collide with a DIFFERENT
    # meeting's PK/string id could match two rows at once, and `.first()` could
    # silently return the wrong meeting's transcript/agenda/decisions/summary.
    meeting_pk = _resolve_meeting_pk(meeting_id, db)
    final_record = None
    meeting = None
    if meeting_pk is not None:
        final_record = (
            db.query(AIMeetingIntelFinal)
            .filter(AIMeetingIntelFinal.meeting_pk == meeting_pk)
            .first()
        )
    if meeting_pk is not None:
        meeting = db.query(Meeting).filter(Meeting.id == meeting_pk).first()
        if meeting and meeting.status == "completed":
            if final_record is None or (final_record.processing_status or "") != "COMPLETED":
                try:
                    from backend.services.final_aggregation_service import build_final_record
                    final_record = build_final_record(meeting.id, db)
                    db.commit()
                except Exception as agg_err:
                    logger.warning(f"[FinalAPI] On-demand final aggregation failed for meeting {meeting.id}: {agg_err}")

    if final_record is not None:
        try:
            response = _attach_language_info(_build_response(final_record), meeting, db)
            return _with_live_status(response, meeting)
        except Exception as build_err:
            logger.error(
                f"[FinalAPI] Failed to build response for final_record {final_record.id}: {build_err}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to assemble the final response.",
            ) from build_err

    # Slow Path: Record not yet aggregated. Resolve meeting_pk to check if meeting exists
    if meeting_pk is None:
        raise HTTPException(
            status_code=404,
            detail=f"Meeting '{meeting_id}' not found.",
        )

    # Meeting exists but final record has not yet been assembled — 202 Accepted. Returned, not raised: a 2xx
    # HTTPException reached clients as {"detail": ...} and the web app rendered it as an empty, completed meeting.
    from backend.utils.processing_status import display_status, user_error

    state = display_status(meeting.status if meeting else None)
    pending: Dict[str, Any] = {
        "processing_status": "PENDING",
        "status": state,
        "message": (
            "Final consolidated record not yet available. "
            "The meeting may still be processing. "
            "Please retry after processing completes."
        ),
        "meeting_pk": meeting_pk,
    }
    if state == "failed":
        pending["user_error"] = user_error(meeting.error_message if meeting else None)
    # "detail" keeps the body every earlier client parsed.
    return JSONResponse(status_code=202, content={**pending, "detail": dict(pending)})


def _with_live_status(response: FinalMeetingResponse, meeting: Optional[Meeting]) -> FinalMeetingResponse:
    """Stamp the meeting's CURRENT processing state on a stored final record.

    The stored record is the last successful build. After a re-run fails (or while one is running) it still says
    COMPLETED, and that was served as the meeting's state; ``display_status`` / ``user_error`` say what is true now.
    """
    from backend.utils.processing_status import display_status, user_error, user_warnings

    if meeting is None:
        return response
    state = display_status(meeting.status)
    response.warnings = user_warnings(meeting.pipeline_stage_states_json) if state == "completed" else []
    response.display_status = state
    response.meeting.status = meeting.status
    response.user_error = user_error(meeting.error_message) if state == "failed" else None
    return response
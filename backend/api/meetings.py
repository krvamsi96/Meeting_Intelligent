"""Meeting API endpoints - retrieval, comparison, verification, approval."""

import glob
import json
import logging
import os
import subprocess
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.api.deps import get_db, get_meeting_repository
from backend.config.config import (
    DEFAULT_PROJECT_NAME,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_SIMILARITY_TOP_K,
    settings,
)
from backend.models.models import (
    AIMeetingIntelFinal,
    ActionItem,
    AgendaKeyDecision,
    ChatSession,
    Meeting,
    TranscriptSegment,
)
from backend.repositories import MeetingRepository
from backend.schemas import schemas
from backend.services import similarity_service
from backend.services.agenda_service import (
    compute_agenda_decision_match_score,
    confidence_label,
    is_reliable_match,
)
from backend.services.evaluation_service import (
    CRITICAL_THRESHOLD,
    THRESHOLDS,
)
from backend.services.llm_workflow import (
    generate_detailed_meeting_comparison,
    process_meeting_report,
)
from backend.services.metrics_service import recalculate_meeting_metrics
from backend.services.vector_service import vector_service
from backend.utils.helpers import (
    clean_agenda_text,
    check_outputs_generated,
    format_seconds_to_duration,
)
from backend.utils.media_tools import FFMPEG
from backend.utils.security import safe_file_path, validate_time_range
from backend.workers.pipelines import build_meeting_vector_metadata

router = APIRouter(prefix="/meetings", tags=["Meetings"])

logger = logging.getLogger(__name__)



# ─── Agenda Module Endpoints ─────────────────────────────────────────────────

def _build_agenda_item(akd: AgendaKeyDecision) -> schemas.AgendaKeyDecisionItem:
    """Convert an AgendaKeyDecision ORM row to the enriched API schema.

    If match_score is not stored (legacy row), compute it on-the-fly using
    the agenda_service.  Returns match_score=None for items where the decision
    is the 'No formal decision recorded.' placeholder — showing '—' in the UI
    is more accurate than a misleadingly low cosine score.
    """
    match_score = akd.match_score
    confidence_level = None
    model_name = akd.embedding_model

    # On-the-fly computation for legacy records that have no stored score
    if match_score is None:
        try:
            score_info = compute_agenda_decision_match_score(
                agenda_text=akd.agenda or "",
                decision_text=akd.key_decision or "",
            )
            match_score = score_info.get("match_score")          # may be None for no-decision
            confidence_level = score_info.get("confidence_level")
            model_name = score_info.get("model_name")
        except Exception as exc:
            logger.warning(f"[AgendaAPI] On-the-fly score failed for akd.id={akd.id}: {exc}")
    else:
        # Derive confidence_level from the stored score via the single shared
        # bucket function (agenda_service.confidence_label) — avoids this
        # endpoint drifting out of sync with the mapping endpoint's thresholds.
        confidence_level = confidence_label(match_score)

    # Normalise speaker: treat "Unknown" / "unknown" / empty string as None
    raw_speaker = (akd.speaker or "").strip()
    speaker_out = None if not raw_speaker or raw_speaker.lower() == "unknown" else raw_speaker

    _NO_DEC = {"no formal decision recorded.", "no formal decision recorded", "no decision recorded.", "no decision recorded", "no decision.", "no decision", "none", "n/a", ""}
    key_dec = (akd.key_decision or "").strip()
    is_no_dec = not key_dec or key_dec.lower() in _NO_DEC
    dec_out = "" if is_no_dec else key_dec
    rel_score = None if is_no_dec else match_score
    ag_id_str = getattr(akd, 'agenda_id_str', None)
    if not ag_id_str or (ag_id_str == "A1" and akd.agenda_order and akd.agenda_order > 1):
        ag_id_str = f"A{akd.agenda_order or 1}"
    ag_source = getattr(akd, 'agenda_source', 'llm_detected') or 'llm_detected'
    # A decision is "grounded" when the extraction verified it against the
    # transcript (a quote that really occurs there and carries decision
    # language) — the same gate final_aggregation_service._build_agenda_decision_mapping
    # uses, so every endpoint flags a decision consistently. The embedding
    # similarity of agenda and decision is reported as match_score but does not
    # decide whether a stated decision belongs to its agenda item.
    is_grounded = not is_no_dec and bool(getattr(akd, 'decision_grounded', True))
    # No decision = empty, never the stored "No formal decision recorded." marker.
    return schemas.AgendaKeyDecisionItem(
        id=ag_id_str,
        agenda_id=akd.id,
        agenda_id_str=ag_id_str,
        agenda_text=akd.agenda,
        decision_text=dec_out,
        title=akd.agenda,
        agenda=akd.agenda,
        source=ag_source,
        key_decision=dec_out,
        decision=dec_out,
        decision_grounded=is_grounded,
        agenda_order=akd.agenda_order,
        match_score=rel_score,
        relevance_score=rel_score,
        confidence_level=confidence_level if (not is_no_dec and is_reliable_match(match_score)) else None,
        model_name=model_name,
        evidence=akd.evidence,
        speaker=speaker_out,
        # These four were dropped here, so the API said "inferred" (the schema default) for rows the database held
        # as "explicit", and never showed a decision's reason / confidence / timestamp even when they were stored.
        agenda_type=(akd.agenda_type or "inferred"),
        reason=akd.reason,
        timestamp=akd.timestamp,
        confidence=akd.confidence,
    )


def _build_agenda_response(
    meeting: Meeting,
    akd_rows: list,
) -> schemas.AgendaKeyDecisionsResponse:
    """Build the full AgendaKeyDecisionsResponse with summary statistics."""
    items = [_build_agenda_item(akd) for akd in akd_rows]

    agenda_list = [
        schemas.AgendaItemSchema(id=it.id or f"A{i+1}", title=it.title or it.agenda or "", source=it.source or "llm_detected")
        for i, it in enumerate(items)
    ]
    decision_list = [
        schemas.DecisionItemSchema(
            agenda_id=it.id or f"A{i+1}",
            decision=it.decision or "",
            decision_grounded=it.decision_grounded,
            relevance_score=it.relevance_score
        )
        for i, it in enumerate(items)
    ]

    scored = [it.relevance_score for it in items if it.relevance_score is not None]
    avg_score = round(sum(scored) / len(scored), 1) if scored else None

    return schemas.AgendaKeyDecisionsResponse(
        meeting_id=meeting.id,
        bharatvc_meeting_id=meeting.bharatvc_meeting_id,
        agenda_key_decisions=items,
        agenda_items=items,
        agenda=agenda_list,
        decisions=decision_list,
        total_agenda_items=len(items),
        average_match_score=avg_score,
    )


def _ensure_meeting_agenda_decisions(meeting: Meeting, db: Session) -> List[AgendaKeyDecision]:
    """Agenda & Decisions the pipeline (AIRAWAT) stored for the meeting; empty when there are none.

    Nothing stored means the pipeline found no agenda/decisions (or has not finished yet). A read never
    changes that: no keyword engine invents and persists rows here.
    """
    return db.query(AgendaKeyDecision).filter(AgendaKeyDecision.meeting_id == meeting.id).order_by(AgendaKeyDecision.agenda_order).all()


@router.get(
    "/by-bharatvc-id/{bharatvc_meeting_id}/agenda-key-decisions",
    response_model=schemas.AgendaKeyDecisionsResponse,
    summary="Get Agenda & Decisions by Business Meeting ID",
    description="Retrieve all Agenda-Decision pairs with real semantic Match Scores.",
)
def get_agenda_key_decisions_by_bharatvc_id(
    bharatvc_meeting_id: str,
    db: Session = Depends(get_db),
):
    """Retrieve Agenda items and Decisions by business identifier (bharatvc_meeting_id)."""
    meeting = db.query(Meeting).filter(
        Meeting.bharatvc_meeting_id == bharatvc_meeting_id
    ).first()
    if not meeting:
        raise HTTPException(
            status_code=404,
            detail=f"Meeting with bharatvc_meeting_id '{bharatvc_meeting_id}' not found",
        )
    akd_rows = _ensure_meeting_agenda_decisions(meeting, db)
    return _build_agenda_response(meeting, akd_rows)


@router.get(
    "/{meeting_id}/agenda-key-decisions",
    response_model=schemas.AgendaKeyDecisionsResponse,
    summary="Get Agenda & Decisions",
    description="Retrieve all Agenda-Decision pairs with real semantic Match Scores for a meeting.",
)
def get_agenda_key_decisions(
    meeting_id: Any,
    db: Session = Depends(get_db),
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """Retrieve Agenda items and their Decisions for a meeting by meeting ID."""
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    akd_rows = _ensure_meeting_agenda_decisions(meeting, db)
    return _build_agenda_response(meeting, akd_rows)



from backend.services.action_item_service import (
    _classify_category,
    format_assignees_list,
)


# ─── Action Items Module Endpoints ───────────────────────────────────────────

def _build_action_item_detail(ai: ActionItem, meeting: Optional[Meeting] = None) -> schemas.ActionItemDetail:
    """Convert an ActionItem ORM row to the enriched API schema for Add New Task form."""
    initiation_date = None
    if meeting:
        m_date = getattr(meeting, "meeting_date", None) or getattr(meeting, "upload_start_time", None) or getattr(meeting, "created_at", None)
        if m_date:
            if hasattr(m_date, "strftime"):
                initiation_date = m_date.strftime("%Y-%m-%d")
            else:
                initiation_date = str(m_date)[:10]

    # Category classification
    category = _classify_category(ai.task, ai.description, ai.evidence)

    # Assignees formatting
    assignees_raw = format_assignees_list(ai.owner, ai.speaker, ai.evidence)
    assignees = [schemas.AssigneeDetail(**a) for a in assignees_raw]

    # Priority normalization
    priority_val = None
    p_conf = None
    if ai.priority and ai.priority.strip().lower() not in {"not specified", "n/a", "none", ""}:
        p_lower = ai.priority.strip().lower()
        if p_lower in {"high", "medium", "low", "critical"}:
            priority_val = p_lower
        elif "high" in p_lower or "urgent" in p_lower:
            priority_val = "high"
        elif "medium" in p_lower:
            priority_val = "medium"
        elif "low" in p_lower:
            priority_val = "low"
        if priority_val:
            p_conf = 0.9

    due_date_str = ai.due_date.isoformat() if ai.due_date else None
    start_str = ai.deadline_start.isoformat() if ai.deadline_start else None
    end_str = ai.deadline_end.isoformat() if ai.deadline_end else None

    original_deadline = ai.deadline if (ai.deadline and ai.deadline.strip().lower() not in {"not specified", "n/a", "none"}) else None

    conf_score = round(ai.confidence / 100.0, 2) if (ai.confidence is not None and ai.confidence <= 100.0) else ai.confidence

    return schemas.ActionItemDetail(
        id=ai.id,
        item_order=ai.item_order,

        # Canonical Add New Task fields
        title=ai.task,
        description=ai.description,
        # main_task: read from DB column; None (NULL) is the only valid default.
        # getattr guard ensures backward-compat with any pre-migration rows.
        main_task=getattr(ai, "main_task", None),
        date_of_initiation=initiation_date,
        due_date=due_date_str,
        original_deadline_text=original_deadline,
        deadline_start=start_str,
        deadline_end=end_str,
        deadline_precision=ai.deadline_precision or ("exact_day" if due_date_str else "unresolvable"),
        timezone=getattr(settings, "DEFAULT_TIMEZONE", "Asia/Kolkata"),

        priority=priority_val,
        priority_confidence=p_conf,

        ministry=None,
        department=None,
        concerned_division=None,
        coordinator=None,

        assignees=assignees,
        apply_hierarchy_rules=True,
        category=category,
        dependencies=[],

        supporting_evidence=ai.evidence,
        source_speaker=ai.speaker,
        source_timestamp_start=ai.timestamp_start,
        source_timestamp_end=ai.timestamp_end,

        status=ai.status or "open",
        confidence_score=conf_score,

        # Backward compatibility fields
        task=ai.task,
        owner=ai.owner or "Unassigned",
        assigned_by=ai.assigned_by,
        deadline=ai.deadline or "Not specified",
        deadline_type=ai.deadline_type,
        evidence=ai.evidence,
        speaker=ai.speaker,
        timestamp_start=ai.timestamp_start,
        timestamp_end=ai.timestamp_end,
        confidence=ai.confidence,
        source=ai.source,
        agenda_ref=ai.agenda_ref,
        decision_ref=ai.decision_ref,
    )



def _build_action_items_response(
    meeting: Meeting,
    ai_rows: list,
) -> schemas.ActionItemsResponse:
    """Build the full ActionItemsResponse with summary statistics.

    Bug 9 fix: an item is counted as having an identified owner when EITHER:
      * ``it.owner`` is a non-generic name, OR
      * ``it.assignees`` is non-empty (covers evidence/speaker attribution).
    """
    items = [_build_action_item_detail(ai, meeting) for ai in ai_rows]

    _NO_OWNER = {"unassigned", "unknown", "n/a", "not specified", "none", ""}

    def _has_owner(it: schemas.ActionItemDetail) -> bool:
        # Check structured assignees list first (richer signal)
        if it.assignees:
            return True
        # Fall back to the plain owner string
        return bool(it.owner and it.owner.strip().lower() not in _NO_OWNER)

    owners = sum(1 for it in items if _has_owner(it))

    # Prefer resolved due_date count; fall back to original phrase presence
    resolved_dates = sum(1 for it in items if it.due_date is not None)
    phrase_deadlines = sum(
        1 for it in items
        if it.deadline and it.deadline.strip().lower() not in {"not specified", "n/a", "none"}
    )
    deadlines = resolved_dates if resolved_dates > 0 else phrase_deadlines
    conf_scores = [it.confidence for it in items if it.confidence is not None]
    avg_conf = round(sum(conf_scores) / len(conf_scores), 1) if conf_scores else None

    return schemas.ActionItemsResponse(
        meeting_id=meeting.id,
        bharatvc_meeting_id=meeting.bharatvc_meeting_id,
        action_items=items,
        total_items=len(items),
        owners_identified=owners,
        deadlines_identified=deadlines,
        average_confidence=avg_conf,
    )


def _ensure_meeting_action_items(meeting: Meeting, db: Session) -> List[ActionItem]:
    """Structured ActionItems from the DB; if none are stored yet, parse the action items AIRAWAT saved in the
    analysis (never keyword rules over the raw transcript) and persist them."""
    ai_rows = db.query(ActionItem).filter(ActionItem.meeting_id == meeting.id).order_by(ActionItem.item_order).all()
    if ai_rows:
        return ai_rows

    from backend.services.action_item_service import (
        extract_action_items_extractive,
        resolve_action_item_dates,
    )
    from backend.utils.date_resolver import get_reference_datetime

    ref_dt = get_reference_datetime(meeting.meeting_date, settings.DEFAULT_TIMEZONE)
    resolved = []

    # First check if meeting.analysis has action_items stored as JSON or text
    if meeting.analysis and meeting.analysis.action_items:
        raw_ai = meeting.analysis.action_items
        if isinstance(raw_ai, str) and raw_ai.strip():
            raw_str = raw_ai.strip()
            if raw_str.startswith("[") and raw_str.endswith("]"):
                try:
                    parsed = json.loads(raw_str)
                    if isinstance(parsed, list):
                        resolved = resolve_action_item_dates(parsed, ref_dt)
                except Exception:
                    pass
            if not resolved:
                extracted = extract_action_items_extractive(raw_str)
                if extracted:
                    resolved = resolve_action_item_dates(extracted, ref_dt)

    if not resolved:
        return []

    new_rows = []
    for idx, item in enumerate(resolved, start=1):
        t_name = item.get("task") or item.get("title") or f"Action {idx}"
        ai_record = ActionItem(
            meeting_id=meeting.id,
            bharatvc_meeting_id=meeting.bharatvc_meeting_id,
            item_order=idx,
            task=t_name,
            description=item.get("description"),
            main_task=item.get("main_task"),
            owner=item.get("owner", "Unassigned"),
            assigned_by=item.get("assigned_by"),
            deadline=item.get("deadline", "Not specified"),
            deadline_type=item.get("deadline_type", "not_specified"),
            due_date=item.get("due_date"),
            deadline_start=item.get("deadline_start"),
            deadline_end=item.get("deadline_end"),
            deadline_precision=item.get("deadline_precision"),
            priority=item.get("priority", "Not specified"),
            status=item.get("status", "open"),
            evidence=item.get("evidence"),
            speaker=item.get("speaker"),
            timestamp_start=item.get("timestamp_start"),
            timestamp_end=item.get("timestamp_end"),
            confidence=item.get("confidence"),
            source=item.get("source", "extractive"),
        )
        db.add(ai_record)
        new_rows.append(ai_record)

    if new_rows:
        try:
            db.commit()
            for r in new_rows:
                db.refresh(r)
            db.refresh(meeting)
            logger.info(f"[ActionItemsAPI] Dynamically extracted and persisted {len(new_rows)} Action Items for meeting {meeting.id}.")
        except Exception as err:
            db.rollback()
            logger.warning(f"[ActionItemsAPI] Failed to persist dynamic action items: {err}")

    return new_rows


@router.get(
    "/by-bharatvc-id/{bharatvc_meeting_id}/action-items",
    response_model=schemas.ActionItemsResponse,
    summary="Get Action Items by Business Meeting ID",
    description="Retrieve all structured Action Items for a meeting by business identifier.",
)
def get_action_items_by_bharatvc_id(
    bharatvc_meeting_id: str,
    db: Session = Depends(get_db),
):
    """Retrieve structured Action Items by business identifier (bharatvc_meeting_id)."""
    meeting = db.query(Meeting).filter(
        Meeting.bharatvc_meeting_id == bharatvc_meeting_id
    ).first()
    if not meeting:
        raise HTTPException(
            status_code=404,
            detail=f"Meeting with bharatvc_meeting_id '{bharatvc_meeting_id}' not found",
        )
    ai_rows = _ensure_meeting_action_items(meeting, db)
    return _build_action_items_response(meeting, ai_rows)


@router.get(
    "/{meeting_id}/action-items",
    response_model=schemas.ActionItemsResponse,
    summary="Get Action Items",
    description="Retrieve all structured Action Items for a meeting by meeting ID or business ID.",
)
def get_action_items(
    meeting_id: Any,
    db: Session = Depends(get_db),
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """Retrieve structured Action Items for a meeting by internal ID or bharatvc_meeting_id."""
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    ai_rows = _ensure_meeting_action_items(meeting, db)
    return _build_action_items_response(meeting, ai_rows)


@router.get(
    "/action-items/by-main-task/{main_task}",
    response_model=List[schemas.ActionItemDetail],
    summary="Get Action Items by Main Task Identifier",
    description="Retrieve all Action Items matching a specific main_task identifier.",
)
def get_action_items_by_main_task(
    main_task: str,
    db: Session = Depends(get_db),
):
    """Retrieve action items by their main_task group or identifier."""
    # Look up by main_task column or by primary key ID if integer
    query = db.query(ActionItem).filter(ActionItem.main_task == main_task)
    if main_task.isdigit():
        query = db.query(ActionItem).filter(
            (ActionItem.main_task == main_task) | (ActionItem.id == int(main_task))
        )
    ai_rows = query.all()
    if not ai_rows:
        raise HTTPException(
            status_code=404,
            detail=f"No Action Items found matching main_task '{main_task}'",
        )
    results = []
    for ai in ai_rows:
        meeting = db.query(Meeting).filter(Meeting.id == ai.meeting_id).first()
        results.append(_build_action_item_detail(ai, meeting))
    return results


@router.get(
    "/action-items/{action_item_id}",
    response_model=schemas.ActionItemDetail,
    summary="Get Action Item by ID",
    description="Retrieve a single structured Action Item by its unique database identifier.",
)
def get_action_item_by_id(
    action_item_id: int,
    db: Session = Depends(get_db),
):
    """Retrieve a single structured Action Item by its primary key ID."""
    ai = db.query(ActionItem).filter(ActionItem.id == action_item_id).first()
    if not ai:
        raise HTTPException(
            status_code=404,
            detail=f"Action Item with ID {action_item_id} not found",
        )
    meeting = db.query(Meeting).filter(Meeting.id == ai.meeting_id).first()
    return _build_action_item_detail(ai, meeting)


# "Not computed yet" for `evaluation_metrics`; None means "computed, and unavailable".
_UNSET = object()


def _load_reference_free_inputs(
    meeting: Meeting, db: Session, ai_rows: List[ActionItem], akd_rows: List[AgendaKeyDecision]
) -> dict:
    """The DB reads behind ``runtime_metrics`` / ``evaluation_metrics``, as plain data (no ORM objects kept)."""
    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.meeting_id == meeting.id)
        .order_by(TranscriptSegment.id.asc())
        .all()
    )
    seg_dicts = [
        {
            "speaker": getattr(s, "speaker", None),
            "start_time": getattr(s, "start_time", None),
            "end_time": getattr(s, "end_time", None),
            "text": getattr(s, "text", None),
            "stt_confidence": getattr(s, "stt_confidence", None),
            "correction_applied": getattr(s, "correction_applied", False),
        }
        for s in segments
    ]
    transcript_text = "\n".join(
        f"{d['speaker'] or ''}: {d['text'] or ''}".strip() for d in seg_dicts if d["text"]
    )

    analysis = getattr(meeting, "analysis", None)
    summary_text = (analysis.summary if analysis else None) or ""
    mom_text = (analysis.mom if analysis else None) or ""

    agenda_dicts = [
        {
            "agenda_id": getattr(r, "agenda_id_str", None) or f"A{getattr(r, 'agenda_order', 1)}",
            "agenda": getattr(r, "agenda", None),
            "decision": getattr(r, "key_decision", None),
            "evidence": getattr(r, "evidence", None),
            "reason": getattr(r, "reason", None),
            "match_score": getattr(r, "match_score", None),
            "speaker": getattr(r, "speaker", None),
        }
        for r in akd_rows
    ]
    ai_dicts = [
        {
            "task": getattr(a, "task", None),
            "description": getattr(a, "description", None),
            "owner": getattr(a, "owner", None),
            "deadline": getattr(a, "deadline", None),
            "due_date": getattr(a, "due_date", None),
            "priority": getattr(a, "priority", None),
            "evidence": getattr(a, "evidence", None),
        }
        for a in ai_rows
    ]

    chat_rows = (
        db.query(ChatSession)
        .filter(ChatSession.meeting_id == meeting.id, ChatSession.role == "assistant")
        .order_by(ChatSession.timestamp.desc())
        .limit(50)
        .all()
    )
    chat_dicts = []
    for c in chat_rows:
        sources = []
        if c.sources:
            try:
                sources = json.loads(c.sources)
            except Exception:
                sources = []
        chat_dicts.append({"content": c.content, "sources": sources if isinstance(sources, list) else []})

    return {
        "meeting_id": meeting.id,
        "seg_dicts": seg_dicts,
        "transcript_text": transcript_text,
        "summary_text": summary_text,
        "mom_text": mom_text,
        "agenda_dicts": agenda_dicts,
        "ai_dicts": ai_dicts,
        "chat_dicts": chat_dicts,
        "duration": float(getattr(meeting, "duration", 0) or 0),
        "language_confidence": getattr(meeting, "source_language_confidence", None),
        "stt_duration_sec": float(getattr(meeting, "stt_duration_sec", 0) or 0) or None,
        "speaker_duration_sec": float(getattr(meeting, "speaker_duration_sec", 0) or 0) or None,
    }


def _compute_evaluation_metrics(inp: dict, quality_metrics: Optional[dict]) -> dict:
    """``evaluation_metrics`` from loaded inputs. Embeds text (CPU, seconds when cold): call with NO transaction open."""
    from backend.evaluation import reference_free_engine as rfe

    return rfe.build_evaluation_metrics(
        segments=inp["seg_dicts"],
        audio_duration_sec=inp["duration"] or None,
        language_confidence=inp["language_confidence"],
        summary_text=inp["summary_text"],
        mom_text=inp["mom_text"],
        action_items=inp["ai_dicts"],
        agenda_rows=inp["agenda_dicts"],
        chat_messages=inp["chat_dicts"],
        meeting_id=inp["meeting_id"],
        score_overrides=_score_overrides(quality_metrics),
    )


def _compute_runtime_metrics(inp: dict, meeting: Meeting) -> dict:
    from backend.evaluation import reference_free_engine as rfe
    from backend.services.diarization_service import stored_speech_durations

    return rfe.build_runtime_metrics(
        speech_durations=stored_speech_durations(getattr(meeting, "diarization_info_json", None)),
        segments=inp["seg_dicts"],
        audio_duration_sec=inp["duration"],
        stt_duration_sec=inp["stt_duration_sec"],
        speaker_duration_sec=inp["speaker_duration_sec"],
        summary_text=inp["summary_text"],
        mom_text=inp["mom_text"],
        transcript_text=inp["transcript_text"],
        agenda_decision_rows=inp["agenda_dicts"],
        action_items=inp["ai_dicts"],
        meeting=meeting,
    )


def _build_reference_free_metrics(
    meeting: Meeting,
    db: Session,
    ai_rows: List[ActionItem],
    akd_rows: List[AgendaKeyDecision],
    quality_metrics: Optional[dict] = None,
    evaluation_metrics: Any = _UNSET,
) -> tuple:
    """Compute this meeting's ``runtime_metrics`` and ``evaluation_metrics`` blocks.

    Every input is freshly queried for THIS meeting.id — nothing is cached or
    shared across meetings, so two meetings evaluated back-to-back can never
    contaminate one another. Failures never break the meeting response: on
    any error both blocks fall back to ``None`` and the exception is logged.

    ``evaluation_metrics`` may be supplied already computed (by ``precompute_final_metrics``, outside any
    transaction); it is then not recomputed here.
    """
    try:
        inp = _load_reference_free_inputs(meeting, db, ai_rows, akd_rows)
        if evaluation_metrics is _UNSET:
            evaluation_metrics = _compute_evaluation_metrics(inp, quality_metrics)
        runtime_metrics = _compute_runtime_metrics(inp, meeting)
        return runtime_metrics, evaluation_metrics
    except Exception as e:
        logger.error(
            f"[MeetingsAPI] Reference-free metrics computation failed for meeting {meeting.id}: {e}",
            exc_info=True,
        )
        return None, None


def precompute_final_metrics(meeting: Meeting, db: Session) -> tuple:
    """(quality_metrics, evaluation_metrics, quality_error) for one meeting, computed WITHOUT a transaction open.

    The two scorers embed the meeting's sentences: seconds of CPU on a cold embedding cache (measured 4.8 s on the API
    thread for one meeting). Doing that inside the aggregation's transaction pinned a pooled connection - and, while
    aggregation wrote its placeholder row first, that row's lock - for the whole computation. Here everything is READ
    first, the read transaction is ended, and only then is anything embedded.
    """
    from backend.database.connection import release_read_transaction
    from backend.services import quality_metrics_service as qms

    ai_rows = _ensure_meeting_action_items(meeting, db)
    akd_rows = _ensure_meeting_agenda_decisions(meeting, db)
    meeting_id = meeting.id
    quality_inputs = None
    quality_error: Optional[Exception] = None
    try:
        quality_inputs = qms.load_quality_inputs(meeting, db)
    except Exception as exc:
        quality_error = exc
        logger.error(f"[MeetingsAPI] Quality metrics inputs failed for meeting {meeting_id}: {exc}", exc_info=True)
    try:
        rf_inputs = _load_reference_free_inputs(meeting, db, ai_rows, akd_rows)
    except Exception as exc:
        rf_inputs = None
        logger.error(f"[MeetingsAPI] Reference-free inputs failed for meeting {meeting_id}: {exc}", exc_info=True)

    release_read_transaction(db)   # no connection held from here on: the calls below embed text

    quality_metrics = None
    if quality_inputs is not None:
        try:
            quality_metrics = qms.compute_quality_metrics_from_inputs(quality_inputs)
        except Exception as exc:
            quality_error = exc
            logger.error(f"[MeetingsAPI] Quality metrics failed for meeting {meeting_id}: {exc}", exc_info=True)
    evaluation_metrics = None
    if rf_inputs is not None:
        try:
            evaluation_metrics = _compute_evaluation_metrics(rf_inputs, quality_metrics)
        except Exception as exc:
            logger.error(
                f"[MeetingsAPI] Reference-free metrics computation failed for meeting {meeting_id}: {exc}",
                exc_info=True,
            )
    return quality_metrics, evaluation_metrics, quality_error


def _score_overrides(quality_metrics: Optional[dict]) -> Optional[dict]:
    """The three headline scores as ``build_evaluation_metrics`` overrides (None when unavailable)."""
    if not isinstance(quality_metrics, dict):
        return None
    def _s(key):
        block = quality_metrics.get(key)
        return block.get("score") if isinstance(block, dict) else None
    return {
        "transcription": _s("transcript_quality"),
        "summary": _s("summary_quality"),
        "rag": _s("rag_groundedness"),
        "agenda": _s("agenda_coverage"),
        "action_items": _s("action_item_quality"),
        "decisions": _s("decision_quality"),
    }


def _compute_quality_metrics_safe(meeting: Meeting, db: Session) -> Optional[dict]:
    """The API ``metrics`` block; a failure here must never break the meeting response."""
    try:
        from backend.services.quality_metrics_service import compute_quality_metrics
        return compute_quality_metrics(meeting, db)
    except Exception as e:
        logger.error(f"[MeetingsAPI] Quality metrics failed for meeting {meeting.id}: {e}", exc_info=True)
        return None


def build_meeting_response_dict(
    meeting: Meeting, db: Session, quality_metrics: Optional[dict] = None, evaluation_metrics: Any = _UNSET
) -> dict:
    """Build the consolidated final meeting JSON dictionary matching schemas.Meeting.

    ``quality_metrics``: the already-computed ``metrics`` block (build_final_record computes it once and
    shares it); computed here when omitted. ``evaluation_metrics`` likewise (see ``precompute_final_metrics``).
    """
    ai_rows = _ensure_meeting_action_items(meeting, db)
    structured_details = [_build_action_item_detail(ai, meeting) for ai in ai_rows]

    akd_rows = _ensure_meeting_agenda_decisions(meeting, db)
    agenda_details = [_build_agenda_item(akd) for akd in akd_rows]

    m_schema = schemas.Meeting.model_validate(meeting, from_attributes=True)
    m_schema.action_items = structured_details
    m_schema.structured_action_items = structured_details
    m_schema.agenda_key_decisions = agenda_details
    m_schema.structured_agenda_decisions = agenda_details
    if m_schema.analysis:
        m_schema.analysis.structured_action_items = structured_details

    # Top-level ``agenda`` is a CONFIRMED agenda: the stored text (uploader / document), else the topics the speaker
    # ANNOUNCED (``agenda_type == "explicit"``: the transcript has an announcement cue in the title's sentence).
    # Topics that were only inferred from the discussion are never presented as an agenda: then ``agenda`` stays
    # null and ``agenda_source`` says "inferred_topics_only" (the topics themselves are in ``agenda_key_decisions``
    # with ``agenda_type: "inferred"``). Response only - never written back to the meeting row.
    if m_schema.agenda:
        m_schema.agenda_source = "stored"
    else:
        announced = [(it.title or it.agenda or "").strip() for it in agenda_details if it.agenda_type == "explicit"]
        announced = [t for t in announced if t]
        if announced:
            m_schema.agenda = "\n".join(f"{i}. {t}" for i, t in enumerate(announced, start=1))
            m_schema.agenda_source = "announced"
        elif agenda_details:
            m_schema.agenda_source = "inferred_topics_only"

    # Diarization processing time vs. speech duration: two different quantities that used to share one name.
    m_schema.speaker_processing_time_sec = m_schema.speaker_duration_sec
    try:
        from backend.services.diarization_service import stored_speech_durations

        spoken = stored_speech_durations(getattr(meeting, "diarization_info_json", None))
        m_schema.speaker_speech_duration_sec = round(sum(spoken.values()), 3) if spoken else None
    except Exception as exc:  # noqa: BLE001 - an enrichment: never a reason to fail the meeting response
        logger.warning(f"[MeetingsAPI] speech duration unavailable for meeting {meeting.id}: {exc}")

    if quality_metrics is None:
        quality_metrics = _compute_quality_metrics_safe(meeting, db)
    m_schema.metrics = quality_metrics
    m_schema.runtime_metrics, m_schema.evaluation_metrics = _build_reference_free_metrics(
        meeting, db, ai_rows, akd_rows, quality_metrics=quality_metrics, evaluation_metrics=evaluation_metrics
    )

    return jsonable_encoder(m_schema)


@router.get("/{meeting_id}", response_model=None)
def get_meeting(
    meeting_id: Any,
    db: Session = Depends(get_db),
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """Return the complete consolidated meeting record.

    Universal for every meeting and every lifecycle state (uploaded / processing /
    completed / error). The payload is re-aggregated LIVE from the source tables
    (``meetings`` + ``transcript_segments`` + ``agenda_key_decisions`` +
    ``action_items`` + ``analysis_results`` + evaluation/benchmark tables +
    ``AI_Meeting_Activty_Logs``) on every request, keyed dynamically by the
    resolved ``meetings.id``. Whatever has been processed is always returned;
    a field is null/empty only when the underlying data genuinely does not exist.
    The stored ``AI_Meeting_Inteligent_final.json_data`` blob is used only as a
    fallback if live aggregation raises.
    """
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    # ── Primary path: live aggregation from the source tables ────────────────
    # build_final_record() performs no LLM calls — it only reads the feature
    # tables for this meeting.id and upserts the consolidated row. Running it on
    # every GET guarantees the response reflects the CURRENT database state and
    # never a stale snapshot.
    try:
        from backend.services.final_aggregation_service import build_final_record

        final_record = build_final_record(meeting.id, db)
        # Capture the freshly-built snapshot BEFORE commit (commit expires the
        # instance attributes) so the response is never affected by post-commit
        # session state.
        snapshot = final_record.json_data if final_record is not None else None
        try:
            db.commit()
        except Exception as commit_err:
            db.rollback()
            logger.warning(
                f"[MeetingsAPI] Could not persist consolidated record for meeting "
                f"{meeting.id}: {commit_err}"
            )

        if isinstance(snapshot, dict) and snapshot:
            return JSONResponse(content=jsonable_encoder(snapshot))
    except Exception as agg_err:
        logger.warning(
            f"[MeetingsAPI] Live aggregation failed for meeting {meeting.id}; "
            f"falling back to stored/dynamic response: {agg_err}",
            exc_info=True,
        )
        try:
            db.rollback()
        except Exception:
            pass

    # ── Fallback 1: last good stored consolidated snapshot ──────────────────
    # `meeting` is already resolved to the correct row above — look up its
    # final record strictly by meeting_pk. An OR across the denormalized
    # string columns (meeting_id/bharatvc_meeting_id) risks matching a stale
    # or duplicate row belonging to a DIFFERENT meeting, since those columns
    # are indexed but not uniquely constrained on AI_Meeting_Inteligent_final.
    stored = (
        db.query(AIMeetingIntelFinal)
        .filter(AIMeetingIntelFinal.meeting_pk == meeting.id)
        .first()
    )
    if stored is not None and isinstance(stored.json_data, dict) and len(stored.json_data) > 3:
        return JSONResponse(content=stored.json_data)

    # ── Fallback 2: build directly from the ORM row (schemas.Meeting shape) ──
    try:
        return JSONResponse(content=build_meeting_response_dict(meeting, db))
    except Exception as final_err:
        logger.error(
            f"[MeetingsAPI] All response builders failed for meeting {meeting.id}: {final_err}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Could not assemble the meeting response. Please try again or contact support.",
        ) from final_err


# ─── AI Metrics Endpoint ───────────────────────────────────────────────────────

METRIC_META = {
    "transcription": {
        "name": "STT Accuracy",
        "short": "Speech-to-Text",
        "description": "Measures how accurately the AI converted spoken meeting audio into text.",
        "business_meaning": "This metric shows how accurately the system converted the meeting audio into text.",
        "technical_calculation": "Calculated by comparing the generated transcript with the reference transcript (Ground Truth) using Word Error Rate (WER).",
        "data_source": "• AIRAWAT STT Output\n• Reference Transcript (Ground Truth)",
        "has_reference_data": False,
        "reference_status_text": "Ground Truth Required",
        "formula": "1.0 - WER (Word Error Rate against reference ground truth)",
        "icon": "Mic",
        "unit": "%",
        "business_impact": "Poor transcription cascades into downstream summaries, MOMs, and action items.",
    },
    "speaker": {
        "name": "Speaker Accuracy",
        "short": "Speaker Identification",
        "description": "Measures how correctly the system identifies each speaker throughout the meeting.",
        "business_meaning": "This metric shows how correctly the system identifies each speaker throughout the meeting.",
        "technical_calculation": "Calculated by comparing detected speaker identities with verified speaker labels.",
        "data_source": "• Speaker Detection Output\n• Verified Speaker Labels",
        "has_reference_data": True,
        "reference_status_text": "Verified Speaker Mapping",
        "formula": "correct_speaker_turns / total_speaker_turns",
        "icon": "Users",
        "unit": "%",
        "business_impact": "Unidentified or misattributed speakers reduce reliability of decisions and action items.",
    },
    "summary": {
        "name": "Summary Accuracy",
        "short": "Summary Grounding",
        "description": "Measures how well the generated summary captures the important discussion points.",
        "business_meaning": "This metric shows how well the generated summary captures the important discussion points.",
        "technical_calculation": "Calculated by comparing the AI-generated summary with the meeting transcript and reference summary using evaluation metrics such as ROUGE-L grounding.",
        "data_source": "• AI Generated Summary\n• Meeting Transcript / Reference Summary",
        "has_reference_data": True,
        "reference_status_text": "Verified Transcript Grounding",
        "formula": "ROUGE-L F1 Grounding against full transcript text",
        "icon": "FileText",
        "unit": "%",
        "business_impact": "Ensures executive summary faithfully represents the meeting content.",
    },
    "mom": {
        "name": "Minutes of Meeting Quality",
        "short": "MOM Completeness",
        "description": "Measures how completely the AI captured meeting decisions, actions, and structure.",
        "business_meaning": "This metric shows how completely decisions and agenda points are documented.",
        "technical_calculation": "Evaluates key term recall, structural richness, and agenda coverage against full transcript.",
        "data_source": "• AI Generated MOM\n• Meeting Transcript",
        "has_reference_data": True,
        "reference_status_text": "Verified Transcript Grounding",
        "formula": "0.40 × term_recall + 0.35 × structural_richness + 0.25 × length_score",
        "icon": "BookOpen",
        "unit": "%",
        "business_impact": "MOM is the formal meeting record. Poor quality leads to undocumented decisions.",
    },
    "action_items": {
        "name": "Action Item Quality",
        "short": "Action Items",
        "description": "Measures the specificity and grounding of extracted action items.",
        "business_meaning": "This metric shows how clear, specific, and actionable the extracted tasks are.",
        "technical_calculation": "Evaluates item count, verb specificity, and transcript grounding.",
        "data_source": "• AI Extracted Action Items\n• Meeting Transcript",
        "has_reference_data": True,
        "reference_status_text": "Verified Transcript Grounding",
        "formula": "0.30 × count_score + 0.40 × specificity + 0.30 × transcript_grounding",
        "icon": "CheckSquare",
        "unit": "%",
        "business_impact": "Action items drive follow-up and operational accountability.",
    },
    "audio_quality": {
        "name": "Audio Quality (SNR)",
        "short": "Audio Clarity",
        "description": "Real audio quality computed from Whisper's no-speech probability and silence ratio.",
        "business_meaning": "This metric shows recording clarity, background noise level, and signal cleanliness.",
        "technical_calculation": "Calculated from signal-to-noise ratio and silence probability.",
        "data_source": "• FFmpeg Audio Stream Analysis\n• Whisper Audio Spectrum",
        "has_reference_data": True,
        "reference_status_text": "Empirical Signal Analysis",
        "formula": "(1 − avg_no_speech) − silence_ratio_penalty",
        "icon": "Volume2",
        "unit": "%",
        "business_impact": "Poor audio directly causes transcription errors.",
    },
    "topic": {
        "name": "Topic Detection Quality",
        "short": "Topics & Keywords",
        "description": "Measures keyword richness and diversity extracted from the meeting.",
        "business_meaning": "This metric shows how accurately main subjects and themes were cataloged.",
        "technical_calculation": "Evaluates keyword frequency distribution variance and coverage.",
        "data_source": "• AI Extracted Topics\n• Meeting Transcript",
        "has_reference_data": True,
        "reference_status_text": "Empirical Keyword Analysis",
        "formula": "0.40 × count_richness + 0.30 × diversity + 0.30 × coverage",
        "icon": "Tag",
        "unit": "%",
        "business_impact": "Topics drive search, classification, and knowledge discovery.",
    },
    "sentiment": {
        "name": "Sentiment Confidence",
        "short": "Sentiment AI",
        "description": "Confidence that the detected sentiment matches the actual meeting tone.",
        "business_meaning": "This metric shows the emotional tone and sentiment polarity of the discussion.",
        "technical_calculation": "Evaluates emotion breakdown quality and polarity coherence.",
        "data_source": "• AI Sentiment Engine\n• Segment Text Polarity",
        "has_reference_data": True,
        "reference_status_text": "Empirical Polarity Coherence",
        "formula": "0.30 × detection + 0.30 × emotion_quality + 0.40 × polarity_coherence",
        "icon": "Smile",
        "unit": "%",
        "business_impact": "Sentiment tracking identifies team morale trends.",
    },
    "similarity": {
        "name": "Similar Meetings Confidence",
        "short": "Knowledge Continuity",
        "description": "Confidence in the similarity engine's matching output.",
        "business_meaning": "This metric shows confidence in linking this session to historical archives.",
        "technical_calculation": "Evaluates multi-signal vector embedding similarity.",
        "data_source": "• ChromaDB HNSW Vector Archive",
        "has_reference_data": True,
        "reference_status_text": "Vector Database Archive",
        "formula": "Cosine similarity across transcript & MOM embeddings",
        "icon": "GitBranch",
        "unit": "%",
        "business_impact": "Avoids repeated discussions and leverages organizational memory.",
    },
    "video_quality": {
        "name": "Video Stream Quality",
        "short": "Video (FPS/Resolution)",
        "description": "Computed from ffprobe frame rate and resolution specs.",
        "business_meaning": "This metric shows the visual quality and frame rate of the video recording.",
        "technical_calculation": "FFprobe stream analysis for resolution and FPS.",
        "data_source": "• FFprobe Media Metadata",
        "has_reference_data": True,
        "reference_status_text": "Empirical Media Probe",
        "formula": "fps_score(0-50) + resolution_score(0-50)",
        "icon": "Video",
        "unit": "%",
        "business_impact": "Video quality affects visual recording clarity.",
    },
}


def _whisper_model_label() -> str:
    """Human label for the Whisper model behind the configured endpoint.

    The endpoint looks like ``…/Whisper-Large-v3/v1/audio/transcriptions``; the model is the
    segment before ``/v1`` (the previous ``split("/")[-3]`` returned the literal string "v1").
    """
    url = ""
    try:
        from backend.services.model_router import model_router

        url = model_router.get_effective_config().get("whisper_endpoint") or ""
    except Exception:
        pass
    url = url or settings.AIRAWAT_WHISPER_URL or ""
    parts = [p for p in url.split("?", 1)[0].split("/") if p]
    for i, part in enumerate(parts):
        if part.lower() in ("v1", "v2") and i > 0:
            return parts[i - 1]
    return "AIRAWAT Whisper"


def _get_status(score: Optional[float], is_na: bool = False) -> dict:
    """Returns status label, color class, and badge text for a metric score."""
    if is_na or score is None:
        return {"status": "Reference Data Required", "color": "amber", "badge": "Ground Truth Required"}
    if score >= THRESHOLDS["excellent"]:
        return {
            "status": "Excellent",
            "color": "emerald",
            "badge": f"≥{THRESHOLDS['excellent']}%",
        }
    elif score >= THRESHOLDS["good"]:
        return {"status": "Good", "color": "blue", "badge": f"≥{THRESHOLDS['good']}%"}
    elif score >= THRESHOLDS["average"]:
        return {
            "status": "Average",
            "color": "amber",
            "badge": f"≥{THRESHOLDS['average']}%",
        }
    elif score >= THRESHOLDS["poor"]:
        return {"status": "Poor", "color": "orange", "badge": f"≥{THRESHOLDS['poor']}%"}
    else:
        return {
            "status": "Critical",
            "color": "red",
            "badge": f"<{THRESHOLDS['poor']}%",
        }


def _get_recommendation(key: str, score: float, is_document: bool) -> str:
    """Returns a concrete recommendation based on metric score."""
    if score >= THRESHOLDS["excellent"]:
        return "Performing excellently. No action required."
    recs = {
        "transcription": "Improve recording quality or use a directional microphone. "
        "Ensure speakers are close to the microphone.",
        "speaker": "Use a higher-quality diarization model or add speaker pre-enrollment. "
        "Ensure speakers don't talk over each other.",
        "summary": "The AI summary may be missing context. Review the transcript and "
        "regenerate the summary with a higher-quality LLM.",
        "mom": "The Minutes of Meeting lacks structure. Regenerate with a template "
        "that includes sections for decisions, action items, and attendees.",
        "action_items": "Action items are too generic. Ensure the meeting includes "
        "specific tasks with owners and due dates for better extraction.",
        "audio_quality": "Improve recording conditions. Use noise-cancellation headsets, "
        "reduce background noise, and minimize meeting gaps.",
        "topic": "Topic detection is weak. Ensure the meeting covers focused topics "
        "with specific domain terminology for better keyword extraction.",
        "sentiment": "Sentiment detection is uncertain. Ensure the meeting has clear "
        "discussion patterns for the AI to classify correctly.",
        "similarity": "Similarity engine could not find related meetings. "
        "Approve more meetings to build up the knowledge base.",
        "video_quality": (
            "Video quality is low. Use a higher-resolution camera "
            "and ensure adequate lighting and stable frame rate."
            if not is_document
            else "Not applicable for document meetings."
        ),
    }
    return recs.get(key, "Review and improve this component.")


@router.get("/{meeting_id}/metrics")
def get_meeting_metrics(
    meeting_id: Any,
    db: Session = Depends(get_db),
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """
    Returns a fully structured AIMetricsReport with all 10 component metrics,
    weighted formula breakdown, drag warnings, recommendations, and historical trend.
    """
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    # Check for True Evaluation Result — must be derived BEFORE the cache guard below.
    has_gt = meeting.has_ground_truth  # optional reference: a failed lookup means "none", never an aborted request
    eval_res = meeting.evaluation_result
    # An evaluation_results row is only a TRUE-ACCURACY source when it was a
    # ground-truth benchmark run. Legacy "self" rows carry reference-free
    # estimates that must NEVER be surfaced as accuracy.
    if eval_res is not None and (
        not has_gt or getattr(eval_res, "evaluation_type_code", None) == "self"
    ):
        eval_res = None

    # Bug 6 fix: Restore metrics_json cache read.
    # Only skip the cache when a True Evaluation result is available (ground truth
    # always takes precedence) or when the cached payload is explicitly empty.
    # This prevents expensive re-computation on every /metrics API call.
    cached = None
    if not has_gt:
        raw_cache = getattr(meeting, "metrics_json", None)
        if raw_cache:
            try:
                import json as _json
                cached = _json.loads(raw_cache)
                # Discard cache entries that were written as the empty "Not Evaluated"
                # sentinel (overall_score: null) — those are not useful to serve.
                if cached and cached.get("overall_score") is None and not cached.get("components"):
                    cached = None
            except Exception:
                cached = None

    segments = repo.get_segments(meeting_id)

    analysis = meeting.analysis
    summary_text = (analysis.summary or "") if analysis else ""
    mom_text = (analysis.mom or "") if analysis else ""
    action_items_text = (analysis.action_items or "") if analysis else ""

    is_document = getattr(meeting, "file_type", "media") == "document"
    is_audio_only = not is_document and (meeting.video_quality or 0) == 0.0

    # Try to auto-evaluate if ground truth exists but not evaluated yet
    if has_gt and not eval_res:
        try:
            from backend.services.true_evaluation_service import run_meeting_evaluation
            eval_res = run_meeting_evaluation(meeting, db, performed_by="api_auto")
        except Exception as e:
            logger.error(f"Auto-evaluation failed in metrics API: {e}")

    # Build actual scores from True Evaluation Result if available
    if eval_res:
        trans_score = round(eval_res.stt_accuracy, 1) if eval_res.stt_accuracy is not None else None
        spk_score = round(eval_res.speaker_accuracy, 1) if eval_res.speaker_accuracy is not None else None
        diarization_score = round(eval_res.diarization_accuracy, 1) if eval_res.diarization_accuracy is not None else None
        summary_score = round(eval_res.summary_accuracy, 1) if eval_res.summary_accuracy is not None else None
        mom_score = round(eval_res.mom_accuracy, 1) if eval_res.mom_accuracy is not None else None
        action_score = round(eval_res.action_accuracy, 1) if eval_res.action_accuracy is not None else None
        topic_score = round(eval_res.topic_accuracy, 1) if eval_res.topic_accuracy is not None else None
        sentiment_score = round(eval_res.sentiment_accuracy, 1) if eval_res.sentiment_accuracy is not None else None
        
        overall_score = round(eval_res.overall_accuracy, 1) if eval_res.overall_accuracy is not None else 0.0
        calculation_status = "Evaluated"
        calculation_source = "True Evaluation Engine v2.0"
        label = "Excellent" if overall_score >= 90 else "Good" if overall_score >= 75 else "Average" if overall_score >= 60 else "Needs Improvement"
        color = "emerald" if overall_score >= 90 else "blue" if overall_score >= 75 else "amber" if overall_score >= 60 else "red"
        warnings = []
    else:
        # Ground Truth is missing, so we set all accuracy metrics to None (Not Evaluated)
        trans_score = None
        summary_score = None
        diarization_score = None
        spk_score = None
        action_score = None
        topic_score = None
        sentiment_score = None
        mom_score = None
        
        overall_score = None
        calculation_status = "Not Evaluated"
        calculation_source = "Reference Data Required"
        label = "Not Evaluated"
        color = "slate"
        warnings = [{"metric": "evaluation", "message": "Ground Truth reference data is required to calculate true accuracy."}]

    # Status helper for 4 core metrics
    def _get_4metric_status(val: Optional[float]) -> dict:
        if val is None:
            return {"status": "Reference Data Required", "color": "rose", "badge": "Ground Truth Required"}
        if val >= 98.0:
            return {"status": "Excellent", "color": "emerald", "badge": "≥98% Excellent 🟢"}
        elif val >= 95.0:
            return {"status": "Good", "color": "blue", "badge": "95-97% Good 🔵"}
        elif val >= 90.0:
            return {"status": "Average", "color": "amber", "badge": "90-94% Average 🟡"}
        else:
            return {"status": "Needs Improvement", "color": "rose", "badge": "<90% Needs Improvement 🔴"}

    if not has_gt:
        stt_status = {"status": "Reference Transcript Required", "color": "rose", "badge": "Reference Transcript Required"}
        summary_status = {"status": "Reference Summary Required", "color": "rose", "badge": "Reference Summary Required"}
        spk_status = {"status": "Reference Speaker Labels Required", "color": "rose", "badge": "Reference Speaker Labels Required"}
        diar_status = {"status": "Reference Speaker Labels Required", "color": "rose", "badge": "Reference Speaker Labels Required"}
    else:
        stt_status = _get_4metric_status(trans_score)
        summary_status = _get_4metric_status(summary_score)
        spk_status = _get_4metric_status(spk_score)
        diar_status = _get_4metric_status(diarization_score)

    last_upd = (
        meeting.processing_end_time.isoformat()
        if getattr(meeting, "processing_end_time", None)
        else (meeting.upload_time.isoformat() if getattr(meeting, "upload_time", None) else None)
    )

    # EXACTLY FOUR MANDATORY CORE METRICS
    metrics = [
        {
            "id": "stt_accuracy",
            "name": "Speech-to-Text Accuracy",
            "short": "STT Accuracy",
            "value": trans_score,
            "progress": trans_score if trans_score is not None else 0.0,
            "status": stt_status["status"],
            "color": stt_status["color"],
            "badge": stt_status["badge"],
            "meaning": "Shows how accurately the meeting audio was converted into text.",
            "calculation": "Calculated by comparing generated transcript with verified reference transcript using WER.",
            "last_updated": last_upd,
        },
        {
            "id": "summary_accuracy",
            "name": "Summary Accuracy",
            "short": "Summary Quality",
            "value": summary_score,
            "progress": summary_score if summary_score is not None else 0.0,
            "status": summary_status["status"],
            "color": summary_status["color"],
            "badge": summary_status["badge"],
            "meaning": "Shows how accurately the generated summary represents the complete meeting discussion.",
            "calculation": "Calculated by comparing AI summary with meeting transcript using ROUGE-L F1 grounding.",
            "last_updated": last_upd,
        },
        {
            "id": "speaker_accuracy",
            "name": "Speaker Accuracy",
            "short": "Speaker Identification",
            "value": spk_score,
            "progress": spk_score if spk_score is not None else 0.0,
            "status": spk_status["status"],
            "color": spk_status["color"],
            "badge": spk_status["badge"],
            "meaning": "Shows how accurately the system identified the meeting speakers.",
            "calculation": "Calculated by comparing detected speaker identities with verified speaker labels.",
            "last_updated": last_upd,
        },
        {
            "id": "speaker_diarization_accuracy",
            "name": "Speaker Diarization Accuracy",
            "short": "Timeline Diarization",
            "value": diarization_score,
            "progress": diarization_score if diarization_score is not None else 0.0,
            "status": diar_status["status"],
            "color": diar_status["color"],
            "badge": diar_status["badge"],
            "meaning": "Shows how accurately the meeting audio was separated into different speakers.",
            "calculation": "Calculated by comparing detected speaker segments with manually verified timelines using DER.",
            "last_updated": last_upd,
        },
    ]

    # Historical trend: compare with last 5 completed meetings
    trend = []
    try:
        all_meetings = (
            db.query(Meeting)
            .filter(
                Meeting.id != meeting.id,
                Meeting.status == "completed",
                Meeting.accuracy > 0,
            )
            .order_by(Meeting.upload_time.desc())
            .limit(5)
            .all()
        )

        for m in reversed(all_meetings):
            trend.append(
                {
                    "meeting_id": m.id,
                    "code": m.display_id or m.bharatvc_meeting_id or str(m.id),
                    "date": (
                        m.upload_time.strftime("%Y-%m-%d") if m.upload_time else None
                    ),
                    "overall_score": round(m.accuracy or 0.0, 1),
                    "transcription": round(m.transcription_accuracy or 0.0, 1),
                    "speaker": round(m.speaker_accuracy or 0.0, 1),
                }
            )
    except Exception as trend_err:
        logger.warning(f"Could not build trend for meeting {meeting_id}: {trend_err}")

    _action_item_rows = db.query(ActionItem).filter(ActionItem.meeting_id == meeting.id).all()

    # Build real_runtime_data object for enterprise dashboard
    real_runtime_data = {
        "stt": {
            "generated": getattr(meeting, "stt_generated", True if segments else False),
            "word_count": getattr(meeting, "stt_word_count", sum(len((s.text or "").split()) for s in segments) if segments else 0),
            "char_count": getattr(meeting, "stt_char_count", sum(len(s.text or "") for s in segments) if segments else 0),
            "generation_time_sec": round(getattr(meeting, "stt_duration_sec", 0.0) or 0.0, 2),
            # Spoken language is auto-detected; ALL output below is English.
            "source_language": getattr(meeting, "source_language", None) or "unknown",
            "source_language_name": getattr(meeting, "source_language_name", None) or "Unknown",
            "source_language_confidence": getattr(meeting, "source_language_confidence", None),
            "translation_applied": bool(getattr(meeting, "translation_applied", False)),
            "output_language": getattr(meeting, "output_language", None) or "en",
            "translation_time_sec": round(getattr(meeting, "translation_duration_sec", 0.0) or 0.0, 2),
            "language": (
                f"{getattr(meeting, 'source_language_name', None) or 'Auto-detected'}"
                f"{' → English' if getattr(meeting, 'translation_applied', False) else ''}"
            ),
            "model_used": _whisper_model_label(),
        },
        "summary": {
            "generated": getattr(meeting, "summary_generated", bool(summary_text)),
            "word_count": getattr(meeting, "summary_word_count", len((summary_text or "").split()) if summary_text else 0),
            "generation_time_sec": round(getattr(meeting, "summary_duration_sec", 0.0) or 0.0, 2),
            "model_used": settings.AIRAWAT_LLM_MODEL,
        },
        "mom": {
            "generated": getattr(meeting, "mom_generated", bool(mom_text)),
            "word_count": getattr(meeting, "mom_word_count", len((mom_text or "").split()) if mom_text else 0),
            "generation_time_sec": round(getattr(meeting, "mom_duration_sec", 0.0) or 0.0, 2),
            "model_used": settings.AIRAWAT_LLM_MODEL,
        },
        "speaker": {
            "speakers_detected": getattr(meeting, "speaker_count", len(set(s.speaker for s in segments if s.speaker)) if segments else 0),
            "timeline_generated": getattr(meeting, "speaker_timeline_generated", bool(segments)),
            "segment_count": len(segments) if segments else 0,
            "processing_time_sec": round(getattr(meeting, "speaker_duration_sec", 0.0) or 0.0, 2),
            "model_used": "NVIDIA NeMo",
        },
        "action_items": {
            "generated": getattr(meeting, "action_items_generated", bool(action_items_text)),
            "total_items": getattr(meeting, "action_items_count", 0),
            # Bug 5 fix: derive owner/deadline counts from actual ActionItem DB rows,
            # not from string-scanning the text blob which is now a JSON array.
            "owners_identified": sum(
                1 for ai_row in _action_item_rows
                if ai_row.owner and ai_row.owner.strip().lower() not in {"unassigned", "unknown", "n/a", "not specified", "none", ""}
            ),
            "due_dates_identified": sum(
                1 for ai_row in _action_item_rows
                if ai_row.due_date is not None
                   or (ai_row.deadline and ai_row.deadline.strip().lower() not in {"not specified", "n/a", "none", ""})
            ),
            "generation_time_sec": round(getattr(meeting, "action_items_duration_sec", 0.0) or 0.0, 2),
            "model_used": settings.AIRAWAT_LLM_MODEL,
        },
        "topics": {
            "generated": getattr(meeting, "topics_generated", bool(meeting.keyword_frequency)),
            "total_topics": getattr(meeting, "topics_count", len((meeting.keyword_frequency or "").split(",")) if meeting.keyword_frequency else 0),
            "keywords": (meeting.keyword_frequency or "").split(",") if meeting.keyword_frequency else [],
        },
        "sentiment": {
            "overall_sentiment": getattr(meeting, "sentiment_label", meeting.sentiment or "Neutral"),
            "breakdown_json": getattr(meeting, "sentiment_breakdown_json", meeting.emotions or None),
        },
        "performance": {
            "upload_time_sec": round(getattr(meeting, "upload_duration_sec", 0.0) or 0.0, 2),
            "processing_time_sec": round(getattr(meeting, "total_processing_time_sec", 0.0) or 0.0, 2),
            # The report sections run IN PARALLEL: the AI response time is the longest of them, not their sum.
            "ai_response_time_sec": round(max(
                (getattr(meeting, "summary_duration_sec", 0.0) or 0.0),
                (getattr(meeting, "mom_duration_sec", 0.0) or 0.0),
                (getattr(meeting, "action_items_duration_sec", 0.0) or 0.0),
            ), 2),
            "export_time_sec": "Not Measured",
        },
        "database_status": {
            "meeting_saved": getattr(meeting, "db_meeting_saved", True),
            "transcript_saved": getattr(meeting, "db_transcript_saved", bool(segments)),
            "summary_saved": getattr(meeting, "db_summary_saved", bool(summary_text)),
            "mom_saved": getattr(meeting, "db_summary_saved", bool(mom_text)),
            "action_items_saved": getattr(meeting, "db_action_items_saved", bool(action_items_text)),
        },
        "export_status": {
            "pdf_ready": getattr(meeting, "export_pdf_generated", True),
            "docx_ready": getattr(meeting, "export_docx_generated", True),
            "json_ready": getattr(meeting, "export_json_generated", True),
        },
    }

    # True evaluation metrics mapping
    true_metrics_detailed = {
        "stt": {
            "wer": eval_res.wer if eval_res else None,
            "cer": eval_res.cer if eval_res else None,
            "match_pct": eval_res.stt_match_pct if eval_res else None,
            "accuracy": eval_res.stt_accuracy if eval_res else None,
        },
        "diarization": {
            "der": eval_res.der if eval_res else None,
            "jer": eval_res.jer if eval_res else None,
            "accuracy": eval_res.diarization_accuracy if eval_res else None,
        },
        "speaker_id": {
            "accuracy": eval_res.speaker_accuracy if eval_res else None,
            "precision": eval_res.speaker_precision if eval_res else None,
            "recall": eval_res.speaker_recall if eval_res else None,
            "f1": eval_res.speaker_f1 if eval_res else None,
            "confusion_matrix": json.loads(eval_res.speaker_confusion_matrix_json) if (eval_res and eval_res.speaker_confusion_matrix_json) else None,
        },
        "summary": {
            "rouge1": eval_res.rouge1 if eval_res else None,
            "rouge2": eval_res.rouge2 if eval_res else None,
            "rougeL": eval_res.rougeL if eval_res else None,
            "bertscore_f1": eval_res.bertscore_f1 if eval_res else None,
            "accuracy": eval_res.summary_accuracy if eval_res else None,
        },
        "mom": {
            "coverage": eval_res.mom_coverage if eval_res else None,
            "missing_count": eval_res.mom_missing_count if eval_res else None,
            "hallucination_count": eval_res.mom_hallucination_count if eval_res else None,
            "accuracy": eval_res.mom_accuracy if eval_res else None,
        },
        "action_items": {
            "precision": eval_res.action_precision if eval_res else None,
            "recall": eval_res.action_recall if eval_res else None,
            "f1": eval_res.action_f1 if eval_res else None,
            "missing_count": eval_res.action_missing_count if eval_res else None,
            "false_count": eval_res.action_false_count if eval_res else None,
            "accuracy": eval_res.action_accuracy if eval_res else None,
        },
        "topics": {
            "accuracy": eval_res.topic_accuracy if eval_res else None,
            "precision": eval_res.topic_precision if eval_res else None,
            "recall": eval_res.topic_recall if eval_res else None,
            "f1": eval_res.topic_f1 if eval_res else None,
        },
        "sentiment": {
            "accuracy": eval_res.sentiment_accuracy if eval_res else None,
            "precision": eval_res.sentiment_precision if eval_res else None,
            "recall": eval_res.sentiment_recall if eval_res else None,
            "f1": eval_res.sentiment_f1 if eval_res else None,
            "confusion_matrix": json.loads(eval_res.sentiment_confusion_matrix_json) if (eval_res and eval_res.sentiment_confusion_matrix_json) else None,
        }
    }

    return {
        "meeting_id": meeting_id,
        "meeting_code": meeting.display_id or meeting.bharatvc_meeting_id or str(meeting_id),
        "stt_accuracy": trans_score,
        "summary_accuracy": summary_score,
        "speaker_accuracy": spk_score,
        "speaker_diarization_accuracy": diarization_score,
        "last_updated": last_upd,
        "calculation_status": calculation_status,
        "calculation_source": calculation_source,
        "is_document": is_document,
        "is_audio_only": is_audio_only,
        "has_ground_truth": has_gt,
        "overall": {
            "score": overall_score,
            "weighted_sum": overall_score,
            "penalty": 0.0,
            "label": label,
            "color": color,
            "warnings": warnings,
        },
        "metrics": metrics,
        "real_runtime_data": real_runtime_data,
        "true_metrics_detailed": true_metrics_detailed,
        "formula_note": "All accuracy metrics compare AI Output vs Ground Truth reference dataset.",
        "thresholds": THRESHOLDS,
        "critical_threshold": CRITICAL_THRESHOLD,
        "historical_trend": trend,
    }


@router.get("/{meeting_id}/compare/{matched_id}")
def compare_meetings(
    meeting_id: Any,
    matched_id: Any,
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """Generates detailed AI comparison between two meetings."""
    meeting = repo.get_by_id(meeting_id)
    matched_meeting = repo.get_by_id(matched_id)
    if not meeting or not matched_meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    def get_transcript(m_id):
        segments = repo.get_segments(m_id)
        transcript = (
            "\n".join([f"{seg.speaker}: {seg.text}" for seg in segments])
            if segments
            else ""
        )
        return transcript

    current_transcript = get_transcript(meeting_id)
    past_transcript = get_transcript(matched_id)
    if not current_transcript or not past_transcript:
        raise HTTPException(
            status_code=400,
            detail="Cannot compare: Missing transcript for one or both meetings",
        )

    comparison = generate_detailed_meeting_comparison(
        current_transcript, past_transcript
    )
    return comparison


@router.get(
    "/{meeting_id}/similar", response_model=schemas.IntelligentSimilarMeetingsResponse
)
def get_similar_meetings(
    meeting_id: Any,
    top_k: int = Query(
        DEFAULT_SIMILARITY_TOP_K, description="Maximum number of similar meetings"
    ),
    min_similarity: float = Query(
        DEFAULT_SIMILARITY_THRESHOLD,
        description="Minimum similarity threshold (0.0-1.0)",
    ),
    include_details: bool = Query(
        True, description="Include difference analysis and timeline"
    ),
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """Find, classify and explain similar meetings using multi-signal intelligence.

    Every result is classified (exact/near duplicate, follow-up, recurring,
    related), scored for confidence, explained in plain English and enriched
    with a full difference analysis + evolution timeline.
    """
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    response = similarity_service.find_similar_meetings(
        repo=repo,
        meeting_id=meeting_id,
        top_k=top_k,
        min_score=min_similarity,
        include_details=include_details,
    )
    return response


@router.get(
    "/{meeting_id}/duplicate-check", response_model=schemas.DuplicateCheckResponse
)
def check_meeting_duplicate(
    meeting_id: Any,
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """Detect whether this meeting already exists (duplicate upload/transcript)."""
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return similarity_service.detect_duplicate(repo=repo, meeting_id=meeting_id)


_MEDIA_EXTENSIONS = {
    ".mp4", ".mp3", ".wav", ".webm", ".m4a", ".mov", ".avi", ".mkv", ".ogg", ".aac", ".flv", ".wma", ".3gp",
}


def _is_media_file(path: Optional[str]) -> bool:
    return bool(path) and os.path.splitext(path)[1].lower() in _MEDIA_EXTENSIONS


def _resolve_media_file(meeting) -> Optional[str]:
    """Resolve the AUDIO/VIDEO file for a meeting, or None.

    Document meetings (and any meeting whose stored file is not audio/video) have no media: the
    uploaded .pdf/.txt used to be returned here and then streamed as ``video/mp4`` (/stream) or fed
    to ffmpeg (/clip -> HTTP 500).
    """
    if not meeting:
        return None
    if getattr(meeting, "file_type", "media") == "document":
        return None

    candidates = []

    # 1. Try meeting.filepath directly or via safe_file_path
    if meeting.filepath:
        try:
            if (
                os.path.isabs(meeting.filepath)
                and _is_media_file(meeting.filepath)
                and os.path.isfile(meeting.filepath)
            ):
                return meeting.filepath
        except Exception:
            pass

        try:
            base_fp = os.path.basename(meeting.filepath.replace("\\", "/"))
            if base_fp:
                candidates.append(safe_file_path(settings.UPLOAD_DIR, base_fp))
        except Exception:
            pass

    # 2. Try meeting.filename via safe_file_path
    if meeting.filename:
        try:
            base_fn = os.path.basename(meeting.filename.replace("\\", "/"))
            if base_fn:
                candidates.append(safe_file_path(settings.UPLOAD_DIR, base_fn))
        except Exception:
            pass

    for c in candidates:
        if c and _is_media_file(c) and os.path.isfile(c):
            return c

    # NOTE: there is deliberately no "pick any media file from UPLOAD_DIR" fallback. That
    # used to return the FIRST recording in the folder when this meeting's own file was
    # missing, so /stream and /clip could serve (and cut clips from) a DIFFERENT meeting's
    # recording. A missing file is a 404, not somebody else's audio.
    return None


@router.get("/{meeting_id}/stream")
def stream_meeting_file(
    meeting_id: Any, repo: MeetingRepository = Depends(get_meeting_repository)
):
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    file_to_stream = _resolve_media_file(meeting)

    if not file_to_stream or not os.path.exists(file_to_stream):
        raise HTTPException(
            status_code=404, detail="Meeting media file not found on server"
        )

    ext = os.path.splitext(file_to_stream)[1].lower()
    media_type_map = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".webm": "video/webm",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }
    media_type = media_type_map.get(ext, "video/mp4")

    headers = {
        "Accept-Ranges": "bytes",
    }

    return FileResponse(path=file_to_stream, media_type=media_type, headers=headers)


_MAX_CLIP_SEC = 3600.0  # a clip is a short excerpt; an unbounded range is a transcode-DoS


@router.get("/{meeting_id}/clip")
def get_meeting_clip(
    meeting_id: Any,
    start: float = Query(...),
    end: float = Query(...),
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    start_val, end_val = validate_time_range(start, end)
    if end_val - start_val > _MAX_CLIP_SEC:
        raise HTTPException(
            status_code=400, detail=f"Clip too long: at most {int(_MAX_CLIP_SEC)} seconds per request"
        )

    source_file = _resolve_media_file(meeting)

    if not source_file or not os.path.exists(source_file):
        raise HTTPException(status_code=404, detail="Meeting media file not found")

    ext = os.path.splitext(source_file)[1].lower()
    # Name the clip from the meeting's integer PK and the VALIDATED range. It used to embed the
    # raw {meeting_id} URL segment, which get_by_id() accepts in decorated forms ("BVC-5",
    # "x\\..\\5") — on Windows that let a caller steer the output path out of the clips folder.
    clip_filename = f"clip_{meeting.id}_{start_val:.3f}_{end_val:.3f}{ext}"
    clip_dir = os.path.join(settings.UPLOAD_DIR, "clips")
    os.makedirs(clip_dir, exist_ok=True)
    clip_path = os.path.join(clip_dir, clip_filename)

    if not os.path.exists(clip_path):
        logger.info(
            f"Generating clip from {source_file} [{start_val:.3f} - {end_val:.3f}]"
        )
        is_audio = ext in [".mp3", ".wav", ".m4a", ".aac", ".ogg", ".wma"]

        # Encode into a temp file and move it into place only on success: a killed / failed
        # ffmpeg used to leave a truncated clip that every later request then served from
        # the cache, and two concurrent requests wrote the same file at once.
        tmp_clip_path = os.path.join(clip_dir, f".tmp-{uuid.uuid4().hex[:8]}-{clip_filename}")

        cmd = [
            FFMPEG,
            "-y",
            "-accurate_seek",
            "-ss",
            f"{start_val:.3f}",
            "-to",
            f"{end_val:.3f}",
            "-i",
            source_file,
        ]
        if is_audio:
            cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero", tmp_clip_path]
        else:
            cmd += [
                "-c:v",
                "libx264",
                "-preset",
                "superfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-avoid_negative_ts",
                "make_zero",
                tmp_clip_path,
            ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=300,
            )
            os.replace(tmp_clip_path, clip_path)
        except Exception as err:
            logger.error(f"FFmpeg clipping failed: {err}")
            try:
                os.remove(tmp_clip_path)
            except OSError:
                pass
            raise HTTPException(status_code=500, detail="Failed to slice media clip") from err

    media_type = "video/mp4"
    if ext == ".mp3":
        media_type = "audio/mpeg"
    elif ext == ".wav":
        media_type = "audio/wav"
    elif ext == ".webm":
        media_type = "video/webm"
    elif ext == ".ogg":
        media_type = "audio/ogg"

    return FileResponse(path=clip_path, media_type=media_type)


# Columns never returned by the meeting list: the server-side storage path of the recording.
_LIST_HIDDEN_COLUMNS = {"filepath"}
_LIST_SORTS = {
    "newest": lambda: (Meeting.upload_time.desc().nullslast(), Meeting.id.desc()),
    "oldest": lambda: (Meeting.upload_time.asc().nullslast(), Meeting.id.asc()),
    "title": lambda: (func.lower(func.coalesce(Meeting.title, Meeting.filename)).asc(), Meeting.id.desc()),
    "duration": lambda: (Meeting.duration.desc().nullslast(), Meeting.id.desc()),
}


def _list_item(m: Meeting) -> Dict[str, Any]:
    """One row of the paginated list: what a user needs to recognise and open a meeting - no internals."""
    from backend.utils.processing_status import display_status, user_error

    state = display_status(m.status)
    return {
        "id": m.id,
        "meeting_id": m.meeting_id,
        "title": m.title or None,
        "filename": m.filename,
        "file_type": m.file_type,
        "status": state,
        "duration_seconds": m.duration if m.duration and m.duration > 0 else None,
        "speaker_count": m.speaker_count if m.speaker_count else None,
        "language": m.source_language or None,
        "uploaded_at": m.upload_time,
        "error": user_error(m.error_message) if state == "failed" else None,
    }


@router.get("", response_model=None)
def list_meetings(
    page: Optional[int] = Query(None, ge=1, description="1-based page; any of page/limit/status/search/sort switches "
                                                         "the answer to the paginated envelope"),
    limit: Optional[int] = Query(None, ge=1, le=100),
    status: Optional[str] = Query(None, description="completed | processing | failed"),
    search: Optional[str] = Query(None, max_length=200, description="matches title or file name"),
    sort: Optional[str] = Query(None, description="newest (default) | oldest | title | duration"),
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """List meetings.

    Without query parameters: every meeting's mapped scalar columns, newest first (the original contract, minus the
    server-side ``filepath``). With any of ``page``/``limit``/``status``/``search``/``sort``: a page of
    ``{items, total, page, limit, pages, status_counts}`` where ``status_counts`` covers ALL meetings, not one page.

    PERFORMANCE: no relationship is loaded (``segments`` / ``analysis`` / ``action_items`` used to trigger an N+1
    load); use ``GET /meetings/{id}/final`` for one meeting's full result.
    """
    from sqlalchemy import inspect as _sa_inspect
    from sqlalchemy import or_

    from backend.utils.processing_status import STATUS_FILTERS, display_status

    if all(v is None for v in (page, limit, status, search, sort)):
        meetings = repo.list_all()
        col_keys = [c.key for c in _sa_inspect(Meeting).mapper.column_attrs if c.key not in _LIST_HIDDEN_COLUMNS]
        return jsonable_encoder([{k: getattr(m, k, None) for k in col_keys} for m in meetings])

    status_key = (status or "").strip().lower() or None
    if status_key and status_key not in STATUS_FILTERS:
        raise HTTPException(status_code=422, detail="status must be one of: completed, processing, failed.")
    sort_key = (sort or "newest").strip().lower()
    if sort_key not in _LIST_SORTS:
        raise HTTPException(status_code=422, detail=f"sort must be one of: {', '.join(_LIST_SORTS)}.")
    page = page or 1
    limit = limit or 10

    db = repo.db
    query = db.query(Meeting)
    term = (search or "").strip()
    if term:
        like = "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        query = query.filter(or_(Meeting.title.ilike(like, escape="\\"), Meeting.filename.ilike(like, escape="\\")))

    counts = {"completed": 0, "processing": 0, "failed": 0}
    for raw_status, n in query.with_entities(Meeting.status, func.count(Meeting.id)).group_by(Meeting.status):
        counts[display_status(raw_status)] += n

    if status_key:
        members = STATUS_FILTERS[status_key]
        query = query.filter(
            Meeting.status.in_(members) if status_key != "processing"
            else or_(Meeting.status.is_(None), ~Meeting.status.in_(STATUS_FILTERS["completed"] + STATUS_FILTERS["failed"]))
        )
    total = counts[status_key] if status_key else sum(counts.values())
    rows = query.order_by(*_LIST_SORTS[sort_key]()).offset((page - 1) * limit).limit(limit).all()
    return jsonable_encoder({
        "items": [_list_item(m) for m in rows],
        "total": total,
        "page": page,
        "limit": limit,
        "pages": max(1, -(-total // limit)),
        "status_counts": counts,
    })


@router.delete("/{meeting_id}")
def delete_meeting(
    meeting_id: Any,
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if (meeting.status or "").lower() in ("uploaded", "processing"):
        # The running pipeline would keep writing results for a row that no longer exists.
        raise HTTPException(status_code=409, detail="This meeting is still being processed. Delete it once processing has finished.")

    pk = meeting.id
    media_path = meeting.filepath
    # Delete the database row FIRST. Removing the vector first meant a failing DB delete left
    # a meeting that still exists but is no longer searchable.
    repo.delete(pk)
    try:
        vector_service.delete_meeting_vector(pk)
    except Exception as vec_err:
        logger.warning(f"Vector cleanup failed for deleted meeting {pk}: {vec_err}")
    _delete_meeting_files(pk, media_path)
    return {"message": "Meeting deleted successfully"}


def _delete_meeting_files(meeting_pk: int, media_path: Optional[str]) -> None:
    """Remove a deleted meeting's upload and generated clips — only ever inside UPLOAD_DIR.

    Deleting the row used to leave the recording on disk forever (storage leak, and a
    "deleted" meeting's audio/video stayed recoverable).
    """
    upload_root = os.path.realpath(settings.UPLOAD_DIR)

    def _inside_upload_dir(path: str) -> bool:
        real = os.path.realpath(path)
        return real == upload_root or real.startswith(upload_root + os.sep)

    targets = []
    if media_path:
        targets.append(media_path)
    targets += glob.glob(os.path.join(glob.escape(os.path.join(settings.UPLOAD_DIR, "clips")), f"clip_{meeting_pk}_*"))
    for path in targets:
        try:
            if os.path.isfile(path) and _inside_upload_dir(path):
                os.remove(path)
        except OSError as file_err:
            logger.warning(f"Could not remove file {path} of deleted meeting {meeting_pk}: {file_err}")


@router.get("/{meeting_id}/verify")
def verify_meeting(
    meeting_id: Any,
    force_regenerate: bool = Query(
        False, description="Force regeneration of missing/placeholder reports"
    ),
    db: Session = Depends(get_db),
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """Run quality checks on meeting outputs — uses live computation, never stale DB cache."""

    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if meeting.status != "completed" and not meeting.analysis:
        raise HTTPException(status_code=400, detail="Meeting processing is not yet completed")

    # Auto-heal placeholder or missing outputs dynamically
    is_sum_ok, is_mom_ok = False, False
    if meeting.analysis:
        is_sum_ok, is_mom_ok = check_outputs_generated(
            meeting.analysis.summary or "", meeting.analysis.mom or "", meeting.filename or ""
        )
    should_regenerate = force_regenerate or (not is_sum_ok or not is_mom_ok)

    recalculate_meeting_metrics(meeting, db, skip_regeneration=not should_regenerate)
    db.commit()

    segments = repo.get_segments(meeting.id)
    analysis = meeting.analysis
    summary_text = analysis.summary if analysis else ""
    mom_text = analysis.mom if analysis else ""
    action_items_text = analysis.action_items if analysis else ""
    decisions_text = analysis.decisions if analysis else ""
    risks_text = analysis.risks if analysis else ""

    is_document = getattr(meeting, "file_type", "media") == "document"
    is_sum_ok, is_mom_ok = check_outputs_generated(
        summary_text, mom_text, meeting.filename or ""
    )

    # Derive total text from segments or fallback to generated analysis text
    total_text = " ".join([s.text for s in segments if s.text]).strip()
    if not total_text and analysis:
        total_text = " ".join(filter(None, [summary_text, mom_text, action_items_text])).strip()

    if is_document:
        # OCR accuracy requires a reference-extracted text; without it, Not Measured.
        live_ta = None
    else:
        live_ta = None  # transcription quality is only measured vs Ground Truth

    # Persist only measured values (Ground-Truth-derived, never fabricated)
    if live_ta is not None and live_ta > (meeting.transcription_accuracy or 0.0):
        meeting.transcription_accuracy = live_ta
        db.commit()

    # Filter valid speaker labels (exclude empty, None, UNKNOWN-only labels)
    speakers = {
        s.speaker
        for s in segments
        if s.speaker
        and s.speaker.strip()
        and s.speaker.strip().upper() not in ("", "NONE")
    }
    unknown_speakers = {
        sp for sp in speakers if "unknown" in sp.lower() or "unidentified" in sp.lower()
    }
    known_speakers = speakers - unknown_speakers

    checks = []

    # Check 1: Transcript / Source Text Coverage
    word_count = len(total_text.split()) if total_text else 0
    if word_count == 0:
        transcript_status = "fail"
        transcript_msg = "Source text appears empty. Re-process the meeting."
    else:
        transcript_status = "pass"
        transcript_msg = f"Source text captured with {word_count} words total."

    checks.append(
        {
            "name": (
                "Recording Transcription Coverage"
                if not is_document
                else "Document Text Extraction Coverage"
            ),
            "status": transcript_status,
            "message": transcript_msg,
            "icon": "FileText",
        }
    )

    # Check 2: Speaker Identification
    if is_document or len(known_speakers) >= 1:
        spk_status = "pass"
        spk_msg = f"Participants processed. {max(len(known_speakers), 1)} speaker/section(s) identified."
    elif len(speakers) >= 1:
        spk_status = "warn"
        spk_msg = f"{len(speakers)} speaker label(s) found but marked as unknown."
    else:
        spk_status = "pass" if (is_sum_ok and is_mom_ok) else "warn"
        spk_msg = "Speaker identification and diarization completed."
    checks.append(
        {
            "name": "Speaker Identification",
            "status": spk_status,
            "message": spk_msg,
            "icon": "Users",
        }
    )

    # Check 3: Audio Quality Score
    aq = round(meeting.audio_quality or 0.0, 1)
    if is_document:
        aq_status = "pass"
        aq_msg = "Audio check skipped for document uploads."
    elif aq >= 75:
        aq_status = "pass"
        aq_msg = f"Audio clarity rated {aq}/100."
    elif aq >= 50:
        aq_status = "warn"
        aq_msg = f"Audio clarity rated {aq}/100. Background noise detected."
    else:
        aq_status = "warn" if (is_sum_ok and is_mom_ok) else "fail"
        aq_msg = f"Audio clarity rated {aq}/100."
    checks.append(
        {
            "name": "Audio Quality Score",
            "status": aq_status,
            "message": aq_msg,
            "icon": "Mic",
        }
    )

    # Check 4: Transcription Accuracy / OCR
    ta = live_ta
    if is_document:
        ta_status = "pass"
        ta_msg = f"OCR text quality rated {ta:.1f}%." if ta is not None else "OCR text extraction completed."
    elif ta is not None and ta >= 70:
        ta_status = "pass"
        ta_msg = f"ASR/OCR accuracy rated {ta:.1f}%. Meets threshold."
    elif is_sum_ok and is_mom_ok:
        ta_status = "pass"
        ta_msg = "Transcript processed. Reference-based accuracy requires Ground Truth data."
    else:
        ta_status = "fail"
        ta_msg = (
            f"ASR/OCR accuracy rated {ta:.1f}%. Accuracy below threshold."
            if ta is not None
            else "Transcript accuracy could not be verified. Add Ground Truth to measure."
        )
    checks.append(
        {
            "name": "Transcription Accuracy / OCR",
            "status": ta_status,
            "message": ta_msg,
            "icon": "CheckCircle",
        }
    )

    # Check 5: Metadata Extraction
    has_sentiment = bool(meeting.sentiment and meeting.sentiment.strip())
    has_keywords = bool(
        meeting.keyword_frequency
        and meeting.keyword_frequency not in ("{}", "null", "")
    )
    meta_ok = has_sentiment or has_keywords or (is_sum_ok and is_mom_ok)
    checks.append(
        {
            "name": "Metadata Extraction",
            "status": "pass" if meta_ok else "warn",
            "message": (
                "Sentiment, emotions, and keyword metadata fully extracted."
                if meta_ok
                else "Some metadata fields could not be fully extracted."
            ),
            "icon": "Database",
        }
    )

    # Check 6: Executive Summary Status
    checks.append(
        {
            "name": "Executive Summary Status",
            "status": "pass" if is_sum_ok else "fail",
            "message": (
                "Executive Summary successfully generated and validated."
                if is_sum_ok
                else "Executive Summary failed to generate or is missing."
            ),
            "icon": "Target",
        }
    )

    # Check 7: Minutes of Meeting Status
    checks.append(
        {
            "name": "Minutes of Meeting Status",
            "status": "pass" if is_mom_ok else "fail",
            "message": (
                "Minutes of Meeting (MoM) successfully generated and validated."
                if is_mom_ok
                else "Minutes of Meeting (MoM) failed to generate or is missing."
            ),
            "icon": "FileText",
        }
    )

    # Check 8: Action Items & Key Insights
    GENERIC_STUBS = {
        "action items to be reviewed",
        "key decisions recorded",
        "no action items identified",
        "no decisions recorded",
        "no major risks",
        "placeholder",
        "tbd",
    }

    def _is_generic(text: str) -> bool:
        return not text or any(stub in text.lower() for stub in GENERIC_STUBS)

    has_actions = bool(action_items_text) and not _is_generic(action_items_text)
    has_decisions = bool(decisions_text) and not _is_generic(decisions_text)
    has_risks = bool(risks_text) and not _is_generic(risks_text)
    insights_ok = has_actions or has_decisions or has_risks or (is_sum_ok and is_mom_ok)
    checks.append(
        {
            "name": "Action Items & Key Insights",
            "status": "pass" if insights_ok else "warn",
            "message": (
                "Action items and key decisions successfully extracted."
                if insights_ok
                else "Some key insights or action items are missing or generic."
            ),
            "icon": "ListTodo",
        }
    )

    has_failures = any(c["status"] == "fail" for c in checks)
    can_approve = not has_failures

    if has_failures and meeting.is_approved:
        meeting.is_approved = False
        db.commit()

    return {
        "checks": checks,
        "overall": "pass" if can_approve else "fail",
        "can_approve": can_approve,
    }



@router.post("/{meeting_id}/approve", response_model=schemas.Meeting)
def approve_meeting(
    meeting_id: Any,
    request: schemas.ApproveRequest,
    db: Session = Depends(get_db),
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """Approve meeting and generate final reports."""
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    if meeting.analysis and (
        not meeting.analysis.summary or meeting.analysis.summary.strip() == ""
    ):
        try:
            logger.info("Generating reports upon human approval...")
            segments = repo.get_segments(meeting_id)
            # segment.speaker already reads "Speaker 1" / a real name — prefixing "Speaker "
            # produced "Speaker Speaker 1:" in the LLM input (and thus in the report).
            full_transcript_text = "\n".join(
                [f"{s.speaker}: {s.text}" for s in segments]
            )
            if full_transcript_text.strip():
                m_date = meeting.meeting_date or "Unknown Date"
                m_time = meeting.meeting_time or "Unknown Time"
                m_dur = format_seconds_to_duration(meeting.duration)

                # FULL transcript — process_meeting_report fits it to the context
                # window per section and condenses a too-long meeting for the
                # summary in parts.
                reports = process_meeting_report(
                    full_transcript_text,
                    meeting_date=m_date,
                    meeting_time=m_time,
                    duration_str=m_dur,
                )
                meeting.analysis.summary = reports["summary"]
                meeting.analysis.mom = reports["mom"]
                meeting.analysis.action_items = reports["action_items"]
                meeting.analysis.decisions = reports["decisions"]
                meeting.analysis.risks = reports["risks"]
                meeting.project_name = reports.get(
                    "project_name", DEFAULT_PROJECT_NAME
                )
        except Exception as e:
            logger.error(f"Failed to generate reports upon approval: {str(e)}")

    meeting.is_approved = True
    if meeting.analysis:
        if request.summary is not None and request.summary.strip() != "":
            meeting.analysis.summary = request.summary
        if request.mom is not None and request.mom.strip() != "":
            meeting.analysis.mom = request.mom
        if request.action_items is not None and request.action_items.strip() != "":
            meeting.analysis.action_items = request.action_items
        if request.decisions is not None and request.decisions.strip() != "":
            meeting.analysis.decisions = request.decisions
        if request.risks is not None and request.risks.strip() != "":
            meeting.analysis.risks = request.risks
        if request.agenda_key_decisions:
            db.query(AgendaKeyDecision).filter(AgendaKeyDecision.meeting_id == meeting.id).delete()
            for idx, item in enumerate(request.agenda_key_decisions, start=1):
                db.add(AgendaKeyDecision(
                    meeting_id=meeting.id,
                    bharatvc_meeting_id=meeting.bharatvc_meeting_id,
                    agenda_order=item.agenda_order or idx,
                    agenda=item.agenda,
                    key_decision=item.key_decision or "No decision recorded."
                ))

    recalculate_meeting_metrics(meeting, db, skip_regeneration=False)
    db.commit()
    db.refresh(meeting)

    # Sync vector store
    try:
        from backend.utils import text_signals as ts

        meeting_code = meeting.display_id or meeting.bharatvc_meeting_id or str(meeting.id)
        segments = repo.get_segments(meeting_id)
        full_transcript_text = "\n".join(
            [f"{s.speaker}: {s.text}" for s in segments]
        ) or (meeting.title or "")
        mom_text = (meeting.analysis.mom if meeting.analysis else "") or (
            clean_agenda_text(meeting.agenda) or ""
        )
        fingerprint = ts.content_fingerprint(full_transcript_text)
        meeting.content_fingerprint = fingerprint
        db.commit()
        vector_service.add_meeting_vector(
            meeting_id=meeting.id,
            meeting_code=meeting_code,
            filename=meeting.filename,
            title=meeting.title or meeting.filename,
            text=full_transcript_text,
            mom_summary=mom_text[:800],
            project_name=meeting.project_name,
            extra_metadata=build_meeting_vector_metadata(meeting, fingerprint),
        )
    except Exception as sync_err:
        logger.warning(
            f"Vector store sync on approval warning for meeting {meeting_id}: {sync_err}"
        )

    # Refresh AI_Meeting_Inteligent_final (and its consolidated json_data snapshot)
    # from the now-approved source tables. build_final_record() re-reads every
    # feature table and rebuilds json_data itself, so no separate override here.
    try:
        from backend.services.final_aggregation_service import build_final_record
        build_final_record(meeting.id, db)
        db.commit()
    except Exception as _agg_err:
        logger.warning(f"Could not refresh AI_Meeting_Inteligent_final on approve: {_agg_err}")

    return meeting


# ─── NVIDIA NeMo Speaker Diarization API Endpoints ──────────────────────────

def _build_nemo_diarization_response(meeting: Meeting, db: Session) -> Dict[str, Any]:
    from backend.models.models import TranscriptSegment
    from backend.services.diarization_service import compute_speaker_diagnostics, stored_speech_durations

    segments = db.query(TranscriptSegment).filter(TranscriptSegment.meeting_id == meeting.id).all()
    seg_dicts = [
        {
            "speaker": s.speaker,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "text": s.text,
        }
        for s in segments
    ]
    # Talk time = the speech-only durations the diarization stored (segment spans include the pauses).
    diag = compute_speaker_diagnostics(
        seg_dicts, meeting.duration or 0.0,
        speech_durations=stored_speech_durations(getattr(meeting, "diarization_info_json", None)),
    )

    speakers_list = []
    for spk_id, info in sorted(diag.get("speaker_breakdown", {}).items()):
        speakers_list.append({
            "speaker_id": spk_id,
            "speaking_duration_seconds": info["speaking_duration_seconds"],
            "speaking_percentage": info["speaking_percentage"],
            "turn_count": info.get("turn_count"),
            "avg_turn_duration_sec": info.get("avg_turn_duration_sec"),
            "first_appearance": info.get("first_appearance"),
            "last_appearance": info.get("last_appearance"),
        })

    status_val = getattr(meeting, "diarization_status", None) or "SUCCESS"

    # Which speaker-embedding model actually produced the speaker count. NeMo TitaNet is a neural voice embedder; if it
    # could not be loaded (not installed, no .nemo file on an offline server) the pipeline runs on a weak acoustic
    # fallback that can merge or split speakers - a consumer must be able to tell (metadata is recorded per meeting).
    try:
        meta = json.loads(getattr(meeting, "diarization_metadata_json", None) or "{}")
    except (TypeError, ValueError):
        meta = {}
    neural = meta.get("neural_embedder_used") if isinstance(meta, dict) else None
    diarization = {
        "status": status_val,
        "engine": "NVIDIA_NEMO",
        "speaker_count": len(speakers_list),
        "speakers": speakers_list,
        "neural_embedder_used": neural,
        "embedding_fidelity": "titanet_neural" if neural is True else "acoustic_fallback" if neural is False else "unknown",
        "selected_engine": meta.get("selected_engine") if isinstance(meta, dict) else None,
        "fallback_used": meta.get("fallback_used") if isinstance(meta, dict) else None,
    }
    if neural is False:
        diarization["warning"] = (
            "The NeMo TitaNet speaker model was not available, so the speaker count comes from a weaker acoustic "
            "fallback embedder and may merge or split speakers. Provide models/diarization/titanet_large.nemo "
            "(python -m backend.scripts.fetch_models --diarization)."
        )

    return {
        "meeting_id": str(meeting.display_id or meeting.bharatvc_meeting_id or meeting.id),
        "diarization": diarization,
    }


@router.get(
    "/{meeting_id}/diarization",
    summary="Get NVIDIA NeMo Diarization & Speaker Analytics",
    description="Returns NVIDIA NeMo speaker diarization status, speaker count, and calculated speaking percentages.",
)
def get_meeting_diarization(
    meeting_id: Any,
    db: Session = Depends(get_db),
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return _build_nemo_diarization_response(meeting, db)


@router.get(
    "/{meeting_id}/speaker-analytics",
    summary="Get Detailed Speaker Analytics",
    description="Returns detailed speaker analytics calculated dynamically from NeMo diarization segments.",
)
def get_speaker_analytics(
    meeting_id: Any,
    db: Session = Depends(get_db),
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return _build_nemo_diarization_response(meeting, db)


@router.get(
    "/{meeting_id}/status",
    summary="Get Meeting Processing Status",
    description="Returns real-time pipeline status and individual stage progress for a meeting.",
)
def get_meeting_status(
    meeting_id: Any,
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting {meeting_id} not found")
    return _processing_status(meeting)


# Stage state (StageTracker) -> state reported to clients.
_STAGE_STATE = {
    "SUCCESS": "completed", "COMPLETED": "completed", "DEGRADED": "completed",
    "FAILED": "failed", "REJECTED": "failed",
    "PROCESSING": "in_progress", "RETRYING": "in_progress",
    "PENDING": "pending",
}
# The three steps a user sees, and the pipeline stages behind each.
_USER_STEPS = (
    ("transcribe", "Processing meeting",
     ("audio_extraction", "stt", "diarization", "alignment", "transcript_correction")),
    ("results", "Preparing results",
     ("summary", "mom", "agenda", "decision", "action_items", "topics", "sentiment", "embeddings_vector", "rag_chatbot")),
)


def _processing_status(meeting: Meeting) -> Dict[str, Any]:
    """Processing progress from the stage states the pipeline actually recorded.

    ``progress_percent`` used to be invented (50 for every running meeting, 100 for a FAILED one) and a stage in
    any state other than success/failed was reported as running - a skipped or pending stage included.
    """
    from backend.loop_engine.stage_manager import ALL_PIPELINE_STAGES
    from backend.utils.processing_status import display_status, user_error

    state = display_status(meeting.status)
    raw: Dict[str, str] = {}
    stage_json = getattr(meeting, "pipeline_stage_states_json", None)
    if stage_json:
        try:
            parsed = json.loads(stage_json) if isinstance(stage_json, str) else stage_json
            if isinstance(parsed, dict):
                raw = {str(k): str(v).upper() for k, v in parsed.items()}
        except (TypeError, ValueError) as exc:
            logger.warning("Meeting %s has unreadable stage states (%s); reporting none.", meeting.id, exc)

    stages = {k: {"stage": k, "status": _STAGE_STATE.get(v, "pending")} for k, v in raw.items()}
    settled = sum(1 for k in ALL_PIPELINE_STAGES if stages.get(k, {}).get("status") in ("completed", "failed"))
    if state == "completed":
        progress = 100
    else:
        progress = min(95, round(100 * settled / len(ALL_PIPELINE_STAGES)))

    def step_state(keys) -> str:
        states = [stages.get(k, {}).get("status", "pending") for k in keys]
        if "failed" in states:
            return "failed"
        if all(s == "completed" for s in states):
            return "completed"
        if any(s in ("completed", "in_progress") for s in states):
            return "in_progress"
        return "pending"

    steps = [{"key": "upload", "label": "Uploading", "state": "completed"}]
    for key, label, keys in _USER_STEPS:
        s = "completed" if state == "completed" else step_state(keys)
        steps.append({"key": key, "label": label, "state": s})
    if state == "processing" and all(s["state"] == "pending" for s in steps[1:]):
        steps[1]["state"] = "in_progress"       # claimed or queued: the first step is what happens next
    steps.append({"key": "done", "label": "Completed" if state != "failed" else "Needs attention",
                  "state": "completed" if state == "completed" else ("failed" if state == "failed" else "pending")})

    err = user_error(meeting.error_message) if state == "failed" else None
    return {
        "meeting_id": meeting.id,
        "public_meeting_id": meeting.meeting_id or meeting.bharatvc_meeting_id,
        "status": state,
        "overall_status": {"completed": "COMPLETED", "failed": "FAILED"}.get(state, "PROCESSING"),
        "progress_percent": progress,
        "has_warnings": any(v == "DEGRADED" for v in raw.values()),
        "steps": steps,
        "stages": stages,
        "error": err["message"] if err else None,
        "error_code": err["code"] if err else None,
    }


@router.post(
    "/{meeting_id}/retry",
    summary="Retry Meeting Processing Pipeline",
    description="Re-triggers the intelligence processing pipeline for a failed or stuck meeting.",
)
def retry_meeting_pipeline(
    meeting_id: Any,
    background_tasks: BackgroundTasks,
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting {meeting_id} not found")

    # A running pipeline must not get a second one on the same meeting (the claim would block it, but the reset
    # below would first overwrite the running run's status).
    if (meeting.status or "").lower() in ("uploaded", "processing"):
        raise HTTPException(status_code=409, detail="This meeting is still being processed.")

    file_path = meeting.filepath
    if not file_path or not os.path.exists(file_path):
        logger.warning("Retry of meeting %s refused: recording missing at %s", meeting.id, file_path)
        raise HTTPException(
            status_code=410,
            detail="The original recording is no longer available, so this meeting cannot be processed again. "
                   "Upload the recording again.",
        )

    # Reset status so _claim_meeting can take ownership
    meeting.status = "uploaded"
    meeting.error_message = None
    repo.db.commit()

    from backend.workers import PIPELINE_REGISTRY, process_meeting_pipeline
    from backend.api.deps import release_session_before_background_task

    file_type = meeting.file_type or "media"
    pipeline = PIPELINE_REGISTRY.get(file_type, process_meeting_pipeline)
    meeting_pk = meeting.id

    release_session_before_background_task(repo.db)
    background_tasks.add_task(pipeline, meeting_pk, file_path)

    return {
        "success": True,
        "message": "Processing has been restarted.",
        "meeting_id": meeting_pk,
        "status": "processing",
    }



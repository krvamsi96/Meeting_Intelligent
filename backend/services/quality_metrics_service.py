"""Database glue for the real-time reference-free quality scores (transcript, summary, RAG, action items,
decisions, agenda coverage).

The scoring itself lives in ``backend.evaluation.quality_scores`` (pure functions). This module only
gathers ONE meeting's own rows — every query is keyed by that meeting's primary key, so another
meeting's transcript, summary or chat can never enter its scores — and stores per-answer RAG results.

Every database read that touches the new ``rag_evaluations`` table runs inside a SAVEPOINT: on a
database that has not been migrated yet the failed query rolls back only its own savepoint and can
never abort the caller's transaction (GET /meetings/{id} rebuilds these scores on every request).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.evaluation import quality_scores as qs
from backend.models.models import (
    ActionItem,
    AgendaKeyDecision,
    ChatSession,
    Meeting,
    RagEvaluation,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

# Meeting-level RAG groundedness averages the most recent answers only.
MAX_RAG_ANSWERS = 50


def _load_rag_evaluations(db: Session, meeting_pk: int) -> Tuple[List[Dict[str, Any]], int]:
    """(stored per-answer evaluations, number of chatbot answers) for THIS meeting only."""
    evaluations: List[Dict[str, Any]] = []
    try:
        with db.begin_nested():
            rows = (
                db.query(RagEvaluation.result_json)
                .join(ChatSession, ChatSession.id == RagEvaluation.chat_message_id)
                .filter(RagEvaluation.meeting_id == meeting_pk, ChatSession.meeting_id == meeting_pk)
                .order_by(RagEvaluation.id.desc())
                .limit(MAX_RAG_ANSWERS)
                .all()
            )
        for (raw,) in rows:
            try:
                evaluations.append(json.loads(raw))
            except (TypeError, ValueError):
                continue
    except Exception as e:
        logger.debug("[QualityMetrics] rag_evaluations unavailable for meeting %s: %s", meeting_pk, e)

    answers = 0
    try:
        with db.begin_nested():
            answers = int(
                db.query(func.count(ChatSession.id))
                .filter(ChatSession.meeting_id == meeting_pk, ChatSession.role == "assistant")
                .scalar() or 0
            )
    except Exception as e:
        logger.debug("[QualityMetrics] chat answer count unavailable for meeting %s: %s", meeting_pk, e)
    return evaluations, answers


def load_quality_inputs(meeting: Meeting, db: Session) -> Dict[str, Any]:
    """Everything ``build_quality_metrics`` needs for ONE meeting, as plain data (no ORM objects).

    This is the only part that touches PostgreSQL. The scoring that follows embeds text (seconds of CPU on a cold
    cache), so callers that care split the two - read here, end the transaction, THEN call
    ``compute_quality_metrics_from_inputs`` - instead of holding a connection through the embedding work.
    """
    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.meeting_id == meeting.id)
        .order_by(TranscriptSegment.id.asc())
        .all()
    )
    seg_dicts = [
        {
            "speaker": s.speaker,
            "text": s.text,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "stt_confidence": s.stt_confidence,
        }
        for s in segments
    ]
    transcript_text = "\n".join(f"{d['speaker'] or ''}: {d['text']}" for d in seg_dicts if (d["text"] or "").strip())

    analysis = getattr(meeting, "analysis", None)
    summary_text = (analysis.summary if analysis else None) or ""

    agenda_rows = (
        db.query(AgendaKeyDecision)
        .filter(AgendaKeyDecision.meeting_id == meeting.id)
        .order_by(AgendaKeyDecision.agenda_order)
        .all()
    )
    agenda_topics = [a.agenda for a in agenda_rows if a.agenda]
    decisions = [
        {"decision": a.key_decision, "evidence": a.evidence, "speaker": a.speaker}
        for a in agenda_rows if (a.key_decision or "").strip()
    ]
    action_rows = (
        db.query(ActionItem).filter(ActionItem.meeting_id == meeting.id).order_by(ActionItem.item_order).all()
    )
    action_items = [
        {"task": r.task, "owner": r.owner, "deadline": r.deadline, "due_date": r.due_date, "evidence": r.evidence}
        for r in action_rows
    ]
    rag_evals, answer_count = _load_rag_evaluations(db, meeting.id)

    is_document = (getattr(meeting, "file_type", "media") or "media") == "document"
    return {
        "segments": seg_dicts,
        "summary_text": summary_text,
        "transcript_text": transcript_text,
        "agenda_topics": agenda_topics,
        "rag_evaluations": rag_evals,
        "assistant_answer_count": answer_count,
        "audio_duration_sec": None if is_document else (getattr(meeting, "duration", None) or None),
        "language_confidence": getattr(meeting, "source_language_confidence", None),
        "is_document": is_document,
        "expected_language": getattr(meeting, "output_language", None) or "en",
        "action_items": action_items,
        "decisions": decisions,
    }


def compute_quality_metrics_from_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """The ``metrics`` block from ``load_quality_inputs`` data. Pure computation: no database access."""
    return qs.build_quality_metrics(**inputs)


def compute_quality_metrics(meeting: Meeting, db: Session) -> Dict[str, Any]:
    """The ``metrics`` block for one meeting, computed live from that meeting's own stored data."""
    return compute_quality_metrics_from_inputs(load_quality_inputs(meeting, db))


def evaluate_rag_answer(
    *,
    meeting_pk: int,
    question: str,
    answer: str,
    context_chunks: Sequence[Any],
    source_chunks: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Score one chatbot answer against the context it was generated from. Never raises."""
    try:
        return qs.rag_answer_groundedness(
            question=question, answer=answer, context_chunks=context_chunks,
            source_chunks=source_chunks, meeting_id=meeting_pk,
        )
    except Exception as e:
        logger.error("[QualityMetrics] RAG answer evaluation failed for meeting %s: %s", meeting_pk, e, exc_info=True)
        return None


def store_rag_evaluation(
    db: Session, *, meeting_pk: int, chat_message_id: Optional[int], evaluation: Optional[Dict[str, Any]]
) -> bool:
    """Persist one answer's evaluation (upsert by chat message). Never raises; returns success."""
    if not evaluation or chat_message_id is None:
        return False
    try:
        with db.begin_nested():
            row = db.query(RagEvaluation).filter(RagEvaluation.chat_message_id == chat_message_id).first()
            payload = json.dumps(evaluation, default=str)
            if row is None:
                row = RagEvaluation(meeting_id=meeting_pk, chat_message_id=chat_message_id)
                db.add(row)
            row.meeting_id = meeting_pk
            row.score = evaluation.get("score")
            row.label = evaluation.get("label")
            row.result_json = payload
            db.flush()
        return True
    except Exception as e:
        logger.warning("[QualityMetrics] Could not store RAG evaluation for message %s: %s", chat_message_id, e)
        return False

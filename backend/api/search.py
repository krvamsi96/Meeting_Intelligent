"""GET /api/v1/search - semantic search across all completed meetings' transcripts, with source references."""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.config.config import settings
from backend.models.models import Meeting
from backend.services.vector_service import VectorSearchUnavailable, vector_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])


def _seconds(value: Any):
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


@router.get("")
def search_meetings(
    q: str = Query(..., min_length=2, max_length=300, description="What to look for, in plain language"),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Transcript passages related to ``q``, each with its meeting, speaker and time range.

    Only meetings that exist and finished processing are returned; a passage of a deleted meeting still in the
    index is dropped. ``similarity`` is topical closeness (cosine), not proof that the passage answers the query.
    """
    min_similarity = float(getattr(settings, "SEARCH_MIN_SIMILARITY", 0.2))
    try:
        hits = vector_service.search_all_meetings(q, top_k=limit * 2, min_similarity=min_similarity)
    except VectorSearchUnavailable as exc:
        logger.warning("Search unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Search is temporarily unavailable. Please try again later.") from exc

    ids = sorted({h["meeting_id"] for h in hits})
    meetings = {
        m.id: m for m in db.query(Meeting).filter(Meeting.id.in_(ids), Meeting.status == "completed").all()
    } if ids else {}
    results: List[Dict[str, Any]] = []
    for h in hits:
        m = meetings.get(h["meeting_id"])
        if m is None:
            continue
        results.append({
            "meeting_id": m.id,
            "meeting_title": m.title or m.filename,
            "uploaded_at": m.upload_time,
            "speaker": h.get("speakers") or None,
            "start_seconds": _seconds(h.get("start_time")),
            "end_seconds": _seconds(h.get("end_time")),
            "text": h.get("text") or "",
            "similarity": h["similarity"],
        })
        if len(results) >= limit:
            break
    return {"query": q, "results": results, "total": len(results)}

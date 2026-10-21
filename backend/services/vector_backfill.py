"""Populate ChromaDB from PostgreSQL for meetings that have no vectors yet.

PostgreSQL holds every transcript, so vectors are derived data. When ChromaDB becomes the vector store (or a new
CHROMA_PERSIST_DIRECTORY is configured) the meetings that were processed earlier are not in it, which would make
similar-meeting / duplicate detection blind to them. This module closes that gap without any hardcoded ids:

* ``start_background_backfill()`` - run at application startup (VECTOR_BACKFILL_ON_STARTUP, default on): a daemon
  thread adds the whole-meeting vector of every meeting that lacks one. One embedding per meeting, so it is quick;
  chunk vectors for RAG are created lazily the first time a meeting is chatted with (or by ``reindex_vectors``).
* ``backfill(...)`` - the same logic on demand; ``python -m backend.scripts.reindex_vectors`` uses it to rebuild
  everything, chunks included.

Re-running is always safe: vector ids are deterministic, so vectors are updated in place, never duplicated.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.utils.helpers import clean_agenda_text

from backend.config.config import settings
from backend.services.vector_service import vector_service

logger = logging.getLogger(__name__)

# Meetings still in the pipeline are indexed by the pipeline itself when it finishes.
_IN_PROGRESS_STATUSES = ("processing", "uploaded")


def index_meeting(repo, meeting, *, chunks: bool = True) -> Tuple[bool, int]:
    """Index ONE meeting from its PostgreSQL transcript; returns ``(indexed, chunk_count)``.

    ``indexed`` is False when the meeting has no transcript or the vector store rejected it.
    """
    from backend.services.meeting_vector_metadata import build_meeting_vector_metadata
    from backend.utils import text_signals as ts

    segments = repo.get_segments(meeting.id)
    if not segments:
        return False, 0
    text = "\n".join(f"{s.speaker}: {s.text}" for s in segments) or (meeting.title or "")
    chunk_count = vector_service.add_transcript_chunks(meeting.id, segments) if chunks else 0
    mom = (meeting.analysis.mom if getattr(meeting, "analysis", None) else "") or (clean_agenda_text(meeting.agenda) or "")
    stored = vector_service.add_meeting_vector(
        meeting_id=meeting.id,
        meeting_code=meeting.display_id or meeting.bharatvc_meeting_id or str(meeting.id),
        filename=meeting.filename,
        title=meeting.title or meeting.filename,
        text=text,
        mom_summary=mom[:800],
        project_name=meeting.project_name,
        extra_metadata=build_meeting_vector_metadata(meeting, meeting.content_fingerprint or ts.content_fingerprint(text)),
    )
    return bool(stored), chunk_count


def _is_indexed(meeting, indexed: Dict[int, str]) -> bool:
    """A whole-meeting vector exists for this meeting id AND belongs to it (same bharatvc id when it has one)."""
    if meeting.id not in indexed:
        return False
    wanted = str(meeting.bharatvc_meeting_id or meeting.meeting_id or "").strip()
    return not wanted or indexed[meeting.id] == wanted


def backfill(
    meeting_ids: Optional[List[int]] = None,
    *,
    chunks: bool = False,
    only_missing: bool = True,
    dry_run: bool = False,
    repo=None,
) -> Dict[str, int]:
    """Index meetings from PostgreSQL into ChromaDB; returns counters.

    ``only_missing``: skip meetings that already have their whole-meeting vector. ``chunks``: also (re)build the
    transcript chunk vectors used for RAG. ``repo``: a MeetingRepository (default: a fresh database session).
    """
    stats = {"meetings": 0, "chunks": 0, "already_indexed": 0, "skipped_no_transcript": 0,
             "skipped_in_progress": 0, "failed": 0}
    if not vector_service.check_ready():
        raise RuntimeError(f"ChromaDB is unavailable: {vector_service.init_error or vector_service.init_state}")

    db = None
    if repo is None:
        from backend.database import SessionLocal
        from backend.repositories import MeetingRepository

        db = SessionLocal()
        repo = MeetingRepository(db)
    try:
        indexed = vector_service.indexed_meeting_vectors() if only_missing else {}
        meetings = repo.list_all()
        if meeting_ids:
            wanted = set(meeting_ids)
            meetings = [m for m in meetings if m.id in wanted]

        todo = []
        for meeting in sorted(meetings, key=lambda m: m.id):
            if (meeting.status or "") in _IN_PROGRESS_STATUSES:
                stats["skipped_in_progress"] += 1
            elif only_missing and _is_indexed(meeting, indexed):
                stats["already_indexed"] += 1
            else:
                todo.append(meeting)
        if todo and not dry_run:
            # Without the model every meeting would fail the same way; stop once with the reason instead.
            from backend.services.model_loader import EmbeddingModelUnavailable, get_embedding_encoder
            try:
                get_embedding_encoder()
            except EmbeddingModelUnavailable as exc:
                raise RuntimeError(f"{len(todo)} meeting(s) not indexed - embedding model unavailable: {exc}") from exc
        if todo:
            logger.info("Vector backfill: %d meeting(s) %s ChromaDB", len(todo),
                        "would be added to" if dry_run else "to add to")

        for meeting in todo:
            label = f"meeting {meeting.id} ({meeting.display_id or meeting.bharatvc_meeting_id or '-'})"
            try:
                if dry_run:
                    if repo.get_segments(meeting.id):
                        stats["meetings"] += 1
                    else:
                        stats["skipped_no_transcript"] += 1
                    continue
                done, chunk_count = index_meeting(repo, meeting, chunks=chunks)
                if done:
                    stats["meetings"] += 1
                    stats["chunks"] += chunk_count
                    logger.debug("%s: indexed (%d chunk(s))", label, chunk_count)
                elif not repo.get_segments(meeting.id):
                    stats["skipped_no_transcript"] += 1
                else:
                    stats["failed"] += 1
                    logger.warning("%s: could not be indexed", label)
            except Exception as exc:
                stats["failed"] += 1
                logger.error("%s: failed - %s", label, exc)
            time.sleep(0.01)                      # stay polite to uploads/requests running in the same process
    finally:
        if db is not None:
            db.close()
    return stats


# State of the startup backfill in this process, reported by /system/health as "vector_indexing".
STARTUP_STATUS: Dict[str, Any] = {"state": "not_started"}


def indexing_status() -> Dict[str, Any]:
    """Startup backfill state plus how many whole-meeting vectors ChromaDB holds now."""
    status = dict(STARTUP_STATUS)
    try:
        if vector_service.check_ready():
            status["meeting_vectors"] = vector_service._meetings.count()  # noqa: SLF001 - count only
    except Exception as exc:  # noqa: BLE001 - health must not raise
        status["count_error"] = str(exc)
    return status


def _run_startup_backfill() -> None:
    STARTUP_STATUS.clear()
    STARTUP_STATUS.update(state="running", started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        stats = backfill(chunks=False, only_missing=True)
        STARTUP_STATUS.update(state="done" if not stats["failed"] else "done_with_failures", **stats)
        if stats["meetings"] or stats["failed"]:
            logger.info(
                "Vector backfill done: %d meeting(s) added to ChromaDB, %d already indexed, %d without transcript, "
                "%d failed", stats["meetings"], stats["already_indexed"], stats["skipped_no_transcript"], stats["failed"])
        else:
            logger.info("Vector backfill: ChromaDB already has every meeting (%d checked)", stats["already_indexed"])
    except Exception as exc:
        STARTUP_STATUS.update(state="skipped", error=str(exc))
        logger.warning("Vector backfill skipped: %s", exc)


def start_background_backfill() -> Optional[threading.Thread]:
    """Start the startup backfill in a daemon thread (VECTOR_BACKFILL_ON_STARTUP=false disables it)."""
    if not getattr(settings, "VECTOR_BACKFILL_ON_STARTUP", True):
        STARTUP_STATUS.update(state="disabled")
        return None
    if not vector_service.check_ready():
        STARTUP_STATUS.update(state="skipped", error="ChromaDB is unavailable")
        return None
    thread = threading.Thread(target=_run_startup_backfill, name="vector-backfill", daemon=True)
    thread.start()
    return thread

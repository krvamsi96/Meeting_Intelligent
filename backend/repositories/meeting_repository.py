"""Meeting Repository - abstracts database operations for meetings."""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, or_
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from backend.models.models import (
    AnalysisResult,
    ChatSession,
    Meeting,
    TranscriptSegment,
)



class MeetingRepository:
    """Repository for meeting-related database operations."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, meeting_id: Any) -> Optional[Meeting]:
        if meeting_id is None:
            return None
        if isinstance(meeting_id, Meeting):
            return meeting_id
        try:
            val = str(meeting_id).strip()
            # Direct integer PK check first if numeric
            if isinstance(meeting_id, int) or val.isdigit():
                meeting = self.db.query(Meeting).filter(Meeting.id == int(val)).first()
                if meeting:
                    return meeting

            # 1. Direct match on bharatvc_meeting_id, meeting_id, or display_id
            meeting = self.db.query(Meeting).filter(Meeting.bharatvc_meeting_id == val).first()
            if meeting:
                return meeting
            meeting = self.db.query(Meeting).filter(Meeting.meeting_id == val).first()
            if meeting:
                return meeting
            meeting = self.db.query(Meeting).filter(Meeting.display_id == val).first()
            if meeting:
                return meeting

            # 2. Try sanitized numeric match if input had legacy prefixes (BVC-, MTG-)
            from backend.utils.helpers import generate_numeric_meeting_id
            clean_val = generate_numeric_meeting_id(val)
            if clean_val and clean_val != val:
                meeting = self.db.query(Meeting).filter(Meeting.bharatvc_meeting_id == clean_val).first()
                if meeting:
                    return meeting
                meeting = self.db.query(Meeting).filter(Meeting.meeting_id == clean_val).first()
                if meeting:
                    return meeting
                meeting = self.db.query(Meeting).filter(Meeting.display_id == clean_val).first()
                if meeting:
                    return meeting
                if clean_val.isdigit():
                    return self.db.query(Meeting).filter(Meeting.id == int(clean_val)).first()

            return None
        except Exception as e:
            logger.warning(f"get_by_id query failed: {e}")
            try:
                self.db.rollback()
            except Exception:
                pass
            return None

    def get_by_public_id(self, public_id: Any) -> Optional[Meeting]:
        """Look a meeting up by its externally visible id ONLY.

        Unlike ``get_by_id`` this never falls back to the internal integer primary key. The
        public ids are numeric strings, so a caller-chosen id such as ``"5"`` would otherwise
        resolve to whichever unrelated meeting happens to own primary key 5 — harmless for a
        read, but destructive when used to decide "this upload is an update of that meeting".
        """
        if public_id is None:
            return None
        val = str(public_id).strip()
        if not val:
            return None
        try:
            return (
                self.db.query(Meeting)
                .filter(
                    or_(
                        Meeting.bharatvc_meeting_id == val,
                        Meeting.meeting_id == val,
                        Meeting.display_id == val,
                    )
                )
                .order_by(Meeting.id)
                .first()
            )
        except Exception as e:
            logger.warning(f"get_by_public_id query failed: {e}")
            try:
                self.db.rollback()
            except Exception:
                pass
            return None

    def list_all(self) -> List[Meeting]:
        return self.db.query(Meeting).order_by(desc(Meeting.upload_time)).all()

    def create(self, **kwargs) -> Meeting:
        from backend.utils.helpers import generate_numeric_meeting_id
        b_id_raw = kwargs.get("bharatvc_meeting_id") or kwargs.get("meeting_id") or kwargs.get("display_id")
        clean_id = generate_numeric_meeting_id(b_id_raw)

        kwargs["bharatvc_meeting_id"] = clean_id
        kwargs["meeting_id"] = clean_id
        kwargs["display_id"] = clean_id

        try:
            # Check if meeting with this clean_id already exists (prevent duplicates). Match on the
            # public id columns only — see get_by_public_id for why the primary key must not be used.
            existing = self.get_by_public_id(clean_id)
            if existing:
                # A re-upload under the same public id REPLACES the file (and, when processed, every result): its
                # upload date is the new upload's. ``upload_time`` was left at the row's first creation while
                # ``upload_start_time`` moved - the API showed 2026-08-27 next to 2026-10-13 for one upload.
                if kwargs.get("upload_time") is None:
                    from backend.models.models import _utcnow

                    kwargs["upload_time"] = kwargs.get("upload_start_time") or _utcnow()
                for k, v in kwargs.items():
                    if v is not None:
                        setattr(existing, k, v)
                self.db.commit()
                self.db.refresh(existing)
                return existing

            meeting = Meeting(**kwargs)
            self.db.add(meeting)
            self.db.commit()
            self.db.refresh(meeting)
            logger.info(f"Database meeting_id = {meeting.bharatvc_meeting_id}")
            return meeting
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create meeting in DB: {e}")
            raise

    def update(self, meeting: Meeting) -> Meeting:
        try:
            self.db.commit()
            self.db.refresh(meeting)
            return meeting
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to update meeting in DB: {e}")
            raise

    def delete(self, meeting_id: Any) -> bool:
        try:
            meeting = self.get_by_id(meeting_id)
            if not meeting:
                return False
            # Two child tables are not ORM relationships of Meeting and depend on a database-level
            # ON DELETE CASCADE. create_all() never adds that constraint to a table that already
            # exists, so on a database created by an older release deleting any completed meeting
            # (each has a consolidated final record) failed with an FK violation. Delete them
            # explicitly; on a fully-cascading schema these are simply no-ops.
            from backend.models.models import AIMeetingIntelFinal, RagEvaluation

            self.db.query(RagEvaluation).filter(RagEvaluation.meeting_id == meeting.id).delete(
                synchronize_session=False
            )
            self.db.query(AIMeetingIntelFinal).filter(AIMeetingIntelFinal.meeting_pk == meeting.id).delete(
                synchronize_session=False
            )
            self.db.delete(meeting)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to delete meeting from DB: {e}")
            raise

    def get_orphaned_processing(self) -> List[Meeting]:
        """Meetings whose background pipeline can no longer be running.

        Called once at startup, when no pipeline exists yet, so BOTH ``processing`` (killed
        mid-run) and ``uploaded`` (accepted, but the restart happened before the queued
        background task started) are orphans. Leaving ``uploaded`` rows alone kept them
        counted as "processing" forever.
        """
        return self.db.query(Meeting).filter(Meeting.status.in_(("processing", "uploaded"))).all()

    # Columns of a meeting that a similarity snapshot reads (see similarity_service._snapshot_from_parts). Selecting
    # just these instead of whole ``Meeting`` rows matters: the row has ~120 columns, several of them large JSON text.
    SNAPSHOT_MEETING_COLUMNS = (
        "id", "display_id", "bharatvc_meeting_id", "title", "filename", "file_type", "status", "is_approved",
        "upload_time", "meeting_date", "meeting_time", "duration", "project_name", "participants",
        "participant_count", "matched_mom", "keyword_frequency", "content_fingerprint",
        "file_size_bytes",
    )

    def get_snapshot_rows(self, meeting_ids: List[int]) -> Dict[str, Any]:
        """Everything a set of meetings' similarity snapshots need, in THREE queries whatever the number of meetings.

        ``{"meetings": {id: row}, "segments": {id: [(speaker, text), ...]}, "analyses": {id: row}}`` where a
        segment list is in transcript order. Replaces one ``get_by_id`` + ``get_segments`` + ``get_analysis`` per
        meeting (6 statements each, 3 of them re-reading the same meeting row), i.e. ~180 statements for the 30
        candidates of a similarity search.
        """
        ids = sorted({int(i) for i in meeting_ids})
        if not ids:
            return {"meetings": {}, "segments": {}, "analyses": {}}
        meeting_cols = [getattr(Meeting, name) for name in self.SNAPSHOT_MEETING_COLUMNS]
        meetings = {row.id: row for row in self.db.query(*meeting_cols).filter(Meeting.id.in_(ids)).all()}
        segments: Dict[int, List[tuple]] = {}
        for meeting_id, speaker, text in (
            self.db.query(TranscriptSegment.meeting_id, TranscriptSegment.speaker, TranscriptSegment.text)
            .filter(TranscriptSegment.meeting_id.in_(ids))
            .order_by(TranscriptSegment.meeting_id, TranscriptSegment.id)
            .all()
        ):
            segments.setdefault(meeting_id, []).append((speaker, text))
        analyses = {
            row.meeting_id: row
            for row in self.db.query(
                AnalysisResult.meeting_id, AnalysisResult.summary, AnalysisResult.mom,
                AnalysisResult.action_items, AnalysisResult.decisions, AnalysisResult.risks,
            ).filter(AnalysisResult.meeting_id.in_(ids)).all()
        }
        return {"meetings": meetings, "segments": segments, "analyses": analyses}

    def get_segments(self, meeting_id: Any) -> List[TranscriptSegment]:
        meeting = self.get_by_id(meeting_id)
        if not meeting:
            return []
        return (
            self.db.query(TranscriptSegment)
            .filter(TranscriptSegment.meeting_id == meeting.id)
            .order_by(TranscriptSegment.id)
            .all()
        )

    def get_segments_text(self, meeting_id: Any, with_speakers: bool = True) -> str:
        segments = self.get_segments(meeting_id)
        if not segments:
            return ""
        if with_speakers:
            return "\n".join([f"{seg.speaker}: {seg.text}" for seg in segments])
        return "\n".join([seg.text for seg in segments])

    def add_segments(self, meeting_id: Any, segments_data: List[Dict[str, Any]]):
        meeting = self.get_by_id(meeting_id)
        if not meeting:
            return
        try:
            for seg_data in segments_data:
                segment = TranscriptSegment(
                    meeting_id=meeting.id,
                    speaker=seg_data["speaker"],
                    start_time=seg_data["start_time"],
                    end_time=seg_data["end_time"],
                    text=seg_data["text"],
                )
                self.db.add(segment)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to add segments to meeting {meeting_id}: {e}")
            raise

    def get_analysis(self, meeting_id: Any) -> Optional[AnalysisResult]:
        meeting = self.get_by_id(meeting_id)
        if not meeting:
            return None
        return (
            self.db.query(AnalysisResult)
            .filter(AnalysisResult.meeting_id == meeting.id)
            .first()
        )

    def create_or_update_analysis(
        self, meeting_id: Any, analysis_data: Dict[str, str]
    ) -> AnalysisResult:
        meeting = self.get_by_id(meeting_id)
        if not meeting:
            raise ValueError(f"Meeting {meeting_id} not found")
        try:
            existing = self.get_analysis(meeting.id)
            if existing:
                for key, value in analysis_data.items():
                    setattr(existing, key, value)
            else:
                existing = AnalysisResult(meeting_id=meeting.id, **analysis_data)
                self.db.add(existing)
            self.db.commit()
            self.db.refresh(existing)
            return existing
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create/update analysis for meeting {meeting_id}: {e}")
            raise

    def get_chat_history(self, meeting_id: Any, session_id: Optional[str] = None) -> List[ChatSession]:
        meeting = self.get_by_id(meeting_id)
        if not meeting:
            return []
        query = self.db.query(ChatSession).filter(ChatSession.meeting_id == meeting.id)
        if session_id:
            query = query.filter(ChatSession.session_id == session_id)
        return query.order_by(ChatSession.timestamp).all()

    def add_chat_message(
        self,
        meeting_id: Any,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        sources: Optional[str] = None,
    ) -> Optional[ChatSession]:
        try:
            int_meeting_id = None
            if isinstance(meeting_id, Meeting):
                int_meeting_id = meeting_id.id
                session_id = session_id or meeting_id.meeting_id or str(meeting_id.id)
            elif isinstance(meeting_id, int):
                int_meeting_id = meeting_id
                session_id = session_id or str(meeting_id)
            else:
                meeting = self.get_by_id(meeting_id)
                if meeting:
                    int_meeting_id = meeting.id
                    session_id = session_id or meeting.meeting_id or str(meeting.id)
                elif str(meeting_id).isdigit():
                    int_meeting_id = int(str(meeting_id))
                    session_id = session_id or str(meeting_id)

            if int_meeting_id is None:
                logger.warning(f"Cannot add chat message: Meeting {meeting_id} not found")
                return None

            msg = ChatSession(
                session_id=session_id or str(int_meeting_id),
                meeting_id=int_meeting_id,
                role=role,
                content=content,
                sources=sources,
            )
            self.db.add(msg)
            self.db.commit()
            self.db.refresh(msg)
            return msg
        except Exception as e:
            try:
                self.db.rollback()
            except Exception:
                pass
            logger.error(f"Failed to add chat message: {e}")
            return None

    def clear_chat_history(self, meeting_id: Any, session_id: Optional[str] = None):
        meeting = self.get_by_id(meeting_id)
        if not meeting:
            return
        try:
            query = self.db.query(ChatSession).filter(ChatSession.meeting_id == meeting.id)
            if session_id:
                query = query.filter(ChatSession.session_id == session_id)
            query.delete()
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to clear chat history: {e}")
            raise

    def get_matched_meeting_analysis(self, matched_id: int) -> Optional[Dict[str, Any]]:
        matched = self.get_by_id(matched_id)
        if not matched:
            return None
        analysis = self.get_analysis(matched_id)
        segments = self.get_segments_text(matched_id)
        return {
            "meeting": matched,
            "analysis": analysis,
            "transcript": segments,
        }

    def get_all_meeting_ids(self) -> List[int]:
        return [m.id for m in self.db.query(Meeting.id).all()]

    def get_meetings_by_status(self, status: str) -> List[Meeting]:
        return self.db.query(Meeting).filter(Meeting.status == status).all()

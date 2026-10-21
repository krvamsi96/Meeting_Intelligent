"""Comparison metadata stored alongside a meeting's whole-transcript vector in ChromaDB."""


def build_meeting_vector_metadata(meeting, fingerprint: str = "") -> dict:
    """Assemble lightweight comparison metadata stored alongside the meeting vector."""
    return {
        "bharatvc_meeting_id": meeting.bharatvc_meeting_id or meeting.meeting_id or "",
        "display_id": meeting.display_id or "",
        "file_type": meeting.file_type or "media",
        "status": meeting.status or "",
        "meeting_date": meeting.meeting_date or "",
        "meeting_time": meeting.meeting_time or "",
        "duration": float(meeting.duration or 0.0),
        "participant_count": int(meeting.participant_count or 0),
        "participants": (meeting.participants or "")[:500],
        "keyword_frequency": (meeting.keyword_frequency or "")[:2000],
        "content_fingerprint": fingerprint or (meeting.content_fingerprint or ""),
    }

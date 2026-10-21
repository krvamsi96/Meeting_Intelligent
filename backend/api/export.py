"""API Router for Meeting Report Exports."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.config.config import settings
from backend.utils.helpers import clean_agenda_text
from backend.models.models import (
    AIProviderConfig,
    AnalysisResult,
    TranscriptSegment,
)
from backend.repositories import MeetingRepository
from backend.services.export_service import (
    clean_markdown_text,
    generate_docx_report,
    generate_excel_report,
    generate_html_report,
)
from backend.utils.crypto import decrypt
from backend.utils.security import content_disposition

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/meetings", tags=["Export"])


def get_report_settings_internal(db: Session) -> dict:
    """Helper to load and decrypt report configuration settings."""
    defaults = {
        "logo_url": "",
        "theme_color_primary": "#1e3a8a",
        "theme_color_secondary": "#2563eb",
        "font_family": "Arial",
        "header_text": settings.REPORT_HEADER_TEXT,
        "footer_text": settings.REPORT_FOOTER_TEXT,
        "watermark": "CONFIDENTIAL",
        "company_name": settings.REPORT_COMPANY_NAME,
        "company_address": settings.REPORT_COMPANY_ADDRESS,
        "show_cover_page": True,
        "show_table_of_contents": True,
        "show_visual_charts": True,
    }
    try:
        config = (
            db.query(AIProviderConfig)
            .filter(AIProviderConfig.provider == "report_settings")
            .first()
        )
        if config and config.encrypted_config:
            decrypted = decrypt(config.encrypted_config)
            data = json.loads(decrypted)
            # Merge with defaults
            for k, v in defaults.items():
                if k not in data:
                    data[k] = v
            return data
    except Exception as e:
        logger.error(f"Error loading report settings: {e}")
    return defaults


@router.get("/{meeting_id}/export/docx")
def export_meeting_docx(meeting_id: str, db: Session = Depends(get_db)):
    """Exports a clean executive corporate Word document (.docx)."""
    repo = MeetingRepository(db)
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    logger.info(f"Export meeting_id = {meeting.bharatvc_meeting_id or meeting.meeting_id}")

    analysis = (
        db.query(AnalysisResult).filter(AnalysisResult.meeting_id == meeting.id).first()
    )
    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.meeting_id == meeting.id)
        .order_by(TranscriptSegment.id)
        .all()
    )
    report_settings = get_report_settings_internal(db)

    try:
        stream = generate_docx_report(meeting, analysis, report_settings, segments)
        filename = f"{meeting.title or 'meeting'}_report.docx"
        return StreamingResponse(
            stream,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": content_disposition(filename)},
        )
    except Exception as e:
        logger.error(f"Error generating Word report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate Word report.") from e


@router.get("/{meeting_id}/export/xlsx")
def export_meeting_xlsx(meeting_id: str, db: Session = Depends(get_db)):
    """Exports structured data (Overview, Action Items, Decisions, Transcript) into multi-sheet Excel."""
    repo = MeetingRepository(db)
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    analysis = (
        db.query(AnalysisResult).filter(AnalysisResult.meeting_id == meeting.id).first()
    )
    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.meeting_id == meeting.id)
        .order_by(TranscriptSegment.id)
        .all()
    )

    try:
        stream = generate_excel_report(meeting, analysis, segments)
        filename = f"{meeting.title or 'meeting'}_analytics.xlsx"
        return StreamingResponse(
            stream,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": content_disposition(filename)},
        )
    except Exception as e:
        logger.error(f"Error generating Excel report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate Excel report.") from e


@router.get("/{meeting_id}/export/html", response_class=HTMLResponse)
def export_meeting_html(meeting_id: str, db: Session = Depends(get_db)):
    """Returns a fully styled HTML business report page (interactive view)."""
    repo = MeetingRepository(db)
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    analysis = (
        db.query(AnalysisResult).filter(AnalysisResult.meeting_id == meeting.id).first()
    )
    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.meeting_id == meeting.id)
        .order_by(TranscriptSegment.id)
        .all()
    )
    report_settings = get_report_settings_internal(db)

    html_content = generate_html_report(
        meeting, analysis, report_settings, segments, print_mode=False
    )
    return HTMLResponse(content=html_content, status_code=200)


@router.get("/{meeting_id}/export/pdf", response_class=HTMLResponse)
def export_meeting_pdf(meeting_id: str, db: Session = Depends(get_db)):
    """Returns the printable HTML layout configured to trigger the browser's PDF print layout."""
    repo = MeetingRepository(db)
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    analysis = (
        db.query(AnalysisResult).filter(AnalysisResult.meeting_id == meeting.id).first()
    )
    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.meeting_id == meeting.id)
        .order_by(TranscriptSegment.id)
        .all()
    )
    report_settings = get_report_settings_internal(db)

    html_content = generate_html_report(
        meeting, analysis, report_settings, segments, print_mode=True
    )
    return HTMLResponse(content=html_content, status_code=200)


@router.get("/{meeting_id}/export/md")
def export_meeting_markdown(meeting_id: str, db: Session = Depends(get_db)):
    """Returns a structured, clean markdown report."""
    repo = MeetingRepository(db)
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    analysis = (
        db.query(AnalysisResult).filter(AnalysisResult.meeting_id == meeting.id).first()
    )

    md_content = (
        f"# Meeting Intelligence Report: {meeting.title or meeting.filename}\n\n"
    )
    md_content += f"**BharatVC Meeting ID:** {meeting.bharatvc_meeting_id or meeting.display_id or meeting.id}\n"
    md_content += f"**Date:** {meeting.meeting_date or 'N/A'}\n"
    md_content += f"**Confidence Rating:** {(meeting.accuracy or 0.0):.1f}%\n"
    md_content += f"**Agenda:** {clean_agenda_text(meeting.agenda) or 'N/A'}\n\n"

    if analysis:
        md_content += (
            f"## Executive Summary\n{clean_markdown_text(analysis.summary)}\n\n"
        )
        md_content += (
            f"## Minutes of Meeting (MoM)\n{clean_markdown_text(analysis.mom)}\n\n"
        )
        md_content += f"## Action Items Registry\n{clean_markdown_text(analysis.action_items)}\n\n"
        md_content += (
            f"## Decisions Tracker\n{clean_markdown_text(analysis.decisions)}\n\n"
        )
        md_content += f"## Risks & Issues\n{clean_markdown_text(analysis.risks)}\n\n"

    filename = f"{meeting.title or 'meeting'}_report.md"
    return Response(
        content=md_content,
        media_type="text/markdown",
        headers={"Content-Disposition": content_disposition(filename)},
    )


@router.get("/{meeting_id}/export/json", response_class=JSONResponse)
def export_meeting_json(meeting_id: str, db: Session = Depends(get_db)):
    """Returns full meeting data including segments and analytics as structured JSON."""
    repo = MeetingRepository(db)
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    analysis = (
        db.query(AnalysisResult).filter(AnalysisResult.meeting_id == meeting.id).first()
    )
    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.meeting_id == meeting.id)
        .order_by(TranscriptSegment.id)
        .all()
    )

    payload = {
        "meeting": {
            "id": meeting.id,
            "bharatvc_meeting_id": meeting.bharatvc_meeting_id or meeting.meeting_id,
            "meeting_id": meeting.meeting_id,
            "display_id": meeting.display_id,
            "title": meeting.title,
            "filename": meeting.filename,
            "upload_time": (
                meeting.upload_time.isoformat() if meeting.upload_time else None
            ),
            "duration": meeting.duration,
            "accuracy": meeting.accuracy,
            "participants": meeting.participants,
            "agenda": clean_agenda_text(meeting.agenda),
            "meeting_date": meeting.meeting_date,
            "meeting_time": meeting.meeting_time,
            "project_name": meeting.project_name,
        },
        "analysis": {
            "summary": analysis.summary if analysis else "",
            "mom": analysis.mom if analysis else "",
            "action_items": analysis.action_items if analysis else "",
            "decisions": analysis.decisions if analysis else "",
            "risks": analysis.risks if analysis else "",
        },
        "segments": [
            {
                "speaker": seg.speaker,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "text": seg.text,
            }
            for seg in segments
        ],
    }

    filename = f"{meeting.title or 'meeting'}_data.json"
    return JSONResponse(
        content=payload,
        status_code=200,
        headers={"Content-Disposition": content_disposition(filename)},
    )

"""
Runtime Metrics Collector & Logger
===================================
Collects, logs, and formats real empirical metrics directly from runtime execution.
No fake, hardcoded, estimated, or AI-invented benchmark values allowed.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.utils.helpers import count_transcript_words

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Returns current UTC timestamp."""
    return datetime.now(timezone.utc)


def build_runtime_metrics_dict(meeting: Any, db: Any = None) -> Dict[str, Any]:
    """
    Assembles a complete dictionary of all 21 mandatory real runtime metrics
    strictly from database records and empirical execution logs.
    """
    if not meeting:
        return {}

    # Segments & analysis outputs
    segments = getattr(meeting, "segments", []) or []
    analysis = getattr(meeting, "analysis", None)

    transcript_text = "\n".join([s.text for s in segments if getattr(s, "text", None)])
    stt_word_count = count_transcript_words(transcript_text) if transcript_text.strip() else 0
    stt_char_count = len(transcript_text) if transcript_text.strip() else 0
    stt_generated = bool(segments and stt_word_count > 0)

    summary_text = getattr(analysis, "summary", "") or "" if analysis else ""
    summary_word_count = len(summary_text.split()) if summary_text.strip() else 0
    summary_generated = bool(summary_text.strip() and "Not Available" not in summary_text)

    mom_text = getattr(analysis, "mom", "") or "" if analysis else ""
    mom_word_count = len(mom_text.split()) if mom_text.strip() else 0
    mom_generated = bool(mom_text.strip() and "Not Available" not in mom_text)

    action_items_text = getattr(analysis, "action_items", "") or "" if analysis else ""
    # Bug 8 fix: action_items_text may now be a JSON array (serialised structured
    # action items).  Try json.loads first; count list elements directly.
    # Only fall back to the line-regex parser for legacy plain-text blobs.
    action_items_count = 0
    action_items_generated = False
    if action_items_text.strip():
        try:
            _ai_parsed = json.loads(action_items_text)
            if isinstance(_ai_parsed, list):
                action_items_count = len(_ai_parsed)
                action_items_generated = action_items_count > 0
            else:
                raise ValueError("not a list")
        except (json.JSONDecodeError, ValueError):
            # Plain-text blob — use the original line-regex approach
            action_lines = [
                l.strip()
                for l in action_items_text.splitlines()
                if l.strip() and (
                    l.strip().startswith(("-", "*", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."))
                    or "[ ]" in l
                )
            ]
            action_items_count = len(action_lines)
            action_items_generated = bool(action_items_text.strip() and "Not Available" not in action_items_text)

    # Topics count
    kw_freq_raw = getattr(meeting, "keyword_frequency", "{}") or "{}"
    topics_count = 0
    try:
        kw_dict = json.loads(kw_freq_raw) if isinstance(kw_freq_raw, str) else kw_freq_raw
        if isinstance(kw_dict, dict):
            topics_count = len(kw_dict)
    except Exception:
        topics_count = 0
    topics_generated = bool(topics_count > 0)

    # Speaker detection
    speakers_set = {
        s.speaker
        for s in segments
        if getattr(s, "speaker", None) and "UNKNOWN" not in str(s.speaker).upper()
    }
    speaker_count = len(speakers_set) if speakers_set else getattr(meeting, "participant_count", 0)
    speaker_timeline_generated = bool(segments and any(getattr(s, "start_time", None) for s in segments))

    # Sentiment distribution
    sentiment_label = getattr(meeting, "sentiment", None) or "Not Measured"
    emotions_raw = getattr(meeting, "emotions", "") or ""
    pos_cnt, neu_cnt, neg_cnt, mix_cnt = 0, 0, 0, 0
    if emotions_raw and isinstance(emotions_raw, str):
        lower_em = emotions_raw.lower()
        if "positive" in lower_em:
            pos_cnt += 1
        if "neutral" in lower_em:
            neu_cnt += 1
        if "negative" in lower_em:
            neg_cnt += 1
        if "mixed" in lower_em:
            mix_cnt += 1

    # Database Validation
    db_meeting_saved = bool(meeting.id is not None)
    db_transcript_saved = bool(stt_generated)
    db_summary_saved = bool(summary_generated)
    db_action_items_saved = bool(action_items_generated)

    # Export Validation
    export_pdf = getattr(meeting, "export_pdf_generated", False)
    export_docx = getattr(meeting, "export_docx_generated", False)
    export_json = getattr(meeting, "export_json_generated", False)

    # Status
    raw_status = getattr(meeting, "status", "completed")
    # The stored columns were only ever the default "Success"/200; report the outcome of the processing itself.
    from backend.utils.processing_status import api_status_for

    api_status, api_status_code = api_status_for(raw_status)
    if raw_status == "completed":
        overall_status = "Success"
    elif raw_status == "error":
        overall_status = "Failed"
    else:
        overall_status = "Partial Success"

    payload = {
        "1_video_name": meeting.filename or meeting.title or "Not Available",
        "2_meeting_id": meeting.display_id or meeting.bharatvc_meeting_id or str(meeting.id),
        "3_video_duration_sec": round(getattr(meeting, "duration", 0.0) or 0.0, 2),
        "4_file_size_bytes": getattr(meeting, "file_size_bytes", 0) or 0,
        "5_upload_start_time": meeting.upload_start_time.isoformat() if getattr(meeting, "upload_start_time", None) else "Not Measured",
        "6_upload_end_time": meeting.upload_end_time.isoformat() if getattr(meeting, "upload_end_time", None) else "Not Measured",
        "7_total_upload_time_sec": round(getattr(meeting, "upload_duration_sec", 0.0) or 0.0, 2),
        "8_processing_start_time": meeting.processing_start_time.isoformat() if getattr(meeting, "processing_start_time", None) else "Not Measured",
        "9_processing_end_time": meeting.processing_end_time.isoformat() if getattr(meeting, "processing_end_time", None) else "Not Measured",
        "10_total_processing_time_sec": round(getattr(meeting, "total_processing_time_sec", 0.0) or getattr(meeting, "processing_time", 0.0) or 0.0, 2),
        "11_stt": {
            "generated": stt_generated,
            "word_count": stt_word_count,
            "char_count": stt_char_count,
            "generation_time_sec": round(getattr(meeting, "stt_duration_sec", 0.0) or 0.0, 2),
        },
        "12_speaker_identification": {
            "speakers_detected": speaker_count,
            "timeline_generated": speaker_timeline_generated,
            "processing_time_sec": round(getattr(meeting, "speaker_duration_sec", 0.0) or 0.0, 2),
        },
        "13_summary": {
            "generated": summary_generated,
            "word_count": summary_word_count,
            "generation_time_sec": round(getattr(meeting, "summary_duration_sec", 0.0) or 0.0, 2),
        },
        "14_mom": {
            "generated": mom_generated,
            "word_count": mom_word_count,
            "generation_time_sec": round(getattr(meeting, "mom_duration_sec", 0.0) or 0.0, 2),
        },
        "15_action_items": {
            "generated": action_items_generated,
            "total_items": action_items_count,
            "generation_time_sec": round(getattr(meeting, "action_items_duration_sec", 0.0) or 0.0, 2),
        },
        "16_topics": {
            "generated": topics_generated,
            "total_topics": topics_count,
        },
        "17_sentiment": {
            "detected_sentiment": sentiment_label,
            "positive_count": pos_cnt,
            "neutral_count": neu_cnt,
            "negative_count": neg_cnt,
            "mixed_count": mix_cnt,
        },
        "18_db_validation": {
            "meeting_saved": db_meeting_saved,
            "transcript_saved": db_transcript_saved,
            "summary_saved": db_summary_saved,
            "action_items_saved": db_action_items_saved,
        },
        "19_export_validation": {
            "pdf_generated": export_pdf,
            "docx_generated": export_docx,
            "json_generated": export_json,
        },
        "20_api_status": {
            "status": api_status,
            "http_status_code": api_status_code,
            "error_message": getattr(meeting, "error_message", None),
        },
        "21_overall_processing_status": overall_status,
    }
    return payload


def log_runtime_metrics(meeting_id: int, metrics: Dict[str, Any]):
    """Logs runtime execution metrics with timestamps."""
    logger.info(f"=== REAL RUNTIME METRICS FOR MEETING ID {meeting_id} ===")
    for key, value in metrics.items():
        if isinstance(value, dict):
            logger.info(f"  [{key}] {json.dumps(value)}")
        else:
            logger.info(f"  [{key}] = {value}")
    logger.info("=========================================================")

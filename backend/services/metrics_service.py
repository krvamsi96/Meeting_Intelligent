"""Stateful meeting metrics recalculation service."""

import json
import logging
import re

from sqlalchemy.orm import Session

from backend.config.config import DEFAULT_PROJECT_NAME, settings
from backend.models.models import Meeting, TranscriptSegment
from backend.services.evaluation_service import _tokenize
from backend.services.llm_workflow import process_meeting_report
from backend.utils.helpers import (
    check_outputs_generated,
    format_seconds_to_duration,
)

logger = logging.getLogger(__name__)

# Shared regex for action item line detection (same pattern as helpers._LIST_PREFIX_RE)
_ACTION_ITEM_RE = re.compile(r'^(?:[-*•]|\d+\.)\s|\[\s?[xX ]?\s?\]')


def recalculate_meeting_metrics(
    meeting: Meeting, db: Session, skip_regeneration: bool = True
):
    """
    Recalculates all quality metrics from actual generated outputs.
    Uses the production-grade weighted formula — never just transcription_accuracy.
    """
    if not meeting.analysis:
        return

    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.meeting_id == meeting.id)
        .order_by(TranscriptSegment.id)
        .all()
    )
    full_transcript_text = "\n".join([f"{s.speaker}: {s.text}" for s in segments])
    if not full_transcript_text.strip():
        full_transcript_text = ""

    summary_text = meeting.analysis.summary or ""
    mom_text = meeting.analysis.mom or ""
    action_items_text = meeting.analysis.action_items or ""

    is_sum_ok, is_mom_ok = check_outputs_generated(
        summary_text, mom_text, meeting.filename
    )

    # If outputs are placeholder/missing, try to regenerate (only if skip_regeneration is False)
    if (not is_sum_ok or not is_mom_ok) and not skip_regeneration:
        try:
            logger.info(
                f"Auto-regenerating placeholder reports for meeting {meeting.id}..."
            )
            if full_transcript_text.strip():
                m_date = meeting.meeting_date or "Unknown Date"
                m_time = meeting.meeting_time or "Unknown Time"
                m_dur = format_seconds_to_duration(meeting.duration)
                # Pass the FULL transcript, exactly like every other pipeline
                # entry point: process_meeting_report() fits it to the context
                # budget per section and condenses a too-long meeting for the
                # summary in parts instead of dropping its middle.
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
                summary_text = meeting.analysis.summary or ""
                mom_text = meeting.analysis.mom or ""
                action_items_text = meeting.analysis.action_items or ""
                is_sum_ok, is_mom_ok = check_outputs_generated(
                    summary_text, mom_text, meeting.filename
                )
        except Exception as e:
            logger.error(f"Failed to auto-regenerate placeholder reports: {str(e)}")

    if not is_sum_ok or not is_mom_ok:
        meeting.accuracy = None
        meeting.confidence_score = None
        meeting.is_approved = False
        return

    meeting.confidence_score = None

    # Calculate real empirical execution counters
    trans_word_cnt = len(_tokenize(full_transcript_text))
    sum_word_cnt = len(_tokenize(summary_text))
    mom_word_cnt = len(_tokenize(mom_text))
    spk_cnt = len({s.speaker for s in segments if s.speaker and "UNKNOWN" not in s.speaker.upper()})
    action_cnt = sum(
        1 for line in action_items_text.splitlines()
        if line.strip() and _ACTION_ITEM_RE.search(line.strip())
    )

    # Store real empirical execution metrics payload
    metrics_payload = {
        "grounding_confidence_rouge": meeting.confidence_score,
        "transcript_word_count": trans_word_cnt,
        "summary_word_count": sum_word_cnt,
        "mom_word_count": mom_word_cnt,
        "speaker_count": spk_cnt,
        "action_items_count": action_cnt,
        "is_approved": True,
        "execution_status": "Success",
    }
    meeting.metrics_json = json.dumps(metrics_payload)
    meeting.is_approved = True

    # If ground truth exists, run true evaluation to set meeting.accuracy
    if meeting.has_ground_truth:
        try:
            from backend.services.true_evaluation_service import run_meeting_evaluation
            run_meeting_evaluation(meeting, db)
        except Exception as e:
            logger.error(f"Auto-evaluation failed on metric recalculation: {e}")
    else:
        # Never invent accuracy from proxy metrics
        meeting.accuracy = None

    logger.info(f"Recalculated meeting {meeting.id} empirical counters: trans_words={trans_word_cnt}, sum_words={sum_word_cnt}, speakers={spk_cnt}")


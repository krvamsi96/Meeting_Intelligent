"""Audit and re-generate stored meeting summaries with the current summary pipeline.

Meetings processed before the summary fix can carry an over-long summary, speaker
labels (Speaker_1 ...), timestamps, markdown, or text that never came from their own
transcript. Reprocessing a meeting already overwrites its summary; this script fixes
the EXISTING rows without reprocessing audio/diarization/MoM/agenda.

It regenerates ONLY ``analysis_results.summary`` — from that meeting's own stored
transcript segments — and then rebuilds its ``AI_Meeting_Inteligent_final`` record so
``json_data`` and the GET API serve the corrected text. Transcripts, speaker labels,
MoM, agenda, decisions and action items are never touched.

The default is a read-only AUDIT: no LLM calls, no writes. Nothing changes without --apply.

Usage:
    python -m backend.scripts.resummarize_meetings                       # audit every completed meeting
    python -m backend.scripts.resummarize_meetings --meeting-id 245      # audit one (repeatable)
    python -m backend.scripts.resummarize_meetings --apply               # regenerate every DIRTY summary
    python -m backend.scripts.resummarize_meetings --apply --all         # regenerate every summary
    python -m backend.scripts.resummarize_meetings --apply --meeting-id 245 --meeting-id 268
"""

import argparse
import logging
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("resummarize_meetings")

_MARKDOWN_RE = re.compile(r"(?m)^\s*(?:#{1,6}\s|[-*•]\s|\d{1,2}[.)]\s)|\*\*")


def _transcript_text(db, meeting_pk: int) -> str:
    from backend.models.models import TranscriptSegment

    segs = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.meeting_id == meeting_pk)
        .order_by(TranscriptSegment.id.asc())
        .all()
    )
    return "\n".join(f"{s.speaker}: {s.text}" for s in segs if (s.text or "").strip())


def audit_summary(summary: str, transcript: str) -> list:
    """Problems with a stored summary, as violation codes (empty list = clean)."""
    from backend.services.summary_service import (
        check_summary, normalize_summary_text, plan_summary_length, prepare_summary_source,
    )

    src = prepare_summary_source(transcript)
    plan = plan_summary_length(src.word_count)
    summary = summary or ""
    codes = [
        v.code
        for v in check_summary(
            summary, plan, source_text=src.text, labels=src.labels, id_labels=src.id_labels, raw=summary
        )
    ]
    if _MARKDOWN_RE.search(summary) or normalize_summary_text(summary)[0] != summary.strip():
        codes.append("formatting")
    return codes


def _select_meetings(db, ids: list):
    from backend.models.models import Meeting
    from backend.repositories import MeetingRepository

    if ids:
        repo = MeetingRepository(db)
        found = []
        for raw in ids:
            m = repo.get_by_id(raw)
            if m is None:
                logger.warning("meeting %s not found — skipped", raw)
            else:
                found.append(m)
        return found
    return db.query(Meeting).filter(Meeting.status == "completed").order_by(Meeting.id.asc()).all()


def main():
    parser = argparse.ArgumentParser(description="Audit / regenerate stored meeting summaries.")
    parser.add_argument("--meeting-id", action="append", default=[],
                        help="Meeting id (any form the API accepts). Repeatable. Default: all completed meetings.")
    parser.add_argument("--apply", action="store_true",
                        help="Regenerate and store summaries. Without it the script only audits.")
    parser.add_argument("--all", action="store_true",
                        help="With --apply: regenerate every selected summary, not just the dirty ones.")
    args = parser.parse_args()

    from backend.database.connection import SessionLocal
    from backend.services.summary_service import finalize_summary, generate_meeting_summary

    stats = {"seen": 0, "clean": 0, "dirty": 0, "skipped": 0, "regenerated": 0, "failed": 0}
    with SessionLocal() as db:
        for meeting in _select_meetings(db, args.meeting_id):
            stats["seen"] += 1
            analysis = meeting.analysis
            transcript = _transcript_text(db, meeting.id)
            if analysis is None or not transcript.strip():
                stats["skipped"] += 1
                logger.info("meeting %-5s SKIP  no analysis row or no transcript", meeting.id)
                continue

            codes = audit_summary(analysis.summary or "", transcript)
            stats["dirty" if codes else "clean"] += 1
            logger.info(
                "meeting %-5s %-6s summary=%4d words  transcript=%6d words  problems=%s",
                meeting.id, "DIRTY" if codes else "clean",
                len((analysis.summary or "").split()), len(transcript.split()), codes or "-",
            )
            if not args.apply or not (codes or args.all):
                continue

            try:
                result = generate_meeting_summary(transcript, meeting_ref=str(meeting.id))
                new_summary = finalize_summary(result.text)
                if not new_summary.strip():
                    raise RuntimeError("no summary could be generated")
                analysis.summary = new_summary
                meeting.summary_word_count = len(new_summary.split())
                meeting.summary_generated = True
                try:
                    from backend.evaluation.metrics import rouge_scores
                    rouge = rouge_scores(new_summary, transcript).get("rougeL")
                    if rouge is not None:
                        meeting.confidence_score = round(rouge, 1)
                except Exception as e:  # metric refresh is best-effort
                    logger.warning("meeting %s: ROUGE refresh skipped (%s)", meeting.id, e)
                db.commit()

                from backend.services.final_aggregation_service import build_final_record
                build_final_record(meeting.id, db)
                db.commit()
                stats["regenerated"] += 1
                logger.info(
                    "meeting %-5s FIXED  method=%s  %d words  (stored in analysis_results + json_data)",
                    meeting.id, result.method, len(new_summary.split()),
                )
            except Exception as e:
                db.rollback()
                stats["failed"] += 1
                logger.error("meeting %s: could not regenerate summary: %s", meeting.id, e)

    logger.info("=" * 60)
    logger.info("%s", stats)
    if not args.apply:
        logger.info("AUDIT ONLY — nothing was changed. Re-run with --apply to regenerate the dirty summaries.")
    if stats["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

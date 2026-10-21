"""API endpoints for Ground Truth management and True Evaluation Engine."""

import json
import logging
from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import io
import csv

from backend.api.deps import get_db
from backend.evaluation import runtime_quality as rq
from backend.models.models import ActionItem, AgendaKeyDecision, Meeting, GroundTruth, EvaluationResult, EvaluationHistory
from backend.schemas import schemas
from backend.services.true_evaluation_service import run_meeting_evaluation
from backend.services.evaluation_service import EvaluationService
from backend.services.live_metrics_service import build_live_evaluation
from backend.evaluation.engine import ENGINE_NAME, ENGINE_VERSION
from backend.utils.db_safe import load_ground_truth


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["True Evaluation Engine"])


@router.post("/{meeting_id}/ground-truth", response_model=schemas.GroundTruthResponse)
def upload_ground_truth(
    meeting_id: int,
    payload: schemas.GroundTruthUpload,
    db: Session = Depends(get_db)
):
    """
    Upload or replace the Ground Truth reference dataset for a meeting.
    """
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    # Serialize structured fields to strings
    ref_speaker_labels_str = json.dumps(payload.ref_speaker_labels) if payload.ref_speaker_labels is not None else None
    ref_speaker_timeline_str = json.dumps(payload.ref_speaker_timeline) if payload.ref_speaker_timeline is not None else None
    ref_action_items_str = json.dumps(payload.ref_action_items) if payload.ref_action_items is not None else None
    ref_topics_str = json.dumps(payload.ref_topics) if payload.ref_topics is not None else None
    ref_similarity_labels_str = json.dumps(payload.ref_similarity_labels) if payload.ref_similarity_labels is not None else None

    gt = db.query(GroundTruth).filter(GroundTruth.meeting_id == meeting_id).first()
    is_new = False
    if not gt:
        gt = GroundTruth(meeting_id=meeting_id)
        db.add(gt)
        is_new = True
    else:
        gt.version += 1

    gt.ref_transcript = payload.ref_transcript
    gt.ref_speaker_labels = ref_speaker_labels_str
    gt.ref_speaker_timeline = ref_speaker_timeline_str
    gt.ref_summary = payload.ref_summary
    gt.ref_mom = payload.ref_mom
    gt.ref_action_items = ref_action_items_str
    gt.ref_topics = ref_topics_str
    gt.ref_sentiment = payload.ref_sentiment
    gt.ref_similarity_labels = ref_similarity_labels_str
    gt.source = payload.source or "manual"
    gt.notes = payload.notes
    gt.uploaded_by = "system"

    db.flush()

    # Log to History
    history = EvaluationHistory(
        meeting_id=meeting_id,
        action="gt_updated" if not is_new else "gt_uploaded",
        performed_by="system",
        details_json=json.dumps({
            "version": gt.version,
            "has_transcript": bool(payload.ref_transcript),
            "has_summary": bool(payload.ref_summary),
            "has_mom": bool(payload.ref_mom)
        })
    )
    db.add(history)
    db.commit()
    db.refresh(gt)
    return gt


@router.get("/{meeting_id}/ground-truth", response_model=schemas.GroundTruthResponse)
def get_ground_truth(meeting_id: int, db: Session = Depends(get_db)):
    """
    Retrieve the Ground Truth dataset for a meeting.
    """
    gt = db.query(GroundTruth).filter(GroundTruth.meeting_id == meeting_id).first()
    if not gt:
        raise HTTPException(status_code=404, detail="Ground Truth not found for this meeting")
    return gt


@router.post("/{meeting_id}/evaluate", response_model=schemas.EvaluationResultResponse)
def evaluate_meeting(meeting_id: int, db: Session = Depends(get_db)):
    """
    Evaluate the AI outputs of a meeting.
    If Ground Truth reference data exists, reference-grounded metrics are computed.
    If no reference data exists, the engine still runs and computes all
    self-contained metrics (e.g. Topic Coherence) while marking reference-grounded
    tasks as not_measured.
    """
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    try:
        eval_res = run_meeting_evaluation(meeting, db, performed_by="api")
        return eval_res
    except Exception as e:
        logger.error(f"Evaluation error for meeting {meeting_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Evaluation failed. See server logs for details.") from e


@router.post("/{meeting_id}/re-evaluate", response_model=schemas.EvaluationResultResponse)
def re_evaluate_meeting(meeting_id: int, db: Session = Depends(get_db)):
    """
    Force re-evaluation of meeting results (useful if transcripts/summaries were edited).
    """
    return evaluate_meeting(meeting_id=meeting_id, db=db)


@router.get("/{meeting_id}/results", response_model=schemas.EvaluationResultResponse)
def get_evaluation_results(meeting_id: int, db: Session = Depends(get_db)):
    """
    Retrieve the latest evaluation results for a meeting.
    """
    res = db.query(EvaluationResult).filter(EvaluationResult.meeting_id == meeting_id).first()
    if not res:
        raise HTTPException(status_code=404, detail="Evaluation results not found. Trigger evaluation first.")
    return res


@router.get("/{meeting_id}/history", response_model=List[schemas.EvaluationHistoryEntry])
def get_evaluation_history(meeting_id: int, db: Session = Depends(get_db)):
    """
    Retrieve the evaluation and Ground Truth change history log for a meeting.
    """
    history = db.query(EvaluationHistory).filter(EvaluationHistory.meeting_id == meeting_id).order_by(EvaluationHistory.timestamp.desc()).all()
    return history


@router.get("/benchmark")
def get_benchmark_report(db: Session = Depends(get_db)):
    """
    Generates a scientific benchmark dataset report aggregating all evaluated meetings.
    """
    results = db.query(EvaluationResult).all()
    if not results:
        return {
            "total_evaluated_meetings": 0,
            "average_overall_accuracy": 0.0,
            "average_stt_accuracy": 0.0,
            "average_diarization_accuracy": 0.0,
            "average_speaker_accuracy": 0.0,
            "average_summary_accuracy": 0.0,
            "average_mom_accuracy": 0.0,
            "average_action_accuracy": 0.0,
            "average_topic_accuracy": 0.0,
            "average_sentiment_accuracy": 0.0,
            "meetings": []
        }

    meeting_details = []
    sums = {
        "overall": 0.0, "stt": 0.0, "diar": 0.0, "spk": 0.0, "sum": 0.0, "mom": 0.0, "act": 0.0, "top": 0.0, "sent": 0.0
    }
    counts = {
        "overall": 0, "stt": 0, "diar": 0, "spk": 0, "sum": 0, "mom": 0, "act": 0, "top": 0, "sent": 0
    }

    meetings_by_id = {
        m.id: m
        for m in db.query(Meeting).filter(Meeting.id.in_([r.meeting_id for r in results])).all()
    }
    for r in results:
        meeting = meetings_by_id.get(r.meeting_id)
        code = (meeting.display_id or meeting.bharatvc_meeting_id or str(meeting.id)) if meeting else str(r.meeting_id)
        filename = meeting.filename if meeting else "Unknown"

        details = {
            "meeting_id": r.meeting_id,
            "meeting_code": code,
            "filename": filename,
            "overall_accuracy": r.overall_accuracy,
            "stt_accuracy": r.stt_accuracy if r.stt_evaluated else None,
            "diarization_accuracy": r.diarization_accuracy if r.diarization_evaluated else None,
            "speaker_accuracy": r.speaker_accuracy if r.speaker_evaluated else None,
            "summary_accuracy": r.summary_accuracy if r.summary_evaluated else None,
            "mom_accuracy": r.mom_accuracy if r.mom_evaluated else None,
            "action_accuracy": r.action_accuracy if r.action_evaluated else None,
            "topic_accuracy": r.topic_accuracy if r.topic_evaluated else None,
            "sentiment_accuracy": r.sentiment_accuracy if r.sentiment_evaluated else None,
        }
        meeting_details.append(details)

        if r.overall_accuracy is not None:
            sums["overall"] += r.overall_accuracy
            counts["overall"] += 1
        if r.stt_evaluated and r.stt_accuracy is not None:
            sums["stt"] += r.stt_accuracy
            counts["stt"] += 1
        if r.diarization_evaluated and r.diarization_accuracy is not None:
            sums["diar"] += r.diarization_accuracy
            counts["diar"] += 1
        if r.speaker_evaluated and r.speaker_accuracy is not None:
            sums["spk"] += r.speaker_accuracy
            counts["spk"] += 1
        if r.summary_evaluated and r.summary_accuracy is not None:
            sums["sum"] += r.summary_accuracy
            counts["sum"] += 1
        if r.mom_evaluated and r.mom_accuracy is not None:
            sums["mom"] += r.mom_accuracy
            counts["mom"] += 1
        if r.action_evaluated and r.action_accuracy is not None:
            sums["act"] += r.action_accuracy
            counts["act"] += 1
        if r.topic_evaluated and r.topic_accuracy is not None:
            sums["top"] += r.topic_accuracy
            counts["top"] += 1
        if r.sentiment_evaluated and r.sentiment_accuracy is not None:
            sums["sent"] += r.sentiment_accuracy
            counts["sent"] += 1

    return {
        "total_evaluated_meetings": len(results),
        # All averages return None when no data exists - NEVER 0.0 fake defaults
        "average_overall_accuracy": round(sums["overall"] / counts["overall"], 2) if counts["overall"] > 0 else None,
        "average_stt_accuracy": round(sums["stt"] / counts["stt"], 2) if counts["stt"] > 0 else None,
        "average_diarization_accuracy": round(sums["diar"] / counts["diar"], 2) if counts["diar"] > 0 else None,
        "average_speaker_accuracy": round(sums["spk"] / counts["spk"], 2) if counts["spk"] > 0 else None,
        "average_summary_accuracy": round(sums["sum"] / counts["sum"], 2) if counts["sum"] > 0 else None,
        "average_mom_accuracy": round(sums["mom"] / counts["mom"], 2) if counts["mom"] > 0 else None,
        "average_action_accuracy": round(sums["act"] / counts["act"], 2) if counts["act"] > 0 else None,
        "average_topic_accuracy": round(sums["top"] / counts["top"], 2) if counts["top"] > 0 else None,
        "average_sentiment_accuracy": round(sums["sent"] / counts["sent"], 2) if counts["sent"] > 0 else None,
        "meetings": meeting_details
    }


def _csv_safe(value):
    """Neutralise spreadsheet formula injection: a cell that starts with = + - @ (or a tab / CR)
    is executed by Excel/LibreOffice. Filenames come from uploads, i.e. from the attacker."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


@router.get("/benchmark/export")
def export_benchmark_report(db: Session = Depends(get_db)):
    """
    Exports aggregate benchmark report as CSV.
    """
    rep = get_benchmark_report(db)
    meetings = rep["meetings"]

    output = io.StringIO()
    writer = csv.writer(output)

    # Headers
    writer.writerow([
        "Meeting Code", "Filename", "Overall Accuracy", "STT Accuracy",
        "Diarization Accuracy", "Speaker Identification Accuracy",
        "Summary Accuracy", "MOM Accuracy", "Action Item Accuracy",
        "Topic Accuracy", "Sentiment Accuracy"
    ])

    for m in meetings:
        writer.writerow([_csv_safe(v) for v in (
            m["meeting_code"], m["filename"], m["overall_accuracy"],
            m.get("stt_accuracy", "N/A"), m.get("diarization_accuracy", "N/A"),
            m.get("speaker_accuracy", "N/A"), m.get("summary_accuracy", "N/A"),
            m.get("mom_accuracy", "N/A"), m.get("action_accuracy", "N/A"),
            m.get("topic_accuracy", "N/A"), m.get("sentiment_accuracy", "N/A")
        )])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=evaluation_benchmark_report.csv"}
    )


@router.get("/{meeting_id}", response_model=schemas.EvaluationReportResponse)
async def get_meeting_evaluation(meeting_id: int, db: Session = Depends(get_db)):
    """
    Run (or re-run) the Universal Evaluation Engine and return the full report.
    Tasks lacking reference data are reported with status='not_measured'.
    """
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    report = await EvaluationService.evaluate_meeting(meeting, db)
    return report


@router.get("/{meeting_id}/runtime-metrics", response_model=schemas.RuntimeMetricsResponse)
async def get_runtime_metrics(meeting_id: int, db: Session = Depends(get_db)):
    """Get Module 1 runtime metrics calculated on-the-fly from actual execution data."""
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    metrics = await EvaluationService.calculate_runtime_metrics(meeting, db)
    return metrics


@router.get("/{meeting_id}/benchmark", response_model=schemas.BenchmarkEvaluationResponse)
def get_benchmark_metrics(meeting_id: int, db: Session = Depends(get_db)):
    """Get Module 2 benchmark accuracy metrics, or status=unavailable if no reference exists."""
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    if not meeting.has_ground_truth:
        return schemas.BenchmarkEvaluationResponse(
            status="unavailable",
            reason="no_reference",
            meeting_id=meeting_id,
            reference_available=False
        )

    eval_res = db.query(EvaluationResult).filter(EvaluationResult.meeting_id == meeting_id).first()
    if not eval_res:
        try:
            from backend.services.true_evaluation_service import run_meeting_evaluation
            run_meeting_evaluation(meeting, db, performed_by="api_auto")
            eval_res = db.query(EvaluationResult).filter(EvaluationResult.meeting_id == meeting_id).first()
        except Exception as e:
            logger.error(f"Auto-evaluation failed in benchmark GET: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate benchmark.") from e

    gt_row = load_ground_truth(db, meeting.id)
    return schemas.BenchmarkEvaluationResponse(
        status="available",
        reason=None,
        benchmark_id=None,
        meeting_id=meeting_id,
        reference_version=str(gt_row.version) if gt_row else None,
        evaluation_version=eval_res.evaluation_version or ENGINE_VERSION,
        reference_available=True,
        stt_accuracy=eval_res.stt_accuracy,
        wer=eval_res.wer,
        cer=eval_res.cer,
        speaker_accuracy=eval_res.speaker_accuracy,
        der=eval_res.der,
        jer=eval_res.jer,
        summary_accuracy=eval_res.summary_accuracy,
        rouge1=eval_res.rouge1,
        rouge2=eval_res.rouge2,
        rougel=eval_res.rougeL,
        bertscore=eval_res.bertscore_f1,
        mom_accuracy=eval_res.mom_accuracy,
        action_precision=eval_res.action_precision,
        action_recall=eval_res.action_recall,
        action_f1=eval_res.action_f1,
        evaluation_time=eval_res.evaluation_timestamp,
        evaluation_status=eval_res.evaluation_status
    )


@router.get("/{meeting_id}/benchmark/history", response_model=List[schemas.BenchmarkHistoryResponse])
def get_benchmark_history(meeting_id: int, db: Session = Depends(get_db)):
    """Get the evaluation run history logs for a meeting."""
    history = db.query(EvaluationHistory).filter(
        EvaluationHistory.meeting_id == meeting_id,
        EvaluationHistory.action.in_(["evaluated", "re_evaluated"])
    ).order_by(EvaluationHistory.timestamp.desc()).all()
    return history


@router.get("/{meeting_id}/benchmark/version", response_model=schemas.BenchmarkVersionResponse)
def get_benchmark_version(meeting_id: int, db: Session = Depends(get_db)):
    """Get evaluation engine metadata and version info."""
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    gt_version = "None"
    gt_row = load_ground_truth(db, meeting.id)
    if gt_row:
        gt_version = str(gt_row.version)

    return schemas.BenchmarkVersionResponse(
        reference_version=gt_version,
        evaluation_version=ENGINE_VERSION,
        engine_name=ENGINE_NAME,
        last_updated=meeting.upload_time
    )



# --- Live Metrics Endpoint ---------------------------------------------------

@router.get("/{meeting_id}/live-metrics", response_model=schemas.LiveEvaluationResponse)
def get_live_metrics(meeting_id: int, db: Session = Depends(get_db)):
    """
    Return the 12 canonical evaluation metric cards for a meeting.

    This endpoint is the authoritative source for the Evaluation Dashboard.
    It runs the Universal Evaluation Engine and assembles structured cards for:
      1.  Speech Recognition   (WER if reference, else runtime quality signals)
      2.  Speaker Intelligence (DER if reference, else diarization stats)
      3.  Executive Summary    (ROUGE-L if reference, else grounding signals)
      4.  Minutes of Meeting   (ROUGE-L if reference, else structural signals)
      5.  Action Items         (F1 if reference, else completeness metrics)
      6.  Topic Detection      (C_v NPMI - always computed when transcript exists)
      7.  Sentiment            (F1 if reference, else detected label)
      8.  Similar Meetings     (nDCG if labels exist, else cosine similarity)
      9.  Chatbot              (operational stats from actual chat sessions)
      10. Audio Signal         (SNR / clipping / codec from ffmpeg+numpy)
      11. Agenda Grounding     (evidence coverage, confidence, unsupported %)
      12. Decision Grounding   (evidence, timestamp, speaker attribution)
      +   Processing Performance (always from real runtime timestamps)

    Meetings in 'processing' or 'error' states return partial metrics with
    a warning note. Only 'pending' (never started) meetings are blocked.
    All values come from actual execution. Metrics without reference data are
    reported as status=not_measured with an explicit reason.
    """
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    # Only block completely unstarted meetings — partial results are better
    # than no information at all for meetings still processing or in error.
    if meeting.status == "pending":
        raise HTTPException(
            status_code=422,
            detail=(
                f"Meeting has not started processing (status: {meeting.status}). "
                "Live metrics are available once processing begins."
            ),
        )

    try:
        return build_live_evaluation(meeting, db)
    except Exception as e:
        logger.error(
            "Live metrics build failed for meeting %s: %s", meeting_id, e, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to build live metrics.") from e




@router.get("/{meeting_id}/runtime-quality-metrics", response_model=schemas.RuntimeQualityMetricsResponse)
def get_runtime_quality_metrics(meeting_id: int, db: Session = Depends(get_db)):
    """
    Compute and return all reference-free runtime quality metrics for a meeting.

    Unlike the benchmark endpoint (which requires ground truth), this endpoint
    always returns metrics calculated directly from actual pipeline execution data:
    - STT: transcript coverage, RTF, timestamp validity, correction rate
    - Speaker: speaker count, coverage, distribution per speaker, turn analysis
    - Content: summary grounding ratio, MoM structure, agenda/decision evidence
    - Action Items: field completeness, evidence coverage, priority validity
    - Performance: per-stage pipeline timing from actual DB columns

    No values are fabricated. None means insufficient data to compute that metric.
    All values change with actual meeting content — two different meetings will
    produce different runtime quality values.
    """
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    if meeting.status not in ("completed", "approved"):
        raise HTTPException(
            status_code=422,
            detail=f"Meeting processing not complete (status: {meeting.status}). "
                   "Runtime quality metrics are available only after processing completes."
        )

    try:
        # ── Gather segment data ──────────────────────────────────────────────
        from backend.models.models import TranscriptSegment
        segs = db.query(TranscriptSegment).filter(TranscriptSegment.meeting_id == meeting_id).all()
        seg_dicts = [
            {
                "speaker":            getattr(s, "speaker", None),
                "start_time":         getattr(s, "start_time", None),
                "end_time":           getattr(s, "end_time", None),
                "text":               getattr(s, "text", None),
                "stt_confidence":     getattr(s, "stt_confidence", None),
                "correction_applied": getattr(s, "correction_applied", False),
            }
            for s in segs
        ]
        audio_dur   = float(getattr(meeting, "duration", 0) or 0)
        stt_dur     = float(getattr(meeting, "stt_duration_sec", 0) or 0) or None
        spk_dur     = float(getattr(meeting, "speaker_duration_sec", 0) or 0) or None

        # ── STT Runtime Metrics ──────────────────────────────────────────────
        stt_rt = rq.stt_runtime_metrics(seg_dicts, audio_dur, stt_dur)

        # ── Speaker Runtime Metrics ──────────────────────────────────────────
        spk_rt = rq.speaker_runtime_metrics(seg_dicts, audio_dur, spk_dur)

        # ── Content Runtime Metrics ──────────────────────────────────────────
        analysis    = getattr(meeting, "analysis", None)
        summary_txt = (analysis.summary if analysis else None) or ""
        mom_txt     = (analysis.mom if analysis else None) or ""
        tx_text     = " ".join(s.get("text", "") or "" for s in seg_dicts)

        sum_rt  = rq.summary_runtime_metrics(summary_txt, tx_text)
        mom_rt  = rq.mom_runtime_metrics(mom_txt)

        akd_rows = db.query(AgendaKeyDecision).filter(
            AgendaKeyDecision.meeting_id == meeting_id
        ).all()
        akd_dicts = [
            {
                "agenda": getattr(r, "agenda", None),
                "key_decision": getattr(r, "key_decision", None),
                "evidence": getattr(r, "evidence", None),
                "match_score": getattr(r, "match_score", None),
            }
            for r in akd_rows
        ]
        agenda_rt = rq.agenda_decision_runtime_metrics(akd_dicts)

        # ── Action Item Runtime Metrics ──────────────────────────────────────
        ai_rows = db.query(ActionItem).filter(ActionItem.meeting_id == meeting_id).all()
        ai_dicts = [
            {
                "task":       getattr(r, "task", None),
                "owner":      getattr(r, "owner", None),
                "deadline":   getattr(r, "deadline", None),
                "due_date":   getattr(r, "due_date", None),
                "priority":   getattr(r, "priority", None),
                "evidence":   getattr(r, "evidence", None),
            }
            for r in ai_rows
        ]
        ai_rt = rq.action_item_runtime_metrics(ai_dicts)

        # ── Processing Performance Metrics ───────────────────────────────────
        perf_rt = rq.processing_performance_metrics(meeting)

        # ── Build and return structured response ─────────────────────────────
        dist_items = [
            schemas.SpeakerDistributionItem(**d)
            for d in spk_rt.get("speaker_distribution", [])
        ]

        return schemas.RuntimeQualityMetricsResponse(
            meeting_id=meeting_id,
            evaluation_type="REFERENCE_FREE",
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            stt=schemas.SttRuntimeQualityMetrics(
                total_segments=stt_rt.get("total_segments"),
                transcript_coverage=stt_rt.get("transcript_coverage"),
                empty_segment_rate=stt_rt.get("empty_segment_rate"),
                timestamp_validity_rate=stt_rt.get("timestamp_validity_rate"),
                repetition_rate=stt_rt.get("repetition_rate"),
                correction_rate=stt_rt.get("correction_rate"),
                avg_stt_confidence=stt_rt.get("avg_stt_confidence"),
                confidence_sample_count=stt_rt.get("confidence_sample_count"),
                rtf=stt_rt.get("rtf"),
                stt_runtime_quality_score=stt_rt.get("stt_runtime_quality_score"),
                weights=stt_rt.get("weights"),
            ),
            speaker=schemas.SpeakerRuntimeQualityMetrics(
                detected_speakers=spk_rt.get("detected_speakers"),
                speaker_coverage=spk_rt.get("speaker_coverage"),
                speaker_assignment_rate=spk_rt.get("speaker_assignment_rate"),
                turn_count=spk_rt.get("turn_count"),
                avg_turn_duration_sec=spk_rt.get("avg_turn_duration_sec"),
                timestamp_consistency_rate=spk_rt.get("timestamp_consistency_rate"),
                unassigned_duration_sec=spk_rt.get("unassigned_duration_sec"),
                micro_segment_count=spk_rt.get("micro_segment_count"),
                speaker_runtime_quality_score=spk_rt.get("speaker_runtime_quality_score"),
                speaker_distribution=dist_items,
                weights=spk_rt.get("weights"),
            ),
            content=schemas.ContentRuntimeQualityMetrics(
                summary_word_overlap_ratio=sum_rt.get("summary_word_overlap_ratio"),
                unsupported_sentence_rate=sum_rt.get("unsupported_sentence_rate"),
                summary_word_count=sum_rt.get("summary_word_count"),
                summary_runtime_quality_score=sum_rt.get("summary_runtime_quality_score"),
                mom_word_count=mom_rt.get("mom_word_count"),
                mom_section_count=mom_rt.get("section_count"),
                agenda_item_count=agenda_rt.get("agenda_item_count"),
                decisions_with_evidence_pct=agenda_rt.get("decisions_with_evidence_pct"),
                avg_decision_match_score=agenda_rt.get("avg_decision_match_score"),
                unsupported_decision_pct=agenda_rt.get("unsupported_decision_pct"),
            ),
            action_items=schemas.ActionItemRuntimeQualityMetrics(
                action_item_count=ai_rt.get("action_item_count"),
                field_completeness_pct=ai_rt.get("field_completeness_pct"),
                evidence_coverage_pct=ai_rt.get("evidence_coverage_pct"),
                assigned_to_coverage_pct=ai_rt.get("assigned_to_coverage_pct"),
                due_date_coverage_pct=ai_rt.get("due_date_coverage_pct"),
                priority_valid_pct=ai_rt.get("priority_valid_pct"),
                duplicate_rate_pct=ai_rt.get("duplicate_rate_pct"),
                action_item_runtime_quality_score=ai_rt.get("action_item_runtime_quality_score"),
            ),
            performance=schemas.ProcessingPerformanceRuntimeMetrics(
                audio_duration_sec=perf_rt.get("audio_duration_sec"),
                stt_duration_sec=perf_rt.get("stt_duration_sec"),
                speaker_duration_sec=perf_rt.get("speaker_duration_sec"),
                summary_duration_sec=perf_rt.get("summary_duration_sec"),
                mom_duration_sec=perf_rt.get("mom_duration_sec"),
                action_items_duration_sec=perf_rt.get("action_items_duration_sec"),
                total_processing_time_sec=perf_rt.get("total_processing_time_sec"),
                rtf=perf_rt.get("rtf"),
                stt_word_count=perf_rt.get("stt_word_count"),
                speaker_count=perf_rt.get("speaker_count"),
            ),
        )

    except Exception as e:
        logger.error("Runtime quality metrics failed for meeting %s: %s", meeting_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to compute runtime quality metrics.") from e

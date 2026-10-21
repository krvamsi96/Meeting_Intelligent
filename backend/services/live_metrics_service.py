"""
Live Metrics Service
====================
Builds the 10 canonical evaluation metric cards for the AI Meeting Intelligence
Platform evaluation dashboard.

Design rules (MANDATORY - never violate):
  * Every metric value comes from real execution: engine report, runtime
    timestamps, or actual model outputs stored in the database.
  * A metric that cannot be computed yields status='not_measured' with an
    explicit reason. The value field is left as None.
  * No hardcoded percentages, no random values, no estimated values.
  * Processing performance timings come exclusively from Meeting model
    columns that are populated by the pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.evaluation import runtime_quality as rq
from backend.evaluation.engine import ENGINE_VERSION, evaluation_engine
from backend.models.models import ActionItem, AgendaKeyDecision, ChatSession, Meeting
from backend.schemas.schemas import (
    LiveEvaluationResponse,
    LiveMetricCard,
    LiveMetricDisplayItem,
    ProcessingPerformance,
)

logger = logging.getLogger(__name__)

MEASURED = "measured"
NOT_MEASURED = "not_measured"


def _safe_float(val: Any) -> Optional[float]:
    """Return float or None - never raises."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _item(
    label: str,
    value: Any,
    unit: Optional[str] = None,
    source: Optional[str] = None,
) -> LiveMetricDisplayItem:
    return LiveMetricDisplayItem(label=label, value=value, unit=unit, source=source)


def _not_measured_card(metric_name: str, metric_key: str, reason: str) -> LiveMetricCard:
    return LiveMetricCard(
        metric_name=metric_name,
        metric_key=metric_key,
        status=NOT_MEASURED,
        reason=reason,
    )


# ---- 1. Speech Recognition --------------------------------------------------

def _build_stt_card(meeting: Meeting, tasks: Dict[str, Any]) -> LiveMetricCard:
    stt = tasks.get("stt", {})
    segments = list(meeting.segments or [])

    if stt.get("status") == MEASURED:
        m = stt.get("metrics", {})
        wer = _safe_float(m.get("wer"))
        raw_wer = _safe_float(m.get("raw_wer"))
        corrected_wer = _safe_float(m.get("corrected_wer"))
        cer = _safe_float(m.get("cer"))
        acc = _safe_float(m.get("accuracy"))
        items = [
            _item("WER (Word Error Rate)", wer, "%", "engine_computed"),
            _item("Raw STT WER", raw_wer, "%", "engine_computed"),
            _item("Corrected STT WER", corrected_wer, "%", "engine_computed"),
            _item("CER (Character Error Rate)", cer, "%", "engine_computed"),
            _item("Accuracy (1-WER)", acc, "%", "engine_computed"),
            _item("Substitutions", m.get("substitutions"), source="engine_computed"),
            _item("Deletions", m.get("deletions"), source="engine_computed"),
            _item("Insertions", m.get("insertions"), source="engine_computed"),
        ]
        return LiveMetricCard(
            metric_name="Speech Recognition",
            metric_key="stt",
            status=MEASURED,
            primary_metric_label="WER",
            primary_metric_value=wer,
            primary_metric_unit="%",
            display_items=[i for i in items if i.value is not None],
            source="reference_transcript_vs_whisper_output",
            execution_time_sec=_safe_float(getattr(meeting, "stt_duration_sec", None)),
        )

    # Ground truth absent — compute reference-free runtime quality metrics.
    # These are REAL measurements from actual pipeline execution, NOT accuracy.
    seg_dicts = [
        {
            "start_time": getattr(s, "start_time", None),
            "end_time":   getattr(s, "end_time", None),
            "text":       getattr(s, "text", None),
            "stt_confidence":    getattr(s, "stt_confidence", None),
            "correction_applied": getattr(s, "correction_applied", False),
        }
        for s in segments
    ]
    audio_dur = _safe_float(getattr(meeting, "duration", None)) or 0.0
    stt_dur   = _safe_float(getattr(meeting, "stt_duration_sec", None))

    rt = rq.stt_runtime_metrics(seg_dicts, audio_dur, stt_dur)

    coverage   = rt.get("transcript_coverage")
    empty_rate = rt.get("empty_segment_rate")
    ts_valid   = rt.get("timestamp_validity_rate")
    rept_rate  = rt.get("repetition_rate")
    corr_rate  = rt.get("correction_rate")
    avg_conf   = rt.get("avg_stt_confidence")
    rtf        = rt.get("rtf")
    q_score    = rt.get("stt_runtime_quality_score")
    total_segs = rt.get("total_segments", 0)

    items = [
        _item("Transcript Coverage",        coverage,   "%",  "reference_free_runtime"),
        _item("Timestamp Validity Rate",     ts_valid,   "%",  "reference_free_runtime"),
        _item("Empty Segment Rate",          empty_rate, "%",  "reference_free_runtime"),
        _item("Correction Rate",             corr_rate,  "%",  "reference_free_runtime"),
        _item("Repetition Rate",             rept_rate,  "%",  "reference_free_runtime"),
        _item("Avg STT Confidence",          avg_conf,   None, "whisper_logprob"),
        _item("Real-Time Factor (RTF)",       rtf,        None, "pipeline_timing"),
        _item("Total Segments",              total_segs, None, "pipeline_output"),
        _item("STT Runtime Quality Score",   q_score,    None, "reference_free_composite"),
    ]

    primary_label = "Transcript Coverage"
    primary_val   = coverage
    if q_score is not None:
        primary_label = "STT Runtime Quality"
        primary_val   = q_score

    if total_segs > 0 or coverage is not None:
        return LiveMetricCard(
            metric_name="Speech Recognition (Runtime Quality)",
            metric_key="stt",
            status=MEASURED,
            primary_metric_label=primary_label,
            primary_metric_value=primary_val,
            primary_metric_unit=None,
            display_items=[i for i in items if i.value is not None],
            source="reference_free_pipeline_telemetry",
            execution_time_sec=stt_dur,
            notes="No reference transcript — showing runtime quality signals. Upload ground truth for WER/CER.",
        )

    return _not_measured_card(
        "Speech Recognition",
        "stt",
        "no_transcript_segments_and_no_reference_transcript",
    )



# ---- 2. Speaker Intelligence ------------------------------------------------

def _build_speaker_card(meeting: Meeting, tasks: Dict[str, Any]) -> LiveMetricCard:
    diar = tasks.get("diarization", {})

    if diar.get("status") == MEASURED:
        m = diar.get("metrics", {})
        der = _safe_float(m.get("der"))
        jer = _safe_float(m.get("jer"))
        acc = _safe_float(m.get("accuracy"))
        items = [
            _item("DER (Diarization Error Rate)", der, "%", "engine_computed"),
            _item("JER (Jaccard Error Rate)", jer, "%", "engine_computed"),
            _item("Accuracy (1-DER)", acc, "%", "engine_computed"),
        ]
        return LiveMetricCard(
            metric_name="Speaker Intelligence",
            metric_key="diarization",
            status=MEASURED,
            primary_metric_label="DER",
            primary_metric_value=der,
            primary_metric_unit="%",
            display_items=[i for i in items if i.value is not None],
            source="reference_timeline_vs_nemo_output",
            execution_time_sec=_safe_float(getattr(meeting, "speaker_duration_sec", None)),
        )

    # Ground truth absent — compute reference-free speaker runtime metrics.
    # All values derived from actual NeMo diarization output timestamps.
    segments = list(meeting.segments or [])
    seg_dicts = [
        {
            "speaker":    getattr(s, "speaker", None),
            "start_time": getattr(s, "start_time", None),
            "end_time":   getattr(s, "end_time", None),
            "text":       getattr(s, "text", None),
        }
        for s in segments
    ]
    audio_dur = _safe_float(getattr(meeting, "duration", None)) or 0.0
    spk_dur   = _safe_float(getattr(meeting, "speaker_duration_sec", None))

    rt = rq.speaker_runtime_metrics(seg_dicts, audio_dur, spk_dur)

    detected   = rt.get("detected_speakers", 0)
    coverage   = rt.get("speaker_coverage")
    assign_rt  = rt.get("speaker_assignment_rate")
    turn_count = rt.get("turn_count")
    avg_turn   = rt.get("avg_turn_duration_sec")
    ts_consist = rt.get("timestamp_consistency_rate")
    q_score    = rt.get("speaker_runtime_quality_score")

    # Build speaker distribution display items from per-speaker stats
    dist = rt.get("speaker_distribution", [])
    items = [
        _item("Detected Speakers",           detected,   None, "nemo_diarization_output"),
        _item("Speaker Coverage",            coverage,   "%",  "reference_free_runtime"),
        _item("Speaker Assignment Rate",     assign_rt,  "%",  "reference_free_runtime"),
        _item("Speaker Turn Count",          turn_count, None, "reference_free_runtime"),
        _item("Avg Turn Duration",           avg_turn,   "s",  "reference_free_runtime"),
        _item("Timestamp Consistency",       ts_consist, "%",  "reference_free_runtime"),
        _item("Speaker Runtime Quality",     q_score,    None, "reference_free_composite"),
    ]
    # Add per-speaker breakdown
    for spk_info in dist[:6]:  # show up to 6 speakers
        spk = spk_info.get("speaker", "?")
        pct = spk_info.get("percentage")
        dur = spk_info.get("duration_seconds")
        items.append(_item(f"{spk} — Speaking %", pct, "%", "nemo_diarization_output"))
        items.append(_item(f"{spk} — Duration",   dur, "s",  "nemo_diarization_output"))
        items.append(_item(f"{spk} — Turns",      spk_info.get("turn_count"), None, "nemo_diarization_output"))

    if detected > 0 or coverage is not None:
        return LiveMetricCard(
            metric_name="Speaker Intelligence (Runtime Quality)",
            metric_key="diarization",
            status=MEASURED,
            primary_metric_label="Detected Speakers",
            primary_metric_value=float(detected),
            primary_metric_unit=None,
            display_items=[i for i in items if i.value is not None],
            source="reference_free_nemo_diarization_output",
            execution_time_sec=spk_dur,
            notes="No reference timeline — showing diarization runtime signals. Upload ground truth for DER/JER.",
        )

    return _not_measured_card(
        "Speaker Intelligence",
        "diarization",
        "reference_speaker_timeline_not_provided (Upload reference speaker timeline ground truth to compute DER/JER)",
    )



# ---- 3. Executive Summary --------------------------------------------------

def _build_summary_card(meeting: Meeting, tasks: Dict[str, Any]) -> LiveMetricCard:
    summ = tasks.get("summary", {})

    if summ.get("status") == MEASURED:
        m = summ.get("metrics", {})
        rougel = _safe_float(m.get("rougeL"))
        items = [
            _item("ROUGE-L", rougel, "%", "engine_computed"),
            _item("ROUGE-1", _safe_float(m.get("rouge1")), "%", "engine_computed"),
            _item("ROUGE-2", _safe_float(m.get("rouge2")), "%", "engine_computed"),
            _item("BERTScore F1", _safe_float(m.get("bertscore_f1")), "%", "engine_computed"),
        ]
        return LiveMetricCard(
            metric_name="Executive Summary",
            metric_key="summary",
            status=MEASURED,
            primary_metric_label="ROUGE-L",
            primary_metric_value=rougel,
            primary_metric_unit="%",
            display_items=[i for i in items if i.value is not None],
            source="reference_summary_vs_llm_output",
            execution_time_sec=_safe_float(getattr(meeting, "summary_duration_sec", None)),
        )

    # No ground truth — show reference-free grounding signal from actual
    # summary content vs transcript. This is NOT ROUGE.
    not_measured_metrics = summ.get("metrics", {})
    overlap = _safe_float(not_measured_metrics.get("summary_word_overlap_ratio"))
    unsp    = _safe_float(not_measured_metrics.get("unsupported_sentence_rate"))
    wc      = not_measured_metrics.get("summary_word_count")
    sc      = not_measured_metrics.get("sentence_count")
    # Also try computing directly if metrics not already in engine result
    if overlap is None:
        analysis = getattr(meeting, "analysis", None)
        gen_summary = (analysis.summary if analysis else None) or ""
        segments = list(meeting.segments or [])
        tx_text = " ".join(getattr(s, "text", "") or "" for s in segments)
        if gen_summary.strip() and tx_text.strip():
            rt = rq.summary_runtime_metrics(gen_summary, tx_text)
            overlap = rt.get("summary_word_overlap_ratio")
            unsp    = rt.get("unsupported_sentence_rate")
            wc      = rt.get("summary_word_count")
            sc      = rt.get("sentence_count")

    if overlap is not None or wc:
        items = [
            _item("Word Overlap w/ Transcript",     overlap, "%",  "reference_free_grounding"),
            _item("Unsupported Sentence Rate",       unsp,   "%",  "reference_free_grounding"),
            _item("Summary Word Count",              wc,     None, "pipeline_output"),
            _item("Summary Sentence Count",          sc,     None, "pipeline_output"),
        ]
        return LiveMetricCard(
            metric_name="Executive Summary (Runtime Grounding)",
            metric_key="summary",
            status=MEASURED,
            primary_metric_label="Word Overlap w/ Transcript",
            primary_metric_value=overlap,
            primary_metric_unit="%",
            display_items=[i for i in items if i.value is not None],
            source="reference_free_transcript_overlap",
            execution_time_sec=_safe_float(getattr(meeting, "summary_duration_sec", None)),
            notes="No reference summary — showing grounding signals (NOT ROUGE). Upload ground truth for ROUGE/BERTScore.",
        )

    return _not_measured_card(
        "Executive Summary",
        "summary",
        "reference_summary_not_provided (Upload reference summary ground truth to compute ROUGE & BERTScore)",
    )



# ---- 4. Minutes of Meeting -------------------------------------------------

def _build_mom_card(meeting: Meeting, tasks: Dict[str, Any]) -> LiveMetricCard:
    mom_task = tasks.get("mom", {})

    if mom_task.get("status") == MEASURED:
        m = mom_task.get("metrics", {})
        rougel = _safe_float(m.get("rougeL"))
        items = [
            _item("ROUGE-L", rougel, "%", "engine_computed"),
            _item("ROUGE-1", _safe_float(m.get("rouge1")), "%", "engine_computed"),
            _item("BERTScore F1", _safe_float(m.get("bertscore_f1")), "%", "engine_computed"),
        ]
        return LiveMetricCard(
            metric_name="Minutes of Meeting",
            metric_key="mom",
            status=MEASURED,
            primary_metric_label="ROUGE-L",
            primary_metric_value=rougel,
            primary_metric_unit="%",
            display_items=[i for i in items if i.value is not None],
            source="reference_mom_vs_llm_output",
            execution_time_sec=_safe_float(getattr(meeting, "mom_duration_sec", None)),
        )

    # No ground truth — show structural runtime signals from actual MoM.
    not_measured_metrics = mom_task.get("metrics", {})
    wc = not_measured_metrics.get("mom_word_count")
    sc = not_measured_metrics.get("section_count")
    # Compute directly if not in engine result
    if wc is None:
        analysis = getattr(meeting, "analysis", None)
        gen_mom = (analysis.mom if analysis else None) or ""
        if gen_mom.strip():
            mom_rt = rq.mom_runtime_metrics(gen_mom)
            wc = mom_rt.get("mom_word_count")
            sc = mom_rt.get("section_count")

    if wc is not None:
        items = [
            _item("MoM Word Count",   wc, None, "pipeline_output"),
            _item("Section Count",    sc, None, "pipeline_output"),
        ]
        return LiveMetricCard(
            metric_name="Minutes of Meeting (Structural)",
            metric_key="mom",
            status=MEASURED,
            primary_metric_label="MoM Word Count",
            primary_metric_value=float(wc) if wc else None,
            primary_metric_unit=None,
            display_items=[i for i in items if i.value is not None],
            source="reference_free_structural_analysis",
            execution_time_sec=_safe_float(getattr(meeting, "mom_duration_sec", None)),
            notes="No reference MoM — showing structural signals (NOT ROUGE). Upload ground truth for ROUGE/BERTScore.",
        )

    return _not_measured_card(
        "Minutes of Meeting",
        "mom",
        "reference_mom_not_provided (Upload reference MoM ground truth to compute ROUGE & BERTScore)",
    )



# ---- 5. Action Items -------------------------------------------------------

def _build_action_items_card(meeting: Meeting, tasks: Dict[str, Any], db: Session) -> LiveMetricCard:
    act = tasks.get("action_items", {})

    if act.get("status") == MEASURED:
        m = act.get("metrics", {})
        f1 = _safe_float(m.get("f1"))
        items = [
            _item("F1 Score", f1, "%", "engine_computed"),
            _item("Precision", _safe_float(m.get("precision")), "%", "engine_computed"),
            _item("Recall", _safe_float(m.get("recall")), "%", "engine_computed"),
            _item("Reference Count", m.get("reference_count"), source="ground_truth"),
            _item("Generated Count", m.get("generated_count"), source="llm_output"),
        ]
        return LiveMetricCard(
            metric_name="Action Items",
            metric_key="action_items",
            status=MEASURED,
            primary_metric_label="F1 Score",
            primary_metric_value=f1,
            primary_metric_unit="%",
            display_items=[i for i in items if i.value is not None],
            source="reference_action_items_vs_llm_output",
            execution_time_sec=_safe_float(getattr(meeting, "action_items_duration_sec", None)),
        )

    # No ground truth — compute reference-free completeness metrics from
    # actual ActionItem rows stored in the database.
    not_measured_metrics = act.get("metrics", {}) or {}
    ai_count = not_measured_metrics.get("action_item_count")

    # Fetch actual ActionItem rows from DB for precise computation
    ai_rows = []
    try:
        ai_db = db.query(ActionItem).filter(ActionItem.meeting_id == meeting.id).all()
        ai_rows = [
            {
                "task":       getattr(r, "task", None),
                "owner":      getattr(r, "owner", None),
                "deadline":   getattr(r, "deadline", None),
                "due_date":   getattr(r, "due_date", None),
                "priority":   getattr(r, "priority", None),
                "evidence":   getattr(r, "evidence", None),
                "assigned_to": getattr(r, "owner", None),
            }
            for r in ai_db
        ]
    except Exception as e:
        logger.warning("ActionItem DB query failed in live metrics card: %s", e)

    if ai_rows:
        ai_m = rq.action_item_runtime_metrics(ai_rows)
    elif ai_count is not None:
        # Use engine-computed stats if DB fetch unavailable
        ai_m = not_measured_metrics
    else:
        ai_m = {}

    count     = ai_m.get("action_item_count") or 0
    field_c   = ai_m.get("field_completeness_pct")
    evidence  = ai_m.get("evidence_coverage_pct")
    owner_c   = ai_m.get("assigned_to_coverage_pct")
    due_c     = ai_m.get("due_date_coverage_pct")
    prio_c    = ai_m.get("priority_valid_pct")
    dup_c     = ai_m.get("duplicate_rate_pct")
    q_score   = ai_m.get("action_item_runtime_quality_score")

    if count > 0 or field_c is not None:
        items = [
            _item("Action Item Count",           count,    None, "pipeline_output"),
            _item("Field Completeness",           field_c,  "%",  "reference_free_completeness"),
            _item("Evidence Coverage",            evidence, "%",  "reference_free_completeness"),
            _item("Assigned-To Coverage",         owner_c,  "%",  "reference_free_completeness"),
            _item("Due Date Coverage",            due_c,    "%",  "reference_free_completeness"),
            _item("Valid Priority Rate",          prio_c,   "%",  "reference_free_completeness"),
            _item("Duplicate Rate",               dup_c,    "%",  "reference_free_completeness"),
            _item("Runtime Quality Score",        q_score,  None, "reference_free_composite"),
        ]
        return LiveMetricCard(
            metric_name="Action Items (Runtime Completeness)",
            metric_key="action_items",
            status=MEASURED,
            primary_metric_label="Action Item Count",
            primary_metric_value=float(count),
            primary_metric_unit=None,
            display_items=[i for i in items if i.value is not None],
            source="reference_free_db_action_items",
            execution_time_sec=_safe_float(getattr(meeting, "action_items_duration_sec", None)),
            notes="No reference action items — showing completeness metrics (NOT F1). Upload ground truth for Precision/Recall/F1.",
        )

    return _not_measured_card(
        "Action Items",
        "action_items",
        "reference_action_items_not_provided (Upload reference action items ground truth to compute F1)",
    )



# ---- 6. Topic Detection ----------------------------------------------------

def _build_topics_card(meeting: Meeting, tasks: Dict[str, Any]) -> LiveMetricCard:
    """Topic Coherence (C_v NPMI) is computed directly when topic analysis exists."""
    top = tasks.get("topics", {})
    if top.get("status") == MEASURED:
        m = top.get("metrics", {})
        coherence = _safe_float(m.get("coherence"))
        topic_count = m.get("topic_count")
        kw_raw = getattr(meeting, "keyword_frequency", None) or ""
        top_topics: List[str] = []
        try:
            kw = json.loads(kw_raw) if isinstance(kw_raw, str) and kw_raw.strip() else kw_raw
            if isinstance(kw, dict):
                top_topics = [str(k) for k in list(kw.keys())[:5]]
            elif isinstance(kw, list):
                top_topics = [str(t) for t in kw[:5]]
        except Exception:
            pass
        items = [
            _item("Topic Count", topic_count, source="corpus_analysis"),
            _item("Topic Coherence (C_v NPMI)", coherence, None, "engine_computed"),
            _item("Top Topics", ", ".join(top_topics) if top_topics else None, source="keyword_frequency"),
        ]
        if m.get("f1") is not None:
            items.append(_item("F1 vs Reference Topics", _safe_float(m.get("f1")), "%", "engine_computed"))
        return LiveMetricCard(
            metric_name="Topic Detection",
            metric_key="topics",
            status=MEASURED,
            primary_metric_label="Topic Coherence (C_v)",
            primary_metric_value=coherence,
            primary_metric_unit=None,
            display_items=[i for i in items if i.value is not None],
            source="corpus_npmi_analysis",
            execution_time_sec=None,
        )

    return _not_measured_card(
        "Topic Detection",
        "topics",
        "topic_analysis_not_performed",
    )


# ---- 7. Sentiment ----------------------------------------------------------

def _build_sentiment_card(meeting: Meeting, tasks: Dict[str, Any]) -> LiveMetricCard:
    sent = tasks.get("sentiment", {})
    if sent.get("status") == MEASURED:
        m = sent.get("metrics", {})
        f1 = _safe_float(m.get("f1"))
        items = [
            _item("F1 Score", f1, "%", "engine_computed"),
            _item("Accuracy", _safe_float(m.get("accuracy")), "%", "engine_computed"),
            _item("Reference Sentiment", m.get("reference"), source="ground_truth"),
            _item("Predicted Sentiment", m.get("predicted"), source="llm_output"),
        ]
        return LiveMetricCard(
            metric_name="Sentiment",
            metric_key="sentiment",
            status=MEASURED,
            primary_metric_label="F1 Score",
            primary_metric_value=f1,
            primary_metric_unit="%",
            display_items=[i for i in items if i.value is not None],
            source="reference_label_vs_llm_output",
            execution_time_sec=None,
        )

    return _not_measured_card(
        "Sentiment",
        "sentiment",
        "reference_sentiment_not_provided (Upload reference sentiment ground truth to compute F1)",
    )


# ---- 8. Similar Meetings ---------------------------------------------------

def _build_similarity_card(meeting: Meeting, tasks: Dict[str, Any]) -> LiveMetricCard:
    sim = tasks.get("similarity", {})
    if sim.get("status") == MEASURED:
        m = sim.get("metrics", {})
        ndcg = _safe_float(m.get("ndcg"))
        items = [
            _item("nDCG@10", ndcg, None, "engine_computed"),
            _item("DCG", _safe_float(m.get("dcg")), None, "engine_computed"),
            _item("IDCG", _safe_float(m.get("idcg")), None, "engine_computed"),
            _item("k", m.get("k"), source="engine_computed"),
        ]
        return LiveMetricCard(
            metric_name="Similar Meetings",
            metric_key="similarity",
            status=MEASURED,
            primary_metric_label="nDCG@10",
            primary_metric_value=ndcg,
            primary_metric_unit=None,
            display_items=[i for i in items if i.value is not None],
            source="reference_labels_vs_vector_index",
            execution_time_sec=None,
        )

    # No reference labels — show raw cosine similarity if available.
    # This is the vector search cosine score, NOT nDCG.
    not_measured_metrics = sim.get("metrics", {}) or {}
    raw_cosine = _safe_float(
        not_measured_metrics.get("cosine_similarity_score")
        or getattr(meeting, "similarity_score", None)
    )
    matched_id   = getattr(meeting, "matched_meeting_id", None)
    matched_code = getattr(meeting, "matched_meeting_code", None)

    if raw_cosine is not None:
        items = [
            _item("Top Match Cosine Similarity", raw_cosine, None, "vector_search_output"),
            _item("Matched Meeting", matched_code or str(matched_id), None, "vector_search_output"),
        ]
        return LiveMetricCard(
            metric_name="Similar Meetings (Cosine Similarity)",
            metric_key="similarity",
            status=MEASURED,
            primary_metric_label="Cosine Similarity Score",
            primary_metric_value=raw_cosine,
            primary_metric_unit=None,
            display_items=[i for i in items if i.value is not None],
            source="vector_search_cosine_score_not_ndcg",
            execution_time_sec=None,
            notes="Cosine similarity from vector search — NOT nDCG. Upload reference labels to compute nDCG.",
        )

    return _not_measured_card(
        "Similar Meetings",
        "similarity",
        "reference_similarity_labels_not_provided (Upload reference similarity labels to compute nDCG@10)",
    )



# ---- 9. Chatbot ------------------------------------------------------------

def _build_chatbot_card(meeting: Meeting, db: Session) -> LiveMetricCard:
    """
    Chatbot evaluation.
    Reports operational stats from database chat sessions.
    """
    try:
        chat_rows = (
            db.query(ChatSession)
            .filter(ChatSession.meeting_id == meeting.id, ChatSession.role == "assistant")
            .all()
        )
    except Exception as e:
        logger.warning("Chatbot card: could not query chat_sessions: %s", e)
        chat_rows = []

    segment_count = len(meeting.segments or [])
    answers_generated = len(chat_rows)
    grounded_sources = 0
    for row in chat_rows:
        if row.sources:
            try:
                src = json.loads(row.sources) if isinstance(row.sources, str) else row.sources
                if isinstance(src, list):
                    grounded_sources += len(src)
            except Exception:
                pass

    if segment_count == 0 and answers_generated == 0:
        return _not_measured_card(
            "Chatbot",
            "chatbot",
            "chatbot_not_indexed_or_queried",
        )

    items = [
        _item("Indexed Transcript Segments", segment_count, source="vector_index"),
        _item("Answers Generated", answers_generated, source="chat_sessions"),
        _item("Grounded Sources Used", grounded_sources, source="chat_sessions"),
    ]
    return LiveMetricCard(
        metric_name="Chatbot",
        metric_key="chatbot",
        status=MEASURED,
        primary_metric_label="Indexed Segments",
        primary_metric_value=segment_count,
        primary_metric_unit=None,
        notes="Evaluated from real database chat logs and vector embedding index.",
        display_items=[i for i in items if i.value is not None],
        source="actual_chat_sessions",
        execution_time_sec=None,
    )


# ---- Audio Signal ----------------------------------------------------------

def _build_audio_card(meeting: Meeting, tasks: Dict[str, Any]) -> LiveMetricCard:
    """Surfaces real SNR/clipping/codec telemetry from media_probe + segment coverage."""
    audio = tasks.get("audio_signal", {})

    if audio.get("status") != MEASURED:
        reason = audio.get("reason", "media_probe_unavailable")
        return _not_measured_card(
            "Audio Signal",
            "audio_signal",
            f"{reason} (requires ffmpeg + numpy; no invented quality mapping applied)",
        )

    m = audio.get("metrics", {})
    snr          = _safe_float(m.get("snr_db"))
    peak_db      = _safe_float(m.get("peak_db"))
    rms_db       = _safe_float(m.get("rms_db"))
    clipping     = _safe_float(m.get("clipping_ratio"))
    silence_sig  = _safe_float(m.get("silence_ratio_signal"))
    speech_pct   = _safe_float(m.get("speech_ratio_pct"))
    silence_seg  = _safe_float(m.get("silence_ratio_seg_pct"))
    codec        = m.get("codec")
    sample_rate  = m.get("sample_rate")
    channels     = m.get("channels")
    bitrate      = m.get("bitrate")
    aud_dur      = _safe_float(m.get("audio_duration_sec")) or _safe_float(getattr(meeting, "duration", None))
    spk_count    = m.get("detected_speaker_count")

    items = [
        _item("SNR (dB)",                    snr,         "dB",   "ffmpeg_numpy_signal"),
        _item("Peak Level (dBFS)",            peak_db,     "dBFS", "ffmpeg_numpy_signal"),
        _item("RMS Level (dBFS)",             rms_db,      "dBFS", "ffmpeg_numpy_signal"),
        _item("Clipping Ratio",               clipping,    None,   "ffmpeg_numpy_signal"),
        _item("Silence Ratio (signal)",        silence_sig, None,   "ffmpeg_numpy_signal"),
        _item("Speech Coverage (segments)",    speech_pct,  "%",    "segment_derived"),
        _item("Silence Ratio (segments)",      silence_seg, "%",    "segment_derived"),
        _item("Detected Speakers",             spk_count,   None,   "segment_derived"),
        _item("Codec",                         codec,       None,   "ffprobe"),
        _item("Sample Rate",                   sample_rate, "Hz",   "ffprobe"),
        _item("Channels",                      channels,    None,   "ffprobe"),
        _item("Bitrate",                       bitrate,     "bps",  "ffprobe"),
        _item("Audio Duration",                aud_dur,     "s",    "ffprobe"),
    ]

    # Primary metric: SNR if available, else speech coverage
    if snr is not None:
        primary_label, primary_val, primary_unit = "SNR", snr, "dB"
    elif speech_pct is not None:
        primary_label, primary_val, primary_unit = "Speech Coverage", speech_pct, "%"
    else:
        primary_label, primary_val, primary_unit = "Codec", codec, None

    return LiveMetricCard(
        metric_name="Audio Signal",
        metric_key="audio_signal",
        status=MEASURED,
        primary_metric_label=primary_label,
        primary_metric_value=primary_val if not isinstance(primary_val, str) else None,
        primary_metric_unit=primary_unit,
        display_items=[i for i in items if i.value is not None],
        source="ffmpeg_numpy_media_probe",
        execution_time_sec=None,
        notes="Audio telemetry from ffmpeg+numpy — NOT an accuracy metric. No invented quality mapping applied.",
    )


# ---- Agenda Grounding -------------------------------------------------------

def _build_agenda_card(meeting: Meeting, tasks: Dict[str, Any]) -> LiveMetricCard:
    """Reference-free grounding metrics for LLM-generated agenda items."""
    ag = tasks.get("agenda", {})

    if ag.get("status") != MEASURED:
        return _not_measured_card(
            "Agenda Grounding",
            "agenda",
            ag.get("reason", "no_agenda_items_generated (No agenda items extracted by LLM)"),
        )

    m = ag.get("metrics", {})
    count      = m.get("item_count", 0)
    evidence   = _safe_float(m.get("evidence_coverage_pct"))
    avg_conf   = _safe_float(m.get("avg_confidence"))
    low_conf   = _safe_float(m.get("low_confidence_pct"))
    unsup      = _safe_float(m.get("unsupported_pct"))
    dup        = _safe_float(m.get("duplicate_pct"))
    ts_cov     = _safe_float(m.get("timestamp_coverage_pct"))
    q_score    = _safe_float(m.get("agenda_runtime_quality_score"))

    items = [
        _item("Agenda Item Count",            count,    None, "pipeline_output"),
        _item("Evidence Coverage",            evidence, "%",  "reference_free_grounding"),
        _item("Avg Confidence",               avg_conf, None, "llm_output"),
        _item("Low Confidence Rate",          low_conf, "%",  "reference_free_grounding"),
        _item("Unsupported Item Rate",         unsup,    "%",  "reference_free_grounding"),
        _item("Duplicate Rate",               dup,      "%",  "reference_free_grounding"),
        _item("Timestamp Coverage",           ts_cov,   "%",  "reference_free_grounding"),
        _item("Grounding Quality Score",       q_score,  None, "reference_free_composite"),
    ]

    return LiveMetricCard(
        metric_name="Agenda Grounding",
        metric_key="agenda",
        status=MEASURED,
        primary_metric_label="Evidence Coverage",
        primary_metric_value=evidence,
        primary_metric_unit="%",
        display_items=[i for i in items if i.value is not None],
        source="reference_free_llm_grounding",
        execution_time_sec=None,
        notes="No reference agenda — showing grounding signals (NOT F1). Upload ground truth for Precision/Recall/F1.",
    )


# ---- Decision Grounding ----------------------------------------------------

def _build_decisions_card(meeting: Meeting, tasks: Dict[str, Any]) -> LiveMetricCard:
    """Reference-free grounding metrics for LLM-generated decisions."""
    dec = tasks.get("decisions", {})

    if dec.get("status") != MEASURED:
        return _not_measured_card(
            "Decision Grounding",
            "decisions",
            dec.get("reason", "no_decision_items_generated (No decisions extracted by LLM)"),
        )

    m = dec.get("metrics", {})
    count      = m.get("item_count", 0)
    evidence   = _safe_float(m.get("evidence_coverage_pct"))
    ts_cov     = _safe_float(m.get("timestamp_coverage_pct"))
    speaker    = _safe_float(m.get("speaker_attribution_pct"))
    avg_conf   = _safe_float(m.get("avg_confidence"))
    unsup      = _safe_float(m.get("unsupported_pct"))
    q_score    = _safe_float(m.get("decisions_runtime_quality_score"))

    items = [
        _item("Decision Count",               count,    None, "pipeline_output"),
        _item("Evidence Coverage",            evidence, "%",  "reference_free_grounding"),
        _item("Timestamp Coverage",           ts_cov,   "%",  "reference_free_grounding"),
        _item("Speaker Attribution",          speaker,  "%",  "reference_free_grounding"),
        _item("Avg Confidence",               avg_conf, None, "llm_output"),
        _item("Unsupported Decision Rate",     unsup,    "%",  "reference_free_grounding"),
        _item("Grounding Quality Score",       q_score,  None, "reference_free_composite"),
    ]

    return LiveMetricCard(
        metric_name="Decision Grounding",
        metric_key="decisions",
        status=MEASURED,
        primary_metric_label="Evidence Coverage",
        primary_metric_value=evidence,
        primary_metric_unit="%",
        display_items=[i for i in items if i.value is not None],
        source="reference_free_llm_grounding",
        execution_time_sec=None,
        notes="No reference decisions — showing grounding signals (NOT F1). Upload ground truth for Precision/Recall/F1.",
    )


# ---- Processing Performance ------------------------------------------------

def _build_processing_performance(meeting: Meeting) -> ProcessingPerformance:
    """
    Processing timing from runtime execution logs only.
    All values come from Meeting model timestamp columns populated by the
    pipeline. None means the timing was not recorded - never fabricated.
    """
    return ProcessingPerformance(
        upload_time_sec=_safe_float(getattr(meeting, "upload_duration_sec", None)),
        stt_time_sec=_safe_float(getattr(meeting, "stt_duration_sec", None)),
        speaker_time_sec=_safe_float(getattr(meeting, "speaker_duration_sec", None)),
        summary_time_sec=_safe_float(getattr(meeting, "summary_duration_sec", None)),
        mom_time_sec=_safe_float(getattr(meeting, "mom_duration_sec", None)),
        action_items_time_sec=_safe_float(getattr(meeting, "action_items_duration_sec", None)),
        total_processing_time_sec=_safe_float(
            getattr(meeting, "total_processing_time_sec", None)
            or getattr(meeting, "processing_time", None)
        ),
        api_time_sec=None,   # not yet instrumented separately
        db_time_sec=None,    # not yet instrumented separately
        file_size_bytes=_safe_int(getattr(meeting, "file_size_bytes", None)),
        video_duration_sec=_safe_float(getattr(meeting, "duration", None)),
    )


# ---- Public API ------------------------------------------------------------

def build_live_evaluation(meeting: Meeting, db: Session) -> LiveEvaluationResponse:
    """
    Run the Universal Evaluation Engine and assemble all canonical metric
    cards for the meeting.

    This is the single authoritative source for the evaluation dashboard.
    All values come from real execution; any metric that cannot be computed
    is reported as status='not_measured' with an explicit reason.
    """
    try:
        report = evaluation_engine.evaluate_meeting(
            meeting, db, performed_by="live_metrics", persist=True
        )
    except Exception as e:
        logger.error("Evaluation engine failed for meeting %s: %s", meeting.id, e, exc_info=True)
        report = {"tasks": {}, "overall": None, "has_reference": False}

    tasks: Dict[str, Any] = report.get("tasks", {})
    overall: Optional[Dict[str, Any]] = report.get("overall")

    metric_cards = [
        _build_stt_card(meeting, tasks),
        _build_speaker_card(meeting, tasks),
        _build_summary_card(meeting, tasks),
        _build_mom_card(meeting, tasks),
        _build_action_items_card(meeting, tasks, db),
        _build_topics_card(meeting, tasks),
        _build_sentiment_card(meeting, tasks),
        _build_similarity_card(meeting, tasks),
        _build_chatbot_card(meeting, db),
        # New cards — Component 1 (audio signal), 2a (agenda), 2b (decisions)
        _build_audio_card(meeting, tasks),
        _build_agenda_card(meeting, tasks),
        _build_decisions_card(meeting, tasks),
    ]

    return LiveEvaluationResponse(
        meeting_id=meeting.id,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        engine_version=ENGINE_VERSION,
        has_reference=bool(report.get("has_reference")),
        metrics=metric_cards,
        processing_performance=_build_processing_performance(meeting),
        overall_score=overall.get("score") if overall else None,
        overall_status=overall.get("status") if overall else None,
        overall_reason=overall.get("reason") if overall else None,
    )


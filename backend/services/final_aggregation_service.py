"""Final Aggregation Service — builds and upserts AI_Meeting_Inteligent_final records.

This service is the single entry point for populating the consolidated final table.
It is called from the processing pipeline after all feature tables have been populated.

Design rules
------------
- ONE row per meeting: upsert prevents duplicates.
- Compute first, write last: nothing is flushed until the record is fully assembled, so no row lock is held while
  it is computed, and an unchanged record writes nothing (see ``build_final_record``).
- The pipeline treats a failure here as a failure of the meeting (it is retried once, then the meeting is marked
  ``error``): a meeting is never ``completed`` without its final JSON.
- No LLM calls: reads only from DB; all AI results must already be persisted.
- No hardcoding: all data is read dynamically from the source tables.
- Validates each feature before writing to the final table.
- Distinguishes reference-based from reference-free metrics.
- Activity logs: only the 18 defined feature steps, filtered and ordered.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from backend.utils.db_safe import has_ground_truth as _db_has_ground_truth
from backend.utils.db_safe import load_ground_truth, optional_db

if TYPE_CHECKING:  # annotation-only; the runtime import happens lazily inside the functions
    from backend.models.models import AIMeetingIntelFinal

logger = logging.getLogger(__name__)

# ── Aggregation version — bump when schema changes ─────────────────────────────
AGGREGATION_VERSION = "1.0"

# ── Step names considered "important" for the final activity log ───────────────
# These map to the 18-feature logical order defined in activity_logger.FEATURE_ORDER.
# All other steps (debug/internal) are excluded from the final table.
_IMPORTANT_STEPS = frozenset({
    "meeting_upload", "input_processing", "video_processing", "document_text_extraction",
    "audio_extraction", "audio_processing", "audio_preprocessing", "voice_activity_detection",
    "document_metadata_extraction",
    "transcription", "speech_to_text",
    "speaker_diarization", "speaker_identification",
    "speaker_duration", "speaker_timeline",
    "transcript_correction", "speaker_mapping", "transcript_save", "document_segments_creation",
    "agenda", "agenda_extraction",
    "decision", "decision_extraction",
    "decision_mapping", "agenda_decision_mapping",
    "action_items", "action_item_extraction",
    "summary", "summary_generation",
    "mom", "mom_generation",
    "sentiment", "sentiment_analysis",
    "metrics", "ai_metrics", "topic_detection",
    "rag", "embedding_generation", "vector_indexing",
    "ai_chatbot", "chatbot_indexing", "similar_meeting_search",
    "database", "data_persistence", "db_save",
    "final_validation", "final_output",
})

# Placeholder decision texts that should NOT be stored as real decisions
_NO_DECISION_PHRASES = frozenset({
    "no formal decision recorded.", "no formal decision recorded",
    "no decision recorded.", "no decision recorded",
    "no decision.", "no decision", "none", "n/a", "",
})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_json_loads(value: Optional[str], fallback: Any = None) -> Any:
    """Safely parse a JSON string; return fallback on any error."""
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _safe_json_dumps(value: Any) -> Optional[str]:
    """Safely serialize value to JSON string; return None on error."""
    if value is None:
        return None
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        return None


def _iso(value: Any) -> Optional[str]:
    """Return an ISO-8601 string for a datetime/date, the value unchanged if it is
    already a primitive, or None. Guarantees the result is JSON-serializable so it
    can be stored in the JSONB ``json_data`` column and returned by JSONResponse.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def _parse_maybe_json(value: Any, fallback: Any = None) -> Any:
    """Parse a value that may be a JSON string, an already-decoded object, or plain
    text. Returns ``fallback`` for empty/None input; returns the original string
    when it is genuine free text (not JSON). Never raises.
    """
    if value is None or value == "":
        return fallback
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s[:1] in ("[", "{"):
            try:
                return json.loads(s)
            except Exception:
                return fallback if fallback is not None else value
        return value
    return value


def _format_duration(seconds: Optional[float]) -> Optional[str]:
    """Convert duration in seconds to 'Xh Ym Zs' human-readable string."""
    if seconds is None or seconds < 0:
        return None
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ── Speaker duration computation from transcript_segments ─────────────────────

def _compute_speaker_duration(
    segments, total_duration_sec: Optional[float], speech_durations: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """Compute per-speaker duration from transcript segment start/end times.

    Uses actual utterance boundaries from transcript_segments.
    Does NOT assume a fixed speaker count — handles 2, 4, 6, N speakers dynamically.
    Handles overlapping segments by summing raw durations (matches diarization behaviour).

    ``speech_durations``: the speech-only talk time per speaker the diarization stored with the meeting. Segment spans
    include every pause inside a segment (22-39 % silence measured), so when it covers exactly the speakers of the
    segments it is used instead; otherwise (legacy meetings) the spans are summed as before.
    """
    durations: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for seg in segments:
        speaker = getattr(seg, "speaker", None) or "Unknown"
        try:
            start = float(seg.start_time or 0)
            end = float(seg.end_time or 0)
            dur = max(0.0, end - start)
        except (TypeError, ValueError):
            dur = 0.0
        durations[speaker] = durations.get(speaker, 0.0) + dur
        counts[speaker] = counts.get(speaker, 0) + 1

    if speech_durations and set(speech_durations) == set(durations):
        durations = {spk: max(0.0, float(speech_durations[spk])) for spk in durations}

    total_spoken = sum(durations.values())
    result = []
    for speaker, dur in sorted(durations.items()):
        pct = round((dur / total_spoken * 100), 1) if total_spoken > 0 else 0.0
        result.append({
            "speaker_id": speaker,
            "duration_seconds": round(dur, 2),
            "percentage": pct,
            "segment_count": counts.get(speaker, 0),
        })

    return sorted(result, key=lambda x: x["duration_seconds"], reverse=True)


# ── Transcript builder ─────────────────────────────────────────────────────────

def _build_transcript(segments) -> List[Dict]:
    """Build a clean transcript array from transcript_segment ORM objects."""
    result = []
    for seg in segments:
        result.append({
            "speaker": getattr(seg, "speaker", "Unknown"),
            "start_time": getattr(seg, "start_time", None),
            "end_time": getattr(seg, "end_time", None),
            "text": getattr(seg, "text", "") or "",
            "confidence": getattr(seg, "stt_confidence", None),
        })
    return result


# ── Speaker summary builder ────────────────────────────────────────────────────

def _build_speakers(segments) -> List[Dict]:
    """Build a speaker summary list. Dynamic — handles any number of speakers."""
    speaker_data: Dict[str, Dict] = {}
    for seg in segments:
        sp = getattr(seg, "speaker", None) or "Unknown"
        if sp not in speaker_data:
            speaker_data[sp] = {"segment_count": 0, "word_count": 0}
        speaker_data[sp]["segment_count"] += 1
        text = getattr(seg, "text", "") or ""
        speaker_data[sp]["word_count"] += len(text.split())

    result = []
    for sp_id, info in sorted(speaker_data.items()):
        result.append({
            "speaker_id": sp_id,
            "label": sp_id,
            "segment_count": info["segment_count"],
            "total_words": info["word_count"],
        })
    return result


# ── Agenda builder ─────────────────────────────────────────────────────────────

def _build_agenda(akd_rows) -> List[Dict]:
    """Build agenda list from agenda_key_decisions rows. Dynamic per meeting."""
    result = []
    for akd in akd_rows:
        ag_id = getattr(akd, "agenda_id_str", None) or f"A{getattr(akd, 'agenda_order', 1)}"
        result.append({
            "agenda_id": ag_id,
            "title": getattr(akd, "agenda", "") or "",
            "type": getattr(akd, "agenda_type", "inferred") or "inferred",
            "order": getattr(akd, "agenda_order", 1),
            "source": getattr(akd, "agenda_source", "llm_detected") or "llm_detected",
        })
    return result


# ── Decisions builder ──────────────────────────────────────────────────────────

def _build_decisions(akd_rows) -> List[Dict]:
    """Build decisions list. Only includes ACTUAL, grounded decisions — filters
    placeholder text AND statements that never cleared the decisive-language/
    evidence bar (discussions, questions, suggestions, proposals are not
    decisions; see llm_workflow.validate_and_format_agenda_key_decisions).
    """
    result = []
    for akd in akd_rows:
        dec_text = (getattr(akd, "key_decision", "") or "").strip()
        if not dec_text or dec_text.lower() in _NO_DECISION_PHRASES:
            continue  # skip non-decisions
        if not bool(getattr(akd, "decision_grounded", False)):
            continue  # skip discussion/suggestion/question text that isn't a confirmed decision
        ag_id = getattr(akd, "agenda_id_str", None) or f"A{getattr(akd, 'agenda_order', 1)}"
        item = {
            "agenda_id": ag_id,
            "decision": dec_text,
            "reason": getattr(akd, "reason", None),
            "evidence": getattr(akd, "evidence", None),
            "speaker": getattr(akd, "speaker", None),
            "timestamp": getattr(akd, "timestamp", None),
            "confidence": getattr(akd, "confidence", None),
            "match_score": getattr(akd, "match_score", None),
        }
        # Only valid key/value data should be shown — no null keys or null/empty values
        clean_item = {
            k: v for k, v in item.items()
            if v is not None and (not isinstance(v, str) or v.strip() != "")
        }
        result.append(clean_item)
    return result


# ── Agenda → Decision mapping builder ─────────────────────────────────────────

def _build_agenda_decision_mapping(akd_rows) -> List[Dict]:
    """Build the semantic agenda-to-decision mapping.

    Uses the stored match_score and agenda_id_str from agenda_key_decisions —
    NOT a positional mapping.  Each agenda topic maps to its actual decision
    (if any) based on the sentence-transformer similarity score already persisted.

    A decision is mapped to its agenda item when it is a GROUNDED decision:
    the extraction paired it with this agenda topic and quoted the words in
    the transcript that state it (llm_workflow verifies the quote occurs in
    the transcript and carries decision language). That pairing is what the
    mapping reports; the stored match_score is shown next to it as the
    embedding similarity of the two texts, not used to veto it — a short
    title ("Migration timeline") and the decision it produced ("Production
    deployment moved to October 15") are rarely close in embedding space even
    when the pairing is exactly right. An agenda item without a grounded
    decision is emitted with has_decision=False (never a forced mapping).
    """
    from backend.services.agenda_service import confidence_label, is_reliable_match

    result = []
    for akd in akd_rows:
        ag_id = getattr(akd, "agenda_id_str", None) or f"A{getattr(akd, 'agenda_order', 1)}"
        dec_text = (getattr(akd, "key_decision", "") or "").strip()
        is_no_dec = not dec_text or dec_text.lower() in _NO_DECISION_PHRASES
        match_score = getattr(akd, "match_score", None)
        is_grounded = bool(getattr(akd, "decision_grounded", False))

        # Mapped only when it is itself a confirmed decision (not a discussion,
        # question, task or suggestion) — see the docstring for why the
        # similarity score no longer vetoes it.
        reliably_mapped = (not is_no_dec) and is_grounded

        # Confidence label from the stored match_score through the ONE shared bucket function (this used to repeat
        # its 95/85/70 thresholds); below the "Fair Match" bar no label is claimed (the item is still mapped).
        confidence_level = confidence_label(match_score) if reliably_mapped and is_reliable_match(match_score) else None

        item = {
            "agenda_id": ag_id,
            "agenda_title": getattr(akd, "agenda", "") or "",
            "decision": dec_text if reliably_mapped else None,
            "has_decision": reliably_mapped,
            "match_score": match_score if reliably_mapped else None,
            "confidence_level": confidence_level,
            "is_grounded": is_grounded and reliably_mapped,
            # What the mapping rests on: the transcript quote (verified by the extraction), who said it, and when.
            "evidence": getattr(akd, "evidence", None) if reliably_mapped else None,
            "speaker": getattr(akd, "speaker", None) if reliably_mapped else None,
            "timestamp": getattr(akd, "timestamp", None) if reliably_mapped else None,
        }
        clean_item = {
            k: v for k, v in item.items()
            if v is not None and (not isinstance(v, str) or v.strip() != "")
        }
        result.append(clean_item)
    return result


# ── Action items builder ───────────────────────────────────────────────────────

# Placeholder values that mean "no real data" — never surfaced as owners/deadlines (§18).
_ACTION_ITEM_PLACEHOLDERS = frozenset({
    "", "unassigned", "not specified", "not_specified", "n/a", "na", "none",
    "tbd", "to be decided", "unknown", "unspecified",
})


def _clean_ai_value(value: Any) -> Any:
    """Return the value, or None if it is empty / a known placeholder string."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in _ACTION_ITEM_PLACEHOLDERS:
            return None
        return value.strip()
    return value


def _build_action_items(ai_rows) -> List[Dict]:
    """Build validated action items array.

    Dynamic — handles any number of items. Only fields backed by real data are
    emitted; null / placeholder ("Unassigned", "Not specified", …) keys are
    dropped from the display structure so the frontend never renders fake
    owners or deadlines (§18). `task`, `status` and `item_order` are always kept.
    """
    result = []
    for ai in ai_rows:
        task = getattr(ai, "task", None)
        if not task or not str(task).strip():
            continue  # task is mandatory; skip invalid rows

        item: Dict[str, Any] = {
            "item_order": getattr(ai, "item_order", None),
            "task": str(task).strip(),
            "status": getattr(ai, "status", None) or "open",
        }
        optional = {
            "description": _clean_ai_value(getattr(ai, "description", None)),
            "owner": _clean_ai_value(getattr(ai, "owner", None)),
            "deadline": _clean_ai_value(getattr(ai, "deadline", None)),
            "deadline_type": _clean_ai_value(getattr(ai, "deadline_type", None)),
            "due_date": str(ai.due_date) if getattr(ai, "due_date", None) else None,
            "priority": _clean_ai_value(getattr(ai, "priority", None)),
            "evidence": _clean_ai_value(getattr(ai, "evidence", None)),
            "speaker": _clean_ai_value(getattr(ai, "speaker", None)),
            "timestamp_start": getattr(ai, "timestamp_start", None),
            "timestamp_end": getattr(ai, "timestamp_end", None),
            "confidence": getattr(ai, "confidence", None),
            "agenda_ref": _clean_ai_value(getattr(ai, "agenda_ref", None)),
            "decision_ref": _clean_ai_value(getattr(ai, "decision_ref", None)),
        }
        item.update({k: v for k, v in optional.items() if v is not None})
        result.append(item)
    return result


# ── Sentiment builder ──────────────────────────────────────────────────────────

def _build_sentiment(meeting) -> Dict:
    """Build sentiment object from meetings table fields.

    Preserves speaker-level breakdown when it exists in sentiment_breakdown_json.
    Never generates arbitrary sentiment — only packages what the pipeline stored.
    """
    label = getattr(meeting, "sentiment_label", None) or getattr(meeting, "sentiment", "Neutral")
    # meeting.emotions is stored as a JSON string (e.g. '{"Joy": 30.0}'). Decode it
    # so the final payload carries a real object, not a JSON string inside JSON.
    emotions_raw = getattr(meeting, "emotions", None) or ""
    if isinstance(emotions_raw, str) and emotions_raw.strip().startswith("{"):
        emotions_parsed = _safe_json_loads(emotions_raw, emotions_raw)
    else:
        emotions_parsed = emotions_raw
    breakdown_raw = _safe_json_loads(getattr(meeting, "sentiment_breakdown_json", None), {})

    # Extract speaker-level breakdown if stored
    speaker_breakdown = []
    if isinstance(breakdown_raw, dict):
        # Check for per-speaker keys like {"Speaker_1": {"sentiment": ..., "confidence": ...}}
        for key, val in breakdown_raw.items():
            if key not in ("sentiment", "emotions") and isinstance(val, dict):
                speaker_breakdown.append({
                    "speaker_id": key,
                    "sentiment": val.get("sentiment"),
                    "confidence": val.get("confidence"),
                    "emotions": val.get("emotions"),
                })

    return {
        "label": label,
        "emotions": emotions_parsed,
        "speaker_breakdown": speaker_breakdown,
        "methodology": "pipeline_llm" if not speaker_breakdown else "per_speaker_llm",
    }


# ── Metrics builder ────────────────────────────────────────────────────────────

def _build_metrics(meeting, db, quality_metrics: Optional[Dict] = None) -> Dict:
    """Build a metrics object with three clearly-separated, non-overlapping groups.

    accuracy         -> TRUE accuracy vs a ground-truth reference (WER/DER/F1/ROUGE).
                        ``value`` is null unless ground truth exists. NEVER estimated.
    reference_based  -> the full per-task ground-truth metric breakdown (null w/o GT).
    reference_free   -> quality / grounding SIGNALS that need no reference. These are
                        NOT accuracy and are labelled as such.
    quality_scores   -> per-feature quality/grounding scores (summary_quality, ...).
                        Explicitly not accuracy.
    technical        -> audio/video telemetry.

    RULE: a metric that is not measured is ``null`` — never 0.0, never a guess.

    ``quality_metrics`` is the real-time reference-free block (transcript quality, summary quality,
    RAG groundedness — backend.evaluation.quality_scores). Its keys are merged in at the top level so
    this object has the SAME ``metrics`` shape as GET /meetings/{id}.
    """
    # Ground truth is optional: the lookup is savepoint-guarded (db_safe), so a failed query answers "none" and the
    # final record is still built instead of the transaction being left aborted for every later step.
    _gt_flag = getattr(meeting, "has_ground_truth", None)
    has_ground_truth = bool(_gt_flag) if _gt_flag is not None else _db_has_ground_truth(db, meeting.id)
    # meeting.accuracy is written ONLY by the ground-truth evaluation engine.
    accuracy_val = getattr(meeting, "accuracy", None) if has_ground_truth else None

    # Reference-free quality SIGNAL produced by the pipeline (never accuracy).
    quality_signal = None
    try:
        mj = _safe_json_loads(getattr(meeting, "metrics_json", None), {})
        if isinstance(mj, dict) and mj.get("metric_type") == "RUNTIME_QUALITY_SIGNAL":
            quality_signal = {
                # accept the legacy key name from rows written before the rename
                "score": mj.get("quality_signal_score", mj.get("overall_score")),
                "label": mj.get("label"),
                "is_accuracy": False,
                "components": mj.get("components"),
                "weights": mj.get("weights"),
                "note": mj.get("note"),
            }
    except Exception:
        quality_signal = None

    result: Dict[str, Any] = {
        "accuracy": {
            "value": accuracy_val,
            "basis": "ground_truth" if has_ground_truth else "not_measured",
            "has_ground_truth": has_ground_truth,
            "note": (
                "True accuracy measured against a human ground-truth reference "
                "(WER / DER / precision / recall / F1 / ROUGE)."
                if has_ground_truth
                else "No ground-truth reference available — true accuracy cannot be "
                     "computed. See metrics.reference_free.quality_signal for a "
                     "reference-free quality estimate (NOT accuracy)."
            ),
        },
        "reference_based": None,    # ground-truth metric breakdown — null unless GT exists
        "reference_free": {"quality_signal": quality_signal},
        "quality_scores": {},
        "technical": {},
    }

    # ── Reference-free quality scores from meetings table ─────────────────────
    raw_scores = {
        "summary_quality": getattr(meeting, "summary_quality", None),
        "mom_quality": getattr(meeting, "mom_quality", None),
        "action_item_quality": getattr(meeting, "action_item_quality", None),
        "topic_quality": getattr(meeting, "topic_quality", None),
        "confidence_score": getattr(meeting, "confidence_score", None),
        "confidence_score_meaning": "ROUGE-L grounding of the summary against the transcript (not accuracy).",
        "audio_quality": getattr(meeting, "audio_quality", None),
        "video_quality": getattr(meeting, "video_quality", None),
    }

    # Enrich from metrics_json if present
    if hasattr(meeting, "metrics_json") and meeting.metrics_json:
        try:
            m_data = json.loads(meeting.metrics_json) if isinstance(meeting.metrics_json, str) else meeting.metrics_json
            if isinstance(m_data, dict):
                comps = m_data.get("components", {})
                for k, v in comps.items():
                    target_k = f"{k}_quality" if not k.endswith("_quality") and k not in ("accuracy", "transcription", "speaker") else k
                    if raw_scores.get(target_k) is None and v is not None:
                        raw_scores[target_k] = v
                if "weights" in m_data:
                    result["technical"]["weights"] = m_data["weights"]
                if "contributions" in m_data:
                    result["technical"]["contributions"] = m_data["contributions"]
        except Exception:
            pass

    result["quality_scores"] = {k: v for k, v in raw_scores.items() if v is not None}

    # ── Reference-free metrics from meeting_evaluations ───────────────────────
    try:
        from backend.models.models import MeetingEvaluation
        me = optional_db(
            db,
            lambda: db.query(MeetingEvaluation).filter(MeetingEvaluation.meeting_id == meeting.id).first(),
            label=f"meeting_evaluations lookup for meeting {meeting.id}",
        )
        if me:
            result["reference_free"] = {
                "source": "meeting_evaluations",
                "note": "Reference-free quality signals — no ground truth required",
                "speech_recognition_quality": me.speech_recognition_quality,
                "speaker_consistency": me.speaker_consistency,
                "summary_grounding": me.summary_grounding,
                "mom_quality": me.mom_quality,
                "action_item_quality": me.action_item_quality,
                "topic_coverage": me.topic_coverage,
                "sentiment_confidence": me.sentiment_confidence,
                "overall_ai_quality": me.overall_ai_quality,
            }
    except Exception as e:
        logger.debug(f"[FinalAggregation] Could not load meeting_evaluations: {e}")

    # ── Reference-based metrics from evaluation_results (only if GT exists) ───
    try:
        from backend.models.models import EvaluationResult, GroundTruth
        gt = load_ground_truth(db, meeting.id)
        if gt:
            er = optional_db(
                db,
                lambda: db.query(EvaluationResult).filter(EvaluationResult.meeting_id == meeting.id).first(),
                label=f"evaluation_results lookup for meeting {meeting.id}",
            )
            if er:
                result["reference_based"] = {
                    "source": "evaluation_results",
                    "note": "Ground-truth-based metrics — computed against human annotations",
                    "ground_truth_version": gt.version,
                    "units": {
                        "wer": "error rate 0-1 (lower is better)",
                        "cer": "error rate 0-1 (lower is better)",
                        "der": "error rate 0-1 (lower is better)",
                        "jer": "error rate 0-1 (lower is better)",
                        "_others": "percent 0-100 (higher is better)",
                    },
                    "wer": er.wer,
                    "cer": er.cer,
                    "stt_accuracy": er.stt_accuracy,
                    "der": er.der,
                    "jer": er.jer,
                    "diarization_accuracy": er.diarization_accuracy,
                    "speaker_accuracy": er.speaker_accuracy,
                    "speaker_f1": er.speaker_f1,
                    "rouge1": er.rouge1,
                    "rouge2": er.rouge2,
                    "rougeL": er.rougeL,
                    "bertscore_f1": er.bertscore_f1,
                    "summary_accuracy": er.summary_accuracy,
                    "mom_coverage": er.mom_coverage,
                    "mom_accuracy": er.mom_accuracy,
                    "action_precision": er.action_precision,
                    "action_recall": er.action_recall,
                    "action_f1": er.action_f1,
                    "action_accuracy": er.action_accuracy,
                    "topic_accuracy": er.topic_accuracy,
                    "topic_f1": er.topic_f1,
                    "sentiment_accuracy": er.sentiment_accuracy,
                    "sentiment_f1": er.sentiment_f1,
                    "overall_accuracy": er.overall_accuracy,
                    "overall_system_score": er.overall_system_score,
                }
    except Exception as e:
        logger.debug(f"[FinalAggregation] Could not load evaluation_results: {e}")

    # ── Technical benchmarks from benchmark_evaluations ───────────────────────
    try:
        from backend.models.models import BenchmarkEvaluation
        be = optional_db(
            db,
            lambda: db.query(BenchmarkEvaluation).filter(BenchmarkEvaluation.meeting_id == meeting.id).first(),
            label=f"benchmark_evaluations lookup for meeting {meeting.id}",
        )
        if be:
            result["technical"] = {
                "source": "benchmark_evaluations",
                "audio_snr": be.audio_snr,
                "audio_silence_ratio": be.audio_silence_ratio,
                "video_fps": be.video_fps,
                "video_resolution": be.video_resolution,
                "video_bitrate": be.video_bitrate,
                "chatbot_faithfulness": be.chatbot_faithfulness,
                "chatbot_answer_relevance": be.chatbot_answer_relevance,
                "similarity_ndcg": be.similarity_ndcg,
            }
    except Exception as e:
        logger.debug(f"[FinalAggregation] Could not load benchmark_evaluations: {e}")

    if isinstance(quality_metrics, dict):
        for key in ("evaluation_type", "ground_truth_available", "transcript_quality",
                    "summary_quality", "rag_groundedness", "method_version"):
            if key in quality_metrics:
                result[key] = quality_metrics[key]
        # One value per metric: the older per-feature "summary_quality" number (a ROUGE-style signal
        # from the pipeline) is replaced by the new reference-free score so this object never carries
        # two different summary-quality values.
        sq = (quality_metrics.get("summary_quality") or {}).get("score")
        if "summary_quality" in result.get("quality_scores", {}):
            if sq is not None:
                result["quality_scores"]["summary_quality"] = sq
            else:
                result["quality_scores"].pop("summary_quality", None)

    return result


# ── Activity logs builder ──────────────────────────────────────────────────────

def _build_activity_logs(db, meeting_pk: int) -> List[Dict]:
    """Build the filtered, ordered activity log for the final table.

    Rules:
    - Only important feature-level steps (step_order 1-18, step_name in _IMPORTANT_STEPS)
    - Ordered by step_order ASC, then started_at ASC
    - Per step_name, keeps only the LATEST run (most recent status)
    - Never mixes logs from different meetings (filtered by meeting_pk)
    """
    try:
        from backend.models.models import MeetingActivityLog
        from backend.services.activity_logger import FEATURE_ORDER
        # Savepoint-guarded: a failing read here must not leave the transaction aborted for build_final_record's
        # remaining steps and its upsert.
        all_logs = optional_db(
            db,
            lambda: (
                db.query(MeetingActivityLog)
                .filter(MeetingActivityLog.meeting_pk == meeting_pk)
                .order_by(MeetingActivityLog.created_at.asc())
                .all()
            ),
            default=None,
            label=f"activity-log load for meeting {meeting_pk}",
        )
    except Exception as e:
        logger.warning(f"[FinalAggregation] Could not load activity logs: {e}")
        return []
    if all_logs is None:
        return []

    if not all_logs:
        return []

    # ── Restrict to the MOST RECENT processing run ────────────────────────────
    # A meeting that was reprocessed has activity rows from several
    # processing_id runs, each possibly written under an older step-order
    # catalogue. Mixing them produces duplicate step_order values and stale
    # timestamps. Keep only the latest run.
    latest_proc_id = None
    latest_ts = None
    for log in all_logs:
        pid = getattr(log, "processing_id", None)
        ts = log.created_at or datetime.min
        if pid and (latest_ts is None or ts > latest_ts):
            latest_ts = ts
            latest_proc_id = pid
    if latest_proc_id:
        run_logs = [l for l in all_logs if getattr(l, "processing_id", None) == latest_proc_id]
    else:
        run_logs = all_logs

    # Filter to important feature-level steps only
    filtered = [
        log for log in run_logs
        if (log.step_name or "").lower() in _IMPORTANT_STEPS
    ]

    def _priority(status: str) -> int:
        return {"COMPLETED": 4, "FAILED": 3, "SKIPPED": 2, "PROCESSING": 1}.get(
            (status or "").upper(), 0
        )

    # Deduplicate: one row per step_name, keeping the best status / latest write.
    step_best: Dict[str, Any] = {}
    for log in filtered:
        name = log.step_name
        existing = step_best.get(name)
        if existing is None:
            step_best[name] = log
            continue
        if _priority(log.status) > _priority(existing.status):
            step_best[name] = log
        elif (
            _priority(log.status) == _priority(existing.status)
            and (log.created_at or datetime.min) > (existing.created_at or datetime.min)
        ):
            step_best[name] = log

    # Re-derive step_order from the CURRENT catalogue (ignore any stale stored
    # value) and collapse to one entry per feature slot (1-18).
    feature_best: Dict[int, Any] = {}
    for name, log in step_best.items():
        order = FEATURE_ORDER.get((name or "").lower(), getattr(log, "step_order", None) or 99)
        cur = feature_best.get(order)
        if cur is None:
            feature_best[order] = (log, order)
        else:
            cur_log, _ = cur
            if _priority(log.status) > _priority(cur_log.status) or (
                _priority(log.status) == _priority(cur_log.status)
                and (log.created_at or datetime.min) > (cur_log.created_at or datetime.min)
            ):
                feature_best[order] = (log, order)

    result = []
    for order in sorted(feature_best):
        log, canonical_order = feature_best[order]
        result.append({
            "step_name": log.step_name,
            "step_order": canonical_order,
            "status": log.status,
            "message": log.message,
            "started_at": log.started_at.isoformat() if log.started_at else None,
            "completed_at": log.completed_at.isoformat() if log.completed_at else None,
            "duration_seconds": log.duration_seconds,
        })

    return result


# ── Self-healing schema guard ─────────────────────────────────────────────────

# Set once per process after the columns are confirmed/created, so the check is
# a single inspector call on the very first aggregation and a no-op thereafter.
_FINAL_SCHEMA_VERIFIED = False


def _ensure_final_table_schema(db) -> None:
    """Guarantee every column of the AIMeetingIntelFinal model exists on the live
    ``AI_Meeting_Inteligent_final`` table, adding any that are missing.

    Why this exists
    ---------------
    ``json_data`` (and ``error_info``) were added to the model after the table
    was first created on some databases. The startup auto-migration in
    ``backend/database/connection.py`` only runs when ``init_db()`` completes; if
    the process was crash-looping (or the table pre-dates the column) the Server
    DB can be left without ``json_data`` — every write then fails with
    ``UndefinedColumn`` and the row (or just the column) is silently rolled back,
    which is exactly the "API has data but the DB column is null/empty" symptom.

    A fresh ``create_all()`` DB (local/dev/SQLite) always has the column, so this
    only ever does work on an out-of-date Server DB. Idempotent, dynamic (reads
    the model, no hardcoded column list), and cheap (runs once per process).
    """
    global _FINAL_SCHEMA_VERIFIED
    if _FINAL_SCHEMA_VERIFIED:
        return

    from sqlalchemy import inspect as _sa_inspect, text as _sa_text
    from backend.models.models import AIMeetingIntelFinal

    table = AIMeetingIntelFinal.__table__
    try:
        bind = db.get_bind()
    except Exception:
        bind = getattr(db, "bind", None)
    if bind is None:
        return
    engine = getattr(bind, "engine", bind)

    try:
        insp = _sa_inspect(engine)
        if not insp.has_table(table.name):
            # Table itself is missing — create just this one from the model.
            table.create(bind=engine, checkfirst=True)
            _FINAL_SCHEMA_VERIFIED = True
            logger.warning(
                f"[FinalAggregation] Created missing table '{table.name}' from the ORM model."
            )
            return

        live_cols = {c["name"] for c in insp.get_columns(table.name)}
        missing = [c for c in table.columns if c.name not in live_cols]
        if not missing:
            _FINAL_SCHEMA_VERIFIED = True
            return

        dialect = engine.dialect
        for col in missing:
            col_type = col.type.compile(dialect)
            ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {col_type}'
            try:
                with engine.begin() as ddl_conn:   # isolated txn, own commit
                    ddl_conn.execute(_sa_text(ddl))
                logger.warning(
                    f"[FinalAggregation] Schema heal: added missing column "
                    f"'{table.name}.{col.name}' ({col_type})."
                )
            except Exception as add_err:
                # Another worker may have added it in parallel — re-check before
                # deciding it is a real failure.
                try:
                    live_cols = {c["name"] for c in _sa_inspect(engine).get_columns(table.name)}
                except Exception:
                    live_cols = set()
                if col.name not in live_cols:
                    logger.error(
                        f"[FinalAggregation] Could NOT add column "
                        f"'{table.name}.{col.name}': {add_err}. Writes to this "
                        f"column will not persist until the schema is fixed.",
                        exc_info=True,
                    )
        _FINAL_SCHEMA_VERIFIED = True
    except Exception as guard_err:
        # Never let the guard itself break aggregation.
        logger.warning(f"[FinalAggregation] Schema guard skipped: {guard_err}")


# ── Main aggregation function ──────────────────────────────────────────────────

def _upsert_final_row(db, meeting, meeting_id: int, values: Dict[str, Any]) -> AIMeetingIntelFinal:
    """Write ``values`` to the meeting's ``AI_Meeting_Inteligent_final`` row: INSERT it, or UPDATE what changed.

    meeting_pk has a UNIQUE constraint, so a concurrent INSERT of the same row (a GET that lands while the pipeline
    finalises, or two GETs) is possible. The INSERT runs in a SAVEPOINT: on an IntegrityError the row the other
    transaction committed is re-read and updated instead - never a 500, never a lost aggregation.

    Only attributes whose value differs are UPDATEd (SQLAlchemy compares against the loaded value), so re-aggregating
    an unchanged meeting emits NO statement at all - and therefore takes no row lock and rewrites no JSON.
    """
    from backend.models.models import AIMeetingIntelFinal, Meeting

    def _find_existing():
        # Strict lookup first: meeting_pk is the unique key for this table.
        row = db.query(AIMeetingIntelFinal).filter(
            AIMeetingIntelFinal.meeting_pk == meeting_id
        ).first()
        if row is not None:
            return row
        # Fall back to the denormalized meeting_id string ONLY when no row
        # owns this meeting_pk yet — e.g. a legacy record written before
        # meeting_pk existed. Never re-parent a row whose meeting_pk already
        # points at a DIFFERENT, still-existing meeting — that would silently
        # steal another meeting's final record.
        candidate = db.query(AIMeetingIntelFinal).filter(
            (AIMeetingIntelFinal.meeting_id == str(meeting.meeting_id))
            & (AIMeetingIntelFinal.meeting_id.isnot(None))
        ).first()
        if candidate is not None and candidate.meeting_pk is not None and candidate.meeting_pk != meeting_id:
            other_meeting_alive = db.query(Meeting.id).filter(Meeting.id == candidate.meeting_pk).first()
            if other_meeting_alive is not None:
                logger.warning(
                    f"[FinalAggregation] meeting_id string '{meeting.meeting_id}' matches an "
                    f"AI_Meeting_Inteligent_final row already owned by a DIFFERENT live meeting "
                    f"(meeting_pk={candidate.meeting_pk}, requested={meeting_id}). Refusing to "
                    f"re-parent it; creating a fresh record for meeting_id={meeting_id} instead."
                )
                return None
        return candidate

    existing = _find_existing()
    if existing is None:
        existing = AIMeetingIntelFinal(meeting_pk=meeting_id, **values)
        db.add(existing)
        try:
            with db.begin_nested():
                db.flush()  # emit the INSERT inside a savepoint
            return existing
        except Exception as insert_race:
            # Someone else inserted it first — roll back just the failed INSERT and update their row.
            db.expunge(existing)
            existing = _find_existing()
            if existing is None:
                raise
            logger.info(
                f"[FinalAggregation] Concurrent insert for meeting {meeting_id} "
                f"resolved by updating the existing final record ({insert_race})."
            )
    existing.meeting_pk = meeting_id
    for column, value in values.items():
        setattr(existing, column, value)
    db.flush()
    return existing


def build_final_record(
    meeting_id: int, db, *, meeting_updates: Optional[Dict[str, Any]] = None
) -> AIMeetingIntelFinal:
    """Read all source tables, validate, normalize, and upsert into AI_Meeting_Inteligent_final.

    Write discipline - why the function is shaped this way
    -----------------------------------------------------
    1. The embedding-heavy metrics are computed FIRST, with the read transaction ended (see
       ``precompute_final_metrics``): no connection is held while text is embedded.
    2. Everything else is read and assembled in memory. NOTHING is written or flushed until the very end, and then
       only once (``_upsert_final_row``). The previous version flushed a ``processing_status='BUILDING'`` placeholder
       FIRST - taking a row lock on the meeting's final row - and then computed for seconds before committing. A
       GET poll that landed then held the lock for 4.3 s and the pipeline's own UPDATE of the same row waited 4.15 s
       behind it (measured, pg_stat_activity: Lock/transactionid, blockers=1).
    3. An unchanged record produces no UPDATE.

    Parameters
    ----------
    meeting_id : int
        The meetings.id (integer PK) of the meeting to aggregate.
    db : Session
        An active SQLAlchemy Session. The caller owns the session lifecycle. Best called with no pending changes
        (then the read transaction can be released during the metric computation).
    meeting_updates : dict, optional
        Column values to apply to the meeting row IN THE SAME TRANSACTION as the final record (the pipeline passes
        ``status`` / ``overall_status`` / ``error_message`` here), so the meeting becomes ``completed`` at the same
        instant its final JSON exists - never before it. They are visible to the JSON built below.

    Returns
    -------
    AIMeetingIntelFinal
        The upserted final record (already added to session; caller must commit).

    Raises
    ------
    ValueError
        If the meeting_id does not exist in the meetings table.
    Exception
        Any other DB/validation error -- the caller is responsible for catching.
    """
    from backend.api.meetings import build_meeting_response_dict, precompute_final_metrics
    from backend.models.models import (
        ActionItem,
        AgendaKeyDecision,
        AnalysisResult,
        Meeting,
        TranscriptSegment,
    )

    logger.info(f"[FinalAggregation] Building final record for meeting_id={meeting_id}")

    # ── 0. Ensure the final table has every column this code writes ───────────
    # Heals a Server DB whose AI_Meeting_Inteligent_final predates json_data /
    # error_info. No-op after the first call and on any create_all() database.
    _ensure_final_table_schema(db)

    # ── 1. Load meeting ────────────────────────────────────────────────────────
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise ValueError(f"Meeting with id={meeting_id} does not exist.")

    errors: List[str] = []

    # ── 2. Heavy metrics first, with NO transaction open while text is embedded ─
    quality_metrics, evaluation_metrics, quality_error = precompute_final_metrics(meeting, db)
    if quality_error is not None:
        logger.warning(f"[FinalAggregation] Quality metrics error for meeting {meeting_id}: {quality_error}")
        errors.append(f"quality_metrics: {quality_error}")

    # In-memory only until the single write at the end (autoflush is off for every session in this application).
    if meeting_updates:
        for _column, _value in meeting_updates.items():
            setattr(meeting, _column, _value)

    # Column values of the final row, assembled below and written once.
    values: Dict[str, Any] = {
        "meeting_id": meeting.meeting_id,
        "bharatvc_meeting_id": meeting.bharatvc_meeting_id,
        "display_id": meeting.display_id,
        "aggregation_version": AGGREGATION_VERSION,
    }

    # ── 3. Meeting metadata ────────────────────────────────────────────────────
    values["title"] = meeting.title or ""
    ref_time = getattr(meeting, "upload_start_time", None) or getattr(meeting, "created_at", None) or _utcnow()
    values["meeting_date"] = meeting.meeting_date or ref_time.strftime("%Y-%m-%d")
    values["meeting_time"] = meeting.meeting_time or ref_time.strftime("%I:%M %p")
    values["duration_seconds"] = meeting.duration
    values["duration_formatted"] = _format_duration(meeting.duration)
    values["status"] = meeting.status
    values["participant_count"] = meeting.participant_count
    values["participants"] = meeting.participants
    values["speaker_count"] = meeting.speaker_count

    # ── 4. Transcript & Speakers ───────────────────────────────────────────────
    # Each READ below runs in its own SAVEPOINT. These steps catch their exception and carry on to the next one; with
    # the statement run bare, the first failing query aborted the PostgreSQL transaction and every following step
    # (and the final upsert) then failed with InFailedSqlTransaction, which buried the real error. Rolling back to
    # the savepoint keeps the transaction usable; the step's own error is still recorded with its true cause.
    try:
        with db.begin_nested():
            segments = (
                db.query(TranscriptSegment)
                .filter(TranscriptSegment.meeting_id == meeting_id)
                .order_by(TranscriptSegment.id.asc())
                .all()
            )
        values["transcript"] = _safe_json_dumps(_build_transcript(segments))
        values["speakers"] = _safe_json_dumps(_build_speakers(segments))
        from backend.services.diarization_service import stored_speech_durations

        values["speaker_duration"] = _safe_json_dumps(
            _compute_speaker_duration(
                segments, meeting.duration, stored_speech_durations(getattr(meeting, "diarization_info_json", None))
            )
        )
    except Exception as e:
        logger.warning(f"[FinalAggregation] Transcript/speakers error for meeting {meeting_id}: {e}")
        errors.append(f"transcript: {e}")

    # ── 5. Agenda + Decisions + Mapping ───────────────────────────────────────
    try:
        with db.begin_nested():
            akd_rows = (
                db.query(AgendaKeyDecision)
                .filter(AgendaKeyDecision.meeting_id == meeting_id)
                .order_by(AgendaKeyDecision.agenda_order.asc())
                .all()
            )
        values["agenda"] = _safe_json_dumps(_build_agenda(akd_rows))
        values["decisions"] = _safe_json_dumps(_build_decisions(akd_rows))
        values["agenda_decision_mapping"] = _safe_json_dumps(
            _build_agenda_decision_mapping(akd_rows)
        )
    except Exception as e:
        logger.warning(f"[FinalAggregation] Agenda/decisions error for meeting {meeting_id}: {e}")
        errors.append(f"agenda_decisions: {e}")

    # ── 6. Action Items ────────────────────────────────────────────────────────
    try:
        with db.begin_nested():
            ai_rows = (
                db.query(ActionItem)
                .filter(ActionItem.meeting_id == meeting_id)
                .order_by(ActionItem.item_order.asc())
                .all()
            )
        values["action_items"] = _safe_json_dumps(_build_action_items(ai_rows))
    except Exception as e:
        logger.warning(f"[FinalAggregation] Action items error for meeting {meeting_id}: {e}")
        errors.append(f"action_items: {e}")

    # ── 7. Summary & MOM ──────────────────────────────────────────────────────
    try:
        with db.begin_nested():
            analysis = (
                db.query(AnalysisResult)
                .filter(AnalysisResult.meeting_id == meeting_id)
                .first()
            )
        if analysis:
            values["summary"] = analysis.summary
            values["mom"] = analysis.mom
        else:
            errors.append("summary/mom: analysis_results not found")
    except Exception as e:
        logger.warning(f"[FinalAggregation] Summary/MOM error for meeting {meeting_id}: {e}")
        errors.append(f"summary_mom: {e}")

    # ── 8. Sentiment ──────────────────────────────────────────────────────────
    try:
        values["sentiment"] = _safe_json_dumps(_build_sentiment(meeting))
    except Exception as e:
        logger.warning(f"[FinalAggregation] Sentiment error for meeting {meeting_id}: {e}")
        errors.append(f"sentiment: {e}")

    # ── 9. Metrics ────────────────────────────────────────────────────────────
    # The three real-time reference-free scores were computed ONCE in step 2 (outside any transaction) and are
    # shared by the `metrics` column and json_data (so both carry identical numbers). A failure never blocks
    # aggregation.
    try:
        values["metrics"] = _safe_json_dumps(_build_metrics(meeting, db, quality_metrics))
    except Exception as e:
        logger.warning(f"[FinalAggregation] Metrics error for meeting {meeting_id}: {e}")
        errors.append(f"metrics: {e}")

    # ── 10. Activity Logs ─────────────────────────────────────────────────────
    try:
        values["activity_logs"] = _safe_json_dumps(
            _build_activity_logs(db, meeting_pk=meeting_id)
        )
    except Exception as e:
        logger.warning(f"[FinalAggregation] Activity logs error for meeting {meeting_id}: {e}")
        errors.append(f"activity_logs: {e}")

    # ── 11. json_data — the EXACT GET /api/v1/meetings/{id} response ─────────
    # REQUIREMENT: AI_Meeting_Inteligent_final.json_data must hold the complete
    # API response in the SAME structure the GET endpoint returns — the flat
    # `schemas.Meeting` shape: top-level scalar columns, the raw *_json string
    # columns, `segments` (list of TranscriptSegment dicts), `analysis`
    # (summary / mom / action_items / structured_action_items), and
    # `agenda_key_decisions` / `structured_agenda_decisions`.
    #
    # `build_meeting_response_dict()` in backend.api.meetings is the single source
    # of that shape and is what GET serves, so we store its output verbatim.
    # It is populated 100% dynamically from THIS meeting's processed rows —
    # nothing hardcoded, no meeting-id branches — so it works for every meeting.
    # Built BEFORE the status is decided: a failure here is an error of the record and must be part of it.
    try:
        # SAVEPOINT: building the response lazy-loads many relationships; a failed statement must roll back to here
        # (and be recorded as this step's error below) rather than abort the transaction the upsert still needs.
        with db.begin_nested():
            values["json_data"] = build_meeting_response_dict(
                meeting, db, quality_metrics=quality_metrics, evaluation_metrics=evaluation_metrics
            )
        logger.info(
            f"[FinalAggregation] json_data (schemas.Meeting API shape) built for "
            f"meeting {meeting_id} — {len(values['json_data'])} top-level keys."
        )
    except Exception as e:
        logger.error(
            f"[FinalAggregation] Could not build schemas.Meeting json_data for "
            f"meeting {meeting_id}: {e}",
            exc_info=True,
        )
        errors.append(f"json_data: {e}")
        # Never leave the column NULL — fall back to a minimal, JSON-safe dict
        # assembled from the identity/metadata already resolved above.
        try:
            values["json_data"] = {
                "id": meeting_id,
                "meeting_id": values.get("meeting_id"),
                "bharatvc_meeting_id": values.get("bharatvc_meeting_id"),
                "display_id": values.get("display_id"),
                "filename": getattr(meeting, "filename", None) or values.get("title") or "",
                "title": values.get("title"),
                "status": values.get("status"),
                "upload_time": _iso(getattr(meeting, "upload_time", None)),
                "duration": values.get("duration_seconds"),
                "participant_count": values.get("participant_count"),
                "participants": values.get("participants"),
                "speaker_count": values.get("speaker_count"),
                "summary": values.get("summary"),
                "mom": values.get("mom"),
                "segments": _parse_maybe_json(values.get("transcript"), []) or [],
                "analysis": {
                    "summary": values.get("summary"),
                    "mom": values.get("mom"),
                    "action_items": _parse_maybe_json(values.get("action_items"), []) or [],
                    "structured_action_items": _parse_maybe_json(values.get("action_items"), []) or [],
                },
                "agenda_key_decisions": _parse_maybe_json(values.get("agenda_decision_mapping"), []) or [],
                "structured_agenda_decisions": _parse_maybe_json(values.get("agenda_decision_mapping"), []) or [],
                "_json_data_note": "fallback shape — build_meeting_response_dict failed; see error_info",
            }
        except Exception as fallback_err:
            # Not swallowed: the record is then marked FAILED below (json_data is a critical part of it).
            logger.error(f"[FinalAggregation] Fallback json_data for meeting {meeting_id} failed too: {fallback_err}")

    # ── 12. Determine final processing_status ───────────────────────────────────
    # FAILED when a *critical* feature raised an exception (not merely "absent"): the transcript, the summary/MoM,
    # or the final JSON itself - a record whose json_data could not be built is not a COMPLETED one, even when the
    # fallback shape was stored. Otherwise it mirrors the meeting lifecycle so the frontend loading-state stays
    # honest: a meeting that is still processing must not be advertised as COMPLETED.
    has_transcript = bool(values.get("transcript") and values.get("transcript") != "[]")
    has_summary = bool((values.get("summary") or "").strip())
    meeting_status = (getattr(meeting, "status", "") or "").lower()

    critical_errors = [e for e in errors if any(
        kw in e for kw in ("transcript:", "summary_mom:", "json_data:")
    )]

    if critical_errors:
        values["processing_status"] = "FAILED"
        values["error_info"] = _safe_json_dumps({
            "stage": "aggregation",
            "errors": errors,
            "message": "One or more critical features failed to aggregate.",
        })
        logger.error(
            f"[FinalAggregation] FAILED for meeting {meeting_id}. Errors: {errors}"
        )
    else:
        if meeting_status == "completed":
            values["processing_status"] = "COMPLETED"
        elif meeting_status in ("error", "failed"):
            values["processing_status"] = "FAILED"
        else:
            # uploaded / processing / unknown — data captured so far is still
            # returned, but the record is not COMPLETED yet.
            values["processing_status"] = "BUILDING"

        if errors:
            values["error_info"] = _safe_json_dumps({
                "stage": "aggregation",
                "errors": errors,
                "message": "Aggregation completed with minor issues in non-critical features.",
            })
        else:
            values["error_info"] = None
        logger.info(
            f"[FinalAggregation] {values['processing_status']} for meeting {meeting_id} "
            f"(transcript={'yes' if has_transcript else 'no'}, "
            f"summary={'yes' if has_summary else 'no'})"
        )

    # ── 13. The one write ─────────────────────────────────────────────────────
    # (`updated_at` moves by itself - `onupdate` - and only when something actually changed.)
    return _upsert_final_row(db, meeting, meeting_id, values)
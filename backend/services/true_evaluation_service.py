"""True Evaluation Service — compatibility adapter over the Universal Evaluation Engine.

All metric computations live in ``backend.evaluation.metrics`` and the
orchestration lives in ``backend.evaluation.engine``. This module keeps the
legacy ``run_meeting_evaluation`` / ``compute_*`` API surface so existing
callers (API endpoints, pipelines, metrics service) keep working unchanged.
No computations are duplicated here; no values are ever fabricated.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.evaluation import metrics
from backend.evaluation.engine import evaluation_engine
from backend.models.models import (
    BenchmarkEvaluation,
    EvaluationHistory,
    EvaluationResult,
    GroundTruth,
    Meeting,
)
from backend.services.vector_service import vector_service

logger = logging.getLogger(__name__)

SEMANTIC_SIMILARITY_THRESHOLD = 0.70

HAS_BLEURT = False  # BLEURT is not part of the default toolchain; BERTScore-style F1 is used.


# ─── Adapter helper ───────────────────────────────────────────────────────────

def _embed_fn(text: str) -> List[float]:
    try:
        return vector_service.generate_embedding(text)
    except Exception as e:
        logger.warning("Embedding failed: %s", e)
        return []


# ─── Compatibility metric calculators ─────────────────────────────────────────

def compute_stt_accuracy(ref_transcript: str, gen_transcript: str) -> Dict[str, Any]:
    """WER / CER via the engine's metrics module (matches old keys)."""
    res = metrics.stt_metrics(ref_transcript, gen_transcript)
    wer = res["wer"]
    res["stt_accuracy"] = res["accuracy"]
    # Legacy match %: exact-word match rate against reference
    n_ref = len(metrics.tokenize_words(ref_transcript))
    correct = max(0, n_ref - (res["deletions"] or 0) - (res["substitutions"] or 0))
    res["stt_match_pct"] = round((correct / n_ref) * 100.0, 2) if n_ref else (
        100.0 if not metrics.tokenize_words(gen_transcript) else 0.0
    )
    return res


def compute_diarization_accuracy(
    ref_timeline: List[Dict[str, Any]], pred_timeline: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """DER / JER via the engine's metrics module (matches old keys)."""
    res = metrics.diarization_metrics(ref_timeline, pred_timeline)
    res["diarization_accuracy"] = res["accuracy"]
    return res


def compute_speaker_id_metrics(
    ref_labels: Dict[str, str], pred_labels: Dict[str, str]
) -> Dict[str, Any]:
    """Speaker ID classification metrics (matches old keys)."""
    res = metrics.classification_metrics(ref_labels, pred_labels)
    if res["f1"] is None:
        return {
            "speaker_accuracy": 0.0, "speaker_precision": 0.0,
            "speaker_recall": 0.0, "speaker_f1": 0.0,
            "speaker_confusion_matrix": {},
        }
    return {
        "speaker_accuracy": res["accuracy"],
        "speaker_precision": res["precision"],
        "speaker_recall": res["recall"],
        "speaker_f1": res["f1"],
        "speaker_confusion_matrix": res["confusion_matrix"],
    }


def compute_summary_scores(ref_summary: str, gen_summary: str) -> Dict[str, Any]:
    """ROUGE-1/2/L + BERTScore-style semantic F1 (matches old keys)."""
    rouge = metrics.rouge_scores(ref_summary, gen_summary)
    semi = metrics.semantic_f1(ref_summary, gen_summary, _embed_fn)
    out = {
        "rouge1": rouge.get("rouge1"),
        "rouge2": rouge.get("rouge2"),
        "rougeL": rouge.get("rougeL"),
        "bertscore_f1": semi.get("bertscore_f1"),
    }
    scores = [v for v in (out["rougeL"], out["bertscore_f1"]) if v is not None]
    out["summary_accuracy"] = round(sum(scores) / len(scores), 2) if scores else 0.0
    return out


def compute_mom_accuracy(ref_mom: str, gen_mom: str) -> Dict[str, Any]:
    """MoM semantic coverage + accuracy (matches old keys)."""
    if not ref_mom or not gen_mom:
        return {"mom_coverage": 0.0, "mom_missing_count": 0, "mom_hallucination_count": 0, "mom_accuracy": 0.0}
    ref_items = [l.strip() for l in ref_mom.split("\n") if len(l.strip()) > 10]
    gen_items = [l.strip() for l in gen_mom.split("\n") if len(l.strip()) > 10]
    # Reference item is "covered" if a generated item is semantically similar.
    threshold = SEMANTIC_SIMILARITY_THRESHOLD
    from backend.evaluation.metrics import _cosine

    ref_embs = [_embed_fn(i) for i in ref_items]
    gen_embs = [_embed_fn(i) for i in gen_items]
    matched = 0
    missing = 0
    for r in ref_embs:
        if r and any(_cosine(r, g) >= threshold for g in gen_embs if g):
            matched += 1
        else:
            missing += 1
    hallucinations = 0
    for g in gen_embs:
        if not g or not any(_cosine(g, r) >= threshold for r in ref_embs if r):
            hallucinations += 1
    coverage = (matched / len(ref_items)) * 100.0 if ref_items else 0.0
    hallucination_penalty = (hallucinations / max(1, len(ref_items))) * 15.0
    return {
        "mom_coverage": round(coverage, 2),
        "mom_missing_count": missing,
        "mom_hallucination_count": hallucinations,
        "mom_accuracy": round(max(0.0, coverage - hallucination_penalty), 2),
    }


def compute_action_item_accuracy(ref_items_raw: Any, gen_items_raw: str) -> Dict[str, Any]:
    """Action-item precision/recall/F1 via semantic matching (matches old keys)."""
    ref_list = []
    if isinstance(ref_items_raw, str):
        ref_list = [l.strip() for l in ref_items_raw.split("\n") if len(l.strip()) > 8]
    elif isinstance(ref_items_raw, list):
        for item in ref_items_raw:
            if isinstance(item, dict):
                desc = item.get("text", "") or item.get("description", "")
                owner = item.get("owner", "") or item.get("assignee", "")
                text = f"{desc} (Owner: {owner})" if owner else desc
                ref_list.append(text)
            else:
                ref_list.append(str(item))
    gen_list = [l.strip() for l in str(gen_items_raw).split("\n") if len(l.strip()) > 8]
    res = metrics.semantic_matching_f1(ref_list, gen_list, _embed_fn)
    if res["f1"] is None:
        return {
            "action_precision": 0.0, "action_recall": 0.0, "action_f1": 0.0,
            "action_missing_count": len(ref_list), "action_false_count": len(gen_list),
            "action_accuracy": 0.0,
        }
    return {
        "action_precision": res["precision"],
        "action_recall": res["recall"],
        "action_f1": res["f1"],
        "action_missing_count": res["reference_count"] - round(res["recall"] / 100.0 * res["reference_count"]),
        "action_false_count": res["generated_count"] - round(res["precision"] / 100.0 * res["reference_count"]),
        "action_accuracy": res["f1"],
    }


def compute_topic_accuracy(ref_topics_raw: Any, gen_topics_raw: str) -> Dict[str, Any]:
    """Topic detection F1 via semantic matching (matches old keys)."""
    ref_list = [str(t).strip().lower() for t in (ref_topics_raw or []) if str(t).strip()]
    if isinstance(ref_topics_raw, str):
        ref_list = [t.strip().lower() for t in ref_topics_raw.replace(",", "\n").split("\n") if t.strip()]
    gen_list = [t.strip().lower() for t in str(gen_topics_raw).replace(",", "\n").split("\n") if t.strip()]
    res = metrics.semantic_matching_f1(ref_list, gen_list, _embed_fn)
    return {
        "topic_accuracy": res["f1"] if res["f1"] is not None else 0.0,
        "topic_precision": res["precision"] if res["precision"] is not None else 0.0,
        "topic_recall": res["recall"] if res["recall"] is not None else 0.0,
        "topic_f1": res["f1"] if res["f1"] is not None else 0.0,
    }


def compute_sentiment_accuracy(ref_sentiment_raw: Any, pred_sentiment_raw: str) -> Dict[str, Any]:
    """Sentiment classification accuracy/F1 (matches old keys)."""
    def normalize(val: str) -> str:
        v = val.lower()
        if "pos" in v:
            return "positive"
        if "neg" in v:
            return "negative"
        return "neutral"

    ref_norm = normalize(str(ref_sentiment_raw or ""))
    pred_norm = normalize(str(pred_sentiment_raw or ""))
    correct = 1 if ref_norm == pred_norm else 0
    labels = ["positive", "neutral", "negative"]
    matrix = {l: {l2: 0 for l2 in labels} for l in labels}
    matrix[ref_norm][pred_norm] = 1
    return {
        "sentiment_accuracy": correct * 100.0,
        "sentiment_precision": correct * 100.0,
        "sentiment_recall": correct * 100.0,
        "sentiment_f1": correct * 100.0,
        "sentiment_confusion_matrix": matrix,
    }


# ─── Orchestration (delegates to the engine) ─────────────────────────────────

def run_meeting_evaluation(
    meeting: Meeting, db: Session, performed_by: str = "system", commit: bool = True
) -> EvaluationResult:
    """Run the Universal Evaluation Engine and return the persisted result.

    ``commit=False`` leaves the caller's transaction open (rows are flushed, not committed) - for callers that run
    the evaluation inside a savepoint as optional work.
    """
    report = evaluation_engine.evaluate_meeting(
        meeting, db, performed_by=performed_by, persist=True, commit=commit
    )
    result = (
        db.query(EvaluationResult)
        .filter(EvaluationResult.meeting_id == meeting.id)
        .first()
    )
    if result is None:
        raise RuntimeError("Engine persisted evaluation but no result row was found.")
    return result
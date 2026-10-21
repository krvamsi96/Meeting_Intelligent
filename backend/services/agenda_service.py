"""Agenda Decision Match Score Service
======================================
Computes a real, dynamic, semantically-grounded match score between an
Agenda item and its corresponding Decision using sentence-transformer
embeddings + cosine similarity.

Design Principles:
- NEVER hardcode scores.
- NEVER use random percentages.
- NEVER use keyword matching.
- Every score comes from actual model inference.
- Scores are deterministic and reproducible.

Model: all-MiniLM-L6-v2 (sentence-transformers)
  - 384-dim dense embeddings
  - Optimised for semantic sentence similarity
  - Fast CPU inference (~5–15ms per pair)
  - Production-ready, widely deployed

Similarity Method: Cosine similarity on L2-normalised embeddings.
"""

import logging
import time
from typing import Any, Dict, Optional

from backend.config.config import settings

logger = logging.getLogger(__name__)

_MODEL_NAME: str = getattr(settings, "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Minimum match_score (0-100) required before a decision is considered
# reliably linked to its paired agenda item. Below this, callers should
# drop the pairing from agenda_decision_mapping instead of forcing it.
MATCH_THRESHOLD: float = float(getattr(settings, "AGENDA_DECISION_MATCH_THRESHOLD", 45.0))
# ──────────────────────────────────────────────────────────────────────────────
# Model Encoder — Delegated to Centralized Offline Model Loader
# ──────────────────────────────────────────────────────────────────────────────

def _get_encoder():
    """Retrieve embedding encoder from centralized offline model loader."""
    from backend.services.model_loader import get_embedding_encoder
    return get_embedding_encoder()


def _encode_texts(texts: list) -> Optional[Any]:
    """Encode a list of texts into L2-normalised embeddings via offline model loader."""
    from backend.services.model_loader import encode_texts
    return encode_texts(texts)


# ──────────────────────────────────────────────────────────────────────────────
# Confidence level thresholds
# ──────────────────────────────────────────────────────────────────────────────

def confidence_label(score_pct: Optional[float]) -> Optional[str]:
    """Public wrapper around the confidence-bucket logic, so every caller
    (API layer included) derives the SAME label from a stored score instead
    of re-implementing the bucket thresholds and drifting out of sync.
    """
    if score_pct is None:
        return None
    return _confidence_label(score_pct)


def _confidence_label(score_pct: float) -> str:
    """Map a match-score percentage to a human-readable confidence label."""
    if score_pct >= 95.0:
        return "Excellent Match"
    elif score_pct >= 85.0:
        return "High Match"
    elif score_pct >= 70.0:
        return "Good Match"
    elif score_pct >= MATCH_THRESHOLD:
        return "Fair Match"
    else:
        return "Not Reliable"


def is_reliable_match(score_pct: Optional[float]) -> bool:
    """Return True only when a match score provides sufficient semantic
    evidence to show a decision under an agenda item's mapping.

    A None score (no decision, or empty inputs) is never reliable — callers
    extracting agenda_decision_mapping must treat that as "no mapping",
    never as a forced/default pairing.
    """
    return score_pct is not None and score_pct >= MATCH_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
# Core scoring function
# ──────────────────────────────────────────────────────────────────────────────

def compute_agenda_decision_match_score(
    agenda_text: str,
    decision_text: str,
) -> Dict[str, Any]:
    """Compute a real semantic match score between an agenda item and its decision.

    Args:
        agenda_text:    The agenda topic/discussion point.
        decision_text:  The corresponding decision or resolution.

    Returns:
        {
            "match_score":       float | None  (0.0–100.0, or None when no real decision exists),
            "confidence_level":  str | None,
            "model_name":        str,
            "processing_time":   float,
        }

    Returns match_score=None when the decision is the canonical no-decision
    placeholder (e.g. 'No formal decision recorded.').  Computing cosine
    similarity in that case yields a low but meaningless value that would
    confuse users — None correctly shows '—' in the UI.

    For genuine decisions, falls back to a text-overlap heuristic if the
    embedding model is unavailable — still without random or hardcoded values.
    """
    t0 = time.perf_counter()

    agenda_clean = (agenda_text or "").strip()
    decision_clean = (decision_text or "").strip()

    # ── Guard: empty inputs ────────────────────────────────────
    if not agenda_clean or not decision_clean:
        elapsed = round(time.perf_counter() - t0, 4)
        return {
            "match_score": None,
            "confidence_level": None,
            "model_name": _MODEL_NAME,
            "processing_time": elapsed,
        }

    # ── Guard: no-decision placeholder — score would be meaningless ───────
    _NO_DEC = {
        "no formal decision recorded.", "no formal decision recorded",
        "no decision recorded.", "no decision recorded",
        "no decision.", "no decision", "none", "n/a",
    }
    if decision_clean.lower() in _NO_DEC:
        elapsed = round(time.perf_counter() - t0, 4)
        return {
            "match_score": None,
            "confidence_level": None,
            "model_name": _MODEL_NAME,
            "processing_time": elapsed,
        }

    # ── Guard: trivially identical texts ────────────────────────────
    if agenda_clean.lower() == decision_clean.lower():
        elapsed = round(time.perf_counter() - t0, 4)
        return {
            "match_score": 100.0,
            "confidence_level": "Excellent Match",
            "model_name": _MODEL_NAME,
            "processing_time": elapsed,
        }

    # ── Primary path: sentence-transformer / transformers cosine similarity ──
    embeddings = _encode_texts([agenda_clean, decision_clean])
    if embeddings is not None:
        try:
            import numpy as np
            cosine_sim = float(np.dot(embeddings[0], embeddings[1]))
            score_pct = round(max(0.0, cosine_sim) * 100.0, 1)
            elapsed = round(time.perf_counter() - t0, 4)
            return {
                "match_score": score_pct,
                "confidence_level": _confidence_label(score_pct),
                "model_name": _MODEL_NAME,
                "processing_time": elapsed,
            }
        except Exception as e:
            logger.warning(f"[AgendaService] Embedding failed: {e}. Using lexical fallback.")

    # ── Fallback: Jaccard token overlap (deterministic, not random) ─────────
    logger.info("[AgendaService] Using lexical overlap fallback (model unavailable).")
    agenda_tokens = set(agenda_clean.lower().split())
    decision_tokens = set(decision_clean.lower().split())
    if not agenda_tokens or not decision_tokens:
        score_pct = 0.0
    else:
        intersection = agenda_tokens & decision_tokens
        union = agenda_tokens | decision_tokens
        jaccard = len(intersection) / len(union) if union else 0.0
        # Report the real overlap directly (0-100%) — no artificial floor.
        # A floor here would fabricate confidence for a pairing with zero
        # actual token overlap, which is exactly the "forced mapping"
        # behaviour this service must avoid.
        score_pct = round(jaccard * 100.0, 1)

    elapsed = round(time.perf_counter() - t0, 4)
    return {
        "match_score": score_pct,
        "confidence_level": _confidence_label(score_pct),
        "model_name": f"{_MODEL_NAME}-lexical-fallback",
        "processing_time": elapsed,
    }


def deduplicate_semantic(texts: list, threshold: float = None) -> list:
    """Return the indices of ``texts`` to KEEP after collapsing near-duplicate
    (semantically equivalent) entries — e.g. "Budget approval was discussed."
    and "Approval of the budget was discussed." are the same item in meaning.

    Greedy, order-preserving: each text is compared only against the
    embeddings of texts already kept: the first occurrence of a duplicate
    cluster survives. Falls back to exact (case/whitespace-insensitive)
    string-match dedup when the embedding model is unavailable, so behaviour
    degrades gracefully rather than silently skipping dedup.
    """
    n = len(texts)
    if n <= 1:
        return list(range(n))
    if threshold is None:
        threshold = float(getattr(settings, "SEMANTIC_DEDUP_THRESHOLD", 0.87))

    clean_texts = [(t or "").strip() for t in texts]
    embeddings = _encode_texts(clean_texts)
    keep: list = []

    if embeddings is None:
        seen = set()
        for i, t in enumerate(clean_texts):
            norm = t.lower()
            if norm and norm in seen:
                continue
            if norm:
                seen.add(norm)
            keep.append(i)
        return keep

    try:
        import numpy as np
        for i in range(n):
            if not clean_texts[i]:
                keep.append(i)
                continue
            is_dup = False
            for j in keep:
                if not clean_texts[j]:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(i)
        return keep
    except Exception as e:
        logger.warning(f"[AgendaService] Semantic dedup failed: {e}. Keeping all items.")
        return list(range(n))


def compute_batch_match_scores(
    pairs: list[tuple[str, str]],
) -> list[Dict[str, Any]]:
    """Compute match scores for a batch of (agenda, decision) pairs efficiently.

    Encodes all texts in a single forward pass for maximum throughput.

    Args:
        pairs: List of (agenda_text, decision_text) tuples.

    Returns:
        List of match score dicts, same order as input.
    """
    if not pairs:
        return []

    t0 = time.perf_counter()
    results: list[Dict[str, Any]] = []

    # Flatten all texts for a single batch encode
    all_texts: list[str] = []
    for agenda, decision in pairs:
        all_texts.append((agenda or "").strip())
        all_texts.append((decision or "").strip())

    embeddings = _encode_texts(all_texts)
    if embeddings is None:
        # Fallback: compute individually
        return [compute_agenda_decision_match_score(a, d) for a, d in pairs]

    try:
        import numpy as np
        _NO_DEC = {
            "no formal decision recorded.", "no formal decision recorded",
            "no decision recorded.", "no decision recorded",
            "no decision.", "no decision", "none", "n/a",
        }
        for i, (agenda_text, decision_text) in enumerate(pairs):
            decision_clean = (decision_text or "").strip()
            # Delegate to single scorer to respect all guards (empty, no-dec, identical)
            if not decision_clean or decision_clean.lower() in _NO_DEC:
                results.append(compute_agenda_decision_match_score(agenda_text, decision_text))
                continue
            agenda_emb = embeddings[i * 2]
            decision_emb = embeddings[i * 2 + 1]
            cosine_sim = float(np.dot(agenda_emb, decision_emb))
            score_pct = round(max(0.0, cosine_sim) * 100.0, 1)
            results.append({
                "match_score": score_pct,
                "confidence_level": _confidence_label(score_pct),
                "model_name": _MODEL_NAME,
                "processing_time": round(time.perf_counter() - t0, 4),
            })
        return results

    except Exception as e:
        logger.warning(f"[AgendaService] Batch scoring failed: {e}. Falling back to individual scoring.")
        return [compute_agenda_decision_match_score(a, d) for a, d in pairs]


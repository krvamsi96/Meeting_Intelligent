"""Reference-Free Evaluation Engine — no Ground Truth required, ever.

This module implements the REFERENCE_FREE evaluation contract: every score is
computed live from the meeting's own transcript, timestamps, speaker labels,
agenda/decision/action-item rows, generated summary/MoM, and chatbot RAG
traffic. Nothing here compares against a human-authored reference, nothing is
hardcoded per meeting, and no WER/CER/DER/ROUGE/F1 style "accuracy" number is
ever produced — those require Ground Truth and are surfaced as ``null``
elsewhere (see ``backend.evaluation.engine`` for the Ground-Truth path).

Design rules
------------
* Every function is a pure computation over its inputs (no DB access, no
  caching, no module-level state) so two calls for two different meetings can
  never contaminate each other.
* A score is ``None`` ("not measured") only when there is genuinely nothing
  to measure (e.g. zero action items). Where the *absence* of content is
  itself a real, observable signal (e.g. an empty transcript), that fact is
  scored honestly (e.g. ``0.0``) — never guessed, never a nice round default.
* Composite scores are a documented, overridable weighted mean of the
  sub-signals that were actually measured; unmeasured sub-signals are
  dropped and the remaining weights renormalized (never blindly averaged,
  never padded with an invented number).
* Embeddings (via ``vector_service``) are used opportunistically for
  semantic similarity/grounding; every embedding-based computation has a
  deterministic word-overlap fallback so a missing/failed embedding model
  degrades gracefully instead of producing ``None`` everywhere.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from backend.evaluation.metrics import _cosine
from backend.utils import text_signals as _text_signals

logger = logging.getLogger(__name__)

REFERENCE_FREE = "REFERENCE_FREE"

# Cosine similarity above this is considered "semantically supported by".
EMBED_SUPPORT_THRESHOLD = float(os.environ.get("RF_EMBED_SUPPORT_THRESHOLD", "0.55"))
# Cosine similarity above this pairs an agenda topic with its decision / mapping.
EMBED_MAPPING_THRESHOLD = float(os.environ.get("RF_EMBED_MAPPING_THRESHOLD", "0.45"))

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")
_ABNORMAL_WS_RE = re.compile(r"[ \t]{3,}|\xa0|\u200b")
_NON_ALNUM_RE = re.compile(r"[^\w\s]", re.UNICODE)

_NO_DECISION_PHRASES = frozenset({
    "", "none", "n/a", "na", "no decision", "no decision.",
    "no decision recorded", "no decision recorded.",
    "no formal decision recorded", "no formal decision recorded.",
})


def _w(env_key: str, default: float) -> float:
    try:
        return float(os.environ.get(env_key, default))
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _words(text: Optional[str]) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _sentences(text: Optional[str]) -> List[str]:
    if not text or not text.strip():
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if len(s.strip()) > 3]


def _clamp_pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(max(0.0, min(100.0, value)), 2)


def _weighted(components: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
    """Weighted mean of 0..1 components (higher = better) -> 0..100, or None.

    Components that are ``None`` are dropped and the remaining weights are
    renormalized — never padded with an invented value.
    """
    total_w = 0.0
    total = 0.0
    for key, val in components.items():
        if val is None:
            continue
        w = weights.get(key, 0.0)
        total += max(0.0, min(1.0, val)) * w
        total_w += w
    if total_w <= 0:
        return None
    return _clamp_pct(total / total_w * 100.0)


def _duplicate_ratio(items: Sequence[str]) -> Optional[float]:
    """Fraction of items that are a near-duplicate of an earlier item (first-6-word key)."""
    keys = []
    for it in items:
        words = _words(it)
        if words:
            keys.append(" ".join(words[:6]))
    if not keys:
        return None
    counts = Counter(keys)
    dup_count = sum(c - 1 for c in counts.values() if c > 1)
    return dup_count / len(keys)


def _embed_batch(texts: Sequence[str]) -> List[List[float]]:
    """Safe batch embedding; returns [] on any failure (never raises)."""
    cleaned = [t for t in texts if t and str(t).strip()]
    if not cleaned:
        return []
    try:
        from backend.services.vector_service import vector_service
        embs = vector_service.generate_embeddings_batch(cleaned)
        return embs or []
    except Exception as e:
        logger.debug("Reference-free engine: embedding batch failed: %s", e)
        return []


def _embed_one(text: Optional[str]) -> List[float]:
    if not text or not text.strip():
        return []
    embs = _embed_batch([text])
    return embs[0] if embs else []


def _max_cosine(vec: Sequence[float], others: Sequence[Sequence[float]]) -> float:
    best = 0.0
    for o in others:
        if o:
            best = max(best, _cosine(vec, o))
    return best


def _holistic_similarity(text_a: Optional[str], text_b: Optional[str]) -> Optional[float]:
    """Single cosine-similarity score between two whole texts, in [0, 1].

    Falls back to Jaccard word overlap when embeddings are unavailable.
    Returns None only when one side has no content at all.
    """
    if not text_a or not text_a.strip() or not text_b or not text_b.strip():
        return None
    # Cap length so a single embedding call stays cheap for long transcripts.
    emb_a = _embed_one(text_a[:4000])
    emb_b = _embed_one(text_b[:4000])
    if emb_a and emb_b:
        return round(_cosine(emb_a, emb_b), 4)
    wa, wb = set(_words(text_a)), set(_words(text_b))
    if not wa or not wb:
        return None
    return round(len(wa & wb) / len(wa | wb), 4)


def _build_source_index(source_text: Optional[str], max_sentences: int = 400) -> Dict[str, Any]:
    """Precompute sentence embeddings + content words for a (potentially large)
    source text ONCE, so many targets can be checked against it without
    re-embedding the source on every check. Build one of these per transcript
    per evaluation run, not per item — see ``_precision_from_index``.
    """
    sentences = _sentences(source_text)[:max_sentences]
    if not sentences and source_text and source_text.strip():
        sentences = [source_text[:2000]]
    embeddings = _embed_batch(sentences) if sentences else []
    words = set(w for w in _words(source_text) if len(w) > 4)
    return {"sentences": sentences, "embeddings": embeddings, "words": words}


def _precision_from_index(index: Dict[str, Any], target_text: Optional[str]) -> Optional[float]:
    """Fraction of ``target_text``'s sentences supported by a prebuilt source index."""
    target_sents = _sentences(target_text)
    if not target_sents or not index["sentences"]:
        return None

    embeddings = index["embeddings"]
    if embeddings and len(embeddings) == len(index["sentences"]):
        tgt_embs = _embed_batch(target_sents[:200])
        if tgt_embs and len(tgt_embs) == len(target_sents[:200]):
            supported = sum(1 for t in tgt_embs if t and _max_cosine(t, embeddings) >= EMBED_SUPPORT_THRESHOLD)
            return supported / len(tgt_embs)

    if not index["words"]:
        return None
    counted = 0
    supported = 0
    for sent in target_sents:
        sw = [w for w in _words(sent) if len(w) > 4]
        if not sw:
            continue
        counted += 1
        if any(w in index["words"] for w in sw):
            supported += 1
    return (supported / counted) if counted else None


def _semantic_precision(source_text: Optional[str], target_text: Optional[str]) -> Optional[float]:
    """One-shot convenience wrapper around ``_build_source_index`` +
    ``_precision_from_index``. "Supported" = max cosine similarity to any
    source sentence >= threshold (embedding path), or >= 1 shared content
    word (fallback path, len > 4).

    For repeated checks against the same (large) source — e.g. a transcript
    checked against many action items/decisions — build the index once with
    ``_build_source_index()`` and reuse it via ``_precision_from_index()``
    instead of calling this per item, to avoid re-embedding the source text
    on every call.
    """
    if not source_text or not source_text.strip():
        return None
    return _precision_from_index(_build_source_index(source_text), target_text)


def _coverage_flags(items: Sequence[str], text: Optional[str]) -> List[bool]:
    """Per-item "is this item present in text?" in ONE pass — embeds ``text``'s
    sentences and all ``items`` exactly once each, regardless of item count.
    Use this (not a per-item loop calling ``_coverage_ratio``) whenever many
    items must be checked against the same text.
    """
    clean_items = [str(i).strip() for i in items]
    if not text or not text.strip():
        return [False] * len(clean_items)

    sents = _sentences(text) or [text]
    item_embs = _embed_batch(clean_items) if any(clean_items) else []
    sent_embs = _embed_batch(sents[:400])
    if item_embs and sent_embs and len(item_embs) == len(clean_items):
        return [bool(ie) and _max_cosine(ie, sent_embs) >= EMBED_SUPPORT_THRESHOLD for ie in item_embs]

    text_lower = text.lower()
    flags = []
    for it in clean_items:
        it_words = [w for w in _words(it) if len(w) > 3]
        if not it_words:
            flags.append(False)
            continue
        hits = sum(1 for w in it_words if w in text_lower)
        flags.append(it.lower() in text_lower or (hits / len(it_words)) >= 0.5)
    return flags


def _coverage_ratio(items: Sequence[str], text: Optional[str]) -> Optional[float]:
    """Fraction of ``items`` that are semantically/lexically present in ``text``."""
    clean_items = [str(i).strip() for i in items if i and str(i).strip()]
    if not clean_items:
        return None
    flags = _coverage_flags(clean_items, text)
    return (sum(1 for f in flags if f) / len(flags)) if flags else 0.0


def _is_placeholder_decision(text: Optional[str]) -> bool:
    return (text or "").strip().lower() in _NO_DECISION_PHRASES


# ─────────────────────────────────────────────────────────────────────────────
# 1. Transcription Quality
# ─────────────────────────────────────────────────────────────────────────────

_TRANSCRIPTION_WEIGHTS = {
    "speech_coverage":        _w("RF_TRANS_W_COVERAGE", 0.25),
    "non_empty_ratio":        _w("RF_TRANS_W_NON_EMPTY", 0.15),
    "timestamp_validity":     _w("RF_TRANS_W_TS_VALID", 0.15),
    "timestamp_continuity":   _w("RF_TRANS_W_TS_CONTINUITY", 0.10),
    "low_duplicate":          _w("RF_TRANS_W_LOW_DUP", 0.15),
    "meaningful_ratio":       _w("RF_TRANS_W_MEANINGFUL", 0.10),
    "language_consistency":   _w("RF_TRANS_W_LANG", 0.10),
}


def transcription_quality(
    segments: Sequence[Dict[str, Any]],
    audio_duration_sec: Optional[float] = None,
    language_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Reference-free transcript quality — see module docstring for the policy.

    ``segments``: dicts with ``text``, ``start_time``, ``end_time`` (seconds
    or numeric strings). ``language_confidence``: the pipeline's own
    source-language detection confidence in [0, 1], if recorded.
    """
    result: Dict[str, Any] = {
        "evaluation_type": REFERENCE_FREE,
        "primary_metric": "transcript_quality_score",
        "formula_notes": {
            "transcript_quality_score": (
                "weighted_mean(speech_coverage, non_empty_ratio, timestamp_validity, "
                "timestamp_continuity, 1-duplicate_ratio, meaningful_ratio, "
                "language_consistency) over whichever signals are measurable"
            ),
            "speech_coverage": "sum(segment durations) / audio_duration x 100",
            "duplicate_ratio": "near-duplicate segments (first-6-word key) / segment_count",
            "timestamp_validity": "segments with 0 <= start < end / segment_count",
            "timestamp_continuity": "consecutive segment pairs with non-decreasing start / (segment_count-1)",
        },
        "weights": _TRANSCRIPTION_WEIGHTS,
    }

    segment_count = len(segments)
    result["segment_count"] = segment_count

    texts = [(s.get("text") or "").strip() for s in segments]
    word_count = sum(len(t.split()) for t in texts)
    char_count = sum(len(t) for t in texts)
    result["word_count"] = word_count
    result["character_count"] = char_count

    if segment_count == 0:
        result.update({
            "is_empty": True,
            "note": "empty_transcript",
            "duplicate_ratio": None,
            "timestamp_validity": None,
            "timestamp_continuity": None,
            "speech_coverage": None,
            "meaningful_segment_ratio": None,
            "language_detection_confidence": language_confidence,
            "score": 0.0,
        })
        return result

    result["is_empty"] = False

    empty_count = sum(1 for t in texts if not t)
    non_empty_ratio = (segment_count - empty_count) / segment_count
    result["empty_segment_count"] = empty_count

    # Meaningful text = more than 2 content words (filters "yes", "ok", "[inaudible]").
    meaningful_count = sum(1 for t in texts if len([w for w in t.split() if len(w) > 2]) > 2)
    meaningful_ratio = meaningful_count / segment_count
    result["meaningful_segment_count"] = meaningful_count
    result["meaningful_segment_ratio"] = _clamp_pct(meaningful_ratio * 100.0)

    duplicate_ratio = _duplicate_ratio(texts)
    result["duplicate_ratio"] = round(duplicate_ratio, 4) if duplicate_ratio is not None else None

    valid_ts = 0
    parsed_ts: List[tuple] = []
    covered = 0.0
    for s in segments:
        try:
            start = float(s.get("start_time"))
            end = float(s.get("end_time"))
        except (TypeError, ValueError):
            continue
        if 0 <= start < end:
            valid_ts += 1
            covered += (end - start)
            parsed_ts.append((start, end))
    ts_validity = valid_ts / segment_count
    result["timestamp_validity"] = _clamp_pct(ts_validity * 100.0)

    continuity = None
    if len(parsed_ts) >= 2:
        ordered = sorted(parsed_ts, key=lambda p: p[0])
        non_decreasing = sum(
            1 for i in range(1, len(ordered)) if ordered[i][0] >= ordered[i - 1][0]
        )
        continuity = non_decreasing / (len(ordered) - 1)
    result["timestamp_continuity"] = _clamp_pct(continuity * 100.0) if continuity is not None else None

    speech_coverage = None
    if audio_duration_sec and audio_duration_sec > 0:
        speech_coverage = min(1.0, covered / audio_duration_sec)
    result["speech_coverage"] = _clamp_pct(speech_coverage * 100.0) if speech_coverage is not None else None
    result["speech_covered_seconds"] = round(covered, 2)

    # 0.0 is what rows written before the fix stored for "no detector ran": not a measured confidence.
    if language_confidence is not None and language_confidence <= 0.0:
        language_confidence = None
    result["language_detection_confidence"] = (
        round(language_confidence, 4) if language_confidence is not None else None
    )

    components = {
        "speech_coverage": speech_coverage,
        "non_empty_ratio": non_empty_ratio,
        "timestamp_validity": ts_validity,
        "timestamp_continuity": continuity,
        "low_duplicate": (1.0 - duplicate_ratio) if duplicate_ratio is not None else None,
        "meaningful_ratio": meaningful_ratio,
        "language_consistency": language_confidence,
    }
    result["score"] = _weighted(components, _TRANSCRIPTION_WEIGHTS)
    result["components"] = {k: _clamp_pct(v * 100.0) if v is not None else None for k, v in components.items()}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text Integrity
# ─────────────────────────────────────────────────────────────────────────────

_INTEGRITY_WEIGHTS = {
    "clean_chars":        _w("RF_INTEG_W_CLEAN_CHARS", 0.25),
    "no_excess_symbols":  _w("RF_INTEG_W_SYMBOLS", 0.15),
    "no_repeated_chars":  _w("RF_INTEG_W_REPEATED", 0.15),
    "no_malformed_words": _w("RF_INTEG_W_MALFORMED", 0.15),
    "non_empty_segments": _w("RF_INTEG_W_EMPTY", 0.15),
    "normal_whitespace":  _w("RF_INTEG_W_WHITESPACE", 0.15),
}


def text_integrity(segments: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Structural text-quality signal — corruption, garbling, encoding damage.

    Entirely independent of any reference text: it only asks "is the text
    itself well-formed?", not "is it correct?".
    """
    result: Dict[str, Any] = {
        "evaluation_type": REFERENCE_FREE,
        "primary_metric": "text_integrity_score",
        "formula_notes": {
            "text_integrity_score": (
                "weighted_mean(1-broken_char_rate, 1-excess_symbol_rate, "
                "1-repeated_char_rate, 1-malformed_word_rate, 1-empty_segment_rate, "
                "1-abnormal_whitespace_rate)"
            ),
        },
        "weights": _INTEGRITY_WEIGHTS,
    }

    texts = [(s.get("text") or "") for s in segments]
    total_segs = len(texts)
    result["segment_count"] = total_segs
    if total_segs == 0:
        result.update({
            "score": None,
            "note": "no_segments_to_check",
            "broken_char_count": None,
            "excess_symbol_ratio": None,
            "repeated_char_ratio": None,
            "malformed_word_ratio": None,
            "empty_segment_ratio": None,
            "abnormal_whitespace_ratio": None,
        })
        return result

    broken_char_count = 0
    excess_symbol_segments = 0
    repeated_char_segments = 0
    malformed_word_segments = 0
    empty_segments = 0
    abnormal_ws_segments = 0
    total_chars = 0

    for text in texts:
        stripped = text.strip()
        total_chars += len(text)
        if not stripped:
            empty_segments += 1
            continue

        broken_char_count += len(_CONTROL_CHAR_RE.findall(text))

        symbol_chars = len(_NON_ALNUM_RE.findall(text.replace(" ", "")))
        if len(text) > 0 and (symbol_chars / len(text)) > 0.30:
            excess_symbol_segments += 1

        if _REPEATED_CHAR_RE.search(text):
            repeated_char_segments += 1

        if _ABNORMAL_WS_RE.search(text):
            abnormal_ws_segments += 1

        words = text.split()
        malformed = any(
            len(w) > 30 or _REPEATED_CHAR_RE.search(w)
            for w in words
        )
        if malformed:
            malformed_word_segments += 1

    excess_symbol_ratio = excess_symbol_segments / total_segs
    repeated_char_ratio = repeated_char_segments / total_segs
    malformed_word_ratio = malformed_word_segments / total_segs
    empty_segment_ratio = empty_segments / total_segs
    abnormal_ws_ratio = abnormal_ws_segments / total_segs
    broken_char_rate = min(1.0, broken_char_count / max(1, total_chars))

    result.update({
        "broken_char_count": broken_char_count,
        "excess_symbol_ratio": round(excess_symbol_ratio, 4),
        "repeated_char_ratio": round(repeated_char_ratio, 4),
        "malformed_word_ratio": round(malformed_word_ratio, 4),
        "empty_segment_ratio": round(empty_segment_ratio, 4),
        "abnormal_whitespace_ratio": round(abnormal_ws_ratio, 4),
    })

    components = {
        "clean_chars": 1.0 - broken_char_rate,
        "no_excess_symbols": 1.0 - excess_symbol_ratio,
        "no_repeated_chars": 1.0 - repeated_char_ratio,
        "no_malformed_words": 1.0 - malformed_word_ratio,
        "non_empty_segments": 1.0 - empty_segment_ratio,
        "normal_whitespace": 1.0 - abnormal_ws_ratio,
    }
    result["score"] = _weighted(components, _INTEGRITY_WEIGHTS)
    result["components"] = {k: _clamp_pct(v * 100.0) for k, v in components.items()}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Speaker Diarization Consistency
# ─────────────────────────────────────────────────────────────────────────────

_DIARIZATION_WEIGHTS = {
    "assignment_rate": _w("RF_DIAR_W_ASSIGN", 0.25),
    "continuity":      _w("RF_DIAR_W_CONTINUITY", 0.20),
    "timestamp_valid": _w("RF_DIAR_W_TS_VALID", 0.20),
    "low_overlap":     _w("RF_DIAR_W_LOW_OVERLAP", 0.15),
    "coverage":        _w("RF_DIAR_W_COVERAGE", 0.20),
}

_MICRO_SEGMENT_SEC = 0.5


def diarization_consistency(
    segments: Sequence[Dict[str, Any]],
    audio_duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Reference-free speaker-diarization consistency.

    The number of speakers, their durations and switch counts are all
    derived from the ACTUAL segment labels for this meeting only — never
    assumed, never clamped to a fixed count.
    """
    result: Dict[str, Any] = {
        "evaluation_type": REFERENCE_FREE,
        "primary_metric": "speaker_consistency_score",
        "formula_notes": {
            "speaker_consistency_score": (
                "weighted_mean(assignment_rate, continuity, timestamp_validity, "
                "1-overlap_ratio, speaker_coverage)"
            ),
            "short_segment_ratio": f"segments shorter than {_MICRO_SEGMENT_SEC}s / total_valid_segments",
            "speaker_switches": "count of consecutive segment pairs whose speaker differs",
        },
        "weights": _DIARIZATION_WEIGHTS,
    }

    total = len(segments)
    result["total_segments"] = total
    if total == 0:
        result.update({
            "score": None,
            "speaker_count": 0,
            "speaker_duration": {},
            "speaker_switches": None,
            "short_segment_ratio": None,
            "note": "no_segments",
        })
        return result

    parsed = []
    invalid_ts = 0
    for s in segments:
        speaker = (s.get("speaker") or "").strip() or "UNKNOWN"
        try:
            start, end = float(s.get("start_time")), float(s.get("end_time"))
        except (TypeError, ValueError):
            invalid_ts += 1
            continue
        if start >= end:
            invalid_ts += 1
            continue
        parsed.append((start, end, speaker))

    valid_count = len(parsed)
    ts_valid_ratio = valid_count / total
    result["invalid_timestamp_count"] = invalid_ts

    if valid_count == 0:
        result.update({
            "score": _clamp_pct(ts_valid_ratio * 100.0 * _DIARIZATION_WEIGHTS.get("timestamp_valid", 0)),
            "speaker_count": 0,
            "speaker_duration": {},
            "speaker_switches": None,
            "short_segment_ratio": None,
            "note": "no_valid_timestamps",
        })
        return result

    parsed.sort(key=lambda p: p[0])

    known_labels = {"unknown", "unidentified"}
    duration_by_speaker: Dict[str, float] = {}
    assigned_count = 0
    for start, end, speaker in parsed:
        dur = end - start
        if speaker.lower() not in known_labels:
            duration_by_speaker[speaker] = duration_by_speaker.get(speaker, 0.0) + dur
            assigned_count += 1

    speaker_count = len(duration_by_speaker)
    result["speaker_count"] = speaker_count
    total_assigned = sum(duration_by_speaker.values()) or 1.0
    result["speaker_duration"] = {
        spk: {
            "duration_seconds": round(dur, 2),
            "percentage": round(dur / total_assigned * 100.0, 2),
        }
        for spk, dur in sorted(duration_by_speaker.items(), key=lambda kv: -kv[1])
    }
    dominant_ratio = (max(duration_by_speaker.values()) / total_assigned) if duration_by_speaker else None
    result["dominant_speaker_share_pct"] = _clamp_pct(dominant_ratio * 100.0) if dominant_ratio is not None else None

    assignment_rate = assigned_count / valid_count
    result["assignment_rate_pct"] = _clamp_pct(assignment_rate * 100.0)

    switches = sum(1 for i in range(1, len(parsed)) if parsed[i][2] != parsed[i - 1][2])
    result["speaker_switches"] = switches

    micro_count = sum(1 for start, end, _ in parsed if (end - start) < _MICRO_SEGMENT_SEC)
    short_segment_ratio = micro_count / valid_count
    result["short_segment_ratio"] = round(short_segment_ratio, 4)
    result["micro_segment_count"] = micro_count

    # Overlap: total pairwise time-overlap between different-speaker segments,
    # relative to the diarized timeline span — real diarization errors show up
    # as > 0 (two speakers "active" over the same audio window).
    overlap_time = 0.0
    for i in range(len(parsed)):
        s1, e1, spk1 = parsed[i]
        for j in range(i + 1, len(parsed)):
            s2, e2, spk2 = parsed[j]
            if s2 >= e1:
                break
            if spk2 != spk1:
                overlap_time += max(0.0, min(e1, e2) - max(s1, s2))
    span = max(1e-6, parsed[-1][1] - parsed[0][0])
    overlap_ratio = min(1.0, overlap_time / span)
    result["speaker_overlap_ratio"] = round(overlap_ratio, 4)

    coverage = None
    if audio_duration_sec and audio_duration_sec > 0:
        coverage = min(1.0, total_assigned / audio_duration_sec)
    result["speaker_coverage_pct"] = _clamp_pct(coverage * 100.0) if coverage is not None else None

    components = {
        "assignment_rate": assignment_rate,
        "continuity": 1.0 - short_segment_ratio,
        "timestamp_valid": ts_valid_ratio,
        "low_overlap": 1.0 - overlap_ratio,
        "coverage": coverage,
    }
    result["score"] = _weighted(components, _DIARIZATION_WEIGHTS)
    # A component can be None (coverage without an audio duration - e.g. a poll that reads the meeting row just before
    # the pipeline commits its duration and the transcript just after): _weighted skips it, and so must this. It
    # used to raise TypeError here and the whole evaluation_metrics block was lost for that response.
    result["components"] = {k: (_clamp_pct(v * 100.0) if v is not None else None) for k, v in components.items()}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Summary Grounding
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARY_WEIGHTS = {
    "semantic_similarity": _w("RF_SUM_W_SIM", 0.35),
    "supported":           _w("RF_SUM_W_SUPPORTED", 0.35),
    "topic_coverage":      _w("RF_SUM_W_TOPIC", 0.15),
    "not_repeated":        _w("RF_SUM_W_NOT_REPEATED", 0.15),
}


def summary_grounding(
    summary_text: Optional[str],
    transcript_text: Optional[str],
    agenda_topics: Optional[Sequence[str]] = None,
    transcript_index: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """How well the generated summary is grounded in the meeting transcript."""
    result: Dict[str, Any] = {
        "evaluation_type": REFERENCE_FREE,
        "primary_metric": "summary_grounding_score",
        "formula_notes": {
            "summary_grounding_score": (
                "weighted_mean(semantic_similarity, 1-unsupported_content_ratio, "
                "topic_coverage, 1-repeated_content_ratio)"
            ),
            "semantic_similarity": "cosine(embed(summary), embed(transcript)); word-overlap fallback",
            "unsupported_content_ratio": "summary sentences with no transcript support / summary sentence count",
            "topic_coverage": "agenda topics semantically present in the summary / total agenda topics",
        },
        "weights": _SUMMARY_WEIGHTS,
    }

    if not summary_text or not summary_text.strip():
        result.update({
            "score": None,
            "note": "no_generated_summary",
            "semantic_similarity": None,
            "topic_coverage": None,
            "unsupported_content_ratio": None,
            "summary_word_count": 0,
        })
        return result

    result["summary_word_count"] = len(summary_text.split())
    sents = _sentences(summary_text)
    result["sentence_count"] = len(sents)

    index = transcript_index if transcript_index is not None else _build_source_index(transcript_text)
    similarity = _holistic_similarity(transcript_text, summary_text)
    precision = _precision_from_index(index, summary_text)
    unsupported_ratio = (1.0 - precision) if precision is not None else None
    topic_cov = _coverage_ratio(list(agenda_topics or []), summary_text)
    dup_ratio = _duplicate_ratio(sents) or 0.0

    result["semantic_similarity"] = round(similarity, 4) if similarity is not None else None
    result["unsupported_content_ratio"] = round(unsupported_ratio, 4) if unsupported_ratio is not None else None
    result["topic_coverage"] = _clamp_pct(topic_cov * 100.0) if topic_cov is not None else None
    result["repeated_content_ratio"] = round(dup_ratio, 4)

    components = {
        "semantic_similarity": similarity,
        "supported": precision,
        "topic_coverage": topic_cov,
        "not_repeated": 1.0 - dup_ratio,
    }
    result["score"] = _weighted(components, _SUMMARY_WEIGHTS)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5 & 6. Action Item / Decision Grounding (shared implementation)
# ─────────────────────────────────────────────────────────────────────────────

def _grounding_report(
    items: Sequence[Dict[str, Any]],
    transcript_text: Optional[str],
    text_fields: Sequence[str],
    evidence_field: str,
    primary_metric: str,
    transcript_index: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Shared grounding checker for action items and decisions.

    An item is "grounded" when either (a) it carries an ``evidence`` quote
    that itself has lexical/semantic support in the transcript, or (b) its
    own text is semantically supported by the transcript directly.

    ``transcript_index``: pass a prebuilt ``_build_source_index(transcript_text)``
    when checking several item lists against the same transcript in one
    evaluation run, so the transcript is embedded once, not once per item.
    """
    total = len(items)
    result: Dict[str, Any] = {
        "evaluation_type": REFERENCE_FREE,
        "primary_metric": primary_metric,
        "formula_notes": {
            primary_metric: (
                "clamp(100 * grounded_items/total_items - 30 * duplicate_ratio, 0, 100)"
            ),
            "grounded": (
                "ONE test shared with metrics.*_quality: the evidence quote occurs verbatim in the transcript "
                "(case, punctuation, speaker labels ignored; '...' gaps checked fragment by fragment); otherwise "
                "the quote / the item's own text is semantically supported by a transcript sentence "
                "(cosine >= RF_EMBED_SUPPORT_THRESHOLD, or a shared content word without embeddings). "
                "A heuristic reference-free signal, not accuracy."
            ),
        },
        "total_items": total,
    }
    if total == 0:
        result.update({"score": None, "grounded_items": 0, "unsupported_items": 0, "duplicate_items": 0})
        return result

    item_texts = []
    for it in items:
        text = " ".join(str(it.get(f) or "") for f in text_fields).strip()
        item_texts.append(text)

    dup_ratio = _duplicate_ratio(item_texts) or 0.0
    dup_count = round(dup_ratio * total)

    index = transcript_index if transcript_index is not None else _build_source_index(transcript_text)
    tx_norm = _text_signals.normalize_text(transcript_text or "")

    grounded = 0
    unsupported = 0
    empty_items = 0
    per_item: List[Dict[str, Any]] = []
    for it, text in zip(items, item_texts):
        if not text.strip():
            empty_items += 1
            per_item.append({"grounded": False, "reason": "empty_item"})
            continue

        evidence = str(it.get(evidence_field) or "").strip()
        is_grounded = False
        basis = None
        if evidence and _text_signals.quote_in_text(evidence, tx_norm):
            # The quote IS in the transcript. Embedding precision over "sentences" cannot be trusted for this: ASR
            # transcripts are often one long unpunctuated run, so a short quote never matches a 100-word "sentence"
            # (measured: 3 verbatim quotes scored 0 grounded here and 100 in metrics.action_item_quality).
            is_grounded, basis = True, "evidence_verbatim"
        elif evidence:
            support = _precision_from_index(index, evidence)
            if support is None:
                ev_words = set(w for w in _words(evidence) if len(w) > 4)
                support = 1.0 if (ev_words and index["words"] and ev_words & index["words"]) else 0.0
            is_grounded = support is not None and support >= 0.5
            basis = "evidence"
        if not is_grounded:
            direct = _precision_from_index(index, text)
            if direct is not None and direct >= 0.5:
                is_grounded = True
                basis = "direct_text_match"

        if is_grounded:
            grounded += 1
        else:
            unsupported += 1
        per_item.append({"grounded": is_grounded, "basis": basis})

    score = _clamp_pct(100.0 * grounded / total - 30.0 * dup_ratio)
    result.update({
        "grounded_items": grounded,
        "unsupported_items": unsupported,
        "empty_items": empty_items,
        "duplicate_items": dup_count,
        "duplicate_ratio": round(dup_ratio, 4),
        "score": score,
    })
    return result


def action_item_grounding(
    action_items: Sequence[Dict[str, Any]],
    transcript_text: Optional[str],
    transcript_index: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Action Item Grounding Score — verifies each extracted item against the transcript."""
    r = _grounding_report(
        action_items, transcript_text,
        text_fields=("task", "description"),
        evidence_field="evidence",
        primary_metric="action_item_grounding_score",
        transcript_index=transcript_index,
    )
    if r["total_items"] > 0:
        owner_supported = 0
        deadline_supported = 0
        owners_checked = 0
        deadlines_checked = 0
        tx_lower = (transcript_text or "").lower()
        for it in action_items:
            owner = str(it.get("owner") or "").strip()
            if owner and owner.lower() not in ("unassigned", "not specified", "none", ""):
                owners_checked += 1
                if owner.lower() in tx_lower:
                    owner_supported += 1
            deadline = str(it.get("deadline") or "").strip()
            if deadline and deadline.lower() not in ("not specified", "none", ""):
                deadlines_checked += 1
                dw = [w for w in _words(deadline) if len(w) > 2]
                if dw and any(w in tx_lower for w in dw):
                    deadline_supported += 1
        r["owner_support_pct"] = _clamp_pct(owner_supported / owners_checked * 100.0) if owners_checked else None
        r["deadline_support_pct"] = _clamp_pct(deadline_supported / deadlines_checked * 100.0) if deadlines_checked else None
    return r


def decision_grounding(
    decisions: Sequence[Dict[str, Any]],
    transcript_text: Optional[str],
    transcript_index: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Decision Grounding Score — verifies each extracted decision against the transcript."""
    r = _grounding_report(
        decisions, transcript_text,
        text_fields=("decision", "reason"),
        evidence_field="evidence",
        primary_metric="decision_grounding_score",
        transcript_index=transcript_index,
    )
    # Re-key to the spec's field names.
    r["total_decisions"] = r.pop("total_items")
    r["grounded_decisions"] = r.pop("grounded_items")
    r["unsupported_decisions"] = r.pop("unsupported_items")
    r.pop("empty_items", None)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# 7. Agenda Coverage
# ─────────────────────────────────────────────────────────────────────────────

def agenda_coverage(agenda_items: Sequence[Dict[str, Any]], transcript_text: Optional[str]) -> Dict[str, Any]:
    """Agenda Coverage Score — are the detected agenda topics actually discussed?

    ``agenda_items`` are read fresh from THIS meeting's agenda_key_decisions
    rows on every call, so two different meetings never share agenda data.
    """
    result: Dict[str, Any] = {
        "evaluation_type": REFERENCE_FREE,
        "primary_metric": "agenda_coverage_score",
        "formula_notes": {
            "agenda_coverage_score": "100 * supported_topics / detected_topics",
            "supported": "agenda title semantically/lexically present in the transcript",
        },
    }
    titles = [str(a.get("title") or "").strip() for a in agenda_items if str(a.get("title") or "").strip()]
    detected = len(titles)
    result["detected_topics"] = detected
    if detected == 0:
        result.update({"score": None, "supported_topics": 0, "unsupported_topics": 0, "note": "no_agenda_items"})
        return result
    if not transcript_text or not transcript_text.strip():
        result.update({"score": None, "supported_topics": 0, "unsupported_topics": detected, "note": "no_transcript"})
        return result

    flags = _coverage_flags(titles, transcript_text)
    supported = sum(1 for f in flags if f)
    unsupported = detected - supported
    result["supported_topics"] = supported
    result["unsupported_topics"] = unsupported
    result["score"] = _clamp_pct(100.0 * supported / detected)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 8. Agenda -> Decision Mapping Consistency
# ─────────────────────────────────────────────────────────────────────────────

def decision_agenda_consistency(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Decision-to-Agenda Consistency Score.

    ``pairs``: one dict per agenda_key_decisions row with ``agenda``,
    ``decision`` (None/blank if no real decision was recorded) and the
    pipeline's own ``match_score`` (0-100 cosine similarity from
    agenda_service), when available.
    """
    result: Dict[str, Any] = {
        "evaluation_type": REFERENCE_FREE,
        "primary_metric": "decision_agenda_consistency_score",
        "formula_notes": {
            "decision_agenda_consistency_score": "100 * supported_mappings / total_mappings",
            "supported": f"cosine(embed(agenda_title), embed(decision_text)) >= {EMBED_MAPPING_THRESHOLD}",
        },
    }
    real_pairs = [p for p in pairs if not _is_placeholder_decision(p.get("decision"))]
    total = len(real_pairs)
    result["total_mappings"] = total
    if total == 0:
        result.update({"score": None, "supported_mappings": 0, "unsupported_mappings": 0, "note": "no_decisions_to_map"})
        return result

    supported = 0
    for p in real_pairs:
        agenda_title = str(p.get("agenda") or "")
        decision_text = str(p.get("decision") or "")
        score = p.get("match_score")
        sim = (float(score) / 100.0) if score is not None else None
        if sim is None:
            sim = _holistic_similarity(agenda_title, decision_text)
        if sim is not None and sim >= EMBED_MAPPING_THRESHOLD:
            supported += 1
    unsupported = total - supported
    result["supported_mappings"] = supported
    result["unsupported_mappings"] = unsupported
    result["score"] = _clamp_pct(100.0 * supported / total)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 9. MOM Grounding
# ─────────────────────────────────────────────────────────────────────────────

_MOM_WEIGHTS = {
    "semantic_similarity": _w("RF_MOM_W_SIM", 0.30),
    "agenda_coverage":     _w("RF_MOM_W_AGENDA", 0.15),
    "decision_coverage":   _w("RF_MOM_W_DECISION", 0.20),
    "action_item_coverage": _w("RF_MOM_W_ACTION", 0.15),
    "supported":           _w("RF_MOM_W_SUPPORTED", 0.20),
}


def mom_grounding(
    mom_text: Optional[str],
    transcript_text: Optional[str],
    agenda_titles: Sequence[str],
    decision_texts: Sequence[str],
    action_item_texts: Sequence[str],
    transcript_index: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """MoM Grounding Score — is the Minutes of Meeting backed by the transcript
    and does it actually cover the agenda/decisions/action items this meeting
    produced?"""
    result: Dict[str, Any] = {
        "evaluation_type": REFERENCE_FREE,
        "primary_metric": "mom_grounding_score",
        "formula_notes": {
            "mom_grounding_score": (
                "weighted_mean(semantic_similarity, agenda_coverage, decision_coverage, "
                "action_item_coverage, 1-unsupported_content_ratio)"
            ),
        },
        "weights": _MOM_WEIGHTS,
    }

    if not mom_text or not mom_text.strip():
        result.update({
            "score": None,
            "note": "no_generated_mom",
            "agenda_coverage": None,
            "decision_coverage": None,
            "action_item_coverage": None,
            "unsupported_content_ratio": None,
        })
        return result

    index = transcript_index if transcript_index is not None else _build_source_index(transcript_text)
    similarity = _holistic_similarity(transcript_text, mom_text)
    precision = _precision_from_index(index, mom_text)
    unsupported_ratio = (1.0 - precision) if precision is not None else None
    agenda_cov = _coverage_ratio(list(agenda_titles), mom_text)
    decision_cov = _coverage_ratio(list(decision_texts), mom_text)
    action_cov = _coverage_ratio(list(action_item_texts), mom_text)
    dup_ratio = _duplicate_ratio(_sentences(mom_text)) or 0.0

    result["semantic_similarity"] = round(similarity, 4) if similarity is not None else None
    result["agenda_coverage"] = _clamp_pct(agenda_cov * 100.0) if agenda_cov is not None else None
    result["decision_coverage"] = _clamp_pct(decision_cov * 100.0) if decision_cov is not None else None
    result["action_item_coverage"] = _clamp_pct(action_cov * 100.0) if action_cov is not None else None
    result["unsupported_content_ratio"] = round(unsupported_ratio, 4) if unsupported_ratio is not None else None
    result["duplicate_content_ratio"] = round(dup_ratio, 4)

    components = {
        "semantic_similarity": similarity,
        "agenda_coverage": agenda_cov,
        "decision_coverage": decision_cov,
        "action_item_coverage": action_cov,
        "supported": precision,
    }
    result["score"] = _weighted(components, _MOM_WEIGHTS)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 10. RAG / Chatbot Groundedness
# ─────────────────────────────────────────────────────────────────────────────

def rag_groundedness(chat_messages: Sequence[Dict[str, Any]], meeting_id: Optional[int] = None) -> Dict[str, Any]:
    """RAG Groundedness Score for this meeting's chatbot traffic only.

    ``chat_messages``: assistant-role ``ChatSession`` rows as dicts with
    ``content`` (the answer) and ``sources`` (already-parsed list of
    retrieved-chunk dicts). Chunks tagged with a different ``meeting_id`` are
    dropped so cross-meeting retrieval (``cross_meeting=True`` chat requests)
    never leaks into this meeting's score.
    """
    result: Dict[str, Any] = {
        "evaluation_type": REFERENCE_FREE,
        "primary_metric": "rag_groundedness_score",
        "formula_notes": {
            "rag_groundedness_score": "100 * mean(per_message semantic support of answer by its own retrieved context)",
            "context_relevance": "mean(similarity_score of retrieved chunks, as returned by the vector search)",
            "unsupported_claim_ratio": "1 - mean(per_message grounding ratio)",
        },
    }
    if not chat_messages:
        result.update({
            "score": None,
            "note": "no_chat_activity",
            "message_count": 0,
            "context_count": None,
            "context_relevance": None,
            "unsupported_claim_ratio": None,
            "retrieval_latency_ms": None,
        })
        return result

    grounding_ratios: List[float] = []
    relevance_scores: List[float] = []
    context_counts: List[int] = []

    for msg in chat_messages:
        answer = str(msg.get("content") or "").strip()
        if not answer:
            continue
        sources = msg.get("sources") or []
        if meeting_id is not None:
            sources = [c for c in sources if c.get("meeting_id") in (None, meeting_id)]
        context_counts.append(len(sources))
        for c in sources:
            score = c.get("similarity_score", c.get("score"))
            if score is not None:
                try:
                    relevance_scores.append(float(score))
                except (TypeError, ValueError):
                    pass

        context_text = " ".join(str(c.get("text") or c.get("content") or "") for c in sources)
        precision = _semantic_precision(context_text, answer) if context_text.strip() else None
        if precision is None:
            answer_words = set(w for w in _words(answer) if len(w) > 4)
            context_words = set(w for w in _words(context_text) if len(w) > 4)
            precision = (len(answer_words & context_words) / len(answer_words)) if answer_words and context_words else None
        if precision is not None:
            grounding_ratios.append(precision)

    result["message_count"] = len(chat_messages)
    result["context_count"] = round(sum(context_counts) / len(context_counts), 2) if context_counts else None
    result["context_relevance"] = round(sum(relevance_scores) / len(relevance_scores), 4) if relevance_scores else None
    result["retrieval_latency_ms"] = None  # not persisted per-message; see live /chat response for real-time latency

    if not grounding_ratios:
        result.update({"score": None, "unsupported_claim_ratio": None, "note": "no_scoreable_answers"})
        return result

    mean_grounding = sum(grounding_ratios) / len(grounding_ratios)
    result["unsupported_claim_ratio"] = round(1.0 - mean_grounding, 4)
    result["score"] = _clamp_pct(mean_grounding * 100.0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 11. Overall Composite Quality
# ─────────────────────────────────────────────────────────────────────────────

_OVERALL_WEIGHTS = {
    "transcription":           _w("RF_OVERALL_W_TRANSCRIPTION", 0.20),
    "text_integrity":          _w("RF_OVERALL_W_TEXT_INTEGRITY", 0.05),
    "diarization":             _w("RF_OVERALL_W_DIARIZATION", 0.15),
    "summary":                 _w("RF_OVERALL_W_SUMMARY", 0.15),
    "action_items":            _w("RF_OVERALL_W_ACTION_ITEMS", 0.10),
    "decisions":               _w("RF_OVERALL_W_DECISIONS", 0.10),
    "agenda":                  _w("RF_OVERALL_W_AGENDA", 0.05),
    "agenda_decision_mapping": _w("RF_OVERALL_W_MAPPING", 0.05),
    "mom":                     _w("RF_OVERALL_W_MOM", 0.10),
    "rag":                     _w("RF_OVERALL_W_RAG", 0.05),
}


def overall_quality(component_scores: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """Composite ``overall_quality_score`` = weighted mean of the measured
    feature-level scores (each already 0-100). Components that are ``None``
    (not measured for this meeting) are dropped and the remaining weights
    are renormalized — this is NOT a blind average of all ten numbers.
    """
    total_w = 0.0
    weighted_sum = 0.0
    used_weights: Dict[str, float] = {}
    for key, score in component_scores.items():
        if score is None:
            continue
        w = _OVERALL_WEIGHTS.get(key, 0.0)
        if w <= 0:
            continue
        weighted_sum += score * w
        total_w += w
        used_weights[key] = w

    overall = _clamp_pct(weighted_sum / total_w) if total_w > 0 else None

    # Gate: everything downstream is built on the transcript, so a poor transcript cannot be averaged away by healthy
    # looking secondary scores (measured: transcript 92.8 with 8.5 words/min and a 92 %-truncated text still gave an
    # overall 86.2). With a transcript score under 60 the overall is held to that score + 10.
    gates: List[str] = []
    transcription = component_scores.get("transcription")
    if overall is not None and transcription is not None and transcription < 60.0:
        ceiling = round(transcription + 10.0, 2)
        if overall > ceiling:
            gates.append(f"capped at transcription score + 10 = {ceiling} (transcription {transcription} < 60)")
            overall = ceiling

    return {
        "evaluation_type": REFERENCE_FREE,
        "overall_quality_score": overall,
        "components": {k: (round(v, 2) if v is not None else None) for k, v in component_scores.items()},
        "weights_used": used_weights,
        "weights_configured": _OVERALL_WEIGHTS,
        "gates_applied": gates,
        "formula": (
            "overall_quality_score = 100 * sum(component_score_i/100 * weight_i for measured i) "
            "/ sum(weight_i for measured i) — unmeasured components are dropped and the "
            "remaining weights renormalized, never padded or defaulted. Gate: when the transcription score is "
            "below 60 the overall score cannot exceed transcription + 10. These are heuristic, reference-free "
            "signals, not measured accuracy."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration — assembles the exact API/DB schema for a single meeting
# ─────────────────────────────────────────────────────────────────────────────

def build_evaluation_metrics(
    *,
    segments: Sequence[Dict[str, Any]],
    audio_duration_sec: Optional[float],
    language_confidence: Optional[float],
    summary_text: Optional[str],
    mom_text: Optional[str],
    action_items: Sequence[Dict[str, Any]],
    agenda_rows: Sequence[Dict[str, Any]],
    chat_messages: Sequence[Dict[str, Any]],
    meeting_id: Optional[int] = None,
    score_overrides: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Any]:
    """Build the full REFERENCE_FREE ``evaluation_metrics`` payload for one meeting.

    ``score_overrides``: authoritative scores for the three headline metrics, keyed
    ``"transcription"`` / ``"summary"`` / ``"rag"`` (value may be ``None`` = not measured). They come
    from ``backend.evaluation.quality_scores`` (the API's ``metrics`` block) and replace this engine's
    own older estimate of the same metric, so a meeting never reports two different values for
    "transcript quality", "summary quality" or "RAG groundedness". The overall composite below is
    computed from the overridden values.

    Every argument is THIS meeting's own data, freshly queried by the caller —
    nothing here is cached, memoized, or shared across calls, so evaluating
    two different meetings back-to-back can never contaminate one another.

    ``agenda_rows``: one dict per ``agenda_key_decisions`` row —
        {"agenda_id", "agenda" (title), "decision", "evidence", "reason",
         "match_score", "speaker"}.
    ``action_items``: one dict per ``action_items`` row —
        {"task", "description", "owner", "deadline", "evidence"}.
    ``chat_messages``: assistant-role ``chat_sessions`` rows —
        {"content", "sources": [{"text"/"content", "similarity_score", "meeting_id"}]}.
    """
    transcript_text = "\n".join(
        f"{s.get('speaker') or ''}: {s.get('text') or ''}".strip()
        for s in segments if s.get("text")
    )

    agenda_titles = [str(r.get("agenda") or "").strip() for r in agenda_rows if str(r.get("agenda") or "").strip()]
    agenda_items_for_coverage = [{"title": t} for t in agenda_titles]

    decision_rows = [
        {"decision": r.get("decision"), "reason": r.get("reason"), "evidence": r.get("evidence")}
        for r in agenda_rows if not _is_placeholder_decision(r.get("decision"))
    ]
    decision_texts = [str(d.get("decision") or "") for d in decision_rows]
    action_item_texts = [str(a.get("task") or "") for a in action_items if str(a.get("task") or "").strip()]

    # Built ONCE and reused by every grounding check below — embedding the
    # transcript per action item / decision / summary / MoM check would be
    # O(items) embedding calls; this keeps it O(1) per evaluation run.
    transcript_index = _build_source_index(transcript_text)

    trans = transcription_quality(segments, audio_duration_sec, language_confidence)
    integ = text_integrity(segments)
    diar = diarization_consistency(segments, audio_duration_sec)
    summ = summary_grounding(summary_text, transcript_text, agenda_titles, transcript_index=transcript_index)
    ai_g = action_item_grounding(action_items, transcript_text, transcript_index=transcript_index)
    dec_g = decision_grounding(decision_rows, transcript_text, transcript_index=transcript_index)
    ag_cov = agenda_coverage(agenda_items_for_coverage, transcript_text)
    mapping = decision_agenda_consistency(agenda_rows)
    mom_g = mom_grounding(
        mom_text, transcript_text, agenda_titles, decision_texts, action_item_texts,
        transcript_index=transcript_index,
    )
    rag = rag_groundedness(chat_messages, meeting_id=meeting_id)

    ov = score_overrides or {}
    if "transcription" in ov:
        trans = {**trans, "score": ov["transcription"], "superseded_by": "metrics.transcript_quality"}
    if "summary" in ov:
        summ = {**summ, "score": ov["summary"], "superseded_by": "metrics.summary_quality"}
    if "rag" in ov:
        rag = {**rag, "score": ov["rag"], "superseded_by": "metrics.rag_groundedness"}
    if "agenda" in ov:
        # One agenda score: metrics.agenda_coverage already includes this block's strict per-topic support share
        # (supported_topics / detected_topics are the same numbers), so the two can no longer disagree.
        ag_cov = {**ag_cov, "strict_topic_support_score": ag_cov.get("score"), "score": ov["agenda"],
                  "superseded_by": "metrics.agenda_coverage"}
    if "action_items" in ov:
        # grounded_items / unsupported_items above use the same verbatim-quote test as metrics.action_item_quality;
        # the headline score is that block's (it also weighs owner, deadline, actionability and uniqueness).
        ai_g = {**ai_g, "grounding_score": ai_g.get("score"), "score": ov["action_items"],
                "superseded_by": "metrics.action_item_quality"}
    if "decisions" in ov:
        dec_g = {**dec_g, "grounding_score": dec_g.get("score"), "score": ov["decisions"],
                 "superseded_by": "metrics.decision_quality"}

    quality = overall_quality({
        "transcription": trans.get("score"),
        "text_integrity": integ.get("score"),
        "diarization": diar.get("score"),
        "summary": summ.get("score"),
        "action_items": ai_g.get("score"),
        "decisions": dec_g.get("score"),
        "agenda": ag_cov.get("score"),
        "agenda_decision_mapping": mapping.get("score"),
        "mom": mom_g.get("score"),
        "rag": rag.get("score"),
    })

    return {
        "evaluation_mode": REFERENCE_FREE,
        "ground_truth_available": False,
        "transcription": {**trans, "wer": None},
        "text_integrity": {**integ, "cer": None},
        "diarization": {**diar, "der": None},
        "summary": {**summ, "rouge_l": None},
        "action_items": {**ai_g, "f1": None},
        "decisions": {**dec_g, "f1": None},
        "agenda": {**ag_cov, "f1": None},
        "agenda_decision_mapping": mapping,
        "mom": mom_g,
        "rag": rag,
        "quality_metrics": quality,
    }


def build_runtime_metrics(
    segments: Sequence[Dict[str, Any]],
    audio_duration_sec: float,
    stt_duration_sec: Optional[float],
    speaker_duration_sec: Optional[float],
    summary_text: str,
    mom_text: str,
    transcript_text: str,
    agenda_decision_rows: Sequence[Dict[str, Any]],
    action_items: Sequence[Dict[str, Any]],
    meeting: Any,
    speech_durations: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Thin wrapper assembling the existing reference-free runtime-quality
    signals (``backend.evaluation.runtime_quality``) into the ``runtime_metrics``
    block of the meeting API response. No new computation — reuses the exact
    formulas already served by ``GET /evaluation/{id}/runtime-quality-metrics``.
    """
    from backend.evaluation import runtime_quality as rq

    stt_rt = rq.stt_runtime_metrics(segments, audio_duration_sec, stt_duration_sec)
    spk_rt = rq.speaker_runtime_metrics(segments, audio_duration_sec, speaker_duration_sec)
    # Talk time per speaker: the diarization's speech-only seconds when it covers exactly the speakers of the
    # segments (the same rule the final record's ``speaker_duration`` uses), so the two never report different
    # numbers; otherwise the segment spans (which include pauses) stay, and ``speech_duration_source`` says so.
    distribution = spk_rt.get("speaker_distribution") or []
    spk_rt["speech_duration_source"] = "segment_spans"
    if speech_durations and distribution and {d["speaker"] for d in distribution} == set(speech_durations):
        total_speech = sum(max(0.0, float(v)) for v in speech_durations.values())
        for d in distribution:
            secs = max(0.0, float(speech_durations[d["speaker"]]))
            d["duration_seconds"] = round(secs, 3)
            d["percentage"] = round(secs / total_speech * 100.0, 2) if total_speech > 0 else 0.0
        distribution.sort(key=lambda d: -d["duration_seconds"])
        spk_rt["speech_duration_source"] = "diarization_speech_regions"
        spk_rt["total_speech_duration_sec"] = round(total_speech, 3)
    # Audio / speech / silence in one place, named for what they are (not measured -> null, never 0).
    audio_len = spk_rt.get("audio_duration_sec")
    speech_len = spk_rt.get("total_speech_duration_sec") if spk_rt["speech_duration_source"] == "diarization_speech_regions" else None
    spk_rt["speech_duration_sec"] = speech_len
    spk_rt["silence_duration_sec"] = (
        round(max(0.0, audio_len - speech_len), 3) if audio_len is not None and speech_len is not None else None
    )
    spk_rt["speech_ratio_pct"] = (
        round(min(100.0, speech_len / audio_len * 100.0), 2) if audio_len and speech_len is not None else None
    )
    # The model's own turn count before alignment (what diarization_metadata_json calls validation segment_count).
    raw_turns = None
    try:
        import json as _json

        meta_raw = getattr(meeting, "diarization_metadata_json", None)
        meta = _json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
        value = (meta.get("validation_diagnostics") or {}).get("segment_count")
        raw_turns = int(value) if value is not None else None
    except Exception:  # noqa: BLE001 - an enrichment: unreadable metadata means "not stored"
        raw_turns = None
    spk_rt["diarization_turn_count"] = raw_turns
    sum_rt = rq.summary_runtime_metrics(summary_text, transcript_text)
    mom_rt = rq.mom_runtime_metrics(mom_text)
    agenda_rt = rq.agenda_decision_runtime_metrics(list(agenda_decision_rows))
    ai_rt = rq.action_item_runtime_metrics(list(action_items))
    perf_rt = rq.processing_performance_metrics(meeting)

    return {
        "evaluation_type": REFERENCE_FREE,
        "stt": stt_rt,
        "speaker": spk_rt,
        "content": {**sum_rt, "mom": mom_rt, "agenda_decision": agenda_rt},
        "action_items": ai_rt,
        "performance": perf_rt,
    }

"""Reference-Free Runtime Quality Metrics
==========================================
Calculates measurable quality signals from actual pipeline outputs
without requiring any ground-truth reference data.

Every function in this module is a PURE computation over its inputs.
- No DB access
- No AI/LLM calls
- No hardcoded quality scores
- No fabricated accuracy percentages

Evaluation type for all metrics: REFERENCE_FREE or RUNTIME_PERFORMANCE

When a required input is missing/empty the metric is returned as None
and the caller reports it as 'not_measured'. A None value is NEVER
replaced with an invented fallback number.

Weight Configuration
--------------------
STT and Speaker quality score component weights can be overridden via
environment variables:
  STT_WEIGHT_COVERAGE       default 0.35
  STT_WEIGHT_VALIDITY       default 0.25
  STT_WEIGHT_CONFIDENCE     default 0.20
  STT_WEIGHT_EMPTY_PENALTY  default 0.10
  STT_WEIGHT_REPETITION     default 0.10

  SPK_WEIGHT_COVERAGE       default 0.35
  SPK_WEIGHT_ASSIGNMENT     default 0.30
  SPK_WEIGHT_CONSISTENCY    default 0.20
  SPK_WEIGHT_CONTINUITY     default 0.15
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Weight configuration (overridable via env vars)
# ---------------------------------------------------------------------------

def _w(env_key: str, default: float) -> float:
    """Read a float weight from environment; fall back to default."""
    try:
        return float(os.environ.get(env_key, default))
    except (TypeError, ValueError):
        return default


STT_W = {
    "coverage":      _w("STT_WEIGHT_COVERAGE",      0.35),
    "validity":      _w("STT_WEIGHT_VALIDITY",       0.25),
    "confidence":    _w("STT_WEIGHT_CONFIDENCE",     0.20),
    "empty_penalty": _w("STT_WEIGHT_EMPTY_PENALTY",  0.10),
    "repetition":    _w("STT_WEIGHT_REPETITION",     0.10),
}

SPK_W = {
    "coverage":    _w("SPK_WEIGHT_COVERAGE",    0.35),
    "assignment":  _w("SPK_WEIGHT_ASSIGNMENT",  0.30),
    "consistency": _w("SPK_WEIGHT_CONSISTENCY", 0.20),
    "continuity":  _w("SPK_WEIGHT_CONTINUITY",  0.15),
}

# Required ActionItem fields for completeness calculation
_REQUIRED_AI_FIELDS = ("task", "owner", "deadline", "priority", "evidence")
_VALID_PRIORITIES   = {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# Sanity validation
# ---------------------------------------------------------------------------

class MetricValidationError(ValueError):
    """Raised when a computed metric value is outside its valid domain."""


def _clamp_pct(value: Optional[float], name: str) -> Optional[float]:
    """Clamp percentage to [0, 100]; raise MetricValidationError on None sentinel abuse."""
    if value is None:
        return None
    if not math.isfinite(value):
        raise MetricValidationError(f"{name}: non-finite value {value!r}")
    return round(max(0.0, min(100.0, value)), 3)


def _clamp_pos(value: Optional[float], name: str) -> Optional[float]:
    """Clamp non-negative value; raise MetricValidationError on negative."""
    if value is None:
        return None
    if not math.isfinite(value):
        raise MetricValidationError(f"{name}: non-finite value {value!r}")
    return round(max(0.0, value), 4)


def validate_metric(name: str, value: Any, kind: str) -> Any:
    """Validate a metric value against domain rules.

    kind options: 'percentage', 'duration', 'speaker_count',
                  'latency', 'rtf', 'wer', 'float', 'int'
    Returns value if valid, raises MetricValidationError otherwise.
    """
    if value is None:
        return None
    if kind == "percentage":
        if not (0 <= float(value) <= 100):
            raise MetricValidationError(f"{name}={value} outside [0, 100]")
    elif kind in ("duration", "latency", "rtf", "wer"):
        if float(value) < 0:
            raise MetricValidationError(f"{name}={value} must be >= 0")
    elif kind == "speaker_count":
        if not isinstance(value, int) or value < 0:
            raise MetricValidationError(f"{name}={value} must be non-negative int")
    return value


# ---------------------------------------------------------------------------
# 1. STT Runtime Quality Metrics
# ---------------------------------------------------------------------------

_LEGACY_PLACEHOLDER_STT_CONFIDENCE = 0.85


def stt_runtime_metrics(
    segments: List[Dict[str, Any]],
    audio_duration_sec: float,
    stt_duration_sec: Optional[float],
) -> Dict[str, Any]:
    """Compute reference-free STT runtime quality metrics from transcript segments.

    Parameters
    ----------
    segments : list of dicts with keys: start_time, end_time, text,
               stt_confidence (optional), correction_applied (optional)
    audio_duration_sec : total audio file duration in seconds
    stt_duration_sec : wall-clock time Whisper/STT engine took

    Returns
    -------
    dict with evaluation_type='REFERENCE_FREE' and all computed signals.
    None values mean that signal could not be computed.
    """
    result: Dict[str, Any] = {
        "evaluation_type": "REFERENCE_FREE",
        "formula_notes": {
            "transcript_coverage": "transcribed_duration / audio_duration × 100",
            "empty_segment_rate":  "empty_segments / total_segments × 100",
            "timestamp_validity_rate": "valid_ts_pairs / total_segments × 100",
            "rtf": "stt_duration_sec / audio_duration_sec (lower=faster)",
            "avg_stt_confidence": "mean(stt_confidence) from Whisper logprob signal",
            "repetition_rate": "repeated_phrase_segments / total_segments × 100",
            "correction_rate": "corrected_segments / total_segments × 100",
            "stt_runtime_quality_score": (
                f"weighted([coverage×{STT_W['coverage']}, "
                f"validity×{STT_W['validity']}, "
                f"confidence_signal×{STT_W['confidence']}, "
                f"(1-empty)×{STT_W['empty_penalty']}, "
                f"(1-repetition)×{STT_W['repetition']}])"
            ),
        },
        "weights": STT_W,
    }

    total_segs = len(segments)
    result["total_segments"] = total_segs
    result["sample_count"] = total_segs

    if total_segs == 0:
        result["transcript_coverage"] = None
        result["empty_segment_rate"] = None
        result["timestamp_validity_rate"] = None
        result["repetition_rate"] = None
        result["correction_rate"] = None
        result["avg_stt_confidence"] = None
        result["rtf"] = None
        result["stt_runtime_quality_score"] = None
        return result

    # ── Transcript coverage ──────────────────────────────────────────────────
    transcribed_dur = 0.0
    valid_ts_count = 0
    invalid_ts_count = 0
    empty_count = 0
    confidence_vals: List[float] = []
    corrected_count = 0
    texts: List[str] = []

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            empty_count += 1

        start_raw = seg.get("start_time") if seg.get("start_time") is not None else seg.get("start")
        end_raw   = seg.get("end_time")   if seg.get("end_time")   is not None else seg.get("end")

        try:
            s, e = float(start_raw), float(end_raw)
            dur = e - s
            if s < e and s >= 0 and (audio_duration_sec <= 0 or e <= audio_duration_sec + 1.0):
                transcribed_dur += dur
                valid_ts_count += 1
            else:
                invalid_ts_count += 1
        except (TypeError, ValueError):
            invalid_ts_count += 1

        # Confidence signal
        conf = seg.get("stt_confidence")
        if conf is not None:
            try:
                confidence_vals.append(float(conf))
            except (TypeError, ValueError):
                pass

        # Correction tracking
        if seg.get("correction_applied"):
            corrected_count += 1

        texts.append(text)

    audio_dur = float(audio_duration_sec) if audio_duration_sec and audio_duration_sec > 0 else None

    # Transcript coverage
    if audio_dur and audio_dur > 0:
        coverage = _clamp_pct(transcribed_dur / audio_dur * 100.0, "transcript_coverage")
    else:
        coverage = None
    result["transcript_coverage"] = coverage
    result["transcribed_duration_sec"] = round(transcribed_dur, 3)

    # Empty segment rate
    empty_rate = _clamp_pct(empty_count / total_segs * 100.0, "empty_segment_rate")
    result["empty_segment_rate"] = empty_rate
    result["empty_segment_count"] = empty_count

    # Timestamp validity rate
    ts_validity = _clamp_pct(valid_ts_count / total_segs * 100.0, "timestamp_validity_rate")
    result["timestamp_validity_rate"] = ts_validity
    result["invalid_timestamp_count"] = invalid_ts_count

    # Rows written by older pipeline versions carry a constant 0.85 on EVERY segment (a default, not a measurement):
    # it is not a confidence signal and must not feed the score.
    if confidence_vals and all(abs(v - _LEGACY_PLACEHOLDER_STT_CONFIDENCE) < 1e-9 for v in confidence_vals):
        confidence_vals = []

    # Average STT confidence (only if the engine actually provided confidence values)
    if confidence_vals:
        avg_conf = round(sum(confidence_vals) / len(confidence_vals), 4)
        result["avg_stt_confidence"] = avg_conf
        result["confidence_sample_count"] = len(confidence_vals)
    else:
        result["avg_stt_confidence"] = None
        result["confidence_sample_count"] = 0

    # Correction rate
    correction_rate = _clamp_pct(corrected_count / total_segs * 100.0, "correction_rate")
    result["correction_rate"] = correction_rate
    result["corrected_segment_count"] = corrected_count

    # RTF — Real-Time Factor: processing_time / audio_duration
    if stt_duration_sec is not None and audio_dur and stt_duration_sec >= 0 and audio_dur > 0:
        rtf = round(stt_duration_sec / audio_dur, 5)
        result["rtf"] = _clamp_pos(rtf, "rtf")
    else:
        result["rtf"] = None

    # Repetition rate — detect abnormally repeated short phrases
    repetition_rate = _compute_repetition_rate(texts)
    result["repetition_rate"] = repetition_rate

    # ── STT Runtime Quality Score ────────────────────────────────────────────
    # Components normalised to [0, 1], all higher = better
    components: Dict[str, Optional[float]] = {}

    components["coverage"]      = coverage / 100.0 if coverage is not None else None
    components["validity"]      = ts_validity / 100.0 if ts_validity is not None else None
    # Confidence: use as-is if in [0,1], else None
    if result["avg_stt_confidence"] is not None:
        c = result["avg_stt_confidence"]
        components["confidence"] = max(0.0, min(1.0, c)) if 0.0 <= c <= 1.0 else None
    else:
        components["confidence"] = None
    components["empty_penalty"] = 1.0 - (empty_rate / 100.0) if empty_rate is not None else None
    components["repetition"]    = 1.0 - (repetition_rate / 100.0) if repetition_rate is not None else None

    score = _weighted_score(components, STT_W)

    # ``transcript_coverage`` is the share of the TIMELINE the segments span: one segment from 0:00 to the end is 100 %
    # whatever it holds. Words per minute shows whether speech was actually transcribed; the same gates as
    # metrics.transcript_quality apply, so the two blocks cannot disagree (a 30-min meeting with 259 words was
    # "stt_runtime_quality_score 97.0" next to transcript_quality "Needs Improvement").
    spoken_words = sum(len((t or "").split()) for t in texts)
    wpm = round(spoken_words / (audio_dur / 60.0), 1) if audio_dur and audio_dur >= 60 else None
    result["words_per_minute"] = wpm
    result["score_gate"] = None
    if score is not None and wpm is not None:
        cap = 55.0 if wpm < 15 else (74.0 if wpm < 30 else None)
        if cap is not None and score > cap:
            result["score_gate"] = f"capped at {cap:g}: only {wpm:g} words per minute were transcribed"
            score = cap
    result["stt_runtime_quality_score"] = score
    result["stt_quality_components"] = {k: round(v * 100, 2) if v is not None else None
                                         for k, v in components.items()}
    return result


def _compute_repetition_rate(texts: List[str]) -> Optional[float]:
    """Detect abnormally repeated short phrases across segments.

    A segment is considered 'repetitive' if its first 6 words exactly match
    those of another segment in the same transcript.
    Returns: repetitive_segments / total_segments × 100
    """
    if not texts:
        return None
    phrase_counter: Counter = Counter()
    for text in texts:
        words = text.lower().split()
        if words:
            key = " ".join(words[:6])
            phrase_counter[key] += 1
    repetitive = sum(count - 1 for count in phrase_counter.values() if count > 1)
    return _clamp_pct(repetitive / len(texts) * 100.0, "repetition_rate")


# ---------------------------------------------------------------------------
# 2. Speaker Diarization Runtime Quality Metrics
# ---------------------------------------------------------------------------

def speaker_runtime_metrics(
    segments: List[Dict[str, Any]],
    audio_duration_sec: float,
    speaker_duration_sec: Optional[float],
) -> Dict[str, Any]:
    """Compute reference-free speaker diarization runtime quality metrics.

    Parameters
    ----------
    segments : list of dicts with keys: speaker, start_time, end_time, text
    audio_duration_sec : total audio duration in seconds
    speaker_duration_sec : wall-clock time diarization engine took

    Returns
    -------
    dict with per-speaker stats and overall diarization quality signals.
    """
    result: Dict[str, Any] = {
        "evaluation_type": "REFERENCE_FREE",
        "formula_notes": {
            "audio_duration_sec": "length of the recording (ffprobe)",
            "speech_duration_sec": "seconds of DETECTED speech (the diarization's VAD regions); null when not measured",
            "speaker_coverage": (
                "share of the audio timeline spanned by the validated segments (end clipped to the audio length; "
                "pauses inside a segment count) - a coverage of the timeline, not a speech measure"
            ),
            "total_diarized_duration_sec": "summed span of the validated segments (pauses included)",
            "diarization_processing_time_sec": "wall-clock time the diarization step took to run",
            "speaker_assignment_rate": "assigned_segments / total_segments × 100",
            "segment_count": "validated transcript segments (a long monologue is cut into several)",
            "turn_count": "speaker turns: runs of consecutive segments by one speaker (a monologue = 1)",
            "speaker_change_count": "turn_count - 1: changes from one speaker to another",
            "diarization_turn_count": "turns the diarization MODEL produced before alignment; null when not stored",
            "speaker_runtime_quality_score": (
                f"weighted([coverage×{SPK_W['coverage']}, "
                f"assignment×{SPK_W['assignment']}, "
                f"consistency×{SPK_W['consistency']}, "
                f"continuity×{SPK_W['continuity']}])"
            ),
        },
        "weights": SPK_W,
    }

    total_segs = len(segments)
    result["total_segments"] = total_segs
    result["sample_count"] = total_segs

    if total_segs == 0:
        result["detected_speakers"] = 0
        result["speaker_count"] = 0
        result["speaker_coverage"] = None
        result["speaker_assignment_rate"] = None
        result["turn_count"] = None
        result["avg_turn_duration_sec"] = None
        result["speaker_distribution"] = []
        result["unassigned_duration_sec"] = None
        result["timestamp_consistency_rate"] = None
        result["speaker_runtime_quality_score"] = None
        return result

    # ── Parse segments ───────────────────────────────────────────────────────
    audio_dur = float(audio_duration_sec) if audio_duration_sec and audio_duration_sec > 0 else None

    speaker_durations: Dict[str, float] = defaultdict(float)
    speaker_turns:     Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    unassigned_dur = 0.0
    assigned_count = 0
    total_diarized = 0.0
    valid_ts = 0
    invalid_ts = 0

    parsed: List[Tuple[float, float, str]] = []  # (start, end, speaker)

    for seg in segments:
        speaker = (seg.get("speaker") or "").strip()
        start   = seg.get("start_time") if seg.get("start_time") is not None else seg.get("start")
        end     = seg.get("end_time")   if seg.get("end_time")   is not None else seg.get("end")

        try:
            s, e = float(start), float(end)
        except (TypeError, ValueError):
            invalid_ts += 1
            continue

        if s >= e:
            invalid_ts += 1
            continue
        if audio_dur is not None:
            if s >= audio_dur:               # starts after the recording ended: not a valid segment
                invalid_ts += 1
                continue
            e = min(e, audio_dur)            # a segment cannot end after the recording does

        valid_ts += 1
        dur = e - s
        total_diarized += dur
        parsed.append((s, e, speaker))

        is_unknown = (not speaker or "unknown" in speaker.lower() or
                      "unidentified" in speaker.lower())
        if is_unknown:
            unassigned_dur += dur
        else:
            speaker_durations[speaker] += dur
            speaker_turns[speaker].append((s, e))
            assigned_count += 1

    # ── Speaker detection ────────────────────────────────────────────────────
    unique_speakers = set(speaker_durations.keys())
    detected_speakers = len(unique_speakers)
    result["detected_speakers"] = detected_speakers
    result["speaker_count"] = detected_speakers

    # ── Speaker Coverage ─────────────────────────────────────────────────────
    if audio_dur and audio_dur > 0 and total_diarized >= 0:
        coverage = _clamp_pct(total_diarized / audio_dur * 100.0, "speaker_coverage")
    else:
        coverage = None
    result["speaker_coverage"] = coverage
    result["total_diarized_duration_sec"] = round(total_diarized, 3)

    # ── Speaker Assignment Rate ──────────────────────────────────────────────
    assignment_rate = _clamp_pct(
        assigned_count / total_segs * 100.0 if total_segs > 0 else None,
        "speaker_assignment_rate"
    )
    result["speaker_assignment_rate"] = assignment_rate
    result["unassigned_duration_sec"] = round(unassigned_dur, 3)

    # ── Timestamp Consistency ────────────────────────────────────────────────
    ts_consistency = _clamp_pct(
        valid_ts / (valid_ts + invalid_ts) * 100.0 if (valid_ts + invalid_ts) > 0 else None,
        "timestamp_consistency_rate"
    )
    result["timestamp_consistency_rate"] = ts_consistency

    # ── Speaker Distribution (per-speaker stats) ─────────────────────────────
    # Turns = runs of consecutive segments by ONE speaker (utils.helpers.speaker_turns), computed once over the
    # validated timeline; per-speaker and overall counts are read from the same list, so they cannot disagree
    # (they were 27 per speaker and 0 overall for one 27-segment monologue).
    from backend.utils.helpers import speaker_turns as _speaker_turns

    all_turns = _speaker_turns(parsed)
    distribution: List[Dict[str, Any]] = []
    total_assigned_dur = sum(speaker_durations.values()) or 1.0
    for spk, dur in sorted(speaker_durations.items(), key=lambda x: -x[1]):
        turns = [t for t in all_turns if t["speaker"] == spk]
        turn_count = len(turns)
        turn_durs  = [t["end"] - t["start"] for t in turns]
        avg_turn   = round(sum(turn_durs) / turn_count, 3) if turn_count else 0.0
        max_turn   = round(max(turn_durs), 3) if turn_durs else 0.0
        min_turn   = round(min(turn_durs), 3) if turn_durs else 0.0
        distribution.append({
            "speaker":              spk,
            "duration_seconds":     round(dur, 3),
            "percentage":           round(dur / total_assigned_dur * 100.0, 2),
            "turn_count":           turn_count,
            "segment_count":        sum(t["segments"] for t in turns),
            "average_turn_sec":     avg_turn,
            "max_turn_sec":         max_turn,
            "min_turn_sec":         min_turn,
            "first_appearance_sec": round(turns[0]["start"], 3) if turns else None,
            "last_appearance_sec":  round(turns[-1]["end"], 3) if turns else None,
        })
    result["speaker_distribution"] = distribution

    # ── Speaker Turn Count & Turn Duration ──────────────────────────────────
    turn_count = len(all_turns)
    result["audio_duration_sec"] = round(audio_dur, 3) if audio_dur is not None else None
    result["segment_count"] = len(parsed)
    result["turn_count"] = turn_count
    result["speaker_change_count"] = max(0, turn_count - 1)
    result["avg_turn_duration_sec"] = (
        round(sum(t["end"] - t["start"] for t in all_turns) / turn_count, 3) if turn_count else 0.0
    )

    # ── Speaker Continuity (penalty for very short < 0.5s micro-segments) ───
    very_short = sum(1 for _, _, spk in parsed if spk in speaker_durations
                     for s, e in [next(((a, b) for a, b, sp in parsed if sp == spk), (0, 0))]
                     if e - s < 0.5)
    # Simpler: count micro-segments across entire parsed list
    micro_count = sum(1 for s, e, _ in parsed if (e - s) < 0.5)
    continuity_score = _clamp_pct(
        (1.0 - micro_count / max(1, len(parsed))) * 100.0,
        "speaker_continuity"
    )
    result["speaker_continuity_score"] = continuity_score
    result["micro_segment_count"] = micro_count

    # ── Diarization Processing Time ──────────────────────────────────────────
    # How long the diarization step RAN (wall clock) - not how long anyone spoke. ``diarization_duration_sec`` is the
    # historical key; ``diarization_processing_time_sec`` says what it is. Talk time is in ``speaker_distribution``.
    result["diarization_duration_sec"] = (
        round(speaker_duration_sec, 3) if speaker_duration_sec is not None else None
    )
    result["diarization_processing_time_sec"] = result["diarization_duration_sec"]

    # ── Speaker Runtime Quality Score ────────────────────────────────────────
    components: Dict[str, Optional[float]] = {
        "coverage":    coverage / 100.0 if coverage is not None else None,
        "assignment":  assignment_rate / 100.0 if assignment_rate is not None else None,
        "consistency": ts_consistency / 100.0 if ts_consistency is not None else None,
        "continuity":  continuity_score / 100.0 if continuity_score is not None else None,
    }
    result["speaker_runtime_quality_score"] = _weighted_score(components, SPK_W)
    result["speaker_quality_components"] = {
        k: round(v * 100, 2) if v is not None else None
        for k, v in components.items()
    }
    return result


# ---------------------------------------------------------------------------
# 3. Summary / MoM Runtime Quality Metrics
# ---------------------------------------------------------------------------

def summary_runtime_metrics(
    summary_text: Optional[str],
    transcript_text: Optional[str],
    agenda_topics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Reference-free summary grounding metrics.

    Measures how much of the generated summary is supported by
    transcript evidence — without requiring a reference summary.
    """
    result: Dict[str, Any] = {
        "evaluation_type": "REFERENCE_FREE",
        "formula_notes": {
            "summary_word_overlap_ratio": "|summary_words ∩ transcript_words| / |summary_words| × 100",
            "unsupported_sentence_rate": "sentences_with_no_transcript_word_match / total_sentences × 100",
            "topic_coverage_rate": "agenda_topics_mentioned_in_summary / total_topics × 100",
        },
    }

    if not summary_text or not summary_text.strip():
        result["summary_word_overlap_ratio"] = None
        result["summary_word_count"] = 0
        result["unsupported_sentence_rate"] = None
        result["topic_coverage_rate"] = None
        result["summary_runtime_quality_score"] = None
        return result

    summary_words = set(_tokenize_words(summary_text))
    tx_words      = set(_tokenize_words(transcript_text or ""))

    result["summary_word_count"] = len(summary_text.split())
    from backend.utils.helpers import count_transcript_words

    result["transcript_word_count"] = count_transcript_words(transcript_text)

    # Word overlap ratio
    if summary_words:
        overlap = len(summary_words & tx_words) / len(summary_words) * 100.0
    else:
        overlap = 0.0
    word_overlap = _clamp_pct(overlap, "summary_word_overlap_ratio")
    result["summary_word_overlap_ratio"] = word_overlap

    # Unsupported sentence rate
    sentences = [s.strip() for s in re.split(r"[.!?\n]", summary_text) if len(s.strip()) > 10]
    if sentences:
        unsupported = sum(
            1 for s in sentences
            if not any(w in tx_words for w in _tokenize_words(s) if len(w) > 4)
        )
        unsp_rate = _clamp_pct(unsupported / len(sentences) * 100.0, "unsupported_sentence_rate")
    else:
        unsupported = 0
        unsp_rate = None
    result["unsupported_sentence_rate"] = unsp_rate
    result["sentence_count"] = len(sentences)
    result["unsupported_sentence_count"] = unsupported

    # Topic coverage
    if agenda_topics:
        summary_lower = summary_text.lower()
        covered = sum(1 for t in agenda_topics if t.lower() in summary_lower)
        topic_cov = _clamp_pct(covered / len(agenda_topics) * 100.0, "topic_coverage_rate")
        result["topic_coverage_rate"] = topic_cov
        result["agenda_topics_covered"] = covered
        result["total_agenda_topics"] = len(agenda_topics)
    else:
        result["topic_coverage_rate"] = None

    # Composite runtime quality — higher overlap and lower unsupported = better
    score_components = [word_overlap]
    if unsp_rate is not None:
        score_components.append(100.0 - unsp_rate)
    if score_components:
        q_score = _clamp_pct(sum(score_components) / len(score_components), "summary_runtime_quality_score")
    else:
        q_score = None
    result["summary_runtime_quality_score"] = q_score
    result["sample_count"] = len(sentences)
    return result


def mom_runtime_metrics(mom_text: Optional[str]) -> Dict[str, Any]:
    """Reference-free MoM structural quality metrics."""
    result: Dict[str, Any] = {
        "evaluation_type": "REFERENCE_FREE",
        "formula_notes": {
            "section_count": "count of markdown headings (# / ## / ###) or numbered sections",
            "mom_word_count": "len(mom_text.split())",
        },
    }
    if not mom_text or not mom_text.strip():
        result["mom_word_count"] = 0
        result["section_count"] = 0
        return result

    result["mom_word_count"] = len(mom_text.split())
    headings = [l for l in mom_text.splitlines()
                if l.strip().startswith("#") or re.match(r"^\d+\.", l.strip())]
    result["section_count"] = len(headings)
    return result


# ---------------------------------------------------------------------------
# 4. Action Item Runtime Quality Metrics
# ---------------------------------------------------------------------------

def action_item_runtime_metrics(
    action_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Reference-free action item completeness and coverage metrics.

    Input: list of action item dicts (from DB ActionItem rows or LLM output).
    """
    result: Dict[str, Any] = {
        "evaluation_type": "REFERENCE_FREE",
        "formula_notes": {
            "field_completeness_pct": "populated_required_fields / (count × num_required_fields) × 100",
            "evidence_coverage_pct":  "items_with_evidence / count × 100",
            "assigned_to_coverage_pct": "items_with_owner / count × 100",
            "due_date_coverage_pct":    "items_with_due_date / count × 100",
            "priority_valid_pct": "items_where_priority ∈ {Low,Medium,High} / count × 100",
            "duplicate_rate_pct": "near-duplicate_items / count × 100 (first-6-word heuristic)",
        },
        "required_fields": list(_REQUIRED_AI_FIELDS),
        "valid_priorities": list(_VALID_PRIORITIES),
    }

    count = len(action_items)
    result["action_item_count"] = count
    result["sample_count"] = count

    if count == 0:
        result["field_completeness_pct"] = None
        result["evidence_coverage_pct"] = None
        result["assigned_to_coverage_pct"] = None
        result["due_date_coverage_pct"] = None
        result["priority_valid_pct"] = None
        result["duplicate_rate_pct"] = None
        result["action_item_runtime_quality_score"] = None
        return result

    # Field completeness
    total_fields = count * len(_REQUIRED_AI_FIELDS)
    populated = 0
    evidence_count = 0
    owner_count = 0
    due_date_count = 0
    priority_valid_count = 0
    task_prefixes: List[str] = []

    for ai in action_items:
        for field in _REQUIRED_AI_FIELDS:
            val = ai.get(field) or ai.get("task") if field == "task" else ai.get(field)
            # map legacy field names
            if field == "owner":
                val = ai.get("owner") or ai.get("assigned_to") or ai.get("assignee")
            if field == "deadline":
                val = ai.get("deadline") or ai.get("due_date") or ai.get("deadline_end")
            if val and str(val).strip() and str(val).strip().lower() not in (
                "not specified", "unassigned", "none", "null", ""
            ):
                populated += 1

        if ai.get("evidence") and str(ai["evidence"]).strip():
            evidence_count += 1
        owner_val = ai.get("owner") or ai.get("assigned_to") or ai.get("assignee") or ""
        if owner_val and str(owner_val).strip().lower() not in ("unassigned", "none", ""):
            owner_count += 1
        due_val = ai.get("due_date") or ai.get("deadline_end") or ai.get("deadline") or ""
        if due_val and str(due_val).strip().lower() not in ("not specified", "none", ""):
            due_date_count += 1
        pri = str(ai.get("priority") or "").strip().lower()
        if pri in _VALID_PRIORITIES:
            priority_valid_count += 1

        # Duplicate detection: first-6-word key of task/title
        task_text = str(ai.get("task") or ai.get("title") or "").lower()
        words = task_text.split()
        task_prefixes.append(" ".join(words[:6]) if words else "")

    dup_counter = Counter(p for p in task_prefixes if p)
    dup_count = sum(v - 1 for v in dup_counter.values() if v > 1)

    result["field_completeness_pct"]   = _clamp_pct(populated / total_fields * 100.0, "field_completeness_pct")
    result["evidence_coverage_pct"]    = _clamp_pct(evidence_count / count * 100.0,    "evidence_coverage_pct")
    result["assigned_to_coverage_pct"] = _clamp_pct(owner_count / count * 100.0,       "assigned_to_coverage_pct")
    result["due_date_coverage_pct"]    = _clamp_pct(due_date_count / count * 100.0,    "due_date_coverage_pct")
    result["priority_valid_pct"]       = _clamp_pct(priority_valid_count / count * 100.0, "priority_valid_pct")
    result["duplicate_rate_pct"]       = _clamp_pct(dup_count / count * 100.0,         "duplicate_rate_pct")
    result["duplicate_count"]          = dup_count

    # Composite score (higher = better): field completeness drives this
    score_components = [
        result["field_completeness_pct"],
        result["evidence_coverage_pct"],
        result["assigned_to_coverage_pct"],
        result["priority_valid_pct"],
    ]
    valid = [s for s in score_components if s is not None]
    ai_score = _clamp_pct(sum(valid) / len(valid), "action_item_runtime_quality_score") if valid else None
    # Penalise duplicates
    if ai_score is not None and result["duplicate_rate_pct"] is not None:
        ai_score = _clamp_pct(ai_score - result["duplicate_rate_pct"] * 0.5, "action_item_runtime_quality_score")
    result["action_item_runtime_quality_score"] = ai_score
    return result


# ---------------------------------------------------------------------------
# 5. Agenda / Decision Runtime Quality Metrics
# ---------------------------------------------------------------------------

def agenda_decision_runtime_metrics(
    agenda_key_decisions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Reference-free agenda & decision quality metrics.

    Input: list of AgendaKeyDecision rows as dicts.
    match_score values come from actual embedding cosine similarity
    computed by agenda_service — never hardcoded here.
    """
    result: Dict[str, Any] = {
        "evaluation_type": "REFERENCE_FREE",
        "formula_notes": {
            "decisions_with_evidence_pct": "decisions_with_non-null_evidence / total × 100",
            "avg_decision_match_score": "mean(match_score) from sentence-transformer cosine similarity",
            "unsupported_decision_pct": "decisions_with_no_evidence / total × 100",
        },
    }

    count = len(agenda_key_decisions)
    result["agenda_item_count"] = count
    result["sample_count"] = count

    if count == 0:
        result["decisions_with_evidence_pct"] = None
        result["unsupported_decision_pct"] = None
        result["avg_decision_match_score"] = None
        result["min_match_score"] = None
        result["max_match_score"] = None
        result["duplicate_agenda_pct"] = None
        return result

    evidence_count = 0
    match_scores: List[float] = []
    agenda_texts: List[str] = []

    for item in agenda_key_decisions:
        ev = item.get("evidence") or ""
        if ev and str(ev).strip():
            evidence_count += 1
        ms = item.get("match_score")
        if ms is not None:
            try:
                match_scores.append(float(ms))
            except (TypeError, ValueError):
                pass
        agenda_texts.append(str(item.get("agenda") or item.get("agenda_text") or "").lower().strip())

    result["decisions_with_evidence_pct"] = _clamp_pct(
        evidence_count / count * 100.0, "decisions_with_evidence_pct"
    )
    result["unsupported_decision_pct"] = _clamp_pct(
        (count - evidence_count) / count * 100.0, "unsupported_decision_pct"
    )

    # Match scores — from actual embedding model (agenda_service)
    if match_scores:
        result["avg_decision_match_score"] = round(sum(match_scores) / len(match_scores), 3)
        result["min_match_score"]          = round(min(match_scores), 3)
        result["max_match_score"]          = round(max(match_scores), 3)
        result["match_score_sample_count"] = len(match_scores)
    else:
        result["avg_decision_match_score"] = None
        result["min_match_score"]          = None
        result["max_match_score"]          = None
        result["match_score_sample_count"] = 0

    # Duplicate agenda ratio
    prefix_counter = Counter(" ".join(t.split()[:5]) for t in agenda_texts if t)
    dup_count = sum(v - 1 for v in prefix_counter.values() if v > 1)
    result["duplicate_agenda_pct"] = _clamp_pct(dup_count / count * 100.0, "duplicate_agenda_pct")
    result["duplicate_agenda_count"] = dup_count
    return result


# ---------------------------------------------------------------------------
# 6. End-to-End Processing Performance Metrics
# ---------------------------------------------------------------------------

def processing_performance_metrics(meeting: Any) -> Dict[str, Any]:
    """Assemble pipeline stage timing metrics from Meeting model columns.

    All values come from actual timestamp columns set by the pipeline.
    None means the stage timing was not recorded — never fabricated.
    """
    def _f(attr: str) -> Optional[float]:
        v = getattr(meeting, attr, None)
        if v is None:
            return None
        try:
            return _clamp_pos(float(v), attr)
        except (TypeError, ValueError, MetricValidationError):
            return None

    audio_dur = _f("duration")
    stt_dur   = _f("stt_duration_sec")
    total_dur = _f("total_processing_time_sec") or _f("processing_time")

    # RTF from stored fields
    rtf = None
    if stt_dur is not None and audio_dur and audio_dur > 0:
        rtf = round(stt_dur / audio_dur, 5)

    speaker_processing = _f("speaker_duration_sec")
    return {
        "evaluation_type": "RUNTIME_PERFORMANCE",
        "audio_duration_sec":       audio_dur,
        "upload_duration_sec":      _f("upload_duration_sec"),
        "stt_duration_sec":         stt_dur,
        # Wall-clock time the diarization step took - NOT how long anyone spoke (that is in the speaker block's
        # speaker_distribution). ``speaker_duration_sec`` is the historical key; the second name says what it is.
        "speaker_duration_sec":     speaker_processing,
        "speaker_processing_time_sec": speaker_processing,
        "summary_duration_sec":     _f("summary_duration_sec"),
        "mom_duration_sec":         _f("mom_duration_sec"),
        "action_items_duration_sec":_f("action_items_duration_sec"),
        "total_processing_time_sec": total_dur,
        "rtf":                      rtf,
        "file_size_bytes":          getattr(meeting, "file_size_bytes", None),
        "stt_word_count":           getattr(meeting, "stt_word_count", None),
        "stt_char_count":           getattr(meeting, "stt_char_count", None),
        "speaker_count":            getattr(meeting, "speaker_count", None),
    }


# ---------------------------------------------------------------------------
# 7. Diarization Quality Signal for pipelines.py
# ---------------------------------------------------------------------------

def diarization_quality_signal(
    segments: List[Dict[str, Any]],
    audio_duration_sec: float,
    diar_status: Optional[str],
) -> float:
    """Dynamic diarization quality signal for the pipeline weighted score.

    Replaces the hardcoded 95/60/20/80/50 constants.
    Returns a float in [0, 100] — NOT an accuracy metric.
    Clearly represents a runtime quality estimate.
    """
    if not segments:
        # No transcript at all — zero quality signal
        return 0.0

    metrics = speaker_runtime_metrics(segments, audio_duration_sec, None)
    score = metrics.get("speaker_runtime_quality_score")

    # Apply diar_status as a reliability multiplier only
    if diar_status == "FAILED":
        multiplier = 0.30
    elif diar_status == "DEGRADED":
        multiplier = 0.70
    else:
        multiplier = 1.00  # SUCCESS or unknown — use computed value as-is

    if score is None:
        # Fallback: partial signal from assignment rate only
        ar = metrics.get("speaker_assignment_rate")
        score = ar if ar is not None else 50.0

    return round(max(0.0, min(100.0, score * multiplier)), 1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b[\w''-]+\b")


def _tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower()) if text else []


def _weighted_score(
    components: Dict[str, Optional[float]],
    weights: Dict[str, float],
) -> Optional[float]:
    """Weighted mean of components that are not None.

    Each component must be in [0, 1] (higher = better).
    Returns score in [0, 100] or None if no components available.
    """
    total_w = 0.0
    weighted_sum = 0.0
    for key, val in components.items():
        if val is None:
            continue
        w = weights.get(key, 0.0)
        weighted_sum += val * w
        total_w += w
    if total_w == 0:
        return None
    return round(max(0.0, min(100.0, weighted_sum / total_w * 100.0)), 2)


# ---------------------------------------------------------------------------
# 8. Agenda Grounding Runtime Metrics
# ---------------------------------------------------------------------------

def agenda_runtime_metrics(
    agenda_items: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Reference-free grounding quality metrics for LLM-generated agenda items.

    Computed from actual AgendaKeyDecision rows and TranscriptSegment rows.
    No ground truth required. Returns None fields when inputs are insufficient.

    Metrics computed:
      - item_count             : total agenda items extracted
      - evidence_coverage_pct  : % of items with a non-empty evidence/source field
      - avg_confidence         : mean confidence score across items (0–1 scale)
      - low_confidence_pct     : % of items with confidence < 0.5
      - unsupported_pct        : % of items with no transcript overlap
      - duplicate_pct          : % of items whose title duplicates another item
      - timestamp_coverage_pct : % of items with valid start/end timestamps
      - agenda_runtime_quality_score : composite [0,100]
    """
    if not agenda_items:
        return {
            "evaluation_type": "REFERENCE_FREE",
            "item_count": 0,
            "evidence_coverage_pct": None,
            "avg_confidence": None,
            "low_confidence_pct": None,
            "unsupported_pct": None,
            "duplicate_pct": None,
            "timestamp_coverage_pct": None,
            "agenda_runtime_quality_score": None,
            "note": "no_agenda_items_generated",
        }

    n = len(agenda_items)

    # Evidence coverage — item has non-empty evidence/source/description field
    with_evidence = sum(
        1 for item in agenda_items
        if (item.get("evidence") or item.get("source") or item.get("description") or "").strip()
    )
    evidence_pct = _clamp_pct(with_evidence / n * 100.0, "evidence_coverage_pct")

    # Confidence
    conf_vals = [
        float(item["confidence"])
        for item in agenda_items
        if item.get("confidence") is not None
    ]
    avg_conf = round(sum(conf_vals) / len(conf_vals), 4) if conf_vals else None
    low_conf = _clamp_pct(
        sum(1 for c in conf_vals if c < 0.5) / len(conf_vals) * 100.0,
        "low_confidence_pct"
    ) if conf_vals else None

    # Duplicate detection — same normalised title
    titles = [
        re.sub(r"\s+", " ", (item.get("title") or item.get("topic") or item.get("text") or "")).strip().lower()
        for item in agenda_items
    ]
    unique_titles = set(titles)
    dup_count = n - len(unique_titles)
    dup_pct = _clamp_pct(dup_count / n * 100.0, "duplicate_pct")

    # Timestamp coverage
    with_ts = sum(
        1 for item in agenda_items
        if item.get("start_time") is not None or item.get("timestamp") is not None
    )
    ts_pct = _clamp_pct(with_ts / n * 100.0, "timestamp_coverage_pct")

    # Unsupported items — title/description has no word overlap with transcript
    tx_text = " ".join(
        (seg.get("text") or "") for seg in transcript_segments
    ).lower()
    tx_words = set(_tokenize_words(tx_text))

    unsupported = 0
    if tx_words:
        for item in agenda_items:
            item_text = " ".join([
                item.get("title") or "",
                item.get("topic") or "",
                item.get("description") or "",
                item.get("evidence") or "",
            ])
            item_words = set(_tokenize_words(item_text))
            meaningful = {w for w in item_words if len(w) > 3}
            if meaningful and not meaningful.intersection(tx_words):
                unsupported += 1
        unsupported_pct = _clamp_pct(unsupported / n * 100.0, "unsupported_pct")
    else:
        unsupported_pct = None  # Cannot measure without transcript

    # Composite quality score
    # Higher evidence + lower duplicates + lower unsupported + higher confidence
    components = {
        "evidence":    (evidence_pct / 100.0) if evidence_pct is not None else None,
        "no_dup":      (1.0 - dup_pct / 100.0) if dup_pct is not None else None,
        "supported":   (1.0 - unsupported_pct / 100.0) if unsupported_pct is not None else None,
        "confidence":  avg_conf if avg_conf is not None else None,
    }
    weights = {"evidence": 0.35, "no_dup": 0.20, "supported": 0.30, "confidence": 0.15}
    q_score = _weighted_score(components, weights)

    return {
        "evaluation_type": "REFERENCE_FREE",
        "item_count": n,
        "evidence_coverage_pct": evidence_pct,
        "avg_confidence": avg_conf,
        "low_confidence_pct": low_conf,
        "unsupported_pct": unsupported_pct,
        "duplicate_pct": dup_pct,
        "timestamp_coverage_pct": ts_pct,
        "agenda_runtime_quality_score": q_score,
        "note": "reference_free_grounding_signal. Upload reference agenda for precision/recall/F1.",
    }


# ---------------------------------------------------------------------------
# 9. Decisions Grounding Runtime Metrics
# ---------------------------------------------------------------------------

def decisions_runtime_metrics(
    decision_items: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Reference-free grounding quality metrics for LLM-generated decisions.

    Computed from actual AgendaKeyDecision rows (agenda_type='decision').
    No ground truth required.

    Metrics:
      - item_count                  : decisions extracted
      - evidence_coverage_pct       : % with non-empty evidence/source
      - timestamp_coverage_pct      : % with valid timestamp
      - speaker_attribution_pct     : % with a non-empty speaker/owner field
      - avg_confidence              : mean confidence
      - unsupported_pct             : % with no transcript word overlap
      - decisions_runtime_quality_score : composite [0,100]
    """
    if not decision_items:
        return {
            "evaluation_type": "REFERENCE_FREE",
            "item_count": 0,
            "evidence_coverage_pct": None,
            "timestamp_coverage_pct": None,
            "speaker_attribution_pct": None,
            "avg_confidence": None,
            "unsupported_pct": None,
            "decisions_runtime_quality_score": None,
            "note": "no_decision_items_generated",
        }

    n = len(decision_items)

    with_evidence = sum(
        1 for d in decision_items
        if (d.get("evidence") or d.get("source") or d.get("reason") or "").strip()
    )
    evidence_pct = _clamp_pct(with_evidence / n * 100.0, "evidence_coverage_pct")

    with_ts = sum(
        1 for d in decision_items
        if d.get("timestamp") is not None or d.get("start_time") is not None
    )
    ts_pct = _clamp_pct(with_ts / n * 100.0, "timestamp_coverage_pct")

    with_speaker = sum(
        1 for d in decision_items
        if (d.get("speaker") or d.get("owner") or d.get("decided_by") or "").strip()
    )
    speaker_pct = _clamp_pct(with_speaker / n * 100.0, "speaker_attribution_pct")

    conf_vals = [
        float(d["confidence"])
        for d in decision_items
        if d.get("confidence") is not None
    ]
    avg_conf = round(sum(conf_vals) / len(conf_vals), 4) if conf_vals else None

    # Unsupported check
    tx_text = " ".join((seg.get("text") or "") for seg in transcript_segments).lower()
    tx_words = set(_tokenize_words(tx_text))
    unsupported_pct: Optional[float] = None
    if tx_words:
        unsupported = 0
        for d in decision_items:
            d_text = " ".join([
                d.get("decision") or d.get("text") or d.get("title") or "",
                d.get("evidence") or "",
                d.get("reason") or "",
            ])
            d_words = set(w for w in _tokenize_words(d_text) if len(w) > 3)
            if d_words and not d_words.intersection(tx_words):
                unsupported += 1
        unsupported_pct = _clamp_pct(unsupported / n * 100.0, "unsupported_pct")

    components = {
        "evidence":   (evidence_pct / 100.0) if evidence_pct is not None else None,
        "timestamp":  (ts_pct / 100.0) if ts_pct is not None else None,
        "speaker":    (speaker_pct / 100.0) if speaker_pct is not None else None,
        "supported":  (1.0 - unsupported_pct / 100.0) if unsupported_pct is not None else None,
        "confidence": avg_conf if avg_conf is not None else None,
    }
    weights = {"evidence": 0.30, "timestamp": 0.15, "speaker": 0.20, "supported": 0.25, "confidence": 0.10}
    q_score = _weighted_score(components, weights)

    return {
        "evaluation_type": "REFERENCE_FREE",
        "item_count": n,
        "evidence_coverage_pct": evidence_pct,
        "timestamp_coverage_pct": ts_pct,
        "speaker_attribution_pct": speaker_pct,
        "avg_confidence": avg_conf,
        "unsupported_pct": unsupported_pct,
        "decisions_runtime_quality_score": q_score,
        "note": "reference_free_grounding_signal. Upload reference decisions for precision/recall/F1.",
    }


# ---------------------------------------------------------------------------
# 10. Audio Runtime Metrics (wraps media_probe + segment speech/silence ratios)
# ---------------------------------------------------------------------------

def audio_runtime_metrics(
    segment_dicts: List[Dict[str, Any]],
    audio_duration_sec: float,
    media_probe_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Combines ffmpeg/numpy media probe telemetry with segment-derived audio metrics.

    Segment-derived (always available when transcript exists):
      - speech_ratio       : speech duration / total audio duration
      - silence_ratio_seg  : estimated from segments not covering audio
      - speaker_count      : distinct speakers

    Media-probe-derived (available when ffmpeg + numpy installed):
      - snr_db, peak_db, rms_db, clipping_ratio, silence_ratio (signal-level)
      - codec, sample_rate, channels, bitrate, duration_sec
    """
    probe = media_probe_result or {}

    # Segment-derived speech coverage
    speech_dur = 0.0
    speakers: set = set()
    for seg in segment_dicts:
        try:
            start = float(seg.get("start_time") or seg.get("start") or 0.0)
            end = float(seg.get("end_time") or seg.get("end") or 0.0)
            spk = seg.get("speaker") or ""
            if end > start:
                speech_dur += end - start
            if spk:
                speakers.add(spk)
        except (TypeError, ValueError):
            continue

    aud_dur = audio_duration_sec or 0.0
    speech_ratio = _clamp_pct(
        speech_dur / aud_dur * 100.0, "speech_ratio"
    ) if aud_dur > 0 else None
    silence_ratio_seg = _clamp_pct(
        max(0.0, aud_dur - speech_dur) / aud_dur * 100.0, "silence_ratio_seg"
    ) if aud_dur > 0 else None

    return {
        "evaluation_type": "REFERENCE_FREE",
        # Segment-derived
        "speech_ratio_pct":       speech_ratio,
        "silence_ratio_seg_pct":  silence_ratio_seg,
        "speech_duration_sec":    round(speech_dur, 2) if speech_dur else None,
        "audio_duration_sec":     round(aud_dur, 2) if aud_dur else None,
        "detected_speaker_count": len(speakers),
        # Media-probe-derived (None when ffmpeg/numpy unavailable)
        "snr_db":          probe.get("snr_db"),
        "peak_db":         probe.get("peak_db"),
        "rms_db":          probe.get("rms_db"),
        "clipping_ratio":  probe.get("clipping_ratio"),
        "silence_ratio_signal": probe.get("silence_ratio"),  # signal-level (ffmpeg)
        "codec":           probe.get("codec"),
        "sample_rate":     probe.get("sample_rate"),
        "channels":        probe.get("channels"),
        "bitrate":         probe.get("bitrate"),
        "note": "reference_free_audio_telemetry. SNR/clipping from ffmpeg+numpy signal analysis.",
    }


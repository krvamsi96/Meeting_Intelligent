"""Real-time, reference-free quality scores: Transcript Quality, Summary Quality, RAG Groundedness,
Action Item Quality, Decision Quality and Agenda Coverage.

No ground truth exists for these meetings, so nothing here is WER/CER/DER/ROUGE or any other
reference-dependent metric. Every score is computed from THIS meeting's own transcript, summary
and chat traffic, on a 0-100 scale, together with a plain-English ``issues`` list explaining
what lowered it.

Design rules (same contract as ``reference_free_engine``):

* Pure functions over their arguments — no DB access, no module-level state, no caching. Two
  meetings evaluated back to back can never contaminate one another.
* A score is a weighted mean of *measured* sub-signals; a sub-signal that cannot be measured
  (e.g. speech coverage without an audio duration, or a constant placeholder ASR confidence)
  is dropped and the remaining weights renormalised — never padded with an invented number.
* Severe defects also apply a documented cap (a "gate") or a subtraction, so that a single
  catastrophic problem cannot hide behind many healthy averages.
* Embeddings are used where semantic support matters; every embedding step has a lexical
  fallback so a missing model degrades the score's precision instead of disabling it.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.evaluation import reference_free_engine as rfe
from backend.evaluation.metrics import _cosine

logger = logging.getLogger(__name__)

EVALUATION_TYPE = "reference_free"
METHOD_VERSION = "1.1"

_LABEL_BANDS = ((90.0, "Excellent"), (75.0, "Good"), (60.0, "Average"))


def _env(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def score_label(score: Optional[float]) -> str:
    """Excellent >= 90, Good >= 75, Average >= 60, otherwise Needs Improvement."""
    if score is None:
        return "Not Available"
    for floor, label in _LABEL_BANDS:
        if score >= floor:
            return label
    return "Needs Improvement"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _pct(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(_clamp01(x) * 100.0, 1)


def _ramp(x: float, lo: float, hi: float) -> float:
    """Linear 0..1 ramp: 0 at/below ``lo``, 1 at/above ``hi``."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return _clamp01((x - lo) / (hi - lo))


def _weighted(components: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
    """Weighted mean of 0..1 components -> 0..100; unmeasured (None) components are dropped."""
    total = total_w = 0.0
    for key, val in components.items():
        if val is None:
            continue
        w = weights.get(key, 0.0)
        if w <= 0:
            continue
        total += _clamp01(val) * w
        total_w += w
    return None if total_w <= 0 else total / total_w * 100.0


def _result(score: Optional[float], issues: List[str], **extra: Any) -> Dict[str, Any]:
    score = None if score is None else round(max(0.0, min(100.0, score)), 1)
    out: Dict[str, Any] = {"score": score, "label": extra.pop("label", None) or score_label(score), "issues": issues}
    out.update(extra)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Text utilities
# ═════════════════════════════════════════════════════════════════════════════
_STOP = frozenset(
    "a an the and or but if so of to in on at for with without from by as is are was were be been being am "
    "do does did done have has had having i you he she it we they me him her us them my your his its our their "
    "this that these those there here what which who whom whose when where why how not no yes can could will "
    "would shall should may might must just very really also then than too about into over under again up "
    "down out off any all some more most other such only own same get got going gonna like okay ok yeah right "
    "well um uh mm hmm actually basically because while during between through after before once each both "
    "few many much one two three".split()
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[a-z]+)?")
_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "s", "ly")


def _stem(word: str) -> str:
    w = word.lower()
    if w.endswith("'s"):
        w = w[:-2]
    for suf in _SUFFIXES:
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
            break
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text or "")


def _content_stems(text: str) -> List[str]:
    return [_stem(t) for t in _tokens(text) if len(t) >= 3 and t.lower() not in _STOP]


def _sentences(text: str) -> List[str]:
    return rfe._sentences(text)


_NUM_WORDS = {
    **{w: i for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen".split())},
    **{w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"},
}
_NUM_SCALE = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000, "lakh": 100_000, "crore": 10_000_000}
_DIGIT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers_in(text: str, include_small_words: bool = True) -> set:
    """Numeric values mentioned in ``text`` as normalised strings, digits AND spelled-out ("fifteen thousand").

    ``include_small_words=False`` ignores spelled-out numbers below ten ("one of them"), which are
    usually pronouns/quantifiers rather than figures worth verifying.
    """
    found = set()
    for m in _DIGIT_RE.finditer(text or ""):
        raw = m.group(0).replace(",", "").rstrip(".")
        if raw:
            found.add(raw.rstrip("0").rstrip(".") if "." in raw else raw)
    words = [w.lower() for w in _tokens(text)]
    i = 0
    while i < len(words):
        if words[i] in _NUM_WORDS or words[i] in _NUM_SCALE:
            total, current, j = 0, 0, i
            while j < len(words) and (words[j] in _NUM_WORDS or words[j] in _NUM_SCALE or words[j] == "and"):
                w = words[j]
                if w == "and":
                    if j + 1 >= len(words) or (words[j + 1] not in _NUM_WORDS and words[j + 1] not in _NUM_SCALE):
                        break
                elif w in _NUM_WORDS:
                    current += _NUM_WORDS[w]
                elif w == "hundred":
                    current = max(current, 1) * 100
                else:
                    total += max(current, 1) * _NUM_SCALE[w]
                    current = 0
                j += 1
            value = total + current
            if include_small_words or value >= 10:
                found.add(str(value))
            i = max(j, i + 1)
        else:
            i += 1
    return found


_COMMON_CAPS = frozenset(
    "I The A An AI API Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May "
    "June July August September October November December English OK".split()
)


def _unsupported_details(target: str, support_text: str, extra_allowed: str = "") -> Tuple[int, List[str]]:
    """(count_checked, unsupported) for numbers and proper nouns in ``target`` absent from ``support_text``.

    A *detail* is a number, a capitalised word that is not the first word of its sentence, or an
    acronym. It counts as supported when it appears in the support text (or in ``extra_allowed``,
    e.g. the user's question for a chat answer). Invented figures and names are the most
    reliable lexical hallucination indicator.
    """
    support_tokens = {t.lower() for t in _tokens(support_text)} | {t.lower() for t in _tokens(extra_allowed)}
    support_stems = {_stem(t) for t in support_tokens}
    support_nums = _numbers_in(support_text) | _numbers_in(extra_allowed)

    details: List[Tuple[str, str]] = []
    for num in _numbers_in(target, include_small_words=False):
        details.append(("num", num))
    for sent in _sentences(target):
        toks = _tokens(sent)
        for idx, tok in enumerate(toks):
            if tok.isdigit() or tok in _COMMON_CAPS:
                continue
            if len(tok) >= 2 and tok.isupper() and tok.isalpha():
                details.append(("name", tok))
            elif idx > 0 and tok[0].isupper() and tok[1:].islower() and len(tok) > 2:
                details.append(("name", tok))

    seen, checked, missing = set(), 0, []
    for kind, val in details:
        if (kind, val.lower()) in seen:
            continue
        seen.add((kind, val.lower()))
        checked += 1
        low = val.lower()
        ok = (val in support_nums) if kind == "num" else (low in support_tokens or _stem(low) in support_stems)
        if not ok:
            missing.append(val)
    return checked, missing


# ── Support index: windows of sentences -> embeddings + vocabulary ───────────────────────────
_MAX_WINDOWS = 500


class _SupportIndex:
    """Embeds a source text ONCE as short sentence windows for many support checks."""

    def __init__(self, text: str):
        sents = _sentences(text) or ([text.strip()] if text and text.strip() else [])
        step = max(1, math.ceil(len(sents) / _MAX_WINDOWS))
        win = 3 * step
        self.windows = [" ".join(sents[i:i + win]) for i in range(0, len(sents), win)] if sents else []
        self.embeddings = rfe._embed_batch(self.windows) if self.windows else []
        if len(self.embeddings) != len(self.windows):
            self.embeddings = []
        self.stems = set(_content_stems(text))
        self.tokens = {t.lower() for t in _tokens(text)}
        self.centroid: List[float] = []
        if self.embeddings:
            dim = len(self.embeddings[0])
            self.centroid = [sum(e[k] for e in self.embeddings) / len(self.embeddings) for k in range(dim)]

    @property
    def semantic(self) -> bool:
        return bool(self.embeddings)

    def lexical_support(self, sentence: str) -> Optional[float]:
        stems = set(_content_stems(sentence))
        if not stems:
            return None
        return len(stems & self.stems) / len(stems)

    def semantic_support(self, vec: Sequence[float]) -> float:
        return rfe._max_cosine(vec, self.embeddings) if self.embeddings else 0.0


def _sentence_support(sentences: Sequence[str], index: _SupportIndex) -> List[float]:
    """Per-sentence support in 0..1: best of lexical overlap and (scaled) embedding similarity."""
    vecs = rfe._embed_batch(list(sentences)) if index.semantic else []
    if vecs and len(vecs) != len(sentences):
        vecs = []
    out = []
    for i, s in enumerate(sentences):
        lex = index.lexical_support(s)
        sem = _ramp(index.semantic_support(vecs[i]), 0.25, 0.65) if vecs else None
        cands = [c for c in (lex, sem) if c is not None]
        out.append(max(cands) if cands else 0.0)
    return out


def _snippet(text: str, n: int = 90) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "..."


# ═════════════════════════════════════════════════════════════════════════════
# 1. TRANSCRIPT QUALITY
# ═════════════════════════════════════════════════════════════════════════════
_TQ_WEIGHTS = {
    "speech_coverage": _env("TQ_W_COVERAGE", 0.14),
    "non_empty": _env("TQ_W_NON_EMPTY", 0.08),
    "meaningful": _env("TQ_W_MEANINGFUL", 0.07),
    "sentence_completeness": _env("TQ_W_SENTENCES", 0.10),
    "repetition": _env("TQ_W_REPETITION", 0.15),
    "text_cleanliness": _env("TQ_W_CLEAN", 0.10),
    "timestamp_consistency": _env("TQ_W_TIMESTAMPS", 0.10),
    "segment_consistency": _env("TQ_W_SEGMENTS", 0.06),
    "language_consistency": _env("TQ_W_LANGUAGE", 0.10),
    "transcription_confidence": _env("TQ_W_CONFIDENCE", 0.10),
}
_COVERAGE_FULL = _env("TQ_COVERAGE_FULL", 0.50)   # >=50% of the recording covered counts as complete
_WPM_FULL = _env("TQ_WPM_FULL", 30.0)              # >=30 transcribed words/min counts as complete

_DANGLING = frozenset(
    "and or but the a an of to in on for with that which as by at from into about is are was were will also "
    "including because if while so then than".split()
)
_TERMINAL_RE = re.compile(r"[.!?][\"')\]]*\s*$")
_LOOP_TOKEN_RE = re.compile(r"\b(\w+)(?:\s+\1\b){3,}", re.IGNORECASE)              # "you you you you"
_LOOP_PHRASE_RE = re.compile(r"\b((?:\w+\s+){1,4}\w+)(?:\s+\1\b){2,}", re.IGNORECASE)  # "thank you thank you thank you"


def _to_seconds(v: Any) -> Optional[float]:
    """Seconds from a number, a numeric string, or 'MM:SS' / 'HH:MM:SS'."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    parts = s.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    return None


def _script_of(ch: str) -> str:
    cp = ord(ch)
    if cp < 0x250:
        return "latin"
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"
    if 0x0980 <= cp <= 0x09FF:
        return "bengali"
    if 0x0B80 <= cp <= 0x0BFF:
        return "tamil"
    if 0x0C00 <= cp <= 0x0C7F:
        return "telugu"
    if 0x0C80 <= cp <= 0x0CFF:
        return "kannada"
    if 0x0D00 <= cp <= 0x0D7F:
        return "malayalam"
    if 0x0600 <= cp <= 0x06FF:
        return "arabic"
    if 0x0400 <= cp <= 0x04FF:
        return "cyrillic"
    if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF:
        return "cjk"
    return "other"


def _merged_coverage(intervals: List[Tuple[float, float]]) -> float:
    """Total seconds covered by the UNION of intervals (overlapping speakers are not double counted)."""
    total = 0.0
    cur_s = cur_e = None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def _informative_confidence(values: List[float], n_segments: int) -> Optional[float]:
    """Mean ASR confidence in 0..1, ONLY when the values are a genuine per-segment signal.

    The pipeline stores a fixed default (0.85) when the STT engine reports no confidence; a
    constant column is a placeholder, not a measurement, and must never feed a score.
    """
    if n_segments == 0 or len(values) < max(3, int(0.3 * n_segments)):
        return None
    if len({round(v, 3) for v in values}) < 3 and (max(values) - min(values)) < 0.01:
        return None
    vals = [math.exp(v) if v < 0 else v for v in values]      # Whisper avg_logprob (<=0) -> probability
    return _clamp01(sum(vals) / len(vals))


def transcript_quality(
    segments: Sequence[Dict[str, Any]],
    *,
    audio_duration_sec: Optional[float] = None,
    language_confidence: Optional[float] = None,
    is_document: bool = False,
    expected_language: Optional[str] = "en",
) -> Dict[str, Any]:
    """0-100 quality of THIS meeting's transcript, from the transcript alone (no reference text).

    ``segments``: dicts with ``text``, ``start_time``, ``end_time``, ``speaker`` and optionally
    ``stt_confidence``. ``is_document``: uploaded documents have synthetic timestamps and no
    audio, so timestamp / coverage / confidence signals are skipped instead of being scored.
    """
    n = len(segments)
    texts = [(s.get("text") or "").strip() for s in segments]
    non_empty = [t for t in texts if t]
    word_count = sum(len(t.split()) for t in non_empty)
    signals: Dict[str, Any] = {
        "segment_count": n, "word_count": word_count, "empty_segment_count": n - len(non_empty),
        "is_document": is_document,
    }

    # ── Failed / empty transcript ────────────────────────────────────────────
    if n == 0 or not non_empty:
        reason = ("Transcript is empty: no transcript segments were produced (transcription failed or found no speech)."
                  if n == 0 else "Transcript is empty: every segment is blank (transcription produced no text).")
        return _result(0.0, [reason], label="Failed", status="failed", components={}, signals=signals)

    issues: List[str] = []
    comps: Dict[str, Optional[float]] = {}

    # ── Timing ───────────────────────────────────────────────────────────────
    parsed: List[Tuple[Optional[float], Optional[float]]] = [
        (_to_seconds(s.get("start_time")), _to_seconds(s.get("end_time"))) for s in segments
    ]
    timing_usable = (not is_document) and any(b is not None and b > 0 for _, b in parsed)
    duration = float(audio_duration_sec) if audio_duration_sec and audio_duration_sec > 0 and not is_document else None

    # 1. Completeness / speech coverage
    duration_mismatch = False
    if timing_usable and duration:
        valid_iv = [(a, b) for a, b in parsed if a is not None and b is not None and 0 <= a < b]
        max_end = max((b for _, b in valid_iv), default=0.0)
        duration_mismatch = max_end > duration * 1.10 + 2.0
        signals["audio_duration_sec"] = round(duration, 1)
        if duration_mismatch:
            # The transcript's timeline does not fit the media it claims to transcribe, so a coverage
            # percentage would be meaningless. Report the inconsistency (it also lowers the timestamp score).
            comps["speech_coverage"] = None
            signals["transcript_end_sec"] = round(max_end, 1)
            issues.append(f"Transcript timestamps run to {max_end / 60:.1f} min but the recording is only "
                          f"{duration / 60:.1f} min long - the transcript timeline does not match the media.")
        else:
            covered = _merged_coverage([(a, min(b, duration)) for a, b in valid_iv])
            cov = covered / duration
            wpm = word_count / (duration / 60.0)
            signals.update({"covered_seconds": round(covered, 1), "speech_coverage_pct": round(min(cov, 1.0) * 100, 1),
                            "words_per_minute": round(wpm, 1)})
            comps["speech_coverage"] = 0.5 * _clamp01(cov / _COVERAGE_FULL) + 0.5 * _clamp01(wpm / _WPM_FULL)
            if cov < 0.2:
                issues.append(f"Transcript covers only {cov * 100:.0f}% of the {duration / 60:.1f}-minute recording.")
            if duration >= 60 and wpm < 15:
                issues.append(f"Very little speech was transcribed ({wpm:.0f} words per minute).")
    else:
        comps["speech_coverage"] = None

    # 2. Non-empty / meaningful segments
    empty_ratio = (n - len(non_empty)) / n
    comps["non_empty"] = 1.0 - empty_ratio
    if empty_ratio > 0.05:
        issues.append(f"{n - len(non_empty)} of {n} segments are empty ({empty_ratio * 100:.0f}%).")
    meaningful = sum(1 for t in non_empty if len(_tokens(t)) >= 3)
    meaningful_ratio = meaningful / len(non_empty)
    comps["meaningful"] = _clamp01(meaningful_ratio / 0.5)          # back-channels ("yes", "ok") are normal up to ~50%
    signals["meaningful_segment_pct"] = round(meaningful_ratio * 100, 1)
    if meaningful_ratio < 0.3:
        issues.append(f"Most segments are fragments of fewer than 3 words ({(1 - meaningful_ratio) * 100:.0f}%).")

    # 3. Broken / incomplete sentences
    broken = 0
    for t in non_empty:
        toks = _tokens(t)
        last = toks[-1].lower() if toks else ""
        # ends on a comma/colon/dash/ellipsis-character or a trailing "..." — explicit escapes on purpose
        ends_open = t.rstrip()[-1:] in ",;:-–—…" or t.rstrip().endswith("...")
        no_punct = len(toks) > 8 and not _TERMINAL_RE.search(t)
        if ends_open or last in _DANGLING or no_punct:
            broken += 1
    broken_ratio = broken / len(non_empty)
    signals["broken_sentence_pct"] = round(broken_ratio * 100, 1)
    comps["sentence_completeness"] = 1.0 - _ramp(broken_ratio, 0.05, 0.50)   # <=5% is natural (turn cut-offs)
    if broken_ratio > 0.15:
        issues.append(f"{broken_ratio * 100:.0f}% of segments end mid-sentence or lack sentence punctuation.")

    # 4. Repetition (duplicate segments, ASR loops, repeated phrases)
    norm = [re.sub(r"\W+", " ", t.lower()).strip() for t in non_empty]
    dup = sum(1 for k, c in Counter(x for x in norm if len(x.split()) >= 3).items() if c > 1
              for _ in range(c - 1))
    dup_ratio = dup / len(non_empty)
    loops = sum(1 for t in non_empty if _LOOP_TOKEN_RE.search(t) or _LOOP_PHRASE_RE.search(t))
    loop_ratio = loops / len(non_empty)
    all_tok = [t.lower() for t in _tokens(" ".join(non_empty))]
    ngram_share = 0.0
    if len(all_tok) >= 40:
        grams = Counter(tuple(all_tok[i:i + 5]) for i in range(len(all_tok) - 4))
        repeated_positions = sum(c * 5 for g, c in grams.items() if c >= 3)
        ngram_share = min(1.0, repeated_positions / len(all_tok))
    p_dup, p_loop, p_ng = _clamp01(dup_ratio * 2), _clamp01(loop_ratio * 3), _ramp(ngram_share, 0.05, 0.40)
    comps["repetition"] = 1.0 - max(p_dup, p_loop, p_ng)
    signals.update({"duplicate_segment_pct": round(dup_ratio * 100, 1), "repetition_loop_pct": round(loop_ratio * 100, 1),
                    "repeated_phrase_share_pct": round(ngram_share * 100, 1)})
    if dup_ratio > 0.10:
        issues.append(f"{dup_ratio * 100:.0f}% of segments are exact duplicates of earlier segments.")
    if loop_ratio > 0.05:
        issues.append(f"{loops} segment(s) contain an ASR repetition loop (a word or phrase repeated many times).")
    if ngram_share > 0.20:
        issues.append(f"{ngram_share * 100:.0f}% of the transcript is made of repeated phrases.")

    # 5. Text cleanliness (abnormal characters). Measured as the SHARE OF SEGMENTS affected, not a per-character
    #    rate: a handful of corrupted characters in a long transcript still means those segments are damaged.
    integ = rfe.text_integrity([{"text": t} for t in non_empty])
    bad_char_segs = sum(1 for t in non_empty if rfe._CONTROL_CHAR_RE.search(t))
    bad_chars = sum(len(rfe._CONTROL_CHAR_RE.findall(t)) for t in non_empty)
    symbol_segs = sum(
        1 for t in non_empty
        if len(t) > 20 and len(rfe._NON_ALNUM_RE.findall(t.replace(" ", ""))) / len(t) > 0.15
    )
    per_segment = {
        "corrupted characters": bad_char_segs / len(non_empty),
        "mostly symbols": symbol_segs / len(non_empty),
        "malformed words": integ.get("malformed_word_ratio") or 0.0,
        "runs of repeated characters": integ.get("repeated_char_ratio") or 0.0,
    }
    comps["text_cleanliness"] = 1.0 - max(_clamp01(v * 2.0) for v in per_segment.values())
    signals["broken_character_count"] = bad_chars
    if bad_chars:
        issues.append(f"{bad_chars} corrupted/control character(s) found in {bad_char_segs} segment(s).")
    for what, v in per_segment.items():
        if what != "corrupted characters" and v > 0.05:
            issues.append(f"{v * 100:.0f}% of segments contain {what}.")

    # 6. Timestamp consistency
    if timing_usable:
        valid = [(a, b) for a, b in parsed if a is not None and b is not None and 0 <= a < b]
        valid_ratio = len(valid) / n
        starts = [a for a, b in parsed if a is not None]
        mono = (sum(1 for i in range(1, len(starts)) if starts[i] >= starts[i - 1]) / (len(starts) - 1)) if len(starts) > 1 else 1.0
        within = None
        if duration:
            within = sum(1 for a, b in valid if b <= duration + 1.0) / max(1, len(valid))
        overlaps = sum(1 for i in range(1, len(valid)) if valid[i][0] < valid[i - 1][1] - 2.0)
        overlap_ratio = overlaps / max(1, len(valid) - 1)
        parts = {"v": (valid_ratio, 0.4), "m": (mono, 0.3), "o": (1.0 - _clamp01(overlap_ratio * 2), 0.15)}
        if within is not None:
            parts["w"] = (within, 0.15)
        ts_comp = sum(v * w for v, w in parts.values()) / sum(w for _, w in parts.values())
        if within is not None:
            # A timeline that does not fit inside the media it transcribes is inconsistent as a whole,
            # not merely one weak sub-check among four.
            ts_comp *= 0.3 + 0.7 * within
        comps["timestamp_consistency"] = ts_comp
        signals.update({"valid_timestamp_pct": round(valid_ratio * 100, 1), "in_order_pct": round(mono * 100, 1)})
        if valid_ratio < 0.9:
            issues.append(f"{(1 - valid_ratio) * 100:.0f}% of segments have invalid timestamps (missing or end before start).")
        if mono < 0.95:
            issues.append("Segment timestamps are out of order.")
        if within is not None and within < 0.95 and not duration_mismatch:
            issues.append("Some segments end after the recording's duration.")
        if overlap_ratio > 0.3:
            issues.append(f"{overlap_ratio * 100:.0f}% of segments overlap the previous one by more than 2 seconds.")
    else:
        comps["timestamp_consistency"] = None

    # 7. Segment consistency
    labelled = sum(1 for s in segments if (s.get("speaker") or "").strip()) / n
    micro = sum(1 for t in non_empty if len(_tokens(t)) < 2) / len(non_empty)
    comps["segment_consistency"] = 0.5 * labelled + 0.5 * (1.0 - _ramp(micro, 0.2, 0.6))
    signals.update({"speaker_labelled_pct": round(labelled * 100, 1), "micro_segment_pct": round(micro * 100, 1)})
    if labelled < 0.95:
        issues.append(f"{(1 - labelled) * 100:.0f}% of segments have no speaker label.")
    if micro > 0.4:
        issues.append(f"{micro * 100:.0f}% of segments are single-word fragments.")

    # 8. Language consistency
    script_chars: Counter = Counter()
    per_seg_share: List[float] = []
    for t in non_empty:
        c = Counter(_script_of(ch) for ch in t if ch.isalpha())
        script_chars.update(c)
        tot = sum(c.values())
        if tot:
            per_seg_share.append(c.most_common(1)[0][1] / tot)
    total_alpha = sum(script_chars.values())
    if total_alpha:
        dominant = script_chars.most_common(1)[0][0]
        non_latin = 1.0 - script_chars.get("latin", 0) / total_alpha
        mixed = sum(1 for s in per_seg_share if s < 0.8) / max(1, len(per_seg_share))
        script_comp = 1.0 - max(_ramp(non_latin, 0.0, 0.2) if (expected_language or "en") == "en" else 0.0,
                                _ramp(mixed, 0.1, 0.5))
        # A confidence of exactly 0.0 is how rows written before the fix stored "no detector ran": not a measurement.
        lc = language_confidence if (language_confidence is not None and 0.0 < language_confidence <= 1.0) else None
        # The detector's confidence reads low even for clean English audio in this pipeline, so on its own it is
        # a weak hint: it only nudges the component, and is reported as an issue only when the text itself
        # corroborates a language problem (non-English or mixed scripts).
        comps["language_consistency"] = script_comp if lc is None else 0.85 * script_comp + 0.15 * lc
        signals.update({"dominant_script": dominant, "non_latin_char_pct": round(non_latin * 100, 1),
                        "mixed_script_segment_pct": round(mixed * 100, 1), "language_detection_confidence": lc})
        if (expected_language or "en") == "en" and non_latin > 0.05:
            issues.append(f"{non_latin * 100:.0f}% of the text is not in the expected (English) script - untranslated speech.")
        elif mixed > 0.2:
            issues.append(f"{mixed * 100:.0f}% of segments mix several scripts/languages.")
        if lc is not None and lc < 0.5 and (non_latin > 0.02 or mixed > 0.1):
            issues.append(f"Language detection confidence is low ({lc * 100:.0f}%) and the text shows language problems.")
    else:
        comps["language_consistency"] = None

    # 9. ASR confidence — only when it is a real per-segment signal
    conf = None
    if not is_document:
        vals = []
        for s in segments:
            v = s.get("stt_confidence")
            try:
                if v is not None:
                    vals.append(float(v))
            except (TypeError, ValueError):
                pass
        conf = _informative_confidence(vals, n)
        signals["stt_confidence_available"] = conf is not None
        if conf is None and vals:
            signals["stt_confidence_note"] = "constant placeholder value — not used"
    comps["transcription_confidence"] = conf
    if conf is not None:
        signals["mean_stt_confidence"] = round(conf, 3)
        if conf < 0.6:
            issues.append(f"Average speech-recognition confidence is low ({conf * 100:.0f}%).")

    score = _weighted(comps, _TQ_WEIGHTS) or 0.0

    # ── Gates: one catastrophic defect cannot hide behind healthy averages ───
    cap, cap_why = 100.0, ""
    def _gate(limit: float, why: str):
        nonlocal cap, cap_why
        if limit < cap:
            cap, cap_why = limit, why
    if word_count < 10:
        _gate(30.0, "the transcript has almost no content")
    elif word_count < 30:
        _gate(70.0, "the transcript is very short")
    if empty_ratio >= 0.9:
        _gate(20.0, "nearly all segments are empty")
    elif empty_ratio >= 0.5:
        _gate(40.0, "half or more of the segments are empty")
    if max(p_dup, p_loop) >= 0.6:
        _gate(55.0, "severe repetition")
    if comps.get("speech_coverage") is not None and (duration or 0) >= 60:
        if signals.get("speech_coverage_pct", 100) < 5:
            _gate(45.0, "almost none of the recording was transcribed")
        elif signals.get("speech_coverage_pct", 100) < 20:
            _gate(55.0, "most of the recording was not transcribed")
        # ``speech_coverage_pct`` is the share of the TIMELINE the segments span - one segment from 0:00 to the end
        # is 100 % whatever it contains. Words per minute is what shows whether speech was actually transcribed
        # (measured: a 30-minute recording with 259 words = 8.5 wpm scored "Excellent 92.8" next to its own issue
        # "Very little speech was transcribed").
        wpm_signal = signals.get("words_per_minute")
        if wpm_signal is not None:
            if wpm_signal < 15:
                _gate(55.0, f"only {wpm_signal:.0f} words per minute were transcribed")
            elif wpm_signal < _WPM_FULL:
                _gate(74.0, f"only {wpm_signal:.0f} words per minute were transcribed")
    if signals.get("non_latin_char_pct", 0) >= 50 and (expected_language or "en") == "en":
        _gate(50.0, "most of the text is untranslated")
    if score > cap:
        issues.append(f"Score capped at {cap:.0f} because {cap_why}.")
        score = cap

    return _result(score, issues, status="ok",
                   components={k: _pct(v) for k, v in comps.items()}, signals=signals)


# ═════════════════════════════════════════════════════════════════════════════
# 2. SUMMARY QUALITY
# ═════════════════════════════════════════════════════════════════════════════
_SQ_WEIGHTS = {
    "factual_consistency": _env("SQ_W_FACTUAL", 0.28),
    "unsupported_details": _env("SQ_W_DETAILS", 0.12),
    "topic_coverage": _env("SQ_W_COVERAGE", 0.20),
    "relevance": _env("SQ_W_RELEVANCE", 0.10),
    "conciseness": _env("SQ_W_CONCISE", 0.10),
    "redundancy": _env("SQ_W_REDUNDANCY", 0.08),
    "readability": _env("SQ_W_READABILITY", 0.12),
}
# Points subtracted for content that must NOT be in a summary (per the product rules).
_SQ_PENALTIES = {
    "speaker_label": 25.0,      # Speaker_0 / SPEAKER_01 / speaker IDs
    "dialogue_format": 20.0,    # "Label: text" lines / speaker-by-speaker conversation
    "timestamp": 10.0,
    "verbatim_copy": 20.0,      # raw transcript
    "mom_or_metadata": 10.0,
    "structured_output": 10.0,
    "non_english": 15.0,
    "markdown": 5.0,
    "speaker_by_speaker": 10.0,
    "too_long": 20.0,           # maximum; scaled by how far past the size limit the summary runs
}
_PENALTY_TEXT = {
    "speaker_label": "Summary contains speaker labels/IDs ({d}).",
    "dialogue_format": "Summary is formatted like a transcript/dialogue ({d}).",
    "timestamp": "Summary contains timestamps ({d}).",
    "verbatim_copy": "Summary copies long stretches of the raw transcript instead of summarising it.",
    "mom_or_metadata": "Summary contains minutes-of-meeting style sections or metadata/IDs.",
    "structured_output": "Summary is structured output (JSON / per-speaker breakdown) instead of prose.",
    "non_english": "Summary is not in English.",
    "markdown": "Summary contains markdown headings/bullets instead of plain prose.",
    "speaker_by_speaker": "Summary reads as a speaker-by-speaker account of who said what.",
    "too_long": "",             # already reported by the conciseness check
}
_MD_RE = re.compile(r"(?m)^\s*(?:#{1,6}\s|[-*•]\s|\d{1,2}[.)]\s)|\*\*")
_SPEAKER_SAYS_RE = re.compile(
    r"(?m)(?:^|(?<=[.!?]\s))(?:the\s+(?:first|second|third|other)\s+speaker|speaker\s*\w+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+"
    r"(?:said|stated|mentioned|asked|replied|responded|added|explained|noted|suggested|agreed|argued|pointed out)\b"
)


def _placeholder_summary(text: str) -> bool:
    t = (text or "").strip().lower()
    return t.startswith("not available") or "summary unavailable" in t or t.startswith("<")


def _salient_terms(text: str, k: int) -> List[str]:
    """Top-k topic terms of a transcript: frequent stems, boosted when spread across the whole meeting."""
    sents = _sentences(text)
    if not sents:
        return []
    n_chunks = 8 if len(sents) >= 16 else max(1, len(sents) // 2)
    size = max(1, math.ceil(len(sents) / n_chunks))
    freq: Counter = Counter()
    where: Dict[str, set] = {}
    for ci in range(0, len(sents), size):
        for st in _content_stems(" ".join(sents[ci:ci + size])):
            freq[st] += 1
            where.setdefault(st, set()).add(ci)
    min_tf = 2 if len(sents) >= 6 else 1
    scored = [(tf * (1.0 + math.log(len(where[st]))), st) for st, tf in freq.items() if tf >= min_tf and len(st) >= 4]
    scored.sort(reverse=True)
    return [st for _, st in scored[:k]]


def summary_quality(
    summary_text: Optional[str],
    transcript_text: Optional[str],
    *,
    agenda_topics: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """0-100 quality of THIS meeting's summary, judged only against THIS meeting's transcript.

    The summary is compared with the transcript's content WITHOUT its speaker labels, so a leaked
    "Speaker_1" can never count as "supported" just because the raw transcript lines contain it.
    """
    from backend.services.summary_service import (
        check_summary, normalize_summary_text, plan_summary_length, prepare_summary_source,
    )

    src = prepare_summary_source(transcript_text or "")
    summary = (summary_text or "").strip()
    signals: Dict[str, Any] = {"summary_word_count": len(summary.split()), "transcript_word_count": src.word_count}

    if src.word_count < 5:
        return _result(None, [], label="Not Available", status="not_available",
                       reason="There is no transcript content to evaluate the summary against.",
                       components={}, signals=signals)
    if not summary:
        return _result(0.0, ["No summary was generated for this meeting."], label="Failed", status="failed",
                       components={}, signals=signals)
    if _placeholder_summary(summary):
        return _result(0.0, ["Summary generation failed (a placeholder was stored instead of a summary)."],
                       label="Failed", status="failed", components={}, signals=signals)

    plan = plan_summary_length(src.word_count)
    # Content checks run on the summary's prose (markdown headings/bullets/preamble stripped, so a
    # "### Executive Summary" title is not judged as a factual statement); the rule checks further
    # down (labels, timestamps, markdown, dialogue...) run on the summary exactly as stored.
    raw_summary = summary
    summary = normalize_summary_text(raw_summary, dedupe=False)[0] or raw_summary
    sents = _sentences(summary) or [summary]
    words = len(summary.split())
    issues: List[str] = []
    comps: Dict[str, Optional[float]] = {}
    index = _SupportIndex(src.text)
    signals["semantic_checks"] = index.semantic

    # 1. Factual consistency: is each statement supported by the transcript?
    support = _sentence_support(sents, index)
    comps["factual_consistency"] = sum(support) / len(support)
    weak = [s for s, v in zip(sents, support) if v < 0.5]
    signals.update({"summary_sentence_count": len(sents), "unsupported_sentence_count": len(weak)})
    for s in weak[:2]:
        issues.append(f"Statement not supported by the transcript: '{_snippet(s)}'")
    if len(weak) > 2:
        issues.append(f"{len(weak)} of {len(sents)} summary statements have little support in the transcript.")

    # 2. Hallucination indicators: invented numbers / names / acronyms
    checked, missing = _unsupported_details(summary, src.text)
    comps["unsupported_details"] = None if checked == 0 else 1.0 - len(missing) / checked
    signals["checked_details"] = checked
    if missing:
        issues.append("Details in the summary that do not appear in the transcript: " + ", ".join(f"'{m}'" for m in missing[:5]) + ".")

    # 3. Important-topic coverage
    k = int(max(4, min(12, round(math.sqrt(src.word_count) / 3))))
    terms = _salient_terms(src.text, k)
    sum_stems = set(_content_stems(summary))
    cov_terms = (sum(1 for t in terms if t in sum_stems) / len(terms)) if terms else None
    agenda = [a for a in (agenda_topics or []) if a and str(a).strip()]
    cov_agenda = rfe._coverage_ratio(agenda, summary) if agenda else None
    parts = [_clamp01(c / 0.6) for c in (cov_terms,) if c is not None] + ([cov_agenda] if cov_agenda is not None else [])
    comps["topic_coverage"] = (sum(parts) / len(parts)) if parts else None
    signals.update({"salient_terms_checked": len(terms),
                    "salient_term_coverage_pct": None if cov_terms is None else round(cov_terms * 100, 1),
                    "agenda_topic_coverage_pct": None if cov_agenda is None else round(cov_agenda * 100, 1)})
    if comps["topic_coverage"] is not None and comps["topic_coverage"] < 0.5:
        missed = [t for t in terms if t not in sum_stems][:4]
        issues.append("Important topics from the meeting are missing from the summary"
                      + (f" (e.g. {', '.join(missed)})." if missed else "."))

    # 4. Relevance: does the summary stay on the meeting's subject?
    rel = None
    if index.semantic:
        svec = rfe._embed_one(summary[:2000])
        if svec and index.centroid:
            rel = _ramp(_cosine(svec, index.centroid), 0.20, 0.60)
    if rel is None:
        a, b = set(_content_stems(summary)), index.stems
        rel = _ramp(len(a & b) / max(1, len(a)), 0.30, 0.90) if a else None
    comps["relevance"] = rel
    if rel is not None and rel < 0.4:
        issues.append("The summary is only loosely related to what was discussed in the meeting.")

    # 5. Conciseness (relative to the meeting's size)
    ratio = words / max(1, src.word_count)
    limit = plan.max_words * 1.15
    concise = 1.0 if words <= limit else 1.0 - _ramp(words, limit, 3.0 * plan.target_words)
    if ratio >= 0.5 and src.word_count >= 40:
        concise = 0.0
    if src.word_count >= 60 and words < plan.min_words * 0.5:
        concise = min(concise, words / (plan.min_words * 0.5))
        issues.append(f"Summary is too short ({words} words) to cover a {src.word_count}-word meeting.")
    comps["conciseness"] = concise
    signals.update({"target_words": plan.target_words, "max_words": plan.max_words,
                    "compression_ratio": round(ratio, 3)})
    if words > limit:
        issues.append(f"Summary is too long: {words} words for a {src.word_count}-word meeting (target about {plan.target_words}).")

    # 6. Redundancy
    stem_sets = [set(_content_stems(s)) for s in sents]
    dup_pairs = 0
    for i in range(1, len(stem_sets)):
        for j in range(i):
            a, b = stem_sets[i], stem_sets[j]
            if len(a) >= 4 and len(b) >= 4 and len(a & b) / len(a | b) >= 0.7:
                dup_pairs += 1
                break
    dup_frac = dup_pairs / max(1, len(sents))
    toks = [t.lower() for t in _tokens(summary)]
    grams = Counter(tuple(toks[i:i + 4]) for i in range(max(0, len(toks) - 3)))
    rep_share = min(1.0, sum(c * 4 for c in grams.values() if c >= 2) / max(1, len(toks)))
    comps["redundancy"] = 1.0 - max(_clamp01(dup_frac * 2), _ramp(rep_share, 0.05, 0.40))
    signals["repeated_sentence_count"] = dup_pairs
    if dup_pairs:
        issues.append(f"{dup_pairs} summary sentence(s) repeat information already stated.")

    # 7. Readability / grammar heuristics
    lens = [len(_tokens(s)) for s in sents]
    avg_len = sum(lens) / len(lens)
    penalties = 0.0
    if avg_len > 40:
        penalties += _ramp(avg_len, 40, 70) * 0.5
        issues.append(f"Sentences are very long on average ({avg_len:.0f} words), which hurts readability.")
    elif avg_len < 4:
        penalties += 0.4
        issues.append("Summary is made of fragments rather than full sentences.")
    cap_ok = sum(1 for s in sents if s[:1].isupper() or not s[:1].isalpha()) / len(sents)
    penalties += (1.0 - cap_ok) * 0.3
    stutter = len(re.findall(r"\b(\w+)\s+\1\b", summary, re.IGNORECASE))
    penalties += min(0.3, stutter * 0.1)
    if stutter:
        issues.append(f"{stutter} repeated word(s) in a row (e.g. 'the the').")
    if not _TERMINAL_RE.search(summary):
        penalties += 0.2
        issues.append("Summary does not end with a complete sentence.")
    comps["readability"] = _clamp01(1.0 - penalties)
    signals["avg_sentence_words"] = round(avg_len, 1)

    score = _weighted(comps, _SQ_WEIGHTS) or 0.0

    # ── Content that must NOT appear in a summary: subtract, and explain ─────
    codes = {v.code: v for v in check_summary(raw_summary, plan, source_text=src.text, labels=src.labels,
                                              id_labels=src.id_labels, raw=raw_summary)}
    applied: Dict[str, float] = {}
    for code, pts in _SQ_PENALTIES.items():
        v = codes.get(code)
        hit = v is not None
        detail = v.detail if v is not None else ""
        if code == "markdown":
            hit = bool(_MD_RE.search(raw_summary))
        elif code == "speaker_by_speaker":
            hit = len(_SPEAKER_SAYS_RE.findall(raw_summary)) >= 3 and "speaker_label" not in codes
        if hit:
            if code == "too_long":
                # Faithful but bloated is still a defect: 5 points just past the limit, rising to 20.
                pts = 5.0 + (pts - 5.0) * _ramp(words, plan.hard_max_words, 3.0 * plan.hard_max_words)
            applied[code] = round(pts, 1)
            if _PENALTY_TEXT[code]:
                issues.append(_PENALTY_TEXT[code].format(d=detail or "see summary"))
    if "speaker_label" in applied and "dialogue_format" in applied:
        applied["dialogue_format"] = 0.0          # same defect — count it once
    score -= sum(applied.values())
    signals["rule_violations"] = sorted(k for k, v in applied.items() if v)

    if comps["factual_consistency"] < 0.4 and score > 60.0:
        issues.append("Score capped at 60 because most of the summary is not supported by the transcript.")
        score = 60.0
    # A faithful but incomplete summary is not an excellent one: when less than 60 % of the meeting's topics
    # (salient terms / agenda topics) are covered the score cannot reach "Good" or "Excellent".
    if comps["topic_coverage"] is not None and comps["topic_coverage"] < 0.6 and score > 74.0:
        issues.append("Score capped at 74 because the summary leaves out much of what the meeting covered.")
        score = 74.0

    return _result(score, issues, status="ok", components={k: _pct(v) for k, v in comps.items()},
                   penalties=applied, signals=signals)


# ═════════════════════════════════════════════════════════════════════════════
# 3. RAG GROUNDEDNESS
# ═════════════════════════════════════════════════════════════════════════════
_RAG_WEIGHTS = {
    "context_relevance": _env("RAG_W_RELEVANCE", 0.15),
    "answer_context_consistency": _env("RAG_W_CONSISTENCY", 0.15),
    "factual_support": _env("RAG_W_SUPPORT", 0.40),
    "unsupported_details": _env("RAG_W_DETAILS", 0.10),
    "context_coverage": _env("RAG_W_COVERAGE", 0.10),
    "meeting_isolation": _env("RAG_W_ISOLATION", 0.10),
}
_ABSTAIN_RE = re.compile(
    r"\b(?:not (?:mentioned|discussed|specified|stated|covered|available|provided|found)|no (?:information|mention|details?|record)|"
    r"(?:could|can)(?:not|n't) (?:find|answer)|does(?:n't| not) (?:mention|contain|include|say|specify)|"
    r"(?:isn't|is not|wasn't|was not) (?:mentioned|discussed|specified)|i (?:don't|do not) (?:know|have))\b",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def rag_answer_groundedness(
    *,
    question: str,
    answer: str,
    context_chunks: Sequence[Any],
    source_chunks: Sequence[Dict[str, Any]] = (),
    meeting_id: Optional[int] = None,
) -> Dict[str, Any]:
    """0-100 groundedness of ONE chatbot answer in the context retrieved for it.

    ``context_chunks``: everything the LLM was actually shown (vector chunks, MoM, action items,
    metadata, ...). ``source_chunks``: the vector-search chunks with provenance
    (``meeting_id``, ``similarity_score``). Any chunk that carries a ``meeting_id`` other than
    ``meeting_id`` belongs to a DIFFERENT meeting: it is excluded from the support context and
    reported as an isolation breach.
    """
    answer = (answer or "").strip()
    issues: List[str] = []
    if not answer:
        return _result(None, [], label="Not Available", status="not_evaluated", reason="Empty answer.")

    # ── Meeting isolation: never let another meeting's chunks support this answer ────
    foreign_texts, own_tagged, foreign_ids, foreign_count = set(), 0, set(), 0
    sims: List[float] = []
    for c in source_chunks or []:
        if not isinstance(c, dict):
            continue
        cid = c.get("meeting_id")
        if meeting_id is not None and cid is not None and str(cid) != str(meeting_id):
            foreign_count += 1
            foreign_ids.add(str(cid))
            foreign_texts.add(_norm(str(c.get("text") or c.get("content") or "")))
        else:
            if cid is not None:
                own_tagged += 1
            s = c.get("similarity_score", c.get("score"))
            try:
                if s is not None:
                    sims.append(float(s))
            except (TypeError, ValueError):
                pass
    ctx_texts: List[str] = []
    for c in context_chunks or []:
        t = c if isinstance(c, str) else str((c or {}).get("text") or (c or {}).get("content") or "")
        if t.strip() and _norm(t) not in foreign_texts:
            ctx_texts.append(t)
    foreign_chunks = foreign_count
    isolation = 1.0 if not foreign_chunks else own_tagged / max(1, own_tagged + foreign_chunks)
    signals: Dict[str, Any] = {
        "context_chunk_count": len(ctx_texts), "foreign_meeting_chunks_excluded": foreign_chunks,
        "answer_word_count": len(answer.split()),
    }
    if foreign_chunks:
        issues.append(f"{foreign_chunks} retrieved chunk(s) belonged to a different meeting "
                      f"({', '.join(sorted(foreign_ids))}) and were excluded from the support context.")

    if not ctx_texts:
        issues.append("No retrieved context was available to support this answer.")
        if _ABSTAIN_RE.search(answer):
            # Saying "the meeting doesn't cover this" with nothing retrieved makes no unsupported claim.
            score = _weighted({"factual_support": 1.0, "meeting_isolation": isolation}, _RAG_WEIGHTS) or 0.0
            return _result(score, issues, status="ok", answer_type="abstention",
                           components={"meeting_isolation": _pct(isolation)}, signals=signals)
        # A substantive answer with zero supporting context is, by definition, ungrounded.
        return _result(0.0, issues, status="ok", answer_type="ungrounded",
                       components={"meeting_isolation": _pct(isolation)}, signals=signals)

    context = "\n".join(ctx_texts)
    index = _SupportIndex(context)
    abstained = bool(_ABSTAIN_RE.search(answer))
    claims = [s for s in _sentences(answer) if len(_tokens(s)) >= 4 and not s.rstrip().endswith("?")
              and not _ABSTAIN_RE.search(s)]
    comps: Dict[str, Optional[float]] = {"meeting_isolation": isolation}

    # Context relevance: is what was retrieved about what was asked?
    qvec = rfe._embed_one(question[:1000]) if index.semantic and question.strip() else []
    if qvec:
        top = sorted((rfe._max_cosine(qvec, [e]) for e in index.embeddings), reverse=True)[:3]
        rel = _ramp(sum(top) / len(top), 0.20, 0.60)
        if sims:
            rel = 0.5 * rel + 0.5 * _ramp(sum(sims) / len(sims), 0.25, 0.65)
    else:
        qs = set(_content_stems(question))
        rel = _ramp(len(qs & index.stems) / len(qs), 0.2, 0.8) if qs else None

    # Context coverage: does the retrieved context contain what the question is about?
    qterms = list(dict.fromkeys(_content_stems(question)))
    cov = (sum(1 for t in qterms if t in index.stems) / len(qterms)) if len(qterms) >= 2 else None

    if abstained and not claims:
        # An honest "the meeting doesn't say" is grounded behaviour; only isolation is judged.
        comps.update({"factual_support": 1.0})
        score = _weighted(comps, _RAG_WEIGHTS) or 0.0
        return _result(score, issues, status="ok", answer_type="abstention",
                       components={k: _pct(v) for k, v in comps.items()}, signals=signals)

    comps["context_relevance"] = rel
    comps["context_coverage"] = None if cov is None else _clamp01(cov / 0.7)
    if rel is not None and rel < 0.4:
        issues.append("The retrieved context is only loosely related to the question.")
    if cov is not None and cov < 0.4:
        issues.append("The retrieved context does not cover the key terms of the question.")

    # Answer/context semantic consistency (holistic) and per-claim factual support
    avec = rfe._embed_one(answer[:2000]) if index.semantic else []
    if avec and index.centroid:
        top = sorted((rfe._max_cosine(avec, [e]) for e in index.embeddings), reverse=True)[:3]
        comps["answer_context_consistency"] = _ramp(sum(top) / len(top), 0.25, 0.70)
    else:
        a = set(_content_stems(answer))
        comps["answer_context_consistency"] = _ramp(len(a & index.stems) / max(1, len(a)), 0.3, 0.9) if a else None

    support = _sentence_support(claims, index) if claims else []
    comps["factual_support"] = (sum(support) / len(support)) if support else None
    unsupported = [c for c, v in zip(claims, support) if v < 0.5]
    signals.update({"claim_count": len(claims), "unsupported_claim_count": len(unsupported)})
    for c in unsupported[:2]:
        issues.append(f"Claim not supported by the retrieved context: '{_snippet(c)}'")
    if len(unsupported) > 2:
        issues.append(f"{len(unsupported)} of {len(claims)} claims have little support in the retrieved context.")

    checked, missing = _unsupported_details(answer, context, extra_allowed=question)
    comps["unsupported_details"] = None if checked == 0 else 1.0 - len(missing) / checked
    if missing:
        issues.append("Numbers/names in the answer that are not in the retrieved context: "
                      + ", ".join(f"'{m}'" for m in missing[:5]) + ".")

    score = _weighted(comps, _RAG_WEIGHTS) or 0.0
    if foreign_chunks and score > 40.0:
        issues.append("Score capped at 40 because context from another meeting was retrieved.")
        score = 40.0
    return _result(score, issues, status="ok", answer_type="answer",
                   components={k: _pct(v) for k, v in comps.items()}, signals=signals)


def rag_not_applicable(reason: str = "RAG was not used for this meeting", label: str = "Not Applicable") -> Dict[str, Any]:
    return {"score": None, "label": label, "reason": reason}


def aggregate_rag_groundedness(
    evaluations: Sequence[Dict[str, Any]],
    *,
    assistant_answer_count: int = 0,
) -> Dict[str, Any]:
    """Meeting-level RAG groundedness = mean of its scored answers (each evaluated when it was generated).

    ``assistant_answer_count`` is how many chatbot answers exist for the meeting; it distinguishes
    "RAG never used" (Not Applicable) from "used, but predates real-time scoring" (Not Evaluated).
    """
    scored = [e for e in evaluations if isinstance(e, dict) and e.get("score") is not None]
    if not scored:
        if assistant_answer_count > 0:
            return rag_not_applicable(
                f"{assistant_answer_count} chatbot answer(s) exist but were generated before real-time RAG scoring "
                "and have no stored retrieval context to evaluate.", label="Not Evaluated")
        return rag_not_applicable()

    mean = sum(e["score"] for e in scored) / len(scored)
    comp_acc: Dict[str, List[float]] = {}
    for e in scored:
        for k, v in (e.get("components") or {}).items():
            if v is not None:
                comp_acc.setdefault(k, []).append(v)
    issue_counts: Counter = Counter()
    for e in scored:
        for i in e.get("issues") or []:
            issue_counts[re.sub(r"['\"].*?['\"]", "...", i)] += 1
    issues = [f"{msg} (in {n} of {len(scored)} answers)" if n > 1 else msg for msg, n in issue_counts.most_common(6)]
    foreign = sum((e.get("signals") or {}).get("foreign_meeting_chunks_excluded", 0) for e in scored)
    return _result(
        mean, issues, status="ok", evaluated_answers=len(scored),
        abstained_answers=sum(1 for e in scored if e.get("answer_type") == "abstention"),
        components={k: round(sum(v) / len(v), 1) for k, v in comp_acc.items()},
        context_isolation={"isolated": foreign == 0, "foreign_meeting_chunks_excluded": foreign},
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4. ACTION ITEM QUALITY   5. DECISION QUALITY   6. AGENDA COVERAGE
#    (reference-free, computed from THIS meeting's own extracted items and transcript)
# ═════════════════════════════════════════════════════════════════════════════
_AQ_WEIGHTS = {
    "grounding": _env("AQ_W_GROUNDING", 0.35),
    "actionability": _env("AQ_W_ACTIONABILITY", 0.10),
    "owner": _env("AQ_W_OWNER", 0.20),
    "deadline": _env("AQ_W_DEADLINE", 0.20),
    "uniqueness": _env("AQ_W_UNIQUENESS", 0.15),
}
_DQ_WEIGHTS = {
    "grounding": _env("DQ_W_GROUNDING", 0.35),
    "decisiveness": _env("DQ_W_DECISIVENESS", 0.30),
    "specificity": _env("DQ_W_SPECIFICITY", 0.10),
    "distinct_from_actions": _env("DQ_W_DISTINCT", 0.15),
    "uniqueness": _env("DQ_W_UNIQUENESS", 0.10),
}
_AG_WEIGHTS = {
    "grounding": _env("AG_W_GROUNDING", 0.25),
    "coverage": _env("AG_W_COVERAGE", 0.25),
    "distinctness": _env("AG_W_DISTINCTNESS", 0.15),
    # Share of topics whose title has a matching statement in the transcript (the strict check the evaluation block
    # used to report as its own, much lower, "agenda score"): both numbers are now one score.
    "topic_support": _env("AG_W_TOPIC_SUPPORT", 0.35),
}

_GENERIC_OWNERS = frozenset({"", "unassigned", "not specified", "unspecified", "unknown", "n/a", "na", "none", "tbd",
                             "team", "everyone", "all", "someone", "anyone"})
_SPEAKER_LABEL_RE = re.compile(r"^\s*(?:speaker|spk)[\s_\-]*\d+\s*$", re.IGNORECASE)
_NO_DEADLINE = frozenset({"", "not specified", "unspecified", "none", "n/a", "na", "tbd", "no deadline"})
_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
_TEMPORAL_RE = re.compile(
    r"\b(?:by|before|until|within|due|on|this|next)\s+(?:the\s+)?(?:end\s+of\s+)?"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|tonight|week|month|quarter|year|"
    r"\d{1,2}(?:st|nd|rd|th)?|(?:" + _MONTHS + r")\b)|\b(?:tomorrow|tonight|asap|immediately|end of (?:the )?(?:day|week|month|quarter|year))\b",
    re.IGNORECASE,
)
_DECISIVE_RE = re.compile(
    r"\b(?:decid\w*|decision|agree[ds]?|agreement|approv\w*|resolv\w*|finali[sz]\w*|confirm\w*|conclud\w*|settled|"
    r"go(?:ing)? with|adopt\w*|chose|chosen|select\w*|award\w*|freeze|frozen|top priority|prioriti[sz]\w*|"
    r"targeted for|is targeted|set for|scheduled for|will be (?:approved|launched|released|migrated|used)|no longer)\b",
    re.IGNORECASE,
)
_COMMITMENT_RE = re.compile(
    r"^\s*(?:i|we|he|she|they|[A-Z][a-z]+|speaker[_\s]?\d+)\s+(?:will|shall|'ll|is going to|are going to|need to|needs to|has to|must)\b",
    re.IGNORECASE,
)
_DECISION_CUE_RE = re.compile(r"\b(?:decision\s*[:\-]|we (?:have )?decided|it was decided|we (?:have )?agreed|is approved|was approved)\b", re.IGNORECASE)


def _not_applicable(reason: str, issues: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"score": None, "label": "Not Applicable", "issues": list(issues or []), "reason": reason}


def _stem_set(text: str) -> frozenset:
    return frozenset(_content_stems(text))


def _overlap(a: frozenset, b: frozenset) -> float:
    return 0.0 if not a or not b else len(a & b) / len(a | b)


def _duplicate_fraction(texts: Sequence[str], threshold: float = 0.8) -> Tuple[float, int]:
    """Share of items that repeat an earlier one (stem overlap >= threshold) and how many do."""
    seen: List[frozenset] = []
    dups = 0
    for t in texts:
        stems = _stem_set(t)
        if stems and any(_overlap(stems, s) >= threshold for s in seen):
            dups += 1
        else:
            seen.append(stems)
    return (dups / len(texts) if texts else 0.0), dups


def _normalize_for_quotes(transcript: str) -> str:
    """The transcript in the form ``text_signals.quote_in_text`` expects."""
    from backend.utils import text_signals as _ts

    return _ts.normalize_text(transcript or "")


def _grounding(text: str, evidence: str, tx_norm: str, index: "_SupportIndex") -> float:
    """0..1: is this item actually said in the transcript? Best of: quoted evidence found verbatim,
    evidence supported (lexically / semantically), the item's own words supported.

    The verbatim test is ``text_signals.quote_in_text`` - the same one ``reference_free_engine`` uses for
    ``evaluation_metrics.action_items`` / ``.decisions`` - so the two blocks cannot disagree on whether a quote is in
    the transcript. ``tx_norm`` is the transcript passed through ``text_signals.normalize_text``.
    """
    from backend.utils import text_signals as _ts

    if evidence and _ts.quote_in_text(evidence, tx_norm):
        return 1.0
    cands = []
    for s in (evidence, text):
        if s and s.strip():
            cands.extend(_sentence_support([s], index))
    return max(cands) if cands else 0.0


def action_item_quality(items: Optional[Sequence[Dict[str, Any]]], transcript_text: Optional[str]) -> Dict[str, Any]:
    """How trustworthy are the action items? Grounded in the transcript, actionable, with a real owner and a
    deadline that is actually stated - and nothing invented. Unmeasurable parts are dropped, never padded."""
    tx = transcript_text or ""
    rows = [i for i in (items or []) if (i.get("task") or "").strip()]
    if not rows:
        cues = [s for s in _sentences(tx) if _COMMITMENT_RE.search(s) and _TEMPORAL_RE.search(s)]
        return _not_applicable(
            "no action items were identified for this meeting",
            [f"{len(cues)} sentence(s) read like dated commitments but no action item was produced "
             f"(e.g. '{_snippet(cues[0])}')"] if cues else [],
        )

    index = _SupportIndex(tx)
    tx_norm = _normalize_for_quotes(tx)
    tx_tokens = {t.lower() for t in _tokens(tx)}
    issues: List[str] = []
    grounding, action, owner_c, deadline_c = [], [], [], []
    for it in rows:
        task = str(it.get("task") or "").strip()
        evidence = str(it.get("evidence") or "")
        g = _grounding(task, evidence, tx_norm, index)
        grounding.append(g)
        if g < 0.5:
            issues.append(f"action item '{_snippet(task, 60)}' is not supported by the transcript")
        elif evidence:
            # Verbatim in the transcript, but is it a commitment of THIS meeting? Two different "this is an example"
            # phrases around the quote say it may be a worked example. A warning only: nothing is removed or re-scored.
            from backend.utils import text_signals as _ts

            markers = _ts.illustration_markers_near(evidence, tx)
            if len(markers) >= 2:
                issues.append(
                    f"action item '{_snippet(task, 60)}' is quoted from the transcript, but the quote sits inside an "
                    f"example or hypothetical ({', '.join(markers)}); it may not be a commitment made in the meeting"
                )

        words = len(task.split())
        action.append(0.0 if task.endswith("?") else (1.0 if 2 <= words <= 25 else 0.5 if words > 25 else 0.0))

        who = str(it.get("owner") or "").strip()
        if who.lower() in _GENERIC_OWNERS:
            owner_c.append(0.4)                                    # nobody was assigned: incomplete, not wrong
        elif _SPEAKER_LABEL_RE.match(who):
            owner_c.append(0.6)                                    # only an anonymous speaker label is known
        else:
            named = [t.lower() for t in _tokens(who) if len(t) >= 3]
            if named and any(t in tx_tokens for t in named):
                owner_c.append(1.0)
            else:
                owner_c.append(0.0)
                issues.append(f"owner '{who}' of '{_snippet(task, 40)}' is never mentioned in the transcript")

        deadline = str(it.get("deadline") or "").strip()
        has_deadline = deadline.lower() not in _NO_DEADLINE or bool(it.get("due_date"))
        said_time = bool(_TEMPORAL_RE.search(evidence)) or bool(_TEMPORAL_RE.search(task))
        if has_deadline:
            deadline_words = [t.lower() for t in _tokens(deadline) if len(t) >= 3 and t.lower() not in _STOP]
            if it.get("due_date") or any(t in tx_tokens for t in deadline_words):
                deadline_c.append(1.0)
            else:
                deadline_c.append(0.3)
                issues.append(f"deadline '{deadline}' of '{_snippet(task, 40)}' is not found in the transcript")
        elif said_time:
            deadline_c.append(0.3)
            issues.append(f"'{_snippet(task, 40)}' states a deadline in its own evidence but none was captured")
        # no deadline and none stated: nothing to measure

    dup_frac, dups = _duplicate_fraction([str(i.get("task") or "") for i in rows])
    if dups:
        issues.append(f"{dups} action item(s) repeat another one")

    def mean(vals):
        return sum(vals) / len(vals) if vals else None

    comps = {
        "grounding": mean(grounding), "actionability": mean(action), "owner": mean(owner_c),
        "deadline": mean(deadline_c), "uniqueness": 1.0 - dup_frac,
    }
    score = _weighted(comps, _AQ_WEIGHTS)
    if score is not None and comps["grounding"] is not None and comps["grounding"] < 0.5:
        score = min(score, 50.0)                                   # gate: mostly unsupported items cannot look fine
    return _result(
        score, issues,
        components={k: _pct(v) for k, v in comps.items()},
        signals={"action_item_count": len(rows), "semantic_checks": index.semantic,
                 "with_named_owner": sum(1 for c in owner_c if c == 1.0),
                 "with_deadline": sum(1 for it in rows if str(it.get("deadline") or "").strip().lower() not in _NO_DEADLINE or it.get("due_date")),
                 "duplicates": dups},
    )


def decision_quality(
    decisions: Optional[Sequence[Dict[str, Any]]],
    transcript_text: Optional[str],
    action_items: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """How trustworthy are the decisions? Stated in the transcript, worded as a group choice (not an individual
    commitment - that is an action item), specific, distinct from the action items and from each other."""
    tx = transcript_text or ""
    rows = [d for d in (decisions or []) if (d.get("decision") or d.get("key_decision") or "").strip()
            and "no formal decision" not in str(d.get("decision") or d.get("key_decision")).lower()]
    if not rows:
        cues = [s for s in _sentences(tx) if _DECISION_CUE_RE.search(s)]
        return _not_applicable(
            "no decisions were identified for this meeting",
            [f"{len(cues)} sentence(s) state a decision but none was extracted (e.g. '{_snippet(cues[0])}')"] if cues else [],
        )

    index = _SupportIndex(tx)
    tx_norm = _normalize_for_quotes(tx)
    action_stems = [_stem_set(str(a.get("task") or "")) for a in (action_items or [])]
    issues: List[str] = []
    grounding, decisive, specific, distinct = [], [], [], []
    texts = []
    for d in rows:
        text = str(d.get("decision") or d.get("key_decision") or "").strip()
        evidence = str(d.get("evidence") or "")
        texts.append(text)
        g = _grounding(text, evidence, tx_norm, index)
        grounding.append(g)
        if g < 0.5:
            issues.append(f"decision '{_snippet(text, 60)}' is not supported by the transcript")

        if _DECISIVE_RE.search(text) or (_DECISIVE_RE.search(evidence) and not _COMMITMENT_RE.search(text)):
            decisive.append(1.0)
        elif _COMMITMENT_RE.search(text):
            decisive.append(0.0)
            issues.append(f"'{_snippet(text, 60)}' reads like an individual commitment (an action item), not a decision")
        else:
            decisive.append(0.5)

        w = len(text.split())
        specific.append(1.0 if 4 <= w <= 40 else 0.5 if w > 40 else 0.3)

        stems = _stem_set(text)
        if action_stems and any(_overlap(stems, a) >= 0.6 for a in action_stems):
            distinct.append(0.0)
            issues.append(f"decision '{_snippet(text, 60)}' duplicates an action item")
        elif action_stems:
            distinct.append(1.0)

    dup_frac, dups = _duplicate_fraction(texts)
    if dups:
        issues.append(f"{dups} decision(s) repeat another one")

    def mean(vals):
        return sum(vals) / len(vals) if vals else None

    comps = {
        "grounding": mean(grounding), "decisiveness": mean(decisive), "specificity": mean(specific),
        "distinct_from_actions": mean(distinct), "uniqueness": 1.0 - dup_frac,
    }
    score = _weighted(comps, _DQ_WEIGHTS)
    if score is not None and comps["grounding"] is not None and comps["grounding"] < 0.5:
        score = min(score, 50.0)
    return _result(
        score, issues,
        components={k: _pct(v) for k, v in comps.items()},
        signals={"decision_count": len(rows), "semantic_checks": index.semantic, "duplicates": dups,
                 "commitment_like": sum(1 for v in decisive if v == 0.0)},
    )


def agenda_coverage(agenda_topics: Optional[Sequence[str]], transcript_text: Optional[str]) -> Dict[str, Any]:
    """How well do the agenda topics describe THIS meeting? Every topic is actually discussed (grounding), the
    discussion is covered by some topic (coverage), and no two topics are the same thing (distinctness)."""
    topics = [t.strip() for t in (agenda_topics or []) if t and str(t).strip()]
    if not topics:
        return _not_applicable("no agenda topics were identified for this meeting")
    tx = transcript_text or ""
    index = _SupportIndex(tx)
    topic_stems = [_stem_set(t) for t in topics]
    issues: List[str] = []

    support = _sentence_support(topics, index)
    for t, s in zip(topics, support):
        if s < 0.4:
            issues.append(f"agenda topic '{_snippet(t, 60)}' is barely discussed in the transcript")

    windows = [(w, _stem_set(w)) for w in index.windows if len(_tokens(w)) >= 6]
    topic_vecs = rfe._embed_batch(topics) if index.semantic else []
    if topic_vecs and len(topic_vecs) != len(topics):
        topic_vecs = []
    covered = 0
    for i, (w, wstems) in enumerate(windows):
        lex = max((len(ts & wstems) / len(ts) for ts in topic_stems if ts), default=0.0)
        sem = 0.0
        if topic_vecs and i < len(index.embeddings):
            sem = rfe._max_cosine(index.embeddings[i], topic_vecs)
        if lex >= 0.5 or sem >= 0.45:
            covered += 1
    coverage = covered / len(windows) if windows else None
    if coverage is not None and coverage < 0.5:
        issues.append(f"only {round(coverage * 100)}% of the discussion falls under an agenda topic")

    dup_frac, dups = _duplicate_fraction(topics, threshold=0.7)
    if dups:
        issues.append(f"{dups} agenda topic(s) repeat another one")

    # Strict per-topic check (same function the evaluation block uses): a topic is "supported" only when some
    # transcript statement matches its title. Without this the soft grounding average hid 4 unsupported topics of 6.
    flags = rfe._coverage_flags(topics, tx)
    supported = sum(1 for f in flags if f)
    topic_support = supported / len(topics) if topics else None
    if supported < len(topics):
        issues.append(f"{len(topics) - supported} of {len(topics)} agenda topic(s) have no matching statement in the transcript")

    comps = {"grounding": sum(support) / len(support), "coverage": coverage, "distinctness": 1.0 - dup_frac,
             "topic_support": topic_support}
    return _result(
        _weighted(comps, _AG_WEIGHTS), issues,
        components={k: _pct(v) for k, v in comps.items()},
        signals={"topic_count": len(topics), "supported_topics": supported, "unsupported_topics": len(topics) - supported,
                 "transcript_windows": len(windows), "covered_windows": covered,
                 "semantic_checks": index.semantic, "duplicates": dups},
    )


# ═════════════════════════════════════════════════════════════════════════════
# The public ``metrics`` block
# ═════════════════════════════════════════════════════════════════════════════
def build_quality_metrics(
    *,
    segments: Sequence[Dict[str, Any]],
    summary_text: Optional[str],
    transcript_text: Optional[str],
    agenda_topics: Optional[Sequence[str]],
    rag_evaluations: Sequence[Dict[str, Any]],
    assistant_answer_count: int,
    audio_duration_sec: Optional[float] = None,
    language_confidence: Optional[float] = None,
    is_document: bool = False,
    expected_language: Optional[str] = "en",
    action_items: Optional[Sequence[Dict[str, Any]]] = None,
    decisions: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble the API/DB ``metrics`` block for one meeting. Each score is computed independently
    and a failure in one never removes the others."""
    def safe(name: str, fn):
        try:
            return fn()
        except Exception as e:                               # never break the meeting response
            logger.error("[QualityMetrics] %s failed: %s", name, e, exc_info=True)
            return _result(None, [f"{name} could not be computed: {e}"], label="Error", status="error")

    return {
        "evaluation_type": EVALUATION_TYPE,
        "ground_truth_available": False,
        # Every score below is a reference-free HEURISTIC computed from this meeting's own transcript/outputs
        # (weighted sub-signals, gates and caps documented in quality_scores.py). None is WER, DER, ROUGE,
        # precision/recall or any other accuracy measured against a reference: those stay null without ground truth.
        "scores_are_heuristic": True,
        "score_meaning": "reference-free heuristic quality signal (0-100), not measured accuracy",
        "transcript_quality": safe("transcript_quality", lambda: transcript_quality(
            segments, audio_duration_sec=audio_duration_sec, language_confidence=language_confidence,
            is_document=is_document, expected_language=expected_language)),
        "summary_quality": safe("summary_quality", lambda: summary_quality(
            summary_text, transcript_text, agenda_topics=agenda_topics)),
        "rag_groundedness": safe("rag_groundedness", lambda: aggregate_rag_groundedness(
            rag_evaluations, assistant_answer_count=assistant_answer_count)),
        "action_item_quality": safe("action_item_quality", lambda: action_item_quality(action_items, transcript_text)),
        "decision_quality": safe("decision_quality", lambda: decision_quality(decisions, transcript_text, action_items)),
        "agenda_coverage": safe("agenda_coverage", lambda: agenda_coverage(agenda_topics, transcript_text)),
        "method_version": METHOD_VERSION,
    }

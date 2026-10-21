"""Pure-Python text & metadata signal helpers for intelligent meeting matching.

These functions provide the *human-thinking* signals that complement raw
embedding similarity: content fingerprints, token/keyword/participant overlap,
title-family detection, temporal cadence, and bullet-list (action item /
decision) comparison.

Everything here is dependency-free (stdlib only), deterministic and works for
any language, any duration and any organization — no hardcoded meeting rules.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

# A compact multi-purpose stopword set. Non-English transcripts are unaffected
# because their function words simply appear on both sides of the comparison.
_STOPWORDS: Set[str] = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "so",
    "as",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "up",
    "about",
    "into",
    "over",
    "after",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "am",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "having",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "this",
    "that",
    "these",
    "those",
    "there",
    "here",
    "what",
    "which",
    "who",
    "whom",
    "will",
    "would",
    "shall",
    "should",
    "can",
    "could",
    "may",
    "might",
    "must",
    "not",
    "no",
    "yes",
    "ok",
    "okay",
    "just",
    "like",
    "get",
    "got",
    "go",
    "going",
    "gonna",
    "want",
    "know",
    "think",
    "really",
    "very",
    "much",
    "some",
    "any",
    "all",
    "more",
    "most",
    "than",
    "too",
    "also",
    "well",
    "right",
    "yeah",
    "um",
    "uh",
    "hmm",
    "actually",
    "basically",
    "kind",
    "sort",
    "thing",
    "things",
    "let",
    "lets",
    "now",
    "one",
    "two",
    "how",
    "when",
    "where",
    "why",
    "speaker",
    "meeting",
}

_WORD_RE = re.compile(r"[a-z0-9]+", re.UNICODE)
_SPEAKER_PREFIX_RE = re.compile(
    r"^\s*(speaker[_\s]?\d+|spk[_\s]?\d+|[A-Z][a-zA-Z .]{0,40})\s*:",
    re.MULTILINE | re.IGNORECASE,
)
_TIMESTAMP_RE = re.compile(r"\[?\d{1,2}:\d{2}(?::\d{2})?\]?")
_SEQUENCE_RE = re.compile(
    r"\b(?:week|day|part|session|episode|sprint|no\.?|number|#|round|q(?:uarter)?|vol(?:ume)?)\s*#?\s*\d+\b",
    re.IGNORECASE,
)
_DATE_IN_TITLE_RE = re.compile(
    r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b"
)
_TRAILING_NUM_RE = re.compile(r"[\s\-_#(]*\d+[)]*\s*$")


# ─────────────────────────────────────────────────────────────────────────────
# Normalization & fingerprints
# ─────────────────────────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """Lowercase, drop speaker labels/timestamps, collapse whitespace."""
    if not text:
        return ""
    text = _SPEAKER_PREFIX_RE.sub(" ", text)
    text = _TIMESTAMP_RE.sub(" ", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_ELLIPSIS_RE = re.compile(r"\.{2,}|…")


def _quote_fragments(quote: str, min_words: int = 3) -> List[str]:
    """The normalised pieces of an evidence quote: it is split at "..." / "…" (a model joins non-adjacent words of
    the transcript that way); pieces shorter than ``min_words`` words carry no verifiable content and are skipped."""
    pieces = [normalize_text(p) for p in _ELLIPSIS_RE.split(quote or "")]
    return [p for p in pieces if len(p.split()) >= min_words]


def quote_in_text(quote: str, normalized_text: str, min_fragment_words: int = 3) -> bool:
    """Is the evidence ``quote`` really in the text? (``normalized_text`` = ``normalize_text(transcript)``.)

    The ONE grounding test for evidence quotes (action items, decisions): verbatim after normalisation (case,
    punctuation, speaker labels and timestamps ignored). A quote with "..." gaps must have every fragment of at
    least ``min_fragment_words`` words occur. A quote with no verifiable fragment is NOT supported - it is never
    guessed to be.
    """
    fragments = _quote_fragments(quote, min_fragment_words)
    if not fragments or not normalized_text:
        return False
    return all(f in normalized_text for f in fragments)


_ILLUSTRATION_MARKERS = (
    ("for example", re.compile(r"\bfor (?:example|instance)\b")),
    # (the text is normalised first: "let's" arrives as "let s")
    ("let's say", re.compile(r"\blet(?: us| s|s)? say\b|\bsuppos(?:e|ing)\b|\bimagine\b|\bassum(?:e|ing)\b")),
    ("sample", re.compile(r"\bsample (?:input|output|text|transcript|sentence|summary)\b")),
    ("possible output", re.compile(r"\bpossible (?:output|summary|structure|result|response)\b|\boriginal sentence\b")),
    ("such as", re.compile(r"\bsuch as\b|\bsay that\b|\bhypothetical(?:ly)?\b")),
)


def illustration_markers_near(
    quote: str, transcript: str, words_before: int = 60, words_after: int = 30
) -> List[str]:
    """Distinct "this is an example" phrases around where ``quote`` occurs in the transcript (empty when the quote
    is not found).

    A task spoken inside a worked example ("for example, let's say a manager says: complete the API testing by
    Friday") has verbatim evidence yet is not a commitment of the meeting. Two or more DIFFERENT markers close to
    the quote make that likely; a single "for example" does not (on real data one marker flagged genuine items of
    recurring meetings), so callers use this as a WARNING, never to delete an item.
    """
    fragments = _quote_fragments(quote)
    if not fragments:
        return []
    words = normalize_text(transcript or "").split()
    first = fragments[0].split()
    n = min(4, len(first))
    start = None
    for i in range(len(words) - n + 1):
        if words[i:i + n] == first[:n]:
            start = i
            break
    if start is None:
        return []
    window = " ".join(words[max(0, start - words_before): start + len(first) + words_after])
    return [name for name, rx in _ILLUSTRATION_MARKERS if rx.search(window)]


def locate_quote_segment(quote: str, segment_texts: List[str]) -> Optional[int]:
    """Index of the first transcript segment whose text contains the quote's first fragment, else None."""
    fragments = _quote_fragments(quote)
    if not fragments:
        return None
    first = fragments[0]
    for i, text in enumerate(segment_texts):
        if first in normalize_text(text or ""):
            return i
    return None


def content_fingerprint(text: str) -> str:
    """Stable SHA-256 fingerprint of the *meaningful* content of a transcript."""
    normalized = normalize_text(text)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def tokens(text: str, keep_stopwords: bool = False) -> List[str]:
    words = _WORD_RE.findall((text or "").lower())
    if keep_stopwords:
        return words
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def token_set(text: str) -> Set[str]:
    return set(tokens(text))


# ─────────────────────────────────────────────────────────────────────────────
# Set-similarity primitives
# ─────────────────────────────────────────────────────────────────────────────
def jaccard(a: Set, b: Set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def overlap_coefficient(a: Set, b: Set) -> float:
    """|a ∩ b| / min(|a|, |b|)  — robust to large size differences."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def content_overlap(text_a: str, text_b: str) -> float:
    """How much of the smaller transcript is contained in the larger one."""
    return overlap_coefficient(token_set(text_a), token_set(text_b))


# ─────────────────────────────────────────────────────────────────────────────
# Titles: recurring / follow-up family detection
# ─────────────────────────────────────────────────────────────────────────────
def title_base(title: str) -> str:
    """Strip sequence markers (Week 3, Part 2, #4, dates, trailing numbers)."""
    if not title:
        return ""
    base = _DATE_IN_TITLE_RE.sub(" ", title)
    base = _SEQUENCE_RE.sub(" ", base)
    base = _TRAILING_NUM_RE.sub(" ", base)
    base = re.sub(r"\s+", " ", base).strip().lower()
    return base


def title_similarity(a: str, b: str) -> float:
    return jaccard(token_set(title_base(a)), token_set(title_base(b)))


def title_has_sequence_marker(title: str) -> bool:
    if not title:
        return False
    return bool(
        _SEQUENCE_RE.search(title)
        or _DATE_IN_TITLE_RE.search(title)
        or _TRAILING_NUM_RE.search(title)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Participants
# ─────────────────────────────────────────────────────────────────────────────
def parse_participants(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[,;/|]", raw)
    cleaned = []
    seen = set()
    for p in parts:
        name = p.strip()
        if not name:
            continue
        key = name.lower()
        if key in {
            "meeting participants",
            "not specified",
            "unknown",
            "n/a",
            "na",
            "n",
            "a",
        }:
            continue
        if key not in seen:
            seen.add(key)
            cleaned.append(name)
    return cleaned


def _participant_key(name: str) -> str:
    return re.sub(r"[^\w]", "", name.lower())


def participant_overlap(a: Optional[str], b: Optional[str]) -> Tuple[float, List[str]]:
    """Returns (jaccard_score, matching_participant_names)."""
    la, lb = parse_participants(a), parse_participants(b)
    if not la or not lb:
        return 0.0, []
    sa = {_participant_key(n): n for n in la}
    sb = {_participant_key(n): n for n in lb}
    common_keys = set(sa) & set(sb)
    union = set(sa) | set(sb)
    matches = [sa[k] for k in common_keys]
    score = len(common_keys) / len(union) if union else 0.0
    return score, matches


# ─────────────────────────────────────────────────────────────────────────────
# Keywords
# ─────────────────────────────────────────────────────────────────────────────
def parse_keywords(raw: Optional[str]) -> Set[str]:
    if not raw:
        return set()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k).lower().strip() for k in data.keys() if str(k).strip()}
        if isinstance(data, list):
            return {str(k).lower().strip() for k in data if str(k).strip()}
    except Exception:
        pass
    return {w for w in tokens(raw)}


def keyword_overlap(a: Optional[str], b: Optional[str]) -> Tuple[float, List[str]]:
    ka, kb = parse_keywords(a), parse_keywords(b)
    common = sorted(ka & kb)
    return jaccard(ka, kb), common


# ─────────────────────────────────────────────────────────────────────────────
# Bullet lists (action items / decisions / risks)
# ─────────────────────────────────────────────────────────────────────────────
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\[[ xX]\]|\d+[.)])\s+", re.MULTILINE)
_CHECKED_ITEM_RE = re.compile(r"^\s*\[\s*[xX]\s*\]")    # an extracted bullet that still carries its "[x]" marker


def extract_bullets(markdown: Optional[str]) -> List[str]:
    """Extract individual items from a markdown list / free text block / JSON action items."""
    if not markdown:
        return []
    text = markdown.strip()
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
        try:
            import json
            parsed = json.loads(text)
            if isinstance(parsed, list):
                bullet_lines = []
                for item in parsed:
                    if isinstance(item, dict):
                        t = item.get("task") or item.get("title") or item.get("description") or ""
                        o = item.get("owner") or item.get("assigned_to") or ""
                        d = item.get("deadline") or item.get("due_date") or ""
                        line = t
                        if o and o.lower() not in ("unassigned", "not specified", "none", ""):
                            line += f" (Assignee: {o})"
                        if d and d.lower() not in ("not specified", "none", ""):
                            line += f" (Deadline: {d})"
                        if line:
                            bullet_lines.append(line)
                    elif isinstance(item, str):
                        cleaned = _BULLET_RE.sub("", item.strip())
                        cleaned = re.sub(r"[\*_`#]+", "", cleaned).strip()
                        if cleaned:
                            bullet_lines.append(cleaned)
                if bullet_lines:
                    return bullet_lines
        except Exception:
            pass

    lines = text.split("\n")
    items: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = _BULLET_RE.sub("", stripped)
        cleaned = re.sub(r"[\*_`#]+", "", cleaned).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        low = cleaned.lower()
        if len(cleaned) < 4:
            continue
        if low.startswith(
            (
                "no ",
                "none",
                "not specified",
                "no formal",
                "no major",
                "no immediate",
                "no open",
            )
        ):
            continue
        items.append(cleaned)
    return items


def _bullet_similarity(a: str, b: str) -> float:
    return jaccard(token_set(a), token_set(b))


def compare_bullet_lists(
    current: Optional[str],
    previous: Optional[str],
    match_threshold: float = 0.5,
) -> Dict[str, List[str]]:
    """Diff two markdown lists into shared / new / resolved / not_carried_over buckets.

    - shared:           items present in both (carried over / still relevant)
    - new:              items only in the current meeting
    - resolved:         items of the previous meeting that were explicitly marked DONE (``[x]``) there and are not
                        repeated now - the only evidence of completion a list can carry
    - not_carried_over: every other previous item that does not reappear (an unchecked ``[ ]`` task, or a bullet with
                        no status at all). It may be done, dropped, deferred or simply not mentioned: absence from
                        the next meeting is NOT completion, so it is never reported as resolved.
    """
    cur = extract_bullets(current)
    prev = extract_bullets(previous)
    shared, new_items = [], []
    matched_prev_idx: Set[int] = set()

    for c in cur:
        best_idx, best_score = -1, 0.0
        for i, p in enumerate(prev):
            s = _bullet_similarity(c, p)
            if s > best_score:
                best_score, best_idx = s, i
        if best_score >= match_threshold and best_idx >= 0:
            shared.append(c)
            matched_prev_idx.add(best_idx)
        else:
            new_items.append(c)

    resolved, not_carried_over = [], []
    for i, p in enumerate(prev):
        if i in matched_prev_idx:
            continue
        (resolved if _CHECKED_ITEM_RE.match(p) else not_carried_over).append(p)
    return {"shared": shared, "new": new_items, "resolved": resolved, "not_carried_over": not_carried_over}


# ─────────────────────────────────────────────────────────────────────────────
# Temporal signals
# ─────────────────────────────────────────────────────────────────────────────
_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%d %B %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %b %Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
]


def parse_date(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    raw = raw.strip()
    if not raw or raw.lower() in {"unknown date", "not specified", "n/a"}:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw[: len(fmt) + 4], fmt)
        except Exception:
            continue
    m = _DATE_IN_TITLE_RE.search(raw)
    if m:
        return parse_date(m.group(1))
    try:
        return datetime.fromisoformat(raw.replace("Z", ""))
    except Exception:
        return None


def temporal_relation(
    current: Optional[str], previous: Optional[str]
) -> Dict[str, Any]:
    """Return ordering + gap between two meeting dates."""
    cd, pd = parse_date(current), parse_date(previous)
    if not cd or not pd:
        return {
            "has_dates": False,
            "gap_days": None,
            "current_is_after": None,
            "cadence": None,
        }
    gap = abs((cd - pd).days)
    cadence = None
    if gap != 0:
        if 5 <= gap <= 9:
            cadence = "Weekly"
        elif gap == 1 or gap == 0:
            cadence = "Daily"
        elif 2 <= gap <= 4:
            cadence = "Every few days"
        elif 12 <= gap <= 16:
            cadence = "Bi-weekly"
        elif 25 <= gap <= 35:
            cadence = "Monthly"
        elif 85 <= gap <= 95:
            cadence = "Quarterly"
    return {
        "has_dates": True,
        "gap_days": gap,
        "current_is_after": cd >= pd,
        "cadence": cadence,
    }


def duration_ratio(a: Optional[float], b: Optional[float]) -> float:
    """1.0 == identical duration, →0 as they diverge."""
    try:
        a, b = float(a or 0), float(b or 0)
    except Exception:
        return 0.0
    if a <= 0 or b <= 0:
        return 0.0
    return min(a, b) / max(a, b)


def humanize_gap(gap_days: Optional[int]) -> str:
    if gap_days is None:
        return "an earlier date"
    if gap_days == 0:
        return "the same day"
    if gap_days == 1:
        return "1 day earlier"
    if gap_days < 7:
        return f"{gap_days} days earlier"
    if gap_days < 30:
        weeks = round(gap_days / 7)
        return f"about {weeks} week{'s' if weeks > 1 else ''} earlier"
    if gap_days < 365:
        months = round(gap_days / 30)
        return f"about {months} month{'s' if months > 1 else ''} earlier"
    years = round(gap_days / 365)
    return f"about {years} year{'s' if years > 1 else ''} earlier"


# ─── Whisper Anti-Hallucination Engine ──────────────────────────────────────
_WHISPER_HALLUCINATION_PATTERNS = [
    r"\b(ywydwyn|ywydw|ywyd|hynwyn|yw'r\s+gweld|yw'n\s+i'n|i'n\s+i'n)\b",
    r"\b(subtitles?\s+by|amara\.org|thank\s+you\s+for\s+watching|subscribe\s+to|like\s+and\s+subscribe)\b",
    r"\b(sous-titres?\s+réalisés?\s+par|untertitel\s+von)\b",
]


# A single word repeated this many times in a row is an ASR loop, not speech (see clean_whisper_hallucinations).
_MIN_LOOP_REPEATS_SINGLE_WORD = 5


def clean_whisper_hallucinations(text: str) -> str:
    """
    Dynamically strips Whisper ASR repetition loops and phantom noise hallucinations.
    Works for any language, any duration, and any video/audio input.
    """
    if not text or not text.strip():
        return ""

    cleaned = text

    # 1. Strip known silent-portion hallucination patterns
    for pat in _WHISPER_HALLUCINATION_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)

    # 2. Collapse consecutive single-word repetition LOOPS (5+ identical consecutive words -> 1 word). Whisper loops
    #    run to dozens of repeats; people really do say "no no no" or "very very very good" (3-4 in a row), and a
    #    3-repeat threshold deleted those words from the transcript.
    cleaned = re.sub(r"\b(\w+)(?:\s+\1){4,}\b", r"\1", cleaned, flags=re.IGNORECASE)

    # 3. Collapse consecutive multi-word phrase repetitions (1 to 5 word phrases repeated 3+ times)
    words = cleaned.split()
    if not words:
        return ""

    n_words = len(words)
    filtered_words = []
    i = 0
    while i < n_words:
        match_len = 0
        max_repeats = 0
        for phrase_len in range(1, 6):
            if i + phrase_len > n_words:
                break
            phrase = [w.lower() for w in words[i : i + phrase_len]]
            repeats = 1
            idx = i + phrase_len
            while idx + phrase_len <= n_words and [w.lower() for w in words[idx : idx + phrase_len]] == phrase:
                repeats += 1
                idx += phrase_len

            min_repeats = _MIN_LOOP_REPEATS_SINGLE_WORD if phrase_len == 1 else 3
            if repeats >= min_repeats and (repeats * phrase_len) > (max_repeats * match_len):
                match_len = phrase_len
                max_repeats = repeats

        if max_repeats >= 3:
            # Keep at most 1 instance of the repeating phrase
            filtered_words.extend(words[i : i + match_len])
            i += max_repeats * match_len
        else:
            filtered_words.append(words[i])
            i += 1

    result = " ".join(filtered_words)

    # 4. Final sanity cleanup of whitespace and punctuation
    result = re.sub(r"\s+", " ", result).strip()
    result = re.sub(r"(\s*[\.\,\?\!]){2,}", r"\1", result)

    # 5. Degenerate-repetition guard.
    #
    # A very low unique/total word ratio means large spans of the text are
    # repeated loops (classic Whisper-on-silence behaviour, common on long,
    # partially-silent recordings). The OLD code returned "" here — which,
    # applied to a whole multi-hour transcript, silently deleted every real
    # sentence too and surfaced as "transcription_returned_no_text".
    #
    # Instead, collapse to the DISTINCT sentences in their original order. This
    # removes the hallucination loops while preserving all genuinely unique
    # content. Only fall through to "" when there is essentially no distinct
    # content at all (pure noise).
    words_res = result.split()
    if len(words_res) > 10:
        unique_tokens = {w.lower() for w in words_res}
        if len(unique_tokens) / len(words_res) < 0.20:
            sentences = re.split(r"(?<=[\.\?\!])\s+", result)
            seen: Set[str] = set()
            kept: List[str] = []
            for s in sentences:
                s_clean = s.strip()
                if not s_clean:
                    continue
                # Normalise for the dedup key; keep the original for output.
                key = re.sub(r"[^\w\s]", "", s_clean.lower())
                key = re.sub(r"\b(\w+)(?:\s+\1)+\b", r"\1", key).strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                kept.append(s_clean)
            deduped = re.sub(r"\s+", " ", " ".join(kept)).strip()
            if deduped:
                deduped_words = deduped.split()
                # Still degenerate after sentence-dedup AND almost no vocabulary
                # → genuine pure-noise; let the caller treat it as empty.
                if (
                    len(deduped_words) <= 3
                    or len({w.lower() for w in deduped_words}) < 4
                ):
                    return ""
                return deduped
            return ""

    return result

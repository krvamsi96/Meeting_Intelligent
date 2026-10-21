"""Meeting summary generation, validation and repair.

A SUMMARY is a short prose overview of what a meeting was about and what came
out of it. Who said what (speaker labels / IDs), timestamps, dialogue and
minutes-of-meeting structure belong to the transcript, speaker analysis and MoM
— never to the summary.

Pipeline (everything is derived from the ONE transcript passed in; there is no
module-level state, cache, RAG lookup or database access, so one meeting's
summary can never leak into another's):

1. ``prepare_summary_source``  — strip speaker labels and timecodes from the
   transcript BEFORE the LLM sees it. A model cannot echo a label it never saw.
2. ``plan_summary_length``     — target size grows sub-linearly with the
   meeting's real word count (short meeting -> short summary).
3. LLM draft                   — one call, or map-reduce (per-part notes, then
   one summary of the notes) when the meeting exceeds the context window. Long
   meetings are compressed semantically, never cut off.
4. ``normalize_summary_text``  — flatten markdown / bullets / preambles / JSON,
   drop repeated sentences.
5. ``check_summary``           — detect speaker labels & IDs (generic patterns
   plus the labels this transcript actually used), timestamps, dialogue
   formatting, MoM/metadata sections, over-length, verbatim copying, non-English
   and truncated output.
6. LLM rewrite (up to ``SUMMARY_MAX_REPAIRS`` passes) that fixes the reported
   problems while keeping the meaning, re-validated after every pass.
7. Deterministic last resort    — only if the LLM could not fix it: replace
   labels with neutral wording (meaning-preserving, never sentence deletion),
   strip timecodes, and as a final bound trim whole trailing sentences.
8. If the LLM is unavailable    — label-free extractive summary at the same
   planned length.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from backend.config.config import CLOUD_MAX_TRANSCRIPT_CHARS, settings

logger = logging.getLogger(__name__)

# (system_prompt, user_text, max_tokens) -> model text. Injectable for tests.
LLMCall = Callable[[str, str, int], str]

# Below this many words of actual meeting content there is nothing to summarise.
MIN_CONTENT_WORDS = 5

# Parts of a too-long meeting are condensed to notes of about this many words.
_NOTES_WORDS_PER_PART = 160
_PART_CHARS = 32000
_MAX_CONDENSE_DEPTH = 3


# ═════════════════════════════════════════════════════════════════════════════
# Speaker-label / timestamp / structure patterns
# ═════════════════════════════════════════════════════════════════════════════
# "speaker" alone is an ordinary word ("speaker diarization", "the speaker") and
# must NOT match — only label-shaped tokens do: Speaker 1, Speaker_2, SPEAKER_00,
# spk3, Participant 4, Speaker A, Unknown Speaker, ...
_SEP = r"[\s_\-]*"
_SPEAKER_ID_PATTERN = (
    r"(?:"
    r"(?i:speaker|spk|spkr)" + _SEP + r"\#?\d+"
    r"|(?i:participant|attendee|talker)" + _SEP + r"\#?\d+"
    r"|(?i:speaker)[\s_\-](?-i:[A-Z])"
    r"|(?i:unknown|unidentified)[\s_\-]*(?i:speaker)"
    r"|(?i:speaker)[\s_\-]*(?i:unknown)"
    r")"
)
_NOT_WORD_BEFORE = r"(?<![A-Za-z0-9_])"
_NOT_WORD_AFTER = r"(?![A-Za-z0-9_])"
_SPEAKER_ID_RE = re.compile(_NOT_WORD_BEFORE + _SPEAKER_ID_PATTERN + _NOT_WORD_AFTER)

_TIMECODE_RE = re.compile(
    r"[\[(]\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*"
    r"(?:[-–—]\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*)?[\])]"       # [00:12] (00:12:30) [00:12 - 00:20]
    r"|(?<![\d:])\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?(?![\d:])"              # 00:12:30
    r"|(?<![\d:])\d{1,2}:\d{2}\s*[-–—]\s*\d{1,2}:\d{2}(?![\d:])"        # 00:12 - 00:20
)

# Leading time codes / labels on a transcript line: "[00:12] Speaker_1: text".
_LEADING_TIMECODE_RE = re.compile(
    r"^\s*(?:[\[(]\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*"
    r"(?:[-–—]\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*)?[\])]\s*[-–—>]?\s*)+"
)
_INLINE_TIMECODE_RE = re.compile(r"\[\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*\]")
_LABEL_LINE_RE = re.compile(r"^\s*(?P<label>[^\s:\[\]][^:\n]{0,38}?)\s*:\s+(?P<text>\S.*)$")

# Line prefix of dialogue / field formatting: "Rajesh: ...", "Action Items: ...".
_LABEL_PREFIX_RE = re.compile(
    r"^[ \t]*(?:[-*•]\s*)?[\[(]?\s*(?P<label>[A-Za-z][\w .\-']{0,30}?)\s*[\])]?\s*:\s+\S", re.M
)

# Minutes-of-meeting / metadata field labels that do not belong inside a summary.
# Anchored to a line start or a sentence start so prose such as "the agenda was
# reviewed" or "the key decision: use X" mid-sentence is not flagged.
_MOM_FIELD_RE = re.compile(
    r"(?im)(?:^|(?<=[.!?]\s))[ \t]*(?:action items?|decisions?(?: made)?|key (?:discussion )?points|"
    r"attendees?|participants?|next meeting|follow[- ]?ups?|agenda|minutes(?: of (?:the )?meeting)?|"
    r"meeting (?:information|details|type|title|id)|date(?: ?(?:&|and) ?time)?|time|duration|status|"
    r"conclusion)[ \t]*:[ \t]"
)
_METADATA_TOKEN_RE = re.compile(
    r"(?i)\bmeeting[\s_\-]?id\b\s*[:#=]?\s*\w*\d|\bbharatvc[\s_\-]?(?:meeting[\s_\-]?)?id\b"
    r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    r"|\bconfidence(?: score)?\s*[:=]\s*\d"
)

_PREAMBLE_RE = re.compile(
    r"^\s*(?:sure|certainly|of course|absolutely|here(?:'s| is| are)|below (?:is|are)|"
    r"the following is|i(?:'ve| have) (?:written|prepared|created)|as requested)\b"
    r"[^\n:]{0,120}[:\-–]\s*\n?",
    re.IGNORECASE,
)
_TITLE_LINE_RE = re.compile(
    r"(?im)^[#*_\s]*(?:(?:executive|meeting|brief|short)\s+)?summary(?:\s+of\s+the\s+meeting)?[#*_\s]*[:\-–]?\s*$"
)
_TITLE_PREFIX_RE = re.compile(r"(?i)^\s*(?:(?:executive|meeting)\s+)?summary\s*[:\-–]\s+")
_THINK_RE = re.compile(r"<think(?:ing)?\b[^>]*>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_DANGLING_WORDS = frozenset(
    "and or the a an of to in on for with that which as by at from into about is are was were "
    "will also including but so because if while".split()
)


# ═════════════════════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SummarySource:
    """The meeting content handed to the summariser — labels and timecodes removed."""

    text: str
    word_count: int
    labels: Tuple[str, ...] = ()       # every speaker label found in the transcript
    id_labels: Tuple[str, ...] = ()    # the ID-shaped subset (contains a digit/underscore)
    fingerprint: str = ""


@dataclass(frozen=True)
class SummaryPlan:
    source_words: int
    target_words: int
    min_words: int
    max_words: int         # what the prompt asks the model not to exceed
    hard_max_words: int    # what validation tolerates before demanding compression
    paragraphs: int
    max_tokens: int        # output ceiling for the LLM call (safety net, not the length control)


@dataclass
class Violation:
    code: str
    detail: str = ""


@dataclass
class SummaryResult:
    text: str
    method: str                      # llm | llm_repaired | llm_neutralized | extractive | empty
    llm_calls: int = 0
    violations: List[str] = field(default_factory=list)   # codes still present in ``text``
    plan: Optional[SummaryPlan] = None


# ═════════════════════════════════════════════════════════════════════════════
# 1. Source preparation
# ═════════════════════════════════════════════════════════════════════════════
def _is_id_like(label: str) -> bool:
    """True for machine-shaped labels (Speaker_1, SPK2, Person_A) but not names or bare digits."""
    label = label.strip()
    if len(label) < 2 or label.isdigit():
        return False
    return bool(re.search(r"\d", label) or "_" in label)


def _plausible_label(label: str) -> bool:
    label = label.strip()
    if not label or len(label.split()) > 5:
        return False
    return not re.search(r"[.!?,;]$", label)


def prepare_summary_source(transcript: str) -> SummarySource:
    """Turn ``Label: text`` transcript lines into label-free content.

    A label is recognised when it is ID-shaped, repeats across the transcript, or
    when most lines carry one — so a lone "Note: ..." line inside ordinary prose is
    left alone. Leading and bracketed timecodes are removed too.
    """
    parsed: List[Tuple[Optional[str], str]] = []
    for raw in (transcript or "").splitlines():
        if not raw.strip():
            continue
        body = _LEADING_TIMECODE_RE.sub("", raw)
        body = _INLINE_TIMECODE_RE.sub("", body).strip()
        m = _LABEL_LINE_RE.match(body)
        if m and _plausible_label(m.group("label")):
            parsed.append((m.group("label").strip(), m.group("text").strip()))
        else:
            parsed.append((None, body))

    counts = Counter(label for label, _ in parsed if label)
    mostly_labelled = bool(parsed) and sum(counts.values()) >= 0.6 * len(parsed)
    accepted = {
        label for label, n in counts.items() if n >= 2 or _is_id_like(label) or mostly_labelled
    }

    lines = []
    for label, text in parsed:
        if label and label not in accepted:
            text = f"{label}: {text}"
        if text.strip():
            lines.append(text.strip())

    content = "\n".join(lines)
    normalized = re.sub(r"\s+", " ", content.lower()).strip()
    return SummarySource(
        text=content,
        word_count=len(content.split()),
        labels=tuple(sorted(accepted)),
        id_labels=tuple(sorted(l for l in accepted if _is_id_like(l))),
        fingerprint=hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12] if normalized else "EMPTY",
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2. Dynamic length
# ═════════════════════════════════════════════════════════════════════════════
def plan_summary_length(source_words: int) -> SummaryPlan:
    """Size the summary from the meeting's real content.

    ``target = coeff * sqrt(words)`` clamped to [min, max]: a meeting 10x longer
    gets a summary ~3x longer, never 10x. A summary is also never allowed to
    approach half of what it summarises.
    """
    words = max(0, int(source_words))
    lo = max(8, int(settings.SUMMARY_MIN_WORDS))
    hi = max(lo, int(settings.SUMMARY_MAX_WORDS))
    target = int(round(min(max(settings.SUMMARY_LENGTH_COEFF * math.sqrt(words), lo), hi)))
    if words:
        target = max(4, min(target, int(words * 0.5)))
    max_words = int(round(target * 1.2))
    hard_max = int(round(max_words * 1.25)) + 5
    return SummaryPlan(
        source_words=words,
        target_words=target,
        min_words=max(3, int(target * 0.6)),
        max_words=max_words,
        hard_max_words=hard_max,
        paragraphs=1 if target <= 110 else 2,
        max_tokens=hard_max * 2 + 80,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3. Normalisation
# ═════════════════════════════════════════════════════════════════════════════
# Only function words are ignored when comparing sentences: numbers, short
# tokens ("Q2", "5%", "B") and names are exactly what distinguishes two otherwise
# similar sentences, so they must count.
_DEDUPE_IGNORE = frozenset("the a an and of to in on for with that this is are was were be it as at by".split())


def _norm_tokens(sentence: str) -> frozenset:
    return frozenset(w for w in re.findall(r"[a-z0-9]+", sentence.lower()) if w not in _DEDUPE_IGNORE)


def _split_sentences(paragraph: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(paragraph) if s.strip()]


def _dedupe_sentences(paragraphs: List[str], jaccard: float = 0.9) -> List[str]:
    """Drop repeated sentences (they add no information).

    Deliberately conservative — a summary sentence is only removed when it is the
    same sentence again (identical content words) or a >=90% overlap of a long
    sentence. Anything that differs by a number, name or date is kept: paraphrased
    repetition is the LLM rewrite's job, not a blind deletion.
    """
    seen: List[frozenset] = []
    out: List[str] = []
    for para in paragraphs:
        kept = []
        for sent in _split_sentences(para):
            toks = _norm_tokens(sent)
            dup = False
            if toks:
                for prev in seen:
                    if toks == prev:
                        dup = True
                        break
                    union = len(toks | prev)
                    if union and len(toks & prev) / union >= jaccard and min(len(toks), len(prev)) >= 8:
                        dup = True
                        break
            if not dup:
                kept.append(sent)
                if toks:
                    seen.append(toks)
        if kept:
            out.append(" ".join(kept))
    return out


def _flatten_json(raw: str) -> Tuple[str, bool]:
    """If the model returned JSON, pull the prose out. Returns (text, was_structured)."""
    s = raw.strip()
    if s[:1] not in "{[":
        return raw, False
    try:
        obj = json.loads(s)
    except Exception:
        return raw, False

    def _texts(o) -> List[str]:
        if isinstance(o, str):
            return [o]
        if isinstance(o, dict):
            return [t for v in o.values() for t in _texts(v)]
        if isinstance(o, list):
            return [t for v in o for t in _texts(v)]
        return []

    if isinstance(obj, dict):
        for key in ("summary", "meeting_summary", "overview", "executive_summary"):
            if isinstance(obj.get(key), str) and len(obj) == 1:
                return obj[key], False     # {"summary": "..."} is just a wrapper
    return " ".join(t.strip() for t in _texts(obj) if t.strip()), True


def normalize_summary_text(
    raw: str, max_paragraphs: Optional[int] = None, dedupe: bool = True
) -> Tuple[str, bool]:
    """Reduce model output to clean prose. Returns (text, was_structured_output).

    Removes think-blocks, code fences, preambles, titles, markdown, bullets and
    numbering (list items become sentences), repeated sentences, and enforces the
    paragraph cap. Content is preserved — this never rewrites or shortens meaning.
    ``dedupe=False`` keeps repeated sentences (used when MEASURING a summary's redundancy).
    """
    text = _THINK_RE.sub("", raw or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*\n?|\n?```\s*$", "", text).strip()
    text, structured = _flatten_json(text)

    for _ in range(2):
        stripped = _PREAMBLE_RE.sub("", text, count=1).strip()
        if stripped == text:
            break
        text = stripped
    text = _TITLE_LINE_RE.sub("", text)
    text = _TITLE_PREFIX_RE.sub("", text, count=1)

    paragraphs: List[str] = []
    current: List[str] = []

    def _flush():
        if current:
            paragraphs.append(" ".join(current).strip())
            current.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            _flush()
            continue
        if re.match(r"^#{1,6}\s", line) or re.match(r"^[-*_=]{3,}$", line):
            _flush()
            continue
        line = re.sub(r"^>\s?", "", line)
        item = re.match(r"^(?:[-*•▪◦]|\d{1,2}[.)])\s+(.*)$", line)
        if item:
            line = item.group(1).strip()
            if line and line[-1] not in ".!?":
                line += "."
        line = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), line)
        line = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", line)
        line = line.replace("`", "").replace("**", "")
        if line:
            current.append(line)
    _flush()

    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs if p.strip()]
    if dedupe:
        paragraphs = _dedupe_sentences(paragraphs)
    if max_paragraphs and len(paragraphs) > max_paragraphs:
        paragraphs = paragraphs[: max_paragraphs - 1] + [" ".join(paragraphs[max_paragraphs - 1:])]

    out = "\n\n".join(paragraphs).strip()
    if out and out[-1].isalnum() and out.rsplit(None, 1)[-1].lower().strip(",;") not in _DANGLING_WORDS:
        out += "."
    return out, structured


# ═════════════════════════════════════════════════════════════════════════════
# 4. Validation
# ═════════════════════════════════════════════════════════════════════════════
def _label_pattern(id_labels: Iterable[str] = ()) -> str:
    """Regex matching any generic speaker label plus this transcript's own ID-shaped labels."""
    alts = [_SPEAKER_ID_PATTERN]
    for label in sorted({l.strip() for l in id_labels if _is_id_like(l)}, key=len, reverse=True):
        tokens = [re.escape(t) for t in re.split(r"[\s_\-]+", label) if t]
        if tokens:
            alts.append("(?i:" + r"[\s_\-]*".join(tokens) + ")")
    return _NOT_WORD_BEFORE + "(?:" + "|".join(alts) + ")" + _NOT_WORD_AFTER


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _verbatim_overlap(summary: str, source: str, n: int = 6) -> float:
    """Share of the summary's n-word runs that appear verbatim in the source."""
    s, src = _words(summary), _words(source)
    if len(s) < n + 4 or len(src) < n:
        return 0.0
    src_runs = {tuple(src[i:i + n]) for i in range(len(src) - n + 1)}
    runs = [tuple(s[i:i + n]) for i in range(len(s) - n + 1)]
    return sum(1 for r in runs if r in src_runs) / len(runs)


def check_summary(
    text: str,
    plan: SummaryPlan,
    *,
    source_text: str = "",
    labels: Sequence[str] = (),
    id_labels: Sequence[str] = (),
    structured: bool = False,
    raw: str = "",
) -> List[Violation]:
    """Return everything wrong with ``text`` as a meeting summary (empty list = clean).

    ``raw`` is the model output BEFORE normalisation. Normalisation folds
    consecutive dialogue lines into one paragraph, so line-prefix (dialogue)
    detection has to look at the raw layout.
    """
    text = text or ""
    words = text.split()
    if len(words) < 3:
        return [Violation("empty", "summary is empty or a fragment")]

    found: List[Violation] = []

    if structured:
        found.append(Violation("structured_output", "model returned JSON / a per-speaker breakdown"))

    hits = [m.group(0) for m in re.finditer(_label_pattern(id_labels), text)]
    if hits:
        uniq = ", ".join(dict.fromkeys(h.strip() for h in hits[:8]))
        found.append(Violation("speaker_label", uniq))

    labels_lower = {l.strip().lower() for l in labels}
    prefixes = [
        m.group("label").strip() for m in _LABEL_PREFIX_RE.finditer(raw or text)
        if len(m.group("label").split()) <= 4
    ]
    dialogue = [
        p for p in prefixes
        if p.lower() in labels_lower or _SPEAKER_ID_RE.fullmatch(p) is not None
    ]
    if dialogue or len(prefixes) >= 2:
        found.append(Violation("dialogue_format", "; ".join(dict.fromkeys(dialogue or prefixes))[:80]))

    if _TIMECODE_RE.search(text):
        found.append(Violation("timestamp", _TIMECODE_RE.search(text).group(0)))

    if _MOM_FIELD_RE.search(text) or _METADATA_TOKEN_RE.search(text):
        found.append(Violation("mom_or_metadata", "minutes-style field labels or IDs"))

    if len(words) > plan.hard_max_words:
        found.append(Violation("too_long", f"{len(words)} words vs target {plan.target_words}"))

    if len(words) >= 25 and source_text and _verbatim_overlap(text, source_text) >= 0.5:
        found.append(Violation("verbatim_copy", "copies phrases from the meeting content"))

    letters = [c for c in text if c.isalpha()]
    if letters and sum(1 for c in letters if ord(c) > 0x24F) / len(letters) > 0.2:
        found.append(Violation("non_english", "non-English text"))

    tail = text.rstrip()
    last_word = re.sub(r"[^\w']", "", tail.rsplit(None, 1)[-1]).lower()
    if tail[-1:] in ",;:-–—" or (tail[-1:] not in ".!?\"')" and last_word in _DANGLING_WORDS):
        found.append(Violation("incomplete", "ends mid-sentence"))

    return found


def _instruction(v: Violation, plan: SummaryPlan) -> str:
    return {
        "speaker_label": f"Remove every speaker label or ID ({v.detail}). Do not attribute statements to "
                         "individual speakers — refer to the participants collectively or leave people out.",
        "dialogue_format": "It reads like a transcript or dialogue. Rewrite it as flowing prose describing "
                           "what was discussed and concluded — no 'Label: text' lines.",
        "timestamp": "Remove all timestamps and time codes.",
        "mom_or_metadata": "Remove minutes-of-meeting headings, field labels (Date, Duration, Participants, "
                           "Action Items, ...) and any IDs or metadata; write plain prose only.",
        "too_long": f"It is too long ({v.detail}). Compress it to about {plan.target_words} words (at most "
                    f"{plan.max_words}) by merging related points and dropping minor detail — do not simply "
                    "cut it off.",
        "verbatim_copy": "It copies wording from the meeting content. Paraphrase in your own words and condense.",
        "non_english": "Write it entirely in English.",
        "incomplete": "It ends mid-sentence. Finish it with complete sentences.",
        "structured_output": "Return one plain-text summary, not JSON, a list, or a per-speaker breakdown.",
        "empty": "Write the summary.",
    }.get(v.code, "Fix the problem and keep it a concise prose summary.")


# ═════════════════════════════════════════════════════════════════════════════
# 5. Deterministic last resort (meaning-preserving)
# ═════════════════════════════════════════════════════════════════════════════
def _sub_cased(pattern: str, replacement: str, text: str) -> str:
    """re.sub that capitalises the replacement when it starts a sentence."""
    def repl(m: "re.Match") -> str:
        before = text[: m.start()]
        at_start = not before.strip() or re.search(r"[.!?][\"')\]]*\s*$|\n\s*$", before) is not None
        return replacement[:1].upper() + replacement[1:] if at_start else replacement
    return re.sub(pattern, repl, text)


def neutralize_speaker_labels(text: str, id_labels: Iterable[str] = ()) -> str:
    """Rewrite speaker references as neutral wording instead of deleting sentences.

    ``Speaker_1: We agreed`` -> ``We agreed``; ``Speaker_1 and Speaker_2 agreed`` ->
    ``The participants agreed``; ``Speaker_3's concern`` -> ``A participant's concern``.
    Used only when the LLM could not produce a clean summary, and as a write-time gate.
    """
    if not text:
        return text
    lab = _label_pattern(id_labels)
    text = re.sub(rf"(?m)^[ \t]*[\[(]?\s*{lab}\s*[\])]?\s*[:\-–—]\s*", "", text)
    text = re.sub(rf"[ \t]*[\[(]\s*{lab}\s*[\])]", "", text)
    run = rf"{lab}(?:\s*(?:,|/|&|\band\b|\bor\b)\s*(?:\band\b\s*)?{lab})+"
    text = _sub_cased(run, "the participants", text)
    text = _sub_cased(rf"{lab}(?=['’]s\b)", "a participant", text)
    text = _sub_cased(lab, "a participant", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)       # horizontal space only: never merge lines
    return text.strip()


def _strip_timecodes(text: str) -> str:
    text = _TIMECODE_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"[ \t]+([,.;:!?])", r"\1", text).strip()


def _strip_field_labels(text: str) -> str:
    text = _MOM_FIELD_RE.sub("", text)
    return _METADATA_TOKEN_RE.sub("", text).strip()


def _trim_to_budget(text: str, max_words: int) -> str:
    """Last-resort bound: keep whole leading sentences (the purpose comes first)."""
    kept: List[str] = []
    total = 0
    for para in text.split("\n\n"):
        for sent in _split_sentences(para):
            n = len(sent.split())
            if kept and total + n > max_words:
                return " ".join(kept)
            kept.append(sent)
            total += n
    return " ".join(kept) if kept else text


def finalize_summary(text: str, id_labels: Iterable[str] = ()) -> str:
    """Deterministic write-time gate (no LLM, no length change).

    Guarantees the stored summary has no markdown/preamble, no repeated
    sentences, no timecodes and no speaker labels — whichever code path produced
    it (LLM, extractive fallback, or a legacy caller).
    """
    if not text or not text.strip():
        return text or ""
    cleaned, _ = normalize_summary_text(text)
    cleaned = _strip_timecodes(cleaned)
    cleaned = neutralize_speaker_labels(cleaned, id_labels)
    cleaned = _strip_field_labels(cleaned)
    return cleaned or text.strip()


def summary_preview(text: str, max_chars: int = 400) -> str:
    """Short, label-free preview of a summary that ends on a sentence boundary."""
    cleaned = finalize_summary(text or "")
    if len(cleaned) <= max_chars:
        return cleaned
    kept = ""
    for sent in _split_sentences(cleaned.replace("\n\n", " ")):
        candidate = f"{kept} {sent}".strip()
        if len(candidate) > max_chars:
            break
        kept = candidate
    if kept:
        return kept
    cut = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip(",;:")
    return cut + "…"


# "### 🎯 EXECUTIVE SUMMARY", "**Executive Summary**", "🎯 EXECUTIVE SUMMARY:" — whichever
# heading style the model used — and the next heading of any of those styles.
_MOM_EXEC_HEAD_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*|\*\*|__)?[^\w\n]{0,6}[ \t]*executive[ \t]+summary[^\n]*\n"
)
_MOM_NEXT_HEADING_RE = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}\s"
    r"|\*\*[^\n]{3,80}\*\*[ \t]*$|__[^\n]{3,80}__[ \t]*$"
    r"|[^\w\s]{0,6}[ \t]*[A-Z][A-Z0-9 &/,'\-]{5,}:?[ \t]*$)"      # bare ALL-CAPS / emoji-led title
)


def scrub_mom_executive_summary(mom: str) -> str:
    """Remove speaker labels from the MoM's own 'Executive Summary' section only.

    The rest of the MoM legitimately attributes points to speakers.
    """
    if not mom:
        return mom
    head = _MOM_EXEC_HEAD_RE.search(mom)
    if not head:
        return mom
    start = head.end()
    nxt = _MOM_NEXT_HEADING_RE.search(mom[start:])
    end = start + nxt.start() if nxt else len(mom)
    section = mom[start:end]
    # neutralize_speaker_labels() strips its input; keep the section's own
    # leading/trailing whitespace so the blank line before the NEXT heading
    # survives (otherwise "### NEXT HEADING" is glued onto this section's text).
    lead = section[: len(section) - len(section.lstrip())]
    trail = section[len(section.rstrip()):]
    return mom[:start] + lead + neutralize_speaker_labels(section.strip()) + trail + mom[end:]


# ═════════════════════════════════════════════════════════════════════════════
# 6. Extractive fallback (LLM unavailable)
# ═════════════════════════════════════════════════════════════════════════════
_STOPWORDS = frozenset(
    "a an the and or but if so of to in on at for with without from by as is are was were be been being "
    "am do does did done have has had having i you he she it we they me him her us them my your his its our "
    "their this that these those there here what which who whom whose when where why how not no yes can could "
    "will would shall should may might must just very really also then than too about into over under again "
    "up down out off any all some more most other such only own same so very get got going gonna like okay ok "
    "yeah right well um uh mm hmm actually basically".split()
)
_FILLER_RE = re.compile(r"(?i)\b(?:um+|uh+|uhm|erm|hmm+|you know|i mean|kind of|sort of)\b[,]?\s*")
_LEAD_MARKER_RE = re.compile(r"(?i)^(?:(?:so|okay|ok|well|yeah|yes|right|alright|and|but|now|then)\b[,]?\s*)+")


def extractive_summary(source_text: str, plan: SummaryPlan) -> str:
    """Pick the most central sentences of the (label-free) content, in original order.

    Frequency-based centrality with a small lead/close bonus and redundancy
    suppression. It is the degraded mode for when no LLM is reachable — it always
    stays within the planned length and can never contain speaker labels because
    the content it reads has none.
    """
    sentences: List[str] = []
    for line in (source_text or "").splitlines():
        for s in _split_sentences(line):
            s = _LEAD_MARKER_RE.sub("", _FILLER_RE.sub("", s)).strip(" ,;-")
            if 6 <= len(s.split()) <= 70:
                sentences.append(s[:1].upper() + s[1:])
    if not sentences:
        return ""

    toks = [[w for w in re.findall(r"[a-z][a-z'\-]+", s.lower()) if w not in _STOPWORDS and len(w) > 2]
            for s in sentences]
    tf = Counter(w for t in toks for w in set(t))
    n = len(sentences)
    scored = []
    for i, t in enumerate(toks):
        if not t:
            continue
        base = sum(tf[w] for w in set(t)) / (len(t) ** 0.7)
        bonus = 1.25 if i < 3 else (1.1 if i >= n - 2 else 1.0)
        scored.append((base * bonus, i))

    chosen: List[int] = []
    total = 0
    for _, i in sorted(scored, reverse=True):
        st = set(toks[i])
        if any(len(st & set(toks[j])) / max(1, len(st | set(toks[j]))) >= 0.5 for j in chosen):
            continue
        w = len(sentences[i].split())
        if chosen and total + w > plan.max_words:
            continue
        chosen.append(i)
        total += w
        if total >= plan.target_words:
            break
    if not chosen:
        chosen = [0]

    out = " ".join(s if s[-1] in ".!?" else s + "." for s in (sentences[i] for i in sorted(chosen)))
    return out.strip()


# ═════════════════════════════════════════════════════════════════════════════
# 7. LLM orchestration
# ═════════════════════════════════════════════════════════════════════════════
def _default_llm_call(system_prompt: str, user_text: str, max_tokens: int) -> str:
    from backend.services.model_router import model_router

    llm = model_router.get_llm(max_tokens=max_tokens)
    result = llm.invoke([("system", system_prompt), ("human", user_text)])
    content = getattr(result, "content", "") or ""
    if isinstance(content, list):
        content = " ".join(
            str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content
        )
    return str(content)


def _split_parts(text: str, part_chars: int) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    size = 0
    for line in text.splitlines():
        if current and size + len(line) + 1 > part_chars:
            parts.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        parts.append("\n".join(current))
    return parts


def _condense(text: str, max_chars: int, call: LLMCall, counter: List[int], depth: int = 0) -> str:
    """Map step: turn each part of an over-long meeting into compact notes (recursing if needed)."""
    from backend.prompts.meeting_prompts import summary_notes_prompt

    parts = _split_parts(text, _PART_CHARS)
    prompt = summary_notes_prompt(_NOTES_WORDS_PER_PART)
    max_tokens = _NOTES_WORDS_PER_PART * 3 + 80

    def _notes(part: str) -> str:
        try:
            counter[0] += 1
            return normalize_summary_text(call(prompt, part, max_tokens))[0]
        except Exception as e:
            logger.warning("[Summary] notes call failed for one part (%s); using extractive notes.", e)
            return extractive_summary(part, plan_summary_length(_NOTES_WORDS_PER_PART * 4))

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(parts))) as ex:
        notes = [n for n in ex.map(_notes, parts) if n.strip()]
    combined = "\n".join(notes)
    if len(combined) > max_chars and depth < _MAX_CONDENSE_DEPTH and len(parts) > 1:
        return _condense(combined, max_chars, call, counter, depth + 1)
    return combined


def _draft(src: SummarySource, plan: SummaryPlan, call: LLMCall, counter: List[int]) -> str:
    from backend.prompts.meeting_prompts import summary_prompt

    max_chars = int(CLOUD_MAX_TRANSCRIPT_CHARS)
    content, from_notes = src.text, False
    if len(content) > max_chars:
        logger.info(
            "[Summary] %d chars exceed the single-pass limit (%d); condensing in parts first.",
            len(content), max_chars,
        )
        content, from_notes = _condense(content, max_chars, call, counter), True
        if not content.strip():
            return ""
    prompt = summary_prompt(plan.target_words, plan.min_words, plan.max_words, plan.paragraphs, from_notes)
    counter[0] += 1
    return call(prompt, content, plan.max_tokens)


def _safe(call: LLMCall, prompt: str, text: str, max_tokens: int, counter: List[int]) -> str:
    try:
        counter[0] += 1
        return call(prompt, text, max_tokens) or ""
    except Exception as e:
        logger.warning("[Summary] LLM call failed: %s", e)
        return ""


def generate_meeting_summary(
    transcript: str,
    *,
    llm_call: Optional[LLMCall] = None,
    meeting_ref: str = "",
) -> SummaryResult:
    """Generate a clean, concise summary of ONE meeting from its own transcript.

    ``transcript`` is the full ``Label: text`` transcript (not pre-truncated).
    ``meeting_ref`` is only used to tag log lines. ``llm_call`` lets tests inject a
    fake model; by default the configured AIRAWAT client is used.
    """
    from backend.prompts.meeting_prompts import summary_rewrite_prompt

    src = prepare_summary_source(transcript)
    if src.word_count < MIN_CONTENT_WORDS:
        return SummaryResult("", "empty")

    plan = plan_summary_length(src.word_count)
    call = llm_call or _default_llm_call
    counter = [0]
    tag = f"meeting={meeting_ref or '?'} fp={src.fingerprint}"
    logger.info(
        "[Summary] %s source_words=%d target=%d (max %d) paragraphs=%d labels=%s",
        tag, src.word_count, plan.target_words, plan.max_words, plan.paragraphs, list(src.labels),
    )

    def _check(text: str, structured: bool = False, raw: str = "") -> List[Violation]:
        return check_summary(
            text, plan, source_text=src.text, labels=src.labels, id_labels=src.id_labels,
            structured=structured, raw=raw,
        )

    text, method, violations = "", "llm", []
    try:
        draft = _draft(src, plan, call, counter)
    except Exception as e:
        logger.warning("[Summary] %s draft generation failed: %s", tag, e)
        draft = ""

    if draft.strip():
        text, structured = normalize_summary_text(draft, plan.paragraphs)
        violations = _check(text, structured, raw=draft)
        for attempt in range(1, max(0, int(settings.SUMMARY_MAX_REPAIRS)) + 1):
            if not violations:
                break
            logger.info(
                "[Summary] %s repair %d: %s", tag, attempt, [f"{v.code}({v.detail})" for v in violations],
            )
            rewrite_prompt = summary_rewrite_prompt(
                [_instruction(v, plan) for v in violations],
                plan.target_words, plan.min_words, plan.max_words, plan.paragraphs,
            )
            rewritten = _safe(call, rewrite_prompt, text or draft, plan.max_tokens, counter)
            if not rewritten.strip():
                break
            text, structured = normalize_summary_text(rewritten, plan.paragraphs)
            violations = _check(text, structured, raw=rewritten)
            method = "llm_repaired"

        if violations:
            codes = [v.code for v in violations]
            logger.warning("[Summary] %s LLM could not clear %s — applying deterministic cleanup.", tag, codes)
            if {"speaker_label", "dialogue_format"} & set(codes):
                text = neutralize_speaker_labels(text, src.id_labels)
            if "timestamp" in codes:
                text = _strip_timecodes(text)
            if "mom_or_metadata" in codes:
                text = _strip_field_labels(text)
            if "too_long" in codes:
                text = _trim_to_budget(text, plan.max_words)
            if "incomplete" in codes:
                sentences = _split_sentences(text.replace("\n\n", " "))
                if len(sentences) > 1 and sentences[-1][-1:] not in ".!?":
                    text = " ".join(sentences[:-1])
            text, _ = normalize_summary_text(text, plan.paragraphs)
            violations = _check(text)
            method = "llm_neutralized"
            if "empty" in codes:
                text = ""    # a fragment is not a summary — take the extractive path below

    if not text.strip():
        # No usable LLM output at all — degrade to a label-free extractive summary.
        logger.warning("[Summary] %s no LLM summary available — using extractive fallback.", tag)
        text = extractive_summary(src.text, plan)
        text, _ = normalize_summary_text(text, plan.paragraphs)
        violations = _check(text)
        method = "extractive"

    remaining = [v.code for v in violations]
    (logger.warning if remaining else logger.info)(
        "[Summary] %s done method=%s llm_calls=%d words=%d remaining_violations=%s",
        tag, method, counter[0], len(text.split()), remaining,
    )
    return SummaryResult(text=text, method=method, llm_calls=counter[0], violations=remaining, plan=plan)


def extractive_summary_for_transcript(transcript: str) -> str:
    """Label-free extractive summary of a raw transcript (no LLM). Used by fallbacks."""
    src = prepare_summary_source(transcript)
    if src.word_count < MIN_CONTENT_WORDS:
        return ""
    text = extractive_summary(src.text, plan_summary_length(src.word_count))
    return normalize_summary_text(text)[0]

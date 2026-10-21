"""Pure helper functions with no pipeline dependencies."""

import json
import re
import threading
import time
from typing import Optional, Tuple

# Regex matching lines that already start with a bullet or numbered list marker
_LIST_PREFIX_RE = re.compile(r"^(?:[-*•]|\d+\.)\s")


def format_seconds_to_duration(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    if not seconds:
        return "Unknown Duration"
    try:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "Unknown Duration"


def check_outputs_generated(summary: str, mom: str, filename: str) -> Tuple[bool, bool]:
    """Check if summary and MoM outputs are valid (not placeholders)."""
    is_sum_ok = True
    is_mom_ok = True

    if not summary or not summary.strip():
        is_sum_ok = False
    elif "summary unavailable" in summary.lower():
        is_sum_ok = False
    elif summary.strip().startswith("<"):
        is_sum_ok = False
    elif summary.strip() == f"Executive summary of meeting recording {filename}":
        is_sum_ok = False

    if not mom or not mom.strip():
        is_mom_ok = False
    elif "meeting minutes unavailable" in mom.lower():
        is_mom_ok = False
    elif mom.strip().startswith("<"):
        is_mom_ok = False
    elif len(mom.strip()) < 30:
        is_mom_ok = False

    return is_sum_ok, is_mom_ok


def truncate_transcript(text: str, max_chars: int) -> str:
    """Fit a transcript within max_chars without silently discarding whatever
    fell in the middle. Keeps the head and tail verbatim (opening framing and
    closing decisions/next-steps are the most context-critical), and for
    anything in between — instead of dropping it outright — stride-samples
    lines evenly across the middle so a very long (e.g. 3-5 hour) meeting's
    mid-session content still reaches the LLM, at reduced density but spanning
    the whole timeline, rather than one contiguous gap.
    """
    if len(text) <= max_chars:
        return text

    head_budget = int(max_chars * 0.35)
    tail_budget = int(max_chars * 0.35)
    middle_budget = max_chars - head_budget - tail_budget

    head = text[:head_budget]
    tail = text[-tail_budget:] if tail_budget > 0 else ""
    middle_text = text[head_budget: max(head_budget, len(text) - tail_budget)]

    middle_digest = ""
    if middle_budget > 200 and middle_text.strip():
        lines = [ln for ln in middle_text.split("\n") if ln.strip()]
        if lines:
            avg_len = max(1, sum(len(l) for l in lines) // len(lines))
            target_lines = max(1, middle_budget // avg_len)
            stride = max(1, len(lines) // target_lines)
            sampled = []
            total_len = 0
            for i in range(0, len(lines), stride):
                ln = lines[i]
                if total_len + len(ln) + 1 > middle_budget:
                    break
                sampled.append(ln)
                total_len += len(ln) + 1
            middle_digest = "\n".join(sampled)

    if middle_digest.strip():
        return (
            head
            + "\n\n[... mid-session content condensed — representative excerpts sampled "
              "across the remaining timeline below ...]\n"
            + middle_digest
            + "\n[... end of condensed excerpts ...]\n\n"
            + tail
        )
    return head + "\n\n[... middle section truncated for context window ...]\n\n" + tail


def extract_json_from_text(content: str) -> str:
    """Robustly extract JSON from LLM output (strip markdown fences, think tags)."""
    content = content.strip()
    content = re.sub(r"<think.*?>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL)

    if "```" in content:
        lines = content.split("\n")
        json_lines = []
        inside_fence = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                inside_fence = not inside_fence
                continue
            if inside_fence or not content.startswith("```"):
                json_lines.append(line)
        content = "\n".join(json_lines).strip()

    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        content = match.group(0)
    return content


class LLMJsonError(ValueError):
    """A model reply that is not a usable JSON object.

    ``kind`` is one of ``empty`` / ``truncated`` / ``malformed`` / ``not_object``. ``reason`` is a short description
    that contains NO transcript or reply content (only positions and token classes), so it is safe to log and to hand
    back to the model as the validation error of a repair attempt.
    """

    def __init__(self, reason: str, kind: str = "malformed"):
        super().__init__(reason)
        self.reason = reason
        self.kind = kind


_JSON_TOKEN_CLASSES = {
    "*": "a list bullet ('*')", "-": "a list bullet ('-')", "•": "a list bullet",
    "'": "a single-quoted string", "`": "a Markdown fence or backtick",
    "/": "a comment", "#": "a comment", ".": "an ellipsis or bare number",
}


def _first_json_object(text: str) -> Tuple[Optional[str], bool]:
    """The first brace-balanced ``{...}`` in ``text`` (string- and escape-aware), and whether it was closed.

    Returns ``(None, False)`` when there is no ``{`` at all, ``(candidate, False)`` when the object is never closed
    (a cut-off reply) and ``(candidate, True)`` otherwise. Unlike a greedy ``{ ... }`` regex it stops at the object's
    own closing brace, so prose or a second object after it cannot corrupt the parse.
    """
    start = text.find("{")
    if start < 0:
        return None, False
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1], True
    return text[start:], False


def parse_llm_json_object(raw) -> dict:
    """Parse the single JSON object an LLM was asked for, or raise ``LLMJsonError`` with a content-free reason.

    Accepts the object bare, inside a Markdown fence, after ``<think>`` blocks, or surrounded by prose. Nothing is
    repaired or guessed: an invalid reply (bullets in an array, a trailing comma, a cut-off string) is reported with
    where and what kind of token broke it, so the caller can ask the model again with that reason.
    """
    if isinstance(raw, list):  # some gateways return content parts
        raw = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw)
    text = (raw or "").strip() if isinstance(raw, str) else ""
    if not text:
        raise LLMJsonError("the reply was empty", "empty")
    text = re.sub(r"<think(?:ing)?\b.*?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    fence = re.search(r"```[A-Za-z0-9_-]*[ \t]*\r?\n(.*?)(?:\r?\n[ \t]*```|\Z)", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    candidate, closed = _first_json_object(text)
    if candidate is None:
        raise LLMJsonError("the reply contains no JSON object", "malformed")
    if not closed:
        raise LLMJsonError(
            f"the JSON object is not closed (the reply was cut off after {len(candidate)} characters)", "truncated"
        )
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        found = candidate[exc.pos] if 0 <= exc.pos < len(candidate) else ""
        token = _JSON_TOKEN_CLASSES.get(found, "an unquoted or misplaced token")
        raise LLMJsonError(
            f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno} - found {token}", "malformed"
        ) from exc
    if not isinstance(parsed, dict):
        raise LLMJsonError(f"expected a JSON object but got {type(parsed).__name__}", "not_object")
    return parsed


def speaker_turns(timeline):
    """The speaker TURNS of a timeline: maximal runs of consecutive segments spoken by the same speaker.

    ``timeline``: iterable of ``(start_sec, end_sec, speaker)`` in any order. Returns a list of dicts
    ``{"speaker", "start", "end", "segments"}`` in time order, so that

    * ``len(turns)`` is the turn count (a monologue is ONE turn however many transcript segments it was cut into -
      it was reported as 0 and as 27 for the same meeting by two code paths),
    * ``len(turns) - 1`` is the number of speaker changes,
    * ``sum(t["segments"])`` is the segment count.
    """
    turns = []
    for start, end, speaker in sorted(timeline, key=lambda t: (t[0], t[1])):
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["end"] = max(turns[-1]["end"], end)
            turns[-1]["segments"] += 1
        else:
            turns.append({"speaker": speaker, "start": start, "end": end, "segments": 1})
    return turns


_SPEAKER_PREFIX_RE = re.compile(r"^[^\n:]{1,60}:[ \t]+", re.MULTILINE)


def count_transcript_words(transcript_text: Optional[str]) -> int:
    """Words that were spoken in a "Speaker_1: text" transcript - the ``Speaker_N:`` line labels are not words.

    The same transcript used to be reported as 259 words (segment text) and 260 (label counted), so the word counts of
    the meeting, the quality score and the runtime metrics disagreed by one word per speaker turn.
    """
    return len(_SPEAKER_PREFIX_RE.sub("", transcript_text or "").split())


def to_string_robust(val) -> str:
    """Convert any JSON value to a markdown-friendly string."""
    if val is None:
        return ""
    if isinstance(val, list):
        lines = []
        for item in val:
            if item is not None:
                item_str = str(item).strip()
                if item_str:
                    # Use regex instead of a fragile hardcoded prefix tuple
                    if not _LIST_PREFIX_RE.match(item_str):
                        lines.append(f"* {item_str}")
                    else:
                        lines.append(item_str)
        return "\n".join(lines)
    if isinstance(val, dict):
        return "\n".join([f"**{k}**: {v}" for k, v in val.items()])
    return str(val).strip()


_id_lock = threading.Lock()
_last_generated_ms = 0


def _next_unique_millis() -> int:
    """Millisecond timestamp that is STRICTLY increasing within this process.

    ``time.time()`` alone can repeat: two uploads accepted in the same millisecond (the server
    handles requests on a thread pool) got ids that differed only by 3 random digits — a 1-in-900
    collision that ``MeetingRepository.create`` would treat as "same meeting" and merge.
    """
    global _last_generated_ms
    with _id_lock:
        now_ms = int(time.time() * 1000)
        _last_generated_ms = now_ms if now_ms > _last_generated_ms else _last_generated_ms + 1
        return _last_generated_ms


def generate_numeric_meeting_id(raw_id: Optional[str] = None) -> str:
    """
    Generate or sanitize a dynamic, unique numeric Meeting ID.
    - Strips legacy prefixes ('BVC-', 'BVC_', 'MTG-', 'MTG_') and non-digit separators.
    - If raw_id contains digits (e.g. '1785956313139501' or 'BVC-1785956313139-501'), extracts pure digits ('1785956313139501').
    - If raw_id is missing or non-numeric, generates a 16-digit timestamp + random sequence numeric ID (e.g., '1785956313139501').
    """
    import random
    if raw_id is not None:
        val = str(raw_id).strip()
        for prefix in ("BVC-", "BVC_", "MTG-", "MTG_"):
            if val.upper().startswith(prefix):
                val = val[len(prefix):]
        digits_only = "".join(c for c in val if c.isdigit())
        if digits_only:
            return digits_only

    timestamp_ms = _next_unique_millis()
    rand_seq = random.randint(100, 999)  # nosec B311 - disambiguates across processes, not a secret
    return f"{timestamp_ms}{rand_seq}"



# Agenda texts the platform itself used to write into ``meetings.agenda`` when the uploader gave none. They were not
# what was discussed, so they are read as "no agenda" (rows written by the old code still hold them).
_LEGACY_GENERATED_AGENDA_RE = re.compile(
    r"^\s*(?:discussion recording:.*|discussion and review of session recording|document analysis and review"
    r"|document content review and analysis)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def clean_agenda_text(value: Optional[str]) -> Optional[str]:
    """The agenda text, or None when it is empty or one of the placeholders the platform used to generate."""
    text = (value or "").strip()
    if not text or _LEGACY_GENERATED_AGENDA_RE.match(text):
        return None
    return text

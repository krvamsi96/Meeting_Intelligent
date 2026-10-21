"""LLM Workflow Service
────────────────────
Generates meeting executive summaries, Minutes of Meeting (MOM), action items, and topic
analytics using the AIRAWAT Enterprise AI provider.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from backend.config.config import settings
from backend.prompts.meeting_prompts import (
    action_items_extraction_prompt,
    agenda_key_decisions_prompt,
    analytics_prompt,
    chatbot_system_prompt,
    comparison_topics_prompt,
    detailed_comparison_prompt,
    get_intent_guidance,
    metadata_json_prompt,
    mom_prompt,
    query_rewrite_prompt,
    similar_meeting_enrichment_prompt,
)
from backend.services.agenda_service import compute_batch_match_scores
from backend.services.model_router import model_router
from backend.services.summary_service import (
    extractive_summary_for_transcript,
    finalize_summary,
    generate_meeting_summary,
    scrub_mom_executive_summary,
)
from backend.utils.exceptions import AirawatError
from backend.utils.helpers import LLMJsonError, extract_json_from_text, parse_llm_json_object, to_string_robust

logger = logging.getLogger(__name__)

# Honest placeholder used wherever no project/organization name could actually
# be extracted from the transcript. Previously these fallbacks used
# DEFAULT_PROJECT_NAME (the PLATFORM's own app name, e.g. "AI Meeting
# Intelligence Platform"), which meant every meeting with no extractable
# project name showed the same fabricated, unrelated "project" — this replaces
# fabricated-but-plausible content with an explicit "not known" value.
_PROJECT_NAME_UNKNOWN = "Not Available"

# Lazy import: avoid circular dependency at module level
def _get_action_item_service():
    from backend.services.action_item_service import (
        validate_action_items,
        deduplicate_action_items,
        resolve_action_item_dates,
    )
    return validate_action_items, deduplicate_action_items, resolve_action_item_dates


# Context window caps
def get_analytics_max_chars() -> int:
    return min(12000, model_router.max_transcript_chars)


def get_report_max_chars() -> int:
    return model_router.max_transcript_chars


def get_llm(temperature: float = 0.1):
    """Returns configured LLM via ModelRouter."""
    return model_router.get_llm(temperature=temperature)


GENERIC_AGENDA_TITLES = {
    "meeting", "discussion", "general", "misc", "miscellaneous",
    "overview", "agenda", "notes", "update", "updates", "general discussion"
}

# Words that name no subject on their own. An agenda title or a decision made ONLY of these ("Review", "Discussion",
# "Review and Discussion", "Team Meeting Update", "Discussed the topic") says nothing about what was actually talked
# about or decided, so it is never shown as if it did. A single content word ("Budget Review", "Q3 Review") is enough
# to keep a title.
_GENERIC_TITLE_WORDS = frozenset({
    "meeting", "meetings", "discussion", "discussions", "discuss", "discussed", "discussing", "review", "reviews",
    "reviewed", "reviewing", "general", "misc", "miscellaneous", "overview", "agenda", "notes", "note", "update",
    "updates", "updated", "session", "sessions", "call", "sync", "recording", "recorded", "topic", "topics", "item",
    "items", "point", "points", "other", "others", "various", "and", "of", "the", "a", "an", "on", "for", "to", "in",
    "about", "with", "team", "weekly", "daily", "regular", "standard", "status", "follow", "followup", "up", "catch",
    "chat", "talk", "talked", "conversation", "summary", "recap", "wrap", "closing", "opening", "introduction",
    "introductions", "intro", "none", "na", "nothing", "no", "decision", "decisions", "decided", "key", "main",
    "project", "matter", "matters", "issue", "issues", "content",
})


def _is_generic_text(text: Any) -> bool:
    """True when ``text`` is empty or built only from subject-less words (see ``_GENERIC_TITLE_WORDS``)."""
    words = re.findall(r"[a-z0-9']+", str(text or "").lower())
    return not words or all(w in _GENERIC_TITLE_WORDS for w in words)


def _locate_evidence(
    evidence: str, transcript: str, before: int = 14, after: int = 4, same_sentence: bool = False
) -> Optional[str]:
    """Find the quote the model gave as evidence in the transcript.

    Returns the quoted words plus ``before`` / ``after`` words of surrounding transcript (so a quote that begins just
    after the words "we decided to" still carries them), or None when the quote does not occur in the transcript.
    ``same_sentence`` keeps the look-back inside the quote's own sentence (a sentence ends at . ! ?).
    Word 3-grams, so punctuation, capitalisation and a dropped filler word do not matter but an invented quote does.
    A mostly non-Latin transcript is not checked (the model quotes a translation there): the quote is returned as is.
    """
    ev = re.findall(r"[a-z0-9]+", evidence.lower())
    if not ev:
        return None
    letters = [c for c in transcript if c.isalpha()]
    if letters and sum(ord(c) > 0x24F for c in letters) / len(letters) > 0.3:
        return evidence
    # Words, with the sentence each belongs to (a sentence ends at . ! ?): the context returned for the cue check must
    # not reach into the NEIGHBOURING sentence, or "we agreed to ship on the 3rd. The vendor contract is still open,
    # we will discuss it next week." validates the deferral with the previous sentence's "agreed".
    tr: List[str] = []
    sentence_of: List[int] = []
    sentence = 0
    for tok in re.findall(r"[a-z0-9]+|[.!?]", transcript.lower()):
        if tok in ".!?":
            sentence += 1
        else:
            tr.append(tok)
            sentence_of.append(sentence)
    n = min(3, len(ev))
    where: Dict[Tuple[str, ...], int] = {}
    for i in range(len(tr) - n + 1):
        where.setdefault(tuple(tr[i:i + n]), i)
    hits = [where[g] for g in (tuple(ev[i:i + n]) for i in range(len(ev) - n + 1)) if g in where]
    if len(hits) / max(1, len(ev) - n + 1) < 0.6:
        return None
    first, last = min(hits), max(hits) + n - 1
    lo = max(0, first - before)
    if same_sentence:
        while lo < first and sentence_of[lo] != sentence_of[first]:
            lo += 1
    hi = min(len(tr), last + 1 + after)
    while hi - 1 > last and sentence_of[hi - 1] != sentence_of[last]:
        hi -= 1
    return " ".join(tr[lo:hi])


def _evidence_in_transcript(evidence: str, transcript: str) -> bool:
    """Whether the quote the model gave as evidence really occurs in the transcript."""
    return _locate_evidence(evidence, transcript) is not None

# Canonical "no decision" strings — normalised for deduplication checks
_NO_DECISION_STRINGS = frozenset({
    "no decision recorded.",
    "no formal decision recorded.",
    "no decision recorded",
    "no formal decision recorded",
    "none", "n/a", "null", "no decision", "no key decision",
})

# Explicit decisive-language cue required for a statement to count as a real
# decision (vs. a discussion, question, suggestion, or proposal). Shared by
# both the LLM-extraction grounding check and the extractive fallback path so
# "decision_grounded" means the same thing in both pipelines.
_DECISION_VERBS = (
    r"decided|we (?:have )?agreed|it was agreed|agreed to|approved|resolved to|will proceed with|final decision|"
    r"concluded that|finalized|the decision is|sign(?:ed)? off|greenlit"
)
_DECISION_VERB_RE = re.compile(rf"\b(?:{_DECISION_VERBS})\b", re.IGNORECASE)

# What a decision sounds like in the words people actually said. It is looked for in the transcript quote the model
# cites as evidence: a task ("we will make a feature list"), a plan ("this has to be integrated") or a deferral ("we
# will discuss it later") contains none of these, so it is never presented as a decision.
_DECISION_CUE_RE = re.compile(
    rf"\b(?:{_DECISION_VERBS}"
    r"|agree(?:d|ment)?|approv(?:e|al)|resolved|concluded|finali[sz]e|confirmed|settled on"
    r"|the decision (?:is|was)|our decision|(?:go|going|proceed|proceeding|move|moving) (?:with|ahead|forward)"
    r"|go(?:ing)? ahead with|(?:we|i)(?:['’]ll| will) (?:use|take|choose|adopt|select|pick)|chosen|chose|selected"
    r"|adopted|rejected|declined|postponed|rescheduled|cancel(?:l?ed)?)\b",
    re.IGNORECASE,
)


def _decision_supported(decision: str, evidence: str, transcript: str) -> Tuple[bool, str]:
    """Is ``decision`` backed by what was actually said? Returns ``(supported, reason_if_not)``.

    With the transcript at hand the model's quote must really occur in it and carry decision language. Without the
    transcript (extractive path, unit callers) only the older evidence-or-decisive-verb rule can be applied.
    """
    if not (transcript or "").strip():
        ok = bool(evidence) or bool(_DECISION_VERB_RE.search(decision))
        return ok, "" if ok else "no evidence and no decisive wording"
    if not evidence:
        return False, "no transcript quote given for it"
    context = _locate_evidence(evidence, transcript)
    if context is None:
        return False, "the quoted evidence does not occur in the transcript"
    # 1. The decision wording in the quote's own sentence (or in the quote / decision text itself).
    own = _locate_evidence(evidence, transcript, same_sentence=True) or ""
    if _DECISION_CUE_RE.search(f"{evidence} {decision} {own}"):
        return True, ""
    # 2. The wording may sit just before the quoted words in the PREVIOUS sentence (a sentence split across two
    #    speaker turns, or a quote that starts after "we decided to"). It only counts when that wording is about THIS
    #    decision - shares content words with it - so a neighbouring decision ("we agreed to ship on the 3rd") cannot
    #    turn "we will discuss the vendor contract next week" into one.
    wide, own_words = context.split(), own.split()
    before_quote = " ".join(wide[: max(0, len(wide) - len(own_words))])
    if before_quote and _DECISION_CUE_RE.search(before_quote):
        decision_words = {w for w in _TITLE_WORD_RE.findall(decision.lower()) if len(w) >= 4 and w not in _GENERIC_TITLE_WORDS}
        if len(decision_words & set(before_quote.split())) >= 2:
            return True, ""
    return False, "the quoted words are a task, plan or discussion, not a decision"


_TITLE_WORD_RE = re.compile(r"[a-z0-9]+")

# Fraction of a generated topic title's informative words that must occur in the transcript.
_MIN_TITLE_SUPPORT = 0.5
# Phrases with which a speaker announces what is going to be covered. An agenda item is "explicit" only when such a
# phrase and (most of) the title's words are in the SAME sentence.
_ANNOUNCE_CUE_RE = re.compile(
    r"\bagenda\b|going to (?:discuss|talk about|cover|go through|look at)|will (?:discuss|cover|go through)|"
    r"meeting is (?:about|on)|topics? (?:for|of|are|today)|let'?s (?:discuss|talk about|start with|begin with|look at|"
    r"move on to)|we (?:need|have|want) to (?:discuss|talk about|cover)|first of all|first item|next (?:topic|item)"
)


def _make_title_checker(transcript: str):
    """``check(title) -> (support, announced)`` for a transcript, or None when there is nothing to check against
    (no transcript, or a mostly non-Latin one whose words these checks cannot compare).

    ``support``: share of the title's informative words found in the transcript (exact or 5-letter stem).
    ``announced``: a cue sentence (see ``_ANNOUNCE_CUE_RE``) holds at least half of those words.
    """
    text = (transcript or "").lower()
    if not text.strip():
        return None
    letters = [c for c in text if c.isalpha()]
    if letters and sum(ord(c) > 0x24F for c in letters) / len(letters) > 0.3:
        return None
    tokens = set(_TITLE_WORD_RE.findall(text))
    stems = {t[:5] for t in tokens}
    cue_sentences = [
        set(_TITLE_WORD_RE.findall(s)) for s in re.split(r"(?<=[.!?])\s+|\n+", text) if _ANNOUNCE_CUE_RE.search(s)
    ]

    def check(title: str):
        words = [w for w in _TITLE_WORD_RE.findall(title.lower()) if len(w) > 2 and w not in _GENERIC_TITLE_WORDS]
        if not words:
            return 1.0, False
        support = sum(1 for w in words if w in tokens or w[:5] in stems) / len(words)
        announced = any(sum(1 for w in words if w in s) / len(words) >= 0.5 for s in cue_sentences)
        return support, announced

    return check


def _arbitrate_shared_evidence(items: List[Dict[str, Any]]) -> None:
    """When several agenda items cite the SAME transcript quote, only the item that quote actually belongs to keeps
    its decision (in place; the other agenda items stay, without a decision).

    A single statement of a decision is evidence for one relationship, not for every topic the model attached it to.
    "Belongs to" is the item whose title shares the largest fraction of its content words with the decision and the
    quote; ties go to the earliest item. Nothing is guessed: a decision that loses is simply left empty.
    """
    groups: Dict[str, List[int]] = {}
    for i, item in enumerate(items):
        if item.get("decision_grounded") and item.get("evidence"):
            key = " ".join(_TITLE_WORD_RE.findall(str(item["evidence"]).lower()))
            if key:
                groups.setdefault(key, []).append(i)

    def _fit(i: int) -> float:
        words = {w for w in _TITLE_WORD_RE.findall(str(items[i].get("agenda", "")).lower()) if w not in _GENERIC_TITLE_WORDS}
        if not words:
            return 0.0
        text = set(_TITLE_WORD_RE.findall(f"{items[i].get('key_decision', '')} {items[i].get('evidence', '')}".lower()))
        return len(words & text) / len(words)

    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        winner = max(idxs, key=lambda i: (_fit(i), -i))
        for i in idxs:
            if i == winner:
                continue
            logger.info(
                "[AgendaExtraction] Decision for agenda %r dropped: its evidence quote is shared with %r, which fits it "
                "better.", str(items[i].get("agenda", ""))[:60], str(items[winner].get("agenda", ""))[:60],
            )
            items[i].update({
                "key_decision": "No formal decision recorded.", "decision": "", "decision_grounded": False,
                "evidence": None, "reason": None, "confidence": None,
            })


def validate_and_format_agenda_key_decisions(
    raw_items: Any, default_source: str = "llm_detected", transcript: str = ""
) -> List[Dict[str, Any]]:
    """Validate, clean, deduplicate, order, and enrich Agenda & Decisions.

    Each validated item gets a real semantic Match Score computed via
    sentence-transformer cosine similarity (no hardcoding, no random values).
    Scores are computed in a single batch pass for efficiency.

    Passes through ``evidence`` (transcript grounding quote), ``speaker``
    (who made the decision), ``reason``, ``timestamp``, ``confidence``, and ``agenda_type``.

    Nothing is invented to fill a gap: an agenda made only of subject-less words ("Review", "Discussion") is dropped,
    and a decision that the transcript does not support is emptied (the agenda item stays, without a decision).
    ``transcript`` is the text the model read; with it, a decision needs a quote that truly occurs in it.
    """
    if not isinstance(raw_items, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    seen_agendas = set()
    seen_decisions = set()
    order = 1
    transcript_check = _make_title_checker(transcript)

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        agenda = str(item.get("agenda") or item.get("title") or "").strip()
        raw_agenda_type = str(item.get("agenda_type") or "inferred").strip().lower()
        agenda_type = "explicit" if "explicit" in raw_agenda_type else "inferred"
        key_decision = str(item.get("key_decision") or item.get("decision") or "").strip()
        reason = str(item.get("reason") or "").strip() or None
        evidence = str(item.get("evidence") or "").strip() or None
        raw_spk = str(item.get("speaker") or "").strip()
        speaker = None if not raw_spk or raw_spk.lower() == "unknown" else raw_spk
        timestamp = str(item.get("timestamp") or "").strip() or None
        # The model's own confidence, or None. It used to become 0.90 / 0.75 / 0.80 depending on whether a quote
        # was present - a number nobody measured, shown as the decision's "confidence".
        raw_conf = item.get("confidence")
        try:
            confidence = float(raw_conf) if raw_conf is not None and str(raw_conf).strip() != "" else None
        except (TypeError, ValueError):
            confidence = None

        source = str(item.get("source") or default_source).strip()

        # 1. Skip empty agenda
        if not agenda:
            continue

        # 2. Skip generic/meaningless agenda titles if generated by LLM ("Review", "Discussion", "Review and
        #    Discussion" ...): they name nothing that was discussed. A title the user typed is always kept.
        if source != "user" and (agenda.lower() in GENERIC_AGENDA_TITLES or _is_generic_text(agenda)):
            logger.info("[AgendaExtraction] Dropped subject-less agenda title %r.", agenda)
            continue

        # 2b. With the transcript at hand a generated topic must be about something that was SAID, and "explicit"
        #     (an announced agenda item) is decided from the transcript, not taken from the model: the model's label
        #     flipped between runs of the same meeting and was stored as "explicit" for topics nobody announced.
        if source != "user" and transcript_check is not None:
            support, announced = transcript_check(agenda)
            if support < _MIN_TITLE_SUPPORT:
                logger.info("[AgendaExtraction] Dropped agenda title %r: its words are not in the transcript.", agenda)
                continue
            agenda_type = "explicit" if announced else "inferred"
        elif source == "user":
            agenda_type = "explicit"

        # 3. Normalise empty / placeholder / subject-less decision, then require the transcript to support it: a
        #    decision with no quote, a quote that is not in the transcript, or a quote that is a task / plan /
        #    deferral rather than a decision is not a decision. The agenda item itself stays.
        is_no_dec = not key_decision or key_decision.lower() in _NO_DECISION_STRINGS or _is_generic_text(key_decision)
        if not is_no_dec:
            supported, why_not = _decision_supported(key_decision, evidence or "", transcript)
            if not supported:
                logger.info("[AgendaExtraction] Dropped decision %r for agenda %r: %s.", key_decision[:80], agenda[:60], why_not)
                is_no_dec = True
                evidence = None
        if is_no_dec:
            key_decision = "No formal decision recorded."
            reason = None
            confidence = None

        agenda_norm = agenda.lower()
        decision_norm = key_decision.lower()

        # 4. Filter duplicate agendas
        if agenda_norm in seen_agendas:
            continue

        # 5. Filter duplicate real decisions (allow multiple "no decision" entries)
        if not is_no_dec and decision_norm in seen_decisions:
            continue

        seen_agendas.add(agenda_norm)
        if not is_no_dec:
            seen_decisions.add(decision_norm)

        # Every decision that survives step 3 is backed by the transcript (see _decision_supported).
        is_grounded = not is_no_dec

        ag_id = f"A{order}"
        cleaned.append({
            "id": ag_id,
            "agenda_id": ag_id,
            "agenda": agenda,
            "title": agenda,
            "agenda_type": agenda_type,
            "source": source,
            "key_decision": key_decision,
            "decision": "" if is_no_dec else key_decision,
            "reason": reason,
            "decision_grounded": is_grounded,
            "evidence": evidence,
            "speaker": speaker,
            "timestamp": timestamp,
            "confidence": confidence,
            "agenda_order": order,
            "match_score": None,
            "relevance_score": None,
            "confidence_level": None,
            "embedding_model": None,
        })
        order += 1

    if not cleaned:
        return []

    # ── Semantic duplicate removal ─────────────────────────────────────────────
    # Exact-string dedup above only catches identical text. Two agenda items
    # that paraphrase the same topic ("Budget approval was discussed." vs.
    # "Approval of the budget was discussed.") must also collapse to one.
    try:
        from backend.services.agenda_service import deduplicate_semantic
        agenda_texts = [item["agenda"] for item in cleaned]
        keep_idx = set(deduplicate_semantic(agenda_texts))
        if len(keep_idx) < len(cleaned):
            logger.info(
                f"[LLMWorkflow] Semantic dedup collapsed {len(cleaned) - len(keep_idx)} "
                f"near-duplicate agenda item(s)."
            )
            cleaned = [item for i, item in enumerate(cleaned) if i in keep_idx]

        # Two DIFFERENT agenda topics can still restate the same decision.
        # Blank out later duplicates' decision (keep the agenda item itself —
        # its topic may be genuinely distinct even if the decision repeats).
        real_dec_positions = [i for i, item in enumerate(cleaned) if item["decision_grounded"]]
        if len(real_dec_positions) > 1:
            dec_texts = [cleaned[i]["key_decision"] for i in real_dec_positions]
            keep_dec_local = set(deduplicate_semantic(dec_texts))
            for local_i, global_i in enumerate(real_dec_positions):
                if local_i not in keep_dec_local:
                    cleaned[global_i]["key_decision"] = "No formal decision recorded."
                    cleaned[global_i]["decision"] = ""
                    cleaned[global_i]["decision_grounded"] = False
                    cleaned[global_i]["evidence"] = None
                    cleaned[global_i]["reason"] = None
                    cleaned[global_i]["confidence"] = None

        # Renumber sequentially after any removals so agenda_order/agenda_id
        # stay contiguous (A1, A2, ...) with no gaps.
        for new_order, item in enumerate(cleaned, start=1):
            new_id = f"A{new_order}"
            item["id"] = new_id
            item["agenda_id"] = new_id
            item["agenda_order"] = new_order
    except Exception as e:
        logger.warning(f"[LLMWorkflow] Semantic dedup failed: {e}")

    # ── One quote supports one relationship ───────────────────────────────────
    _arbitrate_shared_evidence(cleaned)

    # ── Batch compute real semantic match scores ──────────────────────────────
    try:
        pairs = [(item["agenda"], item["key_decision"]) for item in cleaned]
        scores = compute_batch_match_scores(pairs)
        for item, score_info in zip(cleaned, scores):
            item["match_score"] = score_info.get("match_score")
            item["relevance_score"] = score_info.get("match_score")
            item["confidence_level"] = score_info.get("confidence_level")
            item["embedding_model"] = score_info.get("model_name")
    except Exception as e:
        logger.warning(f"[LLMWorkflow] Batch match score computation failed: {e}")

    return cleaned


def _make_sys_prompt(system_prompt: str) -> str:
    """Safely escape curly braces in system prompts for LangChain ChatPromptTemplate."""
    clean = system_prompt.replace("{{", "{").replace("}}", "}")
    return clean.replace("{", "{{").replace("}", "}}")


def escape_template_braces(text) -> str:
    """Escape ``{`` / ``}`` in text that is interpolated INTO a ChatPromptTemplate template string.

    LangChain treats every ``{name}`` in a template as an input variable. Meeting-derived text
    (a title such as "Q3 {budget} review") put into the template itself made every invocation
    fail with "missing input", so the chatbot answered "I encountered an error" for that meeting.
    Text passed as a VARIABLE at invoke() time needs no escaping — only text baked into the template.
    """
    return str(text).replace("{", "{{").replace("}", "}}")


# Conversational preambles some models prepend before the real answer.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:sure|certainly|of course|absolutely|here(?:'s| is| are)|below (?:is|are)|"
    r"the following is|i(?:'ve| have) (?:written|prepared|created)|as requested)\b"
    r"[^\n:]{0,120}[:\-–]\s*\n?",
    re.IGNORECASE,
)


def _polish_prose(text: str) -> str:
    """Light, content-preserving cleanup for Summary / MoM prose.

    Removes model preambles ("Here is a polished summary:"), drops exact
    consecutive duplicate sentences / lines, and normalises whitespace. Does NOT
    paraphrase, reorder, or add anything — the content stays exactly as generated
    from THIS meeting's transcript.
    """
    if not text or not text.strip():
        return text
    out = text.strip()
    # Strip up to two stacked preamble lines.
    for _ in range(2):
        new = _PREAMBLE_RE.sub("", out, count=1).strip()
        if new == out:
            break
        out = new
    # De-duplicate consecutive identical non-empty lines.
    lines = out.split("\n")
    deduped_lines: List[str] = []
    for ln in lines:
        if deduped_lines and ln.strip() and ln.strip() == deduped_lines[-1].strip():
            continue
        deduped_lines.append(ln)
    out = "\n".join(deduped_lines)
    # De-duplicate consecutive identical sentences within a paragraph.
    def _dedup_sentences(par: str) -> str:
        parts = re.split(r'(?<=[.!?])\s+', par)
        seen_prev = None
        kept = []
        for p in parts:
            norm = re.sub(r"\s+", " ", p.strip().lower())
            if norm and norm == seen_prev:
                continue
            seen_prev = norm
            kept.append(p)
        return " ".join(kept)
    out = "\n".join(_dedup_sentences(l) if l.strip() and not l.lstrip().startswith(("#", "-", "*", "|")) else l
                    for l in out.split("\n"))
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def _mom_length_directives(transcript: str):
    """Derive (mom_exec_guidance, mom_max_tokens) for the MoM's own Executive
    Summary section from the actual size of the transcript, so a short meeting
    is not padded and a long one is not cut short.

    (The standalone ``analysis.summary`` is sized separately, and far more
    tightly, by summary_service.plan_summary_length.)
    """
    word_count = len((transcript or "").split())

    if word_count < 150:
        return "2-3 sentences", 500
    if word_count < 800:
        return "1 short paragraph", 800
    if word_count < 3000:
        return "1-2 paragraphs", 1200
    if word_count < 10000:
        return "2 paragraphs", 1700
    return "2-3 paragraphs", 2000


def _strict_llm_failure(section: str, exc: Exception) -> None:
    """A failed AIRAWAT call inside a worker must SURFACE (the pipeline retries the report,
    then fails the meeting with the reason). These workers used to swallow every exception and return an empty /
    default section, which then looked exactly like the model's real answer ("no action items", "Neutral")."""
    if isinstance(exc, AirawatError):
        raise exc
    raise AirawatError(
        f"AI Provider failed while generating the {section}: {type(exc).__name__}: {str(exc)[:200]}"
    ) from exc


def _generate_section(
    llm, system_prompt: str, transcript: str, max_chars: int = None
) -> str:
    """Helper: runs a focused single-section LLM call."""
    if max_chars is None:
        max_chars = model_router.max_transcript_chars
    sys_prompt = _make_sys_prompt(system_prompt)
    prompt = ChatPromptTemplate.from_messages(
        [("system", sys_prompt), ("user", "{transcript}")]
    )
    try:
        result = (prompt | llm).invoke({"transcript": transcript[:max_chars]})
        content = result.content.strip()
        content = re.sub(r"<think.*?>.*?</think>", "", content, flags=re.DOTALL).strip()
        content = re.sub(
            r"<thinking.*?>.*?</thinking>", "", content, flags=re.DOTALL
        ).strip()
        return content
    except Exception as e:
        _strict_llm_failure("minutes of meeting", e)
        logger.warning(f"Section generation failed: {e}")
        return ""


def _extract_agenda_key_decisions(llm, truncated: str, user_agenda: str = "") -> List[Dict[str, Any]]:
    """Dedicated parallel LLM call for Agenda-Decision extraction with evidence.

    Uses the specialized ``agenda_key_decisions_prompt`` which:
    - Supports user-provided agenda topics ('source': 'user') and LLM topic detection ('source': 'llm_detected').
    - Instructs the LLM to extract only genuine decisions (not comments/greetings).
    - Requires exact transcript evidence for every decision.
    - Identifies the speaker who made the decision.
    - Semantically associates each decision to the correct agenda (not by position).
    - Computes a real cosine-similarity match score via sentence-transformers.

    Retry policy:
    - The detailed prompt (evidence required); an empty list is the model's answer.
    - A failed call is retried once, then raised as AirawatError (never replaced by another engine).
    """
    sys_prompt = _make_sys_prompt(agenda_key_decisions_prompt(user_agenda=user_agenda))
    prompt = ChatPromptTemplate.from_messages(
        [("system", sys_prompt), ("user", "{transcript}")]
    )

    def _invoke_and_parse(pr, label: str) -> List[Dict[str, Any]]:
        result = (pr | llm).invoke({"transcript": truncated})
        content = result.content.strip()
        # Strip <think>...</think> or <thinking>...</thinking> blocks (some models)
        content = re.sub(r"<think.*?>.*?</think>", "", content, flags=re.DOTALL).strip()
        content = re.sub(r"<thinking.*?>.*?</thinking>", "", content, flags=re.DOTALL).strip()
        content = extract_json_from_text(content)
        data = json.loads(content)
        raw_akd = data.get("agenda_key_decisions", [])
        if not isinstance(raw_akd, list):
            logger.warning(f"[AgendaExtraction] {label}: LLM returned non-list field.")
            return []
        default_source = "user" if user_agenda and user_agenda.strip() else "llm_detected"
        validated = validate_and_format_agenda_key_decisions(raw_akd, default_source=default_source, transcript=truncated)
        logger.info(f"[AgendaExtraction] {label}: {len(validated)} agenda-decision pair(s).")
        return validated

    # AIRAWAT is the only provider: an empty answer is the answer (no lower-precision "simplified" prompt that
    # squeezes decisions out of a model that found none), and a failure is retried ONCE and then reported.
    last_error: Exception = RuntimeError("no attempt was made")
    for attempt in (1, 2):
        try:
            return _invoke_and_parse(prompt, f"Attempt {attempt} (detailed prompt)")
        except Exception as e:
            last_error = e
            logger.warning(f"[AgendaExtraction] Attempt {attempt} failed: {e}.")
    _strict_llm_failure("agenda and decisions", last_error)
    return []


def generate_extractive_meeting_report(
    transcript: str,
    meeting_date: str = "Unknown Date",
    meeting_time: str = "Unknown Time",
    duration_str: str = "Unknown Duration",
    reference_dt=None,
) -> Dict[str, Any]:
    """
    Intelligent extractive fallback summarizer & MoM generator.
    Parses transcript text and speech segments to build rich, professional,
    structured Executive Summary, MoM, Action Items, Decisions, and Risks.
    Guarantees 100% reliable output even when remote LLM APIs are unreachable.

    ``reference_dt`` is a timezone-aware datetime used for deadline resolution
    (§9.12).  When ``None``, the date_resolver will fall back to current UTC time.
    """
    clean_text = transcript.strip()

    # ── Empty / degenerate transcript guard (BUG-SAMEOUTPUT) ─────────────────
    # Previously this substituted a fixed string ("Meeting recorded with audio
    # content.") when the transcript was empty, which made EVERY content-less
    # meeting produce the *identical* summary / agenda / decisions. That is the
    # "same output for every meeting" symptom. Now we return an honest, empty
    # structure instead — different meetings never share fabricated content.
    _meaningful = re.sub(r"^[A-Za-z0-9_\-. ]{1,30}:\s*", "", clean_text, flags=re.MULTILINE)
    _meaningful = re.sub(r"\s+", " ", _meaningful).strip()
    if len(_meaningful) < 40 or len({w.lower() for w in _meaningful.split() if len(w) > 2}) < 8:
        logger.warning(
            "[ExtractiveReport] Transcript has no usable content (%d meaningful chars). "
            "Returning EMPTY report — not boilerplate.", len(_meaningful),
        )
        _na = "Not Available — transcript did not contain enough content to analyse."
        return {
            "summary": _na,
            "mom": _na,
            "action_items": "No specific action items identified.",
            "structured_action_items": [],
            "decisions": "No formal decisions were recorded.",
            "agenda_key_decisions": [],
            "risks": "No critical risks or blockers identified during this session.",
            "project_name": _PROJECT_NAME_UNKNOWN,
            "sentiment": "Neutral",
            "emotions": "Neutral: 100%",
            "questions_raised": "[]",
            "keyword_frequency": "{}",
        }

    # Parse lines and sentences
    lines = [l.strip() for l in clean_text.splitlines() if l.strip()]

    # Extract speakers
    speakers = set()
    for line in lines:
        if ":" in line[:30]:
            part = line.split(":", 1)[0].strip()
            if len(part) < 30 and not part.startswith("http"):
                speakers.add(part)
    speakers_list = sorted(list(speakers)) if speakers else ["Participants"]
    attendees_str = ", ".join(speakers_list)

    # Key sentences extraction
    sentences = []
    for line in lines:
        content = line.split(":", 1)[-1].strip() if ":" in line[:30] else line
        for s in content.replace("!", ".").replace("?", ".").split("."):
            s_clean = s.strip()
            if len(s_clean) > 15:
                sentences.append(s_clean)

    # Transcript-grounded decision lines (used by both the MoM section and the
    # dedicated decisions field below). SENTENCE-level, explicit-verb match so a
    # decision line is a concise real statement — not a whole speaker paragraph
    # that merely mentions "agreed" once.
    import re as _re_dec
    _dec_prefix_re = _re_dec.compile(r'^[A-Za-z0-9_\- ]{1,30}:\s*')
    decision_lines = []
    _seen_dl = set()
    for l in lines:
        _body = _dec_prefix_re.sub('', l).strip()
        for _sent in re.split(r'(?<=[.!?])\s+', _body):
            _sent = _sent.strip()
            if 8 <= len(_sent.split()) <= 45 and _DECISION_VERB_RE.search(_sent):
                _k = _sent.lower()[:80]
                if _k not in _seen_dl:
                    _seen_dl.add(_k)
                    decision_lines.append(_sent)

    # 1. Summary — a short, label-free, length-planned extractive overview.
    # It deliberately carries NO participants, date/time/duration or headings
    # (that is metadata / MoM content) and is never a raw dump of the opening
    # transcript lines. See summary_service.extractive_summary.
    summary = extractive_summary_for_transcript(clean_text) or (
        "Not Available — transcript did not contain enough content to analyse."
    )

    # 2. Minutes of Meeting (MoM)
    mom_parts = []
    mom_parts.append("# Minutes of Meeting (MoM)\n")
    mom_parts.append(f"**Date:** {meeting_date}  ")
    mom_parts.append(f"**Time:** {meeting_time}  ")
    mom_parts.append(f"**Duration:** {duration_str}  ")
    mom_parts.append(f"**Attendees:** {attendees_str}\n")
    mom_parts.append("---")
    mom_parts.append("### 1. Meeting Objectives")
    # Objectives must come from the transcript — never boilerplate. Use the first
    # few substantive sentences; if there are none, say so honestly.
    if sentences:
        for s in sentences[:3]:
            mom_parts.append(f"- {s}.")
    else:
        mom_parts.append("- Not explicitly stated in the transcript.")
    mom_parts.append("")
    mom_parts.append("### 2. Key Discussion Topics")

    if lines:
        step = max(1, len(lines) // 4)
        for i in range(0, min(len(lines), 4 * step), step):
            chunk = lines[i : i + step]
            topic_num = i // step + 1
            mom_parts.append(f"\n#### Topic {topic_num}: Key Segment Discussion")
            for l in chunk[:3]:
                mom_parts.append(f"- {l}")
    else:
        mom_parts.append("- No discussion content could be extracted from the transcript.")

    mom_parts.append("\n### 3. Decisions & Next Steps")
    # Reuse the transcript-grounded decision lines extracted below; do not fabricate.
    if decision_lines:
        for idx, dl in enumerate(decision_lines[:5], start=1):
            mom_parts.append(f"{idx}. {dl}")
    else:
        mom_parts.append("- No formal decisions were recorded in the transcript.")

    mom = "\n".join(mom_parts)

    # 3. Action Items Extraction
    from backend.services.action_item_service import (
        extract_action_items_extractive,
        resolve_action_item_dates,
    )
    structured_ai = extract_action_items_extractive(transcript)
    # Deterministic date resolution (§9.16)
    if reference_dt is not None:
        structured_ai = resolve_action_item_dates(structured_ai, reference_dt)
    if structured_ai:
        action_items = "\n".join([
            f"- [ ] **{it['task']}** - @{it['owner']} "
            f"(Deadline: {it['deadline']}"
            + (f" | Due: {it['due_date']}" if it.get('due_date') else "")
            + f" | Status: {(it.get('status') or 'open').title()})"
            for it in structured_ai[:5]
        ])
    else:
        # Bug 4 fix: when the extractive parser finds no action items, return an
        # honest empty state — NEVER inject hardcoded sample-specific task strings.
        action_items = "No specific action items identified."
        # structured_ai may already be [] from the extract call above

    # 4. Agenda & Key Decisions Extraction
    import re as _re_early
    # `decision_lines` was already computed (sentence-level, explicit verbs) above.
    if decision_lines:
        decisions = "\n".join([f"{idx+1}. {dl}" for idx, dl in enumerate(decision_lines[:5])])
    else:
        # When no explicit decision was stated, report an honest empty state —
        # NEVER fabricate project-specific decisions.
        decisions = "No formal decisions were recorded."

    # Build structured agenda_key_decisions from the extractive path.
    # Strategy:
    # 1. Identify substantive non-greeting lines as potential agenda topics.
    # 2. For each candidate agenda topic, search for a decision-keyword sentence
    #    that shares significant keyword overlap (semantic association heuristic).
    # 3. Provide the matching transcript line as evidence.
    # 4. Match scores are computed via cosine similarity (sentence-transformers),
    #    NOT hardcoded or random.
    _filler_kws = ["good morning", "hello", "hi ", "welcome", "thanks", "thank you", "okay", "alright", "great", "sure", "yes,", "no,"]
    _decision_kws = [
        "decided", "agreed", "approved", "resolved", "confirmed", "concluded", "finalized",
        "will ", "shall ", "must ", "complete", "deploy", "prepare", "submit", "assign",
        "deadline", "by friday", "by monday", "by tomorrow", "next week",
    ]
    _action_only_kws = ["i will", "i'll", "i am going", "i can"]
    # Regex to strip speaker labels like "Speaker A:", "SPEAKER_00:", "Ravi:"
    import re as _re
    _speaker_prefix_re = _re.compile(r'^[A-Za-z0-9_\- ]{1,30}:\s*')

    agenda_decisions_raw = []

    # ── Topic mining (content-derived, per-transcript) ──────────────────────
    # Instead of using each line's first 60 chars (which yields mid-sentence
    # fragments like "On a case-by-case basis or something."), mine the RECURRING
    # subjects of THIS transcript: capitalised multi-word phrases and repeated
    # significant terms. The most-discussed subjects become the agenda topics.
    _clean_lines = []
    for l in lines:
        stripped = l.strip().lstrip("0123456789.-*•# ").strip()
        if len(stripped) < 15:
            continue
        content = _speaker_prefix_re.sub('', stripped).strip()
        if len(content) >= 10:
            _clean_lines.append(content)

    _full_content = " ".join(_clean_lines)

    # Frequency-based topic mining on the lowercased content — robust to grammar
    # normalisation (which capitalises every sentence, so a "capitalised phrase"
    # heuristic yields junk like "With That"). We take the most-repeated 2–3 word
    # collocations whose words are all content words, plus hot single terms.
    _STOP_TOPIC = {
        "i", "we", "you", "he", "she", "it", "they", "the", "a", "an", "this",
        "that", "these", "those", "there", "here", "then", "than", "with", "from",
        "into", "onto", "your", "our", "their", "his", "her", "its", "and", "but",
        "or", "nor", "for", "yet", "so", "as", "at", "by", "of", "on", "in", "to",
        "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
        "did", "have", "has", "had", "will", "would", "shall", "should", "can",
        "could", "may", "might", "must", "not", "no", "yes", "okay", "yeah",
        "well", "now", "also", "just", "like", "get", "got", "very", "really",
        "today", "think", "know", "want", "going", "said", "says", "one", "two",
        "because", "about", "what", "which", "who", "when", "where", "why", "how",
        "some", "any", "all", "more", "most", "much", "many", "good", "great",
        "sure", "thing", "things", "lot", "kind", "sort", "way", "time", "point",
        "morning", "everyone", "thank", "thanks", "hello", "speaker", "meeting",
        "discuss", "discussed", "discussion", "talk", "talked", "actually",
        "basically", "maybe", "little", "bit", "something", "someone", "anything",
    }

    def _is_content_word(w: str) -> bool:
        return len(w) >= 3 and w not in _STOP_TOPIC and not w.isdigit()

    _norm = _re.sub(r"[^a-z0-9\s]", " ", _full_content.lower())
    _norm = _re.sub(r"\s+", " ", _norm).strip()
    _toks = _norm.split()

    _phrase_freq: Dict[str, int] = {}
    for n in (3, 2):
        for i in range(len(_toks) - n + 1):
            grp = _toks[i:i + n]
            if all(_is_content_word(w) for w in grp):
                key = " ".join(grp)
                _phrase_freq[key] = _phrase_freq.get(key, 0) + 1

    _word_freq: Dict[str, int] = {}
    for w in _toks:
        if _is_content_word(w):
            _word_freq[w] = _word_freq.get(w, 0) + 1

    _phrase_first_line: Dict[str, str] = {}

    def _rep_line(topic_words: List[str]) -> str:
        for cl in _clean_lines:
            low = cl.lower()
            if all(_re.search(rf"\b{_re.escape(w)}\b", low) for w in topic_words):
                return cl
        return " ".join(topic_words)

    _agenda_cap = max(4, min(15, max(1, len(_clean_lines) // 6)))
    ranked_topics: List[str] = []
    _covered: set = set()
    # Multi-word collocations first — must repeat at least twice to be a "topic"
    # (or once if the transcript is very short).
    _min_phrase_freq = 2 if len(_toks) > 200 else 1
    for ph, c in sorted(_phrase_freq.items(), key=lambda kv: (kv[1], len(kv[0].split())), reverse=True):
        if c < _min_phrase_freq:
            continue
        pw = ph.split()
        if any(w in _covered for w in pw):
            continue
        title = ph.title()
        ranked_topics.append(title)
        _phrase_first_line[title] = _rep_line(pw)
        _covered.update(pw)
        if len(ranked_topics) >= _agenda_cap:
            break
    # Fill remaining slots with hot single terms not already covered.
    if len(ranked_topics) < _agenda_cap:
        for w, c in sorted(_word_freq.items(), key=lambda kv: kv[1], reverse=True):
            if c < max(3, len(_toks) // 400) or w in _covered:
                continue
            title = w.title()
            ranked_topics.append(title)
            _phrase_first_line[title] = _rep_line([w])
            _covered.add(w)
            if len(ranked_topics) >= _agenda_cap:
                break

    seen_ag = set()
    ord_idx = 1
    for ag_title in ranked_topics:
        content = _phrase_first_line.get(ag_title, ag_title)
        ag_norm = ag_title.lower().strip()
        if not ag_title or len(ag_title) < 4 or ag_norm in seen_ag or ag_norm in GENERIC_AGENDA_TITLES:
            continue

        seen_ag.add(ag_norm)

        # Classify: is this line itself a decision or an action item?
        is_decision_line = any(kw in content.lower() for kw in _decision_kws)
        is_personal_action = any(kw in content.lower() for kw in _action_only_kws)

        # Find supporting decision line (semantic keyword overlap with agenda topic words)
        dec_text = "No formal decision recorded."
        evidence_text = None
        ag_words = [w for w in ag_norm.split() if len(w) > 3]
        for dl in decision_lines:
            # Strip speaker prefix from decision line too
            dl_clean = _speaker_prefix_re.sub('', dl.strip().lstrip("0123456789.-*• ").strip()).strip()
            dl_lower = dl_clean.lower()
            # Keyword overlap check
            if ag_words and any(w in dl_lower for w in ag_words):
                # Prefer non-personal action lines as decisions
                if not any(kw in dl_lower for kw in _action_only_kws):
                    # Guard: don't copy the agenda text as its own decision (gives misleading 100%)
                    if dl_lower.strip() != ag_norm.strip():
                        dec_text = dl_clean
                        evidence_text = dl_clean
                        break

        # If the candidate line itself is a decision, use it (but not if it's identical to agenda)
        if dec_text == "No formal decision recorded." and is_decision_line and not is_personal_action:
            if content.lower().strip() != ag_norm.strip():
                dec_text = content
                evidence_text = content

        agenda_decisions_raw.append({
            "agenda": ag_title,
            "key_decision": dec_text,
            "evidence": evidence_text,
            "speaker": None,  # speaker attribution not available in extractive path
        })
        ord_idx += 1
        if ord_idx > _agenda_cap:
            break

    # Bug 2 fix: when the extractive parser finds no agenda candidates,
    # return an honest empty list — NEVER inject hardcoded sample-specific
    # agenda/decision pairs that are unrelated to the actual transcript.
    if not agenda_decisions_raw:
        agenda_decisions_raw = []

    agenda_key_decisions = validate_and_format_agenda_key_decisions(agenda_decisions_raw)

    # 5. Risks & Blockers Extraction
    risk_keywords = ["risk", "issue", "concern", "blocker", "challenge", "delay", "dependency"]
    risk_lines = [l for l in lines if any(kw in l.lower() for kw in risk_keywords)]

    if risk_lines:
        risks = "\n".join([f"1. {rl}" for rl in risk_lines[:5]])
    else:
        risks = "No critical risks or blockers identified during this session."

    return {
        "summary": summary,
        "mom": _polish_prose(mom),
        "action_items": action_items,
        "structured_action_items": structured_ai if 'structured_ai' in locals() else [],
        "decisions": decisions,
        "agenda_key_decisions": agenda_key_decisions,
        "risks": risks,
        "project_name": _PROJECT_NAME_UNKNOWN,
        "sentiment": "Neutral",
        "emotions": "Neutral: 100%",
        "questions_raised": "[]",
        "keyword_frequency": "{}",
    }


def _extract_action_items(llm, truncated: str, reference_dt=None) -> List[Dict[str, Any]]:
    """Dedicated 5th parallel worker for structured action item extraction.

    Uses the specialised ``action_items_extraction_prompt()`` which enforces:
    - Classification (action vs decision vs discussion)
    - Structured JSON with owner, deadline, evidence, priority, status
    - No hallucinations (evidence must come from transcript)

    The LLM extracts the natural-language deadline phrase only.  Deterministic
    date resolution is performed by ``resolve_action_item_dates`` (§9.16).

    Returns a validated, deduplicated list of action item dicts with
    ``due_date``, ``deadline_start``, ``deadline_end``, ``deadline_precision``
    populated.

    Falls back to empty list on any failure.
    """
    sys_prompt = _make_sys_prompt(action_items_extraction_prompt())
    prompt = ChatPromptTemplate.from_messages(
        [("system", sys_prompt), ("user", "{transcript}")]
    )
    try:
        result = (prompt | llm).invoke({"transcript": truncated})
        content = extract_json_from_text(result.content)
        data = json.loads(content)
        raw_items = data.get("action_items", [])
        if not isinstance(raw_items, list):
            logger.warning("[ActionItems] LLM returned non-list action_items field.")
            return []
        # Validate, normalise, and deduplicate via service
        validate_fn, dedup_fn, resolve_dates_fn = _get_action_item_service()
        validated = validate_fn(raw_items, truncated)
        deduped = dedup_fn(validated)
        # Deterministic date resolution (§9.16) — no LLM calendar arithmetic
        if reference_dt is not None:
            deduped = resolve_dates_fn(deduped, reference_dt)
        logger.info(f"[ActionItems] Extracted {len(deduped)} action item(s) from LLM.")
        return deduped
    except Exception as e:
        _strict_llm_failure("action items", e)
        logger.warning(f"[ActionItems] Dedicated extraction failed: {e}")
        return []


# ── Meeting metadata: sentiment, risks, topics (+ the two summary text blobs) ─────────────────────────────────
# The model is asked for ONE JSON object. A reply that is not valid JSON, or that breaks the field types below, is
# NOT parsed again as is (at temperature 0.1 / top_p 0.1 the same prompt returns the same bytes - measured: the
# identical "line 24 column 5 (char 660)" error on 3 of 4 live calls): the model is asked once or twice more WITH
# the concise validation error, and when it still cannot comply the section FAILS with the reasons. No default
# ("Neutral", "No major risks...") is ever stored in place of an answer the model did not give.
_METADATA_FORMAT_REPAIRS = 2          # extra calls after an unusable reply (so at most 3 calls per section)
_METADATA_STAGE = "metadata"
_METADATA_FIELDS = (
    "risks", "project_name", "sentiment", "emotions", "questions_raised", "keyword_frequency", "action_items",
    "decisions",
)
_SENTIMENT_LABELS = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
_MAX_TOPICS = 10
_MAX_QUESTIONS = 10
_ECHOED_REPLY_MAX_CHARS = 4000        # how much of the bad reply is shown back to the model


def _validate_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """The metadata object's fields, type-checked and normalised; ``LLMJsonError`` (kind ``schema``) listing EVERY
    problem when it is not what the schema asked for. Reasons name fields and types only, never reply content."""
    problems: List[str] = []
    missing = [k for k in _METADATA_FIELDS if data.get(k) is None]
    if missing:
        problems.append("missing field(s): " + ", ".join(f'"{k}"' for k in missing))

    def text(key: str, allow_list: bool = False, allow_empty_list: bool = False) -> Optional[str]:
        value = data.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            if value.strip():
                return value.strip()
            problems.append(f'"{key}" is an empty string')
            return None
        if allow_list and isinstance(value, list):
            items = [v.strip() for v in value if isinstance(v, str) and v.strip()]
            if len(items) != len(value):
                problems.append(f'"{key}" must contain only non-empty strings')
                return None
            if not items:
                if allow_empty_list:        # "[]" is the model's explicit "none" for action items / decisions
                    return ""
                problems.append(f'"{key}" is an empty array')
                return None
            return to_string_robust(items)
        problems.append(f'"{key}" must be a string' + (" or an array of strings" if allow_list else ""))
        return None

    out: Dict[str, Any] = {
        "risks": text("risks", allow_list=True),
        "project_name": text("project_name"),
        "action_items": text("action_items", allow_list=True, allow_empty_list=True),
        "decisions": text("decisions", allow_list=True, allow_empty_list=True),
    }

    sentiment = data.get("sentiment")
    if sentiment is not None:
        label = _SENTIMENT_LABELS.get(re.sub(r"[^a-z]", "", sentiment.lower())) if isinstance(sentiment, str) else None
        if label is None:
            problems.append('"sentiment" must be exactly one of "Positive", "Neutral", "Negative"')
        out["sentiment"] = label

    emotions = data.get("emotions")
    if emotions is not None:
        if isinstance(emotions, dict) and emotions and all(
            isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool) for k, v in emotions.items()
        ):
            out["emotions"] = ", ".join(f"{k}: {v:g}%" for k, v in emotions.items())
        else:
            out["emotions"] = text("emotions")

    questions = data.get("questions_raised")
    if questions is not None:
        if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
            out["questions_raised"] = [q.strip() for q in questions if q.strip()][:_MAX_QUESTIONS]
        else:
            problems.append('"questions_raised" must be an array of strings')

    topics = data.get("keyword_frequency")
    if topics is not None:
        cleaned: Dict[str, int] = {}
        ok = isinstance(topics, dict)
        for term, count in (topics.items() if ok else []):
            if isinstance(count, str) and count.strip().isdigit():
                count = int(count.strip())
            if not (isinstance(term, str) and term.strip()) or isinstance(count, bool) or not isinstance(count, (int, float)):
                ok = False
                break
            cleaned[term.strip()] = int(count)
        if ok:
            ranked = sorted(cleaned.items(), key=lambda kv: kv[1], reverse=True)[:_MAX_TOPICS]
            out["keyword_frequency"] = dict(ranked)
        else:
            problems.append('"keyword_frequency" must be an object mapping each term (string) to a whole-number count')

    if problems:
        raise LLMJsonError("; ".join(problems), "schema")
    return out


def _reply_text(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content if isinstance(content, str) else ""


def _extract_metadata_json(llm, truncated: str) -> dict:
    """Sentiment, risks, topics (keyword frequency), questions, emotions, project name and the two summary text blobs.

    Raises ``AirawatError`` when AIRAWAT cannot be reached, or does not return a valid, schema-conforming JSON object
    within ``1 + _METADATA_FORMAT_REPAIRS`` calls. Every call logs: stage, attempt, duration, finish reason,
    validation outcome and the failure reason - never the reply or the transcript.
    """
    sys_prompt = _make_sys_prompt(metadata_json_prompt())
    first = ChatPromptTemplate.from_messages([("system", sys_prompt), ("user", "{transcript}")])
    repair = ChatPromptTemplate.from_messages([
        ("system", sys_prompt), ("user", "{transcript}"), ("assistant", "{previous_reply}"), ("user", "{validation_error}"),
    ])
    section = "meeting metadata (sentiment, risks, topics)"
    attempts = 1 + _METADATA_FORMAT_REPAIRS
    reasons: List[str] = []
    previous_reply = ""
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            if attempt == 1:
                result = (first | llm).invoke({"transcript": truncated})
            else:
                result = (repair | llm).invoke({
                    "transcript": truncated,
                    "previous_reply": previous_reply[:_ECHOED_REPLY_MAX_CHARS],
                    "validation_error": (
                        f"Your previous reply could not be used: {reasons[-1]}. Reply again with ONE valid JSON "
                        "object that follows the schema exactly: every array element is a double-quoted string "
                        "(no bullets or bare words), no Markdown fences, no text outside the JSON."
                    ),
                })
        except Exception as exc:
            logger.warning(
                "[%s] attempt=%d/%d duration=%.2fs outcome=request_failed error=%s",
                _METADATA_STAGE, attempt, attempts, time.perf_counter() - started, type(exc).__name__,
            )
            _strict_llm_failure(section, exc)

        reply = _reply_text(result)
        finish = (getattr(result, "response_metadata", None) or {}).get("finish_reason")
        try:
            fields = _validate_metadata(parse_llm_json_object(reply))
        except LLMJsonError as bad:
            reasons.append(bad.reason)
            previous_reply = reply
            logger.warning(
                "[%s] attempt=%d/%d duration=%.2fs reply_chars=%d finish_reason=%s validation=failed kind=%s reason=%s",
                _METADATA_STAGE, attempt, attempts, time.perf_counter() - started, len(reply), finish, bad.kind,
                bad.reason,
            )
            if attempt < attempts:
                time.sleep(min(0.5 * 2 ** (attempt - 1), 2.0))
            continue
        logger.info(
            "[%s] attempt=%d/%d duration=%.2fs reply_chars=%d finish_reason=%s validation=ok",
            _METADATA_STAGE, attempt, attempts, time.perf_counter() - started, len(reply), finish,
        )
        return {
            "action_items": fields["action_items"],
            "decisions": fields["decisions"],
            "agenda_key_decisions": [],     # agenda/decisions come only from the dedicated, transcript-verified call
            "risks": fields["risks"],
            "project_name": fields["project_name"],
            "sentiment": fields["sentiment"],
            "emotions": fields["emotions"],
            "questions_raised": json.dumps(fields["questions_raised"]),
            "keyword_frequency": json.dumps(fields["keyword_frequency"]),
        }

    raise AirawatError(
        f"AI Provider failed while generating the {section}: no valid JSON after {attempts} attempt(s); "
        f"last problem: {reasons[-1]}",
        details={"stage": _METADATA_STAGE, "attempts": attempts, "reasons": reasons},
    )


_MOM_DATE_LINE = re.compile(r"^(\s*[-*]\s*\*\*Date\s*&\s*Time:\*\*).*$", re.IGNORECASE | re.MULTILINE)
_MOM_DURATION_LINE = re.compile(r"^(\s*[-*]\s*\*\*Duration:\*\*).*$", re.IGNORECASE | re.MULTILINE)


def _known_fact(value: Any) -> bool:
    v = str(value or "").strip().lower()
    return bool(v) and not v.startswith("unknown") and v not in ("not specified", "n/a", "none")


def enforce_mom_facts(mom: str, meeting_date: str, meeting_time: str, duration_str: str) -> str:
    """Put the meeting's date / time / duration into the MoM header exactly as the system knows them.

    They are facts (upload time, ffprobe duration), not something the model should infer from what was said, yet the
    model often rewrote the values the prompt gave it as "Not Specified" ("Duration: Not Specified" for a meeting
    whose length is known to the second). Unknown values are left as the model wrote them.
    """
    if not mom:
        return mom
    when = " | ".join(str(v).strip() for v in (meeting_date, meeting_time) if _known_fact(v))
    if when:
        mom = _MOM_DATE_LINE.sub(lambda m: f"{m.group(1)} {when}", mom, count=1)
    if _known_fact(duration_str):
        mom = _MOM_DURATION_LINE.sub(lambda m: f"{m.group(1)} {str(duration_str).strip()}", mom, count=1)
    return mom


def _section_wait_seconds() -> float:
    """Longest a report section may take: every attempt (1 + retries) at the per-attempt timeout, plus slack."""
    try:
        cfg = model_router.get_effective_config()
        per_attempt = int(cfg.get("timeout") or settings.AIRAWAT_TIMEOUT_SEC)
        retries = int(cfg.get("retry_count") or settings.AIRAWAT_RETRY_COUNT)
    except Exception:  # noqa: BLE001 - never let a timeout calculation fail a report
        per_attempt, retries = int(settings.AIRAWAT_TIMEOUT_SEC), int(settings.AIRAWAT_RETRY_COUNT)
    return float(per_attempt * (max(0, retries) + 1) + 30)


def process_meeting_report(
    transcript: str,
    meeting_date: str = "Unknown Date",
    meeting_time: str = "Unknown Time",
    duration_str: str = "Unknown Duration",
    timezone_str: str = "",
    section_cache: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Generates Summary, MoM, and Metadata CONCURRENTLY in parallel for maximum speed.

    ``section_cache`` (optional, owned by the caller's retry loop): results of the sections that already succeeded
    in an earlier attempt of this same report. They are reused, so a retry only asks AIRAWAT again for the
    sections that failed instead of regenerating all five.
    Uses fast-path check to bypass LLM retries if AIRAWAT API key is missing or unconfigured.

    ``timezone_str`` is an IANA timezone name (e.g. ``"Asia/Kolkata"``).  When
    provided, all relative-deadline resolution uses that timezone (§9.13).  If
    empty, the application's configured timezone (``settings.APP_TIMEZONE``) is
    used, falling back to UTC.
    """
    import concurrent.futures
    import hashlib as _hashlib

    # ── Per-call transcript identity (BUG-SAMEOUTPUT diagnostics) ─────────────
    # Every report is derived ONLY from the transcript passed here. Log a
    # fingerprint + head so two meetings producing the same report can be
    # instantly explained (same transcript in) vs investigated (different
    # transcript in, same report out = a real bug).
    _tx = (transcript or "").strip()
    _tx_norm = re.sub(r"\s+", " ", re.sub(r"^[A-Za-z0-9_\-. ]{1,30}:\s*", "", _tx, flags=re.MULTILINE)).strip()
    _tx_fp = _hashlib.sha256(_tx_norm.encode("utf-8")).hexdigest()[:16] if _tx_norm else "EMPTY"
    logger.info(
        "[LLMWorkflow] process_meeting_report | transcript_chars=%d meaningful=%d "
        "fingerprint=%s head=%r",
        len(_tx), len(_tx_norm), _tx_fp, _tx_norm[:100],
    )

    # Degenerate transcript → return an honest empty report. NEVER ask the LLM
    # to summarise nothing (it will invent a plausible-looking generic meeting,
    # and every content-less meeting then looks identical).
    if len(_tx_norm) < 40 or len({w.lower() for w in _tx_norm.split() if len(w) > 2}) < 8:
        logger.warning(
            "[LLMWorkflow] Transcript is empty/degenerate (fingerprint=%s). "
            "Returning empty report instead of a fabricated one.", _tx_fp,
        )
        return {
            "summary": "Not Available — transcript did not contain enough content to analyse.",
            "mom": "Not Available — transcript did not contain enough content to analyse.",
            "action_items": "No specific action items identified.",
            "structured_action_items": [],
            "decisions": "No formal decisions were recorded.",
            "agenda_key_decisions": [],
            "risks": "No critical risks or blockers identified during this session.",
            "project_name": _PROJECT_NAME_UNKNOWN,
            "sentiment": "Neutral",
            "emotions": "Neutral: 100%",
            "questions_raised": "[]",
            "keyword_frequency": "{}",
        }

    # ---- Build the timezone-aware reference datetime (§9.12) ----
    from backend.utils.date_resolver import get_reference_datetime
    _tz = timezone_str or getattr(settings, "APP_TIMEZONE", "") or "UTC"
    reference_dt = get_reference_datetime(
        meeting_date_str=meeting_date,
        timezone_str=_tz,
    )
    logger.info(
        "[LLMWorkflow] Reference datetime for deadline resolution: %s (tz=%s)",
        reference_dt.date(), _tz,
    )

    # AIRAWAT is the only AI provider: an unconfigured or failing AIRAWAT is an ERROR (raised below after the
    # configured retries), never a switch to the rule-based extractive engine whose heuristics can invent
    # decisions and action items the meeting never contained.
    logger.info(
        "Generating meeting reports concurrently (Summary + MoM + Metadata + Agenda-Decision in parallel)..."
    )
    summary = ""
    mom = ""
    metadata = {
        "action_items": "No action items extracted.",
        "decisions": "No decisions extracted.",
        "risks": "No major risks or blockers identified during this session.",
        "project_name": _PROJECT_NAME_UNKNOWN,
        "sentiment": "Neutral",
        "emotions": "Neutral: 100%",
        "questions_raised": "[]",
        "keyword_frequency": "{}",
        "agenda_key_decisions": [],
    }
    # Dedicated agenda-decision extraction result (preferred over metadata prompt output)
    dedicated_akd: List[Dict[str, Any]] = []
    # Dedicated structured action items (preferred over metadata text blob)
    dedicated_ai: List[Dict[str, Any]] = []
    # Wall time of each parallel report section (seconds), returned as ``_section_timings`` (THIS attempt: 0.0 for a
    # section reused from an earlier attempt) and ``_section_durations`` (how long each section really took to
    # generate, also when it was generated by an earlier attempt - what the stored telemetry must report).
    section_timings: Dict[str, float] = {}
    cache = section_cache if section_cache is not None else {}
    durations: Dict[str, float] = cache.setdefault("_durations", {})

    try:
        llm = get_llm()
        from backend.utils.helpers import truncate_transcript as _truncate_transcript
        truncated = _truncate_transcript(transcript, get_report_max_chars())

        mom_exec_guidance, mom_max_tokens = _mom_length_directives(truncated)
        mom_llm = model_router.get_llm(max_tokens=mom_max_tokens)

        def _timed(name, fn, *args, **kwargs):
            if name in cache:  # succeeded in an earlier attempt of this report: not asked again
                section_timings[name] = 0.0
                return cache[name]
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            finally:
                section_timings[name] = round(time.perf_counter() - t0, 3)
            cache[name] = result
            durations[name] = section_timings[name]
            return result

        # Execute all 5 tasks via ThreadPoolExecutor (max_workers=1 respects cloud provider rate limits)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # The summary worker gets the FULL transcript, not ``truncated``: a
            # meeting too long for one call is condensed part-by-part inside
            # generate_meeting_summary instead of having its middle discarded.
            fut_sum = executor.submit(_timed, "summary", generate_meeting_summary, transcript, meeting_ref=_tx_fp)
            fut_mom = executor.submit(
                _timed, "mom", _generate_section, mom_llm,
                mom_prompt(meeting_date, meeting_time, duration_str, mom_exec_guidance), truncated,
            )
            fut_meta = executor.submit(_timed, "decisions_risks_sentiment", _extract_metadata_json, llm, truncated)
            fut_akd = executor.submit(_timed, "agenda_decisions", _extract_agenda_key_decisions, llm, truncated)
            fut_ai = executor.submit(_timed, "action_items", _extract_action_items, llm, truncated, reference_dt)

            # Harvest each future individually so a timeout or error in one section does NOT discard others
            section_errors: Dict[str, Exception] = {}
            # AIRAWAT_TIMEOUT_SEC caps ONE attempt of an LLM call (the client retries a hung/failed attempt
            # AIRAWAT_RETRY_COUNT times). Waiting only one attempt's worth here would abandon a section whose retry -
            # the very thing that rescues a call the gateway silently dropped - was still running.
            section_timeout = _section_wait_seconds()
            try:
                summary_result = fut_sum.result(timeout=section_timeout)
                summary = summary_result.text
                logger.info(
                    "[LLMWorkflow] Summary: method=%s words=%d llm_calls=%d remaining_violations=%s",
                    summary_result.method, len(summary.split()), summary_result.llm_calls,
                    summary_result.violations,
                )
                if summary_result.method == "extractive":
                    # generate_meeting_summary degrades to a rule-based summary when its LLM calls fail.
                    section_errors["summary"] = RuntimeError("the LLM summary calls failed")
                    summary = ""
                    cache.pop("summary", None)
            except Exception as e:
                section_errors["summary"] = e
                logger.warning(f"[LLMWorkflow] Summary worker failed ({e}).")
            try:
                mom = enforce_mom_facts(fut_mom.result(timeout=section_timeout), meeting_date, meeting_time, duration_str)
            except Exception as e:
                section_errors["minutes of meeting"] = e
                logger.warning(f"[LLMWorkflow] MoM worker failed ({e}).")
            try:
                metadata = fut_meta.result(timeout=section_timeout)
            except Exception as e:
                section_errors["metadata"] = e
                logger.warning(f"[LLMWorkflow] Metadata worker failed ({e}); default metadata in local mode.")
            try:
                dedicated_akd = fut_akd.result(timeout=section_timeout)
            except Exception as e:
                section_errors["agenda and decisions"] = e
                logger.warning(f"[LLMWorkflow] Agenda-Decision worker failed ({e}).")
            try:
                dedicated_ai = fut_ai.result(timeout=section_timeout)
            except Exception as e:
                section_errors["action items"] = e
                logger.warning(f"[LLMWorkflow] Action Items worker failed ({e}).")

            if section_errors:
                raise AirawatError(
                    "AI Provider failed to generate: "
                    + "; ".join(f"{name} ({type(err).__name__}: {str(err)[:160]})" for name, err in section_errors.items())
                    + ". No other engine is used; the report is retried and the meeting is marked failed if the provider keeps failing.",
                    # which sections failed (the others were generated): the pipeline records real stage states from it
                    details={"failed_sections": sorted(section_errors)},
                )

    except AirawatError:
        raise
    except Exception as llm_err:
        raise AirawatError(f"AI Provider could not be used to generate the report: {type(llm_err).__name__}: {llm_err}") from llm_err

    # An empty / placeholder summary or MoM is an AI Provider failure (no other engine fills it in).
    from backend.utils.helpers import check_outputs_generated

    is_sum_ok, is_mom_ok = check_outputs_generated(summary, mom, "")
    if not (is_sum_ok and is_mom_ok):
        if not is_sum_ok:
            cache.pop("summary", None)
        if not is_mom_ok:
            cache.pop("mom", None)
        raise AirawatError(
            "AI Provider returned an empty or placeholder "
            + " and ".join(n for n, ok in (("summary", is_sum_ok), ("minutes of meeting", is_mom_ok)) if not ok)
            + ". No other engine is used."
        )

    # The LLM's answer is final, INCLUDING "no decisions" / "no action items": no rule-based engine fills gaps.

    # Prefer the dedicated agenda-decision extraction (richer: includes evidence + speaker).
    # Fall back to the combined metadata prompt output if the dedicated call returned nothing.
    final_akd = dedicated_akd if dedicated_akd else metadata.get("agenda_key_decisions", [])
    logger.info(
        f"[AgendaDecision] Final agenda-decision pairs: {len(final_akd)} "
        f"(source: {'dedicated LLM call' if dedicated_akd else 'metadata prompt fallback'})"
    )
    logger.info(
        f"[ActionItems] Final action items: {len(dedicated_ai)} "
        f"(source: {'dedicated LLM call' if dedicated_ai else 'none'})"
    )

    # The summary was already validated/repaired inside generate_meeting_summary
    # (or is the label-free extractive fallback); finalize_summary is the
    # deterministic write-time gate that guarantees — whichever path produced
    # it — no markdown/preamble, repeated sentences, timecodes or speaker labels.
    summary = finalize_summary(summary)
    # Grammar/clarity polish for the MoM (strip preambles, repeated lines), and
    # keep the MoM's own "Executive Summary" section free of speaker labels —
    # per-speaker attribution belongs only in its discussion sections.
    mom = scrub_mom_executive_summary(_polish_prose(mom))

    return {
        "summary": summary,
        "mom": mom,
        "action_items": metadata["action_items"],          # backward-compat text blob
        "structured_action_items": dedicated_ai,           # new structured array
        "decisions": metadata["decisions"],
        "agenda_key_decisions": final_akd,
        "risks": metadata["risks"],
        "project_name": metadata["project_name"],
        "sentiment": metadata["sentiment"],
        "emotions": metadata["emotions"],
        "questions_raised": metadata["questions_raised"],
        "keyword_frequency": metadata["keyword_frequency"],
        "_section_timings": dict(section_timings),
        "_section_durations": dict(durations),
    }


def process_meeting_analytics(transcript: str) -> Dict[str, Any]:
    """Extracts analytics in 1 fast LLM call using a truncated transcript."""
    logger.info("Extracting meeting analytics (fast truncated mode)...")
    truncated = transcript[: get_analytics_max_chars()]
    llm = get_llm()

    sys_prompt = _make_sys_prompt(analytics_prompt())
    prompt = ChatPromptTemplate.from_messages(
        [("system", sys_prompt), ("user", "{transcript}")]
    )

    try:
        result = (prompt | llm).invoke({"transcript": truncated})
        content = extract_json_from_text(result.content)
        data = json.loads(content)
        return {
            "sentiment": data.get("sentiment", "Neutral"),
            "emotions": data.get("emotions", "Neutral: 100%"),
            "questions_raised": json.dumps(data.get("questions_raised", [])),
            "keyword_frequency": json.dumps(data.get("keyword_frequency", {})),
        }
    except Exception as e:
        logger.error(f"Failed to parse analytics JSON: {e}")
        return {
            "sentiment": "Neutral",
            "emotions": "Neutral: 100%",
            "questions_raised": "[]",
            "keyword_frequency": "{}",
        }


def generate_meeting_comparison_topics(
    current_transcript: str, past_transcript: str
) -> Dict[str, Any]:
    """Extracts common topics and new topics between two meetings."""
    logger.info("Extracting high-level meeting comparison topics...")
    llm = get_llm()
    sys_prompt = _make_sys_prompt(comparison_topics_prompt())
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            (
                "user",
                "PAST MEETING TRANSCRIPT:\n{past_transcript}\n\nCURRENT MEETING TRANSCRIPT:\n{current_transcript}",
            ),
        ]
    )

    max_len = model_router.max_transcript_chars
    try:
        result = (prompt | llm).invoke(
            {
                "past_transcript": past_transcript[:max_len],
                "current_transcript": current_transcript[:max_len],
            }
        )
        content = extract_json_from_text(result.content)
        data = json.loads(content)
        return {
            "common_topics": data.get("common_topics", []),
            "new_topics_introduced": data.get("new_topics_introduced", []),
        }
    except Exception as e:
        logger.error(f"Failed to parse comparison topics JSON: {e}")
        return {
            # DI-06 FIX: Return empty lists instead of fabricating "General Project Updates".
            # Fabricated fallbacks are displayed as real data in the UI, which is dishonest.
            "common_topics": [],
            "new_topics_introduced": [],
        }


def generate_detailed_meeting_comparison(
    current_transcript: str, past_transcript: str
) -> Dict[str, Any]:
    """Generates a detailed comparison between two meetings."""
    logger.info("Generating detailed meeting comparison...")
    llm = get_llm()
    sys_prompt = _make_sys_prompt(detailed_comparison_prompt())
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            (
                "user",
                "PAST MEETING TRANSCRIPT:\n{past_transcript}\n\nCURRENT MEETING TRANSCRIPT:\n{current_transcript}",
            ),
        ]
    )

    max_len = model_router.max_transcript_chars
    try:
        result = (prompt | llm).invoke(
            {
                "past_transcript": past_transcript[:max_len],
                "current_transcript": current_transcript[:max_len],
            }
        )
        content = extract_json_from_text(result.content)
        data = json.loads(content)
        return {
            "past_meeting_topics": data.get("past_meeting_topics", []),
            "current_meeting_topics": data.get("current_meeting_topics", []),
            "completed_since_previous": data.get("completed_since_previous", []),
            "still_pending": data.get("still_pending", []),
            "new_discussions": data.get("new_discussions", []),
            "ai_summary": data.get(
                "ai_summary",
                "The meetings share common project contexts with new updates in the recent session.",
            ),
        }
    except Exception as e:
        logger.error(f"Failed to parse detailed comparison JSON: {e}")
        return {
            "past_meeting_topics": [],
            "current_meeting_topics": [],
            "completed_since_previous": [],
            "still_pending": [],
            "new_discussions": [],
            "ai_summary": "Failed to generate detailed AI summary.",
        }


def generate_similar_meeting_enrichment(
    current_transcript: str, matched_transcript: str
) -> Dict[str, Any]:
    """Analyzes why two meetings are similar."""
    logger.info("Generating similar meeting enrichment metrics...")
    llm = get_llm()
    sys_prompt = _make_sys_prompt(similar_meeting_enrichment_prompt())
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            (
                "user",
                "PREVIOUS MEETING:\n{matched_transcript}\n\nCURRENT MEETING:\n{current_transcript}",
            ),
        ]
    )

    # Use a quarter of the max transcript chars to keep both sides within context
    max_len = max(4000, model_router.max_transcript_chars // 4)
    try:
        result = (prompt | llm).invoke(
            {
                "matched_transcript": matched_transcript[:max_len],
                "current_transcript": current_transcript[:max_len],
            }
        )
        content = extract_json_from_text(result.content)
        data = json.loads(content)
        return {
            "matched_topics": data.get("matched_topics", []),
            "common_discussion_topics": data.get("common_discussion_topics", ""),
            "shared_action_items": data.get("shared_action_items", []),
            "similar_decisions": data.get("similar_decisions", []),
            "overlapping_keywords": data.get("overlapping_keywords", []),
        }
    except Exception as e:
        logger.error(f"Failed to parse similar meeting enrichment JSON: {e}")
        return {
            "matched_topics": [],
            "common_discussion_topics": "Related based on overarching business or project context.",
            "shared_action_items": [],
            "similar_decisions": [],
            "overlapping_keywords": [],
        }


def analyze_query_intent(query: str) -> str:
    """Classifies user query intent for adaptive retrieval.
    
    Delegates to the extended 12-category classifier in agent_router.
    Maintained here for backward compatibility with existing callers.
    """
    try:
        from backend.rag.agent_router import classify_intent
        return classify_intent(query)
    except Exception:
        # Fallback: original keyword matching for safety
        query_lower = query.lower()
        if any(word in query_lower for word in ["summary", "summarize", "overview", "overview of"]):
            return "summary"
        if any(word in query_lower for word in ["decision", "decided", "agreed", "conclusion"]):
            return "decisions"
        if any(word in query_lower for word in ["action item", "task", "todo", "follow-up", "followup", "assignee"]):
            return "action_items"
        if any(word in query_lower for word in ["risk", "blocker", "issue", "concern", "problem"]):
            return "risks"
        if any(word in query_lower for word in ["participant", "attendee", "who was", "who attended"]):
            return "participants"
        if any(word in query_lower for word in ["when", "time", "timestamp", "duration", "long"]):
            return "temporal"
        return "general"


_FOLLOWUP_PRONOUN_PATTERN = re.compile(
    r"\b(it|that|they|them|those|this|these|the same|the first|the second|the third|the last|who did that|when was that|what about that|why is that)\b",
    re.IGNORECASE,
)


def generate_chatbot_response(
    query: str,
    context_chunks: List[str],
    chat_history: List[Dict[str, str]],
    meeting_title: str,
    query_intent: str = "general",
) -> str:
    """Generates a response using retrieved context with strict grounding and dynamic format matching."""
    logger.info(
        f"Generating chatbot response for query: {query} (intent: {query_intent})"
    )
    llm = get_llm()

    formatted_context = "\n\n".join(
        [f"Context Chunk {i + 1}:\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )
    if not formatted_context.strip():
        formatted_context = "No relevant context found from the meeting transcript."

    history_str = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    system_prompt = _make_sys_prompt(chatbot_system_prompt(meeting_title))

    intent_guidance = get_intent_guidance()

    user_prompt = (
        "Here is the conversation history so far:\n{history}\n\n"
        "Here is the relevant meeting transcript and data context:\n{context}\n\n"
        f"Query Intent: {escape_template_braces(query_intent)}\n"
        f"Intent Guidance: {escape_template_braces(intent_guidance.get(query_intent, intent_guidance['general']))}\n\n"
        "User Question: {query}\n\n"
        "Remember: Answer ONLY from the provided meeting context. Do NOT force a fixed template — format your answer naturally according to the question type."
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", user_prompt)]
    )

    try:
        result = (prompt | llm).invoke(
            {
                "history": history_str,
                "context": formatted_context,
                "query": query,
            }
        )
        content = result.content.strip()
        # Clean reasoning tags
        content = re.sub(r"<think.*?>.*?</think>", "", content, flags=re.DOTALL).strip()
        content = re.sub(r"<thinking.*?>.*?</thinking>", "", content, flags=re.DOTALL).strip()
        return content
    except Exception as e:
        logger.error(f"Failed to generate chatbot response: {e}")
        return (
            "I encountered an error trying to process your request. Please try again."
        )


def rewrite_query_with_history(query: str, chat_history: List[Dict[str, str]]) -> str:
    """Rewrites the user's query with conversation history for better RAG retrieval."""
    if not chat_history:
        return query

    # Skip rewrite if query is already standalone and does not contain pronoun references
    query_stripped = query.strip()
    if not _FOLLOWUP_PRONOUN_PATTERN.search(query_stripped) and len(query_stripped.split()) >= 4:
        logger.debug(f"[QueryRewrite] Query is standalone, preserving as is: {query!r}")
        return query

    logger.info("Rewriting user query with conversation history...")
    llm = get_llm()

    history_str = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", query_rewrite_prompt()),
            ("user", "Conversation History:\n{history}\n\nFollow-up Question: {query}"),
        ]
    )

    try:
        result = (prompt | llm).invoke({"history": history_str, "query": query})
        rewritten = result.content.strip()
        rewritten = re.sub(r"<think.*?>.*?</think>", "", rewritten, flags=re.DOTALL).strip()
        rewritten = re.sub(r"<thinking.*?>.*?</thinking>", "", rewritten, flags=re.DOTALL).strip()
        logger.info(f"Original query: '{query}' -> Rewritten: '{rewritten}'")
        return rewritten if rewritten else query
    except Exception as e:
        logger.warning(f"Failed to rewrite query: {e}. Using original.")
        return query

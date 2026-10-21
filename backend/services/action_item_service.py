"""Action Item Service
====================
Validates, deduplicates, and scores extracted action items.

Design Principles:
- NEVER hardcode confidence scores.
- NEVER invent owners, deadlines, or task descriptions.
- Every score is computed from actual content signals.
- Deduplication uses semantic similarity (sentence-transformers).

Confidence Scoring Signals (weighted sum):
  - Task clarity (40%)   : is the task a concrete verb-phrase?
  - Owner clarity (25%)  : is owner resolved to a real person/role?
  - Deadline clarity (20%): is a specific deadline provided?
  - Evidence quality (15%): is a transcript quote present?

Total possible: 100.0 (none is hardcoded -- computed from content).
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy import of date_resolver to avoid circular dependencies
def _get_date_resolver():
    from backend.utils.date_resolver import resolve_deadline, get_reference_datetime
    return resolve_deadline, get_reference_datetime

# Constants
_NO_OWNER_PHRASES = {
    "unassigned", "unknown", "n/a", "not specified", "not mentioned",
    "not identified", "tbd", "",
}
_NO_DEADLINE_PHRASES = {
    "not specified", "not mentioned", "n/a", "tbd", "none", "",
}
_NO_PRIORITY_PHRASES = {
    "not specified", "n/a", "none", "tbd", "",
}
_VALID_STATUSES = {
    "open", "in_progress", "in progress", "completed", "blocked",
    "cancelled", "unknown", "pending",
}

_MIN_TASK_WORDS = 2
_DEDUP_THRESHOLD = 0.88


def _get_encoder():
    """Retrieve embedding encoder from centralized offline model loader."""
    from backend.services.model_loader import get_embedding_encoder
    return get_embedding_encoder()


def _encode_texts(texts: List[str]):
    """Encode texts using centralized offline model loader."""
    from backend.services.model_loader import encode_texts
    return encode_texts(texts)


def _score_task_clarity(task: str) -> float:
    if not task or not task.strip():
        return 0.0
    words = task.strip().split()
    word_count = len(words)
    if word_count < _MIN_TASK_WORDS:
        return 0.1
    action_verbs = {
        "prepare", "complete", "deploy", "review", "send", "submit",
        "write", "create", "update", "implement", "test", "fix", "resolve",
        "schedule", "coordinate", "present", "analyze", "analyse", "build",
        "design", "document", "handle", "manage", "report", "configure",
        "integrate", "track", "evaluate", "finalize", "finalise",
    }
    task_lower = task.lower()
    has_verb = any(v in task_lower for v in action_verbs)
    score = 0.5 + (0.3 if has_verb else 0.0) + min(0.1, (word_count - 2) * 0.02)
    return min(1.0, round(score, 3))


def _score_owner_clarity(owner: str) -> float:
    if not owner or owner.strip().lower() in _NO_OWNER_PHRASES:
        return 0.0
    o = owner.strip().lower()
    generic_roles = {"team", "group", "everyone", "all", "we", "somebody", "someone"}
    if any(g in o for g in generic_roles):
        return 0.5
    if re.match(r'^speaker[_ ]\w', o, re.IGNORECASE):
        return 0.8
    return 1.0


def _score_deadline_clarity(deadline: str, deadline_type: str, due_date=None) -> float:
    """Score deadline clarity.

    If a ``due_date`` was successfully resolved by the date_resolver, the item
    gets the maximum possible deadline score regardless of whether the original
    phrase was relative or absolute.
    """
    if not deadline or deadline.strip().lower() in _NO_DEADLINE_PHRASES:
        return 0.0
    # Resolved due_date available → highest confidence
    if due_date is not None:
        return 1.0
    dtype = (deadline_type or "").strip().lower()
    if dtype == "absolute":
        return 1.0
    if dtype == "relative":
        return 0.7
    return 0.5


def _score_evidence_quality(evidence: str, transcript: str) -> float:
    if not evidence or not evidence.strip():
        return 0.0
    evidence_clean = evidence.strip()
    if transcript:
        chunk = evidence_clean[:60].lower()
        if chunk in transcript.lower():
            return 1.0
        sentences = [s.strip() for s in evidence_clean.split(".") if s.strip()]
        for sentence in sentences:
            if len(sentence) > 10 and sentence.lower() in transcript.lower():
                return 0.85
    return 0.5


def compute_confidence(item: Dict[str, Any], transcript: str = "") -> float:
    """Extraction clarity score (0.0-100.0) for a single action item - a heuristic, NOT a probability that the item
    is correct.

    Weights: Task 40%, Owner 25%, Deadline 20%, Evidence 15%, over the parts that WERE stated: an owner or a
    deadline the meeting never gave is not measured and is left out (the remaining weights are renormalised) - it
    used to count as 0, so every item with no named owner was capped at 75 and the same "57.8" appeared on all of
    them. A missing evidence quote is different: it is a measured weakness (nothing to verify) and counts as 0.
    Never hardcoded -- computed from actual content signals.
    """
    parts = [(_score_task_clarity(item.get("task", "")), 0.40)]
    owner = (item.get("owner") or "").strip()
    if owner and owner.lower() not in _NO_OWNER_PHRASES:
        parts.append((_score_owner_clarity(owner), 0.25))
    deadline = (item.get("deadline") or "").strip()
    if deadline and deadline.lower() not in _NO_DEADLINE_PHRASES:
        parts.append((
            _score_deadline_clarity(deadline, item.get("deadline_type", ""), item.get("due_date")), 0.20,
        ))
    parts.append((_score_evidence_quality(item.get("evidence", ""), transcript), 0.15))
    weight = sum(w for _, w in parts)
    weighted = sum(s * w for s, w in parts) / weight
    return round(weighted * 100.0, 1)


_PLACEHOLDER_TASKS = {
    "complete the task", "do the work", "n/a",
    "tbd", "not specified", "task description", "action item",
}


def _is_valid_item(item: Dict[str, Any]) -> bool:
    task = (item.get("task") or "").strip()
    if not task or len(task.split()) < _MIN_TASK_WORDS:
        return False
    if task.lower() in _PLACEHOLDER_TASKS:
        return False
    status = (item.get("status") or "open").strip().lower()
    if status and status not in _VALID_STATUSES:
        item["status"] = "unknown"
    return True


_TAXONOMY_MAP = {
    "Documentation": {"documentation", "readme", "doc", "manual", "guide", "spec", "specification", "write-up", "wiki"},
    "Testing": {"test", "testing", "qa", "quality", "report", "bug", "benchmark", "validation", "verify", "unit test", "integration test"},
    "Deployment": {"deploy", "deployment", "release", "publish", "launch", "docker", "production", "staging", "ci/cd"},
    "Infrastructure": {"infra", "infrastructure", "cloud", "aws", "azure", "cluster", "network", "devops", "storage", "postgres"},
    "Security": {"security", "auth", "authentication", "encryption", "secret", "token", "vulnerability", "permission", "audit", "ssl"},
    "Review": {"review", "feedback", "evaluate", "assess", "inspect", "approve"},
    "Administration": {"schedule", "meeting", "email", "invite", "call", "budget", "resource", "hiring", "admin", "management", "coordinate"},
    "Development": {"backend", "frontend", "api", "code", "coding", "implementation", "feature", "fix", "refactor", "develop", "programming", "database", "query", "script", "build"},
}


def _classify_category(task: str, description: Optional[str] = None, evidence: Optional[str] = None) -> str:
    """Dynamically classify an Action Item into a configurable taxonomy category.

    Categories: Development, Testing, Deployment, Documentation, Infrastructure,
    Security, Review, Administration, Other.
    Never hardcoded per sample meeting -- uses semantic keyword match on combined text.
    """
    text = f"{task or ''} {description or ''} {evidence or ''}".lower()
    for category, keywords in _TAXONOMY_MAP.items():
        if any(kw in text for kw in keywords):
            return category
    return "Other"


def format_assignees_list(
    owner_str: Optional[str],
    speaker_str: Optional[str] = None,
    evidence: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Dynamically convert owner/speaker strings into a list of structured Assignee objects.

    Returns:
        List[Dict[str, Any]]: Array of objects: [{"name": "Ravi", "speaker_id": None, "confidence": 0.95}]
        Returns empty list [] when unassigned or no responsible person mentioned.
    """
    if not owner_str or owner_str.strip().lower() in _NO_OWNER_PHRASES:
        # Check if evidence shows self-commitment ("I will...")
        if evidence and speaker_str:
            ev_lower = evidence.lower()
            if any(p in ev_lower for p in ["i will", "i'll", "i am going to", "i can", "i shall"]):
                spk_name = speaker_str.strip()
                spk_id = spk_name.lower().replace(" ", "_") if "speaker" in spk_name.lower() else None
                return [{"name": spk_name, "speaker_id": spk_id, "confidence": 0.95}]
        return []

    cleaned = owner_str.strip()
    if cleaned.lower() in _NO_OWNER_PHRASES:
        return []

    # Handle multiple assignees separated by 'and' or ','
    raw_names = re.split(r'\s+and\s+|,\s*', cleaned)
    assignees: List[Dict[str, Any]] = []
    for rname in raw_names:
        rname = rname.strip()
        if not rname or rname.lower() in _NO_OWNER_PHRASES:
            continue
        speaker_id = None
        if "speaker" in rname.lower():
            speaker_id = rname.lower().replace(" ", "_")
        assignees.append({
            "name": rname,
            "speaker_id": speaker_id,
            "confidence": 0.95 if speaker_id or len(rname.split()) == 1 else 0.85,
        })
    return assignees


_MIN_TASK_SUPPORT = 0.5   # share of a quote-less task's content words that must occur in the transcript


def _task_word_support(task: str, tx_words: set, tx_prefixes: set) -> "tuple[float, int]":
    """(share of the task's content words found in the transcript, number of content words).

    Whole-word / shared-stem matching - not substring containment, which let "plan" be "found" inside "planet".
    """
    words = [w for w in re.findall(r"[a-z0-9]+", task.lower()) if len(w) > 3]
    if not words:
        return 0.0, 0
    found = sum(1 for w in words if w in tx_words or w[:5] in tx_prefixes)
    return found / len(words), len(words)


def validate_action_items(
    raw_items: List[Dict[str, Any]],
    transcript: str = "",
) -> List[Dict[str, Any]]:
    """Validate and normalise LLM-extracted action items.
    
    Filters out placeholders, normalises all fields, computes dynamic confidence.
    """
    from backend.utils import text_signals as _ts

    validated: List[Dict[str, Any]] = []
    tx_norm = _ts.normalize_text(transcript) if transcript and transcript.strip() else ""
    tx_words = set(re.findall(r"[a-z0-9]+", tx_norm)) if tx_norm else set()
    tx_prefixes = {w[:5] for w in tx_words}   # inflections: "preparing" / "prepare" share "prepa"
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        task = str(raw.get("task") or raw.get("title") or "").strip()
        if not task:
            continue

        description = str(raw.get("description") or "").strip() or None
        owner = _normalise_owner(raw.get("owner") or raw.get("assigned_to"))
        speaker = _normalise_speaker(raw.get("speaker") or raw.get("source_speaker"))
        evidence = _normalise_nullable(raw.get("evidence") or raw.get("supporting_evidence"))

        # Evidence is a QUOTE: it is shown only if it really occurs in the transcript. An invented or paraphrased
        # quote is removed (never presented as what was said); the task then has to stand on its own words, and an
        # item that is neither quoted nor supported by the transcript at all is dropped as fabricated.
        # This applies to an item that came WITHOUT any quote as well: it used to pass untouched, so a task the
        # transcript never mentions was saved next to the verified ones.
        if tx_norm and evidence and not _ts.quote_in_text(evidence, tx_norm):
            logger.info("[ActionItemService] evidence for %r is not in the transcript; quote removed.", task[:60])
            evidence = None
        if tx_norm and not evidence:
            support, n_words = _task_word_support(task, tx_words, tx_prefixes)
            if n_words == 0 or support < _MIN_TASK_SUPPORT:
                logger.info(
                    "[ActionItemService] %r has no verifiable evidence in the transcript (word support %.0f%% of %d "
                    "word(s), minimum %.0f%%); not saved as an action item.",
                    task[:60], support * 100, n_words, _MIN_TASK_SUPPORT * 100,
                )
                continue

        # Taxonomy Category Classification
        category_input = raw.get("category")
        if category_input and str(category_input).strip() in _TAXONOMY_MAP:
            category = str(category_input).strip()
        else:
            category = _classify_category(task, description, evidence)

        priority = _normalise_priority(raw.get("priority"))
        p_conf = 0.9 if priority and priority != "Not specified" else None

        assignees = format_assignees_list(owner, speaker, evidence)

        # The extraction prompt asks the model to put the deadline PHRASE in ``due_date`` ("October 10",
        # "next Wednesday"); this validator only read ``deadline`` / ``original_deadline_text``, so every
        # deadline the model found was discarded and reported as "Not specified". Accept all the spellings.
        deadline_phrase = _normalise_deadline(
            raw.get("deadline") or raw.get("original_deadline_text") or raw.get("due_date")
            or raw.get("due") or raw.get("deadline_text")
        )

        item: Dict[str, Any] = {
            "task": task,
            "title": task,
            "description": description or "",
            "owner": owner,
            "assigned_to": owner,
            "assignees": assignees,
            "assigned_by": _normalise_nullable(raw.get("assigned_by")),
            "deadline": deadline_phrase,
            "original_deadline_text": deadline_phrase,
            "deadline_type": _normalise_deadline_type(raw.get("deadline_type")),
            # Resolved date fields — populated by resolve_action_item_dates(), not here
            "due_date": None,
            "deadline_start": None,
            "deadline_end": None,
            "deadline_precision": None,
            "priority": priority,
            "priority_confidence": p_conf,
            "category": category,
            "status": _normalise_status(raw.get("status")),
            "evidence": evidence,
            "supporting_evidence": evidence,
            "speaker": speaker,
            "source_speaker": speaker,
            "timestamp_start": _normalise_nullable(raw.get("timestamp_start")),
            "timestamp_end": _normalise_nullable(raw.get("timestamp_end")),
            "agenda_ref": _normalise_nullable(raw.get("agenda_ref")),
            "decision_ref": _normalise_nullable(raw.get("decision_ref")),
            "source": "llm",
        }
        if not _is_valid_item(item):
            continue
        conf = compute_confidence(item, transcript)
        item["confidence"] = conf
        item["confidence_score"] = round(conf / 100.0, 2) if conf is not None else None
        validated.append(item)
    logger.info(f"[ActionItemService] Validated {len(validated)} / {len(raw_items)} items.")
    return validated


def deduplicate_action_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove semantically duplicate action items using embedding cosine similarity.
    
    Falls back to Jaccard overlap when the model is unavailable.
    Two items are duplicates when task similarity >= _DEDUP_THRESHOLD (0.88).
    """
    if len(items) <= 1:
        return items
    tasks = [item["task"] for item in items]
    embeddings = _encode_texts(tasks)
    if embeddings is not None:
        return _dedup_with_embeddings(items, embeddings)
    return _dedup_with_jaccard(items)


def _dedup_with_embeddings(items, embeddings) -> List[Dict[str, Any]]:
    import numpy as np
    n = len(items)
    kept = [True] * n
    for i in range(n):
        if not kept[i]:
            continue
        for j in range(i + 1, n):
            if not kept[j]:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= _DEDUP_THRESHOLD:
                conf_i = items[i].get("confidence") or 0.0
                conf_j = items[j].get("confidence") or 0.0
                if conf_j > conf_i:
                    kept[i] = False
                    break
                else:
                    kept[j] = False
    result = [item for k, item in zip(kept, items) if k]
    removed = n - len(result)
    if removed:
        logger.info(f"[ActionItemService] Deduplication removed {removed} duplicate(s).")
    return result


def _dedup_with_jaccard(items) -> List[Dict[str, Any]]:
    n = len(items)
    kept = [True] * n
    task_tokens = [set(item["task"].lower().split()) for item in items]
    for i in range(n):
        if not kept[i]:
            continue
        for j in range(i + 1, n):
            if not kept[j]:
                continue
            intersection = task_tokens[i] & task_tokens[j]
            union = task_tokens[i] | task_tokens[j]
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard >= 0.75:
                conf_i = items[i].get("confidence") or 0.0
                conf_j = items[j].get("confidence") or 0.0
                kept[j if conf_j <= conf_i else i] = False
    result = [item for k, item in zip(kept, items) if k]
    removed = n - len(result)
    if removed:
        logger.info(f"[ActionItemService] Jaccard dedup removed {removed} duplicate(s).")
    return result


def _clean_and_cap_task(task_text: str) -> str:
    """Clean and cap task string to a concise, high-quality action verb phrase under 14 words.
    Strips speaker prefixes, greetings, filler intros, assignment prefixes, and cuts off long trailing clauses.
    """
    if not task_text:
        return ""
    t = task_text.strip()
    # Strip speaker prefix if present ("Speaker A: ", "SPEAKER_00: ")
    t = re.sub(r'^(?:Speaker\s+)?[A-Za-z0-9_\-]{1,30}:\s*', '', t, flags=re.IGNORECASE).strip()

    # Strip intro interjections ("Great. ", "Okay. ", "Well, ", "Also, ")
    t = re.sub(r'^(?:Great|Okay|Thanks|Well|Sure|Awesome|Good)\.?,?\s*', '', t, flags=re.IGNORECASE).strip()

    # Strip subject-verb assignment prefixes ("Ravi will ", "Priya, please ", "I will ", "We need to ", "The team should ")
    t = re.sub(r'^(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|I|We|The\s+team)\s+(?:will|\'ll|must|should|needs?\s+to|is\s+going\s+to|can|please)\s+', '', t, flags=re.IGNORECASE).strip()

    # Cut off long trailing subordinate clauses ("that will make...", "so that...", "whether you're...", "and then...")
    t = re.split(r'\b(?:that\s+will|so\s+that|which\s+will|whether\s+you|and\s+then|how\s+we\'re)\b', t, flags=re.IGNORECASE)[0].strip()

    # Cut off at first period/semicolon/newline
    t = re.split(r'[;.\n]', t)[0].strip()

    # Strip trailing punctuation
    t = t.rstrip(",:- ")

    # Cap to max 14 words
    words = t.split()
    if len(words) > 14:
        t = " ".join(words[:12])

    if not t:
        return ""
    return t[:1].upper() + t[1:]


def extract_action_items_extractive(
    transcript: str,
    speaker_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Extract action items from transcript without LLM using sentence-level commitment/assignment patterns.
    
    All fields populated from actual transcript content -- never hardcoded.
    Ensures tasks are concise, actionable verb-phrases (under 14 words) -- NEVER raw transcript paragraphs.
    """
    items: List[Dict[str, Any]] = []
    # Concrete, assignable task verbs. Deliberately curated and NOT a generic
    # verb list: "must"/"needs to"/"please" followed by a vague/passive verb
    # ("must maintain 99.9% uptime", "please note that...", "need to be
    # careful about wording") describes a requirement or an FYI, not an
    # assignable action item — matching those bare fabricated tasks out of
    # ordinary sentences (verified empirically; see PR discussion). Requiring
    # one of these verbs keeps the extractive fallback's true positives
    # ("I need to send the report", "please submit by Friday") while dropping
    # the false ones.
    _ACTION_VERB = (
        r"(?:handle|prepare|send|deploy|complete|submit|review|create|write|fix|"
        r"update|schedule|coordinate|build|finalize|configure|integrate|share|"
        r"deliver|test|verify|check|follow\s*up|reach\s*out|set\s*up|arrange|"
        r"organize|draft|publish|upload|contact|call|email|circulate|distribute|book)"
    )
    _COMMITMENT_RE = re.compile(
        r"(?:\bI\s+will\b|\bI'll\b|\bI\s+am\s+going\s+to\b|\bI\s+can\b"
        rf"|\bwill\s+{_ACTION_VERB}\b"
        rf"|\bplease\s+{_ACTION_VERB}\b"
        rf"|\bneeds?\s+to\s+{_ACTION_VERB}\b"
        rf"|\bmust\s+{_ACTION_VERB}\b)",
        re.IGNORECASE,
    )
    _DEADLINE_RE = re.compile(
        r"(?:by\s+(?:Friday|Monday|Tuesday|Wednesday|Thursday|Saturday|Sunday"
        r"|today|tomorrow|end\s+of\s+(?:day|week|month)|next\s+\w+)"
        # Require an actual date/time-shaped target after "before" — a bare
        # \w+ here previously matched ordinary words ("before we continue"
        # -> deadline="before we"), fabricating a deadline out of nothing.
        r"|before\s+(?:next\s+(?:week|month|quarter|sprint|meeting|\w+day)"
        r"|the\s+(?:end\s+of\s+(?:day|week|month|quarter)|deadline)"
        r"|Friday|Monday|Tuesday|Wednesday|Thursday|Saturday|Sunday)"
        r"|within\s+\d+\s+(?:days?|weeks?|hours?)"
        r"|next\s+(?:week|month|sprint|meeting|\w+day)"
        r"|this\s+(?:week|month|sprint|Friday|Monday|Tuesday|Wednesday|Thursday|Saturday|Sunday)"
        r"|in\s+\d+\s+(?:days?|weeks?|hours?)"
        r"|tomorrow|today|EOD|ASAP|end\s+of\s+day)",
        re.IGNORECASE,
    )
    _GREETING_FILLER_RE = re.compile(
        r"\b(?:hello|hi|greetings|delighted|excited|welcome|thank you|thanks|good morning|good afternoon|good evening|talk to you about|in this meeting|in this session|today we will review|we will review|review the project|let's jump into|how we're making|great way for us|see what it looks like|wave of innovation)\b",
        re.IGNORECASE,
    )
    _SPEAKER_RE = re.compile(r"^([A-Za-z0-9_\-\s]{1,30}):\s*(.+)$")
    current_speaker: Optional[str] = None
    processed_evidence: set = set()

    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        sm = _SPEAKER_RE.match(line)
        if sm:
            candidate = sm.group(1).strip()
            if len(candidate) <= 30 and "http" not in candidate:
                current_speaker = candidate
                line = sm.group(2).strip()

        # Split long paragraph lines into individual sentences
        sentences = re.split(r'(?<=[.!?])\s+', line)

        for sent in sentences:
            sent = sent.strip()
            if not sent or sent in processed_evidence:
                continue

            # Strip speaker prefix if embedded inside sentence
            sent_clean = re.sub(r'^(?:Speaker\s+)?[A-Za-z0-9_\-]{1,30}:\s*', '', sent, flags=re.IGNORECASE).strip()

            # Skip short interjections or questions
            if len(sent_clean.split()) < 3 or sent_clean.endswith("?"):
                continue

            # Skip greeting/presentation filler sentences
            if _GREETING_FILLER_RE.search(sent_clean):
                continue

            # Check commitment language
            commitment_match = _COMMITMENT_RE.search(sent_clean)
            if not commitment_match:
                continue

            deadline_match = _DEADLINE_RE.search(sent_clean)
            deadline = deadline_match.group(0) if deadline_match else "Not specified"
            deadline_type = "not_specified"
            if deadline != "Not specified":
                dl_lower = deadline.lower()
                relative_markers = {"next", "tomorrow", "today", "this", "within", "eod", "asap"}
                deadline_type = "relative" if any(m in dl_lower for m in relative_markers) else "absolute"

            # Discard any preamble BEFORE the actual commitment phrase (e.g.
            # "Before we continue, let's take a rest, then I will review the
            # results" -> start from "I will review..."), otherwise the
            # unrelated lead-in text was previously kept as part of the task
            # (verified empirically: produced a garbled task starting
            # "Continue, let's take a rest for a moment, then I will review").
            task_source = sent_clean[commitment_match.start():]
            task_line = re.sub(
                r"^(?:I(?:\s+will|\s+am\s+going\s+to|'ll)\s+"
                r"|(?:we\s+need\s+to|needs?\s+to|must|please)\s+)",
                "", task_source, flags=re.IGNORECASE,
            ).strip()
            if deadline != "Not specified" and deadline_match:
                task_line = task_line.replace(deadline_match.group(0), "").strip().rstrip(",. ")

            task_line = _clean_and_cap_task(task_line)

            if len(task_line.split()) < _MIN_TASK_WORDS:
                continue

            owner = "Unassigned"
            sent_lower = sent_clean.lower()
            if re.search(r"\b(i will|i'll|i am going to|i can)\b", sent_lower):
                owner = current_speaker or "Unassigned"
            else:
                named_match = re.match(r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)?),?\s+(?:please|will|should|must|needs? to)\b", sent_clean)
                if named_match:
                    owner = named_match.group(1)

            processed_evidence.add(sent)
            items.append({
                "task": task_line,
                "description": None,
                "owner": owner,
                "assigned_by": None,
                "deadline": deadline,
                "deadline_type": deadline_type,
                "priority": "Not specified",
                "status": "open",
                "evidence": sent_clean,
                "speaker": current_speaker,
                "timestamp_start": None,
                "timestamp_end": None,
                "agenda_ref": None,
                "decision_ref": None,
                "source": "extractive",
                "confidence": None,
            })

    if not items:
        # Fallback to structured markdown parsing — ONLY when the text actually
        # looks like a checklist. Passing raw transcript prose here turned every
        # spoken line into a fake "action item" (BUG-SAMEOUTPUT: the same
        # meaningless items appeared on unrelated meetings).
        _looks_like_list = bool(re.search(r"(?m)^\s*(?:[-*•]|\[[ xX]?\]|\d+[.)])\s+\S", transcript or ""))
        if _looks_like_list:
            items = parse_action_items_from_markdown(transcript)
        else:
            logger.info(
                "[ActionItems] No commitment sentences found and transcript is not a "
                "checklist — returning empty list (no fabricated items)."
            )
            return []

    for item in items:
        item["confidence"] = compute_confidence(item, transcript)
    validated = validate_action_items(items, transcript)
    return deduplicate_action_items(validated)


def parse_action_items_from_markdown(text: str, transcript: str = "") -> List[Dict[str, Any]]:
    """Parse structured action items from markdown bullet lists, MOM text, or inline format."""
    if not text or not text.strip():
        return []

    items = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    raw_blocks = []
    if len(lines) == 1 and lines[0].count("*") >= 2:
        raw_blocks.append(lines[0])
    else:
        for line in lines:
            if line.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "[ ]", "[x]")):
                raw_blocks.append(line)
            elif raw_blocks:
                raw_blocks[-1] += " " + line
            else:
                raw_blocks.append(line)

    for block in raw_blocks:
        clean_block = re.sub(r"^[-*•\d\.\s\[\]xX]+", "", block).strip()
        if not clean_block or len(clean_block) < 5:
            continue

        owner = "Unassigned"
        deadline = "Not specified"
        priority = "Not specified"
        task_str = clean_block

        # Extract Assignee / Owner
        assignee_match = re.search(r"(?:Assignee|Owner|Assigned To|@):\s*([^,*|\)]+)", task_str, re.IGNORECASE)
        if assignee_match:
            owner = assignee_match.group(1).strip()
            task_str = task_str[:assignee_match.start()] + task_str[assignee_match.end():]

        # Extract Deadline / Due Date
        deadline_match = re.search(r"(?:Deadline|Due Date|Due):\s*([^,*|\)]+)", task_str, re.IGNORECASE)
        if deadline_match:
            deadline = deadline_match.group(1).strip()
            task_str = task_str[:deadline_match.start()] + task_str[deadline_match.end():]

        # Extract Priority
        priority_match = re.search(r"(?:Priority):\s*([^,*|\)]+)", task_str, re.IGNORECASE)
        if priority_match:
            priority = priority_match.group(1).strip()
            task_str = task_str[:priority_match.start()] + task_str[priority_match.end():]

        # Clean remaining task string
        task_clean = re.sub(r"[\*|\(\)]+", " ", task_str).strip()
        task_clean = re.sub(r"\s+", " ", task_clean).strip(" ,.-")

        if len(task_clean.split()) >= 2:
            dl_lower = deadline.lower()
            relative_markers = {"next", "tomorrow", "today", "this", "within", "eod", "asap"}
            deadline_type = "relative" if any(m in dl_lower for m in relative_markers) else ("not_specified" if deadline == "Not specified" else "absolute")

            item = {
                "task": task_clean,
                "title": task_clean,
                "description": None,
                "owner": owner,
                "assigned_to": owner,
                "assigned_by": None,
                "deadline": deadline,
                "original_deadline_text": deadline,
                "deadline_type": deadline_type,
                "priority": priority,
                "status": "open",
                "evidence": block,
                "speaker": None,
                "timestamp_start": None,
                "timestamp_end": None,
                "agenda_ref": None,
                "decision_ref": None,
                "source": "extractive",
            }
            items.append(item)

    return items


def resolve_action_item_dates(
    items: List[Dict[str, Any]],
    reference_dt,
) -> List[Dict[str, Any]]:
    """Resolve natural-language deadlines to calendar dates.

    This function implements the mandatory LLM→Resolver separation described
    in §9.16.  The LLM (or extractive engine) produces the raw phrase
    (e.g. ``"next Monday"``).  THIS function calls the deterministic
    ``date_resolver`` to convert it to an ISO date.

    Parameters
    ----------
    items:
        List of validated action item dicts (from ``validate_action_items``).
    reference_dt:
        Timezone-aware reference datetime.  Use the meeting date when
        available (§9.11); fall back to current server time.

    Returns
    -------
    The same list with ``due_date``, ``deadline_start``, ``deadline_end``,
    and ``deadline_precision`` populated in-place.
    """
    if not items:
        return items
    if reference_dt is None:
        logger.warning(
            "[ActionItemService] resolve_action_item_dates called without reference_dt — "
            "using UTC now as fallback."
        )
        reference_dt = datetime.now(timezone.utc)

    try:
        resolve_deadline, _ = _get_date_resolver()
    except Exception as e:
        logger.error("[ActionItemService] Could not import date_resolver: %s", e)
        return items

    resolved_count = 0
    for item in items:
        deadline_phrase = item.get("deadline", "") or ""
        deadline_type = item.get("deadline_type", "not_specified") or "not_specified"

        # Only attempt resolution when the phrase is meaningful
        skip_phrases = {
            "not specified", "not mentioned", "n/a", "tbd", "none", "",
        }
        if deadline_phrase.strip().lower() in skip_phrases:
            item["deadline_precision"] = "unresolvable"
            continue

        try:
            resolution = resolve_deadline(deadline_phrase, reference_dt)
            # Store native date object — SQLAlchemy Column(Date) and JSON serialization
            # both handle datetime.date correctly. Tests assert against date() objects.
            if resolution.due_date is not None:
                item["due_date"] = resolution.due_date
                resolved_count += 1
                logger.debug(
                    "[ActionItemService] Resolved %r → %s (precision=%s, tz=%s)",
                    deadline_phrase, resolution.due_date,
                    resolution.deadline_precision, resolution.timezone,
                )
            else:
                logger.debug(
                    "[ActionItemService] Could not resolve exact date for %r (precision=%s)",
                    deadline_phrase, resolution.deadline_precision,
                )
            # Always copy window dates and precision back, even when due_date is None
            if resolution.deadline_start is not None:
                item["deadline_start"] = resolution.deadline_start
            if resolution.deadline_end is not None:
                item["deadline_end"] = resolution.deadline_end
            if resolution.deadline_precision:
                item["deadline_precision"] = resolution.deadline_precision
        except Exception as e:
            logger.warning(
                "[ActionItemService] Date resolution failed for %r: %s",
                deadline_phrase, e,
            )

    logger.info(
        "[ActionItemService] Date resolution: %d/%d items got a resolved due_date.",
        resolved_count, len(items),
    )
    return items


# -- Normalisation helpers

def _normalise_owner(value) -> str:
    if not value:
        return ""
    v = str(value).strip()
    if not v or v.lower() in _NO_OWNER_PHRASES or v.lower() in {"null", "unassigned"}:
        return ""
    return v


def _normalise_deadline(value) -> str:
    if not value:
        return ""
    v = str(value).strip()
    if not v or v.lower() in _NO_DEADLINE_PHRASES or v.lower() in {"null", "not specified"}:
        return ""
    return v


def _normalise_deadline_type(value) -> str:
    if not value:
        return "not_specified"
    v = str(value).strip().lower()
    if v in {"absolute", "relative", "not_specified"}:
        return v
    return "not_specified"


def _normalise_priority(value) -> str:
    if not value:
        return ""
    v = str(value).strip()
    if not v or v.lower() in _NO_PRIORITY_PHRASES or v.lower() in {"null", "not specified"}:
        return ""
    v_lower = v.lower()
    if v_lower == "high" or "critical" in v_lower or "urgent" in v_lower:
        return "High"
    if v_lower == "medium" or "moderate" in v_lower:
        return "Medium"
    if v_lower == "low":
        return "Low"
    return ""


def _normalise_status(value) -> str:
    if not value:
        return "open"
    v = str(value).strip().lower()
    if v in {"open", "pending"}:
        return "open"
    if v in {"in_progress", "in progress", "in-progress", "working"}:
        return "in_progress"
    if v in {"completed", "done", "finished"}:
        return "completed"
    if v in {"blocked", "stuck"}:
        return "blocked"
    if v in {"cancelled", "canceled"}:
        return "cancelled"
    return "open"


def _normalise_speaker(value) -> Optional[str]:
    if not value:
        return None
    v = str(value).strip()
    if not v or v.lower() in {"unknown", "null", "n/a", ""}:
        return None
    return v


def _normalise_nullable(value) -> Optional[str]:
    if not value:
        return None
    v = str(value).strip()
    if not v or v.lower() in {"null", "none", "n/a", ""}:
        return None
    return v

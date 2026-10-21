"""
Dynamic STT Correction & Grammar Normalization Service
────────────────────────────────────────────────────────
Production-Grade Architecture:
1. Stage 1: Immutable Raw STT Preservation
2. Stage 2: Dynamic Terminology Discovery (Meeting Metadata, Agenda, Documents, DB Records, Approved Dictionary)
3. Stage 3: Multi-Signal Phonetic & Text Similarity Scorer (Soundex, Metaphone, Levenshtein, Token Similarity)
4. Stage 4: Contextual Validation & Multi-Signal Scoring Engine (Candidate Score Formula with Centralized Weights)
5. Stage 5: AIRAWAT LLM Context-Aware Correction Pass (Meaning Preservation, Zero Hallucination)
6. Stage 6 & 7: Grammar & Punctuation Restoration (Capitalization, Sentence Boundaries, Duplicate Word Cleanup)
7. Stage 8 & 9: Speaker-Safe & Timestamp-Safe Alignment (Speaker IDs and Segment Timestamps Preserved 100%)
8. Stage 10 & 11: Confidence Validation Gating (Threshold >= 0.70) & Audit Trail Generation
9. Stage 16: Safe Error Handling (If Correction Fails, Pipeline Continues with Raw STT)
"""

import difflib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.config.config import settings
from backend.prompts.meeting_prompts import transcript_correction_prompt

logger = logging.getLogger(__name__)

# Protected technical terms & proper acronyms that must NEVER be corrupted (Section 15)
PROTECTED_TERMS: Set[str] = {
    "FastAPI",
    "PostgreSQL",
    "Docker",
    "Python",
    "NVIDIA",
    "NeMo",
    "Whisper",
    "WhisperX",
    "BharatVC",
    "ChromaDB",
    "GovIntranet",
    "JSON",
    "REST",
    "API",
    "STT",
    "ASR",
    "VAD",
    "WER",
    "CER",
    "MOM",
    "HTML",
    "CSS",
    "SQL",
}

# Common English words that must NEVER be force-cased by normalize_sentence_grammar's
# vocabulary-casing pass, even when they collide (case-insensitively) with a
# PROTECTED_TERMS acronym or a Title-Case word captured from meeting metadata.
# Root cause this guards against: a case-insensitive regex substitution cannot
# tell "REST" the API acronym apart from "rest" the ordinary word (e.g. "take a
# rest"), or "MOM" the acronym from "mom" the family word — so without this
# filter, an unrelated sentence gets its casing corrupted into looking like a
# technical acronym. Also used to keep free-text meeting metadata (titles like
# "Weekly Sync Review", which is Title-Cased styling, not proper nouns) from
# polluting the domain vocabulary in the first place.
_COMMON_ENGLISH_WORDS: Set[str] = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "by",
    "at", "from", "is", "are", "was", "were", "be", "been", "being", "this",
    "that", "these", "those", "rest", "mom", "mum", "dad", "sir", "madam",
    "weekly", "monthly", "daily", "quarterly", "annual", "meeting", "meetings",
    "review", "sync", "update", "updates", "team", "project", "discussion",
    "session", "call", "calls", "status", "planning", "kickoff", "follow",
    "followup", "up", "notes", "note", "report", "reports", "summary",
    "overview", "general", "regular", "standup", "agenda", "recap", "must",
    "need", "needs", "please", "will", "can", "should", "task", "tasks",
    "action", "actions", "item", "items", "new", "old", "next", "final",
    "draft", "week", "month", "day", "year", "part", "phase", "stage", "step",
    "topic", "topics", "point", "points", "issue", "issues", "plan", "plans",
    "goal", "goals", "target", "targets",
}


def _soundex(word: str) -> str:
    """Pure-Python Soundex phonetic algorithm."""
    word = re.sub(r"[^a-zA-Z]", "", word).upper()
    if not word:
        return ""
    code_map = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6"
    }
    first = word[0]
    coded = [code_map.get(char, "0") for char in word[1:]]
    # Remove consecutive duplicates
    cleaned = []
    prev = code_map.get(first, "0")
    for digit in coded:
        if digit != "0" and digit != prev:
            cleaned.append(digit)
        prev = digit
    res = first + "".join(cleaned)
    return (res + "000")[:4]


def _simplified_metaphone(word: str) -> str:
    """Pure-Python simplified Metaphone key computation."""
    word = re.sub(r"[^a-zA-Z]", "", word).upper()
    if not word:
        return ""
    # Simplify common phonetic equivalents
    word = re.sub(r"DG", "K", word)
    word = re.sub(r"PH", "F", word)
    word = re.sub(r"CK", "K", word)
    word = re.sub(r"GH", "F", word)
    word = re.sub(r"TH", "0", word)
    word = re.sub(r"[AEIOU]", "", word[1:])
    return (word[:6]) if word else ""


def compute_phonetic_similarity(str1: str, str2: str) -> float:
    """
    Computes pure-Python Soundex & Metaphone phonetic similarity score in [0.0, 1.0].
    """
    s1 = _soundex(str1)
    s2 = _soundex(str2)
    soundex_sim = 1.0 if (s1 and s1 == s2) else (0.5 if (s1 and s2 and s1[0] == s2[0]) else 0.0)

    m1 = _simplified_metaphone(str1)
    m2 = _simplified_metaphone(str2)
    metaphone_sim = difflib.SequenceMatcher(None, m1, m2).ratio() if (m1 and m2) else soundex_sim

    return round(0.5 * soundex_sim + 0.5 * metaphone_sim, 3)


def compute_candidate_score(
    raw_term: str,
    candidate_term: str,
    context_sentence: str = "",
    domain_vocab: Optional[Dict[str, Any]] = None,
    asr_confidence: float = 0.80,
) -> float:
    """
    Computes multi-signal candidate score using centralized weights from settings (Stage 3 & 4):
    Candidate Score = w_p * Phonetic + w_t * Text + w_c * Context + w_d * Domain + w_s * ASR Signal
    """
    if not raw_term or not candidate_term:
        return 0.0

    raw_clean = raw_term.strip()
    cand_clean = candidate_term.strip()

    # 1. Phonetic similarity
    p_sim = compute_phonetic_similarity(raw_clean, cand_clean)

    # 2. Textual sequence similarity (Levenshtein / SequenceMatcher)
    t_sim = difflib.SequenceMatcher(None, raw_clean.lower(), cand_clean.lower()).ratio()

    # 3. Contextual sentence fit
    c_sim = 1.0 if (cand_clean.lower() in context_sentence.lower()) else 0.5

    # 4. Domain relevance
    d_vocab = (domain_vocab or {}).get("vocabulary_terms", [])
    d_sim = 1.0 if any(cand_clean.lower() == v.lower() for v in d_vocab) else 0.3

    # 5. ASR signal
    s_sim = max(0.0, min(1.0, asr_confidence))

    # Apply centralized weights from Settings
    w_p = settings.CORRECTION_WEIGHT_PHONETIC
    w_t = settings.CORRECTION_WEIGHT_TEXT
    w_c = settings.CORRECTION_WEIGHT_CONTEXT
    w_d = settings.CORRECTION_WEIGHT_DOMAIN
    w_s = settings.CORRECTION_WEIGHT_ASR

    total_weight = w_p + w_t + w_c + w_d + w_s
    candidate_score = (
        (w_p * p_sim) + (w_t * t_sim) + (w_c * c_sim) + (w_d * d_sim) + (w_s * s_sim)
    ) / (total_weight if total_weight > 0 else 1.0)

    return round(candidate_score, 3)


def extract_domain_vocabulary(
    meeting_title: Optional[str] = None,
    agenda: Optional[str] = None,
    project_name: Optional[str] = None,
    participants: Optional[str] = None,
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
    db: Optional[Any] = None,
    meeting_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Dynamically extracts valid terminology, entity names, and approved vocabulary from
    meeting title, metadata, agenda, project name, database records, and transcript.
    (Stage 2 & Stage 12 — Dynamic Terminology Learning)

    ``meeting_id``: required whenever ``db`` is supplied — scopes the optional
    database lookup to THIS meeting's own prior action items only, so a task
    word or assignee name from a different meeting can never leak into this
    meeting's correction vocabulary.
    """
    vocab_terms: Set[str] = set(PROTECTED_TERMS)
    entity_names: Set[str] = set()

    # `participants` is a structured name list (comma/space separated) — every
    # capitalized token there is a genuine proper noun, so it is trusted as-is.
    if participants and isinstance(participants, str):
        for w in participants.strip().split():
            clean_w = w.strip(".,;:!?()[]\"'")
            if clean_w and clean_w[0].isupper() and len(clean_w) > 1:
                vocab_terms.add(clean_w)
                if not clean_w.isupper():
                    entity_names.add(clean_w)

    # `meeting_title` / `agenda` / `project_name` are free text, often styled in
    # Title Case ("Weekly Sync Review") — every word there is capitalized
    # because of title styling, NOT because it is a proper noun or acronym.
    # Only capture words that look like a genuine acronym (ALL-CAPS) or contain
    # a digit (e.g. "Q3", "V2"); anything else risks polluting the vocabulary
    # with ordinary English words that later get force-cased onto unrelated
    # transcript text in normalize_sentence_grammar (e.g. "Sync"/"Review").
    for text_source in [meeting_title, agenda, project_name]:
        if text_source and isinstance(text_source, str):
            for w in text_source.strip().split():
                clean_w = w.strip(".,;:!?()[]\"'")
                if not clean_w or len(clean_w) <= 1:
                    continue
                if clean_w.lower() in _COMMON_ENGLISH_WORDS:
                    continue
                looks_like_acronym = clean_w.isupper()
                has_digit = any(c.isdigit() for c in clean_w)
                if looks_like_acronym or has_digit:
                    vocab_terms.add(clean_w)

    # Extract high-frequency domain terms from transcript segments
    if transcript_segments:
        term_freq: Dict[str, int] = {}
        for seg in transcript_segments:
            text = seg.get("text", "") or seg.get("raw_text", "")
            if text:
                tokens = text.split()
                for t in tokens:
                    clean_t = t.strip(".,;:!?()[]\"'")
                    if (
                        clean_t and clean_t[0].isupper() and len(clean_t) > 2
                        and clean_t.lower() not in _COMMON_ENGLISH_WORDS
                    ):
                        term_freq[clean_t] = term_freq.get(clean_t, 0) + 1
        for term, count in term_freq.items():
            if count >= 2:
                vocab_terms.add(term)

    # Optionally extract verified terms from PostgreSQL database if session provided.
    # Scoped strictly to THIS meeting's own action items — an unfiltered query
    # here would leak every other meeting's task words/assignee names into this
    # meeting's correction vocabulary.
    if db is not None and meeting_id is not None:
        try:
            from backend.models.models import ActionItem
            from backend.utils.db_safe import optional_db

            # Optional vocabulary source, so savepoint-guarded: a failed read must not leave the caller's
            # transaction aborted (every later statement would raise InFailedSqlTransaction).
            items = optional_db(
                db,
                lambda: (
                    db.query(ActionItem.task, ActionItem.owner)
                    .filter(ActionItem.meeting_id == meeting_id)
                    .all()
                ),
                default=[],
                label=f"correction vocabulary lookup for meeting {meeting_id}",
            ) or []
            for task, owner in items:
                if task:
                    for token in task.split():
                        c_tok = token.strip(".,;:!?()[]\"'")
                        if (
                            c_tok and c_tok[0].isupper() and len(c_tok) > 2
                            and c_tok.lower() not in _COMMON_ENGLISH_WORDS
                        ):
                            vocab_terms.add(c_tok)
                if owner and owner != "Unassigned":
                    vocab_terms.add(owner)
        except Exception:
            pass
    elif db is not None and meeting_id is None:
        logger.warning(
            "[STT Correction] extract_domain_vocabulary got db= without meeting_id= "
            "— skipping the database vocabulary lookup to avoid pulling every "
            "meeting's action items into this meeting's correction vocabulary."
        )

    return {
        "vocabulary_terms": sorted(list(vocab_terms)),
        "entity_names": sorted(list(entity_names)),
        "project_name": project_name or "",
        "meeting_title": meeting_title or "",
    }


def build_vocabulary_hint(
    meeting_title: Optional[str] = None,
    agenda: Optional[str] = None,
    project_name: Optional[str] = None,
    participants: Optional[str] = None,
    max_chars: int = 220,
) -> str:
    """Build a short, dynamic vocabulary hint for the ASR decoder (Whisper ``initial_prompt`` /
    OpenAI-compatible ``prompt`` field).

    This is a ROOT-CAUSE fix for mis-transcribed proper nouns, organization names, and project
    terminology: biasing the decoder toward the correct spelling *before* it decodes is far more
    reliable than trying to fix a wrong word after the fact. The hint is built entirely from
    THIS meeting's own user-supplied metadata (title/agenda/project/participants) — nothing is
    hardcoded or shared across meetings, so a different meeting produces a different hint (or
    none at all, when no metadata is available yet, which is a safe no-op).

    Whisper only attends to a small trailing window of the prompt, so this is capped short and
    written as plain comma-separated terms/names rather than prose.
    """
    seen: Set[str] = set()
    terms: List[str] = []
    for source in (meeting_title, project_name, agenda, participants):
        if not source or not isinstance(source, str):
            continue
        for raw_word in re.split(r"[,\n;/]+", source):
            word = raw_word.strip().strip(".,;:!?()[]\"'")
            if not word or len(word) < 2:
                continue
            key = word.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(word)

    if not terms:
        return ""

    hint = ", ".join(terms)
    return hint[:max_chars].rstrip(", ")


_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def correction_preserves_content(
    raw_text: str, corrected_text: str, max_word_change: float = 0.15, min_word_overlap: float = 0.70
) -> Tuple[bool, str]:
    """Is ``corrected_text`` still the same speech as ``raw_text`` - only spelled, cased and punctuated better?

    Returns ``(ok, reason_if_not)``. A correction is a light edit, so it keeps (almost) every word: the word count may
    move by at most ``max_word_change`` (never less than 2 words), and at least ``min_word_overlap`` of the raw words
    must still be present. Anything else is a rewrite, a summary or a cut-off answer and is never used as the
    transcript. Measured failure this guards: a 21,504-char / 3,494-word segment came back from the model as 1,673
    chars (260 words, "Corrected minor punctuation and grammar errors"), scored 0.73 >= the 0.70 gate because a
    truncated prefix is a substring of the original (full "contextual fit"), and replaced the transcript - 92 % of
    the meeting gone while every stage reported SUCCESS.
    """
    raw_words = [w.lower() for w in _WORD_RE.findall(raw_text or "")]
    new_words = [w.lower() for w in _WORD_RE.findall(corrected_text or "")]
    if not raw_words:
        return True, ""
    if not new_words:
        return False, f"the correction is empty ({len(raw_words)} words in the transcript)"
    allowed = max(2, int(len(raw_words) * max_word_change + 0.9999))
    if abs(len(new_words) - len(raw_words)) > allowed:
        return False, f"word count changed from {len(raw_words)} to {len(new_words)} (limit +/-{allowed})"
    from collections import Counter

    kept = sum((Counter(raw_words) & Counter(new_words)).values()) / len(raw_words)
    if kept < min_word_overlap:
        return False, f"only {kept:.0%} of the original words remain (minimum {min_word_overlap:.0%})"
    return True, ""


_FILLER_WORDS = frozenset({"uh", "um", "umm", "erm", "er", "ah", "hmm", "mm"})


def _bare(token: str) -> str:
    """The letters/digits of a token, lower-cased (punctuation and case stripped)."""
    return "".join(_WORD_RE.findall(token or "")).lower()


def merge_corrections_with_evidence(
    raw_text: str,
    candidate_text: str,
    domain_vocab: Optional[Dict[str, Any]],
    transcript_words: Optional[Set[str]] = None,
    asr_confidence: float = 0.85,
    min_score: Optional[float] = None,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Word-level merge of an LLM proposal into the raw ASR text - only changes with EVIDENCE are kept.

    The model is asked to fix mis-heard terms, but it will also "fix" words that are merely unclear by writing
    something plausible; a transcript must never contain words the audio does not support. So the proposal is
    diffed against the raw words and each change is judged on its own:

    * unchanged words keep the RAW token (case / punctuation is the grammar pass's job, not the model's);
    * a replaced word is accepted only when it (a) looks or sounds like the raw word AND (b) is backed by this
      meeting's own context - its title / agenda / participants / project vocabulary, or the same spelling already
      occurring elsewhere in the raw transcript - AND (c) scores >= the correction confidence threshold;
    * words the model INSERTED are never accepted (that is invented speech); words it DELETED are accepted only for
      disfluencies (uh / um) and immediate repeats.

    Returns ``(merged_text, accepted_changes, rejected_changes)``. A rejected change leaves the raw word in place.
    The confidence of an accepted change is the measured candidate score, not the model's self-reported number.
    """
    threshold = settings.CORRECTION_CONFIDENCE_THRESHOLD if min_score is None else min_score
    vocab_terms = {_bare(t) for t in ((domain_vocab or {}).get("vocabulary_terms") or []) if _bare(t)}
    seen_words = transcript_words or set()

    raw_tokens = (raw_text or "").split()
    cand_tokens = (candidate_text or "").split()
    raw_keys = [_bare(t) for t in raw_tokens]
    cand_keys = [_bare(t) for t in cand_tokens]

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    merged: List[str] = []

    def _evidence(raw_word: str, cand_word: str) -> Tuple[Optional[Dict[str, Any]], str]:
        raw_key, cand_key = _bare(raw_word), _bare(cand_word)
        if not raw_key or not cand_key:
            return None, "empty_word"
        text_sim = difflib.SequenceMatcher(None, raw_key, cand_key).ratio()
        phon_sim = compute_phonetic_similarity(raw_key, cand_key)
        if text_sim < 0.6 and phon_sim < 0.8:
            return None, f"not_similar_to_the_heard_word (text={text_sim:.2f}, phonetic={phon_sim:.2f})"
        if cand_key in vocab_terms:
            source = "domain_vocabulary"
        elif cand_key in seen_words:
            source = "transcript_consistency"
        else:
            return None, "no_evidence_in_this_meetings_context"
        score = compute_candidate_score(
            raw_term=raw_word.strip(".,;:!?()[]\"'"), candidate_term=cand_word.strip(".,;:!?()[]\"'"),
            context_sentence=raw_text, domain_vocab=domain_vocab, asr_confidence=asr_confidence,
        )
        if score < threshold:
            return None, f"below_confidence_threshold ({score:.2f} < {threshold:.2f})"
        return {"evidence": source, "confidence": score, "text_similarity": round(text_sim, 3)}, ""

    matcher = difflib.SequenceMatcher(None, raw_keys, cand_keys, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            merged.extend(raw_tokens[i1:i2])
        elif tag == "insert":
            for tok in cand_tokens[j1:j2]:
                rejected.append({"original": None, "proposed": tok, "reason": "inserted_word_not_in_the_audio_text"})
        elif tag == "delete":
            for k in range(i1, i2):
                prev = raw_keys[k - 1] if k > 0 else None
                if raw_keys[k] in _FILLER_WORDS or (prev is not None and prev == raw_keys[k]):
                    accepted.append({"original": raw_tokens[k], "corrected": "", "reason": "disfluency removed",
                                     "evidence": "disfluency", "confidence": 0.95})
                else:
                    merged.append(raw_tokens[k])
                    rejected.append({"original": raw_tokens[k], "proposed": "", "reason": "deleted_word_without_evidence"})
        else:  # replace
            raw_block, cand_block = raw_tokens[i1:i2], cand_tokens[j1:j2]
            if len(raw_block) == len(cand_block):
                pairs = list(zip(raw_block, cand_block))
            else:   # words merged or split ("gov in" -> "Govin"): judge the block as one term
                pairs = [(" ".join(raw_block), " ".join(cand_block))]
            for raw_word, cand_word in pairs:
                info, why = _evidence(raw_word.replace(" ", ""), cand_word.replace(" ", ""))
                if info is None:
                    merged.extend(raw_word.split())
                    rejected.append({"original": raw_word, "proposed": cand_word, "reason": why})
                else:
                    # keep the raw token's trailing punctuation: the model's punctuation is not evidence
                    lead = re.match(r"^\W*", raw_word).group(0)
                    trail = re.search(r"\W*$", raw_word).group(0)
                    core = cand_word.strip(".,;:!?()[]\"'")
                    merged.append(f"{lead}{core}{trail}")
                    accepted.append({"original": raw_word.strip(".,;:!?()[]\"'"), "corrected": core,
                                     "reason": f"supported by {info['evidence'].replace('_', ' ')}",
                                     "evidence": info["evidence"], "confidence": info["confidence"]})
    return " ".join(merged), accepted, rejected


def find_phonetic_domain_correction(
    word: str,
    domain_vocab: Optional[Dict[str, Any]],
    context_sentence: str = "",
    min_score: float = 0.65,
    min_text_similarity: float = 0.70,
) -> Optional[Tuple[str, float]]:
    """Proactively search THIS meeting's own dynamic domain vocabulary (Stage 2 output —
    never a cross-meeting or hardcoded dictionary) for a term that is phonetically/textually
    close to ``word`` but not already identical to it.

    This is the active counterpart to ``compute_candidate_score``: previously that scorer
    only ever validated a candidate the LLM had already proposed, so a meeting processed
    with ``use_llm=False`` (or where the LLM pass fails/times out) got zero benefit from the
    domain vocabulary that Stage 2 had already extracted. This closes that gap as a
    conservative fallback — deliberately narrow (capitalized tokens only, bounded length
    delta, and — critically — a hard raw-textual-similarity gate) since an over-eager fuzzy
    match on a real, correctly-heard proper noun would itself corrupt the transcript, which
    is exactly what must be avoided.

    The textual-similarity gate exists because the blended score alone does not safely
    separate a genuine near-miss mishearing from a coincidental partial phonetic overlap
    between two unrelated words: empirically, "Govin" vs "Govind" (a real mishearing) and
    "Ravi" vs "Review" (two unrelated words that must NEVER be conflated) can land close
    together on the blended score, but their raw spelling similarity does not — "Govin"/
    "Govind" and "Gogin"/"Govind" both sit at 0.70+, while "Ravi"/"Review" sits at 0.60.
    Requiring both signals keeps this feature safe rather than merely plausible-looking.

    Returns ``(matched_term, score)`` or ``None`` if no sufficiently confident match exists.
    """
    if not word or not domain_vocab:
        return None
    vocab_terms = domain_vocab.get("vocabulary_terms") or []
    if not vocab_terms:
        return None
    clean_word = word.strip(".,;:!?()[]\"'")
    # Only proper-noun-like tokens: this is the class of word an ASR engine is most
    # likely to mishear (org/product/person names) and where a fuzzy match against
    # THIS meeting's own vocabulary is safe to attempt.
    if len(clean_word) < 3 or not clean_word[0:1].isupper() or not clean_word.replace("-", "").isalpha():
        return None
    if clean_word.lower() in {v.lower() for v in vocab_terms}:
        return None  # already correct — never touch a word that's already right

    best_term, best_score = None, 0.0
    for term in vocab_terms:
        term_clean = term.strip()
        if len(term_clean) < 3 or not term_clean.replace(" ", "").replace("-", "").isalpha():
            continue
        if term_clean.lower() == clean_word.lower():
            continue
        # Guard against nonsensical length mismatches (e.g. "It" fuzzy-matching a
        # long unrelated vocabulary term purely on a coincidental score).
        if abs(len(term_clean) - len(clean_word)) > max(3, len(clean_word) // 2):
            continue
        # Hard safety gate (see docstring): raw spelling similarity must itself be
        # high before the blended score is even considered.
        text_sim = difflib.SequenceMatcher(None, clean_word.lower(), term_clean.lower()).ratio()
        if text_sim < min_text_similarity:
            continue
        score = compute_candidate_score(
            raw_term=clean_word,
            candidate_term=term_clean,
            context_sentence=context_sentence,
            domain_vocab=domain_vocab,
            asr_confidence=0.75,
        )
        if score > best_score:
            best_score, best_term = score, term_clean

    if best_term and best_score >= min_score:
        return best_term, best_score
    return None


def normalize_sentence_grammar(text: str, domain_vocab: Optional[Dict[str, Any]] = None) -> Tuple[str, bool]:
    """
    Performs sentence boundary capitalization, punctuation restoration,
    and 3+ duplicate filler word cleanup without changing facts, dates, names, or meaning.
    (Stage 6 & Stage 7 — Grammar and Sentence Correction)
    """
    if not text or not text.strip():
        return text, False

    original = text
    cleaned = text.strip()

    # 1. Clean ASR repetition loops: 5+ identical consecutive words ("we we we we we need" -> "we need"). A genuine
    #    "no no no" / "very very very" (3-4 in a row) is speech and is kept.
    cleaned = re.sub(r"\b(\w+)(?:\s+\1){4,}\b", r"\1", cleaned, flags=re.IGNORECASE)

    # 2. Capitalize first letter of sentence if uncapitalized
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    # 3. Ensure proper sentence closing punctuation if missing
    if cleaned and not cleaned[-1] in ".!?\":;)":
        cleaned += "."

    # 4. Protect domain terms case from vocabulary dictionary.
    # Defense in depth: never force-case a term that collides with an ordinary
    # English word (e.g. PROTECTED_TERMS' "REST"/"MOM" vs "take a rest"/"my
    # mom") — a case-insensitive match can't tell the acronym from the common
    # word, so blindly substituting corrupts genuine, unrelated sentences.
    if domain_vocab and domain_vocab.get("vocabulary_terms"):
        for term in domain_vocab["vocabulary_terms"]:
            if term.lower() in _COMMON_ENGLISH_WORDS:
                continue
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            cleaned = pattern.sub(term, cleaned)

    changed = (cleaned != original)
    return cleaned, changed


def correct_and_normalize_transcript(
    segments: List[Dict[str, Any]],
    meeting_title: Optional[str] = None,
    agenda: Optional[str] = None,
    project_name: Optional[str] = None,
    participants: Optional[str] = None,
    use_llm: bool = True,
    db: Optional[Any] = None,
    meeting_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Main entry point for dynamic STT correction, multi-signal candidate scoring,
    and grammar normalization.
    Preserves immutable raw_text for every segment.

    ``meeting_id``: forwarded to ``extract_domain_vocabulary`` so an optional
    ``db`` session only ever pulls THIS meeting's own prior action items —
    never every meeting's task words/assignee names.

    Stage 16 Failure Safety: If correction fails, sets correction_status = FAILED
    and safely continues with raw transcript without failing meeting processing.
    """
    if not segments:
        return []

    try:
        # Stage 2: Dynamic Terminology Discovery
        domain_vocab = extract_domain_vocabulary(
            meeting_title=meeting_title,
            agenda=agenda,
            project_name=project_name,
            participants=participants,
            transcript_segments=segments,
            db=db,
            meeting_id=meeting_id,
        )

        corrected_segments: List[Dict[str, Any]] = []
        llm_corrections_map: Dict[int, Dict[str, Any]] = {}
        # Every word the recognizer produced anywhere in this meeting: a proposed spelling that already occurs
        # elsewhere in the raw transcript is evidence (consistency), one that occurs nowhere is not.
        transcript_words: Set[str] = {
            _bare(w)
            for s in segments
            for w in (s.get("raw_text") or s.get("text", "") or "").split()
            if _bare(w)
        }

        # Stage 5: AI LLM Context-Aware Correction Pass
        if use_llm and settings.TRANSCRIPT_CORRECTION_ENABLED:
            try:
                from backend.services.model_router import model_router
                prompt_system = transcript_correction_prompt()
                full_context_text = "\n".join(
                    [f"[{i}] {s.get('speaker', 'Speaker')}: {s.get('text', '') or s.get('raw_text', '')}" for i, s in enumerate(segments)]
                )
                user_msg = (
                    f"Domain Vocabulary: {domain_vocab['vocabulary_terms']}\n"
                    f"Project/Organization: {domain_vocab['project_name']} | {domain_vocab['meeting_title']}\n\n"
                    f"Transcript Segments:\n{full_context_text}"
                )
                # ModelRouter exposes a LangChain chat model via get_llm(); it has
                # no ``completion()`` method. Invoke the chat model directly and
                # read ``.content`` off the returned AIMessage.
                _llm = model_router.get_llm(temperature=0.1)
                _llm_result = _llm.invoke(
                    [
                        {"role": "system", "content": prompt_system},
                        {"role": "user", "content": user_msg},
                    ]
                )
                raw_response = getattr(_llm_result, "content", None) or str(_llm_result)
                if raw_response and raw_response.strip():
                    clean_json = raw_response.strip()
                    if clean_json.startswith("```"):
                        clean_json = re.sub(r"^```(?:json)?\n|\n```$", "", clean_json, flags=re.IGNORECASE).strip()
                    parsed = json.loads(clean_json)
                    if isinstance(parsed, dict) and "segments" in parsed and isinstance(parsed["segments"], list):
                        # Only corrected segments are returned, keyed by their input [index]: echoing every segment
                        # made the answer outgrow the output limit on longer meetings (JSON cut off mid-string,
                        # correction silently skipped after ~45 s). A list without indexes is read positionally.
                        for pos, corr_item in enumerate(parsed["segments"]):
                            if not isinstance(corr_item, dict):
                                continue
                            try:
                                idx = int(corr_item["index"]) if "index" in corr_item else pos
                            except (TypeError, ValueError):
                                continue
                            if 0 <= idx < len(segments):
                                llm_corrections_map[idx] = corr_item
            except Exception as err:
                logger.warning(f"[STT Correction] LLM pass skipped: {err}. Proceeding with multi-signal candidate scoring.")

        # Stages 3-10: Process each segment sequentially
        for idx, seg in enumerate(segments):
            speaker = seg.get("speaker", "Speaker_1")
            start_time = seg.get("start_time", "0.0")
            end_time = seg.get("end_time", "0.0")
            raw_text = seg.get("raw_text") or seg.get("text", "")
            # The STT engine (speech-to-text API) often reports NO confidence. That stays None: a stored
            # 0.85 was read downstream as a measured "85 % confident" on every segment. The candidate scorer below
            # still needs a number, so it uses a neutral prior that is never stored or reported.
            stt_confidence = seg.get("stt_confidence")
            try:
                stt_confidence = None if stt_confidence is None else float(stt_confidence)
            except (TypeError, ValueError):
                stt_confidence = None
            asr_prior = 0.85 if stt_confidence is None else stt_confidence

            final_text = raw_text
            correction_applied = False
            correction_confidence = 1.0
            correction_reason = "no_correction_needed"
            correction_source = "raw_asr"
            metadata_changes: List[Dict[str, Any]] = []

            # Multi-signal phonetic & domain candidate evaluation
            llm_item = llm_corrections_map.get(idx)
            if llm_item and llm_item.get("correction_applied", True):  # listed = corrected, unless it says otherwise
                cand_text = llm_item.get("corrected_text", raw_text)
                if not isinstance(cand_text, str):
                    cand_text = raw_text
                # The model's text is a PROPOSAL, never the transcript. First the coarse guard: it must still be the
                # same speech (see correction_preserves_content) - a rewrite / cut-off answer is dropped whole.
                preserved, why_not = correction_preserves_content(raw_text, cand_text)
                if not preserved:
                    logger.warning(
                        "[STT Correction] segment %d: LLM correction REJECTED, raw text kept (%s; raw_chars=%d "
                        "candidate_chars=%d).", idx, why_not, len(raw_text or ""), len(cand_text or ""),
                    )
                    correction_reason = f"llm_correction_rejected: {why_not}"
                elif cand_text != raw_text:
                    # Then the evidence gate, word by word: a changed word is kept only when it looks/sounds like the
                    # heard word AND is backed by this meeting's own vocabulary or by the same spelling elsewhere in
                    # the transcript, AND scores above the threshold. The model's self-reported confidence is NOT
                    # evidence (it used to default to 0.90 when absent and carried half of the gate), and words
                    # without support stay exactly as the speech recognizer produced them.
                    merged_text, accepted, rejected = merge_corrections_with_evidence(
                        raw_text, cand_text, domain_vocab, transcript_words, asr_confidence=asr_prior,
                    )
                    if rejected:
                        logger.info(
                            "[STT Correction] segment %d: %d proposed change(s) kept as heard - no evidence: %s",
                            idx, len(rejected),
                            "; ".join(f"{r.get('original')!r}->{r.get('proposed')!r} ({r['reason']})" for r in rejected[:5]),
                        )
                    if accepted and merged_text != raw_text:
                        final_text = merged_text
                        correction_applied = True
                        correction_confidence = round(sum(c["confidence"] for c in accepted) / len(accepted), 2)
                        correction_reason = (
                            f"{len(accepted)} term(s) corrected with evidence"
                            + (f"; {len(rejected)} unsupported change(s) not applied" if rejected else "")
                        )
                        correction_source = "phonetic_contextual_llm"
                        metadata_changes = accepted
                    elif rejected:
                        correction_reason = f"llm_changes_unsupported: {len(rejected)} change(s) had no evidence; raw words kept"

            # Stage 3/4 fallback: proactive phonetic domain-vocabulary matching.
            # Only runs when the LLM pass above did NOT already correct this segment
            # (disabled, failed, timed out, or simply didn't flag it) — a safety net,
            # never a second opinion overriding the LLM. Scoped to individual
            # capitalized tokens so it can only ever touch proper-noun-like words,
            # matched against THIS meeting's own dynamically extracted vocabulary.
            if not correction_applied and domain_vocab and domain_vocab.get("vocabulary_terms"):
                tokens_in_text = final_text.split()
                phonetic_changes: List[Dict[str, Any]] = []
                for wi, tok in enumerate(tokens_in_text):
                    core = tok.strip(".,;:!?()[]\"'")
                    if not core:
                        continue
                    match = find_phonetic_domain_correction(core, domain_vocab, context_sentence=final_text)
                    if match:
                        matched_term, score = match
                        tokens_in_text[wi] = tok.replace(core, matched_term, 1)
                        phonetic_changes.append({
                            "original": core,
                            "corrected": matched_term,
                            "reason": "phonetic match against this meeting's own domain vocabulary",
                            "confidence": score,
                        })
                if phonetic_changes:
                    final_text = " ".join(tokens_in_text)
                    correction_applied = True
                    correction_confidence = round(
                        sum(c["confidence"] for c in phonetic_changes) / len(phonetic_changes), 2
                    )
                    correction_reason = "phonetic domain-vocabulary correction"
                    correction_source = "phonetic_domain_vocab"
                    metadata_changes.extend(phonetic_changes)

            # Stage 6 & 7: Grammar and Punctuation Restoration
            norm_text, norm_changed = normalize_sentence_grammar(final_text, domain_vocab)
            if norm_changed:
                final_text = norm_text
                if not correction_applied:
                    correction_applied = True
                    correction_confidence = 0.95
                    # keep the audit trail of a rejected LLM rewrite next to the grammar pass that was applied
                    correction_reason = (
                        f"{correction_reason}; grammar and punctuation restoration"
                        if correction_reason.startswith("llm_correction_rejected")
                        else "grammar and punctuation restoration"
                    )
                    correction_source = "grammar_normalization"
                    metadata_changes.append({
                        "original": raw_text,
                        "corrected": final_text,
                        "reason": "grammar and punctuation restoration",
                        "confidence": 0.95,
                    })

            corrected_segments.append({
                "speaker": speaker,
                # Speaker-identity fields carried through untouched (added by
                # diarization_service.apply_speaker_identity); never guessed here.
                "speaker_id": seg.get("speaker_id"),
                "speaker_name": seg.get("speaker_name"),
                "speaker_confidence": seg.get("speaker_confidence"),
                "label": seg.get("label", speaker),
                "name_source": seg.get("name_source"),
                "start_time": start_time,
                "end_time": end_time,
                "raw_text": raw_text,
                "text": final_text,
                "stt_confidence": stt_confidence,
                "correction_applied": correction_applied,
                "correction_confidence": round(correction_confidence, 2) if correction_applied else 1.0,
                "correction_source": correction_source,
                "correction_reason": correction_reason,
                "correction_metadata": json.dumps(metadata_changes) if metadata_changes else None,
            })

        logger.info(
            f"[STT Correction] Successfully processed {len(segments)} segment(s): "
            f"{sum(1 for s in corrected_segments if s['correction_applied'])} corrected."
        )
        return corrected_segments

    except Exception as exc:
        # Stage 16 Failure Safety: Fallback cleanly to raw transcript without breaking pipeline
        logger.error(f"[STT Correction] Exception encountered: {exc}. Falling back to raw transcript.")
        fallback_segments: List[Dict[str, Any]] = []
        for seg in segments:
            raw_text = seg.get("raw_text") or seg.get("text", "")
            fallback_segments.append({
                "speaker": seg.get("speaker", "Speaker_1"),
                "speaker_id": seg.get("speaker_id"),
                "speaker_name": seg.get("speaker_name"),
                "speaker_confidence": seg.get("speaker_confidence"),
                "label": seg.get("label", seg.get("speaker", "Speaker_1")),
                "name_source": seg.get("name_source"),
                "start_time": seg.get("start_time", "0.0"),
                "end_time": seg.get("end_time", "0.0"),
                "raw_text": raw_text,
                "text": raw_text,
                "stt_confidence": seg.get("stt_confidence"),     # None stays None (see above)
                "correction_applied": False,
                "correction_confidence": 1.0,
                "correction_source": "raw_asr_fallback",
                "correction_reason": f"correction_failed: {exc}",
                "correction_metadata": None,
            })
        return fallback_segments

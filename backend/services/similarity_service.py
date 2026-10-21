"""Intelligent Meeting Similarity Service.

Thinks like a project manager instead of comparing embeddings alone. Every
candidate is scored across multiple independent signals (semantic embedding,
raw content overlap, keyword/topic overlap, participant overlap, title family,
temporal cadence, shared action items & decisions) and then *classified* into a
human-meaningful category with a plain-English explanation, a confidence score,
a full difference analysis and an evolution timeline.

Categories:
    exact_duplicate | near_duplicate | follow_up | recurring | related | not_related

The engine is fully dynamic (no hardcoded meeting rules), language-agnostic and
works for any meeting length, participant count or organization. An optional LLM
layer only *polishes* the prose — it never drives the classification, so the
feature keeps working even fully offline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.config.config import DEFAULT_PROJECT_NAME
from backend.services.summary_service import summary_preview
from backend.services.vector_service import vector_service
from backend.utils import text_signals as ts

logger = logging.getLogger(__name__)


def _as_text(value: Any) -> str:
    """Normalize a model field (str or list of items) into bullet text."""
    if value is None:
        return ""
    import json
    if isinstance(value, str):
        v_str = value.strip()
        if (v_str.startswith("[") and v_str.endswith("]")) or (v_str.startswith("{") and v_str.endswith("}")):
            try:
                parsed = json.loads(v_str)
                return _as_text(parsed)
            except Exception:
                pass
        return value
    if isinstance(value, (list, tuple, set)):
        lines = []
        for v in value:
            if v in (None, ""):
                continue
            if isinstance(v, dict):
                task = v.get("task") or v.get("title") or v.get("description") or v.get("decision") or v.get("agenda") or ""
                owner = v.get("owner") or v.get("assigned_to") or ""
                deadline = v.get("deadline") or v.get("due_date") or ""
                parts = [f"- {task}"] if task else []
                if owner and owner.lower() not in ("unassigned", "none", "not specified", ""):
                    parts.append(f"Assignee: {owner}")
                if deadline and deadline.lower() not in ("not specified", "none", ""):
                    parts.append(f"Deadline: {deadline}")
                if parts:
                    lines.append(" ".join(parts))
                else:
                    lines.append(f"- {json.dumps(v)}")
            else:
                lines.append(str(v))
        return "\n".join(lines)
    return str(value)


# ─── Categories ──────────────────────────────────────────────────────────────
class Category:
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    FOLLOW_UP = "follow_up"
    RECURRING = "recurring"
    RELATED = "related"
    NOT_RELATED = "not_related"


CATEGORY_META = {
    Category.EXACT_DUPLICATE: {
        "label": "Exact Duplicate",
        "tone": "red",
        "hint": "The same meeting appears to have been uploaded more than once.",
    },
    Category.NEAR_DUPLICATE: {
        "label": "Near Duplicate",
        "tone": "orange",
        "hint": "Almost identical with only minor differences.",
    },
    Category.FOLLOW_UP: {
        "label": "Follow-up Meeting",
        "tone": "violet",
        "hint": "A continuation of a previous meeting.",
    },
    Category.RECURRING: {
        "label": "Recurring Meeting",
        "tone": "blue",
        "hint": "Part of a recurring series discussing a similar agenda.",
    },
    Category.RELATED: {
        "label": "Related Meeting",
        "tone": "emerald",
        "hint": "A different meeting discussing similar topics.",
    },
    Category.NOT_RELATED: {
        "label": "Not Related",
        "tone": "slate",
        "hint": "No meaningful relationship.",
    },
}

# Lowest meeting-embedding similarity at which lexical overlap may make two meetings "related".
_RELATED_SEMANTIC_FLOOR = 0.20

# ─── Signal weights for the blended similarity score ─────────────────────────
_WEIGHTS = {
    "semantic": 0.45,
    "content": 0.18,
    "keyword": 0.12,
    "title": 0.10,
    "participant": 0.10,
    "bullets": 0.05,
}

# ─── Meeting-type heuristics (keyword based, dynamic) ────────────────────────
_MEETING_TYPE_HINTS = [
    ("Daily Standup", ("standup", "stand-up", "daily scrum", "daily sync")),
    ("Sprint Planning", ("sprint planning", "backlog grooming", "sprint kickoff")),
    ("Sprint Retrospective", ("retrospective", "retro ")),
    ("Sprint Review / Demo", ("sprint review", "demo", "showcase")),
    ("Project Kickoff", ("kickoff", "kick-off", "kick off")),
    (
        "Status / Sync",
        ("status update", "weekly sync", "sync up", "catch up", "check-in"),
    ),
    (
        "Technical Review",
        ("technical review", "design review", "architecture", "code review"),
    ),
    ("Strategy / Planning", ("strategy", "roadmap", "planning", "quarterly")),
    ("Client / Stakeholder", ("client", "stakeholder", "customer")),
    ("Interview", ("interview", "screening")),
    ("Training / Workshop", ("training", "workshop", "onboarding")),
]


@dataclass
class MeetingSnapshot:
    """Everything needed to reason about a meeting, gathered once from the DB."""

    id: int
    code: str
    title: str = ""
    filename: str = ""
    file_type: str = "media"
    status: str = ""
    is_approved: bool = False
    upload_time: Optional[str] = None
    meeting_date: Optional[str] = None
    meeting_time: Optional[str] = None
    duration: float = 0.0
    project_name: str = DEFAULT_PROJECT_NAME
    participants: Optional[str] = None
    participant_count: int = 0
    transcript: str = ""
    summary: str = ""
    mom: str = ""
    action_items: str = ""
    decisions: str = ""
    risks: str = ""
    keyword_frequency: Optional[str] = None
    fingerprint: str = ""
    file_size: int = 0
    document_names: List[str] = field(default_factory=list)

    @property
    def owner(self) -> str:
        parts = ts.parse_participants(self.participants)
        return parts[0] if parts else "Not identified"

    @property
    def meeting_type(self) -> str:
        haystack = f"{self.title} {self.mom[:1200]} {self.summary[:600]}".lower()
        for label, keys in _MEETING_TYPE_HINTS:
            if any(k in haystack for k in keys):
                return label
        return "Document" if self.file_type == "document" else "General Discussion"


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot loading
# ─────────────────────────────────────────────────────────────────────────────
def build_snapshot(repo, meeting_id: Any) -> Optional[MeetingSnapshot]:
    """Build a MeetingSnapshot from a MeetingRepository."""
    m = repo.get_by_id(meeting_id)
    if not m:
        return None
    # The real repository resolves its argument with get_by_id, which returns a Meeting object as is but re-SELECTs
    # the whole (~120 column) row for an id - twice more per snapshot. Repositories without that (test doubles)
    # keep receiving the id.
    ref = m if hasattr(repo, "get_snapshot_rows") else m.id
    segments = repo.get_segments(ref)
    transcript = (
        "\n".join([f"{s.speaker}: {s.text}" for s in segments]) if segments else ""
    )
    analysis = repo.get_analysis(ref)
    return _snapshot_from_parts(m, transcript, analysis)


def build_snapshots(repo, meeting_ids: List[int]) -> Dict[int, MeetingSnapshot]:
    """Snapshots of many meetings in three queries (``repo.get_snapshot_rows``) instead of ~6 per meeting.

    A repository without ``get_snapshot_rows`` (test doubles, other backends) falls back to one
    ``build_snapshot`` per meeting, so behaviour is identical either way.
    """
    ids = [int(i) for i in meeting_ids]
    loader = getattr(repo, "get_snapshot_rows", None)
    if loader is None:
        return {i: snap for i in ids if (snap := build_snapshot(repo, i)) is not None}
    rows = loader(ids)
    snapshots: Dict[int, MeetingSnapshot] = {}
    for meeting_id in ids:
        m = rows["meetings"].get(meeting_id)
        if m is None:
            continue
        segments = rows["segments"].get(meeting_id) or []
        transcript = "\n".join(f"{speaker}: {text}" for speaker, text in segments)
        snapshots[meeting_id] = _snapshot_from_parts(m, transcript, rows["analyses"].get(meeting_id))
    return snapshots


def _snapshot_from_parts(m, transcript: str, analysis) -> MeetingSnapshot:
    """A MeetingSnapshot from a meeting (ORM object or column row), its joined transcript and its analysis."""
    doc_names = []

    return MeetingSnapshot(
        id=m.id,
        code=m.display_id or m.bharatvc_meeting_id or str(m.id),
        title=m.title or m.filename or "",
        filename=m.filename or "",
        file_type=m.file_type or "media",
        status=m.status or "",
        is_approved=bool(m.is_approved),
        upload_time=(
            m.upload_time.isoformat() if getattr(m, "upload_time", None) else None
        ),
        meeting_date=m.meeting_date,
        meeting_time=m.meeting_time,
        duration=float(m.duration or 0.0),
        project_name=m.project_name or DEFAULT_PROJECT_NAME,
        participants=m.participants,
        participant_count=int(m.participant_count or 0),
        transcript=transcript,
        summary=(analysis.summary if analysis else "") or "",
        # THIS meeting's MoM only. ``matched_mom`` is the MoM of the meeting this one MATCHED: presenting it as this
        # meeting's own minutes (the old fallback when no analysis row exists) copies another meeting's outputs
        # into the comparison.
        mom=(analysis.mom if analysis else "") or "",
        action_items=(analysis.action_items if analysis else "") or "",
        decisions=(analysis.decisions if analysis else "") or "",
        risks=(analysis.risks if analysis else "") or "",
        keyword_frequency=m.keyword_frequency,
        fingerprint=(m.content_fingerprint or "") or ts.content_fingerprint(transcript),
        file_size=int(getattr(m, "file_size_bytes", 0) or 0),
        document_names=doc_names,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core pairwise analysis
# ─────────────────────────────────────────────────────────────────────────────
def _same_recording(cur: MeetingSnapshot, other: MeetingSnapshot) -> bool:
    """Both meetings come from the same uploaded file: same original file name, same byte size, same duration.

    Recording identity is the evidence that one meeting was uploaded twice; the transcripts are NOT part of it (two
    processing runs of one file can differ - one may be incomplete). Not a content hash (the SHA-256 computed at
    upload is only logged): name + exact size + duration within half a second is what is available for every row.
    """
    name_a, name_b = (cur.filename or "").strip().lower(), (other.filename or "").strip().lower()
    if not name_a or name_a != name_b or cur.file_size <= 0 or cur.file_size != other.file_size:
        return False
    if cur.duration > 0 and other.duration > 0 and abs(cur.duration - other.duration) > 0.5:
        return False
    return True


def _compute_signals(
    cur: MeetingSnapshot, other: MeetingSnapshot, semantic: float
) -> Dict[str, Any]:
    content = ts.content_overlap(cur.transcript, other.transcript)
    # Symmetric share of the vocabulary two transcripts have in common. ``content`` (overlap coefficient) divides by
    # the SMALLER side, so a 21-word transcript is "100 % contained" in any long one on the same subject.
    content_jaccard = ts.jaccard(ts.token_set(cur.transcript), ts.token_set(other.transcript))
    kw_score, kw_common = ts.keyword_overlap(
        cur.keyword_frequency, other.keyword_frequency
    )
    title_sim = ts.title_similarity(cur.title, other.title)
    part_score, part_matches = ts.participant_overlap(
        cur.participants, other.participants
    )
    action_cmp = ts.compare_bullet_lists(
        _as_text(cur.action_items), _as_text(other.action_items)
    )
    decision_cmp = ts.compare_bullet_lists(
        _as_text(cur.decisions), _as_text(other.decisions)
    )
    risk_cmp = ts.compare_bullet_lists(_as_text(cur.risks), _as_text(other.risks))
    temporal = ts.temporal_relation(cur.meeting_date, other.meeting_date)
    dur_ratio = ts.duration_ratio(cur.duration, other.duration)
    doc_common = sorted(
        {d.lower() for d in cur.document_names}
        & {d.lower() for d in other.document_names}
    )

    bullets_score = 0.0
    denom = 0
    for cmp_ in (action_cmp, decision_cmp):
        total = (len(cmp_["shared"]) + len(cmp_["new"]) + len(cmp_["resolved"])
                 + len(cmp_.get("not_carried_over", [])))
        if total:
            bullets_score += len(cmp_["shared"]) / total
            denom += 1
    bullets_score = bullets_score / denom if denom else 0.0

    fingerprint_match = bool(
        cur.fingerprint and other.fingerprint and cur.fingerprint == other.fingerprint
    )
    same_recording = _same_recording(cur, other)

    return {
        "semantic": semantic,
        "content": content,
        "content_jaccard": content_jaccard,
        "keyword": kw_score,
        "keyword_common": kw_common,
        "title": title_sim,
        "participant": part_score,
        "participant_matches": part_matches,
        "action_cmp": action_cmp,
        "decision_cmp": decision_cmp,
        "risk_cmp": risk_cmp,
        "temporal": temporal,
        "duration_ratio": dur_ratio,
        "bullets": bullets_score,
        "doc_common": doc_common,
        "fingerprint_match": fingerprint_match,
        # The uploaded FILE is the same (name, byte size and duration) - independent of what each run transcribed.
        "same_recording": same_recording,
        "words_current": len(cur.transcript.split()),
        "words_other": len(other.transcript.split()),
    }


def _blended_score(sig: Dict[str, Any]) -> float:
    score = sum(_WEIGHTS[k] * float(sig.get(k, 0.0)) for k in _WEIGHTS)
    return round(min(1.0, max(0.0, score)), 4)


def _classify(
    cur: MeetingSnapshot, other: MeetingSnapshot, sig: Dict[str, Any], min_score: float
) -> str:
    semantic = sig["semantic"]
    content = sig["content"]
    title_sim = sig["title"]
    temporal = sig["temporal"]
    shared_actions = len(sig["action_cmp"]["shared"])
    topic_overlap = max(sig["keyword"], content)

    # 1. Exact duplicate — same content fingerprint or near-perfect overlap.
    # "Identical" needs the transcripts to AGREE in both directions (Jaccard), not one to be contained in the other
    # (``content`` divides by the smaller side: a 260-word fragment is "100 % contained" in the full 3,448 words).
    jaccard_sim = sig.get("content_jaccard", content)
    if sig["fingerprint_match"]:
        return Category.EXACT_DUPLICATE
    if sig.get("same_recording") and jaccard_sim >= 0.8:
        return Category.EXACT_DUPLICATE
    if semantic >= 0.985 and jaccard_sim >= 0.8 and sig["duration_ratio"] >= 0.97:
        return Category.EXACT_DUPLICATE

    # 2. Near duplicate — overwhelmingly the same content, tiny edits. Keyword containment alone never decides it
    #    (84 of 843 pairs with containment >= 0.88 in the live database were a short transcript "contained" in a long
    #    one on the same subject, Jaccard 0.13): the vocabularies must match in BOTH directions, and the meeting
    #    embeddings must agree.
    if (semantic >= 0.90 and content >= 0.72 and jaccard_sim >= 0.5) or (
        content >= 0.88 and jaccard_sim >= 0.6 and semantic >= 0.80
    ):
        return Category.NEAR_DUPLICATE

    # 2b. The same file uploaded twice whose stored transcripts do NOT agree (one run incomplete): not a content
    #     duplicate and never a "follow-up" / "recurring" meeting - a related record, explained as exactly that.
    if sig.get("same_recording"):
        return Category.RELATED

    # 3. Recurring — same title family with a series / cadence marker.
    series_marker = ts.title_has_sequence_marker(
        cur.title
    ) or ts.title_has_sequence_marker(other.title)
    has_cadence = bool(temporal.get("cadence"))
    if title_sim >= 0.6 and (series_marker or has_cadence) and semantic >= 0.28:
        return Category.RECURRING

    # 4. Follow-up — chronological continuation carrying context forward.
    if (
        temporal.get("has_dates")
        and temporal.get("current_is_after")
        and semantic >= 0.33
        and (title_sim >= 0.5 or shared_actions >= 1 or topic_overlap >= 0.25)
    ):
        return Category.FOLLOW_UP

    # 5. Related — meaningful topical overlap. Shared vocabulary only counts together with SOME semantic similarity
    #    of the meetings themselves: generic keyword overlap between two project meetings is not a relationship.
    if semantic >= max(min_score, 0.30) or (
        semantic >= _RELATED_SEMANTIC_FLOOR and (_blended_score(sig) >= min_score or topic_overlap >= 0.18)
    ):
        return Category.RELATED

    return Category.NOT_RELATED


def _confidence(category: str, sig: Dict[str, Any]) -> float:
    if sig["fingerprint_match"]:
        return 0.99
    if category in (Category.EXACT_DUPLICATE, Category.NEAR_DUPLICATE):
        evidence = [sig["semantic"], sig["content"], sig["duration_ratio"]]
    elif category == Category.RECURRING:
        evidence = [
            sig["title"],
            sig["semantic"],
            0.9 if sig["temporal"].get("cadence") else 0.5,
        ]
    elif category == Category.FOLLOW_UP:
        topic = max(sig["keyword"], sig["content"])
        evidence = [
            sig["semantic"],
            topic,
            0.9,
            min(1.0, len(sig["action_cmp"]["shared"]) / 3.0 + 0.3),
        ]
    else:
        evidence = [
            sig["semantic"],
            sig["content"],
            sig["keyword"],
            sig["title"],
            sig["participant"],
        ]
    evidence = [e for e in evidence if e is not None]
    base = sum(evidence) / len(evidence) if evidence else 0.0
    return round(min(1.0, base * 1.05), 2)


def _confidence_label(c: float) -> str:
    if c >= 0.75:
        return "High"
    if c >= 0.5:
        return "Medium"
    return "Low"


def _top(items: List[str], n: int = 6) -> List[str]:
    seen, out = set(), []
    for it in items:
        k = it.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(it)
        if len(out) >= n:
            break
    return out


def _matching_topics(
    sig: Dict[str, Any], extra: Optional[List[str]] = None
) -> List[str]:
    topics = list(extra or [])
    topics += [k.title() for k in sig["keyword_common"]]
    return _top(topics, 8)


def _build_explanation(
    category: str,
    cur: MeetingSnapshot,
    other: MeetingSnapshot,
    sig: Dict[str, Any],
    blended: float,
) -> str:
    date_str = other.meeting_date or "an earlier date"
    gap = ts.humanize_gap(sig["temporal"].get("gap_days"))
    topics = _matching_topics(sig)

    if category == Category.RELATED and sig.get("same_recording"):
        return (
            f'This is the same recording as "{other.title}" ({other.code}): identical file name, size and duration. '
            f"It is not reported as a duplicate because the two stored transcripts differ a lot "
            f"({sig.get('words_current', 0)} vs {sig.get('words_other', 0)} words, "
            f"{round(sig.get('content_jaccard', 0.0) * 100)}% shared vocabulary), so one of the two processing runs "
            f"is probably incomplete - review meeting {other.code} before deleting or merging anything."
        )
    topic_phrase = ""
    if topics:
        top3 = topics[:3]
        topic_phrase = ", ".join(top3[:-1]) + (
            f" and {top3[-1]}" if len(top3) > 1 else top3[0]
        )

    if category == Category.EXACT_DUPLICATE:
        return (
            f'This appears to be the same meeting as "{other.title}" ({other.code}), '
            f"already in the system from {date_str}. The transcript content is virtually identical, "
            f"so this looks like a duplicate upload."
        )
    if category == Category.NEAR_DUPLICATE:
        pct = round(sig.get("content_jaccard", sig["content"]) * 100)
        return (
            f'This meeting is almost identical to "{other.title}" ({other.code}) from {date_str} — '
            f"about {pct}% of the content matches, with only minor differences in wording or edits."
        )
    if category == Category.RECURRING:
        cadence = sig["temporal"].get("cadence")
        cadence_word = f"{cadence.lower()} " if cadence else ""
        base = (
            f"This looks like a {cadence_word}recurring meeting in the same series as "
            f'"{other.title}" ({other.code}).'
        )
        if topic_phrase:
            base += f" Both sessions cover {topic_phrase}."
        return base
    if category == Category.FOLLOW_UP:
        shared = len(sig["action_cmp"]["shared"])
        new_actions = len(sig["action_cmp"]["new"])
        base = (
            f'This meeting continues the discussion started in "{other.title}" ({other.code}), '
            f"held {gap}."
        )
        if shared:
            base += f" It carries forward {shared} action item{'s' if shared != 1 else ''} from that session"
            base += (
                f" and introduces {new_actions} new one{'s' if new_actions != 1 else ''}."
                if new_actions
                else "."
            )
        elif topic_phrase:
            base += f" The conversation builds on {topic_phrase}."
        return base
    # related
    if topic_phrase:
        return (
            f"Both meetings discuss {topic_phrase}. The discussion overlaps by approximately "
            f'{round(blended * 100)}%, so "{other.title}" ({other.code}) is a useful related reference.'
        )
    return (
        f'"{other.title}" ({other.code}) covers similar ground — the discussions overlap by '
        f"approximately {round(blended * 100)}%."
    )


def _build_difference_analysis(
    sig: Dict[str, Any], cur: MeetingSnapshot, other: MeetingSnapshot
) -> Dict[str, Any]:
    cur_parts = ts.parse_participants(cur.participants)
    other_keys = {
        ts._participant_key(p) for p in ts.parse_participants(other.participants)
    }
    new_participants = [
        p for p in cur_parts if ts._participant_key(p) not in other_keys
    ]

    cur_kw = ts.parse_keywords(cur.keyword_frequency)
    other_kw = ts.parse_keywords(other.keyword_frequency)
    whats_new = _top([k.title() for k in sorted(cur_kw - other_kw)], 8)
    whats_removed = _top([k.title() for k in sorted(other_kw - cur_kw)], 8)

    return {
        "whats_new": whats_new,
        "whats_removed": whats_removed,
        "new_action_items": _top(sig["action_cmp"]["new"], 8),
        # Only items explicitly checked off ("[x]") in the previous meeting; an unchecked "[ ]" task that is simply
        # absent now is in ``not_carried_over_action_items`` - it is not evidence of completion.
        "resolved_action_items": _top(sig["action_cmp"]["resolved"], 8),
        "not_carried_over_action_items": _top(sig["action_cmp"]["not_carried_over"], 8),
        "shared_action_items": _top(sig["action_cmp"]["shared"], 8),
        "new_decisions": _top(sig["decision_cmp"]["new"], 8),
        "new_risks": _top(sig["risk_cmp"]["new"], 6),
        "resolved_issues": _top(sig["risk_cmp"]["resolved"], 6),
        "not_carried_over_issues": _top(sig["risk_cmp"]["not_carried_over"], 6),
        "pending_issues": _top(sig["risk_cmp"]["shared"], 6),
        "new_participants": new_participants,
    }


def _build_timeline(
    sig: Dict[str, Any],
    cur: MeetingSnapshot,
    other: MeetingSnapshot,
    diff: Dict[str, Any],
) -> Dict[str, Any]:
    # Order previous/current chronologically.
    if sig["temporal"].get("has_dates") and not sig["temporal"].get("current_is_after"):
        previous, current = cur, other
    else:
        previous, current = other, cur
    return {
        "previous_meeting": {
            "code": previous.code,
            "title": previous.title,
            "date": previous.meeting_date,
            "id": previous.id,
        },
        "current_meeting": {
            "code": current.code,
            "title": current.title,
            "date": current.meeting_date,
            "id": current.id,
        },
        "whats_new": diff["whats_new"] or diff["new_action_items"],
        "whats_changed": diff["new_decisions"],
        "pending_tasks": diff["shared_action_items"] or diff["pending_issues"],
        "completed_tasks": diff["resolved_action_items"],          # explicitly checked off only
        "not_carried_over_tasks": diff["not_carried_over_action_items"],
    }


def _duplicate_info(other: MeetingSnapshot) -> Dict[str, Any]:
    return {
        "original_meeting_id": other.id,
        "original_meeting_code": other.code,
        "original_title": other.title,
        "upload_date": other.upload_time,
        "existing_location": f"/meetings/{other.id}",
        "uploaded_by": other.owner,
        "already_processed": other.status == "completed",
        "is_approved": other.is_approved,
    }


def analyze_pair(
    cur: MeetingSnapshot,
    other: MeetingSnapshot,
    semantic: float,
    min_score: float,
    extra_topics: Optional[List[str]] = None,
    include_details: bool = True,
) -> Dict[str, Any]:
    """Full multi-signal analysis of one candidate against the current meeting."""
    sig = _compute_signals(cur, other, semantic)
    category = _classify(cur, other, sig, min_score)
    blended = _blended_score(sig)
    # ``similarity_score`` is the MEASURED blended score. It was floored at 0.98 / 0.90 for any exact / near duplicate
    # verdict, so 0.98 meant "classified as duplicate", not "98 % similar" (a 0.84 blend was reported as 0.98). The
    # only certain case - identical transcript fingerprint - keeps the floor.
    if category == Category.EXACT_DUPLICATE and sig["fingerprint_match"]:
        blended = max(blended, 0.98)
    elif category in (Category.EXACT_DUPLICATE, Category.NEAR_DUPLICATE):
        # A duplicate verdict rests on the transcripts agreeing (Jaccard >= 0.5 / 0.8): its score is never lower than
        # that MEASURED agreement - a signal that is simply absent (no keywords stored) must not drag it below.
        blended = max(blended, round(float(sig.get("content_jaccard") or 0.0), 4))
    confidence = _confidence(category, sig)
    meta = CATEGORY_META[category]

    result: Dict[str, Any] = {
        "meeting_id": other.id,
        "meeting_code": other.code,
        "title": other.title,
        "filename": other.filename,
        "file_type": other.file_type,
        "meeting_date": other.meeting_date,
        "meeting_time": other.meeting_time,
        "duration": other.duration,
        "participants": other.participants,
        "participants_count": len(ts.parse_participants(other.participants))
        or other.participant_count,
        "main_participant": other.owner,
        "meeting_owner": other.owner,
        "meeting_type": other.meeting_type,
        "project_name": other.project_name,
        "status": other.status,
        "upload_time": other.upload_time,
        "matched_mom": (other.mom[:800] if other.mom else "")
        or "No prior MoM available.",
        # Sentence-bounded and speaker-label-free (even for a summary stored
        # before summaries were label-free) — never a mid-word 400-char cut.
        "quick_summary": summary_preview(other.summary, 400) or summary_preview(other.mom, 400),
        # Classification
        "category": category,
        "category_label": meta["label"],
        "category_tone": meta["tone"],
        "category_hint": meta["hint"],
        "similarity_score": blended,
        "similarity_percentage": f"{blended * 100:.1f}%",
        "confidence_score": confidence,
        "confidence_label": _confidence_label(confidence),
        "reason": _build_explanation(category, cur, other, sig, blended),
        # Signal breakdown (transparency)
        "signals": {
            "semantic": round(sig["semantic"], 3),
            "content_overlap": round(sig["content"], 3),
            "content_jaccard": round(sig.get("content_jaccard", 0.0), 3),
            "keyword_overlap": round(sig["keyword"], 3),
            "title_similarity": round(sig["title"], 3),
            "participant_overlap": round(sig["participant"], 3),
            "duration_ratio": round(sig["duration_ratio"], 3),
            "same_recording": bool(sig.get("same_recording")),
            "words_current": sig.get("words_current"),
            "words_other": sig.get("words_other"),
        },
        # Matching entities
        "matching_topics": _matching_topics(sig, extra_topics),
        "matching_keywords": _top([k.title() for k in sig["keyword_common"]], 10),
        "matching_participants": sig["participant_matches"],
        "matching_speakers": sig["participant_matches"],
        "matching_decisions": _top(sig["decision_cmp"]["shared"], 6),
        "matching_action_items": _top(sig["action_cmp"]["shared"], 6),
        "matching_documents": sig["doc_common"],
        "is_duplicate": category in (Category.EXACT_DUPLICATE, Category.NEAR_DUPLICATE),
    }

    if include_details:
        diff = _build_difference_analysis(sig, cur, other)
        result["difference_analysis"] = diff
        result["timeline"] = _build_timeline(sig, cur, other, diff)
        if result["is_duplicate"]:
            result["duplicate_info"] = _duplicate_info(other)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────
def find_similar_meetings(
    repo,
    meeting_id: Any,
    top_k: int = 8,
    min_score: float = 0.20,
    include_details: bool = True,
    candidate_pool: int = 30,
    session_factory=None,
) -> Dict[str, Any]:
    """Classify and rank similar meetings for the given meeting id.

    Database access is three reads: the current meeting, then (after the embedding + ChromaDB search) all candidate
    meetings at once (``build_snapshots``; it used to be ~6 queries per candidate, 30 candidates, each re-reading
    the meeting row). With ``session_factory`` (a callable returning a Session) each read gets its own short-lived
    session, so NO connection is held while the transcript is embedded and ChromaDB is queried - this is how the
    pipeline calls it, with ``repo=None``. Without it every read uses ``repo`` (the API endpoints).
    """
    def _read(fn):
        if session_factory is None:
            return fn(repo)
        from backend.repositories import MeetingRepository

        with session_factory() as session:   # closes (and returns its connection) on exit
            return fn(MeetingRepository(session))

    cur = _read(lambda r: build_snapshot(r, meeting_id))
    if cur is None:
        return {
            "meeting_id": meeting_id,
            "total_similar_meetings": 0,
            "average_similarity_score": 0.0,
            "duplicate_detected": False,
            "meetings": [],
        }

    # ``meeting_id`` may be any identifier the API accepts (numeric id, display id, bharatvc id). The vector store
    # only knows primary keys, so query with the RESOLVED meeting and never list it as its own match.
    candidates = vector_service.get_semantic_candidates(
        cur.id, cur.transcript, top_k=candidate_pool
    )

    ordered: List[Dict[str, Any]] = []
    seen = {cur.id}
    for cand in candidates:
        cid = cand["meeting_id"]
        if cid in seen:
            continue
        seen.add(cid)
        ordered.append(cand)
    others = _read(lambda r: build_snapshots(r, [c["meeting_id"] for c in ordered])) if ordered else {}

    analyzed: List[Dict[str, Any]] = []
    for cand in ordered:
        other = others.get(int(cand["meeting_id"]))
        if other is None:
            continue
        res = analyze_pair(
            cur,
            other,
            cand["semantic_score"],
            min_score,
            include_details=include_details,
        )
        if res["category"] == Category.NOT_RELATED:
            continue
        analyzed.append(res)

    # Rank: duplicates first, then by similarity, then confidence.
    _rank_order = {
        Category.EXACT_DUPLICATE: 0,
        Category.NEAR_DUPLICATE: 1,
        Category.FOLLOW_UP: 2,
        Category.RECURRING: 3,
        Category.RELATED: 4,
    }
    analyzed.sort(
        key=lambda x: (
            _rank_order.get(x["category"], 9),
            -x["similarity_score"],
            -x["confidence_score"],
        )
    )
    analyzed = analyzed[:top_k]
    for i, r in enumerate(analyzed):
        r["rank"] = i + 1

    duplicate_detected = any(r["is_duplicate"] for r in analyzed)
    avg = (
        round(sum(r["similarity_score"] for r in analyzed) / len(analyzed), 4)
        if analyzed
        else 0.0
    )

    return {
        "meeting_id": meeting_id,
        "meeting_code": cur.code,
        "total_similar_meetings": len(analyzed),
        "average_similarity_score": avg,
        "duplicate_detected": duplicate_detected,
        "category_counts": _category_counts(analyzed),
        "meetings": analyzed,
    }


def _category_counts(analyzed: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in analyzed:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    return counts


def detect_duplicate(repo, meeting_id: Any) -> Dict[str, Any]:
    """Fast check: is this meeting a duplicate of something already ingested?"""
    cur = build_snapshot(repo, meeting_id)
    if cur is None:
        return {"is_duplicate": False, "duplicates": []}

    candidates = vector_service.get_semantic_candidates(
        cur.id, cur.transcript, top_k=15
    )
    duplicates: List[Dict[str, Any]] = []
    for cand in candidates:
        if cand["meeting_id"] == cur.id:
            continue
        other = build_snapshot(repo, cand["meeting_id"])
        if other is None:
            continue
        sig = _compute_signals(cur, other, cand["semantic_score"])
        category = _classify(cur, other, sig, min_score=0.20)
        if category in (Category.EXACT_DUPLICATE, Category.NEAR_DUPLICATE):
            info = _duplicate_info(other)
            info.update(
                {
                    "category": category,
                    "category_label": CATEGORY_META[category]["label"],
                    "similarity_percentage": f"{max(sig.get('content_jaccard', sig['content']), sig['semantic']) * 100:.1f}%",
                    "confidence_score": _confidence(category, sig),
                    "reason": _build_explanation(
                        category, cur, other, sig, _blended_score(sig)
                    ),
                }
            )
            duplicates.append(info)

    return {
        "meeting_id": meeting_id,
        "is_duplicate": bool(duplicates),
        "message": (
            "This meeting already exists." if duplicates else "No duplicate detected."
        ),
        "duplicates": duplicates,
    }

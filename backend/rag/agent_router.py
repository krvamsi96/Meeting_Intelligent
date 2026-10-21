"""Agentic RAG Router for AI Meeting Intelligence Chatbot.

Implements an intelligent, meeting-aware Agentic RAG architecture:
  1. Intent & Entity Extraction (no LLM overhead, fast pattern & entity matching)
  2. Multi-Tool Selection & Routing (Structured DB, Speaker Analytics, Temporal Search, Vector Search, Dynamic MOM)
  3. Hybrid Retrieval & Cross-Encoder Reranking
  4. Context Assembly & Deduplication with Strict Meeting Isolation
  5. Grounding & Self-Evaluation Verification

Tools:
  1. MetadataTool        — Meeting title, date, duration, participants, project
  2. StructuredDBTool    — Relational DB (action_items, decisions, agenda, risks, sentiment/metrics)
  3. SpeakerAnalysisTool — Speaker talk time, duration calculation, turns, and speaker-filtered utterances
  4. TemporalSearchTool  — Timestamp-aware transcript segment retrieval
  5. DynamicMOMTool      — Structured Minutes of Meeting assembly from all meeting entities
  6. VectorSearchTool    — ChromaDB semantic search + Cross-Encoder reranking
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import text as sa_text

from backend.rag.reranker import get_reranker
from backend.utils.helpers import clean_agenda_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent Classification Patterns (Comprehensive & Dynamic)
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: List[Tuple[str, List[str]]] = [
    ("mom", [
        r"\bmom\b", r"minutes of meeting", r"\bminutes\b",
        r"meeting notes", r"formal record", r"meeting summary and action",
    ]),
    ("speaker_duration", [
        r"who spoke the most", r"who talked the most", r"speaking time",
        r"talk time", r"how long did .* speak", r"duration of speaker",
        r"speaker duration", r"speaker percentage", r"talk duration",
        r"how much did .* talk", r"most active speaker", r"speaking breakdown",
    ]),
    ("action_items", [
        r"action item", r"task", r"\btodo\b", r"follow.?up", r"assignee",
        r"who.*(responsible|assigned|handle|take care|owner)", r"what.*need.*do",
        r"deliverable", r"pending", r"to.?do", r"who is responsible",
        r"who owns", r"deadline", r"due date", r"what needs to be done",
        r"who needs to do what", r"what are the tasks", r"next steps",
    ]),
    ("decisions", [
        r"decision", r"decided", r"\bagreed?\b", r"conclusion", r"resolved",
        r"approved", r"finalized?", r"outcome", r"chosen", r"selected",
        r"what.*decide", r"final.*call", r"consensus", r"what did they decide",
        r"what was decided", r"any decision",
    ]),
    ("risks", [
        r"risk", r"blocker", r"\bissue\b", r"concern", r"problem",
        r"challenge", r"obstacle", r"dependency", r"bottleneck",
        r"complication", r"impediment",
    ]),
    ("sentiment_metrics", [
        r"sentiment", r"emotion", r"mood", r"tone", r"feeling",
        r"questions? (raised|asked)", r"keywords?", r"analytics",
        r"metric", r"positive or negative",
    ]),
    ("temporal", [
        r"\bwhen\b", r"timestamp", r"at what time", r"what time did",
        r"when was .* discussed", r"when did .* mention", r"at \d+:\d+",
        r"what point in time", r"time of discussion",
    ]),
    ("speaker", [
        r"what did (speaker|user|[a-zA-Z0-9_]+) say", r"said by",
        r"according to", r"words of", r"stated by", r"mentioned by",
        r"speaker \d+", r"speaker_[a-zA-Z0-9]+", r"what did .* state",
    ]),
    ("agenda", [
        r"agenda", r"topic", r"subject", r"what was discussed",
        r"points? discussed", r"what.*cover", r"items? on",
        r"important topics", r"main topics", r"key points",
    ]),
    ("summary", [
        r"summar", r"overview", r"what.*about\s*this\s*meeting",
        r"what happened in the meeting", r"brief", r"recap", r"gist",
        r"essence", r"tell me everything important", r"overview of the meeting",
    ]),
    ("participants", [
        r"participant", r"attendee", r"who.*attend", r"who.*present",
        r"who.*there", r"who.*join", r"member", r"team", r"who was in the meeting",
        r"who attended", r"who was present", r"list of people",
    ]),
    ("yes_no", [
        r"\bwas there any\b", r"\bdid they (discuss|talk|mention|decide|agree|cover)\b",
        r"\bis there any\b", r"\bdid anyone\b", r"\bwere there any\b",
        r"\bwas .* (discussed|mentioned|decided|brought up)\b",
        r"\bdid .* happen\b", r"\bhas .* been\b",
    ]),
    ("follow_up", [
        r"\b(tell me more|elaborate|expand on|what about|and what|go on|continue)\b",
        r"\b(the (first|second|third|last) one)\b",
        r"\b(that item|this topic|the same)\b",
        r"\b(when was that|who did that|how about that)\b",
    ]),
    ("general", []),
]


def classify_intent(query: str) -> str:
    """Classify query intent using regex patterns."""
    q = query.lower()
    for intent, patterns in _INTENT_PATTERNS:
        if not patterns:
            continue
        if any(re.search(p, q) for p in patterns):
            return intent
    return "general"


def classify_multi_intent(query: str) -> List[str]:
    """Classify all matching intents for compound queries."""
    q = query.lower()
    matched = []
    for intent, patterns in _INTENT_PATTERNS:
        if not patterns:
            continue
        if any(re.search(p, q) for p in patterns):
            matched.append(intent)
    return matched if matched else ["general"]


def extract_speaker_from_query(query: str) -> Optional[str]:
    """Extract referenced speaker name or label from query."""
    q = query.lower()
    # Match "Speaker 1", "Speaker_01", "Speaker A", "Speaker Rajesh"
    m = re.search(r"speaker\s*([0-9a-zA-Z_]+)", q)
    if m:
        return f"Speaker {m.group(1).replace('_', ' ').strip()}"
    return None


def _fmt_seconds_to_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    if seconds is None or seconds < 0:
        return "00:00"
    total_sec = int(seconds)
    hours, remainder = divmod(total_sec, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _parse_time_str(ts_str: Any) -> float:
    """Convert timestamp string 'HH:MM:SS' or 'MM:SS' or float string to seconds."""
    if ts_str is None:
        return 0.0
    s = str(ts_str).strip()
    try:
        if ":" in s:
            parts = [float(p) for p in s.split(":")]
            if len(parts) == 3:
                return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
            elif len(parts) == 2:
                return parts[0] * 60.0 + parts[1]
        return float(s)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Tool Result Container
# ---------------------------------------------------------------------------

class ToolResult:
    """Container for data returned by a specialized retrieval tool."""

    def __init__(
        self,
        tool_name: str,
        chunks: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        intent: str = "general",
        meeting_id: Any = None,
    ):
        self.tool_name = tool_name
        self.chunks = chunks
        self.metadata = metadata or []
        self.intent = intent
        self.meeting_id = meeting_id


# ---------------------------------------------------------------------------
# Specialized Tools
# ---------------------------------------------------------------------------

class MetadataTool:
    """Retrieves meeting metadata (title, date, duration, participants, project)."""

    def __init__(self, repo):
        self.repo = repo

    def run(self, meeting_id: Any) -> ToolResult:
        chunks = []
        try:
            meeting = self.repo.get_by_id(meeting_id)
            if not meeting:
                return ToolResult("metadata", [])

            duration_str = _fmt_seconds_to_time(float(meeting.duration or 0.0))
            parts = [
                "=== MEETING METADATA ===",
                f"Title: {meeting.title or meeting.filename}",
                f"Meeting Code / ID: {meeting.meeting_id or meeting.bharatvc_meeting_id or meeting.id}",
                f"Date: {meeting.meeting_date or 'N/A'}",
                f"Time: {meeting.meeting_time or 'N/A'}",
                f"Duration: {duration_str}",
                f"Participants: {meeting.participants or 'N/A'}",
                f"Project: {meeting.project_name or 'N/A'}",
                f"Total Speakers: {meeting.speaker_count or 'N/A'}",
            ]
            chunks.append("\n".join(parts))
        except Exception as e:
            logger.warning(f"MetadataTool failed for meeting {meeting_id}: {e}")

        return ToolResult("metadata", chunks, intent="metadata")


class StructuredDBTool:
    """Retrieves structured database records (ActionItem, AgendaKeyDecision, AnalysisResult, Metrics)."""

    def __init__(self, repo):
        self.repo = repo

    def run(self, meeting_id: Any, intent: str, query: str = "") -> ToolResult:
        chunks = []
        try:
            meeting = self.repo.get_by_id(meeting_id)
            if not meeting:
                return ToolResult("structured_db", [], intent=intent)

            analysis = self.repo.get_analysis(meeting.id)

            # ── 1. Action Items ──────────────────────────────────────────────
            if intent in ("action_items", "general", "mom"):
                from backend.models.models import ActionItem
                items = (
                    self.repo.db.query(ActionItem)
                    .filter(ActionItem.meeting_id == meeting.id)
                    .order_by(ActionItem.item_order)
                    .all()
                )
                if items:
                    rows = []
                    for idx, item in enumerate(items, 1):
                        task_text = item.task or item.description or "Task"
                        owner = item.owner or "Unassigned"
                        deadline = item.deadline or (str(item.due_date) if item.due_date else "Not Specified")
                        prio = f", Priority: {item.priority}" if item.priority and item.priority != "Not specified" else ""
                        status = f", Status: {item.status}" if item.status and item.status != "unknown" else ""
                        evidence = f"\n    [Evidence: \"{item.evidence}\"]" if item.evidence else ""
                        rows.append(f"{idx}. Task: {task_text}\n   Assignee / Owner: {owner}\n   Deadline: {deadline}{prio}{status}{evidence}")
                    chunks.append("[STRUCTURED ACTION ITEMS]\n" + "\n".join(rows))
                elif analysis and analysis.action_items:
                    chunks.append(f"[ACTION ITEMS]\n{analysis.action_items}")

            # ── 2. Key Decisions ─────────────────────────────────────────────
            if intent in ("decisions", "general", "mom"):
                from backend.models.models import AgendaKeyDecision
                dec_rows = (
                    self.repo.db.query(AgendaKeyDecision)
                    .filter(AgendaKeyDecision.meeting_id == meeting.id)
                    .order_by(AgendaKeyDecision.agenda_order)
                    .all()
                )
                if dec_rows:
                    rows = []
                    for idx, d in enumerate(dec_rows, 1):
                        # A missing agenda or decision is reported as missing — never replaced by made-up filler
                        # wording that the chatbot would then repeat as if it had been said.
                        agenda_txt = (d.agenda or "").strip() or "not stated"
                        dec_txt = (d.key_decision or "").strip()
                        if dec_txt.lower().rstrip(".") in ("no formal decision recorded", "no decision recorded", "none"):
                            dec_txt = ""
                        dec_txt = dec_txt or "no decision was made"
                        speaker_txt = f" (by {d.speaker})" if d.speaker else ""
                        ts_txt = f" [at {d.timestamp}]" if d.timestamp else ""
                        reason_txt = f"\n   Rationale: {d.reason}" if d.reason else ""
                        ev_txt = f"\n   Evidence: \"{d.evidence}\"" if d.evidence else ""
                        rows.append(f"{idx}. Agenda: {agenda_txt}\n   Decision: {dec_txt}{speaker_txt}{ts_txt}{reason_txt}{ev_txt}")
                    chunks.append("[STRUCTURED AGENDA & KEY DECISIONS]\n" + "\n".join(rows))
                elif analysis and analysis.decisions:
                    chunks.append(f"[KEY DECISIONS]\n{analysis.decisions}")

            # ── 3. Agenda & Topics ───────────────────────────────────────────
            if intent in ("agenda", "topics"):
                from backend.models.models import AgendaKeyDecision
                topics = (
                    self.repo.db.query(AgendaKeyDecision)
                    .filter(AgendaKeyDecision.meeting_id == meeting.id)
                    .order_by(AgendaKeyDecision.agenda_order)
                    .all()
                )
                if topics:
                    top_list = [f"- {t.agenda} ({t.agenda_type or 'detected'})" for t in topics if t.agenda]
                    if top_list:
                        chunks.append("[MEETING AGENDA TOPICS]\n" + "\n".join(top_list))
                _agenda_text = clean_agenda_text(meeting.agenda)      # a generated placeholder is not an agenda
                if _agenda_text:
                    chunks.append(f"[MEETING AGENDA]\n{_agenda_text}")

            # ── 4. Risks & Blockers ──────────────────────────────────────────
            if intent == "risks":
                if analysis and analysis.risks:
                    chunks.append(f"[RISKS & BLOCKERS]\n{analysis.risks}")

            # ── 5. Sentiment & Metrics ───────────────────────────────────────
            if intent == "sentiment_metrics":
                metrics_parts = ["[MEETING SENTIMENT & ANALYTICS]"]
                if meeting.sentiment:
                    metrics_parts.append(f"Overall Sentiment: {meeting.sentiment}")
                if meeting.emotions:
                    metrics_parts.append(f"Emotion Breakdown: {meeting.emotions}")
                if meeting.questions_raised:
                    metrics_parts.append(f"Questions Raised: {meeting.questions_raised}")
                if meeting.keyword_frequency:
                    metrics_parts.append(f"Keywords: {meeting.keyword_frequency}")
                if len(metrics_parts) > 1:
                    chunks.append("\n".join(metrics_parts))

            # ── 6. Summary & MoM ─────────────────────────────────────────────
            if intent == "summary" and analysis and analysis.summary:
                chunks.append(f"[EXECUTIVE SUMMARY]\n{analysis.summary}")
            elif intent == "mom" and analysis and analysis.mom:
                chunks.append(f"[MINUTES OF MEETING]\n{analysis.mom}")

        except Exception as e:
            logger.warning(f"StructuredDBTool error: {e}", exc_info=True)
            try:
                self.repo.db.rollback()
            except Exception:
                pass

        return ToolResult("structured_db", chunks, intent=intent)


class SpeakerAnalysisTool:
    """Analyzes speaker statistics (talk-time, duration, percentage) and speaker utterances."""

    def __init__(self, repo):
        self.repo = repo

    def run(self, meeting_id: Any, speaker_query: Optional[str] = None) -> ToolResult:
        chunks = []
        try:
            meeting = self.repo.get_by_id(meeting_id)
            if not meeting:
                return ToolResult("speaker_analysis", [])

            segments = self.repo.get_segments(meeting.id)
            if not segments:
                return ToolResult("speaker_analysis", [])

            # Compute talk-time statistics per speaker
            speaker_durations: Dict[str, float] = {}
            speaker_turns: Dict[str, int] = {}
            total_duration = 0.0

            for seg in segments:
                spk = seg.speaker or "Unknown Speaker"
                start = _parse_time_str(seg.start_time)
                end = _parse_time_str(seg.end_time)
                dur = max(0.0, end - start)
                speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur
                speaker_turns[spk] = speaker_turns.get(spk, 0) + 1
                total_duration += dur

            if total_duration == 0.0:
                total_duration = float(meeting.duration or 1.0)

            # Sort speakers by talk time descending
            sorted_speakers = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)

            stats_lines = ["[SPEAKER PARTICIPATION & TALK-TIME STATISTICS]"]
            for spk, dur in sorted_speakers:
                pct = (dur / total_duration * 100.0) if total_duration > 0 else 0.0
                time_formatted = _fmt_seconds_to_time(dur)
                turns = speaker_turns.get(spk, 0)
                stats_lines.append(f"- {spk}: {time_formatted} ({pct:.1f}% of speaking time, {turns} turns)")

            if sorted_speakers:
                top_spk, top_dur = sorted_speakers[0]
                stats_lines.append(f"\nMost Active Speaker: {top_spk} ({_fmt_seconds_to_time(top_dur)})")

            chunks.append("\n".join(stats_lines))

            # If a specific speaker was requested, retrieve their statements
            if speaker_query:
                q_clean = speaker_query.lower().replace("speaker", "").strip()
                matched_segs = []
                for s in segments:
                    spk_name = (s.speaker or "").lower()
                    if speaker_query.lower() in spk_name or (q_clean and q_clean in spk_name):
                        matched_segs.append(s)

                if matched_segs:
                    lines = [
                        f"[{s.start_time} - {s.end_time}] {s.speaker}: {s.text}"
                        for s in matched_segs[:25]
                    ]
                    chunks.append(f"[TRANSCRIPT UTTERANCES FOR {speaker_query.upper()}]\n" + "\n".join(lines))

        except Exception as e:
            logger.warning(f"SpeakerAnalysisTool error: {e}", exc_info=True)

        return ToolResult("speaker_analysis", chunks, intent="speaker")


class TemporalSearchTool:
    """Searches transcript segments for time-related queries with timestamps and speaker context."""

    def __init__(self, repo):
        self.repo = repo

    def run(self, meeting_id: Any, query: str) -> ToolResult:
        chunks = []
        try:
            meeting = self.repo.get_by_id(meeting_id)
            if not meeting:
                return ToolResult("temporal_search", [])

            segments = self.repo.get_segments(meeting.id)
            if not segments:
                return ToolResult("temporal_search", [])

            # Extract keywords from query excluding common stop words
            stop_words = {"when", "was", "were", "the", "a", "an", "in", "on", "at", "did", "they", "discuss", "discussed", "talk", "about", "what", "time", "meeting"}
            words = [w for w in re.findall(r"\b\w{3,}\b", query.lower()) if w not in stop_words]

            matched = []
            for seg in segments:
                text_lower = (seg.text or "").lower()
                matches = sum(1 for w in words if w in text_lower)
                if matches > 0:
                    matched.append((matches, seg))

            # Sort by keyword match density
            matched.sort(key=lambda x: x[0], reverse=True)

            if matched:
                lines = []
                for _, s in matched[:10]:
                    ts = s.start_time or "00:00"
                    ts_end = s.end_time or "00:00"
                    lines.append(f"[{ts} - {ts_end}] {s.speaker}: {s.text}")
                chunks.append("[TIMESTAMPS & DISCUSSION SEGMENTS]\n" + "\n".join(lines))

        except Exception as e:
            logger.warning(f"TemporalSearchTool error: {e}")

        return ToolResult("temporal_search", chunks, intent="temporal")


class DynamicMOMTool:
    """Assembles structured Minutes of Meeting (MoM) context dynamically from all meeting entities."""

    def __init__(self, repo):
        self.repo = repo

    def run(self, meeting_id: Any) -> ToolResult:
        chunks = []
        try:
            meeting = self.repo.get_by_id(meeting_id)
            if not meeting:
                return ToolResult("dynamic_mom", [])

            analysis = self.repo.get_analysis(meeting.id)
            from backend.models.models import ActionItem, AgendaKeyDecision

            mom_sections = [f"=== MINUTES OF MEETING (MoM): {meeting.title or meeting.filename} ==="]
            mom_sections.append(
                f"Date: {meeting.meeting_date or 'N/A'} | Duration: {_fmt_seconds_to_time(float(meeting.duration or 0))} | Participants: {meeting.participants or 'N/A'}"
            )

            # 1. Summary
            if analysis and analysis.summary:
                mom_sections.append(f"\n[EXECUTIVE SUMMARY]\n{analysis.summary}")

            # 2. Agenda & Decisions
            dec_rows = (
                self.repo.db.query(AgendaKeyDecision)
                .filter(AgendaKeyDecision.meeting_id == meeting.id)
                .order_by(AgendaKeyDecision.agenda_order)
                .all()
            )
            if dec_rows:
                dec_parts = []
                for idx, d in enumerate(dec_rows, 1):
                    spk_info = f" (by {d.speaker})" if d.speaker else ""
                    ts_info = f" [{d.timestamp}]" if d.timestamp else ""
                    dec_parts.append(f"{idx}. {d.agenda} -> Decision: {d.key_decision}{spk_info}{ts_info}")
                mom_sections.append("\n[AGENDA & KEY DECISIONS]\n" + "\n".join(dec_parts))
            elif analysis and analysis.decisions:
                mom_sections.append(f"\n[KEY DECISIONS]\n{analysis.decisions}")

            # 3. Action Items
            act_rows = (
                self.repo.db.query(ActionItem)
                .filter(ActionItem.meeting_id == meeting.id)
                .order_by(ActionItem.item_order)
                .all()
            )
            if act_rows:
                act_parts = []
                for idx, a in enumerate(act_rows, 1):
                    owner = a.owner or "Unassigned"
                    deadline = a.deadline or (str(a.due_date) if a.due_date else "Not Specified")
                    act_parts.append(f"{idx}. {a.task} | Assignee: {owner} | Deadline: {deadline} | Priority: {a.priority or 'Normal'}")
                mom_sections.append("\n[ACTION ITEMS]\n" + "\n".join(act_parts))
            elif analysis and analysis.action_items:
                mom_sections.append(f"\n[ACTION ITEMS]\n{analysis.action_items}")

            chunks.append("\n".join(mom_sections))
        except Exception as e:
            logger.warning(f"DynamicMOMTool error: {e}", exc_info=True)

        return ToolResult("dynamic_mom", chunks, intent="mom")


# ---------------------------------------------------------------------------
# Context Deduplication
# ---------------------------------------------------------------------------

def deduplicate_context(chunks: List[str], max_total_chars: int = 15000) -> List[str]:
    """Remove redundant chunks and trim within max character budget."""
    seen_signatures: Set[str] = set()
    deduped = []
    total_chars = 0

    for chunk in chunks:
        if not chunk or not chunk.strip():
            continue

        # Signature based on alphanumeric words of 4+ chars
        words = tuple(sorted(set(re.findall(r"\b\w{4,}\b", chunk.lower()))))
        sig = " ".join(words[:20])

        if sig and sig in seen_signatures:
            continue

        remaining = max_total_chars - total_chars
        if remaining <= 0:
            break

        if len(chunk) > remaining:
            chunk = chunk[:remaining] + "\n...[context truncated for length]"

        deduped.append(chunk)
        if sig:
            seen_signatures.add(sig)
        total_chars += len(chunk)

    return deduped


# ---------------------------------------------------------------------------
# Main Agent Router
# ---------------------------------------------------------------------------

class MeetingAgentRouter:
    """Agentic RAG Router with Multi-Intent Classification, Dynamic Tool Routing,
    Cross-Encoder Reranking, and Self-Evaluation Grounding."""

    def __init__(self, repo, vector_service):
        self.repo = repo
        self.vector_service = vector_service
        self._metadata_tool = MetadataTool(repo)
        self._structured_tool = StructuredDBTool(repo)
        self._speaker_tool = SpeakerAnalysisTool(repo)
        self._temporal_tool = TemporalSearchTool(repo)
        self._mom_tool = DynamicMOMTool(repo)
        self._reranker = get_reranker()

    def route(
        self,
        meeting_id: Any,
        query: str,
        top_k: int = 8,
        similarity_threshold: float = 0.25,
        cross_meeting: bool = False,
    ) -> Dict[str, Any]:
        """Main dynamic routing and retrieval entry point.

        Meeting isolation: retrieval is ALWAYS restricted to ``meeting_id``. ``cross_meeting`` is kept in
        the signature for backward compatibility but is ignored - Meeting A must never use Meeting B's
        context. Every vector chunk is additionally checked against the current meeting's primary key
        before it can become context (defence in depth on top of the SQL ``WHERE meeting_id`` filter).
        """
        t0 = time.perf_counter()
        resolved = self.repo.get_by_id(meeting_id)
        meeting_pk = resolved.id if resolved is not None else meeting_id

        # 1. Classify intents & extract query entities
        intents = classify_multi_intent(query)
        primary_intent = intents[0] if intents else "general"
        speaker_entity = extract_speaker_from_query(query)

        tools_used = []
        all_chunks: List[str] = []
        source_chunks: List[dict] = []

        logger.info(
            f"[AgentRouter] query={query!r:.80} primary_intent={primary_intent} "
            f"intents={intents} speaker_entity={speaker_entity}"
        )

        # 2. Metadata Tool (Always provides meeting foundation)
        meta_res = self._metadata_tool.run(meeting_id)
        if meta_res.chunks:
            all_chunks.extend(meta_res.chunks)
            tools_used.append("metadata")

        # 3. Dynamic MoM Tool (Activated on MOM requests)
        if "mom" in intents:
            mom_res = self._mom_tool.run(meeting_id)
            if mom_res.chunks:
                all_chunks.extend(mom_res.chunks)
                tools_used.append("dynamic_mom")

        # 4. Structured DB Tool (Action items, decisions, agenda, risks, metrics, summary)
        structured_intents = {"action_items", "decisions", "agenda", "risks", "sentiment_metrics", "summary"}
        active_struct_intents = set(intents) & structured_intents
        if not active_struct_intents and primary_intent in ("general", "yes_no", "follow_up"):
            active_struct_intents = {"action_items", "decisions", "summary", "agenda"}
        elif primary_intent == "yes_no":
            active_struct_intents.update({"action_items", "decisions", "agenda"})

        for s_intent in active_struct_intents:
            db_res = self._structured_tool.run(meeting_id, s_intent, query=query)
            if db_res.chunks:
                all_chunks.extend(db_res.chunks)
                if "structured_db" not in tools_used:
                    tools_used.append("structured_db")

        # 5. Speaker Analysis Tool (Talk-time, who spoke most, participants, or specific speaker statement)
        if "speaker_duration" in intents or "speaker" in intents or "participants" in intents or speaker_entity:
            spk_res = self._speaker_tool.run(meeting_id, speaker_query=speaker_entity)
            if spk_res.chunks:
                all_chunks.extend(spk_res.chunks)
                tools_used.append("speaker_analysis")

        # 6. Temporal Search Tool (When, timestamps, time range queries)
        if "temporal" in intents:
            temp_res = self._temporal_tool.run(meeting_id, query)
            if temp_res.chunks:
                all_chunks.extend(temp_res.chunks)
                tools_used.append("temporal_search")

        # 7. Semantic Vector Search & Cross-Encoder Reranking
        try:
            # Query vector search with broader candidate pool
            candidate_k = max(top_k * 3, 12)
            vector_candidates = self.vector_service.search_transcript_chunks(
                meeting_id=meeting_pk,
                query_text=query,
                top_k=candidate_k,
                cross_meeting=False,
            )

            # Isolation guard: drop any chunk tagged with a different meeting's id.
            own_candidates = [
                c for c in vector_candidates
                if c.get("meeting_id") is None or str(c.get("meeting_id")) == str(meeting_pk)
            ]
            if len(own_candidates) != len(vector_candidates):
                logger.warning(
                    "[AgentRouter] Dropped %d retrieved chunk(s) that belonged to a different meeting "
                    "(current meeting %s).", len(vector_candidates) - len(own_candidates), meeting_pk,
                )

            # Filter candidates above minimum threshold
            valid_candidates = [
                c for c in own_candidates
                if c.get("similarity_score", 0.0) >= similarity_threshold
            ]

            # If candidates were found, rerank them with CrossEncoder
            if valid_candidates:
                reranked = self._reranker.rerank(query=query, chunks=valid_candidates, top_k=top_k)
                for chunk in reranked:
                    all_chunks.append(chunk["text"])
                    source_chunks.append(chunk)
                tools_used.append("vector_search+rerank")
                logger.info(
                    f"[AgentRouter] Vector: {len(valid_candidates)} candidates -> {len(reranked)} reranked top chunks"
                )

        except Exception as vec_err:
            logger.error(f"[AgentRouter] Vector search / reranker failed: {vec_err}", exc_info=True)

        # 8. Fallback / Secondary Retrieval if context is still sparse
        if len(all_chunks) <= 1:
            logger.info("[AgentRouter] Context is sparse, executing fallback transcript extraction...")
            try:
                segments = self.repo.get_segments(meeting_id)
                if segments:
                    n = len(segments)
                    # Distributed sampling: start, middle, and end
                    indices = list(range(0, min(6, n)))
                    if n > 12:
                        mid_start = max(0, n // 2 - 3)
                        indices.extend(range(mid_start, min(mid_start + 6, n)))
                    if n > 6:
                        indices.extend(range(max(0, n - 6), n))
                    
                    seen_idx = set()
                    sample_segs = []
                    for idx in sorted(indices):
                        if idx not in seen_idx and idx < n:
                            seen_idx.add(idx)
                            sample_segs.append(segments[idx])

                    sample_text = "\n".join([f"[{s.start_time}] {s.speaker}: {s.text}" for s in sample_segs])
                    all_chunks.append(f"[TRANSCRIPT OVERVIEW]\n{sample_text}")
                    tools_used.append("transcript_fallback")
            except Exception as fb_err:
                logger.warning(f"[AgentRouter] Fallback transcript extraction failed: {fb_err}")

        # 9. Deduplicate and budget context length
        deduped_context = deduplicate_context(all_chunks, max_total_chars=15000)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[AgentRouter] Completed in {elapsed_ms:.1f}ms - tools={tools_used} "
            f"chunks_count={len(deduped_context)}"
        )

        return {
            "context_chunks": deduped_context,
            "source_chunks": source_chunks,
            "intents": intents,
            "primary_intent": primary_intent,
            "tools_used": tools_used,
            "elapsed_ms": elapsed_ms,
        }

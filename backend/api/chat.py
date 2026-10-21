"""Chat API endpoints - production-ready chatbot with advanced RAG, lazy indexing, and conversation memory."""

import asyncio
import json
import logging
import re
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.api.deps import get_meeting_repository
from backend.database.connection import SessionLocal
from backend.repositories import MeetingRepository
from backend.schemas import schemas
from backend.services.llm_workflow import (
    _make_sys_prompt,
    analyze_query_intent,
    escape_template_braces,
    generate_chatbot_response,
    get_llm,
    rewrite_query_with_history,
)
from backend.prompts.meeting_prompts import (
    chatbot_system_prompt,
    get_intent_guidance,
)
from backend.services.quality_metrics_service import evaluate_rag_answer, store_rag_evaluation
from backend.services.vector_service import vector_service
from backend.utils.rate_limit import chat_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/meetings", tags=["Chat"])

# Conversation memory cache: {session_id: (deque, last_access_ts)}
_conversation_memory: dict = {}
_conversation_memory_mutex = threading.Lock()
_MEMORY_MAX_LENGTH = 20
_MEMORY_MAX_SESSIONS = 1000       # hard cap on concurrent sessions
_MEMORY_SESSION_TTL_SEC = 3600    # evict sessions inactive for >1 hour

# Thread-safe lazy indexing locks
_indexing_locks = {}
_indexing_lock_mutex = threading.Lock()


def get_indexing_lock(meeting_id: int) -> threading.Lock:
    """Thread-safe access to a lock per meeting to prevent race conditions during lazy indexing."""
    with _indexing_lock_mutex:
        if meeting_id not in _indexing_locks:
            _indexing_locks[meeting_id] = threading.Lock()
        return _indexing_locks[meeting_id]


def _evict_stale_sessions() -> None:
    """Evict sessions older than TTL or when the session cap is exceeded. Call under lock."""
    import time
    now = time.time()
    # Remove expired sessions
    expired = [
        sid for sid, (_, ts) in _conversation_memory.items()
        if now - ts > _MEMORY_SESSION_TTL_SEC
    ]
    for sid in expired:
        del _conversation_memory[sid]
    # If still over cap, evict oldest accessed sessions
    if len(_conversation_memory) >= _MEMORY_MAX_SESSIONS:
        sorted_sessions = sorted(_conversation_memory.items(), key=lambda x: x[1][1])
        evict_count = len(_conversation_memory) - _MEMORY_MAX_SESSIONS + 1
        for sid, _ in sorted_sessions[:evict_count]:
            del _conversation_memory[sid]


def enforce_meeting_isolation(request: "schemas.ChatMessageCreate", meeting_pk: int) -> None:
    """A chat answer may only ever use THIS meeting's context.

    ``cross_meeting`` is still accepted in the request body (so existing clients do not break) but it
    can no longer widen retrieval to other meetings' chunks: Meeting A must never use Meeting B's RAG
    context. Retrieval is always filtered by this meeting's id.
    """
    if getattr(request, "cross_meeting", False):
        logger.warning(
            "[Chat] cross_meeting=true ignored for meeting %s - retrieval is always restricted to "
            "the current meeting.", meeting_pk,
        )
        request.cross_meeting = False


# Helpers for hybrid retrieval & DB updates
def save_chat_to_db_sync(
    meeting_id: int,
    session_id: str,
    user_msg: str,
    assistant_msg: str,
    sources_json: str,
    rag_evaluation: Optional[dict] = None,
):
    db = SessionLocal()
    try:
        repo = MeetingRepository(db)
        repo.add_chat_message(
            session_id=session_id,
            meeting_id=meeting_id,
            role="user",
            content=user_msg,
        )
        saved = repo.add_chat_message(
            session_id=session_id,
            meeting_id=meeting_id,
            role="assistant",
            content=assistant_msg,
            sources=sources_json if sources_json else None,
        )
        if saved is not None and rag_evaluation:
            store_rag_evaluation(db, meeting_pk=meeting_id, chat_message_id=saved.id, evaluation=rag_evaluation)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to save streamed chat message to DB: {e}")
        db.rollback()
    finally:
        db.close()


def parse_time_filters(query: str) -> Optional[tuple[float, float]]:
    """
    Parses start and end seconds from query text.
    Returns (start_sec, end_sec) or None if no time constraint is found.
    """
    query_lower = query.lower()

    # 1. HH:MM:SS format e.g. "01:12:35" or "at 1:05:20"
    match = re.search(r"\b(\d{1,2}):(\d{2}):(\d{2})\b", query_lower)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        target_sec = hours * 3600.0 + minutes * 60.0 + seconds
        return max(0.0, target_sec - 45.0), target_sec + 45.0

    # 2. MM:SS format e.g. "at 12:35" or "around 05:10"
    match = re.search(r"\b(\d{1,2}):(\d{2})\b", query_lower)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        target_sec = minutes * 60.0 + seconds
        return max(0.0, target_sec - 30.0), target_sec + 30.0

    # 3. "between X and Y minutes" / "X to Y minutes"
    match = re.search(
        r"(?:between|from)\s*(\d+(?:\.\d+)?)\s*(?:and|to)\s*(\d+(?:\.\d+)?)\s*(?:minutes|mins|min|m)",
        query_lower,
    )
    if match:
        return float(match.group(1)) * 60.0, float(match.group(2)) * 60.0

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:minutes|mins|min)\s*to\s*(\d+(?:\.\d+)?)", query_lower
    )
    if match:
        return float(match.group(1)) * 60.0, float(match.group(2)) * 60.0

    # 4. "first X minutes"
    match = re.search(r"first\s*(\d+(?:\.\d+)?)\s*(?:minutes|mins|min)", query_lower)
    if match:
        return 0.0, float(match.group(1)) * 60.0

    # 5. "after X minutes" / "after X mins"
    match = re.search(r"after\s*(\d+(?:\.\d+)?)\s*(?:minutes|mins|min|m)", query_lower)
    if match:
        return float(match.group(1)) * 60.0, 999999.0

    # 6. "at minute X" or "around X minutes"
    match = re.search(
        r"(?:at|around|near)\s*(?:minute|min)?\s*(\d+(?:\.\d+)?)\s*(?:minute|min|mins|minutes)?",
        query_lower,
    )
    if match and ("minute" in query_lower or "min" in query_lower):
        mins = float(match.group(1))
        target_sec = mins * 60.0
        return max(0.0, target_sec - 60.0), target_sec + 60.0

    return None


def parse_speaker_filters(query: str, participants_str: str) -> List[str]:
    """
    Finds referenced speakers or participants in the query.
    Returns a list of matching speaker labels or names.
    """
    query_lower = query.lower()
    matches = []

    # 1. Match "speaker X" or "speaker_X" or "speaker A/B"
    spk_matches = re.findall(r"speaker\s*([0-9a-zA-Z]+)", query_lower)
    for spk in spk_matches:
        matches.append(f"SPEAKER_{spk}")
        matches.append(f"Speaker {spk}")
        if spk.isdigit():
            num = int(spk)
            matches.append(f"SPEAKER_{num:02d}")
            matches.append(f"SPEAKER_{num}")
            if num > 0:
                matches.append(f"SPEAKER_{(num-1):02d}")

    # 2. Match participant names
    if participants_str:
        participants = [p.strip() for p in participants_str.split(",") if p.strip()]
        for name in participants:
            if name.lower() in query_lower:
                matches.append(name)

    return list(set(matches))


def format_seconds_to_duration(seconds: float) -> str:
    """Helper to format seconds into MM:SS format."""
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


@router.post(
    "/{meeting_id}/chat",
    response_model=schemas.ChatMessageResponse,
    dependencies=[Depends(chat_rate_limit)],
)
def post_chat_message(
    meeting_id: Any,
    request: schemas.ChatMessageCreate,
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """Sends a message to the chatbot with advanced RAG retrieval, lazy indexing, and context awareness."""
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    # Always use the resolved integer PK for retrieval/isolation. The path param
    # may be a bharatvc_meeting_id / display_id (repo.get_by_id resolves several
    # forms); int()-parsing it would produce a number that is NOT this meeting's
    # PK and could match a *different* meeting's vector chunks (cross-meeting leak).
    meeting_id = meeting.id
    enforce_meeting_isolation(request, meeting_id)
    from backend.utils.security import validate_chat_message

    request.message = validate_chat_message(request.message)
    session_id = request.session_id or meeting.meeting_id or str(meeting.id)

    # 1. Thread-safe Lazy Indexing Check: Index segments on the fly if vector DB is missing them
    lock = get_indexing_lock(meeting.id)
    with lock:
        try:
            if not vector_service.has_transcript_chunks(meeting.id):
                logger.info(
                    f"Lazy indexing: Speech segments for meeting {meeting.meeting_id} not found in vector DB. Re-indexing..."
                )
                segments = repo.get_segments(meeting.id)
                if segments:
                    vector_service.add_transcript_chunks(meeting.id, segments)
                    logger.info(
                        f"Lazy indexing completed: Indexed {len(segments)} segments for meeting {meeting.meeting_id}"
                    )
                else:
                    logger.warning(
                        f"Lazy indexing: No speech segments found in database for meeting {meeting.meeting_id}"
                    )
        except Exception as e:
            logger.error(
                f"Lazy indexing failed for meeting {meeting.meeting_id}: {e}", exc_info=True
            )
            # Roll back any aborted DB transaction so subsequent queries in this
            # request are not blocked by InFailedSqlTransaction.
            try:
                repo.db.rollback()
            except Exception:
                pass

    # Get or build chat history from repo
    try:
        history_records = repo.get_chat_history(meeting.id, session_id)
    except Exception as hist_err:
        logger.warning(f"[Chat] Failed to load chat history: {hist_err}")
        try:
            repo.db.rollback()
        except Exception:
            pass
        history_records = []

    # Convert to format for LLM
    chat_history = [
        {"role": record.role, "content": record.content} for record in history_records
    ]

    # Update conversation memory
    memory = get_conversation_memory(meeting.id, session_id)
    memory.append({"role": "user", "content": request.message})

    # Analyze query intent for better context retrieval
    query_intent = "general"
    try:
        query_intent = analyze_query_intent(request.message)
        logger.debug(f"Query intent detected: {query_intent}")
    except Exception as e:
        logger.warning(f"Failed to analyze query intent: {e}")

    # Rewrite query with history context for better search
    # FIX-C1: Changed > 1 to >= 1 so rewriting activates from turn 2 (not turn 3)
    search_query = request.message
    if len(chat_history) >= 1:
        try:
            search_query = rewrite_query_with_history(request.message, chat_history)
            logger.debug(f"[Chat] Query rewritten: {request.message!r} -> {search_query!r}")
        except Exception as e:
            logger.warning(f"Failed to rephrase follow-up query: {e}")

    # ── Agentic RAG Retrieval ────────────────────────────────────────────────
    # MeetingAgentRouter: intent-driven multi-tool retrieval + deduplication
    context_chunks: list = []
    source_chunks: list = []
    routing_result: dict = {}

    try:
        from backend.rag.agent_router import MeetingAgentRouter
        agent_router = MeetingAgentRouter(repo=repo, vector_service=vector_service)
        routing_result = agent_router.route(
            meeting_id=meeting_id,
            query=search_query,
            top_k=8,
            similarity_threshold=0.25,
            cross_meeting=request.cross_meeting,
        )
        context_chunks = routing_result.get("context_chunks", [])
        source_chunks = routing_result.get("source_chunks", [])
        query_intent = routing_result.get("primary_intent", query_intent)

        logger.info(
            f"[Chat] AgentRouter: intents={routing_result.get('intents')}, "
            f"tools={routing_result.get('tools_used')}, "
            f"context_chunks={len(context_chunks)}, "
            f"elapsed={routing_result.get('elapsed_ms', 0):.0f}ms"
        )
    except Exception as e:
        logger.error(f"[Chat] AgentRouter failed, falling back to legacy retrieval: {e}", exc_info=True)
        # ── Fallback: original retrieval logic ─────────────────────────────
        try:
            top_k = 8 if query_intent in ["summary", "decisions", "action_items", "mom"] else 5
            chunks = vector_service.search_transcript_chunks(
                meeting_id=meeting_id,
                query_text=search_query,
                top_k=top_k,
                cross_meeting=request.cross_meeting,
            )
            for chunk in chunks:
                if chunk.get("similarity_score", 0) >= 0.20:
                    context_chunks.append(chunk["text"])
                    source_chunks.append(chunk)
        except Exception as e2:
            logger.error(f"[Chat] Fallback retrieval also failed: {e2}", exc_info=True)

        # Add static analysis context as fallback
        try:
            analysis = repo.get_analysis(meeting_id)
            if analysis:
                context_parts = []
                if analysis.summary:
                    context_parts.append(f"=== MEETING SUMMARY ===\n{analysis.summary}")
                if analysis.mom:
                    context_parts.append(f"=== MINUTES OF MEETING (MoM) ===\n{analysis.mom}")
                meta_parts = [
                    "=== MEETING METADATA ===",
                    f"Title: {meeting.title or meeting.filename}",
                    f"Date: {meeting.meeting_date or 'N/A'}",
                    f"Duration: {format_seconds_to_duration(float(meeting.duration or 0))}",
                    f"Participants: {meeting.participants or 'N/A'}",
                ]
                context_parts.append("\n".join(meta_parts))
                if analysis.action_items:
                    context_parts.append(f"=== ACTION ITEMS ===\n{analysis.action_items}")
                if analysis.decisions:
                    context_parts.append(f"=== DECISIONS ===\n{analysis.decisions}")
                if analysis.risks:
                    context_parts.append(f"=== RISKS ===\n{analysis.risks}")
                static_ctx = "\n\n".join(context_parts)
                if static_ctx.strip():
                    context_chunks.insert(0, static_ctx)
        except Exception as e3:
            logger.warning(f"[Chat] Fallback static context also failed: {e3}")

    # Generate LLM response with enhanced context
    try:
        response_content = generate_chatbot_response(
            query=request.message,
            context_chunks=context_chunks,
            chat_history=chat_history,
            meeting_title=meeting.title or meeting.filename,
            query_intent=query_intent,
        )
    except Exception as e:
        # A failed generation is an ERROR, not an answer: it used to be returned with HTTP 200, stored in the chat
        # history as the assistant's reply and scored for groundedness like a real one.
        logger.error(f"LLM response generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unavailable, so this question could not be answered. "
                   "Please try again.",
        ) from e

    # ── English-only guarantee ────────────────────────────────────────────────
    # The system prompt already mandates English and the retrieved context is
    # English, but if a model ever emits non-Latin-script text, normalise it.
    try:
        from backend.services.translation_service import looks_non_english, translate_text_to_english
        if looks_non_english(response_content):
            logger.info("[Chat] Non-English answer detected — translating to English.")
            response_content = translate_text_to_english(response_content)
    except Exception:
        pass

    # ── Self-Evaluation + Retry Loop ────────────────────────────────────────
    # If the initial answer is poorly grounded, retry once with a broader query.
    rag_quality_warning: Optional[str] = None
    retry_attempted = False

    try:
        from backend.loop_engine.quality_engine import quality_engine as _qe
        _rag_gate = _qe.validate_rag_chatbot(
            question=request.message,
            retrieved_chunks=[{"text": c} for c in context_chunks if isinstance(c, str)]
                              or [{"content": str(c)} for c in context_chunks],
            answer=response_content,
        )

        grounding_ratio = _rag_gate.diagnostics.get("grounding_ratio", 1.0)
        if (
            not _rag_gate.passed
            and grounding_ratio < 0.25
            and context_chunks
            and not retry_attempted
        ):
            retry_attempted = True
            logger.info(
                f"[Chat] Self-eval triggered retry: grounding_ratio={grounding_ratio:.2f} "
                f"(threshold=0.25). Broadening search query."
            )
            try:
                from backend.rag.agent_router import MeetingAgentRouter
                retry_router = MeetingAgentRouter(repo=repo, vector_service=vector_service)
                retry_result = retry_router.route(
                    meeting_id=meeting_id,
                    query=request.message,
                    top_k=12,
                    similarity_threshold=0.15,
                    cross_meeting=request.cross_meeting,
                )
                retry_chunks = retry_result.get("context_chunks", [])
                retry_sources = retry_result.get("source_chunks", [])

                if retry_chunks:
                    retry_response = generate_chatbot_response(
                        query=request.message,
                        context_chunks=retry_chunks,
                        chat_history=chat_history,
                        meeting_title=meeting.title or meeting.filename,
                        query_intent=query_intent,
                    )
                    retry_gate = _qe.validate_rag_chatbot(
                        question=request.message,
                        retrieved_chunks=[{"text": c} for c in retry_chunks if isinstance(c, str)],
                        answer=retry_response,
                    )
                    retry_grounding = retry_gate.diagnostics.get("grounding_ratio", 0)

                    if retry_grounding > grounding_ratio:
                        logger.info(
                            f"[Chat] Retry accepted: grounding {grounding_ratio:.2f} "
                            f"→ {retry_grounding:.2f}"
                        )
                        response_content = retry_response
                        context_chunks = retry_chunks
                        source_chunks = retry_sources
                        _rag_gate = retry_gate
            except Exception as retry_err:
                logger.warning(f"[Chat] Self-eval retry failed: {retry_err}")

        if not _rag_gate.passed or _rag_gate.status == "DEGRADED":
            rag_quality_warning = _rag_gate.failure_reason
    except Exception as qe_err:
        logger.debug(f"[Chat] RAG quality gate check skipped: {qe_err}")

    # -- Real-time RAG groundedness of THIS answer, judged against the context it was generated from --
    rag_evaluation = evaluate_rag_answer(
        meeting_pk=meeting.id,
        question=request.message,
        answer=response_content,
        context_chunks=context_chunks,
        source_chunks=source_chunks,
    )

    # Save user and assistant messages to DB with rollback protection
    assistant_msg_id = None
    assistant_timestamp = datetime.now(timezone.utc)
    try:
        repo.add_chat_message(
            meeting_id=meeting.id,
            role="user",
            content=request.message,
            session_id=session_id,
        )
        assistant_msg = repo.add_chat_message(
            meeting_id=meeting.id,
            role="assistant",
            content=response_content,
            session_id=session_id,
            sources=json.dumps(source_chunks) if source_chunks else None,
        )
        if assistant_msg:
            assistant_msg_id = assistant_msg.id
            assistant_timestamp = assistant_msg.timestamp
            if store_rag_evaluation(
                repo.db, meeting_pk=meeting.id, chat_message_id=assistant_msg.id, evaluation=rag_evaluation
            ):
                repo.db.commit()
    except Exception as db_err:
        logger.error(f"[Chat] Failed to save chat messages to DB: {db_err}", exc_info=True)
        try:
            repo.db.rollback()
        except Exception:
            pass

    # Update conversation memory
    memory.append({"role": "assistant", "content": response_content})

    response_payload = {
        "id": assistant_msg_id,
        "session_id": session_id,
        "meeting_id": meeting.meeting_id or str(meeting.id),
        "role": "assistant",
        "content": response_content,
        "timestamp": assistant_timestamp,
        "sources": source_chunks if source_chunks else [],
        "rag_groundedness": rag_evaluation,
    }
    if rag_quality_warning:
        response_payload["quality_warning"] = rag_quality_warning
    return response_payload


@router.get(
    "/{meeting_id}/chat/history", response_model=List[schemas.ChatMessageResponse]
)
def get_chat_history(
    meeting_id: Any,
    session_id: Optional[str] = Query(None, description="Optional chat session ID"),
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """Retrieves conversation history with pagination, deserializing source metadata dynamically."""
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    history = repo.get_chat_history(meeting.id, session_id)

    results = []
    for msg in history:
        sources_list = []
        if msg.sources:
            try:
                sources_list = json.loads(msg.sources)
            except Exception:
                pass
        results.append(
            {
                "id": msg.id,
                "session_id": msg.session_id,
                "meeting_id": meeting.meeting_id or str(meeting.id),
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "sources": sources_list,
            }
        )
    return results


@router.post("/{meeting_id}/chat/clear")
def clear_chat_history(
    meeting_id: Any,
    session_id: Optional[str] = Query(None, description="Optional chat session ID"),
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """Clears chat history for a session."""
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    repo.clear_chat_history(meeting.id, session_id)

    with _conversation_memory_mutex:
        if session_id:
            _conversation_memory.pop((meeting.id, session_id), None)
        else:
            # No specific session named — clear every in-memory session
            # belonging to THIS meeting only (never another meeting's).
            for key in [k for k in _conversation_memory if k[0] == meeting.id]:
                del _conversation_memory[key]

    return {"message": f"Cleared message history successfully for meeting {meeting.meeting_id or meeting.id}"}


def get_conversation_memory(meeting_id: int, session_id: Optional[str]) -> deque:
    """Thread-safe conversation memory with TTL eviction and session cap.

    Keyed by (meeting_id, session_id) — NEVER by session_id alone. A request
    that omits session_id is common (client hasn't been issued one yet), and
    keying on the bare session_id would collapse every such request across
    ALL meetings onto a single shared `_conversation_memory[None]` bucket,
    mixing one meeting's conversation into another's.
    """
    import time
    key = (meeting_id, session_id)
    with _conversation_memory_mutex:
        _evict_stale_sessions()
        if key not in _conversation_memory:
            _conversation_memory[key] = (deque(maxlen=_MEMORY_MAX_LENGTH), time.time())
        else:
            mem_deque, _ = _conversation_memory[key]
            _conversation_memory[key] = (mem_deque, time.time())
        return _conversation_memory[key][0]


@router.post("/{meeting_id}/chat/stream", dependencies=[Depends(chat_rate_limit)])
def post_chat_message_stream(
    meeting_id: Any,
    request: schemas.ChatMessageCreate,
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """Sends a message to the chatbot with Agentic RAG retrieval and streams the response tokens."""
    meeting = repo.get_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    # Always use the resolved integer PK (see post_chat_message for rationale —
    # prevents cross-meeting vector leakage when a non-PK id form is supplied).
    meeting_id = meeting.id
    enforce_meeting_isolation(request, meeting_id)

    from backend.utils.security import validate_chat_message

    request.message = validate_chat_message(request.message)

    # 1. Thread-safe Lazy Indexing Check
    lock = get_indexing_lock(meeting_id)
    with lock:
        try:
            if not vector_service.has_transcript_chunks(meeting_id):
                logger.info(
                    f"Lazy indexing: Speech segments for meeting {meeting_id} not found in vector DB. Re-indexing..."
                )
                segments = repo.get_segments(meeting_id)
                if segments:
                    vector_service.add_transcript_chunks(meeting_id, segments)
                    logger.info(
                        f"Lazy indexing completed: Indexed {len(segments)} segments for meeting {meeting_id}"
                    )
        except Exception as e:
            logger.error(
                f"Lazy indexing failed for meeting {meeting_id}: {e}", exc_info=True
            )

    # 2. Get history and query intent
    # Resolve session_id the same way post_chat_message does — a request that
    # omits it must fall back to a meeting-scoped id, never a bare None shared
    # across every meeting (see get_conversation_memory docstring).
    session_id = request.session_id or meeting.meeting_id or str(meeting_id)
    history_records = repo.get_chat_history(meeting_id, session_id)
    chat_history = [
        {"role": record.role, "content": record.content} for record in history_records
    ]

    memory = get_conversation_memory(meeting_id, session_id)
    memory.append({"role": "user", "content": request.message})

    query_intent = "general"
    try:
        query_intent = analyze_query_intent(request.message)
    except Exception as e:
        logger.warning(f"Failed to analyze query intent: {e}")

    # Rewrite query with history for follow-ups
    search_query = request.message
    if len(chat_history) >= 1:
        try:
            search_query = rewrite_query_with_history(request.message, chat_history)
            logger.debug(f"[Stream] Query rewritten: {request.message!r} -> {search_query!r}")
        except Exception as e:
            logger.warning(f"Failed to rephrase follow-up query: {e}")

    # 3. Agentic RAG Retrieval
    context_chunks: list = []
    source_chunks: list = []
    try:
        from backend.rag.agent_router import MeetingAgentRouter
        agent_router = MeetingAgentRouter(repo=repo, vector_service=vector_service)
        routing_result = agent_router.route(
            meeting_id=meeting_id,
            query=search_query,
            top_k=8,
            similarity_threshold=0.25,
            cross_meeting=request.cross_meeting,
        )
        context_chunks = routing_result.get("context_chunks", [])
        source_chunks = routing_result.get("source_chunks", [])
        query_intent = routing_result.get("primary_intent", query_intent)

        logger.info(
            f"[Stream] AgentRouter: intents={routing_result.get('intents')}, "
            f"tools={routing_result.get('tools_used')}, "
            f"context_chunks={len(context_chunks)}, "
            f"elapsed={routing_result.get('elapsed_ms', 0):.0f}ms"
        )
    except Exception as e:
        logger.error(f"[Stream] AgentRouter failed, falling back: {e}", exc_info=True)
        try:
            top_k = 8 if query_intent in ["summary", "decisions", "action_items", "mom"] else 5
            vector_results = vector_service.search_transcript_chunks(
                meeting_id=meeting_id,
                query_text=search_query,
                top_k=top_k,
                cross_meeting=request.cross_meeting,
            )
            for chunk in vector_results:
                if chunk.get("similarity_score", 0) >= 0.20:
                    context_chunks.append(chunk["text"])
                    source_chunks.append(chunk)
        except Exception as vec_err:
            logger.error(f"[Stream] Fallback vector search failed: {vec_err}", exc_info=True)

    # 4. Stream Generator
    async def event_generator():
        llm = get_llm()
        formatted_context = "\n\n".join(
            [
                f"Context Chunk {i + 1}:\n{chunk}"
                for i, chunk in enumerate(context_chunks)
            ]
        )
        if not formatted_context.strip():
            formatted_context = "No relevant context found from the meeting transcript."

        history_str = ""
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

        # The meeting title is baked into the template: escape its braces (see escape_template_braces).
        system_prompt_str = _make_sys_prompt(chatbot_system_prompt(meeting.title or meeting.filename))

        intent_guidance = get_intent_guidance()

        user_prompt_str = (
            "Here is the conversation history so far:\n{history}\n\n"
            "Here is the relevant meeting transcript and data context:\n{context}\n\n"
            f"Query Intent: {escape_template_braces(query_intent)}\n"
            f"Intent Guidance: {escape_template_braces(intent_guidance.get(query_intent, intent_guidance['general']))}\n\n"
            "User Question: {query}\n\n"
            "Remember: Answer ONLY from the provided meeting context. Do NOT force a fixed template — format your answer naturally according to the question type."
        )

        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt_str), ("user", user_prompt_str)]
        )

        chain = prompt | llm
        full_response = ""
        buffer = ""
        in_think = False

        try:
            async for chunk in chain.astream(
                {
                    "history": history_str,
                    "context": formatted_context,
                    "query": request.message,
                }
            ):
                token = chunk.content
                if not token:
                    continue

                buffer += token
                if "<think" in buffer:
                    in_think = True
                    parts = buffer.split("<think", 1)
                    out_token = parts[0]
                    buffer = ""
                    if out_token:
                        full_response += out_token
                        yield f"data: {json.dumps({'text': out_token})}\n\n"
                    if "</think>" in parts[1]:
                        in_think = False
                        buffer = parts[1].split("</think>", 1)[1]
                elif "</think>" in buffer:
                    in_think = False
                    parts = buffer.split("</think>", 1)
                    buffer = parts[1]
                elif not in_think:
                    out_token = buffer
                    buffer = ""
                    full_response += out_token
                    yield f"data: {json.dumps({'text': out_token})}\n\n"

            if not in_think and buffer:
                full_response += buffer
                yield f"data: {json.dumps({'text': buffer})}\n\n"
        except Exception as e:
            logger.error(f"Error during stream generation: {e}", exc_info=True)
            yield f"data: {json.dumps({'text': ' [Error during generation] '})}\n\n"

        sources_str = json.dumps(source_chunks) if source_chunks else None
        # The text has already been streamed; scoring runs off the event loop before the row is stored.
        rag_evaluation = await asyncio.to_thread(
            lambda: evaluate_rag_answer(
                meeting_pk=meeting_id,
                question=request.message,
                answer=full_response,
                context_chunks=context_chunks,
                source_chunks=source_chunks,
            )
        )
        await asyncio.to_thread(
            save_chat_to_db_sync,
            meeting_id,
            session_id,
            request.message,
            full_response,
            sources_str,
            rag_evaluation,
        )

        memory.append({"role": "assistant", "content": full_response})

        yield f"data: {json.dumps({'done': True, 'sources': source_chunks, 'rag_groundedness': rag_evaluation, 'timestamp': datetime.now(timezone.utc).isoformat()}, default=str)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


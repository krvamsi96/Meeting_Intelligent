"""Server-Sent Events (SSE) fan-out manager for real-time activity log updates.

Architecture
------------
- In-process asyncio.Queue fan-out — works for single-worker uvicorn deployments.
- The pipeline workers (sync threads) call publish_sync() which schedules an
  asyncio coroutine on the running event loop via run_coroutine_threadsafe().
- The SSE API endpoint uses async generator subscribe() to receive events.

Design constraints
------------------
- No external broker (Redis, RabbitMQ) required.
- Thread-safe: sync pipeline workers can safely call publish_sync().
- Automatic cleanup: subscriber queues are removed when the HTTP connection closes.
- Max queue depth: 200 events per subscriber (oldest dropped on overflow).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional, Set

logger = logging.getLogger(__name__)

_MAX_QUEUE_DEPTH = 200


class SSEManager:
    """Fan-out manager: pipeline workers publish, HTTP subscribers receive."""

    def __init__(self) -> None:
        # meeting_pk (int) → set of asyncio.Queue
        self._subscribers: Dict[int, Set[asyncio.Queue]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Loop binding ───────────────────────────────────────────────────────────

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind to the running event loop (called once at FastAPI startup)."""
        self._loop = loop

    def _get_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Return the bound loop, or attempt to get the running one."""
        if self._loop is not None:
            return self._loop
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    # ── Subscription management ────────────────────────────────────────────────

    async def subscribe(self, meeting_pk: int) -> AsyncGenerator[str, None]:
        """Async generator yielding SSE-formatted event strings for a meeting.

        Each yielded string is a complete SSE block, e.g.:
            data: {"step": "transcription", "status": "COMPLETED"}\n\n

        Usage in a FastAPI streaming endpoint::

            async def stream(meeting_id: int):
                from fastapi.responses import StreamingResponse
                async def gen():
                    async for event in sse_manager.subscribe(meeting_id):
                        yield event
                return StreamingResponse(gen(), media_type="text/event-stream")
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE_DEPTH)
        self._subscribers.setdefault(meeting_pk, set()).add(queue)
        logger.debug("SSE subscriber added for meeting_pk=%s (total=%d)",
                     meeting_pk, len(self._subscribers[meeting_pk]))
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send a keepalive comment so the HTTP connection stays open
                    yield ": keepalive\n\n"
                    continue
                if payload is None:
                    # Sentinel: producer signalled end of stream
                    break
                yield payload
        finally:
            self._subscribers.get(meeting_pk, set()).discard(queue)
            if meeting_pk in self._subscribers and not self._subscribers[meeting_pk]:
                del self._subscribers[meeting_pk]
            logger.debug("SSE subscriber removed for meeting_pk=%s", meeting_pk)

    # ── Publishing ─────────────────────────────────────────────────────────────

    async def publish(self, meeting_pk: int, data: Dict[str, Any]) -> None:
        """Publish an event to all subscribers of a meeting (async version)."""
        queues = self._subscribers.get(meeting_pk, set())
        if not queues:
            return
        payload = f"data: {json.dumps(data, default=str)}\n\n"
        dead: Set[asyncio.Queue] = set()
        for q in list(queues):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Drop oldest message to make room
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except Exception:
                    dead.add(q)
        for q in dead:
            queues.discard(q)

    def publish_sync(self, meeting_pk: int, data: Dict[str, Any]) -> None:
        """Thread-safe publish for sync pipeline workers.

        Schedules publish() on the event loop from a background thread using
        run_coroutine_threadsafe().  Safe to call from any non-async context.
        """
        loop = self._get_loop()
        if loop is None or not loop.is_running():
            # No event loop available (tests, CLI usage) — silently skip
            return
        asyncio.run_coroutine_threadsafe(self.publish(meeting_pk, data), loop)

    # ── Lifecycle helpers ──────────────────────────────────────────────────────

    def close_meeting(self, meeting_pk: int, final_event: Optional[Dict[str, Any]] = None) -> None:
        """Signal all subscribers that this meeting's stream has ended.

        ``final_event`` (e.g. ``{"event": "done", "status": "completed"}``) is delivered just before the
        end-of-stream sentinel, so a client learns the outcome instead of only seeing the connection drop
        (which a browser EventSource answers by reconnecting).

        Called from pipeline worker THREADS. ``asyncio.Queue`` is not thread-safe: a bare
        ``put_nowait`` from another thread does not wake a consumer blocked in ``get()`` (its
        future is only resolved via the loop), so the stream would linger until the 30 s
        keepalive timeout. Hop onto the event loop instead.
        """
        items = ([f"data: {json.dumps(final_event, default=str)}\n\n"] if final_event else []) + [None]

        def _put(q: asyncio.Queue, item) -> None:
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                # A full queue used to swallow the sentinel and the stream never ended.
                try:
                    q.get_nowait()
                    q.put_nowait(item)
                except Exception:
                    pass
            except Exception:
                pass

        def _close() -> None:
            for q in list(self._subscribers.get(meeting_pk, set())):
                for item in items:  # None = sentinel
                    _put(q, item)

        loop = self._get_loop()
        if loop is not None and loop.is_running():
            try:
                in_loop_thread = asyncio.get_running_loop() is loop
            except RuntimeError:
                in_loop_thread = False
            if in_loop_thread:
                _close()
            else:
                loop.call_soon_threadsafe(_close)
        else:
            _close()  # no loop (tests / CLI): nothing is awaiting, plain call is fine

    @property
    def active_meeting_count(self) -> int:
        return len(self._subscribers)

    def subscriber_count(self, meeting_pk: int) -> int:
        return len(self._subscribers.get(meeting_pk, set()))


# Singleton — imported by both api/activity_logs.py and services/activity_logger.py
sse_manager = SSEManager()

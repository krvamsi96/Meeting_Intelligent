"""In-process sliding-window rate limiting for expensive endpoints.

Upload / URL-ingest / chat each trigger heavy work (ffmpeg, Whisper, LLM calls billed per
token) and were completely unthrottled. This is a small dependency-free limiter suited to
the platform's deliberate single-worker deployment (ChromaDB allows one process, see
main.py). With several workers each would keep its own counters — put a reverse proxy /
API gateway limiter in front in that case.

Configure with ``RATE_LIMIT_CHAT_PER_MIN`` and ``RATE_LIMIT_UPLOAD_PER_MIN`` (0 disables).
The client is identified by socket address; ``X-Forwarded-For`` is only honoured when
``TRUST_FORWARDED_FOR=true`` (otherwise any caller could dodge the limit by spoofing it).
"""

import logging
import math
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict

from fastapi import HTTPException, Request

from backend.config.config import settings

logger = logging.getLogger(__name__)

_WINDOW_SEC = 60.0
_MAX_TRACKED_CLIENTS = 10_000  # bound memory even under an address-spraying attack


class SlidingWindowLimiter:
    def __init__(self, name: str, limit_getter: Callable[[], int], window_sec: float = _WINDOW_SEC):
        self.name = name
        self._limit_getter = limit_getter
        self._window = window_sec
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @staticmethod
    def _client_key(request: Request) -> str:
        if settings.TRUST_FORWARDED_FOR:
            forwarded = request.headers.get("x-forwarded-for", "")
            first = forwarded.split(",")[0].strip()
            if first:
                return first
        return request.client.host if request.client else "unknown"

    def _prune(self, now: float) -> None:
        """Drop idle clients so the table cannot grow without bound. Caller holds the lock."""
        cutoff = now - self._window
        for key in [k for k, dq in self._hits.items() if not dq or dq[-1] < cutoff]:
            del self._hits[key]

    def check(self, request: Request) -> None:
        limit = int(self._limit_getter() or 0)
        if limit <= 0:
            return
        key = self._client_key(request)
        now = time.monotonic()
        with self._lock:
            if len(self._hits) >= _MAX_TRACKED_CLIENTS:
                self._prune(now)
            hits = self._hits[key]
            cutoff = now - self._window
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= limit:
                retry_after = max(1, math.ceil(hits[0] + self._window - now))
                logger.warning("[RateLimit] %s exceeded %d/min for client %s", self.name, limit, key)
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many requests. Limit is {limit} per minute; retry in {retry_after}s.",
                    headers={"Retry-After": str(retry_after)},
                )
            hits.append(now)

    def __call__(self, request: Request) -> None:  # FastAPI dependency
        self.check(request)


chat_rate_limit = SlidingWindowLimiter("chat", lambda: settings.RATE_LIMIT_CHAT_PER_MIN)
upload_rate_limit = SlidingWindowLimiter("upload", lambda: settings.RATE_LIMIT_UPLOAD_PER_MIN)

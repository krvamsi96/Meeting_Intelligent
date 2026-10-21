import json
import logging
import re
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from backend.config.config import settings

logger = logging.getLogger(__name__)

# Sensitive query parameters that must never appear in logs. Matched as SUBSTRINGS of the
# lower-cased name so access_token / x-api-key / apikey / client_secret are covered too.
_REDACT_PARAMS = frozenset({"api_key", "apikey", "token", "password", "passwd", "secret", "key", "auth"})

# A caller-supplied X-Request-ID is echoed into logs and a response header. Accept only a short
# token of safe characters; anything else (CR/LF, spaces, huge values) is a log-forging vector.
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def _redact_query(query: str) -> str:
    """Replace values of sensitive query parameters with [REDACTED]."""
    if not query:
        return ""
    parts = []
    for param in query.split("&"):
        if "=" in param:
            name, _, val = param.partition("=")
            lowered = name.lower()
            if any(sensitive in lowered for sensitive in _REDACT_PARAMS):
                parts.append(f"{name}=[REDACTED]")
            else:
                parts.append(param)
        else:
            parts.append(param)
    return "&".join(parts)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs all incoming requests, tracking elapsed time and correlation IDs.

    Supports two output modes:
    - LOG_FORMAT=text  (default): human-readable single-line log
    - LOG_FORMAT=json           : structured JSON for log aggregation (ELK/Loki/Datadog)
    """

    _json_mode: bool = settings.LOG_FORMAT.lower() == "json"

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", "")
        if not _SAFE_REQUEST_ID.fullmatch(request_id):
            request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        query = _redact_query(request.url.query)

        try:
            response: Response = await call_next(request)
            elapsed = round(time.time() - start_time, 3)
            response.headers["x-request-id"] = request_id

            if self._json_mode:
                logger.info(json.dumps({
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "query": query,
                    "status": response.status_code,
                    "elapsed_sec": elapsed,
                }))
            else:
                path_with_query = f"{request.url.path}{'?' + query if query else ''}"
                logger.info(
                    f"[{request_id}] {request.method} {path_with_query} "
                    f"-> {response.status_code} ({elapsed:.3f}s)"
                )
            return response
        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            if self._json_mode:
                logger.error(json.dumps({
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "elapsed_sec": elapsed,
                    "error": str(e),
                }))
            else:
                logger.error(
                    f"[{request_id}] {request.method} {request.url.path} "
                    f"FAILED in {elapsed:.3f}s: {e}",
                    exc_info=True,
                )
            raise

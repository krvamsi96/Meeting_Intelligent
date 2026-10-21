"""Lightweight authentication guard for administrative endpoints.

The platform has no user/session model, so this is a single shared secret
(``ADMIN_API_KEY``) protecting the endpoints that change server-wide behaviour or
touch the filesystem (AI-provider settings, diarization / loop-engine tuning,
cache clearing).

* Clients send ``X-Admin-Key: <secret>`` or ``Authorization: Bearer <secret>``.
* When ``ADMIN_API_KEY`` is unset the guard lets requests through (backward
  compatible with existing deployments) and logs a one-time warning — set it on
  any deployment reachable by people you do not fully trust.
* Comparison is constant-time.
"""

import hmac
import logging
from typing import Optional

from fastapi import HTTPException, Request

from backend.config.config import settings

logger = logging.getLogger(__name__)

_warned_open = False


def _extract_credential(request: Request) -> Optional[str]:
    supplied = request.headers.get("x-admin-key")
    if supplied:
        return supplied.strip()
    authz = request.headers.get("authorization", "")
    scheme, _, token = authz.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return None


def require_admin(request: Request) -> None:
    """FastAPI dependency: reject the request unless it carries the admin key."""
    global _warned_open
    expected = (settings.ADMIN_API_KEY or "").strip()
    if not expected:
        if not _warned_open:
            _warned_open = True
            logger.warning(
                "ADMIN_API_KEY is not set: administrative endpoints (%s) are UNAUTHENTICATED. "
                "Set ADMIN_API_KEY in .env to protect them.",
                request.url.path,
            )
        return

    supplied = _extract_credential(request)
    if not supplied or not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=401,
            detail="Administrator credentials required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

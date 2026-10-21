"""Security headers middleware for hardening enterprise production HTTP responses."""

import base64
import hashlib
import logging
import re

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.config.config import settings
from backend.utils.security import REPORT_PRINT_SCRIPT

logger = logging.getLogger(__name__)


def _build_swagger_csp() -> str:
    """Build a CSP for Swagger/ReDoc UI that allows their JS/CSS resources.
    connect-src is derived from settings to support any deployment target.
    """
    # Allow the backend itself plus any origin in CORS_ORIGINS
    connect_sources = " ".join(
        [settings.backend_base_url]
        + [o.rstrip("/") for o in settings.cors_origin_list if o]
    )
    return (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' data: https://fastapi.tiangolo.com https://cdn.jsdelivr.net; "
        f"connect-src 'self' {connect_sources}"
    )


def _build_api_csp() -> str:
    """Build a strict CSP for all non-Swagger API endpoints."""
    return (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )


# /meetings/{id}/export/html and /export/pdf return a standalone, self-styled HTML document. The
# strict API policy (no inline style / script) blanked its styling and blocked its print trigger.
_REPORT_PATH_RE = re.compile(r"^/api/v\d+/meetings/[^/]+/export/(html|pdf)/?$")
_PRINT_SCRIPT_HASH = base64.b64encode(hashlib.sha256(REPORT_PRINT_SCRIPT.encode("utf-8")).digest()).decode("ascii")


def _build_report_csp() -> str:
    """CSP for the exported report page.

    Inline CSS is required (the report ships a <style> block) but scripts stay locked down:
    the ONLY script allowed is the exact print trigger, by hash. All report text is HTML-escaped
    by export_service, so style permission cannot be turned into script execution.
    """
    return (
        "default-src 'none'; "
        f"script-src 'sha256-{_PRINT_SCRIPT_HASH}'; "
        "style-src 'unsafe-inline'; "
        "img-src data: https:; "
        "font-src data: https:; "
        "base-uri 'none'; "
        "form-action 'none'; "
        "frame-ancestors 'none'"
    )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enforces essential security headers on all HTTP responses.

    - Swagger/ReDoc paths use relaxed CSP (required for JS-heavy UI).
    - All other paths use a strict CSP with no inline/eval.
    - CSP connect-src is built dynamically from settings.CORS_ORIGINS.
    - Permissions-Policy disables browser APIs not required by this service.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)

        # ── Universal headers ─────────────────────────────────────────────────
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-DNS-Prefetch-Control"] = "off"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), "
            "payment=(), usb=(), interest-cohort=()"
        )

        path = request.url.path
        if path.startswith(("/docs", "/redoc", "/openapi.json")):
            response.headers["Content-Security-Policy"] = _build_swagger_csp()
        else:
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
            if _REPORT_PATH_RE.match(path):
                response.headers["Content-Security-Policy"] = _build_report_csp()
            else:
                response.headers["Content-Security-Policy"] = _build_api_csp()

        return response

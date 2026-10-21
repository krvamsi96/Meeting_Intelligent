"""System-level endpoints such as health check."""

import logging
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.config import settings
from backend.config.config import PROJECT_ROOT
from backend.database.connection import get_database_target, get_engine_pool_status
from backend.repositories import MeetingRepository
from backend.services.model_router import model_router
from backend.services.vector_service import vector_service
from backend.utils.auth import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["System"])

# A health poll must answer quickly even while a dependency is slow or down: the Docker / compose
# healthcheck gives it 10 s, and an unreachable AIRAWAT (30 s probe timeout) or a cold embedding-model
# load (seconds to a minute on a small server) used to hold the whole endpoint past that, so the
# container was flagged unhealthy exactly when an upstream was struggling.
# The two probes are awaited one after the other, so the worst case is 2 x this on the first poll only.
_PROBE_WAIT_SEC = 2.0


class _SlowProbe:
    """Run a slow dependency check off the request path and serve its last completed result.

    * The check runs in a daemon thread, never in the request, and never twice at once.
    * Until a first result exists a poll waits at most ``wait`` seconds for it, then gets ``default``.
    * A result older than ``ttl`` is still served immediately while a fresh one is computed
      (stale-while-revalidate), so a poll never blocks on a refresh.
    * A result is only reused for the same ``fn``, so swapping the check (tests) never serves stale data.
      ``fn`` is compared with ``==``, not ``is``: ``obj.method`` builds a NEW bound-method object on every
      access, so an identity test would see a "different" check each poll and never cache anything.
    """

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
        self._fn: Optional[Callable[[], Any]] = None
        self._value: Any = None
        self._done_at = 0.0  # monotonic time the last result completed; 0 = none yet
        self._thread: Optional[threading.Thread] = None

    def get(self, fn: Callable[[], Any], *, ttl: float, default: Any, on_error: Any, wait: Optional[float] = None) -> Any:
        wait = _PROBE_WAIT_SEC if wait is None else wait
        with self._lock:
            if self._fn != fn:
                self._fn, self._value, self._done_at = fn, None, 0.0
            has_value = self._done_at > 0.0
            stale = not has_value or (time.monotonic() - self._done_at) > ttl
            if stale and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(
                    target=self._run, args=(fn, on_error), name=f"health-probe-{self.name}", daemon=True
                )
                self._thread.start()
            thread = self._thread
        if not has_value and thread is not None:
            thread.join(timeout=wait)  # nothing to serve yet: give the first result a moment
        with self._lock:
            return self._value if self._done_at > 0.0 and self._fn == fn else default

    def _run(self, fn: Callable[[], Any], on_error: Any) -> None:
        try:
            value = fn()
        except Exception:
            logger.warning("Health probe '%s' failed", self.name, exc_info=True)
            value = on_error
        with self._lock:
            if self._fn == fn:
                self._value, self._done_at = value, time.monotonic()


_airawat_probe = _SlowProbe("airawat")
_models_probe = _SlowProbe("models")
_MODELS_TTL_SEC = 10.0


def _effective_airawat_key() -> str:
    """The AIRAWAT key actually in use (SERVER_*/LOCAL_* selection or admin override).

    ``settings.AIRAWAT_API_KEY`` alone is wrong here: a standard deployment defines
    SERVER_AIRAWAT_API_KEY, leaving the unprefixed setting empty.
    """
    try:
        key = model_router.get_effective_config().get("api_key") or ""
    except Exception:
        key = ""
    return (key or settings.AIRAWAT_API_KEY or "").strip()


def _get_airawat_health() -> bool:
    """AIRAWAT LLM connectivity (cached ``AIRAWAT_HEALTH_TTL_SEC``); False without a key (AIRAWAT is the only provider).

    The probe is a slow outbound HTTP call, so it runs in the background (see ``_SlowProbe``). Until the
    very first probe finishes the answer is optimistic (True) so a liveness poll does not fail on an unknown.
    """
    if not _effective_airawat_key():
        # No key means NO AI provider at all (AIRAWAT is the only one): never report that as healthy.
        return False
    return bool(
        _airawat_probe.get(
            model_router.validate_airawat_config,
            ttl=settings.AIRAWAT_HEALTH_TTL_SEC,
            default=True,
            on_error=False,
        )
    )


_provider_probe = _SlowProbe("ai-provider")


def _configured_providers() -> dict:
    """``{"text_generation", "speech_to_text", "problem"}`` - the providers this deployment is set up to use."""
    from backend.providers import provider_factory

    out = {"text_generation": None, "speech_to_text": None, "problem": None}
    try:
        out["text_generation"] = provider_factory.llm_provider_name()
        out["speech_to_text"] = provider_factory.stt_provider_name()
    except Exception as exc:  # noqa: BLE001 - a configuration error is reported, never raised
        out["problem"] = str(exc)
    return out


def _ai_provider_services(providers: dict) -> dict:
    """The ``services`` flag(s) for the AI provider: ``airawat`` (unchanged contract) or ``ai_provider``."""
    name = providers.get("text_generation")
    if name == "airawat":
        return {"airawat": _get_airawat_health()}
    if not name or providers.get("problem"):
        return {"ai_provider": False}
    from backend.providers import provider_factory

    provider = provider_factory.get_provider(name)
    if not provider.is_configured:
        return {"ai_provider": False}
    # Slow outbound call: background probe, optimistic until the first answer (same rule as AIRAWAT).
    return {"ai_provider": bool(_provider_probe.get(
        provider.test_connection, ttl=settings.AIRAWAT_HEALTH_TTL_SEC, default={"ok": True}, on_error={"ok": False},
    ).get("ok"))}


def _airawat_status() -> dict:
    try:
        from backend.services.airawat_health import startup_status

        return {"configured": bool(_effective_airawat_key()), "startup_check": startup_status()}
    except Exception as exc:  # noqa: BLE001 - health must not raise
        return {"configured": bool(_effective_airawat_key()), "startup_check": {"state": "unknown", "error": str(exc)}}


def _vector_indexing_status() -> dict:
    try:
        from backend.services.vector_backfill import indexing_status

        return indexing_status()
    except Exception as exc:  # noqa: BLE001 - health must not raise
        return {"state": "unknown", "error": str(exc)}


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Service health check endpoint for load balancers and monitoring.

    AIRAWAT connectivity is checked at most once every 5 minutes to avoid
    hammering the external API on every load-balancer probe.
    """
    from backend.services.model_loader import get_all_models_health

    pool_status = get_engine_pool_status()
    database_target = get_database_target()
    # Reads the embedding-model state and, on first use, LOADS the model. Never do that inside the poll:
    # while it loads the answer is "loading" (reported as degraded) and turns healthy once it is ready.
    model_health = _models_probe.get(
        get_all_models_health,
        ttl=_MODELS_TTL_SEC,
        default={"status": "loading", "offline_mode": settings.OFFLINE_MODE, "models": {}},
        on_error={"status": "error", "offline_mode": settings.OFFLINE_MODE, "models": {}},
    )

    # ChromaDB is the only vector database; when it is unavailable this carries the reason.
    vector_database = vector_service.health()

    providers = _configured_providers()
    status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "offline_mode": model_health.get("offline_mode", True),
        "services": {
            "database": _check_database(db),
            "vector_database": vector_database["status"] == "available",
            **_ai_provider_services(providers),
            "models": model_health.get("status") == "healthy",
        },
        "models": model_health.get("models", {}),
        # Readiness details: embedding model (loaded engine + startup file verification) and vector indexing progress.
        "embedding_model": {
            "status": model_health.get("status"),
            **{k: (model_health.get("models", {}).get("embedding") or {}).get(k)
               for k in ("active_engine", "resolved_local_path", "dimension", "files", "problem", "fix")},
        },
        "vector_indexing": _vector_indexing_status(),
        # Which provider generates text / transcribes; for AIRAWAT also its startup deep check.
        "ai_providers": providers,
        **({"airawat": _airawat_status()} if "airawat" in (providers["text_generation"], providers["speech_to_text"]) else {}),
        "database_pool": pool_status,
        "database_target": database_target,
        "vector_database": vector_database,
    }
    degraded = any(not v for v in status["services"].values())
    if degraded:
        status["status"] = "degraded"
    if database_target.get("fallback_active"):
        # Connected, but not to the database the deployment is configured for: never report that as healthy.
        status["status"] = "degraded"
        status["warning"] = database_target["warning"]
    # Warn if pool is nearly exhausted (> 80% utilised)
    total = pool_status.get("total_capacity") or 0  # None for pool types without QueuePool metrics
    used = pool_status.get("checked_out") or 0
    if total > 0 and (used / total) >= 0.80:
        status["status"] = "degraded"
        status["warning"] = " | ".join(
            w for w in (status.get("warning"), f"Connection pool near capacity: {used}/{total} connections in use") if w
        )
    return status


@router.get("/status")
def service_status(db: Session = Depends(get_db)):
    """What the web app shows its users: is the service usable? No component names, hosts, paths or versions -
    those stay in ``/system/health`` for operators and monitoring."""
    database_ok = _check_database(db)
    search_ok = vector_service.health()["status"] == "available"
    if not database_ok:
        state = "unavailable"
    elif not search_ok:
        state = "degraded"
    else:
        state = "available"
    return {"status": state, "can_view_meetings": database_ok, "can_upload": database_ok, "can_search": search_ok}


_airawat_diagnostics_probe = _SlowProbe("airawat-diagnostics")


@router.get("/health/airawat")
def airawat_health_check():
    """Detailed AIRAWAT validation (LLM + Whisper endpoints, model, request/response format, latency).

    Never contains the API key (only a short fingerprint). Cached for AIRAWAT_HEALTH_TTL_SEC and refreshed in the
    background, so it always answers immediately: ``{"status": "checking"}`` until the first result exists.
    """
    from backend.services.airawat_health import run_diagnostics

    return _airawat_diagnostics_probe.get(
        run_diagnostics,
        ttl=settings.AIRAWAT_HEALTH_TTL_SEC,
        default={"status": "checking", "provider": "airawat"},
        on_error={"status": "error", "provider": "airawat", "ok": False},
        wait=_PROBE_WAIT_SEC,
    )


@router.get("/health/models")
def models_health_check():
    """Detailed offline AI/ML model health check endpoint."""
    from backend.services.model_loader import get_all_models_health
    return get_all_models_health()


@router.get("/metrics")
def get_system_metrics(db: Session = Depends(get_db)):
    """Get system-wide metrics for the dashboard."""
    try:
        repo = MeetingRepository(db)
        meetings = repo.list_all()

        completed = [m for m in meetings if m.status == "completed"]
        processing = [m for m in meetings if m.status in ("processing", "uploaded")]
        failed = [m for m in meetings if m.status == "error"]

        total_duration = sum(m.duration or 0 for m in completed)
        avg_accuracy = (
            sum(m.accuracy or 0 for m in completed) / len(completed) if completed else 0
        )
        avg_speaker_accuracy = (
            sum(m.speaker_accuracy or 0 for m in completed) / len(completed)
            if completed
            else 0
        )
        avg_transcription_accuracy = (
            sum(m.transcription_accuracy or 0 for m in completed) / len(completed)
            if completed
            else 0
        )

        return {
            "meetings": {
                "total": len(meetings),
                "completed": len(completed),
                "processing": len(processing),
                "failed": len(failed),
            },
            "accuracy": {
                "average": round(avg_accuracy, 1),
                "average_speaker_accuracy": round(avg_speaker_accuracy, 1),
                "average_transcription_accuracy": round(avg_transcription_accuracy, 1),
            },
            "duration": {
                "total_seconds": total_duration,
            },
        }
    except Exception as e:
        logger.error(f"Failed to fetch system metrics: {e}")
        try:
            db.rollback()
        except Exception:
            pass
        return {
            "meetings": {"total": 0, "completed": 0, "processing": 0, "failed": 0},
            "accuracy": {
                "average": 0,
                "average_speaker_accuracy": 0,
                "average_transcription_accuracy": 0,
            },
            "duration": {"total_seconds": 0},
        }


@router.post("/clear-cache", dependencies=[Depends(require_admin)])
def clear_cache():
    """Clear application cache data (Python bytecode caches, Next.js build cache).

    Scope is fixed to the project's own ``backend`` package and ``frontend-web/.next/cache`` —
    never the working directory (which would also walk the virtualenv) and never the built
    ``.next`` output itself. Never touches the ChromaDB vectors.
    """
    cleared_items = []

    # Clear Python bytecode caches, only inside this project's backend package
    backend_dir = os.path.join(PROJECT_ROOT, "backend")
    for dirpath, dirnames, _files in os.walk(backend_dir):
        if os.path.basename(dirpath) == "__pycache__":
            try:
                shutil.rmtree(dirpath)
            except Exception as exc:
                logger.debug("Could not remove %s: %s", dirpath, exc)
            dirnames[:] = []
    cleared_items.append("pycache")

    # Clear the Next.js *cache* (not the compiled build) — anchored to the project root
    next_cache_candidates = [
        os.path.join(PROJECT_ROOT, "frontend-web", ".next", "cache"),
        os.path.join(PROJECT_ROOT, "..", "frontend-web", ".next", "cache"),
    ]
    for next_cache in next_cache_candidates:
        if os.path.isdir(next_cache):
            try:
                shutil.rmtree(next_cache)
                cleared_items.append(".next/cache")
                break
            except Exception as exc:
                logger.debug("Could not remove %s: %s", next_cache, exc)

    return {
        "message": "Cache cleared",
        "cleared": cleared_items,
    }


def _check_database(db: Session) -> bool:
    try:
        res = db.execute(text("SELECT 1")).scalar()
        return res == 1
    except Exception:
        logger.warning("Database health check probe failed", exc_info=True)
        return False

"""AI Meeting Intelligence Platform — FastAPI application entry point.

Clean architecture with CORS, startup tasks, and dynamic router discovery.
"""

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api import discover_routers
from backend.config.config import settings
from backend.database import SessionLocal, init_db
from backend.middleware import RequestLoggingMiddleware, SecurityHeadersMiddleware
from backend.repositories import MeetingRepository
from backend.utils.exceptions import MeetingIntelligenceException
from backend.utils.logging_setup import configure_logging

# ─── Configure logging BEFORE any other module uses it ──────────────────────
configure_logging(
    log_level=settings.LOG_LEVEL,
    log_format=getattr(settings, "LOG_FORMAT", "text"),
)
logger = logging.getLogger(__name__)


def _verify_embedding_model_or_fail() -> None:
    """Real model on disk (auto `git lfs pull`), loaded, producing semantic vectors - or a clear error.

    EMBEDDING_MODEL_REQUIRED=true (default): the API refuses to start, so a broken deployment is seen at once instead
    of silently running without semantic search. false: it starts degraded and /system/health reports the reason.
    """
    from backend.services import model_loader
    from backend.services.embedding_model_guard import ensure_embedding_model

    check = ensure_embedding_model(auto_fetch=True, log=lambda m: logger.warning("[Startup] %s", m))
    problem = "" if check.ok else check.summary()
    if not check.ok and model_loader.resolve_local_embedding_model_path():
        # Configured copy is broken but a complete one exists elsewhere (e.g. the local Hugging Face cache): usable.
        logger.warning("[Startup] %s - using another complete local copy instead.", problem)
        problem = ""
    if not problem:
        report = model_loader.validate_embedding_model()          # loads it and embeds a related / unrelated pair
        if report.get("ok"):
            logger.info("[Startup] Embedding model ready: %s (%s-dim, %s)%s", report.get("model_path"),
                        report.get("dimension"), report.get("engine"),
                        " - fetched with git lfs pull" if check.fetched_with_git_lfs else "")
            return
        problem = f"embedding model does not load / embed: {report.get('error')}. FIX: {report.get('fix') or check.fix}"
    if getattr(settings, "EMBEDDING_MODEL_REQUIRED", True):
        logger.critical("[Startup] REFUSING TO START - %s (set EMBEDDING_MODEL_REQUIRED=false to run without "
                        "semantic search / RAG).", problem)
        raise RuntimeError(problem)
    logger.error("[Startup] EMBEDDING MODEL UNUSABLE - starting WITHOUT semantic search / RAG / indexing: %s", problem)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Memory hygiene for small (≈1 GB) servers ─────────────────────────────
    # 1. Cap torch intra-op threads (env caps above cover numpy/BLAS/OMP).
    # 2. Make the GC a little more eager so the large transient buffers from
    #    each pipeline stage (audio arrays, embedding matrices, transcript
    #    lists) are reclaimed promptly instead of accumulating between cycles.
    try:
        import gc as _gc
        _gc.set_threshold(700, 10, 10)
        try:
            import torch  # noqa: WPS433
            torch.set_num_threads(int(os.getenv("AI_MAX_THREADS", "2")))
        except Exception:
            pass
    except Exception:
        pass

    for _d in (
        settings.UPLOAD_DIR,
        os.path.join(settings.UPLOAD_DIR, settings.UPLOAD_CLIPS_DIR),
        settings.NEMO_CACHE_DIR,
    ):
        try:
            os.makedirs(_d, exist_ok=True)
        except Exception as _mk_err:
            logger.error("Could not create required directory %s: %s", _d, _mk_err)

    # Media tooling: log a clear, actionable message if ffmpeg/ffprobe are absent
    # (they are required for audio extraction, transcription chunking, media probe).
    try:
        from backend.utils.media_tools import ensure_ffmpeg_available
        ensure_ffmpeg_available(strict=False)
    except Exception as _ff_err:
        logger.warning("ffmpeg availability check skipped: %s", _ff_err)

    # Multi-worker note: the in-process SSE manager and chat conversation cache
    # are per-worker. Live activity-log streaming and the in-RAM chat cache are
    # only consistent with a single worker. Chat history itself is DB-backed.
    # The embedded ChromaDB directory can be owned by ONE process only, so every
    # worker after the first runs without vector search (see vector_service).
    try:
        _workers = int(os.getenv("UVICORN_WORKERS", os.getenv("WORKERS", "1")) or "1")
    except ValueError:
        _workers = 1
    if _workers > 1:
        logger.warning(
            "Running with %d uvicorn workers. ChromaDB (the vector database) supports one process per "
            "persistent directory, so only one worker gets RAG / similar-meeting search, and real-time "
            "SSE activity-log streaming is per-worker. Use UVICORN_WORKERS=1.",
            _workers,
        )

    # Bind the SSE fan-out to THIS event loop. Pipeline workers publish from background threads
    # via publish_sync(), which can only reach subscribers through a bound loop; without this
    # call every live activity-log update was silently dropped.
    from backend.services.sse_manager import sse_manager

    sse_manager.bind_loop(asyncio.get_running_loop())

    # ── Startup order: config (imported) -> database -> embedding model -> ChromaDB -> backfill -> API
    logger.info("[Startup 1/4] Database")
    init_db()

    # Embedding model: verified on disk (a Git LFS pointer is fetched with `git lfs pull` when possible, the weights
    # are SHA-256 checked against the repository's LFS oid), then LOADED and proven to embed - before ChromaDB and the
    # backfill need it. Done here as well as in the launch scripts because pm2/systemd may start uvicorn directly.
    logger.info("[Startup 2/4] Embedding model verification")
    _verify_embedding_model_or_fail()

    # Vector database: connect to the persistent ChromaDB (CHROMA_PERSIST_DIRECTORY) and verify its collections.
    # Never fatal - if it is unavailable the API still starts and /system/health reports the reason.
    logger.info("[Startup 4/5] ChromaDB")
    try:
        from backend.services.vector_service import vector_service
        vector_service.ensure_initialized()
        logger.info("[Startup] ChromaDB: %s", vector_service.health().get("status"))
        # Meetings without a vector (new, or indexed while the model was missing) are added in the background.
        logger.info("[Startup 5/5] Vector backfill (runs in the background)")
        from backend.services.vector_backfill import start_background_backfill
        start_background_backfill()
    except Exception as _vec_err:
        logger.warning("Vector database initialization skipped: %s", _vec_err)

    # Speaker-diarization model: loaded in the background so the FIRST meeting after a restart does not spend ~36 s
    # loading it (measured). The API is not held up; a meeting arriving meanwhile waits on the same load lock.
    def _preload_diarization_model() -> None:
        try:
            from backend.services.diarization_service import preload_speaker_model

            t0 = time.perf_counter()
            ready = preload_speaker_model()
            logger.info("[Startup] Speaker model preload: %s in %.1fs", "ready" if ready else "not available",
                        time.perf_counter() - t0)
        except Exception as _dia_err:   # never fatal: the first meeting then loads it itself, as before
            logger.warning("Speaker model preload failed (%s: %s); it will load with the first meeting.",
                           type(_dia_err).__name__, _dia_err)

    threading.Thread(target=_preload_diarization_model, name="diarization-preload", daemon=True).start()

    # Auto-recover orphaned meetings using MeetingRepository
    db = None
    try:
        db = SessionLocal()
        repo = MeetingRepository(db)
        orphaned = repo.get_orphaned_processing()
        for m in orphaned:
            m.status = "error"
            m.error_message = "Server restarted before background processing finished. Please re-upload or re-process file."
            repo.update(m)
        if orphaned:
            logger.info(f"Auto-recovered {len(orphaned)} orphaned meeting(s).")
    except Exception as e:
        logger.warning(f"Startup meeting recovery check failed: {e}")
        if db:
            try:
                db.rollback()
            except Exception:
                pass
    finally:
        if db:
            db.close()
    yield

    # Shutdown: release ChromaDB and its directory lock so the next process can take ownership immediately.
    try:
        from backend.services.vector_service import vector_service
        vector_service.close()
    except Exception as _close_err:
        logger.debug("Vector database close skipped: %s", _close_err)


# Conditionally enable/disable Swagger UI based on ENABLE_SWAGGER setting
_docs_url = "/docs" if settings.ENABLE_SWAGGER else None
_redoc_url = "/redoc" if settings.ENABLE_SWAGGER else None
_openapi_url = "/openapi.json" if settings.ENABLE_SWAGGER else None

app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise Production-Ready AI Meeting Intelligence Platform APIs",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=_docs_url,
    redoc_url=_redoc_url,
    openapi_url=_openapi_url,
)

# ─── Global Exception Handlers ──────────────────────────────────────────────────
@app.exception_handler(MeetingIntelligenceException)
async def custom_exception_handler(request: Request, exc: MeetingIntelligenceException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": "Invalid request payload format.",
            # exc.errors() can carry raw exception objects (ctx={"error": ValueError(...)}) that
            # JSONResponse cannot serialise — that turned a plain 422 into a 500.
            "details": jsonable_encoder(exc.errors()),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    msg = exc.detail if isinstance(exc.detail, str) else "HTTP Exception"
    details = exc.detail if not isinstance(exc.detail, str) else None
    error_code = f"HTTP_{exc.status_code}"
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": error_code,
            "message": msg,
            "details": details,
        },
        # Keep response headers set by the raiser (WWW-Authenticate, Retry-After, Allow …).
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"[UnhandledException] {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred. Please consult application logs.",
            "details": str(exc) if getattr(settings, "DEBUG", False) else None,
        },
    )


# ─── Middleware ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list
    or ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=(
        ["GET", "POST", "PUT", "DELETE", "PATCH"] if settings.CORS_ALLOW_CREDENTIALS else ["*"]
    ),
    allow_headers=["*"],
    max_age=600,
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# ─── Dynamic router registration ────────────────────────────────────────────────
for _router in discover_routers():
    app.include_router(_router, prefix="/api/v1")

logger.info(f"Application started. AI mode: {settings.AI_MODE}")


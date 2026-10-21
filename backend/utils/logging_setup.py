"""
Production-Grade Centralized Logging Setup
Call configure_logging() once at application startup (in main.py).

Log Files:
  logs/application.log  - all INFO+ messages
  logs/error.log        - ERROR+ only (fast triage)
  logs/api.log          - HTTP request/response
  logs/ai_pipeline.log  - pipeline, diarization, transcription, LLM
  logs/database.log     - SQLAlchemy / DB messages

Format (text mode):
  2026-08-20 10:15:21.345 | ERROR | module | func | L245 | job=X | meeting=Y | msg

Rotation: Max 50 MB per file, 10 backup files kept.
"""

import contextvars
import logging
import logging.handlers
import os
import sys
from typing import Optional


# Resolve LOG_DIR to an absolute path anchored at the project root so logging
# works regardless of the process working directory (systemd / PM2 / cron).
try:
    from backend.config.config import _resolve_path as _rp
    _LOG_DIR = _rp(os.getenv("LOG_DIR", ""), "logs")
except Exception:
    _base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _log_dir_raw = os.getenv("LOG_DIR", "logs")
    _LOG_DIR = _log_dir_raw if os.path.isabs(_log_dir_raw) else os.path.join(_base, _log_dir_raw)


# Per-pipeline log context. ContextVars, NOT class attributes: meetings are processed concurrently (FastAPI
# BackgroundTasks run in a thread pool), and a shared class-level value made every pipeline stamp its lines with
# whichever meeting set the context last - and one pipeline finishing blanked the others' labels.
_CTX_JOB: contextvars.ContextVar = contextvars.ContextVar("pipeline_job_id", default=None)
_CTX_MEETING: contextvars.ContextVar = contextvars.ContextVar("pipeline_meeting_id", default=None)
_CTX_STAGE: contextvars.ContextVar = contextvars.ContextVar("pipeline_stage", default=None)


class _PipelineContextFilter(logging.Filter):
    @classmethod
    def set_context(cls, job_id=None, meeting_id=None, stage=None):
        """Set the context for the current thread / task; returns tokens that restore the previous values."""
        return (_CTX_JOB.set(job_id), _CTX_MEETING.set(meeting_id), _CTX_STAGE.set(stage))

    @classmethod
    def clear_context(cls):
        _CTX_JOB.set(None)
        _CTX_MEETING.set(None)
        _CTX_STAGE.set(None)

    def filter(self, record):
        record.job_id = _CTX_JOB.get() or "-"
        record.meeting_id = _CTX_MEETING.get() or "-"
        record.pipeline_stage = _CTX_STAGE.get() or "-"
        return True


_TEXT_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-30s | "
    "%(funcName)-25s | L%(lineno)-4d | "
    "job=%(job_id)s | meeting=%(meeting_id)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_SIMPLE_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-30s | %(message)s"


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        import json
        from datetime import datetime, timezone
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "func": record.funcName,
            "line": record.lineno,
            "job_id": getattr(record, "job_id", "-"),
            "meeting_id": getattr(record, "meeting_id", "-"),
            "stage": getattr(record, "pipeline_stage", "-"),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _make_rotating_handler(filename, level, fmt, json_mode=False):
    handler = logging.handlers.RotatingFileHandler(
        filename, maxBytes=50 * 1024 * 1024, backupCount=10, encoding="utf-8",
    )
    handler.setLevel(level)
    handler.addFilter(_PipelineContextFilter())
    if json_mode:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(fmt, datefmt=_DATE_FORMAT))
    return handler


def configure_logging(log_level="INFO", log_format="text"):
    """Configure root logger with rotating file handlers + console.
    Idempotent - safe to call multiple times."""
    root = logging.getLogger()
    if len(root.handlers) > 1:
        return
    level = getattr(logging, log_level.upper(), logging.INFO)
    json_mode = log_format.lower() == "json"
    root.setLevel(level)
    ctx = _PipelineContextFilter()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.addFilter(ctx)
    if json_mode:
        console.setFormatter(_JsonFormatter())
    else:
        console.setFormatter(logging.Formatter(_TEXT_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    # File handlers are best-effort: a logs/ directory that cannot be written (created by root on a
    # previous run, a read-only mount, a different service user) must not stop the whole API from
    # starting - the console handler above is what Docker / PM2 / journald capture anyway.
    file_logging_error = None
    try:
        _configure_file_logging(root, level, json_mode)
    except OSError as exc:
        file_logging_error = exc

    # Silence noisy third-party libs
    for noisy in ("httpx", "httpcore", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    log = logging.getLogger(__name__)
    if file_logging_error is not None:
        log.warning(
            "File logging DISABLED - cannot write to %s (%s). Logging to the console only. "
            "Fix the directory's permissions or point LOG_DIR at a writable path.",
            os.path.abspath(_LOG_DIR), file_logging_error,
        )
    log.info(
        "Logging configured | level=%s | format=%s | log_dir=%s",
        log_level.upper(), log_format, os.path.abspath(_LOG_DIR),
    )


def _configure_file_logging(root: logging.Logger, level: int, json_mode: bool) -> None:
    """Attach the rotating log files under _LOG_DIR. Raises OSError when the directory is unusable."""
    os.makedirs(_LOG_DIR, exist_ok=True)

    # application.log - all messages
    root.addHandler(_make_rotating_handler(
        os.path.join(_LOG_DIR, "application.log"), level, _TEXT_FORMAT, json_mode))

    # error.log - errors only
    root.addHandler(_make_rotating_handler(
        os.path.join(_LOG_DIR, "error.log"), logging.ERROR, _TEXT_FORMAT, json_mode))

    # api.log
    api_h = _make_rotating_handler(
        os.path.join(_LOG_DIR, "api.log"), level, _SIMPLE_FORMAT, json_mode)
    logging.getLogger("backend.middleware").addHandler(api_h)

    # ai_pipeline.log
    pl_h = _make_rotating_handler(
        os.path.join(_LOG_DIR, "ai_pipeline.log"), level, _TEXT_FORMAT, json_mode)
    for n in ("backend.workers", "backend.services.diarization_service",
              "backend.services.transcription_service", "backend.services.llm_workflow",
              "backend.services.audio_service", "backend.services.similarity_service",
              "backend.services.vector_service", "backend.loop_engine"):
        logging.getLogger(n).addHandler(pl_h)

    # database.log
    db_h = _make_rotating_handler(
        os.path.join(_LOG_DIR, "database.log"), level, _SIMPLE_FORMAT, json_mode)
    for n in ("backend.database", "sqlalchemy.engine", "sqlalchemy.pool"):
        logging.getLogger(n).addHandler(db_h)


class PipelineLogContext:
    """Context manager that injects job_id, meeting_id, stage into every log record.

    Usage:
        with PipelineLogContext(job_id=str(meeting.id), meeting_id=str(meeting.id), stage="TRANSCRIPTION"):
            result = transcribe_audio(audio_path)
            logger.info("Transcription complete")  # automatically tagged with context
    """

    def __init__(self, job_id=None, meeting_id=None, stage=None):
        self._job_id = job_id
        self._meeting_id = meeting_id
        self._stage = stage
        self._tokens = None

    def __enter__(self):
        self._tokens = _PipelineContextFilter.set_context(
            job_id=self._job_id, meeting_id=self._meeting_id, stage=self._stage
        )
        return self

    def __exit__(self, *_):
        # Restore what was there before (nested contexts), instead of blanking the labels.
        tokens, self._tokens = self._tokens, None
        if tokens is None:
            _PipelineContextFilter.clear_context()
            return
        for var, token in zip((_CTX_JOB, _CTX_MEETING, _CTX_STAGE), tokens):
            try:
                var.reset(token)
            except ValueError:                   # token from another context (context copied into a worker thread)
                var.set(None)

    def update_stage(self, stage):
        """Update the stage label without leaving the context."""
        self._stage = stage
        _CTX_STAGE.set(stage)

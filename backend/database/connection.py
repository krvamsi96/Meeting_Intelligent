"""Production PostgreSQL database connection, connection pooling, and schema migration."""

import logging
import os
import platform
from contextlib import contextmanager

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from backend.config.config import mask_database_url, settings, with_psycopg2_driver
from backend.models.models import Base

logger = logging.getLogger(__name__)

# ─── Dynamic PostgreSQL connection resolution ────────────────────────────────
# Resolution order (each step is optional and skipped if unavailable):
#   1. EnvironmentSelector — SERVER_* creds, health-checked; on failure it falls
#      back to LOCAL_* creds, then to a generic DATABASE_URL. This is the
#      "server-first, local-fallback" behaviour the deployment expects.
#   2. settings.dynamic_database_url — plain DATABASE_URL / DB_HOST+DB_NAME etc.
# There is deliberately NO built-in credential fallback: connecting to a database
# nobody configured would ship a real password in source control. If neither step
# yields a URL the application refuses to start with an actionable message.

db_url = None
_db_source = "unresolved"
try:
    from backend.config.environment_selector import get_active_config

    _active = get_active_config()
    if getattr(_active, "database_url", None):
        db_url = _active.database_url
        _db_source = f"environment-selector:{_active.environment}"
except Exception as e:  # import cycle, no .env, all environments unreachable, …
    logger.warning(f"[DB] EnvironmentSelector unavailable ({e}); trying settings/fallback.")

if not db_url:
    try:
        _dyn = settings.dynamic_database_url
        if _dyn:
            db_url = _dyn
            _db_source = "settings.dynamic_database_url"
    except Exception:
        pass

if not db_url:
    raise RuntimeError(
        "No database is configured. Set SERVER_DB_* / LOCAL_DB_* (or SERVER_DATABASE_URL / "
        "LOCAL_DATABASE_URL) in .env, or provide DATABASE_URL, or DB_HOST + DB_DATABASE."
    )

db_url = with_psycopg2_driver(str(db_url))
if not db_url.lower().startswith("postgres"):
    raise RuntimeError(
        f"Database URL must specify a PostgreSQL database (got '{mask_database_url(db_url)}'). "
        "SQLite is not supported."
    )
logger.info(f"[DB] Using database from {_db_source}: {mask_database_url(db_url)}")

connect_args = {
    "connect_timeout": settings.DB_CONNECT_TIMEOUT,
}
if platform.system() != "Windows":
    connect_args.update({
        "keepalives": 1,
        "keepalives_idle": settings.DB_TCP_KEEPALIVES_IDLE,        # 60 s default
        "keepalives_interval": settings.DB_TCP_KEEPALIVES_INTERVAL, # 10 s default
        "keepalives_count": settings.DB_TCP_KEEPALIVES_COUNT,       # 5 probes default
    })
if settings.DB_STATEMENT_TIMEOUT_MS > 0:
    connect_args["options"] = f"-c statement_timeout={settings.DB_STATEMENT_TIMEOUT_MS}"

engine_kwargs = {
    "pool_size": settings.DB_POOL_SIZE,
    "max_overflow": settings.DB_MAX_OVERFLOW,
    "pool_timeout": settings.DB_POOL_TIMEOUT,
    "pool_pre_ping": True,
    "pool_recycle": settings.DB_POOL_RECYCLE,
    "connect_args": connect_args,
    "echo": getattr(settings, "DB_ECHO", False),
}

try:
    # engine_kwargs MUST be applied: pool_pre_ping recovers connections that a
    # firewall / NAT / DB restart silently dropped (without it the first query
    # after an idle period raises "server closed the connection unexpectedly" /
    # "SSL SYSCALL error: EOF detected" on a Linux server), pool_recycle caps
    # connection age, and connect_args carries the TCP keepalives that keep the
    # session alive across the 10-15 min LLM waits in the pipeline.
    engine = create_engine(db_url, **engine_kwargs)
    logger.info(f"PostgreSQL database engine configured for: {mask_database_url(str(db_url))}")
except Exception as e:
    logger.error(f"Failed to create PostgreSQL database engine: {e}")
    raise RuntimeError(f"PostgreSQL engine initialization failed: {e}") from e

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_database_target() -> dict:
    """Which PostgreSQL this process is connected to, and whether that is a FALLBACK. Never contains credentials.

    ``AI_MODE=auto`` tries the SERVER_* database first and, if it does not answer at boot, quietly continues on the
    LOCAL_* one. On a production server that would mean meetings landing in a different database than the one
    configured, so the choice is exposed here (``GET /system/health``) and logged as an ERROR at startup.
    ``AI_MODE=server`` never falls back: the application refuses to start instead.
    """
    environment = _db_source.split(":", 1)[1] if _db_source.startswith("environment-selector:") else _db_source
    server_configured = bool((os.getenv("SERVER_DATABASE_URL") or os.getenv("SERVER_DB_HOST") or "").strip())
    fallback = server_configured and environment != "server"
    url = engine.url
    target = {
        "environment": environment,
        "host": url.host,
        "port": url.port,
        "database": url.database,
        "server_database_configured": server_configured,
        "fallback_active": fallback,
        "ai_mode": (settings.AI_MODE or "").strip().lower(),
    }
    if fallback:
        target["warning"] = (
            "The configured SERVER database was unreachable at startup, so this process is using the "
            f"'{environment}' database instead. Set AI_MODE=server on a production server so it refuses to start "
            "rather than fall back."
        )
    return target


_target_at_boot = get_database_target()
if _target_at_boot["fallback_active"]:
    if platform.system() == "Windows":
        logger.info(
            "[DB] Local development environment: automatically using '%s' at %s:%s/%s.",
            _target_at_boot["environment"], _target_at_boot["host"], _target_at_boot["port"], _target_at_boot["database"],
        )
    else:
        logger.warning(
            "[DB] FALLBACK DATABASE ACTIVE: SERVER_* is configured but unreachable; using '%s' at %s:%s/%s. "
            "Meetings will NOT be stored in the production database. Set AI_MODE=server to fail fast instead.",
            _target_at_boot["environment"], _target_at_boot["host"], _target_at_boot["port"], _target_at_boot["database"],
        )
else:
    logger.info(
        "[DB] Active database: %s at %s:%s/%s (AI_MODE=%s)",
        _target_at_boot["environment"], _target_at_boot["host"], _target_at_boot["port"], _target_at_boot["database"],
        _target_at_boot["ai_mode"],
    )


@contextmanager
def pipeline_db_session():
    """Context manager for background pipeline workers.

    Guarantees the DB connection is ALWAYS returned to the pool after the
    ``with`` block completes, even if an exception is raised.  This prevents
    pool exhaustion when pipeline workers crash mid-execution.

    Usage::

        with pipeline_db_session() as db:
            meeting = db.query(Meeting).filter(...).first()
            meeting.status = "processing"
            db.commit()
        # ← connection is returned to pool here; safe to call slow LLM code next

    For long-running pipelines (LLM calls that take 10-15 min) open the
    session in short-lived segments::

        with pipeline_db_session() as db:
            # fast DB work only
            meeting = db.query(Meeting).filter(...).first()
            meeting.status = "processing"
            db.commit()

        # LLM/Whisper calls here — no DB session held during this block

        with pipeline_db_session() as db:
            # write results
            meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
            meeting.status = "completed"
            db.commit()
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# A session that has FLUSHED changes in its current transaction has written to PostgreSQL without committing; rolling
# it back would silently undo that work. SQLAlchemy does not expose "flushed but uncommitted", so it is tracked here:
# set by any flush that emitted changes, cleared when the session's outermost transaction ends (commit or rollback).
_FLUSHED_KEY = "has_flushed_uncommitted_writes"


@event.listens_for(Session, "after_flush")
def _remember_flushed_writes(session, flush_context):
    session.info[_FLUSHED_KEY] = True


@event.listens_for(Session, "after_transaction_end")
def _forget_flushed_writes(session, transaction):
    if transaction.parent is None:
        session.info.pop(_FLUSHED_KEY, None)


def release_read_transaction(db) -> bool:
    """End the session's transaction and return its connection to the pool - but only if it holds NO writes.

    Use it between the database reads of a computation and the slow, database-free work that follows (embedding,
    ChromaDB, an LLM call), so no PostgreSQL transaction sits "idle in transaction" through that work. Returns False
    (and does nothing) when the session has pending inserts / deletes / modified objects OR has already flushed
    changes it has not committed: rolling those back would silently discard them, so the caller keeps its
    transaction instead. Objects loaded earlier are expired and are transparently re-read on next access.
    """
    if db.new or db.deleted or any(db.is_modified(obj) for obj in db.dirty) or db.info.get(_FLUSHED_KEY):
        return False
    db.rollback()
    return True


def get_engine_pool_status() -> dict:
    """Return current connection pool statistics for health monitoring.

    Tolerates pool implementations that don't expose the QueuePool metrics
    (e.g. StaticPool / NullPool) so /system/health never 500s.
    """
    pool = engine.pool

    def _stat(name):
        fn = getattr(pool, name, None)
        try:
            return fn() if callable(fn) else None
        except Exception:
            return None

    size = _stat("size")
    max_overflow = getattr(pool, "_max_overflow", None)
    return {
        "pool_class": type(pool).__name__,
        "pool_size": size,
        "checked_in": _stat("checkedin"),
        "checked_out": _stat("checkedout"),
        "overflow": _stat("overflow"),
        "max_overflow": max_overflow,
        "total_capacity": (size + max_overflow)
        if isinstance(size, int) and isinstance(max_overflow, int)
        else None,
    }



def init_db():
    """Initialize PostgreSQL database: create tables and run schema migrations.

    Retries the connection with exponential back-off when PostgreSQL is not yet
    ready (e.g. the database service starts after this process on server boot).
    The retry behaviour is controlled by two optional env-vars / settings:

        DB_CONNECT_RETRIES          int   max attempts (default 5)
        DB_CONNECT_RETRY_DELAY_S    float initial wait in seconds (default 2.0)

    Waits are: delay, delay*2, delay*4 … capped at DB_CONNECT_RETRY_MAX_DELAY_S
    (default 60 s).  If all retries are exhausted the original exception is
    re-raised, which will cause the app to exit and PM2 to restart it.
    """
    import time
    import sqlalchemy.exc

    max_retries   = settings.DB_CONNECT_RETRIES
    initial_delay = settings.DB_CONNECT_RETRY_DELAY_S
    max_delay     = settings.DB_CONNECT_RETRY_MAX_DELAY_S
    db_label      = mask_database_url(db_url)

    last_err: Exception = RuntimeError("init_db() never attempted a connection")

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                f"Initializing PostgreSQL database at {db_label} "
                f"(attempt {attempt}/{max_retries})"
            )
            Base.metadata.create_all(bind=engine)
            conn = engine.connect()
            try:
                _run_migrations(conn)
            except Exception as mig_err:
                logger.error(f"Database migration error: {mig_err}", exc_info=True)
                raise mig_err
            finally:
                conn.close()
            logger.info("PostgreSQL database initialized and auto-migrated successfully.")
            return  # ← success — exit immediately

        except (
            sqlalchemy.exc.OperationalError,
            sqlalchemy.exc.DBAPIError,
        ) as db_err:
            last_err = db_err
            logger.error(
                f"PostgreSQL connection / initialization failed: {db_err}",
                exc_info=True,
            )
            if attempt < max_retries:
                wait = min(initial_delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(
                    f"[init_db] PostgreSQL not reachable. "
                    f"Retrying in {wait:.0f}s "
                    f"(attempt {attempt}/{max_retries})…"
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"[init_db] PostgreSQL still not reachable after "
                    f"{max_retries} attempts. Giving up."
                )

        except Exception as other_err:
            # Non-connection errors (e.g., migration bugs) fail immediately —
            # retrying would not help.
            logger.error(
                f"PostgreSQL initialization failed with non-retryable error: {other_err}",
                exc_info=True,
            )
            raise RuntimeError(
                f"PostgreSQL database error: {other_err}. SQLite fallback is disabled."
            ) from other_err

    raise RuntimeError(
        f"PostgreSQL database error: {last_err}. SQLite fallback is disabled."
    ) from last_err



def _rollback_quietly(conn) -> None:
    """Reset the connection's transaction after a failed migration step."""
    try:
        conn.rollback()
    except Exception as rb_err:  # pragma: no cover - best effort
        logger.debug(f"Migration rollback notice: {rb_err}")


def _run_migrations(conn):
    """Auto-migrate PostgreSQL database tables to add missing columns and backfill metadata.

    All migration steps are idempotent and non-blocking — inspecting existing schema
    before executing DDL to avoid unnecessary locks.
    """
    inspector = inspect(conn)

    # 1. Dynamic schema synchronization for all ORM models
    existing_tables = set(inspector.get_table_names())
    for table_name, table in Base.metadata.tables.items():
        if table_name in existing_tables:
            existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
            for column in table.columns:
                if column.name not in existing_cols:
                    col_type = column.type.compile(engine.dialect)
                    logger.info(f"Database Migration: Adding column '{column.name}' ({col_type}) to table '{table_name}'.")
                    try:
                        conn.execute(text(
                            f'ALTER TABLE "{table_name}" ADD COLUMN "{column.name}" {col_type}'
                        ))
                        conn.commit()
                    except Exception as col_err:
                        _rollback_quietly(conn)
                        logger.warning(f"Could not add column '{column.name}' to '{table_name}': {col_err}")

    # 2. Table-specific default values, constraints, and data backfills
    if "meetings" in existing_tables:
        cols_by_name = {c["name"]: c for c in inspector.get_columns("meetings")}
        m_id_col = cols_by_name.get("meeting_id")
        if m_id_col and "varchar" not in str(m_id_col.get("type", "")).lower() and "text" not in str(m_id_col.get("type", "")).lower():
            try:
                conn.execute(text("ALTER TABLE meetings ALTER COLUMN meeting_id TYPE VARCHAR"))
                conn.commit()
            except Exception as alter_err:
                _rollback_quietly(conn)
                logger.debug(f"Column type alter notice: {alter_err}")

        d_id_col = cols_by_name.get("display_id")
        if d_id_col and "varchar" not in str(d_id_col.get("type", "")).lower() and "text" not in str(d_id_col.get("type", "")).lower():
            try:
                conn.execute(text("ALTER TABLE meetings ALTER COLUMN display_id TYPE VARCHAR"))
                conn.commit()
            except Exception as alter_err:
                _rollback_quietly(conn)
                logger.debug(f"Column type alter notice: {alter_err}")

        # Sanitize legacy BVC-/MTG- prefixed IDs ONLY when such rows exist
        try:
            legacy_count_row = conn.execute(text("""
                SELECT COUNT(*) FROM meetings
                WHERE bharatvc_meeting_id LIKE 'BVC%' OR bharatvc_meeting_id LIKE 'MTG%'
                   OR meeting_id LIKE 'BVC%' OR meeting_id LIKE 'MTG%'
                   OR display_id LIKE 'BVC%' OR display_id LIKE 'MTG%'
            """)).scalar()
            if legacy_count_row and int(legacy_count_row) > 0:
                logger.info(
                    f"Database Migration: Sanitizing {legacy_count_row} row(s) with legacy BVC-/MTG- prefixed IDs."
                )
                conn.execute(text("""
                    UPDATE meetings
                    SET
                      bharatvc_meeting_id = REGEXP_REPLACE(bharatvc_meeting_id, '[^0-9]', '', 'g'),
                      meeting_id = REGEXP_REPLACE(meeting_id, '[^0-9]', '', 'g'),
                      display_id = REGEXP_REPLACE(display_id, '[^0-9]', '', 'g')
                    WHERE bharatvc_meeting_id LIKE 'BVC%' OR bharatvc_meeting_id LIKE 'MTG%'
                       OR meeting_id LIKE 'BVC%' OR meeting_id LIKE 'MTG%'
                       OR display_id LIKE 'BVC%' OR display_id LIKE 'MTG%'
                """))
                conn.commit()
            else:
                logger.debug("Database Migration: No legacy-prefixed IDs found — skipping sanitization.")
        except Exception as e:
            _rollback_quietly(conn)
            logger.debug(f"PostgreSQL regex sanitize notice: {e}")

        # Backfill bharatvc_meeting_id, meeting_id, display_id if null using clean numeric IDs
        try:
            conn.execute(text("""
                UPDATE meetings 
                SET bharatvc_meeting_id = COALESCE(meeting_id, CAST(id AS VARCHAR))
                WHERE bharatvc_meeting_id IS NULL OR bharatvc_meeting_id = ''
            """))
            conn.execute(text("""
                UPDATE meetings 
                SET meeting_id = COALESCE(bharatvc_meeting_id, CAST(id AS VARCHAR)),
                    display_id = COALESCE(bharatvc_meeting_id, meeting_id, CAST(id AS VARCHAR))
                WHERE display_id IS NULL OR display_id = '' OR meeting_id IS NULL OR meeting_id = ''
            """))
            conn.commit()
        except Exception as e:
            _rollback_quietly(conn)
            logger.debug(f"ID backfill notice: {e}")

        # Auto-populate null metadata fields
        try:
            conn.execute(text("""
                UPDATE meetings 
                SET title = COALESCE(NULLIF(title, ''), REPLACE(REPLACE(filename, '_', ' '), '-', ' '))
                WHERE title IS NULL OR title = ''
            """))
            conn.execute(text("""
                UPDATE meetings 
                SET agenda = :agenda_text
                WHERE agenda IS NULL OR agenda = ''
            """), {"agenda_text": settings.DEFAULT_AGENDA_TEXT})

            conn.execute(text("""
                UPDATE meetings 
                SET meeting_date = TO_CHAR(upload_time, 'YYYY-MM-DD')
                WHERE meeting_date IS NULL OR meeting_date = '' OR meeting_date = 'Not Specified'
            """))
            conn.execute(text("""
                UPDATE meetings 
                SET meeting_time = TO_CHAR(upload_time, 'HH24:MI')
                WHERE meeting_time IS NULL OR meeting_time = '' OR meeting_time = 'Not Specified'
            """))

            conn.execute(text("""
                UPDATE meetings 
                SET participants = :default_participants
                WHERE participants IS NULL OR participants = ''
            """), {"default_participants": settings.DEFAULT_PARTICIPANTS_TEXT})
            conn.execute(text("""
                UPDATE meetings 
                SET project_name = :default_name
                WHERE project_name IS NULL OR project_name = ''
            """), {"default_name": settings.APP_NAME})
            conn.commit()
        except Exception as e:
            _rollback_quietly(conn)
            logger.debug(f"Metadata backfill notice: {e}")

    if "chat_sessions" in inspector.get_table_names():
        existing_chat_cols = {c["name"] for c in inspector.get_columns("chat_sessions")}
        if "sources" not in existing_chat_cols:
            logger.info("Database Migration: Adding column 'sources' to chat_sessions table.")
            try:
                conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN sources TEXT"))
                conn.commit()
            except Exception as cs_err:
                _rollback_quietly(conn)
                logger.debug(f"chat_sessions sources column notice: {cs_err}")

    if "transcript_segments" in existing_tables:
        try:
            conn.execute(text("""
                UPDATE transcript_segments
                SET raw_text = text
                WHERE raw_text IS NULL OR raw_text = ''
            """))
            conn.commit()
        except Exception as seg_err:
            _rollback_quietly(conn)
            logger.debug(f"Transcript segments backfill notice: {seg_err}")

    # Ensure main_task column exists in action_items
    if "action_items" in existing_tables:
        existing_ai_cols = {c["name"] for c in inspector.get_columns("action_items")}
        if "main_task" not in existing_ai_cols:
            logger.info("Database Migration: Adding column 'main_task' (TEXT, NULL) to action_items.")
            try:
                conn.execute(text("ALTER TABLE action_items ADD COLUMN main_task TEXT DEFAULT NULL"))
                conn.commit()
            except Exception as mt_err:
                _rollback_quietly(conn)
                logger.warning(f"Could not add main_task column: {mt_err}")
        else:
            try:
                conn.execute(text(
                    "UPDATE action_items SET main_task = NULL "
                    "WHERE main_task IN ('Unknown','General Task','N/A','Not specified','Unspecified')"
                ))
                conn.commit()
                logger.debug("action_items main_task: forbidden defaults cleaned up.")
            except Exception as backfill_err:
                _rollback_quietly(conn)
                logger.debug(f"main_task backfill notice: {backfill_err}")

    # Ensure analysis_results.meeting_id has a UNIQUE constraint
    if "analysis_results" in existing_tables:
        existing_uqs = {uq["name"] for uq in inspector.get_unique_constraints("analysis_results")}
        if "uq_analysis_results_meeting_id" not in existing_uqs:
            try:
                conn.execute(text("""
                    ALTER TABLE analysis_results
                    ADD CONSTRAINT uq_analysis_results_meeting_id UNIQUE (meeting_id)
                """))
                conn.commit()
                logger.info("Database Migration: Added UNIQUE constraint on analysis_results.meeting_id.")
            except Exception as uq_err:
                _rollback_quietly(conn)
                logger.debug(f"analysis_results UNIQUE constraint notice: {uq_err}")

    # Ensure transcript_segments.meeting_id is NOT NULL
    if "transcript_segments" in existing_tables:
        t_cols = {c["name"]: c for c in inspector.get_columns("transcript_segments")}
        if "meeting_id" in t_cols and t_cols["meeting_id"].get("nullable", True):
            try:
                conn.execute(text("""
                    ALTER TABLE transcript_segments
                    ALTER COLUMN meeting_id SET NOT NULL
                """))
                conn.commit()
                logger.info("Database Migration: Set transcript_segments.meeting_id to NOT NULL.")
            except Exception as nn_err:
                _rollback_quietly(conn)
                logger.debug(f"transcript_segments NOT NULL notice: {nn_err}")

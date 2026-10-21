"""Shared API dependencies."""

import logging

from fastapi import Depends
from sqlalchemy.orm import Session

from backend.database.connection import SessionLocal
from backend.repositories import MeetingRepository

logger = logging.getLogger(__name__)


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        # Roll back the transaction if an unhandled exception occurs so the
        # connection is returned to the pool in a clean state. Without this,
        # PostgreSQL keeps the session in an "aborted transaction" state and
        # every subsequent query in the same request raises
        # psycopg2.errors.InFailedSqlTransaction.
        db.rollback()
        raise
    finally:
        db.close()


def get_meeting_repository(db: Session = Depends(get_db)) -> MeetingRepository:
    return MeetingRepository(db)


def end_read_transaction(db: Session) -> None:
    """Finish the transaction a read opened, returning its connection to the pool, WITHOUT closing the session.

    The first query of a request opens a PostgreSQL transaction that stays "idle in transaction" until commit,
    rollback or close - through anything slow the endpoint does next (streaming a 500 MB upload, downloading a URL).
    Call this right after the last read that precedes such a wait; the session can be used again afterwards.
    """
    try:
        db.rollback()
    except Exception as exc:
        # The connection is unusable (dropped by the server / firewall): discard it so the next use gets a fresh one.
        logger.warning("Ending the read transaction failed (%s: %s); discarding the connection.", type(exc).__name__, exc)
        db.invalidate()


def release_session_before_background_task(db: Session) -> None:
    """Close the request's session before a BackgroundTask is queued.

    FastAPI keeps a ``yield`` dependency (``get_db``) open until the background tasks of the request have FINISHED.
    A meeting pipeline runs for minutes, so a session that was still open after ``repo.create()`` (its response reads
    ``meeting.id`` after the commit, which starts a new transaction) stayed "idle in transaction" and pinned a pooled
    connection for the whole run - measured at 422.7 s of a 422.6 s pipeline. Read what the response needs first.
    """
    db.close()

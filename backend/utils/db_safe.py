"""Optional database work that must never poison the caller's transaction.

PostgreSQL aborts a whole transaction when ANY statement in it fails: every later statement then raises
``InFailedSqlTransaction`` ("current transaction is aborted, commands ignored until end of transaction block") until a
ROLLBACK. Several code paths wrapped an *optional* query in ``try/except`` and carried on with the same session -
the real error was demoted to a log line, the next statement (often the lazy ``meeting.ground_truth`` load) raised
``InFailedSqlTransaction``, and THAT error is what ended up stored on the meeting.

``optional_db`` runs the work in a SAVEPOINT: a failure rolls back to the savepoint only, the surrounding
transaction (and every row already written in it) stays valid and usable. It is deliberately NOT ``db.rollback()``,
which would also discard everything the pipeline had written so far.

Ground truth is optional by design (accuracy is only measured when a reference exists), so its lookups go through
``has_ground_truth`` / ``load_ground_truth``: a failed lookup means "no ground truth" - reference-free evaluation
proceeds and the accuracy fields stay null.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_failed_transaction_error(exc: BaseException) -> bool:
    """True for PostgreSQL's 25P02 ``InFailedSqlTransaction`` (a previous statement already aborted the transaction)."""
    orig = getattr(exc, "orig", exc)
    return getattr(orig, "pgcode", None) == "25P02" or type(orig).__name__ == "InFailedSqlTransaction"


def describe_db_error(exc: BaseException) -> str:
    """One line: the driver's error class and its first message line (not SQLAlchemy's multi-line SQL dump)."""
    orig = getattr(exc, "orig", exc)
    first = (str(orig).strip().splitlines() or [""])[0]
    return f"{type(orig).__name__}: {first}"[:300]


def optional_db(db, fn: Callable[[], T], *, default: Optional[T] = None, label: str = "optional query") -> Optional[T]:
    """Run ``fn()`` (queries / writes on ``db``) as optional work; return ``default`` if it fails.

    * Pending ORM state is flushed BEFORE the savepoint, outside the try: a failing REQUIRED write is a real error
      and propagates - it is never mistaken for a failed optional query.
    * The work runs in ``db.begin_nested()``: its own failure is rolled back to the savepoint and logged with the
      database's own error text, and the caller's transaction continues untouched.
    * If the transaction was ALREADY aborted by someone else (``InFailedSqlTransaction``) the error is re-raised:
      returning a default there would hide the poisoned session that every following statement will trip over.
    """
    db.flush()
    try:
        with db.begin_nested():
            return fn()
    except Exception as exc:  # noqa: BLE001 - optional by definition; the cause is logged, not hidden
        if is_failed_transaction_error(exc):
            raise
        logger.warning(
            "[OptionalQuery] %s failed (%s) - continuing without it; the transaction was rolled back to its "
            "savepoint and is still usable.", label, describe_db_error(exc),
        )
        return default


def has_ground_truth(db, meeting_id: Any) -> bool:
    """Is there a ground-truth reference for this meeting? A failed lookup is answered with False (and logged)."""
    from backend.models.models import GroundTruth

    found = optional_db(
        db,
        lambda: db.query(GroundTruth.id).filter(GroundTruth.meeting_id == meeting_id).first(),
        default=None,
        label=f"ground-truth lookup for meeting {meeting_id}",
    )
    return found is not None


def load_ground_truth(db, meeting_id: Any):
    """The meeting's GroundTruth row, or None - also None when the lookup itself fails."""
    from backend.models.models import GroundTruth

    return optional_db(
        db,
        lambda: db.query(GroundTruth).filter(GroundTruth.meeting_id == meeting_id).first(),
        default=None,
        label=f"ground-truth load for meeting {meeting_id}",
    )


def ground_truth_of(db, meeting) -> Any:
    """``meeting.ground_truth`` without the unguarded lazy load.

    The relationship value is used when it is already in memory (no query). Otherwise the row is fetched through
    ``load_ground_truth`` so a failed lookup yields None instead of raising out of an evaluation or a response.
    """
    try:
        from sqlalchemy import inspect as _sa_inspect

        if "ground_truth" not in _sa_inspect(meeting).unloaded:
            return meeting.ground_truth
    except Exception:  # noqa: BLE001 - not an ORM instance (e.g. a plain stand-in object): use the attribute
        return getattr(meeting, "ground_truth", None)
    meeting_id = getattr(meeting, "id", None)
    if db is None or meeting_id is None:
        return None
    return load_ground_truth(db, meeting_id)

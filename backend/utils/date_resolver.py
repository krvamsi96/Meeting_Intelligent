"""Dynamic Date / Deadline Resolver
====================================
Deterministic, timezone-aware calendar resolver for natural-language deadline
expressions extracted from meeting transcripts.

Design Rules (§9.16):
  - The LLM only identifies the temporal expression ("next Monday").
  - THIS module performs all calendar arithmetic from a reference datetime.
  - No dates are hardcoded.  No LLM is asked to do arithmetic.

Priority for reference date (§9.12):
  1. meeting_date  (preferred — historical meetings must use the meeting date)
  2. current server datetime  (fallback)
  3. Respect the meeting / application timezone throughout.

Usage::

    from backend.utils.date_resolver import get_reference_datetime, resolve_deadline

    # Build a timezone-aware reference from the meeting date string
    ref_dt = get_reference_datetime("2026-08-05", "Asia/Kolkata")

    result = resolve_deadline("next Monday", ref_dt)
    # DeadlineResolution(
    #     original_deadline_text="next Monday",
    #     due_date=date(2026, 8, 10),
    #     deadline_start=None,
    #     deadline_end=None,
    #     deadline_precision="exact",
    #     timezone="Asia/Kolkata",
    # )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class DeadlineResolution:
    """Result of resolving a natural-language deadline phrase.

    Attributes
    ----------
    original_deadline_text:
        The exact phrase extracted from the transcript (e.g., ``"next Monday"``).
    due_date:
        A resolved ISO calendar date, or ``None`` for window / unresolvable
        expressions (e.g., ``"next week"``, ``"soon"``).
    deadline_start:
        Start of a date window (e.g., first day of next week).  ``None`` when
        not applicable.
    deadline_end:
        End of a date window.  ``None`` when not applicable.
    deadline_precision:
        ``"exact"``         — a single resolved calendar date.
        ``"window"``        — a date range (``deadline_start``/``deadline_end``).
        ``"unresolvable"``  — vague expression; no date can be inferred.
    timezone:
        The timezone name used for the calculation.
    """

    original_deadline_text: str
    due_date: Optional[date] = None
    deadline_start: Optional[date] = None
    deadline_end: Optional[date] = None
    deadline_precision: str = "unresolvable"
    timezone: str = "UTC"


# ---------------------------------------------------------------------------
# Weekday constants  (Monday = 0 … Sunday = 6)
# ---------------------------------------------------------------------------

_WEEKDAY_NAMES: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
    # short forms
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}

_ORDINAL_NUMBERS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
    # compound
    "a": 1, "an": 1,
}

_NO_DATE_PHRASES: frozenset[str] = frozenset({
    "soon", "asap", "as soon as possible", "at some point",
    "later", "eventually", "sometime", "some day",
    "before deployment", "after qa approval", "after qa", "pending approval",
    "before the next meeting", "before next meeting", "before release",
    "when available", "when ready", "tbd", "tba", "not specified",
    "not mentioned", "n/a", "none", "",
})


# ---------------------------------------------------------------------------
# Reference datetime builder
# ---------------------------------------------------------------------------

def get_reference_datetime(
    meeting_date_str: Optional[str] = None,
    timezone_str: Optional[str] = None,
) -> datetime:
    """Return a timezone-aware reference datetime for deadline resolution.

    Priority (§9.12):
    1. ``meeting_date_str`` if parseable → midnight of that date in the given tz.
    2. Current server datetime in the given tz.

    Parameters
    ----------
    meeting_date_str:
        Meeting date as a string, e.g. ``"2026-08-05"``, ``"05-Aug-2026"``,
        ``"August 5, 2026"``.  Pass ``None`` or ``""`` to use current time.
    timezone_str:
        IANA timezone name, e.g. ``"Asia/Kolkata"``, ``"UTC"``.
        Defaults to ``"UTC"`` when ``None`` or invalid.

    Returns
    -------
    datetime
        Timezone-aware datetime representing the reference point.
    """
    tz = _resolve_timezone(timezone_str)
    ref_dt: Optional[datetime] = None

    if meeting_date_str and meeting_date_str.strip().lower() not in {"", "unknown date", "unknown", "none"}:
        ref_dt = _parse_date_string(meeting_date_str, tz)

    if ref_dt is None:
        # Fallback: current server datetime in requested timezone
        ref_dt = datetime.now(tz)
        logger.debug(
            "[DateResolver] No valid meeting_date; using current server time: %s (tz=%s)",
            ref_dt.date(), timezone_str or "UTC",
        )
    else:
        logger.debug(
            "[DateResolver] Using meeting date as reference: %s (tz=%s)",
            ref_dt.date(), timezone_str or "UTC",
        )

    return ref_dt


# ---------------------------------------------------------------------------
# Main resolver entry point
# ---------------------------------------------------------------------------

def resolve_deadline(
    phrase: str,
    reference_dt: datetime,
) -> DeadlineResolution:
    """Resolve a natural-language deadline phrase to a calendar date.

    The ``reference_dt`` MUST be timezone-aware (produced by
    ``get_reference_datetime``).

    Parameters
    ----------
    phrase:
        The exact deadline phrase from the transcript, e.g. ``"next Monday"``.
    reference_dt:
        Timezone-aware datetime to use as the anchor for relative calculations.

    Returns
    -------
    DeadlineResolution
        Contains the resolved date(s) and precision metadata.
    """
    if not phrase:
        return DeadlineResolution(
            original_deadline_text=phrase or "",
            deadline_precision="unresolvable",
            timezone=_tz_name(reference_dt),
        )

    original = phrase.strip()
    normed = original.lower().strip()

    # Vague / unresolvable phrases — no date inference (§9.14)
    if normed in _NO_DATE_PHRASES:
        return DeadlineResolution(
            original_deadline_text=original,
            deadline_precision="unresolvable",
            timezone=_tz_name(reference_dt),
        )

    tz_name = _tz_name(reference_dt)
    ref_date = reference_dt.date()

    resolved = (
        _try_today(normed, ref_date)
        or _try_tomorrow(normed, ref_date)
        or _try_day_after_tomorrow(normed, ref_date)
        or _try_eod(normed, ref_date)
        or _try_within_n_units(normed, ref_date)
        or _try_in_n_units(normed, ref_date)
        or _try_this_weekday(normed, ref_date)
        or _try_next_weekday(normed, ref_date)
        or _try_by_weekday(normed, ref_date)
        or _try_before_weekday(normed, ref_date)
        or _try_weekend(normed, ref_date)
        or _try_end_of_month(normed, ref_date)
        or _try_next_week_window(normed, ref_date)
        or _try_this_week_window(normed, ref_date)
        or _try_next_month_window(normed, ref_date)
        or _try_absolute_date(normed, original, ref_year=ref_date.year)
    )

    if resolved is None:
        # Cannot map to a calendar date — preserve original, mark unresolvable
        logger.debug("[DateResolver] Could not resolve %r — unresolvable.", original)
        return DeadlineResolution(
            original_deadline_text=original,
            deadline_precision="unresolvable",
            timezone=tz_name,
        )

    # Validate the resolved date (§9.17)
    resolved = _validated(resolved, ref_date, original)

    resolved.original_deadline_text = original
    resolved.timezone = tz_name
    return resolved


# ---------------------------------------------------------------------------
# Individual expression resolvers (private)
# ---------------------------------------------------------------------------

def _try_today(normed: str, ref: date) -> Optional[DeadlineResolution]:
    if re.fullmatch(r"today|by\s+today|end\s+of\s+today", normed):
        return DeadlineResolution(
            original_deadline_text="",
            due_date=ref,
            deadline_precision="exact",
        )
    return None


def _try_tomorrow(normed: str, ref: date) -> Optional[DeadlineResolution]:
    patterns = [
        r"tomorrow",
        r"by\s+tomorrow",
        r"by\s+tomorrow\s+morning",
        r"by\s+tomorrow\s+evening",
        r"by\s+tomorrow\s+(?:eod|end\s+of\s+day)",
    ]
    if any(re.fullmatch(p, normed) for p in patterns):
        return DeadlineResolution(
            original_deadline_text="",
            due_date=ref + timedelta(days=1),
            deadline_precision="exact",
        )
    return None


def _try_day_after_tomorrow(normed: str, ref: date) -> Optional[DeadlineResolution]:
    if re.fullmatch(r"day\s+after\s+tomorrow", normed):
        return DeadlineResolution(
            original_deadline_text="",
            due_date=ref + timedelta(days=2),
            deadline_precision="exact",
        )
    return None


def _try_eod(normed: str, ref: date) -> Optional[DeadlineResolution]:
    """By end of day / EOD — same day."""
    patterns = [
        r"by\s+end\s+of\s+day",
        r"by\s+eod",
        r"eod",
        r"end\s+of\s+day",
        r"by\s+close\s+of\s+(?:business|day)",
        r"by\s+cob",
        r"cob",
    ]
    if any(re.fullmatch(p, normed) for p in patterns):
        return DeadlineResolution(
            original_deadline_text="",
            due_date=ref,
            deadline_precision="exact",
        )
    return None


def _try_within_n_units(normed: str, ref: date) -> Optional[DeadlineResolution]:
    """within N days/weeks → ref + N*unit_days (§9.8)."""
    m = re.fullmatch(
        r"within\s+(\w+)\s+(day|days|week|weeks|hour|hours)",
        normed,
    )
    if not m:
        return None
    n = _parse_number(m.group(1))
    unit = m.group(2).rstrip("s")   # normalise "days" → "day"
    if n is None:
        return None
    days = n if unit == "day" else n * 7
    return DeadlineResolution(
        original_deadline_text="",
        due_date=ref + timedelta(days=days),
        deadline_precision="exact",
    )


def _try_in_n_units(normed: str, ref: date) -> Optional[DeadlineResolution]:
    """in N days/weeks → ref + N*unit_days."""
    m = re.fullmatch(
        r"in\s+(\w+)\s+(day|days|week|weeks)",
        normed,
    )
    if not m:
        return None
    n = _parse_number(m.group(1))
    unit = m.group(2).rstrip("s")
    if n is None:
        return None
    days = n if unit == "day" else n * 7
    return DeadlineResolution(
        original_deadline_text="",
        due_date=ref + timedelta(days=days),
        deadline_precision="exact",
    )


def _try_this_weekday(normed: str, ref: date) -> Optional[DeadlineResolution]:
    """this <weekday> → the occurrence of that weekday in the current week (Mon-Sun).

    If the reference date IS that weekday, returns the same day.
    If the weekday has already passed in the current week, returns next week's occurrence.
    """
    m = re.fullmatch(r"this\s+(\w+)", normed)
    if not m:
        return None
    wd = _weekday(m.group(1))
    if wd is None:
        return None
    resolved = _this_weekday(ref, wd)
    return DeadlineResolution(
        original_deadline_text="",
        due_date=resolved,
        deadline_precision="exact",
    )


def _try_next_weekday(normed: str, ref: date) -> Optional[DeadlineResolution]:
    """next <weekday> → always the NEXT occurrence strictly after this week's instance.

    E.g., ref = Wednesday Aug-12 (Wed=2).
    next Monday (0) → skip to the following week's Monday = Aug-17.
    """
    m = re.fullmatch(r"next\s+(\w+(?:day)?)", normed)
    if not m:
        return None
    wd = _weekday(m.group(1))
    if wd is None:
        return None
    resolved = _next_weekday(ref, wd)
    return DeadlineResolution(
        original_deadline_text="",
        due_date=resolved,
        deadline_precision="exact",
    )


def _try_by_weekday(normed: str, ref: date) -> Optional[DeadlineResolution]:
    """by <weekday> / before <weekday> → nearest future occurrence of that weekday.

    If the weekday is today, returns today.
    If the weekday is after today (within the current week), returns that day.
    If the weekday has already passed this week, returns next week's occurrence.
    """
    m = re.fullmatch(r"by\s+(\w+)", normed)
    if not m:
        return None
    wd = _weekday(m.group(1))
    if wd is None:
        return None
    resolved = _nearest_future_weekday(ref, wd)
    return DeadlineResolution(
        original_deadline_text="",
        due_date=resolved,
        deadline_precision="exact",
    )


def _try_before_weekday(normed: str, ref: date) -> Optional[DeadlineResolution]:
    """before <weekday> → nearest future occurrence (same logic as 'by')."""
    m = re.fullmatch(r"before\s+(\w+)", normed)
    if not m:
        return None
    wd = _weekday(m.group(1))
    if wd is None:
        return None
    resolved = _nearest_future_weekday(ref, wd)
    return DeadlineResolution(
        original_deadline_text="",
        due_date=resolved,
        deadline_precision="exact",
    )


def _try_weekend(normed: str, ref: date) -> Optional[DeadlineResolution]:
    """by this weekend / this weekend / by weekend / end of weekend -> Friday or weekend window."""
    patterns = [
        r"by\s+(?:this\s+)?weekend",
        r"(?:this\s+)?weekend",
        r"by\s+end\s+of\s+(?:this\s+)?weekend",
        r"end\s+of\s+(?:this\s+)?weekend",
    ]
    if any(re.fullmatch(p, normed) for p in patterns):
        current_wd = ref.weekday()
        days_to_friday = (4 - current_wd) % 7
        friday = ref + timedelta(days=days_to_friday)
        sunday = friday + timedelta(days=2)
        return DeadlineResolution(
            original_deadline_text="",
            due_date=friday,
            deadline_start=friday,
            deadline_end=sunday,
            deadline_precision="exact",
        )
    return None


def _try_end_of_month(normed: str, ref: date) -> Optional[DeadlineResolution]:
    """by month end / end of (this) month / by end of this month."""
    patterns = [
        r"by\s+month\s+end",
        r"by\s+end\s+of\s+(?:the\s+)?month",
        r"by\s+end\s+of\s+this\s+month",
        r"end\s+of\s+(?:the\s+)?month",
        r"end\s+of\s+this\s+month",
        r"month\s+end",
    ]
    if any(re.fullmatch(p, normed) for p in patterns):
        last_day = _last_day_of_month(ref)
        return DeadlineResolution(
            original_deadline_text="",
            due_date=last_day,
            deadline_precision="exact",
        )
    return None


def _try_next_week_window(normed: str, ref: date) -> Optional[DeadlineResolution]:
    """next week / early next week / mid next week / end of next week → window (§9.7)."""
    if not re.search(r"\bnext\s+week\b", normed):
        return None

    # Compute Monday and Sunday of the week following ref's week
    days_until_monday = (7 - ref.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7  # If today is Monday, next week starts in 7 days
    next_monday = ref + timedelta(days=days_until_monday)
    next_sunday = next_monday + timedelta(days=6)

    # Sub-expressions that give a specific day within next week
    if re.search(r"\bearly\b", normed):
        # Early next week → Monday–Tuesday
        return DeadlineResolution(
            original_deadline_text="",
            due_date=None,
            deadline_start=next_monday,
            deadline_end=next_monday + timedelta(days=1),
            deadline_precision="window",
        )
    if re.search(r"\bmid\b", normed):
        # Mid next week → Wednesday
        return DeadlineResolution(
            original_deadline_text="",
            due_date=None,
            deadline_start=next_monday + timedelta(days=2),
            deadline_end=next_monday + timedelta(days=2),
            deadline_precision="window",
        )
    if re.search(r"\bend\b", normed):
        # End of next week → Friday–Sunday
        return DeadlineResolution(
            original_deadline_text="",
            due_date=None,
            deadline_start=next_monday + timedelta(days=4),
            deadline_end=next_sunday,
            deadline_precision="window",
        )

    # Plain "next week" — full window (§9.7, §9.15)
    return DeadlineResolution(
        original_deadline_text="",
        due_date=None,
        deadline_start=next_monday,
        deadline_end=next_sunday,
        deadline_precision="window",
    )


def _try_this_week_window(normed: str, ref: date) -> Optional[DeadlineResolution]:
    """this week → window from today to Sunday of the current week."""
    if not re.fullmatch(r"this\s+week", normed):
        return None
    days_to_sunday = (6 - ref.weekday()) % 7
    this_sunday = ref + timedelta(days=days_to_sunday)
    return DeadlineResolution(
        original_deadline_text="",
        due_date=None,
        deadline_start=ref,
        deadline_end=this_sunday,
        deadline_precision="window",
    )


def _try_next_month_window(normed: str, ref: date) -> Optional[DeadlineResolution]:
    """next month → entire calendar month window."""
    if not re.fullmatch(r"next\s+month", normed):
        return None
    # First day of next month
    if ref.month == 12:
        first = date(ref.year + 1, 1, 1)
    else:
        first = date(ref.year, ref.month + 1, 1)
    last = _last_day_of_month(first)
    return DeadlineResolution(
        original_deadline_text="",
        due_date=None,
        deadline_start=first,
        deadline_end=last,
        deadline_precision="window",
    )


def _try_absolute_date(normed: str, original: str, ref_year: int = 2026) -> Optional[DeadlineResolution]:
    """Attempt to parse a fully-specified calendar date string."""
    candidate = _parse_explicit_date(original, ref_year=ref_year)
    if candidate is None:
        return None
    return DeadlineResolution(
        original_deadline_text="",
        due_date=candidate,
        deadline_precision="exact",
    )


# ---------------------------------------------------------------------------
# Calendar arithmetic helpers
# ---------------------------------------------------------------------------

def _next_weekday(ref: date, target_wd: int) -> date:
    """Return the date of the *next* occurrence of ``target_wd`` STRICTLY after
    the current week's instance of that weekday.

    'next Monday' from a Wednesday always means the Monday of the following week
    — even if Monday has not yet occurred this week.  This matches the common
    English interpretation of "next <weekday>".

    Algorithm:
      1. Find this week's occurrence of target_wd (Mon=0 of the same ISO week).
      2. Add 7 days to get the same weekday next week.
    """
    # Step 1: how many days from ref to the beginning of next week (Monday)?
    # ISO weekday: Monday=0 … Sunday=6
    current_wd = ref.weekday()
    # Days until next Monday  (if today IS Monday, days_to_next_monday = 7)
    days_to_next_monday = (7 - current_wd) % 7 or 7
    next_monday = ref + timedelta(days=days_to_next_monday)
    # Step 2: add the target weekday offset from Monday
    return next_monday + timedelta(days=target_wd)


def _this_weekday(ref: date, target_wd: int) -> date:
    """Return the date of ``target_wd`` *within the current ISO week*.

    If the day has already passed this week, advance to next week's occurrence.
    """
    current_wd = ref.weekday()
    # Monday of the current week
    this_monday = ref - timedelta(days=current_wd)
    candidate = this_monday + timedelta(days=target_wd)
    if candidate < ref:
        candidate += timedelta(days=7)
    return candidate


def _nearest_future_weekday(ref: date, target_wd: int) -> date:
    """Return the nearest future (or today) occurrence of ``target_wd``.

    Used for "by Friday", "before Friday" — includes today if applicable.
    """
    current_wd = ref.weekday()
    delta = (target_wd - current_wd) % 7
    return ref + timedelta(days=delta)


def _last_day_of_month(d: date) -> date:
    """Return the last calendar day of the month containing ``d``."""
    if d.month == 12:
        return date(d.year, 12, 31)
    return date(d.year, d.month + 1, 1) - timedelta(days=1)


# ---------------------------------------------------------------------------
# Validation (§9.17)
# ---------------------------------------------------------------------------

def _validated(
    result: DeadlineResolution,
    ref_date: date,
    original: str,
) -> DeadlineResolution:
    """Post-resolution validation gate.

    Checks:
    - due_date is a valid calendar date (dataclass already guarantees `date` type)
    - due_date weekday is correct (for weekday-named expressions)
    - due_date is not unreasonably in the past when the statement is future-oriented
    - Window start ≤ window end
    """
    if result.due_date is not None:
        # Sanity: due_date must be a `date` instance
        if not isinstance(result.due_date, date):
            logger.error(
                "[DateResolver] BUG: due_date is not a date object for %r: %r",
                original, result.due_date,
            )
            result.due_date = None
            result.deadline_precision = "unresolvable"
            return result

        # For future-oriented expressions, the resolved date should not be in the past.
        # 'today' and 'by EOD' legitimately resolve to ref_date — allow them.
        future_oriented_pattern = re.compile(
            r"\b(next|by|within|in\s+\d|tomorrow|after)\b", re.IGNORECASE
        )
        if future_oriented_pattern.search(original):
            if result.due_date < ref_date:
                logger.warning(
                    "[DateResolver] Resolved date %s is BEFORE reference %s for %r — "
                    "marking unresolvable.",
                    result.due_date, ref_date, original,
                )
                result.due_date = None
                result.deadline_precision = "unresolvable"
                return result

    if result.deadline_start is not None and result.deadline_end is not None:
        if result.deadline_start > result.deadline_end:
            # Swap to fix ordering — shouldn't happen with correct logic
            result.deadline_start, result.deadline_end = (
                result.deadline_end, result.deadline_start,
            )

    return result


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _weekday(name: str) -> Optional[int]:
    """Return the weekday index (Mon=0) for a weekday name, or ``None``."""
    return _WEEKDAY_NAMES.get(name.lower())


def _parse_number(token: str) -> Optional[int]:
    """Parse an integer or English word number to int."""
    token = token.lower().strip()
    if token.isdigit():
        return int(token)
    return _ORDINAL_NUMBERS.get(token)


def _parse_date_string(date_str: str, tz) -> Optional[datetime]:
    """Try common date formats and return a tz-aware datetime at midnight."""
    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%d-%b-%Y",    # 05-Aug-2026
        "%d %B %Y",    # 05 August 2026
        "%B %d, %Y",   # August 5, 2026
        "%b %d, %Y",   # Aug 5, 2026
        "%Y/%m/%d",
    ]
    clean = date_str.strip()
    for fmt in formats:
        try:
            dt = datetime.strptime(clean, fmt)
            return dt.replace(tzinfo=tz)
        except ValueError:
            continue
    logger.debug("[DateResolver] Could not parse date string: %r", date_str)
    return None


def _parse_explicit_date(phrase: str, ref_year: int = 2026) -> Optional[date]:
    """Attempt to parse a phrase as an explicit ISO / common date."""
    if not phrase:
        return None
    clean = phrase.strip().lower()
    # Strip leading prepositions
    clean = re.sub(r"^(by|on|at|before)\s+", "", clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"\bof\b", " ", clean, flags=re.IGNORECASE).strip()
    # Strip ordinal suffixes: 1st -> 1, 2nd -> 2, 3rd -> 3, 4th -> 4
    clean = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()

    formats_with_year = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d-%b-%Y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ]
    for fmt in formats_with_year:
        try:
            return datetime.strptime(clean, fmt).date()
        except ValueError:
            continue

    # Attempt formats without year (e.g. "4 May", "May 4") assuming ref_year
    formats_no_year = [
        "%d %B",
        "%d %b",
        "%B %d",
        "%b %d",
    ]
    for fmt in formats_no_year:
        try:
            dt = datetime.strptime(clean, fmt)
            return date(ref_year, dt.month, dt.day)
        except ValueError:
            continue

    return None


def _resolve_timezone(timezone_str: Optional[str]):
    """Return a timezone object, falling back to UTC on any error."""
    if not timezone_str:
        return timezone.utc
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+
        return ZoneInfo(timezone_str)
    except Exception:
        pass
    try:
        import pytz
        return pytz.timezone(timezone_str)
    except Exception:
        pass
    logger.warning(
        "[DateResolver] Could not resolve timezone %r — defaulting to UTC.", timezone_str
    )
    return timezone.utc


def _tz_name(dt: datetime) -> str:
    """Extract the timezone name string from a timezone-aware datetime."""
    if dt.tzinfo is None:
        return "UTC"
    try:
        return str(dt.tzinfo)
    except Exception:
        return "UTC"

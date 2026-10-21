"""Deadlines the model extracts must survive validation and resolve to calendar dates.

Found by running the platform for real: AIRAWAT returned ``"due_date": "October 10"`` / ``"next Wednesday"`` exactly as
the extraction prompt asks, but the validator only read ``deadline`` / ``original_deadline_text`` - so every deadline was
discarded and every action item was reported as "Not specified".
"""

from datetime import date

import pytest

from backend.services import action_item_service as ais
from backend.utils.date_resolver import get_reference_datetime

TX = "Speaker_2: I will send the detailed cost breakdown to Priya by next Wednesday. Anita will deliver the final designs by October 10."


def _raw(**over):
    item = {"title": "Send detailed cost breakdown", "description": "Hosting costs", "due_date": "next Wednesday",
            "assigned_to": "Speaker_2", "priority": "", "evidence": "I will send the detailed cost breakdown to Priya by next Wednesday."}
    item.update(over)
    return item


@pytest.mark.parametrize("field", ["due_date", "deadline", "original_deadline_text", "due", "deadline_text"])
def test_the_deadline_phrase_is_kept_whichever_field_carries_it(field):
    raw = _raw()
    raw.pop("due_date")
    raw[field] = "next Wednesday"
    (item,) = ais.validate_action_items([raw], TX)
    assert item["deadline"] == "next Wednesday" and item["original_deadline_text"] == "next Wednesday"


def test_no_deadline_stays_empty_and_is_not_invented():
    (item,) = ais.validate_action_items([_raw(due_date="")], TX)
    assert item["deadline"] == "" and item["deadline_type"] == "not_specified"
    (item,) = ais.validate_action_items([_raw(due_date="Not specified")], TX)
    assert item["deadline"] == ""


def test_a_kept_deadline_resolves_to_a_calendar_date_from_the_meeting_date():
    reference = get_reference_datetime("2026-09-20", "Asia/Kolkata")               # a Sunday
    items = ais.validate_action_items([_raw(due_date="October 10", evidence="Anita will deliver the final designs by October 10.")], TX)
    resolved = ais.resolve_action_item_dates(items, reference)
    assert resolved[0]["due_date"] == date(2026, 10, 10)

"""Agenda -> decision mapping scenarios: a decision is mapped only where the transcript supports the relationship.

Complements test_agenda_decision_grounding.py (hardcoded/generic text, evidence checks) with the mapping-level cases:
direct match, unrelated decision, several agenda items, no agenda, no decisions, ambiguous evidence, insufficient
transcript, and "similarity is a supporting signal, not proof".

The model's answer is data; no AIRAWAT, no embedding model, no database.
"""

from types import SimpleNamespace

import pytest

from backend.services import agenda_service
from backend.services import final_aggregation_service as fas
from backend.services import llm_workflow as wf

TRANSCRIPT = (
    "Speaker_1: Let us start with the database migration. We decided to adopt PostgreSQL and move the staging "
    "environment first.\n"
    "Speaker_2: On the budget, finance said the cloud hosting cost stays flat this quarter and nothing changes.\n"
    "Speaker_1: For the release date, we agreed to ship the new dashboard on November 3.\n"
    "Speaker_2: The vendor contract is still open, we will discuss it next week.\n"
)

MIGRATION = {
    "agenda": "Database migration",
    "key_decision": "Adopt PostgreSQL and migrate staging first",
    "evidence": "We decided to adopt PostgreSQL and move the staging environment first.",
}
RELEASE = {
    "agenda": "Release date",
    "key_decision": "Ship the new dashboard on November 3",
    "evidence": "we agreed to ship the new dashboard on November 3.",
}


@pytest.fixture(autouse=True)
def _model_free(monkeypatch):
    monkeypatch.setattr(
        wf, "compute_batch_match_scores",
        lambda pairs: [{"match_score": 50.0, "confidence_level": "Fair Match", "model_name": "stub"} for _ in pairs],
    )
    monkeypatch.setattr(agenda_service, "deduplicate_semantic", lambda texts, threshold=None: list(range(len(texts))))


def _validate(items, transcript=TRANSCRIPT):
    return wf.validate_and_format_agenda_key_decisions(items, transcript=transcript)


def _rows(validated):
    """What the pipeline stores, as the objects the aggregation reads."""
    return [
        SimpleNamespace(
            agenda_id_str=i["agenda_id"], agenda_order=i["agenda_order"], agenda=i["agenda"],
            key_decision=i["key_decision"], decision_grounded=i["decision_grounded"], match_score=i["match_score"],
            evidence=i["evidence"], speaker=i["speaker"], timestamp=i["timestamp"], reason=i["reason"],
            confidence=i["confidence"], agenda_type=i["agenda_type"], agenda_source=i["source"],
        )
        for i in validated
    ]


def _mapping(items, transcript=TRANSCRIPT):
    return fas._build_agenda_decision_mapping(_rows(_validate(items, transcript)))


# ── direct match ──────────────────────────────────────────────────────────────────────────────────

def test_a_decision_stated_in_the_transcript_is_mapped_with_its_evidence():
    (m,) = _mapping([MIGRATION])
    assert m["has_decision"] is True and m["is_grounded"] is True
    assert m["agenda_title"] == "Database migration" and m["decision"] == MIGRATION["key_decision"]
    assert m["evidence"] == MIGRATION["evidence"]


def test_several_agenda_items_each_keep_their_own_decision():
    out = _mapping([MIGRATION, RELEASE])
    assert [(m["agenda_title"], m["decision"]) for m in out] == [
        ("Database migration", MIGRATION["key_decision"]), ("Release date", RELEASE["key_decision"])]
    assert out[0]["evidence"] != out[1]["evidence"]


# ── no forced mapping ─────────────────────────────────────────────────────────────────────────────

def test_an_agenda_item_without_a_decision_is_listed_but_not_mapped():
    (m,) = _mapping([{"agenda": "Vendor contract", "key_decision": "", "evidence": ""}])
    assert m["has_decision"] is False and m["is_grounded"] is False
    assert "decision" not in m and "evidence" not in m and "match_score" not in m


def test_a_decision_with_a_quote_that_is_not_in_the_transcript_is_dropped_but_the_agenda_stays():
    invented = {"agenda": "Budget", "key_decision": "Increase the hosting budget by 20 percent",
                "evidence": "Everyone voted to increase the hosting budget by twenty percent."}
    (m,) = _mapping([invented])
    assert m["agenda_title"] == "Budget" and m["has_decision"] is False and "evidence" not in m


def test_a_discussion_or_deferral_is_not_a_decision_even_with_a_real_quote():
    deferral = {"agenda": "Vendor contract", "key_decision": "Discuss the vendor contract next week",
                "evidence": "The vendor contract is still open, we will discuss it next week."}
    (m,) = _mapping([deferral])
    assert m["has_decision"] is False


def test_no_agenda_means_an_empty_mapping_not_a_generic_one():
    assert _validate([]) == [] and fas._build_agenda_decision_mapping([]) == []
    assert _validate([{"agenda": "General Discussion", "key_decision": ""}]) == []


def test_no_decisions_at_all_leaves_every_item_unmapped():
    out = _mapping([{"agenda": "Database migration", "key_decision": ""},
                    {"agenda": "Release date", "key_decision": "No formal decision recorded."}])
    assert [m["has_decision"] for m in out] == [False, False]
    decisions = fas._build_decisions(_rows(_validate([{"agenda": "Database migration", "key_decision": ""}])))
    assert decisions == []


# ── ambiguous: one quote cannot support two relationships ────────────────────────────────────────

def test_one_quote_cited_by_two_agenda_items_supports_only_the_one_it_belongs_to():
    quote = "We decided to adopt PostgreSQL and move the staging environment first."
    a = {"agenda": "Database migration", "key_decision": "Adopt PostgreSQL", "evidence": quote}
    b = {"agenda": "Hosting budget", "key_decision": "Move staging first", "evidence": quote}
    out = _mapping([a, b])
    mapped = [m for m in out if m["has_decision"]]
    assert [m["agenda_title"] for m in mapped] == ["Database migration"]
    assert [m["agenda_title"] for m in out] == ["Database migration", "Hosting budget"]   # the topic itself stays


def test_the_same_quote_for_the_same_fit_goes_to_the_first_item_deterministically():
    quote = "we agreed to ship the new dashboard on November 3."
    a = {"agenda": "Dashboard release", "key_decision": "Ship on November 3", "evidence": quote}
    b = {"agenda": "Dashboard launch", "key_decision": "Ship the dashboard November 3", "evidence": quote}
    first = _mapping([a, b])
    second = _mapping([a, b])
    assert first == second and [m["has_decision"] for m in first] == [True, False]


# ── insufficient transcript evidence ──────────────────────────────────────────────────────────────

def test_a_decision_cannot_be_supported_by_an_empty_or_unrelated_transcript():
    out = _mapping([MIGRATION], transcript="Speaker_1: Hello. Speaker_2: Hi, how are you doing today?")
    assert all(m["has_decision"] is False for m in out)       # (the topic itself is not in that transcript either)


def test_a_content_free_transcript_produces_no_agenda_or_decisions_at_all(monkeypatch):
    monkeypatch.setattr(wf.model_router, "get_llm", lambda **kw: (_ for _ in ()).throw(AssertionError("no LLM call")))
    out = wf.process_meeting_report("Speaker_1: hi there ok", "2026-10-12", "10:00", "00:00:20")
    assert out["agenda_key_decisions"] == [] and out["structured_action_items"] == []
    assert out["summary"].startswith("Not Available")


# ── similarity is a supporting signal, never the proof ───────────────────────────────────────────

def test_a_high_similarity_score_cannot_create_a_mapping_without_evidence(monkeypatch):
    monkeypatch.setattr(wf, "compute_batch_match_scores",
                        lambda pairs: [{"match_score": 97.0, "confidence_level": "Excellent Match", "model_name": "stub"}
                                       for _ in pairs])
    no_quote = {"agenda": "Release date", "key_decision": "Ship the new dashboard on November 3", "evidence": ""}
    (m,) = _mapping([no_quote])
    assert m["has_decision"] is False and "match_score" not in m and "confidence_level" not in m


def test_a_low_similarity_does_not_remove_a_supported_decision_it_only_drops_the_label(monkeypatch):
    monkeypatch.setattr(wf, "compute_batch_match_scores",
                        lambda pairs: [{"match_score": 12.0, "confidence_level": "Not Reliable", "model_name": "stub"}
                                       for _ in pairs])
    (m,) = _mapping([MIGRATION])
    assert m["has_decision"] is True and m["match_score"] == 12.0 and "confidence_level" not in m


@pytest.mark.parametrize("score,label", [(97.0, "Excellent Match"), (88.0, "High Match"), (72.0, "Good Match"),
                                         (50.0, "Fair Match")])
def test_the_mapping_label_comes_from_the_one_shared_threshold_function(monkeypatch, score, label):
    monkeypatch.setattr(wf, "compute_batch_match_scores",
                        lambda pairs: [{"match_score": score, "confidence_level": "x", "model_name": "stub"} for _ in pairs])
    (m,) = _mapping([MIGRATION])
    assert m["confidence_level"] == label == agenda_service.confidence_label(score)


def test_scores_below_the_reliability_threshold_carry_no_label():
    assert agenda_service.confidence_label(agenda_service.MATCH_THRESHOLD - 0.1) == "Not Reliable"
    assert not agenda_service.is_reliable_match(agenda_service.MATCH_THRESHOLD - 0.1)
    assert not agenda_service.is_reliable_match(None)

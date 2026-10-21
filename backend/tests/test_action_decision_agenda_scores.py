"""Action Item Quality, Decision Quality and Agenda Coverage: reference-free, explainable, based on the meeting's own output.

Each score is checked against defects seen when the platform was run for real: a decision list that contained
individual commitments ("Ravi will share the schedule ..."), action items whose own evidence states a deadline that
was never captured, and items no one ever said.
"""

import json

import pytest

from backend.evaluation import quality_scores as qs
from backend.evaluation import reference_free_engine as rfe


@pytest.fixture(autouse=True)
def lexical(monkeypatch):
    """Deterministic: no embedding model, every check uses its lexical path."""
    monkeypatch.setattr(rfe, "_embed_batch", lambda texts: [])
    monkeypatch.setattr(rfe, "_embed_one", lambda text: [])


TX = "\n".join([
    "Speaker_1: Good morning. Today we need to agree on the website redesign timeline, the budget, and the launch date.",
    "Speaker_2: On the timeline, engineering needs six weeks after the designs are final. I suggest we lock the designs by October tenth.",
    "Speaker_1: Agreed. Decision: the design freeze date is October tenth. Anita will deliver the final designs by then.",
    "Speaker_2: Hosting will cost about six thousand dollars. I will send the detailed cost breakdown to Priya by next Wednesday.",
    "Speaker_1: Good. Decision: the budget is approved at forty thousand dollars including hosting.",
    "Speaker_2: Launch is targeted for November twenty fifth if there are no scope changes. I will prepare the deployment plan by October twentieth.",
    "Speaker_1: Decision: launch is targeted for November twenty fifth. Thank you everyone, meeting adjourned.",
])

GOOD_ACTIONS = [
    {"task": "Deliver the final designs", "owner": "Anita", "deadline": "October tenth",
     "evidence": "Anita will deliver the final designs by then."},
    {"task": "Send the detailed cost breakdown", "owner": "Speaker_2", "deadline": "next Wednesday",
     "evidence": "I will send the detailed cost breakdown to Priya by next Wednesday."},
    {"task": "Prepare the deployment plan", "owner": "Speaker_2", "deadline": "October twentieth",
     "evidence": "I will prepare the deployment plan by October twentieth."},
]
GOOD_DECISIONS = [
    {"decision": "The design freeze date is October tenth", "evidence": "Decision: the design freeze date is October tenth."},
    {"decision": "The budget is approved at forty thousand dollars including hosting",
     "evidence": "Decision: the budget is approved at forty thousand dollars including hosting."},
    {"decision": "Launch is targeted for November twenty fifth", "evidence": "Decision: launch is targeted for November twenty fifth."},
]
TOPICS = ["Website redesign timeline", "Website redesign budget", "Website launch date"]


# ── action item quality ───────────────────────────────────────────────────────


def test_well_grounded_action_items_score_high_with_no_issues():
    out = qs.action_item_quality(GOOD_ACTIONS, TX)
    assert out["score"] >= 90 and out["label"] == "Excellent" and out["issues"] == []
    assert out["components"]["grounding"] == 100.0 and out["components"]["deadline"] == 100.0
    assert out["signals"]["action_item_count"] == 3 and out["signals"]["with_named_owner"] == 1


def test_an_item_nobody_said_is_flagged_and_caps_the_score():
    invented = [{"task": "Order new servers from the hardware vendor", "owner": "Zorro", "deadline": "Not specified", "evidence": ""}]
    out = qs.action_item_quality(invented, TX)
    assert out["score"] <= 50
    assert any("not supported by the transcript" in i for i in out["issues"])
    assert any("'Zorro'" in i and "never mentioned" in i for i in out["issues"])        # a hallucinated assignee


def test_a_deadline_stated_in_the_evidence_but_not_captured_is_reported():
    """The live defect: 'by next Wednesday' was in the evidence, yet deadline came out 'Not specified'."""
    missed = [dict(GOOD_ACTIONS[1], deadline="Not specified")]
    captured = [GOOD_ACTIONS[1]]
    a, b = qs.action_item_quality(missed, TX), qs.action_item_quality(captured, TX)
    assert any("states a deadline in its own evidence but none was captured" in i for i in a["issues"])
    assert a["score"] < b["score"]


def test_a_deadline_that_appears_nowhere_is_not_trusted():
    made_up = [dict(GOOD_ACTIONS[0], deadline="the fourth of July")]
    assert any("not found in the transcript" in i for i in qs.action_item_quality(made_up, TX)["issues"])


def test_no_deadline_and_none_stated_is_not_penalised():
    plain = [{"task": "Review the plan", "owner": "Anita", "deadline": "Not specified", "evidence": "Anita will deliver the final designs by then."}]
    plain[0]["evidence"] = "We should review the plan"
    out = qs.action_item_quality(plain, TX)
    assert out["components"]["deadline"] is None                 # unmeasurable, dropped, not padded


def test_duplicate_action_items_are_reported():
    out = qs.action_item_quality([GOOD_ACTIONS[0], dict(GOOD_ACTIONS[0])], TX)
    assert out["signals"]["duplicates"] == 1 and any("repeat another one" in i for i in out["issues"])


def test_no_action_items_is_not_applicable_and_notes_missed_commitments():
    assert qs.action_item_quality([], "Speaker_1: We reviewed the servers and everything is fine.")["label"] == "Not Applicable"
    out = qs.action_item_quality([], TX)                         # dated commitments exist, none extracted
    assert out["score"] is None and out["label"] == "Not Applicable"
    assert any("read like dated commitments" in i for i in out["issues"])


# ── decision quality ──────────────────────────────────────────────────────────


def test_real_decisions_score_high():
    out = qs.decision_quality(GOOD_DECISIONS, TX, GOOD_ACTIONS)
    assert out["score"] >= 90 and out["issues"] == [] and out["signals"]["commitment_like"] == 0


def test_an_individual_commitment_is_not_a_decision():
    """The live defect: 'Ravi will share the integration schedule ...' was listed under decisions."""
    listed = GOOD_DECISIONS + [{"decision": "Ravi will share the integration schedule by the end of this month.",
                                "evidence": "Ravi will share the integration schedule by the end of this month."}]
    out = qs.decision_quality(listed, TX, GOOD_ACTIONS)
    assert out["signals"]["commitment_like"] == 1
    assert any("reads like an individual commitment" in i for i in out["issues"])
    assert out["score"] < qs.decision_quality(GOOD_DECISIONS, TX, GOOD_ACTIONS)["score"]


def test_a_decision_that_duplicates_an_action_item_is_reported():
    dup = [{"decision": "Anita will deliver the final designs", "evidence": "Anita will deliver the final designs by then."}]
    out = qs.decision_quality(dup, TX, GOOD_ACTIONS)
    assert any("duplicates an action item" in i for i in out["issues"])


def test_a_decision_nobody_made_is_flagged():
    out = qs.decision_quality([{"decision": "The company will acquire a competitor next quarter", "evidence": ""}], TX)
    assert out["score"] <= 50 and any("not supported by the transcript" in i for i in out["issues"])


def test_no_decisions_is_not_applicable_but_a_missed_stated_decision_is_noted():
    quiet = "Speaker_1: This is a status briefing. The cluster was upgraded and there were no incidents."
    assert qs.decision_quality([], quiet) == {"score": None, "label": "Not Applicable", "issues": [],
                                              "reason": "no decisions were identified for this meeting"}
    out = qs.decision_quality([], TX)
    assert out["score"] is None and any("state a decision but none was extracted" in i for i in out["issues"])


def test_the_no_decision_placeholder_does_not_count_as_a_decision():
    only_placeholder = [{"key_decision": "No formal decision recorded.", "evidence": ""}]
    assert qs.decision_quality(only_placeholder, TX)["label"] == "Not Applicable"


# ── agenda coverage ───────────────────────────────────────────────────────────


def test_topics_that_are_discussed_are_grounded_and_cover_the_meeting():
    out = qs.agenda_coverage(TOPICS, TX)
    assert out["components"]["grounding"] >= 60 and out["components"]["distinctness"] == 100.0
    assert out["signals"]["topic_count"] == 3


def test_an_invented_topic_is_flagged_and_scores_lower():
    invented = TOPICS + ["Quarterly hiring plan for the finance department"]
    a, b = qs.agenda_coverage(invented, TX), qs.agenda_coverage(TOPICS, TX)
    assert any("barely discussed" in i for i in a["issues"])
    assert a["components"]["grounding"] < b["components"]["grounding"]


def test_repeated_topics_are_reported():
    out = qs.agenda_coverage(["Website launch date", "Website launch date planning"], TX)
    assert out["signals"]["duplicates"] == 1 and any("repeat another one" in i for i in out["issues"])


def test_no_agenda_is_not_applicable():
    assert qs.agenda_coverage([], TX)["label"] == "Not Applicable"


# ── the assembled block ───────────────────────────────────────────────────────


def _walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k).lower()
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def test_the_metrics_block_carries_the_three_new_scores_and_no_ground_truth_metric():
    segs = [{"speaker": "Speaker_1", "text": ln.split(": ", 1)[1], "start_time": i * 10.0, "end_time": i * 10.0 + 9.0}
            for i, ln in enumerate(TX.splitlines())]
    block = qs.build_quality_metrics(segments=segs, summary_text="The team set the design freeze, budget and launch date.",
                                     transcript_text=TX, agenda_topics=TOPICS, rag_evaluations=[], assistant_answer_count=0,
                                     audio_duration_sec=70.0, action_items=GOOD_ACTIONS, decisions=GOOD_DECISIONS)
    for name in ("action_item_quality", "decision_quality", "agenda_coverage"):
        assert isinstance(block[name]["score"], float) and block[name]["label"] and isinstance(block[name]["issues"], list)
    forbidden = {"wer", "cer", "der", "rouge", "rouge1", "rouge2", "rougel", "f1", "bertscore"}
    assert forbidden.isdisjoint(set(_walk(block)))
    json.dumps(block)


def test_without_items_the_new_scores_are_not_applicable_never_a_number():
    block = qs.build_quality_metrics(segments=[], summary_text="", transcript_text=TX, agenda_topics=[],
                                     rag_evaluations=[], assistant_answer_count=0)
    assert block["action_item_quality"]["score"] is None or block["action_item_quality"]["label"] in ("Not Applicable", "Error")
    assert block["agenda_coverage"]["label"] == "Not Applicable"


def test_one_meetings_items_never_leak_into_anothers_score():
    """Pure functions: scoring meeting B after meeting A cannot be influenced by A."""
    other_tx = "Speaker_1: We reviewed the servers.\nSpeaker_2: The cluster was upgraded with no downtime."
    a = qs.action_item_quality(GOOD_ACTIONS, TX)["score"]
    b = qs.action_item_quality(GOOD_ACTIONS, other_tx)["score"]          # A's items against B's transcript
    assert b < a
    assert qs.action_item_quality(GOOD_ACTIONS, TX)["score"] == a


def test_the_real_service_reads_only_the_given_meetings_rows(monkeypatch):
    """compute_quality_metrics passes THIS meeting's action items / decisions / agenda, keyed by its primary key."""
    from types import SimpleNamespace

    from backend.services import quality_metrics_service as svc

    captured = {}
    monkeypatch.setattr(svc.qs, "build_quality_metrics", lambda **kw: captured.update(kw) or {})
    monkeypatch.setattr(svc, "_load_rag_evaluations", lambda db, pk: ([], 0))

    class Q:
        def __init__(self, rows):
            self.rows = rows

        def filter(self, *a, **k):
            return self

        def order_by(self, *a, **k):
            return self

        def all(self):
            return self.rows

    rows = {
        "TranscriptSegment": [SimpleNamespace(speaker="Speaker_1", text="We agreed to freeze the design.", start_time=0, end_time=5, stt_confidence=None)],
        "AgendaKeyDecision": [SimpleNamespace(agenda="Design freeze", key_decision="Freeze the design", evidence="We agreed to freeze the design.", speaker="Speaker_1", agenda_order=1)],
        "ActionItem": [SimpleNamespace(task="Deliver designs", owner="Anita", deadline="Friday", due_date=None, evidence="Anita will deliver", item_order=1)],
    }

    class DB:
        def query(self, model, *a, **k):
            return Q(rows.get(getattr(model, "__name__", None) or getattr(getattr(model, "class_", None), "__name__", ""), []))

    meeting = SimpleNamespace(id=7, analysis=None, file_type="media", duration=30, source_language_confidence=None, output_language="en")
    svc.compute_quality_metrics(meeting, DB())
    assert captured["action_items"][0]["task"] == "Deliver designs"
    assert captured["decisions"][0]["decision"] == "Freeze the design"
    assert captured["agenda_topics"] == ["Design freeze"]

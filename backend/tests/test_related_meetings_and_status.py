"""Related meetings / task status / failure status.

Measured BEFORE the fix:

* ``compare_bullet_lists`` returned every item missing from the next meeting under ``resolved`` - an unchecked
  ``[ ] Prepare testing report`` became a "completed task" in ``timeline.completed_tasks``;
* in the live database 84 of the 843 meeting pairs with >= 0.88 keyword containment were a SHORT transcript fully
  "contained" in a long one on the same subject (Jaccard 0.13), and the classifier called them near-duplicates;
* a failed AIRAWAT report left every report stage ``PENDING`` and ``overall_status`` at ``uploaded`` next to
  ``status=error``.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from backend.services import similarity_service as sim
from backend.utils import text_signals as ts
from backend.utils.exceptions import AirawatError

PREVIOUS_ACTIONS = """\
- [ ] **Prepare testing report** — Owner: Anita | Deadline: Friday | Priority: High
- [x] **Ship the API gateway** — Owner: Ravi | Deadline: Monday | Priority: Low
- [ ] **Renew the vendor licence** — Owner: Priya | Deadline: Tuesday | Priority: Medium
* Plain bullet about the hosting budget review
"""
CURRENT_ACTIONS = "- [ ] **Prepare testing report** — Owner: Anita | Deadline: Friday | Priority: High\n- [ ] **Write the audit summary** — Owner: Z"


# ── task status ─────────────────────────────────────────────────────────────────────────────────────

def test_an_unchecked_task_that_is_absent_later_is_not_resolved():
    out = ts.compare_bullet_lists(CURRENT_ACTIONS, PREVIOUS_ACTIONS)
    assert not any("licence" in r.lower() for r in out["resolved"])
    assert any("Renew the vendor licence" in r for r in out["not_carried_over"])


def test_only_an_explicitly_checked_item_is_resolved():
    out = ts.compare_bullet_lists(CURRENT_ACTIONS, PREVIOUS_ACTIONS)
    assert [r for r in out["resolved"]] == [r for r in out["resolved"] if r.startswith("[x]")]
    assert len(out["resolved"]) == 1 and "Ship the API gateway" in out["resolved"][0]


def test_a_task_still_open_in_both_meetings_is_shared_not_resolved():
    out = ts.compare_bullet_lists(CURRENT_ACTIONS, PREVIOUS_ACTIONS)
    assert any("Prepare testing report" in s for s in out["shared"])
    assert not any("Prepare testing report" in s for s in out["resolved"] + out["not_carried_over"])


def test_a_plain_bullet_has_no_status_so_it_is_never_called_resolved():
    out = ts.compare_bullet_lists("", "* Plain bullet about the hosting budget review")
    assert out["resolved"] == [] and len(out["not_carried_over"]) == 1


def test_risks_have_no_checkbox_so_nothing_is_resolved_by_absence():
    out = ts.compare_bullet_lists("- vendor delay", "- storage cost growth\n- certificate expiry")
    assert out["resolved"] == [] and len(out["not_carried_over"]) == 2


def _snapshot(mid, transcript, **kw):
    titles = {1: "Platform migration kickoff", 2: "Quarterly finance and vendor review"}
    base = dict(id=mid, code=str(mid), title=titles.get(mid, f"Unrelated topic {mid}"), transcript=transcript,
                meeting_date=None,
                fingerprint="", action_items="", decisions="", risks="", duration=1800.0)
    base.update(kw)
    return sim.MeetingSnapshot(**base)


def test_the_comparison_output_does_not_call_unchecked_tasks_completed():
    cur = _snapshot(2, "payment gateway budget review notes", action_items=CURRENT_ACTIONS, meeting_date="2026-10-12")
    other = _snapshot(1, "payment gateway budget review notes", action_items=PREVIOUS_ACTIONS)
    res = sim.analyze_pair(cur, other, semantic=0.5, min_score=0.2, include_details=True)
    done = " ".join(res["timeline"]["completed_tasks"])
    assert "Renew the vendor licence" not in done and "Prepare testing report" not in done
    assert any("Renew the vendor licence" in t for t in res["timeline"]["not_carried_over_tasks"])
    assert any("Ship the API gateway" in t for t in res["difference_analysis"]["resolved_action_items"])


# ── duplicate / related classification ──────────────────────────────────────────────────────────────

LONG = " ".join(f"topic{i} platform migration budget review vendor contract" for i in range(1, 60))
SHORT = "platform migration budget review vendor contract topic1 topic2 topic3"


def _category(cur, other, semantic):
    sig = sim._compute_signals(cur, other, semantic)
    return sim._classify(cur, other, sig, min_score=0.2), sig


def test_a_short_transcript_contained_in_a_long_one_is_not_a_near_duplicate():
    cat, sig = _category(_snapshot(1, SHORT), _snapshot(2, LONG), semantic=0.45)
    assert sig["content"] >= 0.88 and sig["content_jaccard"] < 0.5          # the containment trap
    assert cat not in (sim.Category.EXACT_DUPLICATE, sim.Category.NEAR_DUPLICATE)


def test_containment_without_embedding_agreement_is_not_a_duplicate():
    a = " ".join(f"word{i}" for i in range(80))
    cat, sig = _category(_snapshot(1, a), _snapshot(2, a + " extra"), semantic=0.30)
    assert sig["content"] >= 0.88 and sig["content_jaccard"] >= 0.6
    assert cat != sim.Category.NEAR_DUPLICATE                                # lexical only: never "duplicate"


def test_a_real_near_duplicate_is_still_detected():
    a = " ".join(f"word{i}" for i in range(80))
    cat, _ = _category(_snapshot(1, a), _snapshot(2, a + " one more"), semantic=0.96)
    assert cat == sim.Category.NEAR_DUPLICATE


def test_an_identical_transcript_fingerprint_is_an_exact_duplicate():
    cat, _ = _category(_snapshot(1, SHORT, fingerprint="abc"), _snapshot(2, LONG, fingerprint="abc"), semantic=0.1)
    assert cat == sim.Category.EXACT_DUPLICATE


def test_shared_vocabulary_alone_does_not_make_meetings_related():
    cat, sig = _category(_snapshot(1, SHORT), _snapshot(2, LONG), semantic=0.05)
    assert max(sig["content"], sig["keyword"]) >= 0.18                      # plenty of shared words ...
    assert cat == sim.Category.NOT_RELATED                                   # ... but no semantic evidence


def test_semantic_similarity_with_some_topic_overlap_is_related():
    cat, _ = _category(_snapshot(1, SHORT), _snapshot(2, LONG), semantic=0.45)
    assert cat == sim.Category.RELATED


def test_the_duplicate_explanation_quotes_the_symmetric_overlap_not_the_containment():
    cur, other = _snapshot(1, " ".join(f"word{i}" for i in range(80))), _snapshot(2, " ".join(f"word{i}" for i in range(80)) + " one more")
    res = sim.analyze_pair(cur, other, semantic=0.96, min_score=0.2, include_details=False)
    assert res["category"] == sim.Category.NEAR_DUPLICATE and res["is_duplicate"] is True
    assert "content_jaccard" in res["signals"]


# ── failure status: the stages that failed say so ─────────────────────────────────────────────────

class _Session:
    def __init__(self, meeting):
        self.meeting, self.commits = meeting, 0

    def query(self, *a, **k):
        return self

    def filter(self, *a, **k):
        return self

    def first(self):
        return self.meeting

    def commit(self):
        self.commits += 1


@pytest.fixture()
def failing_meeting(monkeypatch):
    from backend.database import connection
    from backend.workers import pipelines

    meeting = SimpleNamespace(id=7, status="processing", overall_status="uploaded", error_message=None,
                              pipeline_stage_states_json=None, retry_history_json=None)
    session = _Session(meeting)

    @contextmanager
    def fake_session():
        yield session

    monkeypatch.setattr(connection, "pipeline_db_session", fake_session)
    return SimpleNamespace(meeting=meeting, session=session, mark=pipelines._mark_meeting_failed)


def _stages(meeting):
    import json
    return json.loads(meeting.pipeline_stage_states_json)


def test_a_failed_metadata_section_marks_exactly_its_stages_failed(failing_meeting):
    err = AirawatError("AIRAWAT LLM failed to generate: metadata", details={"failed_sections": ["metadata"]})
    failing_meeting.mark(7, err)
    m = failing_meeting.meeting
    states = _stages(m)
    assert states["topics"] == "FAILED" and states["sentiment"] == "FAILED"
    assert states["summary"] == "PENDING" and states["mom"] == "PENDING" and states["agenda"] == "PENDING"
    assert states["stt"] == "PENDING" or states.get("stt") is None or states["stt"] != "SUCCESS"
    assert (m.status, m.overall_status) == ("error", "FAILED")
    assert "metadata" in m.error_message and failing_meeting.session.commits == 1


def test_nothing_is_ever_promoted_to_success_by_a_failure(failing_meeting):
    failing_meeting.mark(7, AirawatError("down", details={"failed_sections": ["summary", "action items"]}))
    states = _stages(failing_meeting.meeting)
    assert "SUCCESS" not in states.values()
    assert states["summary"] == "FAILED" and states["action_items"] == "FAILED"


def test_an_airawat_failure_without_section_detail_fails_every_report_stage(failing_meeting):
    failing_meeting.mark(7, AirawatError("AIRAWAT is not configured"))
    states = _stages(failing_meeting.meeting)
    assert all(states[s] == "FAILED" for s in ("summary", "mom", "agenda", "decision", "action_items", "topics", "sentiment"))


def test_a_non_airawat_failure_sets_the_meeting_status_without_inventing_stage_failures(failing_meeting):
    failing_meeting.mark(7, RuntimeError("disk full"))
    m = failing_meeting.meeting
    assert (m.status, m.overall_status) == ("error", "FAILED") and m.pipeline_stage_states_json is None

"""Generated output, stored rows, API response and statuses agree (PostgreSQL, rolled back after each test).

* ``action_items_generated`` / ``action_items_count`` / ``db_action_items_saved`` were derived from the metadata call's
  bullet text while the stage said "N action items saved" (from the dedicated extraction): two LLM answers, two truths;
* ``sentiment_breakdown_json`` held ``emotions`` as a JSON *string* inside JSON (double-encoded);
* top-level ``agenda`` was ``null`` next to six ``agenda_key_decisions``;
* ``speaker_duration_sec`` (diarization processing time) had no honest sibling for speech time.

Skips when the database is not reachable.
"""

import json
import uuid

import pytest


@pytest.fixture()
def db_session():
    from sqlalchemy import inspect
    from sqlalchemy.orm import Session

    from backend.database.connection import engine

    try:
        if not inspect(engine).has_table("meetings"):
            pytest.skip("meetings table does not exist yet (start the app once to create the schema)")
    except Exception as exc:
        pytest.skip(f"database not reachable: {exc}")

    conn = engine.connect()
    outer = conn.begin()
    session = Session(bind=conn, join_transaction_mode="create_savepoint", autoflush=False)
    try:
        yield session
    finally:
        session.close()
        outer.rollback()
        conn.close()


@pytest.fixture(autouse=True)
def lexical(monkeypatch):
    from backend.evaluation import reference_free_engine as rfe

    monkeypatch.setattr(rfe, "_embed_batch", lambda texts: [])
    monkeypatch.setattr(rfe, "_embed_one", lambda text: [])


TRANSCRIPT = (
    "Speaker_1: We decided to adopt PostgreSQL and move the staging environment first.\n"
    "Speaker_1: Priya will prepare the migration plan by Friday and share it with the team.\n"
)


def _pid():
    return "9" + str(uuid.uuid4().int)[:14]


def _meeting(db, **extra):
    from backend.models.models import Meeting, TranscriptSegment

    pid = _pid()
    m = Meeting(bharatvc_meeting_id=pid, meeting_id=pid, display_id=pid, filename=f"{pid}.mp4", file_type="media",
                filepath=f"/tmp/{pid}.mp4", title="Planning sync", status="processing", duration=120.0, **extra)
    db.add(m)
    db.flush()
    for i, line in enumerate(TRANSCRIPT.strip().split("\n")):
        spk, text = line.split(": ", 1)
        db.add(TranscriptSegment(meeting_id=m.id, speaker=spk, start_time=f"{i * 10:.3f}", end_time=f"{i * 10 + 9:.3f}",
                                 text=text))
    db.commit()
    return m


def _reports(**over):
    base = {
        "summary": "The team decided to adopt PostgreSQL and assigned the migration plan to Priya.",
        "mom": "### Minutes\n- PostgreSQL adopted\n- Migration plan owned by Priya",
        "action_items": "No specific action items identified.",       # the metadata call's blob: no bullets at all
        "structured_action_items": [
            {"task": "Prepare the migration plan", "owner": "Priya", "deadline": "Friday", "priority": "High",
             "evidence": "Priya will prepare the migration plan by Friday"},
            {"task": "Share the plan with the team", "owner": "Priya", "deadline": "Friday", "priority": "Medium"},
        ],
        "decisions": "- Adopt PostgreSQL",
        "agenda_key_decisions": [
            {"agenda": "Database migration", "agenda_order": 1, "agenda_id_str": "A1", "key_decision":
             "Adopt PostgreSQL", "evidence": "We decided to adopt PostgreSQL", "decision_grounded": True,
             "match_score": 52.0, "speaker": "Speaker_1"},
            {"agenda": "Hosting budget", "agenda_order": 2, "agenda_id_str": "A2",
             "key_decision": "No formal decision recorded.", "evidence": None, "decision_grounded": False,
             "match_score": None},
        ],
        "risks": "None noted.",
        "project_name": "Migration",
        "sentiment": "Positive",
        "emotions": "Confidence: 70%, Focus: 30%",
        "questions_raised": "[]",
        "keyword_frequency": json.dumps({"postgresql": 3, "migration": 2}),
        "_elapsed_sec": 1.0,
    }
    base.update(over)
    return base


def _run(db, meeting, reports):
    from backend.workers import pipelines

    return pipelines._run_shared_pipeline(meeting.id, TRANSCRIPT, meeting, db, reports=reports, act=None)


# ── action-item flags come from the rows that were saved ─────────────────────────────────────────────

def test_action_item_flags_follow_the_saved_rows_even_when_the_metadata_blob_has_no_bullets(db_session):
    from backend.models.models import ActionItem, Meeting

    m = _meeting(db_session)
    _run(db_session, m, _reports())
    db_session.expire_all()
    m = db_session.get(Meeting, m.id)
    saved = db_session.query(ActionItem).filter(ActionItem.meeting_id == m.id).count()
    assert saved == 2
    assert m.action_items_generated is True and m.action_items_count == 2 and m.db_action_items_saved is True
    assert json.loads(m.pipeline_stage_states_json)["action_items"] == "SUCCESS"


def test_zero_action_items_is_a_successful_stage_with_honest_false_flags(db_session):
    from backend.models.models import ActionItem, Meeting

    m = _meeting(db_session)
    _run(db_session, m, _reports(structured_action_items=[], action_items=""))
    db_session.expire_all()
    m = db_session.get(Meeting, m.id)
    assert db_session.query(ActionItem).filter(ActionItem.meeting_id == m.id).count() == 0
    assert (m.action_items_generated, m.action_items_count, m.db_action_items_saved) == (False, 0, False)


def test_a_reprocess_replaces_action_items_instead_of_duplicating_them(db_session):
    from backend.models.models import ActionItem

    m = _meeting(db_session)
    _run(db_session, m, _reports())
    _run(db_session, m, _reports())
    assert db_session.query(ActionItem).filter(ActionItem.meeting_id == m.id).count() == 2


# ── native JSON where the contract says so ───────────────────────────────────────────────────────────

def test_the_sentiment_breakdown_holds_emotions_as_an_object_not_a_json_string(db_session):
    from backend.models.models import Meeting

    m = _meeting(db_session)
    _run(db_session, m, _reports())
    db_session.expire_all()
    m = db_session.get(Meeting, m.id)
    breakdown = json.loads(m.sentiment_breakdown_json)
    assert breakdown["sentiment"] == "Positive"
    assert breakdown["emotions"] == {"Confidence": 70.0, "Focus": 30.0}          # not '{"Confidence": 70.0, ...}'


def test_an_unparseable_emotions_text_is_kept_as_text_not_double_encoded(db_session):
    from backend.models.models import Meeting

    m = _meeting(db_session)
    _run(db_session, m, _reports(emotions="calm and focused"))
    db_session.expire_all()
    assert json.loads(db_session.get(Meeting, m.id).sentiment_breakdown_json)["emotions"] == "calm and focused"


def test_the_final_json_data_is_a_native_jsonb_object_that_round_trips(db_session):
    from backend.models.models import AIMeetingIntelFinal
    from backend.services.final_aggregation_service import build_final_record

    m = _meeting(db_session)
    _run(db_session, m, _reports())
    record = build_final_record(m.id, db_session)
    db_session.commit()
    db_session.expire_all()
    stored = db_session.get(AIMeetingIntelFinal, record.id).json_data
    assert isinstance(stored, dict)                                              # JSONB object, not a JSON string
    for key in ("segments", "agenda_key_decisions", "structured_agenda_decisions", "action_items"):
        assert isinstance(stored[key], list), key
    assert isinstance(stored["analysis"], dict) and isinstance(stored["metrics"], (dict, type(None)))
    assert json.loads(json.dumps(stored)) == stored
    assert len(stored["segments"]) == 2 and stored["action_items_count"] == 2 and stored["db_action_items_saved"] is True


def test_repeated_aggregation_is_stable_and_never_creates_a_second_row(db_session):
    from backend.models.models import AIMeetingIntelFinal
    from backend.services.final_aggregation_service import build_final_record

    m = _meeting(db_session)
    _run(db_session, m, _reports())
    first = build_final_record(m.id, db_session)
    db_session.commit()
    first_json = json.dumps(first.json_data, sort_keys=True, default=str)
    second = build_final_record(m.id, db_session)
    db_session.commit()
    assert db_session.query(AIMeetingIntelFinal).filter(AIMeetingIntelFinal.meeting_pk == m.id).count() == 1
    assert json.dumps(second.json_data, sort_keys=True, default=str) == first_json


def test_the_final_mapping_carries_the_evidence_and_leaves_unsupported_items_unmapped(db_session):
    from backend.services.final_aggregation_service import build_final_record

    m = _meeting(db_session)
    _run(db_session, m, _reports())
    record = build_final_record(m.id, db_session)
    mapping = json.loads(record.agenda_decision_mapping)
    mapped = {x["agenda_title"]: x for x in mapping}
    assert mapped["Database migration"]["has_decision"] is True
    assert mapped["Database migration"]["evidence"] == "We decided to adopt PostgreSQL"
    assert mapped["Hosting budget"]["has_decision"] is False and "decision" not in mapped["Hosting budget"]


# ── API response: agenda and speaker-time fields ─────────────────────────────────────────────────────

def _response(db, meeting):
    from backend.api.meetings import build_meeting_response_dict

    return build_meeting_response_dict(meeting, db, quality_metrics={}, evaluation_metrics=None)


def test_top_level_agenda_lists_only_announced_topics_and_labels_the_source(db_session):
    reports = _reports()
    reports["agenda_key_decisions"][0]["agenda_type"] = "explicit"      # announced by the speaker
    reports["agenda_key_decisions"][1]["agenda_type"] = "inferred"      # only inferred from the discussion
    m = _meeting(db_session)
    _run(db_session, m, reports)
    db_session.expire_all()
    out = _response(db_session, db_session.get(type(m), m.id))
    assert out["agenda"] == "1. Database migration" and out["agenda_source"] == "announced"
    assert [(i["agenda"], i["agenda_type"]) for i in out["agenda_key_decisions"]] == [
        ("Database migration", "explicit"), ("Hosting budget", "inferred")]      # the API shows the STORED type


def test_inferred_topics_alone_are_never_presented_as_an_agenda(db_session):
    m = _meeting(db_session)
    _run(db_session, m, _reports())                                     # both items inferred (the stored default)
    db_session.expire_all()
    out = _response(db_session, db_session.get(type(m), m.id))
    assert out["agenda"] is None and out["agenda_source"] == "inferred_topics_only"
    assert len(out["agenda_key_decisions"]) == 2
    assert all(i["agenda_type"] == "inferred" for i in out["agenda_key_decisions"])


def test_a_stored_agenda_is_returned_as_is_and_a_meeting_without_any_has_null(db_session):
    stored = _meeting(db_session, agenda="Quarterly planning")
    out = _response(db_session, stored)
    assert out["agenda"] == "Quarterly planning" and out["agenda_source"] == "stored"
    empty = _meeting(db_session)
    out = _response(db_session, empty)
    assert out["agenda"] is None and out["agenda_source"] is None and out["agenda_key_decisions"] == []


def test_speaker_processing_time_and_speech_duration_are_separate_fields(db_session):
    info = {"engine": "nvidia_nemo", "diagnostics": {"speaker_breakdown": {
        "Speaker_1": {"speaking_duration_seconds": 1690.01}}}}
    m = _meeting(db_session, speaker_duration_sec=23.453, diarization_info_json=json.dumps(info))
    out = _response(db_session, m)
    assert out["speaker_duration_sec"] == 23.453                                  # historical field unchanged
    assert out["speaker_processing_time_sec"] == 23.453
    assert out["speaker_speech_duration_sec"] == 1690.01


def test_speech_duration_is_null_when_the_diarization_did_not_measure_it(db_session):
    out = _response(db_session, _meeting(db_session, speaker_duration_sec=4.0))
    assert out["speaker_speech_duration_sec"] is None and out["speaker_processing_time_sec"] == 4.0


def test_a_native_list_stays_a_list_in_the_structured_action_items(db_session):
    m = _meeting(db_session)
    _run(db_session, m, _reports())
    db_session.expire_all()
    out = _response(db_session, db_session.get(type(m), m.id))
    assert isinstance(out["analysis"]["action_items"], list) and len(out["structured_action_items"]) == 2

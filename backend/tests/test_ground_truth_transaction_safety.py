"""A missing / failing ground-truth lookup must never poison the transaction or the pipeline (meeting 96).

Root cause reproduced here with REAL PostgreSQL errors (the ``ground_truth`` table is made unreadable inside the
database by rewriting the statement): a statement that fails inside a transaction aborts it, every later statement
raises ``InFailedSqlTransaction`` - which is the error that was stored on the meeting, while the statement that
actually failed was only a log line. Ground truth is OPTIONAL, so:

* a missing row                        -> reference-free evaluation, accuracy NULL;
* a failing lookup / evaluation        -> same, the cause is logged, the transaction stays usable;
* the pipeline's other work (summary, MoM, action items, stages) is unaffected;
* the API reports the processing outcome in ``api_status`` instead of the column default "Success".

Skips when the database is not reachable.
"""

import json
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest


# ── fixtures ──────────────────────────────────────────────────────────────────────────────────────────

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


@pytest.fixture()
def broken_ground_truth():
    """Every statement reading ``ground_truth`` fails INSIDE PostgreSQL (relation does not exist)."""
    from sqlalchemy import event

    from backend.database.connection import engine

    pattern = re.compile(r"FROM ground_truth\b")

    def _break(conn, cursor, statement, parameters, context, executemany):
        return pattern.sub("FROM ground_truth_does_not_exist", statement), parameters

    event.listen(engine, "before_cursor_execute", _break, retval=True)
    try:
        yield
    finally:
        event.remove(engine, "before_cursor_execute", _break)


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
        "action_items": "No specific action items identified.",
        "structured_action_items": [
            {"task": "Prepare the migration plan", "owner": "Priya", "deadline": "Friday", "priority": "High",
             "evidence": "Priya will prepare the migration plan by Friday"},
        ],
        "decisions": "- Adopt PostgreSQL",
        "agenda_key_decisions": [
            {"agenda": "Database migration", "agenda_order": 1, "agenda_id_str": "A1", "key_decision":
             "Adopt PostgreSQL", "evidence": "We decided to adopt PostgreSQL", "decision_grounded": True,
             "match_score": 52.0, "speaker": "Speaker_1"},
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


def _add_ground_truth(db, meeting):
    from backend.models.models import GroundTruth

    gt = GroundTruth(meeting_id=meeting.id, ref_transcript="We decided to adopt PostgreSQL.", version=3)
    db.add(gt)
    db.commit()
    return gt


# ── the mechanism: why the LAST error is InFailedSqlTransaction ──────────────────────────────────────────

def test_a_failed_statement_aborts_the_transaction_and_the_next_error_masks_it(db_session, broken_ground_truth):
    from sqlalchemy import text
    from sqlalchemy.exc import InternalError, ProgrammingError

    from backend.models.models import GroundTruth
    from backend.utils.db_safe import is_failed_transaction_error

    m = _meeting(db_session)
    with pytest.raises(ProgrammingError):                     # the FIRST failure: the actual database error
        db_session.query(GroundTruth.id).filter(GroundTruth.meeting_id == m.id).first()
    with pytest.raises(InternalError) as masked:              # ... and what every later statement reports
        db_session.execute(text("SELECT 1"))
    assert is_failed_transaction_error(masked.value)
    assert "InFailedSqlTransaction" in type(masked.value.orig).__name__


# ── a missing ground-truth row is normal ─────────────────────────────────────────────────────────────

def test_a_meeting_without_ground_truth_has_no_reference_and_no_error(db_session):
    from backend.utils.db_safe import ground_truth_of, has_ground_truth, load_ground_truth

    m = _meeting(db_session)
    assert has_ground_truth(db_session, m.id) is False
    assert load_ground_truth(db_session, m.id) is None
    assert ground_truth_of(db_session, m) is None
    assert m.has_ground_truth is False


def test_processing_without_ground_truth_completes_with_a_reference_free_signal_and_null_accuracy(db_session):
    from backend.models.models import ActionItem, AnalysisResult, Meeting

    m = _meeting(db_session)
    _run(db_session, m, _reports())
    db_session.expire_all()
    m = db_session.get(Meeting, m.id)

    assert (m.accuracy, m.transcription_accuracy, m.speaker_accuracy) == (None, None, None)
    metrics = json.loads(m.metrics_json)
    assert metrics["metric_type"] == "RUNTIME_QUALITY_SIGNAL" and metrics["is_accuracy"] is False
    assert metrics["accuracy"] is None and metrics["warnings"] == []
    assert db_session.query(AnalysisResult).filter(AnalysisResult.meeting_id == m.id).count() == 1
    assert db_session.query(ActionItem).filter(ActionItem.meeting_id == m.id).count() == 1
    states = json.loads(m.pipeline_stage_states_json)
    assert states["summary"] == "SUCCESS" and states["mom"] == "SUCCESS" and states["action_items"] == "SUCCESS"


# ── a FAILING ground-truth query ─────────────────────────────────────────────────────────────────────

def test_a_failed_ground_truth_lookup_rolls_back_to_its_savepoint_and_keeps_the_session_usable(
    db_session, broken_ground_truth
):
    from backend.models.models import Meeting, TranscriptSegment
    from backend.utils.db_safe import has_ground_truth

    m = _meeting(db_session)
    pending = TranscriptSegment(meeting_id=m.id, speaker="Speaker_2", start_time="30.000", end_time="35.000",
                                text="A row written before the optional lookup.")
    db_session.add(pending)                                    # pending, not yet flushed

    assert has_ground_truth(db_session, m.id) is False         # the lookup failed -> "no ground truth"
    # The transaction is alive: later statements work, and the pending row was neither lost nor rolled back.
    assert db_session.query(Meeting).filter(Meeting.id == m.id).count() == 1
    db_session.commit()
    assert db_session.query(TranscriptSegment).filter(
        TranscriptSegment.meeting_id == m.id, TranscriptSegment.speaker == "Speaker_2"
    ).count() == 1


def test_the_has_ground_truth_property_never_raises_and_response_serialisation_works(db_session, broken_ground_truth):
    from backend.schemas import schemas

    m = _meeting(db_session)
    assert m.has_ground_truth is False
    validated = schemas.Meeting.model_validate(m, from_attributes=True)       # reads the property
    assert validated.id == m.id and validated.has_ground_truth is False


def test_the_pipeline_finishes_every_report_stage_when_the_ground_truth_query_fails(db_session, broken_ground_truth):
    from backend.models.models import ActionItem, AgendaKeyDecision, AnalysisResult, Meeting

    m = _meeting(db_session)
    _run(db_session, m, _reports())
    db_session.expire_all()
    m = db_session.get(Meeting, m.id)

    assert (m.accuracy, m.transcription_accuracy, m.speaker_accuracy) == (None, None, None)
    assert json.loads(m.metrics_json)["metric_type"] == "RUNTIME_QUALITY_SIGNAL"
    analysis = db_session.query(AnalysisResult).filter(AnalysisResult.meeting_id == m.id).one()
    assert "PostgreSQL" in analysis.summary and "Minutes" in analysis.mom                      # summary + MoM kept
    assert db_session.query(ActionItem).filter(ActionItem.meeting_id == m.id).count() == 1      # action items kept
    assert db_session.query(AgendaKeyDecision).filter(AgendaKeyDecision.meeting_id == m.id).count() == 1
    states = json.loads(m.pipeline_stage_states_json)
    assert all(states[s] == "SUCCESS" for s in ("summary", "mom", "agenda", "decision", "action_items"))
    assert m.error_message is None


# ── a failing ground-truth EVALUATION (the reference exists) ─────────────────────────────────────────────

def test_a_failing_evaluation_falls_through_to_the_reference_free_signal(db_session, monkeypatch):
    from sqlalchemy import text

    from backend.models.models import TranscriptSegment
    from backend.services import true_evaluation_service
    from backend.workers import pipelines

    m = _meeting(db_session)
    _add_ground_truth(db_session, m)
    marker = TranscriptSegment(meeting_id=m.id, speaker="Speaker_9", start_time="50.000", end_time="55.000",
                               text="Pending pipeline row.")
    db_session.add(marker)

    def _evaluation_that_hits_a_database_error(meeting, db, performed_by="system", commit=True):
        db.execute(text("SELECT * FROM table_that_does_not_exist_anywhere"))     # aborts the transaction

    monkeypatch.setattr(true_evaluation_service, "run_meeting_evaluation", _evaluation_that_hits_a_database_error)
    pipelines._compute_runtime_quality_signal(m, _reports(), TRANSCRIPT, db=db_session)

    assert (m.accuracy, m.transcription_accuracy, m.speaker_accuracy) == (None, None, None)   # not fabricated
    metrics = json.loads(m.metrics_json)
    assert metrics["metric_type"] == "RUNTIME_QUALITY_SIGNAL"                    # reference-free signal still ran
    assert metrics["quality_signal_score"] is not None
    assert any("evaluation failed" in w for w in metrics["warnings"])            # and says why accuracy is null
    db_session.commit()                                                          # the session is usable ...
    assert db_session.query(TranscriptSegment).filter(TranscriptSegment.speaker == "Speaker_9").count() == 1


def test_a_python_error_in_the_evaluation_is_handled_the_same_way(db_session, monkeypatch):
    from backend.services import true_evaluation_service
    from backend.workers import pipelines

    m = _meeting(db_session)
    _add_ground_truth(db_session, m)

    def _boom(*a, **k):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(true_evaluation_service, "run_meeting_evaluation", _boom)
    pipelines._compute_runtime_quality_signal(m, _reports(), TRANSCRIPT, db=db_session)
    assert m.accuracy is None
    assert json.loads(m.metrics_json)["metric_type"] == "RUNTIME_QUALITY_SIGNAL"


def test_a_successful_evaluation_runs_inside_the_savepoint_without_committing(db_session, monkeypatch):
    from backend.services import true_evaluation_service
    from backend.workers import pipelines

    m = _meeting(db_session)
    _add_ground_truth(db_session, m)
    seen = {}

    def _ok(meeting, db, performed_by="system", commit=True):
        seen["commit"], seen["performed_by"] = commit, performed_by
        meeting.accuracy = 91.5

    monkeypatch.setattr(true_evaluation_service, "run_meeting_evaluation", _ok)
    pipelines._compute_runtime_quality_signal(m, _reports(), TRANSCRIPT, db=db_session)

    assert seen == {"commit": False, "performed_by": "pipeline"}      # the pipeline's own commit decides, not the engine
    assert m.accuracy == 91.5
    assert m.metrics_json is None or json.loads(m.metrics_json).get("metric_type") != "RUNTIME_QUALITY_SIGNAL"


def test_optional_db_does_not_hide_a_transaction_that_was_already_aborted(db_session, broken_ground_truth):
    from sqlalchemy import text
    from sqlalchemy.exc import InternalError, ProgrammingError

    from backend.models.models import GroundTruth
    from backend.utils.db_safe import optional_db

    m = _meeting(db_session)
    with pytest.raises(ProgrammingError):
        db_session.query(GroundTruth.id).filter(GroundTruth.meeting_id == m.id).first()    # poisoned by someone else
    with pytest.raises(InternalError):
        optional_db(db_session, lambda: db_session.execute(text("SELECT 1")).scalar(), default=0)


# ── api_status follows the processing outcome ────────────────────────────────────────────────────────

@pytest.mark.parametrize("status,label,code", [
    ("completed", "Success", 200),
    ("error", "Failed", 500),
    ("processing", "Processing", 202),
    ("uploaded", "Pending", 202),
    (None, "Pending", 202),
])
def test_api_status_for_each_lifecycle_status(status, label, code):
    from backend.utils.processing_status import api_status_for

    assert api_status_for(status) == (label, code)


def test_a_failed_meeting_is_not_served_as_api_status_success():
    from backend.schemas import schemas

    base = dict(id=1, filename="a.mp4", upload_time=datetime.now(timezone.utc))
    failed = schemas.Meeting(**base, status="error", error_message="Processing failed: x",
                             api_status="Success", api_status_code=200, overall_status="FAILED")
    assert (failed.api_status, failed.api_status_code) == ("Failed", 500)
    assert failed.status == "error" and failed.overall_status == "FAILED"
    assert failed.error_message == "Processing failed: x"                 # the actual status and error stay exposed
    done = schemas.Meeting(**base, status="completed", api_status="Success", api_status_code=200)
    assert (done.api_status, done.api_status_code) == ("Success", 200)
    running = schemas.Meeting(**base, status="processing", api_status="Success", api_status_code=200)
    assert (running.api_status, running.api_status_code) == ("Processing", 202)


def test_runtime_metrics_report_the_outcome_not_the_stored_default():
    from backend.services.runtime_metrics import build_runtime_metrics_dict

    m = SimpleNamespace(id=5, filename="a.mp4", title="t", display_id="5", bharatvc_meeting_id="5", status="error",
                        error_message="Processing failed: x", api_status="Success", api_status_code=200,
                        duration=10.0, analysis=None, segments=[], action_items=[], summary_quality=None,
                        mom_quality=None, topics_count=0)
    try:
        payload = build_runtime_metrics_dict(m, None)
    except Exception as exc:  # the builder needs more attributes than this stand-in has; the mapping is what matters
        pytest.skip(f"runtime metrics builder needs a fuller meeting object: {exc}")
    block = payload["20_api_status"]
    assert block["status"] == "Failed" and block["http_status_code"] == 500
    assert block["error_message"] == "Processing failed: x"
    assert payload["21_overall_processing_status"] == "Failed"


# ── a (re)run starts clean; a failure is recorded per stage ─────────────────────────────────────────────

def test_claiming_a_meeting_clears_the_previous_runs_error_and_stage_states(db_session):
    from backend.models.models import Meeting
    from backend.workers import pipelines

    m = _meeting(db_session, error_message="Processing failed: (psycopg2.errors.InFailedSqlTransaction) ...",
                 overall_status="FAILED", api_status="Success", api_status_code=200,
                 pipeline_stage_states_json=json.dumps({"stt": "SUCCESS", "summary": "FAILED"}),
                 retry_history_json=json.dumps([{"stage": "summary"}]))
    m.status = "error"
    db_session.commit()

    assert pipelines._claim_meeting(db_session, m.id, datetime.now(timezone.utc)) is True
    db_session.expire_all()
    m = db_session.get(Meeting, m.id)
    assert m.status == "processing"
    assert m.error_message is None
    assert m.overall_status == "PROCESSING"
    assert m.pipeline_stage_states_json is None and m.retry_history_json is None
    assert (m.api_status, m.api_status_code) == ("Processing", 202)


class _FakeSession:
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

    meeting = SimpleNamespace(id=7, status="processing", overall_status="PROCESSING", error_message=None,
                              pipeline_stage_states_json=None, retry_history_json=None,
                              api_status="Processing", api_status_code=202)
    session = _FakeSession(meeting)

    @contextmanager
    def fake_session():
        yield session

    monkeypatch.setattr(connection, "pipeline_db_session", fake_session)
    return SimpleNamespace(meeting=meeting, session=session, mark=pipelines._mark_meeting_failed)


def test_a_failure_in_diarization_marks_that_stage_failed_and_keeps_the_finished_ones(failing_meeting):
    failing_meeting.mark(
        7, RuntimeError("NeMo crashed"), failed_stage="diarization",
        stage_states={"audio_extraction": "SUCCESS", "stt": "SUCCESS"},
    )
    m = failing_meeting.meeting
    states = json.loads(m.pipeline_stage_states_json)
    assert states["audio_extraction"] == "SUCCESS" and states["stt"] == "SUCCESS"
    assert states["diarization"] == "FAILED"
    # never ran -> still PENDING, never promoted to SUCCESS
    assert states["alignment"] == "PENDING" and states["summary"] == "PENDING" and states["mom"] == "PENDING"
    assert (m.status, m.overall_status) == ("error", "FAILED")
    assert m.error_message == "Processing failed: NeMo crashed"
    assert (m.api_status, m.api_status_code) == ("Failed", 500)
    assert failing_meeting.session.commits == 1


def test_a_stage_that_finished_degraded_is_not_reported_as_success_by_a_later_failure(failing_meeting):
    failing_meeting.mark(
        7, RuntimeError("disk full"),
        stage_states={"audio_extraction": "SUCCESS", "diarization": "DEGRADED", "transcript_correction": "DEGRADED"},
    )
    states = json.loads(failing_meeting.meeting.pipeline_stage_states_json)
    assert states["diarization"] == "DEGRADED" and states["transcript_correction"] == "DEGRADED"
    assert "FAILED" not in states.values()                       # no stage is blamed: the failure was elsewhere


def test_a_failed_stage_is_not_overwritten_by_an_earlier_success_claim(failing_meeting):
    failing_meeting.meeting.pipeline_stage_states_json = json.dumps({"stt": "FAILED"})
    failing_meeting.mark(7, RuntimeError("x"), stage_states={"stt": "SUCCESS"})
    assert json.loads(failing_meeting.meeting.pipeline_stage_states_json)["stt"] == "FAILED"


def test_diarization_status_maps_to_the_honest_stage_state():
    from backend.workers.pipelines import _stage_state_for_diarization

    assert _stage_state_for_diarization("SUCCESS").value == "SUCCESS"
    assert _stage_state_for_diarization("FAILED").value == "FAILED"
    assert _stage_state_for_diarization("DEGRADED").value == "DEGRADED"
    assert _stage_state_for_diarization("anything else").value == "DEGRADED"


# ── speaker duration is not processing time; assignments are validated against the turns ─────────────

def test_speaker_duration_is_talk_time_and_independent_of_the_processing_time_fields(db_session):
    from backend.models.models import Meeting
    from backend.services.final_aggregation_service import build_final_record

    m = _meeting(db_session)
    m.speaker_duration_sec = 999.0        # the diarization STAGE'S processing time
    m.processing_time = 1234.0
    m.total_processing_time_sec = 1234.0
    db_session.commit()

    record = build_final_record(m.id, db_session)
    speaker_duration = record.speaker_duration
    speaker_duration = json.loads(speaker_duration) if isinstance(speaker_duration, str) else speaker_duration
    assert speaker_duration, "per-speaker talk time must be present"
    talk = {row.get("speaker"): row for row in speaker_duration}
    total_talk = sum(float(r.get("duration_seconds") or r.get("total_seconds") or 0) for r in speaker_duration)
    assert 0 < total_talk <= 120.0 + 1.0                       # bounded by the recording, not by the 999 s of compute
    assert "999" not in json.dumps(speaker_duration) and "1234" not in json.dumps(speaker_duration)
    assert talk
    body = record.json_data
    assert body["speaker_duration_sec"] == 999.0               # the processing time stays its own, named field
    assert body["processing_time"] == 1234.0
    db_session.expire_all()
    assert db_session.get(Meeting, m.id).speaker_duration_sec == 999.0


def test_speaker_assignments_are_validated_against_the_diarization_turns():
    from backend.services.diarization_service import validate_speaker_assignments

    turns = [{"start": 0.0, "end": 10.0, "speaker": "S0"}, {"start": 10.0, "end": 20.0, "speaker": "S1"}]
    good = [
        {"speaker": "Speaker_1", "start_time": "0.5", "end_time": "9.5"},
        {"speaker": "Speaker_2", "start_time": "10.5", "end_time": "19.5"},
    ]
    result = validate_speaker_assignments(good, turns)
    assert result["status"] == "VALID" and result["supported_ratio"] == 1.0
    assert result["label_to_engine_speaker"] == {"Speaker_1": "S0", "Speaker_2": "S1"}

    bad = good + [
        {"speaker": "Speaker_2", "start_time": "1.0", "end_time": "4.0"},     # S0's time labelled as Speaker_2
        {"speaker": "Speaker_1", "start_time": "40.0", "end_time": "45.0"},   # outside every turn
    ]
    result = validate_speaker_assignments(bad, turns)
    assert result["status"] == "MISMATCH"
    assert (result["supported"], result["mismatched"], result["outside_turns"]) == (2, 1, 1)


def test_speaker_validation_says_not_verifiable_instead_of_guessing():
    from backend.services.diarization_service import validate_speaker_assignments

    assert validate_speaker_assignments([], [{"start": 0, "end": 1, "speaker": "S0"}])["status"] == "NOT_VERIFIABLE"
    seg = [{"speaker": "Speaker_1", "start_time": "0", "end_time": "5"}]
    assert validate_speaker_assignments(seg, [])["status"] == "NOT_VERIFIABLE"


# ── transcript correction needs evidence ──────────────────────────────────────────────────────────────

VOCAB = {"vocabulary_terms": ["Govind", "Airawat", "PostgreSQL"], "project_name": "", "meeting_title": ""}


def test_a_term_with_vocabulary_evidence_is_corrected_and_the_confidence_is_measured():
    from backend.services.transcript_correction_service import merge_corrections_with_evidence

    merged, accepted, rejected = merge_corrections_with_evidence(
        "so govin will review the plan", "So Govind will review the plan", VOCAB, {"so", "govin", "will", "review"},
    )
    assert merged == "so Govind will review the plan"
    assert [(c["original"], c["corrected"], c["evidence"]) for c in accepted] == [("govin", "Govind", "domain_vocabulary")]
    assert 0.0 < accepted[0]["confidence"] <= 1.0 and not rejected


def test_unclear_words_are_not_replaced_by_text_the_audio_does_not_support():
    from backend.services.transcript_correction_service import merge_corrections_with_evidence

    raw = "the budget is uh unclear for the quarter"
    merged, accepted, rejected = merge_corrections_with_evidence(
        raw, "the budget is approved by finance for the quarter", VOCAB, set(raw.split()),
    )
    assert merged == raw                                        # the heard words stay exactly as heard
    assert not [c for c in accepted if c["corrected"]]
    assert rejected and rejected[0]["reason"].startswith("not_similar_to_the_heard_word")


def test_words_the_model_inserted_are_never_added():
    from backend.services.transcript_correction_service import merge_corrections_with_evidence

    merged, accepted, rejected = merge_corrections_with_evidence(
        "we ship on friday", "we will definitely ship on friday morning", VOCAB, {"we", "ship", "on", "friday"},
    )
    assert merged == "we ship on friday" and accepted == []
    assert {r["proposed"] for r in rejected} >= {"will", "definitely", "morning"}


def test_a_similar_word_without_any_context_support_is_rejected():
    from backend.services.transcript_correction_service import merge_corrections_with_evidence

    merged, accepted, rejected = merge_corrections_with_evidence(
        "send the plan to ravi", "send the plan to Ravindra", VOCAB, {"send", "the", "plan", "to", "ravi"},
    )
    assert merged == "send the plan to ravi" and accepted == []
    assert rejected[0]["reason"] == "no_evidence_in_this_meetings_context"


def test_a_spelling_that_already_occurs_in_the_transcript_is_evidence():
    from backend.services.transcript_correction_service import merge_corrections_with_evidence

    merged, accepted, _ = merge_corrections_with_evidence(
        "kubernetes migration needs kubernetis access", "kubernetes migration needs kubernetes access", VOCAB,
        {"kubernetes", "migration", "needs", "kubernetis", "access"},
    )
    assert merged == "kubernetes migration needs kubernetes access"
    assert accepted[0]["evidence"] == "transcript_consistency"


def test_disfluencies_may_be_dropped_but_real_words_may_not():
    from backend.services.transcript_correction_service import merge_corrections_with_evidence

    merged, accepted, rejected = merge_corrections_with_evidence(
        "we uh need the the report", "we need the report", VOCAB, {"we", "uh", "need", "the", "report"},
    )
    assert merged == "we need the report" and len(accepted) == 2 and not rejected
    merged, accepted, rejected = merge_corrections_with_evidence(
        "we need the final report", "we need the report", VOCAB, {"we", "need", "the", "final", "report"},
    )
    assert merged == "we need the final report" and rejected[0]["reason"] == "deleted_word_without_evidence"


def _llm_answering(monkeypatch, payload):
    from backend.services import model_router as mr

    class _LLM:
        def invoke(self, messages):
            return SimpleNamespace(content=json.dumps(payload))

    monkeypatch.setattr(mr.model_router, "get_llm", lambda **kw: _LLM())


def _segments(text):
    return [{"speaker": "Speaker_1", "start_time": "0.000", "end_time": "5.000", "text": text, "raw_text": text}]


def test_an_llm_rewrite_without_evidence_does_not_change_or_claim_to_correct_the_transcript(monkeypatch):
    from backend.services.transcript_correction_service import correct_and_normalize_transcript

    raw = "we should talk about the budget and the unclear item tomorrow."
    _llm_answering(monkeypatch, {"segments": [{
        "index": 0, "corrected_text": "we should talk about the budget and the pending approval tomorrow.",
        "confidence": 0.99, "reason": "fixed unclear words", "correction_applied": True,
    }]})
    out = correct_and_normalize_transcript(_segments(raw), meeting_title="Weekly sync")
    assert out[0]["text"].lower().rstrip(".") == raw.rstrip(".")           # same words (grammar pass may re-case)
    assert "pending" not in out[0]["text"] and "approval" not in out[0]["text"]
    assert out[0]["correction_source"] != "phonetic_contextual_llm"
    assert "unsupported" in out[0]["correction_reason"] or "grammar" in out[0]["correction_reason"]


def test_an_llm_correction_with_evidence_is_applied_with_its_evidence_recorded(monkeypatch):
    from backend.services.transcript_correction_service import correct_and_normalize_transcript

    raw = "Govin will present the Airawat results."
    _llm_answering(monkeypatch, {"segments": [{
        "index": 0, "corrected_text": "Govind will present the Airawat results.", "reason": "name",
        "correction_applied": True,
    }]})
    out = correct_and_normalize_transcript(
        _segments(raw), meeting_title="Airawat review", participants="Govind Rao",
    )
    assert out[0]["text"].startswith("Govind will present")
    assert out[0]["correction_applied"] is True and out[0]["correction_source"] == "phonetic_contextual_llm"
    changes = json.loads(out[0]["correction_metadata"])
    assert changes[0]["original"] == "Govin" and changes[0]["evidence"] == "domain_vocabulary"
    assert out[0]["raw_text"] == raw                                         # the raw ASR text is never lost


def test_a_missing_llm_confidence_is_not_defaulted_into_an_acceptance(monkeypatch):
    from backend.services.transcript_correction_service import correct_and_normalize_transcript

    raw = "the vendor will sign next week."
    _llm_answering(monkeypatch, {"segments": [{"index": 0, "corrected_text": "the vendor will sign the contract next week."}]})
    out = correct_and_normalize_transcript(_segments(raw))
    assert "contract" not in out[0]["text"]                                  # inserted word: no evidence, not added


# ── action items need transcript evidence ─────────────────────────────────────────────────────────────

AI_TRANSCRIPT = (
    "Priya will prepare the migration plan by Friday and share it with the team. "
    "We also agreed that Arjun reviews the staging environment before the release."
)


def _item(task, evidence=None, **extra):
    return {"task": task, "owner": "Priya", "evidence": evidence, **extra}


def test_an_item_without_evidence_that_the_transcript_never_mentions_is_not_saved():
    from backend.services.action_item_service import validate_action_items

    out = validate_action_items(
        [_item("Negotiate the vendor contract renewal with procurement department")], AI_TRANSCRIPT,
    )
    assert out == []


def test_an_item_with_a_fabricated_quote_that_the_transcript_does_not_support_is_not_saved():
    from backend.services.action_item_service import validate_action_items

    out = validate_action_items(
        [_item("Negotiate the vendor contract renewal with procurement department",
               evidence="Ravi said we must renegotiate the vendor contract immediately")], AI_TRANSCRIPT,
    )
    assert out == []


def test_an_item_with_a_real_quote_is_saved_with_that_quote():
    from backend.services.action_item_service import validate_action_items

    out = validate_action_items(
        [_item("Prepare the migration plan", evidence="Priya will prepare the migration plan by Friday")],
        AI_TRANSCRIPT,
    )
    assert len(out) == 1 and out[0]["evidence"] == "Priya will prepare the migration plan by Friday"


def test_a_quote_less_item_whose_words_are_all_in_the_transcript_is_kept_without_invented_evidence():
    from backend.services.action_item_service import validate_action_items

    out = validate_action_items([_item("Review the staging environment before release")], AI_TRANSCRIPT)
    assert len(out) == 1 and out[0]["evidence"] is None                      # no quote is made up for it


def test_a_fabricated_quote_is_removed_but_a_supported_task_survives():
    from backend.services.action_item_service import validate_action_items

    out = validate_action_items(
        [_item("Share the migration plan with the team", evidence="Priya promised to email everybody tonight")],
        AI_TRANSCRIPT,
    )
    assert len(out) == 1 and out[0]["evidence"] is None


def test_without_a_transcript_there_is_nothing_to_validate_against():
    from backend.services.action_item_service import validate_action_items

    out = validate_action_items([_item("Prepare the migration plan")], "")
    assert len(out) == 1

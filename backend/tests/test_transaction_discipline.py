"""Transaction / lock / connection discipline of the processing pipeline (the "processing is slow" audit).

Each test pins one defect that was MEASURED on a real 8-minute meeting:

* the upload request's DB session stayed "idle in transaction" for the whole pipeline (422 s),
* ``build_final_record`` flushed a row-locking placeholder and then embedded text for seconds; a GET poll and the
  pipeline waited on each other for 4 s,
* every GET poll rewrote ``json_data`` even when nothing had changed,
* a meeting was committed ``completed`` before its final JSON existed,
* ``StageTracker`` committed once per transition and swallowed a failed commit,
* the similarity search ran ~6 queries per candidate (N+1),
* a hung AIRAWAT LLM request was waited on for the full 30 s stream-idle timeout.

DB-backed tests run inside an outer transaction that is always rolled back and skip when PostgreSQL is unreachable.
"""

import http.server
import json
import threading
import time
import types
import uuid

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def db_session():
    from sqlalchemy import inspect
    from sqlalchemy.orm import Session

    from backend.database.connection import engine

    try:
        if not inspect(engine).has_table("meetings"):
            pytest.skip("meetings table does not exist yet (start the app once to create the schema)")
    except Exception as exc:  # database unreachable
        pytest.skip(f"database not reachable: {exc}")

    conn = engine.connect()
    outer = conn.begin()
    session = Session(bind=conn, join_transaction_mode="create_savepoint", autoflush=False)
    session.info["statements"] = statements = []

    from sqlalchemy import event

    @event.listens_for(conn, "before_cursor_execute")
    def _record(_conn, _cursor, statement, _params, _context, _many):
        statements.append(" ".join(statement.split()))

    try:
        yield session
    finally:
        event.remove(conn, "before_cursor_execute", _record)
        session.close()
        outer.rollback()
        conn.close()


def _public_id() -> str:
    return "9" + str(uuid.uuid4().int)[:14]


SENTENCES = [
    "We agreed the release moves to the second week of March because testing needs more time.",
    "Priya will own the migration plan and share it with the whole team by Friday afternoon.",
    "The budget for cloud hosting stays flat, but we must review the storage costs each month.",
    "Rahul raised a risk about the vendor contract and asked legal to review it this week.",
    "Everyone agreed that the customer onboarding flow needs a simpler first screen before launch.",
    "We will meet again next Tuesday to confirm the final scope and the remaining open questions.",
]


def _make_meeting(db, status="processing", n_segments=6, summary="The team moved the release and assigned owners."):
    """A committed (inside the test's outer transaction) meeting with segments and an analysis row."""
    from backend.models.models import AnalysisResult, Meeting, TranscriptSegment

    pid = _public_id()
    meeting = Meeting(
        bharatvc_meeting_id=pid, meeting_id=pid, display_id=pid, filename=f"{pid}.mp4", file_type="media",
        filepath=f"/tmp/{pid}.mp4", title="Planning sync", status=status, duration=120.0,
    )
    db.add(meeting)
    db.flush()
    for i in range(n_segments):
        db.add(TranscriptSegment(
            meeting_id=meeting.id, speaker=f"Speaker {1 + i % 2}", start_time=f"{i * 10:.3f}",
            end_time=f"{i * 10 + 9:.3f}", text=SENTENCES[i % len(SENTENCES)],
        ))
    db.add(AnalysisResult(
        # No parseable action-item text: a legacy meeting with such text and no ActionItem rows has them created (a
        # WRITE) while it is read, which - correctly - keeps the release from ending its transaction.
        meeting_id=meeting.id, summary=summary, mom="### Minutes\n- release moved", action_items="",
        decisions="Release moves to March", risks="Vendor contract",
    ))
    db.commit()
    return meeting.id


@pytest.fixture()
def fake_embedder(monkeypatch):
    """Replaces the embedding call the metric scorers use; records whether the DB session had a transaction open."""
    from backend.evaluation import reference_free_engine as rfe

    seen = {"calls": 0, "in_transaction": [], "session": None}

    def _embed(texts):
        seen["calls"] += 1
        if seen["session"] is not None:
            seen["in_transaction"].append(seen["session"].in_transaction())
        return []          # "embeddings unavailable": the scorers fall back to lexical overlap

    monkeypatch.setattr(rfe, "_embed_batch", _embed)
    return seen


def _final_table_writes(statements):
    return [i for i, s in enumerate(statements)
            if ('AI_Meeting_Inteligent_final' in s) and (s.startswith("INSERT") or s.startswith("UPDATE"))]


# ─────────────────────────────────────────────────────────────────────────────
# release_read_transaction
# ─────────────────────────────────────────────────────────────────────────────
def test_release_read_transaction_ends_a_clean_read_transaction(db_session):
    from backend.database.connection import release_read_transaction
    from backend.models.models import Meeting

    db_session.query(Meeting.id).limit(1).all()
    assert db_session.in_transaction()
    assert release_read_transaction(db_session) is True
    assert not db_session.in_transaction()


def test_release_read_transaction_never_discards_pending_work(db_session):
    from backend.database.connection import release_read_transaction
    from backend.models.models import Meeting

    pid = _public_id()
    pending = Meeting(bharatvc_meeting_id=pid, meeting_id=pid, display_id=pid, filename="p.mp4", file_type="media",
                      filepath="/tmp/p.mp4", title="pending", status="uploaded")
    db_session.add(pending)
    assert release_read_transaction(db_session) is False        # refused: it would have rolled the INSERT back
    assert pending in db_session.new


def test_release_read_transaction_never_discards_flushed_but_uncommitted_writes(db_session):
    """Regression: the first guard only saw UNFLUSHED changes. Rows already flushed (INSERTed, not committed) - which is
    what a legacy read path that creates action items does, and what a caller mid-transaction has - were rolled back."""
    from backend.database.connection import release_read_transaction
    from backend.models.models import Meeting

    pid = _public_id()
    db_session.add(Meeting(bharatvc_meeting_id=pid, meeting_id=pid, display_id=pid, filename="f.mp4", file_type="media",
                           filepath="/tmp/f.mp4", title="flushed", status="uploaded"))
    db_session.flush()                                             # written to PostgreSQL, not committed
    assert not db_session.new                                      # ...so it is no longer "pending" in the session

    assert release_read_transaction(db_session) is False
    assert db_session.query(Meeting.id).filter(Meeting.bharatvc_meeting_id == pid).first() is not None   # survived

    db_session.commit()                                            # once committed the session may release again
    assert release_read_transaction(db_session) is True


# ─────────────────────────────────────────────────────────────────────────────
# StageTracker
# ─────────────────────────────────────────────────────────────────────────────
class _FakeDb:
    def __init__(self, fail_commit=False):
        self.commits = self.rollbacks = 0
        self.fail_commit = fail_commit

    def commit(self):
        self.commits += 1
        if self.fail_commit:
            raise RuntimeError("connection lost")

    def rollback(self):
        self.rollbacks += 1


def _tracker(db):
    from backend.loop_engine.stage_manager import StageTracker

    meeting = types.SimpleNamespace(id=1, pipeline_stage_states_json=None, retry_history_json=None, status="processing",
                                    overall_status=None, error_message=None)
    return StageTracker(meeting, db), meeting


def test_stage_transitions_commit_once_when_batched():
    from backend.loop_engine.stage_manager import StageState

    db = _FakeDb()
    tracker, meeting = _tracker(db)
    for stage in ("summary", "mom", "agenda", "decision"):
        tracker.transition(stage, StageState.SUCCESS, "ok", commit=False)
    assert db.commits == 0                                   # nothing committed per transition
    assert json.loads(meeting.pipeline_stage_states_json)["agenda"] == "SUCCESS"   # but staged on the row
    tracker.save()
    assert db.commits == 1


def test_a_failed_stage_state_commit_is_rolled_back_and_raised_not_swallowed():
    from backend.loop_engine.stage_manager import StageState

    db = _FakeDb(fail_commit=True)
    tracker, _ = _tracker(db)
    with pytest.raises(RuntimeError, match="connection lost"):
        tracker.transition("summary", StageState.SUCCESS, "ok")
    assert db.rollbacks == 1


def test_completion_outcome_is_decided_without_writing():
    from backend.loop_engine.stage_manager import StageState

    db = _FakeDb()
    tracker, meeting = _tracker(db)
    tracker.transition("summary", StageState.SUCCESS, commit=False)
    tracker.transition("diarization", StageState.DEGRADED, commit=False)
    outcome = tracker.evaluate_completion()
    assert outcome == {"status": "completed", "overall_status": "COMPLETED_WITH_WARNINGS", "error_message": None}
    assert meeting.status == "processing" and db.commits == 0      # the decision alone changes nothing

    tracker.transition("stt", StageState.FAILED, commit=False)
    failed = tracker.evaluate_completion()
    assert failed["status"] == "error" and failed["overall_status"] == "FAILED" and "stt" in failed["error_message"]


# ─────────────────────────────────────────────────────────────────────────────
# build_final_record: compute first, embed with no transaction, write once
# ─────────────────────────────────────────────────────────────────────────────
def test_the_final_record_is_written_once_and_only_after_everything_was_read(db_session, fake_embedder):
    from backend.services.final_aggregation_service import build_final_record

    meeting_pk = _make_meeting(db_session)
    fake_embedder["session"] = db_session
    statements = db_session.info["statements"]
    del statements[:]

    record = build_final_record(meeting_pk, db_session)

    writes = _final_table_writes(statements)
    assert len(writes) == 1, statements                          # ONE INSERT; no BUILDING placeholder + UPDATE pair
    last_source_read = max(i for i, s in enumerate(statements)
                           if s.startswith("SELECT") and ("transcript_segments" in s or "analysis_results" in s))
    assert writes[0] > last_source_read                          # nothing locked the row until it was fully computed
    assert isinstance(record.json_data, dict) and record.json_data["segments"]


def test_metric_embedding_runs_with_no_database_transaction_open(db_session, fake_embedder):
    from backend.services.final_aggregation_service import build_final_record

    meeting_pk = _make_meeting(db_session)
    fake_embedder["session"] = db_session
    build_final_record(meeting_pk, db_session)

    assert fake_embedder["calls"] > 0
    assert fake_embedder["in_transaction"] and not any(fake_embedder["in_transaction"])


def test_rebuilding_an_unchanged_record_writes_nothing(db_session, fake_embedder):
    from backend.services.final_aggregation_service import build_final_record

    meeting_pk = _make_meeting(db_session)
    build_final_record(meeting_pk, db_session)
    db_session.commit()
    statements = db_session.info["statements"]
    del statements[:]

    build_final_record(meeting_pk, db_session)                   # what every GET poll used to turn into 2 UPDATEs
    assert _final_table_writes(statements) == []


def test_a_changed_meeting_is_rewritten(db_session, fake_embedder):
    from backend.models.models import AnalysisResult
    from backend.services.final_aggregation_service import build_final_record

    meeting_pk = _make_meeting(db_session)
    build_final_record(meeting_pk, db_session)
    db_session.commit()
    db_session.query(AnalysisResult).filter(AnalysisResult.meeting_id == meeting_pk).update({"summary": "New summary."})
    db_session.commit()
    statements = db_session.info["statements"]
    del statements[:]

    record = build_final_record(meeting_pk, db_session)
    assert len(_final_table_writes(statements)) == 1
    assert record.summary == "New summary."


def test_the_meeting_becomes_completed_in_the_same_write_as_its_final_json(db_session, fake_embedder):
    from backend.models.models import Meeting
    from backend.services.final_aggregation_service import build_final_record

    meeting_pk = _make_meeting(db_session, status="processing")
    outcome = {"status": "completed", "overall_status": "SUCCESS", "error_message": None}

    record = build_final_record(meeting_pk, db_session, meeting_updates=outcome)
    db_session.commit()

    assert record.processing_status == "COMPLETED"
    assert record.json_data["status"] == "completed"             # the stored JSON already says so
    stored = db_session.query(Meeting.status, Meeting.overall_status).filter(Meeting.id == meeting_pk).one()
    assert tuple(stored) == ("completed", "SUCCESS")


def test_a_record_without_final_json_is_failed_never_completed(db_session, fake_embedder, monkeypatch):
    from backend.api import meetings as meetings_api
    from backend.services.final_aggregation_service import build_final_record

    meeting_pk = _make_meeting(db_session, status="processing")

    def _boom(*args, **kwargs):
        raise RuntimeError("cannot serialise")

    monkeypatch.setattr(meetings_api, "build_meeting_response_dict", _boom)
    record = build_final_record(meeting_pk, db_session, meeting_updates={"status": "completed"})
    assert record.processing_status == "FAILED"
    assert "json_data" in (record.error_info or "")


def test_a_processing_meeting_is_not_advertised_as_completed(db_session, fake_embedder):
    from backend.services.final_aggregation_service import build_final_record

    meeting_pk = _make_meeting(db_session, status="processing")
    assert build_final_record(meeting_pk, db_session).processing_status == "BUILDING"


# ─────────────────────────────────────────────────────────────────────────────
# pipeline claim (two workers, one meeting)
# ─────────────────────────────────────────────────────────────────────────────
def test_only_one_worker_can_claim_a_meeting(db_session):
    from datetime import datetime, timezone

    from backend.models.models import Meeting
    from backend.workers.pipelines import _claim_meeting

    meeting_pk = _make_meeting(db_session, status="uploaded")
    now = datetime.now(timezone.utc)
    assert _claim_meeting(db_session, meeting_pk, now) is True
    assert db_session.query(Meeting.status).filter(Meeting.id == meeting_pk).scalar() == "processing"
    assert _claim_meeting(db_session, meeting_pk, now) is False  # the second worker is turned away


def test_claiming_a_missing_meeting_is_refused(db_session):
    from datetime import datetime, timezone

    from backend.workers.pipelines import _claim_meeting

    assert _claim_meeting(db_session, 2_000_000_000, datetime.now(timezone.utc)) is False


# ─────────────────────────────────────────────────────────────────────────────
# similarity: no N+1
# ─────────────────────────────────────────────────────────────────────────────
def test_candidate_snapshots_are_loaded_in_three_queries(db_session):
    from backend.repositories import MeetingRepository
    from backend.services import similarity_service

    ids = [_make_meeting(db_session, status="completed") for _ in range(5)]
    statements = db_session.info["statements"]
    del statements[:]

    snapshots = similarity_service.build_snapshots(MeetingRepository(db_session), ids)

    assert len(snapshots) == 5 and all(s.transcript for s in snapshots.values())
    selects = [s for s in statements if s.startswith("SELECT")]
    assert len(selects) == 3, statements                         # was 6 per meeting = 30 for these five


def test_a_single_snapshot_no_longer_rereads_the_meeting_row(db_session):
    from backend.repositories import MeetingRepository
    from backend.services import similarity_service

    meeting_pk = _make_meeting(db_session, status="completed")
    statements = db_session.info["statements"]
    del statements[:]

    similarity_service.build_snapshot(MeetingRepository(db_session), meeting_pk)
    meeting_selects = [s for s in statements if s.startswith("SELECT") and "FROM meetings" in s]
    assert len(meeting_selects) == 1, statements                 # get_segments / get_analysis used to add two more


def test_find_similar_meetings_uses_short_sessions_and_never_holds_one_across_the_vector_search(db_session, monkeypatch):
    from backend.services import similarity_service

    current = _make_meeting(db_session, status="completed")
    others = [_make_meeting(db_session, status="completed") for _ in range(3)]
    opened = []

    class _Factory:
        """A session factory whose sessions share the test connection; records which are open at any time."""

        def __call__(self):
            from sqlalchemy.orm import Session

            session = Session(bind=db_session.get_bind(), join_transaction_mode="create_savepoint")
            opened.append(session)
            return session

    open_during_search = []

    def _candidates(meeting_pk, text, top_k=25):
        open_during_search.append([s.in_transaction() for s in opened])
        return [{"meeting_id": o, "meeting_code": str(o), "semantic_score": 0.8, "metadata": {}, "preview": ""}
                for o in others]

    monkeypatch.setattr(similarity_service.vector_service, "get_semantic_candidates", _candidates)
    result = similarity_service.find_similar_meetings(
        None, current, top_k=8, min_score=0.0, session_factory=_Factory(),
    )
    assert open_during_search == [[False]]       # the read session that resolved the meeting was closed before the search
    assert {m["meeting_id"] for m in result["meetings"]} <= set(others)


# ─────────────────────────────────────────────────────────────────────────────
# a poll racing the pipeline's commit (observed once in the audit's AFTER run)
# ─────────────────────────────────────────────────────────────────────────────
def test_diarization_consistency_survives_a_missing_audio_duration():
    """A GET can read the meeting row just before the pipeline commits its duration and the transcript just after:
    segments present, audio duration unknown -> coverage None. That used to raise TypeError."""
    from backend.evaluation import reference_free_engine as rfe

    segments = [
        {"speaker": "Speaker 1", "start_time": "0.000", "end_time": "9.000", "text": "one"},
        {"speaker": "Speaker 2", "start_time": "10.000", "end_time": "19.000", "text": "two"},
    ]
    result = rfe.diarization_consistency(segments, None)
    assert result["components"]["coverage"] is None
    assert result["components"]["assignment_rate"] == 100.0
    assert result["score"] is not None                       # the other components still score


# ─────────────────────────────────────────────────────────────────────────────
# retries, diarization overlap
# ─────────────────────────────────────────────────────────────────────────────
def test_a_report_retry_is_reported_to_the_caller(monkeypatch):
    from backend.workers import pipelines

    monkeypatch.setattr(pipelines.time, "sleep", lambda s: None)
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("gateway did not answer")
        return {"summary": "ok"}

    monkeypatch.setattr(pipelines, "process_meeting_report", flaky)
    retries = []
    result = pipelines._generate_reports_with_retry(7, "t", on_retry=lambda attempt, err: retries.append((attempt, str(err))))
    assert result == {"summary": "ok"}
    assert retries == [(2, "gateway did not answer")]


def test_a_precomputed_primary_pass_is_not_run_again_but_the_fallback_still_applies(monkeypatch):
    from backend.config.config import settings
    from backend.diarization.manager import DiarizationManager

    monkeypatch.setattr(settings, "DIARIZATION_MODE", "NEMO_PRIMARY_PYANNOTE_FALLBACK")
    manager = DiarizationManager()
    ran = []
    monkeypatch.setattr(manager.nemo_engine, "diarize", lambda *a, **k: ran.append("nemo") or ([], "FAILED", "x"))
    monkeypatch.setattr(manager.pyannote_engine, "diarize", lambda *a, **k: ran.append("pyannote") or ([], "FAILED", "y"))

    manager.execute("a.wav", primary_result=([], "FAILED", "nemo said no"))
    assert "nemo" not in ran                     # the audio pass that ran alongside transcription is reused
    assert "pyannote" in ran                     # validation + fallback proceed exactly as before


# ─────────────────────────────────────────────────────────────────────────────
# AIRAWAT: response-start timeout for streamed requests
# ─────────────────────────────────────────────────────────────────────────────
class _Gateway:
    """A tiny local HTTP server whose behaviour per request is set by the test."""

    def __init__(self):
        outer = self
        self.header_delay = 0.0
        self.chunk_gap = 0.0
        self.chunks = 1

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                time.sleep(outer.header_delay)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for i in range(outer.chunks):
                    if i:
                        time.sleep(outer.chunk_gap)
                    payload = f"chunk{i}\n".encode()
                    self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")

            def log_message(self, *args):  # silence
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1/chat/completions"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def gateway():
    server = _Gateway()
    yield server
    server.close()


def _transport_client():
    import httpx

    from backend.services.model_router import _AirawatTimedTransport

    return httpx.Client(transport=_AirawatTimedTransport(httpx.HTTPTransport()))


def _stream_call(client, url, stream=True):
    import httpx

    timeout = httpx.Timeout(connect=2, read=5, write=2, pool=2)      # "idle" limit: 5 s
    t0 = time.monotonic()
    with client.stream("POST", url, json={"stream": stream}, timeout=timeout) as response:
        body = b"".join(response.iter_bytes())
    return body, time.monotonic() - t0


def test_a_streamed_request_that_never_starts_fails_at_the_first_byte_limit_not_the_idle_limit(gateway, monkeypatch):
    import httpx

    from backend.config.config import settings

    monkeypatch.setattr(settings, "AIRAWAT_LLM_FIRST_BYTE_TIMEOUT_SEC", 1.0)
    gateway.header_delay = 4.0                                       # headers arrive after 4 s: "hung"
    client = _transport_client()
    t0 = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        _stream_call(client, gateway.url)
    assert time.monotonic() - t0 < 3.0                               # cut at ~1 s, not the 5 s idle limit


def test_a_long_streamed_answer_is_not_cut_by_the_first_byte_limit(gateway, monkeypatch):
    from backend.config.config import settings

    monkeypatch.setattr(settings, "AIRAWAT_LLM_FIRST_BYTE_TIMEOUT_SEC", 1.0)
    gateway.header_delay = 0.0
    gateway.chunks, gateway.chunk_gap = 3, 1.5                       # each gap is longer than the first-byte limit
    body, elapsed = _stream_call(_transport_client(), gateway.url)
    # Each 1.5 s gap exceeds the 1 s first-byte limit yet the stream completes: the idle limit (5 s) governed the body.
    # (elapsed is ~3.0 s; the bound only has to be clearly above the first-byte limit, timer resolution varies.)
    assert body.count(b"chunk") == 3 and elapsed > 2.0


def test_a_non_streamed_request_keeps_waiting_for_its_headers(gateway, monkeypatch):
    from backend.config.config import settings

    monkeypatch.setattr(settings, "AIRAWAT_LLM_FIRST_BYTE_TIMEOUT_SEC", 1.0)
    gateway.header_delay = 1.6                                       # a non-streamed answer legitimately starts late
    body, _ = _stream_call(_transport_client(), gateway.url, stream=False)
    assert b"chunk0" in body


def test_the_first_byte_limit_can_be_disabled(gateway, monkeypatch):
    from backend.config.config import settings

    monkeypatch.setattr(settings, "AIRAWAT_LLM_FIRST_BYTE_TIMEOUT_SEC", 0)
    gateway.header_delay = 1.6
    body, _ = _stream_call(_transport_client(), gateway.url)
    assert b"chunk0" in body


# ─────────────────────────────────────────────────────────────────────────────
# upload: the request's session is closed before the background pipeline starts
# ─────────────────────────────────────────────────────────────────────────────
def test_the_upload_session_is_released_before_the_background_pipeline_runs(db_session, monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api import upload as upload_api
    from backend.api.deps import get_db
    from backend.config.config import settings

    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    state = {}

    def _pipeline(meeting_pk, path):
        # What the pipeline sees when it starts: is the request's session still inside a transaction?
        state["in_transaction_at_pipeline_start"] = db_session.in_transaction()
        state["meeting_pk"] = meeting_pk

    monkeypatch.setitem(upload_api.PIPELINE_REGISTRY, "document", _pipeline)

    def _db():
        yield db_session          # FastAPI closes/keeps it after the response, exactly like get_db

    app = FastAPI()
    app.include_router(upload_api.router)
    app.dependency_overrides[get_db] = _db
    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"Discussed the plan. Priya owns the migration.", "text/plain")},
        data={"bharatvc_meeting_id": _public_id()},
    )
    assert response.status_code == 200, response.text
    assert state["meeting_pk"] == response.json()["id"]
    assert state["in_transaction_at_pipeline_start"] is False


def test_a_meeting_that_is_uploaded_or_processing_cannot_be_resubmitted(db_session, monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api import upload as upload_api
    from backend.api.deps import get_db
    from backend.config.config import settings
    from backend.models.models import Meeting

    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(upload_api.router)
    app.dependency_overrides[get_db] = lambda: (yield db_session)   # noqa: E731 - generator lambda
    client = TestClient(app)

    for status in ("uploaded", "processing"):
        pk = _make_meeting(db_session, status=status)
        public_id = db_session.query(Meeting.bharatvc_meeting_id).filter(Meeting.id == pk).scalar()
        response = client.post(
            "/upload",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
            data={"bharatvc_meeting_id": public_id},
        )
        assert response.status_code == 409, (status, response.text)

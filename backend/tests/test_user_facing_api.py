"""The API contract of the web app's screens (PostgreSQL, rolled back after each test).

Pinned defects of the first web-app integration:
* GET /meetings ignored page/limit/status/search and returned every row (with the server file path);
* /status invented progress (50 % for any running meeting, 100 % for a failed one) and showed the raw exception;
* /final answered "not ready" as a 202 *exception* that the app rendered as an empty COMPLETED meeting, and served a
  stale COMPLETED record as the current state after a re-run failed;
* /retry and DELETE accepted a meeting that was still processing;
* chat stored "I encountered an error" as a real answer (HTTP 200) when the AI call failed;
* there was no cross-meeting search endpoint at all.
"""

import json
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.deps import get_db, get_meeting_repository
from backend.repositories import MeetingRepository
from backend.utils.processing_status import display_status, user_error


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


@pytest.fixture()
def client(db_session, monkeypatch):
    from backend.api import chat, final, meetings, search, system, upload

    app = FastAPI()
    for module in (meetings, final, search, system, upload, chat):
        app.include_router(module.router, prefix="/api/v1")
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_meeting_repository] = lambda: MeetingRepository(db_session)
    monkeypatch.setattr("backend.api.deps.release_session_before_background_task", lambda db: None)
    return TestClient(app)


def _meeting(db, tag, **extra):
    from backend.models.models import Meeting

    pid = "8" + str(uuid.uuid4().int)[:14]
    fields = dict(bharatvc_meeting_id=pid, meeting_id=pid, display_id=pid, filename=f"{tag}-{pid}.mp4",
                  file_type="media", filepath=f"/srv/private/uploads/{pid}.mp4", title=f"{tag} meeting",
                  status="completed", duration=600.0)
    fields.update(extra)
    m = Meeting(**fields)
    db.add(m)
    db.flush()
    return m


# ─── user-facing status vocabulary ────────────────────────────────────────────


@pytest.mark.parametrize("raw, expected", [
    ("completed", "completed"), ("error", "failed"), ("failed", "failed"),
    ("uploaded", "processing"), ("processing", "processing"), (None, "processing"),
])
def test_display_status(raw, expected):
    assert display_status(raw) == expected


@pytest.mark.parametrize("technical, code", [
    ("Processing failed: The spoken language is unknown: AIRAWAT Whisper cannot detect it.", "LANGUAGE_REQUIRED"),
    ("Speech-to-text is not configured: AI_PROVIDER='ollama' does not transcribe audio", "SERVICE_NOT_CONFIGURED"),
    ("Speech-to-text rejected chunk 1/2: HTTP 401 Unauthorized: Invalid API Key", "SERVICE_NOT_CONFIGURED"),
    ("Processing failed: Transcription failed — no text produced (reason: groq_returned_no_speech_text)", "NO_SPEECH"),
    ("Processing failed: ffprobe: Invalid data found when processing input", "MEDIA_UNREADABLE"),
    ("AI Provider failed to generate: summary (TimeoutError: did not answer)", "AI_SERVICE_UNAVAILABLE"),
    ("Processing failed: InFailedSqlTransaction: current transaction is aborted", "PROCESSING_FAILED"),
])
def test_user_error_never_repeats_the_technical_cause(technical, code):
    err = user_error(technical)
    assert err["code"] == code
    for word in ("AIRAWAT", "ollama", "HTTP", "ffprobe", "InFailedSqlTransaction", "groq", "Whisper"):
        assert word.lower() not in err["message"].lower()


# ─── GET /meetings ────────────────────────────────────────────────────────────


def test_the_list_paginates_filters_and_counts_all_matching_meetings(client, db_session):
    tag = "zz" + uuid.uuid4().hex[:8]
    for i in range(3):
        _meeting(db_session, tag, title=f"{tag} done {i}")
    failed = _meeting(db_session, tag, status="error", error_message="Processing failed: ffmpeg exited with 1 at /srv/x")
    _meeting(db_session, tag, status="processing")
    db_session.commit()

    body = client.get("/api/v1/meetings", params={"search": tag, "limit": 2, "page": 1}).json()
    assert body["total"] == 5 and body["pages"] == 3 and len(body["items"]) == 2
    assert body["status_counts"] == {"completed": 3, "processing": 1, "failed": 1}

    failed_page = client.get("/api/v1/meetings", params={"search": tag, "status": "failed"}).json()
    assert [m["id"] for m in failed_page["items"]] == [failed.id] and failed_page["total"] == 1
    item = failed_page["items"][0]
    assert item["status"] == "failed" and item["error"]["code"] == "MEDIA_UNREADABLE"
    assert "filepath" not in item and "/srv" not in json.dumps(item) and "ffmpeg" not in json.dumps(item)

    assert client.get("/api/v1/meetings", params={"status": "bogus"}).status_code == 422


def test_the_unpaginated_list_keeps_its_shape_but_not_the_server_path(client, db_session):
    m = _meeting(db_session, "legacy")
    db_session.commit()
    rows = client.get("/api/v1/meetings").json()
    assert isinstance(rows, list)
    mine = next(r for r in rows if r["id"] == m.id)
    assert "filepath" not in mine and mine["filename"] == m.filename


# ─── GET /meetings/{id}/status ────────────────────────────────────────────────


def test_status_progress_comes_from_the_recorded_stage_states(client, db_session):
    states = {"audio_extraction": "SUCCESS", "stt": "SUCCESS", "diarization": "PROCESSING", "summary": "PENDING"}
    m = _meeting(db_session, "status", status="processing", pipeline_stage_states_json=json.dumps(states))
    db_session.commit()
    body = client.get(f"/api/v1/meetings/{m.id}/status").json()
    assert body["status"] == "processing" and 0 < body["progress_percent"] < 50
    assert body["stages"]["diarization"]["status"] == "in_progress"
    assert body["stages"]["summary"]["status"] == "pending"          # used to be reported as running
    steps = {s["key"]: s["state"] for s in body["steps"]}
    assert steps == {"upload": "completed", "transcribe": "in_progress", "results": "pending", "done": "pending"}


def test_a_failed_meeting_reports_a_safe_reason_and_no_fake_100_percent(client, db_session):
    states = {"audio_extraction": "SUCCESS", "stt": "FAILED"}
    m = _meeting(db_session, "failed", status="error", pipeline_stage_states_json=json.dumps(states),
                 error_message="Processing failed: Speech-to-text did not answer chunk 2/3 after 4 attempt(s) "
                               "(https://api.groq.com/openai/v1/audio/transcriptions)")
    db_session.commit()
    body = client.get(f"/api/v1/meetings/{m.id}/status").json()
    assert body["status"] == "failed" and body["overall_status"] == "FAILED"
    assert body["progress_percent"] < 100
    assert body["error_code"] == "AI_SERVICE_UNAVAILABLE" and "groq" not in body["error"].lower()
    assert body["steps"][1]["state"] == "failed" and body["steps"][-1]["label"] == "Needs attention"


# ─── /final ───────────────────────────────────────────────────────────────────


def test_final_without_results_is_a_real_202_with_the_live_state(client, db_session):
    m = _meeting(db_session, "pending", status="processing")
    db_session.commit()
    r = client.get(f"/api/v1/meetings/{m.id}/final")
    assert r.status_code == 202
    body = r.json()
    assert body["processing_status"] == "PENDING" and body["status"] == "processing"
    assert body["detail"]["processing_status"] == "PENDING"            # the body earlier clients parsed


def test_a_stored_record_is_stamped_with_the_meeting_s_current_failure(client, db_session):
    from backend.models.models import AIMeetingIntelFinal

    m = _meeting(db_session, "stale", status="error", error_message="Processing failed: AI Provider failed to generate: summary")
    db_session.add(AIMeetingIntelFinal(meeting_pk=m.id, processing_status="COMPLETED", meeting_id=m.meeting_id))
    db_session.commit()
    body = client.get(f"/api/v1/meetings/{m.id}/final").json()
    assert body["processing_status"] == "COMPLETED"                      # what the stored record is
    assert body["display_status"] == "failed"                            # what is true now
    assert body["user_error"]["code"] == "AI_SERVICE_UNAVAILABLE"


# ─── retry / delete while processing ──────────────────────────────────────────


def test_retry_and_delete_refuse_a_meeting_that_is_still_processing(client, db_session):
    m = _meeting(db_session, "busy", status="processing")
    db_session.commit()
    assert client.post(f"/api/v1/meetings/{m.id}/retry").status_code == 409
    assert client.delete(f"/api/v1/meetings/{m.id}").status_code == 409


def test_retry_of_a_missing_recording_does_not_reveal_the_server_path(client, db_session):
    m = _meeting(db_session, "gone", status="error")
    db_session.commit()
    r = client.post(f"/api/v1/meetings/{m.id}/retry")
    assert r.status_code == 410 and "/srv/private" not in r.text


# ─── search ───────────────────────────────────────────────────────────────────


def test_search_returns_only_existing_completed_meetings_with_source_references(client, db_session, monkeypatch):
    from backend.api import search as search_api

    done = _meeting(db_session, "srch", title="Budget review")
    running = _meeting(db_session, "srch", status="processing")
    db_session.commit()
    hits = [
        {"meeting_id": done.id, "start_time": "61.5", "end_time": "70.0", "speakers": "Speaker_1", "similarity": 0.62,
         "text": "We approved the budget."},
        {"meeting_id": running.id, "start_time": "1", "end_time": "2", "speakers": None, "similarity": 0.6, "text": "x"},
        {"meeting_id": 987654321, "start_time": "1", "end_time": "2", "speakers": None, "similarity": 0.5, "text": "deleted"},
    ]
    monkeypatch.setattr(search_api.vector_service, "search_all_meetings", lambda q, top_k, min_similarity: hits)
    body = client.get("/api/v1/search", params={"q": "budget approval"}).json()
    assert [r["meeting_id"] for r in body["results"]] == [done.id]
    r = body["results"][0]
    assert r["meeting_title"] == "Budget review" and r["start_seconds"] == 61.5 and r["speaker"] == "Speaker_1"


def test_search_says_unavailable_instead_of_returning_no_results(client, monkeypatch):
    from backend.api import search as search_api
    from backend.services.vector_service import VectorSearchUnavailable

    def down(*a, **k):
        raise VectorSearchUnavailable("ChromaDB directory locked by another process")

    monkeypatch.setattr(search_api.vector_service, "search_all_meetings", down)
    r = client.get("/api/v1/search", params={"q": "budget"})
    assert r.status_code == 503 and "chroma" not in r.text.lower()


# ─── upload options / public status ───────────────────────────────────────────


def test_upload_options_name_no_provider_and_follow_the_stt_capability(client, monkeypatch):
    from backend.config.config import settings
    from backend.providers import provider_factory

    body = client.get("/api/v1/upload/options").json()          # default: AIRAWAT cannot detect languages
    assert body["language_detection"] is False and ".mp4" in body["accepted_extensions"]
    assert body["max_file_size_mb"] == settings.MAX_FILE_SIZE_MB
    assert "airawat" not in json.dumps(body).lower()

    monkeypatch.setattr(settings, "AI_PROVIDER", "groq")
    provider_factory.clear_cache()
    assert client.get("/api/v1/upload/options").json()["language_detection"] is True


def test_public_service_status_exposes_no_internals(client, monkeypatch):
    from backend.api import system as system_api

    monkeypatch.setattr(system_api.vector_service, "health", lambda: {"backend": "chromadb", "status": "unavailable"})
    body = client.get("/api/v1/system/status").json()
    assert body == {"status": "degraded", "can_view_meetings": True, "can_upload": True, "can_search": False}


# ─── chat ─────────────────────────────────────────────────────────────────────


def test_a_failed_chat_answer_is_an_error_and_is_not_stored(client, db_session, monkeypatch):
    from backend.api import chat as chat_api
    from backend.models.models import ChatSession

    m = _meeting(db_session, "chat")
    db_session.commit()
    monkeypatch.setattr(chat_api.vector_service, "has_transcript_chunks", lambda mid: True)
    monkeypatch.setattr(chat_api, "generate_chatbot_response", lambda **k: (_ for _ in ()).throw(TimeoutError("gateway")))
    r = client.post(f"/api/v1/meetings/{m.id}/chat", json={"message": "What was decided?"})
    assert r.status_code == 503 and "gateway" not in r.text
    assert db_session.query(ChatSession).filter(ChatSession.meeting_id == m.id).count() == 0


# ─── reduced-quality speaker separation is reported, not called SUCCESS ───────


def test_an_engine_degraded_verdict_survives_a_passing_validation(monkeypatch):
    """E2E 2026-10-21: without the neural model the acoustic fallback found 1 of 2 voices and it was stored as SUCCESS."""
    from backend.diarization.manager import DiarizationManager
    from backend.diarization.validator import DiarizationValidationResult

    mgr = DiarizationManager()
    monkeypatch.setattr(mgr.validator, "validate", lambda *a, **k: DiarizationValidationResult(passed=True, status="SUCCESS"))
    turns = [{"start": 0.0, "end": 5.0, "speaker": "Speaker_1"}]
    segments, meta, status = mgr.execute("x.wav", primary_result=(turns, "DEGRADED", "reduced-accuracy fallback"))
    assert status == "DEGRADED" and segments == turns and meta["engine_note"] == "reduced-accuracy fallback"
    segments, meta, status = mgr.execute("x.wav", primary_result=(turns, "SUCCESS", None))
    assert status == "SUCCESS"


def test_final_carries_plain_warnings_for_degraded_stages(client, db_session):
    from backend.models.models import AIMeetingIntelFinal
    from backend.utils.processing_status import user_warnings

    assert user_warnings(json.dumps({"stt": "SUCCESS", "diarization": "DEGRADED"})) == [
        "Speakers may not be identified accurately in this recording."]
    assert user_warnings(None) == [] and user_warnings("not json") == []

    m = _meeting(db_session, "warn", pipeline_stage_states_json=json.dumps({"diarization": "DEGRADED", "summary": "SUCCESS"}))
    db_session.add(AIMeetingIntelFinal(meeting_pk=m.id, processing_status="COMPLETED", meeting_id=m.meeting_id))
    db_session.commit()
    body = client.get(f"/api/v1/meetings/{m.id}/final").json()
    assert body["warnings"] == ["Speakers may not be identified accurately in this recording."]
    assert "nemo" not in json.dumps(body["warnings"]).lower()

"""AIRAWAT is the only AI provider: a failure is retried and REPORTED, never silently replaced by another engine.

Found by running the platform for real: one stalled startup probe made the environment selector install Ollama's
dummy key as THE AIRAWAT key (Whisper then answered 401 and local Whisper took over), the LLM asked Ollama for a model
it did not have, and every report came from the rule-based "extractive" engine - which also invents decisions and
action items for meetings that have none. Since then local Whisper, Ollama and the rule-based stand-ins were removed
altogether: EVERY mode, AI_MODE=local included, uses AIRAWAT only (AI_MODE just picks the database / credentials).
"""

import json
from types import SimpleNamespace

import pytest
import requests

from backend.api import meetings as meetings_api
from backend.config import environment_selector as es
from backend.config.config import settings
from backend.services import airawat_health as ah
from backend.services import llm_workflow as wf
from backend.services import transcription_service as ts
from backend.services import translation_service as tr
from backend.services.model_router import model_router
from backend.services.summary_service import SummaryResult, plan_summary_length
from backend.utils.exceptions import AirawatError, AirawatNotConfiguredError, LanguageIdentificationError
from backend.workers import pipelines

SECRET = "sk-very-secret-airawat-key-987654321"


@pytest.fixture()
def strict(monkeypatch):
    monkeypatch.setattr(settings, "AI_MODE", "auto")
    monkeypatch.setattr(settings, "AIRAWAT_API_KEY", "")


@pytest.fixture()
def local(monkeypatch):
    monkeypatch.setattr(settings, "AI_MODE", "local")
    monkeypatch.setattr(settings, "AIRAWAT_API_KEY", "")


def _response(status, body, headers=None):
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(body).encode()
    r.headers["content-type"] = "application/json"
    r.headers.update(headers or {})
    r.url = "https://airawat.example/whisper"
    r.reason = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 500: "Internal Server Error"}.get(status, "")
    return r


# ── environment selector: a failed probe must not re-point the platform ────────────────────────────


def _selector(monkeypatch, *, mode, airawat_ok, key=SECRET):
    sel = es.EnvironmentSelector()
    monkeypatch.setenv("AI_MODE", mode)
    monkeypatch.setenv("LOCAL_AIRAWAT_API_KEY", key)
    monkeypatch.setenv("LOCAL_AIRAWAT_BASE_URL", "https://airawat.example/llm/v1")
    monkeypatch.setenv("LOCAL_DATABASE_URL", "postgresql://u:p@db.example/x")
    monkeypatch.setattr(sel, "_check_database", lambda *a, **k: True)
    monkeypatch.setattr(sel, "_check_airawat", lambda *a, **k: airawat_ok)
    return sel


def test_a_failed_startup_probe_keeps_airawat_in_every_mode(monkeypatch):
    for mode in ("auto", "server", "cloud", "local"):
        cfg = _selector(monkeypatch, mode=mode, airawat_ok=False)._try_local()
        assert cfg.airawat_api_key == SECRET and cfg.airawat_base_url == "https://airawat.example/llm/v1", mode


def test_there_is_no_ollama_substitution_any_more():
    assert not hasattr(es.EnvironmentSelector, "_check_ollama")
    assert not hasattr(settings, "local_ai_allowed")


def test_a_missing_key_is_never_replaced_by_another_engine(monkeypatch):
    for mode in ("auto", "local"):
        cfg = _selector(monkeypatch, mode=mode, airawat_ok=True, key="")._try_local()
        assert cfg.airawat_api_key == "", mode


# ── routing: no key is an error, not a switch to local Whisper ─────────────────────────────────


def test_transcriber_routing_follows_the_policy(monkeypatch, strict):
    def with_key(key):
        monkeypatch.setattr(model_router, "_get_dynamic_config", lambda: {"airawat": {"api_key": key}})

    with_key(SECRET)
    assert model_router.get_transcriber() is ts.transcribe_audio_airawat
    for mode in ("auto", "local"):
        monkeypatch.setattr(settings, "AI_MODE", mode)
        for missing in ("", "   "):
            with_key(missing)
            with pytest.raises(AirawatNotConfiguredError):
                model_router.get_transcriber()


def test_no_local_whisper_code_is_left():
    for name in ("transcribe_audio_local", "_transcribe_chunk_locally", "_get_local_whisper_model",
                 "_transcribe_audio_local_single", "_load_local_whisper_model"):
        assert not hasattr(ts, name), name
    from backend.services import language_service
    assert not hasattr(language_service, "detect_language")


def test_the_llm_client_refuses_to_start_without_a_key(monkeypatch, strict):
    monkeypatch.setattr(model_router, "_get_dynamic_config", lambda: {"airawat": {"api_key": ""}})
    with pytest.raises(AirawatNotConfiguredError):
        model_router.get_llm()


# ── AIRAWAT Whisper request handling ───────────────────────────────────────────────────────────


@pytest.fixture()
def chunk_file(tmp_path):
    path = tmp_path / "chunk.wav"
    path.write_bytes(b"RIFF....WAVEfmt fake")
    return str(path)


@pytest.fixture()
def whisper(monkeypatch, strict):
    """Fake AIRAWAT Whisper: records every request; `script` decides the answers."""
    calls, script = [], []
    monkeypatch.setattr(model_router, "_get_dynamic_config",
                        lambda: {"airawat": {"api_key": SECRET, "retry_count": 1, "base_url": "https://airawat.example/v1"}})
    monkeypatch.setattr(settings, "AIRAWAT_WHISPER_URL", "https://airawat.example/whisper")
    monkeypatch.setattr(ts.time, "sleep", lambda s: None)

    def fake_post(url, headers=None, files=None, data=None, **kw):
        calls.append({"url": url, "data": dict(getattr(data, "fields", data) or {}), "headers": headers})
        item = script[min(len(calls) - 1, len(script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(ts.requests, "post", fake_post)
    return SimpleNamespace(calls=calls, script=script)


def test_whisper_without_a_language_fails_before_any_request(whisper, chunk_file):
    with pytest.raises(LanguageIdentificationError) as err:
        ts.transcribe_single_chunk(chunk_file, 0, 1, language=None)
    assert err.value.error_code == "LANGUAGE_NOT_IDENTIFIED"
    assert "`language`" in str(err.value) and "AIRAWAT_WHISPER_LANGUAGE" in str(err.value)
    assert "local" not in str(err.value).lower() and "models/whisper" not in str(err.value)
    assert whisper.calls == []                      # the gateway would have answered 400 "need model or language"


def test_whisper_sends_the_language_and_never_a_model_field(whisper, chunk_file):
    whisper.script.append(_response(200, {"text": "hello there", "language_code": "hi"}))
    out = ts.transcribe_single_chunk(chunk_file, 0, 1, language="HI")
    assert out["text"] == "hello there"
    assert whisper.calls[0]["data"] == {"language": "hi", "language_code": "hi"}      # a `model` field is REJECTED by the gateway
    assert whisper.calls[0]["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert whisper.calls[0]["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")


@pytest.mark.parametrize("status,detail", [(400, "Bad Request, bad model: whisper-1"), (401, "Bad token; invalid JSON"),
                                           (404, "Model not found for language xx")])
def test_a_permanent_http_error_is_not_retried_and_says_why(whisper, chunk_file, status, detail):
    whisper.script.append(_response(status, {"detail": detail}))
    with pytest.raises(AirawatError) as err:
        ts.transcribe_single_chunk(chunk_file, 0, 1, language="en")
    assert len(whisper.calls) == 1                  # retrying an identical request cannot help
    assert f"HTTP {status}" in str(err.value) and detail in str(err.value)
    assert SECRET not in str(err.value)


def test_a_server_error_is_retried_then_reported(whisper, chunk_file):
    whisper.script.append(_response(500, {"detail": "overloaded"}))
    with pytest.raises(AirawatError) as err:
        ts.transcribe_single_chunk(chunk_file, 0, 1, language="en")
    assert len(whisper.calls) == 3 and "after 3 attempt(s)" in str(err.value)
    assert SECRET not in str(err.value)


def test_a_timeout_is_retried_then_reported(whisper, chunk_file):
    whisper.script.append(requests.exceptions.ReadTimeout("read timed out"))
    with pytest.raises(AirawatError):
        ts.transcribe_single_chunk(chunk_file, 0, 1, language="en")
    assert len(whisper.calls) == 3


def test_a_first_read_timeout_is_retried_with_backoff_and_then_succeeds(whisper, chunk_file, monkeypatch):
    waits = []
    monkeypatch.setattr(ts.time, "sleep", waits.append)
    monkeypatch.setattr(ts.random, "uniform", lambda a, b: 0.0)
    monkeypatch.setattr(settings, "AIRAWAT_RETRY_BACKOFF_BASE_SEC", 2.0)
    whisper.script.extend([requests.exceptions.ReadTimeout("read timed out"),
                           requests.exceptions.ConnectTimeout("connect timed out"),
                           _response(200, {"text": "hello there"})])
    out = ts.transcribe_single_chunk(chunk_file, 0, 1, language="en")
    assert out["text"] == "hello there" and len(whisper.calls) == 3
    assert waits == [2.0, 4.0]                       # exponential: 2 s, then 4 s


def test_whisper_uses_separate_connect_and_read_timeouts_scaled_by_audio_length(monkeypatch):
    monkeypatch.setattr(settings, "AIRAWAT_WHISPER_CONNECT_TIMEOUT_SEC", 20.0)
    monkeypatch.setattr(settings, "AIRAWAT_WHISPER_READ_TIMEOUT_SEC", 180.0)
    monkeypatch.setattr(settings, "AIRAWAT_WHISPER_READ_TIMEOUT_PER_MIN_SEC", 10.0)
    monkeypatch.setattr(settings, "AIRAWAT_WHISPER_READ_TIMEOUT_MAX_SEC", 300.0)
    per_min = settings.BYTES_PER_SEC_WAV * 60
    assert ts._whisper_timeouts(0) == (20.0, 180.0)
    assert ts._whisper_timeouts(8 * per_min) == (20.0, 260.0)     # an 8-minute chunk waits longer
    assert ts._whisper_timeouts(60 * per_min)[1] == 300.0          # never past the gateway's own cut-off


def test_the_audio_is_uploaded_in_blocks_as_valid_multipart(tmp_path):
    # One sendall of the whole body is bounded end-to-end by the connect timeout (a healthy 14.5 MB upload was cut
    # off, a stalled one took 252 s to notice); streamed in blocks, the timeout bounds each block instead.
    from email.parser import BytesParser
    from email.policy import HTTP

    audio = bytes(range(256)) * 400
    path = tmp_path / "chunk.wav"
    path.write_bytes(audio)
    with open(path, "rb") as f:
        body = ts._StreamingMultipartBody({"language": "hi", "prompt": 'Ravi "Q3" roadmap'}, "chunk.wav", f, "audio/wav")
        blocks = iter(lambda: body.read(16384), b"")
        raw = b"".join(blocks)
    assert len(raw) == len(body) and not hasattr(body, "__iter__")   # Content-Length known -> not sent chunked
    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + body.content_type.encode() + b"\r\n\r\n" + raw)
    parts = {p.get_param("name", header="content-disposition"): p for p in msg.iter_parts()}
    assert parts["language"].get_content() == "hi" and parts["prompt"].get_content() == 'Ravi "Q3" roadmap'
    assert parts["file"].get_filename() == "chunk.wav" and parts["file"].get_content_type() == "audio/wav"
    assert parts["file"].get_payload(decode=True) == audio


def test_a_rate_limit_honours_retry_after(whisper, chunk_file, monkeypatch):
    waits = []
    monkeypatch.setattr(ts.time, "sleep", waits.append)
    whisper.script.extend([_response(429, {"detail": "slow down"}, headers={"Retry-After": "7"}),
                           _response(200, {"text": "ok"})])
    assert ts.transcribe_single_chunk(chunk_file, 0, 1, language="en")["text"] == "ok"
    assert waits == [7.0]


@pytest.mark.parametrize("mode", ["auto", "local"])
def test_a_chunk_failure_is_reported_in_every_mode(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(settings, "AI_MODE", mode)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    monkeypatch.setattr(model_router, "_get_dynamic_config", lambda: {"airawat": {"api_key": SECRET}})
    monkeypatch.setattr(ts, "split_audio_into_chunks", lambda p: [(str(audio), 0.0, True)])
    monkeypatch.setattr(ts, "get_audio_info", lambda p: {"duration": 5.0})

    def boom(*a, **k):
        raise AirawatError("AIRAWAT Whisper did not answer")

    monkeypatch.setattr(ts, "transcribe_single_chunk", boom)
    with pytest.raises(AirawatError):
        ts.transcribe_audio_airawat(str(audio), language="en")


def test_an_empty_answer_is_reported_not_recovered_locally(monkeypatch, strict, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    monkeypatch.setattr(model_router, "_get_dynamic_config", lambda: {"airawat": {"api_key": SECRET}})
    monkeypatch.setattr(ts, "split_audio_into_chunks", lambda p: [(str(audio), 0.0, True)])
    monkeypatch.setattr(ts, "get_audio_info", lambda p: {"duration": 5.0})
    monkeypatch.setattr(ts, "transcribe_single_chunk", lambda *a, **k: (_ for _ in ()).throw(ts.EmptyCloudTranscription("no text")))
    with pytest.raises(AirawatError) as err:
        ts.transcribe_audio_airawat(str(audio), language="en")
    assert "returned no text" in str(err.value) and "No other engine" in str(err.value)


def test_all_chunks_empty_is_an_honest_airawat_empty_reason(monkeypatch, local, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    monkeypatch.setattr(model_router, "_get_dynamic_config", lambda: {"airawat": {"api_key": SECRET}})
    monkeypatch.setattr(ts, "split_audio_into_chunks", lambda p: [(str(audio), 0.0, True)])
    monkeypatch.setattr(ts, "get_audio_info", lambda p: {"duration": 5.0})
    monkeypatch.setattr(ts, "transcribe_single_chunk", lambda *a, **k: {"text": "   "})
    out = ts.transcribe_audio_airawat(str(audio), language="en")
    assert out["text"] == "" and out["empty_reason"] == "airawat_returned_no_speech_text"


def test_transcription_without_a_key_or_language_fails_before_touching_the_audio(monkeypatch, strict, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    monkeypatch.setattr(ts, "split_audio_into_chunks", lambda p: (_ for _ in ()).throw(AssertionError("audio must not be split")))
    monkeypatch.setattr(model_router, "_get_dynamic_config", lambda: {"airawat": {"api_key": ""}})
    with pytest.raises(AirawatNotConfiguredError):
        ts.transcribe_audio_airawat(str(audio), language="en")
    monkeypatch.setattr(model_router, "_get_dynamic_config", lambda: {"airawat": {"api_key": SECRET}})
    with pytest.raises(LanguageIdentificationError):
        ts.transcribe_audio_airawat(str(audio), language=None)


def test_a_configured_default_language_is_used_when_the_upload_named_none(whisper, chunk_file, monkeypatch):
    whisper.script.append(_response(200, {"text": "namaste", "language_code": "hi"}))
    monkeypatch.setattr(model_router, "_get_dynamic_config",
                        lambda: {"airawat": {"api_key": SECRET, "retry_count": 1, "language": "hi"}})
    ts.transcribe_single_chunk(chunk_file, 0, 1, language=None)
    assert whisper.calls[0]["data"]["language"] == "hi"


# ── report generation: the LLM's answer is final ───────────────────────────────────────────────

TX = "\n".join([
    "Speaker_1: Today we only review the quarterly server maintenance and nothing needs a decision.",
    "Speaker_2: The database cluster was upgraded last weekend without downtime and response time improved slightly.",
    "Speaker_1: There were no security incidents this quarter and disk usage is at sixty percent.",
])
MOM = "### MEETING INFORMATION\n- **Participants:** Speaker_1, Speaker_2\n\n### EXECUTIVE SUMMARY\nA short status review of server maintenance was held.\n"
GOOD_SUMMARY = SummaryResult(text="The team reviewed quarterly server maintenance: the database cluster was upgraded without downtime.",
                             method="llm", llm_calls=1, plan=plan_summary_length(40))
NO_METADATA = {
    "action_items": "No action items extracted.", "decisions": "No decisions extracted.",
    "risks": "No major risks or blockers identified during this session.", "project_name": "X", "sentiment": "Neutral",
    "emotions": "Neutral: 100%", "questions_raised": "[]", "keyword_frequency": "{}", "agenda_key_decisions": [],
}


@pytest.fixture()
def workflow(monkeypatch):
    """process_meeting_report with fake LLM workers and an extractive engine that must not be reached."""
    state = SimpleNamespace(extractive_calls=0, summary=lambda *a, **k: GOOD_SUMMARY)

    def extractive(**kwargs):
        state.extractive_calls += 1
        return {"summary": "EXTRACTIVE SUMMARY", "mom": "EXTRACTIVE MOM", "action_items": "- [ ] invented", "decisions": "invented decision",
                "risks": "r", "agenda_key_decisions": [{"agenda": "invented", "key_decision": "invented decision"}]}

    monkeypatch.setattr(wf, "generate_extractive_meeting_report", extractive)
    monkeypatch.setattr(wf, "generate_meeting_summary", lambda *a, **k: state.summary(*a, **k))
    monkeypatch.setattr(wf.model_router, "validate_airawat_config", lambda: True)
    monkeypatch.setattr(wf.model_router, "get_llm", lambda **kw: object())
    monkeypatch.setattr(wf, "get_llm", lambda *a, **k: object())
    monkeypatch.setattr(wf, "_generate_section", lambda llm, prompt, transcript, max_chars=None: MOM)
    monkeypatch.setattr(wf, "_extract_metadata_json", lambda llm, t: dict(NO_METADATA))
    monkeypatch.setattr(wf, "_extract_agenda_key_decisions", lambda llm, t, user_agenda="": [])
    monkeypatch.setattr(wf, "_extract_action_items", lambda llm, t, reference_dt=None: [])
    return state


def test_no_decisions_and_no_action_items_stay_empty_outside_development_mode(workflow, strict):
    """The LLM said "none": the keyword engine must not be asked to find some anyway."""
    out = wf.process_meeting_report(TX, meeting_date="2026-09-01", meeting_time="10:00", duration_str="00:05:00")
    assert workflow.extractive_calls == 0
    assert out["agenda_key_decisions"] == [] and out["structured_action_items"] == []
    assert out["decisions"] == "No decisions extracted." and out["action_items"] == "No action items extracted."
    assert "invented" not in json.dumps(out)


def test_local_mode_never_fills_gaps_with_the_extractive_engine_either(workflow, local):
    out = wf.process_meeting_report(TX, meeting_date="2026-09-01", meeting_time="10:00", duration_str="00:05:00")
    assert workflow.extractive_calls == 0 and "invented" not in json.dumps(out)


@pytest.mark.parametrize("broken", ["summary", "mom", "metadata", "agenda", "actions"])
def test_a_failing_llm_section_raises_instead_of_using_the_extractive_engine(workflow, strict, monkeypatch, broken):
    def boom(*a, **k):
        raise TimeoutError("AIRAWAT did not answer")

    if broken == "summary":
        workflow.summary = boom
    elif broken == "mom":
        monkeypatch.setattr(wf, "_generate_section", boom)
    elif broken == "metadata":
        monkeypatch.setattr(wf, "_extract_metadata_json", boom)
    elif broken == "agenda":
        monkeypatch.setattr(wf, "_extract_agenda_key_decisions", boom)
    else:
        monkeypatch.setattr(wf, "_extract_action_items", boom)
    with pytest.raises(AirawatError) as err:
        wf.process_meeting_report(TX, meeting_date="2026-09-01", meeting_time="10:00", duration_str="00:05:00")
    assert workflow.extractive_calls == 0
    assert "failed to generate" in str(err.value) and "No other engine" in str(err.value)


def test_a_report_retry_only_regenerates_the_sections_that_failed(workflow, strict, monkeypatch):
    calls = {"mom": 0, "agenda": 0}
    agenda_answers = [TimeoutError("AIRAWAT did not answer"), []]

    def mom(*a, **k):
        calls["mom"] += 1
        return MOM

    def agenda(*a, **k):
        calls["agenda"] += 1
        answer = agenda_answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(wf, "_generate_section", mom)
    monkeypatch.setattr(wf, "_extract_agenda_key_decisions", agenda)
    cache = {}
    with pytest.raises(AirawatError):
        wf.process_meeting_report(TX, meeting_date="2026-09-01", section_cache=cache)
    out = wf.process_meeting_report(TX, meeting_date="2026-09-01", section_cache=cache)
    assert calls == {"mom": 1, "agenda": 2}          # the MoM that already succeeded is not asked for again
    assert out["mom"] and out["_section_timings"]["mom"] == 0.0


def test_a_summary_that_degraded_to_extractive_inside_the_summary_service_is_a_failure(workflow, strict):
    workflow.summary = lambda *a, **k: SummaryResult(text="rule based", method="extractive", llm_calls=0, plan=plan_summary_length(40))
    with pytest.raises(AirawatError):
        wf.process_meeting_report(TX, meeting_date="2026-09-01", meeting_time="10:00", duration_str="00:05:00")


def test_an_unconfigured_llm_raises_a_clear_error(workflow, strict, monkeypatch):
    monkeypatch.setattr(wf, "get_llm", lambda *a, **k: (_ for _ in ()).throw(AirawatNotConfiguredError("no key")))
    with pytest.raises(AirawatError):
        wf.process_meeting_report(TX, meeting_date="2026-09-01", meeting_time="10:00", duration_str="00:05:00")
    assert workflow.extractive_calls == 0


# ── pipeline: retries, then a clear failure ────────────────────────────────────────────────────


@pytest.fixture()
def retry_env(monkeypatch):
    monkeypatch.setenv("LLM_PIPELINE_MAX_RETRIES", "2")
    monkeypatch.setenv("LLM_PIPELINE_RETRY_DELAY_SEC", "0")
    monkeypatch.setattr(pipelines.time, "sleep", lambda s: None)


def test_reports_are_retried_then_the_meeting_fails_outside_development_mode(retry_env, strict, monkeypatch):
    calls = []
    monkeypatch.setattr(pipelines, "process_meeting_report", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(TimeoutError("slow")))
    with pytest.raises(AirawatError) as err:
        pipelines._generate_reports_with_retry(7, "some transcript", meeting_date="d", meeting_time="t", duration_str="")
    assert len(calls) == 2 and "after 2 attempt(s)" in str(err.value)


def test_a_report_that_succeeds_on_the_second_attempt_is_used(retry_env, strict, monkeypatch):
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("first attempt stalled")
        return {"summary": "ok"}

    monkeypatch.setattr(pipelines, "process_meeting_report", flaky)
    assert pipelines._generate_reports_with_retry(7, "t") == {"summary": "ok"} and len(calls) == 2


def test_local_mode_also_fails_the_meeting_when_reports_fail(retry_env, local, monkeypatch):
    monkeypatch.setattr(pipelines, "process_meeting_report", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("slow")))
    with pytest.raises(AirawatError):
        pipelines._generate_reports_with_retry(7, "t")


# ── translation: outputs must be English, so a failed translation is a failure ─────────────────

HINDI = [{"speaker": "Speaker_1", "start_time": "0.0", "end_time": "3.0", "text": "नमस्ते दुनिया यह एक बैठक है"}]


def test_a_failed_translation_raises_outside_development_mode(strict, monkeypatch):
    monkeypatch.setattr(settings, "TRANSLATION_MAX_RETRIES", 0)
    monkeypatch.setattr(tr, "_get_llm", lambda temperature=0.0: (_ for _ in ()).throw(ConnectionError("AIRAWAT unreachable")))
    with pytest.raises(AirawatError) as err:
        tr.translate_segments_to_english([dict(s) for s in HINDI], "hi")
    assert "could not translate 1/1" in str(err.value) and "ConnectionError" in str(err.value)


def test_local_mode_also_reports_a_failed_translation(local, monkeypatch):
    monkeypatch.setattr(settings, "TRANSLATION_MAX_RETRIES", 0)
    monkeypatch.setattr(tr, "_get_llm", lambda temperature=0.0: (_ for _ in ()).throw(ConnectionError("down")))
    with pytest.raises(AirawatError):
        tr.translate_segments_to_english([dict(s) for s in HINDI], "hi")


def test_translation_that_works_is_unaffected(strict, monkeypatch):
    class LLM:
        def invoke(self, messages):
            return SimpleNamespace(content=json.dumps({"translations": [{"i": 0, "en": "Hello world, this is a meeting"}]}))

    monkeypatch.setattr(tr, "_get_llm", lambda temperature=0.0: LLM())
    out, meta = tr.translate_segments_to_english([dict(s) for s in HINDI], "hi")
    assert out[0]["text"] == "Hello world, this is a meeting" and out[0]["original_text"] == HINDI[0]["text"] and meta["translated"]


# ── reads must not invent data ─────────────────────────────────────────────────────────────────


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return list(self.rows)

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self


class _DB:
    """A meeting WITH a real transcript but no stored agenda / decisions / action items (the pipeline found none)."""

    def __init__(self):
        self.added, self.commits = [], 0
        self.segments = [SimpleNamespace(speaker="Speaker_1", text="We will review the server maintenance report next week."),
                         SimpleNamespace(speaker="Speaker_2", text="I will send the disk usage numbers by Friday.")]

    def query(self, model, *a, **k):
        return _Query(self.segments if getattr(model, "__name__", "") == "TranscriptSegment" else [])

    def add(self, row):
        self.added.append(row)

    def commit(self):
        self.commits += 1


def test_a_read_never_invents_decisions_or_action_items_outside_development_mode(strict, monkeypatch):
    """The transcript below contains 'I will ...' sentences a keyword engine would turn into action items."""
    monkeypatch.setattr(wf, "generate_extractive_meeting_report", lambda **k: (_ for _ in ()).throw(AssertionError("extractive used on a read")))
    monkeypatch.setattr("backend.services.action_item_service.extract_action_items_extractive",
                        lambda text: (_ for _ in ()).throw(AssertionError("extractive used on a read")))
    meeting = SimpleNamespace(id=1, agenda="", bharatvc_meeting_id="1", meeting_date="2026-09-01", analysis=None)
    db = _DB()
    assert meetings_api._ensure_meeting_agenda_decisions(meeting, db) == []
    assert meetings_api._ensure_meeting_action_items(meeting, db) == []
    assert db.added == [] and db.commits == 0                 # and nothing was written


# ── the sanitized AIRAWAT validation ───────────────────────────────────────────────────────────


@pytest.fixture()
def diag(monkeypatch, strict):
    calls = []
    cfg = {"api_key": SECRET, "base_url": "https://airawat.example/llm/v1", "llm_model": "meta/llama-3.3-70b-instruct",
           "llm_endpoint": "/chat/completions", "whisper_url": "https://airawat.example/whisper", "language": ""}
    monkeypatch.setattr(ah, "_resolve_config", lambda: dict(cfg))

    def post(url, headers=None, json=None, files=None, data=None, **kw):
        calls.append({"url": url, "json": json, "data": data, "headers": headers})
        if url.endswith("/chat/completions"):
            return _response(200, {"model": "meta/llama-3.3-70b-instruct", "choices": [{"message": {"content": "pong"}}]})
        return _response(200, {"text": "", "language_code": "en"})

    monkeypatch.setattr(ah.requests, "post", post)
    return SimpleNamespace(calls=calls, cfg=cfg, monkeypatch=monkeypatch)


def test_diagnostics_pass_without_ever_exposing_the_key(diag):
    report = ah.run_diagnostics(deep=True)
    assert report["ok"] is True and report["llm"]["ok"] and report["whisper"]["ok"]
    assert report["key_fingerprint"] == ah.key_fingerprint(SECRET) and len(report["key_fingerprint"]) == 8
    assert SECRET not in json.dumps(report)


def test_the_whisper_check_sends_exactly_what_the_pipeline_sends(diag):
    ah.run_diagnostics(deep=True)
    whisper_call = next(c for c in diag.calls if c["url"].endswith("/whisper"))
    assert whisper_call["data"] == {"language": "en", "language_code": "en"}          # a language, never a model
    llm_call = next(c for c in diag.calls if c["url"].endswith("/chat/completions"))
    assert llm_call["json"]["model"] == "meta/llama-3.3-70b-instruct"


def test_a_rejected_key_is_reported_with_the_reason_and_still_no_secret(diag):
    diag.monkeypatch.setattr(ah.requests, "post", lambda *a, **k: _response(401, {"message": f"Bad token; invalid JSON {SECRET}"}))
    report = ah.run_diagnostics(deep=True)
    assert report["ok"] is False and "HTTP 401" in report["error"]
    assert SECRET not in json.dumps(report)


def test_a_timeout_is_reported_as_a_timeout(diag):
    def slow(*a, **k):
        raise requests.exceptions.ReadTimeout("read timed out")

    diag.monkeypatch.setattr(ah.requests, "post", slow)
    report = ah.run_diagnostics(deep=False, timeout=3)
    assert report["ok"] is False and "timeout" in report["llm"]["error"]


def test_a_missing_key_is_reported_without_any_network_call(monkeypatch, strict):
    monkeypatch.setattr(ah, "_resolve_config", lambda: {"api_key": "", "base_url": "", "llm_model": "m", "llm_endpoint": "/c",
                                                        "whisper_url": "", "language": ""})
    monkeypatch.setattr(ah.requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no request without a key")))
    report = ah.run_diagnostics()
    assert report["configured"] is False and report["ok"] is False and "no API key" in report["error"]
    assert "local_engines_allowed" not in report


def test_an_unexpected_answer_format_fails_the_check(diag):
    diag.monkeypatch.setattr(ah.requests, "post", lambda *a, **k: _response(200, {"unexpected": True}))
    report = ah.run_diagnostics(deep=True)
    assert report["ok"] is False and report["llm"]["response_format_ok"] is False


# ── the real workers must not swallow an AIRAWAT failure ───────────────────────────────────────
# (the stubs above bypass the workers' own try/except; these use the real functions)

from langchain_core.runnables import RunnableLambda  # noqa: E402


def _llm_that_raises(calls=None):
    def boom(_messages):
        if calls is not None:
            calls.append(1)
        raise ConnectionError("AIRAWAT did not answer")

    return RunnableLambda(boom)


def _llm_answering(text, calls=None):
    def answer(_messages):
        if calls is not None:
            calls.append(1)
        return SimpleNamespace(content=text)

    return RunnableLambda(answer)


WORKERS = [
    ("mom", lambda llm: wf._generate_section(llm, "Write the minutes.", TX)),
    ("agenda", lambda llm: wf._extract_agenda_key_decisions(llm, TX)),
    ("metadata", lambda llm: wf._extract_metadata_json(llm, TX)),
    ("actions", lambda llm: wf._extract_action_items(llm, TX)),
]


@pytest.mark.parametrize("name,call", WORKERS, ids=[w[0] for w in WORKERS])
def test_a_real_worker_surfaces_an_airawat_failure_outside_development_mode(strict, name, call):
    with pytest.raises(AirawatError) as err:
        call(_llm_that_raises())
    assert "failed while generating" in str(err.value) and "ConnectionError" in str(err.value)


@pytest.mark.parametrize("name,call", WORKERS, ids=[w[0] for w in WORKERS])
def test_a_real_worker_surfaces_an_airawat_failure_in_local_mode_too(local, name, call):
    with pytest.raises(AirawatError):
        call(_llm_that_raises())


def test_a_model_that_finds_no_agenda_is_not_pushed_to_invent_one_outside_development_mode(strict):
    calls = []
    out = wf._extract_agenda_key_decisions(_llm_answering('{"agenda_key_decisions": []}', calls), TX)
    assert out == [] and len(calls) == 1                       # no second, loosened "higher recall" prompt


def test_the_agenda_extraction_retries_a_failure_once_with_the_same_prompt(strict):
    calls = []
    with pytest.raises(AirawatError):
        wf._extract_agenda_key_decisions(_llm_that_raises(calls), TX)
    assert len(calls) == 2


def test_local_mode_has_no_loose_retry_prompt_either(local):
    calls = []
    assert wf._extract_agenda_key_decisions(_llm_answering('{"agenda_key_decisions": []}', calls), TX) == []
    assert len(calls) == 1


# ── spoken language: from the upload, else the configured default - never guessed ──────────────────

from backend.services import language_service as ls  # noqa: E402


def test_the_upload_language_wins_over_the_configured_default():
    assert ls.resolve_spoken_language("TE", "en") == ("te", "upload")
    assert ls.resolve_spoken_language(None, "hi") == ("hi", "configured")
    assert ls.resolve_spoken_language("auto", "") == (None, "none")          # AIRAWAT answers 404 to "auto"


def test_a_bad_language_code_is_rejected():
    with pytest.raises(ValueError):
        ls.normalize_language_code("telugu please")
    assert ls.normalize_language_code(" En-IN ") == "en-in" and ls.normalize_language_code("") is None


def test_the_env_default_language_reaches_the_airawat_config(monkeypatch):
    from backend.services import model_router as mr_module

    # the router's own settings object (other tests may reload the config module)
    monkeypatch.setattr(mr_module.settings, "AIRAWAT_WHISPER_LANGUAGE", "ml")
    monkeypatch.setattr(mr_module, "SessionLocal", lambda: (_ for _ in ()).throw(RuntimeError("no db")))
    router = mr_module.ModelRouter()
    router.clear_cache()                       # the router caches its config (and may be a shared instance)
    try:
        assert router._get_dynamic_config()["airawat"]["language"] == "ml"
    finally:
        router.clear_cache()


def test_health_reports_airawat_configured_and_both_endpoints(monkeypatch):
    monkeypatch.setattr(ah, "LAST_STARTUP_REPORT", {"ok": False, "configured": True, "llm": {"ok": True},
                                                     "whisper": {"ok": False}, "default_language": "en",
                                                     "error": "whisper: HTTP 401"})
    st = ah.startup_status()
    assert st["configured"] and st["llm_reachable"] and not st["whisper_reachable"] and st["state"] == "failed"
    monkeypatch.setattr(ah, "LAST_STARTUP_REPORT", None)
    assert ah.startup_status() == {"state": "checking"}

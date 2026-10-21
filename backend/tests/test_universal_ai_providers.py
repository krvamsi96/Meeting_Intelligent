"""Ollama / Groq / Gemini providers, provider selection and the administrator settings API.

Policies pinned here (each was violated by the first version of the provider layer):
* only the CONFIGURED provider and model are ever called - no silent switch to AIRAWAT, no hidden fallback models;
* model lists are the provider's live list - never a hardcoded list presented as "available";
* a provider without a tested speech-to-text integration does not claim one (Gemini invented 0.4 s word timings);
* API keys never travel in URLs and are never returned unmasked.
"""

import json
from unittest.mock import MagicMock

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import settings as settings_api
from backend.api.deps import get_db
from backend.config.config import settings
from backend.providers import (
    GeminiProvider,
    GroqProvider,
    OllamaProvider,
    ProviderCapabilityError,
    ProviderConfigurationError,
    ProviderConnectionError,
    provider_factory,
)
from backend.services import transcription_service as ts
from backend.services.model_router import model_router
from backend.utils.exceptions import AIProviderError, AIProviderNotConfiguredError


def _json_response(status, body, headers=None):
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(body).encode()
    r.headers["content-type"] = "application/json"
    r.headers.update(headers or {})
    r.url = "https://provider.example/x"
    return r


def _ollama(**kw):
    return OllamaProvider(**{"base_url": "http://localhost:11434", "model": "llama3.2", **kw})


def _groq(**kw):
    return GroqProvider(**{"api_key": "gsk_test_key_12345", "model": "openai/gpt-oss-120b",
                           "whisper_model": "whisper-large-v3", "base_url": "https://api.groq.com/openai/v1", **kw})


def _gemini(**kw):
    return GeminiProvider(**{"api_key": "AIza-test-key-123", "model": "gemini-2.5-flash", **kw})


# ─── Ollama ────────────────────────────────────────────────────────────────────


def test_ollama_reports_a_configured_model_that_is_not_installed(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _json_response(200, {"models": [{"name": "qwen3:8b"}]}))
    res = _ollama(model="llama3:8b").test_connection()
    assert res["ok"] is False and res["model_available"] is False
    assert "not installed" in res["message"] and "ollama pull llama3:8b" in res["message"]


def test_ollama_accepts_an_untagged_model_name_as_latest(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _json_response(200, {"models": [{"name": "llama3.2:latest"}]}))
    res = _ollama(model="llama3.2").test_connection()
    assert res["ok"] is True and res["model_available"] is True


def test_ollama_unreachable_is_reported_not_raised(monkeypatch):
    def refuse(*a, **k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", refuse)
    res = _ollama().test_connection()
    assert res["ok"] is False and res["models"] == [] and "ollama serve" in res["message"]
    with pytest.raises(ProviderConnectionError):
        _ollama().get_available_models()


def test_ollama_does_not_claim_speech_to_text():
    p = _ollama()
    assert p.supports_transcription is False and p.is_cloud is False
    with pytest.raises(ProviderCapabilityError):
        p.transcribe_chunk("x.wav", 0, 1)


# ─── Groq ──────────────────────────────────────────────────────────────────────


def test_groq_without_a_key_is_not_configured_and_never_reads_the_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_from_the_environment")
    p = _groq(api_key="")
    assert p.is_configured is False
    assert p.test_connection()["ok"] is False
    with pytest.raises(ProviderConfigurationError):
        p.get_llm()


def test_groq_calls_only_the_configured_model_with_no_hidden_fallbacks():
    llm = _groq(model="qwen/qwen3.8-27b").get_llm(max_tokens=123)
    assert type(llm).__name__ == "ChatOpenAI"          # not a RunnableWithFallbacks over other model ids
    assert llm.model_name == "qwen/qwen3.8-27b" and llm.max_tokens == 123


def test_groq_flags_a_retired_model_from_the_live_list(monkeypatch):
    live = {"data": [{"id": "openai/gpt-oss-120b"}, {"id": "whisper-large-v3"}]}
    monkeypatch.setattr(requests, "get", lambda *a, **k: _json_response(200, live))
    res = _groq(model="llama-3.3-70b-versatile").test_connection()
    assert res["ok"] is False and res["model_available"] is False
    assert res["transcription_model_available"] is True
    assert "llama-3.3-70b-versatile" in res["message"]


def test_groq_model_list_failure_raises_instead_of_inventing_a_list(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _json_response(503, {"error": "down"}))
    with pytest.raises(ProviderConnectionError):
        _groq().get_available_models()
    monkeypatch.setattr(requests, "get", lambda *a, **k: _json_response(401, {"error": "bad key"}))
    with pytest.raises(ProviderConfigurationError):
        _groq().get_available_models()


@pytest.fixture()
def no_backoff(monkeypatch):
    monkeypatch.setattr(ts.time, "sleep", lambda s: None)


def _fake_whisper(monkeypatch, answers):
    """requests.post replacement: pops one answer per call (a Response, or an exception to raise)."""
    calls = []

    def fake_post(url, headers=None, data=None, verify=None, timeout=None):
        calls.append({"url": url, "headers": headers, "body": data.read() if data is not None else b""})
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(ts.requests, "post", fake_post)
    return calls


def test_groq_chunk_request_asks_for_word_timestamps_and_lets_groq_detect_the_language(monkeypatch, tmp_path, no_backoff):
    chunk = tmp_path / "c.wav"
    chunk.write_bytes(b"RIFF....WAVEfmt ")
    calls = _fake_whisper(monkeypatch, [_json_response(200, {"text": "hello", "language": "english"})])
    out = _groq().transcribe_chunk(str(chunk), 0, 1, language=None)
    assert out["text"] == "hello"
    body = calls[0]["body"].decode("latin-1")
    assert calls[0]["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert 'name="model"\r\n\r\nwhisper-large-v3' in body and "verbose_json" in body
    assert body.count('name="timestamp_granularities[]"') == 2
    assert 'name="language"' not in body
    assert "gsk_test_key_12345" not in body and calls[0]["headers"]["Authorization"] == "Bearer gsk_test_key_12345"


def test_groq_chunk_is_retried_on_a_server_error_then_succeeds(monkeypatch, tmp_path, no_backoff):
    chunk = tmp_path / "c.wav"
    chunk.write_bytes(b"RIFF")
    calls = _fake_whisper(monkeypatch, [_json_response(500, {"error": "overloaded"}), _json_response(200, {"text": "ok"})])
    assert _groq(retry_count=2).transcribe_chunk(str(chunk), 0, 1, language="en")["text"] == "ok"
    assert len(calls) == 2


def test_groq_chunk_permanent_error_fails_at_once_without_leaking_the_key(monkeypatch, tmp_path, no_backoff):
    chunk = tmp_path / "c.wav"
    chunk.write_bytes(b"RIFF")
    calls = _fake_whisper(monkeypatch, [_json_response(401, {"error": {"message": "Invalid API Key"}})])
    with pytest.raises(AIProviderError) as err:
        _groq().transcribe_chunk(str(chunk), 0, 1)
    assert len(calls) == 1 and "gsk_test_key_12345" not in str(err.value)


def test_groq_multi_chunk_transcript_keeps_absolute_timestamps(monkeypatch, tmp_path, no_backoff):
    """Chunk 2 starts CHUNK_OVERLAP_SEC early; its word times must be shifted back to recording time."""
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"RIFF")
    overlap = ts.CHUNK_OVERLAP_SEC
    chunks = [(str(audio), 0.0, True), (str(tmp_path / "chunk1.wav"), 60.0, False)]
    (tmp_path / "chunk1.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(ts, "split_audio_into_chunks", lambda path: chunks)
    monkeypatch.setattr(ts, "get_audio_info", lambda path: {"duration": 60.0, "size_bytes": 4})
    answers = {
        0: {"text": "first part", "language": "english",
            "words": [{"word": "first", "start": 1.0, "end": 1.4}, {"word": "part", "start": 1.5, "end": 1.9}]},
        1: {"text": "second part",
            "words": [{"word": "second", "start": overlap + 2.0, "end": overlap + 2.4},
                      {"word": "part", "start": overlap + 2.5, "end": overlap + 2.9}]},
    }
    out = ts.transcribe_audio_chunked(
        str(audio), language=None, chunk_transcriber=lambda f, i, n, **k: answers[i],
        provider_label="groq", require_language=False,
    )
    assert out["text"] == "first part second part"
    assert [w["start"] for w in out["words"]] == [1.0, 1.5, 62.0, 62.5]
    assert out["language"] == "english"


# ─── Gemini ────────────────────────────────────────────────────────────────────


def test_gemini_sends_the_key_in_a_header_never_in_the_url(monkeypatch):
    seen = {}

    def fake_get(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers)
        return _json_response(200, {"models": [{"name": "models/gemini-2.5-flash"}]})

    monkeypatch.setattr(requests, "get", fake_get)
    res = _gemini().test_connection()
    assert res["ok"] is True and res["model_available"] is True
    assert "AIza-test-key-123" not in seen["url"] and seen["headers"]["x-goog-api-key"] == "AIza-test-key-123"


def test_gemini_reports_a_retired_model(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _json_response(200, {"models": [{"name": "models/gemini-2.5-flash"}]}))
    res = _gemini(model="gemini-1.5-flash").test_connection()
    assert res["ok"] is False and res["model_available"] is False and "retired" in res["message"]


def test_gemini_does_not_claim_speech_to_text():
    assert _gemini().supports_transcription is False
    with pytest.raises(ProviderCapabilityError):
        _gemini().transcribe_chunk("x.wav", 0, 1)


# ─── provider selection ────────────────────────────────────────────────────────


def test_an_unknown_ai_provider_is_an_error_not_a_silent_default(monkeypatch):
    monkeypatch.setattr(settings, "AI_PROVIDER", "openia")
    provider_factory.clear_cache()
    with pytest.raises(ProviderConfigurationError):
        provider_factory.llm_provider_name()


@pytest.mark.parametrize("llm, stt, expected", [
    ("groq", "", "groq"),
    ("airawat", "", "airawat"),
    ("ollama", "groq", "groq"),
    ("gemini", "airawat", "airawat"),
])
def test_speech_to_text_provider_resolution(monkeypatch, llm, stt, expected):
    monkeypatch.setattr(settings, "AI_PROVIDER", llm)
    monkeypatch.setattr(settings, "STT_PROVIDER", stt)
    provider_factory.clear_cache()
    assert provider_factory.stt_provider_name() == expected


@pytest.mark.parametrize("llm", ["ollama", "gemini"])
def test_a_text_only_provider_without_stt_provider_fails_with_a_configuration_error(monkeypatch, llm):
    monkeypatch.setattr(settings, "AI_PROVIDER", llm)
    provider_factory.clear_cache()
    with pytest.raises(ProviderCapabilityError, match="STT_PROVIDER"):
        provider_factory.stt_provider_name()
    with pytest.raises(AIProviderNotConfiguredError):
        model_router.get_transcriber()


def test_a_failing_provider_is_never_replaced_by_airawat(monkeypatch):
    monkeypatch.setattr(settings, "AI_PROVIDER", "groq")
    monkeypatch.setattr(settings, "AIRAWAT_API_KEY", "sk-airawat-key-that-must-not-be-used")
    provider_factory.clear_cache()
    monkeypatch.setattr(model_router, "_get_airawat_llm", lambda **k: pytest.fail("fell back to AIRAWAT"))
    with pytest.raises(ProviderConfigurationError):
        model_router.get_llm()          # Groq has no key in the test settings


def test_groq_transcriber_requires_a_key(monkeypatch):
    monkeypatch.setattr(settings, "AI_PROVIDER", "groq")
    provider_factory.clear_cache()
    with pytest.raises(AIProviderNotConfiguredError):
        model_router.get_transcriber()


# ─── administrator settings API ────────────────────────────────────────────────


@pytest.fixture()
def admin_client(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_API_KEY", "admin-secret")
    app = FastAPI()
    app.include_router(settings_api.router, prefix="/api/v1")
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = None
    session.query.return_value.all.return_value = []
    app.dependency_overrides[get_db] = lambda: session
    monkeypatch.setattr(model_router, "get_effective_config", lambda: {})
    return TestClient(app), session, {"X-Admin-Key": "admin-secret"}


def test_reading_ai_settings_needs_the_admin_key_and_masks_keys(admin_client, monkeypatch):
    client, _db, auth = admin_client
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk_real_secret_value_9876")
    provider_factory.clear_cache()
    assert client.get("/api/v1/settings/ai").status_code == 401
    data = client.get("/api/v1/settings/ai", headers=auth).json()
    assert data["active_provider"] == "airawat" and {"ollama", "groq", "gemini", "airawat"} <= set(data)
    assert "gsk_real_secret_value_9876" not in json.dumps(data) and data["groq"]["api_key"].endswith("9876")


@pytest.mark.parametrize("payload", [
    {"groq": {"timeout": "abc"}},
    {"groq": {"retry_count": 99}},
    {"ollama": {"temperature": 5}},
    {"gemini": {"base_url": "file:///etc/passwd"}},
    {"groq": {"modle": "typo"}},
    {"active_provider": "openia"},
    {"stt_provider": "gemini"},
])
def test_saving_invalid_ai_settings_is_rejected(admin_client, payload):
    client, db, auth = admin_client
    r = client.post("/api/v1/settings/ai", json=payload, headers=auth)
    assert r.status_code == 400, r.text
    db.commit.assert_not_called()


def test_the_stored_key_never_follows_a_new_host(admin_client):
    client, db, auth = admin_client
    r = client.post("/api/v1/settings/ai", json={"groq": {"base_url": "https://attacker.example/v1"}}, headers=auth)
    assert r.status_code == 400 and "Re-enter the API key" in r.json()["detail"]
    r = client.post("/api/v1/settings/test", headers=auth,
                    json={"provider": "groq", "config": {"base_url": "https://attacker.example/v1"}})
    assert r.status_code == 400


def test_valid_ai_settings_are_saved_encrypted(admin_client):
    client, db, auth = admin_client
    r = client.post("/api/v1/settings/ai", headers=auth,
                    json={"active_provider": "groq", "groq": {"api_key": "gsk_new_key_123456", "timeout": 30}})
    assert r.status_code == 200, r.text
    stored = [c.args[0] for c in db.add.call_args_list]
    assert {row.provider for row in stored} == {"groq", "active_provider"}
    assert all("gsk_new_key_123456" not in row.encrypted_config for row in stored)
    db.commit.assert_called_once()


def test_model_list_endpoint_reports_failure_instead_of_a_fake_list(admin_client, monkeypatch):
    client, _db, auth = admin_client
    monkeypatch.setattr(OllamaProvider, "get_available_models",
                        lambda self, timeout=10.0: (_ for _ in ()).throw(ProviderConnectionError("down")))
    assert client.get("/api/v1/settings/models?provider=ollama", headers=auth).status_code == 502
    monkeypatch.setattr(OllamaProvider, "get_available_models", lambda self, timeout=10.0: ["qwen3:8b"])
    assert client.get("/api/v1/settings/models?provider=ollama", headers=auth).json()["models"] == ["qwen3:8b"]


def test_connection_test_uses_unsaved_form_values(admin_client, monkeypatch):
    client, _db, auth = admin_client
    seen = {}

    def fake_test(self, timeout=10.0):
        seen.update(model=self.model, base_url=self.base_url)
        return {"ok": True, "latency_ms": 12.0, "message": "Connected", "models": ["qwen3:8b"], "model_available": True}

    monkeypatch.setattr(OllamaProvider, "test_connection", fake_test)
    r = client.post("/api/v1/settings/test", headers=auth, json={"provider": "ollama", "config": {"model": "qwen3:8b"}})
    assert r.status_code == 200 and r.json()["connected"] is True and seen["model"] == "qwen3:8b"

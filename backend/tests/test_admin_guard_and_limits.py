"""Admin guard, settings-endpoint credential safety, and rate limiting (no DB needed)."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from backend.config.config import settings
from backend.utils import auth, rate_limit
from backend.utils.auth import require_admin


@pytest.fixture()
def guarded_client():
    app = FastAPI()

    @app.post("/admin-thing", dependencies=[Depends(require_admin)])
    def admin_thing():
        return {"ok": True}

    return TestClient(app)


def test_admin_guard_is_open_when_no_key_is_configured(guarded_client, monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_API_KEY", "")
    monkeypatch.setattr(auth, "_warned_open", False)
    assert guarded_client.post("/admin-thing").status_code == 200


def test_admin_guard_rejects_missing_and_wrong_keys(guarded_client, monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_API_KEY", "s3cret-admin-key")
    r = guarded_client.post("/admin-thing")
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert guarded_client.post("/admin-thing", headers={"X-Admin-Key": "nope"}).status_code == 401
    assert guarded_client.post("/admin-thing", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_admin_guard_accepts_both_header_styles(guarded_client, monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_API_KEY", "s3cret-admin-key")
    assert guarded_client.post("/admin-thing", headers={"X-Admin-Key": "s3cret-admin-key"}).status_code == 200
    assert guarded_client.post("/admin-thing", headers={"Authorization": "Bearer s3cret-admin-key"}).status_code == 200


# ── settings API: the stored API key must never follow an attacker-chosen host ────────


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _FakeDB:
    def __init__(self):
        self.added, self.committed = [], False

    def query(self, *a, **k):
        return _FakeQuery([])

    def add(self, row):
        self.added.append(row)

    def commit(self):
        self.committed = True

    def rollback(self):
        pass


@pytest.fixture()
def settings_client(monkeypatch):
    from backend.api import settings as settings_api
    from backend.api.deps import get_db

    monkeypatch.setattr(settings, "ADMIN_API_KEY", "")
    monkeypatch.setattr(settings, "AIRAWAT_BASE_URL", "https://apis.airawat.example/v1")
    monkeypatch.setattr(settings, "AIRAWAT_WHISPER_URL", "https://apis.airawat.example/whisper")
    monkeypatch.setattr(settings, "AIRAWAT_API_KEY", "server-held-key")
    monkeypatch.setattr(
        settings_api.model_router,
        "get_effective_config",
        lambda: {"api_key": "server-held-key", "base_url": "https://apis.airawat.example/v1"},
    )
    db = _FakeDB()
    app = FastAPI()
    app.include_router(settings_api.router, prefix="/api/v1")
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app), db, settings_api


def test_test_endpoint_refuses_to_send_the_stored_key_to_a_new_host(settings_client, monkeypatch):
    client, _db, settings_api = settings_client
    sent = []
    monkeypatch.setattr(settings_api.requests, "post", lambda *a, **k: sent.append((a, k)))

    r = client.post("/api/v1/settings/test", json={
        "provider": "airawat",
        "config": {"api_key": "", "llama_endpoint": "https://attacker.example/steal"},
    })
    assert r.status_code == 400
    assert sent == [], "the server-held key must never reach an untrusted host"


def test_test_endpoint_allows_probing_a_new_host_with_an_explicitly_supplied_key(settings_client, monkeypatch):
    client, _db, settings_api = settings_client
    seen = {}

    def fake_post(url, headers=None, **kw):
        seen.update(url=url, auth=headers["Authorization"], redirects=kw.get("allow_redirects"))
        return SimpleNamespace(status_code=200, headers={}, text="ok")

    monkeypatch.setattr(settings_api.requests, "post", fake_post)
    r = client.post("/api/v1/settings/test", json={
        "provider": "airawat",
        "config": {"api_key": "user-typed-key", "llama_endpoint": "https://my-own-llm.example/v1/chat/completions"},
    })
    assert r.status_code == 200 and r.json()["connected"] is True
    assert seen["auth"] == "Bearer user-typed-key"
    assert seen["redirects"] is False, "a redirect would forward the Authorization header elsewhere"


def test_test_endpoint_uses_the_stored_key_for_the_configured_host(settings_client, monkeypatch):
    client, _db, settings_api = settings_client
    seen = {}
    monkeypatch.setattr(
        settings_api.requests, "post",
        lambda url, headers=None, **kw: seen.update(auth=headers["Authorization"]) or SimpleNamespace(status_code=200, headers={}, text=""),
    )
    r = client.post("/api/v1/settings/test", json={"provider": "airawat", "config": {"api_key": "sk-a...zzzz"}})
    assert r.status_code == 200
    assert seen["auth"] == "Bearer server-held-key"


def test_a_non_http_endpoint_is_only_ever_a_path_on_the_trusted_base(settings_client, monkeypatch):
    """`file:///…` is not an absolute http(s) URL, so it is appended to the trusted base URL — the
    request can only ever go to the configured host."""
    client, _db, settings_api = settings_client
    seen = {}
    monkeypatch.setattr(
        settings_api.requests, "post",
        lambda url, headers=None, **kw: seen.update(url=url) or SimpleNamespace(status_code=200, headers={}, text=""),
    )
    r = client.post("/api/v1/settings/test", json={"config": {"api_key": "sk-a...zzzz", "llama_endpoint": "file:///etc/passwd"}})
    assert r.status_code == 200
    assert seen["url"].startswith("https://apis.airawat.example/v1/")


def test_save_refuses_a_new_host_when_the_key_is_not_re_entered(settings_client):
    client, db, _ = settings_client
    r = client.post("/api/v1/settings/ai", json={"airawat": {"enabled": True, "api_key": "", "base_url": "https://attacker.example/v1"}})
    assert r.status_code == 400
    assert db.added == [] and db.committed is False


def test_save_validates_numeric_fields(settings_client):
    client, *_ = settings_client
    for field, bad in (("timeout", "abc"), ("timeout", 0), ("max_tokens", 10**9), ("temperature", 9)):
        r = client.post("/api/v1/settings/ai", json={"airawat": {"api_key": "new-key", field: bad}})
        assert r.status_code == 400, (field, bad, r.text)


def test_loop_engine_settings_parse_booleans_and_bounds(settings_client, monkeypatch):
    client, *_ = settings_client
    monkeypatch.setattr(settings, "LOOP_ENGINE_ENABLED", True)
    r = client.post("/api/v1/settings/loop-engine", json={"loop_engine_enabled": "false"})
    assert r.status_code == 200 and settings.LOOP_ENGINE_ENABLED is False, "the string 'false' must not be truthy"
    assert client.post("/api/v1/settings/loop-engine", json={"max_stage_retries": "many"}).status_code == 400
    assert client.post("/api/v1/settings/loop-engine", json={"max_stage_retries": -3}).status_code == 400


def test_admin_key_protects_every_settings_writer(settings_client, monkeypatch):
    client, *_ = settings_client
    monkeypatch.setattr(settings, "ADMIN_API_KEY", "k-admin")
    for path, body in (
        ("/api/v1/settings/ai", {"airawat": {}}),
        ("/api/v1/settings/test", {}),
        ("/api/v1/settings/report", {}),
        ("/api/v1/settings/diarization", {}),
        ("/api/v1/settings/loop-engine", {}),
    ):
        assert client.post(path, json=body).status_code == 401, path


# ── rate limiter ──────────────────────────────────────────────────────────────


def _request(host="1.2.3.4", xff=None):
    headers = {"x-forwarded-for": xff} if xff else {}
    return SimpleNamespace(client=SimpleNamespace(host=host), headers=headers)


def test_rate_limiter_blocks_after_the_limit_and_sets_retry_after():
    from fastapi import HTTPException

    limiter = rate_limit.SlidingWindowLimiter("t", lambda: 3, window_sec=60)
    for _ in range(3):
        limiter.check(_request())
    with pytest.raises(HTTPException) as exc:
        limiter.check(_request())
    assert exc.value.status_code == 429
    assert int(exc.value.headers["Retry-After"]) >= 1
    limiter.check(_request(host="9.9.9.9"))  # a different client is unaffected


def test_rate_limiter_window_slides():
    limiter = rate_limit.SlidingWindowLimiter("t", lambda: 1, window_sec=0.2)
    limiter.check(_request())
    time.sleep(0.25)
    limiter.check(_request())  # window expired -> allowed again


def test_rate_limit_of_zero_disables_it():
    limiter = rate_limit.SlidingWindowLimiter("t", lambda: 0)
    for _ in range(100):
        limiter.check(_request())


def test_forwarded_for_is_ignored_unless_trusted(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(settings, "TRUST_FORWARDED_FOR", False)
    limiter = rate_limit.SlidingWindowLimiter("t", lambda: 1)
    limiter.check(_request(host="5.5.5.5", xff="a"))
    with pytest.raises(HTTPException):
        limiter.check(_request(host="5.5.5.5", xff="b"))  # spoofing XFF must NOT dodge the limit

    monkeypatch.setattr(settings, "TRUST_FORWARDED_FOR", True)
    limiter2 = rate_limit.SlidingWindowLimiter("t2", lambda: 1)
    limiter2.check(_request(host="10.0.0.1", xff="7.7.7.7, 10.0.0.1"))
    limiter2.check(_request(host="10.0.0.1", xff="8.8.8.8"))  # distinct real clients behind the proxy


# ── SSE manager ───────────────────────────────────────────────────────────────


def test_close_meeting_from_a_worker_thread_wakes_the_subscriber_promptly():
    """Regression: put_nowait from a non-loop thread did not wake the consumer (30 s keepalive wait)."""
    from backend.services.sse_manager import SSEManager

    async def scenario():
        mgr = SSEManager()
        mgr.bind_loop(asyncio.get_running_loop())
        received = []

        async def consume():
            async for event in mgr.subscribe(1):
                received.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)  # subscriber registered

        await asyncio.get_running_loop().run_in_executor(None, lambda: mgr.publish_sync(1, {"step": "a"}))
        await asyncio.sleep(0.1)
        t = threading.Thread(target=mgr.close_meeting, args=(1,))
        t.start()
        t.join()
        await asyncio.wait_for(task, timeout=2.0)  # would hang ~30 s before the fix
        return received

    received = asyncio.run(scenario())
    assert any('"step": "a"' in e for e in received), "publish_sync from a thread must reach the subscriber"


def test_closing_a_meeting_stream_delivers_the_done_event_before_ending():
    """The frontend must learn the outcome (status / error), not only see the connection drop and reconnect."""
    from backend.services.sse_manager import SSEManager

    async def scenario():
        mgr = SSEManager()
        mgr.bind_loop(asyncio.get_running_loop())
        received = []

        async def consume():
            async for event in mgr.subscribe(7):
                received.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        done = {"event": "done", "status": "error", "error_message": "AIRAWAT did not answer"}
        await asyncio.get_running_loop().run_in_executor(None, lambda: mgr.close_meeting(7, final_event=done))
        await asyncio.wait_for(task, timeout=2.0)
        return received

    received = asyncio.run(scenario())
    assert len(received) == 1 and '"event": "done"' in received[0] and "AIRAWAT did not answer" in received[0]


def test_the_stream_of_a_finished_meeting_replays_then_ends_with_done(monkeypatch):
    """A client (re)connecting after the run ended used to be held open on keepalives forever: its end-of-stream
    signal was sent before it connected."""
    import json as _json
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api import activity_logs as al
    from backend.api.deps import get_db

    meeting = SimpleNamespace(id=42, bharatvc_meeting_id="M-42", meeting_id="M-42", status="completed", error_message=None)
    monkeypatch.setattr(al, "_resolve_meeting", lambda meeting_id, db: meeting)
    monkeypatch.setattr(al, "_latest_processing_id", lambda pk, db: None)
    app = FastAPI()
    app.include_router(al.router, prefix="/api/v1")
    app.dependency_overrides[get_db] = lambda: None

    with TestClient(app) as client:
        with client.stream("GET", "/api/v1/meetings/M-42/activity-logs/stream") as resp:
            events = [_json.loads(line[5:]) for line in resp.iter_lines() if line.startswith("data:")]
    assert [e["event"] for e in events] == ["connected", "done"] and events[-1]["status"] == "completed"


def test_publish_sync_without_a_bound_loop_is_a_safe_noop():
    from backend.services.sse_manager import SSEManager

    SSEManager().publish_sync(1, {"x": 1})  # must not raise

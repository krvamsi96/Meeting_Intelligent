"""/system/health must answer quickly however slow its dependencies are (Docker gives the probe 10 s).

A cold embedding-model load and an unreachable AIRAWAT (30 s probe timeout) used to be awaited inside the
request, so the container was flagged unhealthy exactly when an upstream was struggling.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import system as system_api
from backend.api.deps import get_db


@pytest.fixture()
def probe():
    return system_api._SlowProbe("test")


def test_a_slow_first_check_is_not_awaited_past_the_wait_budget(probe):
    release = threading.Event()
    started = time.monotonic()
    value = probe.get(lambda: release.wait(5) or "done", ttl=60, default="unknown", on_error="error", wait=0.2)
    assert value == "unknown" and time.monotonic() - started < 2.0
    release.set()


def test_the_result_is_served_once_the_background_check_finishes(probe):
    release = threading.Event()

    def check():
        release.wait(5)
        return "ready"

    assert probe.get(check, ttl=60, default="unknown", on_error="error", wait=0.05) == "unknown"
    release.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if probe.get(check, ttl=60, default="unknown", on_error="error", wait=0.05) == "ready":
            return
        time.sleep(0.02)
    pytest.fail("the finished background result was never served")


def test_a_fast_check_is_answered_inside_the_first_poll(probe):
    assert probe.get(lambda: "ok", ttl=60, default="unknown", on_error="error", wait=2.0) == "ok"


def test_a_stale_result_is_served_immediately_while_it_refreshes(probe):
    calls = []
    gate = threading.Event()

    def check():
        calls.append(1)
        if len(calls) > 1:
            gate.wait(5)  # the refresh is slow
        return f"v{len(calls)}"

    assert probe.get(check, ttl=0.0, default="unknown", on_error="error", wait=2.0) == "v1"
    started = time.monotonic()
    assert probe.get(check, ttl=0.0, default="unknown", on_error="error", wait=2.0) == "v1"  # stale, not awaited
    assert time.monotonic() - started < 1.0
    gate.set()


def test_only_one_check_runs_at_a_time(probe):
    running, peak, lock = 0, 0, threading.Lock()
    gate = threading.Event()

    def check():
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        gate.wait(5)
        with lock:
            running -= 1
        return True

    for _ in range(5):
        probe.get(check, ttl=0.0, default=None, on_error=False, wait=0.01)
    gate.set()
    assert peak == 1


def test_a_failing_check_reports_on_error_instead_of_raising(probe):
    def boom():
        raise RuntimeError("upstream down")

    assert probe.get(boom, ttl=60, default="unknown", on_error="error", wait=2.0) == "error"


def test_a_bound_method_check_is_cached_like_any_other(probe):
    """Regression: `obj.method` is a NEW object on every access, so an identity test never cached it and the
    AIRAWAT probe (model_router.validate_airawat_config) would have run on every single health poll."""

    class Router:
        calls = 0

        def validate(self):
            Router.calls += 1
            return True

    router = Router()
    for _ in range(5):
        assert probe.get(router.validate, ttl=60, default=None, on_error=False, wait=2.0) is True
    assert Router.calls == 1


def test_swapping_the_check_never_serves_the_previous_one(probe):
    assert probe.get(lambda: "first", ttl=60, default="unknown", on_error="error", wait=2.0) == "first"
    assert probe.get(lambda: "second", ttl=60, default="unknown", on_error="error", wait=2.0) == "second"


# ── the endpoint ──────────────────────────────────────────────────────────────


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(system_api, "_PROBE_WAIT_SEC", 0.2)
    monkeypatch.setattr(system_api, "_airawat_probe", system_api._SlowProbe("airawat"))
    monkeypatch.setattr(system_api, "_models_probe", system_api._SlowProbe("models"))
    monkeypatch.setattr(system_api, "vector_service", MagicMock(health=lambda: {"backend": "chromadb", "status": "available"}))
    monkeypatch.setattr(system_api, "get_engine_pool_status", lambda: {"checked_out": 0, "total_capacity": 15})
    session = MagicMock()
    session.execute.return_value.scalar.return_value = 1
    app = FastAPI()
    app.include_router(system_api.router, prefix="/api/v1")
    app.dependency_overrides[get_db] = lambda: session
    return TestClient(app)


def test_health_answers_fast_while_the_model_is_still_loading(client, monkeypatch):
    release = threading.Event()

    def slow_models():
        release.wait(10)
        return {"status": "healthy", "offline_mode": True, "models": {"embedding": {"active_engine": "x"}}}

    monkeypatch.setattr("backend.services.model_loader.get_all_models_health", slow_models)
    monkeypatch.setattr(system_api, "_get_airawat_health", lambda: True)   # this test is about the model probe

    started = time.monotonic()
    body = client.get("/api/v1/system/health").json()
    assert time.monotonic() - started < 3.0
    assert body["services"]["models"] is False and body["status"] == "degraded"  # honest: not ready yet
    assert body["services"]["database"] is True and body["services"]["vector_database"] is True

    release.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        body = client.get("/api/v1/system/health").json()
        if body["services"]["models"]:
            break
        time.sleep(0.05)
    assert body["services"]["models"] is True and body["status"] == "healthy"


def test_health_answers_fast_while_airawat_is_unreachable(client, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr("backend.services.model_loader.get_all_models_health",
                        lambda: {"status": "healthy", "offline_mode": True, "models": {}})
    monkeypatch.setattr(system_api, "_effective_airawat_key", lambda: "a-key")
    monkeypatch.setattr(system_api.model_router, "validate_airawat_config", lambda: release.wait(10) and False)

    started = time.monotonic()
    body = client.get("/api/v1/system/health").json()
    assert time.monotonic() - started < 3.0
    assert body["services"]["airawat"] is True  # unknown yet: informational flag, never fails the poll
    release.set()


def test_a_known_airawat_outage_is_reported_once_the_probe_finished(client, monkeypatch):
    monkeypatch.setattr("backend.services.model_loader.get_all_models_health",
                        lambda: {"status": "healthy", "offline_mode": True, "models": {}})
    monkeypatch.setattr(system_api, "_effective_airawat_key", lambda: "a-key")
    monkeypatch.setattr(system_api.model_router, "validate_airawat_config", lambda: False)

    body = client.get("/api/v1/system/health").json()  # fast probe: answered inside the first poll
    assert body["services"]["airawat"] is False and body["status"] == "degraded"


def test_health_tolerates_a_pool_without_capacity_metrics(client, monkeypatch):
    monkeypatch.setattr(system_api, "get_engine_pool_status", lambda: {"checked_out": None, "total_capacity": None})
    monkeypatch.setattr("backend.services.model_loader.get_all_models_health",
                        lambda: {"status": "healthy", "offline_mode": True, "models": {}})
    monkeypatch.setattr(system_api, "_effective_airawat_key", lambda: "")
    assert client.get("/api/v1/system/health").status_code == 200


def test_a_missing_airawat_key_is_unhealthy_outside_development_mode(monkeypatch):
    """AIRAWAT is the only AI provider: no key means NO AI, which must not be reported as healthy."""
    monkeypatch.setattr(system_api, "_effective_airawat_key", lambda: "")
    monkeypatch.setattr(system_api.settings, "AI_MODE", "auto")
    assert system_api._get_airawat_health() is False
    monkeypatch.setattr(system_api.settings, "AI_MODE", "server")
    assert system_api._get_airawat_health() is False


def test_a_missing_airawat_key_is_unhealthy_in_every_mode(monkeypatch):
    """AIRAWAT is the only AI provider (no local engines in any mode): no key is never reported as healthy."""
    monkeypatch.setattr(system_api, "_effective_airawat_key", lambda: "")
    for mode in ("auto", "local", "server"):
        monkeypatch.setattr(system_api.settings, "AI_MODE", mode)
        assert system_api._get_airawat_health() is False, mode


def test_the_airawat_diagnostics_endpoint_answers_immediately_while_checking(client, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(system_api, "_airawat_diagnostics_probe", system_api._SlowProbe("diag"))
    monkeypatch.setattr("backend.services.airawat_health.run_diagnostics", lambda *a, **k: release.wait(10) and {"ok": True})

    started = time.monotonic()
    body = client.get("/api/v1/system/health/airawat").json()
    assert time.monotonic() - started < 3.0
    assert body["status"] == "checking" and body["provider"] == "airawat"
    release.set()

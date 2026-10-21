"""Regressions found by the Linux-server deployment-readiness audit.

Each test names the failure it prevents. Nothing here needs a database, AIRAWAT, Docker or Linux.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import system as system_api
from backend.api.deps import get_db
from backend.config.config import settings
from backend.scripts import preflight
from backend.services import airawat_health as ah
from backend.services import model_loader as ml
from backend.services.model_router import model_router

REPO_ROOT = Path(__file__).resolve().parents[2]
SECRET = "sk-secret-airawat-key-1234567890"


def _response(status, body):
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(body).encode()
    r.headers["content-type"] = "application/json"
    r.url = "https://airawat.example/x"
    return r


# ── AIRAWAT gateway stalls: the diagnostics retry, the verdict is cached ──────────────────────────
# Measured against the live gateway: ~1 call in 4 was accepted and never answered while the rest took < 1 s, so a
# single-shot probe called a healthy provider "down" a quarter of the time.

CFG = {"api_key": SECRET, "base_url": "https://airawat.example/llm/v1", "llm_model": "meta/llama-3.3-70b-instruct",
       "llm_endpoint": "/chat/completions", "whisper_url": "https://airawat.example/whisper", "language": ""}


@pytest.fixture()
def no_sleep(monkeypatch):
    monkeypatch.setattr(ah.time, "sleep", lambda s: None)
    monkeypatch.setattr("time.sleep", lambda s: None)


def _llm_ok():
    return _response(200, {"model": "m", "choices": [{"message": {"content": "pong"}}]})


def test_a_stalled_request_is_retried_and_the_check_passes(monkeypatch, no_sleep):
    monkeypatch.setattr(ah, "_resolve_config", lambda: dict(CFG))
    calls = []

    def post(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            raise requests.exceptions.ReadTimeout("no answer")
        return _llm_ok() if url.endswith("/chat/completions") else _response(200, {"text": ""})

    monkeypatch.setattr(ah.requests, "post", post)
    report = ah.run_diagnostics(deep=False, attempts=3)
    assert report["ok"] is True and report["llm"]["attempts"] == 2 and len(calls) == 2
    assert "transient" not in report["llm"]


def test_every_attempt_timing_out_is_reported_as_a_timeout_after_the_configured_attempts(monkeypatch, no_sleep):
    monkeypatch.setattr(ah, "_resolve_config", lambda: dict(CFG))
    calls = []

    def post(*a, **k):
        calls.append(1)
        raise requests.exceptions.ReadTimeout("no answer")

    monkeypatch.setattr(ah.requests, "post", post)
    report = ah.run_diagnostics(deep=False, attempts=3)
    assert report["ok"] is False and "timeout" in report["llm"]["error"] and report["llm"]["attempts"] == 3 and len(calls) == 3


@pytest.mark.parametrize("status", [400, 401, 404])
def test_a_permanent_rejection_is_not_retried(monkeypatch, no_sleep, status):
    monkeypatch.setattr(ah, "_resolve_config", lambda: dict(CFG))
    calls = []
    monkeypatch.setattr(ah.requests, "post", lambda *a, **k: calls.append(1) or _response(status, {"message": "nope"}))
    report = ah.run_diagnostics(deep=False, attempts=4)
    assert report["ok"] is False and len(calls) == 1 and report["llm"]["attempts"] == 1


def test_a_gateway_5xx_is_retried(monkeypatch, no_sleep):
    monkeypatch.setattr(ah, "_resolve_config", lambda: dict(CFG))
    answers = iter([_response(503, {"message": "busy"}), _llm_ok()])
    monkeypatch.setattr(ah.requests, "post", lambda *a, **k: next(answers))
    assert ah.run_diagnostics(deep=False, attempts=2)["ok"] is True


def test_the_default_attempts_follow_airawat_retry_count(monkeypatch):
    monkeypatch.setattr(settings, "AIRAWAT_RETRY_COUNT", 2)
    assert ah.diagnostic_attempts() == 3
    monkeypatch.setattr(settings, "AIRAWAT_RETRY_COUNT", 0)
    assert ah.diagnostic_attempts() == 1
    monkeypatch.setattr(settings, "AIRAWAT_RETRY_COUNT", 50)
    assert ah.diagnostic_attempts() == 4                     # bounded: a health probe must not run for minutes


@pytest.fixture()
def router(monkeypatch, no_sleep):
    monkeypatch.setattr(model_router, "_get_dynamic_config",
                        lambda: {"airawat": {"api_key": SECRET, "base_url": "https://airawat.example/llm/v1",
                                             "llama_endpoint": "/chat/completions", "default_llm_model": "m"}})
    monkeypatch.setattr(settings, "AIRAWAT_RETRY_COUNT", 2)
    monkeypatch.setattr(settings, "AIRAWAT_HEALTH_TTL_SEC", 300)
    model_router._validation = (False, 0.0)
    yield model_router
    model_router._validation = (False, 0.0)


def test_validate_retries_a_stall_and_reports_healthy(router, monkeypatch):
    answers = iter([requests.exceptions.ReadTimeout("stall"), _response(200, {"choices": []})])

    def post(*a, **k):
        item = next(answers)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(requests, "post", post)
    assert router.validate_airawat_config() is True


def test_validate_does_not_ping_the_provider_before_every_pipeline_step(router, monkeypatch):
    """diarization's speaker-label step asks per meeting; each ask used to be a live 30 s-timeout request."""
    calls = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(1) or _response(200, {}))
    assert router.validate_airawat_config() and router.validate_airawat_config() and router.validate_airawat_config()
    assert len(calls) == 1
    assert router.validate_airawat_config(use_cache=False) and len(calls) == 2


def test_a_bad_key_is_not_retried_and_the_failure_is_cached_briefly(router, monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(1) or _response(401, {}))
    assert router.validate_airawat_config() is False and len(calls) == 1
    assert router.validate_airawat_config() is False and len(calls) == 1        # negative verdict served from cache
    assert router.validate_airawat_config(use_cache=False) is False and len(calls) == 2


def test_validate_never_logs_the_key(router, monkeypatch, caplog):
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.ConnectionError(f"failed {SECRET}")))
    with caplog.at_level("DEBUG"):
        router.validate_airawat_config(use_cache=False)
    assert SECRET not in caplog.text


# ── which database is in use: a fallback must be visible ─────────────────────────────────────────

def _target_module(monkeypatch, source, env):
    from backend.database import connection as dbc

    monkeypatch.setattr(dbc, "_db_source", source)
    for key in ("SERVER_DATABASE_URL", "SERVER_DB_HOST"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return dbc


def test_the_server_database_is_not_a_fallback(monkeypatch):
    dbc = _target_module(monkeypatch, "environment-selector:server", {"SERVER_DB_HOST": "10.0.0.5"})
    t = dbc.get_database_target()
    assert t["environment"] == "server" and t["fallback_active"] is False and "warning" not in t


def test_using_the_local_database_while_a_server_one_is_configured_is_reported_as_a_fallback(monkeypatch):
    dbc = _target_module(monkeypatch, "environment-selector:local", {"SERVER_DB_HOST": "10.0.0.5"})
    t = dbc.get_database_target()
    assert t["fallback_active"] is True and "AI_MODE=server" in t["warning"]
    assert "password" not in json.dumps(t).lower()


def test_a_machine_with_only_a_local_database_is_not_a_fallback(monkeypatch):
    dbc = _target_module(monkeypatch, "environment-selector:local", {})
    assert dbc.get_database_target()["fallback_active"] is False


@pytest.fixture()
def health_client(monkeypatch):
    monkeypatch.setattr(system_api, "_PROBE_WAIT_SEC", 0.2)
    monkeypatch.setattr(system_api, "_airawat_probe", system_api._SlowProbe("airawat"))
    monkeypatch.setattr(system_api, "_models_probe", system_api._SlowProbe("models"))
    monkeypatch.setattr(system_api, "vector_service", MagicMock(health=lambda: {"backend": "chromadb", "status": "available"}))
    monkeypatch.setattr(system_api, "get_engine_pool_status", lambda: {"checked_out": 0, "total_capacity": 15})
    monkeypatch.setattr(system_api, "_get_airawat_health", lambda: True)
    monkeypatch.setattr("backend.services.model_loader.get_all_models_health",
                        lambda: {"status": "healthy", "offline_mode": True, "models": {}})
    session = MagicMock()
    session.execute.return_value.scalar.return_value = 1
    app = FastAPI()
    app.include_router(system_api.router, prefix="/api/v1")
    app.dependency_overrides[get_db] = lambda: session
    return TestClient(app)


def test_health_reports_the_database_target_and_is_healthy_on_the_configured_server(health_client, monkeypatch):
    target = {"environment": "server", "host": "db.internal", "port": 5432, "database": "meetings",
              "server_database_configured": True, "fallback_active": False, "ai_mode": "server"}
    monkeypatch.setattr(system_api, "get_database_target", lambda: target)
    body = health_client.get("/api/v1/system/health").json()
    assert body["database_target"] == target and body["status"] == "healthy" and "warning" not in body


def test_health_is_degraded_and_says_why_when_a_fallback_database_is_active(health_client, monkeypatch):
    target = {"environment": "local", "host": "127.0.0.1", "port": 5432, "database": "govindb",
              "server_database_configured": True, "fallback_active": True, "ai_mode": "auto",
              "warning": "The configured SERVER database was unreachable at startup"}
    monkeypatch.setattr(system_api, "get_database_target", lambda: target)
    body = health_client.get("/api/v1/system/health").json()
    assert body["status"] == "degraded" and "SERVER database was unreachable" in body["warning"]
    assert body["services"]["database"] is True                    # connected - just not to the intended database


# ── the Git LFS pointer: the origin of "header too large" ───────────────────────────────────────

LFS_POINTER = (b"version https://git-lfs.github.com/spec/v1\n"
               b"oid sha256:53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db\nsize 90868376\n")


def _model_dir(tmp_path, weights: bytes):
    d = tmp_path / "all-MiniLM-L6-v2"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(weights)
    return d


def test_the_lfs_pointer_really_produces_the_reported_safetensors_error(tmp_path):
    """Root cause, reproduced: safetensors reads the pointer's first 8 bytes ('version ') as a header length."""
    from safetensors import safe_open

    d = _model_dir(tmp_path, LFS_POINTER)
    with pytest.raises(Exception, match="header too large"):
        safe_open(str(d / "model.safetensors"), framework="pt")


def test_the_loader_names_the_pointer_instead_of_failing_with_a_safetensors_error(tmp_path):
    problem = ml.inspect_embedding_model_dir(str(_model_dir(tmp_path, LFS_POINTER)))
    assert "Git LFS pointer" in problem and "model.safetensors" in problem


def test_preflight_fails_with_the_fix_when_the_model_is_an_lfs_pointer(tmp_path, monkeypatch):
    d = _model_dir(tmp_path, LFS_POINTER)
    monkeypatch.setattr(ml, "get_embedding_model_search_paths", lambda: [str(d)])
    rep = preflight.Report(startup=False)
    assert preflight.check_embedding(rep, startup=False) is False
    result = rep.results[0]
    assert result.status == preflight.FAIL and "Git LFS pointer" in result.message and "git lfs pull" in result.fix


def test_preflight_names_a_truncated_model_file(tmp_path, monkeypatch):
    d = _model_dir(tmp_path, b"\x10\x00\x00\x00\x00\x00\x00\x00" + b"{}" * 4)     # header claims 16 bytes, file has 16 bytes total
    monkeypatch.setattr(ml, "get_embedding_model_search_paths", lambda: [str(d)])
    rep = preflight.Report(startup=False)
    assert preflight.check_embedding(rep, startup=False) is False and rep.results[0].status == preflight.FAIL


# ── preflight behaviour ─────────────────────────────────────────────────────────────────────────

def test_at_startup_only_python_dependencies_and_postgres_are_fatal(capsys):
    rep = preflight.Report(startup=True)
    rep.add("embedding", preflight.FAIL, "model missing", "copy it")
    rep.add("media", preflight.FAIL, "no ffmpeg")
    assert not rep.failed and [r.status for r in rep.results] == [preflight.WARN, preflight.WARN]
    rep.add("postgres", preflight.FAIL, "cannot connect")
    assert rep.failed


def test_in_the_full_gate_every_failure_is_fatal(capsys):
    rep = preflight.Report(startup=False)
    rep.add("embedding", preflight.FAIL, "model missing")
    assert rep.failed


def test_preflight_refuses_a_chromadb_directory_inside_a_temp_dir(tmp_path, monkeypatch):
    import tempfile

    inside_tmp = os.path.join(tempfile.gettempdir(), "chroma_probe_dir")
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIRECTORY", inside_tmp)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "up"))
    monkeypatch.setattr(settings, "NEMO_CACHE_DIR", str(tmp_path / "nemo"))
    rep = preflight.Report(startup=False)
    preflight.check_directories(rep)
    assert rep.results[-1].status == preflight.FAIL and "temporary" in rep.results[-1].message
    shutil.rmtree(inside_tmp, ignore_errors=True)


def test_preflight_accepts_a_persistent_writable_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIRECTORY", str(tmp_path / "data" / "chroma"))
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "NEMO_CACHE_DIR", str(tmp_path / "nemo"))
    monkeypatch.setattr(preflight, "_is_temp_path", lambda path: False)      # pytest's tmp_path IS a temp dir
    rep = preflight.Report(startup=False)
    preflight.check_directories(rep)
    assert rep.results[-1].status in (preflight.OK,), rep.results[-1].message
    assert (tmp_path / "data" / "chroma").is_dir()


def test_preflight_output_never_contains_the_api_key(monkeypatch, capsys):
    monkeypatch.setattr(ah, "_resolve_config", lambda: dict(CFG))
    rep = preflight.Report(startup=False)
    preflight.check_configuration(rep)
    out = capsys.readouterr().out + json.dumps([r.__dict__ for r in rep.results], default=str)
    assert SECRET not in out and ah.key_fingerprint(SECRET) in out


def test_more_than_one_worker_is_a_failure_because_chromadb_has_one_owner(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "4")
    monkeypatch.setattr(ah, "_resolve_config", lambda: dict(CFG))
    rep = preflight.Report(startup=False)
    preflight.check_configuration(rep)
    assert any(r.check == "config.workers" and r.status == preflight.FAIL for r in rep.results)


def test_preflight_reports_the_spoken_language_source(monkeypatch):
    """AIRAWAT Whisper cannot detect a language: preflight says whether a default exists (no local model involved)."""
    from backend.services.model_router import model_router

    for lang, status in (("en", preflight.OK), ("", preflight.WARN), ("not a code!", preflight.FAIL)):
        monkeypatch.setattr(model_router, "get_effective_config", lambda lang=lang: {"language": lang})
        rep = preflight.Report(startup=False)
        preflight.check_language(rep)
        assert [r.status for r in rep.results if r.check == "language"] == [status], lang


# ── Linux launch scripts ────────────────────────────────────────────────────────────────────────

SHELL_SCRIPTS = ["run_backend.sh", "start.sh", "stop.sh", "healthcheck.sh", "setup_backend.sh", "deploy_docker.sh",
                 "deploy_model.sh", "test_server.sh"]
def _find_usable_bash():
    candidates = [
        shutil.which("bash"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                res = subprocess.run([c, "-c", "echo ok"], capture_output=True, text=True, timeout=5)
                if res.returncode == 0 and res.stdout.strip() == "ok":
                    return c
            except Exception:
                continue
    return None

BASH = _find_usable_bash()


@pytest.mark.parametrize("name", SHELL_SCRIPTS)
def test_shell_scripts_are_lf_only_and_parse(name):
    raw = (REPO_ROOT / name).read_bytes()
    assert b"\r" not in raw, f"{name} has CR characters: bash on Linux fails with '\\r: command not found'"
    assert raw.startswith(b"#!"), f"{name} has no shebang"
    if BASH:
        assert subprocess.run([BASH, "-n", str(REPO_ROOT / name)], capture_output=True).returncode == 0


def test_dockerfile_and_compose_are_lf_only():
    for name in ("Dockerfile.backend", "docker-compose.yml", ".env.example"):
        assert b"\r" not in (REPO_ROOT / name).read_bytes(), name


@pytest.mark.skipif(not BASH, reason="bash not available")
@pytest.mark.parametrize("script", ["run_backend.sh", "start.sh"])
def test_launchers_lowercase_the_log_level_because_uvicorn_rejects_INFO(script):
    """Regression: .env.example says LOG_LEVEL=INFO; `uvicorn --log-level INFO` exits with 'Invalid value'."""
    text = (REPO_ROOT / script).read_text(encoding="utf-8")
    line = next(l for l in text.splitlines() if l.startswith("LOG_LEVEL=") and "tr " in l)
    out = subprocess.run([BASH, "-c", f'LOG_LEVEL=INFO; {line}; echo "$LOG_LEVEL"'], capture_output=True, text=True)
    assert out.stdout.strip() == "info"
    assert '--log-level "${LOG_LEVEL}"' in text


@pytest.mark.skipif(not BASH, reason="bash not available")
def test_docker_entrypoint_is_valid_json_and_lowercases_the_log_level():
    """The ENTRYPOINT is parsed the way Docker does (join continuation lines, JSON-decode) and executed with echo
    standing in for uvicorn."""
    text = (REPO_ROOT / "Dockerfile.backend").read_text(encoding="utf-8")
    joined = re.sub(r"\\\n", "", text)
    match = re.search(r"^ENTRYPOINT (\[.*\])$", joined, re.MULTILINE)
    argv = json.loads(match.group(1))
    assert argv[:2] == ["sh", "-c"]
    script = argv[2].replace("exec gosu appuser python -m uvicorn", "echo uvicorn").replace("gosu root chown", "true")
    out = subprocess.run([BASH, "-c", script], capture_output=True, text=True, env={**os.environ, "LOG_LEVEL": "INFO"})
    assert out.returncode == 0, out.stderr
    assert "--log-level info" in out.stdout and "INFO" not in out.stdout


def test_launchers_run_the_startup_preflight_and_start_from_their_own_directory():
    for name in ("run_backend.sh", "start.sh"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "backend.scripts.preflight --startup" in text, name
        assert 'cd "$(dirname "$(readlink -f' in text, f"{name} must not depend on the caller's working directory"
        assert "SKIP_PREFLIGHT" in text
    setup = (REPO_ROOT / "setup_backend.sh").read_text(encoding="utf-8")
    assert "backend.scripts.preflight" in setup and "git lfs pull" in setup


def test_start_sh_no_longer_tests_for_the_removed_lexical_fallback_engine():
    """Regression: the old check looked for 'lexical_fallback', so a missing model printed '[OK] Embedding Engine'."""
    assert "lexical_fallback" not in (REPO_ROOT / "start.sh").read_text(encoding="utf-8")


def test_stop_script_is_graceful_and_does_not_kill_by_port():
    text = (REPO_ROOT / "stop.sh").read_text(encoding="utf-8")
    assert "kill -TERM" in text and "kill -KILL" in text
    assert not re.search(r"fuser\s+-k|kill\s+-9|next dev", text), "stop.sh must not SIGKILL whatever owns a port"
    assert text.index("kill -TERM") < text.index("kill -KILL")


def test_deploy_model_script_has_no_windows_paths_or_baked_in_host():
    text = (REPO_ROOT / "deploy_model.sh").read_text(encoding="utf-8")
    assert not re.search(r"[A-Za-z]:\\|OneDrive|krvam", text)
    assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text), "no default server address"


def test_jenkins_fails_the_build_on_an_lfs_pointer_and_verifies_the_deployment():
    text = (REPO_ROOT / "Jenkinsfile").read_text(encoding="utf-8")
    assert text.index("stage('Verify Embedding Model')") < text.index("stage('Create Tar')")
    assert "git lfs pull" in text and "exit 1" in text
    assert text.index("stage('Reload via PM2')") < text.index("stage('Post-Deploy Health')") < text.index("stage('Post-Deploy Preflight')")
    assert "backend.scripts.preflight" in text


def test_fetch_models_pins_the_checksum_of_the_weights_the_repository_ships():
    """fetch_models is the LFS-free way to get the real weights; its pin must equal the file the repo ships."""
    import hashlib

    from backend.scripts import fetch_models

    shipped = REPO_ROOT / "models" / "embeddings" / "all-MiniLM-L6-v2" / "model.safetensors"
    if not shipped.is_file() or shipped.stat().st_size < 1_000_000:
        pytest.skip("the shipped weights are not present (a Git LFS pointer or no model directory)")
    assert shipped.stat().st_size == fetch_models.EMBEDDING_WEIGHTS_SIZE
    assert hashlib.sha256(shipped.read_bytes()).hexdigest() == fetch_models.EMBEDDING_WEIGHTS_SHA256


# ── media validation at upload ──────────────────────────────────────────────────────────────────
# python-magic needs libmagic and falls back to the file EXTENSION without it: a corrupt file or a video with no audio
# track was accepted, got a meeting row and a background job, and failed later with "FFmpeg returned non-zero exit code".

from fastapi import HTTPException  # noqa: E402

from backend.utils import media_tools  # noqa: E402
from backend.utils.security import validate_media_file  # noqa: E402

FFMPEG_OK = media_tools.ffmpeg_available() and media_tools.ffprobe_available()
needs_ffmpeg = pytest.mark.skipif(not FFMPEG_OK, reason="ffmpeg / ffprobe not installed")


def _ffmpeg(*args):
    subprocess.run([media_tools.FFMPEG, "-y", "-nostdin", "-loglevel", "error", *args], check=True, timeout=120)


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    if not FFMPEG_OK:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("media")
    with_audio, silent_video, wav, corrupt, empty = (d / n for n in ("a.mp4", "v.mp4", "s.wav", "bad.mp4", "empty.mp4"))
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-f", "lavfi", "-i", "sine=frequency=440", "-t", "1",
            "-pix_fmt", "yuv420p", "-shortest", str(with_audio))
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-t", "1", "-an", "-pix_fmt", "yuv420p", str(silent_video))
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440", "-t", "1", str(wav))
    corrupt.write_bytes(os.urandom(4096))
    empty.write_bytes(b"")
    return SimpleNamespace(video=with_audio, silent=silent_video, wav=wav, corrupt=corrupt, empty=empty)


@needs_ffmpeg
def test_a_video_with_audio_and_an_audio_only_file_pass(samples):
    info = validate_media_file(str(samples.video))
    assert info["has_video"] is True and info["audio_codec"] and 0.5 < info["duration"] < 2.0
    assert validate_media_file(str(samples.wav))["has_video"] is False


@needs_ffmpeg
def test_a_video_without_an_audio_track_is_rejected_with_a_clear_reason(samples):
    with pytest.raises(HTTPException) as err:
        validate_media_file(str(samples.silent))
    assert err.value.status_code == 400 and "no audio track" in err.value.detail


@needs_ffmpeg
@pytest.mark.parametrize("which", ["corrupt", "empty"])
def test_a_corrupt_or_empty_file_is_rejected_without_leaking_paths(samples, which):
    path = getattr(samples, which)
    with pytest.raises(HTTPException) as err:
        validate_media_file(str(path))
    assert err.value.status_code == 400 and "not a readable" in err.value.detail
    assert str(path) not in err.value.detail and os.path.basename(str(path)) not in err.value.detail


def test_validation_is_skipped_not_failed_when_ffprobe_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(media_tools, "ffprobe_available", lambda: False)
    f = tmp_path / "x.mp4"
    f.write_bytes(b"whatever")
    assert validate_media_file(str(f)) == {}


@pytest.fixture()
def upload_client(monkeypatch, tmp_path):
    from backend.api import upload as upload_api
    from backend.api.deps import get_meeting_repository

    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(upload_api, "PIPELINE_REGISTRY", {"media": lambda *a: None, "document": lambda *a: None})
    monkeypatch.setattr(upload_api, "process_meeting_pipeline", lambda *a: None)
    repo = MagicMock()
    repo.get_by_public_id.return_value = None
    repo.db.query.return_value.filter.return_value.first.return_value = None
    repo.create.return_value = SimpleNamespace(id=7)
    app = FastAPI()
    app.include_router(upload_api.router, prefix="/api/v1")
    app.dependency_overrides[get_meeting_repository] = lambda: repo
    return SimpleNamespace(client=TestClient(app), dir=tmp_path, repo=repo)


def _post(uc, path, name, mime="video/mp4"):
    with open(path, "rb") as fh:
        return uc.client.post("/api/v1/upload", files={"file": (name, fh, mime)}, data={"meeting_id": "4242"})


@needs_ffmpeg
def test_upload_rejects_a_corrupt_recording_and_leaves_no_file_or_meeting_behind(upload_client, samples):
    r = _post(upload_client, samples.corrupt, "meeting.mp4")
    assert r.status_code == 400 and "not a readable" in r.json()["detail"]
    assert list(upload_client.dir.iterdir()) == []                      # the partial upload was removed
    upload_client.repo.create.assert_not_called()                       # no meeting row, no background job


@needs_ffmpeg
def test_upload_rejects_a_video_without_audio(upload_client, samples):
    r = _post(upload_client, samples.silent, "muted.mp4")
    assert r.status_code == 400 and "no audio track" in r.json()["detail"]
    upload_client.repo.create.assert_not_called()


@needs_ffmpeg
def test_upload_accepts_a_valid_recording(upload_client, samples):
    r = _post(upload_client, samples.video, "meeting.mp4")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["meeting_id"] == "4242" and body["id"] == 7 and body["status"] == "processing"
    assert (upload_client.dir / "4242_meeting.mp4").is_file()
    upload_client.repo.create.assert_called_once()


def test_upload_rejects_an_unsupported_extension(upload_client, tmp_path):
    f = tmp_path / "x.exe"
    f.write_bytes(b"MZ")
    r = _post(upload_client, f, "run.exe", "application/octet-stream")
    assert r.status_code == 400 and "Unsupported file type" in r.json()["detail"]


# ── report content: facts the system knows must not be "Not Specified" ────────────────────────────

from backend.services import llm_workflow as wf  # noqa: E402

MOM = ("### 📅 MEETING INFORMATION\n- **Date & Time:** Not Specified\n- **Duration:** Not Specified\n"
       "- **Meeting Type:** Review\n- **Participants:** Speaker_1\n\n### 🎯 EXECUTIVE SUMMARY\nText.")


def test_the_mom_header_carries_the_real_date_time_and_duration():
    out = wf.enforce_mom_facts(MOM, "2026-09-26", "11:19 AM", "00:00:55")
    assert "- **Date & Time:** 2026-09-26 | 11:19 AM" in out and "- **Duration:** 00:00:55" in out
    assert "Not Specified" not in out.split("Meeting Type")[0]
    assert out.count("**Duration:**") == 1 and "### 🎯 EXECUTIVE SUMMARY\nText." in out       # nothing else touched


def test_unknown_facts_are_left_as_the_model_wrote_them():
    out = wf.enforce_mom_facts(MOM, "Unknown Date", "Unknown Time", "")
    assert out == MOM
    assert wf.enforce_mom_facts("", "2026-01-01", "10:00 AM", "00:10:00") == ""


def test_a_mom_without_those_lines_is_unchanged():
    text = "### Minutes\nNo header here."
    assert wf.enforce_mom_facts(text, "2026-09-26", "11:19 AM", "00:00:55") == text


def test_a_hung_call_that_is_retried_is_still_waited_for(monkeypatch):
    """AIRAWAT_TIMEOUT_SEC caps ONE attempt. The section futures must wait for every attempt, or the retry that rescues
    a request the gateway silently dropped is abandoned before it can finish."""
    monkeypatch.setattr(model_router, "get_effective_config", lambda: {"timeout": 120, "retry_count": 2})
    assert wf._section_wait_seconds() == 120 * 3 + 30
    monkeypatch.setattr(model_router, "get_effective_config", lambda: (_ for _ in ()).throw(RuntimeError("no db")))
    monkeypatch.setattr(settings, "AIRAWAT_TIMEOUT_SEC", 600)
    monkeypatch.setattr(settings, "AIRAWAT_RETRY_COUNT", 1)
    assert wf._section_wait_seconds() == 600 * 2 + 30


def test_the_diarization_api_discloses_when_the_weak_fallback_embedder_was_used():
    from backend.api import meetings as meetings_api

    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = []
    meeting = SimpleNamespace(display_id="9", bharatvc_meeting_id="9", id=9, duration=10.0, diarization_status="SUCCESS",
                              diarization_metadata_json=json.dumps({"neural_embedder_used": False,
                                                                    "selected_engine": "nvidia_nemo", "fallback_used": False}))
    d = meetings_api._build_nemo_diarization_response(meeting, db)["diarization"]
    assert d["neural_embedder_used"] is False and d["embedding_fidelity"] == "acoustic_fallback"
    assert "titanet_large.nemo" in d["warning"] and d["engine"] == "NVIDIA_NEMO"       # existing key unchanged

    meeting.diarization_metadata_json = json.dumps({"neural_embedder_used": True})
    d = meetings_api._build_nemo_diarization_response(meeting, db)["diarization"]
    assert d["embedding_fidelity"] == "titanet_neural" and "warning" not in d

    meeting.diarization_metadata_json = None
    assert meetings_api._build_nemo_diarization_response(meeting, db)["diarization"]["embedding_fidelity"] == "unknown"


def test_the_upload_stamps_the_meeting_time_in_the_deployment_timezone(upload_client, monkeypatch, tmp_path):
    """Regression: meeting_time was UTC with no label ('05:49 AM' for a 11:19 IST recording)."""
    from datetime import datetime, timezone

    from backend.api import upload as upload_api

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 25, 23, 30, tzinfo=timezone.utc)      # 05:00 next day in Kolkata

    monkeypatch.setattr(upload_api, "datetime", FixedDatetime)
    monkeypatch.setattr(settings, "DEFAULT_TIMEZONE", "Asia/Kolkata")
    monkeypatch.setattr(upload_api, "validate_file_type", lambda *a, **k: None)
    monkeypatch.setattr(upload_api, "validate_media_file", lambda *a, **k: {})
    doc = tmp_path / "n.txt"
    doc.write_text("meeting notes")
    r = _post(upload_client, doc, "notes.txt", "text/plain")
    assert r.status_code == 200, r.text
    created = upload_client.repo.create.call_args.kwargs
    assert created["meeting_date"] == "2026-09-26" and created["meeting_time"] == "05:00 AM"


def test_check_airawat_repeat_reports_the_stall_rate(monkeypatch, capsys):
    from backend.scripts import check_airawat

    monkeypatch.setattr(ah, "_resolve_config", lambda: dict(CFG))
    results = iter([{"ok": True, "latency_ms": 800}, {"ok": False, "error": "no answer within 5s (timeout)"},
                    {"ok": True, "latency_ms": 900}, {"ok": True, "latency_ms": 700}])
    monkeypatch.setattr(ah, "check_llm", lambda cfg, timeout: next(results))
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert check_airawat._measure(4, 5.0) == 0
    out = capsys.readouterr().out
    assert "answered 3" in out and "no answer within the timeout 1" in out and "median 800" in out and "1/4" in out
    assert SECRET not in out


# ── concurrent pipelines must not mislabel each other's log lines ────────────────────────────────

def test_concurrent_pipelines_keep_their_own_log_labels():
    """Regression: the context was class-level state, so meetings processed at the same time stamped each other's job /
    meeting ids (and one finishing blanked the rest) - the logs of a busy server pointed at the wrong meeting."""
    import logging
    import threading

    from backend.utils.logging_setup import PipelineLogContext, _PipelineContextFilter

    seen, lock = [], threading.Lock()

    class Capture(logging.Handler):
        def emit(self, record):
            with lock:
                seen.append((record.job_id, record.meeting_id, record.pipeline_stage, record.getMessage()))

    logger = logging.getLogger("test.pipeline.context")
    handler = Capture()
    handler.addFilter(_PipelineContextFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    barrier = threading.Barrier(4)

    def run(mid):
        with PipelineLogContext(job_id=str(mid), meeting_id=str(mid), stage="AUDIO"):
            barrier.wait()                                   # all four are inside their context at the same moment
            logger.info(f"m{mid}-a")
            time_slice = threading.Event()
            time_slice.wait(0.05)
            logger.info(f"m{mid}-b")
        logger.info(f"m{mid}-after")

    threads = [threading.Thread(target=run, args=(m,)) for m in (101, 102, 103, 104)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    logger.removeHandler(handler)

    for job, meeting, stage, message in seen:
        mid = message.split("-")[0][1:]
        if message.endswith("-after"):
            assert (job, meeting, stage) == ("-", "-", "-"), message
        else:
            assert (job, meeting, stage) == (mid, mid, "AUDIO"), f"{message} was labelled job={job} meeting={meeting}"
    assert len(seen) == 12


def test_a_nested_context_restores_the_outer_one_on_exit():
    import logging

    from backend.utils.logging_setup import PipelineLogContext, _PipelineContextFilter

    record = logging.LogRecord("x", logging.INFO, __file__, 1, "m", None, None)
    flt = _PipelineContextFilter()
    with PipelineLogContext(job_id="7", meeting_id="7", stage="OUTER"):
        with PipelineLogContext(job_id="7", meeting_id="7", stage="INNER"):
            flt.filter(record)
            assert record.pipeline_stage == "INNER"
        flt.filter(record)
        assert (record.job_id, record.pipeline_stage) == ("7", "OUTER")
    flt.filter(record)
    assert record.job_id == "-"


# ── /final carries the language and keeps the original-language transcript alongside the English one ───────────

@pytest.fixture()
def db_session():
    """Real PostgreSQL schema, inside an outer transaction that is ALWAYS rolled back (skipped without a database)."""
    from sqlalchemy import inspect
    from sqlalchemy.orm import Session

    from backend.database.connection import engine

    try:
        if not inspect(engine).has_table("meetings"):
            pytest.skip("meetings table does not exist yet")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database not reachable: {exc}")
    conn = engine.connect()
    outer = conn.begin()
    session = Session(bind=conn, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        outer.rollback()
        conn.close()


def _final_client(session):
    from backend.api import final as final_api

    app = FastAPI()
    app.include_router(final_api.router, prefix="/api/v1")
    app.dependency_overrides[get_db] = lambda: session
    return TestClient(app)


def test_final_reports_the_language_and_the_original_transcript(db_session):
    from backend.models.models import AnalysisResult, Meeting, TranscriptSegment

    m = Meeting(bharatvc_meeting_id="9990001", meeting_id="9990001", display_id="9990001", filename="hi.mp4", file_type="media",
                filepath="/x", title="Hindi", status="completed", duration=20.0, speaker_count=1,
                source_language="hi", source_language_name="Hindi", source_language_confidence=0.97,
                translation_applied=True, output_language="en")
    db_session.add(m)
    db_session.flush()
    db_session.add_all([
        TranscriptSegment(meeting_id=m.id, speaker="Speaker_1", start_time="0.0", end_time="5.0",
                          text="We will migrate the database.", raw_text="हम डेटाबेस माइग्रेट करेंगे।",
                          original_text="हम डेटाबेस माइग्रेट करेंगे।"),
        TranscriptSegment(meeting_id=m.id, speaker="Speaker_1", start_time="5.0", end_time="9.0",
                          text="Ramesh sends the report.", raw_text="रमेश रिपोर्ट भेजेंगे।", original_text=None),
        AnalysisResult(meeting_id=m.id, summary="The team will migrate the database.", mom="### MoM"),
    ])
    db_session.flush()

    body = _final_client(db_session).get("/api/v1/meetings/9990001/final").json()
    meta = body["meeting"]
    assert (meta["source_language"], meta["source_language_name"], meta["translation_applied"], meta["output_language"]) \
        == ("hi", "Hindi", True, "en")
    assert meta["source_language_confidence"] == pytest.approx(0.97)
    english = [s["text"] for s in body["transcript"]]
    assert english == ["We will migrate the database.", "Ramesh sends the report."]      # English stays in `text`
    assert body["transcript"][0]["original_text"] == "हम डेटाबेस माइग्रेट करेंगे।"           # what was actually said
    assert body["transcript"][1]["original_text"] == "रमेश रिपोर्ट भेजेंगे।"               # falls back to the raw ASR text


def test_final_for_an_english_meeting_is_unchanged_apart_from_the_new_fields(db_session):
    from backend.models.models import AnalysisResult, Meeting, TranscriptSegment

    m = Meeting(bharatvc_meeting_id="9990002", meeting_id="9990002", display_id="9990002", filename="en.mp4", file_type="media",
                filepath="/x", title="English", status="completed", duration=10.0, source_language="en",
                source_language_name="English", output_language="en", translation_applied=False)
    db_session.add(m)
    db_session.flush()
    db_session.add_all([TranscriptSegment(meeting_id=m.id, speaker="Speaker_1", start_time="0.0", end_time="3.0", text="Hello team."),
                        AnalysisResult(meeting_id=m.id, summary="A greeting.", mom="### MoM")])
    db_session.flush()
    body = _final_client(db_session).get("/api/v1/meetings/9990002/final").json()
    assert body["meeting"]["source_language"] == "en" and body["meeting"]["translation_applied"] is False
    assert body["transcript"][0]["text"] == "Hello team." and body["transcript"][0]["original_text"] is None
    assert "summary" in body and "action_items" in body and "agenda" in body       # the existing contract is intact


# ── NeMo speaker model: the offline server must never try to download it from NGC ─────────────────
# Seen on the server: "[NeMo] Downloading from: https://api.ngc.nvidia.com/.../titanet-l.nemo" although OFFLINE_MODE=true,
# because titanet_large.nemo was neither in DIARIZATION_MODEL_PATH nor in NeMo's (empty) cache.


def _fake_nemo(monkeypatch, from_pretrained):
    import types

    label_model = SimpleNamespace(
        list_available_models=lambda: [SimpleNamespace(pretrained_model_name="titanet_large",
                                                       location="https://api.ngc.nvidia.com/v2/x/files/titanet-l.nemo")],
        from_pretrained=from_pretrained,
    )
    nemo, collections, asr = types.ModuleType("nemo"), types.ModuleType("nemo.collections"), types.ModuleType("nemo.collections.asr")
    asr.models = SimpleNamespace(EncDecSpeakerLabelModel=label_model)
    nemo.collections, collections.asr = collections, asr
    for name, module in (("nemo", nemo), ("nemo.collections", collections), ("nemo.collections.asr", asr)):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture()
def offline_loader(tmp_path, monkeypatch):
    from backend.services import diarization_service as ds

    (tmp_path / "models").mkdir()
    monkeypatch.setattr(settings, "OFFLINE_MODE", True)
    monkeypatch.setattr(settings, "DIARIZATION_MODEL_PATH", str(tmp_path / "models"))
    monkeypatch.setenv("NEMO_CACHE_DIR", str(tmp_path / "nemo_cache"))
    monkeypatch.setattr(ds, "_get_nemo_device", lambda: "cpu")
    monkeypatch.setattr(ds, "_nemo_speaker_model", None)
    return ds


def test_offline_without_the_speaker_model_fails_at_once_instead_of_downloading(offline_loader, monkeypatch, caplog):
    def _download(**_kw):
        raise AssertionError("OFFLINE_MODE must never call from_pretrained() without a cached model: it downloads from NGC")

    _fake_nemo(monkeypatch, _download)
    with caplog.at_level("WARNING"):
        model = offline_loader._load_nemo_speaker_model_locked()
    assert model == "NEURAL_ACOUSTIC_EMBEDDING" and offline_loader.using_neural_embedder() is False
    assert "OFFLINE_MODE=true forbids downloading it from NGC" in caplog.text      # the real reason is logged, not hidden


def test_offline_still_uses_a_model_already_in_nemos_cache(offline_loader, tmp_path, monkeypatch):
    cached = tmp_path / "nemo_cache" / "titanet-l" / "11ba0924fdf87c049e339adbf6899d48"
    cached.mkdir(parents=True)
    (cached / "titanet-l.nemo").write_bytes(b"x")
    loaded = MagicMock()
    _fake_nemo(monkeypatch, lambda **_kw: loaded)
    assert offline_loader._load_nemo_speaker_model_locked() is loaded and offline_loader.using_neural_embedder() is True

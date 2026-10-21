"""Deployment-configuration regressions found in the server-readiness audit (no database needed)."""

import logging
import logging.handlers
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.config import environment_selector as es
from backend.utils import logging_setup

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── DATABASE_URL built by docker-compose: the RAW password is interpolated into it ─────────────────


@pytest.fixture()
def selector(monkeypatch):
    """An EnvironmentSelector whose reachability check records the URLs it is asked about."""
    sel = es.EnvironmentSelector()
    monkeypatch.setenv("AIRAWAT_API_KEY", "")
    monkeypatch.setattr(sel, "_check_airawat", lambda *a, **k: True)
    return sel


def _settings(**overrides):
    base = dict(DATABASE_URL="", DB_URL="", DB_TYPE="postgresql", DB_CLIENT="postgresql", DB_HOST="", DB_PORT=5432,
                DB_DATABASE="", DB_NAME="", DB_USER="", DB_USERNAME="", DB_PASSWORD="", DB_PASS="", DB_SSL=False,
                DB_SSLMODE="", dynamic_database_url="")
    base.update(overrides)
    return SimpleNamespace(**base)


def test_generic_database_url_credentials_are_percent_encoded(selector, monkeypatch):
    """Regression: 'p@ss#w/rd' in the raw URL made the app report 'No database is available'."""
    import backend.config.config as config

    monkeypatch.setattr(config, "settings", _settings(DATABASE_URL="postgresql://app:p@ss#w/rd!:1@db:5432/meet"))
    tried = []
    monkeypatch.setattr(selector, "_check_database", lambda url, label="": tried.append(url) or True)

    cfg = selector._try_generic()

    assert cfg is not None and cfg.environment == "generic"
    assert tried[0] == "postgresql+psycopg2://app:p%40ss%23w%2Frd%21%3A1@db:5432/meet"
    assert cfg.database_url == tried[0]


def test_generic_database_url_falls_back_to_the_discrete_variables(selector, monkeypatch):
    """A password that itself contains '%41' is ambiguous inside a URL; DB_PASSWORD carries it verbatim."""
    import backend.config.config as config

    monkeypatch.setattr(config, "settings", _settings(
        DATABASE_URL="postgresql://app:pw%41x@db:5432/meet", DB_HOST="db", DB_DATABASE="meet", DB_USER="app",
        DB_PASSWORD="pw%41x"))
    tried = []
    exact = "postgresql+psycopg2://app:pw%2541x@db:5432/meet"
    monkeypatch.setattr(selector, "_check_database", lambda url, label="": tried.append(url) or url == exact)

    cfg = selector._try_generic()

    assert cfg is not None and cfg.database_url == exact
    assert len(tried) == 2 and tried[1] == exact


def test_generic_database_that_is_unreachable_is_not_selected(selector, monkeypatch):
    import backend.config.config as config

    monkeypatch.setattr(config, "settings", _settings(DATABASE_URL="postgresql://app:pw@db:5432/meet"))
    monkeypatch.setattr(selector, "_check_database", lambda url, label="": False)
    assert selector._try_generic() is None


# ── logging must never stop the API from starting ─────────────────────────────


def test_an_unwritable_log_directory_degrades_to_console_logging(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a file where the log directory should be")
    monkeypatch.setattr(logging_setup, "_LOG_DIR", str(blocker / "logs"))  # makedirs -> OSError on every OS

    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers = []  # configure_logging() is idempotent and would return early under pytest's handlers
    try:
        logging_setup.configure_logging("INFO", "text")  # must not raise
        added = list(root.handlers)
        assert any(type(h) is logging.StreamHandler for h in added), "the console handler must still be installed"
        assert not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in added)
        assert "File logging DISABLED" in capsys.readouterr().out
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        root.handlers = saved_handlers
        root.setLevel(saved_level)


# ── shipped templates ─────────────────────────────────────────────────────────


def _dotenv_values(path: Path) -> dict:
    dotenv = pytest.importorskip("dotenv")
    return dotenv.dotenv_values(path)


def test_env_example_ships_a_working_vector_store_path():
    """Regression: '<configure-for-environment>' broke `source .env` and switched the vector store off."""
    value = _dotenv_values(REPO_ROOT / ".env.example")["CHROMA_PERSIST_DIRECTORY"]
    assert value and not re.search(r"<[^<>]*>", value)


def test_env_example_has_no_unquoted_values_a_shell_would_execute():
    """Anyone who `source`s the template (or a copy of it) must not run or choke on its values."""
    hostile = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=[^\"'#\n]*[ |&;()<>`]")
    offenders = [line.split("=", 1)[0] for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
                 if hostile.match(line)]
    assert offenders == []


def test_compose_passes_the_security_settings_to_the_container():
    """The container never sees .env, so a setting missing from compose can never be configured."""
    yaml = pytest.importorskip("yaml")
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    passed = {entry.split("=", 1)[0] for entry in compose["services"]["backend"]["environment"]}
    for name in ("ADMIN_API_KEY", "RATE_LIMIT_CHAT_PER_MIN", "RATE_LIMIT_UPLOAD_PER_MIN", "TRUST_FORWARDED_FOR",
                 "YTDLP_ALLOW_ANY_HOST", "SECRET_KEY", "HF_TOKEN"):
        assert name in passed, f"docker-compose.yml does not pass {name} to the backend container"


def test_compose_does_not_publish_postgres_on_every_interface():
    yaml = pytest.importorskip("yaml")
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    for mapping in compose["services"]["postgres"].get("ports", []):
        assert str(mapping).startswith("127.0.0.1:"), f"PostgreSQL is published beyond loopback: {mapping}"


def test_deployment_guide_has_no_broken_code_fences():
    """Regression: every fence used to close as '```n' (a PowerShell backtick-n leftover)."""
    text = (REPO_ROOT / "LINUX_DEPLOYMENT.md").read_text(encoding="utf-8")
    assert "```n" not in text and "backup_\\." not in text
    assert text.count("```") % 2 == 0


def test_launcher_no_longer_sources_the_env_file():
    """`source .env` executes dotenv values (spaces, '|', '<...>') and aborted the launcher under `set -e`."""
    script = (REPO_ROOT / "run_backend.sh").read_text(encoding="utf-8")
    assert not re.search(r"^\s*source\s+\.env\b", script, re.MULTILINE)
    assert "dotenv_values" in script


def test_launcher_env_loader_survives_hostile_values(tmp_path):
    import shutil
    import subprocess

    candidates = [shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"]
    bash = None
    for c in candidates:
        if c and os.path.exists(c):
            try:
                res = subprocess.run([c, "-c", "echo ok"], capture_output=True, text=True, timeout=5)
                if res.returncode == 0 and res.stdout.strip() == "ok":
                    bash = c
                    break
            except Exception:
                continue
    if not bash:
        pytest.skip("bash not available")
    # The launcher runs inside the activated venv, where `python` is this interpreter.
    path = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    if not shutil.which("python", path=path):
        pytest.skip("no `python` executable next to the test interpreter")
    (tmp_path / ".env").write_text(
        "REPORT_FOOTER_TEXT=Acme Corp | Powered by X\nCHROMA_PERSIST_DIRECTORY=<path>\nSAFE=1\n", encoding="utf-8")
    script = (REPO_ROOT / "run_backend.sh").read_text(encoding="utf-8")
    block = re.search(r'^if \[ -f "\.env" \]; then\n(.*?)^else\n', script, re.DOTALL | re.MULTILINE).group(1)
    probe = f'set -e\ncd "{tmp_path.as_posix()}"\nif true; then\n{block}fi\necho "[$REPORT_FOOTER_TEXT][$SAFE]"\n'
    out = subprocess.run([bash, "-c", probe], capture_output=True, text=True, timeout=60,
                         env={**os.environ, "PATH": path})
    assert out.returncode == 0, out.stderr
    assert "[Acme Corp | Powered by X][1]" in out.stdout

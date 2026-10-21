"""Regression tests: fail-closed encryption, DB-URL handling, CWD-independent .env lookup."""

import importlib

import pytest
from cryptography.fernet import Fernet


@pytest.fixture()
def crypto(monkeypatch, tmp_path):
    """A fresh backend.utils.crypto module with no cached key and an isolated key file."""
    from backend.utils import crypto as crypto_module

    monkeypatch.setattr(crypto_module, "KEY_FILE", str(tmp_path / ".secret_key"))
    monkeypatch.setattr(crypto_module, "_fernet_instance", None)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    yield crypto_module
    monkeypatch.setattr(crypto_module, "_fernet_instance", None)


def test_encrypt_round_trips_with_a_valid_fernet_key(crypto, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", Fernet.generate_key().decode())
    token = crypto.encrypt("api-key-123")
    assert token != "api-key-123"
    assert crypto.decrypt(token) == "api-key-123"


def test_free_form_secret_key_is_derived_not_rejected(crypto, monkeypatch):
    """`openssl rand -hex 32` style values are not valid Fernet keys; that used to break encrypt()."""
    monkeypatch.setenv("SECRET_KEY", "0123456789abcdef" * 4)
    token = crypto.encrypt("hello")
    assert crypto.decrypt(token) == "hello"


def test_same_free_form_secret_gives_the_same_key_across_restarts(crypto, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "a passphrase that is not a fernet key")
    token = crypto.encrypt("persisted")
    monkeypatch.setattr(crypto, "_fernet_instance", None)  # simulate a process restart
    assert crypto.decrypt(token) == "persisted"


def test_encrypt_fails_closed_instead_of_returning_plaintext(crypto, monkeypatch):
    """Regression: encrypt() returned the INPUT on error, so API keys were stored unencrypted."""

    def boom():
        raise RuntimeError("no key")

    monkeypatch.setattr(crypto, "_get_fernet", boom)
    with pytest.raises(RuntimeError):
        crypto.encrypt("super-secret")


def test_key_file_is_created_once_and_reused(crypto, tmp_path):
    first = crypto.encrypt("x")
    key_file = tmp_path / ".secret_key"
    assert key_file.exists()
    created = key_file.read_bytes()
    crypto._fernet_instance = None
    assert crypto.decrypt(first) == "x"
    assert key_file.read_bytes() == created, "the existing key must never be overwritten"


def test_decrypt_of_garbage_returns_empty_string(crypto, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", Fernet.generate_key().decode())
    assert crypto.decrypt("not-a-token") == ""


# ── database URLs ─────────────────────────────────────────────────────────────


def test_bare_postgresql_url_gets_an_explicit_psycopg2_driver():
    """SQLAlchemy 2.1 defaults `postgresql://` to psycopg (v3), which this project does not install."""
    from backend.config.config import with_psycopg2_driver

    assert with_psycopg2_driver("postgresql://u:p@h:5432/db") == "postgresql+psycopg2://u:p@h:5432/db"
    assert with_psycopg2_driver("postgres://u@h/db") == "postgresql+psycopg2://u@h/db"
    assert with_psycopg2_driver("postgresql+psycopg://u@h/db") == "postgresql+psycopg://u@h/db"
    assert with_psycopg2_driver("") == ""


def test_build_database_url_encodes_special_characters_and_pins_driver():
    from sqlalchemy.engine import make_url

    from backend.config.config import build_database_url

    url = build_database_url(db_host="db.example", db_database="meet", db_user="testuser", db_password="Pa@@ss12 3#")
    parsed = make_url(url)
    assert parsed.drivername == "postgresql+psycopg2"
    assert parsed.password == "Pa@@ss12 3#", "password must survive a round trip (space is %20, not '+')"
    assert parsed.host == "db.example"


def test_hand_written_url_with_at_sign_in_password_is_normalised():
    from sqlalchemy.engine import make_url

    from backend.config.config import build_database_url

    fixed = build_database_url(database_url="postgresql://testuser:Pa@@ss123456@10.0.0.5:5432/mydb")
    parsed = make_url(fixed)
    assert (parsed.username, parsed.password, parsed.host, parsed.database) == ("testuser", "Pa@@ss123456", "10.0.0.5", "mydb")


def test_url_normalisation_is_idempotent():
    from backend.config.config import build_database_url

    once = build_database_url(database_url="postgresql://u:p%40ss@h:5432/db")
    assert build_database_url(database_url=once) == once


def test_mask_database_url_hides_the_password():
    from backend.config.config import mask_database_url

    masked = mask_database_url("postgresql+psycopg2://dbadmin:s3cret@host:5432/db")
    assert "s3cret" not in masked and "dbadmin" in masked


# ── .env discovery ────────────────────────────────────────────────────────────


def test_find_env_file_is_anchored_to_the_project_root_not_the_cwd(monkeypatch, tmp_path):
    from backend.config import config as cfg

    (tmp_path / ".env").write_text("X=1\n")
    monkeypatch.setattr(cfg, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("ENV_FILE", raising=False)
    elsewhere = tmp_path / "somewhere" / "else"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)  # what systemd / PM2 do: start the process in another directory
    assert cfg.find_env_file() == str(tmp_path / ".env")


def test_env_file_override_is_honoured(monkeypatch, tmp_path):
    from backend.config import config as cfg

    custom = tmp_path / "custom.env"
    custom.write_text("Y=2\n")
    monkeypatch.setenv("ENV_FILE", str(custom))
    assert cfg.find_env_file() == str(custom)


def test_module_import_keeps_working():
    importlib.import_module("backend.config.environment_selector")

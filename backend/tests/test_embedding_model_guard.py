"""The embedding-model deployment guard: a Git LFS pointer is detected, fetched with `git lfs pull` when possible,
verified byte-for-byte (size + SHA-256 from the repository's pointer), and otherwise reported with the exact fix.
No real network or git is used here: the git calls are replaced; the real model directory is used for the happy path."""

import hashlib
import os
import shutil

import pytest

from backend.services import embedding_model_guard as guard

REAL_DIR = os.path.join(guard.PROJECT_ROOT, "models", "embeddings", "all-MiniLM-L6-v2")
WEIGHTS = b"not really a model, but the bytes whose checksum the pointer pins" * 100


def _pointer(data: bytes) -> str:
    return (f"version https://git-lfs.github.com/spec/v1\noid sha256:{hashlib.sha256(data).hexdigest()}\n"
            f"size {len(data)}\n")


@pytest.fixture()
def model_dir(tmp_path, monkeypatch):
    """A project checkout in tmp_path whose model directory holds the LFS POINTER (what a plain clone ships)."""
    d = tmp_path / "models" / "embeddings" / "all-MiniLM-L6-v2"
    (d / "1_Pooling").mkdir(parents=True)
    for name in guard.REQUIRED_FILES:
        (d / name).write_text("{}")
    (d / "model.safetensors").write_text(_pointer(WEIGHTS))
    monkeypatch.setattr(guard, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("EMBEDDING_MODEL_PATH", raising=False)
    monkeypatch.setattr(guard, "_expected_from_git", lambda _w: None)
    return d


def test_a_git_lfs_pointer_is_detected_and_never_accepted(model_dir, monkeypatch):
    monkeypatch.setattr(guard, "_git_lfs_available", lambda: False)
    check = guard.ensure_embedding_model(auto_fetch=True)
    assert not check.ok and "Git LFS pointer" in check.problem
    assert check.expected_bytes == len(WEIGHTS)


def test_the_pointer_is_replaced_by_git_lfs_pull_and_verified(model_dir, monkeypatch):
    calls = []

    def fake_pull(check):
        calls.append(check.model_dir)
        (model_dir / "model.safetensors").write_bytes(WEIGHTS)          # what git lfs pull writes
        return True, ""

    monkeypatch.setattr(guard, "_git_lfs_available", lambda: True)
    monkeypatch.setattr(guard, "_git_lfs_pull", fake_pull)
    check = guard.ensure_embedding_model(auto_fetch=True)
    assert check.ok and check.fetched_with_git_lfs and check.sha256_verified
    assert calls == [str(model_dir)]
    assert guard.LAST_CHECK is check


def test_a_pulled_file_with_the_wrong_content_is_rejected(model_dir, monkeypatch):
    monkeypatch.setattr(guard, "_git_lfs_available", lambda: True)
    monkeypatch.setattr(guard, "_git_lfs_pull",
                        lambda c: ((model_dir / "model.safetensors").write_bytes(b"x" * len(WEIGHTS)), (True, ""))[1])
    check = guard.ensure_embedding_model(auto_fetch=True)
    assert not check.ok and "SHA-256 does not match" in check.problem


def test_an_unreachable_lfs_remote_gives_a_clear_error(model_dir, monkeypatch):
    monkeypatch.setattr(guard, "_is_git_checkout", lambda: True)
    monkeypatch.setattr(guard, "_git_lfs_available", lambda: True)
    monkeypatch.setattr(guard, "_git_lfs_pull", lambda c: (False, "batch response: Post ... connection refused"))
    check = guard.ensure_embedding_model(auto_fetch=True)
    assert not check.ok and "Git LFS pointer" in check.problem
    assert "git lfs pull` from the repository's LFS remote failed" in check.fix and "connection refused" in check.fix


def test_missing_git_lfs_says_how_to_install_it(model_dir, monkeypatch):
    monkeypatch.setattr(guard, "_is_git_checkout", lambda: True)
    monkeypatch.setattr(guard, "_git_lfs_available", lambda: False)
    check = guard.ensure_embedding_model(auto_fetch=True)
    assert not check.ok and "git-lfs is not installed" in check.fix and "apt-get install -y git-lfs" in check.fix


def test_an_archive_deployment_without_git_is_told_to_ship_the_real_file(model_dir, monkeypatch):
    monkeypatch.setattr(guard, "_is_git_checkout", lambda: False)
    monkeypatch.setattr(guard, "_git_lfs_available", lambda: False)
    check = guard.ensure_embedding_model(auto_fetch=True)
    assert not check.ok and "not a git checkout" in check.fix and "deploy_model.sh" in check.fix


def test_no_fetch_is_attempted_when_disabled(model_dir, monkeypatch):
    monkeypatch.setattr(guard, "_git_lfs_available", lambda: True)
    monkeypatch.setattr(guard, "_git_lfs_pull", lambda c: pytest.fail("must not pull"))
    assert not guard.ensure_embedding_model(auto_fetch=False).ok


def test_a_truncated_copy_is_rejected(model_dir, monkeypatch):
    monkeypatch.setattr(guard, "_expected_from_git", lambda _w: (hashlib.sha256(WEIGHTS).hexdigest(), len(WEIGHTS)))
    (model_dir / "model.safetensors").write_bytes(WEIGHTS[:100])
    check = guard.ensure_embedding_model(auto_fetch=False)
    assert not check.ok and "truncated" in check.problem


def test_missing_config_files_are_named(model_dir):
    os.remove(model_dir / "tokenizer.json")
    check = guard.ensure_embedding_model(auto_fetch=False)
    assert not check.ok and "tokenizer.json" in check.problem


@pytest.mark.skipif(guard.read_lfs_pointer(os.path.join(REAL_DIR, "model.safetensors")) is not None
                    or not os.path.isfile(os.path.join(REAL_DIR, "model.safetensors")),
                    reason="this checkout holds the LFS pointer, not the real weights")
def test_the_real_repository_model_passes_with_its_pinned_checksum(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", REAL_DIR)
    check = guard.ensure_embedding_model(auto_fetch=False)
    assert check.ok and check.sha256_verified and check.weights_bytes == 90868376


def test_the_cli_fails_with_exit_code_1_on_a_pointer(model_dir, monkeypatch, capsys):
    from backend.scripts import ensure_embedding_model as cli

    monkeypatch.setattr(guard, "_git_lfs_available", lambda: False)
    assert cli.main(["--no-fetch"]) == 1
    out = capsys.readouterr().out
    assert "[ERROR]" in out and "FIX:" in out


def test_the_shared_pinned_checksum_is_used_by_fetch_models():
    from backend.scripts import fetch_models

    assert (fetch_models.EMBEDDING_WEIGHTS_SHA256, fetch_models.EMBEDDING_WEIGHTS_SIZE) == \
        guard.PINNED_WEIGHTS[guard.DEFAULT_MODEL_NAME]


def test_nothing_is_copied_or_created_in_place_of_the_model(model_dir, monkeypatch):
    monkeypatch.setattr(guard, "_git_lfs_available", lambda: False)
    before = sorted(os.listdir(model_dir))
    guard.ensure_embedding_model(auto_fetch=True)
    assert sorted(os.listdir(model_dir)) == before
    assert (model_dir / "model.safetensors").read_text().startswith("version https://git-lfs")
    shutil.rmtree(model_dir)                                              # and a missing directory is reported
    assert "does not exist" in guard.ensure_embedding_model(auto_fetch=True).problem

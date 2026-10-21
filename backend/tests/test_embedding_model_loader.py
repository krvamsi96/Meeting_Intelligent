"""The embedding model loader: broken model files are detected up front, the fallback is meaningful, nothing is silent.

Background: the weights are tracked with Git LFS, so a repository cloned without ``git lfs pull`` holds a ~130-byte
text pointer named ``model.safetensors``. safetensors reads its first 8 bytes as a header length and fails with
"Error while deserializing header: header too large", and the platform used to fall back to SHA-256 noise that was
stored in ChromaDB. These tests build real (tiny) safetensors files and real pointer files - nothing about the file
checks is mocked; only the search paths and the (heavy) transformer libraries are substituted.
"""

import json
import logging
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from backend.services import model_loader as ml

REPO_ROOT = Path(__file__).resolve().parents[2]
LFS_POINTER = (b"version https://git-lfs.github.com/spec/v1\n"
               b"oid sha256:53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db\n"
               b"size 90868376\n")


def write_safetensors(path: Path, payload_bytes: int = 64) -> None:
    """A structurally valid safetensors file: 8-byte header length, JSON header, tensor bytes."""
    header = json.dumps({"w": {"dtype": "F32", "shape": [payload_bytes // 4], "data_offsets": [0, payload_bytes]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * payload_bytes)


def make_model_dir(root: Path, name: str = "model", *, weights: str = "valid") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / "vocab.txt").write_text("[PAD]\n[UNK]\n")
    target = directory / "model.safetensors"
    if weights == "valid":
        write_safetensors(target)
    elif weights == "lfs_pointer":
        target.write_bytes(LFS_POINTER)
    elif weights == "truncated":
        write_safetensors(target)
        target.write_bytes(target.read_bytes()[:-20])
    elif weights == "garbage":
        target.write_bytes(b"this is not a safetensors file, just some long text " * 5)
    elif weights == "empty":
        target.write_bytes(b"")
    elif weights == "none":
        pass
    return directory


@pytest.fixture()
def fresh_loader(monkeypatch):
    """Isolated loader state (the real one is a process-wide singleton) and no real transformer library loaded."""
    monkeypatch.setattr(ml, "_embedding_encoder", None)
    monkeypatch.setattr(ml, "_embedding_failure", None)
    monkeypatch.setattr(ml, "_engine_info", {})

    class _MustNotLoad:
        def __getattr__(self, name):
            raise AssertionError("a model that failed validation must not be handed to sentence-transformers")

    monkeypatch.setitem(sys.modules, "sentence_transformers", _MustNotLoad())
    return monkeypatch


# ── model file inspection ─────────────────────────────────────────────────────────────────────────────────────────
def test_a_valid_model_directory_passes(tmp_path):
    assert ml.inspect_embedding_model_dir(str(make_model_dir(tmp_path))) is None


def test_a_git_lfs_pointer_in_place_of_the_weights_is_detected_and_explained(tmp_path):
    problem = ml.inspect_embedding_model_dir(str(make_model_dir(tmp_path, weights="lfs_pointer")))
    assert "Git LFS pointer" in problem and "model.safetensors" in problem


def test_any_lfs_pointer_in_the_model_directory_is_detected_not_only_the_weights(tmp_path):
    directory = make_model_dir(tmp_path)                                 # weights are fine ...
    (directory / "vocab.txt").write_bytes(LFS_POINTER)                   # ... but the tokenizer file is a pointer
    problem = ml.inspect_embedding_model_dir(str(directory))
    assert "Git LFS pointer" in problem and "vocab.txt" in problem and "model.safetensors" not in problem


def test_the_real_size_a_pointer_stands_for_is_read_from_the_pointer(tmp_path):
    pointer = tmp_path / "p.bin"
    pointer.write_bytes(LFS_POINTER)
    assert ml._lfs_pointer_size(str(pointer)) == 90868376
    real = tmp_path / "real.bin"
    real.write_bytes(b"x" * 4096)
    assert ml._lfs_pointer_size(str(real)) is None


@pytest.mark.parametrize("kind, expected", [
    ("truncated", "incomplete"),
    ("garbage", "invalid header"),
    ("empty", "empty or truncated"),
    ("none", "no weights file"),
])
def test_corrupt_incomplete_or_missing_weights_are_reported_with_the_reason(tmp_path, kind, expected):
    assert expected in ml.inspect_embedding_model_dir(str(make_model_dir(tmp_path, weights=kind)))


def test_a_directory_without_a_config_is_reported(tmp_path):
    directory = make_model_dir(tmp_path)
    (directory / "config.json").unlink()
    assert "config.json" in ml.inspect_embedding_model_dir(str(directory))


def test_the_check_is_the_same_error_safetensors_itself_raises(tmp_path):
    """Guards the diagnosis: the pointer file really produces the failure from the server log."""
    safetensors = pytest.importorskip("safetensors")
    pointer = tmp_path / "model.safetensors"
    pointer.write_bytes(LFS_POINTER)
    with pytest.raises(Exception, match="header too large"):
        safetensors.safe_open(str(pointer), framework="np")
    assert "Git LFS pointer" in ml.inspect_embedding_model_dir(str(make_model_dir(tmp_path / "d", weights="lfs_pointer")))


# ── choosing a model directory ────────────────────────────────────────────────────────────────────────────────────
def test_a_broken_copy_is_skipped_and_the_next_valid_copy_is_used(tmp_path, monkeypatch):
    broken = make_model_dir(tmp_path, "project", weights="lfs_pointer")
    good = make_model_dir(tmp_path, "cache")
    monkeypatch.setattr(ml, "get_embedding_model_search_paths", lambda: [str(tmp_path / "missing"), str(broken), str(good)])
    assert ml.resolve_local_embedding_model_path() == str(good)


def test_no_usable_copy_resolves_to_none(tmp_path, monkeypatch):
    broken = make_model_dir(tmp_path, "project", weights="lfs_pointer")
    monkeypatch.setattr(ml, "get_embedding_model_search_paths", lambda: [str(broken)])
    assert ml.resolve_local_embedding_model_path() is None


def test_a_valid_cached_snapshot_is_offered_as_a_last_resort_location(tmp_path, monkeypatch):
    snapshot = tmp_path / "hub" / "models--sentence-transformers--my-model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path / "none"))
    assert str(snapshot) in ml._hf_cache_snapshot_dirs("my-model")
    assert ml._hf_cache_snapshot_dirs("another-model") == []


# ── the loader never goes silent - and never fakes embeddings ────────────────────────────────────────────────────
def test_a_broken_model_is_reported_with_why_and_how_to_fix_it_and_is_never_faked(tmp_path, fresh_loader, caplog):
    broken = make_model_dir(tmp_path, "project", weights="lfs_pointer")
    fresh_loader.setattr(ml, "get_embedding_model_search_paths", lambda: [str(broken)])
    with caplog.at_level(logging.INFO, logger=ml.logger.name):
        with pytest.raises(ml.EmbeddingModelUnavailable) as raised:
            ml.get_embedding_encoder()
    assert "Git LFS pointer" in str(raised.value) and "git lfs pull" in str(raised.value)

    info = ml.get_embedding_engine_info()                    # a diagnostic: never raises
    assert info["engine"] == ml.UNAVAILABLE_ENGINE and info["signature"] is None
    assert "Git LFS pointer" in info["reason"] and "model.safetensors" in info["reason"]
    assert "git lfs pull" in info["fix"] and "restart" in info["fix"]
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("UNAVAILABLE" in e and "FIX:" in e and "no fake vectors" in e and "git lfs pull" in e for e in errors)

    with pytest.raises(ml.EmbeddingModelUnavailable):        # decided once per process, not re-scanned per call
        ml.get_embedding_encoder()
    assert ml.encode_texts(["a b c"]) is None                # lenient callers: no embeddings, nothing invented
    with pytest.raises(ml.EmbeddingModelUnavailable):        # the vector store must not store anything
        ml.encode_texts(["a b c"], strict=True)


def test_the_lexical_stand_in_needs_an_explicit_opt_in_and_says_why(tmp_path, fresh_loader, caplog):
    fresh_loader.setattr(ml.settings, "EMBEDDING_ALLOW_LEXICAL_FALLBACK", True)
    broken = make_model_dir(tmp_path, "project", weights="lfs_pointer")
    fresh_loader.setattr(ml, "get_embedding_model_search_paths", lambda: [str(broken)])
    with caplog.at_level(logging.INFO, logger=ml.logger.name):
        engine, model = ml.get_embedding_encoder()
    assert (engine, model) == (ml.LEXICAL_ENGINE, None)

    info = ml.get_embedding_engine_info()
    assert info["engine"] == ml.LEXICAL_ENGINE
    assert "Git LFS pointer" in info["reason"] and "model.safetensors" in info["reason"]
    assert "git lfs pull" in info["fix"] and "restart" in info["fix"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Embedding engine: lexical_fallback" in w and "FIX:" in w and "git lfs pull" in w for w in warnings)
    assert ml.get_embedding_signature().startswith("lexical_fallback-v")


def test_a_missing_model_is_reported_with_the_locations_searched(tmp_path, fresh_loader):
    fresh_loader.setattr(ml, "get_embedding_model_search_paths", lambda: [str(tmp_path / "nowhere")])
    info = ml.get_embedding_engine_info()
    assert info["engine"] == ml.UNAVAILABLE_ENGINE and "not found" in info["reason"] and "nowhere" in info["reason"]
    assert "EMBEDDING_MODEL_PATH" in info["fix"]


def test_health_reports_the_model_unavailable_with_the_problem(tmp_path, fresh_loader):
    fresh_loader.setattr(ml, "get_embedding_model_search_paths",
                         lambda: [str(make_model_dir(tmp_path, "project", weights="lfs_pointer"))])
    health = ml.get_all_models_health()
    embedding = health["models"]["embedding"]
    assert health["status"] == "unavailable" and embedding["active_engine"] == ml.UNAVAILABLE_ENGINE
    assert embedding["is_local_available"] is False and "Git LFS pointer" in embedding["problem"] and embedding["fix"]


def test_health_reports_degraded_while_on_the_opt_in_lexical_stand_in(tmp_path, fresh_loader):
    fresh_loader.setattr(ml.settings, "EMBEDDING_ALLOW_LEXICAL_FALLBACK", True)
    fresh_loader.setattr(ml, "get_embedding_model_search_paths",
                         lambda: [str(make_model_dir(tmp_path, "project", weights="lfs_pointer"))])
    health = ml.get_all_models_health()
    embedding = health["models"]["embedding"]
    assert health["status"] == "degraded" and embedding["active_engine"] == ml.LEXICAL_ENGINE
    assert embedding["is_local_available"] is False and "Git LFS pointer" in embedding["problem"] and embedding["fix"]


def test_health_is_healthy_and_has_no_problem_field_with_a_working_engine(fresh_loader):
    fresh_loader.setattr(ml, "_embedding_encoder", ("sentence_transformers", object()))
    fresh_loader.setattr(ml, "_engine_info", {"engine": "sentence_transformers", "model_path": "/x", "reason": "", "fix": ""})
    health = ml.get_all_models_health()
    assert health["status"] == "healthy" and "problem" not in health["models"]["embedding"]
    assert ml.get_embedding_signature() == f"{ml._model_name()}:{ml.get_embedding_dimension()}"


def test_both_real_engines_share_one_signature_and_the_fallback_differs(fresh_loader):
    fresh_loader.setattr(ml, "_embedding_encoder", ("sentence_transformers", object()))
    st_signature = ml.get_embedding_signature()
    fresh_loader.setattr(ml, "_embedding_encoder", ("transformers", (object(), object())))
    assert ml.get_embedding_signature() == st_signature
    fresh_loader.setattr(ml, "_embedding_encoder", (ml.LEXICAL_ENGINE, None))
    assert ml.get_embedding_signature() != st_signature


def test_the_model_is_loaded_once_even_when_threads_race(tmp_path, fresh_loader):
    import threading

    calls = []
    fresh_loader.setattr(ml, "_load_embedding_encoder", lambda: calls.append(1) or (ml.LEXICAL_ENGINE, None))
    threads = [threading.Thread(target=ml.get_embedding_encoder) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert calls == [1]


# ── strict vs. lenient encoding ───────────────────────────────────────────────────────────────────────────────────
class _FailingModel:
    def encode(self, *args, **kwargs):
        raise RuntimeError("out of memory")


def test_a_model_failure_gives_lenient_callers_no_vectors_and_the_vector_store_the_error(fresh_loader):
    fresh_loader.setattr(ml, "_embedding_encoder", ("sentence_transformers", _FailingModel()))
    assert ml.encode_texts(["a b c", "d e f"]) is None       # nothing invented for evaluation callers
    with pytest.raises(RuntimeError):
        ml.encode_texts(["a b c"], strict=True)


def test_a_model_failure_can_opt_in_to_lexical_vectors_for_lenient_callers(fresh_loader):
    fresh_loader.setattr(ml.settings, "EMBEDDING_ALLOW_LEXICAL_FALLBACK", True)
    fresh_loader.setattr(ml, "_embedding_encoder", ("sentence_transformers", _FailingModel()))
    lenient = ml.encode_texts(["a b c", "d e f"])
    assert lenient.shape == (2, ml.get_embedding_dimension())
    with pytest.raises(RuntimeError):
        ml.encode_texts(["a b c"], strict=True)


class _FixedOutputModel:
    def __init__(self, array):
        self.array = array

    def encode(self, texts, **kwargs):
        return self.array


@pytest.mark.parametrize("bad", [
    np.full((1, 384), np.nan, dtype=np.float32),            # NaN
    np.full((1, 384), np.inf, dtype=np.float32),            # infinite
    np.ones((1, 10), dtype=np.float32),                     # wrong dimension
    np.ones((2, 384), dtype=np.float32),                    # wrong number of rows
])
def test_vectors_that_are_not_finite_or_the_wrong_shape_are_rejected(fresh_loader, bad):
    fresh_loader.setattr(ml, "_embedding_encoder", ("sentence_transformers", _FixedOutputModel(bad)))
    assert ml.encode_texts(["x"]) is None
    with pytest.raises(ml.EmbeddingOutputError):
        ml.encode_texts(["x"], strict=True)


def test_the_self_test_fails_for_vectors_that_are_not_semantic(fresh_loader):
    """A hash / random / constant vector cannot pass: related sentences must be closer than unrelated ones."""
    constant = np.tile(np.eye(1, 384, dtype=np.float32), (3, 1))          # every sentence -> the same unit vector
    fresh_loader.setattr(ml, "_embedding_encoder", ("sentence_transformers", _FixedOutputModel(constant)))
    fresh_loader.setattr(ml, "_engine_info", {"engine": "sentence_transformers", "model_path": "/x"})
    report = ml.validate_embedding_model()
    assert report["ok"] is False and report["semantic"] is False and report["finite"] and report["unit_norm"]


def test_the_real_model_passes_the_self_test():
    """End to end on the shipped weights: loads, 384-d, finite, unit length, and genuinely semantic."""
    directory = REPO_ROOT / "models" / "embeddings" / "all-MiniLM-L6-v2"
    problem = ml.inspect_embedding_model_dir(str(directory))
    if problem:
        pytest.skip(f"the shipped model is not usable in this checkout ({problem}) - the deployment problem this guards")
    monkey = pytest.MonkeyPatch()
    monkey.setattr(ml, "_embedding_encoder", None)
    monkey.setattr(ml, "_embedding_failure", None)
    monkey.setattr(ml, "_engine_info", {})
    try:
        report = ml.validate_embedding_model()
    finally:
        monkey.undo()
    assert report["ok"] is True, report
    assert report["dimension"] == 384 and report["finite"] and report["unit_norm"] and report["semantic"]
    assert report["similarity_related"] > report["similarity_unrelated"] + 0.1


# ── the lexical fallback embedder ─────────────────────────────────────────────────────────────────────────────────
def test_lexical_vectors_are_unit_length_float32_and_the_right_shape():
    vecs = ml.lexical_embed(["database migration", "", "   ", "!!! ???", "x"])
    assert vecs.shape == (5, ml.get_embedding_dimension()) and vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)          # empty / punctuation-only text stays valid


def test_lexical_vectors_are_a_pure_function_of_the_text():
    a = ml.lexical_embed(["The migration must finish before March."])
    b = ml.lexical_embed(["The migration must finish before March."])
    assert np.array_equal(a, b)


def test_lexical_vectors_do_not_depend_on_python_hash_randomisation():
    """Stored vectors must match the ones computed for queries after a restart (Python's hash() differs per process)."""
    code = ("import numpy as np; from backend.services.model_loader import lexical_embed; "
            "v = lexical_embed(['budget approved for the migration project']); print(round(float(v[0][:8].sum()), 5))")
    outputs = set()
    for seed in ("1", "2", "random"):
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONHASHSEED": seed, "PYTHONUTF8": "1"}
        proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr[-800:]
        outputs.add(proc.stdout.strip().splitlines()[-1])
    assert len(outputs) == 1


def _cos(a: str, b: str) -> float:
    va, vb = ml.lexical_embed([a, b])
    return float(va @ vb)


def test_texts_sharing_vocabulary_are_close_and_unrelated_texts_are_not():
    base = "The database migration to Postgres must finish before March"
    assert _cos(base, base) == pytest.approx(1.0, abs=1e-5)
    related = _cos(base, "Who owns the Postgres database migration deadline in March?")
    unrelated = _cos(base, "Autumn marketing campaign launches in October with influencer partnerships")
    assert related > 0.4 and unrelated < 0.15 and related > 3 * unrelated


def test_word_forms_carry_a_small_but_real_signal():
    """migrate/migration, databases/database share sub-word features: weak (the real model does much better) but
    clearly above the ~0 similarity of texts with nothing in common."""
    assert _cos("we will migrate the databases", "the database migration is planned") > 0.08
    assert abs(_cos("we will migrate the databases", "lunch menu for friday")) < 0.05


def test_the_old_hash_fallback_gave_no_similarity_at_all_for_related_texts():
    """Regression guard for the bug: the SHA-256 fallback returned unrelated noise even for near-identical texts."""
    import hashlib

    def old(text, dim=384):
        chunks = []
        for salt in range((dim // 32) + 1):
            chunks.extend([(b / 127.5) - 1.0 for b in hashlib.sha256(f"{salt}:{text}".encode()).digest()])
        v = np.array(chunks[:dim], dtype=np.float32)
        return v / np.linalg.norm(v)

    a, b = "database migration budget approved", "database migration budget approved."
    assert abs(float(old(a) @ old(b))) < 0.2 and _cos(a, b) > 0.9


def test_retrieval_over_a_small_corpus_puts_the_right_chunk_first():
    corpus = {
        "migration": "We agreed the database migration to Postgres must finish before March, rollback plan owned by Ravi.",
        "marketing": "The autumn marketing campaign launches in October; influencer budget was approved by the brand team.",
        "hiring": "Hiring two backend engineers: the recruiting pipeline and interview panel schedule were reviewed.",
        "security": "The security audit found expired certificates; renewal is due next week and must be tracked.",
    }
    queries = {
        "migration": "when does the postgres migration need to finish",
        "marketing": "what was approved for the influencer campaign",
        "hiring": "interview panel for the backend engineers",
        "security": "which certificates expired in the audit",
    }
    keys = list(corpus)
    chunk_vecs = ml.lexical_embed([corpus[k] for k in keys])
    for expected, question in queries.items():
        scores = chunk_vecs @ ml.lexical_embed([question])[0]
        assert keys[int(np.argmax(scores))] == expected

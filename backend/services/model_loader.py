"""Centralized Offline Model Loader & Registry
=============================================
Manages all local AI/ML models with strict offline enforcement:
- Local path resolution with environment variable overrides
- Zero runtime internet calls to Hugging Face or external CDNs
- Fail-fast validation of the model FILES (a Git LFS pointer, a truncated or corrupt weights file is detected
  up front, not after a cryptic "header too large" from safetensors) with informative, actionable logging
- Deterministic lexical (feature-hashing) fallback when no loadable model exists, so retrieval and similarity
  keep working (with reduced quality) instead of silently turning into noise
- An embedding "signature" naming the vector space in use, so vectors made by different engines are never mixed
- Health check inspection for all models
"""

import glob
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import struct
import threading
import time
from collections import Counter, OrderedDict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from backend.config.config import settings

logger = logging.getLogger(__name__)

# Enforce offline flags at the process level
if getattr(settings, "OFFLINE_MODE", True):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

# ─── Embedding Model Singletons ──────────────────────────────────────────────
LEXICAL_ENGINE = "lexical_fallback"
_LEXICAL_VERSION = 1            # bump when the lexical algorithm changes: it changes the vector space

_embedding_encoder: Optional[Tuple[str, Any]] = None
_embedding_failure: Optional[Exception] = None      # the model could not be loaded (and the fallback is not allowed)
_engine_info: Dict[str, Any] = {}
_load_lock = threading.Lock()
_embedding_dim: int = 384
UNAVAILABLE_ENGINE = "unavailable"


class EmbeddingModelUnavailable(RuntimeError):
    """No usable semantic embedding model. Vectors are never faked: the API reports it and refuses to embed.

    (``EMBEDDING_ALLOW_LEXICAL_FALLBACK=true`` opts in to deterministic word-hash vectors for offline development.)
    """


class EmbeddingOutputError(RuntimeError):
    """The model answered but the vectors are not usable (wrong shape / dimension, NaN or infinite values)."""

# Files that stand in for the real content when a repository is cloned without `git lfs pull`.
_LFS_POINTER_PREFIX = b"version https://git-lfs"
_LFS_POINTER_MAX_BYTES = 1024
_SAFETENSORS_MAX_HEADER = 100 * 1024 * 1024        # the limit safetensors itself enforces ("header too large")
_WEIGHT_FILE_PATTERNS = ("*.safetensors", "pytorch_model*.bin", "rust_model.ot", "tf_model.h5")


def _model_name() -> str:
    return getattr(settings, "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


def _hf_cache_snapshot_dirs(model_name: str) -> List[str]:
    """Snapshot directories of ``model_name`` in the local Hugging Face / sentence-transformers caches (newest first).

    Purely local: nothing is downloaded. Lets a valid cached copy stand in for a broken project copy.
    """
    home = os.path.expanduser("~")
    hf_home = os.environ.get("HF_HOME") or os.path.join(home, ".cache", "huggingface")
    hub_dirs = [os.environ.get("HUGGINGFACE_HUB_CACHE") or os.path.join(hf_home, "hub")]
    st_home = os.environ.get("SENTENCE_TRANSFORMERS_HOME") or os.path.join(home, ".cache", "torch", "sentence_transformers")
    found: List[str] = []
    for hub in hub_dirs + [st_home]:
        pattern = os.path.join(glob.escape(hub), f"models--*--{glob.escape(model_name)}", "snapshots", "*")
        found.extend(sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True))
    return found


def get_embedding_model_search_paths() -> List[str]:
    """Return ordered list of filesystem paths to look for local embedding models."""
    paths = []

    # 1. Explicitly configured path from environment
    if hasattr(settings, "EMBEDDING_MODEL_PATH") and settings.EMBEDDING_MODEL_PATH:
        paths.append(os.path.abspath(settings.EMBEDDING_MODEL_PATH))

    # 2. Path relative to MODEL_BASE_PATH
    base_path = getattr(settings, "MODEL_BASE_PATH", "./models")
    model_name = _model_name()
    paths.append(os.path.abspath(os.path.join(base_path, "embeddings", model_name)))
    paths.append(os.path.abspath(os.path.join(base_path, model_name)))

    # 3. Standard project locations
    paths.append(os.path.abspath(f"./models/embeddings/{model_name}"))
    paths.append(os.path.abspath(f"./models/{model_name}"))
    paths.append(os.path.abspath(f"/app/models/embeddings/{model_name}"))
    paths.append(os.path.abspath(f"/opt/ai_meeting_intelligence/models/embeddings/{model_name}"))

    # 4. Local Hugging Face / sentence-transformers cache snapshots (last resort, still offline)
    paths.extend(os.path.abspath(p) for p in _hf_cache_snapshot_dirs(model_name))

    # Deduplicate while preserving order
    seen = set()
    unique_paths = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    return unique_paths


def _lfs_pointer_size(path: str) -> Optional[int]:
    """If ``path`` is a Git LFS pointer file, the size (bytes) of the real file it stands for (0 when the pointer
    does not say); ``None`` when it is not a pointer. Pointers are ~130-byte text stubs starting with a fixed line."""
    try:
        if os.path.getsize(path) > _LFS_POINTER_MAX_BYTES:
            return None
        with open(path, "rb") as fh:
            head = fh.read(_LFS_POINTER_MAX_BYTES)
    except OSError:
        return None
    if not head.startswith(_LFS_POINTER_PREFIX):
        return None
    match = re.search(rb"^size (\d+)", head, re.MULTILINE)
    return int(match.group(1)) if match else 0


def _safetensors_problem(path: str) -> Optional[str]:
    """Why a ``.safetensors`` file cannot be loaded (checked without loading it); None when it looks complete.

    Format: 8-byte little-endian header length, a JSON header, then the tensor data the header describes.
    """
    size = os.path.getsize(path)
    if size < 8:
        return f"is only {size} byte(s) - empty or truncated"
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        if header_len > _SAFETENSORS_MAX_HEADER or 8 + header_len > size:
            return (f"has an invalid header (it declares {header_len} bytes in a {size}-byte file) - "
                    f"the file is corrupt or is not a safetensors file")
        try:
            header = json.loads(fh.read(header_len))
            data_end = max((int(t["data_offsets"][1]) for k, t in header.items() if k != "__metadata__"), default=0)
        except (ValueError, KeyError, TypeError, IndexError):
            return "has an unreadable header - the file is corrupt"
    if size < 8 + header_len + data_end:
        return f"is incomplete ({size} of {8 + header_len + data_end} bytes) - the copy or download was interrupted"
    return None


def inspect_embedding_model_dir(path: str) -> Optional[str]:
    """Why ``path`` does not hold a loadable embedding model (one line); ``None`` when it does.

    Deliberately inspects file CONTENT, not just presence: a repository cloned without Git LFS contains every
    tracked file (same name) as a tiny text pointer, which a mere ``os.path.isfile`` cannot tell from the model.
    """
    if not os.path.isdir(path):
        return "is not a directory"
    try:
        names = sorted(os.listdir(path))
    except OSError as exc:
        return f"cannot be read ({exc.strerror or exc})"

    pointers = [n for n in names if os.path.isfile(os.path.join(path, n)) and _lfs_pointer_size(os.path.join(path, n)) is not None]
    if pointers:
        return f"holds Git LFS pointer file(s) instead of the real content: {', '.join(pointers)}"
    if not any(n in names for n in ("config.json", "sentence_bert_config.json")):
        return "has no config.json / sentence_bert_config.json"

    weights = sorted({os.path.basename(p) for pattern in _WEIGHT_FILE_PATTERNS
                      for p in glob.glob(os.path.join(glob.escape(path), pattern))})
    if not weights:
        return "has no weights file (model.safetensors / pytorch_model.bin)"
    for name in weights:
        if name.endswith(".safetensors"):
            problem = _safetensors_problem(os.path.join(path, name))
            if problem:
                return f"{name} {problem}"
    return None


def _scan_local_embedding_models() -> Tuple[Optional[str], Dict[str, str]]:
    """First candidate directory holding a loadable model, plus ``{path: problem}`` for every broken one skipped."""
    problems: Dict[str, str] = {}
    for path in get_embedding_model_search_paths():
        if not os.path.isdir(path):
            continue
        problem = inspect_embedding_model_dir(path)
        if problem is None:
            return path, problems
        problems[path] = problem
    return None, problems


def resolve_local_embedding_model_path() -> Optional[str]:
    """Find the first directory containing a LOADABLE local embedding model (broken copies are skipped)."""
    return _scan_local_embedding_models()[0]


def _guard_status() -> Optional[Dict[str, Any]]:
    """The startup file verification (embedding_model_guard) of this process, or None when it has not run."""
    from backend.services import embedding_model_guard

    return embedding_model_guard.LAST_CHECK.as_dict() if embedding_model_guard.LAST_CHECK else None


def _fix_hint(problems: Dict[str, str]) -> str:
    name = _model_name()
    if any("Git LFS pointer" in p for p in problems.values()):
        return (f"The repository was cloned without the large files. Fetch them with `git lfs install && git lfs pull` "
                f"in the project directory (needs network access to the LFS remote), or copy models/embeddings/{name}/ "
                f"from a machine where it works (~91 MB, e.g. `bash deploy_model.sh`), then restart the API. "
                f"Vectors already stored are re-embedded automatically after the restart.")
    if problems:
        return (f"Replace the model directory with a complete copy of '{name}' (~91 MB, e.g. `bash deploy_model.sh`) "
                f"and restart the API. Vectors already stored are re-embedded automatically after the restart.")
    return (f"Copy models/embeddings/{name}/ (~91 MB) to the server (e.g. `bash deploy_model.sh`) or set "
            f"EMBEDDING_MODEL_PATH, then restart the API. Vectors already stored are re-embedded automatically.")


def _load_embedding_encoder() -> Tuple[str, Any]:
    """Load the best available engine (called once, under ``_load_lock``); records why in ``_engine_info``."""
    global _engine_info
    model_name = _model_name()
    local_path, problems = _scan_local_embedding_models()
    reason = ""

    if local_path:
        logger.info(f"[ModelLoader] Found local embedding model at: {local_path}")

        # Strategy 1: SentenceTransformers with local_files_only
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from sentence_transformers import SentenceTransformer

            # Pass absolute local path to SentenceTransformer
            model = SentenceTransformer(local_path, local_files_only=True)
            _engine_info = {"engine": "sentence_transformers", "model_path": local_path, "reason": "", "fix": ""}
            logger.info(f"[ModelLoader] Successfully loaded local SentenceTransformer from: {local_path}")
            return "sentence_transformers", model
        except Exception as st_err:
            reason = f"SentenceTransformer could not load it: {st_err}"
            logger.info(f"[ModelLoader] SentenceTransformer load notice: {st_err}. Trying transformers directly...")

        # Strategy 2: Transformers AutoModel with local_files_only
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
            model = AutoModel.from_pretrained(local_path, local_files_only=True)
            model.eval()
            _engine_info = {"engine": "transformers", "model_path": local_path, "reason": "", "fix": ""}
            logger.info(f"[ModelLoader] Successfully loaded local Transformers AutoModel from: {local_path}")
            return "transformers", (tokenizer, model)
        except Exception as tr_err:
            reason = f"{reason}; Transformers could not load it: {tr_err}"
            problems = {**problems, local_path: reason}
            logger.warning(f"[ModelLoader] Transformers AutoModel load notice: {tr_err}.")

    # No usable model. Say exactly why and how to fix it.
    if problems:
        detail = "; ".join(f"{p} {why}" for p, why in problems.items())
        reason = f"no usable copy of '{model_name}': {detail}"
    else:
        reason = f"'{model_name}' was not found in any local path (searched: {get_embedding_model_search_paths()[:3]}...)"
    fix = _fix_hint(problems)

    if not getattr(settings, "EMBEDDING_ALLOW_LEXICAL_FALLBACK", False):
        # Real semantic embeddings are mandatory. Deterministic word-hash vectors look like embeddings but are not,
        # so they are never substituted silently.
        _engine_info = {"engine": UNAVAILABLE_ENGINE, "model_path": None, "reason": reason, "fix": fix}
        logger.error(
            f"[ModelLoader] Embedding model UNAVAILABLE - {reason}\n  FIX: {fix}\n"
            f"  No embeddings will be generated (no fake vectors). Semantic search / RAG indexing stay off until the model is fixed."
        )
        raise EmbeddingModelUnavailable(f"{reason}. FIX: {fix}")

    # Explicit opt-in (EMBEDDING_ALLOW_LEXICAL_FALLBACK=true): lexical vectors for offline development.
    _engine_info = {"engine": LEXICAL_ENGINE, "model_path": None, "reason": reason, "fix": fix}
    logger.warning(
        f"[ModelLoader] Embedding engine: {LEXICAL_ENGINE} - semantic model NOT loaded. {reason}\n"
        f"  Retrieval and similarity keep working on word overlap only, so paraphrased matches are weaker.\n"
        f"  FIX: {fix}"
    )
    return LEXICAL_ENGINE, None


def get_embedding_encoder() -> Tuple[str, Any]:
    """
    Lazily loads the local embedding encoder with zero network calls.

    Returns:
        Tuple of (engine_type, model_object)
        engine_type can be: 'sentence_transformers', 'transformers', or 'lexical_fallback'
    """
    global _embedding_encoder, _embedding_failure
    if _embedding_encoder is not None:
        return _embedding_encoder
    if _embedding_failure is not None:      # decided once per process: fixing the files needs a restart (see the fix hint)
        raise _embedding_failure
    with _load_lock:                    # the startup backfill thread and a request can race: load the model once
        if _embedding_failure is not None:
            raise _embedding_failure
        if _embedding_encoder is None:
            try:
                _embedding_encoder = _load_embedding_encoder()
            except EmbeddingModelUnavailable as exc:
                _embedding_failure = exc
                raise
    return _embedding_encoder


def get_embedding_dimension() -> int:
    """Return the dimension of vectors produced by encode_texts()."""
    return _embedding_dim


def get_embedding_engine_info() -> Dict[str, Any]:
    """The active engine and, when there is no semantic model, why it is unusable and how to fix it. Never raises."""
    try:
        engine, _ = get_embedding_encoder()
    except EmbeddingModelUnavailable:
        return {"model_name": _model_name(), "dimension": _embedding_dim, "signature": None, **_engine_info,
                "engine": UNAVAILABLE_ENGINE}
    return {"model_name": _model_name(), "dimension": _embedding_dim, "signature": get_embedding_signature(),
            **_engine_info, "engine": engine}


def get_embedding_signature() -> str:
    """Names the VECTOR SPACE ``encode_texts`` currently produces.

    Vectors from different spaces are not comparable (a query embedded by the model against stored vectors from
    the lexical fallback matches nothing meaningful), so the vector store tags every vector with this and
    re-embeds the ones that carry a different tag. Both real engines run the same model, hence one signature.
    """
    engine, _ = get_embedding_encoder()
    if engine == LEXICAL_ENGINE:
        return f"{LEXICAL_ENGINE}-v{_LEXICAL_VERSION}:{_embedding_dim}"
    return f"{_model_name()}:{_embedding_dim}"


# ─── Lexical fallback embedder ───────────────────────────────────────────────
# A feature-hashing bag of words (unigrams + bigrams + character trigrams) into ``dim`` signed buckets, L2-normalised.
# It is a pure function of the text (no corpus statistics, no process state, no Python ``hash()``), so a query and a
# stored chunk are always embedded consistently across restarts, and texts that share vocabulary are close - which
# is what retrieval and near-duplicate detection need. It does not understand paraphrase; the real model does.
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_KIND_WEIGHT = {"w": 1.0, "b": 0.7, "c": 0.15}


@lru_cache(maxsize=131072)
def _feature_slot(kind: str, feature: str, dim: int) -> Tuple[int, float]:
    h = int.from_bytes(hashlib.blake2b(f"{kind}:{feature}".encode("utf-8"), digest_size=8).digest(), "little")
    return h % dim, (1.0 if (h >> 63) & 1 else -1.0)


def _lexical_embed_single(text: str, dim: int = 384) -> np.ndarray:
    tokens = [t for t in _WORD_RE.findall((text or "").lower()) if t.strip("_")]
    features: Counter = Counter()
    for token in tokens:
        features[("w", token, len(token))] += 1
    for left, right in zip(tokens, tokens[1:]):
        features[("b", f"{left} {right}", min(len(left), len(right)))] += 1
    for token in set(tokens):                         # sub-word overlap: "migrate" ~ "migration"
        if len(token) >= 5:
            padded = f"#{token}#"
            for i in range(len(padded) - 2):
                features[("c", padded[i:i + 3], len(token))] += 1

    vec = np.zeros(dim, dtype=np.float32)
    for (kind, feature, length), count in features.items():
        slot, sign = _feature_slot(kind, feature, dim)
        # sublinear term frequency; short words (the, of, to) carry little meaning, so they weigh less
        vec[slot] += sign * _KIND_WEIGHT[kind] * (1.0 + math.log(count)) * (min(length, 6) / 6.0)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:                                   # no words at all: a fixed, valid unit vector
        slot, sign = _feature_slot("w", "<empty>", dim)
        vec[:] = 0.0
        vec[slot] = sign
        return vec
    return vec / norm


def lexical_embed(texts: List[str], dim: Optional[int] = None) -> np.ndarray:
    """Embed ``texts`` with the lexical fallback (N, dim) - deterministic, offline, never fails."""
    dim = dim or _embedding_dim
    return np.array([_lexical_embed_single(t, dim) for t in texts], dtype=np.float32).reshape(len(texts), dim)


_unavailable_warned_at = 0.0


def _validate_vectors(vectors: Any, expected_rows: int) -> np.ndarray:
    """The vectors a model produced must be a (rows, dimension) float matrix of finite numbers."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != expected_rows:
        raise EmbeddingOutputError(f"expected {expected_rows} vector(s), the model returned shape {tuple(arr.shape)}")
    if arr.shape[1] != _embedding_dim:
        raise EmbeddingOutputError(f"expected {_embedding_dim}-dimensional vectors, the model returned {arr.shape[1]}")
    if not np.isfinite(arr).all():
        raise EmbeddingOutputError("the model returned NaN or infinite values")
    return arr


# (id(model), text) -> read-only vector, least recently used first. ~1.5 KB per vector plus its text.
_vector_cache: "OrderedDict[Tuple[int, str], np.ndarray]" = OrderedDict()
_vector_cache_lock = threading.Lock()
_VECTOR_CACHE_MAX_ENTRIES = int(os.getenv("EMBEDDING_CACHE_MAX_ENTRIES", "10000"))


def encode_texts(texts: Union[str, List[str]], strict: bool = False) -> Optional[np.ndarray]:
    """
    Encode a single string or list of strings into L2-normalized dense embedding vectors (N, 384), validated
    (right shape and dimension, finite values). Fully offline.

    Real semantic embeddings only. When the model is unusable: ``strict=True`` callers - the ones that STORE
    vectors (ChromaDB) - get the exception; ``strict=False`` callers get ``None`` (they all have a non-embedding
    path for "embeddings unavailable"). Deterministic lexical vectors are produced only with the explicit
    opt-in ``EMBEDDING_ALLOW_LEXICAL_FALLBACK=true`` (development).
    """
    global _unavailable_warned_at
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return np.zeros((0, _embedding_dim), dtype=np.float32)

    try:
        enc_type, model_obj = get_embedding_encoder()
    except EmbeddingModelUnavailable:
        if strict:
            raise
        now = time.monotonic()
        if now - _unavailable_warned_at > 300:      # one line per 5 minutes, not one per call
            _unavailable_warned_at = now
            logger.warning("[ModelLoader] encode_texts: the semantic embedding model is unavailable - returning no embeddings.")
        return None
    if enc_type == LEXICAL_ENGINE:
        return lexical_embed(texts)

    # A vector is a pure function of (model, text), so texts embedded before are served from the cache and only
    # new ones reach the model. GET /meetings/{id} rebuilds the quality/evaluation metrics on every request and
    # re-embedded the same ~45 texts each time (5-6 s of CPU per request, measured).
    model_key = id(model_obj)
    found: Dict[str, np.ndarray] = {}
    with _vector_cache_lock:
        for t in texts:
            vec = _vector_cache.get((model_key, t))
            if vec is not None:
                _vector_cache.move_to_end((model_key, t))
                found[t] = vec
    missing = list(dict.fromkeys(t for t in texts if t not in found))
    if missing:
        try:
            fresh = _infer_vectors(missing, enc_type, model_obj)
        except Exception as e:
            if strict:
                raise
            if getattr(settings, "EMBEDDING_ALLOW_LEXICAL_FALLBACK", False):
                logger.warning(f"[ModelLoader] Model inference error ({e}). Using the lexical fallback for this call (opt-in).")
                return lexical_embed(texts)
            logger.warning(f"[ModelLoader] Model inference error ({e}). Returning no embeddings for this call.")
            return None
        if fresh is None:
            return None
        with _vector_cache_lock:
            for t, row in zip(missing, fresh):
                vec = np.array(row, dtype=np.float32)
                vec.setflags(write=False)  # shared between callers: nobody may modify a cached vector
                found[t] = vec
                _vector_cache[(model_key, t)] = vec
            while len(_vector_cache) > _VECTOR_CACHE_MAX_ENTRIES:
                _vector_cache.popitem(last=False)
    return np.stack([found[t] for t in texts])


def _infer_vectors(texts: List[str], enc_type: str, model_obj: Any) -> Optional[np.ndarray]:
    """One model pass over ``texts``: validated (len(texts), dim) float32 vectors. Raises on a model error."""
    if enc_type == "sentence_transformers" and model_obj is not None:
        embeddings = model_obj.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _validate_vectors(embeddings, len(texts))

    if enc_type == "transformers" and model_obj is not None:
        import torch
        tokenizer, model = model_obj
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = model(**encoded)
            # Mean pooling
            attn_mask = encoded["attention_mask"].unsqueeze(-1)
            token_embs = out[0]
            sum_embs = torch.sum(token_embs * attn_mask, dim=1)
            sum_mask = torch.clamp(attn_mask.sum(dim=1), min=1e-9)
            mean_pooled = sum_embs / sum_mask
            # L2 normalize
            norm = torch.norm(mean_pooled, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized = (mean_pooled / norm).cpu().numpy()
        return _validate_vectors(normalized, len(texts))

    return None


def validate_embedding_model() -> Dict[str, Any]:
    """Prove the embedding model produces REAL semantic vectors. Never raises.

    Embeds a related pair and an unrelated sentence, then checks: the model loaded, the vectors have the expected
    dimension, every value is a finite number, they are unit length, and the related pair is closer than the
    unrelated one (a hash or random vector cannot pass that). Used by the preflight check and /system/health/models.
    """
    report: Dict[str, Any] = {"ok": False, "model": _model_name(), "expected_dimension": _embedding_dim}
    a = "The team agreed to launch the new website in November."
    b = "The website release was scheduled for the following month by everyone."
    c = "Bananas are yellow and grow in tropical countries."
    try:
        info = get_embedding_engine_info()
        report["engine"] = info.get("engine")
        report["model_path"] = info.get("model_path")
        if info.get("engine") in (UNAVAILABLE_ENGINE, LEXICAL_ENGINE):
            report["error"] = info.get("reason") or "no semantic model loaded"
            report["fix"] = info.get("fix")
            return report
        vectors = encode_texts([a, b, c], strict=True)
        report["dimension"] = int(vectors.shape[1])
        report["finite"] = bool(np.isfinite(vectors).all())
        norms = np.linalg.norm(vectors, axis=1)
        report["unit_norm"] = bool(np.allclose(norms, 1.0, atol=1e-3))
        sim_related = float(vectors[0] @ vectors[1])
        sim_unrelated = float(vectors[0] @ vectors[2])
        report["similarity_related"] = round(sim_related, 4)
        report["similarity_unrelated"] = round(sim_unrelated, 4)
        report["semantic"] = sim_related > sim_unrelated + 0.1
        report["ok"] = bool(report["finite"] and report["unit_norm"] and report["semantic"]
                            and report["dimension"] == _embedding_dim)
        if not report["ok"]:
            report["error"] = "the vectors failed a check (see the individual flags)"
    except Exception as e:  # noqa: BLE001 - a diagnostic must not raise
        report["error"] = f"{type(e).__name__}: {e}"
    return report


# ─── Model Health & Diagnostic Inspection ───────────────────────────────────

def _airawat_key_configured() -> bool:
    """Whether an AIRAWAT key is in use (SERVER_*/LOCAL_* selection or admin override, not just AIRAWAT_API_KEY)."""
    try:
        from backend.services.model_router import model_router

        key = model_router.get_effective_config().get("api_key") or ""
    except Exception:  # noqa: BLE001
        key = ""
    return len((key or settings.AIRAWAT_API_KEY or "").strip()) > 5


def get_all_models_health() -> Dict[str, Any]:
    """
    Comprehensive diagnostic report on all AI/ML models in the system.
    Used by /api/v1/system/health and /api/v1/system/health/models.
    """
    engine = get_embedding_engine_info()
    semantic_model_ready = engine["engine"] not in (LEXICAL_ENGINE, UNAVAILABLE_ENGINE)

    # Embedding model info
    emb_info = {
        "name": _model_name(),
        "configured_path": getattr(settings, "EMBEDDING_MODEL_PATH", "./models/embeddings/all-MiniLM-L6-v2"),
        "resolved_local_path": engine.get("model_path"),
        "is_local_available": semantic_model_ready,
        "active_engine": engine["engine"],
        "signature": engine["signature"],
        "dimension": _embedding_dim,
        "offline_ready": True,
        # Startup verification of the weights on disk (Git LFS pointer / size / SHA-256 / fetched with git lfs pull).
        "files": _guard_status(),
        # Present only when the semantic model could not be loaded: why, and exactly how to fix it.
        **({} if semantic_model_ready else {"problem": engine.get("reason"), "fix": engine.get("fix")}),
    }

    # Speech-to-text is AIRAWAT Whisper only (no local speech model). It cannot detect the language, so the
    # upload names it or the configured default (admin setting / AIRAWAT_WHISPER_LANGUAGE) is used.
    try:
        from backend.services.model_router import model_router as _mr

        _default_lang = (_mr.get_effective_config().get("language") or "").strip().lower() or None
    except Exception:  # noqa: BLE001
        _default_lang = None
    try:
        from backend.providers import provider_factory
        _active_p = provider_factory.get_effective_configs().get("active_provider", "ollama")
    except Exception:
        _active_p = "speech_engine"

    whisper_info = {
        "provider": _active_p,
        "default_language": _default_lang,
        "language_source": "upload form field language, else the default language",
    }

    # Diarization: the NeMo TitaNet speaker embedder decides the speaker count. Without it a weak numpy
    # fallback is used, which cannot tell voices apart (a 4-speaker meeting comes out as 1 speaker).
    nemo_cache = getattr(settings, "NEMO_CACHE_DIR", "./nemo_cache")
    diar_dir = getattr(settings, "DIARIZATION_MODEL_PATH", "./models/diarization")
    nemo_importable = importlib.util.find_spec("nemo") is not None
    nemo_file = None
    if os.path.isdir(diar_dir):
        nemo_file = next((f for f in sorted(os.listdir(diar_dir)) if f.endswith(".nemo")), None)
    neural_loaded = None
    try:
        from backend.services import diarization_service as _ds

        neural_loaded = _ds.using_neural_embedder()
    except Exception:  # noqa: BLE001
        pass
    diar_info = {
        "nemo_enabled": getattr(settings, "NEMO_ENABLED", True),
        "nemo_installed": nemo_importable,
        "speaker_model": getattr(settings, "NEMO_SPEAKER_MODEL", "titanet_large"),
        "speaker_model_file": nemo_file,
        "neural_embedder_loaded": neural_loaded,          # None = not loaded yet in this process
        "pyannote_enabled": getattr(settings, "PYANNOTE_ENABLED", True),
        "configured_mode": getattr(settings, "DIARIZATION_MODE", "NEMO_PRIMARY_PYANNOTE_FALLBACK"),
        "nemo_cache_dir": nemo_cache,
        "nemo_cache_exists": os.path.isdir(nemo_cache),
        "neural_speaker_embedding_ready": bool(nemo_importable and (nemo_file or os.path.isdir(nemo_cache))),
        "offline_ready": bool(nemo_file),
        **({} if nemo_importable else {
            "problem": "the 'nemo' package is not installed: speaker counts come from a weak fallback embedder",
            "fix": "pip install -r backend/requirements.docker.txt (Linux) and provide models/diarization/titanet_large.nemo",
        }),
    }

    return {
        # healthy | degraded (lexical stand-in, development only) | unavailable (no semantic model at all)
        "status": "healthy" if semantic_model_ready else ("unavailable" if engine["engine"] == UNAVAILABLE_ENGINE else "degraded"),
        "offline_mode": getattr(settings, "OFFLINE_MODE", True),
        "hf_hub_offline": os.getenv("HF_HUB_OFFLINE") == "1",
        "transformers_offline": os.getenv("TRANSFORMERS_OFFLINE") == "1",
        "models": {
            "embedding": emb_info,
            "transcription": whisper_info,
            "diarization": diar_info,
        },
    }

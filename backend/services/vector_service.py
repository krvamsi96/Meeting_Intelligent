"""Vector database service - ChromaDB is the ONLY vector store, on a developer machine and on the server.

    PostgreSQL  -> application data (meetings, transcripts, summaries, MoM, action items, metrics, ...)
    ChromaDB    -> embeddings only: RAG retrieval, semantic search, similar-meeting search

The same code runs everywhere. The one environment-specific input is where ChromaDB persists its data:

    VECTOR_DB=chromadb
    CHROMA_PERSIST_DIRECTORY=./data/chroma            # local
    CHROMA_PERSIST_DIRECTORY=/var/lib/<app>/chroma    # server (a persistent volume, never a temp dir)

Collections (cosine distance):
    meeting_embeddings            one vector per meeting (whole transcript)         -> similar-meeting search
    transcript_chunk_embeddings   many vectors per meeting (transcript/doc chunks)  -> per-meeting RAG retrieval

Every stored vector carries the metadata: meeting_id, bharatvc_meeting_id, display_id, chunk_id, source,
document_type, embedding_signature. Ids are deterministic (``<meeting_id>:<chunk_id>``), so processing a meeting
again UPDATES its vectors in place and removes the ones that no longer exist - it can never create duplicates.

The embedding signature names the vector space (model + dimension, or the lexical fallback). If the space changes -
e.g. the model file on a server was broken and is later repaired - the old vectors are no longer comparable with new
query vectors, so they count as "not indexed" (``indexed_meeting_vectors`` / ``has_transcript_chunks``) and the
startup backfill / lazy chat indexing re-embeds them in place, with no manual step.

Meeting isolation: chunk retrieval (the chatbot path) is ALWAYS filtered by ``meeting_id`` - and by the meeting's
``bharatvc_meeting_id`` as well when PostgreSQL knows it, so vectors left behind by a different meeting that once
had the same numeric id (a reset database, or the server and local databases sharing one directory) are never
returned. The legacy ``cross_meeting`` argument is accepted but ignored. Finding OTHER meetings is a separate path
(``get_semantic_candidates`` / ``find_similar_meetings``) that returns meeting-level matches, never chunk context.

One process owns a ChromaDB directory. An embedded ChromaDB keeps its HNSW index in memory, so a second process
writing to the same directory leaves the first one with a stale index (queries fail with "Error finding id").
The owner therefore holds an OS-level lock on ``<dir>/.chroma_owner.lock``; a second process gets a clear
"directory in use" error instead of silently corrupting the store. Run the API with a single uvicorn worker.
"""

import errno
import logging
import os
import re
import threading
import time
import warnings
from typing import Any, Dict, Iterator, List, Optional

from backend.config.config import (
    settings,
    DEFAULT_PROJECT_NAME,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_SIMILARITY_TOP_K,
    RAG_DEFAULT_TOP_K,
    RAG_MAX_CONTEXT_CHARS,
)
from backend.rag.chunker import chunk_transcript_smart

logger = logging.getLogger(__name__)

BACKEND = "chromadb"
MEETING_COLLECTION = "meeting_embeddings"
CHUNK_COLLECTION = "transcript_chunk_embeddings"

SOURCE_TRANSCRIPT = "transcript"
SOURCE_DOCUMENT = "document"
DOC_TYPE_MEETING = "meeting"
DOC_TYPE_TRANSCRIPT_CHUNK = "transcript_chunk"
DOC_TYPE_DOCUMENT_CHUNK = "document_chunk"
MEETING_CHUNK_ID = "full_transcript"        # chunk_id of the single whole-meeting vector

_COSINE = {"hnsw": {"space": "cosine"}}
_UPSERT_BATCH_SIZE = 100                    # bounds request size / memory on a 1 GB RAM server
_RETRY_INTERVAL_SEC = 60.0                  # an unavailable store is re-checked at most this often
_WARN_INTERVAL_SEC = 300.0                  # ... and warns at most this often
_LOCK_FILENAME = ".chroma_owner.lock"


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """ChromaDB metadata values must be str/int/float/bool: drop None, stringify anything else."""
    clean: Dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        clean[str(key)] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return clean


def _doc_id(meeting_id: Any, chunk_id: Any) -> str:
    """Deterministic vector id: the same meeting + chunk always maps to the same id."""
    return f"{int(meeting_id)}:{chunk_id}"


def _batched(items: List[Any], size: int) -> Iterator[List[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _similarity(distance: float) -> float:
    """Cosine distance ([0, 2]) -> similarity in [0, 1] (meeting-level scale, unchanged from earlier releases)."""
    return round(max(0.0, min(1.0, 1.0 - (distance / 2.0 if distance > 1.0 else distance))), 4)


def _collection_space(collection) -> Optional[str]:
    """Distance metric an existing collection was created with (None when it cannot be determined)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            space = ((collection.configuration or {}).get("hnsw") or {}).get("space")
        if space:
            return str(space)
    except Exception:
        pass
    return (getattr(collection, "metadata", None) or {}).get("hnsw:space")


class _StoreUnavailable(Exception):
    """The vector store cannot be opened; ``state`` is a short machine-readable reason."""

    def __init__(self, state: str, message: str):
        super().__init__(message)
        self.state = state


class _DirectoryOwnerLock:
    """Exclusive OS lock on a marker file inside the ChromaDB directory (released automatically if the process dies)."""

    def __init__(self, directory: str):
        self.directory = directory
        self.path = os.path.join(directory, _LOCK_FILENAME)
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        """Take the lock. Raises _StoreUnavailable when another process owns the directory."""
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
        try:
            self._lock_first_byte(fd)
        except OSError as exc:
            holder = self._holder()
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise _StoreUnavailable(
                    "directory_in_use",
                    "The ChromaDB directory is already in use by another process"
                    + (f" (pid {holder})" if holder else "")
                    + ". An embedded ChromaDB directory can be owned by one process only - run the API with a "
                      "single worker (UVICORN_WORKERS=1) and stop any other process using this directory.",
                )
            # A filesystem without lock support (some network mounts): keep working, just unprotected.
            logger.warning("Could not lock the ChromaDB directory (%s); continuing without single-owner protection.", exc)
            return
        self._fd = fd
        try:                                       # the pid lives after the locked byte so other processes can read it
            os.ftruncate(fd, 1)
            os.lseek(fd, 1, os.SEEK_SET)
            os.write(fd, str(os.getpid()).encode("ascii"))
        except OSError:
            pass

    @staticmethod
    def _lock_first_byte(fd: int) -> None:
        if os.name == "nt":
            import msvcrt
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"L")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _holder(self) -> str:
        try:
            with open(self.path, "rb") as fh:
                fh.seek(1)
                return fh.read(20).decode("ascii", "ignore").strip()
        except OSError:
            return ""

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            os.close(fd)


class VectorService:
    """Vector Database Service backed by a persistent ChromaDB."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        # Creating the singleton must be free of side effects (no disk access, no model load): merely importing
        # this module - from a script, a test, a tool - never touches ChromaDB. Initialization happens on first
        # use or explicitly at application startup (ensure_initialized()).
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._reset_state()
            return cls._instance

    _retry_lock = threading.Lock()
    _init_lock = threading.Lock()

    def _reset_state(self):
        self._initialized = False
        self._embedding_dim: Optional[int] = None
        self._client = None
        self._meetings = None
        self._chunks = None
        self._owner_lock: Optional[_DirectoryOwnerLock] = None
        self._op_lock = threading.RLock()          # serialises ChromaDB calls; embeddings are computed outside it
        self.path: Optional[str] = None
        self.is_ready = False
        self.dimension_mismatch = False
        self.init_state = "not_initialized"
        self.init_error = ""
        self._last_init_attempt = 0.0
        self._last_warned = 0.0
        self._logged_states: set = set()

    # ------------------------------------------------------------------ #
    # Initialization and availability
    # ------------------------------------------------------------------ #

    def ensure_initialized(self) -> None:
        """Initialize once (thread-safe). Called at application startup and lazily on first use."""
        if self._initialized:
            return
        with self._init_lock:
            if not self._initialized:
                from backend.services.model_loader import get_embedding_dimension

                self._embedding_dim = get_embedding_dimension() or 384
                self._initialized = True
                self._attempt_init()

    def _attempt_init(self) -> None:
        """Read the configuration, create/connect to ChromaDB and verify the collections. Never raises."""
        self._last_init_attempt = time.monotonic()
        self.path = settings.CHROMA_PERSIST_DIRECTORY
        self._close_client()
        try:
            self._open_store(self.path)
        except _StoreUnavailable as exc:
            self._mark_unavailable(exc.state, str(exc))
        except Exception as exc:
            self._mark_unavailable("init_error", f"ChromaDB could not be initialized: {type(exc).__name__}: "
                                                 f"{str(exc).splitlines()[0][:200] if str(exc) else ''}")
        else:
            self._mark_available()

    def _open_store(self, path: str) -> None:
        backend = (settings.VECTOR_DB or BACKEND).strip().lower()
        if backend != BACKEND:
            raise _StoreUnavailable(
                "unsupported_backend",
                f"VECTOR_DB='{backend}' is not supported: ChromaDB is the only vector database. Set VECTOR_DB=chromadb.")
        if not path:
            raise _StoreUnavailable("path_not_configured", "CHROMA_PERSIST_DIRECTORY is not set.")
        if re.search(r"<[^<>]*>", path):                 # e.g. "<configure-for-environment>" copied from .env.example
            raise _StoreUnavailable(
                "path_not_configured",
                "CHROMA_PERSIST_DIRECTORY is still a placeholder. Set it to a real directory: ./data/chroma locally, "
                "or an absolute path on persistent storage on the server.")
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            raise _StoreUnavailable("directory_error", f"The ChromaDB directory cannot be created: {exc.strerror or exc}.")
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise _StoreUnavailable("chromadb_missing", "The 'chromadb' package is not installed (pip install chromadb).")

        if self._owner_lock is None or self._owner_lock.directory != path:
            if self._owner_lock is not None:
                self._owner_lock.release()
            self._owner_lock = None
            lock = _DirectoryOwnerLock(path)
            try:
                lock.acquire()
            except PermissionError as exc:
                raise _StoreUnavailable("directory_not_writable", f"The ChromaDB directory is not writable: {exc.strerror or exc}.")
            self._owner_lock = lock

        # Anonymous telemetry is off: the platform runs offline and must not make outbound calls.
        self._client = chromadb.PersistentClient(
            path=path, settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False))
        self._client.heartbeat()
        self._meetings = self._get_collection(MEETING_COLLECTION)
        self._chunks = self._get_collection(CHUNK_COLLECTION)
        self._verify_collections()

    def _get_collection(self, name: str):
        # embedding_function=None: vectors always come from the platform's own (offline) embedding model,
        # so ChromaDB must never try to download or run a default one.
        return self._client.get_or_create_collection(name=name, configuration=_COSINE, embedding_function=None)

    def _verify_collections(self) -> None:
        """Both collections must exist, be readable, use cosine distance and match the embedding dimension."""
        for collection in (self._meetings, self._chunks):
            space = _collection_space(collection)
            if space and space != "cosine":
                raise _StoreUnavailable(
                    "collection_space_mismatch",
                    f"ChromaDB collection '{collection.name}' uses '{space}' distance but cosine is required. "
                    f"Rebuild it: python -m backend.scripts.reindex_vectors --reset")
            if collection.count() == 0:
                continue
            sample = collection.get(limit=1, include=["embeddings"]).get("embeddings")
            if sample is not None and len(sample) > 0 and len(sample[0]) != self._embedding_dim:
                self.dimension_mismatch = True
                raise _StoreUnavailable(
                    "dimension_mismatch",
                    f"ChromaDB collection '{collection.name}' holds {len(sample[0])}-dimensional vectors but the "
                    f"embedding model produces {self._embedding_dim}. Re-index after changing the embedding model: "
                    f"python -m backend.scripts.reindex_vectors --reset")
        self.dimension_mismatch = False

    def _mark_available(self) -> None:
        self.is_ready = True
        self.init_state = "ready"
        self.init_error = ""
        self._logged_states.clear()
        logger.info("Vector database: ChromaDB")
        logger.info("ChromaDB status: available")
        logger.info("ChromaDB path: %s", self.path)
        try:
            logger.info("ChromaDB collections: %s=%d, %s=%d (embedding dimension %s)", MEETING_COLLECTION,
                        self._meetings.count(), CHUNK_COLLECTION, self._chunks.count(), self._embedding_dim)
        except Exception:
            pass

    def _mark_unavailable(self, state: str, message: str) -> None:
        self.is_ready = False
        self.init_state = state
        self.init_error = message
        if state not in self._logged_states:        # one clear block per state per process, not per retry
            self._logged_states.add(state)
            logger.error("Vector database: ChromaDB")
            logger.error("ChromaDB status: unavailable")
            logger.error("ChromaDB error: %s", message)
            logger.error("ChromaDB path: %s", self.path)

    def _close_client(self) -> None:
        client, self._client = self._client, None
        self._meetings = self._chunks = None
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.debug("ChromaDB client close skipped: %s", exc)

    def close(self) -> None:
        """Release ChromaDB and the directory lock (application shutdown, tests). A later call re-opens."""
        with self._op_lock:
            self._close_client()
            if self._owner_lock is not None:
                self._owner_lock.release()
                self._owner_lock = None
            self.is_ready = False
            self._initialized = False
            self.init_state = "closed"

    def _available(self) -> bool:
        """True when vector operations can run. An unavailable store is re-checked at most once per interval, so
        it recovers by itself (e.g. the other process released the directory) without a restart."""
        self.ensure_initialized()
        if self.is_ready and self._client is not None:
            return True
        if time.monotonic() - self._last_init_attempt >= _RETRY_INTERVAL_SEC and self._retry_lock.acquire(False):
            try:
                if time.monotonic() - self._last_init_attempt >= _RETRY_INTERVAL_SEC:
                    self._attempt_init()
            finally:
                self._retry_lock.release()
        if self.is_ready and self._client is not None:
            return True
        self._warn_unavailable()
        return False

    def _warn_unavailable(self) -> None:
        now = time.monotonic()
        if now - self._last_warned >= _WARN_INTERVAL_SEC:     # throttled: at most one line per 5 minutes
            self._last_warned = now
            logger.warning("Vector operation skipped: ChromaDB is not available (%s).", self.init_state)

    def check_ready(self) -> bool:
        """Public readiness probe (health checks, scripts): ChromaDB is initialized and answering."""
        if not self._available():
            return False
        try:
            with self._op_lock:
                self._client.heartbeat()
            return True
        except Exception as exc:
            self._mark_unavailable("connection_lost", f"ChromaDB is not responding: {type(exc).__name__}: "
                                                      f"{str(exc).splitlines()[0][:160] if str(exc) else ''}")
            return False

    def health(self) -> Dict[str, Any]:
        """Payload of ``vector_database`` in GET /system/health (no paths or other internals)."""
        if self.check_ready():
            return {"backend": BACKEND, "status": "available"}
        return {"backend": BACKEND, "status": "unavailable",
                "error": self.init_error or "ChromaDB is not initialized."}

    def status(self) -> Dict[str, Any]:
        """Detailed state for logs and the setup-check script."""
        ready = self.check_ready()
        counts = None
        if ready:
            try:
                with self._op_lock:
                    counts = {MEETING_COLLECTION: self._meetings.count(), CHUNK_COLLECTION: self._chunks.count()}
            except Exception:
                counts = None
        return {
            "backend": BACKEND,
            "status": "available" if ready else "unavailable",
            "error": None if ready else (self.init_error or "ChromaDB is not initialized."),
            "state": self.init_state,
            "path": self.path,
            "embedding_dimension": self._embedding_dim,
            "dimension_mismatch": bool(self.dimension_mismatch),
            "collections": counts,
        }

    def reset_collections(self) -> None:
        """Drop and recreate both collections (maintenance: embedding-model change or a full rebuild).

        Vectors are derived data - PostgreSQL still holds every transcript - so they can always be rebuilt with
        ``python -m backend.scripts.reindex_vectors``. Works even when the store was opened in a failed state
        (dimension/distance mismatch), which is exactly when it is needed.
        """
        self.ensure_initialized()
        if self._client is None:
            raise RuntimeError(f"ChromaDB is not open: {self.init_error or self.init_state}")
        with self._op_lock:
            for name in (MEETING_COLLECTION, CHUNK_COLLECTION):
                try:
                    self._client.delete_collection(name)
                except Exception as exc:              # a collection that does not exist yet is fine
                    logger.debug("delete_collection(%s) skipped: %s", name, exc)
            self._meetings = self._get_collection(MEETING_COLLECTION)
            self._chunks = self._get_collection(CHUNK_COLLECTION)
            self._verify_collections()
        self._mark_available()

    # ------------------------------------------------------------------ #
    # Meeting identity (PostgreSQL is the source of truth)
    # ------------------------------------------------------------------ #

    def _lookup_meeting(self, meeting_id: Any) -> Optional[Dict[str, str]]:
        """Identity of a meeting read from PostgreSQL; None when it cannot be read (retrieval then falls back
        to the meeting_id filter alone)."""
        try:
            from backend.database.connection import SessionLocal
            from backend.models.models import Meeting

            with SessionLocal() as session:
                row = (
                    session.query(Meeting.bharatvc_meeting_id, Meeting.meeting_id, Meeting.display_id, Meeting.file_type)
                    .filter(Meeting.id == int(meeting_id))
                    .first()
                )
        except Exception as exc:
            logger.debug("Meeting identity lookup skipped for %s: %s", meeting_id, type(exc).__name__)
            return None
        if row is None:
            return None
        return {
            "bharatvc_meeting_id": _text(row[0] or row[1]),
            "display_id": _text(row[2]),
            "source": SOURCE_DOCUMENT if _text(row[3]) == "document" else SOURCE_TRANSCRIPT,
        }

    def _identity(self, meeting_id: Any, *, bharatvc_meeting_id: Any = None, display_id: Any = None,
                  source: Any = None) -> Dict[str, str]:
        """bharatvc_meeting_id / display_id / source for a meeting: what the caller passed, the rest from PostgreSQL."""
        ident = {"bharatvc_meeting_id": _text(bharatvc_meeting_id), "display_id": _text(display_id),
                 "source": _text(source)}
        if not all(ident.values()):
            row = self._lookup_meeting(meeting_id)
            if row:
                for key, value in row.items():
                    ident[key] = ident[key] or value
        ident["source"] = ident["source"] or SOURCE_TRANSCRIPT
        return ident

    def _scope(self, meeting_id: Any) -> Dict[str, Any]:
        """The ChromaDB ``where`` filter that confines a query to ONE meeting."""
        row = self._lookup_meeting(meeting_id)
        bharatvc = (row or {}).get("bharatvc_meeting_id", "")
        if bharatvc:
            return {"$and": [{"meeting_id": int(meeting_id)}, {"bharatvc_meeting_id": bharatvc}]}
        return {"meeting_id": int(meeting_id)}

    def _base_metadata(self, meeting_id: Any, ident: Dict[str, str], chunk_id: Any, document_type: str) -> Dict[str, Any]:
        return {
            "meeting_id": int(meeting_id),
            "bharatvc_meeting_id": ident["bharatvc_meeting_id"],
            "display_id": ident["display_id"],
            "chunk_id": str(chunk_id),
            "source": ident["source"],
            "document_type": document_type,
            "embedding_signature": self._embedding_signature(),
        }

    # ------------------------------------------------------------------ #
    # Embedding generation (offline model loader)
    # ------------------------------------------------------------------ #

    def _get_embedder(self):
        """Lazy load embedding encoder from centralized offline model loader."""
        from backend.services.model_loader import get_embedding_encoder
        return get_embedding_encoder()

    def _embedding_signature(self) -> str:
        """Names the embedding space new vectors are written in (see model_loader.get_embedding_signature).

        Every stored vector carries it. Vectors made by a different engine - e.g. the lexical fallback used while the
        semantic model file was broken - are not comparable with the current model's query vectors, so they count as
        not indexed and are re-embedded in place instead of silently returning noise.
        """
        from backend.services.model_loader import get_embedding_signature
        return get_embedding_signature()

    def generate_embedding(self, text: str) -> List[float]:
        """Generate dense embedding for text via offline model loader.

        Raises when the model fails mid-run: a vector from another space must never be produced silently (callers
        already treat an exception as "embedding unavailable" - indexing skips the item, evaluation reports it as
        not measured)."""
        cleaned_text = text[:RAG_MAX_CONTEXT_CHARS].strip()
        if not cleaned_text:
            cleaned_text = "Empty meeting transcript"

        from backend.services.model_loader import encode_texts
        embs = encode_texts([cleaned_text], strict=True)
        if embs is None or len(embs) == 0:
            raise RuntimeError("The embedding model returned no vector.")
        return embs[0].tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for a batch of texts in a single model forward pass."""
        if not texts:
            return []

        cleaned_texts = [t[:RAG_MAX_CONTEXT_CHARS].strip() or "Empty transcript segment" for t in texts]
        from backend.services.model_loader import encode_texts
        try:
            embs = encode_texts(cleaned_texts, strict=True)
            if embs is not None and len(embs) == len(cleaned_texts):
                return [row.tolist() for row in embs]
        except Exception as e:
            logger.warning(f"[VectorService] Batch embedding error ({e}). Falling back to per-item.")

        return [self.generate_embedding(t) for t in cleaned_texts]

    # ------------------------------------------------------------------ #
    # Meeting-level vectors (whole-transcript embedding)
    # ------------------------------------------------------------------ #

    def add_meeting_vector(
        self,
        meeting_id: int,
        meeting_code: str,
        filename: str,
        title: str,
        text: str,
        mom_summary: str = "",
        project_name: str = DEFAULT_PROJECT_NAME,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Index (or update in place) a meeting's whole-transcript vector with rich comparison metadata.

        Returns True when the vector was stored, False when it was not (ChromaDB unavailable or the write failed),
        so callers can report indexing honestly instead of assuming success.
        """
        if not self._available():
            logger.warning("ChromaDB not available; skipping add_meeting_vector for meeting %s.", meeting_id)
            return False
        try:
            extra = extra_metadata or {}
            ident = self._identity(
                meeting_id,
                bharatvc_meeting_id=extra.get("bharatvc_meeting_id") or meeting_code,
                display_id=extra.get("display_id"),
                source=SOURCE_DOCUMENT if extra.get("file_type") == "document" else None,
            )
            vector = self.generate_embedding(text)
            metadata = {
                "meeting_code": meeting_code,
                "filename": filename,
                "title": title or filename,
                "mom_summary": (mom_summary or text[:300]).strip(),
                "project_name": project_name or DEFAULT_PROJECT_NAME,
            }
            metadata.update(_clean_metadata(extra))
            metadata.update(self._base_metadata(meeting_id, ident, MEETING_CHUNK_ID, DOC_TYPE_MEETING))
            with self._op_lock:
                self._meetings.upsert(
                    ids=[_doc_id(meeting_id, MEETING_CHUNK_ID)],
                    embeddings=[vector],
                    documents=[text[:6000]],
                    metadatas=[_clean_metadata(metadata)],
                )
            logger.info(f"Vector DB: Upserted meeting {meeting_code} (ID: {meeting_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to add meeting vector: {e}")
            return False

    def indexed_meeting_vectors(self) -> Dict[int, str]:
        """``meeting_id -> bharatvc_meeting_id`` of every whole-meeting vector usable with the CURRENT embedding model.

        Used by backfill to decide what is missing. A vector written by a different embedding engine (e.g. while the
        model file was broken and the lexical fallback was active, or before vectors were tagged) is NOT returned:
        it cannot be compared with today's query vectors, so backfill re-embeds it in place (same deterministic id).
        """
        if not self._available():
            return {}
        try:
            signature = self._embedding_signature()
            with self._op_lock:
                stored = self._meetings.get(include=["metadatas"])
            current: Dict[int, str] = {}
            foreign = 0
            for m in (stored.get("metadatas") or []):
                if not m or m.get("meeting_id") is None:
                    continue
                if m.get("embedding_signature") == signature:
                    current[int(m["meeting_id"])] = _text(m.get("bharatvc_meeting_id"))
                else:
                    foreign += 1
            if foreign:
                logger.info("ChromaDB: %d meeting vector(s) were written by a different embedding engine and will be "
                            "re-embedded with '%s'", foreign, signature)
            return current
        except Exception as e:
            logger.error(f"Failed to list indexed meetings: {e}")
            return {}

    def get_semantic_candidates(
        self,
        meeting_id: int,
        text: str,
        top_k: int = 25,
    ) -> List[Dict[str, Any]]:
        """Return raw semantic neighbours (no thresholding) for multi-signal ranking.

        Each candidate: {meeting_id, meeting_code, semantic_score, metadata, preview}. The current meeting is
        never returned and each other meeting appears once. The heavy classification/ranking is done by the
        similarity_service using these candidates plus authoritative DB data.
        """
        candidates: List[Dict[str, Any]] = []
        if not text or not text.strip() or not self._available():
            return candidates
        try:
            pk = int(meeting_id)                       # the meeting's primary key, never a display/bharatvc id
            query_vector = self.generate_embedding(text)
            with self._op_lock:
                total = self._meetings.count()
                if total == 0:
                    return candidates
                results = self._meetings.query(
                    query_embeddings=[query_vector],
                    n_results=min(max(top_k * 2, top_k), total),
                    where={"meeting_id": {"$ne": pk}},
                    include=["metadatas", "distances", "documents"],
                )
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            documents = (results.get("documents") or [[]])[0]
            seen = set()
            for idx, meta in enumerate(metadatas):
                meta = meta or {}
                m_id = meta.get("meeting_id")
                if m_id is None or m_id == pk or m_id in seen:
                    continue
                seen.add(m_id)
                dist = float(distances[idx]) if idx < len(distances) else 1.0
                candidates.append(
                    {
                        "meeting_id": m_id,
                        "meeting_code": meta.get("meeting_code") or meta.get("bharatvc_meeting_id") or str(m_id),
                        "semantic_score": _similarity(dist),
                        "metadata": meta,
                        "preview": documents[idx] if idx < len(documents) else "",
                    }
                )
        except Exception as e:
            logger.error(f"ChromaDB candidate query failed: {e}")
        candidates.sort(key=lambda x: x["semantic_score"], reverse=True)
        return candidates[:top_k]

    def delete_meeting_vector(self, meeting_id: int):
        """Remove a meeting's whole-transcript vector and all of its transcript chunk vectors."""
        if not self._available():
            return
        try:
            with self._op_lock:
                self._meetings.delete(where={"meeting_id": int(meeting_id)})
                self._chunks.delete(where={"meeting_id": int(meeting_id)})
            logger.info(f"Vector DB: Deleted meeting vector {meeting_id}")
        except Exception as e:
            logger.error(f"Failed to delete meeting vector: {e}")

    def find_similar_meetings(
        self,
        meeting_id: int,
        text: str,
        project_name: str = None,
        top_k: int = DEFAULT_SIMILARITY_TOP_K,
        min_score: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """Find similar meetings using cosine similarity (the current meeting is never returned)."""
        matches: List[Dict[str, Any]] = []
        if not text or not text.strip() or not self._available():
            return matches

        clean_project = (project_name or "").strip().lower()
        _generic_defaults = {
            "ai meeting intelligence platform",
            "meeting intelligence platform",
            "",
        }
        use_project_boost = clean_project and clean_project not in _generic_defaults

        try:
            pk = int(meeting_id)                       # the meeting's primary key, never a display/bharatvc id
            query_vector = self.generate_embedding(text)
            with self._op_lock:
                total = self._meetings.count()
                if total == 0:
                    return matches
                results = self._meetings.query(
                    query_embeddings=[query_vector],
                    n_results=min(top_k * 3, total),
                    where={"meeting_id": {"$ne": pk}},
                    include=["metadatas", "distances"],
                )
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            seen_ids = set()

            for idx, meta in enumerate(metadatas):
                meta = meta or {}
                m_id = meta.get("meeting_id")
                if m_id is None or m_id == pk or m_id in seen_ids:
                    continue
                seen_ids.add(m_id)

                dist = float(distances[idx]) if idx < len(distances) else 0.5
                raw_score = _similarity(dist)
                match_project = (meta.get("project_name") or "").strip().lower()
                project_boost = 0.05 if (use_project_boost and match_project == clean_project) else 0.0
                similarity_score = round(min(1.0, raw_score + project_boost), 4)

                if similarity_score >= min_score:
                    matches.append(
                        {
                            "meeting_id": m_id,
                            "meeting_code": meta.get("meeting_code") or meta.get("bharatvc_meeting_id") or str(m_id),
                            "filename": meta.get("filename"),
                            "title": meta.get("title"),
                            "similarity_score": similarity_score,
                            "similarity_percentage": f"{similarity_score * 100:.1f}%",
                            "matched_mom": meta.get("mom_summary") or "No prior MoM available.",
                        }
                    )
        except Exception as e:
            logger.error(f"ChromaDB similarity query failed: {e}")

        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        return matches[:top_k]

    # ------------------------------------------------------------------ #
    # Transcript chunk vectors (per-meeting RAG retrieval)
    # ------------------------------------------------------------------ #

    def has_transcript_chunks(self, meeting_id: int) -> bool:
        """Existence check used for lazy indexing decisions (scoped to this meeting's identity).

        True only when the meeting has chunk vectors AND every one was written in the CURRENT embedding space; chunks
        from a different engine (see ``_embedding_signature``) count as absent, so callers re-index them in place.
        """
        if not self._available():
            return False
        try:
            where = self._scope(meeting_id)
            signature = self._embedding_signature()
            with self._op_lock:
                found = self._chunks.get(where=where, include=["metadatas"])
            metadatas = found.get("metadatas") or []
            return bool(metadatas) and all((m or {}).get("embedding_signature") == signature for m in metadatas)
        except Exception as e:
            logger.error(f"has_transcript_chunks check failed: {e}")
            return False

    def add_transcript_chunks(
        self,
        meeting_id: int,
        segments: List[Any],
        *,
        source: Optional[str] = None,
        bharatvc_meeting_id: Optional[str] = None,
        display_id: Optional[str] = None,
    ) -> int:
        """Index (or re-index) a meeting's transcript/document chunks for RAG retrieval; returns the chunk count.

        Idempotent: ids are ``<meeting_id>:<chunk_id>``, so processing the same meeting again updates its vectors
        in place and then deletes the ones this transcript no longer produces - the meeting's vectors are always
        exactly the current transcript's, never duplicated and never stale. ``source`` / ``bharatvc_meeting_id`` /
        ``display_id`` default to the meeting's row in PostgreSQL.
        """
        if not self._available():
            logger.warning("ChromaDB not available; skipping add_transcript_chunks for meeting %s.", meeting_id)
            return 0
        if not segments:
            return 0
        try:
            # Smart chunker: structure-aware with anchor chunks for decisions/actions
            chunks = chunk_transcript_smart(segments, meeting_id=meeting_id)
            if not chunks:
                return 0

            ident = self._identity(meeting_id, bharatvc_meeting_id=bharatvc_meeting_id, display_id=display_id,
                                   source=source)
            doc_type = DOC_TYPE_DOCUMENT_CHUNK if ident["source"] == SOURCE_DOCUMENT else DOC_TYPE_TRANSCRIPT_CHUNK

            by_id: Dict[str, Dict[str, Any]] = {}      # a repeated chunk_id would make ChromaDB reject the batch
            for chunk in chunks:
                by_id[_doc_id(meeting_id, chunk["chunk_id"])] = chunk
            ids = list(by_id)
            texts = [by_id[i]["text"] for i in ids]
            vectors = self.generate_embeddings_batch(texts)
            if len(vectors) != len(texts):
                vectors = [self.generate_embedding(t) for t in texts]

            metadatas = []
            for doc_id in ids:
                chunk = by_id[doc_id]
                meta = self._base_metadata(meeting_id, ident, chunk["chunk_id"], doc_type)
                meta.update(_clean_metadata({
                    "chunk_index": chunk.get("chunk_index", 0),
                    "chunk_type": chunk.get("chunk_type", "sliding"),
                    "section_type": chunk.get("section_type", "general"),
                    "start_time": str(chunk["start_time"]),
                    "end_time": str(chunk["end_time"]),
                    "speakers": chunk["speakers"],
                }))
                metadatas.append(meta)

            with self._op_lock:
                for start in range(0, len(ids), _UPSERT_BATCH_SIZE):
                    part = slice(start, start + _UPSERT_BATCH_SIZE)
                    self._chunks.upsert(ids=ids[part], embeddings=vectors[part], documents=texts[part],
                                        metadatas=metadatas[part])
                # New vectors are in place; only now drop this meeting's vectors that the transcript no longer
                # produces (whatever their identity), so retrieval never sees a moment with no chunks at all.
                existing = self._chunks.get(where={"meeting_id": int(meeting_id)}, include=[]).get("ids") or []
                stale = [i for i in existing if i not in by_id]
                for part in _batched(stale, _UPSERT_BATCH_SIZE):
                    self._chunks.delete(ids=part)
            logger.info(f"Indexed {len(ids)} transcript chunks for meeting {meeting_id}"
                        + (f" (removed {len(stale)} stale)" if stale else ""))
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to add transcript chunks: {e}")
            return 0

    def search_transcript_chunks(
        self,
        meeting_id: int,
        query_text: str,
        top_k: int = RAG_DEFAULT_TOP_K,
        cross_meeting: bool = False,
    ) -> List[Dict[str, Any]]:
        """Semantic search over ONE meeting's transcript chunks.

        Meeting isolation: the ChromaDB ``where`` filter is ALWAYS applied (meeting_id, plus the meeting's
        bharatvc_meeting_id when known) and the similarity ranking runs strictly within that filtered set; a
        result whose metadata names a different meeting is dropped as a second guard. ``cross_meeting`` is a
        deprecated no-op kept so old callers do not break.
        """
        matches: List[Dict[str, Any]] = []
        if not query_text or not query_text.strip() or not self._available():
            return matches

        try:
            if cross_meeting:
                logger.warning("search_transcript_chunks(cross_meeting=True) ignored - retrieval is restricted "
                               "to meeting %s.", meeting_id)
            query_vector = self.generate_embedding(query_text)
            where = self._scope(meeting_id)
            with self._op_lock:
                results = self._chunks.query(
                    query_embeddings=[query_vector],
                    n_results=max(1, top_k * 3),
                    where=where,
                    include=["metadatas", "distances", "documents"],
                )
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            documents = (results.get("documents") or [[]])[0]

            for idx, meta in enumerate(metadatas):
                meta = meta or {}
                if meta.get("meeting_id") != int(meeting_id):
                    continue
                dist = float(distances[idx]) if idx < len(distances) else 1.0
                # Cosine similarity: 1 - cosine_distance, normalized to [0, 1]
                similarity_score = round(max(0.0, min(1.0, 1.0 - (dist / 2.0))), 4)
                matches.append(
                    {
                        "meeting_id": meta.get("meeting_id"),
                        "start_time": meta.get("start_time"),
                        "end_time": meta.get("end_time"),
                        "speakers": meta.get("speakers"),
                        "similarity_score": similarity_score,
                        "text": documents[idx] if idx < len(documents) else "",
                        "chunk_id": meta.get("chunk_id"),
                        "chunk_index": meta.get("chunk_index", 0),
                    }
                )

            matches.sort(key=lambda x: x["similarity_score"], reverse=True)

            # Deduplicate: if multiple chunks from same speaker/time window, keep highest scoring
            seen = set()
            deduped_matches = []
            for match in matches:
                key = (match.get("speakers"), match.get("start_time"), match.get("end_time"))
                if key not in seen:
                    seen.add(key)
                    deduped_matches.append(match)
            matches = deduped_matches

        except Exception as e:
            logger.error(f"Failed to search transcript chunks: {e}")

        return matches[:top_k]

    def search_all_meetings(self, query_text: str, top_k: int = 10, min_similarity: float = 0.2) -> List[Dict[str, Any]]:
        """Semantic search across EVERY meeting's transcript chunks (the user-facing search page).

        * Raises ``VectorSearchUnavailable`` when the index cannot be queried - an empty list means "nothing
          relevant", never "the search failed" (``search_transcript_chunks`` returns [] for both).
        * Only chunks embedded with the CURRENT model are compared (``embedding_signature``): vectors from another
          model are not comparable and would rank at random. Re-indexed meetings rejoin the results.
        * ``similarity`` is the cosine similarity (1 - cosine distance). It measures topical closeness only: it is
          not a confidence that the passage answers the query, and never evidence that two recordings are the same.
        """
        if not query_text or not query_text.strip():
            return []
        if not self._available():
            raise VectorSearchUnavailable(self.init_error or "The search index is not available.")
        try:
            query_vector = self.generate_embedding(query_text.strip())
            with self._op_lock:
                results = self._chunks.query(
                    query_embeddings=[query_vector],
                    n_results=max(1, top_k * 3),
                    where={"embedding_signature": self._embedding_signature()},
                    include=["metadatas", "distances", "documents"],
                )
        except Exception as exc:
            logger.error("Cross-meeting search failed: %s", exc, exc_info=True)
            raise VectorSearchUnavailable(f"{type(exc).__name__}: {str(exc)[:160]}") from exc

        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        matches: List[Dict[str, Any]] = []
        seen = set()
        for idx, meta in enumerate(metadatas):
            meta = meta or {}
            similarity = round(1.0 - float(distances[idx]), 4) if idx < len(distances) else 0.0
            key = (meta.get("meeting_id"), meta.get("start_time"), meta.get("end_time"))
            if similarity < min_similarity or meta.get("meeting_id") is None or key in seen:
                continue
            seen.add(key)
            matches.append({
                "meeting_id": int(meta["meeting_id"]),
                "start_time": meta.get("start_time"),
                "end_time": meta.get("end_time"),
                "speakers": meta.get("speakers"),
                "similarity": similarity,
                "text": documents[idx] if idx < len(documents) else "",
            })
        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return matches[:top_k]


class VectorSearchUnavailable(RuntimeError):
    """The search index could not be queried (ChromaDB down / not initialized / embedding model unavailable)."""


vector_service = VectorService()

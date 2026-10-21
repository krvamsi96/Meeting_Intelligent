"""Deployment preflight: prove this machine can run the platform BEFORE (and after) it is started.

    python -m backend.scripts.preflight                  # full gate: every check, incl. live AIRAWAT + round trips
    python -m backend.scripts.preflight --startup        # fast checks run by start.sh / run_backend.sh on every boot
    python -m backend.scripts.preflight --skip-airawat   # no outbound AI calls
    python -m backend.scripts.preflight --json           # machine-readable report

What it checks (no check prints a secret: keys appear only as a short fingerprint, DB URLs are masked):

    python        interpreter version, virtualenv
    dependencies  every package the API imports (+ optional diarization stack)
    config        AI_MODE, AIRAWAT key/URLs/models, database selection (server vs FALLBACK), worker count, ChromaDB path
    directories   uploads / clips / ChromaDB / NeMo cache / logs exist, are writable, have disk space
    media         ffmpeg + ffprobe run; a generated video is converted to 16 kHz mono WAV by the app's own extractor;
                  a corrupt file is rejected
    language      a spoken-language default is configured for AIRAWAT Whisper (it cannot detect one), else uploads must name it
    diarization   the NeMo TitaNet speaker model is present and loads (speaker count), else the weak fallback is reported
    embedding     the model files are a real model (not a Git LFS pointer / truncated file) and produce finite,
                  correctly-sized, semantically meaningful vectors
    chromadb      persistent store opens, vectors written -> retrieved, per-meeting isolation, survives a close/reopen
    postgres      connects, schema creates/migrates, insert -> read -> update -> latest-output round trip for two
                  meetings that must not overwrite each other (all inside a transaction that is rolled back)
    airawat       LLM + Whisper endpoints answer with the configured models (retrying transient gateway stalls)

Exit code: 0 = no failures (warnings allowed), 1 = at least one failure. In ``--startup`` mode only failures that make
the API unable to run at all (python, dependencies, PostgreSQL) are fatal; the rest are reported as warnings because the
API starts and reports them in GET /system/health.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"
_ICON = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[SKIP]"}

# Fatal in --startup mode; everything else is downgraded to a warning there.
_STARTUP_CRITICAL = {"python", "dependencies", "postgres"}


@dataclass
class Result:
    check: str
    status: str
    message: str
    fix: str = ""
    details: Dict[str, object] = field(default_factory=dict)


class Report:
    def __init__(self, startup: bool):
        self.startup = startup
        self.results: List[Result] = []

    def add(self, check: str, status: str, message: str, fix: str = "", **details) -> None:
        if status == FAIL and self.startup and check not in _STARTUP_CRITICAL:
            status, message = WARN, f"{message} (non-fatal at startup)"
        self.results.append(Result(check, status, message, fix, details))
        icon = _ICON[status]
        print(f"  {icon} {check:<13} {message}", flush=True)
        if fix and status in (FAIL, WARN):
            print(f"         FIX: {fix}", flush=True)

    @property
    def failed(self) -> bool:
        return any(r.status == FAIL for r in self.results)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────────────────────

def _first_line(exc: BaseException, limit: int = 220) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {text}"[:limit]


def _version(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "?"


def _run(cmd: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)


def _is_temp_path(path: str) -> bool:
    real = os.path.realpath(path)
    roots = {os.path.realpath(tempfile.gettempdir())}
    if os.name != "nt":
        roots.update({"/tmp", "/var/tmp", "/dev/shm"})
    return any(real == r or real.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# checks
# ─────────────────────────────────────────────────────────────────────────────────────────────

def check_python(rep: Report) -> None:
    ver = sys.version_info
    label = f"Python {ver.major}.{ver.minor}.{ver.micro} on {platform.system()} {platform.machine()}"
    if ver < (3, 10):
        rep.add("python", FAIL, f"{label} - Python 3.10+ is required (3.11 is what the Docker image uses).",
                "Install python3.11 and recreate the virtualenv: python3.11 -m venv venv")
        return
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if not in_venv and not os.path.exists("/.dockerenv"):
        rep.add("python", WARN, f"{label} - not running inside a virtualenv.",
                "Use the project venv: source venv/bin/activate")
        return
    rep.add("python", OK, label + (" (virtualenv)" if in_venv else " (container)"))


# (import name, pip name)
_DEPENDENCIES = [
    ("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("multipart", "python-multipart"),
    ("sqlalchemy", "sqlalchemy"), ("psycopg2", "psycopg2-binary"), ("pydantic_settings", "pydantic-settings"),
    ("dotenv", "python-dotenv"), ("requests", "requests"), ("httpx", "httpx"),
    ("langchain_core", "langchain-core"), ("langchain_openai", "langchain-openai"),
    ("chromadb", "chromadb"), ("sentence_transformers", "sentence-transformers"),
    ("transformers", "transformers"), ("safetensors", "safetensors"), ("torch", "torch"),
    ("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
    ("soundfile", "soundfile"), ("cryptography", "cryptography"),
    ("docx", "python-docx"), ("pptx", "python-pptx"), ("openpyxl", "openpyxl"),
    ("pypdf", "pypdf"), ("nltk", "nltk"), ("yt_dlp", "yt-dlp"),
]


def check_dependencies(rep: Report) -> None:
    missing = []
    for module, pip_name in _DEPENDENCIES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - ImportError, but also broken native libraries
            missing.append(f"{pip_name} ({type(exc).__name__})")
    if missing:
        rep.add("dependencies", FAIL, "cannot import: " + ", ".join(missing),
                "pip install --upgrade pip && pip install torch --extra-index-url https://download.pytorch.org/whl/cpu "
                "&& pip install -r requirements.txt")
    else:
        rep.add("dependencies", OK, "all required packages import "
                f"(fastapi {_version('fastapi')}, sqlalchemy {_version('sqlalchemy')}, chromadb {_version('chromadb')}, "
                f"sentence-transformers {_version('sentence-transformers')}, transformers {_version('transformers')}, "
                f"safetensors {_version('safetensors')}, torch {_version('torch')})")

    on_linux = platform.system() == "Linux"
    try:
        importlib.import_module("nemo")
    except Exception as exc:  # noqa: BLE001
        rep.add("diarization", WARN if on_linux else SKIP,
                f"NeMo (TitaNet speaker embeddings) is not installed ({type(exc).__name__})" +
                (": speaker counts come from a weaker fallback embedder." if on_linux else " (Linux-only package)."),
                "pip install -r requirements.txt on the Linux server (installs nemo_toolkit[asr]), then "
                "python -m backend.scripts.fetch_models --diarization" if on_linux else "")
    try:
        import magic

        magic.Magic(mime=True).from_buffer(b"%PDF-1.4")
    except Exception as exc:  # noqa: BLE001 - ImportError, or libmagic missing
        rep.add("upload-check", WARN if on_linux else SKIP,
                f"python-magic / libmagic unavailable ({type(exc).__name__}): uploads are type-checked by file EXTENSION "
                "only (media content is still validated with ffprobe).",
                "sudo apt-get install -y libmagic1 && pip install python-magic" if on_linux else "")


def check_configuration(rep: Report) -> None:
    from backend.config.config import settings

    mode = (settings.AI_MODE or "auto").strip().lower()
    if mode == "local":
        rep.add("config", WARN, "AI_MODE=local: the LOCAL database and LOCAL_* credentials are used.",
                "Set AI_MODE=server on the production server so the server database is mandatory.")
    elif mode == "auto" and (os.getenv("SERVER_DATABASE_URL") or os.getenv("SERVER_DB_HOST")):
        rep.add("config", WARN, "AI_MODE=auto with a SERVER_* database configured: if it is unreachable at boot the "
                "app silently continues on the LOCAL database.",
                "Set AI_MODE=server in the production .env so the app refuses to start instead of falling back.")
    else:
        rep.add("config", OK, f"AI_MODE={mode}")

    # Universal AI Provider configuration
    try:
        from backend.providers import provider_factory
        active_prov = provider_factory.get_active_provider()
        prov_name = active_prov.name.title()
        if active_prov.is_configured:
            rep.add("config.ai_provider", OK, f"active AI provider: {prov_name} (configured)")
        else:
            rep.add("config.ai_provider", WARN, f"active AI provider '{prov_name}' requires configuration or connection.",
                    f"Configure {prov_name} in AI Settings UI or check server connection.")
    except Exception as exc:
        rep.add("config.ai_provider", WARN, f"AI provider configuration could not be evaluated: {_first_line(exc)}")

    try:
        from backend.services.airawat_health import _resolve_config, key_fingerprint
        _lcfg = _resolve_config()
        if _lcfg.get("api_key"):
            rep.add("config.api_key", OK, f"key fingerprint {key_fingerprint(_lcfg['api_key'])}")
    except Exception:
        pass

    if not settings.SSL_VERIFY:
        rep.add("config.tls", WARN, "SSL_VERIFY=false: TLS certificates are NOT verified.",
                "Fix ca-certificates on the server and set SSL_VERIFY=true")


    try:
        workers = int(os.getenv("UVICORN_WORKERS", os.getenv("WORKERS", "1")) or "1")
    except ValueError:
        workers = 1
    if workers > 1:
        rep.add("config.workers", FAIL, f"UVICORN_WORKERS={workers}: an embedded ChromaDB directory can be owned by one "
                "process only, so RAG / similar-meeting search would work in one worker and fail in the others.",
                "Set UVICORN_WORKERS=1")

    for name in ("SECRET_KEY", "ADMIN_API_KEY"):
        if not os.getenv(name, "").strip():
            what = ("encrypts provider keys saved in the database. Left empty, the launcher generates a NEW one whenever "
                    ".env is replaced (every Jenkins deploy injects it) or the container is rebuilt, and keys saved "
                    "earlier can no longer be decrypted") if name == "SECRET_KEY" else                 "protects POST /settings/* and /system/clear-cache (empty = open)"
            rep.add("config.secrets", WARN, f"{name} is empty: it {what}.",
                    "python -c \"import secrets; print(secrets.token_urlsafe(32))\" and put the result in the .env the deployment "
                    "injects (for Jenkins: the 'aimeet-production-env' credential)")


def check_directories(rep: Report) -> None:
    from backend.config.config import settings

    try:
        from backend.utils import logging_setup
        log_dir = getattr(logging_setup, "_LOG_DIR", "")
    except Exception:  # noqa: BLE001
        log_dir = ""

    targets = {
        "UPLOAD_DIR": settings.UPLOAD_DIR,
        "clips": os.path.join(settings.UPLOAD_DIR, settings.UPLOAD_CLIPS_DIR),
        "CHROMA_PERSIST_DIRECTORY": settings.CHROMA_PERSIST_DIRECTORY,
        "NEMO_CACHE_DIR": settings.NEMO_CACHE_DIR,
    }
    if log_dir:
        targets["logs"] = log_dir

    problems, notes = [], []
    for label, path in targets.items():
        try:
            os.makedirs(path, exist_ok=True)
            probe = os.path.join(path, f".preflight-{uuid.uuid4().hex[:8]}")
            with open(probe, "wb") as fh:
                fh.write(b"ok")
            os.remove(probe)
        except OSError as exc:
            problems.append(f"{label} ({path}): {exc.strerror or exc}")
            continue
        free_gb = shutil.disk_usage(path).free / 1e9
        notes.append(f"{label} {free_gb:.0f} GB free")
        if free_gb < 0.2:
            problems.append(f"{label} ({path}): only {free_gb:.2f} GB free")
    if problems:
        rep.add("directories", FAIL, "; ".join(problems),
                "Create the directories and give the service user write access: "
                "sudo mkdir -p <dir> && sudo chown -R $USER <dir>")
        return

    chroma = settings.CHROMA_PERSIST_DIRECTORY
    if _is_temp_path(chroma):
        rep.add("directories", FAIL, f"CHROMA_PERSIST_DIRECTORY ({chroma}) is inside a temporary directory: the vector "
                "database would be wiped by a reboot or a tmp cleaner.",
                "Use a persistent path, e.g. CHROMA_PERSIST_DIRECTORY=/var/lib/ai_meeting/chroma")
        return
    rep.add("directories", OK, "writable: " + "; ".join(notes), path=chroma)


def check_media(rep: Report, startup: bool) -> None:
    from backend.utils.media_tools import FFMPEG, FFPROBE, ffmpeg_available, ffprobe_available

    if not (ffmpeg_available() and ffprobe_available()):
        rep.add("media", FAIL, f"ffmpeg / ffprobe not found (looked for '{FFMPEG}' / '{FFPROBE}').",
                "sudo apt-get install -y ffmpeg   (or set FFMPEG_BINARY / FFPROBE_BINARY)")
        return
    try:
        first = _run([FFMPEG, "-version"], 20).stdout.splitlines()[0]
        _run([FFPROBE, "-version"], 20)
    except Exception as exc:  # noqa: BLE001
        rep.add("media", FAIL, f"ffmpeg does not run: {_first_line(exc)}", "Reinstall ffmpeg")
        return
    if startup:
        rep.add("media", OK, first[:70])
        return

    from backend.services.audio_service import extract_audio

    tmp = tempfile.mkdtemp(prefix="preflight_media_")
    try:
        video = os.path.join(tmp, "sample.mp4")
        wav = os.path.join(tmp, "sample.wav")
        made = _run([FFMPEG, "-y", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                     "testsrc=size=320x240:rate=15", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
                     "-t", "2", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", "-shortest", video], 120)
        if made.returncode != 0 or not os.path.isfile(video):
            rep.add("media", FAIL, "ffmpeg cannot encode a test video (libx264 / aac missing?): "
                    + (made.stderr or "").strip()[-160:], "Install the full ffmpeg package (not a minimal build)")
            return
        if not extract_audio(video, wav):
            rep.add("media", FAIL, "the app's audio extractor failed on a valid test video.",
                    "Run: ffmpeg -i sample.mp4 -vn -ar 16000 -ac 1 out.wav and read the error")
            return
        probe = json.loads(_run([FFPROBE, "-v", "error", "-show_streams", "-of", "json", wav], 30).stdout or "{}")
        stream = (probe.get("streams") or [{}])[0]
        dur = float(stream.get("duration") or 0)
        if int(stream.get("sample_rate", 0)) != 16000 or int(stream.get("channels", 0)) != 1 or not 1.5 <= dur <= 2.6:
            rep.add("media", FAIL, f"extracted audio has the wrong shape: {stream.get('sample_rate')} Hz, "
                    f"{stream.get('channels')} ch, {dur:.1f}s (expected 16000 Hz mono ~2s).", "Check the ffmpeg build")
            return
        bad = os.path.join(tmp, "corrupt.mp4")
        with open(bad, "wb") as fh:
            fh.write(os.urandom(4096))
        extractor_log = logging.getLogger("backend.services.audio_service")
        previous_level = extractor_log.level
        extractor_log.setLevel(logging.CRITICAL)          # the expected failure below is not an alarm
        try:
            corrupt_accepted = extract_audio(bad, os.path.join(tmp, "bad.wav"))
        finally:
            extractor_log.setLevel(previous_level)
        if corrupt_accepted:
            rep.add("media", FAIL, "a corrupt file was accepted as media (it must be rejected).", "Check the ffmpeg build")
            return
        rep.add("media", OK, f"{first[:60]} | video -> 16 kHz mono WAV ({dur:.1f}s) OK | corrupt file rejected")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_embedding(rep: Report, startup: bool) -> Optional[bool]:
    """Returns True when real semantic embeddings work (the ChromaDB check depends on it)."""
    from backend.services import model_loader as ml

    paths = [p for p in ml.get_embedding_model_search_paths() if os.path.isdir(p)]
    inspected = {p: ml.inspect_embedding_model_dir(p) for p in paths[:6]}
    usable = [p for p, problem in inspected.items() if problem is None]
    if not usable:
        why = "; ".join(f"{p}: {problem}" for p, problem in inspected.items()) or "no model directory found"
        fix = ml._fix_hint({p: pr for p, pr in inspected.items() if pr})   # noqa: SLF001 - same wording the API logs
        rep.add("embedding", FAIL, f"no usable embedding model - {why}", fix)
        return False
    if startup:
        rep.add("embedding", OK, f"model files look complete ({usable[0]})")
        return True

    t0 = time.monotonic()
    result = ml.validate_embedding_model()
    if not result.get("ok"):
        rep.add("embedding", FAIL, f"the model is present but does not produce valid embeddings: "
                f"{result.get('error') or result}", result.get("fix") or "Re-copy models/embeddings/all-MiniLM-L6-v2 "
                "(~91 MB) and retry", **{k: v for k, v in result.items() if k not in ("fix",)})
        return False
    rep.add("embedding", OK,
            f"{result['model']} via {result['engine']}: dimension {result['dimension']}, finite, unit-norm, "
            f"related pair {result['similarity_related']} > unrelated {result['similarity_unrelated']} "
            f"(loaded in {time.monotonic() - t0:.1f}s)")
    return True


def check_language(rep: Report) -> None:
    """Validate spoken language configuration for speech-to-text pipeline."""
    from backend.services.language_service import language_name, normalize_language_code
    from backend.services.model_router import model_router

    raw = model_router.get_effective_config().get("language")
    try:
        code = normalize_language_code(raw)
    except ValueError as exc:
        rep.add("language", FAIL, f"the configured default language is invalid: {exc}",
                "Set DEFAULT_WHISPER_LANGUAGE to an ISO 639-1 code (e.g. en, hi, te, ml)")
        return
    if code:
        rep.add("language", OK, f"default spoken language for speech-to-text: {language_name(code)} ({code}); "
                                "an upload's language field overrides it")
    else:
        rep.add("language", WARN, "no default spoken language: recordings without language field will default to 'en'",
                "Set DEFAULT_WHISPER_LANGUAGE in .env for recordings uploaded without a language")




def check_diarization_model(rep: Report, startup: bool) -> None:
    """Speaker COUNT comes from TitaNet speaker embeddings (NeMo). Without them a weak numpy fallback is used."""
    import importlib.util

    from backend.config.config import settings

    if importlib.util.find_spec("nemo") is None:
        return                                            # already reported by the dependencies check
    files = []
    for directory in (settings.DIARIZATION_MODEL_PATH, settings.NEMO_CACHE_DIR, os.path.expanduser("~/.cache/torch/NeMo")):
        if os.path.isdir(directory):
            for root, _dirs, names in os.walk(directory):
                files.extend(os.path.join(root, n) for n in names if n.endswith(".nemo"))
    if not files:
        rep.add("diarization", WARN, f"no NeMo speaker model ({settings.NEMO_SPEAKER_MODEL}.nemo) found in "
                f"{settings.DIARIZATION_MODEL_PATH} or the NeMo cache: with OFFLINE_MODE it cannot be downloaded, so "
                "speaker counts would come from the weak numpy fallback embedder.",
                "python -m backend.scripts.fetch_models --diarization  (once, with internet), then copy models/diarization/")
        return
    if startup:
        rep.add("diarization", OK, f"speaker model file found ({os.path.basename(files[0])})")
        return
    from backend.services import diarization_service as ds

    t0 = time.monotonic()
    ds._get_nemo_speaker_model()                          # noqa: SLF001 - the same loader the pipeline uses
    if ds.using_neural_embedder():
        rep.add("diarization", OK, f"TitaNet speaker embedder loaded ({os.path.basename(files[0])}, {time.monotonic() - t0:.0f}s)")
    else:
        rep.add("diarization", FAIL, "the NeMo speaker model file exists but did not load: the pipeline would use the "
                "weak numpy fallback (wrong speaker counts).", "Check logs above; re-fetch with "
                "python -m backend.scripts.fetch_models --diarization and match the installed nemo_toolkit version")


def check_chromadb(rep: Report, embeddings_ok: Optional[bool]) -> None:
    from backend.services.vector_service import vector_service

    st = vector_service.status()
    if st["state"] == "directory_in_use":
        rep.add("chromadb", WARN, "the ChromaDB directory is owned by a running API process; persistence round trip "
                "skipped (use GET /api/v1/system/health for its state).", "Stop the API first for a full check")
        return
    if st["status"] != "available":
        rep.add("chromadb", FAIL, f"ChromaDB unavailable ({st['state']}): {st['error']}",
                "Check CHROMA_PERSIST_DIRECTORY exists on persistent storage and is writable by the service user")
        return
    if not embeddings_ok:
        rep.add("chromadb", WARN, f"store opens at {st['path']} but the insert/retrieve test needs the embedding model.")
        return

    import numpy as np
    from backend.services.model_loader import encode_texts

    name = f"preflight_{uuid.uuid4().hex[:8]}"
    docs = [
        ("A1", 101, "The finance team agreed to approve the quarterly budget on Friday."),
        ("A2", 101, "Priya will send the revised budget spreadsheet to the auditors."),
        ("B1", 202, "The engineering team will migrate the database to the new cluster next month."),
        ("B2", 202, "Rahul owns the rollback plan for the PostgreSQL migration."),
    ]
    question = "Who is responsible for the database migration rollback?"
    try:
        vectors = encode_texts([d[2] for d in docs] + [question], strict=True)
        if vectors is None or not np.isfinite(vectors).all() or vectors.shape[1] != st["embedding_dimension"]:
            rep.add("chromadb", FAIL, "embedding output is not finite / has the wrong dimension.")
            return
        client = vector_service._client  # noqa: SLF001 - same access the check_vector_db script uses
        col = client.create_collection(name=name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)
        col.upsert(ids=[d[0] for d in docs], embeddings=vectors[:4].tolist(), documents=[d[2] for d in docs],
                   metadatas=[{"meeting_id": d[1]} for d in docs])
        qv = vectors[4].tolist()

        hit_b = col.query(query_embeddings=[qv], n_results=2, where={"meeting_id": 202}, include=["documents", "distances"])
        ids_b = hit_b["ids"][0]
        hit_a = col.query(query_embeddings=[qv], n_results=4, where={"meeting_id": 101}, include=["documents"])
        if ids_b[0] != "B2" or set(hit_a["ids"][0]) - {"A1", "A2"}:
            rep.add("chromadb", FAIL, f"retrieval is wrong: meeting 202 returned {ids_b}, meeting 101 returned "
                    f"{hit_a['ids'][0]} (expected B2 first and only A* for meeting 101).")
            return

        # restart persistence: close the store completely, open it again from disk, query the same collection
        vector_service.close()
        vector_service.ensure_initialized()
        if not vector_service.check_ready():
            rep.add("chromadb", FAIL, f"store did not reopen after close: {vector_service.init_error}")
            return
        col2 = vector_service._client.get_collection(name=name, embedding_function=None)  # noqa: SLF001
        again = col2.query(query_embeddings=[qv], n_results=1, where={"meeting_id": 202})
        if col2.count() != 4 or again["ids"][0] != ["B2"]:
            rep.add("chromadb", FAIL, "vectors did not survive closing and reopening the store.")
            return
        rep.add("chromadb", OK, f"persistent store at {st['path']}: 4 vectors written, semantic retrieval correct "
                "(B2 for the rollback question), meeting isolation holds, data survived close/reopen")
    except Exception as exc:  # noqa: BLE001
        rep.add("chromadb", FAIL, f"round trip failed: {_first_line(exc)}",
                "Check disk space / permissions on CHROMA_PERSIST_DIRECTORY and that only one process uses it")
    finally:
        try:
            vector_service._client.delete_collection(name)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass
        vector_service.close()


def check_postgres(rep: Report, startup: bool, init_schema: bool) -> None:
    try:
        from backend.database import connection as dbc
    except Exception as exc:  # noqa: BLE001 - "No database is available", bad driver, ...
        rep.add("postgres", FAIL, f"the database layer cannot start: {_first_line(exc, 300)}",
                "Set SERVER_DB_* / DATABASE_URL in .env, check the host is reachable (nc -vz host 5432), the user/"
                "password, pg_hba.conf, and sslmode if the server requires TLS (DB_SSLMODE=require)")
        return
    from sqlalchemy import inspect, text

    target = dbc.get_database_target()
    where = f"{target['environment']} {target['host']}:{target['port']}/{target['database']}"
    if target["fallback_active"]:
        rep.add("postgres", WARN if startup else FAIL,
                f"FALLBACK DATABASE in use ({where}): SERVER_* is configured but was unreachable.",
                "Fix connectivity to the server database, and set AI_MODE=server so the app refuses to start on a "
                "fallback database")
        # keep going: the fallback itself is still checked below
    try:
        with dbc.engine.connect() as conn:
            version = str(conn.execute(text("SELECT version()")).scalar()).split(",")[0]
            who = conn.execute(text("SELECT current_user, current_database()")).one()
            try:
                ssl = conn.execute(text("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()")).scalar()
            except Exception:  # noqa: BLE001 - view not readable for this role
                conn.rollback()
                ssl = None
    except Exception as exc:  # noqa: BLE001
        rep.add("postgres", FAIL, f"cannot connect to {where}: {_first_line(exc, 300)}",
                "Check host/port/credentials, firewall, pg_hba.conf and sslmode")
        return
    pool = dbc.get_engine_pool_status()
    head = (f"{where} as '{who[0]}' | {version} | TLS {'on' if ssl else 'off' if ssl is False else 'unknown'} | "
            f"pool {pool.get('pool_size')}+{pool.get('max_overflow')}")
    if startup:
        if not target["fallback_active"]:
            rep.add("postgres", OK, head)
        return

    from backend.models.models import AnalysisResult, Base, Meeting

    try:
        if init_schema:
            dbc.init_db()
        have = set(inspect(dbc.engine).get_table_names())
        missing = sorted(set(Base.metadata.tables) - have)
        if missing:
            rep.add("postgres", FAIL, f"tables missing: {', '.join(missing[:8])}",
                    "Start the API once (it creates/migrates the schema) or run: python -c \"from backend.database."
                    "connection import init_db; init_db()\" - the DB user needs CREATE on the schema")
            return

        # Round trip inside ONE transaction that is always rolled back, so no data is ever left behind.
        tag = uuid.uuid4().hex[:10]
        session = dbc.SessionLocal()
        try:
            def mk(suffix: str) -> Meeting:
                ident = f"9{int(time.time() * 1000)}{suffix}"
                return Meeting(bharatvc_meeting_id=ident, meeting_id=ident, display_id=ident, filename=f"preflight-{tag}-{suffix}.mp4",
                               file_type="media", filepath="/nonexistent", title=f"preflight {suffix}", status="uploaded")

            a, b = mk("1"), mk("2")
            session.add_all([a, b])
            session.flush()
            session.add_all([AnalysisResult(meeting_id=a.id, summary="summary of A"),
                             AnalysisResult(meeting_id=b.id, summary="summary of B")])
            session.flush()
            a.status, a.title = "completed", "preflight A (updated)"
            session.flush()
            session.expire_all()
            ra = session.query(Meeting).filter(Meeting.id == a.id).one()
            rb = session.query(Meeting).filter(Meeting.id == b.id).one()
            sa = session.query(AnalysisResult).filter(AnalysisResult.meeting_id == a.id).one()
            sb = session.query(AnalysisResult).filter(AnalysisResult.meeting_id == b.id).one()
            good = (ra.status == "completed" and ra.title.endswith("(updated)") and rb.status == "uploaded"
                    and rb.title == "preflight 2" and sa.summary == "summary of A" and sb.summary == "summary of B"
                    and ra.id != rb.id)
        finally:
            session.rollback()
            session.close()
        if not good:
            rep.add("postgres", FAIL, "insert/update/read round trip returned wrong data (meetings overwrote each other).")
            return
        with dbc.engine.connect() as conn:
            leftovers = conn.execute(text("SELECT count(*) FROM meetings WHERE filename LIKE :p"),
                                     {"p": f"preflight-{tag}-%"}).scalar()
        if leftovers:
            rep.add("postgres", FAIL, "the round trip left rows behind (the transaction was not rolled back).")
            return
        rep.add("postgres", OK, head + " | schema complete | insert -> read -> update -> latest-output round trip OK "
                "for two meetings (independent, rolled back)")
    except Exception as exc:  # noqa: BLE001
        rep.add("postgres", FAIL, f"connected ({where}) but the schema / round trip failed: {_first_line(exc, 300)}",
                "The DB user needs CREATE, INSERT, UPDATE, DELETE on the database's public schema")


def check_ai_provider(rep: Report, timeout: Optional[float] = None) -> None:
    try:
        from backend.providers import provider_factory
        active = provider_factory.get_active_provider()
        res = active.test_connection()
        if res.get("ok"):
            rep.add("ai_provider", OK, f"active AI provider '{active.name}' connected ({res.get('latency_ms', 0)} ms)")
        else:
            rep.add("ai_provider", WARN, f"active AI provider '{active.name}': {res.get('message', 'offline or unconfigured')}")
    except Exception as exc:
        rep.add("ai_provider", WARN, f"AI provider connection test: {_first_line(exc)}")



# ─────────────────────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--startup", action="store_true", help="fast boot-time subset; only python/deps/postgres are fatal")
    parser.add_argument("--skip-ai", "--skip-airawat", dest="skip_ai", action="store_true", help="do not call AI provider")
    parser.add_argument("--skip-chromadb", action="store_true", help="skip the ChromaDB round trip (e.g. API is running)")
    parser.add_argument("--no-init-db", action="store_true", help="do not create/migrate the schema (read-only DB check)")
    parser.add_argument("--timeout", type=float, default=None, help="seconds per AI request (default: from config)")
    parser.add_argument("--json", action="store_true", help="print a JSON report at the end")
    args = parser.parse_args(argv)

    rep = Report(startup=args.startup)
    print(("Startup preflight" if args.startup else "Deployment preflight") + f"  ({platform.node()}, {time.strftime('%Y-%m-%d %H:%M:%S')})")

    def guarded(name: str, fn: Callable[[], object]):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - a check must never crash the gate
            rep.add(name, FAIL, f"check crashed: {_first_line(exc, 300)}")
            return None

    guarded("python", lambda: check_python(rep))
    guarded("dependencies", lambda: check_dependencies(rep))
    if any(r.check == "dependencies" and r.status == FAIL for r in rep.results):
        print("  Stopping: install the dependencies first, the remaining checks import them.")
    else:
        guarded("config", lambda: check_configuration(rep))
        guarded("directories", lambda: check_directories(rep))
        guarded("media", lambda: check_media(rep, args.startup))
        embeddings_ok = guarded("embedding", lambda: check_embedding(rep, args.startup))
        guarded("language", lambda: check_language(rep))
        guarded("diarization", lambda: check_diarization_model(rep, args.startup))
        if not args.startup and not args.skip_chromadb:
            guarded("chromadb", lambda: check_chromadb(rep, embeddings_ok))
        guarded("postgres", lambda: check_postgres(rep, args.startup, not args.no_init_db))
        if not args.startup and not getattr(args, "skip_ai", False):
            guarded("ai_provider", lambda: check_ai_provider(rep, args.timeout))


    counts = {s: sum(1 for r in rep.results if r.status == s) for s in (OK, WARN, FAIL, SKIP)}
    print(f"\nRESULT: {counts[OK]} ok, {counts[WARN]} warning(s), {counts[FAIL]} failure(s)"
          + ("  - NOT READY" if rep.failed else "  - ready" + (" (review the warnings)" if counts[WARN] else "")))
    if args.json:
        print(json.dumps({"ready": not rep.failed, "counts": counts, "results": [asdict(r) for r in rep.results]},
                         indent=2, default=str))
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())

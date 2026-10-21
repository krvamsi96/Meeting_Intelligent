"""Centralized configuration management — all values from env vars with sensible defaults.

Environment File Priority (single-file policy)
---------------------------------------------
1. ENV_FILE env var (explicit override — Docker / CI)
2. .env   (the ONE file, with SERVER_* and LOCAL_* sections)
3. OS environment variables only (no file)

Environment selection (server vs local vs generic) is handled by
``backend.config.environment_selector.EnvironmentSelector`` which performs
real connectivity health checks at startup.
"""

import logging
import os
from functools import lru_cache
from typing import List, Optional, Set

from pydantic import ConfigDict
from pydantic_settings import BaseSettings

_config_logger = logging.getLogger(__name__)

# ─── Project root & CWD-independent path resolution ───────────────────────────
# Storage / model / log directories may be given as RELATIVE paths in .env
# (e.g. "./uploads"). On a Linux server started by systemd / PM2 / cron the
# working directory is often NOT the project root, so relative paths must be
# anchored to a stable base. PROJECT_ROOT defaults to the repository root
# (two levels above this file) and can be overridden with PROJECT_ROOT / BASE_DIR.
PROJECT_ROOT: str = os.path.abspath(
    os.getenv("PROJECT_ROOT")
    or os.getenv("BASE_DIR")
    or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _resolve_path(path: str, default: str) -> str:
    """Return an absolute path: expand ~, keep absolutes, anchor relatives to PROJECT_ROOT."""
    raw = (path or default or "").strip()
    if not raw:
        raw = default
    raw = os.path.expanduser(os.path.expandvars(raw))
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(PROJECT_ROOT, raw))


def _env_float(name: str, default: float) -> float:
    """``float(os.environ[name])``, or ``default`` when the variable is unset or blank."""
    raw = (os.getenv(name) or "").strip()
    return float(raw) if raw else default


def find_env_file() -> Optional[str]:
    """Locate the single .env file independent of the process working directory.

    Priority: ENV_FILE override (as given, else anchored to PROJECT_ROOT) →
    PROJECT_ROOT/.env → ./.env (legacy: whatever directory the process started in).
    Shared with environment_selector so pydantic-settings and os.environ agree.
    """
    custom_env = os.getenv("ENV_FILE")
    if custom_env:
        for candidate in (custom_env, os.path.join(PROJECT_ROOT, custom_env)):
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    for candidate in (os.path.join(PROJECT_ROOT, ".env"), os.path.abspath(".env")):
        if os.path.isfile(candidate):
            return candidate
    return None


# ─── Load .env into os.environ ONCE, before anything reads os.getenv() ─────────
# Many defaults below (and the thread caps) use os.getenv() at import time, and
# pydantic-settings alone does not export .env values into os.environ. Without
# this, a value that lives only in .env (e.g. AI_MAX_THREADS, DB_CONNECTION_TIMEOUT)
# was silently ignored. Real OS environment variables still win (override=False).
try:
    from dotenv import load_dotenv as _load_dotenv

    _dotenv_path = find_env_file()
    if _dotenv_path:
        _load_dotenv(dotenv_path=_dotenv_path, override=False)
except ImportError:  # python-dotenv missing: rely on the real environment
    pass

# ─── Thread / BLAS caps (must be set BEFORE numpy / torch / sklearn import) ────
# On a 1 GB / low-core server, OpenMP + MKL default to one worker thread per CPU
# and each carries its own arena / scratch buffers — tens to hundreds of MB of
# otherwise-idle RSS, plus oversubscription slowdowns under BackgroundTasks.
# Pin them to a small value (override with AI_MAX_THREADS). Accuracy is unaffected;
# only intra-op parallelism changes.
_max_threads = os.getenv("AI_MAX_THREADS", "") or str(min(4, os.cpu_count() or 2))
for _tvar in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TOKENIZERS_PARALLELISM",
):
    os.environ.setdefault(_tvar, "false" if _tvar == "TOKENIZERS_PARALLELISM" else _max_threads)


class Settings(BaseSettings):
    # ─── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = os.getenv("APP_NAME", "AI Meeting Intelligence Platform")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    # Bind all interfaces by default so the service is reachable on a server.
    # Override with HOST=127.0.0.1 for loopback-only local development.
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # ─── AI Mode ─────────────────────────────────────────────────────────────
    # AIRAWAT is the ONLY AI provider (speech-to-text and LLM) in EVERY mode. A failed AIRAWAT call is retried
    # (AIRAWAT_RETRY_COUNT) and then FAILS with a clear message: there is no other engine (no local Whisper, no
    # Ollama, no rule-based report stand-in). AI_MODE only selects which database / credential set is used:
    # "auto" (default; server database first, then local), "server", "cloud", "local".
    AI_MODE: str = os.getenv("AI_MODE", "auto")
    # Spoken language sent to AIRAWAT Whisper when an upload does not name one. The gateway cannot detect a
    # language (no code -> "400 need model or language", "auto" -> 404) and the code it is given decides the
    # output script, so it must be known. Per upload: form field `language` (e.g. hi, te, ml, en).
    # Empty = every upload must name its language.
    AIRAWAT_WHISPER_LANGUAGE: str = os.getenv("AIRAWAT_WHISPER_LANGUAGE", "").strip().lower()
    DEFAULT_WHISPER_LANGUAGE: str = os.getenv("DEFAULT_WHISPER_LANGUAGE", "").strip().lower()
    # The semantic embedding model is mandatory: with no usable copy the API reports it (health, logs) and
    # refuses to embed instead of storing deterministic word-hash vectors that only look like embeddings.
    # Set true ONLY for offline development where lexical vectors are an acceptable stand-in.
    EMBEDDING_ALLOW_LEXICAL_FALLBACK: bool = os.getenv("EMBEDDING_ALLOW_LEXICAL_FALLBACK", "false").lower() == "true"
    # The API refuses to start when the real embedding model cannot be verified / fetched (git lfs pull) / loaded.
    # false = start degraded (no semantic search, RAG or indexing) and report it in /system/health.
    EMBEDDING_MODEL_REQUIRED: bool = os.getenv("EMBEDDING_MODEL_REQUIRED", "true").lower() == "true"

    # ─── Dynamic Database Configuration ───────────────────────────────────────
    # All database configuration is 100% dynamic and environment-independent.
    # Supports discrete variables (DB_HOST, DB_PORT, DB_NAME/DB_DATABASE, DB_USER,
    # DB_PASSWORD) or full connection string (DATABASE_URL / DB_URL).
    DB_TYPE: str = os.getenv("DB_TYPE", os.getenv("DB_CLIENT", "postgresql"))
    DB_CLIENT: str = os.getenv("DB_CLIENT", os.getenv("DB_TYPE", "postgresql"))
    DB_HOST: str = os.getenv("DB_HOST", "")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_DATABASE: str = os.getenv("DB_DATABASE", os.getenv("DB_NAME", ""))
    DB_NAME: str = os.getenv("DB_NAME", os.getenv("DB_DATABASE", ""))
    DB_USER: str = os.getenv("DB_USER", os.getenv("DB_USERNAME", ""))
    DB_USERNAME: str = os.getenv("DB_USERNAME", os.getenv("DB_USER", ""))
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", os.getenv("DB_PASS", ""))
    DB_PASS: str = os.getenv("DB_PASS", os.getenv("DB_PASSWORD", ""))
    DB_SSL: bool = os.getenv("DB_SSL", "false").lower() == "true"
    DB_SSLMODE: str = os.getenv("DB_SSLMODE", "")
    DB_ECHO: bool = os.getenv("DB_ECHO", "false").lower() == "true"
    DB_URL: str = os.getenv("DB_URL", os.getenv("DATABASE_URL", ""))
    DATABASE_URL: str = os.getenv("DATABASE_URL", os.getenv("DB_URL", ""))

    # Storage paths are resolved to ABSOLUTE against PROJECT_ROOT so the app is
    # independent of the process working directory on the server.
    UPLOAD_DIR: str = _resolve_path(os.getenv("UPLOAD_DIR", ""), "uploads")

    # ─── Vector database ──────────────────────────────────────────────────────
    # ChromaDB (persistent, on-disk) is the ONLY vector store, locally and on the server; PostgreSQL
    # holds all application data. The path comes from the environment and is anchored to PROJECT_ROOT
    # when relative (see get_settings): local ".env" -> ./data/chroma, server -> e.g. /var/lib/<app>/chroma.
    # CHROMA_PATH is the deprecated name of CHROMA_PERSIST_DIRECTORY, honoured only when the new one is unset.
    VECTOR_DB: str = os.getenv("VECTOR_DB", "chromadb")
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "")
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "")
    # At startup, add every meeting missing from ChromaDB (e.g. processed before it was the vector store) in the
    # background, so similar-meeting / duplicate detection sees them. Idempotent; set false to skip.
    VECTOR_BACKFILL_ON_STARTUP: bool = os.getenv("VECTOR_BACKFILL_ON_STARTUP", "true").lower() == "true"

    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # Minimum cosine-similarity match score (0-100) required before a decision
    # is considered reliably linked to an agenda item. Below this, the pairing
    # is dropped from agenda_decision_mapping rather than shown as a forced
    # match with no real semantic evidence.
    AGENDA_DECISION_MATCH_THRESHOLD: float = float(
        os.getenv("AGENDA_DECISION_MATCH_THRESHOLD", "45.0")
    )
    # Cosine-similarity threshold (0-1) above which two agenda items or two
    # decisions extracted from the same meeting are treated as semantic
    # duplicates and merged.
    SEMANTIC_DEDUP_THRESHOLD: float = float(
        os.getenv("SEMANTIC_DEDUP_THRESHOLD", "0.87")
    )

    # ─── Meeting summary sizing (see services/summary_service.py) ─────────────
    # target_words = SUMMARY_LENGTH_COEFF * sqrt(meeting_words), clamped to
    # [SUMMARY_MIN_WORDS, SUMMARY_MAX_WORDS]: sub-linear growth so a 10x longer
    # meeting gets a ~3x longer summary, never a 10x longer one.
    SUMMARY_LENGTH_COEFF: float = float(os.getenv("SUMMARY_LENGTH_COEFF", "2.4"))
    SUMMARY_MIN_WORDS: int = int(os.getenv("SUMMARY_MIN_WORDS", "30"))
    SUMMARY_MAX_WORDS: int = int(os.getenv("SUMMARY_MAX_WORDS", "200"))
    # How many LLM rewrite passes a summary that fails validation gets before
    # the deterministic last-resort cleanup runs.
    SUMMARY_MAX_REPAIRS: int = int(os.getenv("SUMMARY_MAX_REPAIRS", "2"))

    # ─── Offline Model Paths & Enforced Offline Operation ─────────────────────
    OFFLINE_MODE: bool = os.getenv("OFFLINE_MODE", "true").lower() == "true"
    MODEL_BASE_PATH: str = _resolve_path(os.getenv("MODEL_BASE_PATH", ""), "models")
    EMBEDDING_MODEL_PATH: str = _resolve_path(
        os.getenv("EMBEDDING_MODEL_PATH", ""), "models/embeddings/all-MiniLM-L6-v2"
    )
    DIARIZATION_MODEL_PATH: str = _resolve_path(os.getenv("DIARIZATION_MODEL_PATH", ""), "models/diarization")

    # ─── AI providers ─────────────────────────────────────────────────────────
    # AI_PROVIDER picks the ONE engine that generates text (summary, MoM, agenda, decisions, action items, chat):
    #   airawat (default - the deployment's existing gateway), ollama (local, nothing leaves the host),
    #   groq / gemini (cloud: meeting text is sent to that company's API).
    # STT_PROVIDER picks speech-to-text: airawat or groq. Empty = AI_PROVIDER when that provider can transcribe;
    # with AI_PROVIDER=ollama or gemini it MUST be set, because neither transcribes here - the pipeline then
    # fails with a configuration error instead of quietly uploading the audio to some other service.
    # A failing provider is never replaced by another one: the call is retried, then the meeting fails.
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "airawat").strip().lower()
    STT_PROVIDER: str = os.getenv("STT_PROVIDER", "").strip().lower()

    # Ollama (local). The model must be installed (`ollama pull <model>`); the connection test reports it if not.
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_TIMEOUT: float = _env_float("OLLAMA_TIMEOUT_SEC", _env_float("OLLAMA_TIMEOUT", 120.0))
    OLLAMA_RETRY_COUNT: int = int(os.getenv("OLLAMA_RETRY_COUNT", "2"))
    OLLAMA_MAX_TOKENS: int = int(os.getenv("OLLAMA_MAX_TOKENS", "4096"))

    # Groq (cloud). Model ids change; the connection test checks the configured ones against Groq's live list.
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_WHISPER_MODEL: str = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_TIMEOUT: float = _env_float("GROQ_TIMEOUT_SEC", _env_float("GROQ_TIMEOUT", 60.0))
    GROQ_RETRY_COUNT: int = int(os.getenv("GROQ_RETRY_COUNT", "2"))
    GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "4096"))

    # Google Gemini (cloud, text generation only).
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_BASE_URL: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    GEMINI_TIMEOUT: float = _env_float("GEMINI_TIMEOUT_SEC", _env_float("GEMINI_TIMEOUT", 90.0))
    GEMINI_RETRY_COUNT: int = int(os.getenv("GEMINI_RETRY_COUNT", "2"))
    GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "4096"))

    # AIRAWAT gateway (speech-to-text + LLM)
    AIRAWAT_API_KEY: str = os.getenv("AIRAWAT_API_KEY", "")
    AIRAWAT_BASE_URL: str = os.getenv(
        "AIRAWAT_BASE_URL", "https://apis.airawat.cdac.in/nic/llama70b/v1"
    )
    AIRAWAT_WHISPER_URL: str = os.getenv(
        "AIRAWAT_WHISPER_URL",
        "https://apis.airawat.cdac.in/nic/Whisper-Large-v3/v1/audio/transcriptions",
    )
    AIRAWAT_LLM_MODEL: str = os.getenv(
        "AIRAWAT_LLM_MODEL", "meta/llama-3.3-70b-instruct"
    )
    AIRAWAT_WHISPER_MODEL: str = os.getenv("AIRAWAT_WHISPER_MODEL", "whisper-large-v3")

    CORS_ORIGINS: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:4200,http://127.0.0.1:4200,http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001,http://localhost:80,http://localhost,http://127.0.0.1",
    )
    # Whether CORS allows credentials (cookies/auth headers)
    CORS_ALLOW_CREDENTIALS: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

    # ─── Security Settings ────────────────────────────────────────────────────
    SSL_VERIFY: bool = os.getenv("SSL_VERIFY", "true").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    # Shared secret guarding administrative endpoints (AI-provider settings, diarization /
    # loop-engine tuning, cache clearing). Clients send it as `X-Admin-Key: <value>` (or
    # `Authorization: Bearer <value>`). When EMPTY those endpoints remain open for backward
    # compatibility and a warning is logged on use — set it in any shared deployment.
    ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "")
    # Per-client request ceilings for the expensive endpoints (requests per minute, 0 = off).
    RATE_LIMIT_CHAT_PER_MIN: int = int(os.getenv("RATE_LIMIT_CHAT_PER_MIN", "60"))
    RATE_LIMIT_UPLOAD_PER_MIN: int = int(os.getenv("RATE_LIMIT_UPLOAD_PER_MIN", "20"))
    # Behind a reverse proxy every request shares the proxy's address; set true ONLY when the
    # proxy sets X-Forwarded-For itself (otherwise a client could spoof it to dodge the limit).
    TRUST_FORWARDED_FOR: bool = os.getenv("TRUST_FORWARDED_FOR", "false").lower() == "true"
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
    # yt-dlp resolves URLs (and whatever a page links to) with its own HTTP stack, so the
    # per-hop SSRF validation in url_service.safe_request cannot cover it. It is therefore
    # limited to known video platforms unless this is explicitly enabled.
    YTDLP_ALLOW_ANY_HOST: bool = os.getenv("YTDLP_ALLOW_ANY_HOST", "false").lower() == "true"

    # ─── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    # Set LOG_FORMAT=json for structured JSON logging in production
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "text")

    # ─── Swagger / Docs ───────────────────────────────────────────────────────
    # Set to false in production to hide API documentation from public access.
    ENABLE_SWAGGER: bool = os.getenv("ENABLE_SWAGGER", "true").lower() == "true"

    # ─── AIRAWAT Timeouts & Retry ─────────────────────────────────────────────
    AIRAWAT_TIMEOUT_SEC: int = int(os.getenv("AIRAWAT_TIMEOUT_SEC", "600"))
    AIRAWAT_RETRY_COUNT: int = int(os.getenv("AIRAWAT_RETRY_COUNT", "2"))
    # AIRAWAT Whisper (speech-to-text) per-attempt timeouts. CONNECT is how long to wait for the TCP/TLS handshake;
    # READ is how long to wait for the transcript once the audio is uploaded. The read timeout grows with the chunk's
    # audio length (base + per-minute), capped at MAX so a hung request still gets retried before the gateway's own
    # ~300 s cut-off. Base defaults to AIRAWAT_TIMEOUT_SEC.
    AIRAWAT_WHISPER_CONNECT_TIMEOUT_SEC: float = float(os.getenv("AIRAWAT_WHISPER_CONNECT_TIMEOUT_SEC", "20"))
    AIRAWAT_WHISPER_READ_TIMEOUT_SEC: float = float(
        os.getenv("AIRAWAT_WHISPER_READ_TIMEOUT_SEC", os.getenv("AIRAWAT_TIMEOUT_SEC", "180"))
    )
    AIRAWAT_WHISPER_READ_TIMEOUT_PER_MIN_SEC: float = float(os.getenv("AIRAWAT_WHISPER_READ_TIMEOUT_PER_MIN_SEC", "10"))
    AIRAWAT_WHISPER_READ_TIMEOUT_MAX_SEC: float = float(os.getenv("AIRAWAT_WHISPER_READ_TIMEOUT_MAX_SEC", "300"))
    # AIRAWAT LLM calls are streamed so a request the gateway accepted but never answers is detected by silence
    # (no first byte / no new data for IDLE seconds) and retried, instead of costing the full AIRAWAT_TIMEOUT_SEC.
    # Measured: healthy streams start in 0.3-2 s (5.8 s for a 100k-char prompt); hung requests never send a byte.
    AIRAWAT_LLM_STREAMING: bool = os.getenv("AIRAWAT_LLM_STREAMING", "true").lower() == "true"
    AIRAWAT_LLM_STREAM_IDLE_TIMEOUT_SEC: float = float(os.getenv("AIRAWAT_LLM_STREAM_IDLE_TIMEOUT_SEC", "30"))
    # Longest wait for the response to START (headers) of a streamed LLM call; the gateway either answers within a few
    # seconds (measured max 1.8 s) or never. 0 disables it (the idle timeout above then also covers the start).
    AIRAWAT_LLM_FIRST_BYTE_TIMEOUT_SEC: float = float(os.getenv("AIRAWAT_LLM_FIRST_BYTE_TIMEOUT_SEC", "10"))
    # Run the NeMo speaker pass (audio only, CPU) while the speech-to-text request waits on AIRAWAT (network) instead of
    # after it. Saves up to min(Whisper wait, NeMo compute) per meeting (measured ~37 s) at the cost of a higher CPU/RAM
    # peak during that overlap: set false on a very small server to run them one after the other again.
    DIARIZATION_OVERLAP_WITH_STT: bool = os.getenv("DIARIZATION_OVERLAP_WITH_STT", "true").lower() == "true"
    AIRAWAT_LLM_CONNECT_TIMEOUT_SEC: float = float(os.getenv("AIRAWAT_LLM_CONNECT_TIMEOUT_SEC", "20"))
    # Exponential backoff between AIRAWAT Whisper attempts: BASE * 2^(attempt-1), capped at MAX, plus up to 20% jitter.
    AIRAWAT_RETRY_BACKOFF_BASE_SEC: float = float(os.getenv("AIRAWAT_RETRY_BACKOFF_BASE_SEC", "2"))
    AIRAWAT_RETRY_BACKOFF_MAX_SEC: float = float(os.getenv("AIRAWAT_RETRY_BACKOFF_MAX_SEC", "30"))
    # Default LLM generation parameters (overridable per-request)
    AIRAWAT_MAX_TOKENS: int = int(os.getenv("AIRAWAT_MAX_TOKENS", "2000"))
    AIRAWAT_TEMPERATURE: float = float(os.getenv("AIRAWAT_TEMPERATURE", "0.1"))

    # ─── Health Check ─────────────────────────────────────────────────────────
    # How long (seconds) to cache AIRAWAT connectivity result before re-validating
    AIRAWAT_HEALTH_TTL_SEC: int = int(os.getenv("AIRAWAT_HEALTH_TTL_SEC", "300"))
    # Timeout (seconds) for the health-check POST to /chat/completions.
    # Must be longer than the SSL handshake + inference time on the Linux server.
    # The AIRAWAT endpoint (GlobalSign intermediate CA) adds ~1-3 s SSL overhead on
    # government/NIC networks; a 10 s hard-code was too tight and caused false negatives.
    AIRAWAT_HEALTH_CHECK_TIMEOUT_SEC: int = int(os.getenv("AIRAWAT_HEALTH_CHECK_TIMEOUT_SEC", "30"))

    # ─── Database Connection Pool ─────────────────────────────────────────────
    # pool_size: persistent connections kept alive in the pool.
    # max_overflow: extra temporary connections allowed beyond pool_size.
    # The pipeline worker holds a DB session for up to 10-15 min during LLM calls;
    # to prevent exhaustion, pool_size + max_overflow must comfortably exceed the
    # maximum number of concurrent background pipeline tasks.
    # Defaults tuned for a 1 GB RAM single-worker Linux server. Each pooled
    # psycopg2 connection costs client-side buffers + a backend process on the
    # DB host; 5 + 10 is ample for BackgroundTasks that only hold a session
    # during short DB phases (the pipeline releases it during Whisper/LLM waits).
    # Scale up via env on larger hosts.
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "60"))  # was 30s — too tight
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))  # 30 min
    # PostgreSQL statement timeout in milliseconds (0 = disabled)
    DB_STATEMENT_TIMEOUT_MS: int = int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "0"))
    # Seconds to wait for a TCP connection to PostgreSQL. DB_CONNECTION_TIMEOUT is the
    # name the environment-selector health check has always read; honour both.
    DB_CONNECT_TIMEOUT: int = int(os.getenv("DB_CONNECT_TIMEOUT") or os.getenv("DB_CONNECTION_TIMEOUT") or "5")

    # ─── TCP Keepalive for long-running pipelines ──────────────────────────────
    # During AIRAWAT LLM processing (up to 10–15 min), the PostgreSQL connection
    # sits idle. Firewalls / NAT devices on NIC/government networks silently drop
    # idle TCP connections after ~4 min, causing:
    #   psycopg2.OperationalError: SSL SYSCALL error: Connection timed out
    # These settings tell the OS to send periodic TCP keepalive probes on every
    # psycopg2 connection so the TCP session stays alive across the LLM wait.
    #
    # DB_TCP_KEEPALIVES_IDLE    = seconds idle before first keepalive probe  (default 60)
    # DB_TCP_KEEPALIVES_INTERVAL = seconds between subsequent probes          (default 10)
    # DB_TCP_KEEPALIVES_COUNT   = how many missed probes before giving up     (default 5)
    DB_TCP_KEEPALIVES_IDLE: int = int(os.getenv("DB_TCP_KEEPALIVES_IDLE", "60"))
    DB_TCP_KEEPALIVES_INTERVAL: int = int(os.getenv("DB_TCP_KEEPALIVES_INTERVAL", "10"))
    DB_TCP_KEEPALIVES_COUNT: int = int(os.getenv("DB_TCP_KEEPALIVES_COUNT", "5"))

    # ─── Database Startup Retry (connection refused on boot) ──────────────────
    # When the OS starts PostgreSQL after this process (common on reboot),
    # init_db() retries with exponential back-off instead of crash-looping.
    #
    # DB_CONNECT_RETRIES         = total connection attempts before giving up  (default 5)
    # DB_CONNECT_RETRY_DELAY_S   = initial wait in seconds; doubles each retry  (default 2)
    # DB_CONNECT_RETRY_MAX_DELAY_S = maximum wait cap per retry               (default 60)
    #
    # Total worst-case wait with defaults: 2+4+8+16+60 = 90 s before app exits
    # (PM2 will then restart, giving another 90 s window).
    DB_CONNECT_RETRIES: int = int(os.getenv("DB_CONNECT_RETRIES", "5"))
    DB_CONNECT_RETRY_DELAY_S: float = float(os.getenv("DB_CONNECT_RETRY_DELAY_S", "2.0"))
    DB_CONNECT_RETRY_MAX_DELAY_S: float = float(os.getenv("DB_CONNECT_RETRY_MAX_DELAY_S", "60.0"))

    # ─── Report Customization (enterprise branding) ───────────────────────────
    REPORT_COMPANY_NAME: str = os.getenv("REPORT_COMPANY_NAME", "AI Meeting Intelligence Platform")
    REPORT_COMPANY_ADDRESS: str = os.getenv("REPORT_COMPANY_ADDRESS", "")
    REPORT_HEADER_TEXT: str = os.getenv("REPORT_HEADER_TEXT", "Confidential - Business Report")
    REPORT_FOOTER_TEXT: str = os.getenv("REPORT_FOOTER_TEXT", "AI Meeting Intelligence Platform | Powered by AIRAWAT")

    # ─── Similarity & RAG Config ─────────────────────────────────────────────
    DEFAULT_SIMILARITY_THRESHOLD: float = float(
        os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.20")
    )
    DEFAULT_SIMILARITY_TOP_K: int = int(os.getenv("DEFAULT_SIMILARITY_TOP_K", "10"))

    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "4"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "2"))
    RAG_DEFAULT_TOP_K: int = int(os.getenv("RAG_DEFAULT_TOP_K", "5"))
    RAG_MAX_CONTEXT_CHARS: int = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "15000"))

    # ─── Context Windows ──────────────────────────────────────────────────────
    CLOUD_MAX_LLM_TOKENS: int = int(os.getenv("CLOUD_MAX_LLM_TOKENS", "24000"))
    LOCAL_MAX_LLM_TOKENS: int = int(os.getenv("LOCAL_MAX_LLM_TOKENS", "1500"))
    CLOUD_MAX_TRANSCRIPT_CHARS: int = int(
        os.getenv("CLOUD_MAX_TRANSCRIPT_CHARS", "96000")
    )
    LOCAL_MAX_TRANSCRIPT_CHARS: int = int(
        os.getenv("LOCAL_MAX_TRANSCRIPT_CHARS", "16000")
    )

    # ─── Audio / Media ────────────────────────────────────────────────────────
    # Maximum PCM bytes targeted per Whisper API request. The provider hard limit
    # is 25 MB; the AIRAWAT/NIC gateway is flakier near its ceiling and can answer
    # an oversized upload with HTTP 200 + an EMPTY body (no error), which silently
    # produced empty transcripts on long (multi-chunk) videos. 15 MB leaves room
    # for the WAV header + multipart encoding overhead and keeps every chunk well
    # inside the gateway's reliable range. Override with CLOUD_MAX_BYTES.
    CLOUD_MAX_BYTES: int = int(os.getenv("CLOUD_MAX_BYTES", str(15 * 1024 * 1024)))
    # Hard ceiling accepted by the Whisper endpoint — a produced chunk larger than
    # this is always re-split, never uploaded.
    CLOUD_HARD_MAX_BYTES: int = int(os.getenv("CLOUD_HARD_MAX_BYTES", str(24 * 1024 * 1024)))
    # Extra internal retries for a chunk that returns HTTP 2xx but no usable text
    # (transient gateway emptiness). Independent of AIRAWAT_RETRY_COUNT.
    CLOUD_EMPTY_RESPONSE_RETRIES: int = int(os.getenv("CLOUD_EMPTY_RESPONSE_RETRIES", "2"))
    # 16KHz mono PCM s16le = 32,000 bytes/sec
    BYTES_PER_SEC_WAV: int = int(os.getenv("BYTES_PER_SEC_WAV", "32000"))
    # Overlap between audio chunks to prevent boundary drift (seconds)
    CHUNK_OVERLAP_SEC: float = float(os.getenv("CHUNK_OVERLAP_SEC", "0.5"))
    # Upload file streaming chunk size in bytes (default 4 MB)
    UPLOAD_CHUNK_SIZE_BYTES: int = int(os.getenv("UPLOAD_CHUNK_SIZE_BYTES", str(4 * 1024 * 1024)))
    # Subdirectory under UPLOAD_DIR for audio clip exports
    UPLOAD_CLIPS_DIR: str = os.getenv("UPLOAD_CLIPS_DIR", "clips")

    # ─── Cross-meeting search ─────────────────────────────────────────────────
    # Passages below this cosine similarity are not shown (all-MiniLM-L6-v2: unrelated text scores ~0.0-0.15).
    SEARCH_MIN_SIMILARITY: float = _env_float("SEARCH_MIN_SIMILARITY", 0.2)

    # ─── Transcript segment length ────────────────────────────────────────────
    # A same-speaker run (a one-speaker meeting is ONE run) is cut at sentence ends once it is this long, so a 30-minute
    # monologue is not stored as a single 21 000-character "segment". Text is never changed, only where it is cut.
    TRANSCRIPT_MAX_SEGMENT_SEC: float = float(os.getenv("TRANSCRIPT_MAX_SEGMENT_SEC", "45"))
    TRANSCRIPT_MAX_SEGMENT_WORDS: int = int(os.getenv("TRANSCRIPT_MAX_SEGMENT_WORDS", "120"))

    # ─── Transcript Post-Processing & Candidate Scoring Weights ─────────────
    TRANSCRIPT_CORRECTION_ENABLED: bool = os.getenv("TRANSCRIPT_CORRECTION_ENABLED", "true").lower() == "true"
    CORRECTION_CONFIDENCE_THRESHOLD: float = float(os.getenv("CORRECTION_CONFIDENCE_THRESHOLD", "0.70"))
    CORRECTION_WEIGHT_PHONETIC: float = float(os.getenv("CORRECTION_WEIGHT_PHONETIC", "0.25"))
    CORRECTION_WEIGHT_TEXT: float = float(os.getenv("CORRECTION_WEIGHT_TEXT", "0.25"))
    CORRECTION_WEIGHT_CONTEXT: float = float(os.getenv("CORRECTION_WEIGHT_CONTEXT", "0.25"))
    CORRECTION_WEIGHT_DOMAIN: float = float(os.getenv("CORRECTION_WEIGHT_DOMAIN", "0.15"))
    CORRECTION_WEIGHT_ASR: float = float(os.getenv("CORRECTION_WEIGHT_ASR", "0.10"))

    # ─── AI Evaluation Thresholds & Weights ─────────────────────────────────────
    CRITICAL_THRESHOLD: float = float(os.getenv("CRITICAL_THRESHOLD", "50.0"))
    EVALUATION_WEIGHTS: dict = {
        "speech_recognition_quality": 0.15,
        "speaker_consistency": 0.10,
        "summary_grounding": 0.15,
        "mom_quality": 0.10,
        "action_item_quality": 0.10,
        "topic_coverage": 0.08,
        "sentiment_confidence": 0.07,
        "similarity_confidence": 0.05,
        "audio_signal_quality": 0.10,
        "video_signal_quality": 0.10,
    }

    # ─── Database Defaults (used to backfill null metadata) ───────────────────
    DEFAULT_AGENDA_TEXT: str = os.getenv(
        "DEFAULT_AGENDA_TEXT", "Discussion and review of session recording"
    )
    DEFAULT_PARTICIPANTS_TEXT: str = os.getenv("DEFAULT_PARTICIPANTS_TEXT", "Meeting Participants")
    # Deployment timezone — used for action-item deadline resolution and date display.
    # Must be a valid IANA timezone string (e.g. "Asia/Kolkata", "UTC", "America/New_York").
    DEFAULT_TIMEZONE: str = os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata")
    # BUG-08 FIX: llm_workflow.py reads APP_TIMEZONE but only DEFAULT_TIMEZONE was defined.
    # This alias ensures deadline resolution uses the configured timezone (IST by default)
    # instead of silently falling back to UTC.
    APP_TIMEZONE: str = os.getenv("APP_TIMEZONE", os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata"))

    # ─── NVIDIA NeMo & PyAnnote Speaker Diarization Strategy ───────────────────
    NEMO_ENABLED: bool = os.getenv("NEMO_ENABLED", "true").lower() == "true"
    NEMO_DEVICE: str = os.getenv("NEMO_DEVICE", "auto")
    NEMO_CACHE_DIR: str = _resolve_path(os.getenv("NEMO_CACHE_DIR", ""), "nemo_cache")
    NEMO_SPEAKER_MODEL: str = os.getenv("NEMO_SPEAKER_MODEL", "titanet_large")
    NEMO_VAD_MODEL: str = os.getenv("NEMO_VAD_MODEL", "vad_multilingual_marblenet")
    ENABLE_LLM_SPEAKER_LABELS: bool = os.getenv("ENABLE_LLM_SPEAKER_LABELS", "true").lower() == "true"
    PYANNOTE_ENABLED: bool = os.getenv("PYANNOTE_ENABLED", "true").lower() == "true"
    PYANNOTE_MODEL: str = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
    # SECURITY: No default token — must be set in .env or environment.
    # If PyAnnote is enabled (PYANNOTE_ENABLED=true), HF_TOKEN is required.
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    DIARIZATION_MODE: str = os.getenv("DIARIZATION_MODE", "NEMO_PRIMARY_PYANNOTE_FALLBACK")
    # Speaker-count search bounds. The engine determines the ACTUAL number of
    # speakers dynamically inside [MIN, MAX]; these are only search limits, never
    # a forced/hardcoded count. MIN=1 so a genuine solo recording stays 1.
    DIARIZATION_MIN_SPEAKERS: int = int(os.getenv("DIARIZATION_MIN_SPEAKERS", "1"))
    DIARIZATION_MAX_SPEAKERS: int = int(os.getenv("DIARIZATION_MAX_SPEAKERS", "8"))
    # FALLBACK ONLY: agglomerative cosine-distance cut used when the eigen-based
    # speaker-count estimator (NME-SC, below) is unavailable. Same speaker
    # ≈ 0.15-0.35 apart, different speakers ≈ 0.8-0.95 apart.
    DIARIZATION_NEURAL_DIST_THRESHOLD: float = float(os.getenv("DIARIZATION_NEURAL_DIST_THRESHOLD", "0.62"))
    # Two clusters whose centroids are closer than this cosine distance are the
    # SAME person recorded under slightly different conditions -> merged.
    # CLUSTER_MERGE_COSINE applies to the weak numpy acoustic fallback, whose
    # vectors are all alike. NEURAL_MERGE_COSINE applies to TitaNet: centroids
    # are means over many windows, so two DIFFERENT people with similar voices
    # already reach cosine similarity 0.5-0.7 (distance 0.3-0.5) - merging at 0.5
    # fused real speakers (4 -> 3, 2 -> 1 on real meetings) - while one voice
    # split in two sits at similarity ~0.9. Only genuine duplicates are merged.
    DIARIZATION_CLUSTER_MERGE_COSINE: float = float(os.getenv("DIARIZATION_CLUSTER_MERGE_COSINE", "0.50"))
    DIARIZATION_NEURAL_MERGE_COSINE: float = float(os.getenv("DIARIZATION_NEURAL_MERGE_COSINE", "0.30"))
    # A candidate speaker with less than this much total speech (seconds) AND a
    # trivial share of the talk time is treated as noise / mis-split and folded
    # into the acoustically nearest real speaker.
    DIARIZATION_MIN_SPEAKER_TALK_SEC: float = float(os.getenv("DIARIZATION_MIN_SPEAKER_TALK_SEC", "3.0"))
    # Artefact filter applied AFTER the NME-SC count estimate: a k >= 2 result is
    # kept only when at least two clusters each hold the floor for a real turn
    # (>= MIN_SPEAKER_TALK_SEC of speech AND a contiguous run of
    # >= NEURAL_MIN_SUSTAINED_RUN embedding windows), OR the partition silhouette
    # reaches NEURAL_STRONG_SIL (a genuinely bimodal embedding space). Otherwise a
    # lone talker fragmented by embedding noise collapses back to a single speaker.
    DIARIZATION_NEURAL_MIN_SUSTAINED_RUN: int = int(os.getenv("DIARIZATION_NEURAL_MIN_SUSTAINED_RUN", "2"))
    DIARIZATION_NEURAL_STRONG_SIL: float = float(os.getenv("DIARIZATION_NEURAL_STRONG_SIL", "0.45"))
    # The speaker COUNT is estimated by auto-tuning spectral clustering
    # (Normalized Maximum Eigengap) over the embedding-affinity graph — no fixed
    # or assumed number. On very long meetings the O(p·n^3) eigen search is run
    # on at most this many uniformly sub-sampled windows (the final labelling
    # still uses every window); raise for more precision, lower for speed.
    DIARIZATION_NMESC_MAX_WINDOWS: int = int(os.getenv("DIARIZATION_NMESC_MAX_WINDOWS", "1500"))
    # Energy-VAD margin (dB above the adaptive noise floor) for a frame to count
    # as speech. Lower = more permissive (keeps quiet talkers, risks noise).
    DIARIZATION_VAD_MARGIN_DB: float = float(os.getenv("DIARIZATION_VAD_MARGIN_DB", "9.0"))
    # Speech does not reach further than this many dB below its own loud frames; anything quieter is room tone and is
    # not speech. Keeps the VAD working when the recording's quietest frames are digital silence (a -111..-200 dB
    # "noise floor" used to switch the VAD off). Raise it if very quiet talkers are being cut.
    DIARIZATION_VAD_MAX_RANGE_DB: float = float(os.getenv("DIARIZATION_VAD_MAX_RANGE_DB", "45.0"))
    # A k >= 2 split in which every speaker is ONE unbroken stretch of time (A..A then B..B - nobody ever answers back)
    # and the voices are only this close (centroid cosine distance) is rejected: it is what ONE voice looks like after
    # the microphone, room or phone line changes mid-recording. Different real people sit at >= 0.6.
    DIARIZATION_NEURAL_SINGLE_TURN_SEP: float = float(os.getenv("DIARIZATION_NEURAL_SINGLE_TURN_SEP", "0.50"))
    # Word times from the speech-to-text drift by up to a second near segment ends (and are spread evenly across a
    # segment when the response has no word times), which cut a sentence in two and credited its tail to the NEXT speaker.
    # With this on, only a short fragment at the edge of a sentence (<= 2 s, <= 40 % of it) is given back to the sentence's
    # main speaker; a reply that is a sentence of its own is never touched. Set false to use the raw word/turn overlap.
    DIARIZATION_SENTENCE_SMOOTHING: bool = os.getenv("DIARIZATION_SENTENCE_SMOOTHING", "true").lower() == "true"

    # ─── Voice enrollment & speaker identification ────────────────────────────
    # A participant is registered once (name + a reference voice sample); the
    # same TitaNet embedder used for diarization turns their voice into a stored
    # profile. During meeting processing each diarized speaker's mean embedding
    # is compared (cosine) to every profile. A name is assigned ONLY when the
    # best match clears VOICE_MATCH_THRESHOLD *and* beats the runner-up by
    # VOICE_MATCH_MARGIN — otherwise the speaker stays "Speaker N". Nothing is
    # ever guessed; no names or embeddings are hard-coded.
    VOICE_ENROLLMENT_ENABLED: bool = os.getenv("VOICE_ENROLLMENT_ENABLED", "true").lower() == "true"
    VOICE_PROFILE_DIR: str = _resolve_path(os.getenv("VOICE_PROFILE_DIR", ""), "uploads/voice_profiles")
    # Cosine similarity (TitaNet, L2-normalised) a diarized speaker must reach
    # against a stored profile to be named. Same speaker across recordings
    # typically 0.55-0.85; different speakers 0.0-0.4.
    VOICE_MATCH_THRESHOLD: float = float(os.getenv("VOICE_MATCH_THRESHOLD", "0.62"))
    # The best profile must also beat the second-best by at least this margin,
    # so an ambiguous voice is left unnamed rather than mis-assigned.
    VOICE_MATCH_MARGIN: float = float(os.getenv("VOICE_MATCH_MARGIN", "0.06"))
    # Minimum seconds of detected speech a reference sample / a diarized speaker
    # must contain for a reliable embedding.
    VOICE_MIN_ENROLL_SPEECH_SEC: float = float(os.getenv("VOICE_MIN_ENROLL_SPEECH_SEC", "3.0"))
    VOICE_MIN_MATCH_SPEECH_SEC: float = float(os.getenv("VOICE_MIN_MATCH_SPEECH_SEC", "2.0"))

    # ─── Multilingual pipeline (any spoken language → English output) ─────────
    # The spoken language is auto-detected; the user never selects it. ALL
    # user-facing output (transcript, summary, MoM, action items, agenda,
    # decisions, topics, sentiment text, chatbot answers, API response) is
    # produced in OUTPUT_LANGUAGE. No source language is hardcoded anywhere.
    MULTILINGUAL_ENABLED: bool = os.getenv("MULTILINGUAL_ENABLED", "true").lower() == "true"
    TRANSLATION_ENABLED: bool = os.getenv("TRANSLATION_ENABLED", "true").lower() == "true"
    OUTPUT_LANGUAGE: str = os.getenv("OUTPUT_LANGUAGE", "en")
    # Transcript segments per LLM translation request, and per-batch retries.
    TRANSLATION_BATCH_SIZE: int = int(os.getenv("TRANSLATION_BATCH_SIZE", "20"))
    TRANSLATION_MAX_RETRIES: int = int(os.getenv("TRANSLATION_MAX_RETRIES", "2"))

    # ─── Dynamic Universal Loop Engineering Configuration ─────────────────────
    LOOP_ENGINE_ENABLED: bool = os.getenv("LOOP_ENGINE_ENABLED", "true").lower() == "true"
    MAX_STAGE_RETRIES: int = int(os.getenv("MAX_STAGE_RETRIES", "2"))
    QUALITY_VALIDATION_ENABLED: bool = os.getenv("QUALITY_VALIDATION_ENABLED", "true").lower() == "true"

    @property
    def cors_origin_list(self) -> List[str]:
        """Parse CORS_ORIGINS into a list of clean origin strings."""
        return [
            origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()
        ]

    @property
    def backend_base_url(self) -> str:
        """Canonical backend URL for use in CSP and health checks."""
        return f"http://{self.HOST}:{self.PORT}"

    @property
    def dynamic_database_url(self) -> str:
        """Construct or return the validated dynamic database URL."""
        return build_database_url(
            database_url=self.DATABASE_URL or self.DB_URL,
            db_type=self.DB_TYPE or self.DB_CLIENT,
            db_host=self.DB_HOST,
            db_port=self.DB_PORT,
            db_database=self.DB_DATABASE or self.DB_NAME,
            db_user=self.DB_USER or self.DB_USERNAME,
            db_password=self.DB_PASSWORD or self.DB_PASS,
            db_ssl=self.DB_SSL,
            db_sslmode=self.DB_SSLMODE,
        )

    def _get_active_env_file() -> Optional[str]:
        """Return the single .env file path.

        Priority:
        1. ENV_FILE env var (explicit override — useful in Docker / CI)
        2. PROJECT_ROOT/.env   ← the ONE env file, containing SERVER_* and LOCAL_* sections
        3. None   → rely on OS environment variables only

        NOTE: Environment selection (which credentials to USE — server vs local)
        is a runtime decision made by backend.config.environment_selector, NOT
        by which file is loaded. ALL credentials live in the single .env file.
        """
        env_file = find_env_file()
        if env_file:
            _config_logger.debug("[Config] Loading env file: %s", env_file)
            return env_file
        # No file — OS environment variables only (Docker/systemd/PM2 inject them)
        _config_logger.warning("[Config] No .env file found. Using OS environment variables only.")
        return None

    model_config = ConfigDict(
        env_file=_get_active_env_file(),
        case_sensitive=True,
        extra="ignore",
    )


def with_psycopg2_driver(url: str) -> str:
    """Name the psycopg2 driver explicitly when a PostgreSQL URL does not name one.

    SQLAlchemy 2.1 changed the default driver for a bare ``postgresql://`` URL from psycopg2
    to psycopg (v3). This project installs psycopg2-binary, so on a fresh ``pip install`` every
    ``postgresql://`` URL from .env failed with ``ModuleNotFoundError: No module named
    'psycopg'``. URLs that already choose a driver (``postgresql+psycopg://``, ``+asyncpg`` …)
    are left alone.
    """
    if not url:
        return url
    lowered = url.lower()
    for bare in ("postgresql://", "postgres://"):
        if lowered.startswith(bare):
            return "postgresql+psycopg2://" + url[len(bare):]
    return url


def mask_database_url(url: str) -> str:
    """Mask credentials in database URL for safe logging."""
    if not url:
        return "<empty-database-url>"
    try:
        if "://" in url and "@" in url:
            scheme, rest = url.split("://", 1)
            user_pass, host_db = rest.rsplit("@", 1)
            if ":" in user_pass:
                user, _ = user_pass.split(":", 1)
                return f"{scheme}://{user}:***@{host_db}"
            return f"{scheme}://***@{host_db}"
        return url
    except Exception:
        return "<masked-db-url>"


def build_database_url(
    database_url: str = "",
    db_type: str = "postgresql",
    db_host: str = "",
    db_port: int = 5432,
    db_database: str = "",
    db_user: str = "",
    db_password: str = "",
    db_ssl: bool = False,
    db_sslmode: str = "",
) -> str:
    """Dynamically construct and validate a PostgreSQL database URL.

    Ensures URL encoding of passwords with special characters and supports
    both direct URL and discrete host/port/database/user/password variables.
    """
    import urllib.parse

    raw_url = (database_url or "").strip()
    if raw_url:
        if "://" in raw_url and "@" in raw_url:
            scheme, rest = raw_url.split("://", 1)
            user_pass, host_db = rest.rsplit("@", 1)
            if ":" in user_pass:
                user, raw_pass = user_pass.split(":", 1)
                unquoted_pass = urllib.parse.unquote(raw_pass)
                quoted_pass = urllib.parse.quote(unquoted_pass, safe="")
                unquoted_user = urllib.parse.unquote(user)
                quoted_user = urllib.parse.quote(unquoted_user, safe="")
                raw_url = f"{scheme}://{quoted_user}:{quoted_pass}@{host_db}"
        return with_psycopg2_driver(raw_url)

    host = (db_host or "").strip()
    database = (db_database or "").strip()
    user = (db_user or "").strip()
    password = str(db_password or "")

    if not host or not database:
        raise ValueError(
            "Database configuration error: Missing mandatory database parameters. "
            "Provide DATABASE_URL, or set DB_HOST and DB_DATABASE / DB_NAME in .env."
        )

    scheme = "postgresql"
    if db_type and "postgres" in db_type.lower():
        scheme = "postgresql"
    elif db_type:
        scheme = db_type.lower()

    quoted_user = urllib.parse.quote(user, safe="") if user else ""
    quoted_password = urllib.parse.quote(password, safe="") if password else ""

    auth_part = ""
    if quoted_user and quoted_password:
        auth_part = f"{quoted_user}:{quoted_password}@"
    elif quoted_user:
        auth_part = f"{quoted_user}@"

    port_part = f":{db_port}" if db_port else ""
    constructed = f"{scheme}://{auth_part}{host}{port_part}/{database}"

    ssl_params = []
    if db_sslmode:
        ssl_params.append(f"sslmode={db_sslmode}")
    elif db_ssl:
        ssl_params.append("sslmode=require")

    if ssl_params:
        separator = "&" if "?" in constructed else "?"
        constructed += f"{separator}{'&'.join(ssl_params)}"

    return with_psycopg2_driver(constructed)


DOCUMENT_EXTENSIONS: Set[str] = {
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".txt", ".csv", ".rtf", ".md", ".json", ".log"
}
MEDIA_EXTENSIONS: Set[str] = {
    ".mp4", ".mkv", ".avi", ".mov", ".mp3", ".wav", ".m4a",
    ".webm", ".flv", ".aac", ".ogg", ".wma"
}
ALL_SUPPORTED_EXTENSIONS: Set[str] = DOCUMENT_EXTENSIONS | MEDIA_EXTENSIONS
# All export formats supported by the export service (6 total)
EXPORT_FORMATS = ["pdf", "docx", "xlsx", "md", "html", "json"]


@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    # Resolve dynamic DATABASE_URL if not set directly but discrete variables are provided
    if not s.DATABASE_URL:
        try:
            s.DATABASE_URL = s.dynamic_database_url
        except Exception:
            pass

    # Anchor all storage / model / cache paths to PROJECT_ROOT so the app is
    # independent of the process working directory (pydantic-settings loads the
    # raw ./relative value from .env — resolve it here, post-load).
    # The vector store directory has no in-code path: it is environment-driven (new name first, then the
    # deprecated CHROMA_PATH), with ./data/chroma only as the last-resort local default.
    s.VECTOR_DB = (s.VECTOR_DB or "chromadb").strip().lower()
    s.CHROMA_PERSIST_DIRECTORY = _resolve_path(
        s.CHROMA_PERSIST_DIRECTORY or s.CHROMA_PATH, os.path.join("data", "chroma")
    )

    for _field, _default in (
        ("UPLOAD_DIR", "uploads"),
        ("NEMO_CACHE_DIR", "nemo_cache"),
        ("MODEL_BASE_PATH", "models"),
        ("EMBEDDING_MODEL_PATH", "models/embeddings/all-MiniLM-L6-v2"),
        ("DIARIZATION_MODEL_PATH", "models/diarization"),
        ("VOICE_PROFILE_DIR", "uploads/voice_profiles"),
    ):
        try:
            object.__setattr__(s, _field, _resolve_path(getattr(s, _field, ""), _default))
        except Exception:
            setattr(s, _field, _resolve_path(getattr(s, _field, ""), _default))

    if s.OFFLINE_MODE:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    return s


settings = get_settings()

# ── Environment Selector convenience import ──────────────────────────────────
# Services that need the active runtime config (api_key, database_url, etc.)
# should import get_active_config from here.
try:
    from backend.config.environment_selector import get_active_config as get_active_config  # noqa: F401
except Exception:
    # Startup import cycle guard — environment_selector initializes after config
    pass

# ── Module-level convenience aliases (backward-compatible legacy imports) ───────
DEFAULT_PROJECT_NAME: str = settings.APP_NAME
DEFAULT_SIMILARITY_THRESHOLD = settings.DEFAULT_SIMILARITY_THRESHOLD
DEFAULT_SIMILARITY_TOP_K = settings.DEFAULT_SIMILARITY_TOP_K
RAG_CHUNK_SIZE = settings.RAG_CHUNK_SIZE
RAG_CHUNK_OVERLAP = settings.RAG_CHUNK_OVERLAP
RAG_DEFAULT_TOP_K = settings.RAG_DEFAULT_TOP_K
RAG_MAX_CONTEXT_CHARS = settings.RAG_MAX_CONTEXT_CHARS
CLOUD_MAX_LLM_TOKENS = settings.CLOUD_MAX_LLM_TOKENS
LOCAL_MAX_LLM_TOKENS = settings.LOCAL_MAX_LLM_TOKENS
CLOUD_MAX_TRANSCRIPT_CHARS = settings.CLOUD_MAX_TRANSCRIPT_CHARS
LOCAL_MAX_TRANSCRIPT_CHARS = settings.LOCAL_MAX_TRANSCRIPT_CHARS
CLOUD_MAX_BYTES = settings.CLOUD_MAX_BYTES
BYTES_PER_SEC_WAV = settings.BYTES_PER_SEC_WAV
CHUNK_OVERLAP_SEC = settings.CHUNK_OVERLAP_SEC

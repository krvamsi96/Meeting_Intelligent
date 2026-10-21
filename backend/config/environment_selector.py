"""
Environment Selector — Server-First, Local Fallback Configuration Manager.

Architecture
-----------
.env (unified, single file)
  ↓
EnvironmentSelector (this module)
  ↓
Step 1: Load SERVER_* credentials from .env
Step 2: Health-check server AIRAWAT API + server DB
Step 3: If server reachable → ActiveConfig = SERVER
Step 4: If server unreachable → health-check LOCAL DB
Step 5: If local reachable → ActiveConfig = LOCAL
Step 6: If neither → raise ConfigurationError

The rest of the application calls get_active_config() and uses:
    config.airawat_api_key
    config.airawat_base_url
    config.airawat_whisper_url
    config.database_url
    config.environment   → "server" | "local"

No environment-specific logic leaks into services.

Security
--------
- Credentials are NEVER logged.
- Environment names are logged (e.g. "SERVER" / "LOCAL"), not credential values.
- Health-check result is cached for ENV_HEALTH_CACHE_TTL_SEC (default 300s).
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

# ── Ensure .env values are in os.environ for all os.getenv() calls below ─────
# pydantic_settings loads .env into its model attributes but does NOT inject
# them into os.environ. EnvironmentSelector uses os.getenv() directly, so we
# must load the file explicitly to guarantee the values are available in any
# Python process (uvicorn, pytest, scripts, etc.). The path is resolved against
# the project root (NOT the working directory) — see config.find_env_file().
try:
    from dotenv import load_dotenv as _load_dotenv

    try:
        from backend.config.config import find_env_file as _find_env_file
    except ImportError:  # config still initialising (import cycle) — it loads .env itself
        _find_env_file = None
    _env_file = _find_env_file() if _find_env_file else None
    if _env_file:
        _load_dotenv(dotenv_path=_env_file, override=False)  # OS env vars take precedence
except ImportError:
    pass  # python-dotenv not installed; rely on OS env vars set externally

logger = logging.getLogger(__name__)



class ActiveEnvironment(str, Enum):
    SERVER = "server"
    LOCAL  = "local"
    NONE   = "none"


@dataclass
class ActiveConfig:
    """Single, unified configuration surface consumed by all application services."""

    environment: str                    # "server" | "local"

    # AIRAWAT AI credentials
    airawat_api_key: str
    airawat_base_url: str
    airawat_whisper_url: str
    airawat_llm_model: str
    airawat_whisper_model: str

    # Database
    database_url: str
    db_host: str
    db_port: int
    db_database: str
    db_user: str

    # Shared timeout / retry settings (environment-independent)
    airawat_timeout_sec: int
    airawat_retry_count: int
    airawat_max_tokens: int
    airawat_temperature: float
    airawat_health_check_timeout_sec: int

    # Resolved at runtime — do NOT log
    _db_password: str = ""

    @property
    def db_password(self) -> str:
        return self._db_password


class EnvironmentSelector:
    """
    Determines which environment (SERVER or LOCAL) is available at startup
    and provides a stable ActiveConfig to the rest of the application.

    The selector runs exactly once at application boot (or can be triggered
    to re-check via `refresh()`).  Results are cached for ENV_HEALTH_CACHE_TTL_SEC.
    """

    _instance: Optional["EnvironmentSelector"] = None

    def __new__(cls) -> "EnvironmentSelector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._active_config: Optional[ActiveConfig] = None
        self._last_check_time: float = 0.0
        self._cache_ttl: int = int(os.getenv("ENV_HEALTH_CACHE_TTL_SEC", "300"))
        # Detection does blocking network health checks (DB + AIRAWAT, up to tens of
        # seconds). Serialise it so a burst of concurrent callers triggers ONE probe.
        self._detect_lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────────────────

    def get_active_config(self) -> ActiveConfig:
        """Return the active configuration (server or local), running detection if needed."""
        cached = self._active_config
        if cached is not None and (time.monotonic() - self._last_check_time) < self._cache_ttl:
            return cached
        with self._detect_lock:
            # Another thread may have finished detection while we waited for the lock.
            cached = self._active_config
            if cached is not None and (time.monotonic() - self._last_check_time) < self._cache_ttl:
                return cached
            return self._detect_and_cache()

    def refresh(self) -> ActiveConfig:
        """Force a fresh environment detection, bypassing the cache."""
        with self._detect_lock:
            self._active_config = None
            self._last_check_time = 0.0
            return self._detect_and_cache()

    @property
    def environment(self) -> str:
        """Return the name of the active environment without triggering a full re-detection."""
        if self._active_config:
            return self._active_config.environment
        return self.get_active_config().environment

    # ── Internal Detection Logic ────────────────────────────────────────────────

    def _detect_and_cache(self) -> ActiveConfig:
        ai_mode = os.getenv("AI_MODE", "auto").lower()
        logger.info("[EnvironmentSelector] Starting environment detection. AI_MODE=%s", ai_mode)

        if ai_mode == "server":
            logger.info("[DB] Checking Server DB...")
            config = self._try_server()
            if config is None:
                logger.error("[DB] ERROR: Server DB unavailable.")
                raise RuntimeError(
                    "[EnvironmentSelector] AI_MODE=server but server is unreachable. "
                    "Check SERVER_* credentials and network connectivity."
                )
        elif ai_mode == "local":
            logger.info("[DB] Checking Local DB...")
            config = self._try_local()
            if config is None:
                logger.error("[DB] ERROR: Local DB unavailable.")
                raise RuntimeError(
                    "[EnvironmentSelector] AI_MODE=local but local environment is unavailable. "
                    "Check LOCAL_* credentials (DB, AIRAWAT)."
                )
        else:
            # AUTO: server first, then local
            logger.info("[DB] Checking Server DB...")
            config = self._try_server()
            if config is not None:
                logger.info("[DB] Server DB connection successful.")
                logger.info("[DB] Active Database: SERVER")
            else:
                logger.warning("[DB] Server DB unavailable. Trying Local DB...")
                config = self._try_local()
                if config is not None:
                    logger.info("[DB] Local DB connection successful.")
                    logger.info("[DB] Active Database: LOCAL")
                else:
                    # Final fallback: generic DATABASE_URL / DB_* (no SERVER_/LOCAL_
                    # prefix). This is what docker-compose and plain deployments set.
                    logger.warning("[DB] Local DB unavailable. Trying generic DATABASE_URL / DB_* ...")
                    config = self._try_generic()
                    if config is not None:
                        logger.info("[DB] Generic DB connection successful.")
                        logger.info("[DB] Active Database: GENERIC")
                    else:
                        logger.error("[DB] ERROR: Server DB unavailable.")
                        logger.error("[DB] ERROR: Local DB fallback unavailable.")
                        logger.error("[DB] ERROR: Generic DATABASE_URL fallback unavailable.")
                        logger.error("[DB] Application cannot continue because no database is available.")
                        raise RuntimeError(
                            "[EnvironmentSelector] No database is available. Set SERVER_DB_* / "
                            "LOCAL_DB_* in .env, or a generic DATABASE_URL (or DB_HOST + DB_DATABASE)."
                        )

        self._active_config = config
        self._last_check_time = time.monotonic()
        # Log active environment and DB host only — never log credentials
        logger.info(
            "[EnvironmentSelector] Active environment: %s | DB host: %s | AI endpoint: %s",
            config.environment.upper(),
            config.db_host,
            config.airawat_base_url,
        )
        return config

    def _try_server(self) -> Optional[ActiveConfig]:
        """
        Load SERVER_* credentials and run connectivity health checks.
        Returns ActiveConfig if server is usable, None otherwise.
        Credentials are NEVER logged.
        """
        api_key     = os.getenv("SERVER_AIRAWAT_API_KEY", "")
        base_url    = os.getenv("SERVER_AIRAWAT_BASE_URL", "https://apis.airawat.cdac.in/nic/llama70b/v1")
        whisper_url = os.getenv("SERVER_AIRAWAT_WHISPER_URL", "")
        llm_model   = os.getenv("SERVER_AIRAWAT_LLM_MODEL", "meta/llama-3.3-70b-instruct")
        wh_model    = os.getenv("SERVER_AIRAWAT_WHISPER_MODEL", "whisper-1")
        db_url      = os.getenv("SERVER_DATABASE_URL", "")
        db_host     = os.getenv("SERVER_DB_HOST", "")
        db_port     = int(os.getenv("SERVER_DB_PORT", "5432"))
        db_database = os.getenv("SERVER_DB_DATABASE", "")
        db_user     = os.getenv("SERVER_DB_USER", "")
        db_password = os.getenv("SERVER_DB_PASSWORD", "")

        if not api_key or not api_key.strip():
            logger.debug("[EnvironmentSelector] SERVER_AIRAWAT_API_KEY not set — skipping server.")
            return None
        if not db_url and not db_host:
            logger.debug("[EnvironmentSelector] SERVER_DATABASE_URL not set — skipping server.")
            return None

        # ── Health check: Server database reachability (mandatory) ──────────────────────
        # DB check comes FIRST. If the database is unreachable the server
        # environment is unusable regardless of AIRAWAT status.
        #
        # Policy: the SERVER database gets a bounded number of attempts before we
        # fall back to LOCAL (default 2 = one initial try + one retry). This
        # absorbs a transient network blip without stalling startup when the
        # server is genuinely down. Configurable via SERVER_DB_CONNECT_ATTEMPTS.
        resolved_db_url = self._apply_sslmode(
            self._normalize_db_url(db_url) or self._build_db_url(db_host, db_port, db_database, db_user, db_password)
        )
        is_windows = platform.system() == "Windows"
        server_attempts = max(1, int(os.getenv("SERVER_DB_CONNECT_ATTEMPTS", "1" if is_windows else "2")))
        retry_delay = float(os.getenv("SERVER_DB_RETRY_DELAY_SEC", "0.5" if is_windows else "1.0"))
        db_reachable = False
        for attempt in range(1, server_attempts + 1):
            if self._check_database(resolved_db_url, label="server"):
                db_reachable = True
                if attempt > 1:
                    logger.info(
                        "[EnvironmentSelector] Server database reachable on attempt %d/%d.",
                        attempt, server_attempts,
                    )
                break
            if not is_windows:
                logger.warning(
                    "[EnvironmentSelector] Server database unreachable (attempt %d/%d).",
                    attempt, server_attempts,
                )
            if attempt < server_attempts:
                time.sleep(retry_delay)
        if not db_reachable:
            if is_windows:
                logger.info(
                    "[EnvironmentSelector] Server database unreachable from local machine; automatically selecting LOCAL environment."
                )
            else:
                logger.warning(
                    "[EnvironmentSelector] Server database unreachable after %d attempt(s) — falling back to LOCAL.",
                    server_attempts,
                )
            return None

        # ── Health check: AIRAWAT API reachability (non-blocking) ───────────────
        # If the DB is up but the AIRAWAT API is temporarily down, we still
        # return a valid config so the application can start. AI-dependent
        # endpoints will return an appropriate error at runtime; the rest of
        # the application (upload, DB reads, history) will work normally.
        timeout = int(os.getenv("AIRAWAT_HEALTH_CHECK_TIMEOUT_SEC", "30"))
        if not self._check_airawat(api_key, base_url, llm_model, timeout):
            logger.warning(
                "[EnvironmentSelector] Server AIRAWAT API unreachable. "
                "Continuing with server DB — AI features will be degraded until API is reachable."
            )

        return ActiveConfig(
            environment="server",
            airawat_api_key=api_key,
            airawat_base_url=base_url,
            airawat_whisper_url=whisper_url,
            airawat_llm_model=llm_model,
            airawat_whisper_model=wh_model,
            database_url=resolved_db_url,
            db_host=db_host,
            db_port=db_port,
            db_database=db_database,
            db_user=db_user,
            _db_password=db_password,
            airawat_timeout_sec=int(os.getenv("AIRAWAT_TIMEOUT_SEC", "600")),
            airawat_retry_count=int(os.getenv("AIRAWAT_RETRY_COUNT", "2")),
            airawat_max_tokens=int(os.getenv("AIRAWAT_MAX_TOKENS", "2000")),
            airawat_temperature=float(os.getenv("AIRAWAT_TEMPERATURE", "0.1")),
            airawat_health_check_timeout_sec=timeout,
        )

    def _try_local(self) -> Optional[ActiveConfig]:
        """
        Load LOCAL_* credentials and run connectivity health checks.
        Returns ActiveConfig if local environment is usable, None otherwise.
        """
        api_key     = os.getenv("LOCAL_AIRAWAT_API_KEY", "")
        base_url    = os.getenv("LOCAL_AIRAWAT_BASE_URL", "https://apis.airawat.cdac.in/nic/llama70b/v1")
        whisper_url = os.getenv("LOCAL_AIRAWAT_WHISPER_URL", "")
        llm_model   = os.getenv("LOCAL_AIRAWAT_LLM_MODEL", "meta/llama-3.3-70b-instruct")
        wh_model    = os.getenv("LOCAL_AIRAWAT_WHISPER_MODEL", "whisper-1")
        db_url      = os.getenv("LOCAL_DATABASE_URL", "")
        db_host     = os.getenv("LOCAL_DB_HOST", "127.0.0.1")
        db_port     = int(os.getenv("LOCAL_DB_PORT", "5432"))
        db_database = os.getenv("LOCAL_DB_DATABASE", "")
        db_user     = os.getenv("LOCAL_DB_USER", "")
        db_password = os.getenv("LOCAL_DB_PASSWORD", "")

        if not db_url and not db_host:
            logger.debug("[EnvironmentSelector] LOCAL_DATABASE_URL not set — local unavailable.")
            return None

        # ── Health check: Local database reachability (mandatory for local) ────
        resolved_db_url = self._normalize_db_url(db_url) or self._build_db_url(db_host, db_port, db_database, db_user, db_password)
        if not self._check_database(resolved_db_url, label="local"):
            logger.warning("[EnvironmentSelector] Local database unreachable.")
            return None

        # ── AIRAWAT API ──────────────────────────────────────────────────────────
        # AIRAWAT is the only AI provider in every mode. A failed startup probe (often a single gateway stall)
        # never re-points the platform at another engine: calls are retried and then fail with a clear error.
        timeout = int(os.getenv("AIRAWAT_HEALTH_CHECK_TIMEOUT_SEC", "30"))
        if api_key and api_key.strip():
            if not self._check_airawat(api_key, base_url, llm_model, timeout):
                logger.warning(
                    "[EnvironmentSelector] AIRAWAT did not answer the startup probe. Keeping AIRAWAT "
                    "(AI_MODE=%s): AI calls are retried per AIRAWAT_RETRY_COUNT and fail with a clear "
                    "error until it recovers; no other engine is substituted.",
                    os.getenv("AI_MODE", "auto"),
                )
        else:
            logger.error(
                "[EnvironmentSelector] LOCAL_AIRAWAT_API_KEY is not set: AI features are unavailable "
                "(AI_MODE=%s). Set it in the environment; no other engine is substituted.",
                os.getenv("AI_MODE", "auto"),
            )

        return ActiveConfig(
            environment="local",
            airawat_api_key=api_key,
            airawat_base_url=base_url,
            airawat_whisper_url=whisper_url,
            airawat_llm_model=llm_model,
            airawat_whisper_model=wh_model,
            database_url=resolved_db_url,
            db_host=db_host,
            db_port=db_port,
            db_database=db_database,
            db_user=db_user,
            _db_password=db_password,
            airawat_timeout_sec=int(os.getenv("AIRAWAT_TIMEOUT_SEC", "600")),
            airawat_retry_count=int(os.getenv("AIRAWAT_RETRY_COUNT", "2")),
            airawat_max_tokens=int(os.getenv("AIRAWAT_MAX_TOKENS", "2000")),
            airawat_temperature=float(os.getenv("AIRAWAT_TEMPERATURE", "0.1")),
            airawat_health_check_timeout_sec=timeout,
        )

    def _try_generic(self) -> Optional[ActiveConfig]:
        """Fallback for deployments that set generic (unprefixed) DB / AI env vars.

        Used by docker-compose and plain server installs that provide
        DATABASE_URL / DB_HOST+DB_DATABASE and AIRAWAT_API_KEY without the
        SERVER_* / LOCAL_* prefixes. DB reachability is still mandatory.
        """
        _s = None
        try:
            from backend.config.config import settings as _s
            db_url = (_s.DATABASE_URL or _s.DB_URL or "").strip()
            if not db_url:
                try:
                    db_url = _s.dynamic_database_url
                except Exception:
                    db_url = ""
        except Exception:
            db_url = (os.getenv("DATABASE_URL") or os.getenv("DB_URL") or "").strip()

        if not db_url:
            logger.debug("[EnvironmentSelector] No generic DATABASE_URL / DB_* configured.")
            return None

        # DATABASE_URL is usually assembled by hand — docker-compose interpolates the RAW password into it —
        # so percent-encode its credentials exactly like the SERVER_ / LOCAL_ paths do. Without this an '@',
        # '#', '/' or ':' in the password makes the URL unparseable and the app cannot reach its database.
        candidates = [self._normalize_db_url(db_url)]
        # A password that itself contains a "%XX" sequence is ambiguous inside a URL; the discrete DB_*
        # variables (docker-compose sets them too) carry it verbatim, so they are the exact fallback.
        if _s is not None:
            try:
                from backend.config.config import build_database_url

                discrete = build_database_url(
                    db_type=_s.DB_TYPE or _s.DB_CLIENT,
                    db_host=_s.DB_HOST,
                    db_port=_s.DB_PORT,
                    db_database=_s.DB_DATABASE or _s.DB_NAME,
                    db_user=_s.DB_USER or _s.DB_USERNAME,
                    db_password=_s.DB_PASSWORD or _s.DB_PASS,
                    db_ssl=_s.DB_SSL,
                    db_sslmode=_s.DB_SSLMODE,
                )
            except Exception:
                discrete = ""
            if discrete and discrete not in candidates:
                candidates.append(discrete)

        db_url = next((u for u in candidates if u and self._check_database(u, label="generic")), "")
        if not db_url:
            logger.warning("[EnvironmentSelector] Generic database unreachable.")
            return None

        api_key     = os.getenv("AIRAWAT_API_KEY", "")
        base_url    = os.getenv("AIRAWAT_BASE_URL", "https://apis.airawat.cdac.in/nic/llama70b/v1")
        whisper_url = os.getenv("AIRAWAT_WHISPER_URL", "")
        llm_model   = os.getenv("AIRAWAT_LLM_MODEL", "meta/llama-3.3-70b-instruct")
        wh_model    = os.getenv("AIRAWAT_WHISPER_MODEL", "whisper-1")
        timeout     = int(os.getenv("AIRAWAT_HEALTH_CHECK_TIMEOUT_SEC", "30"))

        # AI is optional here — DB works, so the app can start and serve reads.
        if api_key and api_key.strip() and not self._check_airawat(api_key, base_url, llm_model, timeout):
            logger.warning(
                "[EnvironmentSelector] Generic AIRAWAT API unreachable — AI features degraded until it recovers."
            )

        db_host = ""
        try:
            db_host = db_url.split("@", 1)[1].split("/", 1)[0].split(":", 1)[0] if "@" in db_url else db_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        except Exception:
            db_host = "unknown"

        return ActiveConfig(
            environment="generic",
            airawat_api_key=api_key,
            airawat_base_url=base_url,
            airawat_whisper_url=whisper_url,
            airawat_llm_model=llm_model,
            airawat_whisper_model=wh_model,
            database_url=db_url,
            db_host=db_host,
            db_port=int(os.getenv("DB_PORT", "5432")),
            db_database=os.getenv("DB_DATABASE", os.getenv("DB_NAME", "")),
            db_user=os.getenv("DB_USER", os.getenv("DB_USERNAME", "")),
            _db_password=os.getenv("DB_PASSWORD", os.getenv("DB_PASS", "")),
            airawat_timeout_sec=int(os.getenv("AIRAWAT_TIMEOUT_SEC", "600")),
            airawat_retry_count=int(os.getenv("AIRAWAT_RETRY_COUNT", "2")),
            airawat_max_tokens=int(os.getenv("AIRAWAT_MAX_TOKENS", "2000")),
            airawat_temperature=float(os.getenv("AIRAWAT_TEMPERATURE", "0.1")),
            airawat_health_check_timeout_sec=timeout,
        )

    # ── Connectivity Health Checks ──────────────────────────────────────────────

    def _check_airawat(self, api_key: str, base_url: str, model: str, timeout: int) -> bool:
        """
        Send a minimal POST to the AIRAWAT /chat/completions endpoint.
        Returns True if the response is 2xx (service is reachable and key is valid).
        Credentials are NEVER logged.
        """
        import requests
        ssl_verify = os.getenv("SSL_VERIFY", "true").lower() == "true"
        endpoint = base_url.rstrip("/") + "/chat/completions"
        probe_timeout = min(timeout, 5)
        for attempt in range(2):
            try:
                resp = requests.post(
                    endpoint,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                    json={"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
                    verify=ssl_verify,
                    timeout=probe_timeout,
                )
                if 200 <= resp.status_code < 300:
                    return True
                if resp.status_code not in (408, 429, 500, 502, 503, 504):
                    return False
            except Exception as exc:
                logger.debug("[EnvironmentSelector] AIRAWAT probe attempt %d failed: %s", attempt + 1, type(exc).__name__)
            if attempt == 0:
                time.sleep(0.3)
        return False

    def _check_database(self, database_url: str, label: str = "") -> bool:
        """
        Attempt a minimal TCP connection test against the given database URL.
        Returns True on success. The URL (which may contain credentials) is NEVER logged.
        Timeout is configurable via DB_CONNECTION_TIMEOUT env var (default 5s).

        Uses NullPool so the test connection is immediately closed and NOT held
        in an idle pool — preventing ghost connections from cluttering the DB server.
        """
        # Fast TCP socket pre-check to prevent multi-second timeout stalls on unreachable private IPs
        try:
            raw_url = database_url.replace("postgresql+psycopg2://", "postgresql://")
            parsed = urlparse(raw_url)
            host = parsed.hostname
            port = parsed.port or 5432
            if host:
                is_win = platform.system() == "Windows"
                fast_timeout = float(os.getenv("DB_FAST_PROBE_TIMEOUT", "0.8" if (is_win and label == "server") else "2.0"))
                with socket.create_connection((host, port), timeout=fast_timeout):
                    pass
        except Exception as exc:
            logger.debug("[EnvironmentSelector] Fast TCP probe failed for %s database: %s", label, type(exc).__name__)
            return False

        try:
            import sqlalchemy
            from sqlalchemy.pool import NullPool
            # Use DB_CONNECTION_TIMEOUT (configurable) — default 5 s to avoid long hangs
            # on an unreachable server when the local fallback is needed quickly.
            connect_timeout = int(os.getenv("DB_CONNECTION_TIMEOUT") or os.getenv("DB_CONNECT_TIMEOUT") or "5")
            from backend.config.config import with_psycopg2_driver

            engine = sqlalchemy.create_engine(
                with_psycopg2_driver(database_url),
                poolclass=NullPool,  # Never pool health-check connections
                connect_args={"connect_timeout": connect_timeout},
            )
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            engine.dispose()
            logger.debug("[EnvironmentSelector] %s database reachable.", label.capitalize())
            return True
        except Exception as exc:
            logger.debug(
                "[EnvironmentSelector] %s database unreachable: %s",
                label.capitalize(), type(exc).__name__,
            )
            return False


    def _build_db_url(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> str:
        import urllib.parse
        if not host or not database:
            return ""
        # %XX-encode (not '+' for space): SQLAlchemy's URL parser only un-escapes %XX.
        q_user = urllib.parse.quote(user, safe="") if user else ""
        q_pass = urllib.parse.quote(password, safe="") if password else ""
        auth = f"{q_user}:{q_pass}@" if q_user and q_pass else (f"{q_user}@" if q_user else "")
        return self._apply_sslmode(f"postgresql+psycopg2://{auth}{host}:{port}/{database}")

    @staticmethod
    def _normalize_db_url(url: str) -> str:
        """Percent-encode the credentials of a hand-written DB URL.

        A password containing '@', '#', '/' … (e.g. ``postgresql://u:p@ss@host/db``) is
        not a valid URL as typed and SQLAlchemy would mis-parse the host. Splitting on
        the LAST '@' and re-encoding user/password makes such URLs work; URLs that are
        already correctly encoded are returned unchanged (encoding is idempotent).
        """
        url = (url or "").strip()
        if not url:
            return ""
        try:
            from backend.config.config import build_database_url

            return build_database_url(database_url=url)
        except Exception:
            return url

    @staticmethod
    def _apply_sslmode(url: str) -> str:
        """Append sslmode to a DB URL when DB_SSLMODE / DB_SSL is configured.

        Managed PostgreSQL (and many hardened servers) reject non-SSL
        connections. If the URL already carries an sslmode it is left untouched.
        """
        if not url or "sslmode=" in url:
            return url
        mode = (os.getenv("DB_SSLMODE") or "").strip().lower()
        if not mode:
            if (os.getenv("DB_SSL", "false").strip().lower() in ("1", "true", "yes", "on")):
                mode = "require"
        if not mode or mode in ("disable", "prefer"):
            return url
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}sslmode={mode}"


# ── Singleton instance and public helpers ────────────────────────────────────

environment_selector = EnvironmentSelector()


def get_active_config() -> ActiveConfig:
    """Application-wide accessor. Returns the active environment configuration."""
    return environment_selector.get_active_config()

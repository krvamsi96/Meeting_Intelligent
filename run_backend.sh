#!/usr/bin/env bash
# AI Meeting Intelligence Platform — Backend-Only Linux Runner Script
# Starts ONLY the FastAPI backend (no frontend dependency).
# Usage: bash run_backend.sh

set -e

# Always run from the project directory: PM2 / systemd / cron start this script from an arbitrary working
# directory, and everything below (venv, .env, `-m backend...`) is relative to the project root.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

echo "==============================================================================="
echo "   AI MEETING INTELLIGENCE PLATFORM - BACKEND RUNNER (LINUX)"
echo "==============================================================================="
echo ""

# 1. Check Virtual Environment
if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found."
    echo "        Run 'bash setup_backend.sh' first to set up the environment."
    exit 1
fi

source venv/bin/activate

# 2. Load .env into the shell environment so that pydantic-settings picks up
#    the variables even when pydantic's env_file resolution fails (e.g. CWD mismatch).
#    Falls back gracefully to OS-level env vars (set by PM2 / systemd) when .env
#    is absent — this avoids a crash-loop during the brief deploy window where
#    Jenkinsfile has pushed new code but hasn't yet run 'Inject Production Env'.
if [ -f ".env" ]; then
    # Export every KEY=VALUE line with python-dotenv (the parser the application itself uses).
    # NOT `source .env`: a dotenv value may legally contain spaces, '|', '&', '<', '$' ... e.g.
    #   REPORT_FOOTER_TEXT=Acme Corp | Powered by X      CHROMA_PERSIST_DIRECTORY=<path>
    # and bash would try to EXECUTE or reject those - under `set -e` that aborted this launcher, so
    # PM2 crash-looped. Values are shell-quoted below, so nothing in .env is ever executed.
    if ENV_EXPORTS="$(python - <<'PYEOF'
import shlex
from dotenv import dotenv_values

for key, value in dotenv_values(".env").items():
    if value is not None and key.isidentifier():
        print("export %s=%s" % (key, shlex.quote(value)))
PYEOF
)"; then
        eval "${ENV_EXPORTS}"
        echo "[INFO]  Loaded environment from .env"
    else
        echo "[WARN]  Could not parse .env with python-dotenv (is it installed in the venv?)."
        echo "        The application still reads .env itself; HOST/PORT/WORKERS fall back to defaults."
    fi
else
    echo "[WARN]  .env file not found — falling back to OS environment variables."
    echo "        DATABASE_URL=${DATABASE_URL:-<not set>}"
    if [ -z "${DATABASE_URL:-}" ]; then
        echo "[ERROR] DATABASE_URL is not set and no .env file exists."
        echo "        Deploy the production .env (via Jenkins 'aimeet-production-env') or set DATABASE_URL manually."
        exit 1
    fi
fi

# 3. Load config from environment
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# One worker: the embedded ChromaDB directory (CHROMA_PERSIST_DIRECTORY) can be owned by a single process.
WORKERS="${WORKERS:-${UVICORN_WORKERS:-1}}"
# uvicorn only accepts a LOWERCASE --log-level (critical|error|warning|info|debug|trace), while .env / .env.example
# say LOG_LEVEL=INFO (which the application's own logging accepts): normalise, or uvicorn refuses to start.
LOG_LEVEL="$(printf '%s' "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')"

# 3b. Ensure a stable SECRET_KEY BEFORE workers start (no per-worker race).
if [ -z "${SECRET_KEY:-}" ] && [ -f ".env" ] && ! grep -qE '^SECRET_KEY=.+' .env; then
    GEN_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())' 2>/dev/null || true)"
    if [ -n "${GEN_KEY}" ]; then
        if grep -qE '^SECRET_KEY=' .env; then
            sed -i.bak "s|^SECRET_KEY=.*|SECRET_KEY=${GEN_KEY}|" .env && rm -f .env.bak
        else
            printf '\nSECRET_KEY=%s\n' "${GEN_KEY}" >> .env
        fi
        export SECRET_KEY="${GEN_KEY}"
        echo "[INFO]  Generated and persisted a new SECRET_KEY in .env"
    fi
fi

# 4. Enforce strict offline operation
export OFFLINE_MODE=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 5. Ensure storage and model directories exist
mkdir -p uploads uploads/clips data/chroma nemo_cache logs models/embeddings models/diarization

# 5a. Embedding model (FATAL): the weights are a Git LFS object. If this checkout holds the ~130-byte pointer, fetch the
#     real file with `git lfs pull` from the repository's LFS remote, then verify size + SHA-256. Refuse to start
#     without the real model - the API would run with semantic search / RAG / indexing silently off.
if ! python -m backend.scripts.ensure_embedding_model; then
    echo ""
    echo "[ERROR] The real embedding model could not be obtained - not starting (see the FIX above)."
    exit 1
fi

# 5b. Startup preflight: python, dependencies, PostgreSQL (fatal) + FFmpeg, embedding model files, directories,
#     config (reported, non-fatal: the API starts and GET /system/health shows them). Never prints secrets.
#     It does not call external AI: the API logs its own AI check after boot.
#     Set SKIP_PREFLIGHT=1 to bypass. Run the full gate any time with: python -m backend.scripts.preflight
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    if ! python -m backend.scripts.preflight --startup; then
        echo ""
        echo "[ERROR] Startup preflight failed - fix the problems listed above (FIX: lines), or start anyway with SKIP_PREFLIGHT=1."
        exit 1
    fi
    echo ""
fi

echo "[INFO]  Starting backend server on http://${HOST}:${PORT}"
echo "[INFO]  Workers: ${WORKERS} | Log level: ${LOG_LEVEL}"
echo ""

# 5. Start backend
exec python -m uvicorn backend.main:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --log-level "${LOG_LEVEL}"

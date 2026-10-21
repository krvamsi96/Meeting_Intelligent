#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — Production Offline Startup Script
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Run from the project directory whatever the caller's working directory is (systemd, cron, PM2 ...).
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

echo "================================================================="
echo "   AI MEETING INTELLIGENCE PLATFORM — OFFLINE STARTUP"
echo "================================================================="
echo ""

# 1. Enforce strict offline operation
export OFFLINE_MODE=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# 2. Activate virtual environment if available
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "[OK] Virtual environment activated"
fi

# 3. Check environment file
if [ -f ".env" ]; then
    echo "[OK] Configuration (.env) loaded"
else
    echo "[WARN] .env file not found. Copying from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "[OK] Created .env from template - it still holds PLACEHOLDERS: edit it before relying on this server"
    fi
fi

# 3a. Export .env (HOST, PORT, UVICORN_WORKERS, LOG_LEVEL ...) into this shell with python-dotenv - the parser the
#     application itself uses. NOT `source .env`: a valid dotenv value may contain spaces, '|', '<' ... and bash
#     would execute or reject it. Real environment variables already set win over the file.
if [ -f ".env" ]; then
    if ENV_EXPORTS="$(python3 - <<'PYEOF'
import os
import shlex
from dotenv import dotenv_values

for key, value in dotenv_values(".env").items():
    if value is not None and key.isidentifier() and key not in os.environ:
        print("export %s=%s" % (key, shlex.quote(value)))
PYEOF
)"; then
        eval "${ENV_EXPORTS}"
    else
        echo "[WARN] Could not parse .env with python-dotenv; HOST/PORT/UVICORN_WORKERS fall back to defaults."
    fi
fi

# 3b. Ensure a stable SECRET_KEY exists BEFORE workers start (avoids a race where
#     multiple uvicorn workers each generate a different key on first boot).
if [ -z "${SECRET_KEY:-}" ] && ! grep -qE '^SECRET_KEY=.+' .env 2>/dev/null; then
    GEN_KEY="$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())' 2>/dev/null || true)"
    if [ -n "${GEN_KEY}" ]; then
        if grep -qE '^SECRET_KEY=' .env 2>/dev/null; then
            sed -i.bak "s|^SECRET_KEY=.*|SECRET_KEY=${GEN_KEY}|" .env && rm -f .env.bak
        else
            printf '\nSECRET_KEY=%s\n' "${GEN_KEY}" >> .env
        fi
        export SECRET_KEY="${GEN_KEY}"
        echo "[OK] Generated and persisted a new SECRET_KEY in .env"
    else
        echo "[WARN] Could not generate SECRET_KEY (cryptography missing?). Set it manually in .env."
    fi
fi

# 4. Prepare directories
mkdir -p uploads uploads/clips data/chroma nemo_cache logs models/embeddings models/diarization
chmod -R 755 uploads data nemo_cache logs models 2>/dev/null || true
echo "[OK] Runtime and storage directories prepared"

# 4b. Embedding model (FATAL): fetch the real weights with `git lfs pull` if this checkout holds the Git LFS pointer,
#     verify size + SHA-256, and refuse to start without them.
if ! python3 -m backend.scripts.ensure_embedding_model; then
    echo "[ERROR] The real embedding model could not be obtained - not starting (see the FIX above)."
    exit 1
fi

# 5. Startup preflight: python, dependencies, PostgreSQL (fatal) + FFmpeg, embedding model FILES (a Git LFS pointer
#    or truncated weights file is named precisely), directories, configuration (reported, non-fatal).
#    SKIP_PREFLIGHT=1 bypasses it; the full deployment gate is: python -m backend.scripts.preflight
echo ""
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    if ! python3 -m backend.scripts.preflight --startup; then
        echo "[ERROR] Startup preflight failed - fix the problems listed above (FIX: lines), or start anyway with SKIP_PREFLIGHT=1."
        exit 1
    fi
fi

# 6. Start application
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
# One worker: the embedded ChromaDB directory (CHROMA_PERSIST_DIRECTORY) can be owned by a single process.
WORKERS="${UVICORN_WORKERS:-1}"
# uvicorn only accepts a LOWERCASE --log-level; .env says LOG_LEVEL=INFO (fine for the application's own logging).
LOG_LEVEL="$(printf '%s' "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')"

echo ""
echo "================================================================="
echo "  Starting FastAPI backend on http://${HOST}:${PORT} (${WORKERS} workers)"
echo "================================================================="
echo ""

exec python3 -m uvicorn backend.main:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --log-level "${LOG_LEVEL}"

#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# AI Meeting Intelligence Platform — Docker Deploy Script (Linux)
# Run this ONCE on your Linux server after uploading the project.
# Usage: bash deploy_docker.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "=============================================================="
echo "  AI MEETING INTELLIGENCE PLATFORM — DOCKER DEPLOY"
echo "=============================================================="
echo ""

# ── Step 1: Check Docker ─────────────────────────────────────────
echo "[1/6] Checking Docker installation..."
if ! command -v docker &>/dev/null; then
    echo "  [INFO] Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "  [WARN] Docker installed. You may need to log out and back in."
    echo "         Run: newgrp docker"
fi

if ! command -v docker &>/dev/null; then
    echo "  [ERROR] Docker still not available. Please install manually."
    exit 1
fi
echo "  [OK] Docker: $(docker --version)"

# Verify Docker Compose v2 (plugin or standalone)
if ! docker compose version &>/dev/null; then
    echo "  [ERROR] 'docker compose' (v2) is required. Please install Docker Compose v2."
    exit 1
fi
echo "  [OK] Docker Compose: $(docker compose version)"
echo ""

# ── Step 2: Check .env ──────────────────────────────────────────
echo "[2/6] Checking .env configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.production" ]; then
        cp .env.production .env
        echo "  [INFO] Copied .env.production to .env"
        echo ""
        echo "  ┌─────────────────────────────────────────────────────────┐"
        echo "  │  ACTION REQUIRED: Edit .env before continuing!           │"
        echo "  │                                                           │"
        echo "  │  1. Set AIRAWAT_API_KEY=your_actual_key                  │"
        echo "  │  2. Set CORS_ORIGINS to your server IP/domain            │"
        echo "  │  3. Set REPORT_COMPANY_NAME to your organisation         │"
        echo "  │                                                           │"
        echo "  │  nano .env                                                │"
        echo "  └─────────────────────────────────────────────────────────┘"
        echo ""
        read -rp "  Press Enter after editing .env to continue..." _
    elif [ -f ".env.example" ]; then
        cp .env.example .env
        echo "  [WARN] .env.example copied to .env — configure it before use"
    else
        echo "  [ERROR] No .env, .env.production, or .env.example found!"
        exit 1
    fi
else
    echo "  [OK] .env found"
fi

# Auto-generate SECRET_KEY and JWT_SECRET if missing or empty
SECRET_KEY_VAL=$(grep -E "^SECRET_KEY=" .env | cut -d= -f2 | tr -d ' ' || true)
if [ -z "${SECRET_KEY_VAL:-}" ]; then
    if command -v openssl &>/dev/null; then
        GENERATED_KEY=$(openssl rand -hex 32)
        # Update or append SECRET_KEY
        if grep -qE "^SECRET_KEY=" .env; then
            sed -i "s|^SECRET_KEY=.*|SECRET_KEY=${GENERATED_KEY}|" .env
        else
            echo "SECRET_KEY=${GENERATED_KEY}" >> .env
        fi
        GENERATED_JWT=$(openssl rand -hex 32)
        if grep -qE "^JWT_SECRET=" .env; then
            sed -i "s|^JWT_SECRET=.*|JWT_SECRET=${GENERATED_JWT}|" .env
        else
            echo "JWT_SECRET=${GENERATED_JWT}" >> .env
        fi
        echo "  [OK] Generated SECRET_KEY and JWT_SECRET in .env"
    else
        echo "  [WARN] openssl not found. Set SECRET_KEY and JWT_SECRET manually in .env"
    fi
else
    echo "  [OK] SECRET_KEY already configured"
fi

# Warn if the database password was never set (compose would fall back to a well-known default).
# Only a warning: an existing pgdata volume keeps the password it was first created with.
DB_PASSWORD_VAL=$(grep -E "^DB_PASSWORD=" .env | tail -n 1 | cut -d= -f2- | tr -d ' "'"'"'\r' || true)
if [ -z "${DB_PASSWORD_VAL:-}" ] || [ "${DB_PASSWORD_VAL}" = "CHANGE_ME_IN_PRODUCTION" ]; then
    echo "  [WARN] DB_PASSWORD is not set (or is still CHANGE_ME_IN_PRODUCTION) in .env."
    echo "         Set DB_USER / DB_PASSWORD / DB_DATABASE before the FIRST start - the password is baked into the"
    echo "         pgdata volume on first init and changing .env later does not change it."
fi

# Warn if AIRAWAT key looks unconfigured
AIRAWAT_KEY=$(grep -E "^AIRAWAT_API_KEY=" .env | cut -d= -f2 | tr -d ' ' || true)
if [ -z "${AIRAWAT_KEY:-}" ]; then
    echo "  [WARN] AIRAWAT_API_KEY is empty — cloud AI features will not work."
    echo "         Set it in .env and re-run this script, or run: docker compose restart backend"
elif [[ "${AIRAWAT_KEY}" == eyJ* ]]; then
    echo "  [OK] AIRAWAT_API_KEY is configured"
fi
echo ""

# ── Step 3: Create storage directories ──────────────────────────
echo "[3/6] Creating storage directories..."
mkdir -p uploads uploads/clips data/chroma nemo_cache
chmod -R 755 uploads data nemo_cache
echo "  [OK] Storage directories ready: uploads/, data/chroma/ (ChromaDB), nemo_cache/"

# The image copies ./models and compose mounts it read-only. The weights are tracked with Git LFS: a clone made
# without git-lfs holds a ~130-byte pointer here, which the container would fail on with "Error while deserializing
# header: header too large". Refuse to build around it.
# Standard-library check (no venv needed on the Docker host): git lfs pull if needed, size + SHA-256 verified.
if ! python3 -m backend.scripts.ensure_embedding_model; then
    echo "  [ERROR] Not building an image without the real embedding model (see the FIX above)."
    exit 1
fi
echo ""

# ── Step 4: Build Docker images ─────────────────────────────────
echo "[4/6] Building Docker images (this may take several minutes)..."
docker compose build --no-cache
echo "  [OK] Docker images built"
echo ""

# ── Step 5: Start Services ───────────────────────────────────────
echo "[5/6] Starting PostgreSQL + backend services..."
docker compose up -d
echo ""
echo "  Waiting for PostgreSQL to be ready (up to 30s)..."
# Same user compose gives the postgres container (DB_USER, default postgres) — a hard-coded
# "postgres" never reports ready when DB_USER is customised.
PG_USER=$(grep -E "^DB_USER=" .env | tail -n 1 | cut -d= -f2- | tr -d ' "'"'"'\r' || true)
PG_USER="${PG_USER:-postgres}"
POSTGRES_READY=false
for i in $(seq 1 6); do
    sleep 5
    if docker compose exec -T postgres pg_isready -U "${PG_USER}" &>/dev/null 2>&1; then
        POSTGRES_READY=true
        break
    fi
    echo "    Waiting... ${i}0s"
done

if [ "$POSTGRES_READY" = false ]; then
    echo "  [WARN] PostgreSQL may not be ready yet. Check logs:"
    echo "         docker compose logs postgres"
fi

docker compose ps
echo "  [OK] Services started"
echo ""

# ── Step 6: Health Check ─────────────────────────────────────────
echo "[6/6] Waiting for backend health check (up to 120 seconds)..."
HEALTH_URL="http://localhost:8000/api/v1/system/health"   # compose publishes the backend on 8000
MAX_WAIT=120
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
        echo ""
        echo "  [SUCCESS] Backend is ONLINE!"
        RESPONSE=$(curl -s "$HEALTH_URL")
        echo "  Response: $RESPONSE"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  Waiting... ${ELAPSED}s"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo ""
    echo "  [WARN] Service not responding yet — NeMo model download may still be in progress."
    echo "         Check logs: docker compose logs -f backend"
fi

# Full verification from inside the running container (PostgreSQL round trip, embedding model, AIRAWAT LLM + Whisper,
# FFmpeg; ChromaDB is owned by the running API, so its round trip is skipped here). Prints FIX: lines, never secrets.
if [ $ELAPSED -lt $MAX_WAIT ]; then
    echo ""
    echo "  Running the deployment preflight inside the backend container..."
    docker compose exec -T backend python -m backend.scripts.preflight --skip-chromadb         || echo "  [WARN] Preflight reported problems (see FIX: lines above). The API is up; fix them before real use."
fi

echo ""
echo "=============================================================="
echo "               DEPLOYMENT COMPLETE"
echo "=============================================================="
echo ""
SERVER_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
echo "  Backend API:  http://${SERVER_IP}:8000"
echo "  Health URL:   http://${SERVER_IP}:8000/api/v1/system/health"
echo "  API Docs:     http://${SERVER_IP}:8000/docs  (if ENABLE_SWAGGER=true)"
echo ""
echo "  Useful commands:"
echo "    docker compose logs -f backend    → live backend logs"
echo "    docker compose logs -f postgres   → live DB logs"
echo "    docker compose ps                 → container status"
echo "    docker compose restart backend    → restart backend only"
echo "    docker compose down               → stop all services"
echo "    docker compose down -v            → stop + DELETE all data"
echo "    bash healthcheck.sh               → run full health check"
echo "=============================================================="

#!/usr/bin/env bash
# AI Meeting Intelligence Platform — Backend-Only Linux Setup Script
# Prepares the environment for running the backend only (no frontend).
# Usage: bash setup_backend.sh

set -e

# Run from the project directory whatever the caller's working directory is.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

echo "==============================================================================="
echo "   AI MEETING INTELLIGENCE PLATFORM - BACKEND SETUP (LINUX)"
echo "==============================================================================="
echo ""

# 1. Check Python 3.10+
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.10+:"
    echo "        sudo apt-get install python3.11 python3.11-venv"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[INFO]  Python version: ${PYTHON_VERSION}"
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "[OK]    Python >= 3.10 confirmed"
else
    echo "[ERROR] Python 3.10+ required. Got ${PYTHON_VERSION}"
    exit 1
fi

# 2. Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "[ERROR] ffmpeg not found. Install with:"
    echo "        sudo apt-get install ffmpeg"
    exit 1
fi
echo "[OK]    FFmpeg: $(ffmpeg -version 2>&1 | head -n 1 | cut -d ' ' -f 1-3)"

if ! command -v ffprobe &> /dev/null; then
    echo "[WARN]  ffprobe not found (usually bundled with ffmpeg)"
fi

# 3. Check PostgreSQL client availability (optional — needed if DB is remote)
if command -v psql &> /dev/null; then
    echo "[OK]    PostgreSQL client available"
else
    echo "[WARN]  psql not found. Install if needed: sudo apt-get install postgresql-client"
fi

# 4. Environment configuration
echo ""
echo "[INFO]  Checking .env configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "[INFO]  Copying .env.example to .env"
        cp .env.example .env
        echo "[OK]    Created .env from template"
        echo "[WARN]  Edit .env and configure AI provider credentials, DATABASE_URL, etc."
    else
        echo "[ERROR] Neither .env nor .env.example found."
        echo "        Create a .env file with required variables. See LINUX_DEPLOYMENT.md"
        exit 1
    fi
else
    echo "[OK]    .env file found"
fi

# 5. Python Virtual Environment
echo ""
echo "[INFO]  Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "[OK]    Created virtual environment at ./venv"
else
    echo "[OK]    Virtual environment already exists"
fi

source venv/bin/activate

# 6. Python dependencies
# Installation is deliberately NOT automatic (torch + NeMo are several GB and CPU-only torch must be
# installed first). Stop with instructions instead of failing later with a cryptic ImportError.
if ! python3 -c "import fastapi, uvicorn, sqlalchemy, pydantic_settings, chromadb" 2>/dev/null; then
    echo ""
    echo "[ERROR] Python dependencies are not installed in ./venv. Install them, then re-run this script:"
    echo "          pip install --upgrade pip"
    echo "          pip install torch --extra-index-url https://download.pytorch.org/whl/cpu"
    echo "          pip install -r requirements.txt"
    exit 1
fi

# 7. Create storage and model directories
echo ""
echo "[INFO]  Creating storage and model directories..."
mkdir -p uploads uploads/clips data/chroma nemo_cache logs models/embeddings models/diarization
chmod -R 755 uploads data nemo_cache logs models 2>/dev/null || true
echo "[OK]    Storage and model directories ready"

# 8. Embedding model weights. The repository tracks them with Git LFS: a clone made WITHOUT git-lfs contains a ~130-byte
#    text pointer under the same name, which fails at load time with "Error while deserializing header: header too
#    large" (safetensors reads the pointer's first 8 characters as a length). Detect it now and fetch the real file.
echo ""
echo "[INFO]  Checking embedding model weights (git lfs pull if needed, SHA-256 verified, loaded once)..."
if [ -d ".git" ] && ! git lfs version >/dev/null 2>&1; then
    echo "[WARN]  git-lfs is not installed - it is needed to fetch the model weights from the repository."
    echo "        Ubuntu/Debian: sudo apt-get install -y git-lfs    RHEL: sudo dnf install -y git-lfs"
fi
if ! python -m backend.scripts.ensure_embedding_model --load; then
    echo "[ERROR] The embedding model is not usable (see the FIX above)."
    exit 1
fi

# 9. Full preflight gate: dependencies, config, directories, FFmpeg (converts a generated video), embedding model
#    (real vectors), ChromaDB (write -> retrieve -> reopen), PostgreSQL (schema + insert/update/read round trip,
#    rolled back) and AI Engine (LLM + STT). Prints FIX: lines for every problem; never prints secrets.
echo ""
echo "[INFO]  Running the deployment preflight (python -m backend.scripts.preflight)..."
python -m backend.scripts.preflight

echo ""
echo "==============================================================================="
echo "                BACKEND SETUP COMPLETE"
echo "==============================================================================="
echo ""
echo " Run the backend with:"
echo "   bash run_backend.sh"
echo ""
echo " Or manually:"
echo "   source venv/bin/activate"
echo "   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
echo ""
echo " Health check:"
echo "   curl http://localhost:8000/api/v1/system/health"
echo "==============================================================================="

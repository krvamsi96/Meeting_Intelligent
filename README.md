# Enterprise AI Meeting Intelligence Platform — Universal Enterprise Edition v2.0

Enterprise-grade, high-performance backend platform for automated AI meeting transcription, speaker diarization, executive summaries, Minutes of Meeting (MoM), action item extraction, topic detection, sentiment analysis, similarity retrieval, and interactive RAG chatbot querying powered by Universal Enterprise AI infrastructure.

---

## Technical Stack & Architecture

- **Backend Framework**: Python 3.11+ / FastAPI / Uvicorn (ASGI)
- **Database Layer**: PostgreSQL (Production-only), SQLAlchemy 2.0 ORM, and connection pooling.
- **AI Models & Pipeline**:
  - Speech-to-Text: Whisper-Large-v3 via AIRAWAT or Groq (`STT_PROVIDER`)
  - Spoken language: the upload's `language` form field, else `DEFAULT_WHISPER_LANGUAGE`; Groq can also detect it,
    AIRAWAT cannot (an upload without a language then fails with a clear message)
  - Speaker Diarization: NVIDIA NeMo TitaNet speaker embeddings + spectral clustering (PyAnnote 3.1 as fallback)
  - Text generation (summary, MoM, agenda, decisions, action items, chat): ONE configured provider, `AI_PROVIDER` =
    `airawat` (default) | `ollama` (local) | `groq` | `gemini`. A failing provider is retried and then the meeting
    fails; no other provider is ever used in its place. Gemini and Ollama do not transcribe here, so with them
    `STT_PROVIDER` must be set. Check the configuration with `python -m backend.scripts.check_ai_provider`.
  - Vector Database & RAG: ChromaDB (persistent, the only vector store — locally and on the server) & sentence-transformers (`all-MiniLM-L6-v2`). PostgreSQL keeps all meeting data; ChromaDB keeps only embeddings. Configure with `VECTOR_DB=chromadb` and `CHROMA_PERSIST_DIRECTORY` (`./data/chroma` locally, an absolute path on persistent storage on the server); run a single API worker. See `TROUBLESHOOTING.md` §9.
- **Evaluation Engine**: Universal Evaluation Engine 4.0.0 (WER, CER, DER, ROUGE-1/2/L, BERTScore F1, NPMI C_v Topic Coherence, nDCG@10). Zero hardcoded or synthetic metrics.

---

## Directory Structure

```
.
├── backend/
│   ├── api/             # FastAPI API routers (chat, evaluation, export, meetings, settings, system, upload)
│   ├── config/          # Centralized environment configuration (Settings)
│   ├── database/        # Connection pooling, dialect management & migrations
│   ├── evaluation/      # Reference-grounded AI evaluation engine & metrics
│   ├── middleware/      # Request logging & security headers middleware
│   ├── models/          # SQLAlchemy ORM database models
│   ├── prompts/         # Structured LLM prompt templates
│   ├── rag/             # Document chunking & vector indexing utilities
│   ├── repositories/    # Repository pattern data access layer
│   ├── schemas/         # Pydantic v2 validation schemas
│   ├── services/        # Service layer (audio, diarization, export, live metrics, vector, etc.)
│   ├── utils/           # Security, cryptography, and helper utilities
│   ├── workers/         # Background processing pipelines
│   └── tests/           # Unit, integration, and API test suite
├── frontend/            # Angular web app (dashboard, meetings, upload, search, meeting results)
├── Dockerfile.backend   # Linux production container definition
├── docker-compose.yml   # Multi-service stack composition
├── requirements.txt     # Python production dependencies
├── .env.example         # Environment variables template
└── README.md            # Platform documentation
```

---

## Quick Start Guide

### Prerequisites
- Python 3.11+
- FFmpeg + FFprobe installed and on the system PATH
- PostgreSQL 14+ (application data) - ChromaDB (vectors) is embedded and needs nothing installed
- The two offline models (embedding, speaker) - see `LINUX_DEPLOYMENT.md`, "Before the first start"

### Installation
```bash
# Linux server (full procedure, systemd / PM2 / Docker: LINUX_DEPLOYMENT.md)
python3.11 -m venv venv && source venv/bin/activate
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env            # then edit it (AI_MODE=server in production)
python -m backend.scripts.fetch_models --all      # once, with internet (or `git lfs pull` for the embedding model)
bash setup_backend.sh           # directories, model-file check, and the full deployment preflight

# Windows (development)
setup_windows.bat
```

### Running the Backend
```bash
# Linux
bash run_backend.sh             # runs the startup preflight, then a single uvicorn worker

# Windows
run_backend.bat
```
The server starts at `http://127.0.0.1:8000`. Swagger API documentation is available at `http://127.0.0.1:8000/docs`
(when `ENABLE_SWAGGER=true`).

### Running the web app (Angular)
Requires Node.js 20+.
```bash
cd frontend
npm install                     # first time only
npm start                       # http://127.0.0.1:4200 - /api is proxied to the backend on port 8000 (proxy.conf.json)
npx ng test --watch=false       # unit and component tests
npx ng build                    # production build -> frontend/dist/frontend/browser
```
Windows: `run_all.bat` starts both servers. In production serve `frontend/dist/frontend/browser` and the API from the
SAME origin (a reverse proxy forwards `/api/` to uvicorn); the app calls the relative path `/api/v1` and contains no
backend address, keys or provider settings.

Administrators configure the AI providers at `/admin/settings` (not linked in the navigation; it asks for
`ADMIN_API_KEY`). Ordinary screens never show provider names, model ids, endpoints or technical errors: the API
returns plain-language messages (`backend/utils/processing_status.py`) and keeps the technical cause in the logs.

### Verifying an installation
```bash
python -m backend.scripts.preflight            # dependencies, config, FFmpeg, models, ChromaDB, PostgreSQL, AI Engine
bash healthcheck.sh                            # the running API and the database it is using
python -m backend.scripts.e2e_api_check --url http://127.0.0.1:8000 --file meeting.mp4 --state e2e_state.json
```

### Configuration notes

- **Secrets** live only in `.env` (never committed). Copy `.env.example` and fill it in.
  If secrets were ever committed, follow `SECURITY_SECRET_ROTATION.md`.
- **Admin endpoints** (all of `/settings/ai`, `/settings/test`, `/settings/models`, every `POST /settings/*`,
  `POST /system/clear-cache`) are protected by `ADMIN_API_KEY` when it is set (`X-Admin-Key` header). When it is
  empty they are OPEN - set it on every shared deployment.
- **Health**: `GET /system/health` is the detailed operator/monitoring view (component names, database target);
  restrict it at the reverse proxy. The web app uses `GET /system/status`, which reveals nothing internal.
- **Rate limits** for chat / upload / URL ingest: `RATE_LIMIT_CHAT_PER_MIN`, `RATE_LIMIT_UPLOAD_PER_MIN`.
- **Single worker**: run uvicorn with one worker (ChromaDB and the live activity stream are per-process).
- **Windows / OneDrive**: do not keep the Python virtualenv inside a OneDrive-synced folder — OneDrive
  locks files while `pip` writes them (`WinError 32`) and leaves a half-installed environment.
  Create it elsewhere, e.g. `python -m venv C:\envsi_meeting` and `pip install -r requirements.txt`.
  `nemo_toolkit` is installed on Linux only (it is skipped on Windows automatically).

---

## License & Support
Enterprise Proprietary Software — Universal Enterprise Meeting Intelligence Platform v2.0. All Rights Reserved.
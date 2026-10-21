# Linux Deployment Guide - AI Meeting Intelligence Platform

```text
Linux server ─ FastAPI (one uvicorn process) ─┬─ AIRAWAT   LLM + speech-to-text  (the ONLY AI provider)
                                              ├─ PostgreSQL  application data (meetings, transcripts, MoM, ...)
                                              └─ ChromaDB    vectors for RAG / similar-meeting search (on local disk)
```

## Prerequisites

| Component | Min Version | Install |
|-----------|-------------|---------|
| Ubuntu/Debian | 20.04 LTS+ | - |
| Docker (Option A) | 24.x | curl -fsSL https://get.docker.com \| sh |
| Docker Compose (Option A) | v2.x | Included with Docker |
| Python (Option B) | 3.11 | sudo apt install python3.11 python3.11-venv |
| FFmpeg + FFprobe | 4.x+ | sudo apt install ffmpeg |
| libsndfile, libmagic | any | sudo apt install libsndfile1 libmagic1 |
| curl | any | sudo apt install curl |
| git-lfs (only if you clone the model with git) | any | sudo apt install git-lfs && git lfs install |
| Network | - | outbound HTTPS to `apis.airawat.cdac.in:443`; TCP to PostgreSQL (5432) |

## Before the first start: the three offline models

The API runs fully offline and never downloads anything at run time. These model directories must exist on the server
(none is created by `git clone` alone):

| Model | Directory | Without it |
|-------|-----------|------------|
| Embedding `all-MiniLM-L6-v2` (91 MB, Git LFS) | `models/embeddings/all-MiniLM-L6-v2/` | no embeddings: RAG and similar-meeting search are off (health `degraded`). **A clone without git-lfs holds a 133-byte pointer here - the source of `Error while deserializing header: header too large`.** |
| NeMo TitaNet `titanet_large.nemo` | `models/diarization/` | speaker counts come from a weak fallback embedder |

```bash
# on a machine with internet (the server itself, if it has any); downloads are verified
python -m backend.scripts.fetch_models --all          # or --embedding / --diarization (diarization needs Linux + NeMo)
# copy them to an offline server:
bash deploy_model.sh <user> <server_host> /path/to/project
```

The embedding weights can alternatively come from git: `git lfs install && git lfs pull`. `Jenkinsfile`, `setup_backend.sh`
and `deploy_docker.sh` stop with instructions if they find the pointer instead of the real file.

## Production configuration (`.env`)

Use the project's own variable names (see `.env.example`); nothing environment-specific lives in the code.

```text
AI_MODE=server                          # production: SERVER_* database is mandatory; the app refuses to start otherwise
SERVER_AIRAWAT_API_KEY=...              # the key comes from the environment only; it is never logged or returned
SERVER_AIRAWAT_BASE_URL=https://apis.airawat.cdac.in/nic/llama70b/v1
SERVER_AIRAWAT_WHISPER_URL=https://apis.airawat.cdac.in/nic/Whisper-Large-v3/v1/audio/transcriptions
SERVER_AIRAWAT_LLM_MODEL=meta/llama-3.3-70b-instruct
SERVER_DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME     # or SERVER_DB_HOST/PORT/DATABASE/USER/PASSWORD
DB_SSLMODE=require                      # if the database server requires TLS
CHROMA_PERSIST_DIRECTORY=/var/lib/ai_meeting/chroma                  # persistent disk; never a temp dir
UPLOAD_DIR=/var/lib/ai_meeting/uploads
UVICORN_WORKERS=1                       # ChromaDB's directory has ONE owning process
AIRAWAT_TIMEOUT_SEC=180                 # per ATTEMPT; the gateway sometimes never answers a request (its own cut-off is ~300 s)
AIRAWAT_RETRY_COUNT=3                   # and AIRAWAT_HEALTH_CHECK_TIMEOUT_SEC >= 30; measure: check_airawat --repeat 20
SECRET_KEY=...  ADMIN_API_KEY=...       # python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

`AI_MODE=auto` (the default) tries the server database and, if it does not answer at boot, **continues on the LOCAL
database** - `GET /api/v1/system/health` then shows `database_target.fallback_active: true` and `status: degraded`. AIRAWAT is
the only AI provider in every mode except `local` (development): a failed call is retried per `AIRAWAT_RETRY_COUNT` and then
fails with a clear error; nothing is ever handed to OpenAI, Gemini, Groq, Ollama or any other engine.

## Option A: Docker Deployment (Recommended)

### Step 1: Extract project
```bash
tar -xzf AI_Meeting_Intelligence.tar.gz
cd AI_Meeting_Intelligence
```

### Step 2: Configure environment
```bash
cp .env.example .env
nano .env
```

Docker Compose does **not** read the `SERVER_*` / `LOCAL_*` sections of `.env`, and the container does not
see `.env` at all: `docker-compose.yml` passes an explicit list of variables. Set these (unprefixed) names:

**Required fields:**
- `AIRAWAT_API_KEY=your_actual_key`
- `DB_PASSWORD=your_strong_password` (set it **before the first start** - PostgreSQL stores it in the `pgdata`
  volume on first init; changing `.env` later does not change it). `DB_USER` / `DB_DATABASE` are optional
  (defaults `postgres` / `govindb`). `DATABASE_URL` is built by compose from these - do not set it.
- `CORS_ORIGINS=http://YOUR_SERVER_IP:3000`

**Strongly recommended:**
- `SECRET_KEY` - `deploy_docker.sh` generates it if empty. It encrypts provider API keys stored in the
  database; without it a key file is generated inside the container and is lost when the container is rebuilt.
- `ADMIN_API_KEY` - protects `POST /api/v1/settings/*` and `POST /api/v1/system/clear-cache`
  (clients send `X-Admin-Key: <value>`). Empty means those endpoints are open.

### Step 3: Deploy
```bash
bash deploy_docker.sh
```
It checks the model weights, builds, starts PostgreSQL + the backend, waits for `/system/health`, then runs the preflight
inside the container.

### Step 4: Verify
```bash
docker compose ps
curl http://localhost:8000/api/v1/system/health
docker compose exec backend python -m backend.scripts.preflight --skip-chromadb
docker compose logs -f backend
```

`/system/health` always answers quickly. Right after a start it may report `"status": "degraded"` for a
short while: the embedding model is still loading in the background and the AIRAWAT reachability probe has not
finished. Both settle within a minute; HTTP status stays 200 throughout.

## Option B: Bare-Metal (PM2 or systemd)

```bash
# Install dependencies
sudo apt install -y python3.11 python3.11-venv ffmpeg libsndfile1 libmagic1 git-lfs

# Create virtualenv
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt            # installs NeMo on Linux; backend/requirements.docker.txt is the same list

# Configure (SERVER_* / LOCAL_* sections are used here; see .env.example)
cp .env.example .env && nano .env

# Provision the offline models (see above), then run the full gate:
bash setup_backend.sh                      # dirs, model-weights check, and `python -m backend.scripts.preflight`
```

Start it with **PM2**:

```bash
npm install -g pm2
pm2 start run_backend.sh --name aimeet-backend
pm2 save && pm2 startup
```

or **systemd** (`/etc/systemd/system/aimeet.service`):

```ini
[Unit]
Description=AI Meeting Intelligence backend
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
Type=simple
User=aimeet
WorkingDirectory=/opt/ai_meeting
ExecStart=/bin/bash /opt/ai_meeting/run_backend.sh
Restart=on-failure
RestartSec=10
KillSignal=SIGTERM
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

`run_backend.sh` changes to its own directory, reads `.env` with python-dotenv (so values with spaces or `|`, for example
`REPORT_FOOTER_TEXT=Acme | Powered by X`, are fine), lower-cases `LOG_LEVEL` for uvicorn, and runs
`python -m backend.scripts.preflight --startup` before `exec`-ing uvicorn: it refuses to start on a missing dependency or an
unreachable PostgreSQL, and prints `FIX:` lines for everything else (missing FFmpeg, model files, a fallback database ...)
without blocking. Bypass with `SKIP_PREFLIGHT=1`. Run the API with a **single worker** (`UVICORN_WORKERS=1`).

`bash stop.sh` stops this project's uvicorn gracefully (SIGTERM, then SIGKILL after `STOP_TIMEOUT` seconds); under PM2 /
systemd use their own stop command.

## Verify the deployment

```bash
# 1. everything the platform depends on, from THIS machine (no secrets printed; non-zero exit on failure)
python -m backend.scripts.preflight                 # add --skip-airawat to avoid outbound AI calls
python -m backend.scripts.check_airawat --repeat 20 --timeout 20    # latency / stall rate of the AIRAWAT gateway from here

# 2. the running API
bash healthcheck.sh                                  # shows each service and the database in use

# 3. real recordings end to end (uploads, waits, verifies every endpoint and the content of every result)
python -m backend.scripts.e2e_api_check --url http://SERVER:8000 --file meeting_a.mp4 --file meeting_b.mp4 --state e2e_state.json
# restart the API, then prove nothing was lost or mixed up:
python -m backend.scripts.e2e_api_check --url http://SERVER:8000 --verify-state e2e_state.json
```

`preflight` proves: interpreter and dependencies; configuration (fallback database, worker count, secrets); writable
directories; FFmpeg converting a generated video (and rejecting a corrupt file); the embedding model producing finite,
correctly sized, semantically meaningful vectors; the configured spoken-language default and the speaker model loading; ChromaDB storing, retrieving,
isolating by meeting and surviving a close/reopen; PostgreSQL schema creation and an insert / read / update / latest-output round
trip for two meetings that must not overwrite each other (rolled back); AIRAWAT answering with the configured LLM and Whisper.

## Log Locations

| File | Contents |
|------|----------|
| logs/application.log | All INFO+ messages |
| logs/error.log | ERROR+ only (first check on failure) |
| logs/api.log | HTTP request/response |
| logs/ai_pipeline.log | Pipeline stages, transcription, diarization |
| logs/database.log | DB connection and query logs |

If `logs/` is not writable the API still starts and logs to the console only (a warning says so) - under
PM2 that output is in `pm2 logs`; under Docker in `docker compose logs`.

**Log format:**
```text
2026-08-20 10:15:21.345 | ERROR | diarization_service | run_diarization | L241 | job=98765 | meeting=98765 | msg
```

## Health Checks

```bash
curl http://localhost:8000/api/v1/system/health           # services, database_target (no credentials), warnings
curl http://localhost:8000/api/v1/system/health/airawat   # LLM + Whisper, model, latency, key fingerprint only
tail -f logs/error.log
tail -f logs/ai_pipeline.log
```

## Docker Commands

```bash
docker compose up -d
docker compose down
docker compose restart backend
docker compose logs -f backend --tail=100
docker compose build backend
```

## PM2 Commands

```bash
pm2 status
pm2 restart aimeet-backend
pm2 logs aimeet-backend --lines 200
```

## Database Backup

```bash
# Docker
docker exec mom_postgres pg_dump -U postgres govindb > backup_$(date +%Y%m%d).sql

# Bare-metal
pg_dump -h DB_HOST -U DB_USER DB_NAME > backup_$(date +%Y%m%d).sql

# Restore
psql -h DB_HOST -U DB_USER DB_NAME < backup_20260820.sql
```

(Use your own `DB_USER` / `DB_DATABASE` in place of `postgres` / `govindb` if you changed them.)

ChromaDB holds only derived data: back up its directory like any database directory, or rebuild it from PostgreSQL with
`python -m backend.scripts.reindex_vectors`.

## Firewall

```bash
sudo ufw allow 8000/tcp
sudo ufw allow 3000/tcp
sudo ufw enable
```

PostgreSQL is published on `127.0.0.1:5432` only, so it is not reachable from the network. Note that
`ufw deny 5432` would **not** be enough on its own: ports published by Docker bypass ufw's rules. For
admin access from another machine use an SSH tunnel: `ssh -L 5432:127.0.0.1:5432 user@server`.

See `TROUBLESHOOTING.md` for the embedding-model error, AIRAWAT stalls, fallback databases and ChromaDB.

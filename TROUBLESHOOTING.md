# Troubleshooting Guide — AI Meeting Intelligence Platform

## How to Read Logs

**Error format:**
```
2026-08-20 10:15:21.345 | ERROR | module_name | function_name | L241 | job=98765 | meeting=98765 | message
```

**Find all errors quickly:**
```bash
tail -f logs/error.log
grep "ERROR" logs/application.log | tail -50
grep "job=12345" logs/ai_pipeline.log   # trace a specific job
```

---

## Common Errors & Fixes

### 1. Database Connection Failed at Startup

**Error:**
```
RuntimeError: PostgreSQL database error: connection refused
```

**Causes & Fix:**
- PostgreSQL not running: sudo systemctl start postgresql
- Wrong DATABASE_URL in .env — verify host, port, user, password
- DB_PASSWORD contains special chars (@, !) — URL-encode them in DATABASE_URL (@ -> %40)
- Firewall blocking port 5432: sudo ufw allow from APP_IP to any port 5432

**Verify:**
```bash
psql -h DB_HOST -U DB_USER -d DB_NAME -c "SELECT 1"
```

---

### 2. AIRAWAT API Key Invalid

**Error:**
```
AIRAWAT health check failed: 401 Unauthorized
```

**Fix:**
- Check AIRAWAT_API_KEY in .env — must be full JWT token
- Token may have expired — obtain new key from https://airawat.cdac.in
- Verify with: curl -H "Authorization: Bearer YOUR_KEY" https://apis.airawat.cdac.in/nic/llama70b/v1/models

---

### 3. Audio Extraction Failed (FFmpeg)

**Error:**
```
Audio extraction failed — FFmpeg returned non-zero exit code
```

**Fix:**
```bash
which ffmpeg          # Must return a path
ffmpeg -version       # Must be 4.x+
sudo apt install ffmpeg   # If missing
```

---

### 4. HuggingFace Token Missing (PyAnnote)

**Error:**
```
HF_TOKEN is empty but PYANNOTE_ENABLED=true
RuntimeError: Diarization model initialization failed
```

**Fix:**
- Set HF_TOKEN=your_hf_token in .env
- Get token at https://huggingface.co/settings/tokens
- Accept model license at https://huggingface.co/pyannote/speaker-diarization-3.1
- OR: set PYANNOTE_ENABLED=false and DIARIZATION_MODE=NEMO_ONLY

---

### 5. NeMo Model Download Slow / Failed

**Symptom:** First diarization request takes 5-10 minutes (model downloading ~1-2 GB).

**Fix:**
- Ensure 
emo_cache/ directory is writable and on a fast disk
- In Docker, 
emo_cache volume persists models across restarts
- If behind a proxy: set HTTP_PROXY / HTTPS_PROXY env vars

---

### 6. Speaker Diarization Wrong Results

**Diagnosis — check ai_pipeline.log:**
```
grep "SPEAKER_DIARIZATION" logs/ai_pipeline.log
grep "speakers=" logs/ai_pipeline.log
```

**Common causes:**
- Very short audio (< 2 seconds): minimum ~5 seconds needed per speaker
- All-silence or very low volume: check Audio energy near zero (RMS=...) in logs
- Single speaker detected as multiple: increase NEMO_SPEAKER_MODEL quality (	itanet_large is best)

---

### 7. Application Out of Memory

**Symptom:** Process killed (OOM) during NeMo or transcript processing.

**Fix:**
- NeMo requires at least **8 GB RAM** (16 GB recommended)
- Reduce UVICORN_WORKERS=1 in .env
- Set NEMO_DEVICE=cpu and ensure no concurrent pipelines
- Set NEMO_ENABLED=false and DIARIZATION_MODE=PYANNOTE_ONLY (lower memory)

---

### 8. Uploads Directory Permission Denied

**Error:**
```
PermissionError: [Errno 13] Permission denied: './uploads/file.mp4'
```

**Fix:**
```bash
mkdir -p uploads uploads/clips data/chroma nemo_cache logs
chmod 755 uploads uploads/clips data data/chroma nemo_cache logs
# In Docker: handled automatically by ENTRYPOINT chown
```

---

### 9. ChromaDB / Vector DB unavailable

ChromaDB is the only vector database (RAG, semantic search, similar-meeting search); PostgreSQL holds everything else.
It persists to `CHROMA_PERSIST_DIRECTORY` (local: `./data/chroma`; server: an absolute path on persistent storage).

**Check:**
```bash
curl -s http://<host>:8000/api/v1/system/health
#   "vector_database": {"backend": "chromadb", "status": "available"}
#   "vector_database": {"backend": "chromadb", "status": "unavailable", "error": "<reason>"}
python -m backend.scripts.check_vector_db      # stop the API first; write -> query -> delete round trip
```
The startup log states it too: `Vector database: ChromaDB` / `ChromaDB status: available` / `ChromaDB path: ...`.
The API keeps running while ChromaDB is unavailable (chat then answers without transcript retrieval) and retries
by itself about once a minute.

| `error` says | Cause and fix |
|---|---|
| `already in use by another process` | An embedded ChromaDB directory can be owned by **one process**. Run a single worker (`UVICORN_WORKERS=1`, the default in `start.sh`, `run_backend.sh` and Docker) and stop anything else using the directory (a second API, `reindex_vectors`, `check_vector_db`). The lock is released automatically when that process exits. |
| `still a placeholder` | `.env` still has `CHROMA_PERSIST_DIRECTORY=<configure-for-environment>` (shipped by older `.env.example` templates; the current one uses `./data/chroma`). Set a real path. |
| `cannot be created` / `not writable` | The path is not creatable/writable by the API user: `mkdir -p <dir> && chown <api-user> <dir>`. In Docker it must be the mounted volume (`/app/data/chroma`). |
| `VECTOR_DB='...' is not supported` | Only `VECTOR_DB=chromadb` exists. |
| `package is not installed` | `pip install -r requirements.txt` (needs `chromadb>=1.5,<2`). |
| `...-dimensional vectors` / `distance ... cosine is required` | The store was built with a different embedding model or settings. Rebuild: `python -m backend.scripts.reindex_vectors --reset`. |

**Vectors are derived data.** PostgreSQL holds every transcript, so a lost or corrupted ChromaDB directory is never a data
loss - only a rebuild:
```bash
# stop the API, then (optionally move the broken directory aside first)
python -m backend.scripts.reindex_vectors            # every meeting that has a transcript
python -m backend.scripts.reindex_vectors --meeting-id 12 --meeting-id 15
python -m backend.scripts.reindex_vectors --dry-run  # list what would be indexed
```
Chat also re-indexes a meeting lazily the first time it is asked about and has no vectors.

**Server deployment checklist**
- `CHROMA_PERSIST_DIRECTORY` is an absolute path on a persistent disk/volume (never a temp directory, never a Windows or
  developer path). In Docker the `chroma_data` named volume is mounted there by `docker-compose.yml`.
- Deployments must not overwrite or delete that directory (`Jenkinsfile` already excludes `data/`); back it up like any
  database directory, or just rebuild it with `reindex_vectors`.
- Exactly one API process/worker per directory. Scale out with more meetings per worker, not more workers.

#### Embedding model: `Error while deserializing header: header too large`

**Root cause (reproduced from the committed blob):** the embedding weights (`models/embeddings/all-MiniLM-L6-v2/model.safetensors`,
90 868 376 bytes) are stored in **Git LFS** (`.gitattributes`); what git itself holds for that path is a 133-byte text *pointer*
(`version https://git-lfs.github.com/spec/v1 ...`). Any checkout made without git-lfs - for example a plain `git clone`, or a CI
workspace that is then packed into the deployment tarball - therefore ships the pointer under the real file name. safetensors
reads a model file's first 8 bytes as the length of its JSON header; for the pointer those bytes are the text `version ` =
2 336 927 755 350 992 246, and it answers exactly *"Error while deserializing header: header too large"*. It is not a
transformers / sentence-transformers / huggingface_hub / Python version problem.

The loader now inspects the file **content** up front (LFS pointer, truncated or corrupt weights) and, if the model cannot
be used, logs once at startup:

```
ERROR [ModelLoader] Embedding model UNAVAILABLE - no usable copy of 'all-MiniLM-L6-v2': <dir> holds Git LFS pointer file(s) ... model.safetensors
  FIX: ...
```

**No fake embeddings are ever produced.** The API stays up, but embedding-dependent features (RAG indexing, similar-meeting
search) are off and `GET /api/v1/system/health` reports `status: degraded`, `services.models: false` with
`models.embedding.problem` and `.fix`. (Right after a start it may briefly report `degraded` while the model is still
*loading* in the background - the endpoint never waits for the load. Only a `degraded` that persists together with
`models.embedding.problem` means the model really cannot be loaded.) Deterministic word-hash vectors exist only behind the
explicit development opt-in `EMBEDDING_ALLOW_LEXICAL_FALLBACK=true`.

**Repair** (any one of these), then restart the API:

```bash
python -m backend.scripts.preflight --skip-airawat --skip-chromadb   # names the exact problem and the fix
ls -l models/embeddings/all-MiniLM-L6-v2/model.safetensors           # 133 bytes = pointer; 90868376 = real
git lfs install && git lfs pull                                      # needs git-lfs and the LFS remote
python -m backend.scripts.fetch_models --embedding                   # no LFS: downloads and SHA-256-verifies the same file
bash deploy_model.sh <user> <host> [deploy_path]                     # or copy it from a machine where it works
```

**Prevention:** `Jenkinsfile` now fails the build in *Verify Embedding Model* (after trying `git lfs pull`) instead of
shipping a pointer; `setup_backend.sh`, `deploy_docker.sh`, and `run_backend.sh` / `start.sh` (startup preflight) check the
file too.

After the model is repaired no manual re-index is needed: every stored vector is tagged with the embedding engine that
produced it, and at the next start the backfill (`VECTOR_BACKFILL_ON_STARTUP=true`) adds the meetings that PostgreSQL has but
ChromaDB lacks and re-embeds vectors made by a different engine; a meeting's chunk vectors are rebuilt the next time it is
chatted with. `python -m backend.scripts.reindex_vectors` does everything at once.

#### Other offline models a fresh server needs

| Model | Location | If missing | Provision (once, with internet) |
|-------|----------|------------|--------------------------------|
| NeMo TitaNet `titanet_large.nemo` | `models/diarization/` | speaker counts come from a weak numpy fallback embedder (a 4-speaker meeting can come out as 1) | `python -m backend.scripts.fetch_models --diarization` (on Linux) |

`models/diarization` is git-ignored, so `git clone` never delivers them; copy `models/` to offline servers (`deploy_model.sh`).
Note that a Transformers-format `openai/whisper-*` download is **not** usable here (it has a `config.json` but no `model.bin`).

#### Server starts with `Invalid value for '--log-level': 'INFO'`

uvicorn only accepts a lowercase level, while `.env.example` says `LOG_LEVEL=INFO`. `run_backend.sh`, `start.sh` and the Docker
entrypoint now lower-case it; if you start uvicorn yourself use `--log-level info`.

#### Meetings land in the wrong database (`database_target.fallback_active: true`)

`AI_MODE=auto` tries the `SERVER_*` database first and, if it does not answer at boot, continues on the `LOCAL_*` one.
`GET /api/v1/system/health` shows which database is in use (`database_target`, no credentials) and reports `degraded` with a
warning when that is a fallback; the startup log has an `[DB] FALLBACK DATABASE ACTIVE` ERROR line. On a production server set
`AI_MODE=server`: the API then refuses to start when the server database is unreachable.

#### AIRAWAT: intermittent timeouts

The gateway sometimes accepts a request and never answers it (measured from one workstation: 3 of 12 calls hung for the whole
timeout, the rest answered in under 2.3 s). Health probes and `python -m backend.scripts.check_airawat` now retry such
timeouts / 5xx (`1 + AIRAWAT_RETRY_COUNT` attempts) and report `attempts`; a 401 / 404 (wrong key or model) fails at once.
Use `AIRAWAT_RETRY_COUNT>=2` and `AIRAWAT_HEALTH_CHECK_TIMEOUT_SEC>=30`.

---

### 10. Tracing a Failed Pipeline Job

**Given:** job_id=12345 from API response or error notification.

```bash
# See complete trace for this job
grep "job=12345" logs/ai_pipeline.log

# See what stage it failed at
grep "job=12345" logs/ai_pipeline.log | grep -E "FAILED|ERROR"

# See database logs for this job
grep "12345" logs/database.log
```

**Expected output when pipeline fails:**
```
2026-08-20 10:15:21 | ERROR | diarization_service | run_diarization | L241 | job=12345 | meeting=12345 | SPEAKER_DIARIZATION FAILED
```

This gives you: **file** (diarization_service), **function** (
un_diarization), **line** (241), **stage** (from context), **job** ID.

---

## Quick Diagnostic Checklist

```bash
# 1. All containers running?
docker compose ps

# 2. Backend health
curl http://localhost:8000/api/v1/system/health | python3 -m json.tool

# 3. Recent errors
tail -50 logs/error.log

# 4. Database reachable?
psql -h DB_HOST -U DB_USER -d DB_NAME -c "SELECT COUNT(*) FROM meetings"

# 5. AIRAWAT reachable?
curl -s -o /dev/null -w "%{http_code}" https://apis.airawat.cdac.in/nic/llama70b/v1/models

# 6. FFmpeg installed?
ffmpeg -version | head -1

# 7. Disk space
df -h uploads/ data/ nemo_cache/ logs/
```

---

### 11. Upgrading from a release that stored vectors in PostgreSQL

Earlier releases kept embeddings in the PostgreSQL tables `meeting_vectors` and `transcript_chunk_vectors`. The application
no longer reads them, no PostgreSQL extension is needed, and the application never runs any extension DDL or needs
superuser rights.

**Automatic:** at startup the API adds every meeting that PostgreSQL has but ChromaDB lacks (one embedding per meeting, in a
background thread, so similar-meeting / duplicate detection sees your existing meetings; disable with
`VECTOR_BACKFILL_ON_STARTUP=false`). The startup log says `Vector backfill: N meeting(s) to add to ChromaDB` and then
`Vector backfill done: N meeting(s) added ...`. Transcript chunks for RAG are created the first time a meeting is chatted with.

**Full rebuild** (chunk vectors included; rebuilds ChromaDB from the transcripts already in PostgreSQL - nothing is lost):
```bash
# 1. set CHROMA_PERSIST_DIRECTORY in .env (see section 9), stop the API
python -m backend.scripts.reindex_vectors
# 2. start the API and confirm: curl -s http://<host>:8000/api/v1/system/health   ->  vector_database.status = "available"
```
**Optional clean-up** once you are satisfied (the application never touches these tables, so this is up to your DBA):
```sql
DROP TABLE IF EXISTS meeting_vectors, transcript_chunk_vectors;
```

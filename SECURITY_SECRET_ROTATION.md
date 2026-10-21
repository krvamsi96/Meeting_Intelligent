# Secret Rotation & Git History Scrub

`.env` was committed to this repository: **26 commits** on `main` touch it, and they are pushed to `origin`
(`main` and `ai_meeting_7`) — verified 2026-10-21. Earlier versions of this note said it was already untracked;
that was not true at the time. The removal (`git rm --cached .env`) is now **staged** in the working copy and takes
effect with the next commit; `.gitignore` excludes `.env`. Removing the file from the index does **not** remove the
values from history.

Credential variables that were committed with a value (names only, checked over every historical `.env`):
`AIRAWAT_API_KEY`, `SERVER_AIRAWAT_API_KEY`, `LOCAL_AIRAWAT_API_KEY`, `DB_PASSWORD`, `SERVER_DB_PASSWORD`,
`LOCAL_DB_PASSWORD`, `DATABASE_URL`, `SERVER_DATABASE_URL`, `LOCAL_DATABASE_URL`, `HF_TOKEN`, `OLLAMA_API_KEY`,
`LOCAL_OLLAMA_API_KEY`. The Groq / Gemini / DeepSeek keys in the current workstation `.env` were never committed —
keep it that way.

## 1. Rotate these credentials (do this regardless of history scrub)

| Secret | Where it lives | Action |
|---|---|---|
| `SERVER_AIRAWAT_API_KEY`, `AIRAWAT_API_KEY`, `LOCAL_AIRAWAT_API_KEY` | AIRAWAT / CDAC console | Revoke every leaked token, issue a new one, update the server `.env` |
| `SERVER_DB_PASSWORD` (`dbadmin` on `10.192.110.24`) | PostgreSQL | `ALTER USER dbadmin WITH PASSWORD '<new>';` then update `SERVER_DB_PASSWORD` **and** `SERVER_DATABASE_URL` (URL-encode `@`→`%40`, `#`→`%23`, etc.) |
| `HF_TOKEN` (`hf_…`) | https://huggingface.co/settings/tokens | Delete the leaked token, create a new one (only needed if `PYANNOTE_ENABLED=true`) |
| `LOCAL_DB_PASSWORD` (`root`) | local dev only | low risk; rotate if the dev box is shared |

After rotating, the old history values are useless even if the history is not rewritten.

## 2. (Optional) Scrub the values from git history

Only do this if the repo is private-but-shared or you want the values gone entirely.
**This rewrites history and requires a force-push; every other clone must re-clone.**
Coordinate with anyone else using the repo first.

### Option A — git filter-repo (recommended)

```bash
pip install git-filter-repo

# from a FRESH clone (filter-repo refuses to run on a repo with a remote + local changes)
git clone <repo-url> repo-scrub && cd repo-scrub

# remove the file from all history
git filter-repo --path .env --invert-paths

# (optional) also redact any stray literal secrets that leaked elsewhere
cat > /tmp/redactions.txt <<'EOF'
<LEAKED_AIRAWAT_API_KEY>==>REDACTED_AIRAWAT_TOKEN
<LEAKED_DB_PASSWORD>==>REDACTED_DB_PASSWORD
<LEAKED_HF_TOKEN>==>REDACTED_HF_TOKEN
EOF
git filter-repo --replace-text /tmp/redactions.txt

git remote add origin <repo-url>
git push --force --all
git push --force --tags
```

### Option B — BFG Repo-Cleaner

```bash
# https://rtyley.github.io/bfg-repo-cleaner/
git clone --mirror <repo-url> repo.git
java -jar bfg.jar --delete-files .env repo.git
cd repo.git && git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push --force
```

### After the force-push
- Everyone else: `git fetch --all && git reset --hard origin/<branch>` (or re-clone).
- If the repo is on a host that keeps stale refs / PR snapshots (GitHub/GitLab), open a
  support request to purge cached views, or assume the values are burned and rely on
  step 1 (rotation).

## 3. Prevent recurrence
- `.env` is in `.gitignore` (`.env`, `.env.*`, `*.env`, keep `!.env.example`).
- Server deploy: inject secrets via the environment (systemd `EnvironmentFile=`,
  PM2 `env`, Docker `--env-file` / secrets) or an out-of-band `.env` that is never
  committed. `.env.example` stays as the only tracked template.
- Consider a pre-commit hook (`gitleaks` / `detect-secrets`) to block future leaks.

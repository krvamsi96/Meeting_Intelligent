#!/usr/bin/env bash
# AI Meeting Intelligence Platform — Healthcheck Script
#
# Usage: bash healthcheck.sh [URL]
#   URL defaults to http://127.0.0.1:${PORT:-8000}/api/v1/system/health  (PORT is read from .env when unset).
# Exit code: 0 = API answered HTTP 200 with status "healthy" or "degraded"; 1 = unreachable / not 200 / no status.
# "degraded" is reported with the failing components: a dependency (AIRAWAT, embedding model, ChromaDB, or a FALLBACK
# database) needs attention but the API itself is serving.

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1

echo "==============================================================================="
echo "       AI MEETING INTELLIGENCE PLATFORM - SYSTEM HEALTH CHECK"
echo "==============================================================================="
echo ""

if [ -z "${PORT:-}" ] && [ -f ".env" ]; then
    PORT="$(grep -E '^PORT=' .env | tail -n 1 | cut -d= -f2- | tr -d ' "'"'"'\r')"
fi
HEALTH_URL="${1:-http://127.0.0.1:${PORT:-8000}/api/v1/system/health}"

if ! command -v curl >/dev/null 2>&1; then
    echo "[WARNING] curl command not found."
    exit 1
fi

BODY_FILE="$(mktemp)"
trap 'rm -f "${BODY_FILE}"' EXIT
HTTP_CODE="$(curl -s -o "${BODY_FILE}" -w '%{http_code}' --max-time 15 "${HEALTH_URL}" || echo 000)"

if [ "${HTTP_CODE}" != "200" ]; then
    echo "[ERROR] Backend Service is OFFLINE or unhealthy at ${HEALTH_URL} (HTTP ${HTTP_CODE})"
    exit 1
fi

PY="$(command -v python3 || command -v python)"
if [ -z "${PY}" ]; then
    echo "[SUCCESS] Backend Service is ONLINE (HTTP 200); python not found to summarise the response."
    cat "${BODY_FILE}"; echo
    exit 0
fi

"${PY}" - "${BODY_FILE}" <<'PYEOF'
import json
import sys

try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    print("[ERROR] The health endpoint did not return JSON.")
    sys.exit(1)

status = data.get("status")
services = data.get("services", {})
target = data.get("database_target", {})
if status not in ("healthy", "degraded"):
    print(f"[ERROR] Unexpected health status: {status!r}")
    sys.exit(1)
print(f"[{'SUCCESS' if status == 'healthy' else 'WARNING'}] Backend Service is ONLINE - status: {status}")
for name, ok in services.items():
    print(f"    {'ok  ' if ok else 'DOWN'} {name}")
if target:
    print(f"    database: {target.get('environment')} {target.get('host')}:{target.get('port')}/{target.get('database')}"
          + ("  <-- FALLBACK DATABASE" if target.get("fallback_active") else ""))
if data.get("warning"):
    print(f"    warning: {data['warning']}")
sys.exit(0)
PYEOF

#!/usr/bin/env bash
# AI Meeting Intelligence Platform — Linux Shutdown Script
#
# Stops ONLY this project's uvicorn process(es), gracefully: SIGTERM first so the API can finish its shutdown
# (it closes ChromaDB and releases its directory lock), SIGKILL only if it has not exited after STOP_TIMEOUT
# seconds. It never kills "whatever owns port 8000": that could be another service, and SIGKILL on a running
# embedded ChromaDB / SQLite can leave the vector store needing recovery.
#
# Under PM2 / systemd use their own stop command instead (pm2 stop aimeet-backend / systemctl stop ...).

PROJECT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
STOP_TIMEOUT="${STOP_TIMEOUT:-45}"

echo "==============================================================================="
echo "       AI MEETING INTELLIGENCE PLATFORM - SHUTDOWN SERVICES"
echo "==============================================================================="
echo ""

# PIDs of "uvicorn backend.main:app" whose working directory is THIS project (other checkouts are left alone).
find_pids() {
    local pid
    for pid in $(pgrep -f "uvicorn backend.main:app" 2>/dev/null); do
        if [ "$(readlink -f "/proc/${pid}/cwd" 2>/dev/null)" = "${PROJECT_DIR}" ]; then
            echo "${pid}"
        fi
    done
}

PIDS="$(find_pids)"
if [ -z "${PIDS}" ]; then
    echo "[INFO] No running backend found for ${PROJECT_DIR}."
    exit 0
fi

echo "[INFO] Stopping FastAPI backend (pid: $(echo "${PIDS}" | tr '\n' ' '))..."
# shellcheck disable=SC2086
kill -TERM ${PIDS} 2>/dev/null || true

for _ in $(seq 1 "${STOP_TIMEOUT}"); do
    [ -z "$(find_pids)" ] && break
    sleep 1
done

REMAINING="$(find_pids)"
if [ -n "${REMAINING}" ]; then
    echo "[WARN] Still running after ${STOP_TIMEOUT}s - forcing (SIGKILL): $(echo "${REMAINING}" | tr '\n' ' ')"
    echo "       ChromaDB is repaired automatically on the next start, but check logs/error.log."
    # shellcheck disable=SC2086
    kill -KILL ${REMAINING} 2>/dev/null || true
    exit 1
fi

echo "[SUCCESS] Backend stopped gracefully."

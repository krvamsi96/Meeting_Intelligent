#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# AI Meeting Intelligence Platform — Complete Linux Server Verification Script
# Run this AFTER deploying to verify every feature works end-to-end.
# Usage: bash test_server.sh [SERVER_IP] [PORT]
# Example: bash test_server.sh 192.168.1.100 8000
#          bash test_server.sh localhost 8000
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
SERVER="${1:-localhost}"
PORT="${2:-8000}"
BASE_URL="http://${SERVER}:${PORT}/api/v1"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PASS=0
FAIL=0
WARN=0

pass()  { echo -e "  ${GREEN}[PASS]${NC} $1"; ((PASS++)); }
fail()  { echo -e "  ${RED}[FAIL]${NC} $1"; ((FAIL++)); }
warn()  { echo -e "  ${YELLOW}[WARN]${NC} $1"; ((WARN++)); }
info()  { echo -e "  ${BLUE}[INFO]${NC} $1"; }
section(){ echo -e "\n${CYAN}━━━ $1 ━━━${NC}"; }

# ── Check curl available ───────────────────────────────────────────────────────
if ! command -v curl &>/dev/null; then
    echo "[ERROR] curl not found. Install: sudo apt-get install curl"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  AI MEETING INTELLIGENCE PLATFORM — SERVER VERIFICATION"
echo "  Target: ${BASE_URL}"
echo "═══════════════════════════════════════════════════════════════════"


# ════════════════════════════════════════════════════════════════
# TEST 1 — Docker Containers Running
# ════════════════════════════════════════════════════════════════
section "TEST 1: Docker Container Status"
if command -v docker &>/dev/null; then
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "mom_backend"; then
        pass "mom_backend container is running"
        BACKEND_STATUS=$(docker inspect --format='{{.State.Health.Status}}' mom_backend 2>/dev/null || echo "unknown")
        info "Backend health status: ${BACKEND_STATUS}"
    else
        fail "mom_backend container NOT running — run: docker compose up -d"
    fi

    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "mom_postgres"; then
        pass "mom_postgres container is running"
        PG_STATUS=$(docker inspect --format='{{.State.Health.Status}}' mom_postgres 2>/dev/null || echo "unknown")
        info "PostgreSQL health status: ${PG_STATUS}"
    else
        fail "mom_postgres container NOT running"
    fi
else
    warn "Docker not found on this machine — skipping container checks"
fi


# ════════════════════════════════════════════════════════════════
# TEST 2 — Backend Reachable
# ════════════════════════════════════════════════════════════════
section "TEST 2: Backend Reachability (Port ${PORT})"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${BASE_URL}/system/health" 2>/dev/null || echo "000")
if [[ "$HTTP_CODE" == "200" ]]; then
    pass "Backend reachable — HTTP 200 OK"
elif [[ "$HTTP_CODE" == "000" ]]; then
    fail "Backend NOT reachable — connection refused or timed out at ${BASE_URL}"
    echo ""
    echo "  → Check: docker compose logs backend | tail -30"
    echo "  → Check: docker compose ps"
    exit 1
else
    warn "Backend responded with HTTP ${HTTP_CODE}"
fi


# ════════════════════════════════════════════════════════════════
# TEST 3 — Health Check (DB + Vector DB + AI Models)
# ════════════════════════════════════════════════════════════════
section "TEST 3: System Health Check"
HEALTH=$(curl -s --max-time 15 "${BASE_URL}/system/health" 2>/dev/null || echo "{}")
echo "  Raw response: ${HEALTH}"

if echo "$HEALTH" | grep -q '"status"'; then
    STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "unknown")
    DB_OK=$(echo "$HEALTH"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('services',{}).get('database',False))" 2>/dev/null || echo "False")
    CHROMA_STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('vector_database',{}).get('status','unknown'))" 2>/dev/null || echo "unknown")
    CHROMA_ERR=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('vector_database',{}).get('error',''))" 2>/dev/null || echo "")
    MODELS_OK=$(echo "$HEALTH"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('services',{}).get('models',False))" 2>/dev/null || echo "False")

    [[ "$STATUS" == "healthy" ]]  && pass "Overall status: healthy"  || warn "Overall status: ${STATUS} (degraded — check services below)"
    [[ "$DB_OK"   == "True"   ]]  && pass "PostgreSQL: connected"     || fail "PostgreSQL: NOT connected — check DB credentials in .env"
    [[ "$CHROMA_STATUS" == "available" ]] && pass "Vector Database: available" || fail "Vector Database: ${CHROMA_STATUS} - ${CHROMA_ERR}"
    [[ "$MODELS_OK" == "True"   ]]  && pass "AI Models: ready"          || warn "AI Models: loading or degraded"
else
    fail "Could not parse health response"
fi



# ════════════════════════════════════════════════════════════════
# TEST 4 — System Metrics
# ════════════════════════════════════════════════════════════════
section "TEST 4: System Metrics API"
METRICS=$(curl -s --max-time 10 "${BASE_URL}/system/metrics" 2>/dev/null || echo "{}")
if echo "$METRICS" | grep -q '"meetings"'; then
    pass "System metrics endpoint responding"
    TOTAL=$(echo "$METRICS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('meetings',{}).get('total',0))" 2>/dev/null || echo "0")
    info "Total meetings in DB: ${TOTAL}"
else
    fail "System metrics endpoint not working — response: ${METRICS}"
fi


# ════════════════════════════════════════════════════════════════
# TEST 5 — AI Settings API (Key Masked)
# ════════════════════════════════════════════════════════════════
section "TEST 5: AI Settings API"
SETTINGS=$(curl -s --max-time 10 "${BASE_URL}/settings/ai" 2>/dev/null || echo "{}")
if echo "$SETTINGS" | grep -q '"active_provider"'; then
    pass "AI settings endpoint responding"
    ACTIVE_PROV=$(echo "$SETTINGS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('active_provider',''))" 2>/dev/null || echo "")
    pass "Active AI Provider: ${ACTIVE_PROV}"
else
    fail "AI settings endpoint error — response: ${SETTINGS}"
fi



# ════════════════════════════════════════════════════════════════
# TEST 6 — AIRAWAT Live Connection Test
# ════════════════════════════════════════════════════════════════
section "TEST 6: AI Provider Connection Test"
info "Testing connection to active AI provider..."
CONN_TEST=$(curl -s --max-time 30 -X POST "${BASE_URL}/settings/test-connection" \
    -H "Content-Type: application/json" \
    -d '{"provider":"ollama"}' 2>/dev/null || echo "{}")
if echo "$CONN_TEST" | grep -q '"ok": true\|"ok":true'; then
    pass "AI Provider: CONNECTED"
else
    MSG=$(echo "$CONN_TEST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('message',''))" 2>/dev/null || echo "")
    warn "AI Provider test: ${MSG:-offline/not responding}"
fi



# ════════════════════════════════════════════════════════════════
# TEST 7 — Meetings List API
# ════════════════════════════════════════════════════════════════
section "TEST 7: Meetings List API"
MEETINGS=$(curl -s --max-time 10 "${BASE_URL}/meetings" 2>/dev/null || echo "[]")
if echo "$MEETINGS" | python3 -c "import sys,json; json.load(sys.stdin)" &>/dev/null; then
    COUNT=$(echo "$MEETINGS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d) if isinstance(d,list) else d.get('total',0))" 2>/dev/null || echo "0")
    pass "Meetings API responding — ${COUNT} meeting(s) in DB"
else
    fail "Meetings API error — response: ${MEETINGS:0:200}"
fi


# ════════════════════════════════════════════════════════════════
# TEST 8 — File Upload (Small Test File)
# ════════════════════════════════════════════════════════════════
section "TEST 8: File Upload API (small test file)"
# Create a tiny test audio file using ffmpeg
TEST_FILE="/tmp/test_meeting_audio.wav"
TEST_ID="TEST_$(date +%s)"

if command -v ffmpeg &>/dev/null; then
    # Generate 3-second silent WAV for testing
    ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -t 3 -q:a 9 -acodec pcm_s16le "$TEST_FILE" -y -loglevel quiet 2>/dev/null
    if [[ -f "$TEST_FILE" ]]; then
        info "Test WAV file created (3 seconds silent): ${TEST_FILE}"
        UPLOAD=$(curl -s --max-time 60 -X POST "${BASE_URL}/upload" \
            -F "file=@${TEST_FILE};type=audio/wav" \
            -F "bharatvc_meeting_id=${TEST_ID}" \
            -F "meeting_title=Automated Test Meeting" 2>/dev/null || echo "{}")
        if echo "$UPLOAD" | grep -q '"status"'; then
            UPLOAD_STATUS=$(echo "$UPLOAD" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || echo "")
            MEETING_ID=$(echo "$UPLOAD"  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id',''))" 2>/dev/null || echo "")
            pass "Upload API working — meeting created (DB id=${MEETING_ID}, status=${UPLOAD_STATUS})"
            echo "$MEETING_ID" > /tmp/test_meeting_id.txt
            info "Test meeting ID saved: ${MEETING_ID}"
        else
            fail "Upload failed — response: ${UPLOAD:0:300}"
        fi
        rm -f "$TEST_FILE"
    else
        warn "Could not create test WAV file"
    fi
else
    warn "ffmpeg not found — skipping upload test (ffmpeg required inside Docker for actual processing)"
fi


# ════════════════════════════════════════════════════════════════
# TEST 9 — Export Formats API
# ════════════════════════════════════════════════════════════════
section "TEST 9: Export API Routes Present"
# Check export route responds (even for non-existent meeting — should return 404 not 500)
for FMT in docx xlsx md html json; do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${BASE_URL}/meetings/99999/export/${FMT}" 2>/dev/null || echo "000")
    if [[ "$CODE" == "404" || "$CODE" == "200" ]]; then
        pass "Export route /${FMT} is registered (HTTP ${CODE})"
    else
        warn "Export route /${FMT} returned HTTP ${CODE}"
    fi
done


# ════════════════════════════════════════════════════════════════
# TEST 10 — Report Settings API
# ════════════════════════════════════════════════════════════════
section "TEST 10: Report Settings API"
REPORT=$(curl -s --max-time 10 "${BASE_URL}/settings/report" 2>/dev/null || echo "{}")
if echo "$REPORT" | grep -q '"company_name"\|"header_text"'; then
    COMPANY=$(echo "$REPORT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('company_name',''))" 2>/dev/null || echo "")
    pass "Report settings endpoint working — company: '${COMPANY}'"
else
    fail "Report settings error — response: ${REPORT:0:200}"
fi


# ════════════════════════════════════════════════════════════════
# TEST 11 — PostgreSQL Direct Check
# ════════════════════════════════════════════════════════════════
section "TEST 11: PostgreSQL Direct Connection"
if command -v docker &>/dev/null && docker ps --format '{{.Names}}' 2>/dev/null | grep -q "mom_postgres"; then
    PG_RESULT=$(docker exec mom_postgres psql -U postgres -d govindb -c "SELECT COUNT(*) FROM meetings;" 2>/dev/null | grep -E "^\s+[0-9]" | tr -d ' ' || echo "error")
    if [[ "$PG_RESULT" != "error" ]]; then
        pass "PostgreSQL direct query OK — meetings count: ${PG_RESULT}"
    else
        fail "PostgreSQL direct query failed — DB may not be initialized yet"
    fi

    # Check tables exist
    TABLES=$(docker exec mom_postgres psql -U postgres -d govindb -c "\dt" 2>/dev/null | grep -c "public" || echo "0")
    info "Tables in govindb: ${TABLES}"
else
    warn "Docker not available — skipping direct PostgreSQL check"
fi


# ════════════════════════════════════════════════════════════════
# TEST 12 — Disk Space & Volume Check
# ════════════════════════════════════════════════════════════════
section "TEST 12: Disk Space"
DISK=$(df -h . | tail -1 | awk '{print "Used: "$3" / "$2" ("$5" full)"}')
info "Disk: ${DISK}"
DISK_PCT=$(df . | tail -1 | awk '{print $5}' | tr -d '%')
if [[ "$DISK_PCT" -lt 80 ]]; then
    pass "Disk usage OK (${DISK_PCT}% used)"
elif [[ "$DISK_PCT" -lt 90 ]]; then
    warn "Disk usage HIGH (${DISK_PCT}%) — consider cleanup"
else
    fail "Disk usage CRITICAL (${DISK_PCT}%) — free space immediately"
fi


# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  VERIFICATION COMPLETE"
echo "═══════════════════════════════════════════════════════════════════"
echo -e "  ${GREEN}PASSED${NC}: ${PASS}"
echo -e "  ${YELLOW}WARNINGS${NC}: ${WARN}"
echo -e "  ${RED}FAILED${NC}: ${FAIL}"
echo ""

if [[ $FAIL -eq 0 ]]; then
    echo -e "  ${GREEN}✅ ALL CRITICAL TESTS PASSED — Server is working!${NC}"
else
    echo -e "  ${RED}❌ ${FAIL} CRITICAL FAILURE(S) — Fix the above issues${NC}"
fi

echo ""
echo "  Useful debug commands:"
echo "    docker compose logs -f backend      → live backend logs"
echo "    docker compose logs -f postgres     → PostgreSQL logs"
echo "    docker compose ps                   → container status"
echo "    docker exec mom_postgres psql -U postgres -d govindb -c '\\dt'  → list DB tables"
echo "═══════════════════════════════════════════════════════════════════"

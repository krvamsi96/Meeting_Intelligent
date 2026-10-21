@echo off
REM ===============================================================================
REM   AI MEETING INTELLIGENCE PLATFORM - WINDOWS FRONTEND RUNNER
REM ===============================================================================

echo ===============================================================================
echo    AI MEETING INTELLIGENCE PLATFORM - ANGULAR FRONTEND
echo ===============================================================================
echo.

cd frontend
if not exist "node_modules" (
    echo [INFO] Installing frontend dependencies...
    npm install
)

REM The app needs the backend on port 8000 (requests to /api are forwarded there).
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 http://127.0.0.1:8000/api/v1/system/status | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    echo [WARNING] The backend is not running on http://127.0.0.1:8000.
    echo           Start it with run_backend.bat ^(or use run_all.bat^), otherwise the app shows
    echo           "The service cannot be reached right now."
    echo.
)

echo [INFO] Starting Angular development server at http://127.0.0.1:4200
echo.
npm start -- --host 127.0.0.1 --port 4200

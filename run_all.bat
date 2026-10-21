@echo off
REM ===============================================================================
REM   AI MEETING INTELLIGENCE PLATFORM - COMPLETE RUNNER (BACKEND + FRONTEND)
REM ===============================================================================

echo ===============================================================================
echo    STARTING AI MEETING INTELLIGENCE PLATFORM (FULL STACK)
echo ===============================================================================
echo.

echo [1/2] Launching Backend API server (port 8000)...
start "AI Meeting Backend" run_backend.bat

echo [2/2] Launching Angular Frontend (port 4200)...
start "AI Meeting Frontend" run_frontend.bat

echo.
echo Both servers are launching in separate windows!
echo - Frontend: http://127.0.0.1:4200
echo - Backend API: http://127.0.0.1:8000
echo - Swagger Docs: http://127.0.0.1:8000/docs
echo.
pause

@echo off
REM ===============================================================================
REM   AI MEETING INTELLIGENCE PLATFORM - WINDOWS BACKEND RUNNER
REM ===============================================================================

echo ===============================================================================
echo    AI MEETING INTELLIGENCE PLATFORM - BACKEND (WINDOWS)
echo ===============================================================================
echo.

REM 1. Check Virtual Environment
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo         Please run 'setup_windows.bat' first to set up the environment.
    pause
    exit /b 1
)

REM 2. Check .env
if not exist ".env" (
    echo [ERROR] .env file not found.
    echo         Please create a .env file or run 'setup_windows.bat'.
    pause
    exit /b 1
)

REM 3. Ensure Storage Directories
if not exist "uploads" mkdir uploads
if not exist "uploads\clips" mkdir uploads\clips
if not exist "data\chroma" mkdir data\chroma
if not exist "nemo_cache" mkdir nemo_cache
if not exist "data" mkdir data

REM 4. Activate Virtual Environment & Start Server
call venv\Scripts\activate.bat

REM 5. Embedding model: git lfs pull if this checkout holds the Git LFS pointer, then size + SHA-256 check.
python -m backend.scripts.ensure_embedding_model
if errorlevel 1 (
    echo [ERROR] The real embedding model could not be obtained - see the FIX above.
    pause
    exit /b 1
)

echo [INFO]  Starting backend server on http://127.0.0.1:8000
echo [INFO]  Swagger Documentation: http://127.0.0.1:8000/docs
echo [INFO]  Health Endpoint: http://127.0.0.1:8000/api/v1/system/health
echo.
echo Press Ctrl+C to stop the server.
echo ===============================================================================
echo.

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

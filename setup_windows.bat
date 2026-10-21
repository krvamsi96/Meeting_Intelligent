@echo off
REM ===============================================================================
REM   AI MEETING INTELLIGENCE PLATFORM - WINDOWS SETUP SCRIPT
REM ===============================================================================

echo ===============================================================================
echo    AI MEETING INTELLIGENCE PLATFORM - WINDOWS SETUP
echo ===============================================================================
echo.

REM 1. Check Python 3.10+
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in PATH.
    echo         Please install Python 3.11 from https://www.python.org/downloads/
    echo         Ensure "Add Python to PATH" is checked during installation.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
echo [OK]    Python detected: %PY_VER%

REM 2. Check FFmpeg
ffmpeg -version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARN]  FFmpeg is not detected in PATH.
    echo         Audio/video transcription and diarization require FFmpeg.
    echo         Install via winget: winget install Gyan.FFmpeg
    echo         Or download from https://ffmpeg.org/download.html and add bin to PATH.
    echo.
) else (
    echo [OK]    FFmpeg detected in PATH
)

REM 3. Check .env Configuration
if not exist ".env" (
    if exist ".env.example" (
        echo [INFO]  .env not found. Copying from .env.example...
        copy .env.example .env >nul
        echo [OK]    Created .env from template.
        echo [WARN]  Please edit .env with your PostgreSQL credentials and AI provider settings.
    ) else (
        echo [ERROR] Neither .env nor .env.example found.
        pause
        exit /b 1
    )
) else (
    echo [OK]    .env configuration file found
)

REM 4. Create / Verify Python Virtual Environment
if not exist "venv\Scripts\activate.bat" (
    echo [INFO]  Creating Python virtual environment in .\venv...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK]    Virtual environment created
) else (
    echo [OK]    Virtual environment already exists in .\venv
)

REM 5. Install Dependencies
echo.
echo [INFO]  Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat

python -m pip install --upgrade pip --quiet
echo [INFO]  Installing packages from requirements.txt (this may take a few minutes)...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dependency installation encountered errors.
    pause
    exit /b 1
)
echo [OK]    Dependencies installed successfully

REM 6. Create Storage Directories
if not exist "uploads" mkdir uploads
if not exist "uploads\clips" mkdir uploads\clips
if not exist "data\chroma" mkdir data\chroma
if not exist "nemo_cache" mkdir nemo_cache
if not exist "data" mkdir data
echo [OK]    Storage directories initialized

REM 7. Verify Database Initialization & Migrations
echo.
echo [INFO]  Verifying PostgreSQL database connection and running auto-migrations...
python -c "from backend.database import init_db; init_db()"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARN]  Database initialization failed.
    echo         Please ensure PostgreSQL is running and credentials in .env are correct.
    echo         (DB_HOST, DB_PORT, DB_DATABASE, DB_USER, DB_PASSWORD, DATABASE_URL)
) else (
    echo [OK]    PostgreSQL database connection verified and schema migrated
)

REM 8. Verify Critical Imports
echo.
echo [INFO]  Verifying platform modules...
python -c "from backend.config.config import settings; from backend.main import app; print(f'  [OK]  {settings.APP_NAME} loaded with {len(app.routes)} API routes')"

echo.
echo ===============================================================================
echo                 WINDOWS SETUP COMPLETE
echo ===============================================================================
echo.
echo  To start the backend server, run:
echo      run_backend.bat
echo.
echo  To run tests, run:
echo      test_windows.bat
echo.
echo  Swagger UI: http://127.0.0.1:8000/docs
echo  Health Check: http://127.0.0.1:8000/api/v1/system/health
echo ===============================================================================
pause

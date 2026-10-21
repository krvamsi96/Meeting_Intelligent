@echo off
REM ===============================================================================
REM   AI MEETING INTELLIGENCE PLATFORM - WINDOWS TEST RUNNER
REM ===============================================================================

echo ===============================================================================
echo    RUNNING AI MEETING INTELLIGENCE TEST SUITE
echo ===============================================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please run 'setup_windows.bat' first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

python -m pytest backend/tests -v
pause

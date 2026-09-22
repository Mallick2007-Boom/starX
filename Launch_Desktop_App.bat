@echo off
title Boreas Polar Station — AI Microgrid Launcher
echo ====================================================================
echo   BOREAS POLAR STATION AI MICROGRID — 1-CLICK LAUNCHER
echo ====================================================================
echo.
cd /d "%~dp0"

echo [1/2] Checking BOREAS API Backend (http://localhost:8000)...
powershell -Command "if (!(Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue)) { exit 1 } else { exit 0 }"
if %errorlevel% neq 0 (
    echo Starting backend API server in background...
    start "" ".venv\Scripts\pythonw.exe" api\app.py
    timeout /t 3 >nul
) else (
    echo Backend API server is already active!
)

echo.
echo [2/2] Launching Desktop GUI Application...
start "" ".venv\Scripts\pythonw.exe" desktop_app.py

echo.
echo ====================================================================
echo Mission Control Ready:
echo   * Desktop HUD:           Running
echo   * Web Mission Control:   http://localhost:8000/
echo   * Interactive Swagger:   http://localhost:8000/docs
echo   * ReDoc Documentation:   http://localhost:8000/redoc
echo   * OpenAPI Specification: http://localhost:8000/openapi.json
echo ====================================================================
timeout /t 3 >nul
exit

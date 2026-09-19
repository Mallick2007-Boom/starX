@echo off
title Boreas Polar Station — AI Microgrid Launcher
echo ====================================================================
echo   BOREAS POLAR STATION AI MICROGRID — 1-CLICK LAUNCHER
echo ====================================================================
echo.
echo Starting Desktop GUI Application Prototype...
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" desktop_app.py
echo.
echo Desktop Application is running!
echo You can also access the Web Dashboard at: http://localhost:8000/
echo ====================================================================
timeout /t 3 >nul
exit

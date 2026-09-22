@echo off
title Boreas Polar Station — API & Mission Control Server
echo ====================================================================
echo   BOREAS POLAR RESEARCH STATION — FASTAPI SERVER & MISSION CONTROL
echo ====================================================================
echo.
cd /d "%~dp0"
echo Starting FastAPI backend microservice on http://localhost:8000 ...
echo.
echo   * Web Mission Control:  http://localhost:8000/
echo   * Interactive Swagger:  http://localhost:8000/docs
echo   * ReDoc Documentation:  http://localhost:8000/redoc
echo   * OpenAPI Specification: http://localhost:8000/openapi.json
echo.
echo Press CTRL+C to stop the server.
echo ====================================================================
echo.
".venv\Scripts\python.exe" api\app.py
pause

@echo off
title LungGuard AI Web Server
cd /d "%~dp0"
echo =====================================================================
echo                      LungGuard AI Web Server
echo =====================================================================
echo.
echo Starting FastAPI + Uvicorn server on http://localhost:8000 ...
echo.
echo Press CTRL+C to stop the server at any time.
echo.
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
pause

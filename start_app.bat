@echo off
echo Starting Kawkab AI System...

:: Start Backend
start "Kawkab Backend" cmd /k "pip install -r backend/requirements.txt && uvicorn backend.main:app --reload --port 8000"

:: Start Frontend
start "Kawkab Frontend" cmd /k "cd frontend && npm install && npm run dev"

echo System started!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Don't forget to run Cloudflare Tunnel:
echo cloudflared tunnel --url http://localhost:8000

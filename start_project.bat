@echo off
echo Starting RSNA Aneurysm Detection Project...
start cmd /k "python backend/api.py"
echo AI Backend starting on port 5000...
timeout /t 3
start cmd /k "python -m http.server 8000"
echo Web Dashboard starting on http://localhost:8000/frontend/dashboard.html
timeout /t 2
start http://localhost:8000/frontend/dashboard.html
echo Project is running!

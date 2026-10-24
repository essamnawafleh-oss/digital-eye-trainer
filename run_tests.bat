@echo off
cd /d "%~dp0"
chcp 65001 >nul
.venv\Scripts\python.exe -m pip install -q pytest==9.1.1
.venv\Scripts\python.exe -m pytest %*
pause

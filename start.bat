@echo off
REM Opens the local page (http://127.0.0.1:8765). Works without internet.
cd /d "%~dp0"
chcp 65001 >nul
if not exist .venv\Scripts\python.exe (
  echo Run install.bat first / شغّل install.bat أولاً
  pause & exit /b 1
)
.venv\Scripts\python.exe -m eye_trainer ui %*
pause

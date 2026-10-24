@echo off
REM One-time installation (needs internet ONCE). Windows 10/11, 64-bit.
REM   install.bat        CPU (works everywhere)
REM   install.bat gpu    NVIDIA GPU (needs a matching CUDA driver; see README)
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PY="
py -3.13 --version >nul 2>nul
if %errorlevel%==0 (set "PY=py -3.13" & goto :havepy)
py -3.12 --version >nul 2>nul
if %errorlevel%==0 (set "PY=py -3.12" & goto :havepy)
python --version >nul 2>nul
if %errorlevel%==0 (set "PY=python" & goto :havepy)
echo Python 3.10-3.13 is required: https://www.python.org/downloads/windows/
echo يلزم تثبيت بايثون 3.13 من الموقع الرسمي، مع اختيار "Add python.exe to PATH"
pause
exit /b 1
:havepy
%PY% --version
if not exist .venv %PY% -m venv .venv || goto :fail
call .venv\Scripts\activate.bat || goto :fail
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt || goto :fail
if /i "%1"=="gpu" (
  echo Installing the GPU build of PaddlePaddle ^(CUDA 12.6^). UNVERIFIED by the authors - see README.
  python -m pip uninstall -y paddlepaddle
  python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ || goto :fail
)
echo Downloading the base models (Apache-2.0, ~21 MB)
python -m eye_trainer download-models || goto :fail
python -m eye_trainer check
echo.
echo Done. Double-click start.bat  /  انتهى التثبيت. انقر مرتين على start.bat
pause
exit /b 0
:fail
echo.
echo Installation FAILED. Read the messages above. / فشل التثبيت، راجع الرسائل أعلاه.
pause
exit /b 1

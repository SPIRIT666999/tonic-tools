@echo off
set "PY=python"
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
%PY% --version
if errorlevel 1 (
  echo Python not found. Install it from python.org and enable "Add python.exe to PATH".
  pause
  exit /b 1
)
%PY% -m pip install --upgrade pip
%PY% -m pip install -r "%~dp0requirements.txt"
echo.
echo DONE. Now run ZAPUSK.bat
pause

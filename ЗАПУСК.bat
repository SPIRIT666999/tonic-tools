@echo off
cd /d "%~dp0"
set "PYEXE="
for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
if not defined PYEXE (
  for /f "delims=" %%i in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
)
if defined PYEXE (
  for %%i in ("%PYEXE%") do set "PDIR=%%~dpi"
  if exist "%PDIR%pythonw.exe" (
    start "" "%PDIR%pythonw.exe" "%~dp0tonic_analyzer.py"
    exit /b 0
  )
)
if defined PYEXE (
  "%PYEXE%" "%~dp0tonic_analyzer.py"
) else (
  python "%~dp0tonic_analyzer.py"
)
if errorlevel 1 pause

@echo off
set "PY=python"
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
%PY% "%~dp0tonic_analyzer.py"
if errorlevel 1 pause

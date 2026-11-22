@echo off
chcp 65001 >nul
python "%~dp0tonic_analyzer.py"
if errorlevel 1 pause

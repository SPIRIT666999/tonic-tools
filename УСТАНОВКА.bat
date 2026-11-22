@echo off
chcp 65001 >nul
set "PY=python"
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
%PY% --version || (echo Python не найден. Установи с python.org и поставь галочку Add python.exe to PATH & pause & exit /b 1)
echo Установка зависимостей (нужен интернет)...
%PY% -m pip install --upgrade pip
%PY% -m pip install -r "%~dp0requirements.txt"
echo.
echo Установка завершена. Теперь запусти ЗАПУСК.bat
pause

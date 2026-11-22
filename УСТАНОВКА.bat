@echo off
chcp 65001 >nul
echo Установка зависимостей (нужен интернет)...
python -m pip install --upgrade pip
python -m pip install -r "%~dp0requirements.txt"
echo.
echo Установка завершена. Теперь запусти ЗАПУСК.bat
pause

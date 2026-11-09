@echo off
chcp 65001 >nul
cd /d "%~dp0..\1차"
".venv\Scripts\python.exe" -B -X utf8 server.py --port 8879 --continuous
if errorlevel 1 pause

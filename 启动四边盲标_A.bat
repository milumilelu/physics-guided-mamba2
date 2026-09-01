@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "scripts\15_manual_four_edge_annotator.py" --annotator A
if errorlevel 1 pause

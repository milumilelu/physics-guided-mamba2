@echo off
cd /d "%~dp0"
if not exist "annotations\single_line_range_annotation.csv" (
    echo annotation table missing; run scripts\33_build_single_line_annotation_table.py first.
    ".venv\Scripts\python.exe" "scripts\33_build_single_line_annotation_table.py" --annotator A
    if errorlevel 1 pause
)
".venv\Scripts\python.exe" "scripts\34_manual_single_line_annotator.py" --annotator A
if errorlevel 1 pause

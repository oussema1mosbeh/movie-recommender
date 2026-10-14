@echo off
rem Runs a project script with the project's Python, from any folder.
rem Usage:  run 02_content_based.py "toy story"   (or double-click to run 01_explore.py)
cd /d "%~dp0"
if "%~1"=="" (
    ".venv\Scripts\python.exe" 01_explore.py
    pause
) else (
    ".venv\Scripts\python.exe" %*
)

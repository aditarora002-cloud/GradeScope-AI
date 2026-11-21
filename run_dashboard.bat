@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Setup not found. Please run setup_windows.bat first.
    pause
    exit /b 1
)
echo Starting GradeScope AI...
".venv\Scripts\python.exe" -m streamlit run app.py
pause

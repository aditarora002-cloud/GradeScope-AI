@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)
echo GradeScope AI: Training Portuguese subject with DNN hyperparameter tuning
".venv\Scripts\python.exe" train.py --subject por --tune
if errorlevel 1 (
    echo Model training failed - see error log above.
    pause
    exit /b 1
)
echo Training complete. Open run_dashboard.bat
pause

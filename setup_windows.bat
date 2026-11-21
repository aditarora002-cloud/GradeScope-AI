@echo off
setlocal
cd /d "%~dp0"
echo.
echo  =============================================
echo      GradeScope AI - One-time Windows Setup
 echo  =============================================
echo.
if not exist "pip-temp" mkdir "pip-temp"
set "TEMP=%CD%\pip-temp"
set "TMP=%CD%\pip-temp"
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python environment on the current drive...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Python 3.12 not detected. Trying Python 3.11...
        py -3.11 -m venv .venv
        if errorlevel 1 (
            echo Could not find Python 3.11 or 3.12. Install Python and retry.
            pause
            exit /b 1
        )
    )
)
echo Installing dependencies on current drive (no pip cache)...
".venv\Scripts\python.exe" -m pip install --no-cache-dir -r requirements.txt
if errorlevel 1 (
    echo.
    echo Install failed. Check free disk space and the error above.
    pause
    exit /b 1
)
echo.
echo Setup completed. Double-click run_dashboard.bat to launch.
pause

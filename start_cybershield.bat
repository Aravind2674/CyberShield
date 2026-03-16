@echo off
setlocal EnableExtensions
title CyberShield Startup

REM Move to repository root (location of this .bat)
cd /d "%~dp0"
if errorlevel 1 goto :path_failed

REM Move into the active application directory
cd /d "integrated-video-analytics"
if errorlevel 1 goto :app_dir_missing

REM Prefer workspace-level venv, then local app venv, then system Python
set "PYTHON_EXE=python"
if exist "..\.venv\Scripts\python.exe" set "PYTHON_EXE=..\.venv\Scripts\python.exe"
if not exist "..\.venv\Scripts\python.exe" if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

echo Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 goto :python_missing

echo Starting CyberShield...
echo App URL: http://localhost:8080

REM Ensure required dependencies exist in the selected interpreter
set "MISSING_DEPS=0"
call :check_pkg fastapi
call :check_pkg uvicorn
call :check_pkg opencv-python
call :check_pkg torch
call :check_pkg ultralytics
call :check_pkg easyocr
call :check_pkg fpdf2
call :check_pkg supervision

if not "%MISSING_DEPS%"=="0" call :install_core_deps
if errorlevel 1 goto :dep_failed

set "MISSING_DEPS=0"
call :check_pkg fastapi
call :check_pkg uvicorn
call :check_pkg opencv-python
call :check_pkg torch
call :check_pkg ultralytics
call :check_pkg easyocr
call :check_pkg fpdf2
call :check_pkg supervision
if not "%MISSING_DEPS%"=="0" goto :dep_still_missing

REM Open dashboard in default browser
start "" "http://localhost:8080"

REM Start FastAPI app
"%PYTHON_EXE%" main.py

if errorlevel 1 goto :app_failed

pause
exit /b 0

:python_missing
echo.
echo Python interpreter not found or not runnable: %PYTHON_EXE%
echo Make sure Python is installed, or create .venv first.
pause
exit /b 1

:path_failed
echo.
echo Unable to access the script directory.
pause
exit /b 1

:app_dir_missing
echo.
echo Could not find integrated-video-analytics folder.
echo Verify this file is in the project root.
pause
exit /b 1

:install_core_deps
echo.
echo Required packages not found for this Python environment.
echo Installing core dependencies from requirements.txt...
"%PYTHON_EXE%" -m pip --version >nul 2>&1
if errorlevel 1 "%PYTHON_EXE%" -m ensurepip --upgrade
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
if exist "requirements-face.txt" "%PYTHON_EXE%" -m pip install -r requirements-face.txt >nul 2>&1
exit /b 0

:check_pkg
"%PYTHON_EXE%" -m pip show %~1 >nul 2>&1
if errorlevel 1 set /a MISSING_DEPS+=1
exit /b 0

:dep_failed
echo.
echo Dependency installation failed.
echo Try manually:
echo   "%PYTHON_EXE%" -m pip install -r requirements.txt
pause
exit /b 1

:dep_still_missing
echo.
echo Core dependencies are still missing after installation.
echo Run this command and share output:
echo   "%PYTHON_EXE%" -m pip install -r requirements.txt
pause
exit /b 1

:app_failed
echo.
echo Startup failed. Verify dependencies with:
echo   "%PYTHON_EXE%" -m pip install -r requirements.txt
pause
exit /b 1
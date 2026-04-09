@echo off
setlocal EnableDelayedExpansion
title Voice Model Studio — Windows Installer

echo.
echo  ================================================================
echo   VOICE MODEL STUDIO — One-Click Windows Installer
echo  ================================================================
echo.

:: ── Check for Python ─────────────────────────────────────────────
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo  [WARN] Python not found in PATH.
    echo  [INFO] Opening Python download page …
    start https://www.python.org/downloads/
    echo  Please install Python 3.10 or later, then re-run this script.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  [INFO] Found Python %PY_VER%

:: ── Check Python version >= 3.10 ────────────────────────────────
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if %PY_MAJOR% LSS 3 (
    echo  [ERR ] Python 3.10+ required. Exiting.
    pause
    exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 (
    echo  [ERR ] Python 3.10+ required. You have %PY_VER%. Exiting.
    pause
    exit /b 1
)

:: ── Check / install ffmpeg ───────────────────────────────────────
where ffmpeg >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo  [WARN] ffmpeg not found in PATH.
    echo  [INFO] Download ffmpeg from https://www.gyan.dev/ffmpeg/builds/
    echo         and add the bin/ folder to your PATH for full format support.
)

:: ── Run installer ────────────────────────────────────────────────
echo  [INFO] Running Python installer …
python install.py %*
if %ERRORLEVEL% NEQ 0 (
    echo  [ERR ] Installation failed. See output above.
    pause
    exit /b 1
)

echo.
echo  ================================================================
echo   Installation complete!  Run start.bat to launch the studio.
echo  ================================================================
echo.
pause

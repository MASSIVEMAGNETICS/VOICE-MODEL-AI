@echo off
setlocal
title Voice Model Studio

if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

echo  Starting Voice Model Studio …
"%PYTHON%" app.py %*
if %ERRORLEVEL% NEQ 0 pause

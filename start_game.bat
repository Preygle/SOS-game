@echo off
REM ===========================================================
REM  SOS - Ultimate Edition : complete launcher
REM
REM  Double-click this file to play. It picks an interpreter,
REM  checks the game's dependencies, points the LAYA bot slot at
REM  its own environment, and keeps the window open if anything
REM  goes wrong so you can read the error.
REM
REM  Note on style: this script uses "if errorlevel 1 goto ..."
REM  rather than comparing %ERRORLEVEL% inside ( ) blocks. Batch
REM  expands %VAR% when it PARSES a block, not when it runs, so a
REM  check written the other way reads a stale value and reports
REM  the wrong result.
REM ===========================================================
setlocal

REM Always run from this script's own folder. The game loads art with
REM relative paths (pyglet.resource.path = ['assets']), so launching from
REM anywhere else fails with a missing-resource error.
cd /d "%~dp0"

title SOS - Ultimate Edition

REM --- 1. Pick an interpreter ---------------------------------
REM Prefer the bundled venv (Python 3.11, pyglet 2.1.12); otherwise use
REM whatever "python" resolves to on PATH.
if exist "venv\Scripts\python.exe" goto :use_venv
set "PY=python"
echo [1/3] No venv found, using system Python
goto :check_python

:use_venv
set "PY=venv\Scripts\python.exe"
echo [1/3] Using bundled environment: venv

:check_python
"%PY%" --version >nul 2>&1
if errorlevel 1 goto :no_python

REM --- 2. Check the dependencies the game actually needs -------
REM Only pyglet and numpy are needed to play. torch and matplotlib in
REM requirements.txt are for training and plotting, not for the game.
"%PY%" -c "import pyglet, numpy" >nul 2>&1
if errorlevel 1 goto :install_deps
echo [2/3] Dependencies OK
goto :laya

:install_deps
echo [2/3] Missing dependencies, installing pyglet and numpy...
"%PY%" -m pip install --quiet --upgrade pip
"%PY%" -m pip install --quiet pyglet numpy
if errorlevel 1 goto :no_deps
echo       installed.

:laya
REM --- 3. Optional: the LAYA bot slot --------------------------
REM LAYA runs in its own environment because it needs transformers 5.x while
REM this project pins 4.50. If that environment is present, tell the game
REM where it is; if not, the other two bot slots still work normally.
if not exist "D:\laya\env\Scripts\python.exe" goto :no_laya
set "LAYA_PYTHON=D:\laya\env\Scripts\python.exe"
set "HF_HOME=D:\laya\hf_cache"
echo [3/3] LAYA bot available
goto :run

:no_laya
echo [3/3] LAYA bot not installed - Fast and Deep bots still work

:run
echo.
echo Starting SOS...
echo.
"%PY%" sos.py
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto :crashed
endlocal & exit /b 0

:crashed
echo.
echo ===========================================================
echo  The game exited with an error ^(code %RC%^).
echo  The message above this line says why.
echo ===========================================================
echo.
pause
endlocal & exit /b 1

:no_python
echo.
echo ERROR: no working Python found.
echo Install Python 3.11 or newer from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during setup.
echo.
pause
endlocal & exit /b 1

:no_deps
echo.
echo ERROR: could not install pyglet and numpy.
echo Try it by hand:
echo     %PY% -m pip install -r requirements.txt
echo.
pause
endlocal & exit /b 1

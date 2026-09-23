@echo off
REM ============================================================
REM  Full distillation training run (updated algo).
REM  Outputs go to a NEW timestamped folder every time, so all
REM  previous training data (checkpoints_distill_*, checkpoints,
REM  checkpoints_v2, best.pth, etc.) is left completely intact.
REM ============================================================

REM Reliable timestamp independent of regional settings
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set timestamp=%datetime:~0,4%%datetime:~4,2%%datetime:~6,2%_%datetime:~8,2%%datetime:~10,2%%datetime:~12,2%
set outdir=checkpoints_distill_%timestamp%

echo ========================================================
echo  Starting FULL training run...
echo  Output folder : %outdir%   (new - nothing old is overwritten)
echo ========================================================
echo.

REM Pick the venv Python if present, else system Python
if exist venv\Scripts\python.exe (
    set PY=venv\Scripts\python.exe
) else (
    set PY=python
)

REM --- Full-strength settings (raise --games / --epochs for more) ---
%PY% distill_train.py ^
    --out %outdir% ^
    --games 800 ^
    --teacher-budget 0.8 ^
    --epsilon 0.12 ^
    --blocks 4 ^
    --channels 64 ^
    --epochs 40 ^
    --batch 128 ^
    --lr 1e-3

echo.
echo Training complete. Models and distill_log.csv are in: %outdir%
echo neural_bot.py now auto-loads the newest checkpoints_distill_* folder.
echo.
echo Measure how strong it actually plays (net vs random/greedy/SmartBot):
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe eval_bots.py --games 30
) else (
    python eval_bots.py --games 30
)
pause

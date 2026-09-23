@echo off
REM Get reliable timestamp independent of regional settings
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set timestamp=%datetime:~0,4%%datetime:~4,2%%datetime:~6,2%_%datetime:~8,2%%datetime:~10,2%%datetime:~12,2%
set outdir=checkpoints_distill_%timestamp%

echo ========================================================
echo Starting training run...
echo Outputs will be saved to the new directory: %outdir%
echo This ensures your old training data remains intact.
echo ========================================================
echo.

REM Use the virtual environment Python if it exists, otherwise use system Python
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe distill_train.py --out %outdir%
) else (
    python distill_train.py --out %outdir%
)

echo.
echo Training complete! Check the %outdir% folder for your models and logs.
pause

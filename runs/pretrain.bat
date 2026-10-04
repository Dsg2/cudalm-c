@echo off
rem pretrain <model> <corpus> [epochs=1] : s8w2 recipe from scratch (or resume if models\<model>.bin exists).
rem corpus = a folder from `cudalm data encode` / `data mix`, or a bare name under the data roots.
rem Stop cleanly at any time with Ctrl+C in the TUI, or by creating models\<model>.bin.stop.
setlocal
cd /d %~dp0..
if "%~2"=="" (echo usage: pretrain ^<model^> ^<corpus^> [epochs=1] & exit /b 1)
set EP=%3
if "%EP%"=="" set EP=1
call runs\s8w2env.bat %2
set "CUDALM_SAMPLER=perm"
set "CUDALM_SEED=1234"
set "CUDALM_SAVE_EVERY=2000"
set "CUDALM_LR=7e-4"
set "CUDALM_LRS=1"
set "CUDALM_KEEP_EVERY=40000"
if not exist models\%1.bin cudalm.exe new %1 s8w 32 || exit /b 1
set TRY=0
:again
cudalm.exe train %1 %2 %EP%
set RC=%errorlevel%
rem exit 3 = 20 non-finite steps in a row: resume from the last checkpoint (at most 3 times)
if "%RC%"=="3" if %TRY% LSS 3 (set /a TRY+=1 & goto again)
exit /b %RC%

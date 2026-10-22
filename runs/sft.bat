@echo off
rem sft <base_model> <sft_model> <chat_corpus> <base_iters> <sft_iters>
rem Chat SFT that continues a pretrained model: copies it to <sft_model>, re-warms to LR 1e-4 and runs a WSD
rem schedule for <sft_iters> more steps. <base_iters> = the base run's final iteration (last "it" in its log).
rem Only assistant tokens are trained (chatml loss mask). Compare against the base with bench\chatbench.bat.
setlocal
cd /d %~dp0..
if "%~5"=="" (echo usage: sft ^<base_model^> ^<sft_model^> ^<chat_corpus^> ^<base_iters^> ^<sft_iters^> & exit /b 1)
rem .opt.x1 holds the canon kernels + shared expert (CUDALM_CANON / CUDALM_SHARED_F); .opt/.opt.mu = AdamW moments / Muon momentum of a CUDALM_QLM=0 base
if not exist models\%2.bin for %%x in (.bin .bin.env .bin.opt .bin.opt.mu .bin.opt.q .bin.opt.ng .bin.opt.t1 .bin.opt.x1) do if exist models\%1%%x copy /y models\%1%%x models\%2%%x >nul
set /a ITERS=%4+%5
rem cooldown = last 40%% of the SFT steps (the decay fraction is relative to the whole schedule)
set /a DEC=4000*%5/ITERS
set DEC=0000%DEC%
set "CUDALM_LR=1e-4"
set "CUDALM_SCHED=wsd"
set "CUDALM_REWARMUP=1000"
set "CUDALM_WSD_DECAY=0.%DEC:~-4%"
set "CUDALM_ITERS=%ITERS%"
set "CUDALM_PERM_BASE=%4"
set "CUDALM_SEED=9012"
set "CUDALM_MOE_AUX=0.05"
set "CUDALM_SAVE_EVERY=2000"
set "CUDALM_KEEP_EVERY=0"
rem the copied .env carries the base run's settings (pretrain.bat turns the LR scaler on); SFT does not want its probes
set "CUDALM_LRS=0"
set TRY=0
:again
cudalm.exe train %2 %3 1
set RC=%errorlevel%
if "%RC%"=="3" if %TRY% LSS 3 (set /a TRY+=1 & goto again)
cudalm.exe eval %2 %3 64
exit /b %RC%

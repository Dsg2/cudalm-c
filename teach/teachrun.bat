@echo off
rem teachrun <out_dir> [ndocs=400000] [seed=2] : teacher data from 3 llama-server instances (32 slots each).
rem Writes <out_dir>\teach_s<seed>.jsonl (+ teach.log). Resumable: re-running skips finished docs.
rem   LLAMA_SERVER  path to llama-server.exe (default: on PATH)
rem   TEACHER_GGUF  teacher model, any instruct GGUF (we used a 2B Q4_0 model: ~1450 tok/s on a 16 GB card)
rem Then: cudalm data encode new <corpus> <out_dir>\teach_s<seed>.jsonl <tokenizer>
setlocal
set "D=%~f1"
cd /d %~dp0
if "%~1"=="" (echo usage: teachrun ^<out_dir^> [ndocs] [seed] & exit /b 1)
if not defined TEACHER_GGUF (echo set TEACHER_GGUF to the teacher model .gguf & exit /b 1)
if not defined LLAMA_SERVER set LLAMA_SERVER=llama-server.exe
set N=%2
if "%N%"=="" set N=400000
set SEED=%3
if "%SEED%"=="" set SEED=2
set OUT=%D%\teach_s%SEED%.jsonl
set LOG=%D%\teach.log
if not exist "%D%" mkdir "%D%"
if not exist "%OUT%.topics" copy /y topics.jsonl.topics "%OUT%.topics" > nul
for /l %%r in (1,1,6) do (
  call :srv 8099 & call :srv 8100 & call :srv 8101
  call :wait 8099 & call :wait 8100 & call :wait 8101
  echo [teachrun] attempt %%r >> "%LOG%"
  ..\teachgen.exe "%OUT%" %N% 96 8099,8100,8101 0.5 %SEED% >> "%LOG%" 2>&1
  call :stop
  findstr /c:"kept" "%LOG%" > nul && call :complete && goto done
)
:done
echo done >> "%LOG%"
exit /b 0

:complete
powershell -NoProfile -Command "$n=(Get-Content '%OUT%' | Measure-Object -Line).Lines; $d=(Select-String -Path '%LOG%' -Pattern 'drop (\d+)' | Select -Last 1).Matches.Groups[1].Value; if($n + [int]$d -ge %N% * 0.995){exit 0}else{exit 1}"
exit /b %errorlevel%

:srv
start "" /b cmd /c ""%LLAMA_SERVER%" -m "%TEACHER_GGUF%" -ngl 99 -fa on --host 127.0.0.1 -c 49152 -np 32 --port %1 > "%D%\srv_%1.log" 2>&1"
exit /b 0

:wait
powershell -NoProfile -Command "for($i=0;$i -lt 180;$i++){ try{ if((Invoke-WebRequest http://127.0.0.1:%1/health -UseBasicParsing).StatusCode -eq 200){exit 0} }catch{}; Start-Sleep 1 }; exit 1"
exit /b 0

:stop
powershell -NoProfile -Command "Get-CimInstance Win32_Process | ? { $_.Name -eq 'llama-server.exe' -and $_.CommandLine -match '--port (8099|8100|8101)' } | ForEach-Object { Stop-Process -Id $_.ProcessId }"
powershell -NoProfile -Command "Start-Sleep 3"
exit /b 0

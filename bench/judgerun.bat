@echo off
rem judgerun <chatbench_out.txt> <tag> [student|shuffle|teacher]
rem Coherence judge: starts a llama-server with a judge model on :8099, scores the chatbench replies with judge.exe
rem (teacher NLL per byte + an expected 0-9 coherence score), then stops that server.
rem   LLAMA_SERVER  path to llama-server.exe (default: on PATH)
rem   JUDGE_GGUF    judge model, any instruct GGUF (we used a 2B Q4_0 model)
rem Output: bench\judge\jd_<tag>.tsv (per reply) and jd_<tag>.txt (summary). ~4 min per 96 replies on a mid-range GPU.
setlocal
set "IN=%~f1"
cd /d %~dp0
if "%~2"=="" (echo usage: judgerun ^<chatbench_out.txt^> ^<tag^> [student^|shuffle^|teacher] & exit /b 1)
if not defined JUDGE_GGUF (echo set JUDGE_GGUF to the judge model .gguf & exit /b 1)
if not defined LLAMA_SERVER set LLAMA_SERVER=llama-server.exe
if not exist judge mkdir judge
set MODE=%3
if "%MODE%"=="" set MODE=student
start "" /b cmd /c ""%LLAMA_SERVER%" -m "%JUDGE_GGUF%" -ngl 99 -c 8192 -np 4 --port 8099 --host 127.0.0.1 > judge\srv.log 2>&1"
powershell -NoProfile -Command "for($i=0;$i -lt 120;$i++){ try{ if((Invoke-WebRequest http://127.0.0.1:8099/health -UseBasicParsing).StatusCode -eq 200){exit 0} }catch{}; Start-Sleep 1 }; exit 1"
if errorlevel 1 (echo judge server failed to start - see bench\judge\srv.log & goto stop)
..\judge.exe "%IN%" judge\jd_%2.tsv %MODE% > judge\jd_%2.txt 2>&1
:stop
powershell -NoProfile -Command "Get-CimInstance Win32_Process | ? { $_.Name -eq 'llama-server.exe' -and $_.CommandLine -match '--port 8099' } | ForEach-Object { Stop-Process -Id $_.ProcessId }"
type judge\jd_%2.txt | findstr /v /r "^  *[0-9]*/[0-9]*$"
exit /b 0

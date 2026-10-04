@echo off
rem chatbench <model> : multiple-choice + open-generation chat benchmark (bench\chatbench.txt), sampled and greedy.
rem Output: bench\cb_<model>_samp.txt and bench\cb_<model>_greedy.txt (summary at the end of each).
rem Args of `cudalm chatbench`: <model> <bench.txt> [ngen] [temp] [topp] [reppen] [nsamp]
setlocal
cd /d %~dp0..
if "%~1"=="" (echo usage: chatbench ^<model^> & exit /b 1)
cudalm.exe chatbench %1 bench\chatbench.txt 128 0.7 0.9 1.15 4 > bench\cb_%1_samp.txt 2>&1
cudalm.exe chatbench %1 bench\chatbench.txt 128 0 0.9 1.15 1 > bench\cb_%1_greedy.txt 2>&1
echo results: bench\cb_%1_samp.txt  bench\cb_%1_greedy.txt

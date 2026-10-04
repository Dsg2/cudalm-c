@echo off
rem s8w2env [corpus_dir] : environment of the s8w2 recipe (call it before `cudalm new`).
rem 32-expert top-2 MoE (preset s8w), QLM 4-bit weights + int AdamW, q4 sparse engram (orders 2,3 on layers 1,4).
rem `new` stores these in models\<name>.bin.env, so later train/eval/run calls pick them up by themselves.
set "CUDALM_TOPK=2"
set "CUDALM_BATCH=16"
set "CUDALM_QLM=1"
set "QLM_BITS=4"
set "QLM_OPT=qadam"
set "NGRAM_Q4=1"
set "NGRAM_Q4_FOLD=1"
set "NGRAM_HEADS=4"
set "NGRAM_ORDERS=2,3"
set "NGRAM_DIM=32"
set "NGRAM_ROWS=524287"
set "NGRAM_GATE=1"
set "NGRAM_LAYERS=1,4"
set "NGRAM_CONV=4"
rem canonical-token map (folds case / leading-space variants for the engram hash); made by `cudalm data tokenizer` or `data canon`
set "NGRAM_CANON="
if not "%~1"=="" for %%d in ("%~1" "%~dp0..\data\%~1" "%~dp0..\prep\data\%~1") do if not defined NGRAM_CANON if exist "%%~d\canon.bin" set "NGRAM_CANON=%%~fd\canon.bin"
if not "%~1"=="" if not defined NGRAM_CANON echo [s8w2env] no canon.bin for %~1 - run: cudalm data canon %~1

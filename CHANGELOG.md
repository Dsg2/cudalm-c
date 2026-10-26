# Changelog

## Unreleased - 2026-10-27

- prep/d1build.py: json.dump call
- Add default=int to prevent serialization errors on non-int values

## Unreleased - 2026-10-26

- update(queue): adjust benchmark rows

## Unreleased - 2026-10-26

- Update k_kd_adj to accept rn parameter and compute renormalised mass
- Add HWR5/HWR9 entries to prep/docsel.py for KD arms
- Add new scripts/kdfix.ps1 for KD rerun with build 20o

## Unreleased - 2026-10-26

- Add II_QUEUE.md with stage plan and resume markers
- Add CUDALM_KD env flag and kdcache.py script
- Update cudalm.cu with kd_adj, k_topk_row, vres, kd, kdcache functions
- Update prep scripts: gate, d1build, docsel, gate, hwbuild
- Update scripts: fdoc, master, stage3

## Unreleased - 2026-10-26

- Add FILTER_PLAN.md with block probe results and doc-level filter criteria
- Add CUDALM_EVAL_SPANS flag to cudalm.cu for per-document scoring
- Add prep/docsel.py and prep/fdocsum.py for arm selection and replicate reporting

## Unreleased - 2026-10-25

- Add filtering plan and new pipeline scripts

## Unreleased - 2026-10-25

- add(arith): adjust reward calculation for near misses
- add(cudalm): fix looped model parameter reading
- add(cudalm): adjust Muon loading for n < np
- add(cudalm): adjust g_muon default for per-layer models

## Unreleased - 2026-10-25

- Add num_first support to cudalm serve
- Add gen-arith and arith command handlers to agentenv
- Update README.md with new commands

## Unreleased - 2026-10-25

- Add static const char*pcomma(double v) function to format numbers with thousands separators
- Update pstats_print to use pcomma for numeric values in stat lines
- Modify stat output lines to append formatted numbers to existing values

## Unreleased - 2026-10-25

- Add memset to clear cuda config before parsing header fields in cudalm.cu
- Prevent stack garbage from header parsing in cudalm.cu

## Unreleased - 2026-10-25

- README.md: document new default of 10000 for CUDALM_BENCH_EVERY
- cudalm.cu: change default value from 0 to 10000
- tui.cuh: change default value from 0 to 10000

## Unreleased - 2026-10-24

- Add CUDALM_D, CUDALM_L, CUDALM_H, CUDALM_KV, CUDALM_F to g_envreg
- Update cmd_new to parse env overrides for preset shape
- Add validation for shape constraints (D%H==0, H%KV==0, even dim, L<=64)

## Unreleased - 2026-10-24

- README.md: document new default optimizer sweep results and QLM behavior
- cudalm.cu: update default LR, B2, WD, and QLM defaults in env reg table
- cudalm.cu: add g_muon_exp flag and default Muon off for QLM/looped models
- cudalm.cu: update default_lr_for to use width-based Muon/AdamW scaling
- cudalm.cu: pin QLM and Muon in model config env

## Unreleased - 2026-10-24

- In gpu_muon(): replace hardcoded wd=0.1f with env-based lookup
- In g_envreg[]: update CUDALM_WD description to include Muon and QLM

## Unreleased - 2026-10-24

- Update README.md with new optimizer knobs
- Add CUDALM_LR_MIN, B1, B2, WD env vars to README.md

## Unreleased - 2026-10-24

- add cudalm.cu and qlm.cuh support for CUDALM_LOOP

## Unreleased - 2026-10-23

- README.md: document CUDALM_DATA_QA=1 behavior
- data.cuh: new d_qa_on, d_split_qa, d_qa_ans functions; updated d_doc, cmd_data_encode
- tui.cuh: new Q/A -> chat command with CUDALM_DATA_QA env flag

## Unreleased - 2026-10-23

- data.cuh: new d_chat_fin, d_split_chat, d_chatcol functions and d_doc logic
- tui.cuh: updated --template help text for ChatML markers

## Unreleased - 2026-10-23

- README.md: add documentation for chat template splitting and rendering
- prep/fetch.py: add split_chat, text_doc, ctext functions and update conv_turns, auto_doc, tpl_doc

## Unreleased - 2026-10-23

- README.md: Add usage example for `cudalm data tokenizer-from`
- cudalm.cu: Add `g_tokblob`/`g_tokblobl` globals and `tok_keep()` function
- cudalm.cu: Update `model_load_full()` to call `tok_keep()` after tokenizer parse
- cudalm.cu: Update `info()` to call `tok_keep()` after tokenizer parse
- data.cuh: Add `d_tokjson()` function to generate tokenizer.json

## Unreleased - 2026-10-23

- README.md: document new `cudalm lmeval-vs` command and margin reporting
- cudalm.cu: implement `le_load`, `le_vs`, `cmd_lmeval_vs`, and margin index in `cmd_lmeval`

## Unreleased - 2026-10-23

- Add template parsing and rendering functions to data.cuh
- Add data peek command to data.cuh
- Update data.cuh hf_rows usage to support template
- Add --template flag to fetch.py and update its usage
- Update README.md with template documentation and examples

## Unreleased - 2026-10-23

- Add hf_* CLI commands and fetch.py profile handler for HF datasets
- Add WinINet-based d_http for Hugging Face HTTP calls
- Update data.cuh d_walk to skip sources.json metadata
- Update README.md with HF dataset usage examples
- Update tui.cuh data tools with HF browse option

## Unreleased - 2026-10-23

- feat(data): add d_count() for token/byte counts with k/M/B suffixes
- feat(cudalm): tuicheck now passes all args instead of only the second
- docs: update README.md with new TUI data tools and token count suffixes

## Unreleased - 2026-10-23

- Add ensure_dirs() to cudalm.cu to create these folders next to the exe
- Add README.md entry explaining the new folder structure

## Unreleased - 2026-10-23

- README.md: clarify default stage LR and benchmark task behavior
- cudalm.cu: change default_lr_for to 1/3 of from-scratch peak for new stages
- tui.cuh: update default CUDALM_BENCH_N to 0 and CUDALM_BENCH_TASKS to all tasks

## Unreleased - 2026-10-23

- README.md: add speed switches section
- cudalm.cu: add cp.async tile prefetch support and CUDALM_* flags

## Unreleased - 2026-10-23

- README.md: document CUDALM_BENCH_EVERY, CUDALM_BENCH_TASKS, CUDALM_BENCH_N
- cudalm.cu: add CUDALM_BENCH_* env vars, train_bench() function, and bench_every logic in cmd_train
- tui.cuh: add bench every and bench items flags to train command

## Unreleased - 2026-10-23

- Add DSCALE and DSCALE_INV definitions to k_attn_fwd_tiled
- Add g_dsc array and k_dsc_amax/k_dsc_set functions for adaptive scale
- Replace DSCALE_INV with g_dsc[1] in dQ/dK/dV writes
- Replace DSCALE_INV/DSCALE_INV with g_dsc[1] in backward functions
- Add nan_where locator for non-finite grad analysis

## Unreleased - 2026-10-22

- Add "ufw-l3-qa" and "ufw-l3-ms" entries to S
- Add "ufwl3-en" profile with qa + multi_style sources
- Update save() to write .done file for data encode follow=1

## Unreleased - 2026-10-22

- Add `g_stage0` flag to track first iteration of a training stage
- Update `lr_at()` to use `g_stage0` for relative iteration counting
- Add `CUDALM_PERM_BASE` to `envreg` with `rt=1` to mark it as per-run setting
- Update `cmd_train()` to detect stage boundaries and print stage info
- Update `tui.cuh` `train` command to document stage behavior

## Unreleased - 2026-10-22

- README.md: document new cudalm lmeval output
- cudalm.cu: extend cmd_lmeval with acc_norm index calculation
- cudalm.cu: add index summary to lmeval output

## Unreleased - 2026-10-22

- README.md: update lmeval command line and benchmark table
- bench/lmeval_prep.py: add arcc_test.parquet and arithmark-3.jsonl sources
- cudalm.cu: extend lmeval tasks list and usage help
- tui.cuh: update lmeval tasks choice values

## Unreleased - 2026-10-22

- Change g_asn_ev initialization from NULL to 0 in moe_perm
- Update cmd_bench to print "system RAM" or "VRAM" instead of hardcoded value
- Add GPU build-only message for "decheck" command in main

## Unreleased - 2026-10-22

Add download logic for missing task files in cudalm lmeval
Add lmeval_prep.py download helper for missing benchmarks
Add grow_sidecars() to copy model config and sidecars
Add new lmeval command with missing-file detection
Add new tui.cuh commands for lmeval and diagnostics
Add new cudalm decheck and tokcheck commands
Add new cudalm quantize option for .env files
Add new cudalm data command for tokenizer and corpus tools
Add new cudalm selftest child support
Add utf-8-sig handling in prep/fetch.py readers
Add new cudalm lmeval_prep.py usage and download logic
Add new cudalm lmeval_prep.py benchmark file detection
Add new cudalm lmeval_prep.py missing-file handling
Add new cudalm lmeval_prep.py run logic
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file conversion
Add new cudalm lmeval_prep.py benchmark file download
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py benchmark file handling
Add new cudalm lmeval_prep.py benchmark file fetching
Add new cudalm lmeval_prep.py

## Unreleased - 2026-10-22

- Add cudalm lmeval command to README.md
- Create bench/lmeval_prep.py to convert HellaSwag, ARC-Easy, PIQA and LAMBADA parquet/jsonl files into cudalm lmeval .tsv files
- Add cmd_lmeval function to cudalm.cu with zero-shot lm-eval-harness prompt handling

## Unreleased - 2026-10-22

Add chatbench v2 benchmark file
Add chatbench v2 benchmark file
Add chatbench v2 benchmark file
Add chatbench v2 benchmark file

## Unreleased - 2026-10-18

- Add CUDALM_ROUTER_Z flag to enable router z-loss
- Add zl_fwd/zl_bwd functions and g_zl/g_zlse state
- Wire zl_fwd into moe_forward after router top-k
- Wire zl_bwd into moe_backward after router bwd
- Register CUDALM_ROUTER_Z env var in main

## Unreleased - 2026-10-18

- Add k_rope_h kernel to cpu_kern.h for forward RoPE
- Add k_attn_fwd_fa, k_attn_bwd_dq_fa, k_attn_bwd_dkv_fa kernels to cudalm.cu
- Add attn_fwd_fa, attn_bwd_dq_fa, attn_bwd_dkv_fa templates to cpu_kern.h
- Add cudaEventCreate, cudaEventQuery, cudaEventElapsedTime to cpu_rt.h
- Enhance profiling with drain tracking and async GPU time in cudalm.cu

## Unreleased - 2026-10-18

- Changed sfx array from 6 to 7 entries in cudalm.cu
- Added ".opt.x1" to checkpoint naming at line 6809
- Loop now iterates 7 times instead of 6

## Unreleased - 2026-10-17

- Add agent/agentenv.c, cpu_kern.h, cpu_rt.h for CPU build
- Update README.md with CPU build instructions and layout table
- Update build.py to support --cpu flag
- Update .gitignore with _scratch/ directory


# Changelog

## v0.1.0 - 2026-11-25

- **Faster MoE training without the stash.** On LS4 (top-3, head dim 32) training went from 33.0k to 45.6k tok/s on a
  5060 Ti, and looped LS4 with gradient accumulation from 14.8k to 21.6k, with losses unchanged:
  - top-K passes merge into one grouped pass for any K, stash on or off (`CUDALM_MOE_MERGE`);
  - the expert W13 GEMM has SwiGLU in its epilogue (`CUDALM_SWF`);
  - without the stash, a fused expert backward replaces the forward recompute (`CUDALM_SWB`);
  - head dim 32 uses the FA2 attention kernels (`CUDALM_FA32`);
  - Muon's Newton-Schulz runs in bf16 with the polynomial folded into one small matrix (`CUDALM_MUON_BF16`).
  Each `=0` restores the old path. The selftest checks the new kernels against the old paths.
- **GitHub Actions build** (`.github/workflows/build.yml`): CPU (with selftest) and GPU sm_75/86/89/120 builds on
  Ubuntu and Windows for every push; tags `v*` attach the binaries to the release.
- `CUDALM_CB_B` (default 16): chatbench sequences per batch. Lower it for deep or looped models.
- **Looped layers with attention gates.** `CUDALM_LOOP` now combines with `CUDALM_ATTN_GATE`. Each repeat pass has
  its own gate, copied (with its Adam moments) from the shared layer the first time a plain checkpoint loads looped.
- **Gradient accumulation.** `CUDALM_ACCUM=k` runs k micro-batches of `CUDALM_BATCH` per optimizer step. They see
  the same windows in the same order as one batch of k×B, share one loss denominator (all trainable tokens of the step)
  and add into the same gradients, so the step equals the big-batch step. The Q4_FOLD n-gram table collects every
  micro-batch's keys and row gradients and scatters them once. MoE balance terms (aux loss, aux-free bias step) are
  per micro-batch and scaled by 1/k. Steps per epoch shrink k-fold. Not with QLM_FUSED, KD, TOKW, LRS, engram,
  canon/shared or NGRAM_TRI. On a nano MoE with gated attention and a folded n-gram table, B4, B2×2 and B1×4 (and
  looped B4 vs B2×2) give identical losses over 41 steps.
- `CUDALM_LMEVAL_N`: tokens per lmeval batch (default 16384). Lower it for deep or looped models that would otherwise
  spill out of VRAM.

## 2026-11-22

- **Hugging Face export.** `cudalm hf <name> <dir> [fp32|bf16]` writes a transformers folder with a custom `cudalm`
  architecture (`trust_remote_code`): safetensors weights, config, tokenizer + chat template, model card. Model code in
  `export/hf/`, tokenizer converter `export/tltk2hf.py`. `cudalm logits` dumps full-vocab logits (the CPU build is the
  fp32 reference). `export/hf_check.py` checks the export against it: fp32 logits, greedy decoding with and without the
  KV cache (also across the sliding window), the chat template and bf16 agreement. `hf_tokcheck.py` checks tokenizer
  ids, `hf_diag.py` explains MoE routing ties. lm-evaluation-harness runs on the export: `export\lmeval\` has an
  ArithMark 3.0 task, an EOT-prefix tokenizer that makes lm-eval score as `cudalm lmeval` does, and `compare.py` (per-item
  agreement ≥ 99.9% on LS4-200M-A50M-B).
- **Checkpoints by wall clock.** `train` saves every `CUDALM_SAVE_MIN` minutes (default 30) plus at every keep
  snapshot and bench step, instead of every ~200 steps; a 200M-parameter model wrote ~1.6 GB every 3 minutes.
  `CUDALM_SAVE_EVERY=N` keeps the step cadence. Both are per-run settings, no longer stored in `<model>.bin.env`
  (an old value there is ignored with a note). With timed saves, benches run exactly every `CUDALM_BENCH_EVERY` steps.
- Muon momentum (`.opt.mu`) is saved as bf16, half the size; fp32 files from older builds still load.
- **`cudalm monitor`** (`monitor.cuh`, `monitor.html` → `monitor_html.h` via `embedhtml.py`): read-only HTTP view of
  training runs. Dashboard at `/` (run cards, loss / II / margin charts, log tail, GPU), JSON under `/api/`, optional
  token, `--allow-stop`. `train` writes `<model>.bin.status.json` (`CUDALM_STATUS=0` turns it off).
- `net.cuh`: the socket layer of `serve`, now portable; `cudalm serve` builds and runs on Linux.

## 2026-11-16

- CHANGELOG.md: update Unreleased entry to document tokenizer fix
- CHANGELOG.md: replace old tokenizer selection logic with full tokenizer for bench

## 2026-11-16

- cudalm.cu: fix crash (access violation) at the first in-process bench of a freshly created model. A new model
  file has no tokenizer, so train handed the bench the data tokenizer parsed without its encode tables; it now
  parses a full tokenizer for the bench when benches are on (CUDALM_BENCH_EVERY > 0, the default).

## 2026-11-04

- README.md: document Linux build on Ubuntu 24.04/WSL2 and menu data toolchain
- build.py: add Linux g++ testing and menu help message
- cudalm.cu: remove Windows-only data flag and TUI condition
- data.cuh: add POSIX file operations for Linux data toolchain
- tui.cuh: adjust TUI condition for Linux

## 2026-11-04

- Update README.md with Linux testing info and Windows-only toolchain notes
- Add curl-based download fallback in build.py
- Update cudalm.cu to use lm_exists instead of d_exists for benchmark file checks
- Add Windows-only data toolchain guard in cudalm.cu

## 2026-11-04

- README.md: replace tested-on and GPU support sections with Windows 11, RTX 50 (sm_120) and CUDA 12.9 baseline; add RTX 20/30/40/50 build table for --arch options
- build.py: change --arch flag help text and validation logic to accept RTX series (75/86/89/120) instead of only 120/89/86
- cudalm.cu: add CUDALM_ARCH compile-time path selection via -DCUDALM_ARCH; conditionally include cp.async and bf16 GEMMs for RTX 30/40/50

## 2026-11-04

- README.md: expand tested hardware and add GPU support section
- build.py: extend --arch options and validation
- cudalm.cu: add per-arch feature gating for lower GPUs

## 2026-11-04

- save_rename: retry MoveFileExA 10 times with Sleep(100) for NTFS
- add d2h_write: 64 MB host buffer for save device-to-host
- opt_save: use d2h_write for bf16 optimizer save
- ngram_save: replace malloc/cudaMemcpy chain with t1_rw/ng_tbl_rw
- mu_save: replace malloc/cudaMemcpy with d2h_write

## 2026-11-03

- README.md: update benchmark description to note in-process mode
- cudalm.cu: add CUDALM_BENCH_CHILD env flag
- cudalm.cu: add lmeval_core function for in-process scoring
- cudalm.cu: add LeRes struct and refactor train_bench to use it

## 2026-11-03

- Add static size_t mem_tok_bytes(int N) function
- Add static size_t mem_tok_bytes(N) call in ga_bytes for VRAM report
- Move VRAM headroom check to after ngram/engram load
- Adjust auto_batch and moe_stash_decide call order

## 2026-11-03

- Add hd_warn() to cudalm.cu
- Add hd_warn() to tui.cuh
- Warn when head dims are not 8/16/32/64
- Print warning in cmd_new and cmd_train

## 2026-11-03

- README.md: document CUDALM_STAGE, CUDALM_DECAY_SHAPE, and new TUI form fields
- cudalm.cu: add g_stage_kind, g_dshape, and updated lr_at logic for stable/decay stages
- tui.cuh: add LR schedule, peak LR, warmup, decay frac, decay shape, LR floor form fields and t_train_sched()

## 2026-10-30

- README.md: update tokenizer usage line to include [maxdigits] option
- data.cuh: add maxdigits argument parsing and validation in cmd_data_tokenizer
- data.cuh: skip token pairs exceeding maxdigits limit during merge

## 2026-10-30

- README.md: document new lmeval-vs output lines
- cudalm.cu: implement scale-free and accuracy index calculations in le_vs()

## 2026-10-30

- Update README.md with new loop training options
- Add loop_setk, loop_pass, loop_kmax functions to cudalm.cu
- Add g_lskip and g_lk_init to loop training logic in cudalm.cu
- Add new env flags and training output to cmd_train in cudalm.cu

## 2026-10-30

- README.md: document new flag and factorized embedding option
- cudalm.cu: add fe_init, fe_update, fe_grad, fe_mat, fe_save functions and env var
- tui.cuh: add CUDALM_EMB_E to command table

## 2026-10-29

- Update preset and new model commands
- Add TUI custom preset field
- Add TUI shape validation for new model

## 2026-10-29

- Tidy changelog

## 2026-10-29

- Drop references to private notes from d1build.py and the changelog

## 2026-10-29

- Add _PLAN.md, II_*.md, *Training Analysis.md to .gitignore
- Update .gitignore with new ignore patterns

## 2026-10-29

- README.md: document cudalm lens-tune command
- cpu_kern.h: add k_logsm and k_kld kernels
- cudalm.cu: add blk_reduce, lens_src, tl_apply, tl_load, lens_open, lens_gpu, and cmd_lens_tune

## 2026-10-29

- README.md: document new lens subcommand
- cpu_kern.h: add k_lens kernel
- cudalm.cu: add k_lens kernel and cmd_lens function

## 2026-10-29

- Add reference client script
- Update help text for serve command

## 2026-10-29

- Add cmd_ngfold() to cudalm.cu
- Add ngfold entry to cudalm main()
- Add ngfold.ps1 and scripts/ngfold.ps1
- Add scripts/ngrec.ps1

## 2026-10-27

- New file: scripts/stage4.ps1
- Implements stage4 logic: waits for stage3, runs loop arm and LS-Micro3-D1lr evaluation
- Uses environment variables for paths and seeds
- Outputs results to stage4.res

## 2026-10-27

- prep/d1build.py: json.dump call
- Add default=int to prevent serialization errors on non-int values

## 2026-10-26

- update(queue): adjust benchmark rows

## 2026-10-26

- Update k_kd_adj to accept rn parameter and compute renormalised mass
- Add HWR5/HWR9 entries to prep/docsel.py for KD arms
- Add new scripts/kdfix.ps1 for KD rerun with build 20o

## 2026-10-26

- Add CUDALM_KD env flag and kdcache.py script
- Update cudalm.cu with kd_adj, k_topk_row, vres, kd, kdcache functions
- Update prep scripts: gate, d1build, docsel, gate, hwbuild
- Update scripts: fdoc, master, stage3

## 2026-10-26

- Add CUDALM_EVAL_SPANS flag to cudalm.cu for per-document scoring
- Add prep/docsel.py and prep/fdocsum.py for arm selection and replicate reporting

## 2026-10-25

- Add filtering plan and new pipeline scripts

## 2026-10-25

- add(arith): adjust reward calculation for near misses
- add(cudalm): fix looped model parameter reading
- add(cudalm): adjust Muon loading for n < np
- add(cudalm): adjust g_muon default for per-layer models

## 2026-10-25

- Add num_first support to cudalm serve
- Add gen-arith and arith command handlers to agentenv
- Update README.md with new commands

## 2026-10-25

- Add static const char*pcomma(double v) function to format numbers with thousands separators
- Update pstats_print to use pcomma for numeric values in stat lines
- Modify stat output lines to append formatted numbers to existing values

## 2026-10-25

- Add memset to clear cuda config before parsing header fields in cudalm.cu
- Prevent stack garbage from header parsing in cudalm.cu

## 2026-10-25

- README.md: document new default of 10000 for CUDALM_BENCH_EVERY
- cudalm.cu: change default value from 0 to 10000
- tui.cuh: change default value from 0 to 10000

## 2026-10-24

- Add CUDALM_D, CUDALM_L, CUDALM_H, CUDALM_KV, CUDALM_F to g_envreg
- Update cmd_new to parse env overrides for preset shape
- Add validation for shape constraints (D%H==0, H%KV==0, even dim, L<=64)

## 2026-10-24

- README.md: document new default optimizer sweep results and QLM behavior
- cudalm.cu: update default LR, B2, WD, and QLM defaults in env reg table
- cudalm.cu: add g_muon_exp flag and default Muon off for QLM/looped models
- cudalm.cu: update default_lr_for to use width-based Muon/AdamW scaling
- cudalm.cu: pin QLM and Muon in model config env

## 2026-10-24

- In gpu_muon(): replace hardcoded wd=0.1f with env-based lookup
- In g_envreg[]: update CUDALM_WD description to include Muon and QLM

## 2026-10-24

- Update README.md with new optimizer knobs
- Add CUDALM_LR_MIN, B1, B2, WD env vars to README.md

## 2026-10-24

- add cudalm.cu and qlm.cuh support for CUDALM_LOOP

## 2026-10-23

- README.md: document CUDALM_DATA_QA=1 behavior
- data.cuh: new d_qa_on, d_split_qa, d_qa_ans functions; updated d_doc, cmd_data_encode
- tui.cuh: new Q/A -> chat command with CUDALM_DATA_QA env flag

## 2026-10-23

- data.cuh: new d_chat_fin, d_split_chat, d_chatcol functions and d_doc logic
- tui.cuh: updated --template help text for ChatML markers

## 2026-10-23

- README.md: add documentation for chat template splitting and rendering
- prep/fetch.py: add split_chat, text_doc, ctext functions and update conv_turns, auto_doc, tpl_doc

## 2026-10-23

- README.md: Add usage example for `cudalm data tokenizer-from`
- cudalm.cu: Add `g_tokblob`/`g_tokblobl` globals and `tok_keep()` function
- cudalm.cu: Update `model_load_full()` to call `tok_keep()` after tokenizer parse
- cudalm.cu: Update `info()` to call `tok_keep()` after tokenizer parse
- data.cuh: Add `d_tokjson()` function to generate tokenizer.json

## 2026-10-23

- README.md: document new `cudalm lmeval-vs` command and margin reporting
- cudalm.cu: implement `le_load`, `le_vs`, `cmd_lmeval_vs`, and margin index in `cmd_lmeval`

## 2026-10-23

- Add template parsing and rendering functions to data.cuh
- Add data peek command to data.cuh
- Update data.cuh hf_rows usage to support template
- Add --template flag to fetch.py and update its usage
- Update README.md with template documentation and examples

## 2026-10-23

- Add hf_* CLI commands and fetch.py profile handler for HF datasets
- Add WinINet-based d_http for Hugging Face HTTP calls
- Update data.cuh d_walk to skip sources.json metadata
- Update README.md with HF dataset usage examples
- Update tui.cuh data tools with HF browse option

## 2026-10-23

- feat(data): add d_count() for token/byte counts with k/M/B suffixes
- feat(cudalm): tuicheck now passes all args instead of only the second
- docs: update README.md with new TUI data tools and token count suffixes

## 2026-10-23

- Add ensure_dirs() to cudalm.cu to create these folders next to the exe
- Add README.md entry explaining the new folder structure

## 2026-10-23

- README.md: clarify default stage LR and benchmark task behavior
- cudalm.cu: change default_lr_for to 1/3 of from-scratch peak for new stages
- tui.cuh: update default CUDALM_BENCH_N to 0 and CUDALM_BENCH_TASKS to all tasks

## 2026-10-23

- README.md: add speed switches section
- cudalm.cu: add cp.async tile prefetch support and CUDALM_* flags

## 2026-10-23

- README.md: document CUDALM_BENCH_EVERY, CUDALM_BENCH_TASKS, CUDALM_BENCH_N
- cudalm.cu: add CUDALM_BENCH_* env vars, train_bench() function, and bench_every logic in cmd_train
- tui.cuh: add bench every and bench items flags to train command

## 2026-10-23

- Add DSCALE and DSCALE_INV definitions to k_attn_fwd_tiled
- Add g_dsc array and k_dsc_amax/k_dsc_set functions for adaptive scale
- Replace DSCALE_INV with g_dsc[1] in dQ/dK/dV writes
- Replace DSCALE_INV/DSCALE_INV with g_dsc[1] in backward functions
- Add nan_where locator for non-finite grad analysis

## 2026-10-22

- Add "ufw-l3-qa" and "ufw-l3-ms" entries to S
- Add "ufwl3-en" profile with qa + multi_style sources
- Update save() to write .done file for data encode follow=1

## 2026-10-22

- Add `g_stage0` flag to track first iteration of a training stage
- Update `lr_at()` to use `g_stage0` for relative iteration counting
- Add `CUDALM_PERM_BASE` to `envreg` with `rt=1` to mark it as per-run setting
- Update `cmd_train()` to detect stage boundaries and print stage info
- Update `tui.cuh` `train` command to document stage behavior

## 2026-10-22

- README.md: document new cudalm lmeval output
- cudalm.cu: extend cmd_lmeval with acc_norm index calculation
- cudalm.cu: add index summary to lmeval output

## 2026-10-22

- README.md: update lmeval command line and benchmark table
- bench/lmeval_prep.py: add arcc_test.parquet and arithmark-3.jsonl sources
- cudalm.cu: extend lmeval tasks list and usage help
- tui.cuh: update lmeval tasks choice values

## 2026-10-22

- Change g_asn_ev initialization from NULL to 0 in moe_perm
- Update cmd_bench to print "system RAM" or "VRAM" instead of hardcoded value
- Add GPU build-only message for "decheck" command in main

## 2026-10-22

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

## 2026-10-22

- Add cudalm lmeval command to README.md
- Create bench/lmeval_prep.py to convert HellaSwag, ARC-Easy, PIQA and LAMBADA parquet/jsonl files into cudalm lmeval .tsv files
- Add cmd_lmeval function to cudalm.cu with zero-shot lm-eval-harness prompt handling

## 2026-10-22

Add chatbench v2 benchmark file
Add chatbench v2 benchmark file
Add chatbench v2 benchmark file
Add chatbench v2 benchmark file

## 2026-10-18

- Add CUDALM_ROUTER_Z flag to enable router z-loss
- Add zl_fwd/zl_bwd functions and g_zl/g_zlse state
- Wire zl_fwd into moe_forward after router top-k
- Wire zl_bwd into moe_backward after router bwd
- Register CUDALM_ROUTER_Z env var in main

## 2026-10-18

- Add k_rope_h kernel to cpu_kern.h for forward RoPE
- Add k_attn_fwd_fa, k_attn_bwd_dq_fa, k_attn_bwd_dkv_fa kernels to cudalm.cu
- Add attn_fwd_fa, attn_bwd_dq_fa, attn_bwd_dkv_fa templates to cpu_kern.h
- Add cudaEventCreate, cudaEventQuery, cudaEventElapsedTime to cpu_rt.h
- Enhance profiling with drain tracking and async GPU time in cudalm.cu

## 2026-10-18

- Changed sfx array from 6 to 7 entries in cudalm.cu
- Added ".opt.x1" to checkpoint naming at line 6809
- Loop now iterates 7 times instead of 6

## 2026-10-17

- Add agent/agentenv.c, cpu_kern.h, cpu_rt.h for CPU build
- Update README.md with CPU build instructions and layout table
- Update build.py to support --cpu flag
- Update .gitignore with _scratch/ directory


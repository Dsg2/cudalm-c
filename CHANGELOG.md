# Changelog

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


# Changelog

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


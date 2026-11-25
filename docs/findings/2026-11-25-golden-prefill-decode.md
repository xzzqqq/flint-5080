# Prefill vs. Cached Decode Rounding in the M1.1 Reference

**Date**: 2026-11-25  
**Hardware / Software**: Single RTX 5080 (16,303 MiB, `sm_120a`), PyTorch `2.14.1+cu130`, `safetensors 0.8.0`  
**Checkpoint**: `Maxrio/Inkling-2-Mini-exl3` (`3.3bpw`, 48 logical visits, BF16 residual stream)  
**Artifact**: `tools/weights_cli/_inkling2_ref.py`, `.runs/20261125T032708.760737Z-weights-golden-3788a4b3.json`

## Summary

During M1.1 golden generation, an unconstrained eager PyTorch reference produced identical greedy token IDs between cached step-by-step decode and single-shot teacher-forced prefill, yet exhibited up to **3.850153** maximum absolute logit difference across the 200,058-entry vocabulary. Making GEMM blocking, reductions, and causal attention row slices batch-size-invariant eliminated the discrepancy completely (**0.0** max absolute logit error across all 288 checked positions).

## Measurements

### Initial eager reference (`short_math`, 8 cached decode steps)

| Check | Top-1 Agreement | Max Abs Diff |
|---|---:|---:|
| Full sequence vs. 8 cached steps (9 positions, 200,058 logits) | 100% (9/9) | `3.850153` |
| Single-token isolation at visit 0 residual (cached step 0 vs. last row of `prompt + tok0` prefill) | — | `0.03125` |
| Single-token isolation at LM head logits | — | `0.75925` |

### Root cause

Changing the row count $M$ between prefill ($M = T$) and single-token decode ($M = 1$) caused cuBLAS/PyTorch to select different GEMM tiling and reduction orders. Tiny FP32 rounding differences before BF16 quantization crossed BF16 ULP boundaries at visit 0 (`0.03125`) and compounded through 48 looped visits (including attention residuals and cross-pass controllers) into multi-unit logit drift.

### Stabilized reference (after fixing reduction geometry)

In `tools/weights_cli/_inkling2_ref.py`, we enforced batch-size-invariant execution:

- Fixed 32-row matrix multiplication tiling (`M`-independent reduction order),
- Per-vector norm reductions,
- Per-token block attention residual evaluation,
- Identical visible-key causal slices per attention row.

Re-running `./bin/weights golden --json` on the RTX 5080 across all three cases in `workloads/golden_cases.json`:

| Case | Prompt Tokens | Cached Decode Steps | Checked Positions | Top-1 Match | Max Abs Logit Diff |
|---|---:|---:|---:|---:|---:|
| `short_math` | 27 | 48 | 49 | 100.0% | `0.0` |
| `short_sky` | 24 | 101 | 102 | 100.0% | `0.0` |
| `short_code` | 25 | 136 | 137 | 100.0% | `0.0` |
| **Total** | — | **285** | **288** | **100.0%** | **`0.0`** |

- **Peak GPU memory**: `1,176,753,152` B (**1.10 GiB** allocated, 1.25 GiB reserved) via mmap + on-demand layer/hit-expert EXL3 dequantization.
- **Generator wall time**: `259.91 s` (total CLI `262.50 s`).

## Engineering Takeaway for Phase 1 C++/CUDA

1. **Bit-consistent reference oracle**: The golden `.safetensors` fixtures now have zero internal drift between prefill and incremental cached decode, removing ambiguity when debugging C++/CUDA state management (48 KV slots, 192 short-convolution states, and loop controller caches).
2. **Expected non-zero drift in production kernels**: High-throughput `sm_120a` kernels (different GEMV vs. GEMM schedules, fused Hadamard/EXL3, CUDA Graph) will naturally differ from the fixed-32-row reference in low-order BF16 bits; as `AGENTS.md` prescribes, Phase 1+ validation should judge functional parity via per-layer error bounds, top-k / greedy agreement, and teacher-forced KLD rather than bitwise identity.

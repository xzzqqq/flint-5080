# M1.4b 48-Logical-Visit E2E Alignment & First Real AR Baseline on RTX 5080

**Date**: 2026-11-25  
**Hardware / Software**: Single RTX 5080 (16,303 MiB, `sm_120a`), CUDA 13.0, `libcublas.so.13`, CMake 3.22 + Ninja  
**Checkpoint**: `Maxrio/Inkling-2-Mini-exl3` (`3.3bpw`, 11.57 GiB text weights, 48 logical visits)  
**Artifacts**: `engine/build/forward.so` (`504,424 B`), `.runs/m14b-e2e-golden.txt`, `.runs/20261125T093638.416031Z-forge-profile-04ea984f.json`

## Summary

Wiring the M1.4a `sm_120a` operators into `engine/src/model/inkling2_forward.{h,cpp}` and `forward.so` achieved **100% bit-exact agreement (`max_abs_err = 0`, `RMSE = 0`, `cos_sim = 1`)** against all three M1.1 PyTorch golden fixtures (`short_math`, `short_sky`, `short_code`) across **all 288 generated tokens**, all 48 prefill visits, MTP taps `5 / 23 / 41 / 48`, prefill last-token logits, all 48 `decode_0` visits, and `decode_0` logits.

Partial-range replay (`0..6`, `6..18`, `18..30`, `30..42`, `42..48`, and 48 single-visit calls) reproduced the single-shot run, all 8 carry planes, and the entire KV/convolution arena byte-for-byte.

## Measurements

### 1. Full Prefill + Multi-Step Cached Greedy Decode (`e2e_tests` via `dlopen(forward.so)`)

| Case | Prompt Tokens | Cached Decode Steps | Greedy IDs Matched | Prefill Latency (ms) | Mean C1 Decode (ms/tok) | Prefill Logits Top-1 / Top-5 | Decode 0 Logits Top-1 / Top-5 |
|---|---:|---:|---:|---:|---:|---|---|
| `short_math` | 27 | 48 | **49 / 49** | `2,297.590` | `130.762` | `1/1` · `5/5` | `1/1` · `5/5` |
| `short_sky` | 24 | 101 | **102 / 102** | `2,035.567` | `131.174` | `1/1` · `5/5` | `1/1` · `5/5` |
| `short_code` | 25 | 136 | **137 / 137** | `2,119.452` | `131.498` | `1/1` · `5/5` | `1/1` · `5/5` |
| **Total** | — | **285** | **288 / 288** | — | **~131.1 ms/tok (~7.6 tok/s)** | **Bit-exact (`0` err)** | **Bit-exact (`0` err)** |

- **Decoded answers** (all reached `<|content_model_end_sampling|>`):
  - `short_math`: `391`
  - `short_sky`: *"The sky is usually blue because air molecules scatter shorter blue wavelengths of sunlight more strongly than longer wavelengths, sending blue light toward our eyes from all directions."*
  - `short_code`: `` `f = lambda x: x**2` ``
- **Reduction-order sensitivity note**: An intermediate `short_sky` run diverged on 1/102 positions (position 35, where the top-two logit margin was `0.264946`). Matching the reference 4-element contiguous row reduction order (`engine/src/ops/row_reduction.cuh`) eliminated the divergence completely across all 288 tokens.

### 2. Live Resident `forge` Session (`Maxrio 3.3bpw` 11.57 GiB Resident)

All runs below reported **0 B VRAM loss** on unload (`dlclose`) after warm-up:

| `short_math` Operation | Lazy Cache | Median Latency | Golden Alignment |
|---|---|---:|---|
| `forge run --layers 0..6` (1st call) | Miss | `280.097 ms` | `prefill.step_5` bit-exact |
| `forge run --layers 0..6` (2nd call) | Hit | `280.970 ms` | `prefill.step_5` bit-exact |
| `forge run --layers 18..30` | Miss | `566.573 ms` | `prefill.step_29` bit-exact (restored cross-pass carry) |
| `forge run --e2e` (prefill) | — | `2,261.587 ms` | All 200,058 logits exact, next token `200001` |
| `forge run --e2e --step 1` (cached decode) | — | — | `decode_0.logits` exact, next token `200008` |
| `forge profile --e2e` (50 stages) | — | `2,262.700 ms` | Logits exact, next token `200001` |

### 3. First Real 50-Stage Baseline Profile (`short_math` prefill, 27 tokens)

| Phase | Visits | Measured Time |
|---|---:|---:|
| Embedding (`embed_rmsnorm`) | — | `0.008 ms` |
| Prologue | `0..6` (6) | `282.121 ms` |
| Core Pass 0 | `6..18` (12) | `560.997 ms` |
| Core Pass 1 | `18..30` (12) | `568.504 ms` |
| Core Pass 2 | `30..42` (12) | `568.717 ms` |
| Epilogue | `42..48` (6) | `280.948 ms` |
| LM Head (`unembed[:200058]` + softcap) | — | `1.397 ms` |
| **Total** | **48** | **`2,262.700 ms`** |

## Engineering Implications for M1.5 & Phase 3

1. **Baseline bottleneck is completely clear**: The LM head is already at `1.397 ms` (close to its `0.965 ms` 960 GB/s Roofline floor), whereas each logical visit averages **~2.7 ms/tok** on C1 decode (`~131 ms/tok` total, ~2.7% of the `3.55 ms` AR SOL ceiling) because the correctness-first M1.4 baseline dequantizes full EXL3 matrices (`inner_f16`) to scratch, calls FP32 cuBLAS GEMM per active expert, and synchronizes host-side to read top-8 expert IDs.
2. **Next step (M1.5)**: Make C1 decode MoE dispatch device-driven (eliminating the per-layer host sync), capture a static C1 decode CUDA Graph (`--no-graph` eager toggle + 0 B unload leak), and record the official Phase 1 AR Baseline Roofline & `forge ab` report before closing Phase 1.

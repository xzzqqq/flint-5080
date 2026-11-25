# M1.4a `sm_120a` Operator Alignment Against `ops_golden.safetensors`

**Date**: 2026-11-25  
**Hardware / Software**: Single RTX 5080 (16,303 MiB, `sm_120a`), CUDA 13.0, `libcublas.so.13` (dynamically linked), CMake 3.22 + Ninja (`-rdc=false`, PIC)  
**Checkpoint**: `Maxrio/Inkling-2-Mini-exl3` (`3.3bpw`, 11.57 GiB text weights)  
**Golden Fixture**: `/home/xzq/.cache/flint-5080/golden/Maxrio--Inkling-2-Mini-exl3--3.3bpw/ops_golden.safetensors` (87 tensors)

## Summary

All six `sm_120a` CUDA operator translation units (`exl3_linear.cu`, `norm_embed.cu`, `sconv.cu`, `attention.cu`, `moe.cu`, `loop_res.cu`) plus host cuBLAS/sub-module helpers (`gemm_cublas.{h,cpp}`, `submodules.{h,cpp}`) were verified against every operator and sub-module group in `ops_golden.safetensors` on the RTX 5080.

- All three EXL3 `mul1` (`0x83DCD12D`) bitrates (`3.0 / 3.5 / 6.0`-bit) decoded **bit-exact** full `inner_f16` matrices matching the reference SHA-256 digests.
- Embedding RMSNorm, all six Static/Dynamic SConv steps (with rolling state verification), both SWA and Full Attention modules (prefill + cached decode), all MoE outputs (router logits, top-8 IDs in slot order, weights, expert sum), and all four Message Gates matched **bit-for-bit (`0` max abs error)**.
- The largest FP32 discrepancy across all 25 checked outputs was **`7.63e-6`** (layer-5 3-query block residual, `RMSE = 8.87e-8`, `cos_sim = 1.0`).
- Single-`.cu` incremental compilation of `exl3_linear.cu` took **`0.6725 s`** (strictly header-isolated to `op_params.h` + `<cuda_fp16.h>` / `<cuda_bf16.h>`). Teardown VRAM delta was **`0 B`**.

## Measurements

### 1. EXL3 `mul1` Trellis Decoder Bit-Exactness

| Bitrate | Checkpoint Module | Full `inner_f16` SHA-256 (Match) | `inner_f16_slice` (`32×32`) |
|---:|---|---|---|
| `3.0` | `0.experts.gate_up_proj.0` | `21615d5a737f77639bea4b32d625d29f5949044099d12b6a777e357eff60822b` | `1,024 / 1,024` exact |
| `3.5` | `0.experts.gate_up_proj.1` | `d00934da3e15403adbaa6a4d6831e2de9513e02e230c87828ff7de0e34ddecef` | `1,024 / 1,024` exact |
| `6.0` | `0.self_attn.q_proj.linear` | `dd0df02c99ddb5d6d4fce7730951cbc06bc745b74735be4c21a449adaca62581` | `1,024 / 1,024` exact |

### 2. Per-Operator Numerical Alignment (`ops_tests` on RTX 5080)

| Operator / Sub-Module Output | Max Abs Err | Mean Abs Err | RMSE | Cos Sim | Bit-Exact Elements |
|---|---:|---:|---:|---:|---:|
| Embedding RMSNorm | `0` | `0` | `0` | `1` | `9,216 / 9,216` |
| EXL3 `3.0`-bit linear `y` | `1.90735e-6` | `3.58363e-7` | `4.51852e-7` | `1` | `402 / 6,912` |
| EXL3 `3.5`-bit linear `y` | `1.66893e-6` | `3.64425e-7` | `4.59069e-7` | `1` | `422 / 6,912` |
| EXL3 `6.0`-bit linear `y` | `1.81794e-6` | `3.76653e-7` | `4.77107e-7` | `1` | `590 / 9,216` |
| Static SConv (layer 0 `k_sconv`, `prefill`) | `0` | `0` | `0` | `1` | `4,096 / 4,096` |
| Static SConv (`decode_0`) | `0` | `0` | `0` | `1` | `1,024 / 1,024` |
| Static SConv (`decode_1`) | `0` | `0` | `0` | `1` | `1,024 / 1,024` |
| Dynamic SConv (layer 5 `k_sconv`, `prefill`) | `0` | `0` | `0` | `1` | `4,096 / 4,096` |
| Dynamic SConv (`decode_0`) | `0` | `0` | `0` | `1` | `1,024 / 1,024` |
| Dynamic SConv (`decode_1`) | `0` | `0` | `0` | `1` | `1,024 / 1,024` |
| Sliding Attention (layer 0, `prefill`) | `0` | `0` | `0` | `1` | `9,216 / 9,216` |
| Sliding Attention (layer 0, `decode_0`) | `0` | `0` | `0` | `1` | `2,304 / 2,304` |
| Full Attention (layer 5, `prefill`) | `0` | `0` | `0` | `1` | `9,216 / 9,216` |
| Full Attention (layer 5, `decode_0`) | `0` | `0` | `0` | `1` | `2,304 / 2,304` |
| MoE router logits (layer 0) | `0` | `0` | `0` | `1` | `512 / 512` |
| MoE top-8 IDs (slot order) | — | — | — | — | `32 / 32` |
| MoE routing weights | `0` | `0` | `0` | `1` | `32 / 32` |
| MoE expert output `y` | `0` | `0` | `0` | `1` | `9,216 / 9,216` |
| Block Attention Residual (layer 1, 1 query) | `1.90735e-6` | `1.70939e-8` | `4.67401e-8` | `1` | `4,675 / 9,216` |
| Block Attention Residual (layer 5, 3 queries) | `7.62939e-6` | `2.86995e-8` | `8.87389e-8` | `1` | `12,307 / 27,648` |
| Message Attention Residual (layer 6, pass 1, 4 queries) | `3.81470e-6` | `2.53826e-8` | `7.00323e-8` | `1` | `15,270 / 36,864` |
| Message Gate `attn` / `k` / `v` / `mlp` (layer 6) | `0` (all 4) | `0` (all 4) | `0` (all 4) | `1` | `9,216 / 9,216` each |

Additional synthetic unit coverage in default CTest `ops_tests` (`1.66 s` on RTX 5080): SWA ring-buffer wraparound past `1024`, Full Attention beyond `rel_extent = 1024`, `LogScale` above `pos = 8192`, router cutoff ties, and lowest-ID greedy `argmax` ties.

## Notes for M1.4b & M1.5

- **Baseline MoE host sync**: The M1.4a baseline `RunSparseMoeBlock` copies the selected top-8 expert IDs to host once per layer to loop over hit experts via scratch-staged EXL3 linear + cuBLAS. That is clean and exact for M1.4b eager 48-layer bring-up; for **M1.5 C1 decode CUDA Graph capture**, MoE expert dispatch must become device-driven (indirect pointer table over the 8 selected experts per decode token, zero host sync during graph capture).
- **EXL3 baseline path**: Scratch-staging `inner_f16` (≤ 13.5 MiB) + FP32 cuBLAS GEMM gives clean baseline accuracy now and leaves clear headroom for Phase 3 in-register `sm_120a` trellis GEMV/GEMM fusion.

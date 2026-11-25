# Phase 1 Closing Baseline: C1 Decode CUDA Graph & RTX 5080 Roofline

**Date**: 2026-11-25  
**Hardware / Software**: Single RTX 5080 (16,303 MiB, `sm_120a`, 960 GB/s GDDR7), CUDA 13.0, `libcublas.so.13`, CMake 3.22 + Ninja (`-rdc=false`, PIC, `-lineinfo`)  
**Checkpoint**: `Maxrio/Inkling-2-Mini-exl3` (`3.3bpw`, 11.57 GiB text weights, 48 logical visits)  
**Artifacts**: `.runs/m15/e2e.log`, `.runs/m15/graph.json`, `.runs/m15/eager.json`, `.runs/m15/profile.json`, `.runs/m15/ab.json`, `.runs/m15/hot-edit.json`, `.runs/m15/hot-revert.json`

## Summary

Milestone M1.5 closed Phase 1 by replacing the M1.4b host-synced MoE dispatch with a 100% device-driven indexed EXL3 C1 GEMV and adding a position-indirect static CUDA Graph (`d_decode_pos` + static cuBLAS workspace) to `forward.so`.

- **Zero-recapture C1 decode CUDA Graph**: Each golden case captured **1 static CUDA Graph** and replayed it across all **48 / 101 / 136** cached decode steps with **0 max logit difference** between Graph and Eager and **288 / 288 exact greedy token matches** against M1.1 PyTorch goldens.
- **3.2× C1 decode speedup over M1.4b**: C1 decode latency dropped from `~131.1 ms/tok` (M1.4b eager) to **`50.74 ms/tok`** (M1.5 eager, device-driven EXL3 GEMV) and **`40.88 ms/tok` (`24.46 tok/s`, 8.53% logical SOL)** with CUDA Graph (`1.24×` pure graph gain), with **0 B** VRAM loss on `dlclose`.
- **Hot-reload SLA verified on full real `forward.so`**: Modifying a single operator `.cu` (`norm_embed.cu`) and hot-reloading into the 11.57 GiB resident `forge-host` completed in **`1.71 s`** (warm lazy cache) / **`3.98 s`** (cold lazy record) — well under the 6 s Phase 1 SLA.

## Measurements

### 1. Multi-Step E2E Parity: Eager vs. Single-Graph C1 Decode (`e2e_tests`)

| Case | Greedy IDs | Eager (ms/tok) | Eager (tok/s) | Graph (ms/tok) | Graph (tok/s) | Graph Speedup | Graph Captures / Launches | Graph vs. Eager Max Logit Diff | Unload Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `short_math` | **49 / 49** | `51.21` | `19.53` | **`41.24`** | **`24.25`** | `1.242×` | `1 / 48` | `0` | `0 B` |
| `short_sky` | **102 / 102** | `52.56` | `19.03` | **`44.03`** | **`22.71`** | `1.194×` | `1 / 101` | `0` | `0 B` |
| `short_code` | **137 / 137** | `52.91` | `18.90` | **`44.37`** | **`22.54`** | `1.192×` | `1 / 136` | `0` | `0 B` |

All saved top-64 golden logits matched bit-for-bit across all 285 decode steps, plus all 48 prefill & `decode_0` visit tensors.

### 2. Resident `forge` C1 Decode Roofline (`short_math`, `--e2e --step 1`)

Logical C1 weight traffic per token: `3,347,790,336 B` (active controllers in Pass 1–2 only, mean expert bitrate, unpadded `200,058`-row LM head; `3.487 ms` floor @ 960 GB/s).

| Mode | Median (ms/tok) | Throughput (tok/s) | Effective BW (GB/s) | % of 960 GB/s SOL | Unload Loss |
|---|---:|---:|---:|---:|---:|
| **CUDA Graph** | **`40.883`** | **`24.46`** | `81.89` | `8.53%` | `0 B` |
| **Eager (`--no-graph`)** | `50.744` | `19.71` | `65.97` | `6.87%` | `0 B` |

Interleaved `forge ab baseline phase1-baseline --e2e --step 1` (5 A-B-A-B rounds): `40.892 ms` vs. `40.887 ms` (`1.0001×`, exact parity, `0 B` unload loss).

### 3. 50-Stage Eager C1 Decode Breakdown (`forge profile --e2e --step 1`, `50.780 ms`)

| Phase | Visits | Measured Time (ms) | Share of Total | Logical SOL % |
|---|---|---:|---:|---:|
| Embedding (`embed_rmsnorm`) | — | `0.006` | `0.01%` | `0.16%` |
| Prologue | `0..5` (6) | `5.437` | `10.71%` | `5.54%` |
| Core Pass 0 | `6..17` (12) | `10.876` | `21.42%` | `5.54%` |
| Core Pass 1 | `18..29` (12) | `13.886` | `27.35%` | `4.76%` |
| Core Pass 2 | `30..41` (12) | `13.901` | `27.38%` | `4.76%` |
| Epilogue | `42..47` (6) | `5.433` | `10.70%` | `5.54%` |
| LM Head (`unembed[:200058]` + softcap) | — | `1.230` | `2.42%` | **`78.08%`** |
| **Total** | **48** | **`50.780`** | **100.00%** | **`6.87%`** |

### 4. Single-`.cu` Incremental Hot-Reload SLA (Full Real `forward.so`)

Editing `engine/src/ops/norm_embed.cu` and running `forge run --layers 0..6 --step 1 --case short_math`:

| Breakdown | Comment Edit (Cold Lazy Record) | Revert (Warm Lazy Cache) |
|---|---:|---:|
| Workspace sync (`rsync --checksum --no-times`) | `549.6 ms` | `543.0 ms` |
| Incremental build (`ninja`: `~707 ms` compile + `~43 ms` link) | `893.9 ms` | `885.4 ms` |
| Resident `dlopen` / `dlclose` | `3.1 / 1.5 ms` | `4.8 / 1.8 ms` |
| Lazy record / cache lookup | `2,270.8 ms` (miss) | `0.014 ms` (hit) |
| 6-visit GPU median (`decode_0.step_5` exact, `0 B` loss) | `4.576 ms` | `4.570 ms` |
| **Total external wall clock** | **`3.983 s`** | **`1.711 s`** |

## Key Takeaway for Phase 3 Optimization

The Phase 1 Roofline pinpoints the exact optimization ladder once Phase 2 serving lands:
1. **LM Head is already near SOL (`1.23 ms`, 78.1% of 960 GB/s)** via cuBLAS BF16 GEMM.
2. **97.6% of C1 decode latency sits inside the 48 visits (`~0.91 ms/visit` in Prologue/Pass 0/Epilogue, `~1.16 ms/visit` in Pass 1–2 with controllers)** — dominated by the correctness-first unoptimized EXL3 decode + WHT + separate attention/controller kernel launches (~5.5% SOL).
3. Phase 3 (`dspark` + fused `sm_120a` in-register trellis GEMV/GEMM + even/odd expert `weights repack` + fused attention/controller kernels) has **>10× headroom** inside the 48 visits while guarded by this exact bit-verified Phase 1 baseline.

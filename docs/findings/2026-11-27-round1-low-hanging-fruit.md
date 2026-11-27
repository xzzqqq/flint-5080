# Round 1 low-hanging-fruit optimization — DSpark and AR on RTX 5080

**Date** 2026-11-27  
**Base revision** `cd190a75a8a9df250bebdb6b3a731f518b66aeee` (working tree with Round 1 optimizations)  
**Hardware** One RTX 5080 on `quant3`, 16,303 MiB, `sm_120a`, 960 GB/s, driver 580.105.08, 3,090/15,001 MHz, 360 W limit  
**Software** CUDA 13.0.88, Rust 1.99.1, CMake 3.22.1, Ninja 1.10.1. Inkling 2 Mini (`Maxrio/Inkling-2-Mini-exl3`, 3.3 bpw)  
**HTTP configuration** Greedy C1, `remote_loopback` at `127.0.0.1:8080`, `metrics_version=2`, engine ABI 2, 256-token prefill chunks, 3 prefix-cache slots, CUDA graphs enabled. Auto-sized context: **41,265 tokens** (DSpark, +910 vs Phase 3) and **67,176 tokens** (`--no-spec`).

---

## 1. Iterative `forge` progression (`short_math`, RTX 5080)

Each optimization step was screened rapidly on the resident `forge` daemon before running full acceptance at the end of the round.

| Step | Saved `forge` candidate | Change summary | DSpark step 1 (ms) | Draft / Verify (ms) | AR decode step 1 (ms) | Prefill 27 tok (ms) |
|---|---|---|---:|---:|---:|---:|
| 0 | `baseline` / `round1-exl3-before` | Frozen Phase 3 baseline | 106.86 | 11.39 / 95.34 | 40.90 (8.53% SOL) | 155.83 |
| 1 | `round1-exl3-fwht` | Fused on-the-fly EXL3 dense matmul (`k2 ∈ {6,7,12}`, `ExpandTile`) + 7-stage butterfly FWHT128 | 83.30 | 11.39 / 71.75 | 17.35 (20.10% SOL) | 140.76 |
| 2 | `round1-exl3-moe` | Unified `GroupedGemm` & `IndexedGemv` onto `ExpandTile` with active-row specialization & dual half-warp `m=1` | 67.62 | 11.39 / 56.07 | 15.25 (22.87% SOL) | 121.24 |
| 3 | `round1-dspark-draft` | Folded `unembed @ hidden_out` + `hidden_bias @ unembed^T` in `DsparkRunner` scratch, 256-block `Head` argmax, zero-sync `DraftGreedy` | 58.92 | **2.94** / 55.82 | 15.05 | 121.25 |
| 4 | `round1-flash-attn` | Batched read-before-write GQA Split-KV FlashDecoding for Target & DSpark Draft (1,026-tok DSpark: 137.81 → **55.55 ms**, 2.48×) | 53.83 | 2.91 / 50.77 | 14.57 | 106.65 |
| 5 | `round1-fused-linear-res` | Single-kernel query-major `BlockResidual`, removed 32-row `RunParityRowMatmul` padding & `RunFp32Linear` BF16→FP32 weight casts, direct BF16 `lm_head` | 42.56 | 2.90 / 39.52 | 12.43 | 75.27 |
| 6 | `round1-cuda-graphs` / `round1-causal-verify` | CUDA Graph capture/replay for DSpark Verify (`T=8`) & Draft (`Greedy`), low-register `GroupedGemmSmall<4>` (48 regs/thread, 4-way Split-K), plus causal Verify split-geometry & grouped-MoE fix | **33.03** | **2.74 / 30.13** | **12.38** (28.2% SOL) | 76.17 (103.73 after causal repair) |

---

## 2. End-to-end HTTP `bench` vs frozen Phase 3 baseline (`short` suite)

Paired geometric mean speedups (95% bootstrap CI over 3 cases, fresh server per suite, 1 warmup + 3 measured rounds):

- **DSpark (Phase 3 → Round 1)**:
  - `server_tpot_ms` (ms/round): **3.7875× faster** (`cand/base = 0.2640 [0.2039, 0.3194]`)
  - `server_decode_tok_s`: **3.6288× faster** (`[2.9315, 4.9124]`)
  - `prefill_ms`: **2.0631× faster** (`cand/base = 0.4847 [0.3160, 0.6621]`)
  - `client_e2e_ms`: **2.7886× faster** (`cand/base = 0.3586 [0.2941, 0.4018]`)
- **AR `--no-spec` (Phase 3 → Round 1)**:
  - `server_tpot_ms` / `server_decode_tok_s`: **3.7848× faster** (`[3.3920, 4.4148]`)
  - `prefill_ms`: **2.0617× faster**
  - `client_e2e_ms`: **3.0528× faster**
- **Round 1 AR → DSpark**: **1.4896×** decode throughput (`[1.3398, 1.8302]`).

| Case | Mode | Prompt | Phase 3 ms/round | Round 1 ms/round | Speedup | Phase 3 tok/s | Round 1 tok/s | Round 1 prefill (ms) | Round 1 E2E (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `short_chat` | AR | 63 | 42.113 | **12.416** | 3.39× | 23.75 | **80.54** | 201.70 | 984.98 |
| `short_chat` | DSpark | 63 | 114.095 | **36.445** | 3.13× | 36.81 | **107.91** | 201.18 | 786.91 |
| `short_debug` | AR | 273 | 45.899 | **12.678** | 3.62× | 21.79 | **78.88** | 749.46 | 1,549.51 |
| `short_debug` | DSpark | 273 | 131.079 | **37.031** | 3.54× | 32.04 | **106.32** | 749.59 | 1,344.10 |
| `short_review` | AR | 1091 | 58.846 | **13.329** | 4.41× | 16.99 | **75.02** | 2,940.42 | 3,782.30 |
| `short_review` | DSpark | 1091 | 187.500 | **38.240** | **4.90×** | 27.95 | **137.31** | 2,937.96 | 3,398.30 |

Context-matched SOL now sits at **~28–29% for AR** (`short_chat` 12.42 ms vs 3.58 ms floor) and **~30–34% for DSpark** (`short_chat` 36.45 ms vs ~10.5–12.5 ms floor, up from ~9.5%).

---

## 3. Numerical & acceptance verification

- **Teacher-forced KLD (`weights kld`)** vs pre-round baseline (`round1-exl3-before.so`, 288 positions across `short_math`, `short_sky`, `short_code`):
  - Mean KL: **0.001100**, p95 KL: **0.004722**, top-1 agreement: **288/288 (100%)**, unload loss: **0 B**.
- **Causality & exactness checks**:
  - Greedy AR, DSpark, and independent teacher-forced `T=8` forward agree on **288/288** golden tokens.
  - Eager vs CUDA Graph logits match **bitwise (`max_abs = 0`)**; 48 single-visit staged ranges match bitwise.
  - Wraparound SWA & SConv rollback match direct evaluation **byte-for-byte** for `0, 3, 8`, including rejected-draft suffix isolation.
  - All EXL3 codebook SHA-256 digests remain exact.
- **Full acceptance suites**:
  - `dev check --local`: **PASS** (166 pytest, 29 cargo tests, `clang-format`, `cargo fmt`, `ruff`, `clippy`, hygiene).
  - `dev check --gpu`: **PASS** (9/9 CTest, `ops-golden`, `e2e-golden`, `runtime-golden`, `mtp-golden`, `forge-host`, `server`, chat goldens, 0 B VRAM teardown).
  - HTTP `canary`: **6/6 PASS** (up to 163.7 tok/s on `tool_call_single`).
  - Final GPU state: **0 / 16,303 MiB** VRAM, no compute processes, free GPU lease.

---

## 4. Remaining bottleneck profile & Round 2 direction

Post-Step-6 `nsys` (`32.06 ms` GPU kernel time + `0.97 ms` CPU overhead = `33.03 ms` DSpark round):

- **`GroupedGemmSmall<4>` (MoE EXL3)**: **16.20 ms (50.4% of GPU time)** vs ~6.5 ms HBM floor.
- **`DenseMatmul<12, 8>` (Attention EXL3)**: **4.39 ms (13.7% of GPU time)** vs ~1.0 ms HBM floor.
- **Dense BF16 GEMMs (`lm_head`, Draft `base`, folded Markov head)**: **2.86 ms** — already **~94% SOL (~900 GB/s)**.
- **All remaining ops combined** (`Wht`, `BlockResidual`, `Conv`, `Norm`, `Route`, `GroupTokens`, `Flash`, `Clip`, `Accum`): **~8.6 ms** across 15+ ops.

**64.1% of GPU time is now pure EXL3 dequant + matmul.** Reaching the ≥70% DSpark round SOL target (~15–18 ms/round) in Round 2 centers on custom `sm_120a` Tensor Core (`mma.sync` / `tcgen05`) EXL3 kernels with offline weight repacking (`weights repack`) and fused WHT/epilogue, plus low-precision KV (FP8 / NVFP4) for 128K context capacity.

---

## 5. Run archives (`.runs/`)

| Run | Archive |
|---|---|
| Teacher-forced KLD vs `round1-exl3-before.so` | `20261126T180353.524306Z-weights-kld-6fb71b54.json` |
| Local acceptance (`dev check --local`) | `20261126T181340.122207Z-dev-check-11fcfd83.json` |
| GPU acceptance (`dev check --gpu`) | `20261126T181515.738232Z-dev-check-75aa2585.json` |
| HTTP DSpark `canary` | `20261126T181600.455518Z-bench-run-0480fd41.json` |
| HTTP DSpark `short` | `20261126T181628.567174Z-bench-run-b3af1788.json` |
| HTTP AR `short` | `20261126T181659.596810Z-bench-run-867d5971.json` |
| Compare: Phase 3 DSpark → Round 1 DSpark | `20261126T181728.648484Z-bench-compare-0d77b556.json` |
| Compare: Phase 3 AR → Round 1 AR | `20261126T181728.715904Z-bench-compare-83dba469.json` |
| Compare: Round 1 AR → Round 1 DSpark | `20261126T181728.779118Z-bench-compare-200da05a.json` |

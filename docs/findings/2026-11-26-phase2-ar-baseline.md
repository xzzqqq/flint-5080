# Phase 2 Closing Baseline: End-to-End HTTP Serving (`canary` & `short`)

**Date**: 2026-11-26  
**Revision**: `7af8e39294b2633d7aab37cc98b448d1b06fa568` (clean tree)  
**Hardware / Software**: Single RTX 5080 (`16,303 MiB`, `sm_120a`, 960 GB/s GDDR7), NVIDIA driver `580.105.08`, CUDA 13.0 (`V13.0.88`), max clocks `3090 MHz` graphics / `15001 MHz` mem, power limit `360 W`, idle/post-run GPU temp `30 °C` → `41 °C`, Rust `1.99.1`, CMake `3.22.1` + Ninja `1.10.1`  
**Checkpoint**: `Maxrio/Inkling-2-Mini-exl3` (`3.3bpw`, 11.57 GiB text weights, 48 logical visits, AR-only)  
**Server Config**: `max_context = 67176` (auto-sized), `prefill_chunk_size = 256`, `prefix_cache_slots = 3`, `use_cuda_graph = true`, `request_timeout_sec = 1800`, bound on `127.0.0.1:8080` (`quant3`)  
**Client Location**: Mac laptop over SSH local port forward (`ssh -L 18080:127.0.0.1:8080 quant3`); primary timings come from server-side `x_flint_telemetry`  
**Run Archives**:
- `dev check --local`: `.runs/20261126T104201.177034Z-dev-check-148199ce.json`
- `dev check --gpu`: `.runs/20261126T104616.697437Z-dev-check-fd86193c.json`
- `bench run --suite canary`: `.runs/20261126T105136.914093Z-bench-run-22b63d52.json` (`workload_sha256`: `8955c7cf66370949d32d80b8369c1f6389a13cd3ee88e67021f5ca1ba4038243`)
- `bench run --suite short`: `.runs/20261126T110109.601373Z-bench-run-50ca874a.json` (`workload_sha256`: `a5d139bf5b7313668bf8681ef024691b528f521cf9eeb4dec7cd585f2c36c7cd`)
- Post-`short` server status: `.runs/20261126T110135.794493Z-dev-serve-78feca37.json`

## Summary

This report freezes the **Phase 2 AR HTTP serving baseline** on the RTX 5080 for `canary` and `short` (M2.6 short-context scope). The `agent` suite (4 SWE-bench Verified snapshots totaling 93.1K prompt tokens, max 47.1K) is intentionally deferred until Phase 3 groups prefill MoE experts, because the current correctness-first prefill path runs at `~11–12 tok/s` (projected ~2.4 h per single pass / ~10–12 h for default 1+3 rounds).

- **Full Phase 2 acceptance green**: `dev check --local` (136 pytest, 22 cargo tests, format/lint/hygiene) and `dev check --gpu` (8/8 CTest, `ops-golden`, `e2e-golden`, `runtime-golden`, `forge-host`, `server` smoke & chat golden) all passed.
- **Canary 6/6 passed (`61.8 s`)**: arithmetic reasoning, Python bugfix, single & multi-tool calls (`tool_call_multi` emitted 2 valid calls), stop-sequence truncation, and exact streaming vs. non-streaming UTF-8 parity (`utf8_stream_parity` hit 100% VRAM Prefix Cache on the second request, cutting prefill from `3,033.4 ms` to `1.08 ms`).
- **Frozen `short` baseline (`535.0 s`, empty cache, 1 warmup + 3 rounds)**:
  - `short_chat` (59 prompt tok): **`42.06 ms/round`** (`23.78 tok/s`, **8.44%** of 3.550 ms AR floor / **8.48%** context-matched SOL).
  - `short_debug` (269 prompt tok): **`45.87 ms/round`** (`21.80 tok/s`, **7.74%** of 3.550 ms floor / **7.87%** context-matched SOL).
  - `short_review` (1,087 prompt tok): **`58.98 ms/round`** (`16.95 tok/s`, **6.02%** of 3.550 ms floor / **6.38%** context-matched SOL).
- **Zero recapture & tight within-run stability**: Across all 12 `short` requests (warmup + 3 rounds × 3 cases), the engine executed **1 static CUDA Graph capture** and **756 graph launches** (`12 × 63`), `0` prefix cache hits / `12` misses (confirming cold nonce isolation), and `<0.3%` round-to-round `server_tpot_ms` spread.

## Measurements

### 1. Frozen `short` Suite (Clean Server Cache, 1 Warmup + 3 Measured Rounds)

All three cases emit `64` completion tokens (`63` decode rounds; first token emitted from prefill) with `cached_prefix_ratio = 0.0`. Context-matched AR floor uses the `docs/model.md` short-context weight traffic (`3,407.9 MB`, `3.550 ms` @ 960 GB/s) plus BF16 KV traffic at mid-decode length `L_mid = prompt_tokens + 32` (`40 × min(L_mid, 1024) × 4 KiB` SWA + `8 × L_mid × 4 KiB` Full).

| Case | Prompt / Output (tok) | Prefill Median (ms) | Prefill Rate (tok/s) | TTFT Median (ms) | TPOT Median (ms/round) | TPOT Rounds (ms) | Decode Rate (tok/s) | Context-Matched Floor (ms) | SOL % (3.550 ms / Matched) | Client ITL p50 / p95 (ms) | E2E Median (ms) | HTTP Overhead (ms) |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `short_chat` | 59 / 64 | 4,921.07 | 11.99 | 5,131.30 | **42.059** | 41.952, 42.059, 42.063 | 23.78 | 3.569 | 8.44% / **8.48%** | 42.16 / 44.11 | 7,684.72 | 113.97 |
| `short_debug` | 269 / 64 | 22,823.32 | 11.79 | 23,031.15 | **45.870** | 45.867, 45.870, 45.891 | 21.80 | 3.612 | 7.74% / **7.87%** | 45.95 / 48.67 | 25,833.50 | 117.43 |
| `short_review` | 1,087 / 64 | 96,298.30 | 11.29 | 96,536.12 | **58.984** | 58.984, 58.993, 58.984 | 16.95 | 3.763 | 6.02% / **6.38%** | 58.99 / 60.52 | 100,135.95 | 121.08 |
| **Suite Pooled (`n=9`)** | **269 / 64** | **22,823.32** | — | **23,031.15** | **45.870** | — | **21.80** | — | — | **45.95 / 48.67** | **25,833.50** | **117.43** |

*Notes*:
- `server_decode_tok_s` is `(decode_tokens - 1) / (decode_ms / 1000)`. It excludes the first token emitted during prefill, leaving 63 timed output tokens in each `short` request. `server_tpot_ms` is `decode_ms / decode_rounds` (`decode_ms / 63`).
- `http_overhead_ms` (`114–121 ms`) includes SSH tunnel LAN round-trip from the Mac client plus HTTP/SSE framing.

### 2. Canary Suite (6 Cases, 7 Requests, 100% Pass)

| Case | Mode | Prompt / Output (tok) | Cached Prefix | Prefill (ms) | TPOT (ms/round) | Decode (tok/s) | Finish Reason | Tool Calls | Result |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| `math_reasoning` | stream | 27 / 50 | 0 / 27 | 2,401.14 | 41.654 | 23.53 | `stop` | 0 | PASS |
| `code_bugfix` | stream | 58 / 67 | 0 / 58 | 4,838.17 | 41.961 | 23.48 | `stop` | 0 | PASS |
| `tool_call_single` | stream | 106 / 61 | 0 / 106 | 8,884.10 | 42.805 | 22.98 | `tool_calls` | 1 | PASS |
| `tool_call_multi` | stream | 121 / 95 | 0 / 121 | 10,174.98 | 43.432 | 22.78 | `tool_calls` | 2 | PASS |
| `stop_sequence` | stream | 50 / 118 | 0 / 50 | 4,198.89 | 42.412 | 23.38 | `stop` | 0 | PASS |
| `utf8_stream_parity` | stream | 36 / 123 | 0 / 36 | 3,033.40 | 42.198 | 23.51 | `stop` | 0 | PASS |
| `utf8_stream_parity` | non-stream | 36 / 123 | **36 / 36** | **1.08** | 42.194 | 23.51 | `stop` | 0 | PASS (exact parity) |

### 3. Engine Runtime Telemetry (Post-`short` Run)

- `submitted_requests = 12`, `completed_requests = 12`, `cancelled_requests = 0`
- `graph_captures = 1`, `graph_launches = 756` (zero recapture across 59 → 1,087 prompt tokens)
- `prefix_cache_hits = 0`, `prefix_cache_misses = 12`, `prefix_cache_used_slots = 1 / 3`
- `vram_free_bytes`: `522,059,776 B` (post-startup) → `503,185,408 B` (post-graph capture, `~18.0 MiB` CUDA Graph pool)
- Clean teardown verified: `0 / 16,303 MiB` VRAM used, `GPU lease: free`

## Key Phase 3 Engineering Insights

1. **Prefill MoE is unbatched across tokens (~11.3–12.0 tok/s, `83.4–88.6 ms/tok`)**:  
   In `engine/src/ops/submodules.cpp` (`RunSparseMoeBlock`, `t > 1`), prefill loops over all `t × 8` token-expert pairs and invokes `RunExl3Linear(..., rows=1)` per pair, decoding each expert trellis from scratch on every token. At `prefill_chunk_size = 256`, grouping assignments by expert (`0..127`, ~16 tokens/expert) will amortize trellis decoding by ~15× and unblock freezing the long-context `agent` suite (`8.5K–47.1K` tokens).
2. **Decode attention scales at `+16.5 µs/tok` across 48 visits (~85× above KV bandwidth floor)**:  
   Between `short_chat` (59 tok) and `short_review` (1,087 tok), `server_tpot_ms` increases by **`+16.93 ms`** (`42.06 → 58.98 ms`) for `+1,028` context tokens, whereas the theoretical BF16 KV bandwidth cost for that context delta is only **`0.194 ms`** (`0.19 µs/tok`). Replacing the Phase 1 reference `flint_attention_forward` kernel with a warp/FlashDecoding-style tiled attention kernel (and eventually FP8 KV) is required alongside EXL3 GEMV fusion before `agent` long-context decode can approach SOL.
3. **Baseline decode matches Phase 1 `forge` roofline**:  
   At short context (`short_chat`, 59 tok), end-to-end HTTP `server_tpot_ms = 42.06 ms` closely tracks Phase 1 `e2e_tests` (`41.24–44.37 ms`), confirming negligible Rust/FFI scheduling overhead on the C1 decode hot path (`client_itl_p50 = 42.16 ms`).

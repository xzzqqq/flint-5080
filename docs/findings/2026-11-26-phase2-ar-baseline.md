# Phase 2 Closing Baseline: End-to-End HTTP Serving (`canary` & `short`)

**Date**: 2026-11-26  
**Revision**: `fb90de81b0abc56b3d7ac38540e0ab84ce839264` (clean tree; includes grouped EXL3 MoE prefill `50406f5`, server fixes `7cb3ec3`, and `metrics_version=2` / `remote_loopback` bench fixes `fb90de8`)  
**Hardware / Software**: Single RTX 5080 (`16,303 MiB`, `sm_120a`, 960 GB/s GDDR7), NVIDIA driver `580.105.08`, CUDA 13.0 (`V13.0.88`), max clocks `3090 MHz` graphics / `15001 MHz` mem, power limit `360 W`, idle/post-run GPU temp `30 °C` → `40 °C`, Rust `1.99.1`, CMake `3.22.1` + Ninja `1.10.1`  
**Checkpoint**: `Maxrio/Inkling-2-Mini-exl3` (`3.3bpw`, 11.57 GiB text weights, 48 logical visits, AR-only)  
**Server Config**: `max_context = 67176` (auto-sized), `prefill_chunk_size = 256`, `prefix_cache_slots = 3`, `use_cuda_graph = true`, `request_timeout_sec = 1800`, bound on `127.0.0.1:8080` (`quant3`)  
**Client Location**: `remote_loopback` (dispatched by `./bin/bench run` onto `quant3` loopback via `/home/xzq/.cache/flint-5080/bench-venv/bin/python`, Python `3.12.15`, `httpx 0.28.1`); `metrics_version = 2`  
**Run Archives**:
- `dev check --local` (`fb90de8`): `.runs/20261126T131035.692224Z-dev-check-fc628245.json`
- `dev check --gpu` (engine `50406f5`, unchanged in `fb90de8`): `.runs/20261126T112035.794863Z-dev-check-cbe6930e.json`
- `bench run --suite canary`: `.runs/20261126T131130.756081Z-bench-run-eb72ef27.json` (`workload_sha256`: `8955c7cf66370949d32d80b8369c1f6389a13cd3ee88e67021f5ca1ba4038243`)
- `bench run --suite short` (**Frozen Baseline, Run 1**): `.runs/20261126T131326.144647Z-bench-run-6d4c9ca7.json` (`workload_sha256`: `a5d139bf5b7313668bf8681ef024691b528f521cf9eeb4dec7cd585f2c36c7cd`)
- Post-Run 1 `dev serve status`: `.runs/20261126T131326.589922Z-dev-serve-7c8e60ae.json`
- `bench run --suite short` (Repeatability, Run 2): `.runs/20261126T131453.370677Z-bench-run-e501721c.json`
- `bench compare` (Run 1 vs. Run 2): `.runs/20261126T131454.207227Z-bench-compare-b077c2a3.json`
- Post-teardown `dev status`: `.runs/20261126T131457.340316Z-dev-status-a551c4e9.json`

## Summary

This report freezes the **Phase 2 AR HTTP serving baseline** on the RTX 5080 for `canary` and `short` under `metrics_version = 2` and `client_location = remote_loopback` (eliminating SSH tunnel overhead and fixing per-case cold-cache isolation and decode-rate accounting). The `agent` suite (4 SWE-bench Verified snapshots spanning 10.8K–50K+ tokens) remains deferred until Phase 3 optimizes the reference attention kernel: although grouped MoE prefill (`50406f5`, see [2026-11-26-grouped-moe-prefill.md](2026-11-26-grouped-moe-prefill.md)) accelerated short-context prefill by `10.5–16.9×`, an exploratory probe on `9c002db` confirmed that scalar $O(L^2)$ attention prefill still dominates long contexts (`204.4 s` at `10.8K`, `415.4 s` at `16.9K`, `1,198.2 s` at `27.3K`).

- **Acceptance gates green**: `dev check --local` (142 pytest, 24 cargo tests, format/lint/hygiene) passed on `fb90de8`; `dev check --gpu` (8/8 CTest, `ops-golden`, `e2e-golden`, `runtime-golden`, `forge-host`, `server` smoke & chat golden) passed on the identical engine (`50406f5`).
- **Canary 6/6 passed (`30.56 s`)**: arithmetic reasoning, Python bugfix, single & multi-tool calls (`tool_call_multi` emitted 2 valid calls with per-request IDs), stop-sequence truncation, and exact streaming vs. non-streaming UTF-8 parity (`utf8_stream_parity` hit 100% VRAM Prefix Cache on the second request, cutting prefill from `220.37 ms` to `0.40 ms`).
- **Frozen `short` baseline (`81.77 s`, empty cache, 1 warmup + 3 rounds, `remote_loopback`)**:
  - `short_chat` (63 prompt tok): prefill **`304.23 ms`** (`207.1 tok/s`), decode **`42.017 ms/round`** (`23.80 tok/s`, **8.45%** of 3.550 ms AR floor / **8.49%** context-matched SOL), HTTP overhead **`1.28 ms`**.
  - `short_debug` (273 prompt tok): prefill **`1,375.34 ms`** (`198.5 tok/s`), decode **`45.805 ms/round`** (`21.83 tok/s`, **7.75%** of 3.550 ms floor / **7.89%** context-matched SOL), HTTP overhead **`1.34 ms`**.
  - `short_review` (1,091 prompt tok): prefill **`9,288.42 ms`** (`117.5 tok/s`), decode **`58.839 ms/round`** (`17.00 tok/s`, **6.03%** of 3.550 ms floor / **6.40%** context-matched SOL), HTTP overhead **`2.39 ms`**.
- **Run-to-run repeatability verified across clean server restarts**: Paired `bench compare` between Run 1 and Run 2 reported **0 warnings** and `<0.06%` geometric-mean drift (`server_tpot_ms` ratio **`1.0006`**, 95% CI `[0.9999, 1.0010]`; `prefill_ms` ratio **`1.0004`**, 95% CI `[1.0002, 1.0007]`). This confirms the proposed 5% Phase 3 regression threshold is well above measurement noise.
- **Zero recapture & exact cold isolation**: Across all 12 `short` requests (warmup + 3 rounds × 3 cases), the engine executed **1 static CUDA Graph capture** and **756 graph launches** (`12 × 63`), `0` prefix cache hits / `12` misses.

## Measurements

### 1. Frozen `short` Suite — Run 1 (Clean Server Cache, 1 Warmup + 3 Measured Rounds)

All three cases emit `64` completion tokens (`63` timed decode rounds; first token emitted during prefill) with `cached_prefix_ratio = 0.0`. Cold nonce tags include `<case-id>` (`[bench-short-<case-id>-r01] `), increasing prompt token counts by 4 tokens relative to the pre-`fb90de8` format. Context-matched AR floor uses the `docs/model.md` short-context weight traffic (`3,407.9 MB`, `3.550 ms` @ 960 GB/s) plus BF16 KV traffic at mid-decode length `L_mid = prompt_tokens + 32` (`40 × min(L_mid, 1024) × 4 KiB` SWA + `8 × L_mid × 4 KiB` Full).

| Case | Prompt / Output (tok) | Prefill Median (ms) | Prefill Rate (tok/s) | TTFT Median (ms) | TPOT Median (ms/round) | TPOT Rounds (ms) | Decode Rate (tok/s) | Context-Matched Floor (ms) | SOL % (3.550 ms / Matched) | Client ITL p50 / p95 (ms) | E2E Median (ms) | HTTP Overhead (ms) |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `short_chat` | 63 / 64 | 304.23 | 207.08 | 388.35 | **42.017** | 42.016, 42.019, 42.017 | 23.80 | 3.569 | 8.45% / **8.49%** | 42.06 / 42.54 | 2,952.56 | 1.28 |
| `short_debug` | 273 / 64 | 1,375.34 | 198.50 | 1,467.18 | **45.805** | 45.803, 45.805, 45.806 | 21.83 | 3.612 | 7.75% / **7.89%** | 45.83 / 46.28 | 4,262.28 | 1.34 |
| `short_review` | 1,091 / 64 | 9,288.42 | 117.46 | 9,408.18 | **58.839** | 58.840, 58.836, 58.839 | 17.00 | 3.763 | 6.03% / **6.40%** | 58.85 / 58.94 | 12,997.70 | 2.39 |
| **Suite Pooled (`n=9`)** | **273 / 64** | **1,375.34** | — | **1,467.18** | **45.805** | — | **21.83** | — | — | **45.83 / 46.28** | **4,262.28** | **1.34** |

*Notes*:
- Under `metrics_version = 2`, `server_decode_tok_s` is `(decode_tokens - 1) / (decode_ms / 1000)` (`63 / decode_s`), matching `1000 / server_tpot_ms` for AR decode (`decode_rounds = 63`).
- Running on `remote_loopback` reduced `http_overhead_ms` from `114–121 ms` (SSH tunnel) to `1.28–2.39 ms`, with `client_itl_p95` within `0.1–0.5 ms` of `server_tpot_ms`.

### 2. Repeatability Check: Clean-Restart Run 1 vs. Run 2 (`bench compare`)

Server was stopped and restarted between Run 1 (`6d4c9ca7`) and Run 2 (`e501721c`) to reset Prefix Cache and CUDA Graph state. `bench compare` reported **`warnings = []`**:

| Metric | Pairs | Run 2 / Run 1 (geomean) | Run 1 / Run 2 (geomean) | 95% Bootstrap CI (Run 2 / Run 1) |
|---|---:|---:|---:|---|
| `prefill_ms` | 3 | `1.0004` | `0.9996` | `[1.0002, 1.0007]` |
| `client_ttft_ms` | 3 | `1.0004` | `0.9996` | `[1.0002, 1.0007]` |
| `server_tpot_ms` | 3 | `1.0006` | `0.9994` | `[0.9999, 1.0010]` |
| `server_decode_tok_s` | 3 | `0.9994` | `1.0006` | `[0.9990, 1.0001]` |
| `client_e2e_ms` | 3 | `1.0006` | `0.9994` | `[1.0001, 1.0008]` |

### 3. Canary Suite (6 Cases, 7 Requests, 100% Pass)

| Case | Mode | Prompt / Output (tok) | Cached Prefix | Prefill (ms) | TPOT (ms/round) | Decode (tok/s) | HTTP Overhead (ms) | Finish Reason | Tool Calls | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `math_reasoning` | stream | 27 / 50 | 0 / 27 | 302.11 | 41.559 | 23.58 | 2.16 | `stop` | 0 | PASS |
| `code_bugfix` | stream | 58 / 67 | 0 / 58 | 287.75 | 41.958 | 23.48 | 1.54 | `stop` | 0 | PASS |
| `tool_call_single` | stream | 106 / 61 | 0 / 106 | 501.84 | 42.803 | 22.98 | 1.41 | `tool_calls` | 1 | PASS |
| `tool_call_multi` | stream | 121 / 95 | 0 / 121 | 640.34 | 43.372 | 22.81 | 1.80 | `tool_calls` | 2 | PASS |
| `stop_sequence` | stream | 50 / 118 | 0 / 50 | 252.27 | 42.288 | 23.45 | 1.76 | `stop` | 0 | PASS |
| `utf8_stream_parity` | stream | 36 / 123 | 0 / 36 | 220.37 | 42.071 | 23.58 | 1.35 | `stop` | 0 | PASS |
| `utf8_stream_parity` | non-stream | 36 / 123 | **36 / 36** | **0.40** | 42.074 | 23.57 | 1.46 | `stop` | 0 | PASS (exact parity) |

### 4. Engine Runtime Telemetry (Post-Run 1 `short`)

- `submitted_requests = 12`, `completed_requests = 12`, `cancelled_requests = 0`
- `graph_captures = 1`, `graph_launches = 756` (zero recapture across 63 → 1,091 prompt tokens)
- `prefix_cache_hits = 0`, `prefix_cache_misses = 12`, `prefix_cache_used_slots = 1 / 3`
- `vram_free_bytes`: `522,059,776 B` (post-startup) → `503,185,408 B` (post-graph capture, `~18.0 MiB` CUDA Graph pool)
- Clean teardown verified: `0 / 16,303 MiB` VRAM used, `40 °C`, `GPU lease: free`

### 5. Exploratory `agent` Probe (Diagnostic Only — Why `agent` Awaits Attention Optimization)

An exploratory `--warmup 0 --rounds 1` probe on `9c002db` (identical engine `50406f5`) completed 3/4 cases before being stopped during case 4 (`~50K+` tokens, >32 min elapsed):

| Case | Prompt / Output (tok) | Cached (tok) | Prefill (s) | Prefill Rate (tok/s) | Decode (s) | Approx TPOT (ms) | Finish |
|---|---:|---:|---:|---:|---:|---:|---|
| `django__django-11099` | 10,799 / 54 | 0 | 204.4 | 52.8 | 4.74 | ~89.5 | `ToolCalls` |
| `pydata__xarray-4687` | 16,919 / 124 | 0 | 415.4 | 40.7 | 13.09 | ~106.4 | `ToolCalls` |
| `pallets__flask-5014` | 27,304 / 329 | 1,018* | 1,198.2 | 21.9 | 44.91 | ~136.9 | `ToolCalls` |

*\*Captured prior to `fb90de8`, which added `<case-id>` to cold nonce tags to prevent cross-case prefix sharing across shared tool schemas.*

## Key Phase 3 Engineering Insights

1. **Reference `attention.cu` is now the dominant bottleneck for both long prefill and long-context decode**:  
   With MoE prefill grouped (`50406f5`), prefill scales cleanly at `~200 tok/s` up to 273 tokens but degrades quadratically in the 8 Full-Attention layers (`117 tok/s` at 1K → `53 tok/s` at 10.8K → `22 tok/s` at 27.3K) because `submodules.cpp` invokes `flint_attention_forward(..., tokens=1)` per token and `Score`/`Attend` scan keys serially. On decode, 8 Full-Attention layers add `~30 ms/tok` per 10K context (`~90×` above the `0.34 ms` BF16 KV bandwidth floor), and `short_chat` → `short_review` (`+1,028` tokens) already adds `+16.82 ms` (`~85×` above KV bandwidth). Implementing tiled FlashAttention (prefill) and FlashDecoding (decode) for Inkling's learned relative bias + sink + 3:1 GQA will immediately unblock freezing the full `agent` suite.
2. **Short-context decode remains EXL3 GEMV / non-expert projection bound (~8.5% SOL)**:  
   At 63 prompt tokens (`short_chat`), end-to-end HTTP `server_tpot_ms = 42.02 ms` closely matches Phase 1 `forge` (`40.88 ms`), confirming negligible Rust/HTTP/FFI overhead (`http_overhead_ms = 1.28 ms`).
3. **High measurement precision**:  
   With `remote_loopback`, both within-run and across-restart repeatability sit below `0.1%`, giving Phase 3 optimization rounds high statistical power even at 3 measured rounds per case.

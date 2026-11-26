# Phase 3 frozen baseline — DSpark and AR HTTP serving

**Date** 2026-11-26  
**Revision** `7fc774077a5c0beae3f5e45c1b6ac9b36854e05f` with M3.4 forge evaluation and tooling changes in the working tree.  
**Hardware** One RTX 5080 on `quant3`, 16,303 MiB, `sm_120a`, 960 GB/s. Driver 580.105.08, maximum graphics/memory clocks 3,090/15,001 MHz, 360 W power limit.  
**Software** CUDA 13.0 (V13.0.88), Rust 1.99.1, CMake 3.22.1, Ninja 1.10.1. Inkling 2 Mini, `Maxrio/Inkling-2-Mini-exl3`, 3.3 bpw.  
**HTTP configuration** Greedy C1, `remote_loopback` at `127.0.0.1:8080`, `metrics_version=2`, engine ABI 2, 256-token prefill chunks, three prefix-cache slots, CUDA graphs enabled. Auto-sized context was 40,355 tokens with DSpark and 67,176 with `--no-spec`.

## Frozen short suite

DSpark delivered **1.5536× [1.4707, 1.6448]** the AR decode throughput (paired geometric mean and 95% bootstrap CI). The corresponding client E2E speedup was **1.2820× [1.1254, 1.4671]**.

Each suite started with a fresh server. Each short case had one warmup and three measured requests. All measured requests emitted 64 tokens with zero cached prefix tokens. The first token came from prefill, leaving 63 tokens for the decode counters. Values below are per-case medians.

| Case | Mode | Prompt tokens | spec_accepted | decode_rounds | Tokens/round | ms/round | server_decode_tok_s | DSpark/AR rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `short_chat` | AR | 63 | 0 | 63 | 1.000 | 42.113 | 23.7455 | — |
| `short_chat` | DSpark | 63 | 48 | 15 | 4.200 | 114.095 | 36.8114 | 1.5502× |
| `short_debug` | AR | 273 | 0 | 63 | 1.000 | 45.899 | 21.7868 | — |
| `short_debug` | DSpark | 273 | 49 | 15 | 4.200 | 131.079 | 32.0416 | 1.4707× |
| `short_review` | AR | 1091 | 0 | 63 | 1.000 | 58.846 | 16.9934 | — |
| `short_review` | DSpark | 1091 | 52 | 12 | 5.250 | 187.500 | 27.9513 | 1.6448× |

Tokens/round is timed output tokens divided by decode rounds. The accepted-token count comes directly from server telemetry. On `short_review`, DSpark took 12, 29 and 11 rounds at 27.95, 11.60 and 30.55 tok/s respectively. The bootstrap resamples the three paired case medians 2,000 times (seed 42). It measures variation across cases.

Including warmups, each mode completed 12 requests with 12 prefix-cache misses. AR captured one CUDA graph and launched it 756 times. DSpark recorded 575 accepted tokens and zero graph launches.

## Phase 3 AR non-regression check

| Case | Phase 2 AR ms/round | Phase 3 AR ms/round | Change | Phase 3 AR context-matched SOL |
|---|---:|---:|---:|---:|
| `short_chat` | 42.017 | 42.113 | +0.230% | 8.48% |
| `short_debug` | 45.805 | 45.899 | +0.206% | 7.87% |
| `short_review` | 58.839 | 58.846 | +0.012% | 6.39% |

Phase 3 / Phase 2 AR decode latency was **1.0015× [1.0001, 1.0023]**. The geometric-mean increase was 0.149%. All three cases passed the 5% regression threshold. Context-matched SOL uses the Phase 2 weight and KV traffic model at prompt length + 32.

Both archived comparisons flagged only server metadata. The within-Phase-3 comparison changes speculative mode and auto-sized context. Phase 2 reported engine ABI 1 and omitted speculative-mode fields. Both AR runs used the same 67,176-token context. Workload hashes, prompt lengths, rounds and client location matched.

## Numerical and functional checks

- `weights kld --json` compared the forge startup snapshot (`baseline`) with `current` through a transient host. Both modules consumed the same golden sequence in eight-row chunks, with graphs disabled. Per-case positions were 49 (`short_math`), 102 (`short_sky`) and 137 (`short_code`). Across all 288 positions, mean, nearest-rank p95 and maximum KL were **0.0**, maximum absolute logit difference was **0.0**, top-1 agreement was **288/288**, and unload loss was **0 B**. RPC evaluation took 9.682 s.
- A second run reused a live model-loaded forge-host, incrementally built `current`, and evaluated all 75 `short_math` positions in three-row chunks. It returned exact agreement, 0 B unload loss, and left the same server running.
- The DSpark canary passed **6/6** cases in 22.73 s. Arithmetic, code, single- and multi-tool calls, stop truncation and UTF-8 stream parity all passed. The multi-tool case returned two calls. The parity replay reused all 36 prompt tokens and matched the streamed response.
- Local acceptance passed 166 Python tests, 25 Rust tests, format, lint and hygiene. The full build and GPU acceptance passed, including 9/9 CTest, operator, end-to-end, runtime and MTP goldens, the real-model KLD RPC and server checks. Greedy DSpark matched AR on 288/288 golden tokens.

## Run archives

All records below are under `.runs/`.

| Run | Archive |
|---|---|
| Local acceptance | `20261126T153232.831411Z-dev-check-985d64a6.json` |
| Full engine and server build | `20261126T153202.203745Z-dev-build-cb418351.json` |
| GPU acceptance | `20261126T153336.521405Z-dev-check-05f1bcc8.json` |
| KLD self-baseline | `20261126T152743.456751Z-weights-kld-d9763373.json` |
| Resident KLD, all math positions | `20261126T153355.630429Z-weights-kld-87c93b1b.json` |
| DSpark canary | `20261126T152810.576671Z-bench-run-813be6b3.json` |
| DSpark short | `20261126T152927.735352Z-bench-run-89bc0e9a.json` |
| Phase 3 AR short | `20261126T153056.588783Z-bench-run-c5a73100.json` |
| Phase 3 AR → DSpark | `20261126T153057.831538Z-bench-compare-2e0126f9.json` |
| Frozen Phase 2 AR short | `20261126T131326.144647Z-bench-run-6d4c9ca7.json` |
| Phase 2 AR → Phase 3 AR | `20261126T153057.894890Z-bench-compare-c2d66912.json` |
| Final GPU status | `20261126T153400.139332Z-dev-status-661ef6f6.json` |

SHA-256 for `canary` was `8955c7cf66370949d32d80b8369c1f6389a13cd3ee88e67021f5ca1ba4038243`. All three `short` archives used `a5d139bf5b7313668bf8681ef024691b528f521cf9eeb4dec7cd585f2c36c7cd`.

Final teardown showed a free GPU lease, no compute processes and **0 / 16,303 MiB** VRAM in use.

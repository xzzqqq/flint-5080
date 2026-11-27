# DFlash 2 baseline on RTX 5080

DFlash 2 delivered **1.262× Round 1 DSpark throughput** and **1.880× Round 1 AR throughput** on the three `short` HTTP cases. The corresponding 95% paired bootstrap intervals were [1.160, 1.351] and [1.564, 2.349]. With three prefix-cache slots, the production server auto-sized to **31,508 tokens**. The archived configurations reached 41,265 with DSpark and 67,176 with AR.

Measured 2026-11-27 on `quant3`, one RTX 5080, 16,303 MiB, driver 580.105.08, CUDA 13.0.88, maximum GPU/memory clocks 3,090/15,001 MHz, 360 W limit. The working tree was based on `6b891ef`. Both checkpoints used their published layouts. The SHA-256 of the EXL3 weight file begins `f38e6e79e0a70741`, and the DFlash 2 weight file begins `217980ab00d11669`.

## HTTP measurements

C1, remote loopback, CUDA Graphs enabled, 256-token prefill chunks, telemetry v2. Each greedy result is the median of three measured requests after one warmup. All measured requests had zero cached prompt tokens and returned 64 tokens. The first token comes from prefill. Throughput and emitted tokens per round use the remaining 63 tokens.

| Greedy case | Prompt tokens | Rounds | Accepted drafts | Accepted/round | Emitted/round | ms/round | Decode tok/s | vs DSpark | vs AR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short_chat | 63 | 12 | 52 | 4.33 | 5.25 | 36.019 | 145.76 | 1.351× | 1.810× |
| short_debug | 273 | 14 | 50 | 3.57 | 4.50 | 36.485 | 123.34 | 1.160× | 1.564× |
| short_review | 1,091 | 10 | 53 | 5.30 | 6.30 | 36.572 | 176.22 | 1.283× | 2.349× |

Compared with DSpark, per-round latency improved by a geometric mean of 1.024× and client end-to-end latency by 1.107×. Most of the decode throughput gain came from emitting more tokens per round. Both comparisons matched the workload SHA-256 (`a5d139bf…c36c7cd`), case IDs, client location, and round counts. The only comparison warning concerned server configuration, including context capacity and, for AR, speculative mode. The bootstrap resamples the three case pairs and measures between-case variation.

The deterministic canary passed **6/6**. After correcting the draft sliding window, the streaming smoke used the three short prompts with nonce round 77, temperature 1.0, seed 5080, and `X-Flint-Telemetry: 1`. Each cold request returned usage, 64 completion tokens, a finish reason of `length`, and `[DONE]`. Replaying each request with the same seed reproduced its content, reasoning, tool calls, accepted count, and round count. The timings below are from the cold request for each case.

| Stochastic case | Rounds | Accepted drafts | Emitted/round | ms/round | Decode tok/s |
|---|---:|---:|---:|---:|---:|
| short_chat | 14 | 50 | 4.5000 | 41.180 | 109.28 |
| short_debug | 11 | 53 | 5.7273 | 41.420 | 138.27 |
| short_review | 16 | 47 | 3.9375 | 40.475 | 97.28 |

## Forge timings and memory

`short_math`, 27 prompt tokens, step 1. Forge resident weights included the external drafter for both modes. Each timed `run` used three warmups and 15 samples. Component medians are computed independently. The native first round accepted five drafts and emitted six tokens.

| Mode | Round median (ms) | Draft (ms) | Verify (ms) | Commit + context KV (ms) |
|---|---:|---:|---:|---:|
| DFlash 2, graph | 36.301030 | 2.320666 | 33.692433 | 0.287264 |
| AR, graph | 12.568736 | — | — | — |
| DFlash 2, stage profiling | 41.403344 | 2.318796 | 38.786049 | 0.286947 |
| AR, stage profiling | 16.229856 | — | — | — |

The profile components describe the last profiled sample. Per-stage CUDA events require eager target verification. The graph run supplies the steady-state comparison.

| Watermark | Bytes | MiB |
|---|---:|---:|
| Resident weight arena | 13,418,483,200 | 12,796.86 |
| Forge KV, 48,830-token capacity | 1,813,774,336 | 1,729.75 |
| Forge scratch | 471,859,200 | 450.00 |
| Forge I/O | 72,466,272 | 69.11 |
| Forge free after warmup | 489,553,920 | 466.88 |
| Production free at readiness | 513,671,168 | 489.88 |

The previous Forge default of 65,536 tokens failed allocation with both checkpoints loaded. Forge now subtracts I/O and the 512 MiB reserve from free memory after loading weights and allocating scratch, then computes default KV capacity. Explicit `--kv-seq-len` remains available. Warming batched target execution alongside decode also removed first-use allocation drift from the AR-only KLD RPC unload check.

For the bandwidth model, a separate diagnostic linked a read-only wrapper around the production router in the golden test. For each batch of eight, it counted distinct expert IDs per visit before the grouped GEMM. The native first round used **717 even and 700 odd expert projection pairs** over 48 visits (4.602 GB of expert weights); the fixture proposal used 646 and 644. Using native routing gives **8.463 GB**, **8.815 ms at 960 GB/s**, and **24.3% logical SOL** at the measured 36.301 ms. Routing instrumentation ran separately from timing. [Weight, KV, and roofline ledger](../model.md)

## Numerical verification

- PyTorch 2.14.1+cu130 generated `dflash2_golden.safetensors` (SHA-256 `190e30e7571f263dcf69258c1f6615e6639043c4daf15c37e275f862d80a258b`). Greedy speculative decoding matched AR **49/49**. It took nine rounds, averaging **5.3333 emitted tokens/round** and **63.4921% proposal acceptance** (40/63 slots). Native runtime also accepted 40 and completed in nine rounds. The output cap can truncate the final batch.
- Native speculative output matched AR **49/49 math, 102/102 sky, 137/137 code**. All ten context KV planes and the block embedding matched their fixed-input fixtures bitwise. Eager and CUDA Graph returned identical draft IDs, candidate IDs and probabilities at both temperatures. Target rollback matched direct execution across SWA wrap for 0, 3 and 8 committed rows.
- Substituting the CUDA operations individually into the PyTorch reference localized the initial divergence to split-KV attention. On identical inputs, Q/K norm and RoPE matched bitwise. Attention differed at 7–22 of 32,768 rounded BF16 outputs per layer, with relative L2 at most 0.00335%. Changing only reference attention to FP64 also perturbed later layers. Final-layer and head relative L2 against dense FP32 were 4.13% and 1.70% for the FP64 path, and 3.93% and 1.78% for the native path. Both produced the same seven greedy proposals on reference taps.
- The golden test checks composed outputs, including sorted unary logits (0.779% relative L2). The composite final-layer and head gates are 5% and 2%, with 1% for the first four layers. A second pass supplies each next layer with its reference input after capturing the actual output. That test keeps the 1% gate on **every** layer; the maximum measured error was 0.199%. On the matched head, all 112 ranked candidate IDs and all unary logits were exact; maximum corrected-score error was 0.00000668. On the freely propagated head, 101/112 candidate ranks matched. Scores with different predecessor/successor IDs describe different edges.
- The first live-target round accepted five tokens while the Python first round accepted four. Replaying the **same proposal** verified the fixture acceptance count and bonus exactly. End-to-end gates still compare the full generated sequence.
- Teacher-forced KLD compared the newly frozen baseline with the same current binary (SHA-256 `ca98f71911f5accf2ddd854d0f2b9152c06e3b740654f09f640483fa0bf0d39a`). Across 288 positions, mean/p95/max KL and maximum logit error were zero, top-1 agreement was 288/288, and unload loss was **0 B**. This checks target replay and module lifecycle. The sparse-sampling tests and streaming smoke exercise the proposal path.

Full remote build and GPU acceptance passed all seven groups, including 9/9 CTest tests. Local acceptance passed 203 Python and 29 Rust tests, formatting, lint and hygiene. After the routing diagnostic released its lease, the GPU reported **0 / 16,303 MiB** used, an empty compute-process list and a free lease.

## Sliding-window verification

With a block starting at position 2,064, the eight queries now admit keys starting at positions 17 through 24 respectively. The CUDA test uses distinct values on the window boundary and checks each output against its analytical mean in both single-split and split-KV attention. Regenerating the short Python fixture produced the same SHA-256 and all 49 greedy tokens. The corrected binary (`5f9438a1dc82…`) matched the frozen target logits at all 288 teacher-forced positions, with zero KL, zero maximum logit difference and zero unload loss. All seven GPU acceptance groups, 207 Python tests and 29 Rust tests passed.

A cold 2,148-token HTTP prompt (the `short_review` user message repeated twice) returned identical content and reasoning over 64 greedy completion tokens with DFlash 2 and AR. On the three standard short cases, decode throughput was 0.9992× the pre-fix DFlash 2 baseline, and emitted tokens per round matched on every case.

## Run records

Files are under `.runs/`. The canary command used `--no-archive`; its JSON stdout was retained for the report.

| Measurement | File |
|---|---|
| DFlash 2 and refreshed target fixtures | `20261127T035119.182136Z-weights-golden-570e8489.json`, `20261127T035732.427818Z-weights-golden-6d4ec0f3.json` |
| Final build and GPU acceptance | `20261127T040237.381485Z-dev-build-190e6693.json`, `20261127T040616.620255Z-dev-check-fd605a08.json` |
| Component substitutions, FP64 comparison and routed experts | `20261127T040731.738316Z-dflash2-numerics-and-routing-7475089a.json` |
| Forge startup | `20261127T040242.361616Z-forge-start-a28ca036.json` |
| DFlash 2 run and profile | `20261127T040251.054874Z-forge-run-05ec4911.json`, `20261127T040252.373816Z-forge-profile-af20e4ba.json` |
| AR run and profile | `20261127T040253.704457Z-forge-run-5b904999.json`, `20261127T040254.900321Z-forge-profile-7696a74b.json` |
| KLD and Forge stop | `20261127T040300.973527Z-weights-kld-76bd71ae.json`, `20261127T040301.930402Z-forge-stop-b596d8f5.json` |
| Production start and stop | `20261127T040315.056302Z-dev-serve-e590b57b.json`, `20261127T040443.163417Z-dev-serve-3bf661f3.json` |
| Canary and greedy short | `phase-b-canary.json`, `20261127T040344.658682Z-bench-run-782a00c7.json` |
| Round 1 DSpark and AR | `20261126T181628.567174Z-bench-run-b3af1788.json`, `20261126T181659.596810Z-bench-run-867d5971.json` |
| Paired comparisons, DSpark and AR | `20261127T040436.363691Z-bench-compare-f7f8d1fb.json`, `20261127T040436.430006Z-bench-compare-bc5c1362.json` |
| Temperature 1.0 streaming | `20261127T053400.889591Z-http-stochastic-smoke-401157c8.json` |
| Local acceptance and final GPU status | `20261127T040933.129891Z-dev-check-77191509.json`, `20261127T040923.263133Z-dev-status-4efa91fd.json` |
| Window fix build and fixture regeneration | `20261127T052941.949361Z-dev-build-4bb40d30.json`, `20261127T053139.922645Z-weights-golden-afb57430.json` |
| Window fix local and GPU acceptance | `20261127T053030.460296Z-dev-check-aade0dd8.json`, `20261127T053317.458612Z-dev-check-1f829531.json` |
| Window fix KLD and HTTP canary | `20261127T053336.652259Z-weights-kld-8a2d8fa0.json`, `20261127T053353.275296Z-bench-run-db3111ea.json` |
| Window fix cold long prompt, AR and DFlash 2 | `20261127T053500.638832Z-long-window-parity-deadac78.json`, `20261127T053520.153474Z-long-window-parity-d72d9c0c.json` |
| Window fix greedy short and paired comparison | `20261127T053448.739842Z-bench-run-8716b7c4.json`, `20261127T053516.387304Z-bench-compare-91a7b9ca.json` |

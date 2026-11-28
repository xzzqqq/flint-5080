# Milestone 4 · DFlash 2 verify width on RTX 5080

Keep `spec_verify_len = 8`. On greedy HTTP requests, width 8 had the highest decode throughput on all four agent cases and all three short cases. Across agent greedy rounds, width 4 reached its three-proposal ceiling 147/223 times (65.9%) and width 6 reached its five-proposal ceiling 84/173 times (48.6%). Wider verification cost more per round but emitted enough additional tokens to repay that cost.

The closest greedy result was Q4 at 55,232 prompt tokens (89.51 vs 88.13 tok/s for widths 8 and 6). Width 6 overtook width 8 on Q4 with full sampling (97.51 vs 91.54 tok/s) and filtered sampling (83.33 vs 63.76 tok/s). In the filtered run, width 8 led Q1–Q3 but Q4 produced 147 tokens at 2.50 accepted drafts/round, versus 302 at 3.03 for width 6. That single-seed trajectory moved the four-case filtered geomean in favor of width 6. These samples do not establish a context-length cutoff. The server retains the `--spec-verify-len 2..8` override.

## Sweep conditions

One NVIDIA GeForce RTX 5080 (16,303 MiB, build target `sm_120a`), driver 580.105.08, CUDA 13.0.88, power limit 360 W, maximum graphics/memory clocks 3,090/15,001 MHz. Measurements used the working tree based on `937947275bcd`. Hardware and server records are under `.runs/`. See `20261128T082126.018508Z-dev-status-2bc99403.json` and `20261128T080016.217448Z-dev-serve-d6ecb99c.json`.

The HTTP sweep used Inkling 2 Mini EXL3 3.3bpw, DFlash 2, C1, remote loopback, CUDA Graphs, 256-token prefill chunks and `--prefix-cache-slots 0 --vram-reserve-mib 192`. All three widths reported `max_context = 61,314`. Draft always proposed seven tokens. Target width includes the anchor. After the canary warmup, each agent case ran once per mode with zero cached tokens. Greedy used temperature 0, full used temperature 1.0 and seed 5080, and filtered added top_p 0.95. The short suite used one warmup and three measured rounds per case. SHA-256 for the selected workloads was `b2a965a6fa3eafb90b218ac5613c6d4c499146d0b78c2330dcf3332fe7afc674` (agent) and `a5d139bf5b7313668bf8681ef024691b528f521cf9eeb4dec7cd585f2c36c7cd` (short).

## Decode throughput

| Mode (geometric mean tok/s) | W4 | W6 | W8 |
| --- | --- | --- | --- |
| greedy | 96.73 | 103.60 | 108.95 |
| full | 96.62 | 100.94 | 112.96 |
| filtered | 89.01 | 101.62 | 99.74 |
| agent combined | 94.05 | 102.04 | 107.07 |
| short | 123.36 | 129.20 | 145.97 |

| Case | Snapshot | Prompt tokens | Task | W4 tok/s | W6 tok/s | W8 tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | django__django-11099 | 10817 | edit | 98.77 | 103.98 | 112.24 |
| Q2 | pydata__xarray-4687 | 16937 | explore | 105.90 | 119.44 | 131.82 |
| Q3 | pallets__flask-5014 | 27320 | debug-verify | 89.93 | 97.59 | 110.28 |
| Q4 | psf__requests-1142 | 55232 | debug | 83.17 | 89.47 | 80.54 |

Each mode rate is the geometric mean of case medians. Each snapshot rate above is the geometric mean across its three sampling modes. Task and context length vary together in these four cases.

## Request telemetry

Each histogram counts rounds by emitted draft tokens. N is the sum of the eight `spec_accept_hist` bins. Ceiling % is 100 × bin[W−1] / N. It measures rounds reaching the verify ceiling. The counter cannot show how many additional proposals would have passed. Acc/rnd, tok/rnd and ms/rnd use all decode rounds. Tok/rnd and tok/s exclude the first token from prefill. TTFT is client time to first nonempty delta.

| Mode | Case | W | Histogram (bins 0–7) | N | Ceiling % | Acc/rnd | Tok/rnd | ms/rnd | Tok/s | Prefill ms | TTFT ms | Output tokens | HTTP finish |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| greedy | Q1 | 4 | 2,2,4,9,0,0,0,0 | 17 | 52.94 | 2.176 | 3.118 | 31.203 | 99.91 | 30850.6 | 30921.1 | 54 | tool_calls |
| greedy | Q2 | 4 | 2,3,0,16,0,0,0,0 | 21 | 76.19 | 2.429 | 3.381 | 32.370 | 104.45 | 49901.9 | 49981.2 | 72 | tool_calls |
| greedy | Q3 | 4 | 15,11,7,67,0,0,0,0 | 100 | 67.00 | 2.260 | 3.250 | 33.559 | 96.84 | 84079.7 | 84172.0 | 326 | tool_calls |
| greedy | Q4 | 4 | 10,12,8,55,0,0,0,0 | 85 | 64.71 | 2.271 | 3.259 | 37.624 | 86.62 | 189702.7 | 189823.0 | 278 | tool_calls |
| greedy | Q1 | 6 | 2,2,4,0,0,6,0,0 | 14 | 42.86 | 2.857 | 3.786 | 36.994 | 102.33 | 30869.3 | 30950.9 | 54 | tool_calls |
| greedy | Q2 | 6 | 1,3,0,0,1,10,0,0 | 15 | 66.67 | 3.800 | 4.733 | 38.461 | 123.07 | 49914.5 | 50006.7 | 72 | tool_calls |
| greedy | Q3 | 6 | 14,9,5,5,5,39,0,0 | 77 | 50.65 | 3.234 | 4.221 | 40.666 | 103.79 | 84062.8 | 84169.0 | 326 | tool_calls |
| greedy | Q4 | 6 | 9,10,6,8,5,29,0,0 | 67 | 43.28 | 3.149 | 4.134 | 46.912 | 88.13 | 189613.2 | 189751.5 | 278 | tool_calls |
| greedy | Q1 | 8 | 2,2,3,0,0,0,1,4 | 12 | 33.33 | 3.500 | 4.417 | 40.745 | 108.40 | 30860.4 | 30949.2 | 54 | tool_calls |
| greedy | Q2 | 8 | 1,2,0,0,1,1,0,7 | 12 | 58.33 | 5.000 | 5.917 | 43.539 | 135.89 | 49902.3 | 50002.9 | 72 | tool_calls |
| greedy | Q3 | 8 | 12,9,5,3,6,7,3,22 | 67 | 32.84 | 3.866 | 4.851 | 45.399 | 106.85 | 84042.0 | 84157.8 | 326 | tool_calls |
| greedy | Q4 | 8 | 10,9,5,5,4,1,3,21 | 58 | 36.21 | 3.793 | 4.776 | 53.355 | 89.51 | 189540.0 | 189693.7 | 278 | tool_calls |
| full | Q1 | 4 | 4,2,2,15,0,0,0,0 | 23 | 65.22 | 2.217 | 3.174 | 31.535 | 100.65 | 30857.7 | 30930.0 | 74 | tool_calls |
| full | Q2 | 4 | 2,2,0,18,0,0,0,0 | 22 | 81.82 | 2.545 | 3.500 | 32.518 | 107.63 | 49881.7 | 49961.3 | 78 | tool_calls |
| full | Q3 | 4 | 17,13,7,65,0,0,0,0 | 102 | 63.73 | 2.176 | 3.167 | 33.737 | 93.86 | 83993.5 | 84084.6 | 324 | tool_calls |
| full | Q4 | 4 | 11,13,11,56,0,0,0,0 | 91 | 61.54 | 2.231 | 3.220 | 37.562 | 85.72 | 189392.5 | 189511.6 | 294 | tool_calls |
| full | Q1 | 6 | 3,2,2,2,1,8,0,0 | 18 | 44.44 | 3.111 | 4.056 | 38.223 | 106.10 | 30870.6 | 30969.3 | 74 | tool_calls |
| full | Q2 | 6 | 2,0,2,2,0,9,0,0 | 15 | 60.00 | 3.667 | 4.600 | 39.051 | 117.79 | 49915.2 | 50007.5 | 70 | tool_calls |
| full | Q3 | 6 | 20,18,16,9,8,26,0,0 | 97 | 26.80 | 2.464 | 3.454 | 40.551 | 85.17 | 84058.1 | 84163.1 | 336 | tool_calls |
| full | Q4 | 6 | 6,6,10,4,3,39,0,0 | 68 | 57.35 | 3.603 | 4.588 | 47.052 | 97.51 | 189638.0 | 189776.4 | 313 | tool_calls |
| full | Q1 | 8 | 3,2,1,1,1,0,1,6 | 15 | 40.00 | 3.933 | 4.867 | 42.176 | 115.39 | 30859.2 | 30962.4 | 74 | tool_calls |
| full | Q2 | 8 | 2,0,1,1,1,0,0,7 | 12 | 58.33 | 4.833 | 5.750 | 43.459 | 132.31 | 49897.6 | 49997.9 | 70 | tool_calls |
| full | Q3 | 8 | 11,6,5,7,4,3,5,31 | 72 | 43.06 | 4.375 | 5.361 | 46.027 | 116.48 | 84036.4 | 84151.1 | 387 | tool_calls |
| full | Q4 | 8 | 6,10,8,4,7,3,3,20 | 61 | 32.79 | 3.918 | 4.902 | 53.544 | 91.54 | 189566.6 | 189715.6 | 300 | tool_calls |
| filtered | Q1 | 4 | 4,2,2,15,0,0,0,0 | 23 | 65.22 | 2.217 | 3.174 | 33.123 | 95.82 | 30853.5 | 30942.4 | 74 | tool_calls |
| filtered | Q2 | 4 | 2,0,1,15,0,0,0,0 | 18 | 83.33 | 2.611 | 3.556 | 33.652 | 105.66 | 49888.6 | 49969.8 | 65 | tool_calls |
| filtered | Q3 | 4 | 19,11,15,30,0,0,0,0 | 75 | 40.00 | 1.747 | 2.733 | 34.167 | 80.00 | 84013.2 | 84106.0 | 206 | tool_calls |
| filtered | Q4 | 4 | 7,7,8,22,0,0,0,0 | 44 | 50.00 | 2.023 | 3.000 | 38.713 | 77.49 | 189492.5 | 189611.8 | 133 | tool_calls |
| filtered | Q1 | 6 | 3,2,2,2,1,8,0,0 | 18 | 44.44 | 3.111 | 4.056 | 39.170 | 103.54 | 30870.5 | 30971.2 | 74 | tool_calls |
| filtered | Q2 | 6 | 1,0,3,3,0,9,0,0 | 16 | 56.25 | 3.750 | 4.688 | 39.881 | 117.54 | 49917.7 | 50011.7 | 76 | tool_calls |
| filtered | Q3 | 6 | 9,6,9,8,6,37,0,0 | 75 | 49.33 | 3.427 | 4.413 | 41.974 | 105.14 | 84071.3 | 84178.3 | 332 | tool_calls |
| filtered | Q4 | 6 | 12,9,10,9,4,31,0,0 | 75 | 41.33 | 3.027 | 4.013 | 48.162 | 83.33 | 189554.2 | 189694.4 | 302 | tool_calls |
| filtered | Q1 | 8 | 3,2,1,1,1,0,1,6 | 15 | 40.00 | 3.933 | 4.867 | 43.048 | 113.05 | 30862.4 | 30967.7 | 74 | tool_calls |
| filtered | Q2 | 8 | 2,0,2,1,1,1,1,7 | 15 | 46.67 | 4.733 | 5.667 | 44.484 | 127.39 | 49898.9 | 50001.6 | 86 | tool_calls |
| filtered | Q3 | 8 | 8,7,5,9,7,3,4,22 | 65 | 33.85 | 4.077 | 5.062 | 46.974 | 107.75 | 84042.5 | 84158.3 | 330 | tool_calls |
| filtered | Q4 | 8 | 7,11,6,7,4,2,0,5 | 42 | 11.90 | 2.500 | 3.476 | 54.518 | 63.76 | 189597.9 | 189750.0 | 147 | tool_calls |

Across each four-request suite, the following percentages weight every speculative round equally.

| Mode | W | Spec rounds | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| greedy | 4 | 223 | 13.0 | 12.6 | 8.5 | 65.9 | 0.0 | 0.0 | 0.0 | 0.0 |
| greedy | 6 | 173 | 15.0 | 13.9 | 8.7 | 7.5 | 6.4 | 48.6 | 0.0 | 0.0 |
| greedy | 8 | 149 | 16.8 | 14.8 | 8.7 | 5.4 | 7.4 | 6.0 | 4.7 | 36.2 |
| full | 4 | 238 | 14.3 | 12.6 | 8.4 | 64.7 | 0.0 | 0.0 | 0.0 | 0.0 |
| full | 6 | 198 | 15.7 | 13.1 | 15.2 | 8.6 | 6.1 | 41.4 | 0.0 | 0.0 |
| full | 8 | 160 | 13.8 | 11.2 | 9.4 | 8.1 | 8.1 | 3.8 | 5.6 | 40.0 |
| filtered | 4 | 160 | 20.0 | 12.5 | 16.2 | 51.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| filtered | 6 | 184 | 13.6 | 9.2 | 13.0 | 12.0 | 6.0 | 46.2 | 0.0 | 0.0 |
| filtered | 8 | 137 | 14.6 | 14.6 | 10.2 | 13.1 | 9.5 | 4.4 | 4.4 | 29.2 |

### Short greedy reference

One warmup, three measured requests per case. Counts and N pool the three histograms. Scalar metrics are case medians.

| Case | W | Histogram (bins 0–7) | N | Ceiling % | Acc/rnd | Tok/rnd | ms/rnd | Tok/s | Prefill ms | TTFT ms | Output tokens | HTTP finish |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_chat | 4 | 4,5,4,41,0,0,0,0 | 54 | 75.93 | 2.556 | 3.500 | 28.328 | 123.27 | 202.0 | 233.7 | 64 | length |
| short_debug | 4 | 8,5,8,37,0,0,0,0 | 58 | 63.79 | 2.316 | 3.316 | 28.755 | 115.31 | 753.6 | 783.8 | 64 | length |
| short_review | 4 | 0,0,3,45,0,0,0,0 | 48 | 93.75 | 2.938 | 3.938 | 29.815 | 132.06 | 2949.5 | 2980.9 | 64 | length |
| short_chat | 6 | 4,5,7,3,3,21,0,0 | 43 | 48.84 | 3.200 | 4.200 | 32.347 | 129.84 | 202.1 | 237.2 | 64 | length |
| short_debug | 6 | 11,8,3,6,12,12,0,0 | 52 | 23.08 | 2.765 | 3.706 | 33.514 | 110.12 | 754.0 | 789.0 | 64 | length |
| short_review | 6 | 0,2,3,4,1,26,0,0 | 36 | 72.22 | 4.250 | 5.250 | 34.809 | 150.82 | 2950.4 | 2987.6 | 64 | length |
| short_chat | 8 | 4,5,2,4,0,1,2,17 | 35 | 48.57 | 4.333 | 5.250 | 36.310 | 144.87 | 202.2 | 240.1 | 64 | length |
| short_debug | 8 | 4,8,2,4,12,0,6,6 | 42 | 14.29 | 3.571 | 4.500 | 36.840 | 122.15 | 754.1 | 792.2 | 64 | length |
| short_review | 8 | 0,0,2,3,0,1,4,17 | 27 | 62.96 | 5.300 | 6.300 | 36.674 | 175.76 | 2949.6 | 2989.9 | 64 | length |

## Sweep archives

Filenames are relative to `.runs/`.

| Suite/mode | W4 | W6 | W8 |
| --- | --- | --- | --- |
| greedy | `20261127T081803.836127Z-bench-run-b0bc5d8d.json` | `20261128T074741.098160Z-bench-run-743f911c.json` | `20261128T080626.729473Z-bench-run-264397de.json` |
| full | `20261128T073501.197241Z-bench-run-2a89f925.json` | `20261128T075345.110873Z-bench-run-55c4ce34.json` | `20261128T081229.876311Z-bench-run-77e211e1.json` |
| filtered | `20261128T074102.189655Z-bench-run-f05bdf2c.json` | `20261128T075948.678847Z-bench-run-60d0a7ed.json` | `20261128T081831.960109Z-bench-run-c7035bb5.json` |
| short | `20261128T074125.142330Z-bench-run-bca7c87d.json` | `20261128T080011.322557Z-bench-run-fcf74942.json` | `20261128T081853.897251Z-bench-run-10876591.json` |

| Warmup | W4 | W6 | W8 |
| --- | --- | --- | --- |
| canary | `20261127T081200.558256Z-bench-run-f623b75a.json` | `20261128T074138.228336Z-bench-run-979dd6fe.json` | `20261128T080024.207489Z-bench-run-a6f5a5e0.json` |

## Forge merged graph

Greedy `short_math`, step 1, same saved prefix for each timed replay. The default prefix has 27 tokens (`context_len = 0` selects the case length). The long replay tiles the prompt to 16,384 tokens. Forge allocated 48,830 KV positions and 450 MiB scratch. Each record shows one capture, 18 graph launches and 0 B unload loss. The default-prefix golden check matched all emitted tokens at each width. See `20261128T081859.557910Z-forge-start-a2d63ed8.json`.

GPU ms/round covers the merged graph. Wall ms/round includes commit and context KV store. Replay tok/s is 1,000 × emitted tokens / median wall ms. Forge archives the last replay’s accepted and emitted counts, so the histogram and ceiling below describe that single observed round. The record has no multi-replay histogram or prefill timer.

| Prefix tokens | W | Last-round bins 0–7 | Ceiling % | Acc/round | Tok/round | GPU ms/round | Wall ms/round | Replay tok/s | Prefill ms | Archive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 4 | 0,0,0,1,0,0,0,0 | 100 | 3 | 4 | 27.876 | 28.163 | 142.03 | n/a | `20261128T081900.486457Z-forge-run-0a7a3c31.json` |
| 27 | 6 | 0,0,0,0,0,1,0,0 | 100 | 5 | 6 | 32.525 | 32.819 | 182.82 | n/a | `20261128T081901.501686Z-forge-run-cd992fe0.json` |
| 27 | 8 | 0,0,0,0,0,1,0,0 | 0 | 5 | 6 | 36.485 | 36.781 | 163.13 | n/a | `20261128T081902.575256Z-forge-run-f0079436.json` |
| 16384 | 4 | 0,0,0,1,0,0,0,0 | 100 | 3 | 4 | 32.332 | 32.620 | 122.62 | n/a | `20261128T081950.429293Z-forge-run-53f6dcf2.json` |
| 16384 | 6 | 0,0,0,0,1,0,0,0 | 0 | 4 | 5 | 38.390 | 38.684 | 129.25 | n/a | `20261128T081951.455130Z-forge-run-33f9fbf5.json` |
| 16384 | 8 | 0,0,0,0,1,0,0,0 | 0 | 4 | 5 | 43.685 | 43.981 | 113.69 | n/a | `20261128T081952.560729Z-forge-run-8f2ed61f.json` |

At width 8 on the 27-token prefix, the published 8.815 ms logical floor gives 24.0% SOL against 36.781 ms wall time. Only this cell matches the width and route used for the floor. The routing record is `20261127T040731.738316Z-dflash2-numerics-and-routing-7475089a.json`. [Roofline budget](../model.md)

## Final production acceptance

Production started with `--prefix-cache-slots 3` and engine defaults. The ready probe reported width 8, 512 MiB reserve and 31,508 tokens of context. All six canaries passed. The remote build and all seven GPU groups passed (ctest, ops-golden, e2e-golden, runtime-golden, dflash2-golden, forge-host and server).

Teacher-forced KL was zero (mean, p95 and max), top-1 matched 288/288 positions, max absolute logit difference was zero, and unload loss was 0 B. Reference and candidate had the same SHA-256, `e21f2fb6bbfbf93312f1e82a6c09bfe7c52576ac8779368bb093faf36642537f`. This comparison measures replay repeatability.

### Paired greedy short

Both runs used the same workload hash, one warmup and three measured cold requests per case. Case medians give 146.747 → 146.661 tok/s across cases (geometric mean). Final/baseline was **0.999415×** (95% paired case bootstrap [0.998576, 1.000114], n = 3). The comparison flags a server metadata change. The frozen run used ABI 2. ABI 3 also reports verify width and reserve. Model, context, cache slots, chunk size, graph mode and speculative mode agree.

| Case | Baseline decode tok/s | Final decode tok/s | Final/baseline |
| --- | --- | --- | --- |
| short_chat | 145.526 | 145.461 | 0.999555× |
| short_debug | 123.261 | 123.086 | 0.998576× |
| short_review | 176.173 | 176.193 | 1.000114× |

| Paired metric | Final/baseline geomean | 95% CI |
| --- | --- | --- |
| prefill_ms | 1.000537× | [1.000289, 1.000761] |
| client_ttft_ms | 1.000044× | [0.999101, 1.000777] |
| server_tpot_ms | 1.000134× | [0.998534, 1.001426] |
| server_decode_tok_s | 0.999415× | [0.998576, 1.000114] |
| client_e2e_ms | 1.000103× | [0.999769, 1.000383] |
| tokens_per_round | 1.000000× | [1.000000, 1.000000] |

### Single-request stochastic smoke

One cold streaming request per short case with the round-77 prefix nonce, temperature 1.0, seed 5080 and 64 output tokens. All six measured requests (baseline and final) had zero cached tokens and finished at the length limit. The final fixture had the same workload hash as the archived nonce-77 bench run `20261127T075102.287482Z-bench-run-f9970a4a.json`. The comparison uses the original smoke’s cold samples. Across cases, the geometric mean of final/baseline decode tok/s was **1.098031×**.

| Case | Prompt tokens | Baseline decode tok/s | Final decode tok/s | Final/baseline | Rounds, baseline → final | Accepted, baseline → final |
| --- | --- | --- | --- | --- | --- | --- |
| short_chat | 63 | 109.276 | 118.691 | 1.086154× | 14 → 14 | 50 → 50 |
| short_debug | 273 | 138.274 | 155.547 | 1.124915× | 11 → 11 | 53 → 53 |
| short_review | 1091 | 97.281 | 105.405 | 1.083509× | 16 → 16 | 47 → 47 |

### Acceptance records

Formatting and the five local acceptance groups (pytest, Cargo tests, format, lint and hygiene) passed before the build. The post-documentation local check is recorded below. After server stop, the GPU reported 0/16,303 MiB used, no compute processes and a free lease. Archive filenames are relative to `.runs/`.

| Evidence | Archive |
| --- | --- |
| Formatting | `20261128T082133.091960Z-dev-fmt-5e04a236.json` |
| Local acceptance before build | `20261128T082201.553611Z-dev-check-4bd34daf.json` |
| Remote engine and server build | `20261128T082202.948703Z-dev-build-e1dfdbbd.json` |
| GPU acceptance | `20261128T082449.991231Z-dev-check-67e42908.json` |
| Teacher-forced KLD | `20261128T082501.061720Z-weights-kld-78891592.json` |
| Production restart | `20261128T082514.183511Z-dev-serve-6fc61c7f.json` |
| Production canary (JSON stdout) | `m4-final-canary.json` |
| Pre-round greedy | `20261127T053448.739842Z-bench-run-8716b7c4.json` |
| Final greedy (@m4-final-short) | `20261128T082543.830342Z-bench-run-cc65f42d.json` |
| Paired greedy comparison | `20261128T082543.935010Z-bench-compare-9f4f8fa9.json` |
| Pre-round nonce 77 | `20261127T053400.889591Z-http-stochastic-smoke-401157c8.json` |
| Final nonce 77 (@m4-final-nonce77) | `20261128T082550.233040Z-bench-run-179c0844.json` |
| Production stop | `20261128T082551.070162Z-dev-serve-d7ef1b17.json` |
| Idle GPU and free lease | `20261128T082554.217655Z-dev-status-2b1ccaec.json` |
| Post-documentation local acceptance (JSON stdout) | `m4-final-local.json` |

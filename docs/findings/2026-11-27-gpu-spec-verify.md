# GPU speculative verification on RTX 5080

Each speculative round now copies a 56-byte result to the host. At width 8 the previous stochastic path copied a 924-byte proposal and 6,401,856 bytes of target logits, plus the target runner's timing-path argmax. The verifier reuses target activation scratch on the same stream after forward finishes.

## HTTP smoke

C1 streaming over remote loopback, RTX 5080, CUDA 13.0.88, DFlash 2, verify width 8, CUDA Graphs enabled, 256-token prefill chunks, three prefix slots and a 31,508-token context. The stochastic requests used the archived nonce-77 messages, temperature 1, seed 5080 and full-vocabulary sampling. Each case returned 64 tokens with zero cached prompt tokens. Timings are server decode milliseconds divided by decode rounds, with one cold request per case. [Baseline conditions and timings](2026-11-27-dflash2-pivot-baseline.md)

| Case | Baseline ms/round | GPU ms/round | Decrease | Rounds | Accepted drafts |
|---|---:|---:|---:|---:|---:|
| short_chat | 41.180 | 37.835 | 8.1% | 14 | 50 |
| short_debug | 41.420 | 36.793 | 11.2% | 11 | 53 |
| short_review | 40.475 | 37.411 | 7.6% | 16 | 47 |

Both runs recorded the same round and acceptance counts for each prompt. The greedy short smoke measured 36.15, 36.51 and 36.44 ms/round respectively. All six greedy canaries passed.

## Correctness and teardown

Graph-captured verifier tests matched CPU accepted tokens, corrected or bonus tokens and next seeds, including 4,096 sparse acceptance trials and 200,058-way full and filtered distributions at widths 4, 6 and 8. Filtered bonuses traverse the ranked candidate list used by the CPU sampler. Rejection traverses vocabulary IDs in ascending order. FP64 tile reductions group additions differently from the CPU's sequential sum.

Model-backed seed replay passed in both stochastic modes at each width with the minimum configured scratch budget. All 288 greedy tokens matched the teacher-forced target. The teardown comparison used two independent engine lifetimes at each width. After the first lifetime, free VRAM changed by 0, 4 MiB and 0 respectively. After the second lifetime it changed by 0 B at every width.

Local run records under `.runs/` include `20261127T075047.292393Z-dev-check-6d48f859.json` for GPU acceptance, `20261127T075102.287482Z-bench-run-f9970a4a.json` for stochastic HTTP, `20261127T075109.954845Z-bench-run-713c2465.json` for canary and `20261127T075116.031858Z-bench-run-f8d63741.json` for greedy short.

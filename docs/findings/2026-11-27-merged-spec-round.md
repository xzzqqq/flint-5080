# Merged C1 speculative round on RTX 5080

Measured 2026-11-27 on one RTX 5080 (16,303 MiB), driver 580.105.08, CUDA 13.0.88, working tree based on `48a1a20`. Forge used both resident checkpoints, 48,830 tokens of KV capacity and 450 MiB scratch. Each row is the median of 15 greedy first rounds on the 27-token `short_math` prompt after three warmups. GPU events bracket draft through acceptance. Wall time also includes the transfers, commit and context projection.

| Verify width | GPU round (ms) | Wall (ms) | Eager wall (ms) | Speedup over eager | Accepted / emitted |
|---:|---:|---:|---:|---:|---:|
| 8 | 36.074 | 36.372 | 41.662 | 1.145× | 5 / 6 |
| 6 | 32.493 | 32.786 | 38.829 | 1.184× | 5 / 6 |
| 4 | 27.855 | 28.145 | 33.887 | 1.204× | 3 / 4 |

Each graph captured once and launched 18 times. All three pairs emitted identical IDs and unloaded with 0 B loss. The two width-8 graph measurements were 36.245 and 36.372 ms end to end, bracketing the historical 36.301 ms. At the archived 8.815 ms logical floor, the final width-8 run reached 24.2% logical SOL. [Baseline protocol and roofline](2026-11-27-dflash2-pivot-baseline.md)

The warmed Nsight range contained one graph launch, two asynchronous copies and two stream synchronizations. One sync followed acceptance readback and the other followed commit/store. The trace recorded 4,360 kernel instances and 35.130 ms of kernel time in 36.668 ms wall time. The first profiler capture retained 65,536 B, and the second had 0 B additional loss.

Fixed-proposal replay at width 8 accepted five drafts and emitted the same six IDs. Standalone graph verification took 33.753 ms, commit/store 0.287 ms and total replay 34.058 ms. The stage profile recorded embedding, all 48 visits and the LM head. Fixed-proposal A/B at width 6, merged A/B at width 4, and the width-4 fixed-proposal dump passed with 0 B loss.

Greedy serving matched AR on all 49 math, 102 sky and 137 code tokens. At each width, eager and graph execution matched in greedy and both stochastic modes, and warmed runtime teardown cost 0 B. Switching modes reused the three graph slots and a two-row context tail launched eagerly without evicting them. The HTTP short suite and all six canaries passed. KLD of the reset baseline against the same final binary was zero at all 288 teacher-forced positions, with 288 top-1 matches and 0 B unload loss.

## Local run records

Raw records are under `.runs/`.

| Measurement | Record |
|---|---|
| graph-8 | `20261127T080444.219651Z-forge-run-1e90e7e9.json` |
| eager-8 | `20261127T080445.327452Z-forge-run-b8f88e3e.json` |
| graph-6 | `20261127T080446.299056Z-forge-run-0daf2d41.json` |
| eager-6 | `20261127T080447.463979Z-forge-run-dda5000f.json` |
| graph-4 | `20261127T080448.353308Z-forge-run-263df3fe.json` |
| eager-4 | `20261127T080449.318111Z-forge-run-c745dc3f.json` |
| fixed | `20261127T080449.940624Z-forge-run-7517d852.json` |
| profile | `20261127T080450.536007Z-forge-profile-899cf6f5.json` |
| nsys-cold | `20261127T080454.323966Z-forge-profile-9d5e2d28.json` |
| nsys-warm | `20261127T080458.038687Z-forge-profile-1c8c8882.json` |
| kld | `20261127T080504.062100Z-weights-kld-db2e9d30.json` |
| short | `20261127T080437.536171Z-bench-run-8c436310.json` |
| canary | `20261127T080415.725596Z-bench-run-33a238d0.json` |
| GPU acceptance | `20261127T080403.593521Z-dev-check-80d773b9.json` |

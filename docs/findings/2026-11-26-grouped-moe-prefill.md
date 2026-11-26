# Expert-grouped EXL3 MoE prefill on RTX 5080

Measured on 2026-11-26 with CUDA 13.0.88, `sm_120a`, Inkling 2 mini EXL3 3.3 bpw and the working tree based on `152891247f9b`. HTTP measurements used the Phase 2 configuration (67,176 context, 256-token prefill chunks, three cache slots and CUDA Graph decode). The Mac client used an SSH tunnel. Each suite started with an empty cache.

| Workload | Tokens | Prefill ms, before → after | Prompt tok/s, before → after | Speedup | Decode ms/round, before → after |
|---|---:|---:|---:|---:|---:|
| Forge `short_math` | 27 | 2,256.361 → 155.805 | 11.97 → 173.29 | 14.48× | 40.890 → 40.874 |
| HTTP `short_chat` | 59 | 4,921.069 → 293.060 | 11.99 → 201.32 | 16.79× | 42.059 → 42.079 |
| HTTP `short_debug` | 269 | 22,823.318 → 1,354.335 | 11.79 → 198.62 | 16.85× | 45.870 → 45.878 |
| HTTP `short_review` | 1,087 | 96,298.296 → 9,204.261 | 11.29 → 118.10 | 10.46× | 58.984 → 58.895 |

Forge timed end-to-end prefill with one warmup and three iterations, and the first graphed decode step with two warmups and nine iterations. Its model-budget decode SOL was 8.528% before and 8.532% after. The candidate matched the frozen module and golden logits exactly, including carry, KV and convolution state, with 0 B unload loss.

HTTP medians use server telemetry, one warmup and three measured rounds per case. All 12 requests missed the prefix cache. Decode used one graph capture and 756 launches. The paired prefill speedup was 14.359× (95% case-bootstrap interval 10.462–16.852×). The TPOT ratio was 0.9997 (95% interval 0.9985–1.0005). The largest individual TPOT increase was 0.048%. The comparison reported no configuration or workload warnings.

## Kernel, scratch and parity

The TM=16 kernel decodes each 16-column trellis tile once per expert row tile and reuses its weights across rows. Twelve ordered FP32 split sums remain in registers. The input gather shares the scale/WHT kernel, and final accumulation follows ascending expert IDs through the inverse permutation. NVCC ptxas reported 144 registers per thread, 0 B stack and 0 B spill loads/stores. The release object also reported 0 B local memory.

At the grouped MoE call, 256-byte-aligned allocations including resident runner scratch, forward activations and router buffers total 94.39, 187.37 and 364.89 MiB at 256, 520 and 1,024 tokens. The default arena has 450 MiB. The 1,087-token review request ran with each chunk size. The 520 and 1,024 checks used a 2,048-token context and a one-token output limit to exercise arena capacity independently of decode.

Synthetic tests covered inverse mapping, full and tail tiles, omitted experts, and bitwise agreement with scalar cuBLAS-backed projections at 5 and 35 tokens. Local checks and all GPU acceptance stages passed. The MoE golden output matched 9,216/9,216 BF16 values. Eager and graph each matched 288/288 greedy tokens with zero top-64 error. Runtime and module teardown deltas were 0 B. All six HTTP canaries passed.

## Run records

Archives are under `.runs/`. The frozen HTTP baseline and workload hashes are recorded in the [Phase 2 baseline](2026-11-26-phase2-ar-baseline.md).

| Measurement | Archive filename |
|---|---|
| Forge prefill, before / after | `20261126T111423.198047Z-forge-run-e04a8c95.json` / `20261126T111530.837863Z-forge-run-20aaf156.json` |
| Forge decode, before / after | `20261126T111432.218016Z-forge-run-6b844312.json` / `20261126T111548.559162Z-forge-run-93a229ad.json` |
| Local / GPU acceptance | `20261126T111659.328377Z-dev-check-ae620bd7.json` / `20261126T112035.794863Z-dev-check-cbe6930e.json` |
| HTTP canary / short | `20261126T112110.408918Z-bench-run-320e7b52.json` / `20261126T112238.031186Z-bench-run-a7a59b6a.json` |
| Frozen baseline comparison | `20261126T112239.611923Z-bench-compare-9a60eef9.json` |
| 520 / 1,024 chunk checks | `20261126T112253.531840Z-bench-run-5a6b28f8.json` / `20261126T112307.917315Z-bench-run-9e16906b.json` |
| Free GPU after teardown | `20261126T112317.239725Z-dev-status-d7db7401.json` |

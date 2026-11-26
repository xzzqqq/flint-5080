# DSpark draft and target verification on RTX 5080

Measured on one RTX 5080 with CUDA 13.0.88, driver 580.105.08, and the M3.2 working tree based on `74cde98`. The checkpoint was Maxrio Inkling 2 Mini EXL3 3.3 bpw. The 27-token `short_math` fixture had SHA-256 `71250c6729d4cbf864dfa427e035576dca0b2736366d633496e49f3a9dc6ebf3`.

## Golden replay

| Capture | Maximum absolute error |
| --- | ---: |
| Target prefill taps, all three layer mixtures, three context K/V pairs | 0 |
| Markov mixture across all 27 prompt positions | 2.38419e-7 |
| Anchor/mask embedding, layer outputs 0–2, final head hidden, last-position Markov mixture | 0 |
| Markov FP32 context, BF16 states and latents, all top-64 logits | 0 |

Replaying both sets of fusion coefficients through the same PyTorch operations isolated the residual to host softmax. Two of the four FP32 coefficients in the Markov row differed by one or two FP32 ULPs from CUDA softmax. Across the 27 × 2304 prompt features, host coefficients changed one BF16 value by exactly 2⁻²². The last-position mixture used by the Markov head matched exactly.

All 448 ranked top-64 IDs matched. Each round matched all seven draft IDs, its acceptance count, and its next bonus. The counts were `3, 1, 1, 0, 2, 5, 1, 6, 3, 3, 2, 2, 7`. The 49 output tokens matched both the speculative and AR fixtures, including the initial prefill token and the budget-truncated last round. Steady-state teardown loss was 0 B across a complete model, KV, scratch and runner lifetime.

Starting an eight-row verify at position 1021 exercised target SWA wraparound. After commits of 0, 3, and 8 input rows, all fixed target KV and SConv bytes matched a direct forward over the same prefix. Selecting a different first draft token through the sampling callback changed the second row of proposal logits.

## Attention rounding

The initial warp-level online softmax matched 286 of 448 ranked IDs and produced different drafts in round 11. A sequential FP32 dot product also diverged from PyTorch's strided batched SGEMM. Using the reference matrix strides matched every first-layer QK score. Three BF16 attention outputs still differed. The compiler had contracted score scaling and max subtraction into an FMA. Rounding the scaled score before subtraction made the softmax probabilities and attention output exact in the first layer. All three layer outputs then matched the fixture.

## Forge hot reload

With a 2,048-token KV allocation, 450 MiB scratch, and resident MTP weights, `forge run --e2e --step 1 --case short_math --warmup 1 --iters 3` measured **106.866067 ms** and **106.914362 ms** median on two successive module loads. Each round drafted seven, verified eight inputs, accepted three drafts and emitted four tokens. The timed interval included commit and the draft context update. Both loads reported 0 B VRAM loss. Run IDs were `20261126T150334.053626Z-forge-run-ac5390ab` and `20261126T150334.882535Z-forge-run-e6c89942`.

The corresponding `--no-spec` replay matched all first-decode logits, and `--layers 6..18` matched the recorded visit-17 hidden state. Each reported 0 B unload loss.

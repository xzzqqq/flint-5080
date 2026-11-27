# AGENTS.md

## Project

Flint-5080 is a local inference engine for personal chat and coding agents,
serving **Inkling 2 mini** (with DFlash 2 speculative decoding) on one
**RTX 5080 16 GB** (`sm_120a`).

Optimize for C1 latency with decode approaching SOL (Speed of Light), and C2
when feasible, without artificially hard-capping concurrency.

## Guardrails and optimization freedom

Keep a small set of structural boundaries intact:

- `server/` (Rust) handles HTTP, tokenization, templates, and stream parsing
  only—never scheduling, KV accounting, or model layout. Cross-language calls
  go exclusively through the Request-level C ABI (`engine/include/engine_api.h`).
- Inside `engine/` (C++/CUDA), forbid dynamic `cudaMalloc`/`cudaFree` on
  inference hot paths, and keep `ops/*.cu` header-isolated with lean `sm_120a`
  template instantiations to preserve fast incremental `forward.so` builds.
- Keep `tools/` (Python) out of production serving paths; expose tooling only
  through `dev`, `forge`, `bench`, and `weights` instead of scattering ad-hoc
  scripts.

Inside `engine/`, operator boundaries, layer boundaries, and `forge` tooling
are conveniences, not optimization constraints. Fusion, splitting, cross-layer
changes, multi-stream overlap, host-side scheduling changes, and weight layout
repacking (as standard `safetensors`) are all allowed within task scope.

## Environment and feedback loop

Edit and run CPU checks locally (`dev check --local`); build CUDA and run GPU
workloads on remote Linux through the CLI (`dev`, `forge`, `bench`, `weights`).
Check `dev status` and coordinate ownership of the single 5080 via its lease
lock before claiming the GPU.

During optimization, iterate with `forge` (resident weights + `forward.so`
hot-reload), measuring both latency/SOL and numerical error. Keep candidates
with substantial speedups as saved alternatives (`--save`) while investigating
numerical differences; do not discard them solely for failing bitwise
comparison. Run full acceptance (`bench` and `dev check --gpu`) at the end of
an optimization round on accumulated changes, not after every small edit.

## Numerical correctness and autonomy

Choose checks according to what the transformation changes:

- Changes to data movement, layout, or scheduling that preserve arithmetic and
  rounding order should preserve output bits. Exact semantics (indices, masks,
  state updates) must remain exact.
- For mathematically equivalent transformations, investigate numerical
  differences rather than rejecting the candidate. Bitwise agreement is not
  required.
- Changes to precision, quantization, or approximate algorithms require
  evaluating both numerical error (e.g., teacher-forced KLD) and model-quality
  impact.

Investigate numerical differences and validate or repair candidates
autonomously. Consult the user only when adopting a candidate would exceed
agreed accuracy or quality limits, or when measurements establish a real
quality tradeoff that requires a user decision. If no criterion exists yet,
gather evidence and propose one. A byte mismatch or an unfinished error
analysis is not itself a reason to stop and ask.

## Evidence and documentation

Ground performance and accuracy claims in matched-condition measurements
(hardware/config, timing/SOL, numerical error), never speculation.

Keep documentation minimal:

- `docs/architecture.md` — living system blueprint; update **in place** when
  module boundaries or tooling change (never stack historical patch notes).
- `docs/model.md` — living model structure, weight layout, and 5080 SOL
  roofline budget (maintained alongside model integration).
- `docs/benchmark.md` — fixed evaluation protocol, workloads, and acceptance
  gates.
- `docs/findings/` — dated, evidence-backed archive of GPU experiments,
  numerical error analyses, and **rejected dead ends** so future rounds do not
  repeat them.

## Git and hygiene

- Write commit messages in **English** following **Conventional Commits**
  (`type(scope): summary`, e.g., `feat(forge): add forward.so leak warning`),
  and omit `Co-Authored-By:` trailers.
- Never commit secrets, API keys, tokens, `.dev.toml`, real user traces,
  profiler captures (`*.nsys-rep`, `*.ncu-rep`), model weights, or build
  artifacts. Keep `.githooks/` enabled (`core.hooksPath`) so `gitleaks` and
  commit checks run automatically.

# Flint-5080

A focused local LLM inference engine for serving **Inkling 2 mini** (with DFlash 2 speculative decoding) on a single **RTX 5080 16 GB** (`sm_120a`), built from scratch for low-latency personal chat and coding agents.

The target loads from `--model-dir`. Set `[model].draft_weights_dir` or pass
`--draft-dir` to load `incoai/Inkling-2-Mini-DFlash2`. Use `--no-spec` for AR.
DFlash 2 passed RTX 5080 acceptance and all six HTTP canaries. On the three greedy short cases it delivered 1.262× Round 1 DSpark throughput and 1.880× Round 1 AR throughput. [Measured baseline](docs/findings/2026-11-27-dflash2-pivot-baseline.md)

## Goals

- **Single-GPU focus:** Linux with one RTX 5080 16 GB (`sm_120a`), targeting low concurrency (C1 latency first with decode approaching Speed of Light, C2 when feasible, no artificial concurrency cap, 128K context).
- **Capabilities:** OpenAI-compatible Chat Completions (SSE streaming and non-streaming), reasoning, tool calls, Prometheus `/metrics`, VRAM prefix reuse, and DFlash 2 speculative decoding. Clean extension points are reserved for Responses API, host KV-RAM, Vision, and web UI.
- **Minimal surface:** Standard `safetensors` weights (including offline layout repacks in the same format), zero-bloat single-binary serving, and a fast resident hot-reload workbench (`forge`) for kernel and graph iteration.

---

## Architecture at a Glance

```text
OpenAI HTTP (SSE / JSON)
        │
        ▼
server/   (Thin single-crate Rust shell)
          HTTP, tokenization, Jinja chat templates, reasoning & tool-call parsing, /metrics
        │
        │  Request-level C ABI (engine/include/engine_api.h)
        ▼
engine/   (Self-contained C++/CUDA sm_120a engine)
          safetensors mmap loader, static VRAM arenas, VRAM prefix cache,
          scheduler, CUDA Graphs, AR + DFlash 2 forward & kernels
```

The engine builds into two mutually exclusive runtime modes on the 5080:

- **Serving (`server`)**: Thin Rust HTTP shell statically linked against `libengine` — clean, stable, zero Python runtime dependency.
- **Tuning (`forge-host` + `forward.so`)**: Standalone C++ resident process holding model weights and KV arenas in VRAM while hot-reloading whole-graph `forward.so` builds in seconds—supporting single-op tweaks, cross-operator/cross-layer fusion, CUDA Graph edits, and side-stream overlap without restarting or reloading weights.

---

## Developer Toolchain

Development uses a Mac laptop ↔ remote Linux 5080 workflow backed by four focused CLI entrypoints sharing a lightweight stdlib core (`tools/common/` — `.dev.toml` config, workspace-isolated sync, single-GPU lease lock, and machine-readable `--json` output):

| Command | Role |
|---|---|
| `dev` | Build, production service lifecycle, single-GPU status, and local/GPU verification (`status`, `build`, `serve`, `check`, `ssh`) |
| `forge` | Resident GPU tuning workbench (`start`, `stop`, `run`, `ab`, `profile`, `dump`, `ls`, `rm`) |
| `bench` | Black-box OpenAI HTTP benchmark harness for coding-agent traces, TTFT, canary checks, and paired A/B comparisons |
| `weights` | Offline `safetensors` inspection, golden fixture generation, teacher-forced KLD evaluation, and `sm_120a` layout repacking |

---

## Documentation

- [Architecture](docs/architecture.md) — System boundaries, `engine_api.h`, engine invariants, `forge` hot-reload design, and toolchain.
- [Model & Hardware Budget](docs/model.md) — Inkling 2 mini + DFlash 2 specs, EXL3 subset, 16 GiB / 128K VRAM budget, and 960 GB/s Roofline.
- [Benchmark Protocol](docs/benchmark.md) — Fixed workloads, timing rules, statistical comparisons, and acceptance gates.
- [GPU Findings](docs/findings/README.md) — Dated experiment reports, frozen baselines, numerical analyses, and rejected dead ends.
- [AGENTS.md](AGENTS.md) — Engineering guardrails, optimization workflow, numerical verification rules, and documentation map.

---

## License

Licensed under the [Apache License 2.0](LICENSE).

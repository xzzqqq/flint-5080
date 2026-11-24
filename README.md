# Flint-5080

A focused local LLM inference engine for serving **Inkling 2 mini** (with `dspark` speculative decoding) on a single **RTX 5080 16 GB** (`sm_120a`), built from scratch for low-latency personal chat and coding agents.

> **Status:** Early bootstrap — architecture baseline is settled; engine and server implementation are underway.

---

## Goals

- **Single-GPU focus:** Linux with one RTX 5080 16 GB (`sm_120a`), targeting low concurrency (C1 latency first with decode approaching Speed of Light, C2 when feasible, no artificial concurrency cap).
- **Capabilities:** OpenAI-compatible Chat Completions (SSE streaming and non-streaming), reasoning, tool calls, VRAM prefix reuse, and `dspark` speculative decoding. Clean extension points are reserved for Responses API, host KV-RAM, Vision, metrics, and web UI.
- **Minimal surface:** Standard `safetensors` weights (including offline layout repacks in the same format), zero-bloat single-binary serving, and a fast resident hot-reload workbench (`forge`) for kernel and graph iteration.

---

## Architecture at a Glance

```text
OpenAI HTTP (SSE / JSON)
        │
        ▼
server/   (Thin single-crate Rust shell)
          HTTP, tokenization, Jinja chat templates, reasoning & tool-call stream parsing
        │
        │  Request-level C ABI (engine/include/engine_api.h)
        ▼
engine/   (Self-contained C++/CUDA sm_120a engine)
          safetensors mmap loader, static VRAM arenas, VRAM prefix cache,
          low-concurrency scheduler, CUDA Graphs, AR + dspark forward & kernels
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
| `weights` | Offline `safetensors` inspection, `sm_120a` layout repacking, HuggingFace golden fixture generation, and teacher-forced KLD evaluation |

---

## Roadmap

- **Phase 0 — Scaffold & Tooling:** Repository layout, CMake/Ninja + Cargo skeletons, remote sync, and single-GPU lease locking.
- **Phase 1 — C++ AR Baseline & `forge`:** `safetensors` mmap loader, AR forward pass aligned against HF golden fixtures, and resident `forward.so` hot-reload loop.
- **Phase 2 — Rust HTTP Shell & `bench`:** `engine_api.h` integration, Chat Completions with reasoning/tool calls, VRAM prefix cache, and frozen coding-agent baseline benchmarks.
- **Phase 3 — `dspark` & SOL Optimization:** Speculative decoding integration and sustained kernel/graph optimization toward RTX 5080 hardware limits.

---

## Documentation

- [Architecture Design](docs/architecture.md) — System boundaries, `engine_api.h`, `forge` hot-reload design, CLI toolchain, and roadmap.
- [AGENTS.md](AGENTS.md) — Engineering guardrails, optimization workflow, numerical verification rules, and documentation map.

---

## License

Licensed under the [Apache License 2.0](LICENSE).

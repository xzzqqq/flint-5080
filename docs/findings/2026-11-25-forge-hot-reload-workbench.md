# M1.3 `forge` Resident Workbench & Hot-Reload SLA Verification

**Date**: 2026-11-25  
**Hardware / Software**: Single RTX 5080 (16,303 MiB, `sm_120a`), CUDA 13.0, CMake 3.22 + Ninja (remote host `quant3`, Mac Laptop driver over SSH)  
**Checkpoint**: `Maxrio/Inkling-2-Mini-exl3` (`3.3bpw`, 11.57 GiB text weights resident)  
**Artifacts**: `.runs/20261125T070108.806754Z-forge-start-56ab0758.json`, `.runs/20261125T070122.183987Z-forge-run-94be96c1.json`, `.runs/20261125T070120.214321Z-forge-profile-a4eb0dd7.json`

## Summary

With `forward_abi.h` v2, the resident `forge-host` daemon, and the pure-stdlib `tools/forge_cli/` suite live on the RTX 5080 holding the full 11.57 GiB checkpoint + 64K BF16 KV arena + 450 MiB scratch arena (**14,820 / 16,303 MiB**), an end-to-end single-`.cu` edit → remote sync → incremental compile/link → `dlopen` hot-reload → GPU execution cycle completed in **1.33 s**, well under the 6 s architecture SLA.

## Measurements

### 1. Resident Startup & Arena Footprint (`forge start`)

| Metric | Measured Value |
|---|---:|
| End-to-end `forge start` wall time | `4.267 s` |
| Daemon readiness (incl. `1.496 s` weight load, `261.9 ms` H2D) | `3.249 s` |
| `WeightsArena` | `12,428,319,744 B` (11,852.6 MiB) |
| `KvStateArena` (64K full context) | `2,319,089,664 B` (2,211.7 MiB) |
| `ScratchArena` | `471,859,200 B` (450.0 MiB) |
| Dedicated I/O buffers | `19,678,700 B` (18.8 MiB) |
| Total resident GPU memory | **14,820 / 16,303 MiB** |
| Release artifact sizes (`forward.so` / `forge-host`) | `61,112 B` / `334,560 B` |

### 2. Single-`.cu` Incremental Hot-Reload Breakdown (`forge run --layers 0..4`)

After a deterministic edit to `engine/src/ops/stub_layers.cu` (and verified again on revert):

| Phase | Wall Time |
|---|---:|
| Workspace sync (`rsync --checksum --no-times`) | `514.8 ms` |
| Incremental `cmake --build --target forward` (1 NVCC compile + 1 link, no reconfigure) | `587.7 ms` |
| Resident `dlopen` hot-reload + GPU run IPC round trip | `151.5 ms` |
| CLI startup & orchestration overhead | `76.4 ms` |
| **Total external wall clock** | **`1,330.4 ms` (1.33 s)** |

### 3. Workbench Primitives & Unload Leak Verification (`short_math`)

- **Auto lazy recording**: First `forge run --layers 0..4` recorded entry/reference tensors from `baseline.so` (cache miss, `5.792 µs`); second call hit the cache directly (`5.536 µs`), matching baseline bit-for-bit.
- **CUDA Graph vs. eager toggle**: `run --e2e` ran at `51.040 µs` with CUDA Graph vs. `104.256 µs` with `--no-graph` (exact output parity).
- **Interleaved A/B**: `forge ab baseline stub-v1 --layers 0..4` ran 5 interleaved A-B-A-B rounds (`5.824 µs` vs. `6.752 µs`, exact outputs).
- **Zero-restart profiling**:
  - CUDA Event stage breakdown captured all 4 visit stages (`0..4`) and all 50 e2e stages (`embed` + 48 visits + `lm_head`).
  - `forge profile --layers 0..4 --nsys` captured a `93,642 B` `.nsys-rep` timeline covering all four visit kernels via profiler-API arming on the live resident PID without restarting the daemon.
- **`.safetensors` dump**: `forge dump --case short_math --layers 0..4 --with-state` emitted a `747,092 B` file (`input_ids`, `hidden_in`, `ref_hidden_out`, `cand_hidden_out`, `kv_state`, `conv_state`) verified readable by both C++ `SafetensorsFile` and Python `safetensors`.
- **VRAM leak & lease cleanup**: Ordinary `run` / `ab` / `profile` / `dump` cycles lost **0 B** on `dlclose` (`65,536 B` during Nsight capture, far below the 4 MiB warning limit). After `forge stop`, `dev status` reported `0 / 16,303 MiB` and `GPU lease: free`.

## Pitfall Caught & Fixed

- **macOS `rsync` protocol 29 mtime truncation**: Reverting a single `.cu` edit within the same second or preserving a Mac mtime older than the remote `.o` prevented Ninja from rebuilding `forward.so`. Switching `build_rsync_args` in `tools/common/remote.py` to `--checksum --no-times` compares file content and assigns fresh remote host timestamps to transferred files, guaranteeing every edit and revert triggers incremental compilation reliably.

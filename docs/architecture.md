# Flint-5080 系统架构

> **目标硬件**：单卡 Linux RTX 5080 16 GiB（Blackwell `sm_120a`），配合 Mac Laptop 远程开发  
> **目标场景**：面向 **Inkling 2 mini**（含 DFlash 2 speculative decoding）的个人本地 chat 与 coding-agent 推理服务，C1 极致低延迟优先、兼顾 C2，concurrency 不设硬上限。
> **延伸文档**：模型拓扑、op 语义与 VRAM/Roofline 账本见 [model.md](model.md)，压测协议与门禁见 [benchmark.md](benchmark.md)，优化追踪与实测证据见 [findings/](findings/README.md)，工程红线见 [AGENTS.md](../AGENTS.md)。

---

## 1. 设计原则与系统能力

### 1.1 核心原则

1. **单卡极致精简，拒绝重型抽象**  
   彻底放弃多卡、多硬件档位、多模型热切与重型跨语言 state machine，以最少代码量换取低认知负担、快编译与直接硬件掌控。
2. **面向极限调优的 op 自由与闭环验证**  
   engine 内部不引入锁死 op 演进的抽象——以标准 `safetensors`（支持 `layout_tag` 离线 repack）、静态 Arena 与整图 `forward.so` 秒级 hot-reload 消除重构阻力，并用 `forge`、`weights kld` 与 `bench` 做 latency/SOL 与数值精度闭环验证。

### 1.2 系统能力与扩展预留

**现有能力**
- **Inference engine 与 VRAM 管理**：以标准 `safetensors` 只读 `mmap` 一次载入并静态划分 VRAM Arena，支持 chunked prefill、request cancel 与静态 CUDA Graph。
- **DFlash 2 speculative decoding**：通过 `--draft-dir` 加载独立 BF16 drafter。prefill 仅为 prompt 末尾和中间 prefix snapshot 各自所需的最后 2047 个位置投影四路 target tap，生成各层的 context KV；单个 CUDA Graph 同时支持 greedy 和 stochastic path，每轮返回 7 个 draft token 及各自的 16 个 conditional candidate 和概率（共 924 B）。`--spec-verify-len` 控制 target 批量 verify 的宽度（含 anchor，2–8，默认 8），并按 accept length commit 或恢复 SWA 与 conv state。stochastic mode 在 rejection 处从 normalized `max(p−q, 0)` 采样；`--no-spec` 切换到 AR。
- **VRAM Prefix Cache**：在 VRAM 内复用多轮对话 prefix（含 target 与 drafter 的 SWA KV、target conv state 及末位 logits），exact hit 可零 prefill 直接进入 decode。
- **OpenAI 兼容 server 壳**：提供 Chat Completions（SSE / 非流式 / stop 字符串 / 超时 cancel）、离线 HF golden 对齐的 Tokenizer 与 Jinja 模板、独立 Reasoning/Tool-Call streaming parser 及免鉴权 `/health`、`/ready`、`/metrics`（Prometheus）。
- **调优工作台与精度守门**：通过 `forge` 实现常驻 weights + 整图 `forward.so` hot-reload、A/B 交错与分阶段 profile，配合 `weights golden`/`kld` 和 `bench` 完成数值与性能闭环验收。

**预留扩展**
- **OpenAI Responses API**：`server/` 生成与 streaming parser 层协议无关，可平滑挂载新协议路由。
- **Host KV-RAM cache**：纯 `engine/` 内部 state snapshot 扩展，不触碰跨语言 C ABI。
- **Vision 多模态**：`engine_api.h` 已预留 `MediaSlice` 参数位，当前跳过载入 vision weights。
- **`web/` 前端 UI**：`server/` 可按需挂载静态 `/ui/` 页面。

---

## 2. 分层架构与边界

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  工具链与评测层（tools/ — 轻量 Python，严格隔离于生产路径之外）             │
│  共享纯 stdlib 底座（.dev.toml / 远程同步 / 单卡 lease 锁 / JSON 归档）    │
│  • dev (运维/构建/检查)  • forge (热重载调优)  • bench (压测)  • weights │
└─────────────────────────────────────────────────────────────────────────┘
                                  │ HTTP (OpenAI API)
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  极薄服务壳（server/ — 单 Crate Rust，零调度/KV/CUDA/模型布局逻辑）        │
│  HTTP 协议、HF 对齐 tokenization/模板、独立 streaming parser、断连 cancel 与 /metrics 导出 │
└─────────────────────────────────────────────────────────────────────────┘
                 │ 唯一跨语言边界：engine/include/engine_api.h（Request ABI）
                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  完整推理引擎（engine/ — 纯 C++/CUDA，sm_120a，不依赖 Rust 编译）          │
│  静态 VRAM 分区、VRAM Prefix Cache、调度、AR/DFlash 2 forward 与 sm_120a op │
│  双运行形态：libengine.a（生产） / forge-host + forward.so（调优）         │
└─────────────────────────────────────────────────────────────────────────┘
```

- **跨语言契约（[engine_api.h](../engine/include/engine_api.h)）**：全部推理 runtime state 收归 `engine/` 闭环；跨语言仅暴露 Request 级 C ABI，以 JSON 配置初始化、按轮批量异步回调 token 并回传终态 telemetry 与 stop reason。`server/` 编译为无 Python 依赖的单二进制，对模型、KV 与 speculative decoding 完全无知，文本对齐由 `weights chat-golden` 离线锁死，客户端断连或超时由流内 guard 立即触发 `engine_cancel` 释放单卡。
- **双运行形态**：生产形态由 `server` 静态链接 `libengine.a`；调优形态由纯 C++ 常驻进程 `forge-host` 通过 `dlopen` hot-reload `forward.so`。两者共用同一份 Release（`-O3`、`sm_120a`）engine 核心，通过单卡 lease 锁互斥占用 5080。

---

## 3. Engine 架构不变量

engine 内部不固化特定 op 边界或调度实现，具体 dataflow 见 `engine/src/`、VRAM 账本见 [model.md](model.md)；任何演进只须守住三条结构不变量：

1. **全静态 VRAM 分区**：weights、KV/conv state、prefix snapshot 与 runtime workspace 在启动时按静态 Arena 预分配，推理 hot path 严禁动态 `cudaMalloc` / `cudaFree`。
2. **标准 `safetensors` 一体化**：只认标准 `safetensors` 目录，优化期通过带 `layout_tag` 元数据的离线 repack 满足硬件布局需求，不造私有容器。
3. **内部闭环与重构自由**：调度、prefix cache、CUDA Graph、speculative decoding 与 rejection rollback 全部内聚在 `engine/` 内部；只要对外保持 Request ABI 语义不变，内部 op、层衔接与内存布局可随优化自由演进。

---

## 4. 调优工作台与数值守门（`forge` + `weights`）

- **整图 `forward.so` 秒级 hot-reload**：放弃固定 op slot 插件 ABI，将完整 forward 编排、CUDA Graph、DFlash 2 与全部 op 统一链接为单一 `forward.so`（Forge ABI 见 [forward_abi.h](../engine/src/forge/forward_abi.h)），使跨 op 融合、跨层重排、KV 编码改动或双 stream overlap 均可无条件 hot-reload。借助单架构 `sm_120a`、关闭 `-rdc`、动态链接 CUDA runtime 与头文件隔离的 `ops/*.cu`，单 op 改动可秒级增量重编并进程内热替换。
- **区间懒录制与安全对照**：forward 接受 logical layer 区间 `(begin, end)`，首次局部 replay 时宿主自动用冻结的 `baseline.so` 截获真实 context 与 state snapshot 并缓存，零注册宏；每次 `dlopen`/`dlclose` 前后比对 VRAM 防止泄漏；内置 `--no-graph`、`--no-spec` 与 `return_logits` 零开销对照开关。
- Forge 在分配双模型 weights 和 scratch 后，按剩余 VRAM 扣除 I/O 和 512 MiB 预留量，再确定 KV 容量；可用 `--kv-seq-len` 显式指定。宿主重启保留 baseline。`forge freeze-baseline [candidate]` 原子替换 baseline 并清空 replay cache，`forge start --reset-baseline` 在启动时重新冻结当前构建。`--context-len N` 将用例循环 pad 至 N 个 token，chunked prefill。重复测量复用 VRAM 中的 Full KV，仅恢复当前步可写的 state。
- **三层闭环验证**：
  - `forge`（`start/stop/status`、`run`、`ab`、`profile`、`dump`、`ls/rm`）驱动 latency/SOL 测量与交错 A/B；speculative `run` 同时记录 `draft`、`verify` 和 `commit` 耗时的中位数；`profile --step 1` 输出 Verify 各阶段耗时，`profile --nsys` 在执行机汇总 kernel 和 CUDA API，`profile --compare` 在本地比较两份归档。
  - `weights golden`（`ops`/`e2e`/`dflash2`/`chat` 四级 fixture）锁死语义对齐，`weights kld` 经由常驻 `forge-host`（空闲时自动持锁拉起并回收）以 FP64 计算 teacher-forced KLD，记录两份二进制文件的 SHA-256，digest 相同时提示警告；
  - `bench` 以黑盒 HTTP 压测与配对统计执行端到端验收。

---

## 5. 工具链与顶层目录

开发/评测统一收归 `tools/`，共享零第三方依赖的 `tools/common/`（`.dev.toml` 单一配置、本地/远程 workspace 隔离同步、weights/`ccache` 全局共享、单卡 lease 进程监管与 `--json` / `.runs/` 归档），仅暴露 4 个顶层入口（参数以 `--help` 为准）：

- **`dev`**（纯 stdlib）：远程状态查询、增量构建与生产服务管理（含 `dev serve restart`）。`dev fmt` 统一本地格式化；`dev check --only` 可筛选本地或 GPU 验收项，支持 `ctest:<regex>`。
- **`forge`**（纯 stdlib）：驱动常驻 `forge-host` 的 hot-reload 调优，宿主源文件更新时提示重启。`forge ptxas [files...] [--all]` 根据 compile_commands 单独编译 CUDA 单元并报告各 kernel 的 registers、shared memory 与 spill bytes。默认检查改动的 CUDA 文件，无改动时检查全部 op。
- **`bench`**（轻量异步 HTTP）：黑盒 OpenAI HTTP 压测、canary 回归与配对统计对比。
- **`weights`**（纯 stdlib 前端，离线生成走独立 PyTorch 环境）：weights 检视、四级 golden fixture、teacher-forced KLD 与离线 layout repack。

| 顶层目录 | 职责 |
|---|---|
| `bin/` | `dev`、`forge`、`bench`、`weights` 四个薄 CLI 入口包装 |
| `engine/` | 纯 C++/CUDA `sm_120a` inference engine、独立 CMake/Ninja 构建与 C++ 测试 |
| `server/` | 单 Crate Rust HTTP 服务壳与协议/文本对齐测试 |
| `tools/` | 共享 stdlib 底座与 4 个 CLI 实现（严禁进入生产运行路径） |
| `workloads/` | 固定压测、canary、chat 与模型 golden 用例清单 |
| `docs/` | 活文档（`architecture.md`、`model.md`、`benchmark.md`）与 `findings/` 实测档案库 |

编译速度与跨层边界红线（`ops/*.cu` 头文件隔离、克制 `sm_120a` 模板实例化、hot path 零动态分配、跨层隔离）由 [AGENTS.md](../AGENTS.md) 统一约束。

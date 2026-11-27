# Flint-5080 系统架构

> **目标硬件**：单卡 Linux RTX 5080 16 GiB（Blackwell `sm_120a`），配合 Mac Laptop 远程开发  
> **目标场景**：面向 **Inkling 2 mini**（含 `dspark` 投机解码）的个人本地 chat 与 coding-agent 推理服务，C1 极致低延迟优先、兼顾 C2，并发不设硬上限。  
> **延伸文档**：模型拓扑、算子语义与显存/Roofline 账本见 [model.md](model.md)，压测协议与门禁见 [benchmark.md](benchmark.md)，优化追踪与实测证据见 [findings/](findings/README.md)，工程红线见 [AGENTS.md](../AGENTS.md)。

---

## 1. 设计原则与系统能力

### 1.1 核心原则

1. **单卡极致精简，拒绝重型抽象**  
   彻底放弃多卡、多硬件档位、多模型热切与重型跨语言状态机，以最少代码量换取低认知负担、快编译与直接硬件掌控。
2. **面向极限调优的算子自由与闭环验证**  
   引擎内部不引入锁死算子演进的抽象——以标准 `safetensors`（支持 `layout_tag` 离线重排）、静态 Arena 与整图 `forward.so` 秒级热重载消除重构阻力，并用 `forge`、`weights kld` 与 `bench` 做延迟/SOL 与数值精度闭环验证。

### 1.2 系统能力与扩展预留

**现有能力**
- **推理引擎与显存管理**：以标准 `safetensors` 只读 `mmap` 一次载入并静态划分显存 Arena，支持 chunked prefill、请求取消与静态 CUDA Graph。
- **DSpark 投机解码**：默认加载 MTP 权重执行多 token 草拟与批验证，支持精确状态回滚与贪心/无偏随机拒绝采样，`--no-spec` 可切换回 AR。
- **VRAM Prefix Cache**：在显存内复用多轮对话前缀（含滑窗 KV、卷积状态、末位 logits 与 tap），精确命中可零 prefill 直接进入解码。
- **OpenAI 兼容服务壳**：提供 Chat Completions（SSE / 非流式 / stop 字符串 / 超时取消）、离线 HF 黄金对齐的 Tokenizer 与 Jinja 模板、独立 Reasoning/Tool-Call 流式解析器及免鉴权 `/health`、`/ready`、`/metrics`（Prometheus）。
- **调优工作台与精度守门**：通过 `forge` 实现常驻权重 + 整图 `forward.so` 热重载、A/B 交错与分阶段剖析，配合 `weights golden`/`kld` 和 `bench` 完成数值与性能闭环验收。

**预留扩展**
- **OpenAI Responses API**：`server/` 生成与流式解析层协议无关，可平滑挂载新协议路由。
- **Host KV-RAM 缓存**：纯 `engine/` 内部状态快照扩展，不触碰跨语言 C ABI。
- **Vision 多模态**：`engine_api.h` 已预留 `MediaSlice` 参数位，当前跳过载入视觉权重。
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
│  HTTP 协议、HF 对齐分词/模板、独立流式解析、断连取消与 /metrics 导出       │
└─────────────────────────────────────────────────────────────────────────┘
                 │ 唯一跨语言边界：engine/include/engine_api.h（Request ABI）
                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  完整推理引擎（engine/ — 纯 C++/CUDA，sm_120a，不依赖 Rust 编译）          │
│  静态显存分区、VRAM Prefix Cache、调度、AR/DSpark 前向与 sm_120a 算子     │
│  双运行形态：libengine.a（生产） / forge-host + forward.so（调优）         │
└─────────────────────────────────────────────────────────────────────────┘
```

- **跨语言契约（[engine_api.h](../engine/include/engine_api.h)）**：全部推理运行时状态收归 `engine/` 闭环；跨语言仅暴露 Request 级 C ABI，以 JSON 配置初始化、按轮批量异步回调 token 并回传终态遥测与停止原因。`server/` 编译为无 Python 依赖的单二进制，对模型、KV 与投机完全无知，文本对齐由 `weights chat-golden` 离线锁死，客户端断连或超时由流内 guard 立即触发 `engine_cancel` 释放单卡。
- **双运行形态**：生产形态由 `server` 静态链接 `libengine.a`；调优形态由纯 C++ 常驻进程 `forge-host` 通过 `dlopen` 热加载 `forward.so`。两者共用同一份 Release（`-O3`、`sm_120a`）引擎核心，通过单卡 lease 锁互斥占用 5080。

---

## 3. 引擎架构不变量

引擎内部不固化特定算子边界或调度实现，具体数据流见 `engine/src/`、显存账本见 [model.md](model.md)；任何演进只须守住三条结构不变量：

1. **全静态显存分区**：权重、KV/卷积状态、前缀快照与运行时工作区在启动时按静态 Arena 预分配，推理热路径严禁动态 `cudaMalloc` / `cudaFree`。
2. **标准 `safetensors` 一体化**：只认标准 `safetensors` 目录，优化期通过带 `layout_tag` 元数据的离线派生重排满足硬件布局需求，不造私有容器。
3. **内部闭环与重构自由**：调度、前缀缓存、CUDA Graph、投机解码与拒绝回滚全部内聚在 `engine/` 内部；只要对外保持 Request ABI 语义不变，内部算子、层衔接与内存布局可随优化自由演进。

---

## 4. 调优工作台与数值守门（`forge` + `weights`）

- **整图 `forward.so` 秒级热重载**：放弃固定算子槽位插件 ABI，将完整前向编排、CUDA Graph、DSpark 与全部算子统一链接为单一 `forward.so`（Forge ABI 见 [forward_abi.h](../engine/src/forge/forward_abi.h)），使跨算子融合、跨层重排、KV 编码改动或双流重叠均可无条件热加载。借助单架构 `sm_120a`、关闭 `-rdc`、动态链接 CUDA 运行时与头文件隔离的 `ops/*.cu`，单算子改动可秒级增量重编并进程内热替换。
- **区间懒录制与安全对照**：前向接受逻辑层区间 `(begin, end)`，首次局部回放时宿主自动用冻结的 `baseline.so` 截获真实上下文与状态快照并缓存，零注册宏；每次 `dlopen`/`dlclose` 前后比对显存防止泄漏；内置 `--no-graph`、`--no-spec` 与 `return_logits` 零开销对照开关。
- 宿主重启保留基线。`forge freeze-baseline [candidate]` 原子替换基线并清空回放缓存，`forge start --reset-baseline` 在启动时重新冻结当前构建。`--context-len N` 将用例循环填充至 N 个 token，分块预填充。重复测量复用显存中的 Full KV，仅恢复当前步可写的状态。
- **三层闭环验证**：
  - `forge`（`start/stop/status`、`run`、`ab`、`profile`、`dump`、`ls/rm`）驱动延迟/SOL 测量与交错 A/B；`profile --step 1` 输出 Verify 各阶段耗时，`profile --nsys` 在执行机汇总 kernel 和 CUDA API，`profile --compare` 在本地比较两份归档。
  - `weights golden`（`ops`/`e2e`/`mtp`/`chat` 四级夹具）锁死语义对齐，`weights kld` 经由常驻 `forge-host`（空闲时自动持锁拉起并回收）以 FP64 计算 teacher-forced KLD，记录两份二进制文件的 SHA-256，摘要相同时提示警告；
  - `bench` 以黑盒 HTTP 压测与配对统计执行端到端验收。

---

## 5. 工具链与顶层目录

开发/评测统一收归 `tools/`，共享零第三方依赖的 `tools/common/`（`.dev.toml` 单一配置、本地/远程工作区隔离同步、权重/`ccache` 全局共享、单卡 lease 进程监管与 `--json` / `.runs/` 归档），仅暴露 4 个顶层入口（参数以 `--help` 为准）：

- **`dev`**（纯 stdlib）：远程状态查询、增量构建与生产服务管理（含 `dev serve restart`）。`dev fmt` 统一本地格式化；`dev check --only` 可筛选本地或 GPU 验收项，支持 `ctest:<regex>`。
- **`forge`**（纯 stdlib）：驱动常驻 `forge-host` 的热重载调优，宿主源文件更新时提示重启。`forge ptxas [files...] [--all]` 根据编译数据库单独编译 CUDA 单元并报告各 kernel 的寄存器、共享内存与溢出字节数。默认检查改动的 CUDA 文件，无改动时检查全部算子。
- **`bench`**（轻量异步 HTTP）：黑盒 OpenAI HTTP 压测、canary 回归与配对统计对比。
- **`weights`**（纯 stdlib 前端，离线生成走独立 PyTorch 环境）：权重检视、四级黄金夹具、teacher-forced KLD 与离线布局重排。

| 顶层目录 | 职责 |
|---|---|
| `bin/` | `dev`、`forge`、`bench`、`weights` 四个薄 CLI 入口包装 |
| `engine/` | 纯 C++/CUDA `sm_120a` 推理引擎、独立 CMake/Ninja 构建与 C++ 测试 |
| `server/` | 单 Crate Rust HTTP 服务壳与协议/文本对齐测试 |
| `tools/` | 共享 stdlib 底座与 4 个 CLI 实现（严禁进入生产运行路径） |
| `workloads/` | 固定压测、canary、chat 与模型黄金用例清单 |
| `docs/` | 活文档（`architecture.md`、`model.md`、`benchmark.md`）与 `findings/` 实测档案库 |

编译速度与跨层边界红线（`ops/*.cu` 头文件隔离、克制 `sm_120a` 模板实例化、热路径零动态分配、跨层隔离）由 [AGENTS.md](../AGENTS.md) 统一约束。

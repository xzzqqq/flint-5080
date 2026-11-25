# Flint-5080 总体架构设计

> **状态**：架构基线（Model-Agnostic Blueprint）  
> **目标硬件**：单卡 Linux RTX 5080 16 GB（Blackwell `sm_120a`），配合 Mac Laptop 远程开发  
> **目标模型与场景**：面向 **Inkling 2 mini**（含 `dspark` 投机解码）的个人本地 chat 与 coding-agent 推理服务，追求 **C1 极致低延迟（decode 逼近 SOL）**，兼顾 C2，并发不设硬上限。  
> **范围说明**：本文档沉淀与具体模型细节解耦的总体架构、模块边界、热重载调优台、工具链与工程规范；模型结构、算子排布与 `dspark` 算法细节将在后续文档中补充。

---

## 1. 设计原则与核心取舍

1. **单卡极致精简，拒绝重型抽象**  
   彻底放弃多卡、多硬件档位、多模型热切、发行版级运维设施以及跨语言重型状态机。以最少的代码量换取最低的认知负担、最快的编译速度和最直接的硬件掌控力（面向 Agent-Native 高频迭代）。
2. **单一真相源，边界画在 Request 级**  
   推理引擎的一切运行时状态（权重、显存池、KV/Prefix Cache、调度循环、CUDA Graph、`dspark`）全部收归 C++/CUDA 引擎内部闭环；Rust 服务壳只做协议与文本前/后处理，通过极窄的 Request 级 C ABI 交互，消除跨语言状态同步债。
3. **调优工作台（`forge`）作为一等公民**  
   将“常驻权重 + 整图 `forward.so` 秒级热重载 + 自动懒录制 + 交错 A/B + Roofline 剖析”内建为引擎原生运行形态，而非生产服务的后期补丁。
4. **严格隔离生产路径与外围脚本**  
   生产服务是干净的单二进制（零 Python 依赖）；Python 严格限定在开发、远程编排、黑盒压测与离线权重/参考对齐工具链中，永不侵入在线热路径。

---

## 2. 技术栈与三层职责划分

```text
┌──────────────────────────────────────────────────────────────────────┐
│  工具链与评测层（tools/ — 轻量 Python，仅开发/实验/离线使用）            │
│  4 个顶层命令共享纯 stdlib 底座（配置 / 远程同步 / 单卡 lease 锁 / JSON） │
│  • dev (status/build/serve/check)   • forge (常驻热重载调优工作台)     │
│  • bench (黑盒 HTTP 压测 & canary)  • weights (safetensors/golden/kld) │
└──────────────────────────────────────────────────────────────────────┘
                                 │ HTTP (OpenAI API)
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  极薄服务壳（server/ — 单 Crate Rust，零调度/KV/CUDA 逻辑）             │
│  • OpenAI Chat Completions (SSE / 非流式 / 取消 / 超时) + P1 Responses │
│  • Tokenizer & Jinja Chat Template（离线 HF golden fixture 锁死对齐）  │
│  • 独立可插拔 Reasoning / Tool-Call 流式解析器                         │
│  • 极简 CLI/TOML 配置、/health、/ready、/v1/models、可选 API Key       │
└──────────────────────────────────────────────────────────────────────┘
                     │ 唯一跨语言边界：engine_api.h（Request 级 C ABI）
                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│  完整推理引擎（engine/ — 纯 C++/CUDA，sm_120a，不依赖 Rust 编译）       │
│  • safetensors 只读 mmap loader + 静态显存分区（Arena）                │
│  • VRAM Prefix Cache + 低并发调度（C1 极致、兼顾 C2、超额自然排队）      │
│  • 模型前向（AR baseline + dspark 投机解码）+ CUDA Graph + sm_120a 算子│
│  • 双运行形态：libengine（生产） / forge-host + forward.so（调优）      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. 功能范围与边界（P0 / P1 / Non-goals）

### 3.1 P0（首期必须实现）

| 能力 | 归属层 | 设计要点 |
|---|---|---|
| **OpenAI Chat Completions** | `server/` | SSE 流式与非流式；客户端断连即触发引擎取消；支持请求超时。 |
| **Reasoning & Tool-Call 解析** | `server/` | 独立、可插拔的流式状态机模块，不与 HTTP handler 耦合。 |
| **Tokenizer & Chat Template** | `server/` | `tokenizers` + `minijinja`，由离线 Python 脚本生成的 HF golden fixtures 在 `dev check --local` 中强制锁死 100% 一致。 |
| **极简服务基础** | `server/` | `/health`、`/ready`、`/v1/models`、可选 `--api-key`、结构化 `tracing` 日志、`clap` + 单配置文件。 |
| **单卡低并发推理引擎** | `engine/` | 优化 C1 延迟、有余力优化 C2，不写死并发上限（超额自然排队）；支持 chunked prefill、请求取消与 CUDA Graph。 |
| **VRAM Prefix Cache** | `engine/` | 针对 coding-agent 高前缀复用率场景内置显存级前缀共享。 |
| **投机解码（`dspark`）** | `engine/` | 属于 P0 目标能力；Bootstrap 阶段先跑通 AR baseline，引擎步进结构原生兼容多 token verify，随后接入 `dspark`。 |
| **一等公民 `forge` 工作台** | `engine/` + `tools/` | 独立常驻宿主 + `forward.so` 热重载 + 5 组正交原语（详见第 6 节）。 |
| **4 入口统一底座 CLI 与评测** | `tools/` | `dev` / `forge` / `bench` / `weights`，含固定 coding-agent trace 压测与 canary 回归。 |

### 3.2 P1（首期仅预留干净接口，按需补齐）

| 能力 | 预留方式 |
|---|---|
| **OpenAI Responses API（HTTP + WebSocket）** | `server/` 路由与流式输出层保持协议无关抽象，后续平滑挂载。 |
| **Host KV-RAM 缓存** | 纯 `engine/` 内部状态 `snapshot/restore` 扩展，**不暴露到 C ABI**。 |
| **Vision 多模态** | `engine_api.h` 提交接口预留 `MediaSlice` 参数位（P0 传 `NULL, 0`）；`server/` 预留媒体解码位。 |
| **基础 Prometheus Metrics & `web/` UI** | 引擎通过 `engine_stats_json()` 暴露指标快照，`server/` 后续按需挂 `/metrics` 和静态 `/ui/`。 |

### 3.3 Non-goals（明确移除，绝不带入）

- **Idle Unload**、在线模型切换、模型目录/下载/删除 HTTP API（单模型单卡，直接重启与外部下载）。
- **KV-Disk**（SSD 落盘 / Direct I/O / CRC32）。
- **在线配置热更**（`PATCH /v1/config`）、多卡/多硬件 profile 矩阵。
- **Swagger / OpenAPI 自动代码生成**。
- **数千行模型特化启发式补丁**（如重型 `loop_guard`；先依赖标准 `max_tokens` 与 stop token）。
- **私有权重容器格式**（如 `.ninfer`）与独立 `Makefile` 体系。
- **重型纯 CPU `MockCompute` 状态机测试**（Rust 侧仅需 ~50 行假回调 stub 即可完整覆盖所有 HTTP/SSE/取消/解析单测）。

---

## 4. 权重格式与加载：`safetensors` 一体化

- **唯一格式**：统一只认标准 `safetensors` 目录（`config.json` + `tokenizer.json` + `*.safetensors` + 可选 index）。
- **极简加载器**：`engine/` 内置约 200–300 行只读 `safetensors` `mmap` loader，按规划直接拷贝/映射到 VRAM 或 pinned host memory。
- **优化期重排（Layout Repack）不造新容器**：
  - AR baseline 阶段直接加载原始/量化后的标准 `safetensors`，零转换成本；
  - 后续面向 `sm_120a` 做融合权重、量化 scale 或 MMA 布局重排时，由 `weights repack` 离线输出**派生 `safetensors` 目录**，通过文件 metadata 标明 `layout_tag`（如 `sm120_v1`）。
  - 永远只需维护一套 loader，且 Python 生态（`safetensors` / `torch`）可直接检查与修改任意权重文件。

---

## 5. Rust ↔ C++ 边界：单头文件 Request 级 C ABI

全仓库唯一的跨语言接口收敛至 `engine/include/engine_api.h`（~150 行 C99），参考业界现代引擎（vLLM V1 / SGLang / TensorRT-LLM Executor）的 Request / Token-Stream 边界：

```c
// 1. 生命周期（配置使用 JSON 字符串，字段演进不破 ABI）
EngineHandle* engine_create(const char* config_json, char** err_out);
void          engine_destroy(EngineHandle* h);

// 2. 异步提交请求（引擎内部独立线程驱动调度、prefill、decode 与 dspark）
//    media / n_media 为 P1 Vision 预留（P0 传 NULL, 0）
//    stop_token_ids 下沉到引擎立即终止 decode；stop_strings 与 UTF-8/tool/reasoning 解析留 Rust
uint64_t engine_submit(
    EngineHandle* h,
    const RequestParams* params,
    const int32_t* token_ids, size_t n_tokens,
    const MediaSlice* media, size_t n_media,
    TokenCallback cb, void* user_data
);

// 3. 取消与可观测性
void  engine_cancel(EngineHandle* h, uint64_t req_id);
char* engine_stats_json(EngineHandle* h); // 返回 JSON 快照供 /ready 与 P1 /metrics
```

**接口契约要点**：
- **Rust 侧对底层完全无知**：不感知层数、KV 格式/页表、投机解码细节或 Host KV-RAM。
- **回调事件**：每次产出 token 或结束时回调极小结构体，完成事件附带**服务端精确遥测**（`prefill_tokens`、`cached_prefix_tokens`、`decode_tokens`、`prefill_ms`、`decode_ms`、`decode_rounds`、`spec_accepted`）及**结构化错误码**（`ContextOverflow` / `InvalidInput` / `Cancelled` / `InternalError`）。
- **零热路径开销**：每轮 decode（10–20 ms）仅回调一次推送本轮产出的 1~K 个 token IDs，Rust 侧转发至 `tokio::sync::mpsc` 驱动 SSE。

---

## 6. 一等公民调优工作台：`forge`

### 6.1 两种互斥运行形态（共用同一份 Release 引擎核心）

```text
【生产形态】  server（Rust 薄壳 + 静态链接 libengine）
             → 纯粹、稳定、零热重载与实验负担，面向日常使用

【调优形态】  forge-host（纯 C++ 常驻守护进程，不依赖 Rust 编译）
                └── dlopen ──▶ forward.so（完整前向编排 + CUDA Graph + dspark + 全部 sm_120a 算子）
             → 宿主跨命令常驻 5080 显存（Weights Arena + KV/State Arena + Scratch Workspace Arena）
             → 改计算代码仅增量编单文件并链接为新 forward.so，进程内 3–6 秒热替换生效
```

- **`forge-host` 与 `forward.so` 均为全速 Release（`-O3`、`sm_120a`、CUDA Graph 全开）构建**，绝非 Debug 构建。
- 两种形态互斥使用 5080（由单卡 lease 锁强制防撞车）。

### 6.2 整图 `forward.so` 热重载的关键机制

- **为什么不采用“固定算子槽位 ABI”**：固定槽位在面对跨算子融合、跨层 PDL、双流重叠或 KV 编码变更时极易失效并堆积大量胶水。将完整计算执行体统一链接为单一 `forward.so` 后，**任何计算侧重构均可无条件热加载**。
- **体积与编译速度保障**：
  - 仅编译 `sm_120a` 单架构；CUDA 运行时/数学库动态链接在 `forge-host` 中（不静态编入 `forward.so`）；关闭跨文件设备链接（`-rdc=false`）。
  - 单个 `forward.so` 体积控制在 **5–25 MB**；改单个算子 `.cu` 仅增量编译 1 个编译单元（2–5 s）+ 主机链接（<1 s）+ 热重载（<0.2 s），全流程 **3–6 秒**。
- **无状态算子与零宏层区间执行（`run_layers(begin, end)`）**：
  - 算子本身保持纯无状态函数（不感知层号）；外层前向循环接收 `(begin, end)` 起止参数——完整前向即 `run_layers(0, N)`，Fast Tier 局部回放即 `run_layers(k, m)`。
  - **兼容跨层融合**：区间内部（如 `0..4` 内的相邻层衔接）允许任意跨层融合（fused residual+norm、跨层 PDL 等），仅在区间首尾对齐标准残差流。
  - **自动懒录制**：首次对 `(case, layers)` 执行局部回放时，宿主自动用基准 `baseline.so` 跑一遍真实请求，在 `begin` 层入口自动截获真实 `hidden_state`、位置信息及该区间 KV/状态并缓存，同时存下 `end` 层参考输出；后续所有迭代直接命中缓存复用，**零注册宏、零手工前置**。
- **显存安全与调试对照开关**：
  - **静态 Arena + 热重载泄漏检查**：热路径严禁动态 `cudaMalloc`；`forward.so` 只能使用宿主提供的 Workspace Arena，所有 Graph/stream/event 挂在统一 `ForwardContext` 析构。宿主在 `dlopen` 前与 `dlclose` 后调用 `cudaMemGetInfo` 比对剩余显存，超阈值（如 >4 MB）直接打印 `WARN`。
  - **内置零开销对照选项（`ForwardOptions`）**：支持 `use_cuda_graph`（`--no-graph` 退回 eager）、`use_speculative`（`--no-spec` 退回 AR-only）与 `return_logits`（供 teacher-forcing KLD 与 `dspark` 数学一致性验证）。

### 6.3 `forge` 核心原语（5 组正交命令 + 候选管理）

| 原语 | 职责 | 典型用法 |
|---|---|---|
| **`start` / `stop` / `status`** | 拉起/关闭/查询 `forge-host` 常驻宿主并占/释 5080 lease 锁（自动冻结 `baseline.so`，检测到 `nsys` 时预武装 profiler-API 采集） | `forge start` / `forge status` / `forge stop` |
| **`run`** | 自动增量编并热加载 `forward.so`，输出耗时、5080 SOL 占比与数值误差（支持 `--save <name>` 保存不可变 `.so` 候选） | `forge run --layers 0..4`<br>`forge run --e2e [--full] [--no-graph] [--no-spec]`<br>`forge run --e2e --save fast-v1` |
| **`ab`** | 同常驻会话内交替加载两个候选 `.so` 做 A-B-A-B 交错测量，消除热机与频率漂移 | `forge ab baseline current --layers 0..4`<br>`forge ab fast-v1 fast-v2 --e2e --full` |
| **`profile`** | 默认零重启开启轻量 CUDA Event 阶段标尺，输出分阶段耗时与 960 GB/s Roofline 占比；加 `--nsys` 采集 Nsight Systems timeline | `forge profile --layers 0..1`<br>`forge profile --e2e --nsys` |
| **`dump`** | 指定输入请求（`--case` / `--prompt`）按需导出真实输入、参考/候选输出及可选状态为 `.safetensors`，供 Python 排错或独立 `ncu` 单算子微调 | `forge dump --case short_bug`<br>`forge dump --case long_64k --layers 17..18 --step 5 --with-state` |
| **`ls` / `rm`** | 列出或清理保存在远程 `.forge/candidates/` 下的候选 `.so` | `forge ls` / `forge rm fast-v1` |

---

## 7. 工具链与 CLI 设计（共享底座 + 4 独立入口）

彻底取消 `Makefile` 与散装脚本，底层只留标准 `engine/CMakeLists.txt`（Ninja）与 `server/Cargo.toml`。所有开发/评测能力收归 `tools/`，由**纯 Python stdlib 底座（`tools/common/`）+ 4 个职责清晰、依赖隔离的顶层命令**组成：

### 7.1 共享底座（`tools/common/`，零第三方依赖）
- **单一配置源**：本地 `.dev.toml`（不进 git，仓库提供 `.dev.toml.example`）管理远程 5080 SSH、工作目录、模型权重路径与默认 workload。
- **本地/远程透明一体**：在 Laptop 执行需 GPU/CUDA 的命令时自动增量同步源码 → 校验单卡锁 → 远程执行 → 回传摘要；在 Linux 5080 本机执行完全相同的命令时自动走本机路径。
- **工作区隔离 + 大资产共享**：远程源码与 `build/` 按本地 worktree/workspace 名隔离（支持多分支/多 Agent 并行编辑不互踩）；模型权重目录与 `ccache` 缓存全局共享。
- **带 Lease 的单卡锁**：`forge start` / `serve start` 共用轻量的 5080 占用锁（记录 owner、PID、heartbeat/lease），防止进程崩掉后死锁，也防止 `forge` 与 `server` 撞爆 16 GB 显存。
- **Agent 原生输出契约**：终端默认输出简洁摘要；加 `--json` 时 **`stdout` 仅输出稳定 Schema 的纯净 JSON**（编译与同步日志走 `stderr`），每次运行自动归档至 `.runs/<id>.json`。

### 7.2 四个顶层入口（严格克制，禁止新增散装脚本）

| 命令 | 依赖边界 | 职责 |
|---|---|---|
| **`dev`** | 纯 stdlib | 工程骨架与服务运维：`status`（5080 占用/温度/显存/同步状态）、`build`（`--engine`/`--server`/`--all`）、`serve`（`start`/`stop`/`logs`）、`check`（`--local` Mac 秒级单测 + `--gpu` 远程验收）、`ssh`（逃生舱）。 |
| **`forge`** | 纯 stdlib | 驱动远端/本机 `forge-host` 的全部调优原语（`start/stop/status/run/ab/profile/dump/ls/rm`）。 |
| **`bench`** | 轻量异步 HTTP & 统计 | 纯黑盒 HTTP 压测器（支持 `--url` 测自研或任意 OpenAI 兼容竞品）：固定 coding-agent trace、TTFT、Canary 回归、paired geometric mean + bootstrap CI 报表。以服务端计时为主、客户端校准为辅。 |
| **`weights`** | 纯 stdlib 前端；离线生成子进程通过 `weights.python` 指向独立 PyTorch 环境 | 低频离线权重与参考对齐：`inspect`、`repack`（输出 `safetensors`）、`golden`（持单卡 lease 生成算子与 48 逻辑步 prefill/decode 黄金夹具，详见 [model.md](model.md)）、`kld`（调用常驻引擎做 teacher-forcing KLD 评测）。 |

---

## 8. 仓库目录结构与编码/构建规范

### 8.1 目录布局

```text
flint-5080/
├── bin/                 # 4 个薄入口包装：dev, forge, bench, weights（可加入 PATH 或 ./bin/... 调用）
├── .dev.toml.example    # 唯一环境配置模板
├── AGENTS.md            # Agent-Native 行为宪法、架构边界红线、数值准则与文档地图
├── engine/              # 纯 C++/CUDA 引擎（独立 CMake + Ninja，不依赖 Rust）
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── engine_api.h # 唯一对外 Request 级 C ABI
│   ├── src/
│   │   ├── common/      # 零第三方依赖的轻量内部公共头（如 JSON 解析）
│   │   ├── loader/      # safetensors mmap loader + 静态 Arena 显存规划
│   │   ├── runtime/     # VRAM Prefix Cache, 轻量调度循环, CUDA Graph, P1 KV-RAM 扩展位
│   │   ├── model/       # 前向编排 run_layers(begin, end), AR + dspark verify
│   │   ├── ops/         # sm_120a 无状态 CUDA 算子（严格单文件头文件隔离）
│   │   └── forge/       # forge-host 常驻宿主、dlopen 热重载、懒录制缓存、显存泄漏检测
│   ├── third_party/     # 版本锁定的外部 CUDA 库（如 CUTLASS/CuTe 等）
│   └── tests/           # 引擎与算子 C++ 测试
├── server/              # 极薄单 Crate Rust HTTP 服务
│   ├── Cargo.toml
│   └── src/             # http(chat / P1 responses), tokenizer, template, parser(reasoning/toolcall), engine_ffi
├── tools/               # Python CLI 包（严禁进入生产运行路径）
│   ├── common/          # 纯 stdlib 底座（config, remote sync, gpu lease lock, json output）
│   ├── dev_cli/         # dev 命令实现
│   ├── forge_cli/       # forge 命令实现
│   ├── bench_cli/       # bench 命令实现
│   ├── weights_cli/     # weights 命令实现
│   └── tests/           # pytest 自测（纳入 dev check --local）
├── workloads/           # 固定 coding-agent 压测 trace、canary 与 forge 标准用例
├── web/                 # (P1) 可选前端 UI
└── docs/
    ├── architecture.md  # 活蓝图（边界/工具链变更时原地更新，无 ADR）
    ├── model.md         # (后续) 活文档：模型结构、权重布局与 5080 SOL 账本
    ├── benchmark.md     # (后续) 固定评测口径、workload 与验收门禁
    └── findings/        # 按日期追加的 GPU 实测证据库（含正/负实验与死胡同记录）
```

### 8.2 编译速度与代码卫生红线（写入工程规范）

1. **头文件物理隔离**：`engine/src/ops/*.cu` 仅依赖极薄的纯 C POD 参数头文件，禁止构造“万能 `common.cuh`”，保护改单算子 2–5 秒增量编译红利。
2. **克制模板实例化**：严禁在单个 `.cu` 中展开无用的多架构/多形状笛卡尔积模板，只实例化 `sm_120a` 实际需要的配置。
3. **默认快构工具链**：CMake + Ninja + `ccache` + 快速链接器（`lld`/`mold`），关闭 `-rdc`。
4. **热路径零动态分配**：推理循环内禁止 `cudaMalloc`/`cudaFree`，所有临时缓冲来自预分配 `Scratch Workspace Arena`。
5. **严守跨层边界**：`server/` 禁止引入调度/KV/张量布局逻辑；`tools/` 外禁止散落一次性脚本。

---

## 9. Bootstrap 推进路线图

| 阶段 | 核心目标 | 验收标志 |
|---|---|---|
| **Phase 0：工程骨架与工具底座** | 建好目录、空壳 CMake/Cargo、`tools/common/`（远程同步 + 单卡 lease 锁 + JSON 输出）与 `dev status / build / check --local` | Mac 一键同步并在 5080 增量编通空壳 `libengine` + `forge-host` + `server` |
| **Phase 1：C++ AR Baseline + `forge` 闭环** | 实现 `safetensors` loader、AR 前向、`weights golden` 逐层对齐、`forge start/run/ab/profile/dump` 热重载闭环 | `forge run --e2e` 与 HF 参考 logits/输出对齐；改单算子 6 秒内热重载生效 |
| **Phase 2：Rust 薄壳 + `bench` 端到端** | 打通 `engine_api.h`、Chat Completions (SSE + reasoning + tool calls)、HF tokenizer/template golden 单测、VRAM Prefix Cache 与 `bench` | 真实 coding agent 客户端可流畅对话与调用工具；`bench` 跑出首份冻结基线 |
| **Phase 3：`dspark` 投机解码与 5080 SOL 优化** | 接入 `dspark` verify 路径，全面进入 `forge` 高频优化循环（算子融合、量化/KV 压缩、PDL、布局重排）；按需补齐 P1 能力 | 端到端 decode 稳定逼近 RTX 5080 硬件带宽极限，canary 与 KLD 回归绿灯 |

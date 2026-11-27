# Flint-5080 模型规格与 RTX 5080 硬件账本

> **锁定权重**：`Maxrio/Inkling-2-Mini-exl3` 分支 `3.3bpw`（基座 `thinkingmachines/Inkling-2-Mini` 修订 `ed40e7da`，提交 `b57f9b6c`，15 文件 `13.73 GB`，相对官方混合 NVFP4/MXFP8 参考 teacher-forced KL **0.0121**、Top-1 **95.9%**）。  
> **文档边界**：本文档仅沉淀高频查阅的模型核心规格、非标准结构速记以及 16 GiB 显存 / 960 GB/s Roofline 物理账本。逐行精确数学公式见 [\_inkling2\_ref.py](../tools/weights_cli/_inkling2_ref.py)、[\_dflash2\_ref.py](../tools/weights_cli/_dflash2_ref.py) 与 [exl3\_linear.cu](../engine/src/ops/exl3_linear.cu)；完整张量清单以 `./bin/weights inspect --tensors` 和 [arenas.cpp](../engine/src/loader/arenas.cpp) 为准。

---

## 1. 拓扑速查与非标准结构

### 1.1 核心维度（`text_config` + `dflash_config`）

| 维度 | 文本主干（`model.safetensors`，11.57 GiB） | DFlash 2 草拟器（独立目录的 `model.safetensors`，944.11 MiB） |
|---|---|---|
| 深度与循环 | `24` 物理层，中间 `6..17` 循环 3 遍共 **48 逻辑步**（`visit 0..47`） | `5` 层 Dense BF16 Decoder，配置草拟器目录时载入（`--no-spec` 跳过） |
| 注意力排布 | **40 SWA (`window=1024`) + 8 Full**（物理层 `5,11,17,23`） | 5 SWA（每个 query 最多回看 2047 个位置，块内双向可见） |
| 隐藏维与头 | `hidden=2304`，GQA `24Q / 8KV`，`head_dim=128` | 32Q / 8KV，头维 128，Dense `intermediate=6912` |
| FFN / MoE | `128` 路由专家 / `top-8` / `0` 共享 / `intermediate=1152` | Dense 标准 SwiGLU（无 softcap） |
| 词表与采样点 | `vocab=201024`（有效 `200058`），`embed`/`unembed` 不共享，softcap `30.0` | 共用主干 `embed`/`embed_norm`/`unembed[:200058]`；采样逻辑步 `[5,23,41,48]` |
| 量化构成 | EXL3 `mul1`（偶专家 `3.0b`、奇专家 `3.5b`、Attention `6.0b`）+ 其余 BF16/F32 | 81 个 BF16 张量，候选选择器 rank=256，top-k=16 |

### 1.2 编写算子最易踩坑的 5 项非标准特征

1. **目标 Attention（无 RoPE）**：采用学习相对偏置（`d_rel=16, rel_extent=1024`）+ Q/K 按头 L2 裁剪（`16√2`）+ 无权 V-RMSNorm + Full 物理层对数位置缩放（`floor=8192, alpha=0.1`，同时作用于 Q 和 R）+ 每头 Attention Sink（Softmax 后丢弃）+ `sigmoid` 输出门控 + 因果深度短卷积（`k=4`，SWA 静态除 $\sqrt{5}$，Full 动态除 $3$）。
2. **MoE 路由与 Softcapped SwiGLU**：Router 按 $\sigma(\text{logits}) + b$ 选 top-8，路由权重取**未加偏置**分数归一化乘 `8.0`；专家激活 $\text{cap}(\text{gate},4)\cdot\sigma(\text{gate})\cdot\text{cap}(\text{up},25)$ 中 **$\sigma$ 作用于未截断的原始 `gate`**。
3. **3-Pass 循环与状态隔离**：每 6 物理层重置 Attention-Residual 块边界；Pass 1/2 经低秩控制器（`gate_rank=64`）混入跨轮消息；48 个逻辑步各自持有独立 KV 与 4 份 FP32 卷积 3 步历史（`k/v/attn/mlp_sconv`）。
4. **EXL3 精确子集与双命名空间**：全局码本常量 `mul1 = 0x83dcd12d`（25,582 张量中 6,264 个 `.mul1` 仅在 host 校验不占 VRAM）；统一 `16×16` tile 与 `128`-wide WHT，全部 $K,N \in \{1024,1152,2304,3072\}$ 整除 128；EXL3 张量无 `model.llm.layers.` 前缀，未量化层均有该前缀。
5. **DFlash 2** 将四路目标 tap 拼接后用 `fc` 投影并归一化，五层共享上下文特征，各自投影 KV。块内卷积使用两个因果 tap，每组 16 通道。每个草拟位置从有效词表的原始 logits 选 16 个候选，加上依赖前一 token 的边分数，再做 softcap。

---

## 2. RTX 5080（16,303 MiB）显存实测与 128K 容量账本

### 2.1 静态显存构成与当前实测水位

文本权重按组件拆分为：`moe_experts` **9.30 GiB**（80.4%，有效 `3.266 bpw`）、`embed` **883.4 MiB**、`unembed` **883.4 MiB**、`attention` **492.4 MiB**、`loop_aux/router/norms` **69.1 MiB**；独立草拟器权重占 **944.11 MiB**（`model-vision.safetensors` 607.0 MiB 当前跳过）。

生产服务使用 BF16 KV、450 MiB scratch、3 个前缀槽和 512 MiB 预留。RTX 5080 上，DFlash 2 服务启动后 `/ready` 报告 **31,508 tokens** 和 **489.88 MiB** 空闲显存。Round 1 的相同配置下，AR 容量为 67,176 tokens。板卡总量为 16,303 MiB，CUDA 进程报告可见总量为 15,840.06 MiB。

| BF16 KV 分项 | 容量 |
|---|---:|
| 目标 40 个 SWA 环和卷积状态 | 163.65625 MiB |
| 草拟器 5 个 SWA 环，`5 × 2056 × 2 × 1024 × 2` | 40.15625 MiB |
| 活跃固定状态与 3 份前缀快照 | 815.25 MiB |
| 目标 8 个 Full 层，共享增量 | 32 KiB/token |
| 31,508 tokens 的 Full KV / 含快照 KV 总量 | 984.625 / 1,799.875 MiB |
| 128K tokens 的 Full KV / 含快照 KV 总量 | 4,096 / 4,911.25 MiB |

Forge 有独立的内存布局。双模型权重 Arena 为 **13,418,483,200 B**；启动时自动分配 48,830 tokens，对应 KV **1,813,774,336 B**，scratch 450 MiB，I/O 69.11 MiB，预热后剩余 **489,553,920 B**。

## 3. RTX 5080 带宽下限

### 3.1 DFlash 2 的 `short_math` 首轮

27-token 前缀，草拟块和验证块各 8 行。逻辑流量按每个逻辑层、每种投影的使用量计算，批内相同行或专家只计一次。诊断时从原生目标路由器收集每层的 64 个选择。48 个逻辑层合计命中 717 个偶数和 700 个奇数专家（逐层去重）。带宽采用 960 GB/s，GB 使用十进制。

| 每轮主要读取 | GB | 计算依据 |
|---|---:|---|
| 草拟块、共享词表投影与选择器 | 1.664 | 五层卷积与 Q/K/V/O、MLP、norm、rank-256 投影；词表投影本身 0.922 GB，码本至多 209 次不同行读取 |
| 提交目标 tap 后重建上下文 KV | 0.090 | 一次 `fc` 和五组 K/V 投影 |
| 目标验证，共享权重和词表头 | 2.099 | Forge 的 3.348 GB 单行计数扣除平均 1.248 GB 专家项并计入验证块的 embedding 行 |
| 目标验证，本轮路由命中的专家 | 4.602 | 偶数专家 3,002,112 B/个，奇数专家 3,499,776 B/个 |
| 验证和草拟的可见 KV | 0.008 | `(48 + 5) × (27 + 8) × 4096 B` |
| **合计** | **8.463** | **8.815 ms** 逻辑下限；实测 graph **36.301 ms/round**，即 **24.3% SOL** |

此账本估计权重和 KV 的逻辑读取，实际 DRAM 流量还取决于缓存、临时张量、回滚和写入。其他前缀或草拟路径的专家集合需要重新计数。长上下文中，草拟器每层的 8 行合计最多可见 2,055 个不同的 KV 位置，环容量为 2,056，目标 SWA 环保留 1,024 个，目标 Full 层随上下文增长。[测量与路由记录](findings/2026-11-27-dflash2-pivot-baseline.md)

### 3.2 AR 单 token 对照

Forge 的单 token 权重计数为 **3,347,790,336 B**，短上下文权重下限 **3.487 ms**。在 27-token 前缀首步计入 KV 读取后，下限为 3.493 ms；使用同一常驻模型测得 AR graph **12.569 ms/token**，约 **27.8% SOL**。

下表为给定上下文的带宽估算。KV 读取项为 `(40 × min(L, 1024) + 8 × L) × 2 × 1024 × bytes_per_element`。当前实现使用 BF16；FP8 和分块 NVFP4 列用于容量规划。

| 上下文 | BF16 ms/token | FP8 ms/token | NVFP4 ms/token（0.5625 B/元素） |
|---|---:|---:|---:|
| 1,024 | 3.697 | 3.592 | 3.546 |
| 32,768 | 4.781 | 4.134 | 3.851 |
| 65,536 | 5.899 | 4.693 | 4.166 |
| 131,072 | 8.136 | 5.812 | 4.795 |

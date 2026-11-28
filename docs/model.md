# Flint-5080 模型规格与 RTX 5080 硬件账本

> **锁定权重**：`Maxrio/Inkling-2-Mini-exl3` 分支 `3.3bpw`（base `thinkingmachines/Inkling-2-Mini` 修订 `ed40e7da`，提交 `b57f9b6c`，15 文件 `13.73 GB`，相对官方混合 NVFP4/MXFP8 参考 teacher-forced KL **0.0121**、Top-1 **95.9%**）。  
> **文档边界**：本文档仅沉淀高频查阅的模型核心规格、非标准结构速记以及 16 GiB VRAM / 960 GB/s Roofline 物理账本。逐行精确数学公式见 [\_inkling2\_ref.py](../tools/weights_cli/_inkling2_ref.py)、[\_dflash2\_ref.py](../tools/weights_cli/_dflash2_ref.py) 与 [exl3\_linear.cu](../engine/src/ops/exl3_linear.cu)；完整 tensor 清单以 `./bin/weights inspect --tensors` 和 [arenas.cpp](../engine/src/loader/arenas.cpp) 为准。

---

## 1. 拓扑速查与非标准结构

### 1.1 核心维度（`text_config` + `dflash_config`）

| 维度 | Text backbone（`model.safetensors`，11.57 GiB） | DFlash 2 drafter（独立目录的 `model.safetensors`，944.11 MiB） |
|---|---|---|
| 深度与循环 | `24` physical layers，中间 `6..17` 循环 3 遍共 **48 logical steps**（`visit 0..47`） | `5` 层 Dense BF16 Decoder，配置 `--draft-dir` 时载入（`--no-spec` 跳过） |
| Attention 排布 | **40 SWA (`window=1024`) + 8 Full**（physical layers `5,11,17,23`） | 5 SWA（每个 query 最多回看 2047 个位置，intra-block 双向可见） |
| `hidden_dim` 与 head | `hidden=2304`，GQA `24Q / 8KV`，`head_dim=128` | 32Q / 8KV，`head_dim=128`，Dense `intermediate=6912` |
| FFN / MoE | `128` routed experts / `top-8` / `0` shared / `intermediate=1152` | Dense 标准 SwiGLU（无 softcap） |
| Vocab 与采样点 | `vocab=201024`（有效 `200058`），`embed`/`unembed` 不共享，softcap `30.0` | 共用 backbone `embed`/`embed_norm`/`unembed[:200058]`；采样 logical steps `[5,23,41,48]` |
| 量化构成 | EXL3 `mul1`（偶 expert `3.0b`、奇 expert `3.5b`、Attention `6.0b`）+ 其余 BF16/F32 | 81 个 BF16 tensor，candidate selector `rank=256`, `top-k=16` |

### 1.2 编写 op 最易踩坑的 5 项非标准特征

1. **Target Attention（无 RoPE）**：采用 learned relative bias（`d_rel=16, rel_extent=1024`）+ Q/K per-head L2 clip（`16√2`）+ unweighted V-RMSNorm + Full physical layer log-position scaling（`floor=8192, alpha=0.1`，同时作用于 Q 和 R）+ per-head Attention Sink（Softmax 后丢弃）+ `sigmoid` output gate + causal depthwise short conv（`k=4`，SWA 静态除 $\sqrt{5}$，Full 动态除 $3$）。
2. **MoE Router 与 Softcapped SwiGLU**：Router 按 $\sigma(\text{logits}) + b$ 选 top-8，routing weight 取**未加 bias** 分数 normalize 后乘 `8.0`；expert 激活 $\text{cap}(\text{gate},4)\cdot\sigma(\text{gate})\cdot\text{cap}(\text{up},25)$ 中 **$\sigma$ 作用于未截断的原始 `gate`**。
3. **3-Pass 循环与 State 隔离**：每 6 个 physical layers 重置 Attention-Residual block 边界；Pass 1/2 经 low-rank controller（`gate_rank=64`）混入 cross-pass message；48 个 logical steps 各自持有独立 KV 与 4 份 FP32 conv 3-step 历史（`k/v/attn/mlp_sconv`）。
4. **EXL3 精确子集与双命名空间**：全局 codebook 常量 `mul1 = 0x83dcd12d`（25,582 tensor 中 6,264 个 `.mul1` 仅在 host 校验不占 VRAM）；统一 `16×16` tile 与 `128`-wide WHT，全部 $K,N \in \{1024,1152,2304,3072\}$ 整除 128；EXL3 tensor 无 `model.llm.layers.` 前缀，未量化层均有该前缀。
5. **DFlash 2** 将四路 target tap 拼接后用 `fc` 投影并 normalize，五层共享 context 特征，各自投影 KV。intra-block conv 使用两个 causal tap，每组 16 channels。每个 draft position 从有效 vocab 的原始 logits 选 16 个 candidate，加上依赖前一 token 的 edge score，再做 softcap。

---

## 2. RTX 5080（16,303 MiB）VRAM 实测与 128K 容量账本

### 2.1 静态 VRAM 构成与当前实测水位

Text weights 按组件拆分为：`moe_experts` **9.30 GiB**（80.4%，有效 `3.266 bpw`）、`embed` **883.4 MiB**、`unembed` **883.4 MiB**、`attention` **492.4 MiB**、`loop_aux/router/norms` **69.1 MiB**；独立 drafter weights 占 **944.11 MiB**（`model-vision.safetensors` 607.0 MiB 当前跳过）。

生产服务使用 BF16 KV、450 MiB scratch、3 个 prefix slot 和 512 MiB 预留。RTX 5080 上，DFlash 2 服务启动后 `/ready` 报告 **31,508 tokens** 和 **489.88 MiB** 空闲 VRAM。在 long-context sweep 中，使用 `--prefix-cache-slots 0 --vram-reserve-mib 192` 时，`/ready` 报告 `max_context = 61,314`。[Verify width study](findings/2026-11-28-verify-width-study.md)

Round 1 使用 3 个 prefix slot 和 512 MiB reserve 时，AR 容量为 67,176 tokens。板卡总量为 16,303 MiB，CUDA 进程报告可见总量为 15,840.06 MiB。

| BF16 KV 分项 | 容量 |
|---|---:|
| Target 40 个 SWA ring 和 conv state | 163.65625 MiB |
| Drafter 5 个 SWA ring，`5 × 2056 × 2 × 1024 × 2` | 40.15625 MiB |
| 活跃固定 state 与 3 份 prefix snapshot | 815.25 MiB |
| Target 8 个 Full 层，共享增量 | 32 KiB/token |
| 31,508 tokens 的 Full KV / 含 snapshot KV 总量 | 984.625 / 1,799.875 MiB |
| 128K tokens 的 Full KV / 含 snapshot KV 总量 | 4,096 / 4,911.25 MiB |

Forge 有独立的内存布局。双模型 weights Arena 为 **13,418,483,200 B**；启动时自动分配 48,830 tokens，对应 KV **1,813,774,336 B**，scratch 450 MiB，I/O 69.11 MiB，warmup 后剩余 **489,553,920 B**。

## 3. RTX 5080 带宽下限

### 3.1 DFlash 2 的 `short_math` 首轮

27-token prefix，draft block 和默认 target verify block 各 8 行（1 个 anchor 和 7 个 proposal）。逻辑流量按每个 logical layer、每种投影的使用量计算，batch 内相同行或 expert 只计一次。诊断时从原生 target router 收集每层的 64 个选择。48 个 logical layers 合计命中 717 个偶数和 700 个奇数 expert（逐层去重）。带宽采用 960 GB/s，GB 使用十进制。

| 每轮主要读取 | GB | 计算依据 |
|---|---:|---|
| Draft block、共享 vocab 投影与 selector | 1.664 | 五层 conv 与 Q/K/V/O、MLP、norm、rank-256 投影；vocab 投影本身 0.922 GB，codebook 至多 209 次不同行读取 |
| Commit target tap 后重建 context KV | 0.090 | 一次 `fc` 和五组 K/V 投影 |
| Target verify，共享 weights 和 unembed head | 2.099 | Forge 的 3.348 GB 单行计数扣除平均 1.248 GB expert 项并计入 verify block 的 embedding 行 |
| Target verify，本轮路由命中的 expert | 4.602 | 偶数 expert 3,002,112 B/个，奇数 expert 3,499,776 B/个 |
| Verify 和 draft 的可见 KV | 0.008 | `(48 + 5) × (27 + 8) × 4096 B` |
| **合计** | **8.463** | **8.815 ms** 逻辑下限；width 8 merged graph 的 wall time **36.781 ms/round**，即 **24.0% SOL** |

此账本估计 weights 和 KV 的逻辑读取，实际 DRAM 流量还取决于 cache、临时 tensor、rollback 和写入。其他 prefix 或 draft path 的 expert 集合需要重新计数。长 context 中，drafter 每层的 8 行合计最多可见 2,055 个不同的 KV 位置，ring 容量为 2,056，target SWA ring 保留 1,024 个，target Full 层随 context 增长。[Routing](findings/2026-11-27-dflash2-pivot-baseline.md) · [Merged graph timing](findings/2026-11-28-verify-width-study.md)

### 3.2 AR 单 token 对照

Forge 的单 token weight 计数为 **3,347,790,336 B**，短 context weight 下限 **3.487 ms**。在 27-token prefix 首步计入 KV 读取后，下限为 3.493 ms；使用同一常驻模型测得 AR graph **12.569 ms/token**，约 **27.8% SOL**。

下表为给定 context 的带宽估算。KV 读取项为 `(40 × min(L, 1024) + 8 × L) × 2 × 1024 × bytes_per_element`。当前实现使用 BF16；FP8 和分块 NVFP4 列用于容量规划。

| Context | BF16 ms/token | FP8 ms/token | NVFP4 ms/token（0.5625 B/element） |
|---|---:|---:|---:|
| 1,024 | 3.697 | 3.592 | 3.546 |
| 32,768 | 4.781 | 4.134 | 3.851 |
| 65,536 | 5.899 | 4.693 | 4.166 |
| 131,072 | 8.136 | 5.812 | 4.795 |

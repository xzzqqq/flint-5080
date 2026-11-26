# Flint-5080 模型规格与 RTX 5080 硬件账本

> **锁定权重**：`Maxrio/Inkling-2-Mini-exl3` 分支 `3.3bpw`（基座 `thinkingmachines/Inkling-2-Mini` 修订 `ed40e7da`，提交 `b57f9b6c`，15 文件 `13.73 GB`，相对官方混合 NVFP4/MXFP8 参考 teacher-forced KL **0.0121**、Top-1 **95.9%**）。  
> **文档边界**：本文档仅沉淀高频查阅的模型核心规格、非标准结构速记以及 16 GiB 显存 / 960 GB/s Roofline 物理账本。逐行精确数学公式见 [\_inkling2\_ref.py](../tools/weights_cli/_inkling2_ref.py)、[\_dspark\_ref.py](../tools/weights_cli/_dspark_ref.py) 与 [exl3\_linear.cu](../engine/src/ops/exl3_linear.cu)；完整张量清单以 `./bin/weights inspect --tensors` 和 [arenas.cpp](../engine/src/loader/arenas.cpp) 为准。

---

## 1. 拓扑速查与非标准结构

### 1.1 核心维度（`text_config` + `mtp_config`）

| 维度 | 文本主干（`model.safetensors`，11.57 GiB） | DSpark 草拟器（`model-mtp.safetensors`，606.3 MiB） |
|---|---|---|
| 深度与循环 | `24` 物理层，中间 `6..17` 循环 3 遍共 **48 逻辑步**（`visit 0..47`） | `3` 层 Dense BF16 Decoder，默认载入（`--no-spec` 跳过） |
| 注意力排布 | **40 SWA (`window=1024`) + 8 Full**（物理层 `5,11,17,23`） | **2 SWA (`1032` 环槽 = `1024` 上下文 + `8` 块) + 1 Full**（层 `1`） |
| 隐藏维与头 | `hidden=2304`，GQA `24Q / 8KV`，`head_dim=128` | 同主干，Dense `intermediate=9216` |
| FFN / MoE | `128` 路由专家 / `top-8` / `0` 共享 / `intermediate=1152` | Dense 标准 SwiGLU（无 softcap） |
| 词表与采样点 | `vocab=201024`（有效 `200058`），`embed`/`unembed` 不共享，softcap `30.0` | 共用主干 `embed`/`embed_norm`/`unembed[:200058]`；采样逻辑步 `[5,23,41,48]` |
| 量化构成 | EXL3 `mul1`（偶专家 `3.0b`、奇专家 `3.5b`、Attention `6.0b`）+ 其余 BF16/F32 | 54 张量全 BF16，含 Markov RNN Head（`block=8`, `rank=256`） |

### 1.2 编写算子最易踩坑的 5 项非标准特征

1. **目标 Attention（无 RoPE）**：采用学习相对偏置（`d_rel=16, rel_extent=1024`）+ Q/K 按头 L2 裁剪（`16√2`）+ 无权 V-RMSNorm + Full 物理层对数位置缩放（`floor=8192, alpha=0.1`，同时作用于 Q 和 R）+ 每头 Attention Sink（Softmax 后丢弃）+ `sigmoid` 输出门控 + 因果深度短卷积（`k=4`，SWA 静态除 $\sqrt{5}$，Full 动态除 $3$）。
2. **MoE 路由与 Softcapped SwiGLU**：Router 按 $\sigma(\text{logits}) + b$ 选 top-8，路由权重取**未加偏置**分数归一化乘 `8.0`；专家激活 $\text{cap}(\text{gate},4)\cdot\sigma(\text{gate})\cdot\text{cap}(\text{up},25)$ 中 **$\sigma$ 作用于未截断的原始 `gate`**。
3. **3-Pass 循环与状态隔离**：每 6 物理层重置 Attention-Residual 块边界；Pass 1/2 经低秩控制器（`gate_rank=64`）混入跨轮消息；48 个逻辑步各自持有独立 KV 与 4 份 FP32 卷积 3 步历史（`k/v/attn/mlp_sconv`）。
4. **EXL3 精确子集与双命名空间**：全局码本常量 `mul1 = 0x83dcd12d`（25,582 张量中 6,264 个 `.mul1` 仅在 host 校验不占 VRAM）；统一 `16×16` tile 与 `128`-wide WHT，全部 $K,N \in \{1024,1152,2304,3072\}$ 整除 128；EXL3 张量无 `model.llm.layers.` 前缀，未量化层均有该前缀。
5. **DSpark 草拟块与 Markov Head**：4 路 tap 经 `softmax(fuse_weight)` 加权 RMSNorm 后投影上下文 KV（`k/v_proj_ctx` + `k_norm_ctx` + Neox RoPE `theta=1e6`）；3 层块内双向注意力无偏置/sink/门控；取 7 个 mask 位置经顺序 Markov RNN（`rnn_proj` 列顺序 `[state, prev_w1, hidden, f_rnn]`）按步加偏置并 softcap。

---

## 2. RTX 5080（16,303 MiB）显存实测与 128K 容量账本

### 2.1 静态显存构成与当前实测水位

文本权重按组件拆分为：`moe_experts` **9.30 GiB**（80.4%，有效 `3.266 bpw`）、`embed` **883.4 MiB**、`unembed` **883.4 MiB**、`attention` **492.4 MiB**、`loop_aux/router/norms` **69.1 MiB**；独立文件 `model-mtp.safetensors` 占 **606.3 MiB**（`model-vision.safetensors` 607.0 MiB 当前跳过）。

在默认生产配置（BF16 KV、`ScratchArena = 450 MiB`、`prefix_cache_slots = 3`、`vram_reserve = 512 MiB`）下，headless RTX 5080（驱动初始空闲 `15,571.9 MiB`）实测结果：
- **AR（`--no-spec`）**：权重 `11,852.6 MiB`（H2D `1.51 s` / `46.3 GB/s`），可用 KV + 前缀池 **`2,754.5 MiB`** → 自动定容 **`67,176 tokens`**。
- **AR + DSpark（默认）**：权重 `12,458.9 MiB` + IO/verify 静态缓冲，可用 KV + 前缀池 **`2,106.2 MiB`** → 自动定容 **`40,355 tokens`**。

### 2.2 128K（`131,072` tokens）上下文的 KV 精度物理对比

项目目标支持 **128K 上下文（含 DSpark 与 3 槽 Prefix Cache）**。下表仅列物理容量事实（目标 40 SWA + 8 Full + DSpark 2 SWA + 1 Full + 48×4 FP32 SConv 固定 `3.80 MiB`）：

| KV 精度 | 有效字节/元素 | 活跃固定状态（+DSpark） | Full 增量（9 层） | 128K + DSpark + 3 槽总需求 | 相对当前 `2,106 MiB` 池 |
|---|---:|---:|---:|---:|---|
| **BF16** | `2.0000 B` | 171.9 MiB | 36.0 KiB/tok | **5,296 MiB** | 超出 16 GiB 物理上限 |
| **FP8 (`e4m3`)** | `1.0000 B` | 87.8 MiB | 18.0 KiB/tok | **2,655 MiB** | 差 ~549 MiB（需配合释放显存，如 `embed` 883 MiB 迁 pinned host 或压减 Scratch/预留） |
| **NVFP4 (`e2m1` + blk-16 scale)** | `0.5625 B` | 51.1 MiB | 10.125 KiB/tok | **1,500 MiB** | **无需动 `embed` 直接装入，仍余 ~606 MiB** |

> 最终 KV 精度及显存方案（NVFP4 vs FP8 等）不在此预判，由后续 `forge` 吞吐与 `weights kld` 精度实测决定。

---

## 3. RTX 5080（960 GB/s）Decode Roofline 账本

### 3.1 C1 AR 单 Token 权重流量（48 逻辑步，$L \to 0$）

| 组成部分 | 逻辑读取量 | 占比 | 960 GB/s 下限 |
|---|---:|---:|---:|
| **MoE 路由专家（48 × top-8 均值）** | 1.2484 GB | 36.63% | 1.300 ms |
| **非专家层权重（6-bit Attention + 控制器/SConv/Router）** | 1.2323 GB | 36.16% | 1.284 ms |
| **LM Head（BF16 `unembed`）** | 0.9263 GB | 27.18% | 0.965 ms |
| **Embedding 查表 + 全局查询** | 0.0010 GB | 0.03% | 0.001 ms |
| **合计（短上下文纯 AR 天花板）** | **3.4079 GB** | **100.00%** | **3.550 ms (281.7 tok/s)** |

### 3.2 上下文 KV 带宽叠加与 SOL 区间（AR 单 Token）

| 场景 | 总 HBM 读量 / tok | 100% SOL (960 GB/s) | 80% (768 GB/s) | 75% (720 GB/s) |
|---|---:|---:|---:|---:|
| **短上下文（$L \approx 1\text{K}$）** | ~3.44 GB | 3.58 ms (279 tok/s) | 4.48 ms (223 tok/s) | 4.78 ms (209 tok/s) |
| **32K（BF16 / FP8 / NVFP4 KV）** | 4.65 / 4.03 / 3.76 GB | 206 / 238 / 255 tok/s | 165 / 191 / 204 tok/s | 155 / 179 / 191 tok/s |
| **64K（FP8 / NVFP4 KV）** | 4.57 / 4.06 GB | 210 / 236 tok/s | 168 / 189 tok/s | 158 / 177 tok/s |
| **128K（FP8 / NVFP4 KV）** | 5.64 / 4.66 GB | 170 / 206 tok/s | 136 / 165 tok/s | 128 / 154 tok/s |

### 3.3 核心物理杠杆

1. **AR 天花板清晰，瓶颈三足鼎立**：专家（36.6%）、非专家层（36.2%）、LM Head（27.2%）几乎平分带宽；短上下文纯 AR 80% SOL 约 **223 tok/s**。
2. **EXL3 奇偶成对结构红利**：每层 64 偶（3.0b）+ 64 奇（3.5b）且 `gate_up`/`down` 比特率成对一致、`mul1` 为全局常量，天然适合 `sm_120a` 寄存器内解码与定步长 `repack`。
3. **DSpark + 低精度 KV 的双倍杠杆**：DSpark 单次草拟权重仅 **635.8 MB**（~0.66 ms 下限），一次 8-token verify 可将 LM Head、非专家层及 128K 长上下文 KV 扫描（FP8 `2.23 GB` / NVFP4 `1.26 GB`）同时摊销给整批接受 token，是突破纯 AR 带宽上限的决定性路径。

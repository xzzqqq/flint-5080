# Flint-5080 模型规格、权重布局与 5080 SOL 账本

> **状态**：Phase 0 → Phase 1 模型冻结基线（活文档，随权重布局重排与实测校准原地更新）  
> **目标模型**：`thinkingmachines/Inkling-2-Mini`（官方修订 `ed40e7da`）  
> **首版锁定权重**：`Maxrio/Inkling-2-Mini-exl3` 分支 `3.3bpw`（Git 提交 `b57f9b6c37b9b832a385f2c6d1faa8cc052d7222`）  
> **权重目录约定**：`<shared.models_dir>/Maxrio--Inkling-2-Mini-exl3--3.3bpw`（通过 `.dev.toml` 配置，15 文件，13,733,185,968 B，`SHA256SUMS` 14/14 校验通过）  
> **实测来源**：`./bin/weights inspect --json`（[architecture.md](architecture.md) 约定之纯 `safetensors` 头与配置核验）

---

## 1. 首版模型与量化选型结论

首版锁定 **Maxrio EXL3 `3.3bpw`**，原因如下：

1. **16 GiB 显存下的容量与质量平衡**：文本权重负载为 **12.43 GB（11.57 GiB）**，比 Unsloth `UD-Q3_K_XL` 少约 1.2 GB，能为长上下文 KV/Prefix Cache、内置 DSpark drafter 与运行时 Workspace 留出关键余量；同一组 228 条评测对话（~378K tokens）相对官方混合 NVFP4/MXFP8 参考的 teacher-forced KL 为 **0.0121**（Top-1 一致率 **95.9%**），与 `UD-Q3_K_XL`（0.0120）基本持平。
2. **契合循环 MoE 的 decode 带宽优化方向**：中间 12 个物理层每个 token 循环执行 3 遍，专家投影是反复读取的主力。实测每层专家实际有效存储为 **3.2662 bpw**（含 `suh`/`svh`/`mul1` 元数据），C1 关闭草稿、48 逻辑层 top-8 平均折算的单 token 逻辑专家读取量为 **1.2484 GB**（对比 `Q3_K_XL` ~1.607 GB 少约 22%）。
3. **契合 `safetensors` 一体化与当下场景**：原生标准 `safetensors` 分文件布局（文本 / DSpark / Vision 物理隔离），无需引入 GGUF 解析器；现阶段优先聊天、代码与多轮工具调用，不必为单条 262K 提前承受 2.5-bit 激进量化的质量风险。

---

## 2. Checkpoint 资产与物理分文件清单

该分支共 15 个文件（总计 `13,733,185,968` B），三大权重文件已按模态与职责物理拆分：

| 文件 | 文件字节数 | Payload 字节数 | Tensor 数 | Dtype 构成 | P0 文本加载策略 |
|---|---:|---:|---:|---|---|
| `model.safetensors` | 12,430,993,376 (11.58 GiB) | 12,428,344,800 (11.57 GiB) | 25,582 | `I16` / `F16` / `I32` / `BF16` / `F32` | **必载**（文本主干全量） |
| `model-mtp.safetensors` | 635,801,416 (606.35 MiB) | 635,795,744 (606.34 MiB) | 54 | 全 `BF16` | **Phase 3 载入**（原版 DSpark/MTP，与官方 `mtp.safetensors` 字节一致） |
| `model-vision.safetensors` | 636,563,480 (607.07 MiB) | 636,512,256 (607.03 MiB) | 422 | 全 `BF16` | **P0 跳过**（不打开、不占用显存；P1 Vision 按需载入） |
| 其余 12 个配置/分词文件 | 29,827,696 (28.45 MiB) | — | — | — | `config.json`、`quantization_config.json`、`tokenizer.json`（26.18 MB）、`chat_template.jinja`（6.87 KB）、`tiktoken/tokenizer.model`、`SHA256SUMS` 等 |

### 2.1 组件字节分布（Payload）

| 组件 | Tensor 数 | 总字节数 | 占比（文本内） | EXL3 量化字节 | 未量化字节（BF16/F32） |
|---|---:|---:|---:|---:|---:|
| `moe_experts` | 24,576 | 9,986,924,544 (9.30 GiB) | 80.36% | 9,986,924,544 | 0 |
| `embed` | 1 | 926,318,592 (883.41 MiB) | 7.45% | 0 | 926,318,592 |
| `lm_head` (`unembed`) | 1 | 926,318,592 (883.41 MiB) | 7.45% | 0 | 926,318,592 |
| `attention` | 680 | 516,323,808 (492.40 MiB) | 4.15% | 468,234,720 | 48,089,088 |
| `loop_aux`（控制器/残差查询/`mlp_sconv`） | 218 | 58,023,936 (55.34 MiB) | 0.47% | 0 | 58,023,936 |
| `router`（`mlp.gate`） | 48 | 14,168,064 (13.51 MiB) | 0.11% | 0 | 14,168,064 |
| `norms` | 58 | 267,264 (0.25 MiB) | <0.01% | 0 | 267,264 |
| **文本主干合计** | **25,582** | **12,428,344,800 (11.57 GiB)** | **100.00%** | **10,455,159,264** | **1,973,185,536** |
| `mtp_dspark`（独立文件） | 54 | 635,795,744 (606.34 MiB) | — | 0 | 635,795,744 |
| `vision`（独立文件，P0 跳过） | 422 | 636,512,256 (607.03 MiB) | — | 0 | 636,512,256 |

---

## 3. 模型拓扑与 48 逻辑层算子清单

### 3.1 文本主干核心超参（`config.json -> text_config`）

| 参数 | 值 | 说明 |
|---|---|---|
| `hidden_size` | `2304` | 残差流维度（18 × 128） |
| `num_hidden_layers`（物理层） | `24` | 物理编号 `0..23`，每 6 层为 1 个 Attention-Residual 块 |
| `loop_config` | `core_start=6, core_end=18, num_passes=3, gate_rank=64, is_varloop=false` | 物理层 `6..17`（12 层）循环执行 3 遍，共 **48 逻辑层** |
| `vocab_size` / `unpadded_vocab_size` | `201024` / `200058` | `embed` 与 `unembed` 权重**不共享**；`final_logit_softcapping = 30.0` |
| `eos_token_id` / `pad_token_id` | `200006` (`<\|content_model_end_sampling\|>`) / `199999` (`<\|endoftext\|>`) | 另有 15 个核心角色/thinking/tool 特殊 token（见 `special_tokens_map.json`） |
| Attention 头配置 | `24` Q 头 / `8` KV 头 / `head_dim = 128` | GQA 3:1，SWA 与 Full Attention 维度一致，`q_bias = o_bias = false` |
| 注意力局部/全局排布 | `local_layer_ids = [0..4, 6..10, 12..16, 18..22]`（20 物理层）<br>Full Attention = `[5, 11, 17, 23]`（4 物理层） | `sliding_window_size = 1024`；48 逻辑层中 **40 层 SWA + 8 层 Full** |
| 相对位置与缩放/截断 | `d_rel = 16, rel_extent = 1024`, `log_scaling_n_floor = 8192, log_scaling_alpha = 0.1` | `qk_l2_clip_threshold = 22.627417 (16√2)`, `rel_l2_clip_threshold = 8.0` |
| 归一化与短卷积 | `rms_norm_eps = 1e-6`, `use_value_norm_no_gamma = true` | `sconv_kernel_size = 4`（因果深度短卷积） |
| MoE 路由与激活 | `n_routed_experts = 128`, `num_experts_per_tok = 8`, `n_shared_experts = 0`, `intermediate_size = 1152` | `route_scale = 8.0`, `use_gate_bias = true`, `mlp_gate_softcap = 4.0`, `mlp_up_softcap = 25.0` |

### 3.2 48 逻辑层映射与 `forge` `run_layers(begin, end)` 契约

引擎内部和 `forge run --layers begin..end` 统一采用 **48 逻辑步索引 `step ∈ [0, 48)`**：

| 逻辑步区间 | 阶段 | 物理层映射 | Pass 编号 | SWA 逻辑层数 | Full Attention 逻辑步索引 |
|---|---|---|---:|---:|---|
| `0..6` | Prologue | `0..5` | 0 | 5 | `5`（物理层 5） |
| `6..18` | Core Pass 0 | `6..17` | 0 | 10 | `11, 17`（物理层 11, 17） |
| `18..30` | Core Pass 1 | `6..17` | 1 | 10 | `23, 29`（物理层 11, 17） |
| `30..42` | Core Pass 2 | `6..17` | 2 | 10 | `35, 41`（物理层 11, 17） |
| `42..48` | Epilogue | `18..23` | 0 | 5 | `47`（物理层 23） |
| **合计 `0..48`** | — | **24 物理层** | — | **40** | **8** |

> **关键约束**：
> - 每个逻辑步拥有独立的 KV 槽位与短卷积滚动状态（`k=4` 缓存最近 3 步状态）；循环体物理层 `6..17` 共享权重，但 Pass 0 / 1 / 2 各自维护独立的 KV Cache。
> - `mtp_config.target_layer_ids = [5, 23, 41, 48]` 对应逻辑执行深度的边界采样点（并非物理层编号），在 Phase 1 逐层对齐与 Phase 3 接入 DSpark 时按 HF/ExLlamaV3 参考实现精确锁死。

### 3.3 `model.safetensors` 张量命名与算子清单

`model.safetensors` 内存在**双命名空间约定**（Phase 1 loader 必须按此映射）：
- **EXL3 量化线性层**（6,264 个模块 × 4 张量 = 25,056 张量）：**无 `model.llm.layers.` 前缀**，直接以 `{i}.self_attn.*` 或 `{i}.experts.*` 开头。
- **未量化层（BF16 / F32）**（526 张量）：全部以 `model.llm.` 开头。

| 模块 | 张量命名模板 | 数量 | Dtype | Shape | 单物理层/全局作用 |
|---|---|---:|---|---|---|
| Embedding / Head | `model.llm.embed.weight`, `model.llm.unembed.weight` | 各 1 | `BF16` | `[201024, 2304]` | 词表嵌入（+ `embed_norm`）与输出投影（+ `norm` + softcap 30.0） |
| Norm（局部 20 层） | `model.llm.layers.{i}.attn_norm.weight` | 20 | `BF16` | `[2304]` | SWA 层 Attention 输入 RMSNorm |
| Norm（全局 4 层） | `model.llm.layers.{5,11,17,23}.{q,k,v}_input_norm.weight` | 各 4 | `BF16` | `[2304]` | Full Attention 层分离 Q/K/V 输入 RMSNorm |
| Norm（MLP） | `model.llm.layers.{i}.mlp_norm.weight` | 24 | `BF16` | `[2304]` | MoE MLP 输入 RMSNorm |
| Pre-Sublayer SConv | `model.llm.layers.{i}.{attn,mlp}_sconv.weight` | 各 24 | `BF16` | `[2304, 1, 4]` | Attention / MLP 前置 1D 深度因果短卷积 |
| Attention 量化投影 | `{i}.self_attn.{q,k,v}_proj.linear.*`<br>`{i}.self_attn.o_proj.{gate_proj,linear}.*` | 24 × 5 | EXL3 6-bit | 见 §4.2 | Q (`2304→3072`), K/V (`2304→1024`), Output Gate (`2304→3072`), O (`3072→2304`) |
| K/V SConv（局部 20 层） | `model.llm.layers.{i}.attn.{k,v}_sconv.weight` | 各 20 | `BF16` | `[1024, 1, 4]` | 静态深度因果短卷积 |
| K/V SConv（全局 4 层） | `model.llm.layers.{5,11,17,23}.attn.{k,v}_sconv.{static_w,weight_proj}` | 各 4 | `BF16` | `[4, 1024]`, `[32, 2304]` | 动态输入条件 K/V 因果短卷积 |
| 相对位置 / Clip / Sink | `model.llm.layers.{i}.attn.wr_du.weight`<br>`...rel_logits_proj.proj`<br>`...{q,rel}_l2_clip.scale`<br>`...attn_sink_proj.weight` | 各 24 | `BF16` | `[384, 2304]`<br>`[16, 1024]`<br>`[24, 128]`, `[24, 16]`<br>`[24, 2304]` | 学习相对位置偏置、Q/Rel L2 裁剪缩放、学习 Attention Sink 投影 |
| MoE Router | `model.llm.layers.{i}.mlp.gate.{weight,bias}` | 各 24 | `BF16`, `F32` | `[128, 2304]`, `[128]` | Top-8 路由打分（`route_scale = 8.0`） |
| MoE Experts | `{i}.experts.gate_up_proj.{j}.*`<br>`{i}.experts.down_proj.{j}.*` | 24 × 128 × 2 | EXL3 3.0 / 3.5-bit | 见 §4.2 | Fused `gate_up` (`2304→2304`) + softcap SwiGLU + `down` (`1152→2304`) |
| Attention Residuals | `model.llm.attn_res.attn_queries.{0..22}`<br>`model.llm.attn_res.mlp_queries.{0..23}`<br>`model.llm.attn_res.output_query` | 23 / 24 / 1 | `F32` | `[3, 2304]`（层 `4,10,16,22`）或 `[2304]` | 每 6 层块的跨层残差注意力汇聚（`attn_res_normalize_values = true`） |
| 跨 Pass 查询与控制器 | `model.llm.middle_cycle_cross_pass{,_kv}_queries`<br>`model.llm.layers.{6..17}.controllers.{attn,k,v,mlp}.{down,forget,gain}_weight` | 2 + 144 | `F32`<br>`BF16` | `[24, 2304]`<br>`[64, 4608]`, `[2304, 64]` | 中间 12 层循环的跨轮消息门控与状态混合（低秩 `gate_rank = 64`） |

### 3.4 DSpark / MTP Drafter 结构（`model-mtp.safetensors`，Phase 3 接入）

54 个纯 `BF16` 张量（`635,795,744` B，606.34 MiB）：
- **3 层 Dense Decoder**（`523,807,488` B，499.54 MiB）：`hidden=2304`、`intermediate=9216`、GQA `24Q/8KV` (`head_dim=128`)、RoPE `theta=1e6`；层 `0, 2` 为 SWA (`window=1024`)、层 `1` 为 Full Attention；带上下文条件投影 `k_proj_ctx` / `v_proj_ctx` / `k_norm_ctx` 与 `dflare_norm`。
- **特征融合与掩码**：`fuse_weight` (`[4, 4]`)、`mask_embed` (`[2304]`)、`norm.weight` (`[2304]`)。
- **Markov RNN Head**（`111,979,008` B，106.79 MiB，`block_size = 8`, `markov_rank = 256`, `markov_head_type = "rnn_hidden"`）：`dflare_norm`、`rnn_proj` (`[768, 5120]` + bias)、`hidden_out` (`[2304, 256]` + bias)、`markov_w1.weight` (`[201024, 256]`, 102.92 MB)。

---

## 4. EXL3 精确子集与解码规范

遵循单模型极简原则，Flint-5080 的 loader 与 `sm_120a` 算子**只需支持本 checkpoint 实际用到的精确 EXL3 子集**，禁止展开无用模板组合：

### 4.1 精确格式约束

- **量化方法与码本**：仅 `quant_method = "exl3"`、`codebook = "mul1"`；全部 6,264 个 `.mul1` 张量均为同一全局常量 `0x83dcd12d`（`I32` 标量，`2212286765`）。
- **逻辑 Tile 与 Hadamard 维度**：统一 `16 × 16` 逻辑 tile；输入/输出维 `K, N ∈ {1024, 1152, 2304, 3072}` 全部被 `128` 整除，128-wide Walsh-Hadamard 变换（WHT）无余数、无需零填充。
- **精确比特档位（仅 3 档）**：
  - **`3.0` bits**（3,072 模块）：全部 24 层的**偶数专家** `j ∈ {0, 2, ..., 126}`（`gate_up_proj` 与 `down_proj` 成对一致）。
  - **`3.5` bits**（3,072 模块）：全部 24 层的**奇数专家** `j ∈ {1, 3, ..., 127}`（`gate_up_proj` 与 `down_proj` 成对一致）。
  - **`6.0` bits**（120 模块）：全部 24 层的 5 个 Attention 投影。

> **架构优化提示（Phase 3 `weights repack`）**：由于每层 128 个专家严格呈“64 个偶数 3.0-bit + 64 个奇数 3.5-bit”且 `gate_up` / `down` 比特率成对一致，后续 `weights repack` 可按奇偶拆成两块等步长（fixed-stride）连续设备数组，直接消除 MoE kernel 中的变长指针查找与 warp 分化；同时全局常量 `mul1 = 0x83dcd12d` 可直接编译期常量化。

### 4.2 全部 7 种量化投影形状清单

每个量化模块由 `(.trellis [I16], .suh [F16], .svh [F16], .mul1 [I32])` 组成，其中 `trellis` 形状为 `[K/16, N/16, int(16 * bits)]`：

| 投影模块 | 数量 | Bits | 逻辑形状 `[K → N]` | `trellis` (`I16`) | `suh` / `svh` (`F16`) | 单模块总字节 |
|---|---:|---:|---|---|---|---:|
| `experts.gate_up_proj`（偶 `j`） | 1,536 | `3.0` | `2304 → 2304` | `[144, 144, 48]` | `[2304]` / `[2304]` | 1,999,876 B |
| `experts.gate_up_proj`（奇 `j`） | 1,536 | `3.5` | `2304 → 2304` | `[144, 144, 56]` | `[2304]` / `[2304]` | 2,331,652 B |
| `experts.down_proj`（偶 `j`） | 1,536 | `3.0` | `1152 → 2304` | `[72, 144, 48]` | `[1152]` / `[2304]` | 1,002,244 B |
| `experts.down_proj`（奇 `j`） | 1,536 | `3.5` | `1152 → 2304` | `[72, 144, 56]` | `[1152]` / `[2304]` | 1,168,132 B |
| `self_attn.q_proj.linear` | 24 | `6.0` | `2304 → 3072` | `[144, 192, 96]` | `[2304]` / `[3072]` | 5,319,172 B |
| `self_attn.k_proj.linear` | 24 | `6.0` | `2304 → 1024` | `[144, 64, 96]` | `[2304]` / `[1024]` | 1,776,132 B |
| `self_attn.v_proj.linear` | 24 | `6.0` | `2304 → 1024` | `[144, 64, 96]` | `[2304]` / `[1024]` | 1,776,132 B |
| `self_attn.o_proj.gate_proj` | 24 | `6.0` | `2304 → 3072` | `[144, 192, 96]` | `[2304]` / `[3072]` | 5,319,172 B |
| `self_attn.o_proj.linear` | 24 | `6.0` | `3072 → 2304` | `[192, 144, 96]` | `[3072]` / `[2304]` | 5,319,172 B |

- 单个偶数专家（`gate_up + down`，3.0-bit）：`3,002,120` B（2.863 MiB）
- 单个奇数专家（`gate_up + down`，3.5-bit）：`3,499,784` B（3.338 MiB）
- 每物理层 128 专家合计：`416,121,856` B / `1,019,215,872` 逻辑参数 = **`3.266212` effective bpw**（24 层完全一致）。

### 4.3 线性前向语义（Phase 1 Baseline 基准）

对任意 EXL3 线性投影 $y = x W$（输入 $x \in \mathbb{R}^{M \times K}$，输出 $y \in \mathbb{R}^{M \times N}$）：
1. 输入逐通道乘符号/缩放向量 `suh` 后做 128-wide Walsh-Hadamard 旋转：$\tilde{x} = \text{WHT}_{128}(x \odot \text{suh})$。
2. 从 `trellis` 按 `mul1` 码本解码出 $16 \times 16$ 块权重 $\tilde{W}$，计算 $\tilde{y} = \tilde{x} \tilde{W}$。
3. 输出做 128-wide Walsh-Hadamard 逆旋转并乘 `svh`：$y = \text{WHT}_{128}(\tilde{y}) \odot \text{svh}$。

> **Phase 1 数值对齐策略**：实现正确性验证（`weights golden`）必须使用**同一份 Maxrio 3.3bpw checkpoint** 的 ExLlamaV3 / 精确反量化参考逐层比对，确保 48 逻辑层、SWA/Full、Dynamic K/V SConv、Attention Sink、Attn-Res 与 Loop Controllers 零语义误差；后续 kernel 融合、repack 与 KV 量化再以该 baseline 及 teacher-forced KLD 守门。

---

## 5. RTX 5080（16,303 MiB）静态显存分区预算

实测目标卡为单张 **RTX 5080 16,303 MiB（15.92 GiB）**，无桌面显示占用（headless）。

### 5.1 静态显存账本

| 分区（Arena） | Phase 1–2（AR Only） | Phase 3（AR + DSpark） | 说明 |
|---|---:|---:|---|
| **Weights Arena：文本主干** | 11,853 MiB (11.57 GiB) | 11,853 MiB (11.57 GiB) | `model.safetensors` 全部 payload（含 BF16 `embed`/`unembed`） |
| **Weights Arena：DSpark** | 0 MiB | 606 MiB (0.59 GiB) | `model-mtp.safetensors` 全部 54 个 BF16 张量 |
| **CUDA Context + cuBLAS/运行时 + CUDA Graph** | ~450 MiB | ~550 MiB | 预留驱动上下文、静态 Graph 实例与流缓冲 |
| **Scratch Workspace Arena** | ~450 MiB | ~550 MiB | 激活缓冲、chunked prefill 临时张量（如 chunk=512）、路由/重排缓冲、logits 缓冲 |
| **KV & State Cache Arena（剩余可用）** | **~3,550 MiB (~3.47 GiB)** | **~2,744 MiB (~2.68 GiB)** | 含主干 48 逻辑层 KV、DSpark 3 层 KV 及短卷积状态（`sconv` 状态极小，单序列 <1 MiB） |
| **合计** | **16,303 MiB** | **16,303 MiB** | 静态预分配，热路径严禁动态 `cudaMalloc` |

### 5.2 KV Cache 容量测算（按 40 SWA + 8 Full 精确折算）

Inkling 2 Mini 的 48 逻辑层中仅 **8 层为 Full Attention**（`8 KV heads × 128 head_dim × 2(K,V)`），其余 40 层为固定窗长 `1024` 的 SWA；DSpark 3 层中 2 层 SWA + 1 层 Full：

| KV 精度 | 主干 SWA 固定开销（40×1024） | 主干 Full 增量（8 层） | DSpark SWA / Full | 32K 上下文（AR / +DSpark） | 64K 上下文（AR / +DSpark） | KV Arena 可容纳上限（AR 3.47 GiB / +DSpark 2.68 GiB） |
|---|---:|---:|---|---:|---:|---|
| **BF16 (2 B/elem)** | 160 MiB | 32 KiB/tok | 8 MiB / 4 KiB/tok | 1,184 / 1,320 MiB | 2,208 / 2,472 MiB | **~108K**（AR） / **~73K**（+DSpark） |
| **FP8 (1 B/elem)** | 80 MiB | 16 KiB/tok | 4 MiB / 2 KiB/tok | 592 / 660 MiB | 1,104 / 1,236 MiB | **~222K**（AR） / **~151K**（+DSpark） |

> **工程结论**：
> - Phase 1–2 直接采用 **BF16 KV** 即可在 16 GiB 上轻松覆盖 64K–96K 单序列（或双并发 C2 各 32K+），实现简单且零 KV 量化误差。
> - Phase 3 开启 DSpark 并追求更高 Prefix Cache 命中率与 C2 长上下文时，引入 **FP8 KV** 可将有效 KV 容量翻倍至 ~150K tokens（同时将长上下文 KV 读带宽减半）。
> - 若未来需要额外 ~880 MiB 显存余量，可将 `model.llm.embed.weight`（883 MiB，decode 每 token 仅查表 1 行 4.5 KiB）迁移至 pinned host memory，几乎无延迟损失。

---

## 6. RTX 5080（960 GB/s）Decode Roofline 账本

基于 RTX 5080 理论峰值显存带宽 **960 GB/s**（GDDR7）与实测张量字节，核算 **C1 AR Decode（单 token，48 逻辑层）** 的理论权重读取下限：

### 6.1 单 Token 逻辑权重流量拆分（AR Baseline）

| 组成部分 | 单 token 逻辑读取量 | 占比 | 960 GB/s 理论下限 | 备注 |
|---|---:|---:|---:|---|
| **MoE 路由专家（48 × top-8 均值）** | 1,248,365,568 B (**1.2484 GB**) | 36.63% | **1.300 ms** | 偶 3.0-bit 与奇 3.5-bit 等概率均值（每逻辑层 26.01 MB） |
| **非专家层权重（48 逻辑步）** | 1,232,253,888 B (**1.2323 GB**) | 36.16% | **1.284 ms** | 含 6-bit Attention 5 投影、Router、SConv、Rel/Sink 及 3×12 控制器（若控制器仅在 Pass 1–2 触发，减少 ~56.6 MB / 0.059 ms） |
| **LM Head（BF16 `unembed`）** | 926,318,592 B (**0.9263 GB**) | 27.18% | **0.965 ms** | `[201024, 2304]` BF16 整矩阵读取，**单一项占 27.2% 带宽** |
| **Embedding 查表 + 全局查询** | 972,288 B (**0.0010 GB**) | 0.03% | **0.001 ms** | 1 行 `embed`（4.5 KiB）+ `attn_res` / `cross_pass` / norms |
| **AR 权重流量合计（短上下文 $L \to 0$）** | **3,407,910,336 B (3.4079 GB)** | **100.00%** | **3.550 ms (281.7 tok/s)** | **权重 SOL 天花板** |

### 6.2 目标效率区间与上下文 KV 带宽叠加

| 场景 | 总 HBM 读取量 / token | 100% SOL (960 GB/s) | 80% 有效带宽 (768 GB/s) | 75% 有效带宽 (720 GB/s) |
|---|---:|---:|---:|---:|
| **AR 短上下文（$L \approx 1\text{K}$）** | ~3.44 GB | 3.58 ms (279 tok/s) | 4.48 ms (223 tok/s) | 4.78 ms (209 tok/s) |
| **AR 32K 上下文（BF16 KV, +1.24 GB）** | ~4.65 GB | 4.84 ms (206 tok/s) | 6.05 ms (165 tok/s) | 6.46 ms (155 tok/s) |
| **AR 32K 上下文（FP8 KV, +0.62 GB）** | ~4.03 GB | 4.20 ms (238 tok/s) | 5.25 ms (191 tok/s) | 5.60 ms (179 tok/s) |
| **AR 64K 上下文（FP8 KV, +1.16 GB）** | ~4.57 GB | 4.76 ms (210 tok/s) | 5.95 ms (168 tok/s) | 6.34 ms (158 tok/s) |

### 6.3 对 Phase 1 → Phase 3 优化路线的核心启示

1. **三足鼎立的带宽结构，决定了仅靠 AR 优化上限明确**：MoE 专家（36.6%）、非专家层（36.2%）、BF16 LM Head（27.2%）几乎平分 AR 权重带宽，纯 AR 即使做到 80% 有效带宽也大约在 **220 tok/s**（短上下文）附近。
2. **EXL3 解码效率是核心算子风险点**：Trellis `mul1` 解码 + 128-wide Hadamard 变换如果寄存器压力过大或融合不够紧，极易在 C1 decode 下把 22% 专家带宽优势吃成 ALU/指令延迟瓶颈。Phase 1 先保证简洁、数值严格对齐的 baseline，进入 `forge` 调优期后将 **EXL3 MoE routed GEMV/GEMM（3.0 / 3.5-bit）+ Hadamard 融合** 与 **6-bit Attention QKV/O 融合** 作为首要 `sm_120a` 优化标的。
3. **`dspark`（`block_size = 8`）的倍率杠杆极大**：
   - DSpark 单步 draft（3 层 dense BF16 + Markov head）权重总计仅 **635.8 MB**（~0.66 ms @ 960 GB/s 理论下限），而一次 8-token 并行 verify 可将 **926 MB LM Head** 与 **1,232 MB 非专家层**（以及 8 个 token 间重叠命中的专家权重）全部摊销在多 token 上。
   - 因此 **Phase 3 接入 DSpark 投机解码** 是 C1 实际吞吐突破纯 AR 带宽天花板、逼近 5080 体验极限的决定性手段。

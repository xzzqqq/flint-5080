"""Eager text-only Inkling-2-Loop reference. Sequence tensors omit the singleton batch axis."""

import json
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open

MUL1 = 0x83DCD12D


@lru_cache(maxsize=8)
def _decode_plan(k2, device):
    # Map row-major tile coordinates back to the step in the serial trellis.
    steps = [0] * 256
    for p in range(256):
        lane, j = divmod(p, 8)
        row = 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1)
        col = (lane >> 2) + 8 * (j >> 2)
        steps[16 * row + col] = p
    start = (((torch.tensor(steps, device=device) + 1) * k2) // 2 - 16) % (128 * k2)
    return start // 32, (start // 32 + 1) % (4 * k2), (start % 32).int()


@lru_cache(maxsize=4)
def _codebook(device):
    prod = torch.arange(65536, device=device, dtype=torch.int64) * MUL1
    bsum = sum((prod >> i) & 255 for i in (0, 8, 16, 24))
    inv, bias = torch.tensor([0x1EEE, 0xC931], dtype=torch.uint16).view(torch.float16).tolist()
    return ((1024 + bsum).float() * inv + bias).half()


def decode_inner(trellis, mul1=MUL1):
    rows, cols, width = trellis.shape
    k2 = width // 8
    if width not in (48, 56, 96) or trellis.dtype != torch.int16:
        raise ValueError(f"Unsupported trellis width/dtype: {width}, {trellis.dtype}")
    if int(mul1) & 0xFFFFFFFF != MUL1:
        raise ValueError("Unsupported EXL3 mul1 multiplier")
    words = trellis.contiguous().view(torch.int32)
    a, b, offset = _decode_plan(k2, trellis.device)
    # Take the leading 16 bits of each circular two-word window. Shifts are
    # signed in torch, so explicitly mask the low-word contribution.
    high = (words.index_select(-1, a) << offset) >> 16
    # Only offsets >16 need bits from the next word (at most 15).
    spill = (offset - 16).clamp(min=0)
    low = (words.index_select(-1, b) >> (32 - spill).clamp(max=31)) & ((1 << spill) - 1)
    state = (high | low) & 65535
    tile = _codebook(trellis.device)[state.long()].view(rows, cols, 16, 16)
    return tile.transpose(1, 2).reshape(rows * 16, cols * 16)


@lru_cache(maxsize=8)
def _wht(device, dtype):
    h = torch.ones(1, 1, device=device, dtype=dtype)
    for _ in range(7):
        top = torch.cat((h, h), 1)
        h = torch.cat((top, torch.cat((h, -h), 1)))
    return h / math.sqrt(128)


def row_matmul(x, weight):
    """Use the same GEMM reduction for each token in prefill and decode.

    cuBLAS chooses different kernels for GEMV and for varying GEMM row counts.
    A single BF16 rounding difference can change a later top-8 routing decision.
    """
    rows = x.reshape(-1, x.shape[-1])
    results = []
    for part in rows.split(32):
        padded = F.pad(part, (0, 0, 0, 32 - len(part)))
        results.append((padded @ weight)[: len(part)])
    return torch.cat(results).reshape(*x.shape[:-1], weight.shape[1])


def linear(x, weight):
    return row_matmul(x, weight.T)


def hadamard128(x):
    return (x.reshape(-1, 128) @ _wht(x.device, x.dtype)).reshape_as(x)


def reconstruct_weight(trellis, suh, svh, mul1=MUL1):
    inner = decode_inner(trellis, mul1).float()
    return hadamard128(hadamard128(inner.T).T) * suh.float()[:, None] * svh.float()


def rms(x, weight=None):
    rows = x.float().reshape(-1, x.shape[-1])
    inv = torch.stack([torch.rsqrt(row.square().mean() + 1e-6) for row in rows])
    y = (rows * inv[:, None]).reshape_as(x)
    return (y if weight is None else y * weight).to(x.dtype)


def clip(x, threshold, scale=None):
    y = x.float() if scale is None else x.float() * scale.float()
    rows = y.reshape(-1, y.shape[-1])
    norms = torch.stack([row.norm() for row in rows]).reshape(*y.shape[:-1], 1)
    return (y * (threshold / (norms + 1e-6)).clamp(max=1)).to(x.dtype)


def residual_attention(blocks, partial, query):
    states = torch.stack([*blocks, partial], dim=-2).float()
    q = query.reshape(-1, partial.shape[-1]).float()
    rows = []
    for token in states.unbind(0):
        normalized = rms(token)
        scores = (q @ normalized.T) * partial.shape[-1] ** -0.5
        rows.append(scores.softmax(-1) @ normalized)
    return torch.stack(rows)


def softcap(x, cap):
    return (x.float() / cap).tanh() * cap


class Inkling2LoopReference:
    def __init__(self, path, device):
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        self.c = SimpleNamespace(**config["text_config"])
        self.taps = config["mtp_config"]["target_layer_ids"]
        self.device = torch.device(device)
        self.file = safe_open(path / "model.safetensors", framework="pt", device="cpu")
        self.cache = []
        self.position = 0

    def weight(self, key):
        return self.file.get_tensor("model.llm." + key).to(self.device)

    def layer_weight(self, i, suffix):
        return self.weight(f"layers.{i}.{suffix}")

    def quant_parts(self, base):
        mul1 = self.file.get_tensor(base + ".mul1").item()
        if mul1 & 0xFFFFFFFF != MUL1:
            raise ValueError(f"Unsupported multiplier in {base}")
        return tuple(
            self.file.get_tensor(base + "." + k).to(self.device) for k in ("trellis", "suh", "svh")
        )

    def project(self, x, base):
        trellis, suh, svh = self.quant_parts(base)
        inner = decode_inner(trellis)

        def rotate(rows):
            blocks = rows.reshape(-1, 128)
            return row_matmul(blocks, _wht(rows.device, rows.dtype)).reshape_as(rows)

        rotated = rotate(x.float() * suh)
        return (rotate(row_matmul(rotated, inner.float())) * svh).to(x.dtype)

    def conv(self, i, name, x, cache, condition=None):
        size = self.c.sconv_kernel_size
        prior = cache.get(name)
        context = F.pad(x, (0, 0, size - 1, 0)) if prior is None else torch.cat((prior, x))
        cache[name] = context[-(size - 1) :].clone()
        windows = context.float().unfold(0, size, 1)
        if condition is None:
            weights = self.layer_weight(i, name + ".weight").squeeze(1).float()
            y = (windows * weights).sum(-1)
            divisor = math.sqrt(size + 1)
        else:
            h = self.c.num_key_value_heads
            delta = linear(condition, self.layer_weight(i, name + ".weight_proj")).float()
            weights = self.layer_weight(i, name + ".static_w").T.float().reshape(h, -1, size)
            y = (
                (
                    windows.flip(-1).reshape(len(x), h, -1, size)
                    * (weights + delta.reshape(len(x), h, 1, size))
                )
                .sum(-1)
                .reshape_as(x)
            )
            divisor = math.sqrt(2 * size + 1)
        return ((y + x.float()) / divisor).to(x.dtype)

    def attention(self, i, inputs, cache, position):
        c = self.c
        qx, kx, vx = inputs
        sliding = i in c.local_layer_ids
        heads = c.num_attention_heads
        dim = c.head_dim
        base = f"{i}.self_attn."
        pos = torch.arange(position, position + len(qx), device=self.device)
        tau = (
            1 + c.log_scaling_alpha * ((pos + 1).float() / c.log_scaling_n_floor).clamp(min=1).log()
        )

        def scaled(x, name, threshold):
            y = clip(x, threshold, self.layer_weight(i, f"attn.{name}_l2_clip.scale"))
            return y if sliding else (y.float() * tau[:, None, None]).to(x.dtype)

        q = scaled(
            self.project(qx, base + "q_proj.linear").view(-1, heads, dim),
            "q",
            c.qk_l2_clip_threshold,
        )
        kv = []
        for name, x in (("k", kx), ("v", vx)):
            y = self.project(x, base + name + "_proj.linear")
            y = self.conv(i, f"attn.{name}_sconv", y, cache, None if sliding else x)
            y = y.reshape(len(x), -1, dim)
            y = clip(y, c.qk_l2_clip_threshold) if name == "k" else rms(y)
            if name in cache:
                y = torch.cat((cache[name], y))
            cache[name] = y[-c.sliding_window_size :].clone() if sliding else y
            kv.append(y.repeat_interleave(heads // y.shape[1], dim=1).transpose(0, 1))
        k, v = kv
        dist = pos[:, None] - torch.arange(
            position + len(qx) - k.shape[1], position + len(qx), device=self.device
        )
        rel = linear(qx, self.layer_weight(i, "attn.wr_du.weight")).view(len(qx), heads, -1)
        rel = row_matmul(
            scaled(rel, "rel", c.rel_l2_clip_threshold),
            self.layer_weight(i, "attn.rel_logits_proj.proj"),
        )
        extent = rel.shape[-1]
        bias = rel.transpose(0, 1).gather(-1, dist.clamp(0, extent - 1).expand(heads, -1, -1))
        sink = linear(qx, self.layer_weight(i, "attn.attn_sink_proj.weight"))
        # Evaluate each causal row over precisely its visible keys. Prefill uses
        # the full projected sequence; decode obtains its prefix from the cache.
        rows = []
        for t in range(len(q)):
            end = k.shape[1] - len(q) + t + 1
            begin = max(0, end - c.sliding_window_size) if sliding else 0
            keys = k[:, begin:end].contiguous()
            values = v[:, begin:end].contiguous()
            dots = (q[t, :, None, :].float() * keys.float()).sum(-1).to(q.dtype)
            scores = dots * dim**-0.5 + bias[:, t, begin:end].masked_fill(
                dist[t, begin:end] >= extent, 0
            )
            probs = torch.cat((scores, sink[t, :, None]), -1).float().softmax(-1)[:, :-1]
            rows.append(
                (probs.to(v.dtype).float().unsqueeze(-1) * values.float()).sum(1).to(v.dtype)
            )
        y = torch.stack(rows).reshape(len(qx), -1)
        gate = self.project(qx, base + "o_proj.gate_proj").float().sigmoid()
        return self.project((y.float() * gate).to(y.dtype), base + "o_proj.linear")

    def moe(self, i, x):
        logits = linear(x.float(), self.layer_weight(i, "mlp.gate.weight").float())
        scores = logits.sigmoid()
        ids = (
            (scores + self.layer_weight(i, "mlp.gate.bias"))
            .topk(self.c.num_experts_per_tok, sorted=False)
            .indices
        )
        weights = scores.gather(1, ids)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * self.c.route_scale
        out = torch.zeros_like(x, dtype=torch.float32)
        # Group the small routing table on the host once, avoiding a device
        # synchronization for every expert and loading only selected experts.
        hits = {}
        for token, row in enumerate(ids.tolist()):
            for slot, expert in enumerate(row):
                hits.setdefault(expert, []).append((token, slot))
        for expert in sorted(hits):
            tokens, slots = zip(*hits[expert], strict=True)
            idx = torch.tensor(tokens, device=x.device)
            gate, up = self.project(x[idx], f"{i}.experts.gate_up_proj.{expert}").chunk(2, -1)
            act = (
                softcap(gate, self.c.mlp_gate_softcap)
                * gate.float().sigmoid()
                * softcap(up, self.c.mlp_up_softcap)
            )
            y = self.project(act.to(x.dtype), f"{i}.experts.down_proj.{expert}")
            out[idx] += y.float() * weights[idx, list(slots), None]
        return out.to(x.dtype), logits, ids, weights

    def message_queries(self, i):
        j = 2 * (i - self.c.loop_config["core_start"])
        q, m = self.weight("middle_cycle_cross_pass_queries")[j : j + 2]
        k, v = self.weight("middle_cycle_cross_pass_kv_queries")[j : j + 2]
        return torch.stack((q, k, v, m))

    def gate(self, i, name, x, message):
        base = f"controllers.{name}."
        down = self.layer_weight(i, base + "down_weight")
        z = linear(
            torch.cat((rms(x.float()), rms(message.float())), -1).to(down.dtype), down
        ).float()
        forget = linear(z, self.layer_weight(i, base + "forget_weight").float()).tanh()
        gain = linear(z, self.layer_weight(i, base + "gain_weight").float()).tanh()
        return (1 - forget) * x.float() + gain * message.float()

    def layer(self, i, x, blocks, messages, cache):
        inputs = (
            residual_attention(blocks, x, self.weight(f"attn_res.attn_queries.{i - 1}")).unbind(-2)
            if i
            else (x,)
        )
        if len(inputs) == 1:
            inputs = inputs * 3
        normalized = []
        for j, (name, y) in enumerate(zip(("q", "k", "v"), inputs, strict=True)):
            if messages is not None:
                y = self.gate(i, "attn" if j == 0 else name, y, messages[:, j])
            norm = "attn_norm" if i in self.c.local_layer_ids else name + "_input_norm"
            normalized.append(rms(y, self.layer_weight(i, norm + ".weight")).to(x.dtype))
        attn = self.attention(i, normalized, cache, self.position)
        partial = self.conv(i, "attn_sconv", attn, cache).float()
        if i % self.c.n_layers_per_attn_res_block == 0:
            blocks = [*blocks, x]
        else:
            partial = x.float() + partial
        y = residual_attention(blocks, partial, self.weight(f"attn_res.mlp_queries.{i}")).squeeze(
            -2
        )
        if messages is not None:
            y = self.gate(i, "mlp", y, messages[:, 3])
        y = rms(y, self.layer_weight(i, "mlp_norm.weight")).to(x.dtype)
        y = self.conv(i, "mlp_sconv", self.moe(i, y)[0], cache)
        return (partial + y.float()).to(x.dtype), blocks

    @torch.inference_mode()
    def _forward(self, ids):
        x = self.file.get_tensor("model.llm.embed.weight")[torch.as_tensor(ids, device="cpu")].to(
            self.device
        )
        x = rms(x, self.weight("embed_norm.weight"))
        states = {"embed": x}
        blocks = []
        loop = self.c.loop_config
        start, end = loop["core_start"], loop["core_end"]
        visit = 0

        def run(i, previous=None):
            nonlocal visit, x, blocks
            if visit in self.taps:
                states[f"mtp_tap_{visit}"] = x
            messages = (
                residual_attention(*previous, self.message_queries(i))
                if previous is not None
                else None
            )
            if visit == len(self.cache):
                self.cache.append({})
            x, blocks = self.layer(i, x, blocks, messages, self.cache[visit])
            states[f"step_{visit}"] = x
            visit += 1

        for i in range(start):
            run(i)
        entry, prefix = x, blocks
        for p in range(loop["num_passes"]):
            previous = (blocks[len(prefix) + 1 :], x) if p else None
            x, blocks = entry, prefix
            for i in range(start, end):
                run(i, previous)
        for i in range(end, self.c.num_hidden_layers):
            run(i)
        final = residual_attention(blocks, x, self.weight("attn_res.output_query")).squeeze(-2)
        final = rms(final, self.weight("norm.weight")).to(x.dtype)
        states["final_hidden"] = final
        if visit in self.taps:
            states[f"mtp_tap_{visit}"] = final
        head = self.file.get_tensor("model.llm.unembed.weight")[: self.c.unpadded_vocab_size]
        logits = linear(final, head.to(self.device))
        states["logits"] = softcap(logits, self.c.final_logit_softcapping)
        self.position += len(ids)
        return states

    def prefill(self, ids):
        self.cache = []
        self.position = 0
        return self._forward(ids)

    def decode_step(self, token):
        if not self.cache:
            raise ValueError("prefill must precede decode_step")
        return self._forward([token])

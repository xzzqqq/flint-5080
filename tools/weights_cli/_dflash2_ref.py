"""Eager, single-request DFlash2 reference with a committed target-context cache."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open

from ._inkling2_ref import softcap


def neox_rope(x, positions, theta):
    dim = x.shape[-1]
    inv_freq = 1 / theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim)
    phase = torch.outer(positions.float(), inv_freq)[:, None, :]
    cos, sin = phase.cos(), phase.sin()
    a, b = x.float().chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)


class Inkling2DFlash2Reference:
    def __init__(self, path, device, target):
        path = Path(path)
        self.device = torch.device(device)
        config = json.loads((path / "config.json").read_text())
        self.c = SimpleNamespace(
            **{**config, **config["dflash_config"], **config["rope_parameters"]}
        )
        self.target = target
        self.unembed = self.target.unembedding()
        with safe_open(path / "model.safetensors", framework="pt") as f:
            self.weights = {k: f.get_tensor(k).to(device) for k in f.keys()}
        self.reset_cache()

    def reset_cache(self):
        self.cache = {}
        self.position = 0

    def norm(self, x, name):
        x = x.float()
        inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.c.rms_norm_eps)
        return (x * inv * self.weights[name + ".weight"].float()).bfloat16()

    def project(self, x, name):
        return F.linear(x, self.weights[name + ".weight"])

    def store_context_kv(self, taps, ctx_pos, trace=None):
        ctx = self.norm(self.project(taps.flatten(1), "fc"), "hidden_norm")
        for i in range(self.c.num_hidden_layers):
            attn = f"layers.{i}.self_attn."
            shape = (len(ctx), self.c.num_key_value_heads, self.c.head_dim)
            k = self.project(ctx, attn + "k_proj").view(shape)
            k = neox_rope(self.norm(k, attn + "k_norm"), ctx_pos, self.c.rope_theta)
            v = self.project(ctx, attn + "v_proj").view(shape)
            if i in self.cache:
                old_k, old_v = self.cache[i]
                k, v = torch.cat((old_k, k)), torch.cat((old_v, v))
            k, v = (a[-self.c.sliding_window :].clone() for a in (k, v))
            self.cache[i] = k, v
            if trace is not None:
                trace[f"ctx_kv.layer_{i}.k"] = k
                trace[f"ctx_kv.layer_{i}.v"] = v
        self.position = int(ctx_pos[-1]) + 1

    def embed_block(self, anchor_token_id):
        return self.target.embed_tokens(
            [anchor_token_id] + [self.c.mask_token_id] * (self.c.block_size - 1)
        )

    def grouped_conv(self, u, delta, base, side):
        n, h = u.shape
        g = h // self.c.conv_group_size
        coeff = (
            base[side].reshape(2, g, -1).float()
            + delta.reshape(n, 2, 2, g)[:, side, :, :, None].float()
        )
        values = u.reshape(n, g, -1).float()
        out = values * coeff[:, 0]
        out[1:] += values[:-1] * coeff[1:, 1]
        return out.reshape_as(u).to(u.dtype)

    def attention(self, i, h, q_pos):
        base = f"layers.{i}.self_attn."
        q, k, v = [
            self.project(h, base + name + "_proj").view(len(h), -1, self.c.head_dim)
            for name in ("q", "k", "v")
        ]
        q = neox_rope(self.norm(q, base + "q_norm"), q_pos, self.c.rope_theta)
        k = neox_rope(self.norm(k, base + "k_norm"), q_pos, self.c.rope_theta)
        ctx_k, ctx_v = self.cache[i]
        groups = self.c.num_attention_heads // self.c.num_key_value_heads
        k, v = [
            torch.cat(pair).repeat_interleave(groups, dim=1).transpose(0, 1).float()
            for pair in ((ctx_k, k), (ctx_v, v))
        ]
        scores = (q.transpose(0, 1).float() @ k.transpose(1, 2)) * self.c.head_dim**-0.5
        key_positions = torch.arange(k.shape[1], device=q.device) + q_pos[0] - len(ctx_k)
        distance = q_pos[:, None] - key_positions[None, :]
        scores.masked_fill_(distance.abs() >= self.c.sliding_window, -torch.inf)
        y = (scores.softmax(-1) @ v).transpose(0, 1).reshape(len(h), -1)
        return self.project(y.bfloat16(), base + "o_proj")

    def forward_block(self, block, q_pos, trace=None):
        x = block
        for i in range(self.c.num_hidden_layers):
            base = f"layers.{i}."
            for name, norm in (
                ("attention", "input_layernorm"),
                ("mlp", "post_attention_layernorm"),
            ):
                h = self.norm(x, base + norm)
                conv = base + name + "_conv."
                kernel = self.weights[conv + "base_kernel"]
                delta = self.project(h, conv + "kernel_projection")
                h = self.grouped_conv(h, delta, kernel, 0)
                if name == "attention":
                    y = self.attention(i, h, q_pos)
                else:
                    gate = self.project(h, base + "mlp.gate_proj").float()
                    up = self.project(h, base + "mlp.up_proj").float()
                    y = self.project((F.silu(gate) * up).bfloat16(), base + "mlp.down_proj")
                x = x + self.grouped_conv(y, delta, kernel, 1)
            if trace is not None:
                trace[f"layer_{i}_out"] = x
        return self.norm(x, "norm")

    def select(self, h, anchor, trace=None, temperature=0.0, seed=0):
        h = h[1:]
        logits = F.linear(h, self.unembed).float()
        cand = logits.argsort(dim=-1, descending=True, stable=True)[:, : self.c.selector_top_k]
        unary = logits.gather(1, cand)
        projected = self.project(h, "candidate_selector.hidden_projection").float()
        a = self.weights["candidate_selector.predecessor_codebook"]
        b = self.weights["candidate_selector.successor_codebook"]
        n, k = cand.shape
        scores = unary.new_zeros(n, k, k)
        ids = []
        prev = 0
        for e in range(n):
            pred = cand[e - 1] if e else cand.new_tensor([anchor])
            edges = (a[pred].float() * projected[e]) @ b[cand[e]].float().T
            scores[e, : len(pred)] = softcap(unary[e] + edges, self.c.final_logit_softcapping)
            row = scores[e, prev]
            if temperature > 0:
                seed = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
                z = ((seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
                z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
                u = ((z ^ (z >> 31)) >> 11) * 2**-53
                weights = ((row - row.max()) / temperature).exp().double()
                weights = (weights / weights.sum()).float().double()
                prev = int(
                    torch.searchsorted(weights.cumsum(0), u * weights.sum(), right=True).clamp_max(
                        k - 1
                    )
                )
            else:
                prev = int(row.argmax())
            ids.append(cand[e, prev])
        result = torch.stack(ids).int()
        if trace is not None:
            trace.update(
                cand_ids=cand.int(), unary_logits=unary, selector_scores=scores, draft_ids=result
            )
        return result

    def draft_greedy(self, anchor, trace=None):
        block = self.embed_block(anchor)
        positions = torch.arange(len(block), device=self.device) + self.position
        head = self.forward_block(block, positions, trace)
        if trace is not None:
            trace.update(embed_block=block, head_hidden=head)
        return self.select(head, anchor, trace)

    @torch.inference_mode()
    def speculative_decode_greedy(self, target, prompt_ids, max_new_tokens):
        """Return generated IDs and a replayable trace. The last emitted token is pending."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.reset_cache()

        def taps(output):
            return torch.stack([output[f"draft_tap_{i}"] for i in self.c.target_layer_ids], dim=1)

        pre = target.prefill(prompt_ids)
        committed = taps(pre)
        anchor = int(pre["logits"][-1].argmax())

        def ids(values):
            return torch.tensor(values, dtype=torch.int32, device=self.device)

        trace = dict(
            prompt_ids=ids(prompt_ids), prefill_taps=committed, prefill_bonus_id=ids([anchor])
        )
        self.store_context_kv(committed, torch.arange(len(prompt_ids), device=self.device), trace)
        generated = [anchor]
        round_index = 0
        while len(generated) < max_new_tokens:
            details = {} if round_index == 0 else None
            draft = self.draft_greedy(anchor, details)
            block = [anchor, *draft.tolist()]
            snapshot = target.snapshot_cache()
            verified = target.verify_block(block)
            target_ids = verified["logits"].argmax(-1).int()
            mismatches = (draft != target_ids[:-1]).nonzero()
            accepted = int(mismatches[0, 0]) if len(mismatches) else len(draft)
            bonus = int(target_ids[accepted])
            prefix = f"round_{round_index}."
            for k, value in (details or {}).items():
                trace[prefix + k] = value
            trace.update(
                {
                    prefix + k: v
                    for k, v in dict(
                        anchor_id=ids([anchor]),
                        draft_ids=draft,
                        target_ids=target_ids,
                        accepted_count=ids([accepted]),
                        next_bonus_id=ids([bonus]),
                    ).items()
                }
            )
            count = min(1 + accepted, max_new_tokens - len(generated))
            if count < len(block):
                target.restore_cache(snapshot)
                verified = target.verify_block(block[:count])
            committed = taps(verified)
            self.store_context_kv(
                committed, torch.arange(count, device=self.device) + self.position
            )
            generated.extend([*block[1 : accepted + 1], bonus][:count])
            anchor = generated[-1]
            round_index += 1
        trace["spec_generated_ids"] = ids(generated)
        return generated, trace

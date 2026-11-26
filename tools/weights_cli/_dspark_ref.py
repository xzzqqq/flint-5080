"""Eager, single-request DSpark reference with a committed target-context cache."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open

from ._inkling2_ref import Inkling2LoopReference, softcap


def neox_rope(x, positions, theta):
    dim = x.shape[-1]
    inv_freq = 1 / theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim)
    phase = torch.outer(positions.float(), inv_freq)[:, None, :]
    cos, sin = phase.cos(), phase.sin()
    a, b = x.float().chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)


class Inkling2DSparkReference:
    def __init__(self, path, device, target=None):
        path = Path(path)
        self.device = torch.device(device)
        self.c = SimpleNamespace(**json.loads((path / "config.json").read_text())["mtp_config"])
        self.target = target if target is not None else Inkling2LoopReference(path, device)
        self.unembed = self.target.unembedding()
        with safe_open(path / "model-mtp.safetensors", framework="pt") as f:
            self.weights = {k: f.get_tensor(k).to(device) for k in f.keys()}
        self.alpha = self.weights["fuse_weight"].float().softmax(-1)
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

    def mix_rmsnorm(self, taps, row):
        prefix = "markov_head" if row == self.c.num_hidden_layers else f"layers.{row}"
        mixed = (taps.float() * self.alpha[row, :, None]).sum(1)
        return self.norm(mixed, prefix + ".dflare_norm")

    def compute_f_rnn(self, taps):
        return self.mix_rmsnorm(taps[-1:], self.c.num_hidden_layers)

    def store_context_kv(self, taps, ctx_pos, trace=None):
        for i in range(self.c.num_hidden_layers):
            attn = f"layers.{i}.self_attn."
            ctx = self.mix_rmsnorm(taps, i)
            shape = (len(ctx), self.c.num_key_value_heads, self.c.head_dim)
            k = self.project(ctx, attn + "k_proj_ctx").view(shape)
            k = neox_rope(self.norm(k, attn + "k_norm_ctx"), ctx_pos, self.c.rope_theta)
            v = self.project(ctx, attn + "v_proj_ctx").view(shape)
            if i in self.cache:
                old_k, old_v = self.cache[i]
                k, v = torch.cat((old_k, k)), torch.cat((old_v, v))
            if i in self.c.local_layer_ids:
                k, v = (a[-self.c.sliding_window_size :].clone() for a in (k, v))
            self.cache[i] = k, v
            if trace is not None:
                trace[f"mix_rmsnorm.layer_{i}"] = ctx
                trace[f"ctx_kv.layer_{i}.k"] = k
                trace[f"ctx_kv.layer_{i}.v"] = v
        self.position = int(ctx_pos[-1]) + 1

    def embed_block(self, anchor_token_id):
        anchor = self.target.embed_tokens([anchor_token_id])
        return torch.cat((anchor, self.weights["mask_embed"].expand(self.c.block_size - 1, -1)))

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
        y = (scores.softmax(-1) @ v).transpose(0, 1).reshape(len(h), -1)
        return self.project(y.bfloat16(), base + "o_proj")

    def forward_block(self, block, q_pos, trace=None):
        x = block
        for i in range(self.c.num_hidden_layers):
            base = f"layers.{i}."
            x = x + self.attention(i, self.norm(x, base + "input_layernorm"), q_pos)
            h = self.norm(x, base + "post_attention_layernorm")
            gate = self.project(h, base + "mlp.gate_proj").float()
            up = self.project(h, base + "mlp.up_proj").float()
            x = x + self.project((F.silu(gate) * up).bfloat16(), base + "mlp.down_proj")
            if trace is not None:
                trace[f"layer_{i}_out"] = x
        return self.norm(x, "norm")

    def markov_sample_greedy(self, head_hidden, f_rnn, anchor_token_id, trace=None):
        h = head_hidden[1:]
        base_logits = F.linear(h, self.unembed)
        rank = self.c.markov_rank
        w = self.weights["markov_head.rnn_proj.weight"].float()
        bias = self.weights["markov_head.rnn_proj.bias"].float()
        features = torch.cat((h, f_rnn.expand_as(h)), dim=-1)
        context = F.linear(features.float(), w[:, 2 * rank :], bias)
        prev = anchor_token_id
        state = h.new_zeros(rank)
        ids, states, latents, top_ids, top_logits = [], [], [], [], []
        for row, base in zip(context, base_logits, strict=True):
            token_feature = self.weights["markov_head.markov_w1.weight"][prev]
            raw = row + F.linear(torch.cat((state, token_feature)).float(), w[:, : 2 * rank])
            gate, cand, out = raw.bfloat16().chunk(3)
            gate = gate.float().sigmoid().bfloat16()
            state = gate * state + (1 - gate) * cand.float().tanh().bfloat16()
            latent = out.float().tanh().bfloat16()
            hidden = self.project(latent, "markov_head.hidden_out")
            hidden = hidden + self.weights["markov_head.hidden_out.bias"]
            bias_logits = F.linear(hidden, self.unembed)
            logits = softcap(base + bias_logits, self.target.c.final_logit_softcapping).bfloat16()
            prev = logits.argmax()
            ids.append(prev)
            if trace is not None:
                states.append(state)
                latents.append(latent)
                top = logits.argsort(descending=True, stable=True)[:64]
                top_ids.append(top.int())
                top_logits.append(logits[top])
        draft = torch.stack(ids).int()
        if trace is not None:
            trace.update(
                {
                    "markov.context": context,
                    "markov.states": torch.stack(states),
                    "markov.latents": torch.stack(latents),
                    "markov.draft_ids": draft,
                    "markov.top64_ids": torch.stack(top_ids),
                    "markov.top64_logits": torch.stack(top_logits),
                }
            )
        return draft

    def folded_logit_error(self, latents):
        """Check affine folding in FP64, then measure BF16 projection rounding error."""
        w = self.weights["markov_head.hidden_out.weight"]
        bias = self.weights["markov_head.hidden_out.bias"]
        exact_hidden = F.linear(latents.double(), w.double(), bias.double())
        hidden = torch.stack([F.linear(z, w) + bias for z in latents])
        unfolded = torch.stack([F.linear(h, self.unembed) for h in hidden])
        maximum = 0.0
        # Bound the FP64 working set independently of vocabulary size.
        for start in range(0, len(self.unembed), 4096):
            u = self.unembed[start : start + 4096].double()
            folded = F.linear(latents.double(), u @ w.double(), u @ bias.double())
            torch.testing.assert_close(folded, exact_hidden @ u.T, atol=1e-9, rtol=1e-10)
            maximum = max(
                maximum, (folded - unfolded[:, start : start + len(u)]).abs().max().item()
            )
        return maximum

    def draft_greedy(self, anchor_token_id, f_rnn, trace=None):
        block = self.embed_block(anchor_token_id)
        positions = torch.arange(self.c.block_size, device=self.device) + self.position
        head = self.forward_block(block, positions, trace)
        if trace is not None:
            trace.update(embed_block=block, head_hidden=head, f_rnn=f_rnn)
        return self.markov_sample_greedy(head, f_rnn, anchor_token_id, trace)

    @torch.inference_mode()
    def speculative_decode_greedy(self, target, prompt_ids, max_new_tokens):
        """Return generated IDs and a replayable trace. The last emitted token is pending."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.reset_cache()

        def taps(output):
            return torch.stack([output[f"mtp_tap_{i}"] for i in self.c.target_layer_ids], dim=1)

        pre = target.prefill(prompt_ids)
        committed = taps(pre)
        anchor = int(pre["logits"][-1].argmax())

        def ids(values):
            return torch.tensor(values, dtype=torch.int32, device=self.device)

        trace = dict(
            prompt_ids=ids(prompt_ids), prefill_taps=committed, prefill_bonus_id=ids([anchor])
        )
        self.store_context_kv(committed, torch.arange(len(prompt_ids), device=self.device), trace)
        trace["mix_rmsnorm.markov"] = self.mix_rmsnorm(committed, self.c.num_hidden_layers)
        generated = [anchor]
        round_index = 0
        while len(generated) < max_new_tokens:
            details = {} if round_index == 0 else None
            draft = self.draft_greedy(anchor, self.compute_f_rnn(committed), details)
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

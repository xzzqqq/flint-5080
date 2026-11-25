"""Generate portable, checkpoint-backed operator and cached decoding fixtures."""

import fcntl
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import safetensors
import torch
from safetensors.torch import save_file

from tools.common.config import load_config

from .__main__ import build_parser
from ._inkling2_ref import (
    Inkling2LoopReference,
    decode_inner,
    reconstruct_weight,
    residual_attention,
)


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class OperatorTrace(Inkling2LoopReference):
    """Capture submodule call boundaries during a real prefill and two cache updates."""

    def __init__(self, path, device):
        super().__init__(path, device)
        self.tensors = {}
        self.phase = "prefill"

    def record(self, group, **values):
        self.tensors.update({group + "." + k: v.cpu().contiguous() for k, v in values.items()})

    def conv(self, i, name, x, cache, condition=None):
        y = super().conv(i, name, x, cache, condition)
        if i in (0, 5) and name == "attn.k_sconv":
            group = "static_sconv" if i == 0 else "dynamic_sconv"
            self.record(
                group + "." + self.phase, x=x, y=y, **({"condition": condition} if i else {})
            )
        return y

    def attention(self, i, inputs, cache, position):
        y = super().attention(i, inputs, cache, position)
        if i in (0, 5) and self.phase != "decode_1":
            group = "sliding_attention" if i == 0 else "full_attention"
            q, k, v = inputs
            self.record(group + "." + self.phase, q=q, k=k, v=v, y=y)
        return y

    def moe(self, i, x):
        result = super().moe(i, x)
        if i == 0 and self.phase == "prefill":
            y, logits, ids, weights = result
            self.record("moe", x=x, y=y, router_logits=logits, top8_ids=ids.int(), weights=weights)
        return result

    def gate(self, i, name, x, message):
        y = super().gate(i, name, x, message)
        group = "message_gate." + name
        # The first call to a layer-6 controller is on core pass 1.
        if i == 6 and self.phase == "prefill" and group + ".y" not in self.tensors:
            self.record(group, x=x, message=message, y=y)
        return y


def operator_fixtures(path, device, ids):
    model = OperatorTrace(path, device)
    model.tensors["input_ids"] = torch.tensor(ids, dtype=torch.int32)
    pre = model.prefill(ids)
    for i in (1, 5):
        block, partial = pre["embed"], pre[f"step_{i - 1}"]
        query = model.weight(f"attn_res.attn_queries.{i - 1}")
        model.record(
            f"block_residual.layer_{i}",
            block=block,
            partial=partial,
            query=query,
            y=residual_attention([block], partial, query),
        )
    # The block saved at entry to layer 12 and the partial after layer 17
    # survive the prefix slice when forming messages for pass 1.
    block, partial = pre["step_11"], pre["step_17"]
    query = model.message_queries(6)
    model.record(
        "message_residual.layer_6_pass_1",
        block=block,
        partial=partial,
        query=query,
        y=residual_attention([block], partial, query),
    )
    state = pre
    for step in range(2):
        model.phase = f"decode_{step}"
        token = int(state["logits"][-1].argmax())
        model.tensors[model.phase + ".input_ids"] = torch.tensor([token], dtype=torch.int32)
        state = model.decode_step(token)

    rng = torch.Generator().manual_seed(5080)
    for bits, base in (
        ("3_0", "0.experts.gate_up_proj.0"),
        ("3_5", "0.experts.gate_up_proj.1"),
        ("6_0", "0.self_attn.q_proj.linear"),
    ):
        tr, su, sv = model.quant_parts(base)
        x = torch.randn(3, len(su), generator=rng).to(device)
        inner = decode_inner(tr)
        digest = hashlib.sha256(inner.cpu().numpy().tobytes()).digest()
        model.record(
            "exl3." + bits,
            trellis=tr,
            suh=su,
            svh=sv,
            mul1=model.file.get_tensor(base + ".mul1"),
            inner_f16_slice=inner[:32, :32],
            inner_sha256=torch.tensor(list(digest), dtype=torch.uint8),
            x=x,
            y=x @ reconstruct_weight(tr, su, sv),
        )
    return model.tensors


def e2e_case(model, case, steps):
    prompt = case["input_ids"]
    pre = model.prefill(prompt)
    saved = {"input_ids": torch.tensor(prompt, dtype=torch.int32)}
    for k, v in pre.items():
        if k != "logits":
            saved["prefill." + k] = v.cpu()
    last = pre.pop("logits")[-1].cpu()
    saved["prefill.last_logits"] = last
    comparison = [last]
    generated = [int(last.argmax())]

    def top(prefix, logits):
        vals, ids = logits.topk(64)
        saved[prefix + ".top64_ids"] = ids.int()
        saved[prefix + ".top64_logits"] = vals

    top("prefill", last)
    for s in range(steps):
        row = model.decode_step(generated[-1])
        prefix = f"decode_{s}"
        last = row["logits"][-1].cpu()
        generated.append(int(last.argmax()))
        comparison.append(last)
        if s == 0:
            for k, v in row.items():
                if k.startswith("step_"):
                    saved[prefix + "." + k] = v.cpu()
            saved[prefix + ".logits"] = last
        saved[prefix + ".final_hidden"] = row["final_hidden"].cpu()
        top(prefix, last)
    saved["generated_ids"] = torch.tensor(generated, dtype=torch.int32)
    forced = model.prefill(prompt + generated[:-1])["logits"][len(prompt) - 1 :].cpu()
    cached = torch.stack(comparison)
    match = forced.argmax(-1) == cached.argmax(-1)
    metrics = dict(
        prefill_vs_decode_max_abs_err=(forced - cached).abs().max().item(),
        prefill_vs_decode_top1_match=match.float().mean().item(),
        positions=len(match),
        case=case["name"],
    )
    if not torch.isfinite(cached).all() or not torch.isfinite(forced).all() or not match.all():
        raise ValueError(f"Cached decode self-verification failed: {metrics}")
    return saved, generated, metrics


def write_fixture(directory, name, tensors):
    # safetensors rejects overlapping storage (e.g. the final MTP tap and final_hidden).
    values = {k: v.cpu().contiguous().clone() for k, v in tensors.items()}
    path = directory / name
    save_file(values, path)
    return dict(
        sha256=sha256(path),
        size_bytes=path.stat().st_size,
        tensors={
            k: dict(shape=list(v.shape), dtype=str(v.dtype).removeprefix("torch."))
            for k, v in values.items()
        },
    )


@torch.inference_mode()
def generate(args, staging):
    from tokenizers import Tokenizer
    from transformers.utils.chat_template_utils import render_jinja_template

    start = time.monotonic()
    path = Path(args.path).resolve()
    cases_path = Path(args.cases)
    cases = json.loads(cases_path.read_text())["cases"]
    names = [c["name"] for c in cases]
    if len(set(names)) != len(names) or not all(re.fullmatch(r"[a-zA-Z0-9_-]+", n) for n in names):
        raise ValueError("Invalid or duplicate case name")
    if missing := set(args.case) - set(names):
        raise ValueError(f"Unknown cases: {sorted(missing)}")
    if not cases:
        raise ValueError("Empty golden case manifest")
    operator_ids = cases[0]["input_ids"][:4]
    if args.case:
        cases = [c for c in cases if c["name"] in args.case]
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    torch.set_float32_matmul_precision("highest")
    model = Inkling2LoopReference(path, device)
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    template = (path / "chat_template.jinja").read_text()
    files, outputs, checks = {}, [], []
    if not args.skip_ops:
        print("Generating operator fixtures", file=sys.stderr, flush=True)
        files["ops_golden.safetensors"] = write_fixture(
            staging, "ops_golden.safetensors", operator_fixtures(path, device, operator_ids)
        )
    if not args.skip_e2e:
        for case in cases:
            steps = case["decode_steps"] if args.steps is None else args.steps
            if steps < 1:
                raise ValueError("decode_steps must be positive")
            rendered, _ = render_jinja_template(
                conversations=[case["messages"]],
                chat_template=template,
                reasoning_effort=case.get("reasoning_effort"),
                add_generation_prompt=True,
            )
            if tok.encode(rendered[0], add_special_tokens=False).ids != case["input_ids"]:
                raise ValueError(f"Checkpoint tokenizer disagrees with {case['name']}.input_ids")
            print(f"Generating {case['name']} ({steps} cached steps)", file=sys.stderr, flush=True)
            tensors, ids, check = e2e_case(model, case, steps)
            checks.append(check)
            name = f"e2e_{case['name']}.safetensors"
            files[name] = write_fixture(staging, name, tensors)
            outputs.append(
                dict(
                    **case,
                    executed_decode_steps=steps,
                    generated_ids=ids,
                    text=tok.decode(ids, skip_special_tokens=False),
                )
            )
    metrics = {}
    if checks:
        metrics = dict(
            positions=sum(c["positions"] for c in checks),
            prefill_vs_decode_top1_match=min(c["prefill_vs_decode_top1_match"] for c in checks),
            prefill_vs_decode_max_abs_err=max(c["prefill_vs_decode_max_abs_err"] for c in checks),
            by_case=checks,
        )
    manifest = dict(
        schema_version=1,
        created_at=datetime.now(UTC).isoformat(),
        weights_dir=str(path),
        checkpoint={
            p: sha256(path / p)
            for p in ("model.safetensors", "config.json", "tokenizer.json", "chat_template.jinja")
        },
        cases_sha256=sha256(cases_path),
        generator_sha256={
            p: sha256(Path(__file__).with_name(p)) for p in ("_golden.py", "_inkling2_ref.py")
        },
        transformers_reference="430050f6f71195b76004b5d61bbb446ec0508fb3",
        python=platform.python_version(),
        torch=torch.__version__,
        transformers=version("transformers"),
        tokenizers=version("tokenizers"),
        jinja2=version("jinja2"),
        operator_seed=5080,
        safetensors=safetensors.__version__,
        cuda=torch.version.cuda,
        device=device,
        gpu=torch.cuda.get_device_name() if device == "cuda" else None,
        files=files,
        cases=outputs,
        self_verification=metrics,
        peak_vram_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else 0,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        wall_seconds=time.monotonic() - start,
        out_dir=str(Path(args.out).resolve()),
    )
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main(argv=None):
    args = build_parser().parse_args(["golden", *(sys.argv[1:] if argv is None else argv)])
    if not args.path:
        raise ValueError("--path is required for the standalone generator")
    if args.out is None:
        cfg = load_config(allow_missing=True)
        base = Path(cfg.weights.golden_dir or cfg.repo_root / ".forge/golden")
        args.out = str(base / Path(args.path).name)
    out = Path(args.out)
    if args.skip_ops and args.skip_e2e:
        raise ValueError("Select at least one fixture tier.")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        staging = Path(tmp)
        manifest = generate(args, staging)
        # Publish the manifest last. A failed publication must never leave an
        # old manifest advertising files from two different runs.
        descriptor = os.open(out, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            (out / "manifest.json").unlink(missing_ok=True)
            for name in [*manifest["files"], "manifest.json"]:
                (staging / name).replace(out / name)
        finally:
            os.close(descriptor)
    summary = {
        k: manifest[k]
        for k in ("out_dir", "cases", "self_verification", "wall_seconds", "peak_vram_bytes")
    }
    print(json.dumps(dict(**summary, files=[*manifest["files"], "manifest.json"]), allow_nan=False))


if __name__ == "__main__":
    main()

import json
import math
import shlex
import struct
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.weights_cli import __main__ as cli
from tools.weights_cli import commands


def test_lightweight_cli():
    subprocess.run(
        [sys.executable, "-S", "-m", "tools.weights_cli", "golden", "--help"],
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize("mode", ["remote", "remote_cpu", "local_override", "local_default"])
def test_golden_dispatch(cfg, monkeypatch, capsys, mode):
    cfg.remote.enabled = mode != "local_default"
    cfg.model.weights_dir = "model name"
    cfg.model.draft_weights_dir = "separate draft"
    cfg.weights.golden_dir = "" if mode == "local_default" else "/goldens"
    events = []

    @contextmanager
    def hold(label):
        events.append(label)
        yield
        events.append("released")

    executor = Mock()
    executor.sync.side_effect = lambda: events.append("sync")
    data = dict(
        out_dir="/out",
        files=["manifest.json"],
        cases=[dict(name="case1", text="answer")],
        wall_seconds=3,
        peak_vram_bytes=1048576,
    )

    def run(command, check, stream_stderr, echo_stdout):
        events.append("run")
        assert check and stream_stderr and not echo_stdout
        args = shlex.split(command)
        assert args[1:3] == ["-m", "tools.weights_cli._golden"]
        if mode == "local_override":
            assert args[0] == "my python" and args[args.index("--path") + 1] == "/other"
            assert args[args.index("--out") + 1] == "/dir"
        else:
            assert args[0] == (sys.executable if mode == "local_default" else "/venv/python")
            assert args[args.index("--path") + 1] == cfg.shared.models_dir + "/model name"
            directory = ".forge/golden" if mode == "local_default" else "/goldens"
            assert args[args.index("--out") + 1] == directory + "/model name"
        assert args.count("--case") == 2 and "--skip-ops" in args and "--steps" in args
        assert args[args.index("--draft-path") + 1] == cfg.shared.models_dir + "/separate draft"
        assert "--skip-draft" in args
        return SimpleNamespace(stdout=json.dumps(data))

    executor.run.side_effect = run

    def make_executor(config):
        assert config.is_remote == mode.startswith("remote")
        return executor

    monkeypatch.setattr(cli, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(cli, "find_repo_root", lambda _: cfg.repo_root)
    monkeypatch.setattr(commands, "Executor", make_executor)
    monkeypatch.setattr(commands, "GpuLeaseManager", lambda *args: SimpleNamespace(hold=hold))
    args = ["golden", "--case", "a", "--case", "b", "--steps", "2", "--skip-ops", "--skip-draft"]
    if mode == "local_override":
        args += ["--local", "--path", "/other", "--out", "/dir", "--python", "my python"]
    if mode.startswith("remote"):
        cfg.weights.python = "/venv/python"
        args += ["--json"]
    if mode == "remote_cpu":
        args += ["--device", "cpu"]
    assert cli.main(args) == 0
    assert events == (["sync"] if mode.startswith("remote") else []) + (
        ["run"] if mode == "remote_cpu" else ["weights-golden", "run", "released"]
    )
    output = capsys.readouterr().out
    if mode.startswith("remote"):
        assert json.loads(output)["data"] == data
    else:
        assert "manifest.json" in output and "answer" in output


@pytest.fixture
def pt():
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield torch
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("k2", [6, 7, 12])
def test_exl3_scalar_oracle(pt, k2):
    from tools.weights_cli._inkling2_ref import decode_inner, hadamard128, reconstruct_weight

    rng = pt.Generator().manual_seed(5080)
    tr = pt.randint(-32768, 32768, (8, 8, 8 * k2), generator=rng, dtype=pt.int16)
    inner = decode_inner(tr)
    inv, bias = struct.unpack("<ee", struct.pack("<HH", 0x1EEE, 0xC931))
    # Independent scalar bitstream decoder, including cross-word and wraparound states.
    for ti, tj in [(0, 0), (2, 7), (7, 1)]:
        words = tr[ti, tj].tolist()
        stream = "".join(
            f"{(a & 65535) | ((b & 65535) << 16):032b}"
            for a, b in zip(words[::2], words[1::2], strict=True)
        )
        for p in range(256):
            end = (p + 1) * k2 // 2
            state = int("".join(stream[b % len(stream)] for b in range(end - 16, end)), 2)
            byte_sum = sum(((state * 0x83DCD12D) & 0xFFFFFFFF).to_bytes(4, "little"))
            (value,) = struct.unpack("<e", struct.pack("<e", (1024 + byte_sum) * inv + bias))
            lane, j = divmod(p, 8)
            row = 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1)
            col = (lane >> 2) + 8 * (j >> 2)
            assert inner[ti * 16 + row, tj * 16 + col].item() == value
    h = pt.tensor(
        [[(-1) ** ((i & j).bit_count()) for j in range(128)] for i in range(128)]
    ) / math.sqrt(128)
    pt.testing.assert_close(hadamard128(pt.eye(128)), h)
    su, sv = pt.randn(2, 128, generator=rng).half()
    pt.testing.assert_close(
        reconstruct_weight(tr, su, sv),
        su[:, None] * (h @ inner.float() @ h) * sv,
        atol=4e-6,
        rtol=1e-5,
    )
    with pytest.raises(ValueError, match="multiplier"):
        decode_inner(tr, 0)


def tiny_checkpoint(pt, path):
    from safetensors.torch import save_file

    rng = pt.Generator().manual_seed(23)
    layers = 8
    c = dict(
        hidden_size=128,
        num_hidden_layers=layers,
        local_layer_ids=[0, 2, 4, 6],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        log_scaling_alpha=0.1,
        log_scaling_n_floor=2,
        qk_l2_clip_threshold=10,
        rel_l2_clip_threshold=2,
        num_experts_per_tok=2,
        route_scale=2,
        sliding_window_size=3,
        sconv_kernel_size=4,
        n_layers_per_attn_res_block=2,
        mlp_gate_softcap=4,
        mlp_up_softcap=25,
        loop_config=dict(core_start=2, core_end=6, num_passes=3),
        unpadded_vocab_size=70,
        final_logit_softcapping=30,
    )
    config = dict(text_config=c)
    (path / "config.json").write_text(json.dumps(config))
    tensors = {}

    def weight(name, shape, scale=0.1, offset=0, dtype=None):
        tensors["model.llm." + name] = (pt.randn(shape, generator=rng) * scale + offset).to(
            dtype or pt.bfloat16
        )

    def quant(base, k, n, k2):
        tensors[base + ".trellis"] = pt.randint(
            -32768, 32768, (k // 16, n // 16, 8 * k2), generator=rng, dtype=pt.int16
        )
        tensors[base + ".suh"] = pt.ones(k, dtype=pt.float16) / math.sqrt(k)
        tensors[base + ".svh"] = pt.ones(n, dtype=pt.float16)
        tensors[base + ".mul1"] = pt.tensor(0x83DCD12D - 2**32, dtype=pt.int32)

    for name in ("embed", "unembed"):
        weight(name + ".weight", (72, 128))
    for name in ("embed_norm", "norm"):
        weight(name + ".weight", (128,), offset=1)
    weight("attn_res.output_query", (128,), dtype=pt.float32)
    for name in ("middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"):
        weight(name, (8, 128), dtype=pt.float32)
    for i in range(layers):
        base = f"layers.{i}."
        if i:
            weight(
                f"attn_res.attn_queries.{i - 1}", (3, 128) if i % 2 else (128,), dtype=pt.float32
            )
        weight(f"attn_res.mlp_queries.{i}", (128,), dtype=pt.float32)
        for name in (
            ["q_input_norm", "k_input_norm", "v_input_norm"] if i % 2 else ["attn_norm"]
        ) + ["mlp_norm"]:
            weight(base + name + ".weight", (128,), offset=1)
        for name in ("attn_sconv", "mlp_sconv"):
            weight(base + name + ".weight", (128, 1, 4))
        for name in ("k", "v"):
            prefix = base + "attn." + name + "_sconv."
            if i % 2:
                weight(prefix + "static_w", (4, 128))
                weight(prefix + "weight_proj", (4, 128))
            else:
                weight(prefix + "weight", (128, 1, 4))
        weight(base + "attn.q_l2_clip.scale", (2, 128), offset=1)
        weight(base + "attn.rel_l2_clip.scale", (2, 4), offset=1)
        weight(base + "attn.wr_du.weight", (8, 128))
        weight(base + "attn.rel_logits_proj.proj", (4, 3))
        weight(base + "attn.attn_sink_proj.weight", (2, 128))
        weight(base + "mlp.gate.weight", (3, 128))
        weight(base + "mlp.gate.bias", (3,), dtype=pt.float32)
        for name, k, n in (
            ("q_proj.linear", 128, 256),
            ("k_proj.linear", 128, 128),
            ("v_proj.linear", 128, 128),
            ("o_proj.gate_proj", 128, 256),
            ("o_proj.linear", 256, 128),
        ):
            quant(f"{i}.self_attn." + name, k, n, 12)
        for e in range(3):
            quant(f"{i}.experts.gate_up_proj.{e}", 128, 256, 6 + e % 2)
            quant(f"{i}.experts.down_proj.{e}", 128, 128, 6 + e % 2)
        if 2 <= i < 6:
            for gate in ("attn", "k", "v", "mlp"):
                for name, shape in (("down", (8, 256)), ("gain", (128, 8)), ("forget", (128, 8))):
                    weight(base + f"controllers.{gate}.{name}_weight", shape)
    save_file(tensors, path / "model.safetensors")


def test_loop_prefill_and_cached_decode(pt, tmp_path):
    from tools.weights_cli._inkling2_ref import Inkling2LoopReference

    tiny_checkpoint(pt, tmp_path)
    model = Inkling2LoopReference(tmp_path, "cpu")
    tokens = [2, 4, 6, 3, 5, 7]
    model.taps = [1, 7, 13, 16]
    expected = model.prefill(tokens)
    assert expected["logits"].shape == (6, 70)
    # Physical layer 2 restarts from the same prologue on all three passes.
    # Its outputs differ only when the previous pass supplies its messages.
    assert not pt.equal(expected["step_2"], expected["step_6"])
    assert not pt.equal(expected["step_6"], expected["step_10"])
    model.prefill(tokens[:2])
    for pos in range(2, len(tokens)):
        got = model.decode_step(tokens[pos])
        for k, v in got.items():
            pt.testing.assert_close(v, expected[k][pos : pos + 1], atol=0, rtol=0)
        for visit in model.taps[:-1]:
            assert pt.equal(got[f"draft_tap_{visit}"], got[f"step_{visit - 1}"])

        assert pt.equal(got["draft_tap_16"], got["final_hidden"])
    assert len(model.cache) == 16 and len({id(s) for s in model.cache}) == 16
    for i, slot in enumerate(model.cache):
        assert len(slot["k"]) == (6 if i % 2 else 3)
        for name in ("attn.k_sconv", "attn.v_sconv", "attn_sconv", "mlp_sconv"):
            assert len(slot[name]) == 3

    from safetensors.torch import load_file

    from tools.weights_cli._golden import e2e_case, write_fixture

    case = dict(name="tiny", input_ids=tokens)
    saved, generated, check = e2e_case(model, case, 2)
    assert check["positions"] == 3 and check["prefill_vs_decode_top1_match"] == 1
    assert saved["prefill.top64_ids"][0].item() == generated[0]
    assert saved["decode_0.top64_ids"][0].item() == generated[1]
    assert saved["decode_1.top64_ids"][0].item() == generated[2]
    inventory = write_fixture(tmp_path, "e2e_tiny.safetensors", saved)
    fixture = load_file(tmp_path / "e2e_tiny.safetensors")
    assert fixture["prefill.step_15"].shape == (6, 128)
    assert fixture["decode_0.step_15"].dtype == pt.bfloat16
    assert fixture["decode_0.logits"].dtype == pt.float32
    assert inventory["tensors"]["generated_ids"] == dict(shape=[3], dtype="int32")


@pytest.fixture
def dflash2(pt, tmp_path):
    from safetensors.torch import save_file

    from tools.weights_cli._dflash2_ref import Inkling2DFlash2Reference

    tiny_checkpoint(pt, tmp_path)
    from tools.weights_cli._inkling2_ref import Inkling2LoopReference

    target = Inkling2LoopReference(tmp_path, "cpu")
    target.taps = [1, 7, 13, 16]
    path = tmp_path / "draft"
    path.mkdir()
    cfg = dict(
        hidden_size=128,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        sliding_window=16,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000},
        dflash_config=dict(
            block_size=8,
            selector_top_k=16,
            conv_group_size=16,
            mask_token_id=64,
            final_logit_softcapping=30,
            target_layer_ids=target.taps,
        ),
    )
    (path / "config.json").write_text(json.dumps(cfg))
    rng = pt.Generator().manual_seed(21)
    weights = {}

    def add(key, *shape):
        offset = 1 if "norm" in key else 0
        weights[key] = (pt.randn(shape, generator=rng) / 8 + offset).bfloat16()

    add("fc.weight", 128, 512)
    add("hidden_norm.weight", 128)
    add("norm.weight", 128)
    add("candidate_selector.hidden_projection.weight", 4, 128)
    for book in ("predecessor", "successor"):
        add(f"candidate_selector.{book}_codebook", 72, 4)
    for i in range(5):
        p = f"layers.{i}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            add(p + norm + ".weight", 128)
        for conv in ("mlp_conv", "attention_conv"):
            add(p + conv + ".base_kernel", 2, 2, 128)
            add(p + conv + ".kernel_projection.weight", 32, 128)
        for name, shape in (
            ("q_proj", (16, 128)),
            ("o_proj", (128, 16)),
            ("k_proj", (8, 128)),
            ("v_proj", (8, 128)),
            ("k_norm", (4,)),
            ("q_norm", (4,)),
        ):
            add(p + f"self_attn.{name}.weight", *shape)
        for name in ("gate", "up", "down"):
            add(p + f"mlp.{name}_proj.weight", *((128, 16) if name == "down" else (16, 128)))
    save_file(weights, path / "model.safetensors")
    return Inkling2DFlash2Reference(path, "cpu", target)


def test_dflash2_selector(pt, dflash2):
    h = pt.randn(8, 128, generator=pt.Generator().manual_seed(7)).bfloat16()
    trace = {}
    ids = dflash2.select(h, 2, trace)
    cand = trace["cand_ids"].long()
    unary = (h[1:] @ dflash2.unembed.T).float()
    assert pt.equal(cand, unary.argsort(descending=True, stable=True)[:, :16])
    w = dflash2.weights
    prev = 2
    for e in range(7):
        proj = h[e + 1] @ w["candidate_selector.hidden_projection.weight"].T
        pred = w["candidate_selector.predecessor_codebook"][prev].float()
        succ = w["candidate_selector.successor_codebook"][cand[e]].float()
        score = 30 * ((unary[e, cand[e]] + (succ * pred * proj.float()).sum(-1)) / 30).tanh()
        prev = int(cand[e, score.argmax()])
        assert ids[e] == prev
    assert pt.equal(
        dflash2.select(h, 2, temperature=1, seed=42), dflash2.select(h, 2, temperature=1, seed=42)
    )

    dflash2.unembed = pt.zeros_like(dflash2.unembed)
    dflash2.select(h, 2, trace)
    assert pt.equal(trace["cand_ids"], pt.arange(16).expand(7, -1))


def test_dflash2_grouped_conv_and_context(pt, dflash2):
    # Make the tap and group axes distinguishable and exercise a previous-only filter.
    rng = pt.Generator().manual_seed(5)
    u = pt.randn(8, 128, generator=rng).bfloat16()
    base = pt.randn(2, 2, 128, generator=rng).bfloat16()
    delta = pt.randn(8, 32, generator=rng).bfloat16()
    for side in (0, 1):
        got = dflash2.grouped_conv(u, delta, base, side)
        for row in range(8):
            expected = pt.zeros(128)
            for tap in range(min(row + 1, 2)):
                group = delta[
                    row, (2 * side + tap) * 8 : (2 * side + tap + 1) * 8
                ].repeat_interleave(16)
                expected += u[row - tap].float() * (base[side, tap].float() + group.float())
            assert pt.equal(got[row], expected.bfloat16())
    taps = pt.randn(22, 4, 128, generator=rng).bfloat16()
    dflash2.store_context_kv(taps[:16], pt.arange(16))
    dflash2.store_context_kv(taps[16:], pt.arange(16, 22))
    chunked = dflash2.cache.copy()
    dflash2.reset_cache()
    dflash2.store_context_kv(taps, pt.arange(22))
    for i, (k, v) in dflash2.cache.items():
        assert len(k) == len(v) == 16
        for one, two in zip((k, v), chunked[i], strict=True):
            pt.testing.assert_close(one, two, atol=0, rtol=0)


def test_dflash2_window_per_query(pt, dflash2):
    # Uniform scores. Context values identify the surviving absolute positions.
    ctx, block, window = 22, 8, dflash2.c.sliding_window
    positions = pt.arange(ctx - window, ctx, dtype=pt.bfloat16)
    values = positions[:, None, None].expand(window, 2, 4)
    dflash2.cache[0] = pt.zeros_like(values), values
    weight = dflash2.weights["layers.0.self_attn.o_proj.weight"]
    weight.zero_()
    weight[0, 0] = 1
    result = dflash2.attention(0, pt.zeros(block, 128).bfloat16(), pt.arange(ctx, ctx + block))
    for row in range(block):
        history = range(ctx + row - window + 1, ctx)
        mean = sum(history) / (len(history) + block)
        assert result[row, 0] == pt.tensor(mean).bfloat16()


def test_dflash2_golden_file(pt, dflash2, tmp_path):
    from safetensors.torch import load_file

    from tools.weights_cli._golden import draft_fixture, write_fixture

    data, report = draft_fixture(
        tmp_path / "draft", "cpu", dflash2.target, {"input_ids": [2, 4, 6, 3]}, 2
    )
    write_fixture(tmp_path, "dflash2_golden.safetensors", data)
    saved = load_file(tmp_path / "dflash2_golden.safetensors")
    assert report["greedy_match_rate"] == 1 and pt.equal(
        saved["ar_generated_ids"], saved["spec_generated_ids"]
    )
    assert all(
        f"ctx_kv.layer_{i}.k" in saved and f"round_0.layer_{i}_out" in saved for i in range(5)
    )
    assert saved["round_0.cand_ids"].shape == (7, 16)


def test_dflash2_target_rollback_and_parity(pt, dflash2, monkeypatch):
    model = dflash2.target
    prompt = [2, 4, 6, 3]
    ids = [int(model.prefill(prompt)["logits"][-1].argmax())]
    for _ in range(11):
        ids.append(int(model.decode_step(ids[-1])["logits"][-1].argmax()))
    position, expected_cache = model.snapshot_cache()

    def compare(count):
        generated, trace = dflash2.speculative_decode_greedy(model, prompt, count)
        assert trace["spec_generated_ids"].tolist() == generated == ids[:count]
        assert dflash2.position == model.position == len(prompt) + count - 1
        return trace

    trace = compare(3)
    assert trace["round_0.target_ids"].shape == (8,)
    assert trace["round_0.selector_scores"].shape == (7, 16, 16)

    # Force 2/7 accepted (replay past SWA eviction), then 7/7 (no rollback).
    def propose(anchor, details):
        offset = model.position - len(prompt)
        candidates = ids[offset + 1 : offset + 8]
        if offset == 0:
            candidates[2] = (candidates[2] + 1) % len(dflash2.unembed)
        return pt.tensor(candidates, dtype=pt.int32)

    monkeypatch.setattr(dflash2, "draft_greedy", propose)
    trace = compare(12)
    assert [int(trace[f"round_{r}.accepted_count"]) for r in range(2)] == [2, 7]
    assert model.position == position
    for got, expected in zip(model.cache, expected_cache, strict=True):
        for key in expected:
            assert pt.equal(got[key], expected[key]), key


@pytest.mark.parametrize("same_checkpoint", [True, False])
def test_partial_manifest(pt, tmp_path, monkeypatch, same_checkpoint):
    from tools.weights_cli import _golden

    out = tmp_path / "out"
    out.mkdir()
    checkpoint = {"model.safetensors": "text", "config.json": "cfg"}
    cases = [{"name": "short_math", "generated_ids": [1, 2]}]
    files = {"ops_golden.safetensors": {}, "e2e_short_math.safetensors": {}}
    previous = dict(
        checkpoint=checkpoint.copy(), files=files, cases=cases, self_verification={"positions": 2}
    )
    if not same_checkpoint:
        previous["checkpoint"]["config.json"] = "other"
    for name in files:
        (out / name).write_bytes(b"original")
    (out / "manifest.json").write_text(json.dumps(previous))

    def generate(args, staging):
        assert args.skip_e2e and args.skip_ops and not args.skip_draft
        (staging / "dflash2_golden.safetensors").write_bytes(b"new")
        return dict(
            files={"dflash2_golden.safetensors": {}},
            checkpoint=checkpoint,
            out_dir=str(out),
            cases=[],
            self_verification={},
            wall_seconds=1,
            peak_vram_bytes=1,
        )

    monkeypatch.setattr(_golden, "generate", generate)
    args = [
        "--path",
        str(tmp_path),
        "--draft-path",
        str(tmp_path / "draft"),
        "--out",
        str(out),
        "--skip-ops",
        "--skip-e2e",
    ]
    _golden.main(args)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["cases"] == (cases if same_checkpoint else [])
    assert manifest["self_verification"] == (
        previous["self_verification"] if same_checkpoint else {}
    )
    assert set(manifest["files"]) == {"dflash2_golden.safetensors"} | (
        set(files) if same_checkpoint else set()
    )
    for name in files:
        assert (out / name).read_bytes() == b"original"
    with pytest.raises(ValueError, match="tier"):
        _golden.main([*args, "--skip-draft"])


@pytest.mark.parametrize("changed_file", [None, "config.json", "model.safetensors"])
def test_preserve_skipped_draft(pt, tmp_path, changed_file):
    from tools.weights_cli import _golden

    checkpoint = {"config.json": "target-config", "model.safetensors": "target-weights"}
    previous = {
        "schema_version": 2,
        "checkpoint": checkpoint.copy(),
        "draft_weights_dir": str(tmp_path),
        "draft_verification": {"greedy_match_rate": 1.0},
        "files": {"dflash2_golden.safetensors": {"sha256": "fixture"}},
    }
    for name in checkpoint:
        path = tmp_path / name
        path.write_bytes(b"original")
        previous["checkpoint"]["draft/" + name] = _golden.sha256(path)
    if changed_file:
        (tmp_path / changed_file).write_bytes(b"modified")
    args = SimpleNamespace(
        skip_ops=False, skip_draft=True, draft_path=tmp_path, skip_e2e=False, case=[]
    )
    manifest = {"schema_version": 2, "checkpoint": checkpoint.copy(), "files": {}}
    _golden.preserve_skipped(manifest, previous, args)
    assert ("draft_verification" in manifest) == (changed_file is None)
    assert manifest["files"] == ({} if changed_file else previous["files"])
    assert manifest["checkpoint"] == (checkpoint if changed_file else previous["checkpoint"])

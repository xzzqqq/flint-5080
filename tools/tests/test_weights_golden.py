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
        return SimpleNamespace(stdout=json.dumps(data))

    executor.run.side_effect = run

    def make_executor(config):
        assert config.is_remote == mode.startswith("remote")
        return executor

    monkeypatch.setattr(cli, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(cli, "find_repo_root", lambda _: cfg.repo_root)
    monkeypatch.setattr(commands, "Executor", make_executor)
    monkeypatch.setattr(commands, "GpuLeaseManager", lambda *args: SimpleNamespace(hold=hold))
    args = ["golden", "--case", "a", "--case", "b", "--steps", "2", "--skip-ops"]
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
    config = dict(text_config=c, mtp_config=dict(target_layer_ids=[1, 7, 13, 16]))
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
        for visit in (1, 7, 13):
            assert pt.equal(got[f"mtp_tap_{visit}"], got[f"step_{visit - 1}"])
        assert pt.equal(got["mtp_tap_16"], got["final_hidden"])
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

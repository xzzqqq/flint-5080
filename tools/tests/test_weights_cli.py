import json
import math
import shlex
import struct
import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.common.config import ConfigError
from tools.common.output import RunResult
from tools.weights_cli import __main__ as cli
from tools.weights_cli import _inspect, commands


def write_weights(path, rows):
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, dtype, shape in rows:
        size = math.prod(shape) * {"I16": 2, "F16": 2, "I32": 4, "BF16": 2}[dtype]
        header[name] = dict(shape=shape, dtype=dtype, data_offsets=[offset, offset + size])
        offset += size
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "test model"
    (root / "tiktoken").mkdir(parents=True)
    (root / "tiktoken/tokenizer.model").write_bytes(b"vocab")
    (root / "chat_template.jinja").write_text("hello")
    (root / "tokenizer.json").write_text("{}")
    (root / "SHA256SUMS").write_text("0" * 64 + "  config.json\n")
    (root / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {"codebook": "mul1", "quant_method": "exl3"},
                "text_config": {"hidden_size": 16},
            }
        )
    )
    (root / "tokenizer_config.json").write_text(
        '{"tokenizer_class":"Test","added_tokens_decoder":{}}'
    )
    for key in ("generation_config", "processor_config", "special_tokens_map"):
        (root / f"{key}.json").write_text('{"example":42}')

    rows = []
    bits = {}
    for layer in (0, 12):
        for expert, width in enumerate((3.0, 3.5)):
            module = f"{layer}.experts.down_proj.{expert}"
            bits[module] = width
            for suffix, dtype, shape in (
                ("trellis", "I16", [1, 1, int(width * 16)]),
                ("suh", "F16", [16]),
                ("svh", "F16", [16]),
                ("mul1", "I32", []),
            ):
                rows.append((f"{module}.{suffix}", dtype, shape))
        rows.extend(
            (f"model.llm.layers.{layer}.{name}", "BF16", [16])
            for name in (
                "mlp.gate.bias",
                "attn.q_l2_clip.scale",
                "attn_norm.weight",
                "controllers.attn.down_weight",
            )
        )
    rows += [
        ("model.llm.embed.weight", "BF16", [32, 16]),
        ("model.llm.unembed.weight", "BF16", [32, 16]),
    ]
    write_weights(root / "model.safetensors", rows)
    write_weights(
        root / "model-mtp.safetensors", [("layers.0.self_attn.q_proj.weight", "BF16", [4])]
    )
    write_weights(root / "model-vision.safetensors", [("model.visual.proj_in.weight", "BF16", [8])])
    (root / "quantization_config.json").write_text(json.dumps({"bits": bits}))
    return root


def test_inspect_headers_and_files(checkpoint):
    data = _inspect.inspect_directory(checkpoint, include_tensors=True)
    assert data["weights_dir"] == str(checkpoint)
    paths = [f["path"] for f in data["files"]]
    assert paths == sorted(paths) and "tiktoken/tokenizer.model" in paths
    assert data["total_dir_bytes"] == sum((checkpoint / p).stat().st_size for p in paths)
    assert data["sha256sums_present"] and data["sha256sums_entry_count"] == 1
    h = next(h for h in data["safetensors"] if h["file"] == "model.safetensors")
    assert h["metadata"] == {"format": "pt"}
    assert h["tensor_count"] == 26
    assert h["dtypes"]["I16"] == {"count": 4, "bytes": 416}
    assert h["dtypes"]["I32"] == {"count": 4, "bytes": 16}
    assert h["file_size_bytes"] == 8 + h["header_bytes"] + h["total_tensor_bytes"]
    assert len(data["tensors"]) == 28
    config = data["configs"]
    assert config["config"]["text_config"]["hidden_size"] == 16
    assert config["processor_config"] == config["special_tokens_map"] == {"example": 42}
    assert config["generation_config"] == {"example": 42}
    assert config["tokenizer_config_summary"] == {
        "tokenizer_class": "Test",
        "added_tokens_count": 0,
        "chat_template_jinja_bytes": 5,
        "tokenizer_json_bytes": 2,
    }


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        struct.pack("<Q", 15) + b"{}",
        struct.pack("<Q", 1) + b"{",
        struct.pack("<Q", 2) + b"[]",
        struct.pack("<Q", 8) + b'{"x":{}}',
    ],
)
def test_invalid_headers(tmp_path, raw):
    (tmp_path / "broken.safetensors").write_bytes(raw)
    with pytest.raises(ValueError, match="broken.safetensors"):
        _inspect.inspect_directory(tmp_path)


def test_truncated_payload(checkpoint):
    path = checkpoint / "model.safetensors"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="invalid tensor header"):
        _inspect.inspect_directory(checkpoint)


def test_exl3_inventory(checkpoint):
    data = _inspect.inspect_directory(checkpoint)
    assert "tensors" not in data
    exl3 = data["exl3"]
    assert exl3["suffix_breakdown"]["trellis"] == {"count": 4, "bytes": 416}
    comps = exl3["component_breakdown"]
    assert {k: v["bytes"] for k, v in comps.items()} == {
        "embed": 1024,
        "lm_head": 1024,
        "attention": 64,
        "router": 64,
        "moe_experts": 688,
        "norms": 64,
        "loop_or_other": 64,
        "mtp_dspark": 8,
        "vision": 16,
    }
    assert comps["moe_experts"]["quantized"] == {"count": 16, "bytes": 688}
    assert comps["mtp_dspark"]["unquantized"] == {"count": 1, "bytes": 8}
    assert exl3["loop_or_other_patterns"] == {
        "model.llm.layers.{i}.controllers.attn.down_weight": {"count": 2, "bytes": 64}
    }
    assert exl3["per_layer_summary"] == [
        dict(
            layer=i,
            total_bytes=472,
            expert_bytes=344,
            non_expert_bytes=128,
            expert_tensor_count=8,
            expert_parameters=512,
            effective_expert_bpw=5.375,
        )
        for i in (0, 12)
    ]
    tiles = exl3["codebook_and_tile_summary"]
    assert tiles["codebooks"] == ["mul1"]
    assert tiles["bits"] == [3, 3.5]
    assert tiles["by_suffix"]["trellis"] == [
        {"dtype": "I16", "shape": [1, 1, 48]},
        {"dtype": "I16", "shape": [1, 1, 56]},
    ]
    group = next(g for g in data["groups"] if g["pattern"] == "{i}.experts.down_proj.{j}.trellis")
    assert group == dict(
        file="model.safetensors",
        pattern="{i}.experts.down_proj.{j}.trellis",
        count=4,
        dtypes=["I16"],
        shapes=[[1, 1, 48], [1, 1, 56]],
        total_bytes=416,
        bytes_per_tensor=[96, 112],
        sample_name="0.experts.down_proj.0.trellis",
    )
    assert sum(g["count"] for g in data["groups"]) == 28
    assert sum(g["total_bytes"] for g in data["groups"]) == sum(c["bytes"] for c in comps.values())
    q = data["configs"]["quantization_summary"]
    assert q["top_level_fields"] == ["bits"] and q["methods"] == ["exl3"]
    assert q["counts_by_codebook"] == {"mul1": 4}
    assert q["counts_by_bits"] == {"3.0": 2, "3.5": 2}
    assert q["counts_by_module_pattern"] == {"{i}.experts.down_proj.{j}": 4}
    assert q["raw_examples"]["bits"]["0.experts.down_proj.1"] == 3.5


@pytest.mark.parametrize("mode", ["relative", "absolute", "override", "local"])
def test_path_resolution(cfg, checkpoint, mode):
    cfg.shared.models_dir = str(checkpoint.parent)
    cfg.model.weights_dir = checkpoint.name if mode == "relative" else str(checkpoint)
    override = None
    if mode == "override":
        cfg.model.weights_dir = "wrong"
        override = str(checkpoint)
    cfg.remote.enabled = mode == "local"
    result = RunResult("weights inspect", True, {})
    commands.inspect(cfg, Namespace(path=override, local=mode == "local", tensors=False), result)
    assert result.data["weights_dir"] == str(checkpoint)


def test_missing_weights_directory(cfg):
    with pytest.raises(ConfigError, match="model.weights_dir.*--path"):
        commands.inspect(cfg, Namespace(path=None), RunResult("weights inspect", True, {}))


def test_remote(cfg, checkpoint):
    cfg.remote.enabled = True
    result = RunResult("weights inspect", True, {})
    args = Namespace(path=str(checkpoint), local=False, tensors=True)
    with patch.object(commands, "Executor") as executor:

        def remote(command, *, cwd, check):
            argv = shlex.split(command)
            assert argv[:2] == ["python3", "-c"] and cwd == "." and check
            return SimpleNamespace(stdout=subprocess.check_output(argv, text=True))

        executor.return_value.run.side_effect = remote
        commands.inspect(cfg, args, result)
        executor.assert_called_once_with(cfg)
        executor.return_value.run.assert_called_once()
    assert result.data == _inspect.inspect_directory(checkpoint, include_tensors=True)


@pytest.mark.parametrize(
    ("flags", "raw"), [(["--json", "inspect"], False), (["inspect", "--json", "--tensors"], True)]
)
def test_cli_json(repo, checkpoint, capsys, monkeypatch, flags, raw):
    monkeypatch.setattr(cli, "find_repo_root", lambda _: repo)
    assert cli.main([*flags, "--local", "--path", str(checkpoint)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] and output["command"] == "weights inspect"
    assert ("tensors" in output["data"]) == raw
    assert json.loads(Path(output["run_file"]).read_text())["data"] == output["data"]


def test_human_and_error(repo, checkpoint, capsys, monkeypatch):
    monkeypatch.setattr(cli, "find_repo_root", lambda _: repo)
    assert cli.main(["inspect", "--local", "--path", str(checkpoint)]) == 0
    output = capsys.readouterr().out
    for expected in (
        "Files",
        "Safetensors payloads",
        "model-mtp.safetensors",
        "moe_experts",
        "mul1",
    ):
        assert expected in output
    assert cli.main(["inspect", "--local", "--json"]) == 1
    assert "--path" in json.loads(capsys.readouterr().out)["error"]


def test_absent_configs(tmp_path):
    data = _inspect.inspect_directory(tmp_path)
    assert data["configs"]["config"] is None
    assert data["groups"] == [] and not data["sha256sums_present"]


def test_expert_bpw_without_sign_vectors(tmp_path):
    write_weights(
        tmp_path / "model.safetensors", [("0.experts.down_proj.1.trellis", "I16", [1, 1, 56])]
    )
    path = tmp_path / "quantization_config.json"
    path.write_text('{"bits":{"0.experts.down_proj.1":3.5}}')
    (layer,) = _inspect.inspect_directory(tmp_path)["exl3"]["per_layer_summary"]
    assert layer["expert_parameters"] == 256 and layer["effective_expert_bpw"] == 3.5
    path.unlink()
    (layer,) = _inspect.inspect_directory(tmp_path)["exl3"]["per_layer_summary"]
    assert layer["effective_expert_bpw"] is None


def test_module_quantization_metadata():
    data = _inspect.quantization_summary(
        {},
        {
            "quant_method": "exl3",
            "revision": 1,
            "modules": {
                "0.self_attn.k_proj.linear": {
                    "bits": 6,
                    "bpw": 6.02,
                    "group_size": 32,
                    "codebook": "mul1",
                }
            },
        },
    )
    assert data["metadata"] == {"quant_method": "exl3", "revision": 1}
    assert data["bpw"] == [6.02] and data["group_sizes"] == [32]
    assert data["bits"] == [6] and data["counts_by_codebook"] == {"mul1": 1}
    assert data["raw_examples"]["modules"]["0.self_attn.k_proj.linear"]["bits"] == 6


def test_kld_parser():
    parse = cli.build_parser().parse_args
    args = parse(["kld"])
    assert (args.ref, args.cand, args.case, args.chunk_size) == ("baseline", "current", [], None)
    assert not any((args.local, args.json, args.all_positions, args.no_build))
    args = parse(
        [
            "--json",
            "kld",
            "--ref",
            "v1",
            "--cand",
            "v2",
            "--path",
            "model",
            "--case",
            "math",
            "--case",
            "code",
            "--chunk-size",
            "3",
            "--all-positions",
            "--no-build",
            "--local",
        ]
    )
    assert args.json and args.all_positions and args.no_build and args.local
    assert (args.ref, args.cand, args.path, args.chunk_size, args.case) == (
        "v1",
        "v2",
        "model",
        3,
        ["math", "code"],
    )
    with pytest.raises(SystemExit):
        parse(["kld", "--ref", "../candidate"])


def test_kld_sequences(tmp_path):
    from tools.weights_cli._kld import sequences

    cases = [dict(name="a", input_ids=[20]), dict(name="b", input_ids=[30, 40])]
    path = tmp_path / "e2e_a.safetensors"
    write_weights(
        path, [("generated_ids", "I32", [3]), ("unused", "F16", [2]), ("input_ids", "I32", [2])]
    )
    with path.open("r+b") as stream:
        stream.seek(-24, 2)
        stream.write(struct.pack("<6i", 3, 4, 5, 999, 1, 2))
    assert sequences(tmp_path, cases, [], False) == [
        dict(name="a", token_ids=[1, 2, 3, 4], eval_start=1),
        dict(name="b", token_ids=[30, 40], eval_start=0),
    ]
    assert sequences(tmp_path, cases, ["a"], True) == [
        dict(name="a", token_ids=[1, 2, 3, 4], eval_start=0)
    ]
    with pytest.raises(ValueError, match="Unknown workload"):
        sequences(tmp_path, cases, ["absent"], False)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="invalid input_ids"):
        sequences(tmp_path, cases, [], False)


@pytest.mark.parametrize(
    ("resident", "flags", "build", "failure"),
    [
        (True, [], True, False),
        (True, ["--no-build"], False, False),
        (True, ["--cand", "saved"], False, False),
        (False, [], False, False),
        (False, ["--no-build"], False, True),
        (True, [], True, True),
    ],
)
def test_kld_lifecycle(cfg, resident, flags, build, failure):
    args = cli.build_parser().parse_args(["kld", "--local", "--path", "model", *flags])
    cases = [dict(name="a", token_ids=[1, 2, 3], eval_start=1)]
    result = RunResult("weights kld", True, {})
    calls = []

    def host_call(ex, op, payload):
        calls.append(op)
        if op == "status":
            return dict(running=resident, weights_bytes=12, model_dir="model")
        if op == "kld":
            assert payload["cases"] == cases
            assert payload["so_ref"] == ".forge/candidates/baseline.so"
            if failure:
                raise RuntimeError("bad candidate")
            return {"num_positions": 2}
        assert op == "stop"

    with (
        patch.object(commands.forge, "host_call", side_effect=host_call),
        patch.object(commands, "Executor") as executor,
        patch.object(commands.forge, "start") as start,
        patch.object(commands.forge, "build") as rebuild,
    ):
        executor.return_value.run.return_value.stdout = json.dumps(cases)
        if failure:
            with pytest.raises(RuntimeError, match="bad candidate"):
                commands.kld(cfg, args, result)
        else:
            commands.kld(cfg, args, result)
            assert result.data == {"num_positions": 2, "resident": resident}
        assert calls == (["status", "kld"] if resident else ["status", "kld", "stop"])
        assert rebuild.called is build
        if build:
            assert rebuild.call_args.args[-1] == ["forward"]
        assert start.called is not resident
        if not resident:
            options = start.call_args.args[1]
            assert not options.reset_baseline
            assert options.kv_seq_len == 3 and options.no_build == args.no_build


def test_kld_rejects_no_model(cfg):
    args = cli.build_parser().parse_args(["kld", "--local"])
    with patch.object(
        commands.forge, "host_call", return_value=dict(running=True, weights_bytes=0)
    ):
        with pytest.raises(ValueError, match="--no-model"):
            commands.kld(cfg, args, RunResult("weights kld", True, {}))


def test_kld_human_format():
    metrics = dict(
        num_positions=3, mean_kl=0, p95_kl=0, max_kl=0, top1_match_rate=1, max_abs_logit_diff=0
    )
    data = dict(
        metrics,
        ref="baseline",
        cand="current",
        ref_sha256="a" * 64,
        cand_sha256="a" * 64,
        same_binary=True,
        elapsed_ms=42,
        unload_loss_bytes=0,
        cases=[dict(metrics, name="math")],
    )
    out = commands.human_format(RunResult("weights kld", True, data))
    for field in (
        "baseline → current",
        "ref aaaaaaaaaaaa  cand aaaaaaaaaaaa",
        "WARNING: ref and cand are identical binaries (aaaaaaaaaaaa)",
        "math: 3 positions",
        "total: 3 positions",
        "top-1 100.00%",
        "unload loss 0 B",
    ):
        assert field in out

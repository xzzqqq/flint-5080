"""Offline weight inspection and reference fixture commands."""

import json
import shlex
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from tools.common.config import ConfigError
from tools.common.gpu_lock import GpuLeaseManager
from tools.common.remote import Executor

from . import _inspect


def inspect(cfg, args, result):
    path = args.path
    if not path:
        if not cfg.model.weights_dir:
            raise ConfigError("Set model.weights_dir in .dev.toml or pass --path.")
        path = Path(cfg.shared.models_dir) / cfg.model.weights_dir
    if args.local or not cfg.is_remote:
        result.data.update(_inspect.inspect_directory(path, include_tensors=args.tensors))
    else:
        source = Path(_inspect.__file__).read_text(encoding="utf-8")
        command = ["python3", "-c", source, str(path)]
        if args.tensors:
            command.append("--tensors")
        response = Executor(cfg).run(shlex.join(command), cwd=".", check=True)
        result.data.update(json.loads(response.stdout))


def golden(cfg, args, result):
    if args.local:
        cfg = replace(cfg, remote=replace(cfg.remote, enabled=False))
    if not args.path and not cfg.model.weights_dir:
        raise ConfigError("Set model.weights_dir in .dev.toml or pass --path.")
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.skip_e2e and args.skip_ops:
        raise ValueError("Select at least one fixture tier.")
    path = Path(args.path or Path(cfg.shared.models_dir) / cfg.model.weights_dir)
    out = args.out or str(Path(cfg.weights.golden_dir or ".forge/golden") / path.name)
    python = args.python or cfg.weights.python or ("python3" if cfg.is_remote else sys.executable)
    command = [
        python,
        "-m",
        "tools.weights_cli._golden",
        "--path",
        str(path),
        "--out",
        out,
        "--device",
        args.device,
        "--cases",
        args.cases,
    ]
    for case in args.case:
        command.extend(["--case", case])
    if args.steps is not None:
        command.extend(["--steps", str(args.steps)])
    if args.skip_ops:
        command.append("--skip-ops")
    if args.skip_e2e:
        command.append("--skip-e2e")
    executor = Executor(cfg)
    if cfg.is_remote:
        executor.sync()
    lease = (
        GpuLeaseManager(cfg, executor).hold("weights-golden")
        if args.device != "cpu"
        else nullcontext()
    )
    with lease:
        reply = executor.run(shlex.join(command), stream_stderr=True, echo_stdout=False, check=True)
    result.data.update(json.loads(reply.stdout))


def chat_golden(cfg, args, result):
    if args.local:
        cfg = replace(cfg, remote=replace(cfg.remote, enabled=False))
    if not args.path and not cfg.model.weights_dir:
        raise ConfigError("Set model.weights_dir in .dev.toml or pass --path.")
    path = args.path or str(Path(cfg.shared.models_dir) / cfg.model.weights_dir)
    python = args.python or cfg.weights.python or ("python3" if cfg.is_remote else sys.executable)
    executor = Executor(cfg)
    if cfg.is_remote:
        executor.sync()
    command = shlex.join([python, "-m", "tools.weights_cli._chat_golden", path])
    reply = executor.run(command, check=True, stream_stderr=True, echo_stdout=False)
    data = json.loads(reply.stdout)
    counts = {
        key: len(data[key]) for key in ("template_cases", "template_error_cases", "tokenizer_cases")
    }
    out = cfg.repo_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(reply.stdout, encoding="utf-8")
    result.data.update(out=str(out), **counts)


def human_format(result):
    if not result.ok:
        return f"{result.command} FAILED\n{result.error}"
    data = result.data
    if result.command == "weights chat-golden":
        return f"weights chat-golden OK  {data['out']}"
    if result.command == "weights golden":
        lines = [f"weights golden OK  {data['out_dir']}", *data["files"]]
        for case in data["cases"]:
            lines.append(f"  {case['name']}  {case['text']!r}")
        if check := data.get("self_verification"):
            lines.append(
                f"Cached decode: {check['prefill_vs_decode_top1_match']:.1%} top-1, "
                f"max logit difference {check['prefill_vs_decode_max_abs_err']:.6g}"
            )
        peak_mib = data["peak_vram_bytes"] / 2**20
        lines.append(f"{data['wall_seconds']:.1f} s, peak GPU allocation {peak_mib:.1f} MiB")
        return "\n".join(lines)
    lines = [
        f"{result.command} OK",
        data["weights_dir"],
        f"Files ({len(data['files'])}, {data['total_dir_bytes']:,} bytes)",
    ]
    lines.extend(f"  {f['size_bytes']:>16,}  {f['path']}" for f in data["files"])
    tensor_count = sum(h["tensor_count"] for h in data["safetensors"])
    tensor_bytes = sum(h["total_tensor_bytes"] for h in data["safetensors"])
    lines.append(f"Safetensors payloads ({tensor_count:,} tensors, {tensor_bytes:,} bytes)")
    lines.extend(
        f"  {h['file']}  {h['tensor_count']:,} tensors  {h['total_tensor_bytes']:,} bytes"
        for h in data["safetensors"]
    )
    lines.append("Components")
    for name, values in data["exl3"]["component_breakdown"].items():
        lines.append(f"  {name:<16} {values['count']:>6,} tensors  {values['bytes']:>16,} bytes")
    tile = data["exl3"]["codebook_and_tile_summary"]
    layers = data["exl3"]["per_layer_summary"]
    lines.append(f"EXL3  codebooks={tile['codebooks']}  bits={tile['bits']}  layers={len(layers)}")
    return "\n".join(lines)

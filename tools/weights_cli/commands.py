"""Offline weight inspection and reference fixture commands."""

from __future__ import annotations

import json
import shlex
import sys
from argparse import Namespace
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from tools.common.gpu_lock import GpuLeaseManager
from tools.common.remote import Executor
from tools.forge_cli import commands as forge

from . import _inspect


def inspect(cfg, args, result):
    path = cfg.resolve_model_dir(args.path)
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
    path = cfg.resolve_model_dir(args.path)
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.skip_e2e and args.skip_ops and args.skip_mtp:
        raise ValueError("Select at least one fixture tier.")
    out = args.out or str(cfg.resolve_golden_dir(path))
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
    if args.skip_mtp:
        command.append("--skip-mtp")
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
    path = cfg.resolve_model_dir(args.path)
    python = args.python or cfg.weights.python or ("python3" if cfg.is_remote else sys.executable)
    executor = Executor(cfg)
    if cfg.is_remote:
        executor.sync()
    command = shlex.join([python, "-m", "tools.weights_cli._chat_golden", str(path)])
    reply = executor.run(command, check=True, stream_stderr=True, echo_stdout=False)
    data = json.loads(reply.stdout)
    counts = {
        key: len(data[key]) for key in ("template_cases", "template_error_cases", "tokenizer_cases")
    }
    out = cfg.repo_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(reply.stdout, encoding="utf-8")
    result.data.update(out=str(out), **counts)


def kld(cfg, args, result):
    if args.local:
        cfg = replace(cfg, remote=replace(cfg.remote, enabled=False))
    ex = Executor(cfg)
    if cfg.is_remote:
        ex.sync()
    status = forge.host_call(ex, "status", {})
    resident = status["running"]
    if resident and not status["weights_bytes"]:
        raise ValueError("Resident forge-host uses --no-model. Stop it before running KLD.")
    path = cfg.resolve_model_dir(args.path or (status.get("model_dir") if resident else None))
    if resident and str(path) != status["model_dir"]:
        raise ValueError(f"Resident forge-host has a different model loaded: {status['model_dir']}")
    python = "python3" if cfg.is_remote else sys.executable
    command = shlex.join(
        [python, "-m", "tools.weights_cli._kld", str(cfg.resolve_golden_dir(path))]
    )
    selection = dict(names=args.case, all_positions=args.all_positions)
    reply = ex.run(command, input_text=json.dumps(selection), check=True)
    cases = json.loads(reply.stdout)
    request = dict(
        command="kld",
        so_ref=forge.candidate_path(args.ref),
        so_cand=forge.candidate_path(args.cand),
        cases=cases,
    )
    if args.chunk_size is not None:
        request["chunk_size"] = args.chunk_size
    if resident:
        if "current" in (args.ref, args.cand) and not args.no_build:
            forge.build(cfg, ex, result.data, ["forward"])
    else:
        start_args = Namespace(
            path=str(path),
            no_model=False,
            no_build=args.no_build,
            kv_seq_len=max(len(case["token_ids"]) for case in cases),
            scratch_mb=None,
        )
        forge.start(cfg, start_args, ex, GpuLeaseManager(cfg, ex), {}, transient=True)
    try:
        result.data.update(forge.host_call(ex, "kld", request))
        result.data["resident"] = resident
    finally:
        if not resident:
            forge.host_call(ex, "stop", {"force": True})


def human_format(result):
    if not result.ok:
        return f"{result.command} FAILED\n{result.error}"
    data = result.data
    if result.command == "weights kld":
        lines = [f"weights kld OK  {data['ref']} → {data['cand']}"]
        for row in [*data["cases"], dict(data, name="total")]:
            lines.append(
                f"  {row['name']}: {row['num_positions']} positions, KL mean/p95/max "
                f"{row['mean_kl']:.6g}/{row['p95_kl']:.6g}/{row['max_kl']:.6g}, "
                f"top-1 {row['top1_match_rate']:.2%}, max |Δlogit| {row['max_abs_logit_diff']:.6g}"
            )
        lines.append(f"{data['elapsed_ms']:.1f} ms, unload loss {data['unload_loss_bytes']} B")
        return "\n".join(lines)
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

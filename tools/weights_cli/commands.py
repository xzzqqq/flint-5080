"""Offline weight inspection commands."""

import json
import shlex
from pathlib import Path

from tools.common.config import ConfigError
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


def human_format(result):
    if not result.ok:
        return f"{result.command} FAILED\n{result.error}"
    data = result.data
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

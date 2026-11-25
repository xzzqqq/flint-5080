"""Read checkpoint metadata. This module also runs verbatim on the remote Python 3.10."""

import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

LAYER = re.compile(r"^(?:(?:.*\.)?layers\.)?(\d+)\.")
PACKED = {"trellis", "suh", "svh", "mcg_mult", "mul1"}
FIELDS = {
    "quant_method": "methods",
    "method": "methods",
    "codebook": "codebooks",
    "bits": "bits",
    "bpw": "bpw",
    "group_size": "group_sizes",
}


def pattern(name):
    parts = name.split(".")
    for index, part in enumerate(parts):
        if part.isdecimal():
            parts[index] = "{j}" if "experts" in parts[:index] else "{i}"
    return ".".join(parts)


def component(file, name):
    if file == "model-mtp.safetensors":
        return "mtp_dspark"
    if file == "model-vision.safetensors":
        return "vision"
    parts = name.split(".")
    if "experts" in parts:
        return "moe_experts"
    if "controllers" in parts:
        return "loop_or_other"
    if "embed" in parts or "embed_tokens" in parts:
        return "embed"
    if "unembed" in parts or "lm_head" in parts:
        return "lm_head"
    if "router" in parts or ".mlp.gate." in name:
        return "router"
    if any(part.endswith("norm") for part in parts):
        return "norms"
    if {"attn", "self_attn", "attn_sconv"}.intersection(parts):
        return "attention"
    return "loop_or_other"


def tally():
    return {"count": 0, "bytes": 0}


def add(bucket, nbytes):
    bucket["count"] += 1
    bucket["bytes"] += nbytes


def read_safetensors(path, file):
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{file}: truncated safetensors length")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= min(size - 8, 100_000_000):
            raise ValueError(f"{file}: invalid safetensors header length {length}")
        try:
            header = json.loads(stream.read(length))
        except ValueError as exc:
            raise ValueError(f"{file}: invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{file}: expected safetensors header object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(isinstance(v, str) for v in metadata.values()):
        raise ValueError(f"{file}: invalid __metadata__")

    tensors = []
    ranges = []
    dtypes = defaultdict(tally)
    payload_size = size - 8 - length
    for name, entry in header.items():
        try:
            dtype, shape = entry["dtype"], entry["shape"]
            start, end = entry["data_offsets"]
            if not (
                isinstance(dtype, str)
                and isinstance(shape, list)
                and all(type(n) is int and n >= 0 for n in shape)
                and type(start) is int
                and type(end) is int
                and 0 <= start <= end <= payload_size
            ):
                raise ValueError
            width = re.fullmatch(r"[UIF](\d+)(?:_.*)?", dtype)
            bits = int(width[1]) if width else {"BF16": 16, "BOOL": 8}.get(dtype)
            if bits and math.prod(shape) * bits != (end - start) * 8:
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{file}: invalid tensor header for {name}") from exc
        ranges.append((start, end))
        add(dtypes[dtype], end - start)
        tensors.append(dict(file=file, name=name, dtype=dtype, shape=shape, nbytes=end - start))
    offset = 0
    for start, end in sorted(ranges):
        if start != offset:
            raise ValueError(f"{file}: overlapping or missing tensor data at {offset}")
        offset = end
    if offset != payload_size:
        raise ValueError(f"{file}: tensor offsets do not cover the payload")
    return dict(
        file=file,
        file_size_bytes=size,
        header_bytes=length,
        metadata=metadata,
        tensor_count=len(tensors),
        total_tensor_bytes=offset,
        dtypes=dict(dtypes),
    ), tensors


def quantization_summary(config, quant):
    """The Inkling export stores per-module bits separately from global EXL3 metadata."""
    metadata = (config or {}).get("quantization_config") or {}
    fields = {key: set() for key in FIELDS.values()}
    modules = defaultdict(dict)
    top = {}
    for key, value in (quant or {}).items():
        if key in FIELDS and isinstance(value, dict):
            for name, setting in value.items():
                modules[name][key] = setting
        elif key in ("modules", "entries") and isinstance(value, dict):
            for name, settings in value.items():
                modules[name].update(settings)
        else:
            top[key] = value
    for settings in (metadata, top, *modules.values()):
        for key, field in FIELDS.items():
            value = settings.get(key)
            if isinstance(value, (str, int, float)):
                fields[field].add(value)
    defaults = {**metadata, **top}
    samples = {}
    by_bit, by_codebook, by_pattern = Counter(), Counter(), Counter()
    for name, settings in modules.items():
        bits = settings.get("bits", defaults.get("bits"))
        codebook = settings.get("codebook", defaults.get("codebook"))
        if bits is not None:
            by_bit[str(bits)] += 1
        if codebook is not None:
            by_codebook[codebook] += 1
        normalized = pattern(name)
        by_pattern[normalized] += 1
        samples.setdefault((normalized, bits, codebook), name)
    # Preserve the exact field -> module -> value representation for mapped fields.
    examples = {}
    for name in list(samples.values())[:8]:
        for key, value in quant.items():
            if isinstance(value, dict) and name in value:
                examples.setdefault(key, {})[name] = value[name]
    return {
        "top_level_fields": list(quant or {}),
        "metadata": top,
        "config_metadata": metadata,
        **{key: sorted(values) for key, values in fields.items()},
        "entry_count": len(modules),
        "counts_by_bits": dict(by_bit),
        "counts_by_codebook": dict(by_codebook),
        "counts_by_module_pattern": dict(by_pattern),
        "raw_examples": examples,
    }


def expert_parameters(rows, bits_by_module):
    """Count logical matrix elements once, including dense experts when present."""
    modules = defaultdict(dict)
    total = 0
    for t in rows:
        module, _, suffix = t["name"].rpartition(".")
        if suffix in PACKED:
            modules[module][suffix] = t
        else:
            total += math.prod(t["shape"])
    for module, tensors in modules.items():
        if "suh" in tensors and "svh" in tensors:
            total += math.prod(tensors["suh"]["shape"]) * math.prod(tensors["svh"]["shape"])
        else:
            bits = (
                bits_by_module.get(module) if isinstance(bits_by_module, dict) else bits_by_module
            )
            if "trellis" not in tensors or not isinstance(bits, (int, float)) or bits <= 0:
                return None
            total += tensors["trellis"]["nbytes"] * 8 / bits
    return total


def pattern_groups(tensors):
    groups = {}
    for tensor in tensors:
        file, name = tensor["file"], tensor["name"]
        normalized = pattern(name)
        key = file, normalized
        if key not in groups:
            groups[key] = dict(
                file=file,
                pattern=normalized,
                count=0,
                dtypes=set(),
                shapes=set(),
                bytes_per_tensor=set(),
                total_bytes=0,
                sample_name=name,
            )
        group = groups[key]
        group["count"] += 1
        group["total_bytes"] += tensor["nbytes"]
        group["bytes_per_tensor"].add(tensor["nbytes"])
        group["dtypes"].add(tensor["dtype"])
        group["shapes"].add(tuple(tensor["shape"]))

    for group in groups.values():
        group["dtypes"] = sorted(group["dtypes"])
        group["shapes"] = [list(shape) for shape in sorted(group["shapes"])]
        group["bytes_per_tensor"] = sorted(group["bytes_per_tensor"])

    return [groups[key] for key in sorted(groups)]


def layer_summary(layers, quant):
    bits = (quant or {}).get("bits", {})
    per_layer = []
    for layer, rows in sorted(layers.items()):
        experts = [t for part, t in rows if part == "moe_experts"]
        nbytes = sum(t["nbytes"] for t in experts)
        total = sum(t["nbytes"] for _, t in rows)
        params = expert_parameters(experts, bits)
        per_layer.append(
            dict(
                layer=layer,
                total_bytes=total,
                expert_bytes=nbytes,
                non_expert_bytes=total - nbytes,
                expert_tensor_count=len(experts),
                expert_parameters=params,
                effective_expert_bpw=8 * nbytes / params if params else None,
            )
        )
    return per_layer


def inventory(tensors, quant, summary):
    layers = defaultdict(list)
    suffixes, components = defaultdict(tally), {}
    remainder = defaultdict(tally)
    tiles = defaultdict(set)
    for tensor in tensors:
        file, name = tensor["file"], tensor["name"]
        dtype, shape, size = tensor["dtype"], tuple(tensor["shape"]), tensor["nbytes"]
        suffix = name.split(".")[-1]
        part = component(file, name)
        add(suffixes[suffix], size)
        if part == "loop_or_other":
            add(remainder[pattern(name)], size)
        if part not in components:
            components[part] = {**tally(), "quantized": tally(), "unquantized": tally()}
        bucket = components[part]
        add(bucket, size)
        add(bucket["quantized" if suffix in PACKED else "unquantized"], size)
        if suffix in PACKED or suffix == "scale":
            tiles[suffix].add((dtype, shape))
        if file == "model.safetensors" and (match := LAYER.match(name)):
            layers[int(match[1])].append((part, tensor))

    codebooks = set(summary["codebooks"]) | ({"mul1"} & tiles.keys())
    return {
        "suffix_breakdown": dict(suffixes),
        "component_breakdown": components,
        "loop_or_other_patterns": dict(remainder),
        "per_layer_summary": layer_summary(layers, quant),
        "codebook_and_tile_summary": {
            "codebooks": sorted(codebooks),
            "bits": summary["bits"],
            "by_suffix": {
                key: [dict(dtype=dtype, shape=list(shape)) for dtype, shape in sorted(values)]
                for key, values in tiles.items()
            },
        },
    }


def inspect_directory(directory, include_tensors=False):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Checkpoint directory does not exist or is not a directory: {root}")
    files, tensors, headers = [], [], []
    sizes = {}
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        file = path.relative_to(root).as_posix()
        sizes[file] = path.stat().st_size
        kind = path.suffix.lstrip(".") or path.name
        files.append(dict(path=file, size_bytes=sizes[file], kind=kind))
        if path.suffix == ".safetensors":
            header, rows = read_safetensors(path, file)
            headers.append(header)
            tensors.extend(rows)

    def read(key):
        file = key + ".json"
        if file not in sizes:
            return None
        value = json.loads((root / file).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{file}: expected a JSON object")
        return value

    configs = {
        k: read(k)
        for k in ("config", "generation_config", "processor_config", "special_tokens_map")
    }
    tokenizer = read("tokenizer_config") or {}
    configs["tokenizer_config_summary"] = {
        **{
            k: v
            for k, v in tokenizer.items()
            if k not in ("added_tokens_decoder", "chat_template", "response_template")
        },
        "added_tokens_count": len(tokenizer.get("added_tokens_decoder", {})),
        "chat_template_jinja_bytes": sizes.get("chat_template.jinja"),
        "tokenizer_json_bytes": sizes.get("tokenizer.json"),
    }
    quant = read("quantization_config")
    summary = configs["quantization_summary"] = quantization_summary(configs["config"], quant)
    result = dict(
        weights_dir=str(root),
        files=files,
        total_dir_bytes=sum(sizes.values()),
        configs=configs,
        safetensors=headers,
        exl3=inventory(tensors, quant, summary),
        groups=pattern_groups(tensors),
        sha256sums_present="SHA256SUMS" in sizes,
    )
    if result["sha256sums_present"]:
        lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        result["sha256sums_entry_count"] = sum(
            bool(re.match(r"^\\?[0-9a-fA-F]{64} [ *]", line)) for line in lines
        )
    if include_tensors:
        result["tensors"] = tensors
    return result


if __name__ == "__main__":
    print(json.dumps(inspect_directory(sys.argv[1], "--tensors" in sys.argv[2:])))

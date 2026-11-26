"""Prepare teacher-forcing sequences on the execution host using only the standard library."""

import json
import struct
import sys
from pathlib import Path

from ._inspect import read_header


def golden_ids(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        header, length = read_header(stream, size, path)
        body = 8 + length

        def read_ids(name):
            try:
                tensor = header[name]
                (count,) = tensor["shape"]
                code = {"I32": "i", "I64": "q"}[tensor["dtype"]]
                begin, end = tensor["data_offsets"]
                if not (
                    type(count) is int
                    and count > 0
                    and type(begin) is type(end) is int
                    and 0 <= begin <= end <= size - body
                    and end - begin == count * struct.calcsize("<" + code)
                ):
                    raise ValueError
                stream.seek(body + begin)
                return list(struct.unpack(f"<{count}{code}", stream.read(end - begin)))
            except (KeyError, TypeError, ValueError, struct.error) as exc:
                raise ValueError(f"{path}: invalid {name} tensor") from exc

        return read_ids("input_ids"), read_ids("generated_ids")


def sequences(golden_dir, cases, names, all_positions):
    by_name = {case["name"]: case for case in cases}
    selected = []
    for name in names or by_name:
        if name not in by_name:
            raise ValueError(f"Unknown workload case {name!r}")
        path = golden_dir / f"e2e_{name}.safetensors"
        if path.exists():
            prompt, generated = golden_ids(path)
            tokens, start = prompt + generated[:-1], 0 if all_positions else len(prompt) - 1
        else:
            tokens, start = by_name[name]["input_ids"], 0
        selected.append(dict(name=name, token_ids=tokens, eval_start=start))
    if not selected:
        raise ValueError("No KLD cases selected")
    return selected


if __name__ == "__main__":
    try:
        cases = json.loads(Path("workloads/golden_cases.json").read_text())["cases"]
        print(json.dumps(sequences(Path(sys.argv[1]), cases, **json.load(sys.stdin))))
    except (OSError, ValueError) as exc:
        sys.exit(str(exc))

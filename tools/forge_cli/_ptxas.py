"""Inspect CUDA resources using the execution host's CMake compilation database."""

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

RESOURCE_FIELDS = {
    "registers": "regs",
    "barriers": "barriers",
    "bytes smem": "smem_bytes",
    "bytes stack frame": "stack_bytes",
    "bytes spill stores": "spill_stores_bytes",
    "bytes spill loads": "spill_loads_bytes",
}
RESOURCE = re.compile(r"(\d+) (" + "|".join(RESOURCE_FIELDS) + r"|bytes cmem\[\d+\])")


def parse_resources(stderr, file):
    records = []
    entry = None
    for line in stderr.splitlines():
        if match := re.search(r"Compiling entry function '([^']+)' for '([^']+)'", line):
            symbol, arch = match.groups()
            entry = dict.fromkeys(RESOURCE_FIELDS.values(), 0)
            entry.update(file=file, kernel=symbol, arch=arch, cmem_bytes=0)
            records.append(entry)
        elif entry is not None:
            if "Function properties for " in line and line.split()[-1] != entry["kernel"]:
                entry = None  # Device functions may follow the kernel's own statistics.
                continue
            for amount, unit in RESOURCE.findall(line):
                if unit.startswith("bytes cmem"):
                    entry["cmem_bytes"] += int(amount)
                else:
                    entry[RESOURCE_FIELDS[unit]] = int(amount)
    return records


def cubin_command(entry):
    argv = entry.get("arguments") or shlex.split(entry["command"])
    if Path(argv[0]).name == "ccache":
        argv = argv[1:]
    if Path(argv[0]).name != "nvcc":
        raise ValueError(f"Expected nvcc for {entry['file']}, found {argv[0]}")
    args = []
    tokens = iter(argv)
    for token in tokens:
        if token in {
            "-o",
            "--output-file",
            "-MF",
            "-MT",
            "--dependency-output",
            "-keep-dir",
            "--keep-dir",
        }:
            next(tokens)
        elif token not in {
            "-c",
            "--compile",
            "--keep",
            "-keep",
            "--save-temps",
            "-save-temps",
            "-MD",
            "-MMD",
            "--generate-dependencies-with-compile",
            "--generate-nonsystem-dependencies-with-compile",
        } and not token.startswith(
            ("--output-file=", "--dependency-output=", "--keep-dir=", "-keep-dir=", "-MF", "-MT")
        ):
            args.append(token)
    return [*args, "-cubin", "-o", "/dev/null", "-Xptxas=-v"]


def inspect(root, files, all_ops=False):
    root = root.resolve()
    database = json.loads((root / "engine/build/compile_commands.json").read_text())
    units = {}
    for entry in database:
        path = (Path(entry["directory"]) / entry["file"]).resolve()
        if path.suffix == ".cu" and path.is_relative_to(root):
            units.setdefault(path.relative_to(root), entry)
    if all_ops or not files:
        selected = sorted(path for path in units if path.parent == Path("engine/src/ops"))
    else:
        selected = []
        for name in files:
            path = Path(name)
            absolute = (root / path).resolve()
            matches = [
                p
                for p in units
                if root / p == absolute or (path.parent == Path(".") and p.name == name)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{name}: {'ambiguous basename' if matches else 'no CUDA compile entry'}"
                )
            if matches[0] not in selected:
                selected.append(matches[0])
    if not selected:
        raise ValueError("No CUDA compile entries for engine/src/ops/*.cu")
    records = []
    for path in selected:
        entry = units[path]
        run = subprocess.run(
            cubin_command(entry), cwd=entry["directory"], text=True, capture_output=True
        )
        if run.returncode:
            raise RuntimeError(f"{path}: nvcc failed ({run.returncode})\n{run.stderr.strip()}")
        records.extend(parse_resources(run.stderr, str(path)))
    if records and (filt := shutil.which("c++filt")):
        symbols = "".join(row["kernel"] + "\n" for row in records)
        demangled = subprocess.run([filt], input=symbols, text=True, capture_output=True)
        if demangled.returncode == 0:
            for row, name in zip(records, demangled.stdout.splitlines(), strict=True):
                row["kernel"] = name
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(Path.cwd(), args.files, args.all)))
    except (ValueError, RuntimeError, OSError) as exc:
        sys.exit(str(exc))

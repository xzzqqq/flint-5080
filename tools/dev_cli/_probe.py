"""Read-only host inventory, executable before the remote checkout exists."""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path


def query(args):
    try:
        proc = subprocess.run(args, text=True, errors="replace", capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if proc.returncode:
        return None, proc.stderr.strip() or f"exit {proc.returncode}"
    return proc.stdout.strip(), None


def smi(table, columns):
    output, error = query(["nvidia-smi", f"--query-{table}=" + ",".join(columns),
                           "--format=csv,noheader,nounits"])
    if error:
        return {"error": error}
    rows = csv.reader(io.StringIO(output), skipinitialspace=True)
    return {"rows": [dict(zip(columns, row)) for row in rows]}


def inventory(workspace):
    versions = {}
    for tool in ("nvcc", "cmake", "ninja", "ccache", "cargo", "rustc", "mold", "ld.lld"):
        output, error = query([tool, "--version"])
        lines = (output or "").splitlines()
        version = next((s for s in lines if "release " in s), lines[0] if lines else "")
        versions[tool] = {"error": error} if error else {"version": version}
    return {
        "workspace_exists": Path(workspace).is_dir(),
        "gpu": smi("gpu", ["index", "uuid", "name", "temperature.gpu", "utilization.gpu",
                           "memory.used", "memory.total"]),
        "compute_processes": smi("compute-apps", ["gpu_uuid", "pid", "process_name", "used_memory"]),
        "toolchain": versions,
    }


if __name__ == "__main__":
    print(json.dumps(inventory(sys.argv[1])))

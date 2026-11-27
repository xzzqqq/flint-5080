"""Summarize a single Nsight Systems capture and compare archived captures."""

import csv
import json
import subprocess
import tempfile
from pathlib import Path


def _read_stats(path: Path, *, kernel: bool) -> list[dict]:
    rows = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                item = dict(
                    name=row["Name"],
                    total_ms=float(row["Total Time (ns)"]) / 1e6,
                    avg_ms=float(row["Avg (ns)"]) / 1e6,
                )
                pct = float(row["Time (%)"])
                if kernel:
                    item.update(
                        instances=int(row["Instances"]),
                        min_ms=float(row["Min (ns)"]) / 1e6,
                        max_ms=float(row["Max (ns)"]) / 1e6,
                        pct_gpu=pct,
                    )
                else:
                    item.update(calls=int(row["Num Calls"]), pct_api=pct)
                rows.append(item)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise ValueError(f"Invalid or missing Nsight table {path.name}: {exc}") from exc
    if not rows:
        raise ValueError(f"Empty Nsight table {path.name}")
    return sorted(rows, key=lambda row: row["total_ms"], reverse=True)


def analyze_report(report_path: Path, wall_ms: float | None = None, top_k: int = 20) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory) / "stats"
        run = subprocess.run(
            [
                "nsys",
                "stats",
                "--sqlite",
                str(base.with_suffix(".sqlite")),
                "--report",
                "cuda_gpu_kern_sum,cuda_api_sum",
                "--format",
                "csv",
                "--timeunit",
                "ns",
                "--output",
                str(base),
                str(report_path),
            ],
            capture_output=True,
            text=True,
        )
        if run.returncode:
            raise RuntimeError(f"nsys stats failed: {run.stderr.strip() or run.stdout.strip()}")
        kernels = _read_stats(base.with_name("stats_cuda_gpu_kern_sum.csv"), kernel=True)
        api = _read_stats(base.with_name("stats_cuda_api_sum.csv"), kernel=False)
    total = sum(row["total_ms"] for row in kernels)
    # Summed durations can exceed wall time when streams overlap.
    return dict(
        top_kernels=kernels[:top_k],
        top_cuda_api=api[:10],
        kernel_totals_ms={row["name"]: row["total_ms"] for row in kernels},
        summary=dict(
            total_gpu_kernel_ms=total,
            kernel_instances=sum(row["instances"] for row in kernels),
            wall_ms=wall_ms,
            cpu_launch_gap_ms=max(0.0, wall_ms - total) if wall_ms is not None else None,
            gpu_active_ratio_pct=100 * total / wall_ms if wall_ms else None,
            overlap_detected=wall_ms is not None and total > wall_ms,
        ),
    )


def compare_profiles(path_a: Path, path_b: Path, top_k: int = 20) -> dict:
    def load(path):
        try:
            profile = json.loads(path.read_text())["data"]["nsys_summary"]
            return profile["summary"], profile.get(
                "kernel_totals_ms", {row["name"]: row["total_ms"] for row in profile["top_kernels"]}
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path} is missing forge profile --nsys data") from exc

    a, kernels_a = load(path_a)
    b, kernels_b = load(path_b)
    diffs = []
    for name in sorted(kernels_a.keys() | kernels_b.keys()):
        first, second = kernels_a.get(name, 0), kernels_b.get(name, 0)
        diffs.append(
            dict(
                name=name,
                a_ms=first,
                b_ms=second,
                delta_ms=second - first,
                speedup=first / second if second else None,
            )
        )
    diffs.sort(key=lambda row: abs(row["delta_ms"]), reverse=True)
    summary = {}
    for key in a:
        if key not in b:
            continue
        x, y = a[key], b[key]
        numeric = type(x) in (float, int) and type(y) in (float, int)
        summary[key] = dict(a=x, b=y, delta=y - x if numeric else None)
    return dict(summary=summary, kernel_diffs=diffs[:top_k])

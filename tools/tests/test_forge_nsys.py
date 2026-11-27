import json
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from tools.common.output import RunResult
from tools.forge_cli import _nsys, commands
from tools.forge_cli.__main__ import build_parser


def fake_stats(cmd, **kwargs):
    base = Path(cmd[cmd.index("--output") + 1])
    base.with_name("stats_cuda_gpu_kern_sum.csv").write_text(
        "Time (%),Total Time (ns),Instances,Avg (ns),Min (ns),Max (ns),Name\n"
        "25,1000000,1,1000000,1000000,1000000,small\n"
        '75,3000000,2,1500000,1000000,2000000,"gemm<float, 2>"\n'
    )
    base.with_name("stats_cuda_api_sum.csv").write_text(
        "Time (%),Total Time (ns),Num Calls,Avg (ns),Name\n100,2000,2,1000,launch\n"
    )
    return CompletedProcess(cmd, 0)


@pytest.mark.parametrize(
    "wall,gap,ratio,overlap", [(None, None, None, False), (10, 6, 40, False), (2, 0, 200, True)]
)
def test_stats(wall, gap, ratio, overlap):
    with patch.object(_nsys.subprocess, "run", side_effect=fake_stats) as run:
        data = _nsys.analyze_report(Path("capture.nsys-rep"), wall, 1)
    run.assert_called_once()
    assert data["summary"] == dict(
        total_gpu_kernel_ms=4,
        kernel_instances=3,
        wall_ms=wall,
        cpu_launch_gap_ms=gap,
        gpu_active_ratio_pct=ratio,
        overlap_detected=overlap,
    )
    assert data["top_kernels"] == [
        dict(
            name="gemm<float, 2>",
            instances=2,
            total_ms=3,
            avg_ms=1.5,
            min_ms=1,
            max_ms=2,
            pct_gpu=75,
        )
    ]
    assert data["top_cuda_api"] == [
        dict(name="launch", calls=2, total_ms=0.002, avg_ms=0.001, pct_api=100)
    ]
    output = commands.human_format(RunResult("forge profile", True, {"nsys_summary": data}))
    assert "GPU kernels" in output and "CUDA API" in output


def test_compare_is_local(tmp_path):
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, totals in zip(paths, [{"k": 8, "removed": 1}, {"k": 2, "new": 4}], strict=True):
        profile = dict(
            top_kernels=[], kernel_totals_ms=totals, summary={"wall_ms": sum(totals.values())}
        )
        path.write_text(json.dumps({"data": {"nsys_summary": profile}}))
    args = build_parser().parse_args(["profile", "--compare", *map(str, paths)])
    result = RunResult("forge profile", True, {})
    with patch.object(commands, "Executor", side_effect=AssertionError):
        commands.execute(None, args, result)
    compare = result.data["nsys_compare"]
    rows = compare["kernel_diffs"]
    assert rows[0] == dict(name="k", a_ms=8, b_ms=2, delta_ms=-6, speedup=4)
    assert rows[1]["name"] == "new" and rows[2]["speedup"] is None
    assert compare["summary"]["wall_ms"]["delta"] == -3
    assert "B − A" in commands.human_format(result)
    paths[0].write_text("{}")
    with pytest.raises(ValueError, match="nsys data"):
        _nsys.compare_profiles(*paths)

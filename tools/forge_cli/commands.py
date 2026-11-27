"""Controller-side orchestration, build and workload selection."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from uuid import uuid4

from tools.common.build import engine_build_script, engine_configure_script
from tools.common.daemon import atomic_json
from tools.common.gpu_lock import GpuLeaseManager
from tools.common.remote import Executor

from ._nsys import compare_profiles
from .host import INFO

CURRENT = "engine/build/forward.so"


def candidate_path(name, current=CURRENT):
    return current if name == "current" else f".forge/candidates/{name}.so"


def host_call(executor: Executor, cmd: str, payload: dict) -> dict:
    python = "python3" if executor.cfg.is_remote else sys.executable
    command = shlex.join([python, "-m", "tools.forge_cli.host", cmd])
    reply = executor.run(command, input_text=json.dumps(payload), check=True)
    return json.loads(reply.stdout)


def build(cfg, executor, data, targets):
    sync = executor.sync()
    data["sync_ms"] = sync.duration_ms
    script = engine_build_script(cfg, targets=targets, incremental=True)
    out = executor.run("set -e\n" + script, check=True, stream_stderr=True)
    data["build_ms"] = out.duration_ms


def start(cfg, args, ex, lease, data, *, transient=False):
    if not args.no_build:
        build(cfg, ex, data, ["forge-host", "stub_forward" if args.no_model else "forward"])
    else:
        ex.sync()
    argv = [
        "engine/build/forge-host",
        "--daemon",
        "--socket",
        ".forge/host.sock",
        "--baseline",
        "engine/build/stub_forward.so" if args.no_model else CURRENT,
        "--state-dir",
        ".forge",
    ]
    if not transient and args.reset_baseline:
        argv.append("--reset-baseline")
    if args.no_model:
        argv.append("--no-model")
    else:
        path = cfg.resolve_model_dir(
            args.path, error_message="Set model.weights_dir or pass --path / --no-model"
        )
        if draft := cfg.resolve_draft_dir(args.draft_dir):
            argv.extend(["--draft-dir", str(draft)])
        golden = cfg.resolve_golden_dir(path)
        argv.extend(["--model-dir", str(path), "--golden-dir", str(golden)])
    for key in ("kv_seq_len", "scratch_mb"):
        value = getattr(args, key)
        if value:
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    record = lease.acquire("forge")
    try:
        data.update(
            host_call(
                ex,
                "start",
                dict(
                    argv=argv,
                    token=record.token,
                    lock=cfg.shared.lock_file,
                    ttl=cfg.gpu.lease_ttl_sec,
                ),
            )
        )
    except BaseException:
        lease.release_unbound(record.token)
        raise
    if cfg.is_remote and not transient:
        atomic_json(
            cfg.repo_root / ".forge/daemon.json",
            {
                "pid": data["pid"],
                "heartbeat": data["heartbeat"],
                "host_started_epoch": data["host_started_epoch"],
                "remote": cfg.remote_workspace_dir,
                "no_model": args.no_model,
            },
        )


def modified_cuda_files(root):
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "-z",
            "--no-renames",
            "--untracked-files=all",
            "--",
            "engine",
        ],
        cwd=root,
        capture_output=True,
        check=True,
    )
    return [
        name
        for record in os.fsdecode(status.stdout).split("\0")
        if (name := record[3:]).endswith(".cu") and (root / name).is_file()
    ]


def ptxas(cfg, args, ex):
    files = args.files or ([] if args.all else modified_cuda_files(cfg.repo_root))
    if cfg.is_remote and not args.no_sync:
        ex.sync()
    configure = engine_configure_script(cfg)
    ex.run(
        f"set -e\nif ! test -f engine/build/compile_commands.json; then\n{configure}\nfi",
        check=True,
    )
    cmd = ["python3", "-m", "tools.forge_cli._ptxas"]
    if args.all:
        cmd.append("--all")
    reply = ex.run(shlex.join([*cmd, "--", *files]), check=True)
    return json.loads(reply.stdout)


def execute(cfg, args, result):
    if args.command == "profile" and args.compare:
        result.data["nsys_compare"] = compare_profiles(*args.compare)
        return
    ex = Executor(cfg)
    cmd, data = args.command, result.data
    if cmd == "ptxas":
        data["kernels"] = ptxas(cfg, args, ex)
        return
    lease = GpuLeaseManager(cfg, ex)
    if cmd == "start":
        start(cfg, args, ex, lease, data)
    elif cmd == "stop":
        data.update(host_call(ex, cmd, {"force": args.force}))
        if cfg.is_remote:
            (cfg.repo_root / ".forge/daemon.json").unlink(missing_ok=True)
    elif cmd == "status":
        data.update(lease=lease.inspect_public(), **host_call(ex, cmd, {}))
    elif cmd == "ls":
        data.update(host_call(ex, cmd, {}))
    elif cmd == "rm":
        if args.name == "current" or (args.name == "baseline" and not args.force):
            raise ValueError("current is a build target; removing baseline requires --force")
        data.update(host_call(ex, cmd, {"name": args.name}))
    else:
        info = cfg.repo_root / INFO
        no_model = info.exists() and json.loads(info.read_text()).get("no_model", False)
        target = "stub_forward" if no_model else "forward"
        current = f"engine/build/{target}.so"
        if cmd == "freeze-baseline":
            if args.candidate == "current" and not args.no_build:
                build(cfg, ex, data, [target])
            else:
                ex.sync()
            data.update(host_call(ex, cmd, {"so_path": candidate_path(args.candidate, current)}))
            return
        cases = json.loads((cfg.repo_root / "workloads/golden_cases.json").read_text())["cases"]
        name = args.case or cfg.workload.default_case or cases[0]["name"]
        case = next((case for case in cases if case["name"] == name), None)
        if not case:
            raise ValueError(f"Unknown workload case {name!r}")
        names = (args.cand_a, args.cand_b) if cmd == "ab" else (args.candidate,)
        if "current" in names and not args.no_build:
            build(cfg, ex, data, [target])
        begin, end = args.layers
        q = dict(
            command=cmd,
            case=name,
            step=args.step,
            verify_len=args.verify_len,
            fixed_proposal=args.fixed_proposal,
            input_ids=case["input_ids"],
            begin_visit=begin,
            end_visit=end,
            use_cuda_graph=not getattr(args, "no_graph", False),
            use_speculative=not getattr(args, "no_spec", False),
        )
        if getattr(args, "context_len", None):
            q["context_len"] = args.context_len
        if cmd == "ab":
            q.update(
                so_a=candidate_path(names[0], current),
                so_b=candidate_path(names[1], current),
                rounds=args.rounds,
            )
        else:
            q["so_path"] = candidate_path(names[0], current)
        if cmd in {"run", "ab"}:
            q.update(
                warmup=args.warmup if args.warmup is not None else (10 if args.full else 2),
                iters=args.iters if args.iters is not None else (50 if args.full else 5),
            )
        if cmd == "run":
            q.update(save=args.save)
        elif cmd == "profile":
            q.update(nsys=args.nsys)
        elif cmd == "dump":
            q.update(
                with_state=args.with_state,
                iters=1,
                warmup=0,
                out=args.out or f".forge/dumps/{name}-{begin}-{end}-{uuid4().hex[:8]}.safetensors",
            )
        start_time = time.monotonic()
        data.update(host_call(ex, cmd, q))
        data["reload_run_ms"] = (time.monotonic() - start_time) * 1000


def _table(columns, rows):
    cells = [columns, *rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(columns))]
    return [
        "  " + "  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row[:-1])) + "  " + row[-1]
        for row in cells
    ]


def _number(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _profile_tables(profile):
    lines = ["Nsight Systems", *[f"  {k:<22} {_number(v)}" for k, v in profile["summary"].items()]]
    for title, key, count, pct in [
        ("GPU kernels", "top_kernels", "instances", "pct_gpu"),
        ("CUDA API", "top_cuda_api", "calls", "pct_api"),
    ]:
        lines.append(title)
        fields = [count, "total_ms", "avg_ms", pct, "name"]
        lines.extend(_table(fields, [[_number(row[f]) for f in fields] for row in profile[key]]))
    return lines


def _comparison_tables(compare):
    rows = [
        [_number(delta[f]) for f in ("a", "b", "delta")] + [metric]
        for metric, delta in compare["summary"].items()
    ]
    fields = ["a_ms", "b_ms", "delta_ms", "speedup", "name"]
    return [
        "Nsight Systems comparison (delta = B − A)",
        *_table(["A", "B", "delta", "metric"], rows),
        "GPU kernels",
        *_table(fields, [[_number(row[f]) for f in fields] for row in compare["kernel_diffs"]]),
    ]


def human_format(result):
    lines = [result.command + (" OK" if result.ok else " FAILED")]
    if result.error:
        lines.append(result.error)
    for key, value in result.data.items():
        if key == "kernels":
            fields = [
                "file",
                "arch",
                "regs",
                "smem_bytes",
                "cmem_bytes",
                "stack_bytes",
                "spill_stores_bytes",
                "spill_loads_bytes",
                "barriers",
                "kernel",
            ]
            lines.extend(_table(fields, [[str(row[f]) for f in fields] for row in value]))
        elif key == "host_stale_warning":
            lines.append("WARNING " + value)
        elif key == "nsys_summary":
            lines.extend(_profile_tables(value))
        elif key == "nsys_compare":
            lines.extend(_comparison_tables(value))
        elif key == "stages":
            lines.extend(
                f"  {s['name']:12} {s['elapsed_ms']:.4f} ms  {s['sol_pct']:.2f}% logical SOL"
                for s in value
            )
        else:
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(lines)

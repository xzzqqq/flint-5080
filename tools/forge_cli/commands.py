"""Controller-side orchestration, build and workload selection."""

import json
import shlex
import sys
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from tools.common.gpu_lock import GpuLeaseManager
from tools.common.remote import Executor
from tools.dev_cli.commands import _engine_script

from .host import atomic_json

CURRENT = "engine/build/forward.so"


def candidate_path(name):
    return CURRENT if name == "current" else f".forge/candidates/{name}.so"


def host_call(executor, cmd, payload):
    python = "python3" if executor.cfg.is_remote else sys.executable
    command = shlex.join([python, "-m", "tools.forge_cli.host", cmd, json.dumps(payload)])
    reply = executor.run(command, check=True)
    return json.loads(reply.stdout)


def build(cfg, executor, data, targets):
    sync = executor.sync()
    data["sync_ms"] = sync.duration_ms
    script = _engine_script(cfg, False, targets=targets, incremental=True)
    out = executor.run("set -e\n" + script, check=True, stream_stderr=True)
    data["build_ms"] = out.duration_ms


def start(cfg, args, ex, lease, data):
    if not args.no_build:
        build(cfg, ex, data, ["forge-host", "forward"])
    else:
        ex.sync()
    argv = [
        "engine/build/forge-host",
        "--daemon",
        "--socket",
        ".forge/host.sock",
        "--baseline",
        CURRENT,
        "--state-dir",
        ".forge",
    ]
    if args.no_model:
        argv.append("--no-model")
    else:
        if not args.path and not cfg.model.weights_dir:
            raise ValueError("Set model.weights_dir or pass --path / --no-model")
        path = Path(args.path or Path(cfg.shared.models_dir) / cfg.model.weights_dir)
        golden = Path(cfg.weights.golden_dir or ".forge/golden") / path.name
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
    except Exception:
        lease.release_unbound(record.token)
        raise
    if cfg.is_remote:
        atomic_json(
            cfg.repo_root / ".forge/daemon.json",
            {
                "pid": data["pid"],
                "heartbeat": data["heartbeat"],
                "remote": cfg.remote_workspace_dir,
            },
        )


def execute(cfg, args, result):
    ex = Executor(cfg)
    lease = GpuLeaseManager(cfg, ex)
    cmd, data = args.command, result.data
    if cmd == "start":
        start(cfg, args, ex, lease, data)
    elif cmd == "stop":
        data.update(host_call(ex, cmd, {"force": args.force}))
        if cfg.is_remote:
            (cfg.repo_root / ".forge/daemon.json").unlink(missing_ok=True)
    elif cmd == "status":
        status = asdict(lease.inspect())
        if status["record"]:
            del status["record"]["token"]
        data.update(lease=status, **host_call(ex, cmd, {}))
    elif cmd == "ls":
        data.update(host_call(ex, cmd, {}))
    elif cmd == "rm":
        if args.name == "current" or (args.name == "baseline" and not args.force):
            raise ValueError("current is a build target; removing baseline requires --force")
        data.update(host_call(ex, cmd, {"name": args.name}))
    else:
        cases = json.loads((cfg.repo_root / "workloads/golden_cases.json").read_text())["cases"]
        name = args.case or cfg.workload.default_case or cases[0]["name"]
        case = next((case for case in cases if case["name"] == name), None)
        if not case:
            raise ValueError(f"Unknown workload case {name!r}")
        names = (args.cand_a, args.cand_b) if cmd == "ab" else (args.candidate,)
        if "current" in names and not args.no_build:
            build(cfg, ex, data, ["forward"])
        begin, end = args.layers
        q = dict(
            command=cmd,
            case=name,
            input_ids=case["input_ids"],
            begin_visit=begin,
            end_visit=end,
            use_cuda_graph=not getattr(args, "no_graph", False),
            use_speculative=not getattr(args, "no_spec", False),
        )
        if cmd == "ab":
            q.update(
                so_a=candidate_path(names[0]), so_b=candidate_path(names[1]), rounds=args.rounds
            )
        else:
            q["so_path"] = candidate_path(names[0])
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
                step=args.step,
                with_state=args.with_state,
                iters=1,
                warmup=0,
                out=args.out or f".forge/dumps/{name}-{begin}-{end}-{uuid4().hex[:8]}.safetensors",
            )
        start_time = time.monotonic()
        data.update(host_call(ex, cmd, q))
        data["reload_run_ms"] = (time.monotonic() - start_time) * 1000


def human_format(result):
    lines = [result.command + (" OK" if result.ok else " FAILED")]
    if result.error:
        lines.append(result.error)
    for key, value in result.data.items():
        if key == "stages":
            lines.extend(
                f"  {s['name']:12} {s['elapsed_ms']:.4f} ms  {s['sol_pct']:.2f}% logical SOL"
                for s in value
            )
        else:
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(lines)

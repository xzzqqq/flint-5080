"""Execution-host operations. Also runs on the GPU host's system Python 3.10."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from tools.common.daemon import (
    ReadyProcess,
    spawn_supervisor,
    stop_supervised_process,
    supervise_leased_process,
    wait_for,
)

STATE = Path(".forge")
INFO = STATE / "daemon.json"
SOCKET = STATE / "host.sock"


def ipc(request: dict, timeout: float = 1800) -> dict:
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(timeout)
        client.connect(str(SOCKET))
        client.sendall(json.dumps(request).encode() + b"\n")
        with client.makefile("rb") as reader:
            data = json.loads(reader.readline())
    if not data.pop("ok"):
        raise RuntimeError(data["error"])
    return data


def supervise(cfg: dict) -> None:
    session = "flint-" + uuid4().hex if shutil.which("nsys") else ""
    command = cfg["argv"]
    if session:
        profiles = STATE / "profiles"
        profiles.mkdir(exist_ok=True)
        # Arm the resident process once. Only an explicit profiler API range collects data.
        command = [
            "nsys",
            "profile",
            "--trace=cuda",
            "--cuda-graph-trace=node",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=repeat:sync",
            "-o",
            str(profiles / session),
            *command,
        ]
    started = time.monotonic()

    def check_ready(proc: subprocess.Popen) -> ReadyProcess | None:
        try:
            status = ipc({"command": "status"}, timeout=min(1, cfg["ttl"] / 4))
        except (FileNotFoundError, ConnectionRefusedError, TimeoutError):
            return None
        return ReadyProcess(
            status["pid"],
            dict(
                session=session,
                no_model="--no-model" in cfg["argv"],
                load_ms=(time.monotonic() - started) * 1000,
            ),
        )

    supervise_leased_process(
        STATE,
        command,
        check_ready,
        lock=cfg["lock"],
        token=cfg["token"],
        ttl=cfg["ttl"],
        name="forge-host",
        startup_timeout=90,
        cleanup_paths=(SOCKET,),
    )


def start(cfg: dict) -> dict:
    def started(info: dict) -> dict:
        return dict(
            **ipc({"command": "status"}, timeout=5),
            load_ms=info["load_ms"],
            heartbeat=info["heartbeat"],
        )

    return spawn_supervisor(
        STATE,
        "tools.forge_cli.host",
        cfg,
        name="forge-host",
        log_path=STATE / "host.log",
        startup_timeout=130,
        prepare=lambda: SOCKET.unlink(missing_ok=True),
        started=started,
    )


def stop(force: bool) -> dict:
    def shutdown() -> None:
        ipc({"command": "shutdown"}, timeout=5 if force else 180)

    return stop_supervised_process(
        STATE, force=force, graceful_stop=shutdown, cleanup_paths=(SOCKET,)
    )


def candidates():
    return sorted(p.stem for p in (STATE / "candidates").glob("*.so"))


def run_and_save(q, name):
    dst = STATE / "candidates" / (name + ".so")
    with tempfile.NamedTemporaryFile(dir=dst.parent, suffix=".tmp") as tmp:
        with Path(q["so_path"]).open("rb") as src:
            shutil.copyfileobj(src, tmp)
        tmp.flush()
        data = ipc(dict(q, so_path=tmp.name))
        os.link(tmp.name, dst)
    return dict(data, saved=name)


def capture(q):
    if not shutil.which("nsys"):
        raise RuntimeError("nsys is missing on the execution host")
    session = json.loads(INFO.read_text())["session"]
    if not session:
        raise RuntimeError("Restart forge after installing nsys to enable CUDA collection")
    directory = STATE / "profiles"
    with (STATE / "profile.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        pattern = session + "*.nsys-rep"
        previous = set(directory.glob(pattern))
        data = ipc(dict(q, profiler_range=True, warmup=1, iters=1))
        path = wait_for(
            lambda: next(iter(set(directory.glob(pattern)) - previous), None), 30, "Nsight report"
        )
    data.update(report=str(path), report_bytes=path.stat().st_size)
    return data


def dispatch(op: str, q: dict) -> dict | None:
    if op == "supervise":
        return supervise(q)
    if op == "start":
        return start(q)
    if op == "stop":
        return stop(q["force"])
    if op == "ls":
        return {"candidates": candidates()}
    if op == "rm":
        (STATE / "candidates" / (q["name"] + ".so")).unlink()
        return {"removed": q["name"]}
    if op == "status":
        try:
            return dict(running=True, candidates=candidates(), **ipc({"command": "status"}))
        except (FileNotFoundError, ConnectionRefusedError):
            return dict(running=False, candidates=candidates())
    saved = q.pop("save", None)
    if saved:
        return run_and_save(q, saved)
    data = capture(q) if q.pop("nsys", False) else ipc(q)
    if op == "dump" and not Path(data["path"]).is_file():
        raise RuntimeError("Missing dump on execution host")
    return data


if __name__ == "__main__":
    try:
        result = dispatch(sys.argv[1], json.load(sys.stdin))
        if result is not None:
            print(json.dumps(result))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))

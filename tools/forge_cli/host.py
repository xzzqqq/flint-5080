"""Execution-host operations. Also runs on the GPU host's system Python 3.10."""

import contextlib
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from tools.common._lease import GpuLockError, release_local, renew_local
from tools.common.daemon import alive, atomic_json, identity, wait_for

STATE = Path(".forge")
INFO = STATE / "daemon.json"
SOCKET = STATE / "host.sock"


def ipc(request, timeout=1800):
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(timeout)
        client.connect(str(SOCKET))
        client.sendall(json.dumps(request).encode() + b"\n")
        with client.makefile("rb") as reader:
            data = json.loads(reader.readline())
    if not data.pop("ok"):
        raise RuntimeError(data["error"])
    return data


def supervise():
    cfg = json.loads(sys.stdin.readline())
    token, lock, ttl = cfg["token"], cfg["lock"], cfg["ttl"]
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
    proc = None
    info = None
    try:
        renew_local(lock, token, ttl, pid=os.getpid())
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL)
        started = time.monotonic()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"forge-host launcher exited ({proc.returncode})")
            try:
                status = ipc({"command": "status"})
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() - started > 90:
                    raise RuntimeError("forge-host failed to become ready") from None
                renew_local(lock, token, ttl)
                time.sleep(min(0.2, ttl / 4))
        pid = status["pid"]
        renew_local(lock, token, ttl, pid=pid)
        info = dict(
            pid=pid,
            pid_start=identity(pid),
            heartbeat=os.getpid(),
            heartbeat_start=identity(os.getpid()),
            token=token,
            lock=lock,
            session=session,
            no_model="--no-model" in cfg["argv"],
            load_ms=(time.monotonic() - started) * 1000,
        )
        atomic_json(INFO, info)
        renewed = time.monotonic()
        while alive(info, "pid"):
            proc.poll()
            if time.monotonic() - renewed >= ttl / 3:
                renew_local(lock, token, ttl)
                renewed = time.monotonic()
            time.sleep(min(0.2, ttl / 4))
    finally:
        if info and alive(info, "pid"):
            os.kill(info["pid"], signal.SIGTERM)
        if proc and proc.poll() is None:
            if not info:
                proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        with contextlib.suppress(GpuLockError):
            release_local(lock, token)


def start(cfg):
    STATE.mkdir(mode=0o700, exist_ok=True)
    if INFO.exists():
        existing = json.loads(INFO.read_text())
        if "remote" not in existing and alive(existing, "pid"):
            raise RuntimeError("forge-host is already running")
    INFO.unlink(missing_ok=True)
    SOCKET.unlink(missing_ok=True)
    log_path = STATE / "host.log"
    with log_path.open("w") as log:
        supervisor = subprocess.Popen(
            [sys.executable, "-m", "tools.forge_cli.host", "supervise"],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    supervisor.stdin.write(json.dumps(cfg).encode() + b"\n")
    supervisor.stdin.close()

    def ready():
        if supervisor.poll() is not None:
            raise RuntimeError(log_path.read_text()[-2000:] or "forge-host startup failed")
        return INFO.is_file()

    wait_for(ready, 130, "forge-host startup")
    info = json.loads(INFO.read_text())
    return dict(**ipc({"command": "status"}), load_ms=info["load_ms"], heartbeat=info["heartbeat"])


def stop(force):
    if not INFO.exists():
        return {"stopped": False}
    info = json.loads(INFO.read_text())
    if "remote" in info:
        return {"stopped": False}
    if alive(info, "pid"):
        try:
            ipc({"command": "shutdown"}, timeout=5 if force else 180)
            wait_for(lambda: not alive(info, "pid"), 20, "forge-host exit")
        except (OSError, ValueError, RuntimeError):
            if not force:
                raise
            if alive(info, "pid"):
                os.kill(info["pid"], signal.SIGTERM)
        wait_for(lambda: not alive(info, "pid"), 20, "forge-host exit")
    wait_for(lambda: not alive(info, "heartbeat"), 45, "lease heartbeat exit")
    # Covers unexpected heartbeat termination. A replacement lease owns a different token.
    with contextlib.suppress(GpuLockError):
        release_local(info["lock"], info["token"])
    INFO.unlink(missing_ok=True)
    SOCKET.unlink(missing_ok=True)
    return {"stopped": True}


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


def dispatch(op, q):
    if op in {"start", "stop"}:
        STATE.mkdir(mode=0o700, exist_ok=True)
        with (STATE / "lifecycle.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            return start(q) if op == "start" else stop(q["force"])
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
    if sys.argv[1] == "supervise":
        supervise()
    else:
        try:
            print(json.dumps(dispatch(sys.argv[1], json.loads(sys.argv[2]))))
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            sys.exit(str(exc))

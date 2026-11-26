"""Manage the server and its transferable GPU lease on the execution host."""

import contextlib
import fcntl
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

from tools.common._lease import GpuLockError, release_local, renew_local
from tools.common.daemon import alive, atomic_json, identity, wait_for

STATE = Path(".serve")
INFO = STATE / "daemon.json"
LOG = STATE / "server.log"


def address(bind):
    url = urlsplit("//" + bind)
    if not url.hostname or not url.port:
        raise ValueError("bind must specify a host and a non-zero port")
    host = "127.0.0.1" if url.hostname in {"0.0.0.0", "::"} else url.hostname
    return host, url.port


def ready(bind, timeout=1):
    conn = http.client.HTTPConnection(*address(bind), timeout=timeout)
    try:
        conn.request("GET", "/ready")
        reply = conn.getresponse()
        if reply.status == 200:
            return json.load(reply)
    except (OSError, ValueError, http.client.HTTPException):
        pass
    finally:
        conn.close()
    return None


def supervise(cfg):
    lock, token, ttl = cfg["lock"], cfg["token"], cfg["ttl"]
    proc = None
    published = False
    stopping = False

    def shutdown(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        renew_local(lock, token, ttl, pid=os.getpid())
        env = os.environ.copy()
        if cfg.get("api_key") is not None:
            env["FLINT_API_KEY"] = cfg["api_key"]
        proc = subprocess.Popen([cfg["binary"], *cfg["argv"]], stdin=subprocess.DEVNULL, env=env)
        deadline = time.monotonic() + 120
        while not stopping:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"server exited during startup ({proc.returncode}); see dev serve logs"
                )
            snapshot = ready(cfg["bind"], timeout=min(1, ttl / 4))
            if snapshot is not None and proc.poll() is None:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("server failed to become ready; see dev serve logs")
            renew_local(lock, token, ttl)
            time.sleep(min(0.2, ttl / 4))
        if stopping:
            return
        renew_local(lock, token, ttl, pid=proc.pid)
        atomic_json(
            INFO,
            dict(
                pid=proc.pid,
                pid_start=identity(proc.pid),
                heartbeat=os.getpid(),
                heartbeat_start=identity(os.getpid()),
                bind=cfg["bind"],
                lock=lock,
                token=token,
            ),
        )
        published = True
        renewed = time.monotonic()
        while not stopping and proc.poll() is None:
            if time.monotonic() - renewed >= ttl / 3:
                renew_local(lock, token, ttl)
                renewed = time.monotonic()
            time.sleep(min(0.1, ttl / 6))
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        with contextlib.suppress(GpuLockError):
            release_local(lock, token)
        if published:
            # A concurrent stop owns cleanup under the lifecycle lock. Never
            # unlink state that a subsequent start has already published.
            with (STATE / "lifecycle.lock").open("a") as guard:
                with contextlib.suppress(BlockingIOError, FileNotFoundError):
                    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if json.loads(INFO.read_text())["token"] == token:
                        INFO.unlink()


def status():
    try:
        info = json.loads(INFO.read_text())
    except FileNotFoundError:
        return dict(running=False, ready=None)
    if alive(info, "pid"):
        return dict(running=True, pid=info["pid"], bind=info["bind"], ready=ready(info["bind"]))
    return dict(running=False, ready=None)


def start(cfg):
    if status()["running"]:
        raise RuntimeError("server is already running")
    binary = Path(
        cfg.get("binary")
        or Path("server/target") / os.environ.get("CARGO_BUILD_TARGET", "") / "release/server"
    ).absolute()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"Missing executable {binary}. Run dev build first.")
    cfg["binary"] = str(binary)
    try:
        sock = socket.create_connection(address(cfg["bind"]), timeout=1)
    except ConnectionRefusedError:
        pass
    else:
        sock.close()
        raise RuntimeError(f"Address already in use: {cfg['bind']}")
    INFO.unlink(missing_ok=True)
    with LOG.open("w") as log:
        supervisor = subprocess.Popen(
            [sys.executable, "-m", "tools.dev_cli._serve_host", "supervise"],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    try:
        with supervisor.stdin as writer:
            writer.write(json.dumps(cfg).encode())

        def started():
            if supervisor.poll() is not None:
                raise RuntimeError("Server startup failed. See dev serve logs.")
            return INFO.exists()

        wait_for(started, 125, "server readiness")
        snapshot = status()
        if not snapshot["running"] or snapshot["ready"] is None:
            raise RuntimeError("Server lost readiness during startup. See dev serve logs.")
        return snapshot
    except BaseException:
        if supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait(timeout=20)
        raise


def stop(force):
    if not INFO.exists():
        return {"stopped": False}
    info = json.loads(INFO.read_text())
    if alive(info, "pid"):
        with contextlib.suppress(ProcessLookupError):
            os.kill(info["pid"], signal.SIGTERM)
        try:
            wait_for(lambda: not alive(info, "pid"), 1 if force else 30, "server shutdown")
        except RuntimeError:
            if not force:
                raise
            if alive(info, "pid"):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(info["pid"], signal.SIGKILL)
            wait_for(lambda: not alive(info, "pid"), 10, "forced server shutdown")
    if "heartbeat" in info:
        wait_for(lambda: not alive(info, "heartbeat"), 45, "server supervisor exit")
    with contextlib.suppress(GpuLockError):
        release_local(info["lock"], info["token"])
    INFO.unlink(missing_ok=True)
    return {"stopped": True}


def dispatch(action, cfg):
    if action == "supervise":
        return supervise(cfg)
    if action == "status":
        return status()
    if action == "logs":
        if cfg["lines"] < 0:
            raise ValueError("lines must be non-negative")
        if not LOG.exists():
            return {"lines": []}
        with LOG.open(errors="replace") as reader:
            return {"lines": [line.rstrip("\n") for line in deque(reader, maxlen=cfg["lines"])]}
    STATE.mkdir(mode=0o700, exist_ok=True)
    with (STATE / "lifecycle.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        return start(cfg) if action == "start" else stop(cfg["force"])


if __name__ == "__main__":
    try:
        result = dispatch(sys.argv[1], json.load(sys.stdin))
        if result is not None:
            print(json.dumps(result))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))

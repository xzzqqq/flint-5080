"""Lease-bound process supervision on the execution host (Python 3.10+)."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from ._lease import GpuLockError, release_local, renew_local

T = TypeVar("T")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as tmp:
        try:
            json.dump(value, tmp)
            tmp.flush()
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def identity(pid: int) -> str | None:
    if not Path("/proc/self/stat").exists():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        except PermissionError:
            pass
        return "alive"
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(info: dict, key: str) -> bool:
    start = info[key + "_start"]
    return start is not None and identity(info[key]) == start


def wait_for(fn: Callable[[], T | None], timeout: float, label: str) -> T:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.05)
    raise TimeoutError("Timed out waiting for " + label)


@contextmanager
def lifecycle_lock(state_dir: Path) -> Iterator[None]:
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (state_dir / "lifecycle.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        yield


def _read_info(state_dir: Path) -> dict | None:
    try:
        return json.loads((state_dir / "daemon.json").read_text())
    except FileNotFoundError:
        return None


def _remove_owned_state(state_dir: Path, token: str, cleanup_paths: Sequence[Path]) -> None:
    """Caller holds the lifecycle lock. A later start may already own these paths."""
    info = _read_info(state_dir)
    if info is not None and info.get("token") == token:
        for path in cleanup_paths:
            path.unlink(missing_ok=True)
        (state_dir / "daemon.json").unlink()


def _signal_pid(info: dict, sig: signal.Signals) -> None:
    if alive(info, "pid"):
        with suppress(ProcessLookupError):
            os.kill(info["pid"], sig)


def _terminate_target(info: dict, timeout: float, *, force: bool) -> None:
    _signal_pid(info, signal.SIGTERM)
    try:
        wait_for(lambda: not alive(info, "pid"), timeout, "daemon exit")
    except TimeoutError:
        if not force:
            raise
        _signal_pid(info, signal.SIGKILL)
        wait_for(lambda: not alive(info, "pid"), 10, "forced daemon exit")


@dataclass
class ReadyProcess:
    """A successful readiness probe identifies the target behind any launcher."""

    pid: int
    metadata: dict


def _reap_child(proc: subprocess.Popen, info: dict | None) -> None:
    # A profiler launcher may need time to flush its report after the target exits.
    if info and info["pid"] != proc.pid:
        _terminate_target(info, 15, force=True)
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=15)
    if info and proc.poll() is not None:
        return

    # Include descendants of a launcher interrupted before its target reported a PID.
    def group_exited() -> bool:
        if proc.poll() is None:
            return False
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            # Darwin can report EPERM while the group's last member is exiting.
            if sys.platform == "darwin":
                return True
            raise
        return False

    with suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        wait_for(group_exited, 15, "child process group exit")
    except TimeoutError:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def supervise_leased_process(
    state_dir: Path,
    command: Sequence[str],
    check_ready: Callable[[subprocess.Popen], ReadyProcess | None],
    *,
    lock: str,
    token: str,
    ttl: int,
    name: str,
    startup_timeout: float,
    env: dict[str, str] | None = None,
    cleanup_paths: Sequence[Path] = (),
) -> None:
    stopping = False

    def shutdown(signum: int, frame: object) -> None:
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, shutdown) for sig in (signal.SIGTERM, signal.SIGINT)}
    proc = None
    info = None
    published = False
    try:
        renew_local(lock, token, ttl, pid=os.getpid())
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        deadline = time.monotonic() + startup_timeout
        while not stopping:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{name} failed to become ready")
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited during startup ({proc.returncode})")
            ready = check_ready(proc)
            if ready is not None and proc.poll() is None:
                info = dict(
                    **ready.metadata,
                    pid=ready.pid,
                    pid_start=identity(ready.pid),
                    heartbeat=os.getpid(),
                    heartbeat_start=identity(os.getpid()),
                    lock=lock,
                    token=token,
                )
                break
            renew_local(lock, token, ttl)
            time.sleep(min(0.2, ttl / 4))
        if stopping or info is None:
            return
        if not alive(info, "pid"):
            raise RuntimeError(f"{name} exited during startup")
        renew_local(lock, token, ttl, pid=info["pid"])
        atomic_json(state_dir / "daemon.json", info)
        published = True
        renewed = time.monotonic()
        while not stopping:
            proc.poll()  # Reap a direct child so identity() also works off Linux.
            if not alive(info, "pid"):
                break
            if time.monotonic() - renewed >= ttl / 3:
                renew_local(lock, token, ttl)
                renewed = time.monotonic()
            time.sleep(min(0.1, ttl / 6))
    finally:
        try:
            if proc is not None:
                _reap_child(proc, info)
        finally:
            with suppress(GpuLockError):
                release_local(lock, token)
            if published:
                with lifecycle_lock(state_dir):
                    _remove_owned_state(state_dir, token, cleanup_paths)
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def spawn_supervisor(
    state_dir: Path,
    module: str,
    cfg: dict,
    *,
    name: str,
    log_path: Path,
    startup_timeout: float,
    prepare: Callable[[], None],
    started: Callable[[dict], dict],
) -> dict:
    supervisor = None
    try:
        with lifecycle_lock(state_dir):
            info = _read_info(state_dir)
            if info and "remote" not in info and alive(info, "pid"):
                raise RuntimeError(f"{name} is already running")
            prepare()
            (state_dir / "daemon.json").unlink(missing_ok=True)
            with log_path.open("w") as log:
                supervisor = subprocess.Popen(
                    [sys.executable, "-m", module, "supervise"],
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=log,
                    text=True,
                    start_new_session=True,
                )
            with supervisor.stdin as writer:
                json.dump(cfg, writer)

            def ready() -> dict | None:
                if supervisor.poll() is not None:
                    raise RuntimeError(f"{name} startup failed. See {log_path}.")
                info = _read_info(state_dir)
                return info if info and info["token"] == cfg["token"] else None

            return started(wait_for(ready, startup_timeout, name + " readiness"))
    except BaseException:
        # The supervisor may itself need the lifecycle lock to finish cleanup.
        if supervisor is not None and supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait()
        raise


def stop_supervised_process(
    state_dir: Path,
    *,
    force: bool,
    graceful_stop: Callable[[], None] | None = None,
    cleanup_paths: Sequence[Path] = (),
) -> dict[str, bool]:
    with lifecycle_lock(state_dir):
        info = _read_info(state_dir)
        if info is None or "remote" in info:
            return {"stopped": False}
        if graceful_stop and alive(info, "pid"):
            try:
                graceful_stop()
                wait_for(lambda: not alive(info, "pid"), 20, "daemon shutdown")
            except (OSError, ValueError, RuntimeError):
                if not force:
                    raise
        if alive(info, "pid"):
            _terminate_target(info, 1 if force else 30, force=force)

    if "heartbeat" in info:
        wait_for(lambda: not alive(info, "heartbeat"), 45, "supervisor exit")
    # Recover a dead supervisor's state. Its lease or workspace may have been reused.
    with lifecycle_lock(state_dir):
        with suppress(GpuLockError):
            release_local(info["lock"], info["token"])
        _remove_owned_state(state_dir, info["token"], cleanup_paths)
    return {"stopped": True}

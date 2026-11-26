"""Single-host lease operations. Also sent verbatim to Python on the GPU host."""

from __future__ import annotations

import fcntl
import getpass
import json
import os
import socket
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class LeaseRecord:
    token: str
    owner: str
    workspace: str
    mode: str
    pid: int
    hostname: str
    acquired_at: float
    heartbeat_at: float
    expires_at: float


@dataclass
class LockStatus:
    locked: bool
    stale: bool
    stale_reason: str | None
    record: LeaseRecord | None
    lock_file: str

    def to_public_dict(self) -> dict:
        snapshot = asdict(self)
        if snapshot["record"] is not None:
            del snapshot["record"]["token"]
        return snapshot


class GpuLockError(RuntimeError):
    pass


class GpuLockBusyError(GpuLockError):
    def __init__(self, record: LeaseRecord):
        self.record = record
        super().__init__(f"GPU held by {record.owner} in {record.workspace} ({record.mode})")


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


@contextmanager
def _locked(lock_file):
    path = Path(lock_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + ".flock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        yield path


def _inspect(path: Path) -> LockStatus:
    try:
        record = LeaseRecord(**json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return LockStatus(False, False, None, None, str(path))
    except (ValueError, TypeError) as exc:
        raise GpuLockError(f"Invalid GPU lease in {path}: {exc}") from exc
    reason = None
    if time.time() >= record.expires_at:
        reason = "expired"
    elif (
        record.pid > 0 and record.hostname == socket.gethostname() and not is_pid_alive(record.pid)
    ):
        reason = "dead_pid"
    return LockStatus(reason is None, reason is not None, reason, record, str(path))


def _write(path: Path, record: LeaseRecord):
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as tmp:
        try:
            json.dump(asdict(record), tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def inspect_local(lock_file: str | Path) -> LockStatus:
    with _locked(lock_file) as path:
        return _inspect(path)


def acquire_local(
    lock_file: str | Path,
    mode: str,
    workspace: str,
    ttl_sec: int,
    owner: str | None = None,
    force: bool = False,
    pid: int = 0,
) -> LeaseRecord:
    if ttl_sec <= 0:
        raise GpuLockError("GPU lease TTL must be positive")
    with _locked(lock_file) as path:
        status = _inspect(path)
        if status.locked and not force:
            raise GpuLockBusyError(status.record)
        now = time.time()
        record = LeaseRecord(
            uuid.uuid4().hex,
            owner or getpass.getuser(),
            workspace,
            mode,
            pid,
            socket.gethostname(),
            now,
            now,
            now + ttl_sec,
        )
        _write(path, record)
        return record


def renew_local(
    lock_file: str | Path, token: str, ttl_sec: int, pid: int | None = None
) -> LeaseRecord:
    if ttl_sec <= 0:
        raise GpuLockError("GPU lease TTL must be positive")
    with _locked(lock_file) as path:
        status = _inspect(path)
        if not status.locked:
            raise GpuLockError("No active GPU lease to renew")
        record = status.record
        if token != record.token:
            raise GpuLockError("GPU lease token mismatch")
        if pid is not None:
            record.pid = pid
        record.heartbeat_at = time.time()
        record.expires_at = record.heartbeat_at + ttl_sec
        _write(path, record)
        return record


def release_local(
    lock_file: str | Path, token: str | None = None, force: bool = False, only_unbound: bool = False
) -> None:
    with _locked(lock_file) as path:
        record = _inspect(path).record
        if record is None:
            return
        if only_unbound and (record.pid or record.token != token):
            return
        if not force and token != record.token:
            raise GpuLockError("GPU lease token mismatch")
        path.unlink()


if __name__ == "__main__":
    import sys

    operation, arguments = json.loads(sys.argv[1])
    operations = {
        "inspect": inspect_local,
        "acquire": acquire_local,
        "renew": renew_local,
        "release": release_local,
    }
    try:
        result = operations[operation](**arguments)
        print(json.dumps({"result": asdict(result) if result else None}))
    except GpuLockBusyError as exc:
        print(json.dumps({"busy": asdict(exc.record)}))
    except (GpuLockError, OSError) as exc:
        print(json.dumps({"error": str(exc)}))

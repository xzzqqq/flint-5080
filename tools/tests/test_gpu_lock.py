import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tools.common import _lease
from tools.common.gpu_lock import GpuLeaseManager, GpuLockBusyError, GpuLockError
from tools.common.remote import Executor


@pytest.fixture
def manager(cfg):
    return GpuLeaseManager(cfg, Executor(cfg))


def test_lifecycle(manager, monkeypatch):
    clock = [100]
    monkeypatch.setattr(_lease.time, "time", lambda: clock[0])
    assert not manager.inspect().locked
    rec = manager.acquire("forge", 10)
    assert manager.inspect().record == rec
    assert rec.expires_at == 110 and rec.owner and rec.workspace
    clock[0] += 3
    new = manager.renew(rec.token, 20)
    assert new.acquired_at == 100 and new.heartbeat_at == 103 and new.expires_at == 123
    manager.release(rec.token)
    assert manager.inspect().record is None


def test_contention(manager):
    def attempt(_):
        try:
            return manager.acquire("bench").token, True
        except GpuLockBusyError as exc:
            return exc.record.token, False

    with ThreadPoolExecutor(max_workers=8) as pool:
        attempts = list(pool.map(attempt, range(24)))
    assert sum(won for _, won in attempts) == 1
    assert len({token for token, _ in attempts}) == 1


def test_expired(manager, monkeypatch):
    rec = manager.acquire("old", 5)
    monkeypatch.setattr(_lease.time, "time", lambda: rec.expires_at)
    status = manager.inspect()
    assert status.stale and not status.locked and status.stale_reason == "expired"
    with pytest.raises(GpuLockError):
        manager.renew(rec.token)
    assert manager.acquire("new").token != rec.token


def test_dead_pid(manager):
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    manager.acquire("old", pid=proc.pid)
    assert manager.inspect().stale_reason == "dead_pid"
    manager.acquire("new", pid=os.getpid())
    assert manager.inspect().locked


def test_pid_check_permission(monkeypatch):
    def denied(pid, signal):
        raise PermissionError()

    monkeypatch.setattr(os, "kill", denied)
    assert _lease.is_pid_alive(1)


def test_foreign_hostname(manager, monkeypatch):
    manager.acquire("other", pid=os.getpid())
    monkeypatch.setattr(_lease.socket, "gethostname", lambda: "different-host")
    monkeypatch.setattr(_lease, "is_pid_alive", lambda pid: pytest.fail("foreign PID probed"))
    assert manager.inspect().locked


def test_force_and_token_mismatch(manager):
    old = manager.acquire("first")
    new = manager.acquire("replacement", force=True)
    for action in (manager.release, manager.renew):
        with pytest.raises(GpuLockError, match="token"):
            action(old.token)
    assert manager.inspect().record == new
    with pytest.raises(GpuLockError):
        manager.release()
    manager.release(force=True)
    assert not manager.inspect().locked


def test_hold_releases_on_exception(manager):
    with pytest.raises(ValueError):
        with manager.hold("dev-check") as rec:
            assert manager.inspect().record == rec
            raise ValueError("smoke failed")
    assert not manager.inspect().locked


def test_heartbeat(manager, monkeypatch):
    import threading

    renewed = threading.Event()
    renew = manager.renew

    def observe(*args):
        rec = renew(*args)
        renewed.set()
        return rec

    monkeypatch.setattr(manager, "renew", observe)
    with manager.hold("check", ttl_sec=1) as rec:
        assert renewed.wait(2)
        assert manager.inspect().record.heartbeat_at > rec.heartbeat_at


def test_remote_before_sync(cfg, tmp_path, monkeypatch):
    # Replace only the SSH transport. Bash and the embedded Python helper run for real.
    ssh = tmp_path / "ssh"
    ssh.write_text('#!/bin/sh\nfor last do :; done\nexec /bin/bash -c "$last"\n')
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    cfg.remote.enabled = True
    cfg.remote.workspaces_root = str(tmp_path / "unsynced")
    manager = GpuLeaseManager(cfg, Executor(cfg))
    assert not manager.inspect().locked
    rec = manager.acquire("remote")
    with pytest.raises(GpuLockBusyError) as err:
        manager.acquire("collision")
    assert err.value.record == rec
    assert manager.renew(rec.token).token == rec.token
    with pytest.raises(GpuLockError, match="token"):
        manager.release("other")
    manager.release(rec.token)
    assert not manager.inspect().locked
    assert not Path(cfg.remote_workspace_dir).exists()

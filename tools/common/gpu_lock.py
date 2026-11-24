"""Shared GPU ownership, using the same atomic operations locally and over SSH."""

import getpass
import json
import os
import shlex
import threading
from contextlib import contextmanager
from pathlib import Path

from . import _lease
from ._lease import (
    GpuLockBusyError,
    GpuLockError,
    LeaseRecord,
    LockStatus,
    acquire_local,
    inspect_local,
    release_local,
    renew_local,
)
from .config import DevConfig
from .remote import Executor


class GpuLeaseManager:
    def __init__(self, cfg: DevConfig, executor: Executor):
        self.cfg = cfg
        self.executor = executor

    def _call(self, operation, **kwargs):
        kwargs["lock_file"] = self.cfg.shared.lock_file
        if not self.cfg.is_remote:
            fn = {
                "inspect": inspect_local,
                "acquire": acquire_local,
                "renew": renew_local,
                "release": release_local,
            }[operation]
            return fn(**kwargs)

        # cwd="." works even when rsync has never created the workspace.
        source = Path(_lease.__file__).read_text(encoding="utf-8")
        command = shlex.join(["python3", "-c", source, json.dumps([operation, kwargs])])
        reply = self.executor.run(command, cwd=".", timeout=30)
        if reply.returncode:
            raise GpuLockError(reply.stderr.strip() or "Remote lease helper failed")
        try:
            payload = json.loads(reply.stdout)
        except ValueError as exc:
            raise GpuLockError("Invalid JSON from remote lease helper") from exc
        if "busy" in payload:
            raise GpuLockBusyError(LeaseRecord(**payload["busy"]))
        if "error" in payload:
            raise GpuLockError(payload["error"])
        result = payload["result"]
        if operation == "inspect":
            if result["record"]:
                result["record"] = LeaseRecord(**result["record"])
            return LockStatus(**result)
        return LeaseRecord(**result) if result else None

    def inspect(self) -> LockStatus:
        return self._call("inspect")

    def acquire(
        self, mode: str, ttl_sec: int | None = None, force: bool = False, pid: int = 0
    ) -> LeaseRecord:
        ttl = self.cfg.gpu.lease_ttl_sec if ttl_sec is None else ttl_sec
        return self._call(
            "acquire",
            mode=mode,
            ttl_sec=ttl,
            force=force,
            pid=pid,
            owner=getpass.getuser(),
            workspace=self.cfg.effective_workspace_name,
        )

    def renew(self, token: str, ttl_sec: int | None = None) -> LeaseRecord:
        ttl = self.cfg.gpu.lease_ttl_sec if ttl_sec is None else ttl_sec
        return self._call("renew", token=token, ttl_sec=ttl)

    def release(self, token: str | None = None, force: bool = False) -> None:
        self._call("release", token=token, force=force)

    @contextmanager
    def hold(self, mode: str, ttl_sec: int | None = None, force: bool = False):
        ttl = self.cfg.gpu.lease_ttl_sec if ttl_sec is None else ttl_sec
        record = self.acquire(mode, ttl, force, pid=0 if self.cfg.is_remote else os.getpid())
        done = threading.Event()
        errors = []

        def heartbeat():
            while not done.wait(ttl / 3):
                try:
                    self.renew(record.token, ttl)
                except Exception as exc:
                    errors.append(exc)
                    return

        worker = threading.Thread(target=heartbeat, daemon=True)
        try:
            worker.start()
            yield record
        finally:
            done.set()
            if worker.ident is not None:
                worker.join()
            self.release(record.token)
        if errors:
            raise GpuLockError(f"GPU lease heartbeat failed: {errors[0]}")

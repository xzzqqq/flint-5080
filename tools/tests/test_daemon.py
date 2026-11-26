from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

from tools.common._lease import acquire_local, inspect_local
from tools.common.daemon import supervise_leased_process


@pytest.mark.parametrize("interrupt", [False, True], ids=["timeout", "sigint"])
def test_startup_renews_until_ready_and_tears_down(tmp_path, interrupt):
    lock = str(tmp_path / "lease")
    record = acquire_local(lock, "test", "workspace", 1)
    observations = []
    children = []

    def not_ready(proc: subprocess.Popen) -> None:
        lease = inspect_local(lock).record
        assert lease.pid == os.getpid()
        observations.append(lease.heartbeat_at)
        children.append(proc)
        if interrupt and len(observations) == 2:
            os.kill(os.getpid(), signal.SIGINT)

    def supervise():
        supervise_leased_process(
            tmp_path,
            [sys.executable, "-c", "import time; time.sleep(30)"],
            not_ready,
            lock=lock,
            token=record.token,
            ttl=1,
            name="test process",
            startup_timeout=0.3,
        )

    if interrupt:
        supervise()
    else:
        with pytest.raises(RuntimeError, match="failed to become ready"):
            supervise()
    assert observations[-1] > observations[0]
    assert children[-1].returncode == -signal.SIGTERM
    assert inspect_local(lock).record is None
    assert not (tmp_path / "daemon.json").exists()

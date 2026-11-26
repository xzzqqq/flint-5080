"""Process identity, state publication, and polling on the execution host."""

import json
import os
import tempfile
import time
from pathlib import Path


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as tmp:
        try:
            json.dump(value, tmp)
            tmp.flush()
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def identity(pid):
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


def alive(info, key):
    return identity(info[key]) == info[key + "_start"] and info[key + "_start"] is not None


def wait_for(fn, timeout, label):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.05)
    raise RuntimeError("Timed out waiting for " + label)

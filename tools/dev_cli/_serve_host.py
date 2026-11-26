"""Manage the server and its transferable GPU lease on the execution host."""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

from tools.common.daemon import (
    ReadyProcess,
    alive,
    spawn_supervisor,
    stop_supervised_process,
    supervise_leased_process,
)

STATE = Path(".serve")
INFO = STATE / "daemon.json"
LOG = STATE / "server.log"


def address(bind: str) -> tuple[str, int]:
    url = urlsplit("//" + bind)
    if not url.hostname or not url.port:
        raise ValueError("bind must specify a host and a non-zero port")
    host = "127.0.0.1" if url.hostname in {"0.0.0.0", "::"} else url.hostname
    return host, url.port


def ready(bind: str, timeout: float = 1) -> dict | None:
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


def supervise(cfg: dict) -> None:
    env = os.environ.copy()
    if cfg.get("api_key") is not None:
        env["FLINT_API_KEY"] = cfg["api_key"]

    def check_ready(proc: subprocess.Popen) -> ReadyProcess | None:
        if ready(cfg["bind"], timeout=min(1, cfg["ttl"] / 4)) is not None:
            return ReadyProcess(proc.pid, {"bind": cfg["bind"]})
        return None

    supervise_leased_process(
        STATE,
        [cfg["binary"], *cfg["argv"]],
        check_ready,
        lock=cfg["lock"],
        token=cfg["token"],
        ttl=cfg["ttl"],
        name="server",
        startup_timeout=120,
        env=env,
    )


def status() -> dict:
    try:
        info = json.loads(INFO.read_text())
    except FileNotFoundError:
        return dict(running=False, ready=None)
    return server_status(info)


def server_status(info: dict) -> dict:
    if alive(info, "pid"):
        return dict(running=True, pid=info["pid"], bind=info["bind"], ready=ready(info["bind"]))
    return dict(running=False, ready=None)


def start(cfg: dict) -> dict:
    def prepare() -> None:
        binary = Path(
            cfg.get("binary")
            or Path("server/target") / os.environ.get("CARGO_BUILD_TARGET", "") / "release/server"
        ).absolute()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(f"Missing executable {binary}. Run dev build first.")
        cfg["binary"] = str(binary)
        try:
            with socket.create_connection(address(cfg["bind"]), timeout=1):
                raise RuntimeError(f"Address already in use: {cfg['bind']}")
        except ConnectionRefusedError:
            pass

    def started(info: dict) -> dict:
        snapshot = server_status(info)
        if not snapshot["running"] or snapshot["ready"] is None:
            raise RuntimeError("Server lost readiness during startup. See dev serve logs.")
        return snapshot

    return spawn_supervisor(
        STATE,
        "tools.dev_cli._serve_host",
        cfg,
        name="server",
        log_path=LOG,
        startup_timeout=125,
        prepare=prepare,
        started=started,
    )


def dispatch(action: str, cfg: dict) -> dict | None:
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
    if action == "start":
        return start(cfg)
    return stop_supervised_process(STATE, force=cfg["force"])


if __name__ == "__main__":
    try:
        result = dispatch(sys.argv[1], json.load(sys.stdin))
        if result is not None:
            print(json.dumps(result))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))

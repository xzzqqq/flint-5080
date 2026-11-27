import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

import pytest

from tools.common.daemon import alive, atomic_json, lifecycle_lock, wait_for
from tools.common.gpu_lock import GpuLeaseManager, GpuLockBusyError
from tools.common.remote import Executor
from tools.dev_cli import __main__ as cli
from tools.dev_cli import _serve_host, commands

STUB = """
import argparse, json, os, signal, time
from http.server import BaseHTTPRequestHandler, HTTPServer
p = argparse.ArgumentParser()
p.add_argument("--bind")
p.add_argument("--model-dir")
p.add_argument("--model-name")
args, other = p.parse_known_args()
if args.model_name == "broken":
    raise SystemExit(3)
if args.model_name == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/ready" else 404)
        self.end_headers()
        self.wfile.write(json.dumps(dict(status="ready", model=args.model_name,
            key_set=bool(os.getenv("FLINT_API_KEY")), argv=other,
            model_dir=args.model_dir)).encode())
    def log_message(self, *args):
        pass

host, port = args.bind.split(":")
print("stub booting", flush=True)
server = HTTPServer((host, int(port)), Handler)
time.sleep(0.1)
print("stub ready", flush=True)
server.serve_forever()
"""


@pytest.fixture
def local(cfg, monkeypatch, capsys):
    cfg.remote.enabled = False
    cfg.gpu.lease_ttl_sec = 1
    cfg.model.weights_dir = "fixture"
    cfg.model.draft_weights_dir = "draft"
    cfg.shared.models_dir = str(cfg.repo_root / "weights")
    monkeypatch.setattr(cli, "find_repo_root", lambda _: cfg.repo_root)
    monkeypatch.setattr(cli, "load_config", lambda **_: cfg)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    monkeypatch.setenv("CARGO_BUILD_TARGET", "test-target")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        cfg.server.bind = f"0.0.0.0:{sock.getsockname()[1]}"
    binary = cfg.repo_root / "server/target/test-target/release/server"
    binary.parent.mkdir(parents=True)

    def run(*args, ok=True):
        code = cli.main(["serve", *args, "--json"])
        reply = json.loads(capsys.readouterr().out)
        assert code == (0 if ok else 1), reply
        return reply["data"] if ok else reply["error"]

    yield cfg, run, binary
    if (cfg.repo_root / ".serve/daemon.json").exists():
        run("stop", "--force")


def install(binary):
    binary.write_text(f"#!{sys.executable}\n" + STUB)
    binary.chmod(0o755)


def test_serve_lifecycle(local, monkeypatch):
    cfg, run, binary = local
    install(binary)
    monkeypatch.setattr(commands, "build", lambda *_: pytest.fail("unexpected build"))
    lease = GpuLeaseManager(cfg, Executor(cfg))
    assert not run("status")["running"]
    assert run("logs")["lines"] == []
    data = run(
        "start",
        "--api-key",
        "fixture-key",
        "--model-name",
        "mini",
        "--max-context",
        "512",
        "--prefill-chunk-size",
        "64",
        "--prefix-cache-slots",
        "1",
        "--request-timeout-sec",
        "0",
        "--no-cuda-graph",
        "--no-spec",
    )
    assert data["running"]
    ready = data["ready"]
    assert ready["key_set"] and ready["model"] == "mini"
    assert ready["model_dir"] == str(Path(cfg.shared.models_dir) / cfg.model.weights_dir)
    assert ready["argv"] == [
        "--draft-dir",
        str(Path(cfg.shared.models_dir) / "draft"),
        "--max-context",
        "512",
        "--prefill-chunk-size",
        "64",
        "--prefix-cache-slots",
        "1",
        "--request-timeout-sec",
        "0",
        "--no-cuda-graph",
        "--no-spec",
    ]
    time.sleep(1.1)  # Outlive the original lease and verify renewal by the execution host.
    status = run("status")
    assert status["running"] and status["ready"] == ready
    record = status["lease"]["record"]
    assert record["pid"] == data["pid"] and record["mode"] == "serve" and "token" not in record
    assert status["lease"]["locked"]
    with pytest.raises(GpuLockBusyError):
        lease.acquire("forge")
    assert run("logs", "--lines", "1")["lines"] == ["stub ready"]
    assert run("stop")["stopped"]
    assert not run("stop")["stopped"]
    status = run("status")
    assert not status["running"] and not status["lease"]["locked"]
    with lease.hold("forge"):
        assert "forge" in run("start", ok=False)


def test_force_and_build(local, monkeypatch):
    cfg, run, binary = local
    calls = []

    def build(build_cfg, args, result):
        calls.append((build_cfg, args.target))
        assert GpuLeaseManager(cfg, Executor(cfg)).inspect().record.mode == "serve"
        install(binary)

    monkeypatch.setattr(commands, "build", build)
    run("start", "--build", "--model-name", "stubborn")
    assert calls == [(cfg, "all")]
    assert run("stop", "--force")["stopped"]
    status = run("status")
    assert not status["running"] and not status["lease"]["locked"]


def test_startup_failure_rolls_back(local, monkeypatch):
    cfg, run, binary = local
    assert "Run dev build" in run("start", ok=False)
    assert not run("status")["lease"]["locked"]
    install(binary)
    assert "startup failed" in run(
        "start", "--binary", str(binary), "--model-name", "broken", ok=False
    )
    assert "exited during startup" in run("logs")["lines"][-1]
    assert not run("status")["lease"]["locked"]

    def fail_build(cfg, args, result):
        result.ok, result.error = False, "broken build"

    monkeypatch.setattr(commands, "build", fail_build)
    assert "broken build" in run("start", "--build", ok=False)
    assert not run("status")["lease"]["locked"]


def test_wildcard_connect_address():
    assert _serve_host.address("[::]:8080") == ("127.0.0.1", 8080)
    assert _serve_host.address("[::1]:8080") == ("::1", 8080)


def test_server_exit_releases_lease_and_cleans_state(local):
    cfg, run, binary = local
    install(binary)
    data = run("start")
    os.kill(data["pid"], signal.SIGTERM)
    info = cfg.repo_root / ".serve/daemon.json"
    deadline = time.monotonic() + 5
    while info.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not info.exists()
    assert not run("status")["lease"]["locked"]


def test_supervisor_signal_preserves_replacement_state(local):
    cfg, run, binary = local
    install(binary)
    run("start")
    state = cfg.repo_root / ".serve"
    path = state / "daemon.json"
    info = json.loads(path.read_text())
    manager = GpuLeaseManager(cfg, Executor(cfg))

    # Hold the lock until the old supervisor has released its lease, then publish a new owner.
    with lifecycle_lock(state):
        os.kill(info["heartbeat"], signal.SIGTERM)
        wait_for(lambda: manager.inspect().record is None, 5, "old lease release")
        replacement = manager.acquire("replacement")
        new_info = dict(info, token=replacement.token)
        atomic_json(path, new_info)
    try:
        wait_for(lambda: not alive(info, "heartbeat"), 5, "old supervisor exit")
        assert not alive(info, "pid")
        assert json.loads(path.read_text()) == new_info
        assert manager.inspect().record == replacement
    finally:
        path.unlink(missing_ok=True)
        manager.release(replacement.token)


def test_restart_stopped_and_running(local):
    cfg, run, binary = local
    install(binary)
    first = run("restart", "--model-name", "stubborn")
    fresh = run("restart", "--force", "--model-name", "fresh", "--no-spec")
    assert first["pid"] != fresh["pid"] and fresh["action"] == "restart"
    assert fresh["ready"]["model"] == "fresh" and "--no-spec" in fresh["ready"]["argv"]
    assert run("status")["lease"]["record"]["pid"] == fresh["pid"]
    run("stop")
    assert not Path(cfg.shared.lock_file).exists()

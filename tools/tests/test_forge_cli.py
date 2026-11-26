"""CPU coverage for the workbench, runnable with python -m unittest."""

import argparse
import io
import json
import os
import shlex
import sys
import tempfile
import unittest
from contextlib import chdir, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.common._lease import inspect_local, renew_local
from tools.common.config import DevConfig
from tools.common.daemon import alive, atomic_json
from tools.common.gpu_lock import GpuLeaseManager
from tools.common.output import RunResult
from tools.common.remote import Executor
from tools.forge_cli import __main__ as cli
from tools.forge_cli import commands, host

HOST_STUB = """
import json, os, signal, socket, sys
from pathlib import Path

if "--stubborn" in sys.argv:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

with socket.socket(socket.AF_UNIX) as server:
    server.bind(".forge/host.sock")
    server.listen()
    while True:
        conn, _ = server.accept()
        with conn, conn.makefile("rb") as reader:
            request = json.loads(reader.readline())
            cmd = request["command"]
            reply = {"ok": True, "pid": os.getpid(), "launcher": os.getppid()}
            if cmd == "run":
                reply["count"] = len(request["input_ids"])
                reply["sum"] = sum(request["input_ids"])
            if cmd == "shutdown":
                Path("shutdown").touch()
                if "--stubborn" in sys.argv:
                    continue
            conn.sendall(json.dumps(reply).encode() + b"\\n")
            if cmd == "shutdown":
                break
"""


class ForgeCLI(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cfg = DevConfig(Path(directory.name))
        self.cfg.remote.enabled = False
        self.cfg.shared.lock_file = str(self.cfg.repo_root / "lease.json")
        self.parser = cli.build_parser()

    def test_parser(self):
        for args in [
            ["start", "--no-model", "--kv-seq-len", "256", "--scratch-mb", "16"],
            ["stop", "--force"],
            ["status"],
            ["ls"],
            ["rm", "trial"],
            ["run", "--layers", "0..4", "--save", "trial", "--iters", "2"],
            ["ab", "baseline", "trial", "--full", "--no-graph", "--rounds", "3"],
            ["profile", "--e2e", "--nsys", "--no-spec"],
            ["dump", "--with-state", "--step", "5", "--out", "out.safetensors"],
        ]:
            with self.subTest(args=args):
                parsed = self.parser.parse_args([*args, "--local", "--json"])
                self.assertTrue(parsed.json and parsed.local)
                self.assertEqual(parsed.command, args[0])

    def test_layers_and_names(self):
        self.assertEqual(cli.layers("0..4"), (0, 4))
        self.assertEqual(cli.layers("47..48"), (47, 48))
        for value in ["-1..4", "0..49", "2..2", "3..2", "4", "1:3", "1..3x"]:
            with self.assertRaises(argparse.ArgumentTypeError):
                cli.layers(value)
        self.assertEqual(cli.savable("v1.0-fp_16"), "v1.0-fp_16")
        for value in [".", "..", "", "../a", "a/b", "x y", "bad;name", "current", "baseline"]:
            with self.assertRaises(argparse.ArgumentTypeError):
                cli.savable(value)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parser.parse_args(["run", "--e2e", "--layers", "0..1"])

    def test_execution_host_and_incremental_build(self):
        self.cfg.build.jobs = 4
        cases = self.cfg.repo_root / "workloads/golden_cases.json"
        cases.parent.mkdir()
        cases.write_text('{"cases":[{"name":"tiny","input_ids":[4,5]}]}')
        result = RunResult("forge run", True, {})
        for remote in [True, False]:
            self.cfg.remote.enabled = remote
            with self.subTest(remote=remote), patch.object(commands, "Executor") as ctor:
                ex = ctor.return_value
                ex.cfg = self.cfg
                ex.sync.return_value = SimpleNamespace(duration_ms=1)
                ex.run.return_value = SimpleNamespace(stdout="{}", duration_ms=2)

                def execute(args):
                    commands.execute(self.cfg, self.parser.parse_args(args), result)

                execute(["run", "--no-graph", "--layers", "0..4"])
                ex.sync.assert_called_once_with()
                build_call, request = ex.run.call_args_list
                build = build_call.args[0]
                self.assertIn("if ! test -f engine/build/build.ninja", build)
                self.assertIn("cmake --build engine/build --target forward --parallel 4", build)
                argv = shlex.split(request.args[0])
                self.assertEqual(argv[0], "python3" if remote else commands.sys.executable)
                self.assertEqual(argv[1:], ["-m", "tools.forge_cli.host", "run"])
                q = json.loads(request.kwargs["input_text"])
                self.assertEqual(q["input_ids"], [4, 5])
                self.assertFalse(q["use_cuda_graph"])
                self.assertEqual((q["begin_visit"], q["end_visit"]), (0, 4))
                for args in [
                    ["ab", "baseline", "trial"],
                    ["profile", "--no-build"],
                    ["dump", "--candidate", "x"],
                ]:
                    ex.reset_mock()
                    execute(args)
                    ex.sync.assert_not_called()
                    ex.run.assert_called_once()
                execute(["ab", "current", "baseline"])
                ex.sync.assert_called_once()

    def test_no_model_build_and_current_candidate(self):
        cases = self.cfg.repo_root / "workloads/golden_cases.json"
        cases.parent.mkdir()
        cases.write_text('{"cases":[{"name":"tiny","input_ids":[0]}]}')
        self.cfg.remote.enabled = True
        result = RunResult("forge", True, {})
        with (
            patch.object(commands, "build") as build,
            patch.object(commands, "GpuLeaseManager"),
            patch.object(commands, "host_call", return_value={"pid": 1, "heartbeat": 2}) as call,
        ):
            commands.execute(self.cfg, self.parser.parse_args(["start", "--no-model"]), result)
            self.assertEqual(build.call_args.args[-1], ["forge-host", "stub_forward"])
            argv = call.call_args.args[-1]["argv"]
            stub = "engine/build/stub_forward.so"
            self.assertEqual(argv[argv.index("--baseline") + 1], stub)
            commands.execute(self.cfg, self.parser.parse_args(["run", "--save", "trial"]), result)
            self.assertEqual(build.call_args.args[-1], ["stub_forward"])
            payload = call.call_args.args[-1]
            self.assertEqual(payload["so_path"], stub)
            self.assertNotIn("current_path", payload)

    def test_candidate_lifecycle(self):
        with chdir(self.cfg.repo_root):
            dst = Path(".forge/candidates")
            dst.mkdir(parents=True)
            current = Path(commands.CURRENT)
            current.parent.mkdir(parents=True)
            current.write_bytes(b"first")
            (dst / "baseline.so").write_bytes(b"baseline")

            def run(q):
                self.assertEqual(Path(q["so_path"]).read_bytes(), current.read_bytes())
                current.write_bytes(b"concurrent rebuild")
                return {}

            with patch.object(host, "ipc", run):
                request = {"save": "trial", "so_path": commands.CURRENT}
                self.assertEqual(host.dispatch("run", request.copy())["saved"], "trial")
                self.assertEqual((dst / "trial.so").read_bytes(), b"first")
                with self.assertRaises(FileExistsError):
                    host.dispatch("run", request.copy())
            self.assertEqual(host.dispatch("ls", {})["candidates"], ["baseline", "trial"])
            host.dispatch("rm", {"name": "trial"})
            self.assertEqual(host.candidates(), ["baseline"])

    def test_save_named_candidate_measures_snapshot(self):
        with chdir(self.cfg.repo_root):
            baseline = Path(".forge/candidates/baseline.so")
            baseline.parent.mkdir(parents=True)
            baseline.write_bytes(b"baseline")
            current = Path(commands.CURRENT)
            current.parent.mkdir(parents=True)
            current.write_bytes(b"current")

            def measure(q):
                baseline.write_bytes(b"replacement")
                self.assertEqual(Path(q["so_path"]).read_bytes(), b"baseline")
                return {}

            with patch.object(host, "ipc", measure):
                host.dispatch("run", {"save": "trial", "so_path": str(baseline)})
            self.assertEqual((baseline.parent / "trial.so").read_bytes(), b"baseline")

    def test_lease_handoff_and_release(self):
        lock = self.cfg.shared.lock_file
        args = self.parser.parse_args(["start", "--no-model", "--no-build"])
        result = RunResult("forge start", True, {})
        manager = GpuLeaseManager(self.cfg, MagicMock())

        def launch(ex, op, q):
            self.assertEqual(op, "start")
            self.assertIn("--no-model", q["argv"])
            record = inspect_local(lock).record
            self.assertEqual((record.mode, record.token), ("forge", q["token"]))
            renew_local(lock, record.token, 600, os.getpid())
            atomic_json(
                host.INFO, dict(token=record.token, lock=lock, pid=os.getpid(), pid_start="old")
            )
            return {"pid": os.getpid(), "heartbeat": 1}

        with chdir(self.cfg.repo_root), patch.object(commands, "host_call", launch):
            commands.execute(self.cfg, args, result)
            rec = manager.inspect().record
            self.assertEqual(rec.pid, os.getpid())
            manager.release_unbound(rec.token)
            self.assertTrue(manager.inspect().locked)
            self.assertTrue(host.stop(False)["stopped"])
            self.assertIsNone(manager.inspect().record)
            self.assertFalse(host.INFO.exists())
        with patch.object(commands, "host_call", side_effect=RuntimeError("launch failed")):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                commands.execute(self.cfg, args, result)
        self.assertFalse(manager.inspect().locked)

    def test_supervised_host_rpc_and_shutdown(self):
        # A local profiler stand-in keeps the launcher PID distinct from the IPC target.
        root = self.cfg.repo_root
        nsys = root / "nsys"
        nsys.write_text(
            f"#!{sys.executable}\nimport subprocess, sys\n"
            "sys.exit(subprocess.call(sys.argv[sys.argv.index('-o') + 2:]))\n"
        )
        nsys.chmod(0o755)
        stub = root / "host.py"
        stub.write_text(HOST_STUB)
        ex = Executor(self.cfg)
        lease = GpuLeaseManager(self.cfg, ex)
        env = dict(
            PATH=str(root) + os.pathsep + os.environ["PATH"],
            PYTHONPATH=str(Path(__file__).resolve().parents[2]),
        )
        with patch.dict(os.environ, env):
            for force in (False, True):
                with self.subTest(force=force):
                    args = [sys.executable, str(stub), "--no-model"]
                    if force:
                        args.append("--stubborn")
                    record = lease.acquire("forge")
                    cfg = dict(argv=args, lock=self.cfg.shared.lock_file, token=record.token, ttl=2)
                    info_path = root / host.INFO
                    started = commands.host_call(ex, "start", cfg)
                    try:
                        info = json.loads(info_path.read_text())
                        self.assertTrue(info["session"] and info["no_model"])
                        self.assertEqual(lease.inspect().record.pid, started["pid"])
                        self.assertNotEqual(started["pid"], started["launcher"])
                        ids = list(range(80_000))
                        reply = commands.host_call(ex, "run", {"command": "run", "input_ids": ids})
                        self.assertEqual((reply["count"], reply["sum"]), (len(ids), sum(ids)))
                        self.assertTrue(commands.host_call(ex, "stop", {"force": force})["stopped"])
                    finally:
                        if info_path.exists():
                            commands.host_call(ex, "stop", {"force": True})
                    self.assertFalse(alive(info, "pid") or alive(info, "heartbeat"))
                    self.assertFalse(info_path.exists() or (root / host.SOCKET).exists())
                    self.assertIsNone(lease.inspect().record)
                    (root / "shutdown").unlink()  # Both stop paths attempted IPC first.

    def test_json_archive_human_output_and_errors(self):
        self.cfg.remote.enabled = True
        calls = []

        def reply(ex, cmd, q):
            calls.append((ex.cfg.is_remote, cmd, q))
            return {"candidates": ["baseline"]}

        with (
            patch.object(cli, "find_repo_root", return_value=self.cfg.repo_root),
            patch.object(cli, "load_config", return_value=self.cfg),
            patch.object(commands, "host_call", reply),
        ):
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cli.main(["ls", "--local", "--json"]), 0)
            record = json.loads(out.getvalue())
            self.assertTrue(record["ok"])
            self.assertEqual(record["command"], "forge ls")
            self.assertEqual(
                json.loads(Path(record["run_file"]).read_text())["data"],
                {"candidates": ["baseline"]},
            )
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cli.main(["ls"]), 0)
            self.assertIn("forge ls OK", out.getvalue())
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cli.main(["rm", "baseline", "--json"]), 1)
            self.assertIn("--force", json.loads(out.getvalue())["error"])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["rm", "baseline", "--force", "--json"]), 0)
            self.assertEqual(
                calls, [(False, "ls", {}), (True, "ls", {}), (True, "rm", {"name": "baseline"})]
            )

    def test_missing_nsys(self):
        with patch.object(host.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "nsys is missing"):
                host.capture({})

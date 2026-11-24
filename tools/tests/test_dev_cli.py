import json
from pathlib import Path

import pytest

from tools.common.remote import CommandResult, SyncResult
from tools.dev_cli import __main__ as cli
from tools.dev_cli import _probe, commands


@pytest.fixture
def entry(cfg, monkeypatch):
    monkeypatch.setattr(cli, "find_repo_root", lambda _: cfg.repo_root)
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: cfg)
    return cli.main


class FakeExecutor:
    def __init__(self):
        self.runs = []
        self.syncs = 0

    def sync(self):
        self.syncs += 1
        return SyncResult(False, 2.5, "/workspace", 3, [])

    def run(self, cmd, **kwargs):
        self.runs.append((cmd, kwargs))
        stdout = '{"engine/build/libengine.a": 123}' if cmd.startswith("python3 -c") else ""
        return CommandResult(0, stdout, "", 4.5, [])


@pytest.fixture
def executor(monkeypatch):
    fake = FakeExecutor()
    monkeypatch.setattr(commands, "Executor", lambda cfg: fake)
    return fake


@pytest.mark.parametrize("argv", [["--json", "status"], ["status", "--json"],
                                  ["check", "--local", "--json"], ["serve", "logs", "--json"]])
def test_json_placement(argv):
    assert cli.build_parser().parse_args(argv).json


@pytest.mark.parametrize("argv", [["check"], ["check", "--gpu", "--local"],
                                  ["build", "--all", "--engine"]])
def test_mutually_exclusive(argv):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv)


def test_build_commands(cfg, entry, executor, capsys):
    cfg.remote.enabled = True
    cfg.build.jobs = 4
    assert entry(["build", "--clean", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert executor.syncs == 1 and result["data"]["sync"]["transferred_files"] == 3
    engine, server = [cmd for cmd, kwargs in executor.runs if kwargs.get("stream_stderr")]
    for part in ("-G Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=120a",
                 "command -v mold", "command -v ld.lld", "command -v ccache",
                 "-DCMAKE_CUDA_COMPILER_LAUNCHER=$launcher",
                 "rm -rf engine/build", "cmake --build engine/build --parallel 4"):
        assert part in engine
    assert "FLINT_ENGINE_BUILD_DIR=" in server and "cd server" in server
    assert "cargo clean" in server and "cargo build --release --features link-engine" in server
    targets = result["data"]["targets"]
    assert targets["engine"]["artifacts_bytes"] == {"engine/build/libengine.a": 123}
    assert targets["server"]["duration_ms"] == 4.5


@pytest.mark.parametrize("target", ["server", "engine"])
def test_build_no_sync(cfg, entry, executor, capsys, target):
    cfg.remote.enabled = True
    assert entry(["build", "--" + target, "--no-sync", "--json"]) == 0
    assert executor.syncs == 0
    assert set(json.loads(capsys.readouterr().out)["data"]["targets"]) == {target}


@pytest.mark.parametrize("target,manifest", [("engine", "engine/CMakeLists.txt"), ("server", "server/Cargo.toml")])
def test_missing_manifest(entry, capsys, target, manifest):
    assert entry(["build", "--" + target, "--json"]) == 1
    assert manifest in json.loads(capsys.readouterr().out)["error"]


def test_status(cfg, entry, capsys, monkeypatch):
    def query(args):
        if "--query-gpu=" in " ".join(args):
            return "0, GPU-1, RTX 5080, 40, 15, 100, 16384", None
        if args[0] == "nvidia-smi":
            return "GPU-1, 123, server, 50", None
        return None, "missing tool"
    monkeypatch.setattr(_probe, "query", query)
    assert entry(["--json", "status"]) == 0
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["mode"] == "local" and data["remote_workspace"] == cfg.remote_workspace_dir
    assert data["gpu"]["rows"][0]["name"] == "RTX 5080"
    assert data["compute_processes"]["rows"][0]["pid"] == "123"
    assert set(data["toolchain"]) == set("nvcc cmake ninja ccache cargo rustc mold ld.lld".split())
    assert data["toolchain"]["nvcc"]["error"] == "missing tool" and not data["lease"]["locked"]
    assert entry(["status"]) == 0
    output = capsys.readouterr().out
    assert "mode: local" in output and "RTX 5080" in output


def test_status_unreachable(cfg, entry, monkeypatch, capsys):
    cfg.remote.enabled = True
    monkeypatch.setattr(commands.Executor, "run", lambda *a, **kw: CommandResult(255, "", "offline", 1, []))
    assert entry(["status", "--json"]) == 1
    data = json.loads(capsys.readouterr().out)["data"]
    for key in ("gpu", "lease", "compute_processes", "toolchain"):
        assert data[key]["error"] == "offline"
    assert "git" in data


@pytest.mark.parametrize("uv,cargo", [(True, False), (False, True)])
def test_local_without_config(repo, executor, capsys, monkeypatch, uv, cargo):
    monkeypatch.setattr(cli, "find_repo_root", lambda _: repo)
    monkeypatch.setattr(commands.shutil, "which", lambda _: "/bin/uv" if uv else None)
    if cargo:
        (repo / "server").mkdir()
        (repo / "server/Cargo.toml").touch()
    assert cli.main(["check", "--local", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] and executor.syncs == 0
    assert executor.runs[0][0] == ("uv run pytest tools/tests" if uv else "pytest tools/tests")
    checks = result["data"]["checks"]
    assert checks["hygiene"]["status"] == "passed"
    assert checks["cargo"]["status"] == ("passed" if cargo else "skipped")
    if cargo:
        assert "cargo test" in executor.runs[1][0]


def test_local_bad_hygiene(repo, entry, executor, capsys):
    (repo / "bin/bench").chmod(0o644)
    (repo / ".dev.toml.example").write_text("[bad")
    assert entry(["check", "--local", "--json"]) == 1
    errors = json.loads(capsys.readouterr().out)["data"]["checks"]["hygiene"]["errors"]
    assert "bin/bench" in errors[0] and ".dev.toml.example" in errors[1]


def test_gpu_missing_builds_and_cleanup(cfg, entry, capsys):
    assert entry(["check", "--gpu", "--json"]) == 1
    checks = json.loads(capsys.readouterr().out)["data"]["checks"]
    assert set(checks) == {"ctest", "server", "forge-host"}
    assert all("dev build" in check["error"] for check in checks.values())
    assert not Path(cfg.shared.lock_file).exists()


@pytest.mark.parametrize("action", ["start", "stop", "status", "logs"])
def test_serve(entry, capsys, action):
    assert entry(["serve", action, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["action"] == action and "Phase 2" in data["message"]


def test_ssh_argv(entry, capsys):
    assert entry(["--json", "ssh", "printf", "%s", "arg with spaces"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["stdout"] == "arg with spaces"


@pytest.mark.parametrize("interrupted", [False, True])
def test_gpu_steps_run_under_lease(cfg, entry, executor, capsys, interrupted):
    runs = []

    def run(cmd, **kwargs):
        assert executor.syncs == 1
        record = json.loads(Path(cfg.shared.lock_file).read_text())
        assert record["mode"] == "dev-check"
        runs.append(cmd)
        if interrupted:
            raise OSError("interrupted transport")
        return CommandResult(0, "", "", 1, [])

    executor.run = run
    assert entry(["check", "--gpu", "--json"]) == int(interrupted)
    result = json.loads(capsys.readouterr().out)
    assert not Path(cfg.shared.lock_file).exists()
    if interrupted:
        assert result["error"] == "interrupted transport"
    else:
        assert len(runs) == 3
        for cmd, expected in zip(runs, ["ctest --test-dir", "forge-host --smoke", 'server" --smoke']):
            assert expected in cmd


def test_missing_required_config(repo, monkeypatch, capsys):
    monkeypatch.setattr(cli, "find_repo_root", lambda _: repo)
    assert cli.main(["status", "--json"]) == 1
    assert ".dev.toml.example" in json.loads(capsys.readouterr().out)["error"]


def test_release_artifact_sizes(cfg, monkeypatch):
    monkeypatch.setenv("CARGO_BUILD_TARGET", "linux-target")
    for profile in ("debug", "release"):
        folder = cfg.repo_root / "server/target/linux-target" / profile
        folder.mkdir(parents=True)
        (folder / "server").write_bytes(b"binary")
    sizes = commands._artifact_sizes(commands.Executor(cfg), "server")
    assert sizes == {"server/target/linux-target/release/server": 6}

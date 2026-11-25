import json
import shlex
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


@pytest.mark.parametrize(
    "argv",
    [
        ["--json", "status"],
        ["status", "--json"],
        ["check", "--local", "--json"],
        ["serve", "logs", "--json"],
    ],
)
def test_json_placement(argv):
    assert cli.build_parser().parse_args(argv).json


@pytest.mark.parametrize(
    "argv", [["check"], ["check", "--gpu", "--local"], ["build", "--all", "--engine"]]
)
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
    for part in (
        "-G Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=120a",
        "command -v mold",
        "command -v ld.lld",
        "command -v ccache",
        "-DCMAKE_CUDA_COMPILER_LAUNCHER=$launcher",
        "rm -rf engine/build",
        "cmake --build engine/build --parallel 4",
    ):
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


@pytest.mark.parametrize(
    "target,manifest", [("engine", "engine/CMakeLists.txt"), ("server", "server/Cargo.toml")]
)
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
    monkeypatch.setattr(
        commands.Executor, "run", lambda *a, **kw: CommandResult(255, "", "offline", 1, [])
    )
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
    assert checks["format"]["status"] == ("passed" if cargo else "skipped")
    assert checks["format"]["tools"]["clang-format"]["status"] == "skipped"
    assert checks["format"]["tools"]["ruff format"]["status"] == "skipped"
    assert checks["lint"]["status"] == ("passed" if cargo else "skipped")
    assert checks["lint"]["tools"]["ruff check"]["status"] == "skipped"
    if cargo:
        assert "cargo test" in executor.runs[1][0]
        assert any("cargo clippy" in cmd for cmd, _ in executor.runs)


@pytest.mark.parametrize(
    "outcome",
    ["passed", "clang-format", "missing", "cargo fmt", "ruff format", "ruff check", "cargo clippy"],
)
def test_local_format_and_lint_checks(repo, entry, executor, capsys, monkeypatch, outcome):
    names = [
        "engine/src/op with spaces.cu",
        "engine/include/api.h",
        "engine/src/ignore.md",
        "engine/build/obj.c",
        "engine/build-debug/nested/generated.cuh",
        "engine/third_party/lib/vendored.hpp",
        "server/Cargo.toml",
        "tools/sample.py",
    ]
    for name in names:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def fake_which(tool):
        if outcome == "missing":
            return None
        if tool in ("clang-format", "uv"):
            return f"/bin/{tool}"
        return None

    monkeypatch.setattr(commands.shutil, "which", fake_which)
    invoked = []

    def run(command, **kwargs):
        invoked.append(command)
        if (
            (outcome == "clang-format" and command.startswith("clang-format"))
            or (outcome == "cargo fmt" and "cargo fmt" in command)
            or (outcome == "ruff format" and "ruff format" in command)
            or (outcome == "ruff check" and "ruff check" in command)
            or (outcome == "cargo clippy" and "cargo clippy" in command)
        ):
            return CommandResult(1, "", "style violation", 1, [])
        return CommandResult(0, "", "", 1, [])

    executor.run = run
    assert entry(["check", "--local", "--json"]) == int(outcome != "passed")
    checks = json.loads(capsys.readouterr().out)["data"]["checks"]
    fmt = checks["format"]
    lint = checks["lint"]
    fmt_failed = outcome in ("clang-format", "missing", "cargo fmt", "ruff format")
    lint_failed = outcome in ("missing", "ruff check", "cargo clippy")
    assert fmt["status"] == ("failed" if fmt_failed else "passed")
    assert lint["status"] == ("failed" if lint_failed else "passed")
    assert "cd server && cargo fmt --check" in invoked
    assert "cd server && cargo clippy --all-targets -- -D warnings" in invoked
    clang = [shlex.split(command) for command in invoked if command.startswith("clang-format")]
    if outcome == "missing":
        assert not clang and "missing" in fmt["tools"]["clang-format"]["error"]
        assert "missing" in fmt["tools"]["ruff format"]["error"]
        assert "missing" in lint["tools"]["ruff check"]["error"]
    else:
        assert clang == [["clang-format", "--dry-run", "--Werror", *sorted(names[:2])]]
        assert "uv run ruff format --check tools" in invoked
        assert "uv run ruff check tools" in invoked
    if outcome in ("clang-format", "cargo fmt", "ruff format"):
        assert fmt["tools"][outcome]["error"] == "style violation"
    if outcome in ("ruff check", "cargo clippy"):
        assert lint["tools"][outcome]["error"] == "style violation"


def test_local_bad_hygiene(repo, entry, executor, capsys):
    (repo / "bin/bench").chmod(0o644)
    (repo / ".dev.toml.example").write_text("[bad")
    assert entry(["check", "--local", "--json"]) == 1
    errors = json.loads(capsys.readouterr().out)["data"]["checks"]["hygiene"]["errors"]
    assert "bin/bench" in errors[0] and ".dev.toml.example" in errors[1]


def test_gpu_missing_weights_config(cfg, entry, capsys, monkeypatch):
    cfg.model.weights_dir = ""
    monkeypatch.setattr(commands, "GpuLeaseManager", lambda *_: pytest.fail("lease requested"))
    assert entry(["check", "--gpu", "--json"]) == 1
    assert "Set model.weights_dir" in json.loads(capsys.readouterr().out)["error"]


def test_gpu_missing_builds_and_cleanup(cfg, entry, capsys):
    cfg.model.weights_dir = "tiny"
    assert entry(["check", "--gpu", "--json"]) == 1
    checks = json.loads(capsys.readouterr().out)["data"]["checks"]
    assert set(checks) == {"ctest", "ops-golden", "e2e-golden", "server", "forge-host"}
    assert all("dev build" in check["error"] for check in checks.values())
    assert not Path(cfg.shared.lock_file).exists()


@pytest.mark.parametrize(
    "missing",
    [
        "forward.so",
        "model",
        "golden",
        "ops_golden.safetensors",
        "e2e_short_math.safetensors",
        "e2e_short_sky.safetensors",
        "e2e_short_code.safetensors",
    ],
)
def test_gpu_missing_golden_prerequisite(cfg, entry, capsys, missing):
    cfg.shared.models_dir = str(cfg.repo_root / "models")
    cfg.model.weights_dir = "tiny"
    build = cfg.repo_root / "engine/build"
    build.mkdir(parents=True)
    for name in ("ops_tests", "e2e_tests"):
        (build / name).write_text("#!/bin/sh\nexit 0\n")
        (build / name).chmod(0o755)
    if missing != "forward.so":
        (build / "forward.so").touch()
    if missing != "model":
        (Path(cfg.shared.models_dir) / "tiny").mkdir(parents=True)
    golden = cfg.repo_root / ".forge/golden/tiny"
    if missing != "golden":
        golden.mkdir(parents=True)
        for name in ("ops_golden", "e2e_short_math", "e2e_short_sky", "e2e_short_code"):
            path = golden / (name + ".safetensors")
            if path.name != missing:
                path.touch()
    assert entry(["check", "--gpu", "--json"]) == 1
    checks = json.loads(capsys.readouterr().out)["data"]["checks"]
    if missing == "model":
        for name in ("ops-golden", "e2e-golden"):
            assert "Missing model directory" in checks[name]["error"]
    elif missing == "forward.so":
        assert "Missing engine/build/forward.so. Run dev build" in checks["e2e-golden"]["error"]
    else:
        name = "ops-golden" if missing.startswith("ops") else "e2e-golden"
        error = checks[name]["error"]
        assert (".forge/golden/tiny" if missing == "golden" else missing) in error
        assert "Run weights golden" in error


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
    cfg.model.weights_dir = "model's weights"
    cfg.shared.models_dir = "/shared models"
    cfg.weights.golden_dir = "golden fixtures"
    model = "/shared models/model's weights"
    golden = "golden fixtures/model's weights"
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
        assert set(result["data"]["checks"]) == {
            "ctest",
            "ops-golden",
            "e2e-golden",
            "forge-host",
            "server",
        }
        assert len(runs) == 5
        assert "ctest --test-dir" in runs[0]
        assert shlex.split(runs[1].splitlines()[-1]) == [
            "engine/build/ops_tests",
            "--model-dir",
            model,
            "--golden-file",
            golden + "/ops_golden.safetensors",
        ]
        assert shlex.split(runs[2].splitlines()[-1]) == [
            "engine/build/e2e_tests",
            "--forward",
            "engine/build/forward.so",
            "--model-dir",
            model,
            "--golden-dir",
            golden,
        ]
        assert "forge-host --smoke" in runs[3]
        assert '"$server" --smoke' in runs[4]


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

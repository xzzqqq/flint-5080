"""Implementation of the dev subcommands."""

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath

from tools.common.config import ConfigError, load_config
from tools.common.gpu_lock import GpuLeaseManager, GpuLockError
from tools.common.output import RunResult, log
from tools.common.remote import Executor

from . import _probe


def _git(root):
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()

    try:
        return {"head": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, RuntimeError) as exc:
        return {"error": str(exc)}


def status(cfg, args, result):
    executor = Executor(cfg)
    data = result.data
    data.update(
        mode="remote" if cfg.is_remote else "local",
        config_path=str(cfg.config_path) if cfg.config_path else None,
        remote_workspace=cfg.remote_workspace_dir,
        git=_git(cfg.repo_root),
    )
    if cfg.is_remote:
        source = Path(_probe.__file__).read_text(encoding="utf-8")
        command = shlex.join(["python3", "-c", source, cfg.remote_workspace_dir])
        try:
            probe = executor.run(command, cwd=".", timeout=70)
            if probe.returncode:
                raise RuntimeError(probe.stderr.strip() or "Remote inventory failed")
            data.update(json.loads(probe.stdout))
        except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            error = (
                str(exc)
                if not isinstance(exc, subprocess.TimeoutExpired)
                else "Host probe timed out"
            )
            for key in ("gpu", "compute_processes", "toolchain"):
                data[key] = {"error": error}
            result.ok, result.error = False, error
    else:
        data.update(_probe.inventory(cfg.repo_root))
    try:
        lease = asdict(GpuLeaseManager(cfg, executor).inspect())
        if lease["record"]:
            del lease["record"]["token"]
        data["lease"] = lease
    except (GpuLockError, OSError, subprocess.TimeoutExpired) as exc:
        error = (
            "Lease inspection timed out" if isinstance(exc, subprocess.TimeoutExpired) else str(exc)
        )
        data["lease"] = {"error": error}
        result.ok, result.error = False, result.error or error


def _engine_script(cfg, clean, targets=(), incremental=False):
    build = cfg.build
    flags = [
        "cmake",
        "-S",
        "engine",
        "-B",
        "engine/build",
        "-G",
        "Ninja",
        f"-DCMAKE_BUILD_TYPE={build.build_type}",
        f"-DCMAKE_CUDA_ARCHITECTURES={build.cuda_arch}",
    ]
    configure = shlex.join(flags)
    for lang in ("C", "CXX", "CUDA"):
        configure += f' "-DCMAKE_{lang}_COMPILER_LAUNCHER=$launcher"'
    for kind in ("EXE", "SHARED", "MODULE"):
        configure += f' "-DCMAKE_{kind}_LINKER_FLAGS=$link_flags"'
    compile_args = ["cmake", "--build", "engine/build"]
    if targets:
        compile_args.extend(["--target", *targets])
    compile_args.append("--parallel")
    if build.jobs:
        compile_args.append(str(build.jobs))
    lines = [
        f"use_ccache={1 if build.use_ccache else 0}",
        'launcher=""',
        'if [ "$use_ccache" = 1 ] && command -v ccache >/dev/null; then launcher=ccache; fi',
        f"linker={shlex.quote(build.linker)}",
        'if [ "$linker" = auto ]; then',
        "    if command -v mold >/dev/null; then linker=mold",
        "    elif command -v ld.lld >/dev/null; then linker=lld",
        "    else linker=system",
        "    fi",
        "fi",
        'link_flags=""',
        'if [ "$linker" != system ] && [ -n "$linker" ]; then link_flags="-fuse-ld=$linker"; fi',
    ]
    if clean:
        lines.append("rm -rf engine/build")
    if incremental:
        configure = f"if ! test -f engine/build/build.ninja; then {configure}; fi"
    lines.extend([configure, shlex.join(compile_args)])
    return "\n".join(lines)


def _server_script(cfg, clean):
    commands = ['export FLINT_ENGINE_BUILD_DIR="$PWD/engine/build"', "cd server"]
    if clean:
        commands.append("cargo clean --release --target-dir target")
    cargo = "cargo build --release --features link-engine --target-dir target"
    if cfg.build.jobs:
        cargo += f" --jobs {cfg.build.jobs}"
    return "\n".join([*commands, cargo])


def _artifact_sizes(executor, target):
    if target == "engine":
        paths = [
            "engine/build/" + name
            for name in ("libengine.a", "libengine.so", "forward.so", "forge-host")
        ]
        select = f"paths = {paths!r}"
    else:
        select = (
            'paths = [Path("server/target") / os.environ.get("CARGO_BUILD_TARGET", "") '
            '/ "release/server"]'
        )
    source = (
        "import json, os\nfrom pathlib import Path\n" + select + "\n"
        "print(json.dumps({str(p): p.stat().st_size for p in map(Path, paths) if p.is_file()}))"
    )
    out = executor.run(shlex.join(["python3", "-c", source]), check=True)
    return json.loads(out.stdout)


def build(cfg, args, result):
    executor = Executor(cfg)
    if args.no_sync or not cfg.is_remote:
        result.data["sync"] = {"skipped": True}
    else:
        sync = executor.sync()
        result.data["sync"] = {
            "skipped": sync.skipped,
            "duration_ms": sync.duration_ms,
            "transferred_files": sync.transferred_files,
        }
    targets = result.data["targets"] = {}
    for target, manifest, script in (
        ("engine", "engine/CMakeLists.txt", _engine_script),
        ("server", "server/Cargo.toml", _server_script),
    ):
        if args.target not in ("all", target):
            continue
        check = (
            f"test -f {manifest} || "
            f'{{ echo "Missing {manifest}. Add the {target} build manifest first." >&2; exit 1; }}'
        )
        out = executor.run("set -e\n" + check + "\n" + script(cfg, args.clean), stream_stderr=True)
        targets[target] = {"duration_ms": out.duration_ms, "returncode": out.returncode}
        if out.returncode:
            result.ok = False
            result.error = (
                out.stderr.strip()[-2000:] or f"{target} build failed (exit {out.returncode})"
            )
            return
        targets[target]["artifacts_bytes"] = _artifact_sizes(executor, target)


def _check_step(executor, checks, name, command):
    log(f"Running {name}")
    out = executor.run(command, stream_stderr=True)
    checks[name] = {
        "status": "failed" if out.returncode else "passed",
        "returncode": out.returncode,
        "duration_ms": out.duration_ms,
    }
    if out.returncode:
        checks[name]["error"] = out.stderr.strip()[-2000:] or f"exit {out.returncode}"


def _ruff_runner():
    if shutil.which("uv"):
        return "uv run ruff"
    if shutil.which("ruff"):
        return "ruff"
    return None


def _has_python_tools(root):
    tools_dir = root / "tools"
    return tools_dir.is_dir() and any(tools_dir.rglob("*.py"))


def _aggregate_tool_checks(tools, empty_reason):
    errors = [
        f"{name}: {step['error']}" for name, step in tools.items() if step["status"] == "failed"
    ]
    if errors:
        return {"status": "failed", "errors": errors, "tools": tools}
    if all(step["status"] == "skipped" for step in tools.values()):
        return {"status": "skipped", "reason": empty_reason, "tools": tools}
    return {"status": "passed", "tools": tools}


def _check_format(root, executor):
    suffixes = {".h", ".hpp", ".c", ".cc", ".cpp", ".cu", ".cuh"}
    engine = root / "engine"
    sources = []
    for directory, children, filenames in os.walk(engine):
        directory = Path(directory)
        if directory == engine:
            children[:] = [
                name for name in children if name != "third_party" and not name.startswith("build")
            ]
        sources.extend(
            str((directory / name).relative_to(root))
            for name in filenames
            if Path(name).suffix in suffixes
        )

    tools = {}
    if not sources:
        tools["clang-format"] = {"status": "skipped", "reason": "no engine sources"}
    elif not shutil.which("clang-format"):
        tools["clang-format"] = {"status": "failed", "error": "clang-format is missing"}
    else:
        command = shlex.join(["clang-format", "--dry-run", "--Werror", *sorted(sources)])
        _check_step(executor, tools, "clang-format", command)

    if (root / "server/Cargo.toml").is_file():
        _check_step(executor, tools, "cargo fmt", "cd server && cargo fmt --check")
    else:
        tools["cargo fmt"] = {"status": "skipped", "reason": "server/Cargo.toml is missing"}

    if not _has_python_tools(root):
        tools["ruff format"] = {"status": "skipped", "reason": "no python tool sources"}
    elif not (ruff := _ruff_runner()):
        tools["ruff format"] = {"status": "failed", "error": "ruff is missing"}
    else:
        _check_step(executor, tools, "ruff format", f"{ruff} format --check tools")

    return _aggregate_tool_checks(tools, "no engine, server, or python tool sources")


def _check_lint(root, executor):
    tools = {}
    if (root / "server/Cargo.toml").is_file():
        _check_step(
            executor,
            tools,
            "cargo clippy",
            "cd server && cargo clippy --all-targets -- -D warnings",
        )
    else:
        tools["cargo clippy"] = {"status": "skipped", "reason": "server/Cargo.toml is missing"}

    if not _has_python_tools(root):
        tools["ruff check"] = {"status": "skipped", "reason": "no python tool sources"}
    elif not (ruff := _ruff_runner()):
        tools["ruff check"] = {"status": "failed", "error": "ruff is missing"}
    else:
        _check_step(executor, tools, "ruff check", f"{ruff} check tools")

    return _aggregate_tool_checks(tools, "no server or python tool sources")


def _check_local(cfg, checks):
    local = replace(cfg, remote=replace(cfg.remote, enabled=False))
    executor = Executor(local)
    runner = "uv run pytest" if shutil.which("uv") else "pytest"
    _check_step(executor, checks, "pytest", f"{runner} tools/tests")
    if (cfg.repo_root / "server/Cargo.toml").is_file():
        _check_step(executor, checks, "cargo", "cd server && cargo test")
    else:
        checks["cargo"] = {"status": "skipped", "reason": "server/Cargo.toml is missing"}
    checks["format"] = _check_format(cfg.repo_root, executor)
    checks["lint"] = _check_lint(cfg.repo_root, executor)
    errors = []
    for name in ("dev", "forge", "bench", "weights"):
        path = cfg.repo_root / "bin" / name
        if not path.is_file() or not os.access(path, os.X_OK):
            errors.append(f"{path} must be executable")
    try:
        load_config(cfg.repo_root / ".dev.toml.example", repo_root=cfg.repo_root)
    except ConfigError as exc:
        errors.append(str(exc))
    checks["hygiene"] = {"status": "failed" if errors else "passed", "errors": errors}


def _gpu_check_commands(model_dir, golden_dir):
    def require(flag, path, message):
        test = shlex.join(["test", flag, str(path)])
        return f"{test} || {{ echo {shlex.quote(message)} >&2; exit 1; }}"

    def build(flag, path):
        return require(flag, path, f"Missing {path}. Run dev build first.")

    def golden(flag, path):
        return require(flag, path, f"Missing {path}. Run weights golden first.")

    model = require(
        "-d", model_dir, f"Missing model directory {model_dir}. Check model.weights_dir."
    )
    ops_file = golden_dir / "ops_golden.safetensors"
    ops = "engine/build/ops_tests"
    e2e = "engine/build/e2e_tests"
    forward = "engine/build/forward.so"
    return {
        "ctest": [
            build("-f", "engine/build/CTestTestfile.cmake"),
            "ctest --test-dir engine/build --output-on-failure --no-tests=error",
        ],
        "ops-golden": [
            build("-x", ops),
            model,
            golden("-f", ops_file),
            shlex.join([ops, "--model-dir", str(model_dir), "--golden-file", str(ops_file)]),
        ],
        "e2e-golden": [
            build("-x", e2e),
            build("-f", forward),
            model,
            golden("-d", golden_dir),
            *(
                golden("-f", golden_dir / f"e2e_short_{case}.safetensors")
                for case in ("math", "sky", "code")
            ),
            shlex.join(
                [
                    e2e,
                    "--forward",
                    forward,
                    "--model-dir",
                    str(model_dir),
                    "--golden-dir",
                    str(golden_dir),
                ]
            ),
        ],
        "forge-host": [build("-x", "engine/build/forge-host"), "engine/build/forge-host --smoke"],
        "server": [
            'server="server/target/${CARGO_BUILD_TARGET:+$CARGO_BUILD_TARGET/}release/server"',
            'test -x "$server" || { echo "Missing $server. Run dev build first." >&2; exit 1; }',
            '"$server" --smoke',
        ],
    }


def check(cfg, args, result):
    result.data["mode"] = "local" if args.local else "gpu"
    checks = result.data["checks"] = {}
    if args.local:
        _check_local(cfg, checks)
    else:
        if not cfg.model.weights_dir:
            raise ValueError("Set model.weights_dir in .dev.toml for GPU acceptance checks")
        model_dir = PurePosixPath(cfg.shared.models_dir) / cfg.model.weights_dir
        golden_dir = PurePosixPath(cfg.weights.golden_dir or ".forge/golden") / model_dir.name
        executor = Executor(cfg)
        executor.sync()
        with GpuLeaseManager(cfg, executor).hold("dev-check"):
            for name, lines in _gpu_check_commands(model_dir, golden_dir).items():
                _check_step(executor, checks, name, "\n".join(lines))
    failed = [name for name, detail in checks.items() if detail["status"] == "failed"]
    if failed:
        result.ok, result.error = False, "Failed checks: " + ", ".join(failed)


def serve(cfg, args, result):
    result.data.update(
        action=args.action, status="unavailable", message="The server runtime arrives in Phase 2."
    )


def _status_lines(data):
    lines = [
        f"mode: {data['mode']}",
        f"config: {data['config_path']}",
        f"remote workspace: {data['remote_workspace']}",
    ]
    git = data["git"]
    if "error" in git:
        lines.append(f"git: {git['error']}")
    else:
        state = "dirty" if git["dirty"] else "clean"
        lines.append(f"git: {git['head']} ({state})")

    gpu = data["gpu"]
    if not gpu.get("rows"):
        lines.append("GPU: " + gpu.get("error", "none reported"))
    for row in gpu.get("rows", []):
        lines.append(
            f"GPU {row['index']}: {row['name']}, {row['memory.used']}/{row['memory.total']} MiB, "
            f"{row['utilization.gpu']}% utilization, {row['temperature.gpu']} °C"
        )
    processes = data["compute_processes"]
    if not processes.get("rows"):
        lines.append("compute processes: " + processes.get("error", "none"))
    for row in processes.get("rows", []):
        lines.append(f"PID {row['pid']}: {row['process_name']} ({row['used_memory']} MiB)")
    lease = data["lease"]
    if lease.get("locked"):
        record = lease["record"]
        lines.append(f"GPU lease: {record['owner']} in {record['workspace']} ({record['mode']})")
    else:
        state = lease.get("error") or lease.get("stale_reason") or "free"
        lines.append(f"GPU lease: {state}")
    if "error" in data["toolchain"]:
        lines.append("toolchain: " + data["toolchain"]["error"])
    else:
        for name, info in data["toolchain"].items():
            lines.append(f"{name}: {info.get('version') or info.get('error')}")
    return lines


def human_format(result: RunResult) -> str:
    lines = [result.command + (" OK" if result.ok else " FAILED")]
    if result.error:
        lines.append(result.error)
    data = result.data
    if result.command == "dev status" and "lease" in data:
        lines.extend(_status_lines(data))
    elif "checks" in data:
        for name, step in data["checks"].items():
            detail = step.get("reason") or step.get("error") or ", ".join(step.get("errors", []))
            lines.append(f"{name}: {step['status']}" + (f" ({detail})" if detail else ""))
    elif "targets" in data:
        for target, detail in data["targets"].items():
            lines.append(f"{target}: {detail['duration_ms']:.0f} ms (exit {detail['returncode']})")
            for path, size in detail.get("artifacts_bytes", {}).items():
                lines.append(f"  {path}: {size:,} bytes")
    elif "message" in data:
        lines.append(data["message"])
    return "\n".join(lines)

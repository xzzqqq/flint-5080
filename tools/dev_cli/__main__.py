"""Entrypoint for bin/dev."""

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from tools.common.config import find_repo_root, load_config
from tools.common.output import RunResult, archive_run, emit_result
from tools.common.remote import Executor

from . import commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dev", description="Manage the Flint-5080 development workspace."
    )
    json_help = "Emit machine-readable JSON on stdout."
    parser.add_argument("--json", action="store_true", help=json_help)
    subs = parser.add_subparsers(dest="command", required=True)

    def sub(name, description):
        child = subs.add_parser(name, help=description, description=description)
        child.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help=json_help)
        return child

    sub("status", "Inspect the workspace, GPU lease, utilization, and toolchain.")
    build = sub("build", "Sync and build the engine and server.")
    targets = build.add_mutually_exclusive_group()
    for name in ("engine", "server", "all"):
        targets.add_argument("--" + name, dest="target", action="store_const", const=name)
    build.set_defaults(target="all")
    build.add_argument("--clean", action="store_true", help="Rebuild targets from a clean state.")
    build.add_argument(
        "--no-sync", action="store_true", help="Use sources already on the build host."
    )
    check = sub(
        "check",
        "Run local tests or the GPU acceptance checks with model weights and golden fixtures.",
    )
    check.add_argument(
        "--only", action="append", help="Run selected checks (repeat or comma-separate)."
    )
    check.add_argument("--draft-dir")
    fmt = sub("fmt", "Format engine, Rust, and Python sources locally.")
    fmt.add_argument("--check", action="store_true", help="Report needed changes without writing.")
    fmt.add_argument("--fix", action="store_true", help="Also apply Ruff lint fixes.")
    modes = check.add_mutually_exclusive_group(required=True)
    modes.add_argument("--local", action="store_true")
    modes.add_argument("--gpu", action="store_true")
    ssh = sub("ssh", "Open a workspace shell or execute a command there.")
    ssh.add_argument("cmd", nargs=argparse.REMAINDER, help="Command and its arguments")
    serve = sub("serve", "Manage the inference server on the execution host.")
    actions = serve.add_subparsers(dest="action", required=True)
    start = actions.add_parser("start", help="Launch the server and wait for readiness.")
    restart = actions.add_parser("restart", help="Stop and start a fresh server.")
    stop = actions.add_parser("stop", help="Stop the server and release its GPU lease.")
    status = actions.add_parser("status", help="Show the GPU lease and live readiness.")
    logs = actions.add_parser("logs", help="Read recent server log lines.")
    for action in (start, restart, stop, status, logs):
        action.add_argument(
            "--json", action="store_true", default=argparse.SUPPRESS, help=json_help
        )
    for action in (start, restart):
        for flag in ("bind", "model-dir", "draft-dir", "model-name", "api-key", "binary"):
            action.add_argument("--" + flag)
        for flag in (
            "max-context",
            "prefill-chunk-size",
            "prefix-cache-slots",
            "request-timeout-sec",
        ):
            action.add_argument("--" + flag, type=int)
        action.add_argument("--no-cuda-graph", action="store_true")
        action.add_argument("--no-spec", action="store_true")
        action.add_argument(
            "--build", action="store_true", help="Rebuild the engine and server first."
        )
    restart.add_argument("--force", action="store_true")
    stop.add_argument("--force", action="store_true")
    logs.add_argument("--lines", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start = time.monotonic()
    result = RunResult(f"dev {args.command}", True, {})
    root = find_repo_root(Path(__file__))
    try:
        optional = args.command in {"serve", "fmt"} or (args.command == "check" and args.local)
        cfg = load_config(repo_root=root, allow_missing=optional)
        if args.command == "ssh":
            executor = Executor(cfg)
            cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
            if args.json and cmd:
                out = executor.run(shlex.join(cmd))
                code = out.returncode
                result.data.update(stdout=out.stdout, stderr=out.stderr)
            else:
                code = executor.interactive_ssh(cmd, stdout=sys.stderr if args.json else None)
            result.data["returncode"] = code
            result.ok = code == 0
            if not args.json:
                result.duration_ms = (time.monotonic() - start) * 1000
                archive_run(root, result)
                return code
        else:
            handler = {
                "build": commands.build,
                "check": commands.check,
                "fmt": commands.fmt,
                "status": commands.status,
                "serve": commands.serve,
            }[args.command]
            handler(cfg, args, result)
    except subprocess.TimeoutExpired:
        result.ok, result.error = False, "Command timed out"
    except (OSError, RuntimeError, ValueError) as exc:
        result.ok, result.error = False, str(exc)
    result.duration_ms = (time.monotonic() - start) * 1000
    return emit_result(
        result, json_mode=args.json, human_formatter=commands.human_format, repo_root=root
    )


if __name__ == "__main__":
    raise SystemExit(main())

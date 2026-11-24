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
    check = sub("check", "Run local tests or the GPU smoke checks.")
    modes = check.add_mutually_exclusive_group(required=True)
    modes.add_argument("--local", action="store_true")
    modes.add_argument("--gpu", action="store_true")
    ssh = sub("ssh", "Open a workspace shell or execute a command there.")
    ssh.add_argument("cmd", nargs=argparse.REMAINDER, help="Command and its arguments")
    serve = sub("serve", "Report availability of the Phase 2 server runtime.")
    serve.add_argument("action", choices=("start", "stop", "status", "logs"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start = time.monotonic()
    result = RunResult(f"dev {args.command}", True, {})
    root = find_repo_root(Path(__file__))
    try:
        optional = args.command == "serve" or (args.command == "check" and args.local)
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

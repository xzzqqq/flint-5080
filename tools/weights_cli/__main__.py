"""Entrypoint for bin/weights."""

import argparse
import time
from pathlib import Path

from tools.common.config import find_repo_root, load_config
from tools.common.output import RunResult, emit_result

from . import commands


def build_parser():
    parser = argparse.ArgumentParser(prog="weights", description="Inspect Flint-5080 checkpoints.")
    json_help = "Emit machine-readable JSON on stdout."
    parser.add_argument("--json", action="store_true", help=json_help)
    subs = parser.add_subparsers(dest="command", required=True)
    child = subs.add_parser("inspect", help="Inventory checkpoint files and tensor headers.")
    child.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help=json_help)
    child.add_argument("--path", help="Inspect this directory instead of model.weights_dir.")
    child.add_argument("--local", action="store_true", help="Inspect the local filesystem.")
    child.add_argument("--tensors", action="store_true", help="Include individual tensor records.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    start = time.monotonic()
    result = RunResult(f"weights {args.command}", True, {})
    root = find_repo_root(Path(__file__))
    try:
        cfg = load_config(repo_root=root, allow_missing=args.local)
        commands.inspect(cfg, args, result)
    except (OSError, RuntimeError, ValueError) as exc:
        result.ok, result.error = False, str(exc)
    result.duration_ms = (time.monotonic() - start) * 1000
    return emit_result(
        result, json_mode=args.json, human_formatter=commands.human_format, repo_root=root
    )


if __name__ == "__main__":
    raise SystemExit(main())

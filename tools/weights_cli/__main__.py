"""Entrypoint for bin/weights."""

import argparse
import time
from pathlib import Path

from tools.common.config import find_repo_root, load_config
from tools.common.output import RunResult, emit_result

from . import commands


def build_parser():
    parser = argparse.ArgumentParser(
        prog="weights",
        description="Inspect checkpoints and generate Flint-5080 reference fixtures.",
    )
    json_help = "Emit machine-readable JSON on stdout."
    parser.add_argument("--json", action="store_true", help=json_help)
    subs = parser.add_subparsers(dest="command", required=True)
    child = subs.add_parser("inspect", help="Inventory checkpoint files and tensor headers.")
    child.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help=json_help)
    child.add_argument("--path", help="Inspect this directory instead of model.weights_dir.")
    child.add_argument("--local", action="store_true", help="Inspect the local filesystem.")
    child.add_argument("--tensors", action="store_true", help="Include individual tensor records.")
    golden = subs.add_parser("golden", help="Generate reference fixtures from a checkpoint.")
    golden.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help=json_help)
    golden.add_argument("--local", action="store_true", help="Run on this host.")
    golden.add_argument("--python", help="Python interpreter with the reference dependencies.")
    golden.add_argument("--path", help="Checkpoint directory on the execution host.")
    golden.add_argument("--out", help="Destination fixture directory.")
    golden.add_argument("--cases", default="workloads/golden_cases.json")
    golden.add_argument("--case", action="append", default=[], help="Select a named case.")
    golden.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    golden.add_argument("--steps", type=int, help="Override the number of cached decode steps.")
    golden.add_argument("--skip-ops", action="store_true")
    golden.add_argument("--skip-e2e", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    start = time.monotonic()
    result = RunResult(f"weights {args.command}", True, {})
    root = find_repo_root(Path(__file__))
    try:
        cfg = load_config(repo_root=root, allow_missing=args.local)
        getattr(commands, args.command)(cfg, args, result)
    except (OSError, RuntimeError, ValueError) as exc:
        result.ok, result.error = False, str(exc)
    result.duration_ms = (time.monotonic() - start) * 1000
    return emit_result(
        result, json_mode=args.json, human_formatter=commands.human_format, repo_root=root
    )


if __name__ == "__main__":
    raise SystemExit(main())

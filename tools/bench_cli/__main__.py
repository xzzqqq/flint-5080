"""Black-box OpenAI HTTP benchmark entry point."""

import argparse
import math
import os
import sys
from pathlib import Path
from time import perf_counter

import httpx

from tools.common.config import find_repo_root
from tools.common.output import RunResult, emit_result

from . import commands
from .workloads import PERF


def build_parser():
    parser = argparse.ArgumentParser(prog="bench", description="Black-box OpenAI HTTP benchmark")
    parser.add_argument("--json", action="store_true", help="Emit a JSON run record.")
    subs = parser.add_subparsers(dest="command", required=True)
    run = subs.add_parser(
        "run",
        help="Measure performance and run deterministic canaries.",
    )
    compare = subs.add_parser("compare", help="Compare paired performance case medians.")
    for sub in (run, compare):
        sub.add_argument(
            "--json", action="store_true", default=argparse.SUPPRESS, help="Emit a JSON run record."
        )
    run.add_argument("--url", help="Base URL (defaults to http://<server.bind> or 127.0.0.1:8080).")
    run.add_argument("--api-key", help="Bearer token (defaults to FLINT_API_KEY).")
    run.add_argument("--case", action="append", help="Run only this case ID (repeatable).")
    run.add_argument("--tag", help="Name this archived run for bench compare.")
    run.add_argument("--model", help="Model ID (defaults to first /v1/models entry).")
    run.add_argument(
        "--suite",
        choices=(*PERF, "canary", "all"),
        default="all",
        help="Suite to run (default all).",
    )
    run.add_argument(
        "--rounds", type=int, default=3, help="Measured performance rounds (default 3)."
    )
    run.add_argument(
        "--warmup", type=int, default=1, help="Unmeasured rounds per performance case (default 1)."
    )
    run.add_argument(
        "--telemetry",
        choices=("auto", "off"),
        default="auto",
        help="Request Flint telemetry (default auto).",
    )
    run.add_argument(
        "--timeout",
        type=float,
        default=1800,
        help="Per-request deadline in seconds (default 1800).",
    )
    run.add_argument("--bench-file", type=Path, help="Override workloads/bench_cases.json.")
    run.add_argument("--canary-file", type=Path, help="Override workloads/canary_cases.json.")
    run.add_argument(
        "--local",
        action="store_true",
        help="Run on this machine instead of dispatching to the configured remote host.",
    )
    run.add_argument(
        "--python",
        help="Python interpreter with httpx on the remote host (defaults to [bench].python).",
    )
    run.add_argument("--api-key-stdin", action="store_true", help=argparse.SUPPRESS)
    run.add_argument(
        "--no-archive", dest="archive", action="store_false", help="Skip .runs archive."
    )
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument(
        "--archive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save comparison in .runs (default off).",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    result = RunResult(f"bench {args.command}", True, {})
    root = find_repo_root(Path(__file__))
    start = perf_counter()
    try:
        if args.command == "run":
            if args.tag is not None:
                commands.validate_tag(args.tag)
            if args.rounds < 1 or args.warmup < 0:
                parser.error("rounds must be positive, warmup must be nonnegative")
            if not math.isfinite(args.timeout) or args.timeout <= 0:
                parser.error("timeout must be finite and positive")
            if args.api_key_stdin and not args.api_key:
                args.api_key = sys.stdin.read().strip() or None
            else:
                args.api_key = args.api_key or os.getenv("FLINT_API_KEY")
            commands.dispatch_run(args, root, result)
        else:
            commands.compare(args, result, root)
    except (
        OSError,
        RuntimeError,
        ValueError,
        httpx.HTTPError,
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        result.ok, result.error = False, f"{type(exc).__name__}: {exc}"
    result.duration_ms = (perf_counter() - start) * 1000
    return emit_result(
        result,
        json_mode=args.json,
        human_formatter=commands.human_format,
        repo_root=root,
        archive=args.archive,
        after_archive=(lambda path: commands.write_tag(path, result.data["metadata"]))
        if result.data.get("metadata", {}).get("tag")
        else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())

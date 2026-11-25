"""Resident GPU workbench."""

import argparse
import re
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from tools.common.config import find_repo_root, load_config
from tools.common.output import RunResult, emit_result

from . import commands


def layers(value):
    match = re.fullmatch(r"(\d+)\.\.(\d+)", value)
    if match:
        begin, end = map(int, match.groups())
        if 0 <= begin < end <= 48:
            return begin, end
    raise argparse.ArgumentTypeError("expected BEGIN..END with 0 <= BEGIN < END <= 48")


def candidate(value):
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", value) or value in {".", ".."}:
        raise argparse.ArgumentTypeError("invalid candidate name")
    return value


def savable(value):
    if candidate(value) in {"baseline", "current"}:
        raise argparse.ArgumentTypeError("reserved candidate name")
    return value


def natural(value):
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative count")
    return n


def positive(value):
    n = natural(value)
    if not n:
        raise argparse.ArgumentTypeError("expected a positive count")
    return n


def build_parser():
    parser = argparse.ArgumentParser(prog="forge", description=__doc__)
    parser.add_argument("--json", action="store_true")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "stop", "status", "run", "ab", "profile", "dump", "ls", "rm"):
        p = subs.add_parser(name)
        p.add_argument("--local", action="store_true")
        p.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
        if name in {"start", "run", "ab", "profile", "dump"}:
            p.add_argument("--no-build", action="store_true")
        if name == "start":
            model = p.add_mutually_exclusive_group()
            model.add_argument("--path")
            model.add_argument("--no-model", action="store_true")
            p.add_argument("--kv-seq-len", type=positive)
            p.add_argument("--scratch-mb", type=positive)
        if name in {"stop", "rm"}:
            p.add_argument("--force", action="store_true")
        if name == "rm":
            p.add_argument("name", type=candidate)
        if name in {"run", "ab", "profile", "dump"}:
            span = p.add_mutually_exclusive_group()
            span.add_argument("--layers", type=layers, default=(0, 48))
            if name != "dump":
                span.add_argument("--e2e", action="store_true")
                p.add_argument("--no-graph", action="store_true")
                p.add_argument("--no-spec", action="store_true")
            p.add_argument("--step", type=natural, default=0, help="0=prefill, 1=first decode")
            p.add_argument("--case")
            if name != "ab":
                p.add_argument("--candidate", type=candidate, default="current")
        if name in {"run", "ab"}:
            p.add_argument("--full", action="store_true", help="Use the extended timing sample")
            p.add_argument("--warmup", type=natural)
            p.add_argument("--iters", type=positive)
        if name == "ab":
            p.add_argument("cand_a", type=candidate)
            p.add_argument("cand_b", type=candidate)
            p.add_argument("--rounds", type=positive, default=5)
        if name == "run":
            p.add_argument("--save", type=savable)
        if name == "profile":
            p.add_argument("--nsys", action="store_true")
        if name == "dump":
            p.add_argument("--with-state", action="store_true")
            p.add_argument("--out")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = find_repo_root(Path(__file__))
    start = time.monotonic()
    result = RunResult(f"forge {args.command}", True, {})
    try:
        cfg = load_config(repo_root=root, allow_missing=args.local)
        if args.local:
            cfg = replace(cfg, remote=replace(cfg.remote, enabled=False))
        commands.execute(cfg, args, result)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        result.ok, result.error = False, str(exc)
    result.duration_ms = (time.monotonic() - start) * 1000
    return emit_result(
        result, json_mode=args.json, human_formatter=commands.human_format, repo_root=root
    )


if __name__ == "__main__":
    raise SystemExit(main())

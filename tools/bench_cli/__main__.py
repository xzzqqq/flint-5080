"""Entrypoint for `bin/bench` (placeholder until Phase 2)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench",
        description=(
            "Flint-5080 black-box OpenAI HTTP benchmark and canary harness. "
            "Scheduled for Phase 2."
        ),
    )
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

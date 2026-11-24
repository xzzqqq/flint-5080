"""Entrypoint for `bin/forge` (placeholder until Phase 1)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="forge",
        description=(
            "Flint-5080 resident GPU tuning workbench "
            "(start, stop, run, ab, profile, dump, ls, rm). "
            "Scheduled for Phase 1."
        ),
    )
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

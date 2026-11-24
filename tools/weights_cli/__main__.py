"""Entrypoint for `bin/weights` (placeholder until Phase 1)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="weights",
        description=(
            "Flint-5080 offline safetensors inspection, repacking, golden fixture generation, "
            "and KLD evaluation CLI. Scheduled for Phase 1."
        ),
    )
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Machine-readable run records and terminal output."""

import json
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import find_repo_root


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunResult:
    command: str
    ok: bool
    data: dict[str, Any]
    error: str | None = None
    schema_version: int = 1
    started_at: str = field(default_factory=utc_now)
    duration_ms: float = 0.0


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def archive_run(repo_root: Path, result: RunResult) -> Path:
    directory = repo_root / ".runs"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    command = re.sub(r"[^a-zA-Z0-9_]+", "-", result.command).strip("-")
    path = directory / f"{stamp}-{command}-{uuid4().hex[:8]}.json"
    path.write_text(json.dumps(asdict(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def emit_result(result: RunResult, *, json_mode: bool, human_formatter: Callable[[RunResult], str],
                archive: bool = True, repo_root: Path | None = None) -> int:
    run_file = None
    if archive:
        try:
            run_file = archive_run(repo_root or find_repo_root(), result)
        except OSError as exc:
            result.ok = False
            error = f"Cannot archive run: {exc}"
            result.error = f"{result.error}. {error}" if result.error else error
    payload = asdict(result)
    if run_file is not None:
        payload["run_file"] = str(run_file)
    print(json.dumps(payload, allow_nan=False) if json_mode else human_formatter(result))
    return 0 if result.ok else 1

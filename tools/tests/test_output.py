import json
from pathlib import Path

import pytest

from tools.common.output import RunResult, emit_result, log


@pytest.mark.parametrize("ok,code", [(True, 0), (False, 1)])
def test_json_contract(repo, capsys, ok, code):
    result = RunResult("dev/check --local", ok, {"test": "running"}, error=None if ok else "failed")
    log("build log")
    assert (
        emit_result(result, json_mode=True, human_formatter=lambda _: pytest.fail(), repo_root=repo)
        == code
    )
    std = capsys.readouterr()
    data = json.loads(std.out)
    assert std.err == "build log\n" and "build log" not in std.out
    path = Path(data.pop("run_file"))
    assert path.parent == repo / ".runs" and "dev-check-local" in path.name
    assert data == json.loads(path.read_text())
    assert data["schema_version"] == 1 and data["started_at"] and data["ok"] is ok


def test_human(repo, capsys):
    result = RunResult("dev status", True, {})
    assert (
        emit_result(
            result,
            json_mode=False,
            human_formatter=lambda r: r.command,
            archive=False,
            repo_root=repo,
        )
        == 0
    )
    assert capsys.readouterr().out == "dev status\n" and not (repo / ".runs").exists()


def test_archive_error_still_emits_json(repo, capsys):
    (repo / ".runs").touch()
    result = RunResult("dev status", True, {})
    assert emit_result(result, json_mode=True, repo_root=repo, human_formatter=str) == 1
    payload = json.loads(capsys.readouterr().out)
    assert not payload["ok"] and "Cannot archive" in payload["error"]

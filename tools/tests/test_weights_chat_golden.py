import hashlib
import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.weights_cli import __main__ as cli
from tools.weights_cli import commands

WORKLOADS = Path(__file__).resolve().parents[2] / "workloads"


@pytest.mark.parametrize("local", [False, True])
def test_chat_golden_cli(cfg, monkeypatch, capsys, local):
    cfg.remote.enabled = True
    cfg.weights.python = "/reference env/python"
    cfg.model.weights_dir = "checkpoint"
    monkeypatch.setattr(cli, "find_repo_root", lambda _: cfg.repo_root)
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: cfg)
    fixture = (WORKLOADS / "chat_golden.json").read_text()
    executor = Mock()
    executor.run.return_value = SimpleNamespace(stdout=fixture)

    def make_executor(effective):
        assert effective.is_remote != local
        return executor

    monkeypatch.setattr(commands, "Executor", make_executor)
    monkeypatch.setattr(commands, "GpuLeaseManager", lambda *args: pytest.fail("CPU-only command"))
    flags = ["--local", "--path", "my weights", "--python", "/local env/python"] if local else []
    destination = cfg.repo_root / "nested" / "chat.json"
    assert cli.main(["chat-golden", "--json", "--out", str(destination), *flags]) == 0
    assert destination.read_text() == fixture
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] and result["data"]["out"] == str(destination)
    assert executor.sync.call_count == int(not local)
    command = shlex.split(executor.run.call_args.args[0])
    assert command == [
        "/local env/python" if local else cfg.weights.python,
        "-m",
        "tools.weights_cli._chat_golden",
        "my weights" if local else str(Path(cfg.shared.models_dir) / cfg.model.weights_dir),
    ]
    assert executor.run.call_args.kwargs["check"]
    assert not executor.run.call_args.kwargs["echo_stdout"]
    assert (
        cli.build_parser().parse_args(["--json", "chat-golden"]).out == "workloads/chat_golden.json"
    )


def test_fixture_integrity():
    fixture = json.loads((WORKLOADS / "chat_golden.json").read_bytes())
    provenance = fixture["provenance"]
    assert (
        provenance["chat_template_sha256"]
        == hashlib.sha256(fixture["chat_template"].encode()).hexdigest()
    )
    for key in ("tokenizer_sha256", "tokenizer_config_sha256"):
        assert len(bytes.fromhex(provenance[key])) == 32
    assert provenance["transformers_version"] and provenance["model_name"]
    special = fixture["special_tokens"]
    assert special["eos_token_id"] == special["content_model_end_sampling"]
    assert special["pad_token_id"] == special["endoftext"]
    for group in ("template_cases", "template_error_cases", "tokenizer_cases"):
        rows = fixture[group]
        assert len({row["name"] for row in rows}) == len(rows)
    templates = {row["name"]: row for row in fixture["template_cases"]}
    assert len(templates) >= 12
    for row in templates.values():
        assert row["messages"] and row["rendered"] and row["input_ids"]
        assert all(isinstance(id_, int) for id_ in row["input_ids"])
    for ref in json.loads((WORKLOADS / "golden_cases.json").read_bytes())["cases"]:
        row = templates[ref["name"]]
        for key in ("input_ids", "messages", "reasoning_effort"):
            assert row[key] == ref[key]
    parallel = templates["parallel_tools"]
    calls = parallel["messages"][1]["tool_calls"]
    assert len(calls) == 2 and all(isinstance(tc["function"]["arguments"], str) for tc in calls)
    assert len(parallel["tools"]) == 2
    assert {row["name"] for row in fixture["template_error_cases"]} >= {
        "audio",
        "assistant_image",
        "unknown_role",
        "invalid_effort_unknown",
        "invalid_effort_1.5",
        "invalid_effort_-0.1",
        "non_object",
        "invalid_json",
    }
    texts = fixture["tokenizer_cases"]
    assert {row["name"] for row in texts} >= {
        "ascii",
        "rust",
        "python",
        "chinese",
        "emoji",
        "special",
        "spaces",
        "empty",
    }
    for row in texts:
        assert isinstance(row["token_ids"], list)
        assert row["decoded_keep_special"] == row["text"]
        assert row["decoded_skip_special"] == ("hi" if row["name"] == "special" else row["text"])

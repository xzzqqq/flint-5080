import shlex
import subprocess

import pytest

from tools.common.remote import RSYNC_EXCLUDES, Executor, build_rsync_args, ssh_base_args


def test_transport(cfg):
    assert "-p" not in ssh_base_args(cfg)
    cfg.remote.port = 2222
    ssh = ssh_base_args(cfg)
    for option in (
        "ControlMaster=auto",
        "ControlPersist=60",
        "ControlPath=/tmp/flint-ssh-%C",
        "BatchMode=yes",
        "ConnectTimeout=10",
    ):
        assert ssh[ssh.index(option) - 1] == "-o"
    assert ssh[ssh.index("-p") + 1] == "2222"
    args = build_rsync_args(cfg, dry_run=True)
    assert set(("-az", "--delete", "--itemize-changes", "--dry-run")) <= set(args)
    assert "--dry-run" not in build_rsync_args(cfg)
    excludes = {v for k, v in zip(args, args[1:], strict=False) if k == "--exclude"}
    required = """.git/ .dev.toml .env .env.* build/ engine/build*/ server/target/ target/
        .forge/ .runs/ .scratch/ models/ *.safetensors *.gguf *.nsys-rep* *.ncu-rep *.qdrep
        __pycache__/ *.pyc .pytest_cache/ .ruff_cache/ .venv/ .claude/ .codex/ .qwen/ .DS_Store
        *.log *.log.err workloads/private/""".split()
    assert set(required) <= excludes == set(RSYNC_EXCLUDES)
    assert shlex.split(args[args.index("-e") + 1]) == ssh
    assert args[-2:] == [str(cfg.repo_root) + "/", f"{cfg.remote.host}:{cfg.remote_workspace_dir}/"]


def test_local_run(cfg, capsys):
    ex = Executor(cfg)
    out = ex.run("pwd; echo err >&2", stream_stderr=True, check=True)
    assert out.stdout.strip() == str(cfg.repo_root) and out.stderr == "err\n"
    captured = capsys.readouterr()
    assert captured.out == "" and "err\n" in captured.err and out.stdout in captured.err
    assert ex.sync().skipped
    assert ex.run("pwd", cwd="/").stdout == "/\n"
    assert ex.run("exit 17").returncode == 17
    with pytest.raises(RuntimeError, match="problem"):
        ex.run("echo problem >&2; exit 3", check=True)


@pytest.mark.parametrize("stream", [False, True])
def test_timeout(cfg, stream):
    with pytest.raises(subprocess.TimeoutExpired):
        Executor(cfg).run("sleep 10", stream_stderr=stream, timeout=0)


def test_remote_sync(cfg, tmp_path, monkeypatch):
    cfg.remote.enabled = True
    cfg.remote.workspaces_root = str(tmp_path / "remote")
    ssh, rsync = tmp_path / "ssh", tmp_path / "rsync"
    ssh.write_text('#!/bin/sh\nfor last do :; done\nexec bash -c "$last"\n')
    rsync.write_text('#!/bin/sh\nprintf "%s\\n" "<f+++++++++ x" ".d..t...... ./" "<f.st...... y"\n')
    import os

    for path in (ssh, rsync):
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ["PATH"])
    result = Executor(cfg).sync(dry_run=True)
    assert not result.skipped and result.transferred_files == 2
    assert "--dry-run" in result.cmd and (tmp_path / "remote/my-worktree").is_dir()

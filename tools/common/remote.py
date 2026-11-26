"""Workspace synchronization and execution on the local or configured SSH host."""

import codecs
import os
import selectors
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

from .config import DevConfig, _shell_path


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float
    cmd: list[str]


@dataclass
class SyncResult:
    skipped: bool
    duration_ms: float
    remote_dir: str
    transferred_files: int
    cmd: list[str]


RSYNC_EXCLUDES = (
    ".git/",
    ".git",
    ".dev.toml",
    ".env",
    ".env.*",
    "build/",
    "engine/build*/",
    "server/target/",
    "target/",
    ".forge/",
    ".serve/",
    ".runs/",
    ".scratch/",
    "models/",
    "*.safetensors",
    "*.gguf",
    "*.nsys-rep*",
    "*.ncu-rep",
    "*.qdrep",
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
    ".ruff_cache/",
    ".venv/",
    ".claude/",
    ".codex/",
    ".qwen/",
    ".DS_Store",
    "*.log",
    "*.log.err",
    "workloads/private/",
)


def ssh_base_args(cfg: DevConfig) -> list[str]:
    args = [
        "ssh",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=60",
        "-o",
        "ControlPath=/tmp/flint-ssh-%C",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
    ]
    if cfg.remote.port > 0:
        args.extend(["-p", str(cfg.remote.port)])
    return args


def build_rsync_args(cfg: DevConfig, *, dry_run: bool = False) -> list[str]:
    # Protocol 29 truncates mtimes to seconds. Preserve same-size fast edits and
    # let Ninja see each changed input as newer than its last remote object.
    args = ["rsync", "-az", "--checksum", "--no-times", "--delete", "--itemize-changes"]
    if dry_run:
        args.append("--dry-run")
    for pattern in RSYNC_EXCLUDES:
        args.extend(["--exclude", pattern])
    return args + [
        "-e",
        shlex.join(ssh_base_args(cfg)),
        str(cfg.repo_root) + "/",
        f"{cfg.remote.host}:{cfg.remote_workspace_dir}/",
    ]


def _capture(
    cmd: list[str],
    *,
    stream: bool,
    timeout: int | None,
    echo_stdout: bool = True,
    input_text: str | None = None,
) -> CommandResult:
    start = time.monotonic()
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    ) as proc:
        try:
            if stream:
                chunks = [[], []]
                with selectors.DefaultSelector() as selector:
                    for index, pipe in enumerate((proc.stdout, proc.stderr)):
                        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                        selector.register(pipe, selectors.EVENT_READ, (index, decoder))
                    if proc.stdin is not None:
                        pending = memoryview(input_text.encode())
                        if pending:
                            os.set_blocking(proc.stdin.fileno(), False)
                            selector.register(proc.stdin, selectors.EVENT_WRITE)
                        else:
                            proc.stdin.close()
                    while selector.get_map():
                        remaining = None if timeout is None else start + timeout - time.monotonic()
                        if remaining is not None and remaining <= 0:
                            raise subprocess.TimeoutExpired(cmd[0], timeout)
                        for key, events in selector.select(remaining):
                            if events & selectors.EVENT_WRITE:
                                try:
                                    written = os.write(key.fd, pending[:65536])
                                    pending = pending[written:]
                                    if pending:
                                        continue
                                except BlockingIOError:
                                    continue
                                except BrokenPipeError:
                                    pass
                                selector.unregister(key.fileobj)
                                key.fileobj.close()
                                continue
                            index, decoder = key.data
                            block = os.read(key.fd, 65536)
                            value = decoder.decode(block, final=not block)
                            chunks[index].append(value)
                            if index == 1 or echo_stdout:
                                sys.stderr.write(value)
                                sys.stderr.flush()
                            if not block:
                                selector.unregister(key.fileobj)
                stdout, stderr = ("".join(parts) for parts in chunks)
                remaining = None if timeout is None else max(0, start + timeout - time.monotonic())
                proc.wait(timeout=remaining)
            else:
                out, err = proc.communicate(
                    input=input_text.encode() if input_text is not None else None, timeout=timeout
                )
                stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise
    return CommandResult(proc.returncode, stdout, stderr, (time.monotonic() - start) * 1000, cmd)


class Executor:
    def __init__(self, cfg: DevConfig):
        self.cfg = cfg

    def sync(self, *, dry_run: bool = False) -> SyncResult:
        cfg = self.cfg
        directory = cfg.remote_workspace_dir
        if not cfg.is_remote:
            return SyncResult(True, 0, directory, 0, [])
        start = time.monotonic()
        self.run("mkdir -p -- " + _shell_path(directory), cwd=".", check=True)
        args = build_rsync_args(cfg, dry_run=dry_run)
        result = _capture(args, stream=True, timeout=None)
        if result.returncode:
            raise RuntimeError(f"rsync failed ({result.returncode}): {result.stderr.strip()}")
        transferred = sum(line.startswith(("<f", ">f")) for line in result.stdout.splitlines())
        return SyncResult(False, (time.monotonic() - start) * 1000, directory, transferred, args)

    def _args(self, shell_cmd: str, cwd: str | None, *, tty: bool = False) -> list[str]:
        cfg = self.cfg
        if not cfg.is_remote:
            return [
                "bash",
                "-c",
                f"cd -- {shlex.quote(cwd or str(cfg.repo_root))} || exit 1\n{shell_cmd}",
            ]
        path = _shell_path(cwd or cfg.remote_workspace_dir)
        script = f"{cfg.remote_env_prefix()}\ncd -- {path} || exit 1\n{shell_cmd}"
        return [
            *ssh_base_args(cfg),
            *(["-t"] if tty else []),
            cfg.remote.host,
            "bash -lc " + shlex.quote(script),
        ]

    def run(
        self,
        shell_cmd: str,
        *,
        cwd: str | None = None,
        check: bool = False,
        stream_stderr: bool = False,
        timeout: int | None = None,
        echo_stdout: bool = True,
        input_text: str | None = None,
    ) -> CommandResult:
        result = _capture(
            self._args(shell_cmd, cwd),
            stream=stream_stderr,
            timeout=timeout,
            echo_stdout=echo_stdout,
            input_text=input_text,
        )
        if check and result.returncode:
            raise RuntimeError(
                result.stderr.strip() or f"Command failed (exit {result.returncode})"
            )
        return result

    def interactive_ssh(self, extra_cmd: list[str] | None = None, *, stdout=None) -> int:
        command = shlex.join(extra_cmd) if extra_cmd else 'exec "${SHELL:-/bin/bash}" -i'
        return subprocess.call(self._args(command, None, tty=not extra_cmd), stdout=stdout)

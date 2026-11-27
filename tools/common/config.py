"""Local configuration and shell environment shared by the tooling CLIs."""

from __future__ import annotations

import os
import re
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import get_type_hints


class ConfigError(RuntimeError):
    pass


@dataclass
class RemoteConfig:
    enabled: bool = True
    host: str = "gpu-dev"
    port: int = 0
    workspaces_root: str = "/home/user/workspace/flint-5080-workspaces"
    workspace_name: str = ""
    cuda_home: str = "/usr/local/cuda"
    path_prepend: list[str] = field(default_factory=lambda: ["$HOME/.cargo/bin"])
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class SharedConfig:
    models_dir: str = "/home/user/models"
    ccache_dir: str = "/home/user/.cache/flint-5080/ccache"
    lock_file: str = "/tmp/flint-5080-gpu.lock.json"


@dataclass
class BuildConfig:
    build_type: str = "Release"
    cuda_arch: str = "120a"
    jobs: int = 0
    linker: str = "auto"
    use_ccache: bool = True


@dataclass
class GpuConfig:
    lease_ttl_sec: int = 600


@dataclass
class ModelConfig:
    draft_weights_dir: str = ""
    weights_dir: str = ""


@dataclass
class WeightsConfig:
    python: str = ""
    golden_dir: str = ""


@dataclass
class BenchConfig:
    python: str = ""


@dataclass
class ServerConfig:
    bind: str = "127.0.0.1:8080"


@dataclass
class WorkloadConfig:
    default_case: str = ""


def _shell_path(value: str) -> str:
    for prefix in ("$HOME", "${HOME}", "~"):
        if value == prefix or value.startswith(prefix + "/"):
            return '"$HOME"' + shlex.quote(value[len(prefix) :])
    return shlex.quote(value)


@dataclass
class DevConfig:
    repo_root: Path
    config_path: Path | None = None
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    shared: SharedConfig = field(default_factory=SharedConfig)
    build: BuildConfig = field(default_factory=BuildConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)

    @property
    def is_remote(self) -> bool:
        return self.remote.enabled

    @property
    def effective_workspace_name(self) -> str:
        name = self.remote.workspace_name or self.repo_root.name
        return re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip(".-") or "workspace"

    @property
    def remote_workspace_dir(self) -> str:
        return str(PurePosixPath(self.remote.workspaces_root) / self.effective_workspace_name)

    def resolve_model_dir(
        self, override: str | Path | None = None, *, error_message: str | None = None
    ) -> PurePosixPath:
        if override:
            return PurePosixPath(override)
        if self.model.weights_dir:
            return PurePosixPath(self.shared.models_dir) / self.model.weights_dir
        raise ConfigError(error_message or "Set model.weights_dir in .dev.toml or pass --path.")

    def resolve_draft_dir(self, override: str | Path | None = None) -> PurePosixPath | None:
        value = self.model.draft_weights_dir if override is None else override
        return PurePosixPath(self.shared.models_dir) / value if value else None

    def resolve_golden_dir(self, model_dir: str | PurePosixPath) -> PurePosixPath:
        return (
            PurePosixPath(self.weights.golden_dir or ".forge/golden")
            / PurePosixPath(model_dir).name
        )

    def remote_env_prefix(self) -> str:
        exports = []
        paths = [_shell_path(p) for p in self.remote.path_prepend]
        if self.remote.cuda_home:
            exports.append("export CUDA_HOME=" + _shell_path(self.remote.cuda_home))
            exports.append(
                'export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
            )
            paths.append('"$CUDA_HOME/bin"')
        if paths:
            exports.append("export PATH=" + ":".join([*paths, '"$PATH"']))
        if self.shared.ccache_dir:
            exports.append("export CCACHE_DIR=" + _shell_path(self.shared.ccache_dir))
        exports.extend(f"export {k}={shlex.quote(v)}" for k, v in self.remote.env.items())
        return "\n".join(exports)


def find_repo_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    for parent in (path, *path.parents):
        if (parent / ".dev.toml.example").is_file() or (parent / ".git").exists():
            return parent
    raise ConfigError(f"No Flint repository found above {path}")


def _section(name, cls, raw):
    if not isinstance(raw, dict):
        raise ConfigError(f"[{name}] must be a table")
    schema = get_type_hints(cls)
    for key, value in raw.items():
        expected = schema.get(key)
        if expected == list[str]:
            valid = isinstance(value, list) and all(isinstance(v, str) for v in value)
        elif expected == dict[str, str]:
            valid = isinstance(value, dict) and all(
                re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", k) and isinstance(v, str)
                for k, v in value.items()
            )
        else:
            valid = type(value) is expected
        if not valid:
            raise ConfigError(f"Invalid {name}.{key} (expected {expected})")
    return cls(**raw)


def load_config(
    config_path: Path | None = None, repo_root: Path | None = None, allow_missing: bool = False
) -> DevConfig:
    root = repo_root.resolve() if repo_root else find_repo_root()
    path = (
        Path(config_path or os.environ.get("FLINT_DEV_CONFIG") or root / ".dev.toml")
        .expanduser()
        .resolve()
    )
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as exc:
        if not allow_missing:
            raise ConfigError(
                f"Missing {path}. Copy .dev.toml.example to .dev.toml and configure it."
            ) from exc
        raw, path = {}, None
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc

    cfg = DevConfig(root, path)
    sections = get_type_hints(DevConfig)
    for name, value in raw.items():
        if name not in sections or name in {"repo_root", "config_path"}:
            raise ConfigError(f"Unknown config section {name}")
        setattr(cfg, name, _section(name, sections[name], value))
    if path is None or os.environ.get("FLINT_LOCAL") == "1":
        cfg.remote.enabled = False
    if host := os.environ.get("FLINT_REMOTE_HOST"):
        cfg.remote.host = host
    if cfg.is_remote and (not cfg.remote.host.strip() or cfg.remote.host.startswith("-")):
        raise ConfigError("remote.host must be an SSH destination")
    if not 0 <= cfg.remote.port <= 65535 or cfg.build.jobs < 0 or cfg.gpu.lease_ttl_sec <= 0:
        raise ConfigError(
            "Check remote.port (0..65535), build.jobs (>=0), and gpu.lease_ttl_sec (>0)"
        )
    if not cfg.shared.lock_file or not cfg.remote.workspaces_root:
        raise ConfigError("shared.lock_file and remote.workspaces_root must be nonempty")
    return cfg

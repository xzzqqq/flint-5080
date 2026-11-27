import os
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from tools.common.config import ConfigError, find_repo_root, load_config


def test_example(repo):
    cfg = load_config(repo / ".dev.toml.example", repo)
    assert cfg.is_remote and cfg.remote.host == "gpu-dev"
    assert cfg.build.cuda_arch == "120a" and cfg.build.use_ccache
    assert cfg.shared.models_dir and cfg.gpu.lease_ttl_sec == 600
    assert cfg.weights.python == cfg.weights.golden_dir == cfg.bench.python == ""
    assert cfg.server.bind == "127.0.0.1:8080"
    assert cfg.model.weights_dir == cfg.workload.default_case == ""
    assert cfg.remote_workspace_dir == cfg.remote.workspaces_root + "/my-worktree"


def test_weights_and_bench_sections(repo):
    (repo / ".dev.toml").write_text(
        '[weights]\npython = "/venv/python"\ngolden_dir = "/golden"\n'
        '[bench]\npython = "/bench/python"\n'
    )
    cfg = load_config(repo_root=repo)
    assert (cfg.weights.python, cfg.weights.golden_dir) == ("/venv/python", "/golden")
    assert cfg.bench.python == "/bench/python"


def test_root(repo):
    assert find_repo_root(repo / "bin") == repo
    (repo / ".dev.toml.example").unlink()
    (repo / ".git").write_text("gitdir: elsewhere")
    assert find_repo_root(repo / "bin") == repo


def test_missing(repo):
    with pytest.raises(ConfigError, match=r"\.dev\.toml\.example"):
        load_config(repo_root=repo)
    cfg = load_config(repo_root=repo, allow_missing=True)
    assert not cfg.is_remote and cfg.config_path is None


def test_overrides(repo, monkeypatch):
    path = repo / "alternate.toml"
    path.write_text('[remote]\nworkspace_name = "../../trial A"\n')
    monkeypatch.setenv("FLINT_DEV_CONFIG", str(path))
    monkeypatch.setenv("FLINT_LOCAL", "1")
    monkeypatch.setenv("FLINT_REMOTE_HOST", "other-host")
    cfg = load_config(repo_root=repo)
    assert cfg.config_path == path and not cfg.is_remote
    assert cfg.remote.host == "other-host"
    assert cfg.effective_workspace_name == "trial-A"


@pytest.mark.parametrize(
    "text",
    [
        "bad =",
        '[remote]\nenabled = "yes"',
        "[build]\njobs = true",
        "[weights]\npython = 3",
        "[bench]\npython = 3",
        '[remote.env]\n"bad-key" = "x"',
        "[gpu]\nlease_ttl_sec = 0",
    ],
)
def test_invalid_config(repo, text):
    (repo / ".dev.toml").write_text(text)
    with pytest.raises(ConfigError):
        load_config(repo_root=repo)


def test_remote_env_is_shell_safe(cfg):
    cfg.remote.path_prepend = ["$HOME/.cargo/bin", "/path with space/$(false)"]
    cfg.remote.cuda_home = "/cuda 'special'"
    cfg.shared.ccache_dir = "$HOME/cache"
    cfg.remote.env = {"EXTRA": "literal $PATH 'quoted'"}
    script = (
        cfg.remote_env_prefix()
        + '\nprintf "%s\\n" "$PATH" "$LD_LIBRARY_PATH" "$CCACHE_DIR" "$EXTRA"'
    )
    env = dict(os.environ, PATH="/bin", HOME="/my home", LD_LIBRARY_PATH="/old/lib")
    out = subprocess.check_output(["bash", "-c", script], text=True, env=env)
    assert out.splitlines() == [
        "/my home/.cargo/bin:/path with space/$(false):/cuda 'special'/bin:/bin",
        "/cuda 'special'/lib64:/old/lib",
        "/my home/cache",
        "literal $PATH 'quoted'",
    ]
    cfg.remote.cuda_home = cfg.shared.ccache_dir = ""
    assert "CUDA_HOME" not in cfg.remote_env_prefix()
    assert "CCACHE_DIR" not in cfg.remote_env_prefix()


def test_resolve_model_dir(cfg):
    with pytest.raises(ConfigError, match="model.weights_dir"):
        cfg.resolve_model_dir()
    with pytest.raises(ConfigError, match="Use --model-dir"):
        cfg.resolve_model_dir(error_message="Use --model-dir")
    assert cfg.resolve_model_dir(Path("custom/model")) == PurePosixPath("custom/model")
    cfg.shared.models_dir = "/shared models"
    cfg.model.weights_dir = "weights/mini"
    assert cfg.resolve_model_dir() == PurePosixPath("/shared models/weights/mini")
    assert cfg.resolve_model_dir("/override") == PurePosixPath("/override")
    cfg.model.weights_dir = "/absolute/mini"
    assert cfg.resolve_model_dir() == PurePosixPath("/absolute/mini")


def test_resolve_golden_dir(cfg):
    assert cfg.resolve_golden_dir("/models/mini/") == PurePosixPath(".forge/golden/mini")
    cfg.weights.golden_dir = "/reference fixtures"
    assert cfg.resolve_golden_dir(PurePosixPath("mini")) == PurePosixPath(
        "/reference fixtures/mini"
    )


def test_draft_directory(cfg):
    cfg.shared.models_dir = "/weights"
    assert cfg.resolve_draft_dir() is None
    cfg.model.draft_weights_dir = "draft"
    assert cfg.resolve_draft_dir() == PurePosixPath("/weights/draft")
    assert cfg.resolve_draft_dir("/external") == PurePosixPath("/external")
    assert cfg.resolve_draft_dir("override") == PurePosixPath("/weights/override")
    assert cfg.resolve_draft_dir("") is None

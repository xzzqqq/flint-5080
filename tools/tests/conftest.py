import shutil
from pathlib import Path

import pytest

from tools.common.config import load_config


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    for name in ("FLINT_DEV_CONFIG", "FLINT_LOCAL", "FLINT_REMOTE_HOST"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "my worktree"
    root.mkdir()
    shutil.copy(ROOT / ".dev.toml.example", root)
    shutil.copytree(ROOT / "bin", root / "bin")
    return root


@pytest.fixture
def cfg(repo):
    cfg = load_config(repo_root=repo, allow_missing=True)
    cfg.shared.lock_file = str(repo / "locks" / "gpu.json")
    return cfg

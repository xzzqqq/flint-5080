import os
import shlex
import subprocess

import pytest

from tools.common.build import engine_build_script


def test_engine_build_script(cfg):
    default = engine_build_script(cfg)
    assert "rm -rf" not in default and "build.ninja" not in default
    assert "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON" in default
    assert default.splitlines()[-1] == "cmake --build engine/build --parallel"

    cfg.build.jobs = 4
    cfg.build.use_ccache = False
    script = engine_build_script(
        cfg, clean=True, targets=("forward", "second target"), incremental=True
    )
    assert script.startswith("rm -rf engine/build\n")
    assert "use_ccache=0" in script
    assert shlex.split(script.splitlines()[-1]) == [
        "cmake",
        "--build",
        "engine/build",
        "--target",
        "forward",
        "second target",
        "--parallel",
        "4",
    ]


@pytest.mark.parametrize("ninja", [False, True])
@pytest.mark.parametrize("database", [False, True])
def test_incremental_configure(cfg, tmp_path, ninja, database):
    build = tmp_path / "engine/build"
    build.mkdir(parents=True)
    if ninja:
        (build / "build.ninja").touch()
    if database:
        (build / "compile_commands.json").touch()
    cmake = tmp_path / "cmake"
    cmake.write_text('#!/bin/sh\necho "$@"\n')
    cmake.chmod(0o755)
    run = subprocess.run(
        ["sh", "-ec", engine_build_script(cfg, incremental=True)],
        cwd=tmp_path,
        env=dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}"),
        check=True,
        capture_output=True,
        text=True,
    )
    assert ("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON" in run.stdout) == (not ninja or not database)
    assert run.stdout.endswith("--build engine/build --parallel\n")

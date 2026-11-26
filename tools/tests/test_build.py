import shlex

from tools.common.build import engine_build_script


def test_engine_build_script(cfg):
    default = engine_build_script(cfg)
    assert "rm -rf" not in default and "build.ninja" not in default
    assert default.splitlines()[-1] == "cmake --build engine/build --parallel"

    cfg.build.jobs = 4
    cfg.build.use_ccache = False
    script = engine_build_script(
        cfg, clean=True, targets=("forward", "second target"), incremental=True
    )
    assert script.startswith("use_ccache=0\n")
    clean, configure, build = script.splitlines()[-3:]
    assert clean == "rm -rf engine/build"
    assert configure.startswith("if ! test -f engine/build/build.ninja; then cmake ")
    assert shlex.split(build) == [
        "cmake",
        "--build",
        "engine/build",
        "--target",
        "forward",
        "second target",
        "--parallel",
        "4",
    ]

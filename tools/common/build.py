"""CMake configure and build commands shared by the engine and forge builds."""

from __future__ import annotations

import shlex
from collections.abc import Sequence

from .config import DevConfig


def engine_configure_script(cfg: DevConfig) -> str:
    build = cfg.build
    flags = [
        "cmake",
        "-S",
        "engine",
        "-B",
        "engine/build",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-G",
        "Ninja",
        f"-DCMAKE_BUILD_TYPE={build.build_type}",
        f"-DCMAKE_CUDA_ARCHITECTURES={build.cuda_arch}",
    ]
    configure = shlex.join(flags)
    for lang in ("C", "CXX", "CUDA"):
        configure += f' "-DCMAKE_{lang}_COMPILER_LAUNCHER=$launcher"'
    for kind in ("EXE", "SHARED", "MODULE"):
        configure += f' "-DCMAKE_{kind}_LINKER_FLAGS=$link_flags"'
    lines = [
        f"use_ccache={1 if build.use_ccache else 0}",
        'launcher=""',
        'if [ "$use_ccache" = 1 ] && command -v ccache >/dev/null; then launcher=ccache; fi',
        f"linker={shlex.quote(build.linker)}",
        'if [ "$linker" = auto ]; then',
        "    if command -v mold >/dev/null; then linker=mold",
        "    elif command -v ld.lld >/dev/null; then linker=lld",
        "    else linker=system",
        "    fi",
        "fi",
        'link_flags=""',
        'if [ "$linker" != system ] && [ -n "$linker" ]; then link_flags="-fuse-ld=$linker"; fi',
    ]
    return "\n".join([*lines, configure])


def engine_build_script(
    cfg: DevConfig, *, clean: bool = False, targets: Sequence[str] = (), incremental: bool = False
) -> str:
    configure = engine_configure_script(cfg)
    if incremental:
        configure = (
            "if ! test -f engine/build/build.ninja || "
            f"! test -f engine/build/compile_commands.json; then\n{configure}\nfi"
        )
    compile_args = ["cmake", "--build", "engine/build"]
    if targets:
        compile_args.extend(["--target", *targets])
    compile_args.append("--parallel")
    if cfg.build.jobs:
        compile_args.append(str(cfg.build.jobs))
    lines = ["rm -rf engine/build"] if clean else []
    return "\n".join([*lines, configure, shlex.join(compile_args)])

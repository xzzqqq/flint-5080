"""CMake configure and build commands shared by the engine and forge builds."""

from __future__ import annotations

import shlex
from collections.abc import Sequence

from .config import DevConfig


def engine_build_script(
    cfg: DevConfig, *, clean: bool = False, targets: Sequence[str] = (), incremental: bool = False
) -> str:
    build = cfg.build
    flags = [
        "cmake",
        "-S",
        "engine",
        "-B",
        "engine/build",
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
    compile_args = ["cmake", "--build", "engine/build"]
    if targets:
        compile_args.extend(["--target", *targets])
    compile_args.append("--parallel")
    if build.jobs:
        compile_args.append(str(build.jobs))
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
    if clean:
        lines.append("rm -rf engine/build")
    if incremental:
        configure = f"if ! test -f engine/build/build.ninja; then {configure}; fi"
    lines.extend([configure, shlex.join(compile_args)])
    return "\n".join(lines)

import json
import shlex
import subprocess

import pytest

from tools.forge_cli import _ptxas
from tools.forge_cli.commands import modified_cuda_files

REPORT = """ptxas info    : 16 bytes gmem
ptxas info    : Compiling entry function '_Z4testv' for 'sm_120a'
ptxas info    : Function properties for _Z4testv
    48 bytes stack frame, 16 bytes spill stores, 8 bytes spill loads
ptxas info : Used 42 registers, used 2 barriers, 128 bytes smem, 16 bytes cmem[0], 4 bytes cmem[2]
ptxas info    : Function properties for helper
    72 bytes stack frame, 40 bytes spill stores, 32 bytes spill loads
ptxas info    : Compiling entry function 'plain' for 'sm_120a'
ptxas info    : Used 8 registers, used 0 barriers
"""


def test_resources_per_entry():
    a, b = _ptxas.parse_resources(REPORT, "test.cu")
    assert a == dict(
        file="test.cu",
        kernel="_Z4testv",
        arch="sm_120a",
        regs=42,
        barriers=2,
        smem_bytes=128,
        cmem_bytes=20,
        stack_bytes=48,
        spill_stores_bytes=16,
        spill_loads_bytes=8,
    )
    assert b == dict(
        a,
        kernel="plain",
        regs=8,
        barriers=0,
        smem_bytes=0,
        cmem_bytes=0,
        stack_bytes=0,
        spill_stores_bytes=0,
        spill_loads_bytes=0,
    )


@pytest.fixture
def database(tmp_path):
    build = tmp_path / "engine/build"
    build.mkdir(parents=True)
    compiler = tmp_path / "nvcc"
    compiler.write_text(
        "#!/bin/sh\npwd >> calls\nprintf '%s\\n' \"$@\" >> calls\ncat >&2 <<'EOF'\n"
        + REPORT
        + "EOF\n"
    )
    compiler.chmod(0o755)
    entries = []
    for path in ["ops/a b.cu", "ops/second.cu", "runtime/second.cu", "ops/other.cpp"]:
        file = "../src/" + path
        args = [
            "/usr/bin/ccache",
            str(compiler),
            "-DFLAG=two words",
            "-MD",
            "-MF",
            "compiled.d",
            "-c",
            file,
            "-o",
            "compiled.o",
        ]
        command = {"arguments": args} if entries else {"command": shlex.join(args)}
        entries.append(dict(directory=str(build), file=file, **command))
    (build / "compile_commands.json").write_text(json.dumps(entries))
    return build


def test_inspect_command(database, monkeypatch):
    root = database.parents[1]
    monkeypatch.setattr(_ptxas.shutil, "which", lambda _: None)
    records = _ptxas.inspect(root, ["a b.cu", "engine/src/ops/a b.cu"])
    assert len(records) == 2 and records[0]["kernel"] == "_Z4testv"
    assert records[0]["file"] == "engine/src/ops/a b.cu"
    assert (database / "calls").read_text().splitlines() == [
        str(database),
        "-DFLAG=two words",
        "../src/ops/a b.cu",
        "-cubin",
        "-o",
        "/dev/null",
        "-Xptxas=-v",
    ]
    filt = root / "c++filt"
    filt.write_text("#!/bin/sh\nsed 's/_Z4testv/test()/'\n")
    filt.chmod(0o755)
    monkeypatch.setattr(_ptxas.shutil, "which", lambda _: str(filt))
    for files, all_ops in [([], False), (["a b.cu"], True)]:
        rows = _ptxas.inspect(root, files, all_ops)
        assert len(rows) == 4
        assert {row["file"] for row in rows} == {
            "engine/src/ops/a b.cu",
            "engine/src/ops/second.cu",
        }
        assert rows[0]["kernel"] == "test()"


def test_inspect_errors(database):
    root = database.parents[1]
    for path, error in [("second.cu", "ambiguous"), ("missing.cu", "no CUDA compile entry")]:
        with pytest.raises(ValueError, match=error):
            _ptxas.inspect(root, [path])
    (root / "nvcc").write_text("#!/bin/sh\necho diagnostic >&2\nexit 1\n")
    with pytest.raises(RuntimeError, match="diagnostic"):
        _ptxas.inspect(root, ["a b.cu"])


def test_modified_cuda_files(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for name in ["engine/src/ops/tracked.cu", "engine/src/ops/deleted.cu"]:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    subprocess.run(["git", "add", "engine"], cwd=tmp_path, check=True)
    (tmp_path / "engine/src/ops/deleted.cu").unlink()
    (tmp_path / "engine/src/ops/space name.cu").touch()
    (tmp_path / "engine/src/ops/unrelated.cpp").touch()
    (tmp_path / "other.cu").touch()
    assert sorted(modified_cuda_files(tmp_path)) == [
        "engine/src/ops/space name.cu",
        "engine/src/ops/tracked.cu",
    ]

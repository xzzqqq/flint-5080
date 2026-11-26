import asyncio
import io
import json
import shlex
import shutil
import sys
from pathlib import Path

import httpx
import pytest

from tools.bench_cli import __main__ as cli
from tools.bench_cli import client
from tools.bench_cli.stats import paired_bootstrap_ci
from tools.bench_cli.workloads import PERF, evaluate, load_workloads
from tools.common.remote import Executor

ROOT = Path(__file__).resolve().parents[2]


class Endpoint:
    """Serve Flint-shaped terminal frames and fragmented OpenAI tool deltas."""

    def __init__(self):
        self.requests = []
        self.active = self.peak = 0
        self.model = "test-model"
        self.external = False
        self.fault = None
        self.prefill = 20
        self.chat = []

    async def __call__(self, req):
        self.requests.append(req)
        if req.url.path == "/ready":
            if self.external:
                return httpx.Response(404)
            return httpx.Response(
                200,
                json={
                    "model": self.model,
                    "engine": {
                        "gpu_name": "mock",
                        "max_context": 4096,
                        "prefix_cache_slots": 3,
                        "use_cuda_graph": True,
                        "use_speculative": True,
                        "speculative_active": True,
                        "completed_requests": 101,
                    },
                },
            )
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        assert req.url.path == "/v1/chat/completions"
        body = json.loads(req.content)
        self.chat.append(body)
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        prompt = body["messages"][-1]["content"]
        calls = []
        if "17 * 23" in prompt:
            content = "392" if self.fault == "math" else "391"
        elif "Python loop" in prompt:
            content = "for i in range(len(items))"
        elif "get_weather" in prompt:
            content = ""
            for i, city in enumerate(["Tokyo", "Paris"] if "Paris" in prompt else ["Tokyo"]):
                calls.append(
                    {
                        "id": f"call-{len(self.chat)}-{i}",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": city}),
                        },
                    }
                )
            if self.fault == "missing":
                calls = []
            if self.fault == "arguments" and len(calls) > 1:
                calls[1]["function"]["arguments"] = '{"city":'
        elif "exactly these numbers" in prompt:
            content = "1, 2, 3, 4, " + ("5" if self.fault == "stop" else "")
        elif "Hash Table" in prompt:
            content = "哈希表 Hash Table 🧩"
            if self.fault == "parity" and not body["stream"]:
                content += "!"
        else:
            content = "ok"
        reason = "tool_calls" if "get_weather" in prompt else "stop"
        reasoning = "思考" + ("\ufffd" if self.fault == "replacement" else "")
        end = {}
        if not body["stream"] or body.get("stream_options", {}).get("include_usage"):
            end["usage"] = {
                "prompt_tokens": 80,
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 24},
            }
        if not self.external and req.headers.get("X-Flint-Telemetry") == "1":
            end["x_flint_telemetry"] = dict(
                prefill_tokens=80,
                cached_prefix_tokens=24,
                decode_tokens=8,
                prefill_ms=self.prefill,
                decode_ms=40,
                decode_rounds=4,
                spec_accepted=3,
            )
        if not body["stream"]:
            msg = dict(content=content, reasoning_content=reasoning, tool_calls=calls)
            return httpx.Response(
                200, json=dict(choices=[dict(message=msg, finish_reason=reason)], **end)
            )

        def frame(delta, finish=None):
            return {"choices": [dict(delta=delta, finish_reason=finish)]}

        frames = [frame({"role": "assistant"}), frame({"reasoning_content": reasoning})]
        if content:
            frames.append(frame(dict(content=content)))
        for i, call in enumerate(calls):
            args = call["function"]["arguments"]
            frames.append(
                frame(
                    {
                        "tool_calls": [
                            dict(call, index=i, function=dict(name="get_", arguments=args[:4]))
                        ]
                    }
                )
            )
        for i, call in reversed(list(enumerate(calls))):
            frames.append(
                frame(
                    {
                        "tool_calls": [
                            dict(
                                index=i,
                                function=dict(
                                    name="weather", arguments=call["function"]["arguments"][4:]
                                ),
                            )
                        ]
                    }
                )
            )
        frames.extend([frame({}, reason), dict(choices=[], **end)])
        sse = ": keepalive\n\n" + "".join(f"data: {json.dumps(f)}\n\n" for f in frames)
        if self.fault != "truncated":
            sse += "data: [DONE]\n\n"
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})


@pytest.fixture
def harness(repo, monkeypatch, capsys):
    shutil.copytree(ROOT / "workloads", repo / "workloads")
    monkeypatch.setattr(cli, "find_repo_root", lambda _: repo)
    endpoint = Endpoint()
    cls = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: cls(transport=httpx.MockTransport(endpoint), **kw)
    )

    def invoke(*args, code=0, human=False):
        actual = cli.main(list(args))
        output = capsys.readouterr().out
        assert actual == code, output
        return output if human else json.loads(output)

    return endpoint, invoke, repo


@pytest.mark.parametrize("suite", [*PERF, "canary", "all"])
def test_suites(suite, harness, monkeypatch):
    endpoint, invoke, repo = harness
    monkeypatch.setenv("FLINT_API_KEY", "mock-credential")
    result = invoke("--json", "run", "--suite", suite, "--rounds", "2", "--warmup", "1")
    path = Path(result.pop("run_file"))
    assert path.parent == repo / ".runs"
    assert json.loads(path.read_text()) == result
    assert "mock-credential" not in path.read_text()
    assert all(
        req.headers["Authorization"] == "Bearer mock-credential"
        for req in endpoint.requests
        if req.url.path.startswith("/v1/")
    )
    assert all(b["model"] == "test-model" for b in endpoint.chat)
    data = result["data"]
    suites = [*PERF, "canary"] if suite == "all" else [suite]
    assert list(data["cases"]) == suites
    assert data["metadata"]["suites"] == suites
    assert data["metadata"]["metrics_version"] == 2
    assert data["metadata"]["client_location"] == "local"
    assert data["metadata"]["server"] == dict(
        model="test-model",
        gpu_name="mock",
        max_context=4096,
        prefix_cache_slots=3,
        use_cuda_graph=True,
        use_speculative=True,
        speculative_active=True,
    )
    assert data["metadata"]["canary_passed"] == (True if "canary" in suites else None)
    if "short" in suites:
        short = [
            b["messages"][0]["content"]
            for b in endpoint.chat
            if b["messages"][0]["content"].startswith("[bench-short")
        ]
        assert [s.split()[0] for s in short] == [
            f"[bench-short-{case_id}-{phase}]"
            for case_id in data["cases"]["short"]
            for phase in ("w00", "r01", "r02")
        ]
        for case in data["cases"]["short"].values():
            assert [r["round"] for r in case["samples"]] == [1, 2]
            assert case["summary"]["prefill_ms"] == {"median": 20, "mean": 20, "n": 2}
        assert data["suite_summary"]["short"]["prefill_ms"]["n"] == 6
    if "agent" in suites:
        agent_reqs = [
            b for b in endpoint.chat if b["messages"][0]["content"].startswith("[bench-agent")
        ]
        assert len(agent_reqs) == 12
        tags = [
            f"[bench-agent-{case_id}-{phase}] "
            for case_id in data["cases"]["agent"]
            for phase in ("w00", "r01", "r02")
        ]
        for b, tag in zip(agent_reqs, tags, strict=True):
            assert b["messages"][0]["content"].startswith(tag)
            assert b["tools"][0]["function"]["description"].startswith(tag)
        for case in data["cases"]["agent"].values():
            assert [r["round"] for r in case["samples"]] == [1, 2]
            assert case["summary"]["prefill_ms"] == {"median": 20, "mean": 20, "n": 2}
        assert data["suite_summary"]["agent"]["prefill_ms"]["n"] == 8
    if "canary" in suites:
        cases = data["cases"]["canary"]
        assert len(cases) == 6 and all(case["passed"] for case in cases.values())
        assert cases["tool_call_multi"]["samples"][0]["tool_call_count"] == 2
        assert sum(len(c["samples"]) for c in cases.values()) == 7
        stream, plain = cases["utf8_stream_parity"]["samples"]
        assert stream["metrics"]["chunk_count"] == 2
        m = plain["metrics"]
        assert m["client_ttft_ms"] == m["client_e2e_ms"] and m["client_itl_p50_ms"] is None
        assert m["server_tpot_ms"] == 10 and m["server_decode_tok_s"] == 175
        assert m["cached_prefix_ratio"] == 0.3 and m["spec_accepted"] == 3
        assert m["http_overhead_ms"] == max(0, m["client_e2e_ms"] - 60)


@pytest.mark.parametrize(("external", "off"), [(True, False), (False, True)])
def test_client_fallback(harness, external, off):
    endpoint, invoke, repo = harness
    endpoint.external = external
    (repo / ".dev.toml").write_text(
        '[remote]\nenabled = false\n[server]\nbind="custom.test:1234"\n'
    )
    result = invoke(
        "run",
        "--json",
        "--suite",
        "short",
        "--model",
        "explicit" if external else "test-model",
        "--telemetry",
        "off" if off else "auto",
        "--rounds",
        "1",
        "--warmup",
        "0",
        "--no-archive",
    )
    assert not (repo / ".runs").exists()
    assert endpoint.peak == 1
    assert all(
        req.url.host == "custom.test" and req.url.path != "/v1/models" for req in endpoint.requests
    )
    assert (result["data"]["metadata"]["server"] is None) == external
    report = result["data"]["cases"]["short"]["short_chat"]
    for row in report["samples"]:
        m = row["metrics"]
        assert m["telemetry_source"] == "client" and "server_tpot_ms" not in m
        assert 0 <= m["client_ttft_ms"] <= m["client_e2e_ms"]
        assert m["client_itl_p95_ms"] >= m["client_itl_p50_ms"] >= 0
        if not off:
            assert (m["prompt_tokens"], m["completion_tokens"], m["cached_tokens"]) == (80, 8, 24)
    for b in endpoint.chat:
        assert ("stream_options" in b) != off
    for req in endpoint.requests:
        if req.method == "POST":
            assert ("X-Flint-Telemetry" in req.headers) != off


@pytest.mark.parametrize(
    ("fault", "case", "error"),
    [
        ("math", "math_reasoning", "missing '391'"),
        ("missing", "tool_call_single", "too few tool calls"),
        ("arguments", "tool_call_multi", "arguments must be a JSON object"),
        ("stop", "stop_sequence", "forbidden '5'"),
        ("parity", "utf8_stream_parity", "output mismatch"),
        ("replacement", "code_bugfix", "U+FFFD"),
        ("truncated", "math_reasoning", "before [DONE]"),
    ],
)
def test_canary_gate(harness, fault, case, error):
    endpoint, invoke, _ = harness
    endpoint.fault = fault
    result = invoke("run", "--suite", "canary", "--json", code=1)
    assert result["data"]["metadata"]["canary_passed"] is False
    report = result["data"]["cases"]["canary"][case]
    assert case in result["error"] and not report["passed"]
    assert error in " ".join(report["failures"])
    assert json.loads(Path(result["run_file"]).read_text())["ok"] is False


def test_model_typo(harness):
    endpoint, invoke, _ = harness
    result = invoke("run", "--json", "--model", "tset-model", code=1)
    assert (
        "Requested model 'tset-model' does not match server model 'test-model'" in result["error"]
    )
    assert not endpoint.chat


def test_compare(harness):
    endpoint, invoke, repo = harness
    base = invoke("run", "--json", "--suite", "short", "--rounds", "2")
    endpoint.model = "other"
    candidate = invoke("run", "--json", "--suite", "short", "--rounds", "1", "--model", "other")
    # JSON stdout is also a supported input. Give each case a different ratio.
    for i, case in enumerate(candidate["data"]["cases"]["short"].values()):
        case["summary"]["prefill_ms"]["median"] = 20 * (2**i)
    path = repo / "candidate.json"
    path.write_text(json.dumps(candidate))
    # Earlier archives have no metric version. Exercise a mismatched client location too.
    del base["data"]["metadata"]["metrics_version"]
    base["data"]["metadata"]["client_location"] = "remote_loopback"
    older = repo / "baseline.json"
    older.write_text(json.dumps(base))
    args = ["compare", str(older), str(path)]
    result = invoke("--json", *args)
    assert "run_file" not in result
    data = result["data"]
    assert any("model:" in w for w in data["warnings"])
    assert any("rounds:" in w for w in data["warnings"])
    assert "metrics_version: baseline=None, candidate=2" in data["warnings"]
    assert "client_location: baseline='remote_loopback', candidate='local'" in data["warnings"]
    metric = data["metrics"]["prefill_ms"]
    assert sorted(p["ratio"] for p in metric["pairs"]) == [1, 2, 4]
    assert metric["candidate_over_baseline"] == pytest.approx(
        dict(geomean=2, n=3, ci95_low=1, ci95_high=4)
    )
    assert metric["baseline_over_candidate"]["geomean"] == pytest.approx(0.5)
    assert invoke(*args, "--json", "--no-archive")["data"] == data
    human = invoke(*args, human=True)
    assert "WARNING model" in human and "95% CI" in human and "prefill_ms" in human
    assert Path(invoke(*args, "--json", "--archive")["run_file"]).is_file()


def test_timing_and_fragmented_tools(monkeypatch):
    # Role/finish/usage frames must never advance the chunk clock.
    clock = iter([5.0, 5.125, 5.375, 5.5, 6.0])
    monkeypatch.setattr(client, "perf_counter", lambda: next(clock))
    frames = [
        {"choices": [{"delta": d, "finish_reason": reason}]}
        for d, reason in [
            ({"role": "assistant"}, None),
            ({"content": ""}, None),
            (
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "t",
                            "type": "function",
                            "function": {"name": "run", "arguments": "{"},
                        }
                    ]
                },
                None,
            ),
            ({"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}, None),
            ({"content": "ok"}, None),
            ({}, "tool_calls"),
        ]
    ]
    raw = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test",
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=raw)),
        ) as http:
            return await client.chat_completion(http, {}, stream=True, telemetry=False, timeout=5)

    result = asyncio.run(run())
    assert result.tool_calls == [
        {"id": "t", "type": "function", "function": {"name": "run", "arguments": "{}"}}
    ]
    assert result.metrics == pytest.approx(
        dict(
            telemetry_source="client",
            client_ttft_ms=125,
            client_e2e_ms=1000,
            chunk_count=3,
            client_itl_p50_ms=187.5,
            client_itl_p95_ms=243.75,
        )
    )


def test_real_workloads_and_invalid_schema(tmp_path):
    perf = load_workloads(ROOT / "workloads/bench_cases.json", PERF)
    assert [c["id"] for c in perf["short"]] == ["short_chat", "short_debug", "short_review"]
    assert all(c["cold_nonce"] and c["max_tokens"] == 64 for c in perf["short"])
    assert [c["id"] for c in perf["agent"]] == [
        "django__django-11099",
        "pydata__xarray-4687",
        "pallets__flask-5014",
        "psf__requests-1142",
    ]
    assert all(
        c["cold_nonce"] and c["max_tokens"] == 8192 and c["reasoning_effort"] == "max"
        for c in perf["agent"]
    )
    assert all(
        [f["function"]["name"] for f in c["tools"]] == ["bash", "str_replace_editor", "submit"]
        for c in perf["agent"]
    )
    data = load_workloads(ROOT / "workloads/canary_cases.json", ["canary"])
    case = data["canary"][0]
    path = tmp_path / "cases.json"
    for bad in [
        dict(temperature=1),
        dict(messages=[]),
        dict(checks={"unexpected_check": True}),
        dict(checks={"expected_tool_calls": {"names": []}}),
        dict(checks={"regex": "("}),
    ]:
        path.write_text(json.dumps(dict(canary=[dict(case, **bad)])))
        with pytest.raises(ValueError, match="math_reasoning"):
            load_workloads(path, ["canary"])
    path.write_text(json.dumps(dict(canary=[case, case])))
    with pytest.raises(ValueError, match="case id"):
        load_workloads(path, ["canary"])


def test_reasoning_regex_and_zero_ratios():
    checks = dict(require_reasoning=True, regex="^391$", expected_finish_reason="length")
    response = client.Completion(
        content="391", reasoning_content="arithmetic", finish_reason="length"
    )
    assert not evaluate(checks, response)
    response.reasoning_content = ""
    response.content = "392"
    assert len(evaluate(checks, response)) == 2
    assert paired_bootstrap_ci([]) == dict(n=0, geomean=None, ci95_low=None, ci95_high=None)


def test_compare_handles_zero_missing_and_unmatched_conditions(harness):
    _, invoke, repo = harness
    base = invoke("run", "--suite", "all", "--json", "--rounds", "1", "--warmup", "0")
    original = base["run_file"]
    case = base["data"]["cases"]["agent"]["django__django-11099"]
    case["summary"]["cached_prefix_ratio"]["median"] = 0
    del case["summary"]["server_tpot_ms"]
    base["data"]["metadata"]["server"]["gpu_name"] = "different"
    base["data"]["metadata"]["canary_passed"] = False
    path = repo / "candidate.json"
    path.write_text(json.dumps(base))
    data = invoke("compare", original, str(path), "--json")["data"]
    assert any("gpu_name" in w for w in data["warnings"])
    assert any("canary_passed" in w for w in data["warnings"])
    for metric, value in [("cached_prefix_ratio", 0), ("server_tpot_ms", None)]:
        comparison = data["metrics"][metric]
        assert comparison["candidate_over_baseline"]["n"] == 6
        (excluded,) = comparison["excluded"]
        assert excluded["candidate"] == value and excluded["suite"] == "agent"


def test_cli_overrides_and_custom_bench(harness):
    endpoint, invoke, repo = harness
    source = json.loads((repo / "workloads/bench_cases.json").read_text())
    case = source["short"][0]
    case["cold_nonce"] = False
    path = repo / "custom.json"
    path.write_text(json.dumps({"short": [case]}))
    output = invoke(
        "run",
        "--suite",
        "short",
        "--bench-file",
        str(path),
        "--url",
        "http://external.test",
        "--api-key",
        "explicit",
        "--rounds",
        "1",
        "--warmup",
        "0",
        "--no-archive",
        human=True,
    )
    assert "short_chat" in output
    assert endpoint.chat[0]["messages"] == case["messages"]
    assert endpoint.requests[0].url.host == "external.test"
    assert endpoint.requests[0].headers["authorization"] == "Bearer explicit"

    source = json.loads((repo / "workloads/canary_cases.json").read_text())
    math_case = source["canary"][0]
    math_case["checks"]["require_reasoning"] = True
    path.write_text(json.dumps({"canary": [math_case]}))
    data = invoke("--json", "run", "--suite", "canary", "--canary-file", str(path), "--no-archive")[
        "data"
    ]
    assert list(data["cases"]["canary"]) == ["math_reasoning"]
    assert data["metadata"]["canary_passed"]


@pytest.mark.parametrize(("decode_tokens", "decode_ms"), [(0, 0), (1, 20)])
def test_zero_decode_and_empty_stream(decode_tokens, decode_ms):
    async def run():
        frame = {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "x_flint_telemetry": dict(
                prefill_tokens=0,
                cached_prefix_tokens=0,
                decode_tokens=decode_tokens,
                prefill_ms=0,
                decode_ms=decode_ms,
                decode_rounds=0,
                spec_accepted=0,
            ),
        }
        raw = f"data: {json.dumps(frame)}\n\ndata: [DONE]\n\n"
        async with httpx.AsyncClient(
            base_url="http://test",
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=raw)),
        ) as http:
            return await client.chat_completion(http, {}, stream=True, telemetry=True, timeout=5)

    m = asyncio.run(run()).metrics
    assert m["cached_prefix_ratio"] == 0
    assert m["chunk_count"] == 0 and m["client_ttft_ms"] is None
    assert "server_tpot_ms" not in m and "server_decode_tok_s" not in m


def test_remote_dispatch_and_local_overrides(harness, monkeypatch, tmp_path):
    endpoint, invoke, repo = harness
    (repo / ".dev.toml").write_text(
        '[remote]\nenabled = true\nhost = "gpu-5080"\n'
        '[server]\nbind = "0.0.0.0:9090"\n'
        '[bench]\npython = "/remote/bench/python"\n'
    )
    calls = []

    # Substitute the SSH transport only. The real Executor streams to and from the child.
    def local_transport(self, cmd, cwd):
        argv = shlex.split(cmd)
        calls.append(("run", argv))
        return [
            sys.executable,
            "-c",
            """
import json, sys
assert "--api-key-stdin" in sys.argv and sys.stdin.read() == "secret-token\\n"
failed = "canary" in sys.argv
data = {"metadata": {"client_location": "local", "suites": ["short"]}}
error = "Canary failed: math_reasoning" if failed else None
print(json.dumps({"ok": not failed, "data": data, "error": error}))
sys.exit(int(failed))
""",
            *argv,
        ]

    monkeypatch.setattr(Executor, "_args", local_transport)
    monkeypatch.setattr(Executor, "sync", lambda self: calls.append(("sync", self.cfg.remote.host)))
    monkeypatch.setenv("FLINT_API_KEY", "secret-token")
    res = invoke("run", "--suite", "short", "--rounds", "2", "--warmup", "1", "--json")
    assert res["ok"] and Path(res["run_file"]).parent == repo / ".runs"
    assert calls[0] == ("sync", "gpu-5080")
    _, argv = calls[1]
    assert argv[:9] == [
        "/remote/bench/python",
        "-m",
        "tools.bench_cli",
        "--json",
        "run",
        "--local",
        "--no-archive",
        "--url",
        "http://127.0.0.1:9090",
    ]
    assert "--api-key-stdin" in argv and "secret-token" not in argv
    assert res["data"]["metadata"]["client_location"] == "remote_loopback"

    # Canary failure propagates remote exit code and diagnostics cleanly.
    failed = invoke("run", "--suite", "canary", "--json", code=1)
    assert not failed["ok"] and "math_reasoning" in failed["error"]

    # Explicit --local or --url bypasses remote dispatch; --api-key-stdin reads stdin.
    calls.clear()
    monkeypatch.delenv("FLINT_API_KEY")
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("stdin-secret\n"))
    invoke(
        "run",
        "--suite",
        "short",
        "--local",
        "--api-key-stdin",
        "--rounds",
        "1",
        "--warmup",
        "0",
        "--no-archive",
        "--json",
    )
    assert calls == []
    assert endpoint.requests[0].headers["authorization"] == "Bearer stdin-secret"

    # External workload file is rejected in remote mode.
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    err = invoke("run", "--suite", "short", "--bench-file", str(outside), "--json", code=1)
    assert "inside the repository" in err["error"]


def test_tool_call_parity_uses_payload(harness):
    _, invoke, repo = harness
    source = json.loads((repo / "workloads/canary_cases.json").read_text())
    case = next(c for c in source["canary"] if c["id"] == "tool_call_single")
    case["checks"]["check_stream_parity"] = True
    path = repo / "parity.json"
    path.write_text(json.dumps({"canary": [case]}))
    result = invoke("run", "--json", "--suite", "canary", "--canary-file", str(path))
    assert result["data"]["metadata"]["canary_passed"]

"""Execute suites, archive their measured samples and compare matched cases."""

import asyncio
import hashlib
import json
import shlex
from urllib.parse import urlsplit

import httpx

from tools.common.config import load_config
from tools.common.remote import Executor

from . import client
from .stats import paired_bootstrap_ci, summarize
from .workloads import PERF, evaluate, load_workloads, requests_for


async def performance(send, suite, case, args):
    rows = []
    width = max(2, len(str(max(args.warmup, args.rounds))))
    for i in range(-args.warmup, args.rounds):
        warmup = i < 0
        number = i + args.warmup if warmup else i + 1
        bodies = requests_for(suite, case, number, warmup, width)
        responses = [await send(body) for body in bodies]
        if not warmup:
            for response in responses:
                rows.append(dict(round=number, metrics=response.metrics))
    return dict(samples=rows, summary=summarize(r["metrics"] for r in rows))


async def canary(send, case):
    checks = case["checks"]
    (body,) = requests_for("canary", case, 1, False)
    samples, failures = [], []
    outputs = []
    for stream in [True, False] if checks.get("check_stream_parity") else [True]:
        try:
            response = await send(body, stream=stream)
            failures.extend(evaluate(checks, response))
            outputs.append((response.content, response.reasoning_content, response.tool_calls))
            samples.append(
                dict(
                    stream=stream,
                    metrics=response.metrics,
                    finish_reason=response.finish_reason,
                    tool_call_count=len(response.tool_calls),
                )
            )
        except (httpx.HTTPError, ValueError, TimeoutError, KeyError, TypeError, IndexError) as exc:
            failures.append(f"stream={stream}: {type(exc).__name__}: {exc}")
    if len(outputs) == 2 and outputs[0] != outputs[1]:
        failures.append("stream/non-stream output mismatch")
    return dict(
        samples=samples,
        passed=not failures,
        failures=failures,
        summary=summarize(s["metrics"] for s in samples),
    )


def _default_url(bind):
    parts = urlsplit("//" + bind)
    if parts.hostname in {"0.0.0.0", "::"} and parts.port:
        return f"http://127.0.0.1:{parts.port}"
    return "http://" + bind


def _remote_workload_path(path, root, flag):
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Remote bench run requires {flag} inside the repository (or pass --local)"
        ) from exc


def run_remote(cfg, args, root, result):
    bench_rel = _remote_workload_path(args.bench_file, root, "--bench-file")
    canary_rel = _remote_workload_path(args.canary_file, root, "--canary-file")
    python = args.python or cfg.bench.python or "python3"
    cmd = [
        python,
        "-m",
        "tools.bench_cli",
        "--json",
        "run",
        "--local",
        "--no-archive",
        "--url",
        _default_url(cfg.server.bind),
        "--suite",
        args.suite,
        "--rounds",
        str(args.rounds),
        "--warmup",
        str(args.warmup),
        "--telemetry",
        args.telemetry,
        "--timeout",
        str(args.timeout),
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    if bench_rel:
        cmd.extend(["--bench-file", bench_rel])
    if canary_rel:
        cmd.extend(["--canary-file", canary_rel])
    input_text = None
    if args.api_key:
        cmd.append("--api-key-stdin")
        input_text = args.api_key + "\n"
    executor = Executor(cfg)
    executor.sync()
    reply = executor.run(
        shlex.join(cmd), input_text=input_text, stream_stderr=True, echo_stdout=False
    )
    try:
        remote = json.loads(reply.stdout)
    except json.JSONDecodeError as exc:
        detail = reply.stderr.strip() or f"Remote bench failed (exit {reply.returncode})"
        raise RuntimeError(detail) from exc
    if not isinstance(remote, dict) or not {"ok", "data"} <= remote.keys():
        raise RuntimeError("Unexpected remote bench output")
    result.ok = bool(remote["ok"])
    result.error = remote.get("error")
    result.data.update(remote["data"])


def dispatch_run(args, root, result):
    cfg = load_config(repo_root=root, allow_missing=True)
    if cfg.is_remote and not args.local and not args.url:
        run_remote(cfg, args, root, result)
    else:
        asyncio.run(run(args, root, result, cfg=cfg))


async def run(args, root, result, *, cfg=None):
    suites = (*PERF, "canary") if args.suite == "all" else (args.suite,)
    workloads = {}
    for filename, selected in [
        (args.bench_file or root / "workloads/bench_cases.json", [s for s in suites if s in PERF]),
        (
            args.canary_file or root / "workloads/canary_cases.json",
            [s for s in suites if s == "canary"],
        ),
    ]:
        if selected:
            workloads.update(load_workloads(filename, selected))
    cfg = cfg or load_config(repo_root=root, allow_missing=True)
    url = (args.url or _default_url(cfg.server.bind)).rstrip("/")
    model = args.model or await client.discover_model(url, args.api_key, args.timeout)
    server = await client.probe_ready(url, args.timeout)
    hashes = {
        s: hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
        for s, cases in workloads.items()
    }
    meta = dict(
        model=model,
        suites=list(suites),
        rounds=args.rounds,
        warmup=args.warmup,
        telemetry=args.telemetry,
        server=server,
        workload_sha256=hashes,
        canary_passed=None,
    )
    results = {}
    result.data.update(metadata=meta, cases=results, suite_summary={})
    async with httpx.AsyncClient(
        base_url=url,
        headers=client.auth(args.api_key),
        timeout=args.timeout,
        limits=httpx.Limits(max_connections=None),
    ) as http:

        async def send(body, stream=True):
            return await client.chat_completion(
                http,
                dict(body, model=model),
                stream=stream,
                telemetry=args.telemetry == "auto",
                timeout=args.timeout,
            )

        failed = []
        for suite, cases in workloads.items():
            results[suite] = {}
            for case in cases:
                if suite == "canary":
                    report = await canary(send, case)
                    if not report["passed"]:
                        failed.append(case["id"])
                else:
                    report = await performance(send, suite, case, args)
                results[suite][case["id"]] = report
            result.data["suite_summary"][suite] = summarize(
                row["metrics"] for case in results[suite].values() for row in case["samples"]
            )
            if suite == "canary":
                meta["canary_passed"] = not failed
                if failed:
                    result.ok, result.error = False, "Canary failed: " + ", ".join(failed)


METRICS = (
    "prefill_ms",
    "client_ttft_ms",
    "server_tpot_ms",
    "server_decode_tok_s",
    "client_e2e_ms",
    "cached_prefix_ratio",
)


def read_run(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("command") != "bench run" or not {"metadata", "cases"} <= record["data"].keys():
        raise ValueError(f"{path} is not a bench run record")
    return record


def compare(args, result):
    baseline, candidate = read_run(args.baseline), read_run(args.candidate)
    warnings = []
    for label, record in (("baseline", baseline), ("candidate", candidate)):
        if not record["ok"]:
            warnings.append(f"{label} run failed: {record.get('error')}")
    left, right = baseline["data"], candidate["data"]
    a, b = left["metadata"], right["metadata"]
    for field in (
        "model",
        "suites",
        "rounds",
        "warmup",
        "telemetry",
        "canary_passed",
        "workload_sha256",
        "server",
    ):
        if a.get(field) != b.get(field):
            warnings.append(f"{field}: baseline={a.get(field)!r}, candidate={b.get(field)!r}")

    def medians(run):
        return {
            (s, k): case["summary"]
            for s, cases in run["cases"].items()
            if s in PERF
            for k, case in cases.items()
        }

    base, cand = medians(left), medians(right)
    if base.keys() != cand.keys():
        warnings.append(f"unpaired cases: {sorted(base.keys() ^ cand.keys())}")
    comparisons = {}
    for metric in METRICS:
        pairs, excluded = [], []
        for suite, case_id in sorted(base.keys() & cand.keys()):
            key = suite, case_id
            x = base[key].get(metric, {}).get("median")
            y = cand[key].get(metric, {}).get("median")
            if x is None or y is None or min(x, y) <= 0:
                excluded.append(dict(suite=suite, case_id=case_id, baseline=x, candidate=y))
                continue
            pairs.append(dict(suite=suite, case_id=case_id, baseline=x, candidate=y, ratio=y / x))
        ratios = [pair["ratio"] for pair in pairs]
        comparisons[metric] = dict(
            candidate_over_baseline=paired_bootstrap_ci(ratios),
            baseline_over_candidate=paired_bootstrap_ci([1 / r for r in ratios]),
            pairs=pairs,
            excluded=excluded,
        )
    result.data.update(warnings=warnings, metrics=comparisons)


def show_median(summary, key):
    value = summary.get(key, {}).get("median")
    return f"{value:.2f}" if value is not None else "-"


def human_format(result):
    lines = []
    if result.command == "bench compare":
        lines.extend("WARNING " + w for w in result.data.get("warnings", []))
        lines.append(
            f"{'metric':22} {'pairs':>5}  {'cand/base':>10}  {'base/cand':>10}  95% CI (cand/base)"
        )
        for name, report in result.data.get("metrics", {}).items():
            ci = report["candidate_over_baseline"]
            if ci["n"]:
                lines.append(
                    f"{name:22} {ci['n']:5d}  {ci['geomean']:10.4f}  "
                    f"{report['baseline_over_candidate']['geomean']:10.4f}  "
                    f"[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]"
                )
            else:
                lines.append(f"{name:22}     0  n/a")
    else:
        columns = (
            "prefill_ms",
            "client_ttft_ms",
            "client_e2e_ms",
            "server_tpot_ms",
            "server_decode_tok_s",
            "cached_prefix_ratio",
        )
        lines.append(
            "suite   case                     prefill ms   TTFT ms    E2E ms   TPOT ms  "
            "    tok/s    cached"
        )
        for suite, cases in result.data.get("cases", {}).items():
            for name, case in cases.items():
                fields = " ".join(f"{show_median(case['summary'], k):>9}" for k in columns)
                lines.append(f"{suite:7} {name:24} {fields}")
                if "passed" in case:
                    lines.append(
                        "        PASS"
                        if case["passed"]
                        else "        FAIL " + ", ".join(case["failures"])
                    )
    if result.error:
        lines.append(result.error)
    return "\n".join(lines)

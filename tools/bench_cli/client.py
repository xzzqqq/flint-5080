"""OpenAI chat transport and timings at the HTTP boundary."""

import asyncio
import json
from dataclasses import dataclass, field
from time import perf_counter

import httpx

from .stats import percentile


def auth(api_key):
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


async def discover_model(url, api_key, timeout):
    async with httpx.AsyncClient(timeout=timeout) as client, asyncio.timeout(timeout):
        response = await client.get(url + "/v1/models", headers=auth(api_key))
        response.raise_for_status()
        for entry in response.json().get("data", []):
            if isinstance(entry.get("id"), str) and entry["id"]:
                return entry["id"]
    raise ValueError("No model ID returned by /v1/models")


async def probe_ready(url, timeout):
    fields = (
        "gpu_name",
        "max_context",
        "prefix_cache_slots",
        "use_cuda_graph",
        "prefill_chunk_size",
        "mode",
        "abi_version",
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client, asyncio.timeout(timeout):
            response = await client.get(url + "/ready")
            response.raise_for_status()
            body = response.json()
            if isinstance(body, dict):
                engine = body.get("engine")
                server = (
                    {k: engine[k] for k in fields if k in engine}
                    if isinstance(engine, dict)
                    else {}
                )
                if isinstance(body.get("model"), str) and body["model"]:
                    server["model"] = body["model"]
                return server or None
    except (httpx.HTTPError, TimeoutError, ValueError):
        pass
    return None


@dataclass
class Completion:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list = field(default_factory=list)
    finish_reason: str | None = None
    metrics: dict = field(default_factory=dict)


def record_usage(metrics, frame, telemetry):
    usage = frame.get("usage") or {}
    for key in ("prompt_tokens", "completion_tokens"):
        if key in usage:
            metrics[key] = usage[key]
    details = usage.get("prompt_tokens_details") or {}
    if "cached_tokens" in details:
        metrics["cached_tokens"] = details["cached_tokens"]
    t = frame.get("x_flint_telemetry")
    if telemetry and t:
        for key in (
            "prefill_tokens",
            "cached_prefix_tokens",
            "decode_tokens",
            "prefill_ms",
            "decode_ms",
            "decode_rounds",
            "spec_accepted",
        ):
            metrics[key] = t[key]
        if "queue_wait_ms" in t:
            metrics["queue_wait_ms"] = t["queue_wait_ms"]
        metrics["telemetry_source"] = "server"
        if t["decode_rounds"] > 0:
            metrics["server_tpot_ms"] = t["decode_ms"] / t["decode_rounds"]
        decode_generated = max(0, t["decode_tokens"] - 1)
        if t["decode_ms"] > 0 and decode_generated > 0:
            metrics["server_decode_tok_s"] = decode_generated / (t["decode_ms"] / 1000)
        metrics["cached_prefix_ratio"] = t["cached_prefix_tokens"] / max(1, t["prefill_tokens"])


def merge_calls(calls, deltas):
    for delta in deltas:
        index = delta["index"]
        call = calls.setdefault(index, {"function": {"name": "", "arguments": ""}})
        for key in ("id", "type"):
            if delta.get(key):
                call[key] = delta[key]
        for key in ("name", "arguments"):
            call["function"][key] += (delta.get("function") or {}).get(key) or ""


async def read_stream(response, result, times, telemetry):
    calls = {}
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            result.tool_calls = [calls[i] for i in sorted(calls)]
            return
        frame = json.loads(data)
        if "error" in frame:
            raise ValueError(f"Chat stream error: {frame['error']}")
        record_usage(result.metrics, frame, telemetry)
        for choice in frame["choices"]:
            delta = choice.get("delta") or {}
            if any(delta.get(k) for k in ("content", "reasoning_content", "tool_calls")):
                times.append(perf_counter())
                result.content += delta.get("content") or ""
                result.reasoning_content += delta.get("reasoning_content") or ""
                merge_calls(calls, delta.get("tool_calls") or [])
            if choice.get("finish_reason"):
                result.finish_reason = choice["finish_reason"]
    raise ValueError("Chat stream ended before [DONE]")


async def chat_completion(client, request, *, stream, telemetry, timeout):
    body = dict(request, stream=stream)
    headers = {}
    if telemetry:
        headers["X-Flint-Telemetry"] = "1"
        if stream:
            body["stream_options"] = {"include_usage": True}
    result = Completion(metrics={"telemetry_source": "client"})
    times = []
    start = perf_counter()
    async with asyncio.timeout(timeout):
        async with client.stream("POST", "v1/chat/completions", json=body, headers=headers) as resp:
            resp.raise_for_status()
            if stream:
                await read_stream(resp, result, times, telemetry)
            else:
                frame = json.loads(await resp.aread())
                choice = frame["choices"][0]
                message = choice["message"]
                result.content = message.get("content") or ""
                result.reasoning_content = message.get("reasoning_content") or ""
                result.tool_calls = message.get("tool_calls") or []
                result.finish_reason = choice["finish_reason"]
                record_usage(result.metrics, frame, telemetry)
            end = perf_counter()
    if not result.finish_reason:
        raise ValueError("Completion missing finish_reason")
    e2e = (end - start) * 1000
    gaps = [(b - a) * 1000 for a, b in zip(times, times[1:], strict=False)]
    m = result.metrics
    m.update(
        client_e2e_ms=e2e,
        client_ttft_ms=(times[0] - start) * 1000 if times else (None if stream else e2e),
        chunk_count=len(times) if stream else 1,
        client_itl_p50_ms=percentile(gaps, 50) if gaps else None,
        client_itl_p95_ms=percentile(gaps, 95) if gaps else None,
    )
    if m["telemetry_source"] == "server":
        m["http_overhead_ms"] = max(0.0, e2e - (m["prefill_ms"] + m["decode_ms"]))
    return result

"""Fixed workload format, session isolation, and deterministic canary assertions."""

import json
import re
from copy import deepcopy

PERF = ("short", "agent")
OPTIONS = {"temperature", "max_tokens", "tools", "reasoning_effort", "stop"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strings(value):
    return isinstance(value, list) and all(isinstance(s, str) and s for s in value)


def validate_messages(messages):
    require(isinstance(messages, list) and messages, "expected a nonempty message list")
    for m in messages:
        require(
            isinstance(m, dict) and m.get("role") in ("system", "user", "assistant", "tool"),
            "invalid message role",
        )
        require(
            isinstance(m.get("content"), (str, list))
            or (
                m["role"] == "assistant"
                and isinstance(m.get("tool_calls"), list)
                and m["tool_calls"]
            ),
            "invalid content",
        )


def validate_checks(checks):
    require(isinstance(checks, dict) and checks, "canary requires checks")
    for key, value in checks.items():
        if key in ("require_reasoning", "check_stream_parity", "forbid_replacement_char"):
            require(type(value) is bool, f"invalid {key}")
        elif key in ("must_contain", "must_not_contain"):
            require(strings(value), f"invalid {key}")
        elif key == "regex":
            require(isinstance(value, str), "regex must be a pattern")
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"invalid regex: {exc}") from exc
        elif key == "expected_finish_reason":
            require(value in ("stop", "length", "tool_calls"), "invalid expected_finish_reason")
        elif key == "expected_tool_calls":
            require(
                isinstance(value, dict) and strings(value.get("names")) and value["names"],
                "expected_tool_calls requires names",
            )
            require(
                not value.keys() - {"min_calls", "names", "required_keys", "arguments_contain"},
                "unknown tool check",
            )
            minimum = value.get("min_calls", 1)
            require(type(minimum) is int and minimum >= 1, "invalid min_calls")
            for k in ("required_keys", "arguments_contain"):
                require(strings(value.get(k, [])), f"invalid {k}")
        else:
            raise ValueError(f"unknown canary check {key}")


def load_workloads(path, suites):
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), f"{path}: expected suite object")
    selected = {}
    for suite in suites:
        cases = data.get(suite)
        require(isinstance(cases, list) and cases, f"{path}: missing or empty suite {suite}")
        ids = set()
        for case in cases:
            require(isinstance(case, dict), f"{suite}: expected case object")
            name = case.get("id")
            require(isinstance(name, str) and name and name not in ids, f"{suite}: invalid case id")
            ids.add(name)
            try:
                fields = {
                    "short": {"messages", "cold_nonce"},
                    "agent": {"messages", "cold_nonce"},
                    "canary": {"messages", "checks"},
                }
                unknown = case.keys() - OPTIONS - fields[suite] - {"id"}
                require(not unknown, f"unknown case fields: {sorted(unknown)}")
                require(
                    type(case.get("temperature")) in (float, int) and case["temperature"] == 0,
                    "temperature must be zero",
                )
                require(
                    type(case.get("max_tokens")) is int and case["max_tokens"] > 0,
                    "max_tokens must be positive",
                )
                if "stop" in case:
                    require(strings(case["stop"]) and case["stop"], "invalid stop sequences")
                require(isinstance(case.get("tools", []), list), "tools must be a list")
                for tool in case.get("tools", []):
                    require(
                        tool["type"] == "function"
                        and isinstance(tool["function"]["name"], str)
                        and isinstance(tool["function"]["parameters"], dict),
                        "invalid tool",
                    )
                validate_messages(case["messages"])
                require(
                    isinstance(case["messages"][0]["content"], str), "missing first message text"
                )
                if suite == "canary":
                    validate_checks(case.get("checks"))
                if suite in PERF:
                    require(type(case.get("cold_nonce")) is bool, "expected cold_nonce boolean")
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}: {suite}/{name}: {exc}") from exc
        selected[suite] = cases
    return selected


def requests_for(suite, case, number, warmup, width=2):
    """Yield request bodies for a single round, isolating cold runs with a prefix nonce."""
    options = {k: case[k] for k in OPTIONS if k in case}
    phase = "w" if warmup else "r"
    index = f"{number:0{width}d}"
    if suite in PERF and case.get("cold_nonce"):
        tag = f"[bench-{suite}-{case['id']}-{phase}{index}] "
        # Inkling renders tool declarations before the system message. Isolate those too.
        if options.get("tools"):
            options["tools"] = deepcopy(options["tools"])
            fn = options["tools"][0]["function"]
            fn["description"] = tag + fn.get("description", "")
        yield dict(options, messages=tagged(case["messages"], tag))
    else:
        yield dict(options, messages=case["messages"])


def tagged(messages, tag):
    return [dict(messages[0], content=tag + messages[0]["content"]), *messages[1:]]


def invalid_json_constant(value):
    raise ValueError(f"invalid JSON constant {value}")


def evaluate(checks, response):
    failures = []
    content = response.content
    for fragment in checks.get("must_contain", []):
        if fragment not in content:
            failures.append(f"missing {fragment!r}")
    for fragment in checks.get("must_not_contain", []):
        if fragment in content:
            failures.append(f"forbidden {fragment!r}")
    if "regex" in checks and not re.search(checks["regex"], content):
        failures.append("content regex mismatch")
    if checks.get("require_reasoning") and not response.reasoning_content.strip():
        failures.append("missing reasoning_content")
    if checks.get("forbid_replacement_char") and "\ufffd" in content + response.reasoning_content:
        failures.append("replacement character U+FFFD")
    if (
        "expected_finish_reason" in checks
        and checks["expected_finish_reason"] != response.finish_reason
    ):
        failures.append(f"unexpected finish_reason {response.finish_reason!r}")
    spec = checks.get("expected_tool_calls")
    if spec:
        if len(response.tool_calls) < spec.get("min_calls", 1):
            failures.append("too few tool calls")
        for index, call in enumerate(response.tool_calls):
            fn = call.get("function") if isinstance(call, dict) else None
            if not isinstance(fn, dict):
                failures.append(f"tool {index}: missing function")
                continue
            if fn.get("name") not in spec["names"]:
                failures.append(f"tool {index}: unexpected name")
            try:
                args = json.loads(fn["arguments"], parse_constant=invalid_json_constant)
            except (TypeError, ValueError, KeyError):
                args = None
            if not isinstance(args, dict):
                failures.append(f"tool {index}: arguments must be a JSON object")
                continue
            if any(k not in args for k in spec.get("required_keys", [])):
                failures.append(f"tool {index}: missing argument key")
            if any(
                s not in json.dumps(args, ensure_ascii=False)
                for s in spec.get("arguments_contain", [])
            ):
                failures.append(f"tool {index}: missing argument substring")
    return failures

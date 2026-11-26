# HTTP benchmark protocol

Run `./bin/bench run --suite canary --no-archive` for a quick check of a running
endpoint. Run `./bin/bench run --suite all --json` for the full protocol.
`--url` takes the base URL before `/v1` and defaults to the local server bind.
Set `FLINT_API_KEY` for authenticated endpoints. The harness discovers the first
model in `/v1/models` unless given `--model`. It samples static engine metadata
from Flint's `/ready`. All generation goes through OpenAI Chat Completions.

## Fixed synthetic inputs

| Suite | Request schedule | Readout |
| --- | --- | --- |
| short | Serial short chat, medium debugging, long code review. Nominal input sizes ~40/256/1024 tokens, maximum 64 output tokens. | Short-context steady C1 decode and SOL |
| agent | Four frozen SWE-bench Verified coding-agent snapshots spanning Q1–Q4 context lengths (8.5K / 15.1K / 22.4K / 47.1K tokens) across edit, explore and debug stages. `reasoning_effort: "max"`, maximum 8,192 output tokens, natural EOS. | Realistic long-context C1 decode under Full-Attention KV load and DSpark acceptance |
| canary | One greedy pass over arithmetic, code, single/multiple tool calls, stop strings and bilingual UTF-8. The parity case also sends one non-streaming request. | All assertions pass, including exact reconstructed content, reasoning and tool calls for parity |

Inputs and checks live in `workloads/{bench,canary}_cases.json`. All cases use
temperature zero. The `agent` cases are a 4-request stratified subset extracted
from `coding_agent_48` (`django__django-11099`, `pydata__xarray-4687`,
`pallets__flask-5014`, `psf__requests-1142`) with `cache_control` hints stripped.
The multi-city canary validates every call and records the actual count,
accepting one or more valid calls.

Cold requests prefix the first message with `[bench-<suite>-w00] ` during warmup
and `[bench-<suite>-r01] ` in the first measured round. For `agent` cases with
tools, the first tool description carries that tag too because Inkling
serializes tools before messages. Tags repeat across invocations. For official
comparisons start from equivalent, empty server caches. Inspect actual cached
token counts on cold requests: templates may contribute a common prefix. Record
the returned prompt lengths.

## Timing and comparisons

Use one warmup and three measured rounds per performance case by default.
Exclude all warmups from samples, counts and statistics. For each available
numeric metric report median, arithmetic mean and sample count by case and
suite. A suite summary pools its measured requests.

Request `X-Flint-Telemetry: 1` and streaming `stream_options.include_usage`.
Use the returned `x_flint_telemetry` as the primary engine measurement.
TPOT is decode ms per decode **round**. Output rate is decode tokens per decode
second. These can differ under speculative decoding. Report cached tokens over
total prefill tokens and the nonnegative residual of client E2E minus server
prefill and decode time. That residual includes queueing, transport and server
overhead. Exact queue wait requires an explicit server timer.

Time every request with a client monotonic clock. TTFT ends at the first
nonempty content, reasoning or tool-call delta. ITL p50/p95 measure gaps between
those output chunks. Role, finish and usage frames do not count. E2E includes
the terminal response. TTFT is null for an empty stream, ITL is null with fewer
than two chunks. A non-streaming response counts as one chunk and TTFT equals
E2E. Retain standard usage when supplied. With external endpoints or
`--telemetry off`, use `telemetry_source=client` when engine timers are absent.
Server-only metrics and token rates with unknown counts remain unavailable.

Use `./bin/bench compare BASELINE.json CANDIDATE.json` on archived runs or JSON
stdout. Pair **performance case medians** by suite and case ID. Each matched
case contributes one ratio, independent of rounds, slots or session length.
Report the geometric mean of candidate/baseline and its inverse orientation.
For latencies, baseline/candidate is speedup. For throughput,
candidate/baseline is speedup. Resample case pairs with replacement 2,000
times (seed 42) and take linearly interpolated 2.5th and 97.5th percentiles.
This interval reflects variation across the fixed cases. A single pair gives
a point interval. Zero or missing values have no log ratio and appear with
their original values in the metric's exclusions.

Check warnings for model, selected suites, workload hashes, round and warmup
counts, telemetry mode, canary status and available GPU/engine configuration.
Also match weights, precision, client location, cache state, driver, GPU
clocks/power and background load. Record deliberate changes. A comparison with
unmatched conditions is diagnostic.

## Acceptance and evidence

- Require every canary to pass. A failed assertion or request yields a nonzero
  run exit code and a case-level diagnostic.
- M2.6 freezes Phase 2 AR with all suites on the RTX 5080. Record C1 decode
  SOL as `100 × model AR floor_ms / server_tpot_ms` at matched context length
  (short-context floor 3.550 ms at 960 GB/s, see `docs/model.md`). Compare
  Phase 3 speculative output in tokens/s as well as ms/round. Its output rate
  relative to the AR ceiling is an AR-equivalent rate, not measured bandwidth.
- Proposed Phase 3 review thresholds, pending M2.6 repeatability measurements:
  target at least 75% of the context-matched logical bandwidth SOL for optimized
  AR C1 decode. For regression review use 5% on `short` and `agent` decode
  latency against frozen Phase 2. Inspect individual cases. Repeat a matched run when a regression crosses the threshold and the
  interval overlaps it. Confirm sustained changes before promotion. Verify
  numerical accuracy and model quality separately for precision or algorithm changes.
- Keep raw run records under `.runs/`. Add `--archive` to save a comparison as well.
  Publish the frozen baseline and accepted or rejected experiments as dated
  reports in `docs/findings/`, citing run IDs, source revisions, hardware,
  configuration, workload hashes, medians, intervals, SOL and correctness.

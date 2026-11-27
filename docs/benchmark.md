# HTTP benchmark protocol

Run `./bin/bench run --suite canary --no-archive` for a quick check of a running
endpoint. Run `./bin/bench run --suite all --json` for the full protocol.
`--url` takes the base URL before `/v1` and defaults to the local server bind.
When `.dev.toml` enables `[remote]` and `--url`/`--local` are omitted, `bench run`
syncs the workspace, runs on the execution host over loopback using `[bench].python`,
and archives the run record locally under `.runs/`.
Set `FLINT_API_KEY` for authenticated endpoints. The harness discovers the first
model in `/v1/models` unless given `--model`. It checks an explicit model against
the name reported by Flint's `/ready` and samples static engine metadata there.
All generation goes through OpenAI Chat Completions. Repeat `--case ID` to run
specific cases from the selected suites. Use `--tag NAME` to name an archived run.

## Fixed synthetic inputs

| Suite | Request schedule | Readout |
| --- | --- | --- |
| short | Serial short chat, medium debugging, long code review. Nominal input sizes ~40/256/1024 tokens, maximum 64 output tokens. | Short-context steady C1 decode and SOL |
| agent | Four frozen SWE-bench Verified coding-agent snapshots spanning Q1–Q4 context lengths (8.5K / 15.1K / 22.4K / 47.1K tokens) across edit, explore and debug stages. `reasoning_effort: "max"`, maximum 8,192 output tokens, natural EOS. | Realistic long-context C1 decode under Full-Attention KV load and DFlash 2 acceptance |
| canary | One greedy pass over arithmetic, code, single/multiple tool calls, stop strings and bilingual UTF-8. The parity case also sends one non-streaming request. | All assertions pass, including exact reconstructed content, reasoning and tool-call payloads for parity, excluding request-specific IDs |

Inputs and checks live in `workloads/{bench,canary}_cases.json`. All cases use
temperature zero. The `agent` cases are a 4-request stratified subset extracted
from `coding_agent_48` (`django__django-11099`, `pydata__xarray-4687`,
`pallets__flask-5014`, `psf__requests-1142`) with `cache_control` hints stripped.
The multi-city canary validates every call and records the actual count,
accepting one or more valid calls.

Cold requests prefix the first message with `[bench-<suite>-<case-id>-w00] `
during warmup and `[bench-<suite>-<case-id>-r01] ` in the first measured round.
For cases with tools, the first tool description carries the same tag because
Inkling serializes tools before messages. The case ID isolates cases sharing
tool definitions and system prompts. Tags repeat across invocations. For
official comparisons start from equivalent, empty server caches. Inspect actual cached
token counts on cold requests: templates may contribute a common prefix. Record
the returned prompt lengths.

## Timing and comparisons

Use one warmup and three measured rounds per performance case by default.
Exclude all warmups from samples, counts and statistics. For each available
numeric metric report median, arithmetic mean and sample count by case and
suite. A suite summary pools its measured requests.

Request `X-Flint-Telemetry: 1` and streaming `stream_options.include_usage`.
Use the returned `x_flint_telemetry` as the primary engine measurement.
TPOT is decode ms per decode **round**. With `metrics_version=2`,
`server_decode_tok_s` is
`(decode_tokens - 1) / (decode_ms / 1000)`. It counts tokens emitted during
decode rounds, excluding the first token generated during prefill. The rate is
unavailable when no such tokens were emitted or decode time is zero. For positive decode rounds, `tok/rnd` reports decode-generated tokens per round
and `acc` reports accepted speculative tokens per round. Report cached tokens over
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
stdout. Either argument also accepts `@tag`, `latest` or `prev`. Append a suite
to a recency alias (for example, `prev:short`) to count only runs containing it.
Pair **performance case medians** by suite and case ID. Each matched
case contributes one ratio, independent of rounds, slots or session length.
Report the geometric mean of candidate/baseline and its inverse orientation.
For latencies, baseline/candidate is speedup. For throughput,
candidate/baseline is speedup. Resample case pairs with replacement 2,000
times (seed 42) and take linearly interpolated 2.5th and 97.5th percentiles.
This interval reflects variation across the fixed cases. A single pair gives
a point interval. Zero or missing values have no log ratio and appear with
their original values in the metric's exclusions.

Check warnings for model, selected suites, case filters, workload hashes, round and warmup
counts, telemetry mode, canary status and available GPU/engine configuration.
`bench compare` also warns on mismatched `metrics_version` or `client_location`
(`local` or `remote_loopback`), including fields missing from older archives.
Also match weights, precision, cache state, driver, GPU
clocks/power and background load. Record deliberate changes. A comparison with
unmatched conditions is diagnostic.

## Acceptance and evidence

- Require every canary to pass. A failed assertion or request yields a nonzero
  run exit code and a case-level diagnostic.
- Report speculative decoding in `server_decode_tok_s`, `ms/round`
  (`server_tpot_ms`), `tokens/round`, `spec_accepted`, and context-matched
  DFlash 2 round SOL (`100 × draft_round_floor_ms / server_tpot_ms`, accounting
  for draft + verify unique-expert weight and KV traffic at 960 GB/s; see
  `docs/model.md`). AR (`--no-spec`) context-matched SOL (`3.487 ms`
  short-context weight floor) may be recorded alongside as a diagnostic
  reference.
- Freeze the greedy DFlash 2 baseline with one warmup and three cold-nonce requests
  per case. Record accepted drafts/round as well as emitted tokens/round. A separate
  temperature-1.0 streaming smoke uses the short prompts, seed 5080, and the
  server telemetry header. Seed 0 asks the engine to choose a fresh seed. Keep these single-request samples outside greedy medians.
  The verified 5080 baseline uses 31,508 tokens of context and three prefix slots.
  [Baseline measurements and run IDs](findings/2026-11-27-dflash2-pivot-baseline.md)
- Optimization and regression thresholds: target at least **70% of
  context-matched DFlash 2 round SOL** to maximize end-to-end
  `server_decode_tok_s`. Use a 5% regression review threshold on `short` and
  `agent` decode latency/throughput against the active frozen baseline
  (clean-restart repeatability over `remote_loopback` is verified below 0.1%).
  Inspect individual cases and repeat matched runs when a regression crosses
  the threshold and the interval overlaps it. Confirm sustained changes before
  promotion.
- Whenever an optimization alters arithmetic order, precision (such as KV
  compression), or algorithms, verify numerical accuracy via `./bin/weights kld`
  (teacher-forced KLD and top-1 agreement) alongside `canary` and
  `dev check --gpu`, and evaluate model quality where precision is traded.
- Keep raw run records under `.runs/`. Add `--archive` to save a comparison as
  well. Publish frozen baselines and accepted or rejected experiments as dated
  reports in `docs/findings/`, citing run IDs, source revisions, hardware,
  configuration, workload hashes, medians, intervals, SOL, and numerical checks.

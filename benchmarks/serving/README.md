# Serving measurements and client tasks

These tools contain original, handwritten Python code and fixtures.
They use Python standard library APIs. They add no runtime dependency.
HTTP clients accept loopback endpoints only.

## Clef HTTP acceptance

`clef_acceptance.py` checks an existing `python-clef` service with the 4096-token profile.
It starts no processes and imports no model libraries.
The caller owns service startup, shutdown, the accelerator lock, and external pressure monitoring.

Prepare an independent registration before the HTTP run.
Use schema `apxinf-clef-acceptance-registration-v1`.
The registration contains exactly `schema`, `model`, `model_revision`, `capability_revision`, and `encodings`.
Pin both revisions to the approved worker identity.
The `encodings` object contains exactly `outage`, `healthy`, `boundary`, and `overflow`.
Each entry contains exactly `request_sha256`, `input_tokens`, `input_ids_sha256`, and `source_receipt_sha256`.
Every digest uses lowercase SHA-256 hexadecimal text.
Every count uses a positive JSON integer.
The boundary and overflow counts must equal 4096 and 4097, respectively.

The maintained `cases(model)` function defines the original requests and their serial order.
Encode each distinct valid request independently with the approved tokenizer and public encoder.
Retain the input ID arrays and their source receipt outside the registration.
Hash canonical request bytes with UTF-8, sorted keys, `ensure_ascii=False`, and separators `(',', ':')`.
Hash the complete input ID array as ASCII JSON with separators `(',', ':')`.
Hash the retained receipt bytes for `source_receipt_sha256`.
Add no domain prefix or final newline to the request or array bytes.
The registration binds requests but cannot prove historical independence or agreement between counts and retained arrays.

```sh
python3 benchmarks/serving/clef_acceptance.py \
  --base-url http://127.0.0.1:8080 \
  --registration /absolute/path/to/registration.json \
  --output-dir devlocal/clef-acceptance/run-001
```

Use a new output directory for each run.
`--request-timeout` bounds each HTTP exchange and defaults to 120 seconds.
`--settlement-timeout` bounds each joint observation and defaults to 180 seconds.
Settlement GET requests use the smaller remaining budget.
Bodies have a 262144-byte limit. Python's HTTP parser also bounds header lines and header counts.
The client accepts numeric HTTP loopback addresses without DNS resolution.

The nine cases cover invalid streaming, an unknown alias, an image field, outage, healthy state, repetition, context boundary, overflow, and recovery.
The tool checks all question projections, independent input counts, pinned identities, exact outcome deltas, worker replacement, and resource settlement.
Score projection checks permit absolute error of at most `1e-12`.
It stops after the first failure and never retries a POST.
`report.json` uses atomic checkpoints and retains bounded response bodies in adjacent files.
`post_attempts` counts dispatch attempts, which might not reach the service.
Each case records its dispatch state, including uncertain transport failures.

A successful run sets `passed` to `true` and exits with status zero.
Default qualification requires observed `macos` pressure policy with normal, fresh samples.
For an explicitly disabled policy, add `--diagnostic-disabled-pressure`.
A diagnostic pass always leaves `default_pressure_qualified` set to `false`.
Routing quality and repeated answer equality remain observations.
The tool does not check independent logits, hidden `noul` probabilities, process reaping, or generation performance.

## Offline memory calibration

`memory_calibration.py` records raw observations from the pinned local MLX adapter.
The [calibration profile](../../doc/serving/memory-calibration-v0.1.md) defines the artifact and its acceptance boundaries.
This command owns the existing Metal lock and starts one isolated model process.
The process requires normal macOS pressure before loading and at execution checkpoints.
Unknown pressure blocks execution.

Run a small explicit shape subset first:

```sh
python3 benchmarks/serving/memory_calibration.py \
  --model /absolute/path/to/approved/model \
  --input-lengths 1,255,256,257 --max-tokens 8 --repetitions 1 \
  --output-dir /tmp/apxinf-memory-subset
```

Omitting `--input-lengths` selects the default chunk, cache growth, and context boundaries.
The default context is 16384 tokens, with a 32-token generation allowance and two repetitions.
Additional cases stop during prefill, stop after output, or request zero output.
Every case remains in the report even when an earlier case fails or pressure blocks loading.
`--timeout` bounds the complete probe process, including model loading.

Use a new directory under `devlocal/<feature>/` or outside the checkout.
`calibration.json` contains identity, profiles, case results, and journal references.
`samples.jsonl` retains flushed observations, including partial evidence from interrupted runs.
`process.json` records the parent outcome, timeout, and reaping evidence.
An unconfirmed process exit leaves the probe report incomplete, even if its last case completed.
The parent never overwrites a report while its probe can still write it.

The tool separates allocator counters, cache payload, process peak RSS, and system swap.
Synchronized samples can change timing and allocation overlap.
They do not establish serving latency, model quality, or an approved admission budget.

Use an explicit plan for mixed output allowances and context boundaries:

```sh
python3 benchmarks/serving/memory_calibration.py \
  --model /absolute/path/to/approved/model \
  --plan-file benchmarks/serving/plans/01-smoke.json \
  --seed-messages benchmarks/serving/plans/seed-primary.json \
  --output-dir /tmp/apxinf-memory-explicit
```

Do not combine `--plan-file` with `--input-lengths`, `--max-tokens`, or `--repetitions`.
The parent freezes the complete plan and seed messages before it starts the probe.
The probe checks that snapshot and never rereads the external source files.
Both successful and blocked reports retain the frozen input identity.

The [staged evidence plan](../../doc/serving/memory-evidence-plan-20261010.md) defines boundary, repetition, stopping, and independent input cases.
Its fixtures define planned work and contain no measurement results.
The [cache geometry analysis](../../doc/serving/qwen35-memory-geometry-20261010.md) explains the selected allocation boundaries.
Those static calculations do not establish complete allocation peaks.
The [measurement record](../../doc/serving/memory-evidence-20261010.md) separates completed short requests from blocked stages and unreached outputs.
The [production validation profile](../../doc/serving/production-memory-validation-v0.1.md) defines checks without additional device sampling synchronization.

## Offline memory coverage

`memory_coverage.py` checks an existing calibration directory without loading MLX.
The [coverage profile](../../doc/serving/memory-coverage-v0.1.md) defines validation and reached targets.

```sh
python3 benchmarks/serving/memory_coverage.py \
  /tmp/apxinf-memory-subset --output /tmp/apxinf-memory-coverage.json
```

The output path must not exist.
The report separates planned cases, executed cases, generation observations, and reached targets.
Declared frozen inputs require their retained snapshot, matching digests, and consistent plan, profile, messages, and template options.
Legacy artifacts report `frozen_input_validation=unavailable` and cannot establish held-out input provenance.
It preserves early EOS, stopping, cleanup, physical cache offsets, and missing shape intervals.
Peak ownership uses the run identity and reset epoch.
Inherited peak observations do not establish a new case peak.

Exit status 0 means the recorded evidence passes consistency checks.
It does not establish complete coverage or approve an admission budget.
Invalid evidence produces exit status 1 without derived coverage claims.
The analyzer does not change source artifacts or service configuration.

## Production memory observations

`production_memory_validation.py` attaches to an existing local service and sends ordered streaming Messages requests.
The [production profile](../../doc/serving/production-memory-validation-v0.1.md#initial-http-runner) defines its plan, checks, and evidence files.
The runner adds no device synchronization or worker inspection calls.
It records external process RSS, host observations, raw service metrics, response bytes, and matching diagnostic records.
Unavailable allocator counters and request cache payload remain unknown.

Prepare the plan before execution.
Use a new directory under `devlocal/<feature>/` or outside the checkout.
The service launcher must hold the existing Metal lock during loading, requests, and cleanup.
Pass its locked descriptor to the runner through `subprocess.run(..., pass_fds=(descriptor,))` or an equivalent parent interface.
The runner checks the descriptor's inode and exclusive ownership.
It rejects a Boolean ownership declaration or a separately opened descriptor.

```sh
python3 benchmarks/serving/production_memory_validation.py \
  --plan-file devlocal/serving-upstream/production-plan.json \
  --base-url http://127.0.0.1:18080 \
  --service-pid 12345 --worker-pid 12346 \
  --service-binary /absolute/path/to/apxinf-serve \
  --diagnostics-file /absolute/path/to/service-stderr.log \
  --metal-lock /absolute/path/to/metal-measure.lock --lock-fd 3 \
  --output-dir devlocal/serving-upstream/production-run-01
```

Replace the example identities, paths, and descriptor with values from the owning launcher.
Do not run this command from an unrelated shell without the inherited descriptor.
Exit status zero means all declared case checks passed and collection finished.
Complete validation remains incomplete because this attached runner cannot establish process reaping or candidate validation.
Retain the launcher's shutdown and reaping report separately.

Run the CPU fixture tests without loading a model:

```sh
python3 -m unittest discover -s tests/python -p test_serving_production_memory_validation.py -v
```

These tests use temporary loopback HTTP servers and original fixture responses.

Run the service's CPU HTTP integration checks separately:

```sh
cargo test --locked -p apxinf-serving --test http_gateway -- --nocapture
```

This suite uses the real CLI and TCP transport with an original CPU worker fixture.
It loads no model or MLX runtime.
Its default output directory is `devlocal/serving-http-integration/`.
Set `APXINF_HTTP_TEST_OUTPUT` to select another output directory.

## Boundaries

`http_benchmark.py` measures HTTP requests and streamed responses.
`claude_code_tasks.py` runs the installed Claude Code CLI against the local Messages endpoint.
The runner flushes captured stdout and stderr after each received chunk.
The captured files remain available for inspection while the client runs.
The runner records the client PID and process group in `client.process.json` before it reads output.
The client task report records actual tool calls, tool results, and task assertions.
The fixture files contain no repository content or credentials.

The client deadline remains active until the leader exits and both output pipes close.
At timeout, the runner stops its client process group and confirms that no live members remain.
Signal grace periods are 15 seconds for SIGINT, 5 seconds for SIGTERM, and 10 seconds for SIGKILL.
These cleanup periods can extend capture beyond the task deadline.
After group cleanup, pipe draining has a 0.5-second limit.
An open pipe or a cleanup failure raises an error instead of returning a successful task capture.

The runner does not signal descendants that create another process group.
Such descendants can keep pipes open, which causes the bounded drain to fail.

The harness gives each client process a separate configuration directory.
It supplies a dummy local API key and excludes inherited provider credentials and proxy settings.
It retains the existing `HOME` value. It does not change global Claude Code settings.
The `--bare` and `--restricted` flags limit client features and file access.
Explicit tool lists and permission rules limit each task.
The harness reads the deployment context limit from readiness and sets the client window for that process.
The `--max-context` option permits a smaller explicit window.
These bounded tasks disable automatic client compaction.
The 16K MiniCPM run otherwise entered repeated compaction after a successful file edit.
The service still rejects requests that exceed the actual input and output allowance.
These tests do not measure automatic compaction quality or unbounded agent sessions.
The client disables Anthropic extended-thinking requests in its isolated environment.
An explicitly selected native model-thinking deployment still enables its model template.
Its output allowance includes hidden reasoning tokens.
The `--max-tokens` option limits request output. It does not declare model capabilities in client metadata.

Claude Code 2.1.289 can count hidden output usage as retained context and block a later request locally.
For this version, `--acceptance-claude-context-compat` enables a separate acceptance mode.
The runner verifies the client version and requires a positive deployment context limit from readiness.
It sets the internal `CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE` to the effective client window.
The report records the version, deployment window, effective window, and threshold.
The default mode does not set this override.

This control is an undocumented client test hook, not a stable public interface.
The compatibility mode supports only bounded loopback acceptance runs.
It does not change model capacity, requested output, or reported usage.
The service still rejects an exact prepared input plus output allowance above its context limit.
The mode does not provide automatic compaction or indefinite sessions.

Anthropic does not officially support Claude Code with non-Claude models.
These tasks measure observed behavior with the installed client version.
They do not establish complete Claude Code compatibility.

## Offline checks

Run the parser, accounting, and task boundary checks:

```sh
python3 -m unittest discover -s tests/python -p test_serving_http.py -v
```

The live tests remain skipped unless `APXINF_SERVING_URL` exists.
Set that variable only after the service becomes ready and the measurement device becomes available.

## Performance run

Use the existing device lock when another application coordinates Metal measurements:

```sh
python3 benchmarks/serving/http_benchmark.py \
  --base-url http://127.0.0.1:8080 \
  --model qwen3.5-2b \
  --samples 32 --warmup 2 --concurrency 1 2 4 \
  --workload mixed --max-tokens 64 \
  --metal-lock /Users/haiyan-mini/.cache/enginetailor/metal-measure.lock \
  --output /tmp/apxinf-serving-benchmark.json
```

The harness opens the existing lock file for reading and requests an exclusive, nonblocking lock.
It fails if another process owns the lock. It does not stop that process.
The lock covers warmup and measured requests. It does not coordinate service startup.
The service operator must also coordinate model loading and other device work.

Each concurrency level uses a fixed number of simultaneous clients.
Each client sends its next request after its previous request finishes.
This closed-loop load does not represent an independent arrival process.
The mixed workload uses one long prompt for every four requests.
The long prompts share an original prefix. This property does not establish a cache hit.

The report contains raw request samples and server metric snapshots.
Each snapshot retains its timestamp and original metric text.
Metric summaries report first, last, minimum, and maximum observed values.
Memory counters retain their server names and units. They do not replace whole-system memory measurements.
Warmup requests appear separately and do not enter measured statistics.
`warmup_summary` reports warmup attempts, completions, and errors.
Any warmup failure sets exit status 1. The report retains all warmup and measured samples.

| Metric | Definition |
| --- | --- |
| TTFT | Request start to the first nonempty text, thinking, or tool argument delta. |
| E2E | Request start to complete response consumption or failure. |
| Inter-output latency | Time between nonempty SSE content events. |
| TPOT | E2E minus TTFT, divided by reported output tokens minus one. |
| Output throughput | Successful reported output tokens divided by measured wall time. |
| Attempt rate | All attempted requests divided by measured wall time. |
| Completion rate | Successful requests divided by measured wall time. |
| Error rate | Failed attempts divided by all attempts. |
| Goodput | Successful requests within every configured latency target, divided by measured wall time. |
| Usage coverage | Successful responses with output token counts, divided by successful responses. |

An SSE event can contain multiple tokens or a partial character.
Inter-output latency therefore differs from model token latency.
TPOT uses server token counts and includes the final response tail.
The report leaves unavailable values empty. It never substitutes character counts for token counts.

Set `--slo-ttft-ms` and `--slo-e2e-ms` to record explicit goodput targets.
Each supplied target must be finite and positive.
The harness reports no goodput target when both options remain absent.
P95 uses the nearest rank. Thirty-two samples provide exploratory evidence, not a reliable tail guarantee.
Record the model revision, client version, service revision, hardware, and background load with each final report.

## Fixed arrival rate

The default load mode remains `closed_loop`, with concurrency levels 1, 2, and 4.
Use both `--arrival-rate` and `--max-in-flight` to select `open_loop`.
The arrival rate is a finite, positive number of planned requests per second.
The maximum in-flight count is a positive integer.
The harness rejects an incomplete option pair, non-finite rates, and non-positive values.

Open-loop load uses one run with exactly `--samples` planned arrivals.
Omit `--concurrency` in this mode, or supply one value equal to `--max-in-flight`.
The harness rejects other concurrency settings in this mode.
The existing run field `concurrency` records the in-flight limit.
The report retains format `apxinf-serving-benchmark-v1` and adds `load_mode`.
It also records `arrival_rate_per_s` and `max_in_flight` for open-loop runs.

An arrival is a planned request time, independent of earlier request completion.
The first arrival starts the measurement clock. Arrival `i` occurs at `i / arrival_rate` seconds.
The in-flight limit includes all submitted requests that did not finish.
At capacity, the harness records a `client_capacity_drop` failure immediately.
It does not queue that arrival for later execution.
After the final arrival, the harness waits for every dispatched request to finish.

Run a fixed arrival rate against a service whose operator already holds the Metal lock:

```sh
python3 benchmarks/serving/http_benchmark.py \
  --base-url http://127.0.0.1:18080 --model apxinf-local \
  --samples 16 --warmup 2 --arrival-rate 1 --max-in-flight 4 \
  --workload mixed --max-tokens 32 --timeout 30 \
  --slo-ttft-ms 3000 --slo-e2e-ms 15000 \
  --output /tmp/apxinf-serving-arrivals.json
```

Each open-loop sample adds these fields:

| Field | Meaning |
| --- | --- |
| `scheduled_monotonic_s` | Planned arrival on the client's monotonic clock. |
| `scheduled_offset_s` | Planned arrival relative to the first arrival. |
| `dispatch_monotonic_s` | Time when the request thread starts, or null for an undispatched arrival. |
| `dispatch_delay_s` | Request thread start minus planned arrival, or null for an undispatched arrival. |
| `client_capacity_drop` | True when the in-flight limit prevents dispatch. |
| `arrival_ttft_s` | Planned arrival to first visible content, or null when no content arrives. |
| `arrival_to_completion_s` | Planned arrival to response completion or failure observation. |

The summary records `offered_requests`, `dispatched_requests`, and `client_capacity_drops` separately.
The existing attempt count includes every planned arrival in this mode.
Dropped arrivals contribute to the error count. They never contribute to successful throughput or goodput.
The summary also reports distributions for dispatch delay, arrival TTFT, and arrival completion time.
Arrival completion distributions include failures and capacity drops.

Request TTFT, E2E, and `goodput_requests_per_s` retain their existing meanings.
`arrival_slo_passing_requests` applies the same configured thresholds from each planned arrival.
`arrival_goodput_requests_per_s` divides that count by the full measurement time.
Both goodput denominators include arrival scheduling and the final wait for dispatched requests.
Absent latency targets produce null values for both goodput measures.

This final wait covers client HTTP completion, not server resource settlement.
Public expiry can precede worker cleanup.
The metric monitor ends when the client requests finish, so later cleanup can fall outside its snapshots.
Nonzero active requests or `apxinf_cleanup_pending` do not establish a drained server.
Operators must observe resource settlement separately before another device experiment.

## Absolute HTTP deadline

`--timeout` sets an absolute deadline for each HTTP request, including request transmission, headers, and the response body.
The socket also uses that value as an idle timeout.
Heartbeats and partial SSE lines do not extend the absolute deadline.
The client closes only the connection owned by the expired request.
An expired request remains a failed sample, including when content arrived before expiry.

## Claude Code tasks

The `expenses` and `meeting-actions` tasks measure bounded daily file work.
Both tasks permit only `Read` and `Edit`.
Each task creates original input files and an empty JSON output file.
The prompt defines the schema and rules, but does not contain the expected answers.
The task specification retains original input bytes, SHA-256 digests, an output limit, and a task revision.

The `expenses-v1` input contains dated CSV transactions and expense rules.
Include only posted CNY transactions in the listed categories.
Include refunds as negative amounts. Exclude other currencies, pending rows, and unlisted categories.
Convert decimal amounts to integer cents without rounding.
The output file is `expenses.json` with exactly these fields:

- `currency` is the string `CNY`.
- `included_rows` contains included transaction IDs in source order.
- `category_totals_cents` maps each listed category to its integer total.
- `total_cents` is the integer sum across categories.

The `meeting-actions-v1` input contains original meeting notes and extraction rules.
Extract explicit actions and decisions in source order.
Exclude suggestions and discussion without an explicit action.
Use JSON null for an absent owner or due date.
The output file is `meeting-actions.json` with exactly these fields:

- `meeting_date` is the ISO date from the notes.
- `actions` contains objects with exactly `task`, `owner`, and `due_date`.
- `decisions` contains explicit decision strings.

Both outputs must contain one strict JSON object within 8192 bytes.
Duplicate fields, unknown fields, nonfinite numbers, and incorrect value types fail validation.
The checker compares the complete structured result with independent, handwritten expectations.
The checker never executes model output.

A pass requires the actual initial tool inventory and a successful final client result.
The client must exit with code zero without timeout or cancellation.
Every input requires a successful `Read` before a successful output `Edit`.
Tool IDs must be unique. Assistant messages contain calls. User messages contain results.
All tool paths must resolve to an input or the fixed output.
Every tool result must indicate success. Every input must retain its original bytes.
The output must be a bounded regular file without a symbolic link.
The report separates `tool_cycle_passed`, `inputs_unchanged`, `artifact_check`, `observed_input_usage`, and `passed`.
These checks establish exact fixture completion, not general financial or meeting analysis quality.

### Expense calculation tools

The `expenses-tools-cny` and `expenses-tools-usd` tasks add original cases with a calculation tool.
Their revisions are `expenses-tools-cny-v1` and `expenses-tools-usd-v1`.
Each workspace contains a CSV, selection rules, usage instructions, and the same standard-library calculator.
The model reads these inputs and both empty JSON targets before its first Edit.
The model writes `calculation-plan.json` with exactly `currency`, `status`, and `categories`.
Category order is free, but selected transaction IDs must retain CSV source order.

The model then runs exactly `python3 -I expense_calculator.py` through Bash.
The calculator prints a receipt and writes no files.
The model saves the receipt's `summary` into `expenses.json` with Edit.

The calculator validates every CSV row before selection and uses signed integer cents.
It includes eligible refunds and zero amounts without currency conversion or rounding.
Limits are 100 rows, 16 categories, and 12 integer amount digits.
Money integers must remain within the JSON safe-integer range.

Byte limits are 65536 for CSV, 8192 each for rules and usage, and 32768 for calculator source.
The plan, receipt, and artifact limits are 4096, 16384, and 8192 bytes.
These utility limits do not measure model capacity.

The checker binds the complete receipt to original file hashes, the successful plan Edit, and independent literal expectations.
It requires successful results before dependent mutations and rejects failed attempts, pending calls, and repeated final results.
The pinned Claude Code 2.1.289 Bash result supplies stdout, stderr, and execution flags.
The checker requires matching stdout content, empty stderr, and false interruption, image, and no-output flags.
It permits a bounded Bash description and timeout, and an optional false Edit `replace_all` flag.

The requested inventory must contain exactly Bash, Edit, and Read. The advertised inventory may contain additional tools.
The report adds `calculation_receipt` and `tool_cycle_reason` to the existing evidence fields.

Run these cases with `--tasks expenses-tools-cny expenses-tools-usd`.
CPU tests verify calculator behavior and trace rejection. Actual model completion requires a separate captured client run.
The [M4 expense trial report](../../doc/serving/expense-tools-validation-20261023.md) records six failed MiniCPM5-2B task attempts.
Successful HTTP requests and CPU arithmetic do not establish task completion.
The evidence assumes trusted capture and no external writer. Tool permissions do not provide an operating-system sandbox.
These cases do not establish long-context capacity or general financial quality.

The cases add no dependencies or serving protocol changes.

Run the three small tasks after service readiness:

```sh
python3 benchmarks/serving/claude_code_tasks.py \
  --base-url http://127.0.0.1:8080 \
  --model qwen3.5-2b \
  --tasks read edit check --repeats 1 \
  --metal-lock /Users/haiyan-mini/.cache/enginetailor/metal-measure.lock \
  --output-dir /tmp/apxinf-claude-tasks
```

The read task requires a Read call and the marker from a newly created file.
The edit task requires Read and Edit calls. A restricted expression evaluator checks the changed function.
The check task requires Read and Bash calls. It checks the output from an original local script.
A claimed result without the required tool call fails the task.

Use `--tasks cancel` for the separate cancellation experiment.
The harness interrupts only its own process group after the first content delta.
A passing result requires a larger cancellation counter and zero active and queued requests.
If generation finishes before interruption, the harness does not report successful cancellation.
Run this experiment without other requests to make the counter evidence attributable.

The report separates CLI elapsed time from HTTP latency.
CLI time includes process startup, prompt construction, model requests, and tool execution.
Each task directory retains client output, client errors, fixture files, and an isolated configuration directory.
The harness preserves failed results. It does not replace them with expected output.

## Lifecycle checks

`lifecycle_checks.py` checks cancellation, bounded admission, and an optional idle worker failure.
Run it after the Claude Code tasks and before the operator unloads the service.
The service operator must hold the device lock. The lifecycle tool does not acquire another lock.

```sh
python3 benchmarks/serving/lifecycle_checks.py \
  --base-url http://127.0.0.1:8080 --model apxinf-local \
  --checks cancel queue --queue-capacity 16 --timeout 15 \
  --output /tmp/apxinf-lifecycle.json
```

Set `--queue-capacity` to the actual deployment configuration.
The current readiness document does not expose that value.
The tool does not infer a capacity when neither the flag nor readiness supplies one.
It fills the waiting queue behind one active stream and requires an overflow response with HTTP 429 and `queue_full`.
It closes its own sockets after the observation, including when an assertion fails.
Settlement requires zero active requests, queued requests, and reserved bytes, plus a larger cancellation counter.
The report retains request IDs when the server exposes them, local probe IDs, and metric snapshots.
The test uses short observation deadlines. It does not wait for the deployment's full request timeout.

The optional worker failure check sends SIGTERM and leaves the service unavailable.
Use the explicit service and worker PIDs recorded by the launcher:

```sh
python3 benchmarks/serving/lifecycle_checks.py \
  --base-url http://127.0.0.1:8080 --checks worker-failure \
  --service-pid "${APXINF_SERVICE_PID:?Set the recorded service PID}" \
  --worker-pid "${APXINF_WORKER_PID:?Set the recorded worker PID}" \
  --allow-worker-termination --timeout 15 \
  --output /tmp/apxinf-worker-failure.json
```

This macOS check uses `/bin/ps` and `/usr/sbin/lsof` for the supplied PIDs only.
It checks ownership, parentage, worker command, model path, worker epoch, and the service listener before signaling the worker.
It does not discover or stop other services.
It then requires readiness HTTP 503, health HTTP 200, zero reservations, and the same service process identity.
An identity mismatch prevents the signal. A final operating-system PID reuse race remains possible between the last check and the signal.
The operator restarts or unloads the service after this check.

## Protocol references

These official documents supplied behavioral requirements. No source implementation or example code entered the harness.
The research date is 2026-10-09.

- [Gateway protocol](https://code.claude.com/docs/en/llm-gateway-protocol) defines optional fields and response behavior.
- [Messages streaming](https://platform.claude.com/docs/en/build-with-claude/streaming) defines SSE ordering and incremental tool arguments.
- [CLI reference](https://code.claude.com/docs/en/cli-reference) defines the client flags.
- [Environment variables](https://code.claude.com/docs/en/env-vars) defines process configuration controls.
- [Headless operation](https://code.claude.com/docs/en/headless) defines structured client output.
- [Gateway support](https://code.claude.com/docs/en/llm-gateway) states the model support boundary.

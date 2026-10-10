# Local serving profile 0.1

This profile implements P1a and an initial P2a client adapter.
It uses the [serial worker profile](serial-profile-v0.1.md).
The implementation does not complete the full roadmap.
The measurement report records actual model and client results separately.

## Components

`apxinf-serve` owns HTTP, request admission, deadlines, output parsing, and worker supervision.
One Python worker owns the model and all MLX operations.
One bounded queue precedes that worker.
The default queue holds 16 waiting requests.
The worker executes one request at a time.

The coordinator removes cancelled and expired waiting requests every 20 milliseconds and before enqueue.
This cleanup proceeds during active execution and retains FIFO order for remaining requests.
Queue and public result counters change once for each removed request.

The supervisor checks model artifacts and runtime files independently after worker loading.
The content manifest includes the model template, tokenizer, adapter, interpreter, and dependency pins.
An identity mismatch prevents readiness.
The service never selects another model or precision automatically.

The service supports stateless text requests.
The existing v1 CLI and exact append interface retain their behavior.
This HTTP profile does not expose sessions, images, embeddings, batch execution, or prefix reuse.

## Diagnostic output

The process uses one dedicated diagnostic writer thread and one queue with 256 records.
Each record contains at most 4096 bytes, including its final newline.
The writer can hold one additional record while its destination blocks.
Full queues discard new records without waiting.
Oversized records disappear completely, without partial JSON output.
A write failure disables the writer and discards its remaining records.
Thread creation failure prevents worker startup.

Worker stderr uses a separate pipe and an asynchronous reader with 512-byte chunks.
Each chunk becomes a diagnostic record with its worker epoch.
Chunk boundaries do not represent complete lines or complete UTF-8 characters.
Invalid or split UTF-8 sequences use replacement characters.
The supervisor stops the reader when it drops that worker instance.
Unread bytes can disappear during worker recovery or rotation.

The service never requires diagnostic delivery for a request, recovery, or shutdown result.
At process exit, the binary allows up to 100 milliseconds for remaining diagnostics.
It does not join a blocked diagnostic writer.
This output is unsuitable for a durable audit trail.

Diagnostic metrics describe the complete process, including worker rotations:

| Metric | Meaning |
| --- | --- |
| `apxinf_diagnostic_written_total` | Records whose complete write succeeded. |
| `apxinf_diagnostic_dropped_total{reason}` | Discarded records, with `full`, `oversized`, or `unavailable` as the reason. |
| `apxinf_diagnostic_write_errors_total` | Failed destination writes. |
| `apxinf_diagnostic_worker_read_errors_total` | Failed reads from worker stderr. |
| `apxinf_diagnostic_writer_available` | Whether the writer thread remains available. |

These counters do not measure delivery beyond the destination write.
A failed write can leave partial bytes at that destination.
Metrics can remain available after diagnostic output fails.
Writer availability does not establish destination progress.
Loss counters exclude unread stderr bytes and records without final accounting when the process exits.

## Build and start

Run the following commands from the repository root on an Apple Silicon Mac.
The worker requires the exact interpreter and package versions below.
The `.apxinf/toolchains/` environment is a local dependency directory, not a checked-in runtime.
Create it before using the startup command.

1. Install Python 3.14.3 with the macOS installer from the [official release page](https://www.python.org/downloads/release/python-3143/).
2. Set `APXINF_PYTHON` to that installation's Python executable.
3. Check its version before creating the environment.

```sh
APXINF_PYTHON=/absolute/path/to/python3.14
"$APXINF_PYTHON" - <<'PY'
import platform
import sys

if platform.python_version() != "3.14.3" or platform.machine() != "arm64":
    sys.exit("Use Python 3.14.3 running as arm64.")
PY
```

Create the environment with copied interpreter files:

```sh
mkdir -p .apxinf/toolchains
"$APXINF_PYTHON" -m venv --copies .apxinf/toolchains/mlx-lm-0.31.3-copies
```

Install all eight package pins in that environment:

```sh
.apxinf/toolchains/mlx-lm-0.31.3-copies/bin/python -m pip install \
  huggingface-hub==1.28.0 \
  mlx==0.32.1 \
  mlx-lm==0.31.3 \
  mlx-metal==0.32.1 \
  numpy==2.5.2 \
  safetensors==0.8.0 \
  tokenizers==0.22.2 \
  transformers==5.15.1
.apxinf/toolchains/mlx-lm-0.31.3-copies/bin/python -m pip check
```

Check the versions with the same first-party helper that the worker uses:

```sh
.apxinf/toolchains/mlx-lm-0.31.3-copies/bin/python - <<'PY'
import importlib.util
import json

spec = importlib.util.spec_from_file_location("apxinf_runtime_pins", "scripts/apxinf_mlx_generate.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
print(json.dumps(helper._pinned_toolchain_versions(), indent=2, sort_keys=True))
PY
```

This check reads package metadata without loading a model.
The [dependency record](dependencies.md#python-worker) describes the helper and public runtime APIs.
These eight pins do not lock every transitive dependency.
Retain an installed-package record with each experiment.

Store task logs and benchmark results under `devlocal/serving-upstream/`.
Use a new subdirectory for each run to preserve earlier evidence.
Keep the managed Python environment in `.apxinf/toolchains/`.

```sh
mkdir -p devlocal/serving-upstream
APXINF_RUN_DIR="$(mktemp -d "$PWD/devlocal/serving-upstream/run-XXXXXX")"
.apxinf/toolchains/mlx-lm-0.31.3-copies/bin/python -m pip freeze \
  > "$APXINF_RUN_DIR/python-packages.txt"
```

Build the new binary:

```sh
cargo build --release -p apxinf-serving
```

This Mac requires the installed macOS 15.4 SDK for its current linker.
The default macOS 27 SDK contains architecture declarations that this linker cannot read.
Use this command on the recorded test host:

```sh
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk \
  cargo build --release -p apxinf-serving --offline
```

Start a local model:

```sh
target/release/apxinf-serve \
  --model /absolute/path/to/local/model \
  --model-id apxinf-local \
  --python .apxinf/toolchains/mlx-lm-0.31.3-copies/bin/python \
  --port 8080
```

The process requires a loopback address.
The process exposes its listener only after model loading and identity checks.
The default context limit is 16384 tokens.
The default output limit is 2048 tokens.
The default request timeout is 300 seconds, including queue time.

## HTTP interface

| Route | Operation |
| --- | --- |
| `GET /healthz` | Check the HTTP process. |
| `GET /readyz` | Check readiness and inspect the exact capability profile. |
| `GET /v1/models` | Obtain the configured model alias and model revision. |
| `POST /v1/messages` | Generate an Anthropic Messages response, with optional SSE output. |
| `POST /v1/messages/count_tokens` | Apply the same template and count its tokens. |
| `POST /v1/chat/completions` | Generate an OpenAI Chat Completions response, with optional SSE output. |
| `GET /metrics` | Read counters in Prometheus text format. |

These routes implement the declared subset only.
They do not establish compatibility with the OpenAI Responses API or every Anthropic extension.

The request body limit is 1000000 bytes.
The decoder rejects duplicate JSON keys and excessive nesting.
The gateway checks the complete normalized input before queue admission.
Malformed client input must not stop a healthy worker.

The request model must equal the configured alias.
Generation uses greedy decoding.
The gateway accepts omitted temperature or temperature zero.
The gateway accepts omitted `top_p` or `top_p` one.
Other sampling settings receive an explicit error.
OpenAI requests can set either `max_tokens` or `max_completion_tokens`.
The gateway rejects requests that set both fields.

Anthropic messages support text, `tool_use`, and `tool_result` blocks.
Tool results must precede text within their user message.
An error result adds `[tool_result is_error=true]` before its text during normalization.
The service accepts `auto` and `none` tool choice.
It rejects forced tools, server tools, and enforced single-tool generation.
The optional `disable_parallel_tool_use` field must contain a Boolean.

The selected model template defines the tool grammar.
The parser accepts Qwen XML calls and JSON calls inside `tool_call` markers.
The parser rejects undeclared tools and incomplete tool blocks.
It checks object arguments, required properties, basic types, enums, and additional properties.
It does not claim complete JSON Schema enforcement or constrained decoding.
The gateway rejects `strict: true` on OpenAI function tools.
The service never executes a tool.
The client applies its own tool permissions and executes permitted calls.

Thinking must remain disabled in this profile.
The gateway treats `output_config.effort` as advisory and performs no corresponding model adjustment.
Accepted effort values are `low`, `medium`, `high`, `xhigh`, and `max`.
The gateway accepts client metadata and cache hints without creating a prompt cache.
Readiness explicitly reports no prompt-cache capability.

## Streaming and failures

Anthropic SSE output uses matching event names and payload types.
Each content block has a stable, contiguous index.
The gateway emits tool arguments only after a complete call passes parser checks.
Clients must not expect one SSE delta per model token.

OpenAI clients can request `stream_options.include_usage` with streaming enabled.
The gateway then adds a separate final usage chunk with an empty `choices` array.
Other chunks contain null usage in this mode.
Omitted or false `include_usage` omits usage fields from the stream.
This behavior follows the [official Chat API reference](https://developers.openai.com/api/reference/resources/chat).

Output usage counts model tokens, including hidden EOS and reasoning tokens.
It excludes internal lookahead predictions.
The final usage count comes from the worker terminal result.
The gateway waits for resource settlement before it emits public success.
The original public deadline remains active during generation and cleanup waiting.
Worker completion or a stop match cannot permit success after that deadline.
At expiry, the service returns an error without waiting for cleanup.
It reports an already observed parser or worker failure as failed instead.
Resources remain reserved until checked cleanup or worker process reaping.
Late worker events cannot replace that public result or add public output.

A stop match requests normal completion.
A disconnect requests cancellation.
The gateway does not convert worker failure or expiry into successful stopping.
The output channel holds at most 64 content events and one reserved final event.
The reserved slot prevents final-result loss when the content queue is full.
Content overflow requests cancellation with cause `slow_consumer`.
Closed receivers count as client cancellation.
Parser errors and worker protocol faults remain failures after a disconnect.
Request counters describe the public result.
Settlement logs record both the public status and worker status.
An output block cannot exceed 262144 bytes.
Total decoded output cannot exceed 4194304 bytes.

The gateway rejects a full queue with HTTP 429.
It rejects context overflow before generation.
A worker protocol fault fences the worker epoch.
The supervisor stops the failed worker before releasing uncertain reservations.
This revision requires a service restart after a worker fault.
It does not replay requests automatically.

Planned command-history rotation uses a separate lifecycle path.
Before preparation, the supervisor reserves four command records for preparation, submission, one control, and shutdown.
It rotates the worker when fewer records remain within the 10000-record limit.
The old process exits before replacement loading starts.
The replacement must match the original approved model and capability revisions.
Readiness exposes the new worker epoch after identity checks pass.
It publishes the complete replacement `ready` snapshot, including current memory values.

Command exhaustion triggers worker shutdown before the coordinator removes another waiting request.
The old worker can drain and exit while host pressure prevents replacement loading.
The supervisor then waits for pressure recovery before it retries replacement loading.
Waiting requests retain their deadlines and independent queue removal during this wait.
New requests receive `model_unavailable` while the worker rotates.

Pressure after replacement loading requires process reaping before another load attempt.
Service shutdown ends the pressure wait without a worker fault.
Worker faults and failed process reaping remain failures, including during shutdown.

A request can expire or receive cancellation while its template and tokenizer run.
The service ends that public request immediately and retains the execution position.
It waits up to 20 additional seconds for the matching preparation response.
It discards that response before it dispatches another request.
A request deadline alone does not make this path a worker fault.

Public deadline and cancellation checks continue while the supervisor writes preparation to stdin.
Each stdin write has a separate 20-second health timeout.
The preparation settlement grace includes any unfinished write.
The service retains the execution position until safe settlement or worker recovery completes.
These control-plane policies retain protocol `apxinf-worker/2.0` and its existing wire fields.

## Metrics and shutdown

`apxinf_requests_total` reports separate `completed`, `cancelled`, `failed`, and `expired` status counters.
These counters include terminal outcomes during queue waiting, preparation, and token counting.
Each queued request increments one status counter at its public terminal result.
`apxinf_worker_rotations_total` counts successful planned rotations.
`apxinf_worker_faults_total` counts worker faults separately from public request outcomes.

The [observability contract](contracts-v0.1.md#observability-contract) defines the new duration histograms and fixed bucket boundaries.
The operation label is `generate` or `count_tokens`.
`apxinf_request_outcomes_total` adds operation and status labels without replacing the existing status counter.
Its operation totals equal each existing status total.
Requests rejected before enqueue do not enter either outcome counter.
Existing metric names and meanings remain available.

Legacy `apxinf_queue_seconds_sum` covers requests selected for preparation.
Legacy `apxinf_request_seconds_sum` covers generation attempts with checked worker settlement.
These legacy sums remain separate from the new histogram families.

Queue duration includes requests that end without execution.
Preparation duration continues until the matching response or successful process recovery.
It includes blocked stdin transmission and late responses after public expiry or cancellation.
Service request duration starts at enqueue and ends at the public terminal result.
These durations exclude earlier HTTP input reading and normalization.

The first worker event and first public output use distinct histograms.
Only a non-empty text or complete tool call accepted by the result channel establishes public output readiness.
This point does not establish an HTTP flush or client TTFT.
Adjacent worker event intervals include IPC and event aggregation, not only model computation.
The worker first-token histogram uses its existing local `ttft_ns` value when output tokens exist.
Missing events do not create zero-duration samples.

`apxinf_cleanup_pending` counts public terminal requests that still need resource settlement.
Its count remains until a checked cleanup event, settled preparation, or successful process recovery.
Failed process recovery produces no settlement sample and does not clear unresolved cleanup.
Success after settlement produces a zero public-terminal settlement duration.
Separate worker-terminal settlement timing measures cleanup before public success.

Histograms use fixed operation labels and the required `le` bucket label.
The service request histogram also uses the four public status values.
The cleanup gauge uses only the operation label.
Request identities and error messages remain in logs rather than metric labels.
These measurements do not establish a performance SLO, memory estimator, or batch capability.

An embedding host must call `Service::shutdown` explicitly.
This operation closes admission and the waiting queue.
Active work retains its original deadline and cancellation rules.
After active settlement, the coordinator stops and reaps the worker, then exits.
Dropping external `Arc<Service>` references alone does not guarantee shutdown.
The CLI calls `Service::shutdown` on Ctrl-C.

The host must then await `Service::wait_stopped` before it stops the async runtime.
This operation returns success only after the coordinator reaps all owned worker processes.
The operation returns an error if the coordinator fails or cannot reap an owned worker process.
The CLI awaits `Service::wait_stopped` after HTTP termination, including HTTP errors.

## Memory and device controls

The default MLX memory guideline is 10 GiB.
The default sequence reservation is 1 GiB.
Startup checks resident allocations plus that reservation against the configured budget.

The embedding API checks both memory limits before worker startup.
Each limit must be between 1 and 9007199254740991 bytes.
The sequence reservation must not exceed the memory budget.
Budget arithmetic rejects integer overflow.

The profile retains one sequence credit and one compute permit during each request.
The worker synchronizes device work before resource settlement.

`apxinf_worker_peak_bytes` retains a conservative allocator peak across worker epochs.
Each `ready` contributes the larger of `peak_bytes` and the sum of `active_bytes` and `cache_bytes`.
Each checked terminal contributes its peak before resource settlement, including publicly expired or cancelled requests.
A later cleanup failure does not remove that observation.

These controls do not establish an operating-system memory limit.
MLX documents its memory limit as a guideline.
The reservation is a deployment estimate, not a proof for every model shape.
Peak metrics describe MLX allocator memory rather than complete system memory use.

### Host pressure policy

The CLI defaults to `--host-pressure-policy macos`.
The library caller selects `HostPressurePolicy` explicitly in `Config`.
The `macos` policy reads `kern.memorystatus_vm_pressure_level` through the system `sysctlbyname` function.
The sensor uses the returned dispatch values: 1 means normal, 2 means warning, and 4 means critical.
Other values and read failures produce `unknown`.
Unsupported platforms require explicit `--host-pressure-policy disabled` or fail startup.

The supervisor checks pressure before initial loading and before replacement loading.
A warning, critical, or unknown result prevents loading.
It checks again after loading, before it publishes worker readiness.
A failed check stops and reaps that worker.
These checks cannot prevent pressure changes during loading.

The monitor samples once per second without invoking a subprocess.
Sampling continues during graceful shutdown until the coordinator finishes active work and worker recovery.
A sample becomes stale after three seconds, including exactly three seconds.
Warning, critical, unknown, and stale samples block admission.
A normal first sample permits initial loading.
Recovery requires two consecutive normal samples at least one second apart.
Additional normal samples within that interval do not reset its start time.
A stale gap resets this recovery sequence.

Admission checks run before enqueue, before preparation, and before generation dispatch.
Blocked requests receive `capacity_unavailable` with HTTP 503.
Waiting requests keep their original deadlines and cancellation rules until selection.
Selection rejects a blocked request without worker dispatch.
A blocked pressure snapshot prevents generation dispatch after preparation settles.
An active generation retains its existing deadline, cancellation, and resource settlement rules.

`GET /readyz` returns 503 when host admission blocks or the worker is unavailable.
Its `host_pressure` object reports `policy`, `state`, `admission_allowed`, `recovering`, and nullable `sample_age_seconds`.
The `state` is `disabled`, `normal`, `warning`, `critical`, `unknown`, or `stale`.
The top-level `worker_available` field reports the separate worker lifecycle state.
`GET /healthz` remains independent of pressure.
Pressure recovery cannot reopen a closed service or fix a worker fault.

The pressure metrics use fixed labels only:

| Metric | Meaning |
| --- | --- |
| `apxinf_host_pressure_enabled` | One for the enabled policy, otherwise zero. |
| `apxinf_host_pressure_state{state}` | One for the current state, otherwise zero, for each of the six states. |
| `apxinf_host_admission_allowed` | One when the pressure policy allows admission, otherwise zero. |
| `apxinf_host_pressure_sample_age_seconds` | Sample age, or `NaN` when no sample exists. |
| `apxinf_host_pressure_read_errors_total` | Failed or invalid sensor reads. |
| `apxinf_host_pressure_rejections_total{stage,reason}` | Pressure rejections at `enqueue` or `dispatch`. |

The rejection reason is `warning`, `critical`, `unknown`, `stale`, or `recovering`.
The `dispatch` stage includes preparation and generation checks.
Metrics describe a snapshot at scrape time.
The disabled policy reports no sample and permits admission without pressure checks.

A normal pressure signal does not prove sufficient headroom or absence of swap activity.
This policy does not replace a measured request memory estimate.
It does not change MLX limits, model precision, or worker capability identity.

## Validation

Run the shared protocol and gateway checks:

```sh
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk \
  cargo test -p apxinf-serving --offline
python3 -m pytest -q \
  tests/python/test_serving_contracts.py \
  tests/python/test_serving_text_worker.py \
  tests/python/test_serving_http.py
```

Follow [the benchmark guide](../../benchmarks/serving/README.md) for real model and Claude Code tests.
Follow [the Metal experiments](metal-experiments.md) for controlled parameter comparisons.
Tests must preserve raw failures and actual outputs.
CPU test doubles cannot establish real inference or client compatibility.
The [lifecycle report](lifecycle-hardening-20261009.md) separates the latest CPU checks from historical model measurements.

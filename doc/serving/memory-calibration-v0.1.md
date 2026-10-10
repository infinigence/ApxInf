# Memory calibration profile 0.1

This profile defines offline observations for the serial MLX adapter.
It does not define an approved admission estimate.
The Python measurement tool owns this artifact format.
The Rust service does not consume this format or change its budget from it.
The worker protocol remains `apxinf-worker/2.0`.

## Runtime observations

The execution owner can call `MLXRuntime.inspect_memory` between generation steps and after cleanup.
This operation waits for all streams that the runtime owns before it reads counters.
The runtime captures the generation stream through the public progress callback context.
Cleanup waits for that stream before it releases cache references or clears allocator storage.
A missing generation stream is valid only before generation submits work.

The observation contains these fields:

| Field | Meaning |
| --- | --- |
| `allocator` | Existing `active_bytes`, `cache_bytes`, and `peak_bytes` counters. |
| `peak_epoch` | Zero for loading, then one additional epoch for each generation peak reset. |
| `runtime_consumed_position` | The adapter's tracked position, which can differ from physical cache offsets. |
| `cache_payload_bytes` | The sum of supported cache `nbytes` values, or null if any value is unknown. |
| `layers` | Ordered cache observations with `type`, nullable `offset`, and nullable `nbytes`. |

Cache payload is a logical array observation, not a complete allocation charge.
It can already contribute to allocator active memory.
The tool must not add cache payload to allocator active memory.
The tool must not add a peak counter to current active or cache memory.
An unknown cache property remains null instead of zero.
The adapter must not invent offsets for recurrent cache layers.

The tool records allocator peaks in their original reset intervals.
It does not reset peaks at each sampling checkpoint.
Synchronized sampling can change allocation overlap and execution timing.
These observations do not establish uninstrumented serving latency or a universal peak bound.

## Calibration artifact

The format name is `apxinf-memory-calibration-v1`.
The artifact kind is `raw_observations`.
The tool records its digest, model manifest, runtime versions, hardware, execution profile, and the complete case plan.
It records the input token digest and exact input length for each case.
Each case specifies its output allowance, stop condition, and repetition index.
Synthetic token inputs remain explicit and do not establish template or model quality.

Every planned case has one result: `completed`, `blocked`, `failed`, or `skipped`.
The tool preserves errors, their phase, partial samples, and cleanup results.
An earlier failure must not remove later planned cases from the artifact.
An interrupted run must retain its completed observations.
Only exclusive creation may start a new output artifact.

The command creates a new output directory and writes `calibration.json` there.
A parent process acquires the device lock and starts one isolated probe process.
Both processes retain the same inherited lock descriptor until the probe exits.
The parent checks that the probe exited before it returns the lock.
The probe records request settlement separately from process exit and model reclamation.

The top-level fields are `format`, `artifact_kind`, `run_id`, `created_at`, `tool_sha256`, `requested_profile`, and `model_path`.
`tool_files` records separate SHA-256 digests for the driver and its host and process helpers.
Additional fields are `hardware`, `identity`, `input_source`, `plan`, `cases`, `run_samples`, `status`, `error`, `error_phase`, and `process`.

`load_started` records entry into the loader, not successful loading.
`run_cleanup` records final settlement separately from process reaping.
`recovery` records journal recovery errors, or remains null during normal probe execution.
Identity remains null until the model loads and its manifest becomes available.

The process result records its PID, return code, and reaping evidence.
The parent marks an incomplete report failed if the probe exits before a final result.
Timeouts, launcher errors, and unconfirmed process reaping also fail the run.
The parent preserves a failed report even when the probe never starts.
`process.json` is the authoritative parent outcome.

After reaping, a structurally invalid probe report fails the run.
The parent preserves its original bytes in `calibration.invalid-<UUID>.json` when filesystem access permits.
The parent uses its initial report and complete plan for journal recovery.
Recovered samples cannot reconstruct missing model identity, output counts, or successful settlement.
The parent records the preserved filename or preservation error in its process outcome.
`process.invalid_report` contains nullable `file` and `error` fields for this operation.

The parent attempts to write its outcome even if it cannot save the final calibration report.
`process.report_write_error` records that write failure separately from any earlier parent error.

If reaping remains unconfirmed, the parent leaves the probe report untouched and marks its own outcome failed.
This rule prevents concurrent report writes by the parent and a surviving probe.

Each plan entry contains `case_id`, `input_tokens`, `max_tokens`, `stop_after_prefill`, `stop_after_output`, and `repetition`.
Unused stop boundaries are null.
Case results reference their plan entry by `case_id` and retain their execution index.
Results include input and output digests, actual output count, EOS observation, and whether the requested stop boundary occurred.
They also contain samples, status, reason, error phase, error text, and separate generator closure and settlement results.

An executing case temporarily uses status `running`.
An interrupted case uses null for an output count that the tool cannot recover exactly.
The parent retains its observed count separately through journal samples.

The input source tiles tokens from one original prepared prompt and truncates them to each planned length.
It records the seed prompt, template options, prepared seed tokens, and their digest.
This synthetic workload measures shapes, not conversational quality.
The complete case plan exists before model loading, including cases that never execute.

## Explicit plans and seed messages

`--plan-file` accepts a JSON array of one to 256 cases.
Each case contains exactly the six plan fields defined above.
The tool preserves case order and adds no cases or repetitions.
Case IDs are unique and match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`.
The repetition index is an integer from zero through three.
Booleans and floating-point values are not integers.

Input length is between one and `max_context`.
Output allowance is between zero and `max_output_tokens`.
Each case requires `input_tokens + max_tokens <= max_context`.
The tool checks this condition per case, without applying the largest output allowance to every input length.

Both stop fields are present and use null when unused.
A case has at most one stop boundary.
A prefill stop is an integer from zero through the input length.
An output stop is an integer from one through the output allowance.
Zero-output cases cannot specify a stop boundary.
Prefill stops refer to reported callback positions, not pure prefill completion or physical cache offsets.

`--plan-file` conflicts with explicit `--input-lengths`, `--max-tokens`, and `--repetitions` options.
Without a plan file, the existing default plan and options retain their behavior.
Deployment limits and runtime options still apply to an explicit plan.

`--seed-messages` accepts a JSON messages array.
The existing public `prepare_input` validator checks this array and its encoded frame size.
The tool fixes `tools` to an empty array and `enable_thinking` to false.
The messages file cannot override the provider, precision, template options, or tool definitions.
Without this option, the tool uses its existing original seed message.

Each external plan or messages file has a 1 MiB read limit.
The reader rejects duplicate object keys, non-finite numbers, invalid UTF-8, and malformed JSON.
The tool checks both files before it acquires the device lock or creates the output directory.

The parent freezes checked inputs in `measurement-inputs.json` before it starts the probe.
It creates this file exclusively and never rewrites it.
The frozen format is `apxinf-calibration-inputs-v1`.
It contains `profile`, `plan`, `seed_messages`, `plan_source`, and `seed_source`.
The frozen file has a 4 MiB read limit.

Each source record contains `kind`, `path`, and `sha256`.
The kind is `default` or `file`.
File sources record their absolute source path and original byte digest.
Default sources use a null path and the digest of their normalized JSON content.
Normalized JSON uses sorted keys, compact separators, UTF-8, and no final newline.
The source path is provenance, not a later read instruction.

The child receives the expected frozen-file digest through an internal argument.
It reads only the fixed frozen path in its output directory.
It checks the digest, profile, cases, and messages before model loading.
It does not read mutable external plan or seed files.

CLI reports retain `measurement_inputs` before loading, including blocked runs and spawn failures.
This field records the frozen filename, digest, source records, and seed messages.
Direct calls without a frozen specification use null for this field.
The existing `plan` field retains every case.
Prepared input evidence must match the frozen messages and fixed template options.
After preparation, `input_source` retains the actual messages, template options, seed tokens, and token digest.
Changing seed messages does not bypass EOS or establish held-out validation by itself.

After reaping, the parent checks the report's plan, profile, and frozen-input metadata against its original snapshot.
A mismatch makes the run fail without discarding its journal.
These additions retain raw format v1, default execution behavior, and the worker protocol.

Each sample includes its phase, monotonic timestamp, output count, memory observation, and host pressure observation.
The phase vocabulary is `before_load`, `loaded`, `before_request`, `prefill_progress`, `first_output`, `output_checkpoint`, `terminal`, and `settled`.
Progress samples include the callback's reported prompt position.
That position does not establish a pure prefill boundary or a physical cache offset.
The pinned generator can compute ahead before it yields an output token.

The output count counts actual generator yields, including EOS.
The tool applies the configured EOS policy and records early EOS explicitly.
Zero-output cases follow the service's no-generation behavior.
They do not call the runtime generator as an independent prefill probe.
Stopping during prefill and after output uses explicit case conditions.
The tool closes each generator before it requests resource settlement.
Failed settlement stops subsequent cases.

## Host snapshot

`host_memory.snapshot()` returns one JSON-safe host observation.
Its `monotonic_ns` field records when the observation starts.
The individual reads are sequential and do not form an atomic system snapshot.

| Field | Meaning |
| --- | --- |
| `pressure.state` | `normal`, `warning`, `critical`, or `unknown`. |
| `pressure.dispatch_value` | The macOS dispatch value, or null when the read fails. |
| `pressure.error` | The sensor error, or null after a valid read. |
| `process_peak_rss_bytes` | The observing process's lifetime RSS maximum on macOS, or null. |
| `process_peak_rss_error` | The unavailable or invalid RSS reason, or null after a valid read. |
| `swap.total_bytes` | The system's configured swap size, or null. |
| `swap.used_bytes` | The system's used swap size, or null. |
| `swap.available_bytes` | The system's available swap size, or null. |
| `swap.page_size_bytes` | The swap page size, or null. |
| `swap.encrypted` | The reported swap encryption flag, or null. |
| `swap.error` | The swap read error, or null after a valid read. |

Pressure uses `kern.memorystatus_vm_pressure_level` through the read-only `sysctlbyname` system interface.
The exact dispatch values are 1 for normal, 2 for warning, and 4 for critical.
These known states require their matching dispatch value and a null error.

Other values, invalid sizes, and system errors produce unknown pressure.
Unknown pressure requires a nonempty error and either a null dispatch value or an unknown unsigned 32-bit value.
Combined bit values do not represent a valid pressure state.
An unavailable observation must not become zero or normal.

Swap uses the public `vm.swapusage` structure.
RSS uses `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`.
The RSS value already uses bytes on macOS.
Other platforms produce a null RSS value without an assumed conversion.
These process and system values remain separate from allocator memory.

`host_memory.hardware_snapshot()` returns nullable `model` and `memory_bytes` fields, plus an `errors` map.
Each failed field has one entry in that map.
The function reads `hw.model` with a 256-byte limit and reads `hw.memsize` as an unsigned 64-bit value.
Missing hardware fields do not become invented defaults.

The calibration caller records `before_load`, `loaded`, or the current checkpoint phase with each pressure decision.
A rejected postload decision does not imply that model allocation never occurred.
Host snapshots do not add worker protocol fields or approve an admission estimate.

## Host and process controls

The tool requires ownership of the existing Metal measurement lock before model loading.
The tool must fail if it cannot acquire that lock immediately.
It keeps the lock until model cleanup or process exit.
The tool checks real macOS pressure before loading, after loading, and at sampling boundaries.
Only normal pressure permits further model execution.
Unknown values and sensor failures block execution.
The tool records the failing phase before cleanup.
It does not change system wired-memory settings or silently disable pressure checks.

The run records process peak RSS separately from MLX allocator counters.
Peak RSS is a process lifetime maximum, not current physical footprint.
The tool records system swap observations separately from model memory.
Neither measurement can attribute all host activity to the model.

## Sample journal

The tool appends observations to `samples.jsonl` and flushes each complete record.
Each record contains `record_index`, nullable `case_id`, `phase`, `monotonic_ns`, and `output_tokens`.
It also contains nullable `reported_prompt_position`, `memory`, `memory_error`, and `host`.
Before model loading, `memory` is null.
An unsampled output pressure failure records null memory and an explicit `memory_error`.

Journal timestamps use unsigned 64-bit nanoseconds, from zero through 18446744073709551615.
This offline range is independent of the worker protocol's safe-integer limit.
The producer and analyzer retain integer precision without floating-point conversion.
Journal timestamps cannot decrease within one run.
Their clock origin does not define an absolute date or permit comparisons between different runs.

Each case stores its first record index and record count in `samples`.
The artifact stores the journal filename, record count, and SHA-256 digest in `journal`.
Run samples store their record indices separately.
This arrangement avoids retaining all observations or rewriting them after every sample.
The observer's CPU memory and filesystem activity remain part of the instrumented process.

After successful process reaping, the parent recovers complete journal records and updates case ranges.
It retains malformed or incomplete bytes in the original journal and reports recovery errors.
The journal digest covers all retained bytes, including an incomplete final record.
An interrupted case never becomes completed from recovered samples alone.

The tool permits at most 4096 samples per case.
Exceeding that limit fails the case and stops the matrix.
It checks pressure at each progress callback and output yield.
It records output checkpoints at the first output and every sixteen outputs, plus terminal and settlement samples.

## Acceptance boundaries

The initial matrix covers prefill chunk boundaries, cache growth boundaries, and the configured context limit.
It also covers zero output, one output, longer output, prefill stopping, output stopping, and repeated requests.
The first request in a fresh process and later requests have distinct labels.
Clearing allocator cache does not establish a cold GPU hardware cache.

Valid raw observations require matching identity, complete case accounting, explicit reset intervals, and truthful cleanup results.
Successful completion requires loaded identity, input evidence, every planned case result, and successful request settlement.
Each completed case requires token digests, an output count within its allowance, and successful generator closure.
The parent checks these conditions after journal recovery and before it reports success.
An admission envelope requires a separate versioned contract and held-out validation.
It must define its supported model, profile, shape domain, margin, and identity mismatch behavior.
Incomplete cases, unexplained failures, or unconfirmed cleanup cannot establish an approved envelope.
The calibration tool does not automatically produce or approve such an envelope.

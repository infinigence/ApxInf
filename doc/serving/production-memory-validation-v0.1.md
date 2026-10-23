# Production memory validation profile 0.1

Status: P1 evidence procedure with an initial HTTP runner and CPU tests, without production results or approved limits.
This profile adds no worker fields, HTTP fields, or metrics.
It does not start batching, persistent caching, or another dependent stage.
The [contract](contracts-v0.1.md) and [local service profile](local-service-v0.1.md) retain authority.

## Execution boundary

Use the normal production HTTP path with one model and one active sequence.
Keep the approved model, precision, runtime, generation settings, and cleanup behavior.
Keep all synchronization that production already performs.
Do not call `inspect_memory` during production validation.
Do not insert `synchronize`, tensor evaluation, or peak resets for sampling.
Do not remove production synchronization to improve a result.

External observation can still affect scheduling and process memory.
Record the observation interval, collector versions, and collection overhead limits before execution.
Do not describe this procedure as observation without overhead.

## Identity and inputs

Freeze these records before each run:

| Record | Required evidence |
| --- | --- |
| Model and runtime | Bundle manifest, model revision, capability revision, provider, precision, package pins, and worker epoch. |
| Product and tools | Service binary hash, adapter hash, benchmark hash, external collector hashes, and exact arguments. |
| Host | Host model, device identity, physical memory, operating-system version, and operating-system build. |
| Environment | Pressure policy, memory settings, process limits, relevant environment settings, and competing workload policy. |
| Observation | Collector sources, units, timestamps, intervals, metric snapshots, and diagnostic retention policy. |
| Requests | Ordered request bodies, content digests, expected input bands, output allowances, deadlines, and stop conditions. |

Bind process samples to verified PID, start identity, parent identity, and worker epoch.
Record worker replacement as a new process identity.
Do not combine measurements from different identities into one request result.
Record actual input and output counts from available production results.
Do not treat estimated prompt lengths or planned output allowances as observed counts.

Synthetic token inputs from calibration do not establish equivalent production HTTP inputs.
Retain the production request bodies and their actual prepared lengths separately.

## Available observations

These sources have different scopes.
Keep their original names and units in the evidence.

| Existing source | Meaning and limit |
| --- | --- |
| `ready.memory` | Startup allocator `active_bytes`, `cache_bytes`, and `peak_bytes`. The readiness snapshot is not a current allocator sample. |
| `request_settled` diagnostic | Request identity, public and worker status, token counts, and `worker_metrics`. Diagnostic delivery can fail. |
| `worker_metrics.peak_memory_bytes` | Existing worker terminal peak. The worker reads it before resource settlement. It does not cover later cleanup allocations. |
| `apxinf_worker_peak_bytes` | A retained maximum from readiness and checked terminals across worker epochs. It is not an individual request peak. |
| External process observation | OS RSS or footprint for the identified gateway or worker. Record the counter source, units, and observation interval. |
| Host observation | Pressure and system swap. Normal pressure does not establish available headroom. Swap includes other processes. |

The current runtime resets its allocator peak when generation starts.
Zero-output requests do not start generation and do not establish a new generation peak.
Keep startup, generation, and inherited peak observations separate.
Do not subtract retained peak counters to calculate request memory.

Mark a missing, ambiguous, or failed peak observation as unknown.
Do not interpret a missing diagnostic or a fallback zero as zero memory use.
Retain diagnostic loss and write-error counters with the collected records.
Zero diagnostic loss does not prove that every record reached storage before exit.

The current production interfaces do not expose request cache payload or current allocator counters after settlement.
Do not claim those values from RSS, readiness, or the retained peak metric.
Keep these observations unknown when the required source is absent.
An external sampled maximum can miss allocations between samples.
Process lifetime peak RSS does not establish current RSS or a new request peak.
Do not report the collector's own RSS as worker RSS.
Do not add RSS, cache payload, allocator active, allocator cache, and allocator peak together.

## Decision table

Declare the required checks before execution.
A case passes only its declared checks, not an unrestricted memory domain.

| Check | Required evidence | Decision when evidence is absent or conflicting |
| --- | --- | --- |
| Identity | All required identities and settings match the frozen plan. | Reject comparison after a mismatch. Leave missing identity unresolved. |
| Production execution | The service uses its normal path without additional device sampling synchronization. | Reject the production claim after instrumentation changes execution. |
| Full output | Observed output reaches the planned allowance with a recorded terminal outcome. | Keep early EOS as completed execution but unresolved allowance coverage. |
| Stop boundary | Evidence identifies the actual stop position and subsequent cleanup. | Leave an unobserved stop boundary unresolved. |
| Runtime peak | The source identifies a valid request peak and its interval. | Keep missing or ambiguous peaks unknown. Never substitute an aggregate maximum. |
| External memory | Bound process samples include their units, timestamps, intervals, and failures. | Leave unsupported counters or collection gaps unresolved. |
| Resource settlement | Public outcomes and service observations establish checked settlement or successful process reaping. | Public expiry, cancellation, or disconnect alone does not pass. |
| Process reaping | Shutdown results and parent records show every owned worker exit. | Fail cleanup when reaping fails. Leave missing exit evidence unresolved. |
| Host pressure | The normal production pressure policy permits execution through the required observation period. | Record warning, critical, unknown, or stale pressure as blocked. |
| Candidate validation | A frozen candidate passes independent inputs within its declared domain and margins. | Leave acceptance unresolved without a candidate or sufficient evidence. |

Do not suppress EOS to reach an output allowance.
Do not extrapolate short output to a longer allowance.
Separate stop cases from full-output cases.
Preserve failures, pressure blocks, collection gaps, and unreached targets.

## Settlement and shutdown

Collect service metrics after the HTTP client finishes.
HTTP expiry or disconnect can precede worker cleanup.
Check `apxinf_active_requests`, `apxinf_queued_requests`, `apxinf_capacity_reserved_bytes`, and both operation values of `apxinf_cleanup_pending`.
For an isolated drained run, require all these values to reach zero.
Correlate counter changes with the owned requests and their recorded outcomes.
Do not infer per-request settlement from aggregate zeros during unrelated traffic.

Keep observation active until checked settlement or successful process reaping.
Record a bounded waiting period before execution.
Treat an exhausted waiting period as unresolved cleanup, not successful settlement.
Call `Service::shutdown` and await `Service::wait_stopped` when using the embedding API.
For the CLI, retain its shutdown exit result and the parent process report.
Check owned process identities before any cleanup action.
Retain the process-group membership result after reaping.
Process exit does not establish successful model execution or output coverage.

## Acceptance order

1. Freeze the production procedure, inputs, collectors, and required checks.
2. Retain calibration evidence separately from production observations.
3. Define a versioned candidate with its domain, margins, source artifacts, and identity mismatch behavior.
4. Freeze the candidate before independent input measurements.
5. Run independent inputs without fitting the candidate to their results.
6. Use new independent inputs after validation results change the candidate.
7. Report passed checks, unknown observations, blocked cases, and unresolved targets separately.

Candidate margins require measured justification.
This profile approves no numerical margin, memory reservation, or latency threshold.
Exploratory inputs do not become independent validation after candidate fitting.
The [independent input gate](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/memory-evidence-plan-20261010.md#independent-input-gate) also applies to stage 08.

Reserve the existing Metal device lock before model execution.
Do not run competing GPU work during a measurement.
Do not disable pressure checks or increase system memory settings to continue.
Stop progression after a pressure block, unexplained failure, invalid evidence, or incomplete cleanup evidence.
A blocked run can retain earlier observations, but the blocked case does not pass.
Successful checks do not approve an admission envelope without its separate contract and independent validation.

## Initial HTTP runner

`benchmarks/serving/production_memory_validation.py` attaches to an existing local service.
It sends ordered Anthropic Messages requests and records their original SSE bytes.
It does not start, signal, restart, or stop the service.
Process reaping remains `unknown` because the runner does not own those processes.
Its complete validation remains `incomplete`, even when all required case checks pass.

Supply a plan file, base URL, process IDs, service binary, diagnostic file, and new output directory.
Use a new run directory under `devlocal/<feature>/` or outside the checkout.
Also supply the existing Metal lock path and an inherited `--lock-fd` descriptor.
The runner checks the descriptor's inode and exclusive ownership before each request.
A Boolean ownership claim or a lock held by another descriptor cannot satisfy this check.
The launcher must retain the inherited descriptor until observation finishes.

The plan format is `apxinf-production-memory-plan-v1`.
Its fields are `format`, `expected`, `limits`, `environment`, and `cases`.
`expected` contains `model_revision`, `capability_revision`, `worker_epoch`, and `service_binary_sha256`.
The runner checks these values against readiness and the identified service executable.
It checks the executable's mapped device and inode against the hashed file.

The initial identity check covers those revisions, the worker epoch, the service executable, and the observed processes.
Device identity remains unknown in this runner.
The environment contains declared `process_limits`, `runtime_settings`, and `competing_workload_policy` records.
These declarations do not prove the service's environment or process limits.

Limits contain integer milliseconds for `request_timeout_ms`, `settlement_timeout_ms`, `observation_interval_ms`, and `observation_budget_ms`.
They also contain `max_sample_gap_ms`, `max_response_bytes`, `max_samples`, and `max_diagnostic_bytes`.
The runner records collection duration and gaps against these frozen limits.
Each case contains `case_id`, `body`, `input_tokens_min`, `input_tokens_max`, and `required_checks`.
The initial body permits plain text messages, greedy generation, and a positive output allowance, with streaming enabled.
Stop conditions and tool requests require a later runner extension.

Required check names are `identity`, `execution`, `input_band`, `full_output`, `settlement`, `host_pressure`, `external_memory`, and `runtime_peak`.
Each case must require identity, execution, settlement, and host pressure.
Case checks report `passed`, `failed`, `blocked`, `unreached`, or `unknown`.
An early EOS leaves `full_output` unreached but permits the next case after settlement.
Pressure blocks, failed checks, and required unknown observations stop later requests.
The runner retains all planned cases, including those it skips.

Before any request, the runner saves the source plan, frozen limits, readiness, process identities, tool hashes, and diagnostic offset.
The output contains `frozen-plan.json`, `observations.jsonl`, `diagnostics.jsonl`, one SSE file per executed case, and `report.json`.
The report records file digests and actual token counts from checked responses.
Readiness memory remains a startup snapshot, and the aggregate peak remains an aggregate metric.
Current allocator values after settlement and request cache payload remain unknown.

The external collector checks process identity before and after each RSS read.
It records `ps` RSS in KiB and converts that value to bytes on macOS.
It never uses the collector's own RSS as service memory.
PID, parent PID, start time, command, listener ownership, and worker epoch bind the observations.
These checks reduce attribution errors but cannot eliminate every operating-system identity race.

Host pressure must remain normal in both service readiness and the external host sensor.
The external pressure tuple must contain state `normal`, dispatch value `1`, and no sensor error.
The observation budget limits acceptable collection duration, not operating-system scheduling delay.
An excessive duration or sample gap leaves external memory checks unknown.

Settlement requires drained resource gauges and exactly one matching public outcome increment during the isolated case.
The final metric observation must start after the HTTP client finishes.
The checks include both `cleanup_pending` operations and reject missing metrics.
Unexpected outcome increments invalidate attribution to the case.

A diagnostic peak requires one matching request record, consistent usage, successful worker completion, and a positive peak value.
Missing, duplicate, invalid, or incomplete diagnostics leave that peak unknown.
Diagnostics alone never establish settlement.

The current diagnostic record contains no worker epoch.
The runner correlates its request ID with the response, diagnostic offset, and stable process observations.
These sources do not provide an atomic process snapshot.

Exit code zero means collection finished and each required case check passed.
It does not establish process reaping, candidate validation, or admission approval.
Other case outcomes return exit code one and preserve partial evidence.
Argument errors before collection return exit code two.
The operator retains shutdown and parent reaping evidence separately.

## Implementation references

- [Worker peak reset](../../python/apxinf/apxinf/serving/text_worker.py#L184) and [terminal before settlement](../../python/apxinf/apxinf/serving/text_worker.py#L687).
- [Service metrics](../../crates/apxinf-serving/src/supervisor.rs#L275) and [settlement diagnostic](../../crates/apxinf-serving/src/supervisor.rs#L1607).
- [Diagnostic loss counters](../../crates/apxinf-serving/src/diagnostics.rs#L131).
- [Shutdown contract](local-service-v0.1.md#metrics-and-shutdown).
- [Calibration acceptance boundary](memory-calibration-v0.1.md#acceptance-boundaries).
- [P1 gates](implementation-plan.md#p1-serial-streaming-and-exact-session-append).

This document applies structural ASD-STE100 principles.
Structural checks do not establish full dictionary compliance or certification.

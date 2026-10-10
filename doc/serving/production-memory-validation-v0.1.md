# Production memory validation profile 0.1

Status: proposed P1 evidence procedure, with no execution results or approved limits.
This profile defines no runner, worker fields, HTTP fields, or metrics.
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
The [independent input gate](memory-evidence-plan-20261010.md#independent-input-gate) also applies to stage 08.

Reserve the existing Metal device lock before model execution.
Do not run competing GPU work during a measurement.
Do not disable pressure checks or increase system memory settings to continue.
Stop progression after a pressure block, unexplained failure, invalid evidence, or incomplete cleanup evidence.
A blocked run can retain earlier observations, but the blocked case does not pass.
Successful checks do not approve an admission envelope without its separate contract and independent validation.

## Implementation references

- [Worker peak reset](../../python/apxinf/apxinf/serving/text_worker.py#L184) and [terminal before settlement](../../python/apxinf/apxinf/serving/text_worker.py#L687).
- [Service metrics](../../crates/apxinf-serving/src/supervisor.rs#L275) and [settlement diagnostic](../../crates/apxinf-serving/src/supervisor.rs#L1607).
- [Diagnostic loss counters](../../crates/apxinf-serving/src/diagnostics.rs#L131).
- [Shutdown contract](local-service-v0.1.md#metrics-and-shutdown).
- [Calibration acceptance boundary](memory-calibration-v0.1.md#acceptance-boundaries).
- [P1 gates](implementation-plan.md#p1-serial-streaming-and-exact-session-append).

This document applies structural ASD-STE100 principles.
Structural checks do not establish full dictionary compliance or certification.

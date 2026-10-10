# ApxInf serving implementation plan

Status: proposed work, version 0.1, 2026-10-09.

This plan defines the full implementation sequence.
The [serial profile](serial-profile-v0.1.md) defines the first implemented subset.
The [local service](local-service-v0.1.md) records its boundaries.
P1a code and CPU checks do not complete the P1 acceptance gate.
The plan applies structural principles from ASD-STE100 Issue 9. It does not claim full dictionary compliance or certification.

## Authority and implementation policy

[The contract](contracts-v0.1.md) defines the canonical semantics for serving. [The system design](../serving-system-design-20261009.md) records the research, architecture choices, and supporting evidence. This plan defines work order, module boundaries, and phase gates. A conflict requires a contract correction before implementation continues.

Contributors write the implementation directly from the ApxInf contract. Contributors do not copy, translate, or vendor implementation code from external serving engines.
Public dependency APIs remain permitted. Each adapter records its dependency versions, API calls, and observed limitations. The project does not use code generators for serving types, adapters, or test stubs.

Rust and Python use separate handwritten types. Both implementations follow the same contract and original conformance fixtures. The fixtures contain input cases, expected events, and expected resource outcomes. They do not define a second semantic specification.
Changes to fields, states, or ownership rules start in the contract. The same change includes the affected types and fixtures.

The existing service v1 and session v1 interfaces remain frozen. New streaming and control events use a separate protocol version. Existing commands retain their documented behavior.

## Proposed module boundaries

Every path in this table describes a future module. The first implementation creates only modules that the serial service needs.

| Proposed path | Owner and boundary |
| --- | --- |
| `src/serve.rs` | This entry point reads configuration and connects the components. |
| `crates/apxinf-serving/src/contracts.rs` | These handwritten types represent the canonical contract in Rust. |
| `crates/apxinf-serving/src/gateway.rs` | Gateway owns HTTP parsing, response formatting, and bounded output streams. |
| `crates/apxinf-serving/src/coordinator.rs` | Coordinator owns the request registry, admission decisions, and deadlines. |
| `crates/apxinf-serving/src/model_supervisor.rs` | ModelSupervisor owns worker identity, readiness, drain, restart, and process recovery. |
| `crates/apxinf-serving/src/device_arbiter.rs` | DeviceArbiter owns capacity reservations and compute permits for each device. |
| `crates/apxinf-serving/src/prompt_adapter.rs` | PromptAdapter owns templates, tokenization, media normalization, and effective input length. |
| `crates/apxinf-serving/src/state_store.rs` | StateStore owns snapshot identity, references, byte accounting, and eviction policy. |
| `crates/apxinf-serving/src/metrics.rs` | This module records contract events and separate latency intervals. |
| `crates/apxinf-serving/src/workers/mod.rs` | This module defines handwritten interfaces for TextWorker and PolicyWorker. |
| `crates/apxinf-serving/src/workers/mlx.rs` | This adapter owns the new MLX protocol connection. |
| `crates/apxinf-serving/src/workers/native.rs` | This adapter owns native model execution on its assigned thread. |
| `python/apxinf/apxinf/serving/contracts.py` | These handwritten types represent the same contract in Python. |
| `python/apxinf/apxinf/serving/text_worker.py` | TextWorker owns model execution, active sequence state, and dependency calls. |
| `python/apxinf/apxinf/serving/policy_worker.py` | PolicyWorker owns policy execution and observation deadlines. |
| `tests/fixtures/serving/` | Original fixtures describe cases that both languages consume. |
| `tests/serving_contracts.rs` | Rust tests consume the shared fixtures through public boundaries. |
| `tests/python/test_serving_contracts.py` | Python tests consume the same fixtures through public boundaries. |

Gateway does not access tensors. Coordinator does not schedule individual MLX tokens. TextWorker retains one execution owner for each model instance. PolicyWorker uses an independent execution path.
StateStore handles opaque state references. Provider adapters define physical state layouts and execute state operations.

ModelSupervisor starts workers only after DeviceArbiter grants the required resources.
Load, warmup, execution, and restore obey the same device ownership rules.

## Dependency order

P0 precedes P1. P1a provides the reliability foundation for stateless P2a, P2b, and P2c work. These three branches can deliver independently. P2a takes priority when the first target workload uses a coding Agent.

P1b provides exact session append. Features that use explicit sessions require P1b.
P2b and P3 also require measured memory estimates, fixed latency targets, and model quality gates.
Passing P1a does not satisfy these additional gates or complete P1.

P3 depends on stable identity and lifecycle contracts. P3 does not require P2b when the serial adapter supports snapshots.
Each P4 feature requires its own measured benefit. P5 does not block the Mac service.

## P0: specification and conformance table

P0 defines component interfaces before the first runtime skeleton. It also records the reference environment and future acceptance measurements. The environment record includes hardware, model bundle, quantization, runtime versions, and the target client. The baseline compares existing ApxInf behavior with the proposed service under the same workload.
Numerical latency and memory targets require measurements. The plan does not substitute estimates for these measurements.

The conformance table maps each contract requirement to a future shared fixture and an observation.
Initial cases cover successful requests, malformed events, cancellation, timeouts, resource recovery, session conflicts, and unsupported capabilities. Additional cases cover duplicate terminal events, stale worker epochs, missing cleanup acknowledgement, and output congestion.
The table distinguishes a logical terminal event from acknowledged resource release.

P0 exit checks:

- Review each component interface against the canonical contract.
- Record the owner of every mutable state object.
- Record the release condition for every reservation, credit, and permit.
- Map each required behavior to an original conformance case.
- Record the baseline procedure and unresolved acceptance values.
- Freeze acceptance values from measurements before performance claims or default optimization changes.

## P1: serial streaming and exact session append

P1 contains two separate gates: P1a for stateless reliability and P1b for exact session append.
Existing P1a evidence covers only the checks that its validation report records.
The reliability changes below require new verification records before their gate closes.

P1 creates the smallest complete service through Gateway, Coordinator, ModelSupervisor, DeviceArbiter, PromptAdapter, and TextWorker. The initial deployment uses one model instance and one active sequence. Gateway accepts additional requests through a bounded queue.
The service exposes the selected text endpoint, SSE output, request errors, model identity, readiness, and metrics.
PromptAdapter calculates input length before Coordinator checks context capacity and output reservation.

The MLX adapter retains the existing runtime identity checks. Its new protocol separates output events from control messages. An independent control reader receives cancellation while the execution owner processes a request.
TextWorker applies cancellation at a documented execution boundary. It acknowledges cleanup only after device work and state references permit release.

P1 holds the compute permit for the complete request. A cancellation request alone does not return that permit.

The existing exact append path retains its current restrictions and byte limit semantics. The service reports the actual execution path. It does not imply snapshot, fork, or automatic prefix support.
Gateway buffers output within a fixed limit. A slow connection cannot block the execution owner indefinitely.

P1a adds planned worker rotation before command history exhaustion.
Rotation preserves approved model and capability identities and exposes the replacement epoch.
It waits for old process exit before replacement loading and does not replay completed requests.
Other worker faults retain their documented manual restart policy.

P1a separates public preparation expiry from operation settlement.
A cancelled or expired request ends before the supervisor consumes its late preparation response.
The supervisor retains the execution position for up to 20 additional seconds.
It dispatches no later request until settlement or worker recovery completes.
Independent queue sweeps remove cancelled and expired entries every 20 milliseconds and before enqueue.

Preparation checks include stdin transmission, with a separate 20-second write health timeout.
Unconfirmed transmission or response retains the execution position until settlement or worker recovery.
P1a also requires explicit host shutdown and complete public result counters.
These changes retain protocol `apxinf-worker/2.0` and its existing wire fields.

P1a reliability exit checks:

- Exercise command history exhaustion through planned rotation without request replay.
- Check replacement identities, epoch visibility, and old process exit before replacement loading.
- Check immediate public expiry during preparation and safe disposal of its late response.
- Exercise cancellation during preparation and worker fencing after the 20-second settlement grace.
- Check queue capacity recovery during active execution, FIFO order, and one counter update per result.
- Exercise blocked stdin writes while public deadlines and cancellation remain active.
- Check expiry counters, successful rotation counters, and separate worker fault counters.
- Check complete readiness snapshots after rotation, including the replacement memory values.
- Check rotation after command exhaustion while pressure prevents replacement loading.
- Check queue expiry and shutdown during pressure waits without false worker faults.
- Check replacement process reaping before pressure retries and after loading failures.
- Exercise explicit service shutdown and accepted worker shutdown with stdin still open.
- Repeat streaming, disconnect, resource settlement, worker failure, and v1 regression checks.

P1a observability uses the [observability contract](contracts-v0.1.md#observability-contract) before implementation.
It measures queue waiting, preparation, first output, event intervals, public completion, and resource settlement separately.
The worker protocol remains `apxinf-worker/2.0` with no new fields.
These measurements support later latency targets and memory experiments without establishing their acceptance values.

P1a observability exit checks:

- Check queue durations for execution, cancellation, and expiry.
- Check preparation duration through blocked transmission, late responses, and successful process recovery.
- Distinguish first worker output from first public output readiness without claiming client TTFT.
- Check event aggregation, hidden output, and absent output without invented timing samples.
- Check both terminal-to-settlement durations and pending cleanup after failed process recovery.
- Check service duration and one public outcome for generation and token counting.
- Check operation totals against legacy status counters.
- Check fixed cumulative histogram buckets, units, and permitted labels.
- Preserve existing metric names, meanings, and worker fields.

Generation deadline exit checks:

- Check expiry before acceptance, during generation, during control transmission, and after worker completion before cleanup.
- Preserve the deadline after a stop control without sending a second control.
- Suppress late public output while checking worker frames and recording worker measurements.
- Retain resources and queue order until settlement or process reaping.
- Record each public outcome once, including a later worker fault.
- Preserve public success only after checked settlement before the deadline.

Diagnostic isolation exit checks:

- Check worker fencing, request settlement, and process exit with closed or blocked diagnostic output.
- Bound record size, queue capacity, and worker stderr reads.
- Check diagnostic loss counters without additional diagnostic writes.
- Preserve worker stdout checking and resource recovery when diagnostic output fails.

P1a host pressure checks use the [host pressure contract](contracts-v0.1.md#host-pressure-admission).
The local profile specifies sensor values, freshness, recovery, and HTTP behavior.
This guard supplements the fixed reservation without completing the measured memory estimator gate.

P1a host pressure exit checks:

- Check normal, warning, critical, invalid, failed, and stale sensor results.
- Check two-sample recovery, including stale gaps and repeated reads.
- Check rejection before enqueue, preparation, generation, and model loading.
- Check active settlement, queue deadlines, readiness, shutdown, and worker faults under pressure changes.
- Check disabled policy visibility and bounded metric labels.
- Check the real sensor without creating artificial host memory pressure.

The [memory calibration profile](memory-calibration-v0.1.md) defines the next estimator evidence step.
Its raw observations cannot complete the measured memory estimator gate by themselves.

Memory calibration exit checks:

- Check generation stream synchronization before state release and allocator cleanup.
- Preserve cache payload, allocator counters, and process memory as separate measurements.
- Check input, output, consumed position, and cache offset accounting at each checkpoint.
- Preserve blocked, failed, skipped, and interrupted cases with their cleanup results.
- Exercise chunk boundaries, context boundaries, stopping, and repeated requests under the approved execution profile.

The [memory coverage profile](memory-coverage-v0.1.md) defines offline evidence checks before estimator design.
It distinguishes completed runs from reached shapes and retains missing input and output intervals.
Neither complete planned cases nor complete shape enumeration establishes a universal memory bound.

### Planned long-context support

Long-context serving is a future capability goal.
This plan does not enable a larger context profile or change an interface limit.
The current deployment defaults to 16384 context tokens and 2048 output tokens.
These limits do not establish measured memory coverage for every permitted request.
The protocol ceiling does not establish a supported inference length.

First complete measurements within the current deployment profile.
Then evaluate larger context profiles against explicit quality, latency, and memory criteria.
Each profile must retain its model, runtime, input, and observation identities.
Exact sessions, batching, and snapshots retain their separate acceptance gates.

Long-context exit checks:

- Check prepared input plus reserved output below, at, and above the effective context limit.
- Check KV capacity growth and additional prefill calls at their actual execution boundaries.
- Exercise long input, maximum output allowance, late stopping, and context-edge requests.
- Compare fresh-process requests with repeated requests in one process.
- Keep early EOS targets unresolved instead of suppressing EOS to reach an allowance.
- Check cancellation, deadlines, backpressure, settlement, and process recovery during long requests.
- Measure production HTTP execution without additional device synchronization for memory sampling.
- Freeze estimator scope and margins before independent input validation.
- Check model quality and client context metadata before publishing a larger capability profile.
- Check mixed short and long workloads when a future batch adapter passes its own correctness gate.

The [memory evidence plan](memory-evidence-plan-20261010.md) defines the current measurement sequence.
The [production validation profile](production-memory-validation-v0.1.md) defines the separate production evidence boundary.
Neither document approves a memory budget or promises a larger supported context size.

### Remaining P1 checks

P1b exit checks:

- Define the exact append profile before implementation.
- Check session identity, version conflicts, exclusive writes, and state invalidation after partial mutation.
- Compare appended generation with the fixed fresh-generation reference.

P1 exit checks:

- Compare serial output with the fixed reference profile.
- Run the existing CLI and v1 regression checks.
- Exercise disconnects, full queues, expired deadlines, and slow connections.
- Check one logical terminal outcome for each accepted request.
- Check resource recovery after normal completion, cancellation, and worker failure.
- Check exact append isolation and explicit session conflict errors.
- Measure queue time, first output latency, output intervals, and memory use separately.

## P2a: Agent compatibility

P2a extends PromptAdapter and Gateway for one selected Agent client. The adapter represents tool definitions, tool results, reasoning fields, and completion reasons according to that client contract.
The model template and incremental parser form one tested compatibility profile.
An OpenAI chat endpoint alone does not establish compatibility with Responses or Anthropic Messages.

P2a exit checks:

- Complete a real tool request and tool result cycle with the selected client.
- Exercise fragmented tool arguments, stop sequences, and cancellation.
- Check unsupported parameters produce explicit errors.
- Preserve exact session append only when the rendered token prefix matches.

## P2b: MLX continuous batching

P2b uses public `mlx_lm.generate.BatchGenerator` APIs through the original TextWorker adapter.
The adapter records the pinned version and supported model cache types. Dependency upgrades require separate quality and boundary checks.
The worker uses bounded advancement through `next()` when the pinned API supports that boundary.
It does not treat `next_generated()` as proof of bounded prefill latency.
The prefill budget includes the number of prompts and the chunk size for each prompt.

The adapter records constructor effects on memory limits and generator lifetime.

Coordinator limits admission through worker credits. TextWorker owns insertion, removal, and token scheduling within the batch.
DeviceArbiter transfers a compute permit between steps only after the adapter reports completion of prior device work.
An exact append session retains the serial adapter until the batch adapter proves compatible state transfer.

P2b exit checks:

- Freeze memory estimator limits, latency targets, and quality criteria before performance acceptance.

- Compare model quality across serial and batch profiles.
- Exercise sequence insertion, completion, cancellation, and removal during prefill and decode.
- Measure short requests beside long prompts at several concurrency levels.
- Check bounded waits, output latency, and memory use against P0 targets.
- Reject unsupported cache, sampling, and model combinations explicitly.

## P2c: serial vision requests

P2c adds one vision model through PromptAdapter and a serial TextWorker profile. PromptAdapter identifies image content, processor version, insertion positions, and effective context length.
The worker reports media memory separately from text state. PolicyWorker does not process vision chat requests.
The first profile uses only the selected model's public dependency APIs.

P2c exit checks:

- Compare image processing and model output with the selected reference.
- Exercise media limits, context limits, cancellation, and repeated requests.
- Check position state and memory accounting for each supported input shape.
- Keep vision batching and state reuse disabled until their separate capability checks pass.

## P3: immutable snapshots in memory

P3 adds snapshot creation, restore, and prefix lookup through StateStore.
StateStore uses complete model identity and the provider's state schema. It pins references during active use. The provider reports the common restore boundary across attention, convolution, recurrent, and position state.
The position contract distinguishes consumed input tokens from emitted output tokens and any pending token.

Snapshot support does not imply arbitrary trim, fork, or shared execution buffers. Shared snapshots remain immutable. A request owns each mutable continuation.

P3 exit checks:

- Freeze snapshot memory estimates, latency targets, and quality criteria before cache acceptance.

- Compare restored continuation with uncached continuation at each supported boundary.
- Exercise branch creation, eviction, cancellation, and failed restore.
- Reject incompatible model identities, state schemas, and incomplete state components.
- Check reference recovery and byte accounting after each terminal path.
- Measure saved prefill work and Agent latency against snapshot costs.

## P4: SSD storage and multiple models

P4 contains independent features for SSD snapshots and model residency. Actual workloads determine their delivery order. SSD snapshots use atomic publication, integrity checks, version checks, and storage quotas.
ModelSupervisor combines duplicate load requests and applies TTL or LRU policy only to eligible model instances.
DeviceArbiter counts resident weights, active state, snapshots, temporary tensors, and runtime reserves against one physical memory budget.
ModelSupervisor limits new admission when another model waits for device access.

Logical storage blocks organize snapshots. They do not establish kernel paging or zero-copy sharing during execution.

P4 exit checks:

- Measure restore cost against recompute cost before enabling SSD reuse.
- Exercise interrupted writes, incompatible snapshots, full storage, and recovery.
- Measure model switching under memory pressure and repeated client requests.
- Check bounded waits during model drain and load failure.

## P5: native batching and platform extensions

P5 separates native weights from request state before introducing multiple active sequences.
Native batch interfaces remain within model and runtime modules. Backend operator interfaces do not own request admission.
Kernel paging requires explicit attention kernel support and its own correctness and performance evidence.
CUDA, Jetson, and Metal profiles each have independent capability and quality gates.

PolicyWorker adds observation freshness, bounded inputs, and deadline handling to the existing policy path. Process isolation alone does not establish a real-time execution guarantee.

P5 exit checks:

- Check state independence before enabling native concurrency.
- Check each platform against its fixed model and quality profile.
- Measure policy deadline behavior under the intended device workload.
- Retain the accepted serial profile when a new profile fails its acceptance gate.

# Serving support and verification boundaries

The serial service implements explicit deployment profiles.
Each profile declares its provider, model identity, limits, and worker protocol.
The [local service guide](local-service-v0.1.md) defines startup and public HTTP behavior.
The [interface contract](contracts-v0.1.md) and profile documents define request and settlement semantics.

## Execution scope

| Profile | Execution owner | Supported scope and limits |
| --- | --- | --- |
| MLX-LM text | Separate Python worker and external MLX-LM runtime. | Serial text generation under the checked runtime identity. |
| Native MiniCPM5 | ApxInf Rust model with the native MLX SDK. | Explicit greedy, sampling, or thinking deployment profiles. Each profile retains its own limits. |
| Native prepared MiniCPM5 | ApxInf Rust model with the native MLX SDK. | One ordinary BF16 greedy sequence and one retained code entry. Request KV is fresh. |
| MiniCPM-V | Separate Python worker and external MLX-VLM runtime. | The declared PNG profile, checked dependency state, and serial requests. |
| Clef text | Separate Python worker and the checked external Clef loader. | Structured text decisions for the declared question types. This is not the native Metal Clef model. |

Sessions, shared prefix reuse, and continuous batching remain outside these serial profiles.
An external provider does not establish native ApxInf model support.
Native prepared execution does not qualify sampling, thinking, vision, or DSpark.
Model-family limits and deployment limits are distinct.
The effective request limit is their checked intersection, including reserved output tokens.

## Source-bound observations

The [serving PR](https://github.com/infinigence/ApxInf/pull/116) records historical text, vision, Clef, and client observations.
Those observations span different source and runtime revisions.
They are not one qualification matrix and must not be combined into one acceptance claim.
Their development records remain separate from the maintained interface documents.

The [native runtime PR](https://github.com/infinigence/ApxInf/pull/183) reports M4 checks for source `73c1f521a18657873414bea07191fc89c78fe267`.
The reported scope is ordinary MiniCPM5 BF16 greedy execution, direct and HTTP output comparison, cancellation, and allocator settlement.
These observations do not establish a paired performance improvement, a process-RSS bound, or qualification of other providers.
This documentation cleanup does not rerun or extend those device observations.
Later runtime source changes require verification for their own exact candidate.

Earlier parameter trials observed output differences across prefill chunk selections.
Those trials did not justify a general performance or quality claim.
Historical client trials also include failed drawing and expense tasks.
Successful HTTP processing, correct tool arithmetic, and released request resources do not establish task completion.

## Remaining qualification

The [context acceptance profile](context-acceptance-v0.1.md) separates token capacity from task quality.
Long input admission does not establish factual recall or general long-context quality.
Thinking-budget control does not establish drawing or tool-use quality.
Vision transport and image grounding do not establish successful image-driven task revision.
Clef diagnostic requests do not establish admission under the default host-pressure policy.

CPU contract tests check schemas, shared fixtures, protocol ordering, and controlled failure paths.
Test workers do not prove model computation, dependency identity, or device cancellation.
Actual model and client checks use the declared checkpoint, runtime, source, inputs, and generation policy.
Keep numerical comparisons, complete task outcomes, performance measurements, and resource observations separate.

## Memory evidence

The [calibration profile](memory-calibration-v0.1.md) defines synchronized diagnostic observations.
The [coverage profile](memory-coverage-v0.1.md) distinguishes planned, executed, stopped, and reached cases.
The [production profile](production-memory-validation-v0.1.md) retains normal execution and cleanup synchronization.
Calibration synchronization can change overlap and allocation peaks.
It does not replace production observations.

Allocator active bytes, cached bytes, peaks, logical cache payload, process RSS, and system swap describe different quantities.
A cache-size formula or a normal pressure signal does not prove physical memory headroom.
An observed stop does not establish execution through the original output allowance.
Inherited peaks do not establish a new case peak.
No historical report approves a general admission envelope or process physical-memory guarantee.

## Reproduction

The [benchmark guide](../../benchmarks/serving/README.md) describes the maintained clients and calibration tools.
Use the actual profile's public entry and retain complete inputs, outputs, identities, and cleanup results with the run.
Raw outputs, environment snapshots, traces, and experiment histories belong in the ignored development workspace.
The source revision and Git blobs identify repository files.
Private evidence paths are not portable links for a fresh checkout.

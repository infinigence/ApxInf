# Core ML and Apple Neural Engine execution (proposal)

Status: design proposal. No Core ML code is merged. This document asks
maintainers to agree on ownership, scope and evidence before implementation.

## Motivation

Apple Silicon has a Neural Engine (ANE) next to the GPU. This proposal uses
the public Core ML framework to request ANE execution. A model can split one
stage between MLX on the GPU and Core ML. Whether the Core ML share runs on
the ANE requires the placement evidence defined below.

The first candidate is Clef Flash on the M1 Pro. A GPU+ANE composition keeps
the optimized GPU kernels and moves 5632 of each MLP's 12288 intermediate
columns to the ANE, as per-layer int8 programs. Owner evidence (M1 Pro,
2026-10-17, same session after a reboot):

| Implementation | 32-layer complete request, weighted |
| --- | --- |
| GPU only | 3603 ms |
| GPU + ANE | 2790 ms (1.29x) |

The GPU+ANE composition passed the owner's qualification with two recorded
accepted decision changes. Its MLX peak memory is 8.68 GB, against 7.22 GB for
the GPU-only composition.

The reference implementation generates the ANE programs with Python
coremltools. ApxInf does not execute Python, so a port needs its own program
writer.

## Requirements

- No Python and no coremltools at build time or run time.
- Native execution on Apple Silicon macOS only. On other targets the feature
  is inert, as the `mlx` feature is.
- No silent acceptance of an unqualified path. Compilation or loading errors
  fail preparation. Successful loading alone does not qualify ANE execution.
  An explicitly requested GPU+ANE path must fail preparation or request admission
  when its placement evidence or approved host-handoff exception is missing.
- No application compilation during a request. Native framework readiness
  also needs warm execution evidence, not merely a sealed wrapper.
- Qualification binds the program, hardware, OS, configuration and fixed
  profile, not just the Metal architecture name.

## Scope

In scope:

- A crate, provisionally `apxinf-coreml`, behind a `coreml` feature.
- A Rust writer for ML Program packages (specification version 9, opset
  `CoreML8`) that covers a fixed list of operations.
- An Objective-C++ bridge that compiles, caches, loads and runs these
  packages through the Core ML framework.

Out of scope:

- General model conversion and arbitrary MIL operations.
- An implementation of the `Backend` trait.
- Dynamic shapes and stateful programs.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `apxinf-coreml` program writer | Typed operation builder for the supported list, ML Program protobuf, MILBlob weight file, package layout |
| `apxinf-coreml` runtime | Compile, content-addressed cache, load with requested compute units, seal after load, FP16 `MLMultiArray` host input and output with explicit strides, worker-thread prediction, `NSError` conversion |
| `apxinf-mlx` | Asynchronous evaluation, so the GPU share runs while the ANE predicts; host transfer of the exchanged tiles |
| Model family | What runs on the ANE, host-side quantization and its numerics, tiling, the GPU/ANE overlap schedule, residency budget, architecture policy |

The family composes MLX arrays and Core ML programs. Neither crate depends on
the other.

## Program writer

The protobuf messages are a hand-written subset of coremltools 9.0
`Model.proto`, `FeatureTypes.proto` and `MIL.proto` (BSD-3-Clause). They use
`prost` derive macros, which the workspace already locks, so no `protoc`, build
script code generation or new dependency is needed.

Operations: `const`, `constexpr_blockwise_shift_scale`, `conv`, `mul`, `add`,
`tanh`, `real_div`, `maximum`, `abs`, `reduce_max`, `split` and `identity`.
These are the operations in the GPU+ANE composition's programs. The builder rejects any other
operation. A new operation is added with its own test.

The weight file uses MILBlob storage version 2:

- a 64-byte header with the blob count;
- for each blob, 64 bytes of metadata (sentinel `0xDEADBEEF`, data type, size,
  data offset), then the data, aligned to 64 bytes.

A `blobFileValue` offset in the program points at the blob's metadata.

**Equivalence target.** coremltools 9.0 writes protobuf maps in hash order,
so its `model.mlmodel` cannot be reproduced byte for byte. The reference is
coremltools 9.0 output from `mb.program` followed by
`ct.convert(convert_to="mlprogram", compute_precision=FLOAT16,
minimum_deployment_target=iOS18)`. For the same weights, the writer's output
matches it in three ways:

- the decoded program, excluding writer metadata and `buildInfo`;
- the weight file, byte for byte;
- prediction outputs, bit for bit, on the same device.

The tests use recorded coremltools packages. The packages themselves are not
committed, because they contain checkpoint-derived weights.

## Runtime

- **Compile.** Core ML compiles a package with `MLModel compileModelAtURL:`.
  The compiled `.mlmodelc` is stored in a cache directory. The key is the
  program and weight digests plus the macOS build. The writer creates the
  entry in a temporary directory and renames it into place, so concurrent
  loads never read a partial entry.
- **Load.** The model is loaded with the compute units the family requests
  (the GPU+ANE composition uses `MLComputeUnitsCPUAndNeuralEngine`). After loading, the runtime is
  sealed: a later application compile request is an error. Loaded and sealed
  does not mean placement-qualified or ready for requests.
- **Predict.** Inputs and outputs are FP16 host buffers. Output multi-arrays
  are read with their reported strides, because the ANE may pad them.
  Prediction runs on a worker thread that the runtime owns. MLX arrays never
  cross threads, so MLX's thread affinity is preserved.
- **Memory.** The ANE keeps only a limited amount of program weights resident;
  about 2.4 GB was observed on the M1 Pro. Exceeding it reloads programs on
  every switch. The family owns the per-layer budget.

### Stage placement and permitted CPU work

Apple's [CPU and Neural Engine compute-unit option](https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine)
permits both processors and excludes the GPU from that Core ML model. It
does not guarantee ANE placement. The public compute-unit API has no
ANE-only setting. This proposal does not promise enforced ANE-only execution
for every invocation.

The family must define a placement ledger for every admitted fixed profile:

| Stage | Required placement and CPU allowance |
| --- | --- |
| GPU share and remainder of the model | MLX GPU computation, including the merge of the two MLP shares. No new CPU layer mathematics. |
| Core ML share | The declared 5632-column MLP share in each of 32 layers. All runtime tensor arithmetic, including projections, activation, reductions and arithmetic quantization operations inside the program, requires ANE placement evidence. |
| Host boundary | Only the declared tile copies, stride-aware packing/unpacking and per-row symmetric int8 quantization with FP16 scales and round-half-even. Record exact formulas, extents and frequency. This is a proposed exception, not permission for arbitrary CPU operators. |
| Preparation and orchestration | CPU weight conversion, constant materialization before readiness, static metadata, validation, scheduling and error handling are permitted. They do not authorize moving runtime layer mathematics to the CPU. |

The ledger must map source operations to the compiled structure, including
fused or eliminated operations. Any CPU work inside Core ML must be listed
by operation and purpose and limited to non-arithmetic data movement or
metadata. Unknown mappings or unlisted CPU operations block qualification.
The writer's supported-operation list is not a CPU-placement allowlist.

### Placement qualification

Preparation must inspect the compiled `.mlmodelc` with the same
`MLModelConfiguration` used to load the model. The public Objective-C
[`MLComputePlan`](https://developer.apple.com/documentation/coreml/mlcomputeplan-85vdw?language=objc)
API provides `loadContentsOfURL:configuration:completionHandler:` and
`computeDeviceUsageForMLProgramOperation:`. Its
[`MLComputePlanDeviceUsage`](https://developer.apple.com/documentation/coreml/mlcomputeplandeviceusage)
reports `supportedComputeDevices` and `preferredComputeDevice`. For each
ANE-required operation, the proposed gate requires ANE support and ANE
preference. A missing plan, unknown usage, unavailable API or unacceptable
CPU assignment leaves the path unqualified. `MLComputePlan` is available
from macOS 14.4; the program format may require a newer deployment target.

A compute plan describes anticipated placement, not an execution certificate.
Qualification also requires saved execution traces for every admitted profile
and per-layer program through the complete request, including program switches
and GPU overlap. Apple's [Core ML and Neural Engine Instruments workflow](https://developer.apple.com/videos/play/wwdc2022/10027/)
can correlate Core ML compute requests with hardware activity. Evidence must
identify the traced program and stage, establish ANE activity for its required
computation and account for CPU work against the ledger. Aggregate ANE
utilization, prediction success or a preferred-device report alone is not
enough. If the available diagnostics cannot resolve the required attribution,
qualification remains blocked rather than inferring placement.

The supported claim is measured ANE execution for an identified profile and
environment, not a guarantee about every future prediction. Public APIs do
not provide a strict per-invocation ANE-only enforcement mechanism here. If
that stronger guarantee is required, this design cannot satisfy it and must
remain unqualified. Qualification records must state the trace coverage and
its limits.

Until the evidence and maintainer decision below exist, report the path as
unqualified with the blocking reason. An explicit GPU+ANE request must fail, not
silently select CPU-only Core ML or the GPU-only composition, which remains a separate
selection. Recheck the plan during preparation and invalidate qualification
when any bound identity, profile or configuration changes, or when contrary
placement evidence appears. These are proposed acceptance rules, not an
implemented status API or a claim that loading enforces placement.

## Proposed host-handoff exception

The canonical [accelerator port acceptance contract](model-execution-wiring.md#accelerator-port-acceptance)
excludes external execution plans and requires intermediate GPU residency.
The [MLX prepared execution contract](mlx-backend.md#prepared-compiled-execution)
also requires device-resident handoff. The GPU+ANE composition's native Core ML programs and host
tile exchange do not satisfy those rules as written.

This document requests a maintainer decision for only the fixed-profile
`metal_clef` GPU+ANE composition described here. The proposed exception would
permit the native Core ML stage and the bounded host work in the placement
ledger. It does not permit arbitrary external engines, CPU model execution
or unbounded intermediate transfers. GPU-only and CUDA requirements remain
unchanged, including device residency and CUDA Graph capture. MLX stages
outside the listed boundaries retain their existing preparation requirements.

Approval is pending. Before calling this a completed heterogeneous port,
maintainers must record the accepted scope and evidence and link that decision
from the canonical acceptance contract. Until then the existing contract
remains authoritative and the exception is an acceptance blocker. This design
change neither implements nor approves the exception.

The approval record must include the following evidence, with immutable
artifact digests and replay instructions. Missing budgets or evidence fail
the gate; a compile/load/seal/predict smoke test is not sufficient.

| Acceptance boundary | Required record and checks |
| --- | --- |
| Source and binary | Exact ApxInf and reference source revisions, dirty patch digest if present, lockfile, build command, toolchain, SDK and feature flags; executable and loaded native-library digests; checkpoint, package, weight and compiled-artifact digests. Bind the traces and outputs to those bytes, not merely to a current clean head. |
| Device and configuration | Physical Mac model and SoC, macOS build, Core ML configuration and function, MLX version, device identity and dispatch overrides. Bind the saved plan and execution traces to the same hardware and configuration. Metal architecture strings are selection hints, not ANE attestation. |
| I/O and transfers | Every public input/output and exchanged tile: shape, dtype, strides, byte extent, source/destination, direction, frequency and numerical conversion. Declare maximum copies, bytes, synchronizations, queue depth and live host/device memory per layer and complete request, then compare observations to those limits. Include packing, quantization and padding. Unified memory does not make these handoffs free; distinguish logical transfers from measured physical traffic and mark opaque framework copies as unknown. |
| Ownership and lifetimes | Name the owner of every host buffer, `MLMultiArray`, MLX array, loaded program and worker task, with preparation/request/layer lifetime and reuse point. Retain immutable inputs through prediction completion. Read GPU-produced input only after producer completion; release or reuse output storage only after its consumer completes. Validate dtype, dimensions, strides and accessible byte bounds before access. Keep MLX objects on their owning thread. Test alias rejection, retained outputs, changed inputs, cancellation, errors and teardown while work is in flight. |
| Warm readiness | Prepare all admitted shapes and programs, then exercise repeated complete requests and layer switches. Record cold compilation separately, application compile-call counts, observable framework compilation/load events, resident-program churn, allocator peaks and retained memory. Require the declared warm bounds with no observed hot compilation or repeated load churn. A sealed wrapper proves only that it rejects application compile calls; insufficient native visibility blocks the stronger readiness claim. |
| Complete request | Use the maintained public path from real input through final output, including all layers, boundary CPU work, transfers, overlap waits and state updates. Check the writer equivalence target, declared intermediate/final numerical tolerances, changed inputs, reset, invalidation and failures. Record any accepted decision changes with maintainer approval. Pair GPU+ANE and GPU-only timings on the same source/environment and workload, with raw samples and actual output completion at the timing boundary. Predeclare any performance release threshold. |

The initial host-buffer design must not claim zero-copy transfer or direct
GPU/ANE storage sharing. For caller-backed `MLMultiArray` storage, the wrapper
must retain its backing owner and use an explicit deallocation policy.
Core ML's [multi-array interface](https://developer.apple.com/documentation/coreml/mlmultiarray)
does not make a borrowed pointer outlive its storage. Scoped byte-access
pointers must not escape their access block. Copy results to owned handoff
storage before passing them across a worker boundary when ownership cannot be
retained safely. Account for that copy in the transfer budget.

## Numerics and qualification

ANE arithmetic is FP16, with device-specific reductions. Outputs are not
bit-equal to the GPU and can differ between ANE generations.

- Families may use `MetalDeviceCaps` architecture to select a candidate, but
  it is not ANE placement evidence. Admission also requires the bound placement
  and complete-request qualification above, with paired timing.
- Host-side quantization (here: per-row symmetric int8 with FP16 scales,
  round-half-even) reproduces the reference exactly. Tests check it against
  recorded digests.
- A fixed program, weights and input on the same device must give
  bit-identical outputs. A test checks this, and also checks ApxInf's
  program against the recorded coremltools program on that device.

## Proposed sequence

1. This document.
2. `apxinf-coreml`: the writer with the equivalence tests above, and the bridge
   with one device smoke test (compile, load, seal, predict). This establishes
   neither ANE placement nor heterogeneous-port acceptance.
3. `apxinf-mlx`: asynchronous evaluation.
4. `metal_clef` GPU+ANE composition. It depends on the GPU composition, the approved
   exception and all qualification gates above. `applegpu_g13s` is only the
   initial candidate dispatch identity, not sufficient evidence to enable it.

## Questions for maintainers

- Is the narrowly scoped Core ML and bounded host-handoff exception acceptable,
  with measured placement rather than an unenforceable ANE-only promise?
- Is the crate name `apxinf-coreml` and the feature name `coreml` acceptable?
- Where should the cache live by default? The composition compiles 32 programs; the first
  load takes seconds per layer.
- Should first-load compilation be an explicit preparation step rather than
  part of loading?
- CI builds the crate on Linux, where it is inert. Who owns the versioned
  macOS placement and complete-request qualification evidence until a macOS
  runner exists? A manual smoke test alone cannot qualify it.

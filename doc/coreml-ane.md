# Core ML and Apple Neural Engine execution (proposal)

Status: design proposal. No Core ML code is merged. This document asks
maintainers to agree on ownership, scope and evidence before implementation.

## Motivation

Apple Silicon has a Neural Engine (ANE) next to the GPU. Core ML is the only
supported way to run work on it. A model can split one stage between the two
devices: MLX computes part of it on the GPU while Core ML runs the rest on the
ANE.

The first candidate is Clef Flash on the M1 Pro. The EngineTailor GPU+ANE
frontier A04 keeps the C14 GPU kernels and moves 5632 of each MLP's 12288
intermediate columns to the ANE, as per-layer int8 programs. Owner evidence
(M1 Pro, 2026-10-17, same session after a reboot):

| Implementation | 32-layer complete request, weighted |
| --- | --- |
| C14 (GPU only) | 3603 ms |
| A04 (GPU + ANE) | 2790 ms (1.29x) |

A04 passed the owner's qualification with two recorded accepted decision
changes. Its MLX peak memory is 8.68 GB, against 7.22 GB for C14.

The EngineTailor implementation generates the ANE programs with Python
coremltools. ApxInf does not execute Python, so a port needs its own program
writer.

## Requirements

- No Python and no coremltools at build time or run time.
- Native execution on Apple Silicon macOS only. On other targets the feature
  is inert, as the `mlx` feature is.
- No silent fallback. If a requested program cannot be compiled or loaded for
  the ANE, loading fails.
- No compilation while a request runs.
- Each architecture is qualified separately, as for the Metal kernels.

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
These are the operations in A04's programs. The builder rejects any other
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
  (A04 uses CPU and Neural Engine). After loading, the runtime is sealed: a
  later compile request is an error.
- **Predict.** Inputs and outputs are FP16 host buffers. Output multi-arrays
  are read with their reported strides, because the ANE may pad them.
  Prediction runs on a worker thread that the runtime owns. MLX arrays never
  cross threads, so MLX's thread affinity is preserved.
- **Memory.** The ANE keeps only a limited amount of program weights resident;
  about 2.4 GB was observed on the M1 Pro. Exceeding it reloads programs on
  every switch. The family owns the per-layer budget.

## Numerics and qualification

ANE arithmetic is FP16, with device-specific reductions. Outputs are not
bit-equal to the GPU and can differ between ANE generations.

- Families gate ANE use by `MetalDeviceCaps` architecture. Each architecture
  needs its own complete-model qualification and paired timing.
- Host-side quantization (A04: per-row symmetric int8 with FP16 scales,
  round-half-even) reproduces the reference exactly. Tests check it against
  recorded digests.
- A fixed program, weights and input on the same device must give
  bit-identical outputs. A test checks this, and also checks ApxInf's
  program against the recorded coremltools program on that device.

## Proposed sequence

1. This document.
2. `apxinf-coreml`: the writer with byte-equality tests, and the bridge with
   one device smoke test (compile, load, seal, predict).
3. `apxinf-mlx`: asynchronous evaluation.
4. `metal_clef` `a04` composition. It depends on the C14 port and is enabled
   only on `applegpu_g13s`.

## Questions for maintainers

- Is the crate name `apxinf-coreml` and the feature name `coreml` acceptable?
- Where should the cache live by default? A04 compiles 32 programs; the first
  load takes seconds per layer.
- Should first-load compilation be an explicit preparation step rather than
  part of loading?
- CI builds the crate on Linux, where it is inert. Is a manually run macOS
  device smoke test enough until a macOS runner exists?

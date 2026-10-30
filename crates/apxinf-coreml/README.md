# Core ML programs for the Apple Neural Engine

This crate writes Core ML ML Program packages in Rust and runs them through
the Core ML framework. It is the native path to the Apple Neural Engine (ANE);
see the [design](../../doc/coreml-ane.md). No Python or coremltools is used.

## Writer (portable)

`ProgramBuilder` builds one FP16 `main` function (specification 9, opset
`CoreML8`) from a fixed operation list:

- `constexpr_blockwise_shift_scale` (INT8 weights with FP16 scales),
- `conv` (stride 1, `valid`, one group),
- `mul`, `add`, `real_div`, `maximum` (with an FP16 scalar or a tensor),
- `tanh`, `abs`, `identity`, `reduce_max`, `split`.

Other operations are rejected until added with a test. Operations and their
constants are named as coremltools 9.0 `mb.program` names them.

`Package::write` writes the `.mlpackage` directory: a fixed `Manifest.json`,
`model.mlmodel` and a MILBlob version 2 `weight.bin`. The protobuf subset in
`proto.rs` uses `prost` derive macros; no code generator runs at build time.

coremltools serializes protobuf maps in hash order, so its `model.mlmodel` is
not byte-reproducible. The writer instead matches it in three ways:

- the decoded program, excluding the writer's metadata and `buildInfo`;
- the weight file, byte for byte;
- prediction outputs, bit for bit, on the same device.

## Runtime (`native`, Apple Silicon macOS)

- `ProgramCache` compiles a package once into `<dir>/<digest>.mlmodelc`. The
  digest covers the program, the weights and the macOS build. The compiled
  directory is created under a temporary name and renamed into place.
- `ProgramCache::seal` makes a later cache miss an error, so nothing compiles
  while requests run.
- `Model::load` takes the compute units (for the ANE,
  `ComputeUnits::CpuAndNeuralEngine`).
- `ProgramCache::load_with_placement` resolves a compiled artifact once, inspects
  its ML Program structure, loads an unpublished model with the same native
  configuration factory, and invokes the family's fallible admission predicate.
  A rejected plan publishes no model. Owned facts include the resolved path,
  requested units, every function/nested operation and raw input/device usage.
  Missing usage, missing preference, empty support and unknown classes remain
  distinct. Inspection requires macOS 14.4 or newer and never substitutes for
  observed execution or warming.
- `Model::predict` exchanges contiguous FP16 host arrays and honours the output
  strides Core ML reports.
- `Model` is not `Send`. A family that overlaps ANE work with the GPU owns
  that schedule.

The placement timeout bounds the asynchronous plan wait only. It neither
cancels framework work nor bounds compilation, model loading, traversal or the
family predicate. A late completion retains only native state and cannot call
the Rust visitor. The native boundary contains Objective-C/C++ exceptions; the
visitor contains Rust panics and refuses partial traversal. Facts are copied
before their borrowed native views expire.

The family must bind the package/weights, compiled contents, configuration,
OS/hardware and fixed profile under trusted immutable cache ownership.
`compiled_path` and cache sealing alone do not establish content provenance.
Plan devices are anticipated placement, not proof of actual or exclusive ANE
execution. A `const`/constexpr name or opaque literal-presence flag is not
sufficient proof of static bytes or preparation-only materialization. The
family owns that mapping, exact-profile warm predictions and execution,
transfer, quality, resource and lifetime gates. There is no generic Ready flag.

On other targets, and without `native`, only the writer is built and no
framework is linked.

## Tests

- `cargo test -p apxinf-coreml` runs the portable writer tests.
- With `native`, library tests also cover owned facts, nested/empty traversal,
  distinct unknown states, malformed views, completeness and timeout checks.
  The ignored `tests/placement.rs` exercises real CPU-only plan/configuration,
  prediction and rejection without advertising ANE placement.
- The ignored `tests/recorded_package.rs` takes a coremltools 9.0 package of
  a Clef int8 MLP program from `APXINF_COREML_REFERENCE`. It rebuilds that
  program from the package's own weights and checks the three properties
  above, cache sealing, and repeated predictions. The package is not
  committed, because its weights derive from the checkpoint.

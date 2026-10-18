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
- `Model::predict` exchanges contiguous FP16 host arrays and honours the output
  strides Core ML reports.
- `Model` is not `Send`. A family that overlaps ANE work with the GPU owns
  that schedule.

On other targets, and without `native`, only the writer is built and no
framework is linked.

## Tests

- `cargo test -p apxinf-coreml` runs the portable writer tests.
- The ignored `tests/recorded_package.rs` takes a coremltools 9.0 package of
  a Clef int8 MLP program from `APXINF_COREML_REFERENCE`. It rebuilds that
  program from the package's own weights and checks the three properties
  above, cache sealing, and repeated predictions. The package is not
  committed, because its weights derive from the checkpoint.

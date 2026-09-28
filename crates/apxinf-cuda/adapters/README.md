# CUDA host adapters

This directory is the CUDA/C++ host boundary consumed by Rust FFI:

- `custom_kernels.cu` owns the stable C ABI and launch configuration for
  custom CUDA operators.
- `core_kernels_adapter.cu`, `static_bf16_adapter.cu`, and
  `w8a8_adapter.cu` preserve the remaining legacy C ABI surfaces while
  including pure operators from `kernels/custom/`.
- `cublas_adapter.cu` owns the cuBLAS MQA adapter and its logits workspace.
- `cublaslt_adapter.cu` owns cuBLASLt plans, heuristics, workspace, and its
  stable C ABI.
- `cutlass_*_adapter.cu` and `fa2_adapter.cu` expose stable C ABI shims around
  the C++ operators under `kernels/cutlass/`.

Rust safe kernel contracts call the symbols defined here directly through the
private `src/ffi/` declarations. Operator implementation files do not export C
ABI symbols.

The SM110 AOT adapters (`gdn_bf16_aot_adapter.cu`, `fa4_*_adapter.cu`,
`quack_m256n256_adapter.cu`) load the exported modules before graph capture,
fill their tensor descriptors, and launch them on the caller's stream.
`fa4_d256_split_batch_merge.cu` provides the split-attention packing/merge
launches. These are operator integration code, not model runners or alternative
model implementations. The six exported kernels share five maintained adapter
translation units; both vision geometries use the same adapter.

Generated CuTe `.h` declarations and `.o` device code live in the external AOT
artifact directory described by `aot/manifest.json`, not in this directory.
There is no separate handwritten declaration header per adapter: Rust's private
`src/ffi/` declares the C boundary. `fixed_profile.h` is shared compile-time
geometry, checked against the Rust and export recipe constants by a test.
Device implementations such as GDN QKV fusion belong in `kernels/custom/`.

`nvtx.c` supplies two C symbols around the NVTX3 header-only API for Rust's
existing profiling ranges. CUDA installations with these headers need no legacy
NVTX shared library. It is built by the C compiler only with the `nvtx` feature;
`--no-default-features` disables this instrumentation. It does not run kernels
or change numerical dispatch.

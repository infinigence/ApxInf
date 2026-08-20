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

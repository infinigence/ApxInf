# Vendored Marlin kernel core

These CUDA template files are derived without semantic modification from
`vllm-project/vllm` tag `v0.27.1`, commit
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, directory
`csrc/libtorch_stable/quantization/marlin/`.

The upstream files are licensed under Apache-2.0. The original Marlin
copyright and license header is retained in `marlin_template.h`; the repository
root license applies to the remaining derived files. `core/scalar_type.hpp` is
an ApxInf compatibility shim containing only the compile-time type surface
needed by the BF16/U4 template; it does not import the PyTorch exception or
tensor ABI.

The production carrier is the raw-pointer C ABI in
`adapters/marlin_adapter.cu`. Unsupported architectures, dtypes, group sizes,
shapes, or tile sizes fail closed.

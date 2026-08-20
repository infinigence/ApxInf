# CUTLASS and FMHA kernel sources

This directory contains ApxInf's static-inference CUTLASS C++ operators and the
pinned headers they compile against. Stable C ABI shims live separately under
`crates/apxinf-cuda/adapters/`; `crates/apxinf-cuda/build.rs` compiles those
adapters against only these repository-local operator sources and headers.

## Source snapshot

The contents were imported without modification from an upstream snapshot at
commit `7434950d5b3a73dcbf810a797e772cebdb869598`:

- `include/` from `3rdparty/cutlass/include/`
- `tools/util/include/` from
  `3rdparty/cutlass/tools/util/include/`
- `fmha/` from `csrc/kernels/nvidia/attention/fmha/`
- `extensions/` from
  `csrc/kernels/nvidia/cutlass/cutlass_extensions/`; only the three headers
  required by the SM80-family W8A8 scale epilogue are included

The CUTLASS version header identifies release 4.3.0. ApxInf intentionally pins
this complete header set because the SM101/SM110 FMHA templates and the local
operators must be compiled against a mutually compatible CUTLASS snapshot.

## Licensing

The imported CUTLASS and FMHA headers retain their original NVIDIA copyright
and BSD-3-Clause notices. The full CUTLASS license and the source repository's
MIT license are preserved under `licenses/`. The imported scale-epilogue
extension headers retain their SGLang copyright and Apache-2.0 license text.

ApxInf-specific C++ operators are `fp8_gemm_sm100.cu` and
`fmha_sm100.cu` for SM100-family devices and
`w8a8_gemm_sm80.cu` for SM80-family devices. They do not export
`extern "C"`; the corresponding shims are in `adapters/`.

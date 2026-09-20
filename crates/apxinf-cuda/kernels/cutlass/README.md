# CUTLASS and FMHA kernel sources

This directory contains ApxInf's static-inference CUTLASS C++ operators and the
pinned headers they compile against. Stable C ABI shims live separately under
`crates/apxinf-cuda/adapters/`; `crates/apxinf-cuda/build.rs` compiles those
adapters against only these repository-local operator sources and headers.

## Source snapshot

Except for the explicitly pinned FMHA headers below, the contents were imported
without modification from an upstream snapshot at commit
`7434950d5b3a73dcbf810a797e772cebdb869598`:

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

These files are unmodified NVIDIA CUTLASS `v4.4.2` example-77 sources:

- `fmha/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`, SHA-256
  `9cceff09555a9ac4b379cbfaa92fe6b76372b4b6c65f9bca0faabd699143891f`
- `fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp`, SHA-256
  `8d7289812f7513c527987291125c820bb1385233780fe2dc0cf6b6e369cbbdbc`

Keeping the digests here makes accidental local algorithm patches visible.

## Attention integration policy

Existing BF16 dispatch remains unchanged; configurations that select NVIDIA
FMHA now use its official mainloop. The production FP16 vision shape uses the
separately vendored, upstream Flash-Attention 2 `v2.7.4.post1` kernels
documented in `fa2/VENDOR.md`, including packed QKV through an ApxInf-owned
raw-pointer adapter. This avoids modifying the NVIDIA mainloop to rescale
softmax probabilities before its probability-storage conversion. The official
mainloop's FP16 narrowing has unacceptable error for that shape, while FA2
preserves accuracy and was faster in the Thor comparison.

## Licensing

The imported CUTLASS and FMHA headers retain their original NVIDIA copyright
and BSD-3-Clause notices. The full CUTLASS license is preserved under
`licenses/`. The imported scale-epilogue extension headers retain their SGLang
copyright and Apache-2.0 license text.

ApxInf-specific C++ operators are `fp8_gemm_sm100.cu` and
`fmha_sm100.cu` for SM100-family devices and
`w8a8_gemm_sm80.cu` for SM80-family devices. They do not export
`extern "C"`; the corresponding shims are in `adapters/`.

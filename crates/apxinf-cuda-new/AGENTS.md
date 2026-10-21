# apxinf-cuda-new agent rules

Current CUDA backend. New kernels and L3 semantics go here, never into the
legacy `crates/apxinf-cuda`.

- Every public L3 semantic has exactly one named candidate-accuracy test with
  an independent Torch oracle (see the `l3-reference-testing` skill). Organize
  tests by semantic, not by dtype.
- `test-new.sh` is a deprecated shim that forwards to `just test-l3`. Its
  defaults (`APXINF_CUDA_ARCH=sm_110`, `CARGO_TARGET_DIR=target-gemm-pilot`)
  are explicit overridable defaults in the recipe, not magic.
- The crate's `nvtx` feature is defined but currently unused by any cfg site.
- Vendored CUTLASS / fa2 sources under `native/kernels/{cutlass,fa2*}` are
  upstream code: do not edit; patch via `native/patches/` if unavoidable.
- `qwen38_gdn_batched_ops` test target does not compile on main
  (references removed `ops::gdn_prepare_flashinfer`); excluded from
  `just check`'s target set. Fix in a code PR.

# Vendor patches

Vendor source under `native/kernels/fa2/flash_attn` and
`native/kernels/cutlass/fmha` is kept byte-identical to its documented upstream
snapshot whenever possible. Any unavoidable source-level change must be stored
as a reviewable patch in this directory rather than silently edited in the
vendor tree.

Each patch must document its upstream revision, affected files, reason, and
validation. Compatibility shims belong in an ApxInf-owned sibling directory,
and architecture dispatch belongs in an operator wrapper.

## `fa2-direct-e4m3-output.patch`

- Upstream: Flash-Attention `v2.7.4.post1`
- Affected file: `flash_attn/flash_fwd_kernel.h`
- Purpose: add the PI0.5 static-scale FP16-to-E4M3 output epilogue without
  changing the public FA2 parameter ABI.
- Application: `build.rs` copies the FA2 headers into `OUT_DIR` and applies the
  patch there only for the dedicated E4M3 translation units. The checked-in
  vendor tree remains pristine.
- Validation: Thor SM110 all-candidate precision tests, raw E4M3 byte checks,
  and the PI0.5 Attention shape profile.

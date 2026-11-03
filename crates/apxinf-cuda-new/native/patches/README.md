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

## `attention-indexed-sm110.patch`

- Source: existing CUTLASS FMHA snapshot and Flash-Attention `v2.7.4.post1`.
  Exact canonical revisions/hashes, licenses, and affected files are recorded in
  `../kernels/attention/INDEXED_VENDOR.md`.
- Purpose: isolate native midpoint row flags and indexed FA2 Q/O addressing
  without changing canonical vendor files or the public Attention ABI.
- Application: `build.rs` stages five generated headers from four canonical
  sources in a private `OUT_DIR` directory. The single-Q mainloop and kernel
  copies have separate namespaces. The patch applies only to the dedicated
  indexed translation units; canonical vendor sources remain unchanged.
- Validation of the original configuration 0: staging byte identity,
  canonical-source preservation and repeated staging passed. Thor compilation
  and 12 production-kernel closed-form
  eager/Graph cases passed, including prepared-launch attribute instrumentation
  with a negative control. Corrected backend regression tests, the original
  56 complete-model action/video checks, and full public acceptance passed.
  The frozen public ABBA result and its limits are recorded in `doc/dreamzero.md`.

- Configuration 1 validation: the maintained single-Q implementation passes
  450 native output/flags/LSE/guard checks, the full backend suite (170 passed,
  0 failed, 2 ignored), and all 56 original joint action/video comparisons with
  zero recorded error. Full public quality and a frozen 16-process paired
  comparison pass independent audits; warm median latency is 0.8483% lower
  with a positive one-sided 95% process-group bootstrap lower bound. Cold
  observations are higher; detailed scope and limits are in `doc/dreamzero.md`.

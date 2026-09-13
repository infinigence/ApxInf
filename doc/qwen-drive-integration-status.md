# Qwen-Drive native integration checkpoint

This is an incomplete implementation checkpoint, not a qualified deployment or performance release.

The model target is Qwen/Qwen-Drive-1.0-4B, checkpoint revision
`28484089a7cc8c335cf5089fb0745cf7c49b6eaa`. The official reference source
revision is `28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0`.
The tested native path uses RTX 4090, BF16, ApxInf Rust/CUDA and AutoPolicy.

## Observed validation

- CUDA release build and native Python extension loading succeeded.
- VQA scenes 0 and 1 passed the fixed reference comparison; scene 2 failed
  first at zero-based generated token index 87 (native 2763, reference 12682).
  Native and reference output lengths were 542 and 568.
- At that step the native logits for both tokens were 24.125. The reference
  logits were 24.125 for 2763 and 24.25 for 12682.
- Feeding native hidden state into the official LM head reproduced the native
  tie. The head input differed in 2339/2560 elements, max absolute difference
  0.25. The earliest upstream divergence remains unresolved.
- Direct scene 0 failed the trajectory gate, max absolute error 0.040283203125.
- Reasoning scene 0 passed token comparison but failed trajectory comparison,
  max absolute error 0.08056640625.
- Perception inference currently raises an explicit pending-implementation
  error. Partial perception modules do not establish end-to-end support.
- No all-mode acceptance or performance improvement is claimed.

## Retained diagnostics and rejected candidates

`APXINF_QWEN_TRACE_DECODE_STEP` enables a logits readback at one decode step.
`APXINF_QWEN_HEAD_INPUT` names an output file for single-row LM-head inputs;
when enabled it is overwritten on subsequent calls. Both are unset by default.
Existing development diagnostics remain and need cleanup before production.

Two direct-takeover experiments regressed VQA scene 0 and were reverted:
recurrent decay via approximate exp2, and 128-dimensional recurrent tree
reductions. Earlier sequential inverse reduction, diagonal TF32 truncation and
A1 BF16 round-trip candidates were also rejected. Do not change sampling tie
policy or relax token/trajectory gates to make these examples pass.

## Next acceptance work

Locate the first differing model-layer output under an identical token prefix,
repair that numerical boundary, and repeat full VQA. Then repair and validate
planning trajectories and implement the missing perception execution path.
Final acceptance requires the same source candidate to pass all four native
modes and the fixed Host deployment verifier.

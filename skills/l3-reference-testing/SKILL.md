---
name: l3-reference-testing
description: Add or update numerical reference tests for ApxInf CUDA L3 operators. Use when introducing an L3 semantic API, adding candidates to an L3 operator, or reviewing whether candidate tests use an independent Torch oracle and the exact public input contract.
---

# L3 Reference Testing

For every public L3 semantic, maintain one clearly named candidate-accuracy
test, such as `gemm_bias_all_candidates_match_torch`. Keep operator semantics
as the primary organization; cover supported dtypes and quantization variants
inside that test instead of organizing tests by dtype.

## Oracle contract

- Generate deterministic fixtures with PyTorch and a fixed seed. Check the
  exact typed input bytes, scales, bias, scalar parameters, and FP32 expected
  outputs into the repository. Runtime tests must not require PyTorch.
- Start the Torch calculation from the exact tensors accepted by the L3 API.
  For BF16, use the stored BF16 values converted to FP32. For FP8 or INT8,
  decode the stored values and apply the same scale tensors supplied to the L3
  call. Do not compare against the FP32 values that existed before
  quantization; quantizer error belongs to separate quantization-operator
  tests.
- Implement the public mathematical meaning independently. Do not reproduce a
  candidate's intermediate buffers, packing, tiling, or launch decomposition.
  For GeGLU with public row-major `B=[K,2N]`, compute `A @ B_gate` and
  `A @ B_up` as two logical projections, where the first `N` columns are
  `B_gate` and the remaining `N` columns are `B_up`; do not build and split the
  candidate's `[M,2N]` intermediate projection.
- Include every semantic operand and option that changes numerical results,
  including scales, bias, alpha, output scale, activation convention,
  accumulation dtype, and output dtype.

## Candidate coverage and acceptance

- Run every candidate compatible with the device, Spec, alignment, resource
  policy, and execution mode. Fail when an expected candidate is silently not
  visited or when any visited candidate fails numerical validation.
- Convert candidate output to FP32 only for comparison. Reject NaN and Inf.
- Apply the same immutable thresholds to every provider and candidate:
  maximum scaled element error `<= 0.08`, relative L2 error `<= 0.05`, and
  cosine similarity `>= 0.9999`. A new candidate must not weaken these limits.
- Exercise non-unit scales whenever the L3 contract supports them. Exercise
  both ordinary and quantized input contracts that the L3 API accepts.

## Adding a new L3 semantic

1. Document its logical input and output contract at the public Args type.
2. Extend the PyTorch fixture generator with a direct expression of that
   contract and regenerate the checked-in fixtures.
3. Add one `<semantic>_all_candidates_match_torch` test using the exact same
   typed inputs and bindings for Torch and the candidates.
4. Assert that all expected compatible backends were visited and passed the
   shared thresholds.
5. Run the focused test, then the complete `test-new.sh` suite serially on the
   target GPU. Reuse the existing target directory for incremental builds.

If the L3 operator performs quantization internally, test that quantization as
part of its declared semantic. Otherwise keep FP32-to-low-precision conversion
out of the L3 accuracy test.

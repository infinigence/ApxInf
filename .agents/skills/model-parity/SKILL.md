---
name: model-parity
description: Compare model outputs between two ApxInf builds or between ApxInf and a reference implementation, and diagnose numeric gaps. Use for refactor byte-identity gates, reference cosine/rel-L2 acceptance, new input shapes or precisions, and any "outputs differ" investigation at model level (not single L3 operators).
---

# Model Parity

Two different questions; do not mix their evidence:

- **Build parity**: ApxInf build A vs build B (refactor, rebase, kernel swap).
  Gate is byte identity of actions on fixed inputs and fixed noise, eager and
  graph, per precision. Any difference is a regression until explained.
- **Reference parity**: ApxInf vs the reference implementation of record.
  Gate is a cosine / relative-L2 threshold fixed before measuring. For
  single-operator accuracy use `l3-reference-testing` instead.

## Protocol (stop at the first failing step)

Before experiments, create a missing task plan with `just acceptance init
<feature>` (it creates the ignored directory), or reuse the existing plan.
Fill in the acceptance matrix and public callers using
[agent acceptance](../../../doc/guides/agent-acceptance.md). Include each changed
public path and each affected family when shared binding/backend code changes.
Private harness results diagnose internals; they cannot substitute for the
public policy/binding call. A skip or absent device is UNVERIFIED, not a pass.

1. **Name the reference of record.** Repository or checkout, revision,
   checkpoint identity, precision, attention backend. A public upstream that
   only defines preprocessing is not automatically the forward-pass
   reference. Never declare the reference "unavailable" before checking the
   brief, the port records and the original port PR.
2. **Reproduce a known anchor first.** Feed the fixture that previously
   matched (stored inputs: token IDs, pixel values or patches, grids, masks,
   position IDs, state, initial noise) to both stacks and reproduce the
   recorded number. Do not test new shapes, precisions or code until the
   anchor reproduces; a harness that cannot reproduce it is not yet qualified
   to diagnose new shapes. Record the anchor as blocked when it cannot run.
3. **Measure the noise floor.** Run the reference against itself in FP32 vs
   the target precision on the same fixture. A gap well above this floor is
   semantic, not rounding. "BF16 accumulation" is a hypothesis until this
   measurement says otherwise.
4. **Make inputs identical and prove it.** Same uint8 images or same
   preprocessed tensors, same prompt tokens, same state encoding, same masks.
   Pass the *same initial-noise tensor* to both stacks, then hash the tensor
   each stack actually consumed: references often ignore a `noise=` argument
   and draw their own. Same seed in two RNGs is not the same noise.
5. **Record the effective shape.** Read back the grid / image size / token
   count each stack actually used and assert it against the intended mode.
   Checkpoint configs can silently resize inputs (for example a training
   resolution that maps 448 to 256); an unasserted shape test proves nothing.
6. **Compare in one declared output domain.** Normalized model output vs
   normalized, or physical vs physical, stated in every table. Report active
   action dimensions separately from padded ones; padded dims with near-zero
   normalizer range explode in the physical domain.
7. **Bisect by boundary.** Tokens (exact) → preprocessing output → vision
   embeddings → prefix hidden/KV → per-step velocity → final actions. Change
   one boundary at a time, keep the rest fixed, and record each control.
8. **Gate and report.** Use the thresholds fixed in step 1; never relax them
   after seeing the numbers.

## Build-parity mechanics

- Build baseline and candidate in separate source trees and separate cargo
  target dirs. Syncing with mtime-preserving copies can reuse stale nvcc
  objects; sync with checksums and confirm the extension hash changed.
- Run the identical harness file on both trees (compare its checksum) with
  the same tuning database; without tactics, kernels are untuned and timing
  and numerics differ.
- Confirm the baseline is bitwise repeatable before comparing to it.
- Latency: interleave baseline and candidate processes on the same idle
  device, same session, and report every round.
- Before claiming "the gap predates this branch", run the baseline build on the
  exact failing input through its own public entry point.

## Claim discipline

Label every conclusion MEASURED (with artifact path), INFERRED (reasoning
only) or UNVERIFIED. A root cause found by elimination is INFERRED. Never
write COMPLETE or CONFIRMED next to a result whose inputs, shape or output
domain were not checked by steps 4 to 6. When a contract rejects your input,
first check whether your branch changed that contract. Withdraw a superseded
claim explicitly instead of leaving it in the record.

## Final evidence

Freeze implementation and harnesses before final measurements. Record their
hashes and the loaded extension path/hash; rerun after edits. Compare full action
tensors, not log tails. Test unchanged inputs twice, changed images and prompts,
and shape A -> B -> A (including equal-sized buffers with different grids).
Test explicit preparation separately from implicit inference: required graph
mode, fallback reason, updated inputs on replay, and preservation of a healthy
plan after a rejected request are distinct contracts.

For timing/tuning, read back actual execution mode and fallback reason before
using a sample. Diagnostic event synchronization belongs outside graph capture.
Require a nonzero count of newly tuned keys for a tuning experiment; unchanged
keys mean the experiment did not exercise tuning. Reconcile stage timings with
the measured scope and board bandwidth before explaining a speedup. Validate an
optimization on its actual caller path (bias, shape, precision), then rerun
shared callers; compiling a new kernel does not prove it is used.

## Timebox

Set a time limit for each diagnosis before starting it. When it expires,
write down what is established (with labels) and what is not, then return to
the remaining goals instead of extending the investigation silently.

## Report table

One row per (mode, precision, entry path): effective grids or shape, output
domain, full and active-dimension cosine and rel-L2, noise source, baseline
or reference revision, artifact path.

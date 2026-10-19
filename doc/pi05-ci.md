# PI05 CI: offline deployment and rollout

This PR prepares the CI tools and workflows. It does **not** register runners,
enable GPU dispatch, or change branch protection. An incomplete board matrix,
missing asset, uncalibrated budget, failed preflight or numerical failure cannot pass.

## Acceptance status

**Not accepted / not online.** CPU contracts and the shared-board harness are
implemented. All 12 hardware/precision/view cells have been measured on the
clean implementation commit `ad35a46d`; numerical qualification and performance
budgets remain incomplete. The PR stays a draft until the required gates pass.

### Clean implementation-head verification (2026-10-19)

The complete Thor BF16/FP8 and Orin BF16/INT8 campaigns use clean `ad35a46d`
checkouts, eight frozen observations per view, fresh recipe preparation, the
revised consecutive/revisit protocol, and independent `bench_pi05.py` runs
with 30 warmups and 50 samples per observation. Input, checkpoint, config,
loaded CUDA libraries, extension and recipe identities pass independent checks.
All 96 outputs match the preceding curated-build outputs bitwise, with zero
consecutive/revisit drift. Independent local recomputation also verifies all
whole-chunk metrics and P50/P95 values from the raw latency samples.

| Hardware / precision | Historical whole-chunk passes, 1/2/3 views | Additional timestep/max-abs diagnostic passes | Measured P50 ranges in ms, 1/2/3 views |
| --- | --- | --- | --- |
| Thor BF16 | 8/8, 8/8, 8/8 | 8/8, 8/8, 8/8 | 61.26–62.28 / 69.48–70.69 / 85.53–86.01 |
| Thor FP8 | 7/8, 7/8, 6/8 | 3/8, 5/8, 4/8 | 38.73–38.91 / 44.91–45.22 / 52.91–53.88 |
| Orin BF16 | 8/8, 8/8, 8/8 | 8/8, 8/8, 8/8 | 136.48–137.92 / 170.50–171.93 / 216.21–217.82 |
| Orin INT8 | 7/8, 7/8, 6/8 | 4/8, 5/8, 5/8 | 89.09–89.98 / 128.88–129.59 / 167.73–168.82 |

Both quantized variants still fail the historical whole-chunk rules for
record0/frame165 in all three views and record1/frame169 in three views.
The additional requirements are distinguished in the budget audit below;
no failure or limit has been removed. Earlier constructed-input failures
also remain unresolved. Desktop GPU clients and unqualified power conditions
mean these latency samples do not establish an exclusive performance budget
or a paired before/after regression verdict. Completed measurement is not
release acceptance. The Thor campaign used the alternate shared board and
released its host GPU lock after all six cells completed.

### Measured state inputs (2026-10-19)

The eight DROID observations now also have their synchronized seven joint
positions and gripper position, normalized with the official
[pi05_droid quantile statistics](https://storage.googleapis.com/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json)
and tokenized by the pinned OpenPI Pi05 state-token path before state padding.
Images, noise, base checkpoint and original numerical limits are unchanged.
The prompts contain 44–54 valid tokens. This evaluates the base model's
implementation, not the fine-tuned DROID policy's task success. Third-camera
remapping remains shape coverage.

| Hardware / precision | 1 view | 2 views | 3 views | Total |
| --- | --- | --- | --- | --- |
| Thor BF16 | 8/8 | 8/8 | 8/8 | 24/24 |
| Thor FP8, original profile | 3/8 | 5/8 | 4/8 | 12/24 |
| Orin BF16 | 8/8 | 8/8 | 8/8 | 24/24 |
| Orin INT8 | 4/8 | 5/8 | 5/8 | 14/24 |

These are diagnostic reruns with existing curated native builds, not clean
final-PR-source acceptance. Every count includes all action timesteps. Graph
execution, fixed recipes and matching CUDA-library identities are verified.
Repeated/revisited outputs under the earlier protocol are bitwise stable; that
protocol does not qualify the revised consecutive-call gate below. The original
text-only and constructed failures remain recorded and unresolved.

The largest new FP8 discrepancy is `droid-record0-frame165`: two-view cosine
0.9766 / relative L2 0.2337, and three-view cosine 0.9541 / relative L2 0.3148.
The corresponding BF16 cases pass. Other quantized cases fail only an individual
timestep despite passing whole-chunk limits.

A controlled calibration experiment uses four fixed real frames with identical
images/noise, varying only text-only versus measured-state tokenization. Both
new profiles pass 0/8 one-view evaluation cases, versus 3/8 for the original
profile. Original outputs restore bitwise before and after the factors, and the
new profiles are rejected. These calibration frames are disjoint by frame but
share evaluation episodes; they are diagnostics, not qualified calibration data.
A separate candidate uses sixteen observations from four different DROID
trajectories, with episode identities checked against evaluation. Its fixed
selection is the union of initial, maximum gripper position, largest gripper
closure change and final frames; missing recorded prompts are excluded and
selection does not use model outputs. This candidate also passes 0/8, versus
3/8 before and after with the original profile. Both original controls restore
bitwise. It is rejected; additional calibration data alone has not repaired FP8.

Single-factor experiments keep the original profile and change only scales
that decreased after recalibration. Changing the language sites makes
`droid-record1-frame169` fall from cosine 0.99806 / relative L2 0.06224 to
0.27909 / 1.09974. Changing only the remaining sites gives 0.99759 / 0.07023,
which still fails individual timesteps. This localizes the severe degradation
to language scale changes; it does not establish a particular faulty kernel.
Read-only hooks in the official reference preserve both selected outputs
bitwise. After excluding masked camera slots and their prefix tokens, the new
state profile clips zero observed values for frame169, versus 2,800 projection
input values outside the original profile's ranges. Counts include shared
Q/K/V inputs separately. These are official BF16 activations, not propagated
native FP8 activations; they do not prove the native FP8 path avoids clipping.

The subsequent native FP8 capture identifies a calibration-coordinate bug.
FP8 folds each language RMSNorm's trained scale into QKV and gate/up weights,
then quantizes the RMS output **before** that scale. The BF16 reference path
keeps the scale explicit, but the collector was recording the post-scale
projection input. For frame169, native layer 1's raw RMS maximum is about
44.39 while the reference's scaled maximum is 13.31. Lowering the bound to
14.64 clips this native input. The original profile has no raw-RMS range
violations on this capture; the reduced language profile has 6,163. Traced
eager and ordinary graph outputs both restore the frozen outputs bitwise.

The collector now separately computes identity-weight RMS at the language
projection sites and excludes their post-scale linear inputs from collection.
It uses dedicated CUDA scratch storage, preserving prepared inference arena
addresses. The real GPU regression fails before the fix (range 2 instead of
approximately 16), then passes after it with identical model outputs. The
prepared eager-calibration regression also passes. This repairs observation
coordinates without changing FP8 inference, its weight folding, or the limits.
A correct calibration coordinate is necessary but does not establish that a
new profile meets the full-chunk accuracy gate. Failed profiles remain unapproved.

Recollecting the same 48 episode-disjoint projections with the corrected
observer improves the one-view evaluation from 0/8 to 6/8. The original profile
still gives 3/8 before and after, with bitwise controls. Frame137 fails only
zero-based timestep 48 (cosine 0.996208); frame169 still fails whole-chunk limits
(cosine 0.978581, relative L2 0.217067, maximum absolute error 0.400615), although
its previous erroneous-profile cosine was 0.259216. With identical frozen GEMM
recipes, a separate before/after calibration replay verifies that every other
site is bitwise unchanged and both collectors repeat exactly. The new profile
is not approved. The repair's paired performance qualification remains open:
Thor sampling encountered foreign GPU workloads, followed by Python and
`nvidia-smi` waiting in `os_acquire_rwlock_read`. Those measurements cannot
establish a regression verdict; the task's affected submissions were stopped,
without terminating foreign jobs or resetting the device.

A subsequent same-board Thor pair completes the missing warm-inference check
for this calibration repair. It compares the verified pre-fix `d9424a83`
extension with the clean `ad35a46d` extension, then revisits the pre-fix
extension. The same two frozen one-view observations, original FP8 profile,
prepared recipes and CUDA libraries are used throughout. The maintained
benchmark takes 30 warmups and 50 samples; its timed inference method is
unchanged. Output/repeat/revisit checks run after measurement and restore
bitwise in every phase. All 497 activity samples contain only this task's
compute processes and no external compiler.

| Precision / observation | P50 ms: before / fix / before again | P50 change vs baseline mean | P95 change vs baseline mean |
| --- | --- | --- | --- |
| BF16 / record0-frame070 | 62.130 / 62.182 / 62.174 | +0.05% | +0.30% |
| BF16 / record1-frame169 | 61.223 / 61.357 / 61.372 | +0.10% | -0.02% |
| FP8 / record0-frame070 | 38.736 / 38.814 / 38.819 | +0.09% | +0.31% |
| FP8 / record1-frame169 | 38.804 / 38.850 / 38.854 | +0.05% | +0.12% |

No substantial warm regression is observed in this pilot. First-call timing
has a wider spread: the BF16 triplets are 1040/1101/1191 ms and
1013/1018/995 ms, so the warm result is not a blanket cold-start verdict.
Persistent desktop graphics and power conditions remain unqualified; this
pair does not approve an exclusive performance budget or replace all-view
performance qualification. The independently checked raw samples retain
first-call measurements and baseline revisit drift.

On Orin, the isolated language-normalization weight preparation correction
passes 5/8 one-view state cases versus 4/8 originally, but introduces a new
failure. Keeping only the flow state in FP32 preserves the original failure
set. Neither factor is promoted. Quantizing/dequantizing FP32 checkpoint
weights and rowwise activations inside official OpenPI passes all four selected
state observations; this retains the official floating-point GEMMs and is not
an integer-GEMM oracle. The original reference restores bitwise after each
factor. A separate integer-GEMM factor uses signed INT8 inputs, INT32
accumulation, FP32 row-then-channel scaling and BF16 output. Independent FP64
dot products validate its padded irregular shapes before model execution.
After correcting the scalar-division rounding described below, quantizing
only language projections or all transformer projections each passes 3/4
observations. For frame169, relative L2 is 0.1194 and 0.1399 respectively.
The earlier reciprocal-multiply factor passed 3/4 and 2/4, with relative L2
0.1840 and 0.2102; its raw results remain diagnostic evidence. Unmodified reference controls restore bitwise. This
reproduces some failures outside ApxInf, but leaves the reference's other
operations unchanged and does not establish native arithmetic equivalence.

A native precision factor preserves only the language projections as BF16,
keeping the other INT8 computations unchanged. It passes all four selected
observations, versus 1/4 for its bitwise-restored INT8 controls. Retaining only
QKV or only the remaining language projections each passes 3/4; frame169
still fails. The broad retention candidate is rejected for performance:

| One-view state observation | Original P50 | Candidate P50 | Original again P50 |
| --- | --- | --- | --- |
| record0 / frame070 | 90.059 ms | 122.527 ms | 90.056 ms |
| record1 / frame169 | 89.701 ms | 121.695 ms | 90.106 ms |

These measurements use `bench_pi05.py`, 30 warmups and 50 samples, fixed
recipes, graph execution and identical inputs. The candidate regresses by
35–36% against the mean of the bracketing originals. They are a shared-board
pilot, not an exclusive performance budget. The candidate is a private mixed
precision diagnostic, not an accepted INT8 implementation.

A subsequent eight-observation bisection retains only language layers 0–8 or
9–17. Each passes 7/8, versus 4/8 for bitwise-restored original controls. The
first half still fails frame169; the second half fails frame137. Splitting the
layer groups further on four fixed observations gives 3/4 for layers 0–4,
3/4 for 5–8, 0/4 for 9–12 and 3/4 for 13–17. The first two groups fail frame169;
the last group fails frame137. These are sensitivity experiments, not fixes or
accepted mixed precision policies. No smaller group yet passes this pilot.

A subsequent capture tests the actual native language arithmetic on frame169.
Both the trace-disabled graph output and traced eager output restore the
original INT8 chunk bitwise. All 69 language projections, containing
208,558,080 output elements, match an independent INT8 dot product and FP32
row-then-channel dequantization to BF16 exactly; sampled integer dots also
match FP64 arithmetic. Every captured activation byte and scale matches a
CPU correctly rounded division oracle. An earlier PyTorch factor differs at
1,127 scale values and 687 quantized bytes because scalar division by 127
uses reciprocal multiplication. Those earlier factors cannot establish exact
native quantizer equivalence; the corrected reference experiment requires
all 69 captured activation quantizers to match bitwise before evaluation.
Using the same captured inputs, all 17 GeGLU blocks also match the pinned
official operation bitwise. The combined RoPE/Attention boundary differs;
including the reference's two masked camera slots produces the same observed
errors as compact active slots. This separates shape padding from the remaining
arithmetic difference on this case. These operator checks narrow attribution
to remaining boundaries and quantization loss; they do not waive full-chunk
failures.

For FP8, applying the fixed quantize/dequantize transforms to all transformer
projections inside official OpenPI does not reproduce the severe native
language-scale failure: the decreased-language profile passes both selected
observations. This factor still uses official floating-point GEMMs and other
BF16 boundaries. Actual native intermediate tensors are needed to distinguish
clipping, GEMM arithmetic and propagated floating-point differences.

### Earlier text-only and constructed inputs (2026-10-17)

Each numerical run covers eight real observations and thirteen constructed
diagnostics per view: 63 cases per board/precision. The existing limits apply
to the whole chunk and every action timestep. Counts below are diagnostic
results, **not approved baselines**.

| Build / hardware / precision | Real images, text-only, 1 / 2 / 3 views | Constructed, 1 / 2 / 3 views | Total |
| --- | --- | --- | --- |
| Maintained TF32 and split-K repair / Thor BF16 | 7/8, 8/8, 8/8 | 11/13, 8/13, 12/13 | 54/63 |
| Maintained TF32 build / FP8, original calibration | 0/8, 5/8, 5/8 | 1/13, 0/13, 0/13 | 11/63 |
| Maintained TF32 and split-K repair / Orin BF16 | 6/8, 8/8, 8/8 | 9/13, 8/13, 10/13 | 49/63 |
| Same Orin build / INT8 | 0/8, 6/8, 4/8 | 1/13, 0/13, 0/13 | 11/63 |
| Complete private arithmetic diagnostic / Thor BF16 | 8/8, 8/8, 8/8 | 13/13, 13/13, 13/13 | 63/63 |
| Complete private arithmetic diagnostic / Orin BF16 | 7/8, 8/8, 8/8 | 10/13, 8/13, 9/13 | 50/63 |

An independent offline audit of these archived maintained BF16 receipts checks
all 126 input-file digests, paired checkpoint/config/suite identities and loaded
CUDA-library digests, then recomputes whole-chunk and timestep metrics from raw
outputs. Under the historical whole-chunk policy, Thor passes **59/63** and
Orin **52/63**, versus 54/63 and 49/63 under the additional checks above. Five
Thor failures and three Orin failures arise only from those additional checks.
The remaining four Thor and eleven Orin failures still fail whole-chunk limits;
they cannot be dismissed by removing timestep checks. These are archived
pre-final-head runs, not additional current-head qualification. The original
receipts, thresholds and failure verdicts are preserved.

The private Thor result uses staged Attention, ordered patch projection and an
action-down provider control. It is not a maintained execution path. Its 60/63
bitwise outputs, unchanged limits, graph execution and zero repeated/revisited
drift establish diagnostic evidence only. Both engines load identical CUDA
library digests. The three non-bitwise cases also pass every timestep. Earlier
Thor runs with different loaded cuBLAS libraries are ineligible for formal
acceptance; the aggregator rejects that mismatch.

### Numerical findings and remaining work

- Maintained fixes cover timestep generation, unfolded normalization, staged
  BF16 RoPE/GeGLU, FP32 frontend boundaries, Welford LayerNorm, bias before BF16
  rounding, vision Attention arithmetic, RGB reciprocal multiplication and
  pageable-upload completion. Independent GPU negative controls exercise these
  boundaries. Fixtures use synthetic inputs rather than checkpoint captures.
- FP32 auxiliary projections now explicitly request `GemmMath::Tf32`; storage,
  accumulation, resources, graph lifetime and recipe identity remain in the
  safe CUDA operator layer. Replacing the private TF32 adapter preserves four
  difficult Thor outputs bitwise. This alone does not fix the full model.
- Orin's private white/gradient traces first diverge at the first vision output
  projection; input, LayerNorm, QKV and Attention are bitwise equal. The selected
  cuBLASLt candidate uses in-place split-K reduction in BF16. An independent
  cancellation case loses `0.99609375` despite requesting FP32 accumulation.
  Filtering BF16 partial reductions to compute precision passes the actual GPU
  RED/GREEN test and six existing GEMM regressions. The two-view private
  diagnostic improves from 8/13 to 9/13; white-image relative L2 falls from
  0.290 to 0.052 and still fails. The full maintained matrix is recorded above;
  Orin paired performance diagnostics are recorded below.
- After that fix, Orin's first vision FC2 difference is attributable to the
  **reference's** reduced-precision BF16 partial sums. With frozen identical
  inputs/weights, disabling PyTorch's `allow_bf16_reduced_precision_reduction`
  makes FC2 bitwise equal to ApxInf. An independent FP64 calculation also shows
  lower error for ApxInf. Whole-model white and camera-gray diagnostics then
  pass, but zero-noise cases still fail. The same reference flag changes none
  of the four Thor outputs. Original reference receipts remain unchanged;
  choosing a different arithmetic profile requires explicit bank qualification.
- FP32 frontend and TF32 auxiliary projections bypassed six calibration
  observation sites. Restoring observation at those GEMMs passes a small
  checkpoint-free CUDA RED/GREEN regression on both Thor and Orin. The observer
  already supports FP32 tensors. All 63 Thor outputs remain bitwise unchanged
  with observation inactive. Paired measurements across all three views show
  no clear steady-state slowdown (maximum P50 increase 0.016 ms). Three
  independent two-view repeats also overlap in first-call latency, while
  shape-return tails remain higher in some candidate runs (1.078–1.146 s
  versus 1.058–1.082 s); these shared-desktop results are not accepted budgets.
- The original FP8 calibration used seven synthetic inputs with horizon 10.
  Its metadata does not qualify it for the horizon-50 acceptance matrix.
  A new calibration factor uses four separate real observations, disjoint from
  evaluation, across 1–3 views. It is an unapproved experiment and does not
  replace the existing profile or reference. On fifteen pilot inputs, the
  original profile passes 2/15 and the new one passes 0/15; the new profile is
  rejected. Keeping only the FP8 flow state in FP32 leaves the 1/2-view pass counts unchanged (6/42); that isolated change
  does not resolve quantization failures.
- A separate Thor action-down difference comes from **FP32 reduction order**:
  several split-K candidates match OpenPI while the selected unsplit candidate
  differs in 287 BF16 elements. FP64 differs from both. This is distinct from
  Orin's reduced-precision partial sums and does not justify a provider override.
- The next private Orin zero-noise trace first diverges at action layer 1 QKV;
  its input, style and normalization are bitwise equal. An independent operator
  replay reproduces all 50 differing BF16 elements: PyTorch separate Q/K/V
  projections match OpenPI, while a packed projection matches ApxInf bitwise.
  Two of eight packed cuBLASLt candidates also match OpenPI. Both original
  outputs differ from an independent FP64 dot-product oracle. This identifies
  a shape/algorithm-dependent arithmetic difference, not a packing error or
  authority to hardcode a model-specific provider.
- Reference sensitivity is measured separately from implementation error. With
  the original OpenPI model, changing normalized FP32 patch inputs by one ULP
  (at most `1.1921e-7`) makes the two-view black/zero-noise diagnostic reach
  cosine 0.8664 and relative L2 0.5094. Original controls before and after the
  perturbation are bitwise equal to the saved reference. Across eight real
  observations per view and both perturbation directions, 13/16 one-view
  comparisons pass the existing limits; two and three views each pass 16/16.
  These failures do not coincide with every native failure. This establishes
  input sensitivity, not an implementation waiver, a new tolerance, or authority
  to remove failing observations from the bank.
- Quantization loss is also observable without ApxInf: in a separate OpenPI
  weight-only experiment, quantizing/dequantizing the language transformer
  weights per output channel makes a one-view real observation fail the
  historical INT8 limits (cosine 0.9924, relative L2 0.1255). The two-view
  black/zero-noise diagnostic reaches cosine -0.1101. Activations, original
  operators and auxiliary projections remain unchanged, and restoring weights
  reproduces every saved control bitwise. This is a sensitivity experiment,
  not an exact emulation or qualification of the native INT8 implementation.
  A separate activation-only factor quantizes/dequantizes language Linear
  inputs row-wise with unchanged weights and operators. Two one-view real-image
  inputs reach cosine 0.8513 and 0.8695; restoring the original computation is
  bitwise equal to every control. These fixed computation fixtures do not
  establish deployed policy quality or native INT8 arithmetic equivalence.
  Separately, 60 native auxiliary-projection factors (input/output, time,
  modulation, all, and control across two views) resolve none of the selected
  failures; those private substitutions are rejected.
- INT8 also retains a distinct RMSNorm preparation difference: it rounds
  `1 + offset` to BF16, whereas the reference adds one in FP32 after rounding
  the offset to BF16. A private change reuses the existing BF16-path scale
  preparation without changing quantized GEMMs. A checkpoint-free test of the
  actual INT8 language-layer preparation fails before the change and passes
  afterward on Orin. Reusing unchanged original recipes, the full 63-case
  comparison resolves three failures but adds two: one-view black/normal noise
  and two-view `droid-record0-frame000`. Repeats and revisits remain bitwise.
  The change is not promoted; its performance run was canceled before execution
  because it failed the numerical regression requirement.
- Remaining maintained BF16 failures include zero-noise gradients, black and
  white images, distinct camera gray levels and one-view real observations.
  Some fail only an individual timestep. FP8/INT8 failures also remain unresolved;
  they are not waived as out-of-distribution inputs or accepted by replacing
  reference outputs.

### INT8 initialization regression

The CPU quantizer now scans canonical `[input, output]` weights contiguously
instead of traversing one strided column at a time. Each channel retains its
original maximum-reduction order, scale, rounding and clipping. A column-wise
oracle checks quantized bytes and scale bits, including odd shapes and zero
channels. All four related native tests pass on Orin.

Reusing unchanged recipes, all 63 one-, two- and three-view INT8 outputs are
bitwise equal to the preceding implementation; repeated and revisited outputs
also have zero drift. This preserves the existing OpenPI failure set and does
not qualify the quantized model. Paired model construction takes 93.175 / 34.195 / 92.862 seconds for baseline /
candidate / baseline revisit. With 30 warmups and 50 samples, one-view T10 P50
is 87.609 / 87.762 / 87.881 ms and T200 is 105.322 / 105.111 / 105.154 ms.
There is no clear steady-state regression in this paired diagnostic. T10 P95
is 87.945 / 87.993 / 87.968 ms; the small tail increase is retained. These shared
board measurements are not exclusive-run deployment budgets. T10 first-call
latency is 6.758 / 6.788 / 6.688 seconds; T200 is 7.489 / 7.722 / 7.750 seconds,
and shape return is 7.614 / 7.801 / 8.067 seconds. The initialization improvement
does not eliminate those graph/shape-preparation costs.

### Performance evidence

Paired measurements use 30 warmups and 50 samples, with baseline, candidate and
baseline revisit. They retain shared desktop clients and are diagnostic, not
exclusive-run budgets. Latest maintained Thor split-K measurements, against the preceding TF32 build, are:

| Views | T10 baseline / repair / revisit P50 | T200 baseline / repair / revisit P50 |
| --- | --- | --- |
| 1 | 59.266 / 59.281 / 59.286 ms | 64.652 / 64.599 / 64.673 ms |
| 2 | 69.572 / 70.305 / 69.479 ms | 76.346 / 76.305 / 76.183 ms |
| 3 | 87.064 / 84.686 / 86.322 ms | 91.028 / 91.291 / 91.061 ms |

The two-view T10 increase is approximately 1%. An isolated Attention recipe
factor recovers it. A general diagnostic change then measures graph-safe
candidates using CUDA Graph replay instead of eager host submissions. An
independent host-submission-delay fixture fails with the old tuner and passes
with the change. The two-view BF16 pilot preserves the failure set and measures
70.433 / 69.549 / 70.287 ms for baseline / candidate / revisit. The expanded
Thor BF16/FP8 matrix across all three views adds no failures and resolves one
two-view FP8 real case. However, full paired performance exposes a three-view
BF16 T10 regression: 84.694 / 86.398 / 84.783 ms, approximately 2% slower.
Orin one-view BF16 adds a real-observation failure: whole-chunk cosine remains
0.99994, but the worst timestep relative L2 increases from 0.0323 to 0.0591,
above the existing 0.05 limit. The pre-performance regression check stops that
candidate's Orin timing run. The tuner change is rejected. Its Orin CUDA library
run passes 90 tests, fails the same twelve unsupported-provider tests as the
baseline, and ignores two; it introduces no new library-test failures.

The complete private Thor arithmetic repair adds approximately 9–11 ms. A
single-factor measurement attributes about 8–9 ms to staged Attention; unrolling
its softmax saves about 1.65 ms in the full two-view model. This remains private
performance debt. Latest maintained Orin measurements include TF32 and split-K
repairs, against `35bcbf59`, including earlier INT8 repairs:

| Precision / views | T10 baseline / repair / revisit P50 | T200 baseline / repair / revisit P50 |
| --- | --- | --- |
| INT8 / 1 | 87.346 / 87.762 / 87.327 ms | 104.473 / 105.242 / 104.649 ms |
| INT8 / 2 | 127.663 / 127.995 / 127.536 ms | 144.173 / 144.856 / 144.544 ms |
| INT8 / 3 | 166.969 / 167.020 / 166.634 ms | 183.842 / 183.700 / 183.487 ms |
| BF16 / 1 | 131.422 / 134.144 / 131.621 ms | 130.595 / 131.038 / 130.480 ms |
| BF16 / 2 | 193.621 / 195.532 / 193.958 ms | 185.045 / 183.063 / 185.442 ms |
| BF16 / 3 | 217.307 / 218.723 / 216.186 ms | 231.073 / 229.380 / 230.616 ms |

Small increases, shape-return costs and baseline drift are retained, not waived.
Final-source BF16/FP8/INT8 accuracy, stability, performance and hosted aggregation
must all be accepted before rollout. CPU CI currently has 51 passing tests,
including real-process cancellation, repeated cancellation, unresponsive-worker
and blocked-auditor cleanup. The standalone comparator also rejects missing,
invalid or mismatched checkpoint/config identities. Replaying 252 saved GPU
outputs preserves every case verdict. Thor CUDA library tests pass 101 tests
with two ignored; the wider integration suite stops on missing external Qwen
reference tensors. Orin has twelve failures in Thor-specific provider tests,
so the full cross-model CUDA suite is not reported as passing. No online
configuration has been enabled.

## Scope

The required matrix is Thor BF16/FP8 and Orin BF16/INT8, each with 1, 2 and 3
views: twelve cells. One real three-camera checkpoint is sufficient to protect
these shared computation paths. Add a `libero` profile with `views: [2]` to also
protect the native LIBERO checkpoint; this produces sixteen cells. Every
profile is run through the same script, with no case-specific model code.

Use the native action horizon (normally 50) and ten flow steps. The output under
test is the normalized `[H,32]` model action chunk, through `infer_rgb` including
vision, input upload and blocking output readback. This does not certify camera
name adapters, tokenization, action unnormalization, physical units, or closed
loop success. Keep policy/adapter CPU tests and LIBERO deployment acceptance.
1/2 views mean the first slots of base/left/right; arbitrary missing middle
camera masks are not exposed by the current ApxInf direct interface.

## Fixed inputs and references

`compare_pi05_openpi.py` has four phases, usable in separate environments:
`prepare`, `openpi`, `apxinf`, `compare`. Daily CI uses saved OpenPI and approved
ApxInf outputs; it neither installs OpenPI nor downloads weights.

Preferred preparation is `--case-npz`: one frozen model observation per file,
with `images` (V,224,224,3 uint8), `token_ids` (T integer), and `noise` (H,32
float32). Preserve the actual preprocessed camera order and tokenized
prompt/state from the same observation. Gaussian noise must be saved, not
regenerated separately by two frameworks. Loading a frozen suite validates
uint8 NHWC image shape and view count, nonempty in-vocabulary integer tokens,
and finite FP32 noise with the declared action horizon. Each case digest is
checked again when loading it for inference, so a changed archive cannot reuse
a prior suite-validation result. These checks run outside latency sampling. The same observation can produce
1/2/3 view shape fixtures by retaining the first V cameras. This is a computation
test, not evidence that a deployed policy with removed cameras succeeds.

```bash
python scripts/compare_pi05_openpi.py prepare \
  --suite-dir devlocal/pi05-ci-gate/bank/base-3view \
  --image-keys base,left,right --horizon 50 \
  --case-npz /existing/observations/grasp.npz \
  --case-npz /existing/observations/release.npz

JAX_PLATFORMS=cpu python scripts/compare_pi05_openpi.py openpi \
  --suite-dir devlocal/pi05-ci-gate/bank/base-3view \
  --checkpoint-dir /existing/pi05-base --openpi-revision FULL_REFERENCE_SHA

python scripts/compare_pi05_openpi.py apxinf \
  --suite-dir devlocal/pi05-ci-gate/bank/base-3view \
  --checkpoint-dir /existing/pi05-base --hardware thor --precision bf16 \
  --revision FULL_APPROVED_APXINF_SHA --stability-repeats MEASURED_STABILITY_COUNT \
  --output devlocal/pi05-ci-gate/bank/base-3view/thor-bf16-approved.json
```

Choose observations by explicit coverage: each supported LIBERO task, grasp and
release transitions, occlusion, small predicted actions, longest deployed token
profile, and historical failures. Calibration episodes and acceptance episodes
must be separate. Record the source episode/frame and coverage in the bank's
review notes. The implementation accepts any number of frozen observations;
there is no arbitrary 48-case target or claim of statistical task coverage.

`--diagnostic` is an implementation diagnosis set, not the merge-gate input
bank. It contains thirteen explicitly constructed cases: gradient/T10, a second
noise draw, T21, the configured token-length boundary, black image only, zero
noise only, black+zero together, white image only, negative noise only,
white+negative together, distinct constant gray levels per camera, and
gradient+zero / black+zero at the configured length boundary. Distinct gray
levels do **not** swap camera order. T21 is a concrete workload, not a
length boundary; `--max-token-len` must match the checkpoint's
`tokenizer_max_length`/`max_token_len` (200 for the tested checkpoints).
The boundary uses repeated valid tokens and tests shape handling, not natural
language quality. Float CHW conversion has CPU coverage and adds no GPU case.

`--source-npz` replays every supplied RGB observation with canned T10 tokens and
fixed noise; it is image replay only. `--case-npz` preserves actual tokens and
noise and is preferred for a formal bank. Neither mode is automatically labelled
representative. Record task, trial, trajectory frame/time, camera ordering,
preprocessing/source revision and original asset digest for every real
observation. Different token modes on the same image are one observed scene.
`--provenance-json` accepts a case-stem-to-metadata mapping and freezes it
inside the suite manifest. Use a common `observation_id` for token variants of
one frame. Unknown frame/time metadata must remain unknown; do not invent provenance.
Three-camera real coverage requires a source containing three actual cameras;
adding a synthetic/duplicated third camera is a separately labelled diagnostic.

Select a small held-out bank by input differences that can affect execution:
actual camera count, deployed token lengths/state-token mode, visual diversity,
small reference action magnitudes, and minimized historical failures. Retain
which case covers each difference, and remove cases with identical canonical
model input. Determine the resulting count from measured runtime and coverage;
synthetic diagnostic counts do not prescribe the formal bank count.

The eight existing DROID image fixtures use the pinned tokenizer with
`state=None`: they contain the recorded task text but no state tokens. The
pinned official tokenizer identifies that branch as the Pi0 prompt format;
Pi05 state input uses `Task: ..., State: ...; Action: ...`. Therefore those
fixtures are text-only computation diagnostics, not coverage of the normal
Pi05 state-token path. Additional controlled zero-state fixtures must also be
labelled as constructed states, not measured normalized DROID observations.
Neither category replaces or waives previously failing numerical cases.

The official adapter uses OpenPI PyTorch PI0.5 with compilation disabled,
selected BF16 parameters, zero direct state, explicit prompt tokens/masks and
the same flow noise. PI0.5 state information must already be tokenized into
the prompt when required by the policy. Do not feed a different model family,
expert size, solver, or checkpoint configuration into this reference adapter.

## Numerical and performance budgets

For vectors a,b, cosine is `dot(a,b)/(norm(a)*norm(b))`; relative L2 is
`norm(b-a)/norm(a)`. Cosine alone accepts b=2a, so the gate also uses relative
L2 and maximum absolute error. It checks the whole chunk **and each timestep**,
and fails on NaN, Inf, shape mismatch or receipt mismatch. Undefined zero-norm
metrics are JSON null. Only a zero reference uses the explicit `zero_max_abs`
budget; an all-zero candidate against a nonzero reference fails.

There are two independent comparisons: official OpenPI and a frozen approved
implementation on the same hardware/precision/calibration. Do not advance the
approved baseline automatically with main: this would permit gradual drift.
Per-channel physical accuracy still requires policy-level evaluation; padded
32-dimensional latent output cannot certify robot units.

The existing `pi05_bench.rs` checks whole-chunk cosine/relative L2 at
.999/.05 for BF16, .997/.10 for FP8, and .995/.10 for INT8. Only INT8 has
its historical reference max-abs limit of .125; BF16 and FP8 have no such
limit in that benchmark. Applying these floors to every timestep and adding
.125 max-abs to BF16/FP8 are additional diagnostic requirements introduced by
this work, not unchanged historical gates. Those additions have not been
independently calibrated and must not be described as approved merge budgets.
No saved failure or measured output is changed by this clarification.

For example, re-scoring the clean `ad35a46d` Orin state-bank outputs against
the original whole-chunk policy gives INT8 7/8, 7/8 and 6/8 across views,
versus 4/8, 5/8 and 5/8 under the additional timestep checks. Four whole-chunk
failures remain; the six additional failures are attributable to the stronger
protocol, not evidence of six new implementation regressions.

[Edge-LLM's PI05 design at revision 0f527057](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/0f52705723abfa8c6702654bfdcc082a728ed353/docs/source/developer_guide/models/pi05.md)
describes FP16 engines compared with the OpenPI FP32 policy on each
configuration's own embodiment observations. Its comparator uses .99999
whole-chunk cosine per batch entry and .005 maximum absolute error, with no
per-timestep cosine gate. Its PI05 path does not implement FP8/INT8
quantization. These are not evidence for applying the same budgets to ApxInf
quantized models, all artificial inputs, or every rebuilt engine; the design
also records build-dependent numerical variation.

The example definition deliberately has exact-equality numerical placeholders
and null performance budgets. Replace them only after measuring same-build and
rebuilt baseline variability, checking known-error mutations, and approving the
application's allowable error and latency. `budget_reason` records that evidence.
An uncalibrated cell fails the aggregate even when its numerical comparison
passes. No fabricated 5%/10% latency allowance is supplied.

Accuracy uses exactly one finite complete action chunk per saved observation
from each engine, with identical images, tokens and FP32 source noise. It saves
that first valid output; later calls cannot replace it. This is fixed-observation
chunk agreement, not a multi-chunk closed-loop rollout or LIBERO success-rate
evaluation. Official selected-BF16 is mixed precision; do not describe it as
all operations being BF16.

Stability is a separate phase enabled by `--stability-repeats`. For each frozen
input, it first captures a stability anchor and then performs the requested
number of consecutive repeat calls. Drift includes differences from that anchor
and from the saved accuracy output. The same pass visits A and all other inputs;
one final A call measures revisit drift against both A anchors. This preserves
`N * (repeats + 1) + 1` stability calls without a redundant extra traversal.
Accuracy outputs remain the original first valid chunks; stability never replaces
them or supplies latency samples.

Receipts identify this as `consecutive-and-revisit-v1`; the aggregator rejects
missing or older stability protocols. Previously, `repeats=1` compared calls
separated by other inputs and did not establish consecutive repeat stability.
Historical numerical outputs remain diagnostic evidence, but those stability
receipts cannot qualify the revised gate. CPU negative controls cover the actual
call sequence, state leaked only after another input, and opposite deviations
around the saved accuracy output.

Both boards also pass an actual two-input, old/new/old GPU protocol check for
both supported precisions. Accuracy outputs restore bitwise, repeat/revisit
drift is zero, and graph mode, recipes and loaded CUDA libraries are unchanged.
The same five stability calls take the following phase wall times in milliseconds:

| Board / precision | Old protocol | Consecutive protocol | Old protocol revisit |
| --- | --- | --- | --- |
| Thor BF16 | 5279 | 2273 | 4123 |
| Thor FP8 | 2149 | 1052 | 1789 |
| Orin BF16 | 16940 | 7887 | 13519 |
| Orin INT8 | 37942 | 14533 | 25398 |

This measures harness work, including fewer shape recaptures; it is not warm
model latency, a performance budget, or full-bank stability qualification.

Daily bank policies require an independent `stability_max_abs` budget and an
explicit positive stability count, chosen after same-input and rebuilt-extension
measurements. The example's one repetition is an unapproved placeholder, not an
established protocol. Stability never borrows the OpenPI or baseline accuracy
tolerance; an uncalibrated stability budget fails pre-build validation and hosted
aggregation.

Performance runs in a separate process through `bench_pi05.py --suite-dir ...
--model-dir ... --layer l1`, with independent `--warmup` and `--samples`. This
reuses that benchmark's timer loop and empirical order-statistic P50/P95
(no interpolated quantiles). It records first-call time, warm samples and their
statistics per frozen input. Each board/precision/view cell has its own protocol
and budget; pooling cannot hide a slow input. Accuracy results contain no
latency samples. Both receipts must agree on input, extension, checkpoint,
configuration, calibration, hardware and exact source SHA. Accuracy and performance
also record paths and SHA256 of the **loaded** CUDA driver, runtime, cuBLAS and
cuBLASLt libraries. The aggregator compares their digests against each other
and the approved ApxInf baseline; paths can differ after relocation. The
OpenPI reference must be generated on the same hardware family as its cell,
recording GPU name/capability, PyTorch/CUDA build, actual math flags and loaded
library digests. A Thor reference cannot qualify an Orin cell: identical OpenPI
source/weights/inputs already produced failures against each other on these
boards in the recorded diagnostics. Different approved reference/runtime recipes
remain explicit per-board bank assets; never silently replace a golden.
Installed toolkit version alone is insufficient.
Old receipts without these fields must be rerun before gate acceptance.

The suite benchmark and accuracy CLI forward `--autotune` and `--tactics`
explicitly to the native loader. The v6 daily runner creates an isolated recipe
directory per cell and first runs the entire suite with tuning enabled. A fresh
process then collects accuracy/stability with tuning disabled; another process
measures performance against the same directory. Preparation has its own output
receipt and is excluded from latency sampling. Use a new output directory for
each run; an existing per-cell recipe directory is rejected.

Receipts hash sorted relative filenames and contents of persisted `*.recipe`
files before and after inference. The evaluator requires nonempty, unchanged
caches for baseline, accuracy and performance, and matching caches for candidate
accuracy and performance. Aggregation also verifies the preparation receipt's
source, extension, inputs, checkpoint, configuration, precision, runtime libraries
and resulting cache against the candidate. A baseline built from empty-cache
fallbacks, a missing receipt or tuning during measurement cannot qualify.
Every measured input and shape return also records the native runner's
`execution_mode()`. Baseline and candidate receipts must report `graph`; eager
fallback, runtime-managed and missing modes are rejected. The query runs outside
the timed warm sample loop. Old receipts require a new run. The
baseline and candidate may have different algorithm selections; their own caches
must remain fixed during measurement. A nonempty cache alone does not prove that
every operator has an optimized implementation; measured budgets and operator
coverage review remain necessary.

A shared Orin diagnostic confirmed why this matters: a 1-view empty-cache BF16
run measured about 2.05/3.23 seconds for T10/T200. A separately prepared 2-view
baseline measured about 194/182 milliseconds. These are different view counts,
so they are evidence of a configuration problem, not a paired speedup claim.
Neither diagnostic defines the formal performance budget.

Calibrate sample and warmup counts using repeated independent runs: verify
warmup convergence, estimate per-case variation and tail behavior, and test
whether the intended regression is reliably distinguishable. Ten repetitions
are not an approved P95 protocol. Keep first/cold/shape-switch and warm budgets
separate. The example's zero warmups/one sample is deliberately unqualified and
cannot establish a deployable performance baseline. Missing budgets still fail.
`performance_limits` maps **each case name** to its approved `p50_ms`, `p95_ms`
and `first_call_ms` ceilings. The names must match the frozen suite exactly.
A common ceiling derived from the slowest case could let a formerly fast case
regress substantially; pooled statistics are informational and never approve a
case. For example, a 10ms baseline and a 100ms baseline need separate measured
budgets, rather than giving both cases the latter's ceiling. These numbers
illustrate the failure mode and are not proposed gate thresholds.

The implicit runtime caches only the latest prepared token shape. A different
token count can rebuild its CUDA graph; constant-shape warm samples cannot
detect a regression when returning to an earlier shape. Performance therefore
adds one B -> A shape-first call after the warm phase, using the first frozen
case and the last case with a different token length. It records both input
hashes and checks a separate `shape_switch_limit_ms` budget per cell. This is
one first-event measurement, rather than a stationary latency percentile.
Repeated independent runs must establish its budget. Suites without two token
lengths or cells without an approved switch budget fail before compilation.

An Orin one-view BF16 diagnostic pilot collected 100 warm samples per input
for eight observations: per-case P50 was 167.9–168.6ms and P95 169.2–171.9ms.
The B -> A shape-first call took 791.7ms, versus 850.0–860.1ms for initial
preparation of the two token lengths. Before/after activity snapshots both
contained desktop GPU clients, so this pilot is not an accepted performance
baseline or paired regression. It demonstrates why the switch needs its own
measurement; it does not identify the cause of an uninstrumented robot log.

The v4 parity, v2 performance and v5 bank schemas require regeneration; old mixed-purpose
receipts are historical evidence and cannot be silently promoted into the new
protocol.

## Freeze the approved bank

Copy `configs/pi05/ci-bank.example.json` into the ignored bank directory and
edit its profiles/policies. Paths may reuse external checkpoint caches in place.
The generator expands profiles × board/precision × views and records SHA256
for checkpoint weights, config, calibration, suite manifest and both outputs.
Suite manifests separately bind every input NPZ.

```bash
python scripts/pi05_ci.py freeze \
  --definition devlocal/pi05-ci-gate/bank/definition.json \
  --bank devlocal/pi05-ci-gate/bank/bank.json
```

Review and distribute this small bank descriptor and frozen input/output files;
weights stay in existing board caches. Absolute paths in the approved descriptor
must resolve consistently on the corresponding board. The full bank SHA is an
operator setting; a PR cannot choose another bank or edit the trusted harness.
Changing checkpoint, preprocessing, precision recipe or reference semantics
requires an explicit bank review and regeneration with pinned OpenPI.

Reference receipts are board-specific: the template uses
`{hardware}-openpi.json` so Thor and Orin cannot silently reuse one receipt.
Before compiling, validation checks the frozen reference/baseline provenance,
checkpoint configuration, runtime identities and complete output shapes. An
approved baseline must itself pass the official-reference limits for every
chunk and action step; hashing a known failing baseline does not qualify it.

## Offline board deployment

Linux ARM64 runners need Git, Bash, flock, realpath, Rust/Cargo, the supported
CUDA toolchain, `nvpmodel`, `jetson_clocks`, Thor's `nvidia-smi` or Orin's
`tegrastats`/`stdbuf`, and a Python environment
with `scripts/requirements-pi05-ci.txt` (NumPy and Pillow; Pillow is also needed
by the CPU benchmark-import tests). The wrapper builds `apxinf-py` with
`cuda,extension-module` against that Python. CUDA builds need the repository's
native-kernel dependencies and may take substantially longer than inference.
The GPU job's 90-minute timeout is an operational cap, not a latency budget:
the first SM87 build exceeded 40 minutes before the six model runs started.
The candidate checkout retains ignored Cargo caches between runs; tracked or
untracked source dirt is still rejected. The wrapper selects the extension from
Cargo's successful artifact record rather than assuming a pre-existing filename.

Reference generation additionally needs pinned OpenPI, CUDA PyTorch,
safetensors, Transformers 4.53.2 with OpenPI's replacement modules, and the
OpenPI import dependencies (JAX/Flax, augmax, dm-tree). These are **reference-side
extras**, not daily ApxInf dependencies. Use the installation procedure of the
pinned OpenPI revision; do not install incompatible generic Transformers into
the daily runtime.

Create an operator-owned environment JSON per board with `hardware`,
`nvpmodel_sha256`, `clocks_sha256`, and `temperature_ceiling_c`. The digests are
of stripped `nvpmodel -q` and `jetson_clocks --show` stdout, excluding lines
starting with `FAN Dynamic Speed Control=` (the automatic fan PWM varies).
Configure stable
clocks before collecting baseline; this script never changes power settings.
On Jetson, the clocks query requires root. Give the runner narrowly scoped
passwordless permission for `sudo -n /usr/bin/jetson_clocks --show`, not broad
sudo access. The stock preflight uses that read-only command.
The temperature ceiling must come from the approved performance envelope.
Exact clock receipt checks intentionally reject a changed operating mode.

Set these environment variables in the runner service or local shell:

| Variable | Value |
| --- | --- |
| `APXINF_CI_PYTHON` | Absolute runtime Python path |
| `APXINF_CI_BANK` / `APXINF_CI_BANK_SHA256` | Approved bank and digest |
| `APXINF_CI_PREFLIGHT` / `APXINF_CI_PREFLIGHT_SHA256` | Operator-owned copy of `preflight_pi05_ci.py` and digest |
| `APXINF_CI_ACTIVITY_AUDIT` / `APXINF_CI_ACTIVITY_AUDIT_SHA256` | Operator-owned read-only activity auditor and digest |
| `APXINF_CI_ENVIRONMENT` / `APXINF_CI_ENVIRONMENT_SHA256` | Board environment JSON and digest |
| `APXINF_CI_GPU_LOCK` | Shared developer/CI lock; default `/tmp/apxinf-thor-gpu.lock` or `/tmp/apxinf-orin-gpu.lock` |

Make the preflight executable and ensure its interpreter can read the vendor
tools. Run `bash /trusted/scripts/run_pi05_ci.sh thor SHA /candidate /candidate/devlocal/pi05-ci-gate/run-001`.
The candidate must be a clean checkout at SHA. The wrapper validates the bank,
per-case budgets and every required board asset before compiling; an unqualified
template or missing file therefore fails without spending a full CUDA build.
GPU reservation uses nonblocking
flock; busy returns 75 and does not kill/preempt jobs. Developers must use the
same lock. It also checks sampled GPU utilization and approved clocks/power/
temperature before and after inference. The wrapper releases the GPU reservation
during CPU/CUDA compilation, then reacquires it and repeats preflight before
inference; a developer who takes the GPU meanwhile causes exit 75.
The trusted wrapper also polls GPU device users and compiler processes during
inference. A foreign process invalidates the run and stops only the CI process
group. The hosted aggregate requires `activity.json` and checks its raw samples;
a green summary cannot override interference or missing coverage. SIGINT/SIGTERM
cancellation saves a failed receipt and terminates the monitored process group;
repeated cancellation cannot interrupt that cleanup. A child that ignores
SIGTERM is killed after a five-second grace period. Sampling does
not prove that no sub-second interference occurred; developers must still use
the lock and coordinate a quiet measurement window.

Install a root-owned copy of `monitor_pi05_ci.py` under `/opt/apxinf-ci/`.
The same operator-owned auditor executable works on both boards:

```sh
#!/bin/sh
case "$(tr '\0' ' ' < /proc/device-tree/compatible)" in
  *tegra234*) exec sudo -n /usr/bin/python3 /opt/apxinf-ci/monitor_pi05_ci.py audit ;;
  *tegra264*) exec /usr/bin/python3 /opt/apxinf-ci/monitor_pi05_ci.py audit ;;
  *) exit 1 ;;
esac
```

On Orin, give the runner passwordless permission for that exact read-only command only.
The root-owned script and its parent directory must not be writable by the
runner. Full `/proc` descriptor visibility is required on Orin: an unprivileged
scan can silently miss another user's GPU process. Thor uses the unprivileged XML process inventory from `nvidia-smi -q -x`,
including graphics clients that the compute-apps query omits;
Orin checks compute device descriptors, including `/dev/nvhost-gpu`,
`/dev/nvhost-as-gpu`, `/dev/nvgpu/igpu0/ctrl`, and anonymous nvgpu/host1x GPU
channels. Missing telemetry or permission errors reject the run
instead of accepting a partial scan. Approve the executable digest as `PI05_CI_ACTIVITY_AUDIT_SHA256` in GitHub
and set its path in the runner environment. This PR does not install it or grant
sudo access.

Save build/preflight logs, raw outputs, activity samples and summary JSON.

## GitHub rollout after this PR

1. Register two dedicated Linux ARM64 runner services on the local boards,
   labels `pi05-ci,thor` and `pi05-ci,orin`. They connect outbound to GitHub;
   no inbound public GPU endpoint is needed. Use operator-owned assets and
   service configuration. Do not put registration tokens into the repository.
2. Set repository variables `PI05_CI_BANK_SHA256`, `PI05_CI_PREFLIGHT_SHA256`,
   `PI05_CI_ACTIVITY_AUDIT_SHA256`, `PI05_CI_thor_ENVIRONMENT_SHA256`,
   `PI05_CI_orin_ENVIRONMENT_SHA256`. After
   offline acceptance, set `PI05_CI_ENABLED=true`. Without it dispatch fails.
3. A maintainer reviews an **internal** PR and dispatches `PI05 GPU gate` from
   the default branch with PR number and the reviewed full head SHA. CPU
   contracts run automatically. Dispatch checks that the PR head is current.
4. Each board builds the candidate while running the harness from the reviewed
   default branch. Board concurrency and the shared lock reserve hardware.
   Missing runners queue; busy/invalid environments fail and must be resubmitted.
5. A GitHub-hosted publisher validates both complete reports and their bank/SHA,
   verifies artifact input/golden hashes and recomputes accuracy/performance
   from the raw actions and latency samples rather than trusting green summaries,
   then writes `pi05/gpu` on the **tested PR SHA**. Only this hosted job has
   status-write permission. A newer PR commit needs a new dispatch. Failure or
   stale/partial evidence produces failure, not a green partial matrix.
6. Once budgets, both boards, representative fixtures and reporting are accepted,
   make `pi05/gpu` required in repository rules. Keep CPU contracts required as
   appropriate. Neither setting is changed by this PR.

These persistent shared development machines are not a sandbox for hostile PR
code. Maintainer dispatch is authorization to run trusted internal changes,
not isolation. GPU jobs have no status-write credentials, but compromised jobs
can still damage same-user assets and forge same-host results. Do not enable
automatic fork PR execution; use disposable isolated workers for that threat
model. Host hardening, read-only mounts and resource ownership remain operator
responsibilities. See [GitHub's self-hosted runner security guidance](https://docs.github.com/en/actions/reference/security/secure-use).

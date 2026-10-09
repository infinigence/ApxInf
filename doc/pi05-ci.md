# PI05 CI: offline deployment and rollout

This PR prepares the CI tools and workflows. It does **not** register runners,
enable GPU dispatch, or change branch protection. An incomplete board matrix,
missing asset, uncalibrated performance budget, or failed preflight cannot pass.

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
regenerated separately by two frameworks. The same observation can produce
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
bank. It contains eleven explicitly constructed cases: gradient/T10, a second
noise draw, T21, the configured token-length boundary, black image only, zero
noise only, black+zero together, white image only, negative noise only,
white+negative together, and distinct constant gray levels per camera. The
last case does **not** swap camera order. T21 is a concrete workload, not a
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
metrics are JSON null and require the explicit `zero_max_abs` budget.

There are two independent comparisons: official OpenPI and a frozen approved
implementation on the same hardware/precision/calibration. Do not advance the
approved baseline automatically with main: this would permit gradual drift.
Per-channel physical accuracy still requires policy-level evaluation; padded
32-dimensional latent output cannot certify robot units.

The existing `pi05_bench.rs` reference floors (.999/.05 BF16, .997/.10 FP8,
.995/.10 INT8 with .125 max abs) are historical comparison policies, not newly
calibrated deployment guarantees. [Edge-LLM's PI05 comparator](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/experimental_models/pi05/scripts/compare_pi05_actions.py)
uses .99999 cosine and .005 max abs for its configuration. Do not apply those
numbers blindly to ApxInf's different dtype and implementation.

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

Stability is a separate phase enabled by `--stability-repeats`: compare repeated
calls against the saved accuracy output, then explicitly run A -> other suite
inputs -> A. Its drifts are separate from accuracy outputs and latency. Daily
bank policies require an explicit positive stability count, chosen after
same-input and rebuilt-extension measurements; the example's one repetition is
an unapproved placeholder, not an established protocol.

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
boards (see the diagnosis below). Different approved reference/runtime recipes
remain explicit per-board bank assets; never silently replace a golden.
Installed toolkit version alone is insufficient.
Old receipts without these fields must be rerun before gate acceptance.

Calibrate sample and warmup counts using repeated independent runs: verify
warmup convergence, estimate per-case variation and tail behavior, and test
whether the intended regression is reliably distinguishable. Ten repetitions
are not an approved P95 protocol. Keep first/cold/shape-switch and warm budgets
separate. The example's zero warmups/one sample is deliberately unqualified and
cannot establish a deployable performance baseline. Missing budgets still fail.

The v4 parity and v2 bank schemas require regeneration; old mixed-purpose
receipts are historical evidence and cannot be silently promoted into the new
protocol.

## Qualification snapshot (2026-10-09)

Product math at `5c7ac664` completed the twelve board/precision/view cells with
148 first accuracy outputs and 4,440 independent warm samples. Two stability
repeats and an A -> other inputs -> A revisit measured zero drift throughout.
These pilot counts are exploratory, not an approved protocol. Twenty-one CPU
contract tests and the focused Rust weight-folding regression pass.

The real LIBERO 2-view INT8 failure was traced to folding learned language
RMSNorm into projections before per-output-channel weight quantization. Keeping
raw weights and explicit normalization improves state cosine .926409 -> .999824
(relative L2 .383409 -> .018752) and text .901550 -> .999696
(.439377 -> .024661). Both complete chunks and every action timestep meet the
historical INT8 comparison floors. BF16/FP8 retain folding in their own packing
paths; all nine corresponding board/view cells are bitwise unchanged on the
historical overlapping inputs. Random unit-normalization weights would miss
this learned-checkpoint failure.

This does not qualify the entire model: base BF16 still fails black+zero,
white+normal, zero-noise and some real token-mode comparisons. Base FP8 and
INT8 also fail ordinary gradient/noise and real-input cases. Official prefix
KV substitution improves several failures but leaves action-side error.
Layer/flow traces measure iterative amplification. Separate GeGLU gives the
same output; split normalization or an FP32 patch projection does not repair
all cases. FP8 group ablations identify strong action-activation sensitivity
on gradient+zero and vision/language sensitivity on other inputs. Increasing
the old H10 calibration margin is insufficient. A per-site H50 recalibration
also improves some inputs and worsens others; it is a diagnosis experiment with
overlapping calibration/test inputs, not a production profile. No numerical
limit was relaxed.

The full v4/v2 raw receipt matrix was revalidated through the maintained
aggregator. Exact same-build comparisons and stability pass, while missing
performance budgets correctly prevent acceptance. A representative held-out
bank, genuine three-camera observations, rebuilt variability and performance
qualification remain outstanding. The system is **not accepted and not online**.
Fixed-observation chunk agreement is not closed-loop or LIBERO task-success
certification.

## Matched-operand operator diagnosis

Further native instrumentation at product math `5c7ac664` captures operands,
quantized codes/scales and outputs, then recomputes them independently with
PyTorch. Instrumented complete action outputs are bitwise identical to the
product receipts on the tested inputs. This controls for changes introduced by
the probes. No new product arithmetic change follows from these experiments.

| Path | Matched-operand evidence | Interpretation and scope |
| --- | --- | --- |
| BF16 language | QKV/output/down GEMM and residual sums match exactly on three inputs across all 18 layers; a complete layer recomputed from its input has relative L2 at most .000652 under the native recipe, versus .003526 under the selected official recipe. | Explicit versus folded RMSNorm, FP32 RoPE and GeGLU intermediates introduce different rounding boundaries. Sampled GEMM is not the source of that difference. |
| BF16 vision | Layers 0/13/26, three inputs: sampled GEMM, fused residual and FP32 GELU match; native attention agrees with SDPA within .000022 relative L2. | Official sequential BF16 bias/residual or activation rounding differs by roughly .001–.003 locally. FP32 patch embedding is another measured contributor, but changing it alone does not repair all chunks. |
| BF16 action | Layers 0/8/16 at first and last flow steps, three inputs: GEMM, fused gate/residual and native GeGLU match exactly; adaptive norm relative L2 is at most .000061. | Official separate BF16 gate multiplication/residual addition and activation rounding differ by roughly .002–.004 locally. |
| FP8 prefix/action | Independent E4M3 code/scale reconstruction agrees with sampled native GEMM to relative L2 at most .000066; GeGLU quantization differences are at most about .0002. | Quantization and iterative amplification remain causes of quality loss; these probes do not establish an FP8 GEMM defect. |
| Orin INT8 | 108 sampled GEMMs: 36 prefix and 72 action, three inputs, layers 0/8/16 and first/last action flow steps; actual integer codes/scales produce bitwise identical outputs. Activation codes, sampled weight codes and weight-scale bits also match. Native gate/residual and GeGLU match; norm differences are at most .000037. | The confirmed learned-norm packing bug was repaired separately. No integer GEMM/quantizer defect is found in these samples; quantization recipe and BF16 rounding remain contributors to base-model failures. |

The actual official `sample_actions` path forces **eager language/action
attention**, despite the constructor's SDPA config. Eager-only interventions
leave the output unchanged and are not evidence for an attention-backend fix.
Full-chunk single-factor reference changes have mixed effects: black+zero
cosine .865597 becomes .938741 with prefix FP32 GeGLU, .888755 with prefix FP32
RoPE, and .921981 with combined prefix/action changes. Other cases can worsen.
Intermediate rounding differences and their amplification are measured; no
single cast or fusion rewrite has been proved to repair every failure.

FP8 action layer 0 MLP norm at the first flow step clips about .2% of values on
black+zero and gradient+zero. Its local quantization relative L2 is about .176
and .178, versus about .026–.029 at typical unclipped sampled sites. A controlled
scale change at only that site changes gradient+zero cosine .464966 -> .361706
(relative L2 3.767167 -> 6.754926); also changing layer 16 yields .362401 /
6.707144. All three profiles still fail all 13 two-view diagnostic inputs under
historical whole-chunk plus timestep floors. Full per-site H50 recalibration
has different mixed results, documented above. Eliminating clipping is
insufficient: changing static range also changes resolution and the iterative
trajectory. These overlapping calibration/test probes are not production
profiles and are not a reason to relax limits.

These operator samples narrow the causes; they do not exhaust every layer,
token, checkpoint, view or board, prove complete port correctness, or certify
remaining failed cases as acceptable. The OpenPI disagreement stays red.

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

## Offline board deployment

Linux ARM64 runners need Git, Bash, flock, realpath, Rust/Cargo, the supported
CUDA toolchain, `nvpmodel`, `jetson_clocks`, Thor's `nvidia-smi` or Orin's
`tegrastats`/`stdbuf`, and a Python environment
with `scripts/requirements-pi05-ci.txt`. The wrapper builds `apxinf-py` with
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
| `APXINF_CI_ENVIRONMENT` / `APXINF_CI_ENVIRONMENT_SHA256` | Board environment JSON and digest |
| `APXINF_CI_GPU_LOCK` | Shared developer/CI lock; default `/tmp/apxinf-thor-gpu.lock` or `/tmp/apxinf-orin-gpu.lock` |

Make the preflight executable and ensure its interpreter can read the vendor
tools. Run `bash /trusted/scripts/run_pi05_ci.sh thor SHA /candidate /candidate/devlocal/pi05-ci-gate/run-001`.
The candidate must be a clean checkout at SHA. GPU reservation uses nonblocking
flock; busy returns 75 and does not kill/preempt jobs. Developers must use the
same lock. It also checks sampled GPU utilization and approved clocks/power/
temperature before and after inference. The wrapper releases the GPU reservation
during CPU/CUDA compilation, then reacquires it and repeats preflight before
inference; a developer who takes the GPU meanwhile causes exit 75.
These checks cannot prevent a developer
who bypasses the reservation from starting work mid-run; coordinate a quiet
measurement window. Save build/preflight logs, raw outputs and summary JSON.

## GitHub rollout after this PR

1. Register two dedicated Linux ARM64 runner services on the local boards,
   labels `pi05-ci,thor` and `pi05-ci,orin`. They connect outbound to GitHub;
   no inbound public GPU endpoint is needed. Use operator-owned assets and
   service configuration. Do not put registration tokens into the repository.
2. Set repository variables `PI05_CI_BANK_SHA256`, `PI05_CI_PREFLIGHT_SHA256`,
   `PI05_CI_thor_ENVIRONMENT_SHA256`, `PI05_CI_orin_ENVIRONMENT_SHA256`. After
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


## Repair performance regression (2026-10-09)

The learned-RMSNorm INT8 packing repair was rerun on Orin with separate fresh
processes in **before -> after -> before** order. Only one model is resident at
once. Each checkpoint/view profile uses two fixed inputs, three timing blocks
per input per process, 30 warmups and 100 measured samples per block. The
maintained `bench_pi05.py` timer and empirical P50/P95 definitions are reused;
all 7,200 raw INT8 samples are retained without tail filtering. This is a repair
regression experiment, not an approved merge-gate sample count or budget.

| Orin INT8 profile/input | Before P50/P95 ms | After P50/P95 ms | P50/P95 change |
| --- | --- | --- | --- |
| LIBERO 2-view state | 130.244 / 130.639 | 129.747 / 129.995 | -0.382% / -0.493% |
| LIBERO 2-view text | 127.871 / 128.085 | 127.717 / 127.876 | -0.120% / -0.163% |
| Base 1-view T10 | 87.921 / 88.190 | 87.944 / 88.146 | +0.026% / -0.050% |
| Base 1-view T200 | 105.014 / 105.296 | 105.087 / 105.354 | +0.069% / +0.055% |
| Base 2-view T10 | 127.839 / 128.215 | 127.866 / 128.335 | +0.021% / +0.093% |
| Base 2-view T200 | 145.007 / 145.398 | 144.987 / 145.321 | -0.014% / -0.052% |
| Base 3-view T10 | 167.107 / 167.532 | 166.972 / 167.511 | -0.081% / -0.012% |
| Base 3-view T200 | 184.264 / 184.656 | 184.198 / 184.575 | -0.036% / -0.044% |

The comparison shows no performance regression distinguishable from measured
run variation on these inputs. Every block reports zero major page faults and
zero process swap; rebuilding the original model reproduces identical actions.
The independent LIBERO accuracy replay also reproduces the improved cosine,
relative L2 and all-timestep verdicts stated above. This does not certify every
input or a zero-cost repair. An earlier two-model-resident attempt experienced
swap and a worker exit; it is preserved as invalid evidence and excluded.

Further BF16 experiments explicitly reproduce the official weight casts,
normalization, intermediate BF16 GeGLU/gated-residual/RoPE rounding and FP32
auxiliary projections/flow state. Ordinary inputs improve, but complete
whole-chunk plus timestep checks still fail. These private candidates are not
product changes. An initial auxiliary projection lost graph capture and was
rejected for its latency increase; a workspace-based version restored capture.
Remaining candidates must establish precision and controlled performance
before promotion. The gate remains unaccepted and offline.

The same A/B/A protocol also completed 10,800 Thor BF16/FP8 samples and 5,400
Orin BF16 samples, covering base 1–3 views at T10 and T200. All paired actions
are bitwise identical; all blocks report zero major faults and zero process swap.
Orin BF16 P50/P95 deltas range from -0.370% to +0.384%. Thor BF16/FP8 deltas
range from -0.286% to +0.469%. Two Thor FP8 input profiles initially had non-overlapping block intervals.
A further 3,600-sample **after -> before -> after** run on FP8 2/3 views
produces overlapping block intervals throughout; 3-view T10 changes sign.
Fixed/original P50/P95 differences in that run range from -0.064% to +0.180%.
Together, 27,000 retained paired samples show no reproducible regression beyond
measured process variation on the selected inputs. They do not prove identical
latency or supply an approved performance budget.
Longest text is not necessarily slowest: Orin 2-view BF16 T10 is slower than
T200. These two inputs do not constitute worst-case performance coverage.


## Further numerical controls (2026-10-09)

All following controls are private experiments, with unchanged official
goldens and historical limits. No candidate below has been promoted.

| Thor BF16 control | Full chunk and all-step passes | T10 warm P50 ms |
| --- | --- | --- |
| Complete staged arithmetic, 2 views | 8/13 | 90.100 |
| Complete staged arithmetic, 3 views | 10/11 | 100.945 |
| 2-view prefix row padding alone | 8/13 | 95.964 |
| 2-view precise attention softmax | 8/13 | 93.299 |
| 2-view explicit masked attention columns | 8/13 | about 99 |

These are exploratory separate benchmark runs (10 warmups, 30 retained
samples/input), not the controlled A/B/A repair qualification above. Their
latency increases and remaining numerical failures prevent promotion.
The three-view staged control passes black+zero but still fails distinct
camera gray on timestep checks. Success on three views does not establish
why a different two-view input fails.

Reference-only controls distinguish several causes. Batching camera encoding
changes none of ten selected official outputs. Removing masked image tokens
changes output despite identical valid images/tokens/noise: white-input whole
relative L2 reaches .133. The official golden is retained. Native row padding
alone does not reproduce masked attention or cure black+zero.

Replacing official vision output restores white and distinct-gray all-step
checks in the complete staged two-view control. Replacing all official KV
still leaves black+zero and gradient+zero red. The numerical differences
span prefix and action paths; no single successful local operator comparison
proves a repaired complete chunk. Precise MQA and RoPE controls on 24 matched
action sites are bitwise equal to Torch, but their full chunks remain red.

Loaded-library inspection finds system CUDA 13.2/cuBLAS 13.4 in standalone
ApxInf and bundled CUDA 13.0 libraries in OpenPI. Live cuBLAS operator probes
inside a Torch process inherit those bundled libraries; this does not
automatically establish their standalone behavior. Comparisons against saved
standalone native operator outputs remain separate evidence. Runtime alignment
does not repair the full two-view chunks: product BF16 retains 6/13 passes
and the masked-attention diagnostic retains 8/13. It is not adopted as a
repair or a relaxed threshold.


### Continued auxiliary and receipt regressions

Matched F32 auxiliary tests cover 83 actual linear operands and 20 time-MLP
SiLU operands. Checkpoint-layout weights remove 29 BF16 rounding differences
under Torch's bundled cuBLAS; system cuBLAS still has 19 differences. Precise
SiLU is bitwise equal to Torch on those operands; fast math has small F32
changes despite equal BF16-rounded outputs. These local controls do not certify
a complete model. The checkpoint-layout two-view candidate retains 8/13 passes
and T10/T200 warm P50 89.834/94.679ms; it is not promoted.

The loaded-library receipt change is tested separately from the 27,000-sample
INT8 repair qualification. New Thor BF16 and Orin INT8 two-view receipts have
13 first outputs, zero repeat/revisit drift and matching accuracy/performance
DSO hashes. Each board has two opposite-order three-process sweeps (30 warmups,
100 retained samples/input, T10/T200). Thor additionally has six alternating
processes, with 1,200 retained samples and board telemetry. An initial Thor
T200 P50/P95 increase of 1.298%/1.454% shrinks in reverse order and changes
sign in the alternating run: T10 -0.043%/-0.108%, T200 -0.015%/-0.249%.
Orin reverse-order differences are T10 +0.031%/+0.052%, T200 +0.103%/+0.083%.
The evidence does not reproduce a systematic warm regression. Raw tails and
initial asymmetric results are retained; no acceptance budget was loosened.
These selected-input repair experiments do not approve the full deployment
matrix or replace the pending representative bank and measured budgets.


The precise-SiLU full intervention yields 13 bitwise-equal chunks to the
checkpoint-layout candidate; it remains 8/13 and is not promoted. With the
auxiliary-layout candidate, replacing all official KV makes black+zero pass
every action timestep (.999628 cosine/.029057 relative L2), while gradient+zero
still fails (.993072/.122654). Official vision replacement restores white and
gray. These controls identify contributions, not a product repair. Replacing
each flow input with the official state reduces gradient+zero final relative
L2 to .037334; resetting only flow step 3/6/9 gives .085689/.037853/.037334.
These whole metrics improve, but per-action-timestep checks still fail.
Single-step discrepancies remain and accumulate. The official capture
reproduces both tested original goldens bitwise. Injected/trace runs disable
capture and have no product performance meaning.


### Actual reference precision policy

The pinned OpenPI constructor sets float32 matmul precision to `high`. Reference
receipts now record the resulting matmul policy and CUDA/cuDNN TF32 switches.
A reconstructed `highest`-precision F32 oracle does not reproduce this policy.
On ten actual action-input operands, `high` reproduces the observed reference
BF16 embeddings exactly; full F32 GEMM differs in up to 5,706 BF16 elements,
and cuBLAS TF32 mode reproduces every observed embedding. Both actual language
and expert `inv_freq` buffers are BF16 after model conversion; 127/128 values
differ from the earlier F32-formula control. These findings supersede broad
reference-equality claims from reconstructed high-precision/RoPE oracles.
The corrected-buffer and TF32 interventions remain private pending complete
output and performance qualification; no thresholds or goldens were changed.


### TF32 action-path isolation and performance rejection

With official full prefix KV and each official flow input injected, the staged
BF16-buffer/physical-mask/checkpoint-layout/TF32 control produces all ten next
flow states bitwise equal to actual OpenPI under its bundled CUDA libraries.
The same control under system libraries still has final relative L2 .023268.
This isolates reference policy and runtime contributions; injected states do
not establish complete native inference correctness.

The non-injected two-view candidate passes 9/13 diagnostic cases under either
runtime. Under bundled libraries, gradient+zero remains .995106 cosine/.099584
relative L2, black+zero .828698/.564806 and white .999138/.077142. Black with
normal noise also fails per-timestep checks despite .999836/.018779 whole
metrics. System-library warm T10/T200 P50 is 99.143/104.040ms versus the existing
approximately 83/86ms product; bundled-library T10 is approximately 123ms.
These exploratory 10-warmup/30-sample measurements reject promotion as-is;
they are not paired performance qualification or an approved budget. Remaining
prefix errors and the latency increase must both be repaired.


### Complete two-view diagnostic parity

A private native control now passes all 13 complete chunks and every action
step at the historical BF16 comparison floors. Eleven chunks are bitwise
OpenPI; T21 and T200 have cosine .99999165/.99999221 and relative L2
.008466/.004432. Two independent repeat calls per input and the revisit phase
have zero drift. Official outputs and thresholds are unchanged.

The control combines actual BF16 RoPE buffers, OpenPI's TF32 auxiliary policy,
physical masked attention columns, staged BF16 operator boundaries, the
`UNFUSE_FMA` Flash softmax with precise division, reference RGB normalization,
and F32 cuDNN patch convolution under the aligned runtime. Native parameters
calling Torch's existing Flash binary were an isolation experiment; the
complete control uses separately compiled native Flash and no Torch model.
Convolution controls reproduce actual patch embeddings with cuDNN algorithms
0/1/2; ordinary GEMM has different F32 reduction rounding. Replacing only
remaining official vision output makes gradient+zero bitwise OpenPI, which
isolates its remaining numerical failure to vision before the conv control.

This is not a deployable repair. The private implementation has fixed-shape
and process-lifetime diagnostic resources, and would need maintained operator
contracts and explicit cuDNN/runtime prerequisites before adoption. Independent
30-warmup/100-sample performance has T10 P50/P95 120.125/120.478ms and T200
101.191/101.335ms, above the approximately 83/86ms product. Nsight graph-node
traces identify slower GEMM schedules and decomposed GeGLU as major costs.
The initial unfused arithmetic candidate was rejected on performance.
Subsequent fusion controls below remain private and unaccepted.


### Fusion recovery and one-to-three-view isolation

A private rounded language GeGLU fusion preserves every element of the full
2-view conv control. Extending its tested shape range and packing the staged
activation kernel retains 13/13 whole-chunk/all-step passes and zero repeat /
revisit drift. Independent w30/n100 T10 P50/P95 is 83.927/84.063ms; T200 is
91.725/91.865ms. This recovers most of the initial 120/101ms cost, but has not
yet demonstrated non-regression against the current approximately 83/86ms
runtime. The completed counterbalanced comparison below rejects this version.

The prior all-view diagnostic control passes 11/11 in 3 view, but only 11/13
in 1 view: gradient+zero fails an action-step limit (whole cosine .999367,
relative L2 .036021); white+normal has cosine .998721 / L2 .050575. Disabling
the rounded fusion reproduces every 1-view output exactly, excluding that
fusion as their cause. Padding prefix query rows does not repair them (10/13),
and some inputs exhaust graph workspace and fall back to eager. It is rejected.
A private mixed-cuBLAS-version auxiliary control exits with SIGSEGV before
accuracy/performance results; it is rejected, with crash evidence retained.
No private arithmetic or runtime recipe is promoted, and no limit is loosened.


### Physical KV caching and cuBLAS workspace isolation

Three blocks of fresh-process, counterbalanced runs (w30/n100, T10/T200,
1,800 raw samples) compared the packed control against both the deployed
runtime and the reference-aligned runtime. Packed P50 is 84.028/91.707ms;
deployed product is 83.838/86.453ms. The T200 regression rejects adoption.
Unrolling the fixed softmax reduction preserves every output and reduces
T10/T200 P50 to 81.529/88.775ms. Removing physical masked columns is faster
but fails three cases, so that control is rejected.

Caching physical prefix KV padding once per layer, with an explicit active
camera mask boundary, retains all 13 two-view whole-chunk/all-step passes,
bitwise identical to the packed control, with zero repeat/revisit drift.
Three further counterbalanced blocks (1,800 samples) give cached P50
79.388/86.197ms versus deployed product 83.771/86.230ms. This is still not
same-library non-regression: aligned product T200 is 84.473ms. The first
cached-padding attempt exposed dummy cameras, failed all cases and is excluded.

One-view white with official vision output first differs at the language
layer-0 down projection: 3,890 BF16 elements. Shape padding alone is disproved
by a standalone actual-tensor oracle. Identical `cublasGemmEx` arguments using
a fresh handle differ; the prepared Torch handle matches. An owned explicit
128KiB/1MiB/4MiB/8MiB workspace matches; default and 16/32/64MiB differ. This
is evidence of workspace-dependent numerical scheduling, not a reason to
allocate arbitrarily larger workspaces. [PyTorch 2.9.1](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/cuda/CublasHandlePool.cpp)
sets workspace after setting the stream; its non-SM90 default is 8MiB+128KiB.

The private cached-prefix/checkpoint-TN/owned-8MiB control passes 13/13 in
2 view (12 bitwise chunks) and 11/11 in 3 view (10 bitwise). One view remains
11/13: black+zero fails step limits and white+normal cosine .993169 / relative
L2 .121118 fails. Independent two-view w30/n100 P50/P95 is 79.799/80.022ms
(T10) and 84.340/84.571ms (T200); the completed three-block counterbalanced regression (1,800 raw samples) gives
candidate P50/P95 79.804/80.048ms T10 and 84.300/84.631ms T200, versus deployed
83.785/84.077ms and 86.516/87.241ms. Same-library T200 product is
84.376/84.597ms, with overlapping block medians (product 84.227–84.509,
candidate 84.203–84.467ms). No reproducible warm regression is measured on
these two inputs. Aligned product T10 107.872ms includes its graph-workspace
fallback and is not evidence of a kernel-only speedup.
This recipe is diagnostic: it adds checkpoint-layout weights and changes
handle resources, and has no maintained operator/lifecycle contract yet.
No numerical prototype is promoted, no threshold is loosened and GPU CI stays
disabled until all required cells and performance budgets are qualified.


### One-view vision recipe and per-view performance rejection

Actual one-view hooks reproduce all three diagnostic goldens unchanged.
Gradient/black/white first differ at vision layer-0 biased FC2 (685/752/809
BF16 elements); all earlier operators in that layer are bitwise. Replacing
official vision makes each complete chunk bitwise. Actual-operand cuBLASLt
BIAS oracles on both NN/TN layouts find all first eight recipes bitwise with
0/128KiB/1MiB limits, but the first recipe at 4–16MiB produces the 809 white
mismatches. The existing native preparation uses a 32MiB preference. A private
zero-workspace FC2 control restores one-view **13/13**, twelve bitwise chunks;
T21 cosine .99997503 / relative L2 .017476. This remains a numerical prototype.

Other-view A/B/A w30/n100 (1,200 raw samples) finds clear warm regressions
before enabling actual all-view fusion: one-view T10/T200 74.460/78.225ms
versus repeated product 69.322/70.546ms; three-view 96.413/106.736ms versus
92.848/96.008ms. Neither a two-view speedup nor an all-view numerical pass
justifies accepting this global recipe. FC2 repair alone lowers one-view to
73.530/77.296ms, still above product.

The first fusion toggle experiment did not cover real fusion in one/three
view: `language_dual_geglu_shape_possible` only prepares M522/533 weights.
Both switch positions therefore executed the same path. A subsequent private
BF16 loading control prepares the already tested M<=1024 rounded fusion for
all views; complete one/three-view outputs remain bitwise the FC2 control.
Independent w30/n100 P50 T10/T200 is 71.750/75.753ms (one view),
79.563/84.164ms (two), and 92.873/98.658ms (three). One-view and three-view
T200 still have measurable performance debt. No prototype is promoted.

The same candidate under deployed CUDA/cuBLAS passes 12/13, 11/13, 11/11
for one/two/three view. Remaining failures are gradient+zero action steps in
one/two view, plus black+zero in two view (cosine .998798 / L2 .059593).
Aligned-library results cannot certify the deployed-library contract. The
first independently compiled SM87 Orin candidate passes 0/13, 0/13, 10/11
for one/two/three view despite matching checkpoint/config/input hashes.
Actual one-view OpenPI hooks reproduce the unchanged white golden. First
native action input relative L2 is .001067 and modulation about 1e-7, but
first attention L2 is .31912. Official vision/all-prefix-KV replacements do
not remove the large error. A targeted adapter dump then finds that SM87
selects FA2 and bypasses the private paired mask/cache adapter, exposing dummy
camera positions. This is an experimental adaptation error, not a regression
in the shipped INT8 fix. Correct private dispatch recovers **11/13, 9/13,
10/11**. Remaining failures include one-view task text/gradient-zero step
limits; two-view gradient-zero, black-zero, white and distinct camera gray;
three-view white step limits. These candidates are not accepted repairs.

Corrected SM87 dispatch w30/n100 P50 T10/T200 is 173.428/197.961ms,
255.797/263.143ms and 293.998/327.525ms for one/two/three view. Prior fixed
product paired medians are 134.507/150.185ms, 213.449/203.234ms and
231.341/256.228ms. The pilot candidate is clearly slower; it does not meet the
performance requirement. It is not promoted or used as a CI baseline.

Actual-operand FC2 CUDA-event microbench (w10/n100 per heuristic) finds the
exact NN zero-workspace recipe about .0225ms versus .0436–.0439ms for the
inexact first 4–16MiB recipe. The FC2 repair itself is faster. One-view T200
whole-model profiles instead expose additional physical attention and precise
RoPE work. A four-warp softmax scheduling control preserves all 37 chunks
bitwise but yields 71.796/75.683ms, 79.776/84.230ms and 93.187/98.458ms:
it does not remove the remaining one/three-view performance debt.


### Same-board references and lower-overhead controls (2026-10-10)

A private Thor BF16 control retains the repaired FC2 recipe, staged rounding,
checkpoint-layout prefix projection and physical masked KV layout. It prepares
BF16 RoPE sine/cosine tables once and uses matched-operand cuBLASLt recipes for
50-row action QK/PV (zero workspace, heuristic indices 4/1). All 37 diagnostic
chunks pass the unchanged whole-chunk and all-timestep floors under the bundled
CUDA 13.0 runtime: 13/13, 13/13, 11/11 for 1/2/3 views. Of these, 34 chunks are
bitwise equal to OpenPI; the three T21 chunks are not bitwise equal.

Separate 30-warmup/100-sample pilots measure T10/T200 P50 of
69.399/73.023, 77.079/81.314 and 90.020/95.485 ms. Two/three-view pilots improve
on the deployed implementation; one-view T200 remains about 3.5% slower than
its deployed 70.546 ms pilot. These are not a new paired qualification.
The same control with system CUDA libraries retains only 12/13, 11/13, 11/11
passes; gradient+zero and two-view black+zero remain failures. Reducing the
fusion pipeline from three stages to two preserves all 37 numerical verdicts
but slows the measured profiles. None of these private controls is promoted.

Orin now also has a separately saved OpenPI reference generated on SM87 with
PyTorch 2.9.1+cu130, the same pinned OpenPI source, checkpoint and exact input
hashes. The existing Thor OpenPI outputs pass only 9/13, 9/13 and 9/11 against
this Orin reference at the unchanged BF16 floors. Two-view black+zero has
cosine .986077 and relative L2 .220721 between the two official executions.
The comparison is evidence of architecture/runtime recipe sensitivity, not
proof that either port is correct, and does not replace either golden.

Against the same-board Orin reference, the shipped BF16 outputs pass
7/13, 6/13 and 7/11; the private explicitly masked candidate passes
9/13, 9/13 and 9/11. For two-view black+zero, shipped/private cosine is
.865164/.978608 and relative L2 is .501508/.279529. The private candidate also
has a measured latency increase, so it remains rejected. Explicit TF32 operand
rounding and compact FA2 controls do not cure the full matrix.

The maintained reference collector now records the actual GPU name/capability,
PyTorch/CUDA build and hardware family. The gate rejects a reference from a
different board family; CPU tests cover that rejection. Existing references
without hardware provenance cannot qualify and must be regenerated rather than
edited to claim a board. Cross-board comparisons remain useful diagnostics.

### Continued precision isolation and paired performance checks (2026-10-10)

All candidates in this section are private diagnostic controls. The maintained
INT8 learned-norm packing repair keeps its previously recorded 27,000-sample
qualification; none of the following BF16 recipes replaces that implementation,
changes a golden, relaxes a floor, or supplies an approved performance budget.

On Orin, actual first-layer operands distinguish vision projection rounding
from activation math. GELU is bitwise equal to PyTorch when fed the same FC1
output. Default two-camera packed QKV and FC1 projections differ from the
same-board reference in 403,368 and 204,372 values respectively (including both
repeated camera operands in the microcontrol). The vendor algorithm reports
`splitK=2`, `reduction=INPLACE` for both default recipes. Restricting cuBLASLt's
reduction preference to compute-type partial sums removes those differences
in the tested operands, as do selected non-split recipes. Repeated operands
establish a matrix-shape control, not three distinct real camera observations.
A numerically correct recipe's first heuristic rank is not necessarily fastest;
local projection equality is also insufficient for full-model acceptance.

A private candidate with exact vision QKV/FC1/FC2 recipes and an NN-layout
language down projection passes 10/13, 8/13 and 11/11 same-board tests for
1/2/3 views. Two-camera black+zero still fails badly (cosine .552661, relative
L2 .909827). Instrumentation verifies that its final output is unchanged.
For this input, the vision embedding and first language normalization are
bitwise equal, but the first language QKV projection has 221,994 differing
values; later layer residual errors accumulate. This separates the remaining
language/action error from the repaired first vision projection. It does not
excuse the failure or certify the remaining vision cases.

This candidate's 30-warmup/100-sample T10/T200 latency pilots are
146.104/164.854, 224.381/220.752 and 244.078/271.777ms on Orin. Corresponding
deployed BF16 pilots are 134.507/150.185, 213.449/203.234 and
231.341/256.228ms. These unpaired pilots reject the candidate as a performance
solution; they are not evidence that a maintained fix regressed.

Thor has an additional 5,400 raw samples across three interleaved blocks,
three views and two token lengths, using one resident model per process,
30 warmups and 100 measured calls per input. The fixed two-view implementation
in that experiment deliberately has dual GeGLU disabled and is a fusion-off
control. A separate 1,800-sample, three-block two-view run preserves its default
`auto` fusion selection. In that default run, candidate T10 P50 is
76.462–76.627ms versus 73.592–73.791ms for the fixed implementation under the
same bundled libraries. T200 improves (80.619–80.747ms versus
84.296–84.469ms), but that does not cancel the short-token regression.
Fixed system-library measurements are 83.703–83.770ms and
86.253–87.054ms; changing the library environment cannot be used to hide the
same-environment short-token debt.

One-view candidate T200 P50 is 73.860/73.881/70.802ms across the first
three blocks, against fixed system 70.446/70.573/70.652ms. The faster candidate
process cannot be averaged with the slower ones to claim qualification. Removing
an unused transposed weight copy preserves all 37 outputs bitwise but still
measures 73.946ms T200 in a fresh process. Four additional candidate profiles
all capture the slow process; they do not identify the occasional fast-process
cause. A subsequent same-input baseline/candidate profile measures total GPU
kernel time of 71.097/74.691ms per call. Masked softmax accounts for
1.068/2.827ms respectively; it is a measured contributor, not the whole cause.
Profiling overhead means these numbers are diagnostic, not latency budgets.

Uniform reduction-policy and isolated softmax controls are being evaluated
with the same whole-chunk, every-timestep and independent latency protocols.
No candidate is accepted until the remaining precision and performance debts
are resolved. The PR remains draft and GPU CI remains disabled.


### Reference reduction-mode provenance

Actual Orin language operands isolate a distinct source of reference sensitivity.
With the saved first-layer normalized input, separate PyTorch Q, K and V
projections each reproduce the saved official result bitwise for both compact
522-row and padded 778-row matrices. Disabling PyTorch's BF16 reduced-precision
reduction changes 221,994 Q values but leaves K/V bitwise equal to their saved
outputs. A single packed QKV `F.linear` also differs in 221,994 values. Thus
matching FP32 accumulation everywhere does not reproduce this reference's
shape-dependent Q reduction. This is evidence about this layer/input; it is
not a justification for passing an end-to-end failure.

The OpenPI receipt now additionally records the actual BF16 and FP16
reduced-precision reduction flags. These flags are recorded without changing
the reference computation. Their receipt is covered by the reviewed reference
file digest. A diagnostic alternative reduction mode must be saved separately;
it must not silently replace the approved reference. The separate-language-QKV
candidate is undergoing whole-chunk and independent performance regression.

The softmax reciprocal-multiply control speeds up the Thor one-view T200
pilot to 72.369ms but fails the unchanged floors (11/13, 11/13, 10/11), so it is
rejected. Skipping masked exp/div work preserves all 37 prior candidate outputs
bitwise but measures 70.685/74.450, 76.824/81.173 and 89.252/95.551ms for
T10/T200: this is also rejected as a performance repair. No numerical control
from these experiments has been promoted.


### CUDA library alignment is required for reference qualification

A controlled Orin first-language-layer reconstruction keeps Q/K/V after RoPE
bitwise equal. With the reference's bundled cuBLAS 13.0 libraries, padded
batched attention, flattened QK, softmax and PV each reproduce the saved
attention bitwise. With system cuBLAS 13.4, the same reconstruction has exactly
714 differing output values, matching the private native trace. This identifies
a library-recipe difference for this operator/input; it does not certify the
whole model. Separate full references are being generated under the actual
system libraries without replacing the older experiment outputs.

The gate now requires official-reference and candidate CUDA runtime fingerprints
to match, in addition to their hardware families. Loaded DSO content hashes
identify the environment; changing an installation path alone does not fail.
Reference collection must use the board's reviewed CUDA libraries explicitly
when PyTorch would otherwise prefer bundled copies, then inspect the actual
loaded-library receipt. The receipt also records the actual cuDNN version.
A toolkit version label or requested library path is insufficient. CPU coverage
rejects a same-board reference with a changed cuBLAS digest. Cross-runtime
comparisons remain diagnostics, and cannot qualify a merge gate.

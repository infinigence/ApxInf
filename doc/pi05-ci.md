# PI05 CI: offline deployment and rollout

This PR prepares the CI tools and workflows. It does **not** register runners,
enable GPU dispatch, or change branch protection. An incomplete board matrix,
missing asset, uncalibrated performance or shape-switch budget, or failed preflight cannot pass.

## Acceptance status

**Not accepted / not online.** The shared-board harness is implemented. The
branch now incorporates upstream's direct cuda-new PI05 migration (`#88`).
Merged-source GPU accuracy, stability and performance require fresh validation;
results from the old backend do not qualify the current implementation.
The merge preserves explicit INT8 language normalization and its conservative
allocation-schedule reservation. BF16/FP8 retain their folded weight recipes.

### Historical numerical evidence

The following real-image results describe the pre-migration implementation at
`e811c4f6`, using the historical limits below. They are not current-head results:

| Board / precision | 1 view | 2 views | 3 views |
| --- | --- | --- | --- |
| Thor BF16 | 6/8 | 8/8 | 8/8 |
| Thor existing FP8 calibration | 0/8 | 4/8 | 4/8 |
| Orin BF16 | 7/8 | 8/8 | 8/8 |
| Orin INT8 | 0/8 | 6/8 | 4/8 |

The eight observations come from two official DROID episodes. The third exterior
camera is explicitly mapped to the third model slot for shape coverage. These
are fixed-observation implementation diagnostics, not normalized DROID-policy
or closed-loop success evaluations. The additional 13 constructed cases per
view cover separate image, noise and token-length factors, including T200.

Controlled Orin BF16 experiments establish that multiple numerical boundaries
matter together: FP32 patch/auxiliary computations, staged BF16 RoPE/GeGLU,
unfolded normalization, and reduction/GEMM conventions. An unfolded-norm plus
staged-GeGLU subset passes all 24 real observations but fails 17 of 39 constructed
cases. It is insufficient as a repair. The complete private candidate passes
all 63 inputs and every action step; 57 outputs are bitwise official. These
private adapters are not shipped or substituted for the official reference.

Replacing that complete candidate's language RoPE with a table-based device
operator preserves all 63 outputs bitwise. The inverse-frequency buffer must
first undergo the reference's BF16 conversion. A subsequent norm/time rewrite
initially fails four cases with a different FP32 reduction schedule; changing
only that schedule restores all 63 outputs bitwise. Independent GPU operator
fixtures and deliberate incorrect-rounding controls support these findings.
Time embeddings can be generated internally using FP64 periods and sequential
FP32 timesteps: all 10,240 values match the pinned reference on Orin. These
experiments identify semantics to carry into current interfaces; they do not
prove that the newly merged runtime already implements them.

INT8 diagnostic adapters still fail full-model comparison after correcting
weight packing, scale division, patch dtype and flow-rounding conventions.
Some operator discrepancies disappear without fixing complete action chunks.
FP8 calibration with additional observations improves the historical real-image
count to 16/24 but still fails real and constructed cases. Neither quantized
path has an approved replacement reference or a relaxed gate.

### Performance and infrastructure evidence

The earlier INT8 capture-allocation repair has 1,000 paired timed calls:
P50/P95 changes are within 0.437%, but the three-view shape return remains
15–17 ms (0.66–0.78%) slower in both orders. This residual is not waived.
The private RoPE replacement has 900 timed calls across 1–3 views and short/long
tokens: steady-state P50/P95 improve, while some first calls increase by
4–17 ms beyond both controls. Norm replacement performance is evaluated
separately. All these measurements retain their original source and extension
identities and do not establish current-head performance.

Shared desktop clients remain on Orin, and Thor also has external GPU services.
An exclusive maintenance window, current-source matrix and measured budgets
are required for formal acceptance. The harness never stops other users' jobs.

The merged source passes 32 PI05 Rust CPU tests and module-boundary checks.
The CI suite has 39 CPU tests. Real-process tests cover normal/repeated
cancellation, an unresponsive worker and a blocked activity auditor. Cancellation
uses a dedicated exception: `InterruptedError` can be swallowed as a retryable
syscall interruption inside subprocess selectors, delaying worker cleanup.

No runners, dispatch variable or branch-protection rule have been enabled.

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
bank policies require an independent `stability_max_abs` budget and an explicit
positive stability count, chosen after
same-input and rebuilt-extension measurements; the example's one repetition is
an unapproved placeholder, not an established protocol. Stability never borrows
the OpenPI or baseline accuracy tolerance; an uncalibrated stability budget
fails pre-build validation and hosted aggregation.

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

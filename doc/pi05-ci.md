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
  --revision FULL_APPROVED_APXINF_SHA --repeats MEASURED_SAMPLE_COUNT \
  --output devlocal/pi05-ci-gate/bank/base-3view/thor-bf16-approved.json
```

Choose observations by explicit coverage: each supported LIBERO task, grasp and
release transitions, occlusion, small predicted actions, longest deployed token
profile, and historical failures. Calibration episodes and acceptance episodes
must be separate. Record the source episode/frame and coverage in the bank's
review notes. The implementation accepts any number of frozen observations;
there is no arbitrary 48-case target or claim of statistical task coverage.

`--diagnostic` retains seven synthetic cases: gradient/short prompt, second
Gaussian noise, longer prompt, black/zero noise, white/negative noise,
float CHW equivalent, and camera-order contrast. `--source-npz` replaces the
gradient images with saved RGB observations but retains canned tokens; it is
image replay only. Neither mode is labelled representative. Float CHW tests the
shared harness conversion, not independent policy preprocessing. Prefer the
frozen model input mode for deployment regression.

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

The first inference for each input is reported separately from repeated calls;
shape switches can include graph creation. Each saved input is repeated with
the same noise, and repeat drift is checked. Repeated-call P50/P95 and raw
samples are recorded; performance budgets apply to each case, so pooling cannot
hide a slow input. A fixed sample count is a measurement protocol, not a
confidence guarantee for P95. Calibrate counts from observed variability and
the desired detection power. Choose cold/switch and warm budgets separately.

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
temperature before and after inference. These checks cannot prevent a developer
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

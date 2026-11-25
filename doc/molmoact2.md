# MolmoAct2: LIBERO continuous-action policy

The `cuda_molmoact2` family runs observation-to-action inference of the
[allenai/MolmoAct2-LIBERO](https://huggingface.co/allenai/MolmoAct2-LIBERO)
checkpoint through `AutoModel`, `LoadedModel::Vla`, `VlaRuntime` and Python
`AutoPolicy`. It is the plain **continuous-action** model: the VLM prefill
produces per-layer K/V, an action expert turns them into a `[10, 32]` normalized
action chunk by ten forward Euler flow steps, and the Python policy maps that
chunk to physical `[10, 7]` LIBERO actions. Official PyTorch model code is a
reference for numerical comparison only and is never imported; `torch` and
`torchvision` are a host-preprocessing dependency of the Python policy (the
camera resize, see below).

## Checkpoint and profile

| Item | Selection |
| --- | --- |
| Checkpoint | `allenai/MolmoAct2-LIBERO`, revision `0d24a92bd1faf321ef497c3bbd5681af97c65aa2`, read in place |
| Official reference code | `allenai/lerobot` (MolmoAct2 policy), revision `28b4f721f931aab554cb176412223f098023705f` |
| Backend / dtype | CUDA, BF16, batch 1; whole-model CUDA Graph by default, prepared eager execution selectable (see *Execution and lifecycle*) |
| Observation | Two cameras, one 378×378 crop each: `observation.images.image`, then `observation.images.wrist_image`; 8-D robot state; task text |
| Sampling | Ten forward Euler steps, `t_i = i/10`, `dt = 0.1`; initial noise `[1, 10, 32]` |
| Model output | Normalized padded actions, FP32 `[10, 32]` (dimensions 7..31 are exactly zero) |
| Public output | Physical FP32 `[10, 7]` actions (`actions`; values are BF16-representable, see *Post-processing*), plus `normalized_actions`, `token_ids`, `noise`, `timing`, `metadata` |
| Excluded | Think / depth reasoning, cross-frame cache, FAST discrete actions, training, quantization, other backends and robots, a graph split into separately captured segments |

The registry keys are `molmoact2`, `molmoact2-cuda` and `cuda_molmoact2`. Central
`model_variant` admission accepts all three spellings; the family loader owns what
the variant means: `None`/`"bf16"` is the BF16 profile, anything else (and any
other precision or `LoadOptions` override) is rejected by the loader before the
device or any weight is touched. Which **checkpoint** is accepted is decided
while `config.json` is parsed, also before any weight is read: a checkpoint with
`model_type: molmoact2` must describe this profile (`add_action_expert`,
`action_mode: both`, `state_format: discrete`, no depth reasoning or depth gate,
`mask_action_dim_padding`, one observation step, expert geometry that matches the
text model with one block per text layer, and a `libero` tag in `norm_stats.json`
with 7-wide action and 8-wide state statistics); a pure VLM, pretrain or depth
checkpoint is rejected both by the native loader and by the Python policy. What
a **request** may carry (image count and size, prompt length, no continuous state,
noise shape and finiteness) is validated per call, at inference time.

## Usage

```python
from apxinf.policies import AutoPolicy

policy = AutoPolicy.from_pretrained("/path/to/MolmoAct2-LIBERO", device="cuda:0")  # needs apxinf[molmoact2]
result = policy.infer({
    "observation.images.image": agentview_rgb,        # uint8 HWC (or CHW), any size
    "observation.images.wrist_image": wrist_rgb,
    "observation.state": state_8,                      # float32 [8]
    "task": "put the white mug on the left plate",
}, noise=fixed_noise)                                  # optional float32 [1, 10, 32]
actions = result["actions"]                            # float32 [10, 7], physical units
policy.close()
```

Without `noise` the runner draws it on the device from a seeded generator
(`seed=` of `from_pretrained`); successive calls differ. Image, state and prompt
keys can be renamed with `image_keys=`, `state_key=`, `prompt_key=`. Images must
be `uint8`; floating-point images are rejected instead of being rescaled by a
guess. A runner can be injected with `model_runner=`. A runnable example is
`python/apxinf/examples/molmoact2policy_infer.py` (needs the CUDA binding, the
checkpoint and `torch`/`torchvision`; options `--execution`, `--tactics`,
`--autotune`).

## Where each step lives

| Step | Owner |
| --- | --- |
| Resize to 378×378 (bilinear, no antialias), prompt text, state normalization and 256-bin discretization, tokenization, image-token expansion, BOS, post-processing | Python policy (`python/apxinf/apxinf/policies/impls/molmoact2.py`) |
| uint8 upload, normalization and patchification on the GPU (`RgbToPatchesReciprocal`), vision encoder, pooling connector, VLM prefill | native family (`crates/apxinf-model/src/cuda_molmoact2/model/blocks/bf16.rs`) |
| Context adapter (36 layers of K/V → per-block cross-attention K/V), action expert, ten Euler steps, publish | native family (`model/blocks/bf16/expert.rs`, `model/mod.rs`) |
| Request validation, noise handling, plan cache, result ownership | `model_runner/{runner,prepare}.rs` |
| Checkpoint config, loader options, host/device weights | `config.rs`, `load.rs`, `weights/{host,bf16}.rs` |

**Prompt and state.** The task text is normalized like the official
`_normalize_question_text`. The state is normalized with the `libero` q01/q99
statistics (first seven entries; the eighth, the gripper, is kept), clipped,
mapped to 256 bins with round-half-even, and written into the prompt as
`<state_N>` tokens. The model has no continuous state input, so the native runner
rejects a continuous `state` argument. These helpers re-implement functions of
the Python files shipped with the checkpoint; see *Provenance and license*.

**Host resize.** The official pipeline resizes `uint8` images with torchvision's
CPU bilinear kernel (`Resize([378, 378], BILINEAR, antialias=False)` on a
channel-first view of the HWC frame, then a clip to `[0, 255]`). `resize_bilinear_u8`
runs exactly those calls, so the default is the official resize and not an
approximation (a NumPy re-implementation was tried and differed from the kernel by
single codes on real frames, and one code changes the actions). It needs `torch`
and `torchvision` (`pip install 'apxinf[molmoact2]'`); when they are missing the
policy raises (also in `from_pretrained`, before any weight is read) and never
falls back to another resize. Only this one image operation uses torch on the
host; no official model code runs. The `image_transform=` hook still accepts any
callable returning a 378×378 `uint8` image and replaces the default; its
exactness is then the caller's.

**Post-processing.** The runner returns the trajectory in BF16 values widened to
FP32. The policy crops seven dimensions, clips to `[-1, 1]`, unnormalizes in
FP32 NumPy like the official `_FeatureNormalizer`, and then rounds the result to
BF16 (round to nearest even) and widens it again to FP32. This is the last step
of the official BF16 model, which returns its physical actions in the trajectory
dtype, so the public values are BF16-representable, and the policy accepts only
the `bf16` runner profile. Without the final rounding the result would differ
from the official BF16 physical actions by up to 1.7e-3.

**Patch normalization.** The official GPU path normalizes `uint8` pixels as
`float(v) * float(1/255) * 2 - 1` in FP32 with one BF16 rounding. The shared gather
semantic `RgbToPatchesReciprocal` (see `crates/apxinf-cuda-new/cuda-operator.md`)
expresses exactly that; the existing true-division recipe differs from it at
byte value 127 after BF16 rounding, and that single value changes the actions.

**Context adapter and rounding.** The expert attends to the visible VLM rows only
(valid tokens, minus every EOS token and the action span); the rows are gathered
before the shared `context_{k,v}_proj`, so masked keys never enter the attention.
BF16 roundings follow the official execution: non-affine RMS norms,
`x * (1 + scale) + shift` as separately rounded operations, residual
`x + gate * out` as rounded product then rounded sum, RoPE products each rounded
to BF16, the Euler update `x + bf16(dt * v)` with the velocity masked on the
padded dimensions, initial noise rounded FP32 → BF16. Timestep conditioning and
the per-step modulation tables depend only on the checkpoint and are built once
at load; the context adapter and everything downstream are rebuilt for every
observation, so no state of a previous observation survives a call.

## Execution and lifecycle

Three paths share one traversal function (`MolmoAct2Model::forward`), so the
model math exists once. They differ in what is prepared and kept:

| Path | What is kept between requests | How it runs | Where it is used |
| --- | --- | --- | --- |
| **Raw eager** (`ModelRunner::run`, internal) | nothing: every request creates and destroys its own native GEMM/attention executions; no `ExecutionSession`, one synchronization per GEMM | host-enqueued, unprepared | the tuning pre-run and the reference of the maintained test; **not** an execution mode of `infer` |
| **Prepared eager** (`execution="eager"` / `ExecutionPolicy::Eager`; also the fallback of `PreferGraph`) | a plan: stable input/output buffers, an `ExecutionSession` workspace arena and the prepared native executions | the same traversal enqueued from the host on every request, no capture, one synchronization at the end | the public eager path |
| **Prepared graph** (default `PreferGraph`, or `RequireGraph`) | the same plan plus one captured whole-model graph (uploaded uint8 images → F32 output) | one replay per request | the public default path |

* The graph is a single graph, not Vision/Language/Action segments. Intermediates
  live in a pre-sized workspace arena owned by the plan; no device allocation takes
  place while it replays. If capture fails, `PreferGraph` falls back to the
  prepared-eager plan and reports the reason.
* A plan is built from a sample request: a sizing probe traversal measures the
  arena, a prepare traversal creates the native executions, then (graph only) the
  identical sequence is captured. Raw eager is **not** a baseline for eager
  inference: it creates every execution per call.
* **Structure** (camera count and layout, prefill segments, number of visible
  context rows) is fixed by a plan; **content** (image bytes, token ids,
  visible-row indices, noise) is rewritten in place before each run.

Consequences for the public interface:

* `infer` keeps one implicit plan for every policy. A request with the same
  structure only rewrites content and runs; another structure (for example a
  prompt of another length) releases the plan before the new one is built, so two
  complete plans never exist at once.
* `prepare_for(sample, policy)` returns a caller-owned plan. Its `run` rejects a
  request of another structure instead of running it against stale tables, and the
  plan remains usable after a rejected request. `RequireGraph` fails when capture
  fails; `PreferGraph` returns a prepared-eager plan whose status carries the
  reason; `Eager` never captures.
* `prepare(spec)` and `prepare_with_policy(spec, …)` are shape-only and **remain
  unsupported** (the structure cannot be derived from a token count); they return
  an explicit error that points to `prepare_for`.
* `clear_prepared` synchronizes and drops the implicit plan.
* Both noise inputs work: host-supplied full FP32 `[1, 10, 32]` noise (staged on
  the device, cast to BF16 outside the graph) and the device generator, which
  advances with every run's RNG key.
* Each run copies the plan's stable output buffer into a fresh allocation, so
  earlier results survive later runs and the plan being dropped.
* GEMM/attention tactics are keyed exactly. A plan fixed at preparation does not
  see recipes added later; tune first, then load with autotune off.

Python: `MolmoAct2Policy.from_pretrained(..., execution=None | "prefer-graph" |
"require-graph" | "eager")` forwards the policy to the native loader;
`policy.execution_mode` and every `infer` result report `unprepared` (a graph
runner before its first call), `graph` or `eager`. Tuning:
`from_pretrained(..., tactics=<recipe dir>, autotune=False | True)`. Without a
recipe directory the backends use their untuned default algorithms; a cold
directory plus `autotune=True` writes recipes (the first request runs once through
the raw path so that recipes are tuned on real data, then the plan is prepared),
and a later process with `autotune=False` only reads them. Tuned recipes select
different algorithms, so **tuned outputs are not bit-equal to untuned ones**.

### Resources and transfers

Measured on Jetson Thor (unified memory), before the test-only cleanup of this
change, on the same model code. Jetson readings mix everything, so the quantities
below are different kinds and must not be added up or substituted for one another:

| Kind | Amount | Meaning |
| --- | --- | --- |
| Checkpoint weights on disk | 21,768,955,912 bytes (20.27 GiB), five safetensors files | the payload read; not a measurement of what is resident |
| Plan workspace arena | 186,646,528–190,840,832 bytes per plan | allocated once at prepare |
| Explicit native GEMM workspace, one traversal | 7,668,895,104 bytes (7.142215 GiB) for the 1,461 GEMM executions of one traversal; 39 construction-time executions add 173,654,016 bytes | exact integer sum of the per-execution `workspace=` reports (`APXINF_GEMM_REPORT=1`); a *reservation* made at creation, not a measured resident size |
| Process RSS | high-water mark about 10.8 GB (`VmHWM`) | host figure; does not contain the device allocations |
| System-wide `MemAvailable` | fell by about 3.5 GiB while a plan was built | whole-machine reading next to the run (other tenants and page cache included) |

**Execution counts.** Building a whole-model plan *creates* 39 (model
construction) + 1,461 (sizing probe) + 1,461 (real prepare) = 2,961 native GEMM
executions in total. That is a **cumulative number of creations, not a number of
executions alive together**: the probe's executions are destroyed before the real
session is built. Execution reuse was counted, not assumed: with
`APXINF_GEMM_REPORT=1` every created GEMM execution prints one line, and in the
measured runs seven same-structure prepared-eager `infer` calls (including changed
images, prompt and noise) created 0 of them while one raw eager call created
3,743. Only GEMM creation is counted; attention executions follow the same session
cache but were not counted separately.

**Per request (provided noise).** Host → device 862,552 bytes (the two uint8
images are 857,304 of them), device → device 1,280 bytes (the action copy),
device → host 1,280 bytes (the action, when the caller reads it). **Inside the
captured graph no device allocation takes place.** That is a statement about the
graph region only: a whole request still analyzes the prompt on the host, builds
byte vectors for the uploads, stages host noise, allocates a fresh device output
per returned `Action`, and the Python policy builds its own arrays. No
allocation-free request is claimed, and prepared eager additionally enqueues every
operator from the host.

**Shared cuBLAS handle (fix in `apxinf-cuda-new`).** The cuBLAS provider created
one `cublasHandle_t` per GEMM execution; on Jetson Thor a handle cost about
65–70 MB of device-visible memory, and a whole-model prepare drove the system
`MemAvailable` down to 3.0–3.3 GiB and took 12.6 s for the probe. The provider now
shares one handle per (device, stream, thread, math mode); the math mode is set once
at creation (changing it per launch fails graph capture with cuBLAS status 14),
stream and workspace are still set per launch, and the handle is destroyed with its
last execution. In the measured runs after the fix `MemAvailable` stayed above
94 GiB during a whole-model prepare and the probe took 1–4 s (observations of those
runs on that machine, not a lower bound). Selection and numerics are unchanged: the
untuned outputs are bit-identical to those obtained before the fix. Attention and
cuBLASLt executions still own one handle each. Other model families that use this
provider were not smoke-tested for this change; multi-device, multi-stream and
mixed-math-mode use have no dedicated test.

## Validation

The numbers below were measured on Jetson Thor (SM110), the original LIBERO BF16
checkpoint above, three real LIBERO observations (the "fixtures"), and the
identical full noise tensor `[1, 10, 32]` fed to the official run and to the
native run (a seed alone does not give the same noise across generators); host
preprocessing used `torch 2.10.0` / `torchvision 0.25.0`. The references are the
official BF16 + autocast run and the official FP32 run of the pinned
`allenai/lerobot` revision. The reference captures, run directories and logs are
private development evidence and are **not** part of the repository; the
repository carries the maintained tests in *Tests and commands*.

The tolerances are **fixed engineering screens**, not derived error bounds. They
are not derived from closed-loop sensitivity (never measured), do not bound inputs
other than the three fixtures, and say nothing about task success.

1. **Exact (an oracle exists, no tolerance).** Prompt token ids against the
   official processor; the noise fed to the native run against the official
   run's noise; padding mask and
   zero padded dimensions; the host-resized images against the official processor's
   378×378 frames; the post-processing applied to the official normalized
   trajectories against the official physical actions, bit for bit.
2. **End-to-end screen.** Every Euler trajectory step and the final normalized
   actions: max-abs at most 0.04 against both references; physical actions: per
   dimension at most 2 % of that dimension's range (`q99 - q01`; 2.0 for the
   gripper). The screen is coarse on purpose: the reference's own BF16-vs-FP32
   difference must pass it too.
3. **Local sanity.** Context adapter layers 0/17/35 fed the *official* K/V, and
   expert block 0 fed the *official* context, at relative L2 at most 2^-8.
4. **Diagnostics only.** Every other tensor, all velocities, relative L2 values and
   the ratio of a native-vs-BF16 distance to the reference's BF16-vs-FP32 distance.

Final normalized actions `[10, 32]` / physical actions `[10, 7]`, native vs the
references, official noise (max-abs / relative L2):

| Fixture | native vs BF16 (normalized) | native vs FP32 (normalized) | BF16 vs FP32 (normalized) | native vs BF16 (physical) | native vs FP32 (physical) | BF16 vs FP32 (physical) |
| --- | --- | --- | --- | --- | --- | --- |
| `f0_readme_libero10` | 7.8e-3 / 3.7e-3 | 4.5e-3 / 4.1e-3 | 4.4e-3 / 3.8e-3 | 3.9e-3 / 2.3e-3 | 3.5e-3 / 2.6e-3 | 2.7e-3 / 2.3e-3 |
| `f1_object_t24_mid` | 3.9e-3 / 1.9e-3 | 4.4e-3 / 2.6e-3 | 4.6e-3 / 2.6e-3 | 3.9e-3 / 1.3e-3 | 3.3e-3 / 1.8e-3 | 4.5e-3 / 2.0e-3 |
| `f2_goal_t12_third` | 3.9e-3 / 2.2e-3 | 5.5e-3 / 3.6e-3 | 5.5e-3 / 3.4e-3 | 3.9e-3 / 2.1e-3 | 3.3e-3 / 2.4e-3 | 3.3e-3 / 2.1e-3 |

The largest native-vs-reference max-abs over the three fixtures is 0.0078125
(normalized, vs BF16) and 0.0055 (vs FP32); the largest per-dimension physical
difference is about 0.26 % of the range vs BF16 and 0.352 % vs FP32. The native
distances are **not** uniformly below the reference's own BF16-vs-FP32 distance
(max-abs up to 1.79× for the first fixture's normalized actions; relative L2 0.65×
to 0.99×). Physical actions are BF16 values, so 3.9e-3 is a single BF16 step for a
magnitude between 0.5 and 1. The source of the residual differences to the BF16
reference was not isolated.

Coverage of the 36 VLM layers: the official K/V were re-captured for layers 0, 17
and 35 only (the same-operand probes above). All 36 layers of the native K/V were
checked for repeat identity (same process and across processes) and, in the patch
normalization work, against an ablation that replaced only the patch normalization;
the other layers' agreement with the official model rests on the end-to-end
comparison.

Execution-path results on the three fixtures (untuned default algorithms): raw
eager, prepared eager and the whole-model graph give bit-identical actions, equal
to the earlier eager-only native run; debug and release builds are bit-identical;
through the public policy, `execution="require-graph"` and `"eager"` give
bit-identical actions on the original frames. Repeating a request is bitwise
identical, changed image / wrist image / task / state / noise take effect and
restoring them restores the result, held results survive later runs and plan
teardown, a different structure replaces the plan, an injected capture failure
makes `RequireGraph` fail and `PreferGraph` fall back with the reason, and rejected
requests leave the plan usable. Cold-cache tuning wrote 51 native / 41 public-path
recipes; a fresh process with tuning off read them, took every GEMM selection from
them, and reproduced the tuned actions bit for bit (tuned against the references:
normalized max-abs 3.9e-3 to 7.8e-3 vs BF16 and 4.3e-3 to 5.3e-3 vs FP32).

These results were measured on the development source before the test-only code
(acceptance entries, probes, progress logging, numerical taps) was removed; that
cleanup changed no model computation or execution path. After it, the maintained
test described below was run on the final source (see the review description for
its result), and the earlier numbers were not re-measured.

### Performance

Release build on Jetson Thor, one action chunk = all ten Euler steps. The two
paths are timed at the same boundary: `Plan::run` binds the inputs in place, executes
(graph replay, or the host-enqueued traversal of the eager plan), copies the output
into a fresh device allocation and synchronizes. Median ms, untuned default
algorithms (graph: 10 warm-up + 100 samples; prepared eager: 5 + 30); p99 within
about 0.6 % of the median in every row:

| Fixture | prepared eager | prepared graph | eager / graph |
| --- | --- | --- | --- |
| f0 | 230.45 | 178.32 | 1.29× |
| f1 | 227.56 | 176.99 | 1.29× |
| f2 | 227.98 | 177.07 | 1.29× |

With tuned recipes: 197.4–198.4 ms (prepared eager) against 151.9–158.1 ms
(graph), 1.25–1.30×. Two graph plans of the same structure differed by about 4 %
(152 vs 158 ms), each stable within itself; the cause was not examined. The gap is
consistent with the host dispatch the graph removes, which was not measured
separately. Public policy including host preprocessing, untuned, median wall ms:
graph 184–191, prepared eager 234–237. Building a plan takes about 1.1–1.2 s
(probe + prepare, + capture for the graph), the first `infer` about 1.3 s, loading
the checkpoint about 17–20 s.

Raw eager takes 14.9–17.5 s per chunk untuned. It creates every native execution
per call and is **not** an eager baseline; it is not divided into the graph figure.
No official-implementation timing, roofline analysis or closed-loop measurement
exists in this work. These are single-machine measurements on a shared Jetson
Thor, not latency guarantees. A tuned run of the first fixture in an earlier
measurement series had an unexplained slow phase (graph chunk median 335 ms) and
is not a steady-state figure; a later series did not reproduce it, which does not
explain it.

**Memory screen.** The development runs applied a screen on system-wide
`MemAvailable` over up to three windows (accepted when the drop was at most 8 MiB;
an unreadable sample counts as a failure). It failed on the third fixture in three
earlier runs that reused recipes in a fresh process; **the cause is unknown**
(nothing observed the machine during them) and later passing runs do not explain
it. Passing means only that those windows did not trigger the screen on a shared
machine; **no leak-free claim is made.** The 50 GB free-memory condition used to
start those runs was a launch guard on a shared machine, not a requirement of the
model.

## Tests and commands

CPU-only unit tests (no GPU): configuration, geometry, weight mapping, request
validation, loader option rejection, registry and admission:

```bash
cargo test -p apxinf-model --features cuda --lib cuda_molmoact2
cargo test -p apxinf-model --lib auto::tests::molmoact2
cargo check -p apxinf-model                      # without the cuda feature
cargo test -p apxinf-cuda-new --lib operator_doc # the operator catalogue test
cargo test -p apxinf-cuda-new --test rgb_patch_normalization --no-run   # compile check only, nothing is executed
cd python/apxinf && python -m pytest tests/test_molmoact2_policy.py \
    tests/test_auto_registry.py tests/test_policy_layering.py
```

The policy tests need `torch` and `torchvision` (the module skips without them).
`--no-run` only builds the shared-operator integration test; running it needs a
CUDA GPU (next section).

CUDA GPU operator test (needs a CUDA device; it replays the RGB-to-patch
normalization kernel, including from a captured graph, against independent CPU
references for every byte value):

```bash
cargo test -p apxinf-cuda-new --test rgb_patch_normalization
```

For this change the test was **compiled only** on the final source; it was not
executed on the GPU in the final round. Its earlier passing result on the
development source is reused on the strength of source continuity (no edit of the
gather sources it covers is recorded after that result; the hashes were not
re-compared mechanically in the final round), not as a new pass.

Checkpoint behaviour test (`#[ignore]`d; needs a CUDA device and the public
checkpoint; it builds its inputs from the checkpoint config, so no exported data is
needed, and holds one prepared plan at a time):

```bash
APXINF_MOLMOACT2_TEST_CHECKPOINT=/path/to/MolmoAct2-LIBERO \
cargo test -p apxinf-model --features cuda --release --lib \
    cuda_molmoact2::model_runner::tests -- --ignored --nocapture
```

It checks that raw eager, prepared eager and the graph give identical actions,
changed inputs take effect and restore, results outlive later requests and plan
teardown, structure changes replace (and release) the plan, rejected requests keep
it usable, capture failure behaves as described above, and device-generated noise is
deterministic per key. It does not compare against the official model.

## Provenance and license

The Python prompt, state and normalization helpers follow functions of the same
names in `modeling_molmoact2.py`, `processing_molmoact2.py` and
`image_processing_molmoact2.py`, shipped with the checkpoint revision above. Those
files carry no per-file license header; the publisher's repository
`allenai/molmoact2` states "This model is licensed under Apache 2.0" and carries the
Apache License 2.0 as `LICENSE`. That repository's `main` is mutable, so this is the
statement as found when this change was prepared, not a statement pinned to the
checkpoint revision. The `allenai/lerobot` reference revision above (Apache-2.0) is
used to produce reference outputs only. See `NOTICE`. Weights are not distributed.

## Limits

* Closed-loop behaviour, task success rates and any error bound beyond the three
  fixtures were not measured.
* Results are for the fixed profile (original LIBERO checkpoint, BF16, batch 1, two
  cameras, one crop, horizon 10, ten flow steps) on one Jetson Thor with these
  driver and library versions. Bitwise equality of paths, builds and repeats is a
  recorded result of these runs, not a guarantee across devices, drivers or library
  versions. The CUDA path is not Thor-specific by construction, but no other device
  was tested.
* A prepared plan is bound to one token structure; another structure rebuilds it
  (about 1 s). Recipes tuned after a plan was prepared are not seen by it. A recipe
  directory is exact-keyed; a missing key uses the untuned default.
* Prepared eager is slower than the graph (1.25–1.30× measured) because it enqueues
  every operator from the host; `PreferGraph` fallback carries the reason.
* Unexplained observations: three failed memory screens, a slow first-fixture tuned
  run, and a 4 % difference between two graph plans of one structure.
* Attention and cuBLASLt executions still own one handle each; the shared cuBLAS
  handle change has no dedicated multi-device / mixed-math-mode test and other
  families were not smoke-tested.
* The host resize equals the installed torch kernel; another torch build or platform
  is only as exact as its own kernel.
* Physical actions follow the official BF16 output boundary; a different output
  precision would be a different profile.
* Batch 1, two views, one crop, horizon 10, ten flow steps, LIBERO 7-D actions and
  8-D state only.

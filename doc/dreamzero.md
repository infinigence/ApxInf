# DreamZero

DreamZero uses the existing `AutoPolicy → DreamZeroPolicy → ModelRunner →
AutoModel → LoadedModel::Vla → DreamZeroModelRunner` path. Its native network
jointly predicts future video and actions; video tokens participate in every
self-attention and denoising step. It does not use an action-only WAM runtime.

## Checkpoint and precision scope

The admitted profile is the released
[DreamZero-DROID](https://huggingface.co/GEAR-Dreams/DreamZero-DROID/tree/96ad344138c66e82536422432ad742f015784942)
checkpoint at `96ad344138c66e82536422432ad742f015784942`, with official source
[dreamzero](https://github.com/dreamzero0/dreamzero/tree/ab790c198fbce33503358efbbd4187ce9a89adf3)
at `ab790c198fbce33503358efbbd4187ce9a89adf3`. The loader requires the complete
BF16 checkpoint, the DROID architecture, and all three camera views. The
checkpoint's CC-BY-NC-4.0 license remains applicable.

The baseline uses 16 complete DiT evaluations per conditional/unconditional
branch with flow UniPC, video guidance 5, and conditional actions. The upstream
configuration's training/default step field does not by itself define this
inference profile. The native arithmetic follows the official autocast scopes:
initial CLIP resize/normalization and VAE row normalization use FP32, casting
back at the learned BF16 boundaries. History VAE encodes outside that scope
retain their separate BF16 rounding behavior. Approximate DiT caching, quantization, and action smoothing
are unavailable and cannot be enabled implicitly.

Flash is a separately trained checkpoint with distinct video/action noise-time
arrangements. The network represents separate modality time embeddings and
schedules, but the loader admits only the released standard profile. A reduced
step count is not Flash support. AgiBot and Flash checkpoints are not accepted;
adding them requires their own weight, normalization, schedule and device
validation. No released official Flash checkpoint was verified for this port.

## Ownership and interfaces

| Owner | Responsibility |
| --- | --- |
| `cuda_dreamzero/config.rs`, `load.rs` | Strict architecture/profile admission, execution policy defaults, recipe options and construction |
| `cuda_dreamzero/weights/host.rs` | CPU checkpoint read, name/shape/dtype validation and unused decoder validation; loading only |
| `cuda_dreamzero/weights/bf16.rs` | One BF16 device representation, one-time transposes, convolution packing and fixed position tables |
| `cuda_dreamzero/model/mod.rs` | Model-domain profile, vision/language/action topology, standard denoising loop and opaque workspace interface |
| `cuda_dreamzero/model/blocks/bf16.rs` | Single `Bf16Blocks` implementation: UMT5, CLIP, temporal VAE, joint DiT, UniPC tensor operations and physical intermediate storage |
| `cuda_dreamzero/model_runner/runner.rs` | `VlaRuntime`, request validation, RNG binding, transactional episode history and bounded implicit plan cache |
| `cuda_dreamzero/model_runner/prepare.rs` | Stable buffers, real warmup, three graph captures and retained prepared resources |
| `policies/impls/dreamzero.py` | DROID camera/task/state mapping, UMT5 tokenizer, q99 normalization, relative joint-action decoding |

The family imports model-neutral safe operations through its `backend.rs`.
`check_dreamzero_module_boundaries.py` checks the model/runner/weights direction.
The model owns stage and sampling order; Blocks owns numerical/operator composition.
The runner owns graph preparation. The single BF16 file uses private modality
namespaces to distinguish tensor/weight types, rather than separate implementations.
Host weight code is never part of forward execution. Neither the
backend nor another family owns DreamZero-specific execution.

The policy accepts uint8 camera arrays `[T,H,W,3]`, with `T` equal to 1, 4, or 9.
Default camera keys are `video.exterior_image_1_left`,
`video.exterior_image_2_left`, and `video.wrist_image_left`; state keys are
`state.joint_position` (7) and `state.gripper_position` (1), with a `task` string.
Raw frames undergo the official CPU center crop and antialiased resize.
`images_preprocessed=True` requires 176×320 frames. The policy forms a 352×640
montage and rounds its normalization to BF16. Learned computation runs natively.
The native boundary receives flattened montage rows, 64 normalized state
values, and two padded 512-token prompts with masks.

Native actions have shape `[24,32]`. The policy decodes all 24 steps and the
first eight robot coordinates, adding current joint positions to the seven
relative joint coordinates. Supply the official UMT5 tokenizer through
`tokenizer_path`; assets are reused in place and are not downloaded by inference.
The joint network computes future video latents internally. The action API does
not decode or return RGB video; the loader validates the checkpoint's unused
VAE decoder tensors before releasing them.

```python
from apxinf import AutoPolicy

policy = AutoPolicy.from_pretrained(
    "/path/to/DreamZero-DROID",
    tokenizer_path="/path/to/official-umt5-tokenizer",
    seed=17,
)
result = policy.infer(observation, episode="trial-1")
actions = result["actions"]  # [24, 8], joint positions and gripper
policy.reset_history(episode="trial-2")
```

`VlaRuntime::initial_latent_shape()` declares stochastic input separately from
output actions. DreamZero's shape is `[113408]`: contiguous `[16,2,44,80]` video
noise followed by `[24,32]` action noise. The Python `noise_shape` getter and
validation use this declaration. `noise=None` generates device noise from the
binding's RNG; explicitly supplied joint noise does not advance that RNG.
Existing families retain their action-shaped default.
A seed identifies ApxInf's native RNG stream; comparisons with the reference
freeze the complete joint noise explicitly.
The native Python binding also accepts a leading batch dimension of one and
canonicalizes it to the declared shape. The policy's `noise` argument retains
the explicit one-dimensional joint layout.

## History and preparation

A history identity is a nonempty `(session, episode)` pair. Each Python policy
has a unique session; callers can pass `episode=` to `infer` or call
`reset_history(episode=...)`. A changed identity, changed prompt/mask, single
frame observation, or exhausted temporal window starts a fresh history. Calls
without identity are standalone. Only observed video K/V is committed, after
successful inference; generated video/action K/V is never published as history.
One policy instance represents one logical stream. The generic shared WebSocket
server does not assign connection identities, so it is not an admitted
multi-client DreamZero serving path; use separate policy instances or explicit
episode identities in an application adapter.
Positive and negative branches retain separate K/V. Warmup and graph capture
cannot mutate previously committed history.

`reset_history()` removes semantic history; `clear_prepared()` evicts only the
implicit executable plan. Neither operation resets the RNG. `ModelRunner.reset_sampling(seed=...)` changes RNG state separately. Explicit plans retain their model and
resources, but reject a request whose current history requires another profile.
An action tensor is a view of a plan's reusable output; copy it before reusing
that same plan if the old values must remain stable.

Rust `prepare_for(real_request, policy)` supports `Eager`, `PreferGraph`, and
`RequireGraph`. Spec-only preparation is rejected because the real montage,
prompt and history are required. Implicit inference requires all three graphs:
Vision (CLIP/temporal VAE), Language (both prompts, context projections and
per-layer text/image cross-attention K/V), and
Action (observed K/V computation plus joint denoising). A compatible prepared
run rebinds stable inputs and executes without capture or autotuning. The
implicit cache holds one profile; switching profiles replaces it.
On an implicit profile miss, real preparation already evaluates the exact
request's complete schedule. After all three graphs capture successfully, that
same call publishes the computed result and observed history once, without a
second evaluation. This reuse is confined to the uninterrupted preparation call;
no request-value cache or validity token is introduced. Subsequent compatible
calls rebind and replay normally. Explicit `prepare_for` still publishes no
history, and every explicit prepared `run` rebinds and executes its request.
The `checkpoint_joint_implicit_history_preparation_result` opt-in test checks
the complete action/video history sequence and generated-noise/cache replay.
The explicit and implicit complete-checkpoint suites each pass all 14 scenarios,
with all 56 action/video metric rows zero. Earlier performance comparisons below
describe their recorded frozen builds; preparation-result reuse is measured
separately.
Self-attention K/V projections write directly into the current-token rows of
the contiguous attention buffers. Only that K view receives staged RMS/complex
RoPE; committed history is copied into the separate prefix before projection
and is never normalized or overwritten. Observed K/V publication still copies
only observed rows into independent history storage. This removes the temporary
current K/V matrices and their two copies per block evaluation.

DiT modulation and gated residual updates use staged pointwise fusions. Each
operation retains the original dtype casts, including `1 + scale` and the
product before shift/residual addition. Video and action/state keep independent
modulation vectors. The final FFN residual writes hidden directly; no full
hidden copy is needed. These operations remove the two block workspaces' gated
matrix and scale-vector scratch, without changing LayerNorm reduction order.

Language recomputes both CFG branches' cross-attention K/V on every request.
Each layer retains its own four BF16 tensors; observed and denoising evaluations
only read them. Raw key-projection scratch is shared across layers within each
branch. Biased projections and staged RMS normalization include all padded text
rows. These tensors belong to the prepared workspace and are independent of
committed video history. Relative to recomputing K/V in each block evaluation,
this adds 1,228,431,360 bytes of tensor storage for the fixed 40-layer profile;
provider execution resources are additional.

Streaming observations advance the history length and therefore require a new
profile until window rollover. Preparation recomputes the three stages, and a
successful call copies committed K/V into independent device storage. A warm
single-observation replay measurement does not represent streaming latency.

`autotune` and `tactics` use cuda-new's recipe directory contract. Tuning options
are immutable for a loaded runner. Construct a new runner to select another
store/configuration; legacy tuning JSON is not interpreted as native recipes.

DreamZero keeps `ColdBaseline` GEMM selection for encoder, context-embedding,
timestep and output-head projections. Bulk DiT block projections, including
per-layer cross-attention K/V produced by Language, use `Any` within the same
BF16 staged-bias, deterministic and 1 MiB workspace constraints. Online tuning
runs during preparation; replay retains the selected executions. Recipe/session
keys distinguish these domains. This contract is scoped to the qualified device,
library and build; it is not a cross-hardware arithmetic guarantee. Attention
continues to use its existing candidate domain.

Unrestricted tuning across all GEMMs failed full-trajectory action/video gates
despite zero-error isolated component captures. The bulk-block-only relaxation
was subsequently qualified separately, as recorded in the performance section.

## Qualified device profile

The full qualification and performance figures in this section were collected
for `4af38ed3`, before the architecture revision and main integration below.
They establish the historical admitted profile; current-source revalidation is
reported separately under Architecture revision.

Validation uses NVIDIA Thor (SM110), driver 595.78, CUDA runtime 13.2.86,
cuBLAS 13.4 and cuDNN 9.20.0, in the existing MAXN power mode. The native CUDA
code is compiled with NVCC 13.0.88, matching the reference PyTorch build; the
Rust model uses dev-profile `opt-level=2`. CUDA 13.2 compilation changes BF16
activation results and has not passed this profile. These results do not qualify
other devices, compiler/library versions, checkpoints or sampling profiles.

All ten checkpoint payloads were verified against the pinned official LFS
SHA-256 identities. The reference uses PyTorch 2.11.0+cu130 and torchvision 0.26,
rather than upstream 2.8/0.23, with transformers 4.51.3, peft 0.5 and diffusers
0.30.2. It retains the official forward and sampling bodies, disables approximate
caching and unwraps two UniPC compile decorators for an eager BF16 baseline. An
allocation adapter skips redundant base-Wan loading and assigns every tensor
from the complete checkpoint. These differences are part of the qualified scope.

The following table retains the pre-performance integration evidence. The
2026-10-30 performance revision has separate, current acceptance below.

| Validation | Pre-performance integration result |
| --- | --- |
| Full cold, tuned and fresh-process restored model | Each passes 14 scenarios and 28 action/video metrics, all finite with relative L2, RMSE and max-absolute error 0 |
| Final-binary components | 9 tests and 122 unique encoder, VAE, DiT and sampler checkpoints, all numerical errors 0 |
| Graphs and lifecycle | Three graphs; Eager/PreferGraph parity; changed inputs; retained outputs/owners; stale-profile rejection; RNG; history commits, rollover, task/episode/session isolation and failed-call rollback pass |
| Public AutoPolicy | Raw cameras/task/state, all 24×32 normalized actions and 24×8 decoded robot actions agree with independent official captures; batched joint noise, input rebinding and RNG checks pass |
| Actual recipe restoration | Full model: 71 GEMM and 18 Attention disk hits; public profile: 59 GEMM and 9 Attention disk hits. No fallback/tuning in fresh processes; content hashes and mtimes unchanged |

The maintained gates are action relative L2/RMSE 0.02 and max-absolute 0.10,
future-video relative L2 0.02, full CLIP/UMT5 encoder relative L2 0.01, and isolated
learned-block relative L2 0.005. Eager/captured parity is exact. The full runs used
a frozen binary whose test-only video predicate was temporarily 0.04; every raw
metric was independently re-evaluated against the original 0.02 gate and the
stronger zero-error predicate before restoring 0.02 in the maintained test. For that pre-performance qualification, only
that test predicate/comment and its documentation changed afterward;
the executed source/binary identities and earlier failures remain preserved.

The 14 scenarios cover initial/repeat, three successive history observations,
window rollover, history reset, changed task, changed episode, nine-frame input,
independently changed video/action noise, raw-public-input equivalence and a
separate session. Fixed joint noise establishes reference/native agreement;
separate perturbations demonstrate cross-modal influence. Inputs are synthetic
camera/state/task fixtures, not evidence of closed-loop robot task success.

## Performance revision acceptance (2026-10-30)

The optimized native build preserves standard BF16 joint future-video/action
inference with 16 evaluations per CFG branch. Its full real-checkpoint test
passes all 14 scenarios and 28 action/video metrics with relative L2, RMSE and
max-absolute error zero, using the maintained gates directly. History rollover,
request rebinding, graph ownership, RNG and failed-call rollback checks pass.
The CUDA backend suite passes 167 tests (2 ignored; the unchanged long
convolution-candidate test is filtered); all 9 family contract tests pass.
These are results for the frozen performance build, rather than a new run of
the historical 122 component checkpoints above.

The changes use exact indexed/vector transport for unfold and pointwise, plus
certified FP32 intervals with the original FP64 fallback for complex rotation.
Staged RMS and complex RoPE are fused for self-attention Q/K, retaining both
normalization casts and the FP64 fallback. Language computes all layers' text/image
K/V once per request; observed and denoising blocks reuse those exact tensors.
Staged modulation and gated residual fuse the existing elementwise operations,
preserving each BF16 cast. Current self-attention K/V projections write directly
to their final history-concatenated buffer views; only current K is normalized
and rotated. FFN residuals write directly to hidden storage.
A same-profile public/canonical input rebind matches an independent capture with
zero action error, then restores the original public output exactly. CUDA capture
counts remain 3 before rebind, after changed input and after restoration.
Bulk DiT block projections may tune within deterministic `StagedBias` and the
1 MiB workspace limit. Encoders, context/timestep projections and output heads
retain `ColdBaseline`. Algorithm selection is qualified for the recorded Thor,
CUDA and library build; deterministic execution alone does not promise identical
arithmetic across every future library tactic.

The preceding performance build (`b2179f9`) also has an accepted fresh-process
CUDA/NVTX trace: 47 disk-recipe and 1,431 memory-recipe preparation summaries,
exact public outputs, and the strict process, event-loss, transfer and three-graph
audit described below. Those trace counts describe that build, not the subsequent
RMS/RoPE and request-context K/V revision.

The subsequent RMS/RoPE + request-K/V build also has an accepted fresh-process
full-request trace. Its synchronized diagnostic window is 22.639 s with 73,402
kernels and three graph launches; public outputs remain exact and the strict
copy/helper/event-loss audit passes. Relative to the RMS/RoPE-only intermediate
trace (81,082 kernels), K/V reuse removes exactly 7,680 kernel launches. These
instrumented timings are separate from the matched uninstrumented results below.

The staged-affine/direct-KV revision has an accepted full-request trace with a
21.611 s diagnostic window, 60,122 kernels and three graph launches. Relative to
the preceding request-KV trace, it removes exactly 13,280 arithmetic kernels and
4,080 D2D copies. D2D payload falls from 109,428,796,032 to 37,077,052,032 bytes
(10,063 remaining copies). Kernel and copy counts match the source-derived
schedule independently; payload is not measured physical DRAM traffic. Public
outputs and the strict copy/helper/event-loss audit pass.

## Measured performance and residency

Uninstrumented public-boundary A/B results on Thor in MAXN, in seconds. Four
fresh processes run in A→B→B→A order with three warm samples each. Both paths use
the same raw-input script, fixed joint noise and numerical checks; online tuning
is disabled and the optimized path restores its previously qualified recipes.

| Path | Median load | Median first prepare/infer | Warm median | Warm p95 | Warm samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native baseline, before performance changes | 37.448 | 107.525 | 47.541 | 47.577 | 6 |
| Native optimized, restored recipes | 34.869 | 65.164 | 24.975 | 25.179 | 6 |

The matched warm speedup is 1.904×, a 47.5% latency reduction. Every
process returns exact normalized and robot actions, including exact replay and
leading-batch noise checks. GPU frequency samples inside the measured calls are
1,575 MHz for both paths. Default memory DVFS is retained: baseline EMC samples
range from 3.2 to 4.266 GHz; optimized samples stay at 4.266 GHz. These are stock
MAXN end-to-end results, rather than an experiment that locks both memory clocks.

A second matched A→B→B→A experiment compares the published performance build
(`b2179f9`) with the RMS/RoPE fusion and per-request K/V revision. Both use frozen
extensions and qualified recipes, the same public-input script and no online
tuning; each variant has two fresh processes and six warm samples.

| Path | Warm median | Warm p95 | Warm samples |
| --- | ---: | ---: | ---: |
| Published performance build (`b2179f9`) | 25.085 | 25.181 | 6 |
| RMS/RoPE fusion + per-request cross-attention K/V | 22.650 | 22.714 | 6 |

This matched improvement is 1.108×, or 9.7% less latency. All normalized and robot
output errors are zero, with exact repeated replay and leading-batch noise
checks. During measured windows, all GPU frequency samples are 1,575 MHz and all
EMC samples are 4.266 GHz for both variants; clocks were not reconfigured. The
old 47.541→24.975 s comparison and this 25.085→22.650 s comparison are separate
experiments and should not be combined into a new matched-baseline speedup.

A third matched A→B→B→A experiment compares the published request-KV revision
(`2f6be45`) with staged-affine fusion and direct current-KV placement. The same
public script, frozen extensions/recipes and fixed joint workload are used, with
no online tuning and six warm samples per variant across two fresh processes.

| Path | Warm median | Warm p95 | Warm samples |
| --- | ---: | ---: | ---: |
| Published request-KV revision (`2f6be45`) | 22.604 | 22.857 | 6 |
| Staged affine + direct current-KV placement | 21.670 | 21.765 | 6 |

This matched improvement is 1.043×, or 4.1% less latency. All public action
metrics are zero, with exact replay and leading-batch noise checks. All measured
GPU/EMC samples remain 1,575 MHz / 4.266 GHz for both variants. This is a separate
experiment from the earlier comparisons. These warm T=1 measurements do not
establish streaming history performance.

A fourth public A→B→B→A comparison isolates implicit preparation-result reuse
against the staged-affine/direct-KV build (`0cd880b`). Each variant has two fresh
processes, two first-request/first-cached observations and six warm samples.
The same raw-input script, fixed joint noise and frozen recipes are used with
online tuning disabled. Values below are median seconds.

| Path | First request | First cached request | First two requests | Warm replay |
| --- | ---: | ---: | ---: | ---: |
| Staged affine + direct KV | 53.176 | 23.489 | 76.665 | 21.701 |
| Implicit preparation-result reuse | 31.040 | 22.696 | 53.736 | 21.686 |

The first cached request includes the first graph replay; the first-two total
accounts for work deferred to that call. Observed median latency is 41.6% lower
for the first request and 29.9% lower for the first two requests together. Warm
replay is effectively unchanged in this experiment. All 12 normalized/deployed/
robot action metric rows are zero, including leading-batch noise and exact replay
checks. The full BF16 joint schedule remains 16 evaluations per CFG branch.

Cold observations are limited: baseline first/cached ranges are 50.273–56.078 /
21.635–25.343 s; reuse ranges are 30.164–31.916 / 22.666–22.725 s. Default DVFS
remains enabled; first/cached windows include CPU work and sample GPU 315–1,575
MHz and EMC 2.75–4.266 GHz. All warm-window samples are 1,575 MHz GPU and 4.266
GHz EMC. These observations establish neither a serving SLA nor a hardware
minimum. The first canonical streaming comparison below predates preparation-result
reuse and measures the separately identified revisions.

A fifth public A→B→B→A experiment isolates initial VAE suffix reuse
(`f042a56`) against the published preparation-result build (`f9c6ba4`). Both
use the same raw observation, fixed joint noise, frozen extensions/recipes and
validation script, with online tuning disabled. Each variant has two fresh
processes, two first/first-cached observations and six warm samples. Values below
are median seconds; warm p95 is an empirical small-sample percentile.

| Path | First request | First cached | First two requests | Warm median | Warm p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published preparation-result build | 33.330 | 22.936 | 56.265 | 21.588 | 21.738 |
| Initial VAE suffix reuse | 29.408 | 21.815 | 51.223 | 20.504 | 20.562 |

Observed warm latency falls by 5.0% (1.053×). Every candidate warm sample
(20.346–20.565 s) is below every baseline sample
(21.518–21.741 s). All 246 warm-window GPU/EMC observations are
1,575 MHz / 4.266 GHz; no observer timeout or foreign compute PID is observed.
Recipes are unchanged after the experiment, and artifact offloads finish before
the first A/B process. No profiler or inference hook is loaded.

First/cached observations remain limited and use default DVFS: baseline ranges
are 31.537–35.122 / 22.395–23.477 s, and candidate ranges are
28.706–30.111 / 21.664–21.967 s. These windows include CPU work and sample GPU
315–1,575 MHz / EMC 2.75–4.266 GHz. First-two totals include the first graph
replay. This standalone T=1 comparison does not establish streaming speed or a
hardware minimum, and is separate from earlier matched experiments.

The unchanged BF16 joint schedule still runs 16 evaluations per CFG branch.
Explicit and implicit complete-checkpoint suites each pass all 14 scenarios,
with all 56 action/video metric rows zero. Five full-output VAE comparisons
(each 506,880 raw BF16 elements) are bitwise exact across changed RGB input and
retained graph replay; six independent 1/9/33-frame reference comparisons are
zero. Full public rebind, robot decoding, leading-batch noise, RNG and resource
lifetime checks pass, and all 12 A/B action metric rows are zero. The native
backend revision is unchanged; this comparison adds only model-owned VAE work
reuse while retaining all nine initial latent frames.

A sixth public A→B→B→A experiment extends that reuse through the first
spatial-only downsample and following 192-channel residual blocks (`45697c8`),
against the published initial-prefix build (`75afbb1`, code `f042a56`). The
validation script, raw input, joint noise and all 90 recipe files are identical;
recipe files remain unchanged afterward. Two fresh processes per variant each
record three warm samples, with online tuning disabled.

| Path | First request | First cached | First two requests | Warm median | Warm p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published initial-prefix build | 30.204 | 21.425 | 51.628 | 20.322 | 20.366 |
| Extended initial VAE prefix | 30.205 | 21.588 | 51.793 | 19.987 | 20.039 |

Values are median seconds. Paired warm latency falls by
1.65% (1.017×). Every candidate sample
(19.957–20.043 s) is below every baseline sample
(20.267–20.377 s). All 237 warm-window GPU/EMC
observations are 1,575 MHz / 4.266 GHz, with no warm observer timeout or sampled
foreign compute PID. No sample is discarded; all artifact offloads finish
before the first A/B process. First/cached observations use default DVFS and
show no established improvement; their two observations per variant and the
empirical warm p95 do not establish an SLA, streaming speed or a hardware minimum.

This extended build independently passes all 14 explicit and 14 implicit
checkpoint scenarios: all 56 action/video metric rows are zero, with the original
16-evaluation-per-CFG-branch schedule. All five complete-output VAE bitwise
checks and six independent 1/9/33-frame reference metrics pass. Full public
input rebinding, robot decoding, noise/RNG and cleanup checks pass; all 12 A/B
public metric rows are zero. Normalization and the native backend are unchanged.

A separate canonical-native streaming A→B→B→A comparison exercises the same six
history scenarios in order, with one warmup sequence followed by two measured
sequences per fresh process. Each variant has two processes and four measured
calls per scenario. Calls include implicit profile preparation, binding, replay,
action download and history publication; they exclude policy camera resizing
and tokenization. Joint future video is computed, but this experiment gates only
the returned 24×32 normalized actions against the independent fixtures; video
qualification remains the separate full-checkpoint acceptance above.

| Canonical scenario | Published request-KV (`2f6be45`) | Staged affine + direct KV (`0cd880b`) |
| --- | ---: | ---: |
| initial | 53.959 | 48.389 |
| repeat | 22.578 | 21.711 |
| history | 53.549 | 48.128 |
| history2 | 57.913 | 52.425 |
| history3 | 62.322 | 57.861 |
| window | 54.212 | 49.065 |

Values are median seconds. All 72 calls, including warmups, have zero action
relative L2, RMSE and max-absolute error. The measured scenario reductions range
from 3.8% to 10.3%; the controlled history3 comparison does not reproduce the
apparent regression suggested by isolated first-replay acceptance timers.
Default DVFS remains enabled: measured-call frequency samples range from
315 to 1,575 MHz GPU and 2.75 to 4.266 GHz EMC for both variants, including CPU
preparation intervals. No fixed-clock or physical-traffic claim follows.

The first baseline process completed and saved all 18 calls, then exited with a
cleanup-only error from an unsupported `ModelRunner.close()` call. Its original
script, report and exit status are retained. Cleanup was corrected for the
remaining processes; the program preceding cleanup is byte-identical. Those
already completed timed calls remain in this comparison, with the exception
recorded explicitly rather than normalized to a successful process exit.

A second canonical-native streaming A→B→B→A comparison measures preparation-result
reuse against `0cd880b`, using the same six-case sequence and one warmup plus two
measured sequences per fresh process. Each variant again has four measured calls
per scenario; this comparison includes profile preparation and history publication.

| Canonical scenario | Staged affine + direct KV | Preparation-result reuse |
| --- | ---: | ---: |
| initial | 48.247 | 26.597 |
| repeat | 21.733 | 21.735 |
| history | 48.146 | 25.850 |
| history2 | 52.818 | 27.988 |
| history3 | 57.362 | 30.471 |
| window | 49.026 | 26.888 |

Values are median seconds. All 72 calls, including warmups, have zero normalized
action relative L2, RMSE and max-absolute error. Profile-changing calls have
44.9–47.0% lower medians; repeat is effectively unchanged (21.733 versus 21.735 s).
The explicit and implicit full-checkpoint suites separately qualify future video
and lifecycle behavior with all 56 action/video metric rows zero. This streaming
benchmark returns actions; it does not independently revalidate future video.

Default DVFS is retained: measured windows sample GPU 315–1,575 MHz and EMC
2.75–4.266 GHz for both variants, including CPU preparation. No foreign compute
PID or observer timeout occurs in the measured-window samples. The first baseline
warmup history3 call takes 92.614 s and contains two observer timeouts; the
predeclared warmup exclusion applies to every process. All four processes finish
successfully. These small-sample observations establish neither a serving SLA nor
a physical hardware minimum.

Earlier pre-performance runs measured 47.617 s cold and 47.756 s restored public
medians (5 samples each). The historical official preprocessed BF16-head median
is 39.875 s (5 samples); it has a different API boundary and experiment context.

Load and first inference are separate intervals. Processes are fresh, but OS file
caches were retained. Native timings include raw camera preprocessing/tokenizing,
input upload and returned robot actions. Reference timings begin with preprocessed
BF16 head inputs and include upload through action download; the joint video path
is computed in both. The native/reference figures have different API boundaries; the speedup above
compares only the two matched native public paths. The small-sample p95 is
empirical interpolation, not a serving SLA. Profiler runs are excluded. No other
compute PID was observed in the approximately one-second A/B samples; this does not exclude shorter activity or the
existing graphics desktop.

The pre-performance complete-process allocation trace records a Device
allocation high-water
of 79,935,015,722 bytes (74.45 GiB), with Device Static and Pinned categories
reported separately. This is not total physical memory or driver overhead.
That allocation high-water has not been remeasured for the optimized build.
Reference PyTorch peaks are 53,079,793,664 allocated bytes (49.43 GiB) and
56,358,862,848 reserved bytes (52.49 GiB); allocator statistics are not equivalent
to native traced allocation usage. System `MemAvailable` is tracked separately
for run admission/protection and is not a process memory measurement.

The optimized complete-process CUDA/NVTX trace contains one synchronized warm
public request lasting 25.182 s. Summed kernel durations are 12.974 s GEMM,
3.666 s attention, 2.920 s pointwise, 1.975 s normalization, 1.394 s unfold and
0.922 s complex rotation. It contains
three graph launches and 83,802 GPU kernels, nine declared H2D uploads totaling
1,586,816 bytes before computation, and one 1,536-byte action download afterward.
All recorded kernels, copies and memsets associated with that request are inside
its synchronized range; there is no intermediate host tensor transfer. The
request also records 14,143 D2D copies (109,428,796,032 bytes), 186 memsets and
162 `cudaMalloc`/`cudaFree` pairs outside the graphs. These are performance debt.

An earlier capture-range trace warned of incomplete CUDA collection and was
rejected. The accepted full-process trace has no event-loss diagnostic. Its
no-CUDA/no-NVTX warnings are retained and attributed by same-run child PID/command
records to `nvidia-smi` and the `lscpu` inventory command; unknown-process warnings
are not exempted. The driver reports Unified Memory tracing unavailable. No
managed allocation API or managed allocation event was recorded, but physical
page migration, page faults and mapped-host accesses were not measured. Explicit
CUDA copies and allocation events do not establish total physical memory traffic.

Performance status remains **best effort with remaining debt**. The paired public
comparisons above identify each source and measurement scope. A physical
DRAM-counter roofline and a complete-model hardware minimum have not been
established; the earlier optimistic DiT-only arithmetic floor is not an achieved
whole-model latency. The historical native/reference allocation figures use
different accounting and do not establish current optimized residency. Encoder
and output-head reduction orders remain constrained after earlier complete-model
failures; only the qualified bulk DiT domain is open to tuning.
Streaming history changes require new preparation, and transactional observed-K/V
snapshots allocate/copy outside the captured graphs. Warm standalone T=1 replay
is not streaming latency or allocation-free replay.

## Build and regression requirements

Build with cuDNN 9 development headers and provide `libcudnn.so.9` at runtime;
installing headers after a build requires rebuilding cuda-new. The native provider
is loaded only when queried, so other families do not gain a mandatory runtime
dependency. DreamZero uses deterministic FP32-reduction convolution plans within
a caller-owned 256 MiB workspace for the RGB stem and four temporal convolutions.
Biased projections use a 1 MiB cuBLASLt bound and FP32 accumulation, including
FP32 split-K partial reduction before one BF16 output conversion. The family-default
`ColdBaseline` policy anchors encoders and output heads. Bulk DiT block
projections explicitly select `Any` while preserving `StagedBias`, determinism
and the 1 MiB bound; a workspace limit alone does not specify a reduction order.

Initial vision binds one real RGB montage followed by 32 zero frames. The
model reuses the identical temporal suffix in the first two VAE resolutions and
the intervening spatial-only downsampling. It computes the transient and one
complete steady frame, then fills repeated output frames with nonoverlapping device copies. Retained
convolution tiles keep their original GEMM row count, including the final
partial tile. Reuse ends before temporal downsampling. Normalization and later
VAE stages keep their full execution; all nine initial latent frames remain available to conditioning. History and
general 1/9/33-frame encoding use the ordinary full path. No activation is
reused across requests.

The opt-in `checkpoint_vae_initial_steady_prefix_bitwise_and_graph` test compares
raw BF16 bits against full VAE execution for two RGB inputs, changed-input graph
replay, and replay after releasing the original workspace owners. The existing
`checkpoint_vae_temporal_profiles_eager_capture_and_lifetime` test retains the
independent 1/9/33-frame reference coverage. Complete joint explicit/implicit
history suites and public request timing are still required for a new build;
component parity alone does not qualify it.

VAE spatial attention uses native memory-efficient CUTLASS at head width 384.
Pinned upstream dependency headers, license and provenance are retained under
`apxinf-cuda-new/native/kernels/efficient_attention`; inference links no
PyTorch/libtorch runtime. Full initial-frame and 33-frame VAE captures, including
changed input, pass in the qualified complete-model build.

Regression results: cuda-new 121 passed/2 ignored (including 480 convolution
candidate checks across 12 profiles); CPU model library 153 passed/1 ignored;
Python with the real CUDA extension 306 passed/4 skipped/30 subtests, plus one
Pi0Fast option-rejection-order failure reproduced on the unchanged base. Linux
MLX feature isolation, opaque storage and both tokenizer CI commands pass.
Model-family boundaries and diff checks pass. Opt-in real-checkpoint tests require
external weights and fixed official captures; compilation or ordinary unit tests
alone do not establish this device qualification.

## Architecture revision

The `cuda_dreamzero` layout follows the Fast-WAM/Cosmos review contract and
uses the current cuda-new interfaces after merging main. Precision choices are
explicit L3 parameters: `GemmNumerics::StagedBias` on existing GEMM+bias,
`NormNumerics::Staged` for RMS/L2 and `NormNumerics::Welford` for LayerNorm.
The separate additive-attention contract supplies UMT5's learned bias and
materialized logits/probabilities; ordinary attention has no additive-bias input.
The existing default numerical paths remain the choices for other callers.
The VAE tiled unfold/GEMM path explicitly requests `GemmNumerics::F32Reduction`
for its FP32 partial-reduction and zero-workspace contract; its native candidate
is retained because that path uses it.

The joint noise shape, native action shape, Python policy/registry names,
history transaction and separate `reset_history`/`clear_prepared` operations
remain the public contract. Historical measurements above are tied to their
recorded source; they do not by themselves validate this revision.

Current-source validation at `59a2a683` on the same Thor2 / SM110 and NVCC
13.0.88 profile passes all 14 joint-history scenarios and all 28 action/video
comparisons: relative L2, RMSE and max-absolute error are all exactly zero.
Every scenario passes exact eager/three-graph parity. Repeated replay, native
RNG behavior, changed history profiles, window rollover, reset and
session/episode/task isolation pass. Real device inference followed by injected
action-download errors or nonfinite host results preserves the committed
history. Eager/PreferGraph plans and returned outputs retain their resources
after their original owners are dropped.

The current backend run passes 161 tests (2 ignored); the unchanged convolution
suite separately passes all 480 candidate checks across 12 dtype/profile
configurations. All 9 non-ignored DreamZero model contracts pass, along with
4 CPU family tests, 9 loader tests, 2 shared VLA tests and 18 Python policy tests.
CUDA-feature model/backend/test and binding type checks, model-family/CUDA
architecture guards and GitHub CI pass. The native and fresh extension builds
record source/binary SHA-256 identities; all 157 transferred source files match
the recorded revision. Evidence is retained under the ignored
`devlocal/dreamzero-port/architecture-20261030/` directory. These checks do not
rerun the historical whole-model tuned/restored, component, performance or
Nsight measurements above.

The fresh PyO3 extension at the same revision passes the registered AutoPolicy
path from raw cameras/task/state: all 24×32 normalized actions and 24×8 decoded
robot actions match the official captures with zero error. Leading-batch joint
noise `[1,113408]` and one warm replay produce exactly identical outputs. The
actually imported extension path and SHA-256 match the freshly built artifact.
This public rerun covers raw-input parity, batch-noise canonicalization and
repeatability; the broader history/RNG/lifecycle coverage above is the native
acceptance run.

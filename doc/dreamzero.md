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
| `dreamzero/config.rs`, `load.rs` | Strict architecture/profile admission, cuda-new recipe options, construction |
| `dreamzero/weights/` | Complete checkpoint mapping, BF16 transposes and convolution packing, fixed relative-position tables |
| `dreamzero/model/` | UMT5, CLIP, causal temporal VAE, joint DiT, independent video/action UniPC and observed-history computation |
| `dreamzero/model_runner.rs` | Request validation, RNG binding, history identity/transaction, stable workspaces, three graph stages |
| `policies/impls/dreamzero.py` | DROID camera/task/state mapping, UMT5 tokenizer, q99 normalization, relative joint-action decoding |

The family imports model-neutral safe operations through its `backend.rs`.
`check_dreamzero_module_boundaries.py` checks the model/runner/weights direction.
The model owns computation and the runner owns graph preparation. Neither the
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
Vision (CLIP/temporal VAE), Language (both prompts/context projections), and
Action (observed K/V computation plus joint denoising). A compatible prepared
run rebinds stable inputs and executes without capture or autotuning. The
implicit cache holds one profile; switching profiles replaces it.
Streaming observations advance the history length and therefore require a new
profile until window rollover. Preparation recomputes the three stages, and a
successful call copies committed K/V into independent device storage. A warm
single-observation replay measurement does not represent streaming latency.

`autotune` and `tactics` use cuda-new's recipe directory contract. Tuning options
are immutable for a loaded runner. Construct a new runner to select another
store/configuration; legacy tuning JSON is not interpreted as native recipes.

DreamZero selects the backend's `ColdBaseline` GEMM algorithm policy. All
projections use the first successfully prepared cold-fallback algorithm within
their declared workspace and precision constraints. Online tuning performs real
timing and capture validation of that one eligible algorithm; it cannot trade
reduction order for speed. Other families retain unrestricted selection by
default. Recipe/session keys distinguish these domains. This contract is scoped
to the qualified device, library and build; it is not a cross-hardware arithmetic
guarantee. Attention continues to use its existing candidate domain.

Unrestricted GEMM tuning failed full-trajectory action/video gates despite
zero-error isolated component captures. A controlled replay reproduced the error
with the original recipes and restored zero-error action/video outputs by replacing
only GEMM choices with measured cold-baseline choices. Unrestricted tuning is
therefore not an admitted DreamZero numerical profile. The constrained policy is qualified below for the pinned device and build.

## Qualified device profile

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

| Validation | Result |
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
stronger zero-error predicate before restoring 0.02 in the maintained test. Only
that test predicate/comment and this documentation changed after qualification;
the executed source/binary identities and earlier failures remain preserved.

The 14 scenarios cover initial/repeat, three successive history observations,
window rollover, history reset, changed task, changed episode, nine-frame input,
independently changed video/action noise, raw-public-input equivalence and a
separate session. Fixed joint noise establishes reference/native agreement;
separate perturbations demonstrate cross-modal influence. Inputs are synthetic
camera/state/task fixtures, not evidence of closed-loop robot task success.

## Measured performance and residency

Uninstrumented wall times on the qualified device, in seconds:

| Path | Load | First prepare/infer | Warm median | Warm p95 | Warm samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native AutoPolicy, cold recipes | 38.664 | 109.011 | 47.617 | 47.623 | 5 |
| Native AutoPolicy, fresh process with recipes | 32.041 | 109.060 | 47.756 | 47.772 | 5 |
| Official preprocessed BF16 head | 22.446 | 40.542 | 39.875 | 39.893 | 5 |

Load and first inference are separate intervals. Processes are fresh, but OS file
caches were retained. Native timings include raw camera preprocessing/tokenizing,
input upload and returned robot actions. Reference timings begin with preprocessed
BF16 head inputs and include upload through action download; the joint video path
is computed in both. These are different API boundaries, so this is not a speedup
claim. The small-sample p95 is empirical interpolation, not a serving SLA. Profiler
runs are excluded. No other compute PID was observed in the approximately
two-second whole-process samples; this does not exclude shorter activity or the
existing graphics desktop.

Nsight's complete-process allocation trace records a Device allocation high-water
of 79,935,015,722 bytes (74.45 GiB), with Device Static and Pinned categories
reported separately. This is not total physical memory or driver overhead.
Reference PyTorch peaks are 53,079,793,664 allocated bytes (49.43 GiB) and
56,358,862,848 reserved bytes (52.49 GiB); allocator statistics are not equivalent
to native traced allocation usage. System `MemAvailable` is tracked separately
for run admission/protection and is not a process memory measurement.

The complete-process CUDA/NVTX trace contains one marked warm public request:
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

Performance status is **best effort with remaining debt**. The exact native path
is slower at the measured public boundary and has a larger allocation high-water
than the reference allocator reports. Its constrained GEMM domain deliberately
excludes faster alternative reduction orders that failed complete trajectories.
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
FP32 split-K partial reduction before one BF16 output conversion. The explicit
`ColdBaseline` policy, not the workspace bound alone, constrains the algorithm.

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

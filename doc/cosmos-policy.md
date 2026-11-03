# Cosmos Policy: LIBERO direct policy

The `cosmos-policy` family implements observation-to-action inference through
`AutoModel`, `LoadedModel::Vla`, `VlaRuntime` and Python `AutoPolicy`. It is
Cosmos **Policy**, not Cosmos Reason or a generic video-generation endpoint.

## Checkpoint and profile

| Item | Selection |
| --- | --- |
| Official source | [NVlabs/cosmos-policy](https://github.com/NVlabs/cosmos-policy/tree/18a2accadf4e7a3531e56754102af5a24d2316da), revision `18a2accadf4e7a3531e56754102af5a24d2316da` |
| Policy checkpoint, statistics, cached T5 | [nvidia/Cosmos-Policy-LIBERO-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B/tree/cb689ec0e3347c13667d70a78a3447388f5c3bb8), revision `cb689ec0e3347c13667d70a78a3447388f5c3bb8` |
| Wan2.1 VAE | `tokenizer/tokenizer.pth` from NVIDIA Cosmos-Predict2-2B-Video2World, revision `f50c09f5d8ab133a90cac3f4886a6471e9ba3f18`; the byte-identical official [Wan-AI original](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/37ec512624d61f7aa208f7ea8140a131f93afc9a) is also usable |
| Prepared profile | `libero-2b-direct-v1`, batch 1, BF16 model/VAE |
| Observation | Wrist then primary RGB; deterministic 224×224 preprocessing; 9 proprioceptive values in the order `gripper_qpos[0..1], eef_pos[0..2], eef_quat[0..3]`; one of the official 40 cached task texts |
| Sampling | Five denoiser calls, rectified-flow preconditioning, FP64 2ab solver state with FP32 denoiser boundaries |
| Supplied noise | Standard-normal FP32 `[1,16,9,28,28]`; the family applies sigma-max 80 |
| Public output | FP32 `[16,7]` actions, unnormalized by the Python policy |

The proprioception order is positional, not named: the checkpoint normalizes
and the denoiser consumes the nine values in this order, so a permuted vector
still has a valid shape and silently changes the action. The Python policy
exports it as `PROPRIO_LAYOUT`; an adapter that reads LIBERO's
`robot0_gripper_qpos`, `robot0_eef_pos` and `robot0_eef_quat` must concatenate
them in exactly that order.

The converter verifies SHA256 before reading any source asset. Its accepted
identities are maintained in
[`checkpoints/cosmos_policy.py`](../python/apxinf/apxinf/checkpoints/cosmos_policy.py).
The VAE identity is
`38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`.
Source assets are read in place. An existing output directory is rejected;
`config.json` is written last as the completion marker.

```bash
python -m apxinf.checkpoints.cosmos_policy \
  --assets /path/to/official-assets --output /path/to/prepared-cosmos-policy
```

The assets directory contains `Cosmos-Policy-LIBERO-Predict2-2B.pt`,
`tokenizer.pth`, `libero_dataset_statistics.json` and `libero_t5_embeddings.pkl`.
Conversion requires PyTorch and safetensors; native inference does not import the reference
model or use PyTorch as an execution engine.

```python
from apxinf.policies import AutoPolicy

policy = AutoPolicy.from_pretrained(
    "/path/to/prepared-cosmos-policy",
    device="cuda:0",
    model_variant="bf16",
    autotune=True,
    tactics="/path/to/cosmos-policy-recipes",
    state_key="proprio",
    image_keys=("wrist_image", "primary_image"),
)
result = policy.infer({
    "wrist_image": wrist_rgb,
    "primary_image": primary_rgb,
    "proprio": proprio_9,
    "prompt": "put both the alphabet soup and the tomato sauce in the basket",
}, noise=fixed_standard_normal)  # optional complete joint noise
policy.close()
```

The policy accepts **decoded HWC uint8 RGB**, state and a prompt, like the
other observation-to-action policies. It does not accept encoded JPEG bytes.
The default LIBERO image transform performs a JPEG-quality-95 encode/decode
round trip, resize, center crop and final resize to reproduce the reference
training-data recipe. This is a policy/profile transform, not a VAE or DiT
input-format requirement. Transport decoding, simulator image orientation,
camera mapping and robot conventions belong to the application/robo adapter.

`image_transform=` injects a Cosmos-local image transform into the policy;
it is not forwarded to native model loading. The callback receives one decoded
RGB image and must return HWC uint8 `[224,224,3]`. An adapter which already
applied the full profile recipe can pass `image_transform=lambda image: image`
to avoid applying JPEG/crop/resize twice. An image being 224x224 alone does not
mean that recipe has been applied. Overriding it changes the policy input recipe
and requires application-level validation; the native RGB boundary is unchanged.
The default callback preserves the existing reference pixel checks.

The native boundary is resized RGB plus a normalized FP32 `[9]` state, one
cached-task ID and optional joint noise; it returns normalized FP32 `[16,7]`
actions. The policy owns prompt mapping, state normalization and physical-unit
action unnormalization. This matches the layer responsibilities of other
families without sharing their model implementations. Unlike token-based
policies, the cached-task ID is not a text-token sequence, and the joint latent
noise shape is independent of the action shape.

The caller owns simulator image orientation and robot-specific action adapters.
Unknown task text is rejected; this profile does not run a general T5 encoder.
Use `autotune=True` to populate device-specific exact recipes on the first real
request, then ship or reuse that `tactics` directory: it is the only thing that
carries per-shape tuner decisions. Fresh processes can use `autotune=False` with
the same directory.

Without a recipe the untuned path no longer falls through to the generic
baseline for the shapes this family owns. The backend prefers, in order, the
persistent-scheduler FMHA candidate for 1764-token attention, the cuBLASLt
128x224 2-CTA tile for the `M=1764,N=2048,K=8192` MLP-down projection, the
vendor heuristic rank 0 for the `1764/2048/2048` square projections, and the
CUTLASS lookup-table exact-GELU epilogue for the `1764x8192x2048` MLP up. These
are compile-time-verifiable shortcuts, not measured picks, so a real recipe can
still win over them; they exist so that a fresh process is close to the accepted
latency rather than silently several times slower.
`model_runner=` injection uses the same policy interface. The policy validates
joint-noise geometry independently of the action shape.

## Computation and ownership

The native family is `cuda_cosmos_policy/`; the existing public `cosmos-policy`
selection and checkpoint metadata remain compatible. `cuda_cosmos_policy` is
also an explicit native loader alias.

```text
cuda_cosmos_policy/
  config.rs / load.rs          validation and construction
  backend.rs                  safe CUDA API reexports only
  weights/host.rs              host checkpoint storage and validation
  weights/bf16.rs              BF16 device types, mapping, packing and upload
  weights/mod.rs               deliberate weight exports
  model/mod.rs                modality connection and five-step RF/2ab schedule
  model/blocks/bf16.rs         Bf16Blocks, private VAE/DiT and tensor composition
  model/blocks/mod.rs          one concrete precision implementation
  model_runner/runner.rs       VlaRuntime, request validation and implicit cache
  model_runner/prepare.rs      device inputs, binding/RNG, sessions and graph lifetime
  model_runner/mod.rs          runtime export
```

`Bf16Blocks` owns intermediate representations and reports workspace requirements;
the runner allocates canonical device inputs and execution resources. The model
calls observation encoding, noise initialization, denoising steps and action
readout through that boundary. Subnetworks remain private inside the single
BF16 file. Host storage does not depend on CUDA; weights never execute model
forward methods. There is no additional WAM runtime, variant abstraction or
import from another model family. The maintained
[`ownership guard`](../scripts/check_cosmos_policy_module_boundaries.py) checks
these dependencies and file placement; numerical tests establish behavior.

The latent has nine frames: blank, current proprioception, current wrist image,
current primary image, action, future proprioception, future wrist image,
future primary image and value. All 1,764 patched tokens pass through all 28
transformer blocks at every denoiser call. Action readout averages all 112
repeated action chunks in latent frame 4. Future modalities participate in
joint denoising even though the public `Action` exposes only actions.

The direct path does not require future RGB decoding. Native assets therefore
contain the VAE encoder and mean projection, not its decoder. Planning/search,
RoboCasa/ALOHA checkpoints, arbitrary prompts, alternate image geometry,
quantized variants and altered sampling schedules are outside this profile.

Preparation binds precision, full latent geometry, checkpoint weights, device
stream and the fixed sampling schedule. `Eager`, `PreferGraph` and
`RequireGraph` are explicit policies; default inference/preparation uses
`PreferGraph`, recording a fallback reason if capture fails. Explicit
`RequireGraph` still rejects a capture failure; placeholder preparation disables online
tuning. `prepare_for` may tune on a real sample when loading enabled autotune.
Compatible runs perform no tuning or capture. Prepared device addresses remain
stable while each run rebinds RGB, state, task ID and noise; changing values at
fixed geometry does not rebuild the graph. The entire observation encoding and
five-step sampler are captured, while Python transforms, request upload and host
result conversion remain outside it.

Cross-attention currently reuses temporary K/V storage but recomputes text
projections and K normalization per layer and denoising step. It does not cache
projected text K/V across steps or requests. That invariant text work can later
be cached by layer, task, weights, device and precision. Self-attention K/V change
with the noisy latent and timestep, so an autoregressive append-only KV cache is
not applicable. Fixed prompt and image geometry enable graph reuse; they do not
make changing visual/latent activations cacheable. Every request recomputes causal
VAE caches from the singleton initial chunk and resets solver state. Explicit
plans retain weights after runner destruction; returned actions own storage
independent of subsequent replay. `clear_prepared` evicts only implicit plans.

New backend operations remain model-neutral: causal THWC convolution lowering,
staged channel L2 normalization, staged RGB normalization, elementwise multiply,
FP64 linear recurrence and indexed column mean. GEMM, attention, normalization,
activation, gathering, casting and device copies use existing safe CUDA APIs.
Convolution tiles reuse GEMM bindings and copy their outputs on device so vendor
resource lifetime is bounded by layer geometry rather than tile count.

## Validation and limits

Validated on NVIDIA Thor (SM110), CUDA 13.2 / driver 595.78, using the pinned
BF16 profile above. Reference captures use PyTorch 2.9.0+cu130 and
Transformer Engine 2.8. Native measurements below use a Rust debug build.
Private inputs, scripts, captures, recipes and logs are kept in the port's
ignored `devlocal/cosmos-policy/` evidence directory; none is required at runtime.

| Check | Result |
| --- | --- |
| Reference repeatability | 128 captured tensors bitwise identical across reruns, including after device reboot |
| Public preprocessing | RGB bytes and FP32 proprioception bitwise identical; NumPy 1.26.4 and 2.2.6 checked |
| VAE | All five encoded prefix frames: relative L2 ≤0.00527, cosine ≥0.999986; eager/Graph bitwise |
| Full 28-layer DiT | Patch and time projections exact; observed layers and all nine final frames pass relative L2 ≤0.02 / cosine ≥0.999; eager/Graph bitwise |
| Five-call joint sampler | Every semantic frame passes; tuned run maximum frame relative L2 0.00994, minimum cosine 0.999950 |
| Normalized actions, fixed seed 1 | Maximum absolute error 0.004625, RMSE 0.001129; gripper decisions agree |
| Public actions, independent fixed seed 7 | Maximum normalized error 0.003891, RMSE 0.000978; gripper decisions agree; no Torch import during inference |
| Whole-model preparation | Eager/Graph bitwise; repeated replay, changed RGB/state/task, supplied/generated noise, retained actions, reset/drop, implicit-cache clearing and placeholder preparation pass |
| Cold/fresh-process recipes | Online tuning passes; fresh process with tuning disabled loads 50 GEMM recipes and reproduces the same actions |
| Public entry point | Real `AutoPolicy → ModelRunner → AutoModel → VlaRuntime` inference, repetition, RNG reset/advance and close/lifetime checks pass |
| Backend checks | 11 all-candidate Torch comparisons, 7 device integration tests, shape/alias/capture contracts and operator catalog checks pass |

The VAE prefix optimization was checked with the reference's original
channels-last input strides: encoding the first 17 RGB frames exactly reproduces
the first five latents of its complete 33-frame encoding. Changing the reference
probe to contiguous NCTHW changed cuDNN rounding and was not used as the golden.
The instantiated checkpoint uses spatial RoPE NTK `3^(42/40)` and temporal `1`.

At initial port commit `544865c0`, with two warmups and five timed calls, the native whole-model graph measured
**2455.89 ms median / 2502.46 ms p95**, including input binding, independent
Action publication and synchronization. The Python policy measured a 2517.16 ms
median including preprocessing and action unnormalization. The official direct
reference, with optional future-image decoding disabled, measured 1197.35 ms
median / 1198.45 ms p95. Functional acceptance passes; performance remains
**best effort with performance debt** (roughly twice reference latency).
Existing GEMM and attention candidate tuning was exercised; the convolution
lowering and repeated text projections remain optimization opportunities.

A 2026-10-30 release-build rerun on Thor MAXN compared the post-merge baseline
with vectorized, bounds-checked causal volume lowering. With five warmups and
20 measured requests per variant, no other GPU process observed during either
run, median/p95 changed from **2428.23 / 2428.78 ms** to
**1410.86 / 1411.78 ms** (1.72x median speedup). This measures native request
binding, Graph replay, independent Action publication and synchronization;
Python preprocessing and D2H conversion are excluded. The complete joint latent
was bitwise identical between variants; normalized action maximum error against
the official fixture was 0.00459105 and RMSE 0.00108809. Updated causal-volume
padding/history/stride/unaligned-view tests and the full preparation/lifecycle
regression passed. This is an optimization checkpoint, not a claim of reaching
the BF16 roofline; convolution materialization and transformer memory passes
remain significant.

A subsequent same-device A/B added a 128-column RMSNorm candidate that assigns
one warp per row while preserving the previous reduction order. Five warmups
and 20 requests per variant measured **1410.29 / 1410.93 ms** median/p95 before
and **1311.46 / 1312.01 ms** after (7.0% lower median latency). The complete
joint output remained bitwise identical, and FP16/BF16 short-row reference,
in-place Graph, and full-model lifecycle tests passed. This remains a measured
optimization milestone, not evidence that the original 220–300 ms ideal
roofline estimate is attainable.

The next iteration replaced the seven-pass BF16/FP32 rotary chain with the
backend's standalone precomputed half-split rotation, preserving separately
rounded FP32 products and addition. Another uncontended 5-warmup/20-request
A/B measured **1306.99 / 1312.69 ms** median/p95 before and
**1106.51 / 1107.83 ms** after. Complete joint output and public action arrays
were again bitwise identical. Multiple rotary shapes, CPU reference, in-place
Graph, malformed contracts, and full-model preparation/lifecycle passed.
The coefficient table is broadcast over heads; index and FP32 intermediate
buffers are no longer materialized. All measurements retain BF16, 1764 tokens,
28 layers and five denoiser calls.

Vectorizing aligned embedding gathers subsequently reduced median/p95 from
**1106.72 / 1107.19 ms** to **1009.51 / 1009.90 ms**, using the same five warmups
and 20-request A/B procedure without observed GPU contention. Full joint output
and action arrays remained bitwise identical; alignment fallback, invalid-ID
clamping, Graph replay and model lifecycle tests passed. Across these iterations,
BF16 latency is about 2.41x faster than the original release baseline. The
remaining attention and memory passes still prevent a roofline-limit claim.

Parallelizing the 384-dimensional VAE attention fallback's score generation,
while preserving dot-product and softmax accumulation order, then reduced
median/p95 from **1009.99 / 1010.49 ms** to **934.02 / 934.18 ms** in another
5-warmup/20-request A/B. Attention recipes were refreshed after the kernel build
fingerprint changed. The complete joint output and actions remained bitwise
identical, with unchanged reference errors and passing lifecycle tests. The
specialized path reuses the existing probability workspace; no model geometry,
precision, sampling count or attention formula changes.

Warp-per-row Channel L2 normalization for 96/192/384 channels and at least
1024 rows reduced the next same-device A/B from **930.60 / 933.40 ms**
median/p95 to **895.92 / 896.53 ms**. Both runs used five warmups and 20 measured
requests without observed GPU contention. Small row counts retain the original
kernel. Every staged BF16 boundary and the reduction tree are preserved; full
joint output, actions and fixture errors remain unchanged. Threshold/tail,
clamping, zero-input and in-place Graph tests passed. This is approximately
2.71x faster than the original 2428.23 ms release baseline, with the same model
precision, geometry and sampling count.

Extending the existing CUTLASS dense-attention candidate to the DiT's BF16
H16/D128 self/cross shapes, with residual key masking, reduced the next
same-device median/p95 from **898.58 / 898.96 ms** to **844.75 / 845.34 ms**
(five warmups, 20 requests, no observed GPU contention). Cold preparation
retuned the attention recipes; warm Graph replay selected CUTLASS for all
280 DiT attention calls. The accumulation order differs from FA2, so joint
outputs are no longer bitwise identical to that baseline. All nine joint-frame
reference checks and lifecycle tests pass; normalized action max absolute
error is **0.00597128**, RMSE **0.00116459**, with deterministic replay. The
largest frame relative L2 is **0.0113869**, within the existing 0.02 threshold.
These remain fixture-level precision checks, not episode success-rate results.
Model precision, geometry and denoiser count are unchanged.

Specializing precomputed rotary index arithmetic for H16/D128 then reduced a
paired A/B from **850.18 / 850.93 ms** median/p95 to **833.19 / 833.75 ms**,
again using five warmups and 20 uncontended requests. Full joint output is
bitwise identical to the CUTLASS-attention version, with unchanged fixture
errors and passing rotary/capture/lifecycle tests. Other rotary geometries
retain runtime indexing. This is approximately **2.91x** faster than the
original 2428.23 ms baseline; the ideal roofline estimate remains unachieved.

Vectorized BF16 Multiply/EulerUpdate loads and stores reduced the next paired
median/p95 from **834.04 / 834.31 ms** to **767.95 / 768.23 ms** (five warmups,
20 measured requests, no observed GPU contention). The scalar arithmetic and
rounding remain unchanged; the reference/Graph tests and full model lifecycle
checks pass, with unchanged action fixture errors. This optimization changes
backend dispatch only, not model structure or public input/output contracts.
The current user target is **200 ms**; it has not been achieved.

The VAE residual and head normalization workspaces share a chronological THWC
allocation across the singleton and remaining chunk. Two immutable zero frames
precede the timeline. Each producer writes a disjoint current-chunk view; its
convolution reads the same allocation through a view including the preceding
two frames, using symmetric spatial padding. This removes per-request history
concatenation for those convolutions while preserving the BF16 inputs supplied
to each convolution, causal edges, and output intermediates. Temporal downsampling reduces the prepared
timeline capacity at the corresponding model stage. Buffer views retain their
shared allocation through Graph execution and model lifetime.

The contiguous-history version passes the unchanged VAE, DiT, joint-frame,
action, task-update, Graph/RNG and lifetime gates, plus all 214 backend tests.
Matched five-warmup, fifty-warmup and reversed-order comparisons improve by
11.19, 7.19 and 5.76 ms respectively. The combined eighty measured requests per
variant give **368.08 ms** for the candidate versus **374.73 ms** for the
contemporaneous baseline. NSYS confirms that hot input/history materialization
falls from 26 vectorized launches (10.78 ms) to 5 (2.17 ms); 21 concatenations
are absent. The target below 200 ms remains unmet.

Fourteen new exact Conv3d geometry keys arise from the contiguous views; all
333 inherited recipes are unchanged. Ordinary tuning selects two previously
qualified convolution configurations differently, so outputs differ from the
preceding version. Public normalized maximum error/RMSE against the original
reference are **0.00395649 / 0.00108858**, within the unchanged limits, and each
version remains bitwise stable across replay and eager/Graph execution. A
separate diagnostic holding the convolution configurations equal restores
bitwise equality of actions and normalized outputs, isolating storage layout
from accumulation-order changes. Those diagnostic recipe overrides are not
used for production selection or performance acceptance.

The model-fixed black first VAE frame is now encoded during plan preparation.
Its output and causal history tensors remain model-owned and are reused by
requests; request RGB only overwrites frames 5–12 and never frame zero. The
runner prepares constant operators in a separate traversal before recording the
request sequence, so eager execution and Graph capture see the same operator
order. The paired median/p95 decreased from **765.53 / 765.88 ms** to
**747.75 / 748.34 ms**, using five warmups and 20 uncontended requests. Full
joint output remains bitwise identical; changed-RGB/state/task and RNG/lifetime
tests pass. This moves invariant computation out of requests; it does not cache
observations, alter checkpoint arithmetic or remove denoiser steps. Preparation
cost is excluded from these steady-state timings. **200 ms remains unmet.**

A single tuned plan's process GPU-memory samples reached 14,044 MiB (3-second
polling). This is sampled driver accounting, not an exact allocator peak.
The reference reported 6,866,727,424 bytes peak PyTorch allocation; these two
memory measurements have different scopes. The lifecycle test that temporarily held two plans reached 23,962 MiB in
process GPU-memory samples. The initial per-tile GEMM binding memory growth was fixed
by sharing tile bindings and retested successfully.

The 2026-10-29 ownership refactor was rebuilt on Thor and explicitly reran all
eight family tests, including the three fixture-dependent device tests: VAE,
DiT, and full joint actions/preparation/lifetime. The VAE prefix and complete
joint output are bitwise identical to the retained pre-refactor outputs. The
11 Python policy tests and the module ownership guard also pass. These checks
validate the new module/interface boundaries; the latency figures above remain
labelled with their original measured revision.

This establishes fixed-observation numerical/runtime acceptance, not a LIBERO
simulation success-rate benchmark. The broad cuda-new integration-test build
also encounters pre-existing unresolved helpers in `qwen38_gdn_batched_ops.rs`;
the targeted operator and model checks above were run explicitly.
No closed-loop robot rollout, other GPU,
other checkpoint, precision, arbitrary text encoder or planning-search mode is
claimed. Future image/state/value latents are genuinely jointly denoised; the
optional future RGB decoder is outside the direct Action path.

## Reproducible native performance measurement

`cosmos_policy_bench` uses the public `AutoModel` / `VlaRuntime` contract with
prepared BF16 assets and captured canonical input. Its fixture directory must
contain `joint-fixture.safetensors` (normalized state, joint noise and reference
actions), `public-rgb.bin` and `public-task-id.txt`. It does not download weights
or synthesize a reference. For example, from the project root:

```bash
cargo run --release -p apxinf-model --features cuda \
  --example cosmos_policy_bench -- /path/to/prepared-assets /path/to/fixtures \
  devlocal/cosmos-thor-roofline/run-name/results/native.json \
  --recipes /path/to/recipes --warmup 5 --iterations 20
```

Use `--autotune` once for missing exact recipes; preparation/tuning is excluded
from timed samples. The benchmark requires successful Graph capture, checks
reference action error/gripper decisions and fixed-input replay, and reports
individual wall-clock samples plus median/p95. These samples include request
binding, model Graph, independent Action publication and synchronization; they
exclude Python preprocessing and action D2H conversion. Device-wide contention,
power mode and clocks must be recorded separately before interpreting a speedup.
`--profile` brackets only timed requests with the CUDA profiler API, allowing
Nsight Systems `--capture-range=cudaProfilerApi --capture-range-end=stop` without
including model loading or preparation. Use `--iterations 1` for an individual
steady-state trace. This timing tool is not a substitute for full joint-latent
numerical and lifecycle tests.


The BF16 DiT uses grouped adaptive LayerNorm with explicit staged BF16 numerics.
Its nine frame groups share shift/scale conditioning over 196 tokens per frame,
without expanding full shift/scale tensors. Grouped gated residual updates also read the nine compact modulation groups
directly, retaining the BF16 product rounding before residual addition. This preserves the normalization, scale-plus-one, product and
shift rounding boundaries; it does not change the five-step schedule or token
geometry. Performance and complete-model acceptance for this change are tracked
in the current optimization run, not inferred from operator microbenchmarks.

A 2026-10-31 Thor release-build validation of this path measured true AutoPolicy
observation-to-action hot latency at **695.74 ms median / 696.22 ms p95** (five
warmups, twenty samples, fixed official seed-1 observation/noise). This includes
Python processing and action download. Matched native request latency improved
from 743.39 ms to 688.03 ms; complete joint outputs remained bitwise identical.
The <200 ms optimization objective is still unmet. These are fixed-fixture
numerical/performance results, not episode success-rate measurements.

The subsequent grouped gated-residual validation measured **668.68 ms median /
669.42 ms p95** through AutoPolicy (same five warmups/twenty samples). Matched
native requests improved from 681.31 to 650.16 ms. Complete joint outputs and
actions remained bitwise unchanged; the BF16 five-step profile is unchanged.


The BF16 VAE now expresses causal 3D convolution through the backend's prepared
`Conv3d` semantic. It supplies THWC history/input geometry and immutable flattened
weights; native preparation owns packing, algorithm choice and fallback. The
model applies bias separately after the convolution's BF16 output rounding.
The five-step full joint denoising profile and acceptance gates are unchanged.
On Thor, matched native graph medians improved from 649.81ms to 542.70ms;
actual AutoPolicy hot median is 564.85ms (p95 565.49ms, five warmups and twenty
samples). The fixed-reference normalized action max error is 0.00357 and RMSE
0.000810. VAE, full joint, eager/graph equivalence, observation updates, RNG and
output-lifetime gates pass. Direct convolution changes accumulation order, so
outputs are not bitwise identical to the previous lowering implementation.
The target below 200ms has not been reached. These are fixed-fixture numerical
and latency results, not an episode success-rate claim.

Each `ConvWorkspace` keeps its lowered tile buffers alive for the plan's
lifetime, so the per-convolution workspace limit is effectively multiplied by
the ~24 VAE convolutions. The family now passes 32 MiB instead of 512 MiB:
lowering derives the tile height from that budget, and 512 MiB let every stage
run at its 4096-row cap and retained 543 MiB of scratch, while 32 MiB retains
about 280 MiB and costs roughly 210 additional tile launches across the whole
request. Smaller values trade launches faster than they release bytes. The prefix chunk's
head tail is deliberately *not* skipped even though `condition_map` substitutes
proprioception at t == 1 and never reads latent 1: removing it changes the
operator sequence the shared session observes, which shifts the buffer offsets
the following chunk sees and selects different vectorized kernels. That is
numerically visible (joint frame 2 relative L2 0.46 versus 0.005), so the
redundant copy stays as a stability constraint, not dead work.


Vectorized Conv3d input/history materialization reduces a subsequent matched
native median from543.77ms to523.53ms. Actual AutoPolicy median is541.62ms
(p95542.28ms, five warmups/twenty samples), with identical full joint output
bytes and unchanged action errors. Aligned channels use eight-element BF16
transfers; odd channels and nonaligned history retain the scalar path.


Caching AdaLN row values in registers further reduces matched native median
from 521.77ms to 505.79ms. Actual AutoPolicy median is 522.61ms (p95 523.65ms,
five warmups/twenty samples), with fulljoint output bytes unchanged and the
same numerical/lifecycle acceptance. This remains above the 200ms target.


Self-attention Q/K now express per-head RMSNorm followed by precomputed rotary
embedding through the backend compound operator, retaining BF16 rounding at
the stage boundary. Cross-attention continues to apply RMSNorm without rotary
embedding. Matched native latency decreases from 505.26ms to 484.75ms, and
actual AutoPolicy median is 502.25ms (p95 502.71ms, five warmups/twenty samples).
Full joint output bytes remain identical; numerical, input-update, graph and
lifecycle checks pass. The target below 200ms remains unmet.

Self-attention prepares V first, then schedules each Q/K RMSNorm and rotary
operation immediately after its projection. The independent Q, K and V buffers retain their existing
BF16 shapes; all three projections read the same normalized input. The model
blocks own this order, and eager execution and graph capture use the same path.

Fixed schedule modulation is prepared once per model plan. For each of the five
timesteps, the model retains BF16 shift/scale/gate tensors for all 28 blocks and
the final layer (47,370,240 bytes including observer time-global tensors).
Preparation uses the original shared GEMM scratch, activation, normalization
and BF16 addition sequence before copying each result into stable model-owned
storage. Requests still execute every denoising step and every token; RGB,
state, task embeddings and sampled noise remain request inputs. The cache is
owned by `Bf16Blocks` and follows the prepared model lifetime.

On Thor, the fixed-modulation paired native median/p95 changed from
**484.95 / 486.99 ms** to **480.96 / 482.10 ms** (five warmups, 20 samples).
The real AutoPolicy observation-to-action median/p95 is **498.61 / 499.77 ms**,
including preprocessing and D2H output; first call was 2549.91 ms. Official
DiT/VAE fixtures and full joint lifecycle checks pass. The complete joint
output remains bitwise identical to the preceding RMSNorm/RoPE version, and
public normalized action max error/RMSE remain 0.00356975 / 0.00080957.
The **200 ms target remains unmet**.

Cross-attention text K/V projections and K normalization now execute once per
request after gathering that request's task embedding. All five denoising steps
reuse the 28 model-owned K/V pairs; self-attention and per-step query computation
remain unchanged. Refresh is part of both eager execution and the captured
request graph, so task changes do not reuse stale projections. The buffers use
117,440,512 bytes in total, replacing the previous 4,194,304-byte shared scratch.

The text K/V reuse A/B on Thor reduced native median/p95 from
**481.07 / 482.61 ms** to **470.71 / 471.33 ms** (five warmups, 20 samples).
The real AutoPolicy median/p95 is **493.73 / 494.48 ms**, with first call
2551.86 ms. Full joint output is bitwise unchanged; official layer fixtures,
RNG/lifetime tests, and eager-versus-Graph task transitions pass. The public
action errors remain unchanged. **200 ms remains unmet.**

A subsequent GEMM recipe A/B kept the same BF16 implementation and changed two
cuBLASLt selections (MLP down projection to heuristic rank 5, 2048-wide square
projection to rank 2). Two public runs per configuration gave pooled medians
**489.69 ms baseline / 487.36 ms candidate** (40 samples each); individual
candidate medians were 488.24 and 486.97 ms. The small gain is less than the
roughly 5 ms native-only improvement. Runtime reports confirmed both selections
in the actual AutoPolicy path, and full joint bits, task transitions and public
accuracy/lifecycle checks remained unchanged. This is an independently measured
recipe configuration, not a new default backend dispatch. **200 ms remains unmet.**

The BF16 MLP up projection now expresses GEMM plus staged exact erf GELU as
a single backend semantic. It retains the original BF16 projection rounding
before evaluating GELU. Backend preparation owns algorithm selection and all
execution resources; the model does not select a GPU kernel. The full five-step
schedule and text K/V refresh remain unchanged. On Thor, a fresh task-local
recipe selected CUTLASS implementation 11, version 1, configuration 1
(256×256×64 tile, 2×2 cluster). The first real AutoPolicy run measured
**470.59 ms median / 471.66 ms p95**, versus a contemporaneous **488.00 ms**
baseline median (five warmups, 20 samples each). Full joint output remains
bitwise identical; official DiT/VAE, changed-task eager/Graph, action error,
RNG and retained-output checks pass. Backend all-candidate, recipe, resource
and graph tests pass; the complete backend regression suite passes with the
existing external Qwen reference fixture linked into the checkout. A second
20-sample public run gives 470.63 ms median; pooled median is **470.62 ms**.
**200 ms remains unmet.**

The DiT residual prologue now requests grouped gated residual plus the next
adaptive LayerNorm through `fused_add_norm`. The initial normalization remains
separate; each residual retains hidden state and prepares the next sublayer's
normalized input, including the final two-part style. On Thor, two real AutoPolicy runs measure **464.34 / 464.21 ms** medians,
with pooled median **464.27 ms**, versus a contemporaneous **470.65 ms**
baseline. Native request latency is **442.90 ms** median. The full joint
output remains bitwise identical; all model numerical and graph/lifetime
gates pass. The complete backend suite reports 198 passed, 0 failed,
20 ignored. NSYS confirms 420 fused launches plus five initial normalizations.
**200 ms remains unmet.**

The VAE d384 single-head attention now has an SM110 backend candidate using
strided-batched cuBLAS QK, FP32 softmax, exact BF16-to-FP32 V conversion, and
pedantic FP32 probability-times-V before BF16 output. It supports batches 1–4;
preparation owns all buffers and the cuBLAS handle. The model continues to
request ordinary dense attention. Fresh recipes select parallel softmax for
both VAE shapes and include the cuBLAS runtime version.

Two real five-step BF16 AutoPolicy runs measure **448.37 / 451.69 ms** medians,
with pooled median **449.90 ms**, versus a contemporaneous **463.77 ms**
baseline. Native request latency is **435.55 ms** median. DiT-only output is
bitwise unchanged; VAE/full-joint output changes within the established
reference tolerances. Public normalized action max error/RMSE are
**0.00298630 / 0.00079586**, with physical action max error **0.00279966**.
All model numerical, task-transition, graph/RNG/lifetime gates pass; backend
regression reports 201 passed, 0 failed, 20 ignored. NSYS confirms the old
batch-four scalar attention QK/PV kernels are absent from the hot graph.
**200 ms remains unmet.**

VAE normalization followed by SiLU now requests `channel_l2_norm_silu`.
The backend preserves every staged BF16 normalization rounding, including the
weighted result before SiLU, and fuses the activation into generic/warp
normalization kernels. Two real AutoPolicy runs measure **447.78 / 444.52 ms**
medians, pooled **445.39 ms**, versus a contemporaneous **452.68 ms** baseline.
Native request median is **426.14 ms**. VAE and complete joint outputs remain
bitwise unchanged from the batched-attention endpoint; public action errors,
fixed-noise replay, task transitions, RNG and retained-output checks pass.
Backend regression reports 202 passed, 0 failed, 20 ignored. The standalone
fusion test covers the generic path and warp threshold, exact aliasing,
independent SiLU reference values and invalid epsilon.
**200 ms remains unmet.**

VAE residual blocks and attention output projections now combine convolution
bias and residual addition through existing `residual_add` with
`BiasOrder::BiasThenResidual`. This preserves the BF16 convolution and bias
rounding boundaries; temporal cache edges still refer to normalized inputs.
Two public AutoPolicy medians are **440.91 / 443.00 ms**, pooled **441.51 ms**,
versus a contemporaneous **447.27 ms** baseline. Native request median is
**421.30 ms**. VAE/full-joint output bytes and action errors remain unchanged;
all model numerical, task-transition, graph, RNG and lifetime gates pass.
NSYS confirms 11 separate bias launches are removed. This is model composition
of existing backend semantics; backend sources are unchanged from the preceding
202-pass regression. **200 ms remains unmet.**

The VAE preparation path also propagates the runner's tuning/cache/fallback
settings through the provider-neutral `Conv3dPolicy`. Sample-free preparation
disables online tuning before constructing the model. Convolution choices are
prepared and retained before recording the request graph; request replay never
selects a provider. The optional backend engine candidates and exact recipes
are owned by `apxinf-cuda-new`, with legacy direct/lowered implementations as
fallbacks. This does not change the five-step schedule, precision or observation
contract. End-to-end performance acceptance remains separate from operator
selection and compilation.

The reference crop resize is implemented once, in NumPy
(`_resize_crop` in the Python policy). It preserves FP32 horizontal
intermediates, separately rounded first taps, fused second taps, and final
nearest-even uint8 rounding. A C++ CPU extension with the same algorithm was
evaluated and removed: it saved about 2.6 ms per request (~0.8% of the
measured 335.81 ms), only applied when the optional build succeeded, and
duplicated the NumPy path line for line. JPEG quality, crop geometry, and
custom-transform behavior are unchanged.


The VAE convolution producer can now request
`fused_add_channel_l2_norm_silu` to publish its biased/residual result together
with the next normalization. The input convolution prepares the first residual
block's `norm1`; each first residual block prepares the second block's `norm1`;
the final downsample convolution prepares the next level's `norm1`. Each
residual block's first convolution similarly prepares `norm2`. Middle blocks
still normalize their inputs explicitly. Every original intermediate output is
materialized, and bias-before-residual BF16 rounding is preserved. Model/Blocks
owns this ordering through the safe L3 API; alignment and kernel selection remain
inside the model-neutral Norm backend. This changes neither denoising steps nor
model precision.

The plain 128-column BF16 RMS path packs eight elements per lane for
16-byte-aligned operands. Its reduction tree matches the scalar path; FP16
and independently unaligned input, weight, or output retain the scalar path.
This backend dispatch is model-neutral and preserves in-place Graph use.

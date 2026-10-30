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
| Observation | Wrist then primary RGB; deterministic 224×224 preprocessing; 9 proprioceptive values; one of the official 40 cached task texts |
| Sampling | Five denoiser calls, rectified-flow preconditioning, FP64 2ab solver state with FP32 denoiser boundaries |
| Supplied noise | Standard-normal FP32 `[1,16,9,28,28]`; the family applies sigma-max 80 |
| Public output | FP32 `[16,7]` actions, unnormalized by the Python policy |

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
request. Fresh processes can use `autotune=False` with the same `tactics`
directory; untuned fallback attention is substantially slower.
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

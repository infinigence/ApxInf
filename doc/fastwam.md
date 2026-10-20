# Fast-WAM action inference

The `fastwam` family implements the action-only profile of
[`lerobot/fastwam_libero_uncond_2cam224`](https://huggingface.co/lerobot/fastwam_libero_uncond_2cam224)
through the existing `VlaRuntime` and Python `AutoPolicy` interfaces. It does not
extend the shared observation, action or prepared-inference contracts.

## Supported profile

| Property | Value |
| --- | --- |
| Checkpoint revision | `53983e1249b4eb4d89ab42b40b8678a882d331ad` |
| Reference implementation | LeRobot `146d8474932c8b566717eb2738ba459337ab71c4` |
| Backend / variant | CUDA through `apxinf-cuda-new`; `bf16` |
| Images | Two already resized 224 × 224 RGB uint8 views, NHWC or NCHW |
| Text | UMT5 tokenizer, right-padded to 128 positions |
| State / actions | 8 normalized state values; 32 × 7 normalized native actions |
| Schedule | Ten shifted flow steps; supplied noise or keyed native RNG |
| Scope | One observed frame; action prediction without generated video |

The frozen Wan VAE, UMT5 encoder, video conditioning and all action steps execute
natively on the GPU. Wan attention preserves its F32 arithmetic; UMT5 preserves
its BF16 intermediate rounding. DiT residual state and LayerNorm use F32 to
avoid repeated BF16 rounding across blocks. The small action input and output
projections preserve F32 state through Euler integration and the next step. Branch
projections, gates and schedule deltas retain BF16 rounding. The variant name
describes checkpoint weights, not the dtype of every intermediate; both small
action projections' BF16 checkpoint values are promoted exactly to F32 when loading.

Other image resolutions, batch sizes, context lengths, action schedules, model
variants and video generation are rejected. Image resizing belongs before the
policy's declared image boundary; this implementation requires the caller to
supply the reference-preprocessed 224 × 224 images.

## Assets and policy

The model directory must contain the original `config.json`, safetensors and
LeRobot pre/postprocessor files, including their normalization state files.
Supply the frozen encoders and tokenizer through the existing `assets` mapping:

```python
from apxinf import AutoPolicy

policy = AutoPolicy.from_pretrained(
    "/models/fastwam_libero_uncond_2cam224",
    device="cuda:0",
    model_variant="bf16",
    assets={
        "text_encoder": "/models/wan22/text_encoder",
        "vae": "/models/wan22/vae",
        "tokenizer": "/models/umt5-xxl",
    },
)
result = policy.infer({
    "observation.images.image": camera_rgb_u8,      # [224,224,3]
    "observation.images.image2": wrist_rgb_u8,      # [224,224,3]
    "observation.state": proprioception_f32,        # [8]
    "task": "pick up the red block and place it in the bowl",
})
actions = result["actions"]                         # [32,7], task convention
```

Alternatively place the assets in `text_encoder/`, `vae/` and `tokenizer/` under
the checkpoint directory. Loading does not download assets. The encoders are
from `Wan-AI/Wan2.2-TI2V-5B-Diffusers` revision
`b8fff7315c768468a5333511427288870b2e9635`; the tokenizer is `google/umt5-xxl`
revision `66cb9e7e85526fe440a945569e42c72fb6cbc0ad`.

The policy orders camera keys, applies the checkpoint task template, tokenizes,
normalizes state, unnormalizes actions and applies the checkpoint's gripper
convention. The Rust boundary receives normalized state and returns normalized
actions. Direct native callers must provide the same preprocessing. Optional
`noise=` is a finite float32 array of shape `[32,7]` for deterministic comparisons.

## Execution ownership

`config.rs` and `load.rs` validate the supported architecture and resolve assets.
`weights/` maps and packs checkpoint tensors, including fixed time and position
tables. `model.rs`, `text.rs`, `vae.rs` and `dit.rs` compose the computation.
`model_runner.rs` owns request validation, input/RNG binding, workspace and CUDA
Graph preparation. There are no imports from another model family.

Raw RGB bytes are uploaded unchanged; camera concatenation, layout conversion
and pixel normalization execute on the device. Encoded conditioning and its
cross-attention projections are computed once per request and reused across
all ten action steps. `Model` reports a conservative workspace requirement for
the fixed profile, including the monotonic graph arena's intermediate tensors.
Native operator executions own their provider scratch separately.

The native runner supports `Eager`, `RequireGraph` and `PreferGraph` preparation.
A preferred-graph fallback is exposed through `PreparationStatus`. The default
inference path requires the complete model graph. Prepared outputs use stable
storage and may be overwritten by the next run; copy an output before reusing
the same prepared object when a snapshot is needed.

## Validation

The opt-in checkpoint tests in `fastwam/{vae,text,model_runner}.rs` require local
assets and independent reference captures. They cover encoder boundaries,
canonical action outputs, complete graph parity, changed image/text/state/noise,
keyed RNG, invalid specifications and output ownership. The CUDA backend tests
cover additive/F32 attention, rounded arithmetic, pairwise rotary embeddings,
image transforms, alias/alignment validation and capture lifetimes.

Qualification on NVIDIA Thor (SM110, CUDA 13.2) uses the pinned checkpoint and
LeRobot source above, with PyTorch 2.11.0, Transformers 5.5.4 and Diffusers 0.39.0.
Twelve deterministic synthetic cases vary both cameras, prompt, state and noise.
For each case, both relative L2 and maximum absolute error against independent
official FP32 outputs must be no worse than official BF16 error, with a fixed
`1e-6` comparison allowance. Every passing result below also meets the gate
without that allowance.

| Path | Cases passed | Relative L2 range | Maximum absolute error range |
| --- | ---: | ---: | ---: |
| Native graph and default AutoPolicy | 12/12 | 0.002958–0.007074 | 0.003039–0.009395 |
| AutoPolicy with cold tuning | 12/12 | 0.002867–0.007082 | 0.003094–0.008858 |
| AutoPolicy with persisted recipes in a new process | 12/12 | 0.002867–0.007082 | 0.003094–0.008858 |

Canonical eager and whole-model graph outputs are exact. Input updates, keyed RNG,
invalid-request rejection and retained output/resource lifetimes pass. The new
process reuses all 45 recipe files without changing their bytes or timestamps;
canonical actions are exact across cold/warm processes and all twelve error
measurements agree. UMT5 final output matches the reference exactly; the VAE
boundary passes its independent precision check. The backend regression suite
passes 111 library tests and eight public integration tests; two unrelated
opt-in backend tests remain ignored.

The native UMT5 tokenizer matches 594 independent official token sequences,
including Unicode whitespace and special-token boundaries. To run its maintained
asset-backed regression, set `APXINF_FASTWAM_TOKENIZER` to the pinned
`tokenizer.json` and run `python -m pytest python/apxinf/tests/test_fastwam_policy.py`.
The policy restores the reference T5 pre-tokenization configuration in memory;
it does not require Transformers or Python tokenizers at inference time.

These checks establish numerical and runtime behavior for the declared profile.
Follow the [porting workflow](porting-workflow.md) when qualifying another profile
or device.

### LIBERO-Spatial rollout pilot

A separate closed-loop pilot on the same Thor ran all ten LIBERO-Spatial tasks,
using bundled initial state 0 for each task and environment seed 0:

| Runtime | Successful episodes | Pilot success rate |
| --- | ---: | ---: |
| Official LeRobot BF16 | 10/10 | 100% |
| Native default AutoPolicy | 10/10 | 100% |

Both paths used the same checkpoint, fixed action-noise seed 42, ten inference
steps, and the pinned LeRobot observation processor. The simulator rendered both
cameras directly at 224 × 224, with relative control at 20 Hz, hard resets, ten
settling steps, and a 300-action limit. Each prediction produced 32 actions; the
first ten were executed before replanning. MuJoCo 2.3.7 and robosuite 1.4.1 ran
in a separate simulator process. Initial-state files and starting observation
hashes matched exactly across both runs. Success was the simulator's
`check_success()` result. Native completion took 75–125 steps; official BF16 took
76–144 steps. The native extension was unchanged from numerical qualification.

This is a ten-episode pilot with one fixed initial state per task, not a full
LIBERO benchmark or evidence of statistical equivalence. Object, Goal and Long
suites, additional initial states, and real-robot performance remain untested.
It does not reproduce published FP32/256-pixel multi-suite benchmark conditions.

## Measured performance and remaining work

On the same Thor, twenty alternating official/native pairs measured host RGB,
state and task inputs through host action outputs. Both models remained loaded,
with three warmup pairs after first-call checks. Every official output matched
its frozen BF16 oracle exactly; every native output repeated exactly and passed
the FP32 error gate. The native extension used the CUDA build with Rust
model/core/loader packages at optimization level 2.

| Runtime | Median | p95 |
| --- | ---: | ---: |
| Official BF16 | 1091.60 ms | 1152.14 ms |
| Native default AutoPolicy | 987.10 ms | 1007.05 ms |

The median paired speedup was 1.11×. Another GPU workload was present throughout
the measured samples: these are shared-device results, not isolated latency or
a production SLO. The official call includes seeded noise generation and takes
normalized state; native receives the same supplied noise and additionally
normalizes state and unnormalizes actions. Both include all model computation
and host output transfer.

Functional acceptance passes; optimization status is **best effort with
performance debt**. Independent native processes took about 69 seconds to load
and 10.85 seconds for the first default/cached call, including graph preparation.
Cold tuning raised that first call to 14.00 seconds. With both runtimes resident
in the paired experiment, the first native call took 236.05 seconds; its cause
has not been isolated. These setup times are excluded from the warm table.

The prepared native process reported approximately 36 GiB of CUDA memory in
standalone snapshots. Its conservative graph arena reserves 12.14 GiB; neither
number is a measured allocation peak. Further work should profile first-call
resource/graph preparation and reuse intermediates between action steps to
reduce the monotonic arena. The required encoders, ten-step loop and graph
replay already execute on the GPU without intermediate host round trips.

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
Absent `num_inference_steps` defaults to 10; any explicit value must be 10.
Absent/null `sigma_shift` uses the scheduler shift of 5; an explicit override
must equal 5. Other values are rejected before device weights are loaded.
`weights/` maps and packs checkpoint tensors, including time-projection weights
and fixed position tables. `model/mod.rs` owns model construction, modality connections and the
flow schedule. `model/blocks/bf16.rs` owns UMT5, VAE and DiT composition,
projection/bias rounding and fixed time-modulation preparation. Its BF16
implementation includes the declared F32 residual and action-projection paths.
Temporary time-projection weights are released after modulation preparation.
`backend.rs` only imports safe CUDA resources/operators and adapts host uploads;
weights never call model forward methods.
The [family boundary guard](../scripts/check_fastwam_module_boundaries.py)
checks these owners recursively; it does not replace numerical validation.
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

The opt-in checkpoint tests in `cuda_fastwam/model/blocks/bf16.rs` and
`cuda_fastwam/model_runner/runner.rs` require local
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

### FP32 attention optimization (2026-10-26)

A subsequent Thor measurement identified the default FP32 attention provider as
24.4% of traced kernel time: 660 calls and approximately 104 ms per request.
The CUDA backend now supplies a pedantic FP32 cuBLAS candidate for dense MHA
with at most 1024 keys. QK and PV operate directly on interleaved head strides;
masking and softmax share one kernel. TF32 conversion is disabled. The model's
precision boundaries, public interfaces and ten-step schedule are unchanged.
Resource limits, graph preparation and exact recipe selection remain native
operator responsibilities; GQA and longer keys retain the existing candidates.

Twenty alternating baseline/optimized pairs measured default AutoPolicy with
autotuning disabled, host RGB/state/task inputs, supplied canonical noise and
host action outputs. Both models remained resident. Three joint warmup pairs
followed loading and graph preparation; pair order alternated, and GPU process
checks before, during and after timing detected only the two benchmark workers.

| Native implementation | Median | p95 |
| --- | ---: | ---: |
| Before FP32 attention optimization | 390.64 ms | 395.09 ms |
| With FP32 cuBLAS attention | 321.85 ms | 322.34 ms |

Median paired speedup is 1.214×, with a 17.6% reduction in median latency.
These paired results are separate from the earlier shared-device comparison
against official LeRobot. Loading and first-call setup remain excluded.

Native eager/graph lifecycle, default AutoPolicy, cold tuning and a separate
warm-cache process each pass all twelve frozen FP32 noninferiority cases without
using the comparison allowance. Eager, graph and public default outputs agree
exactly. Cold/warm actions and all error metrics agree exactly, and all 45 recipe
files retain identical contents and modification times. Backend validation passes
112 library tests and eight public integration tests; two unrelated opt-in tests
remain ignored. This optimization did not rerun the LIBERO rollout pilot.

Roofline analysis uses measured streaming-copy bandwidth (231.38 GB/s), pedantic
SGEMM throughput (6.18 TFLOP/s), actual attention shapes and Nsight Systems
latencies. Arithmetic intensity uses ideal minimum Q/K/V/output traffic, omitting
scratch and repeated reads. NCU counter collection was denied by device permissions,
so these are analytical roofline estimates, not measured DRAM utilization or
stall attribution. Shared-score-memory, split-value-reduction and static-unrolling
experiments did not improve all representative shapes and were not adopted.

### Fused attention and conversion reuse (2026-10-27)

The next iteration adds a short dense F32 attention candidate that fuses QK,
softmax and PV. Four queries reuse vectorized K/V loads, eight-lane groups
reduce QK and four warps reduce PV; scores stay in shared memory. It needs no
global score workspace. The provider checks B1, Q16–128, K1–160, D128, equal
heads, no mask and 16-byte binding alignment; other cases retain existing
providers. Products and accumulators remain F32, with a different reduction
order. The model schedule and checkpoint precision boundaries are unchanged.

BF16-to-F32 conversion now uses aligned four-element vectors with scalar
fallback for tails and unaligned views. Each request also promotes its rounded
cross-attention context K/V once, then reuses those exact F32 values across the
ten action steps. This is not a cross-request text cache: changed observations,
state and prompts still recompute their context on every graph replay.

Two separate twenty-pair, alternating-order experiments on Thor use the same
default AutoPolicy boundary and GPU-exclusivity checks described above:

| Experiment | Previous cuBLAS median / p95 | Candidate median / p95 |
| --- | ---: | ---: |
| Fused attention only | 322.16 / 334.08 ms | 307.88 / 318.82 ms |
| Fused attention + vector conversion + context promotion reuse | 322.42 / 322.63 ms | 294.05 / 294.62 ms |

The combined change reduces median latency by 8.8% (1.096×); do not combine
ratios from different experiments. Under separate Nsight Systems tracing,
F32 attention kernel time falls from 35.49 to 23.79 ms/request and BF16-to-F32
conversion from 24.11 to 5.79 ms/request. There are 540 fewer conversions and
1,860 fewer total kernel nodes per request. Trace timings are diagnostic, not
the unprofiled end-to-end benchmark.

All twelve original FP32 noninferiority cases pass for default, native
eager/graph, cold tuning and a separate warm-cache process without using the
comparison allowance. The conversion/reuse changes preserve the fused-only
canonical actions and all twelve error metrics exactly. Cold/warm actions,
error metrics and recipe contents/mtimes agree. Backend validation passes
114 library tests, an additional focused attention-alignment fallback test and
eight public integration tests; two unrelated opt-in tests remain ignored.
Compute Sanitizer memcheck and synccheck pass the fused kernel's boundary
shapes. No additional LIBERO rollouts were performed for this iteration.

The roofline interpretation now distinguishes cache-resident and streaming
workloads. Thor reports 32 MiB L2; empirical copy throughput is 933–1090 GB/s
for 4–16 MiB working sets, versus approximately 231 GB/s for streaming data.
Warm attention microbenchmarks reach 29–33% of the measured 6.18 TFLOP/s F32
compute ceiling; actual full-model attention reaches about 23–30%. The larger
32–64% fractions against ideal minimum DRAM traffic are not measured hardware
utilization. Actual cache/DRAM traffic and stalls remain unknown without NCU
counters. Larger tiles, register/shared-memory variants and pedantic cuBLASLt
heuristics were evaluated; none beat the selected candidate across this work.

### Source ownership and capture boundary

The native implementation is `crates/apxinf-model/src/cuda_fastwam/`.
`model/` connects modalities and owns the fixed flow schedule;
`model/blocks/bf16.rs` composes DiT, UMT5 and VAE operators. Private inline modules hold the subnetwork implementations; only `Bf16Blocks`
is exposed to the model. It owns physical conditioning caches, dtype boundaries
and workspace requirements. `weights/host.rs` owns
checkpoint storage and source validation; `weights/bf16/` owns device weight
representations and upload/packing, including explicitly F32 tensors used by
this implementation. Text and VAE computation objects retain their weight trees.
`model_runner/runner.rs` validates requests and owns the plan cache;
`model_runner/prepare.rs` owns stable buffers, preparation, capture and replay.
`backend.rs` remains a safe CUDA import/transfer seam.

The captured graph covers the complete model forward from already bound device
inputs: RGB canonicalization, VAE, UMT5, video conditioning, all ten action steps
and the device output copy. Request validation, CPU token/mask construction,
state/provided-noise conversion and uploads are outside capture. Generated noise
uses a GPU RNG launch before replay. Python resize/tokenization, policy
normalization and action postprocessing, and host output download are also
outside this graph. Whole-model capture does not mean whole-request capture.

The native loader accepts `cuda_fastwam`; existing `fastwam` and `fastwam-cuda`
selection spellings remain compatible. The Python policy name stays `fastwam`.

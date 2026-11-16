# Cosmos3-Edge DROID policy

This is the native CUDA port of **Cosmos3-Edge-Policy-DROID**, distinct from
[Cosmos Policy LIBERO](cosmos-policy.md) introduced by PR154. It builds on that
PR's safe operators, prepared execution sessions, Conv3d lifetime/cache fixes,
wide-head attention and normalization APIs. It does not import the Cosmos
Policy model, weights or sampler.

## Supported profile

| Contract | Supported value |
| --- | --- |
| Checkpoint | [nvidia/Cosmos3-Edge-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID), revision `0110e64ca982589f1a3d856cbdd7d465dccb79bc` |
| Reference architecture | [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM), export contract revision `a16b2d162bb12245f98ebef4227a1d7fe9c67757` |
| Prepared format | `droid-policy-v1`, BF16, batch 1, DROID domain 8 |
| Observation | One canonical NHWC uint8 RGB image, **544×736**, no proprioception |
| Language | One cached instruction, exactly 126 chat-templated tokens |
| Model | 28 UND/GEN layers, width 2048, MLP 9216, 16 query / 8 KV heads, head dimension 128 |
| Denoising | Four UniPC flow-matching steps, shift 5, five latent frames |
| Joint noise | Standard-normal FP32 `[1,376384]`: CTHW video `[48,5,34,46]`, followed by action `[16,64]` |
| Output | FP32 `[16,10]`, DROID reference gripper convention (`1 - value` in the final dimension) |

Configuration admission checks every fixed geometry, numerical constant and
revision. LeRobot LIBERO checkpoints, arbitrary token lengths, FP8, FP16 and
other sampling schedules are unsupported. Legacy prepared files that declare
17 VAE pixel frames remain readable by Rust; newly converted assets declare
one frame. This compatibility does not alter the five-frame GEN latent.

## Loading and inference

Offline conversion depends on Torch, safetensors and Transformers. Inference
uses the native runner and NumPy; it does not import the reference engine.

```bash
python -m apxinf.checkpoints.cosmos3_edge --assets /path/to/hf-assets --output /path/to/prepared
```

```python
import numpy as np
from apxinf import AutoPolicy

policy = AutoPolicy.from_pretrained(
    "/path/to/prepared", device="cuda:0", model_variant="bf16",
    autotune=True, tactics="/path/to/recipes",
)
result = policy.infer({
    "image": canonical_rgb,  # uint8 [544,736,3], correct camera orientation
    "prompt": "Pick up the banana and place it in the bowl.",
}, noise=np.random.default_rng(0).standard_normal((1, 376384)).astype(np.float32))
actions = result["actions"]
```

The caller owns camera choice/orientation and resizing to the declared RGB
profile. Historical fixtures use the reference STB linear resize; another
resize implementation changes the input and requires its own validation.
The policy passes the fixed grid `[[1,544,736]]` through the existing rectangular
RGB binding. Rust accepts only that grid or its fixed-profile omission.

The converter restricts the prompt table to one 126-token instruction, selects
domain-8 action matrices, omits the reasoner head and VAE decoder, and gathers
the cached embeddings once. The Python loader verifies manifest hashes for
weights, VAE, text and prompt order before native loading. Changing a prompt
requires regenerating its embeddings, not replacing `prompts.json` alone.

## Execution and optimization ownership

`cuda_cosmos3_edge/weights` validates and packs UND/GEN/VAE tensors once.
`model/blocks/bf16.rs` owns tensor layouts, VAE, per-request UND prefill and
joint GEN denoising. `model_runner` owns request validation, bounded implicit
plan, same-stream uploads, RNG, workspace and the whole-model Graph. It honors
`eager`, `prefer_graph` and `require_graph`; a required capture failure is an
error. Returned actions own independent device storage and survive replay,
plan eviction and runner destruction.

Between RGB/noise upload and action download, computation stays on the GPU.
RGB normalization and `(channel,x,y)` 2×2 packing use a safe backend operator.
Only the singleton observation is VAE-encoded: causal frame zero does not depend
on the unused 16-frame rollout encoding. All 1955 video and 16 action tokens
remain in GEN joint attention. UND runs once per request and stores each layer's
KV for all four GEN steps.

The additional model-neutral backend work is fractional mRoPE (integer callers
retain their original path), d640 wide attention, exact-shape joint GQA FMHA,
vectorized volume lowering for 160/320/640 channels, warp channel-L2/SiLU,
packed8 residual-add/RMSNorm, and BF16 GEMM with squared-ReLU epilogue.
The epilogue preserves projection-to-BF16 rounding before the activation;
a staged baseline serves unsupported configurations. BF16 Lt split-K candidates
are available to tuning for the exact down-projection geometry. Experimental
FP8 and unsuccessful standalone persistent GEMM candidates are excluded.

## Historical best operator measurements

Thor SM110, CUDA 13.2, batch 1 and four steps. These measurements belong to the
Edge optimization work before this PR; they are not fresh PR-head measurements.
Kernel traces originally called “in-graph” were subsequently identified as
preparation/eager session traces. Their kernel durations are labelled here;
only the endpoint rows are production Graph measurements.

| Operator | Geometry / request count | Accepted implementation | Measured duration |
| --- | --- | --- | ---: |
| GEN MLP up + ReLU² | M1971/N9216/K2048, 112 | CUTLASS config 4, 256×192×64, two SM, cluster 2×1 | 468.3 µs median / 476.8 µs mean in later eager/session trace; earlier aggregate 52.7 ms |
| GEN MLP down | M1971/N2048/K9216, 112 | Lt rank 0, ID 66, nvjet 256×200 | ~79.5 ms per request, ~710 µs per node |
| GEN residual + RMSNorm | rows1971/cols2048, 224 | packed8, unchanged FP32 reduction tree | 85.5 µs per call / 19.2 ms eager/session aggregate |
| Same norm, standalone | rows1971/cols2048 | packed8 versus scalar | 61.6 versus 158.3 µs |
| Joint attention | Q1971/KV2097, GQA16/8, d128, 112 | FMHA | ~298.8 µs / 33.46 ms trace aggregate |
| GEN Q / O projection | M1971/N2048/K2048, 112 each | native recipes | 9.98 / 10.17 ms trace aggregate |
| GEN K / V projection | M1971/N1024/K2048, 112 each | native recipes | 5.65 / 5.65 ms trace aggregate |
| VAE singleton / UND | one observation / 126 tokens | direct Conv3d / cached prefill | ~22.0 / 14.6 ms eager phases |
| Full forward Graph | VAE + UND + four GEN steps | fused ReLU² + packed8 norm | **268.52 ms** mean of two paired run medians, versus 294.07 ms with both fusions disabled |

The fusion A/B improves the Graph endpoint by **25.6 ms (8.7%)** under the same
thermal conditions. It excludes Python preprocessing, request uploads, action
publication and preparation. Individual operator savings cannot be added to
construct a hypothetical observation-to-action latency.

The fastest standalone DOWN candidate measured **0.573536 ms warm /
0.611168 ms cold**, but its first full-model A/B regressed from 268.55 to
271.13 ms. Further static-persistent DOWN configurations also regressed actual
Graph node sums. The latest UP experiment lowered 112 UP calls from 56.04 to
54.86 ms but failed overall stationarity/displacement screens. Neither was
promoted. The **200 ms whole-policy** and **35 ms UP/DOWN aggregate** targets
remain unmet. Lower precision, fewer steps or pruning joint video tokens would
change the model contract and require separate accuracy acceptance.

## Validation and reproducibility

The maintained device gate loads through `AutoModel`, compares all 160 action
values with an independent FP32 reference, checks eager/Graph bitwise parity,
RGB-only/noise-only/combined rebinding, placeholder preparation, repeated and
advanced RNG keys, rejected noise, and retained actions after runner/Graph drop.
Current validation gates are cosine ≥0.9999, max-abs ≤0.03 and RMSE ≤0.01;
they do not claim robot-task success. Run with assets in place and separate
fixture/output/recipe directories:

```bash
APXINF_COSMOS3_ASSETS=/path/to/prepared \
APXINF_COSMOS3_FIXTURE=/path/to/fixture \
APXINF_COSMOS3_OUTPUT_DIR=/path/to/devlocal/cosmos-edge/results \
APXINF_COSMOS3_RECIPES=/path/to/devlocal/cosmos-edge/recipes \
APXINF_COSMOS3_AUTOTUNE=1 \
cargo test --release --features cuda -p apxinf-model --lib \
  cosmos3_edge_reference_and_preparation_lifetime -- --ignored --nocapture --test-threads=1
```

Fixtures contain canonical `rgb.u8`, packed `noise.f32` and independent
`actions.f32`. Offline reference scripts, source/asset hashes, captures, rejected
experiments and complete logs stay in Git-ignored `devlocal/`. Historical native
outputs against FP32 reference achieved cosine 0.9999938–0.9999950,
max-abs 0.01169–0.01616 and RMSE 0.00362–0.00411 on obs/nav0/dark inputs.
TensorRT FP16 comparisons are a different reference and must not replace these
FP32 accuracy measurements.

No closed-loop DROID or LIBERO success rate, general prompt encoding, other
checkpoint, GPU or precision is claimed.

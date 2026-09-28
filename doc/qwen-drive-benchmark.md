# Qwen-Drive BF16 benchmark and validation

## Results

Jetson AGX Thor SM110 (20 SMs), CUDA 13.2, BF16 `planner-sft`, direct
planning, batch one and ten flow steps. Request latency includes policy
preprocessing, native execution and the materialized `[50, 3]` host trajectory;
images are already decoded and model loading is excluded. This is an L2 request
measurement, whereas the PI0.5 README table measures CUDA Graph replay.

| Measurement | Result |
| --- | ---: |
| Request P50 / P95 | 498.78 / 508.42 ms |
| Request throughput | 2.00 Hz |
| Fixed NAVSIM scenes | 242 |
| PDM score, 0–100 | 85.6786 |
| Trajectories identical to accepted padded implementation | 242 / 242 |

The fixed subset's official-code SFT direct reference scores **84.6974**.
The accepted unpadded optimized version scores **85.6723**. Padding changes 125
of its trajectories and leaves 117 exact, without changing any discrete PDM
component. The current implementation matches the accepted padded version
exactly, including every per-scene score.

These are fixed-subset results, not the published full-navtest scores. The
[official release](https://github.com/QwenLM/Qwen-Drive-1.0#planning) reports SFT/RL
88.2/90.7, or 89.3/91.4 with best-of-six selection. Checkpoint, scene selection,
image profile and sampling protocol must match before comparing scores. The
local gain over official code includes three binary metric flips and does not
establish a systematic quality improvement.

### Recorded timing conditions

Measured on 2026-09-28 with CPU min=max=2.601 GHz, GPU min=max=1.575 GHz,
EMC min=max=4.266 GHz, MAXN and fan PWM 255 with automatic fan control disabled.
The GPU lock covered the entire run; configuration was restored afterwards.
The table uses the pooled 60 measured requests from both candidate arms.

| ABBA arm | Request P50 | Request P95 |
| --- | ---: | ---: |
| Accepted padded reference, first | 497.00 ms | 503.95 ms |
| Maintained candidate, first | 499.25 ms | 508.68 ms |
| Maintained candidate, second | 497.58 ms | 501.43 ms |
| Accepted padded reference, last | 499.83 ms | 508.68 ms |

Reference pooled P50/P95 is 498.31/507.19 ms. The maintained implementation
preserves the accepted performance within this run's variation. The historical
482.98 ms result is not substituted for this fresh measurement.

## Build and load

Use Linux AArch64, CUDA 13.2 and the native SM110 operator artifacts. The offline
exporter requires CuTe DSL 4.7.0 and pinned upstream sources; normal engine builds
consume the generated artifacts. Follow the
[AOT export instructions](../crates/apxinf-cuda/aot/README.md), then build:

```sh
APXINF_CUDA_ARCH=sm_110 \
APXINF_CUDA_AOT_MANIFEST=/path/to/artifacts/manifest.json \
  maturin build --release --features cuda,extension-module \
  --compatibility linux -m crates/apxinf-py/Cargo.toml
```

Install the resulting wheel and `python/apxinf` in the test environment. The
checkpoint, planner, tokenizer and test fixtures are external assets.

```python
from pathlib import Path
from apxinf import AutoPolicy

model = Path("/path/to/Qwen-Drive-1.0-4B")
policy = AutoPolicy.from_pretrained(
    model, model_type="qwen_drive", model_variant="bf16",
    planner=model / "planner-sft", mode="direct_planning", num_steps=10,
    tactics="/path/to/empty-vendor.json",
)
```

Use the same explicit tactic store for both comparison arms. The empty vendor
store in this acceptance exercises the operator defaults. `reasoning_planning`
is also supported, but is outside this direct-planning latency result.
VQA and BEV perception are not exposed by this planning runtime.

## Latency procedure

1. Check the GPU process list, machine load and shared GPU lock. Do not benchmark
   alongside other GPU work or CPU compilation/scoring.
2. Save clock configuration with `sudo jetson_clocks --store <file>` and set
   `sudo jetson_clocks --fan`. Read back CPU, GPU, EMC and fan settings; restore
   the saved configuration after testing.
3. Use the same checkpoint, decoded frames, target sizes, prompt and initial
   noise in both arms. The primary fixture has three cameras and four frames
   per camera, 3385 real prompt tokens padded to 3387.
4. Warm up ten calls; measure thirty calls per arm. Alternate reference,
   candidate, candidate, reference. Keep all arms and report P50/P95 plus spread.

For the pinned fixture format (`scenes.json`, image `.npy` files and
`initial-noise.npy`), the core loop is:

```python
import json
import time
import numpy as np

fixtures = Path("/path/to/public-inputs")
scene = json.loads((fixtures / "scenes.json").read_text())[0]
observation = dict(scene)
observation["views"] = {
    camera: [
        {"image": np.load(fixtures / frame["image"]),
         "target_size": frame["target_size"]}
        for frame in frames
    ]
    for camera, frames in scene["views"].items()
}
noise = np.load(fixtures / "initial-noise.npy")
for _ in range(10):
    policy.infer(observation, noise=noise)
samples, actions = [], []
for _ in range(30):
    start = time.perf_counter()
    result = policy.infer(observation, noise=noise)
    samples.append((time.perf_counter() - start) * 1000)
    actions.append(np.asarray(result["actions"]).copy())
assert all(np.array_equal(actions[0], x) for x in actions)
ordered = sorted(samples)
print({"p50_ms": ordered[int(.50 * (len(ordered) - 1))],
       "p95_ms": ordered[int(.95 * (len(ordered) - 1))]})
policy.close()
```

Capture native-library, model, tokenizer, artifact-manifest, tactic-store and
input hashes with raw timings and clock readbacks. Acceptance evidence stays
under `devlocal/qwen-drive-performance/`; it is not part of the source distribution.

## Accuracy procedure

Use the frozen NAVSIM v1.1 242-scene subset, the same metric cache/maps, ten
flow steps and one supplied initial-noise tensor per scene. The official arm
uses `planner-sft`, BF16, `direct_planning`, one sample and seed 42; this seed's
noise was checked against the native supplied tensor. Both arms use the released
benchmark image target sizes. The subset without those targets is a different
image profile and must not be substituted silently.

| Pinned input | SHA-256 |
| --- | --- |
| `navtest-interp-242.jsonl` | `bf55f27e931664f9c81e870ab375cf7a0d05949b8877e41964f4f6abd5955ce5` |
| Same scenes with benchmark target sizes | `1d86691ccbfa8df57eb28d2de204fd1c75d718dfc5b043d9d84461e5feb07930` |
| Metric-cache `metadata/cache.csv` | `d7d80f11e0fdfc7856cf1e267593d6a9ce0d2d1c90f499c9201de3453275f99e` |

Require 242 unique tokens and finite `[50, 3]` trajectories. Report position
errors in metres separately from heading in radians, then score the new
predictions with the frozen PDM evaluator. Compare each safety/comfort component
per scene; an unchanged aggregate score alone is insufficient. The
[official evaluation guide](https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/docs/evaluation.md)
describes the NAVSIM trajectory conversion and scoring interfaces.

Additional regression covers four scenes, 1/4/10 steps, changed noise, repeated
calls, invalid masks and logical-length switching. Portable checks:

```sh
PYTHONPATH=python/apxinf python -m pytest \
  python/apxinf/tests/test_qwen_drive_policy.py \
  python/apxinf/tests/test_qwen_drive_padding.py
cargo test -p apxinf-cuda --example build-aot
cargo check -p apxinf-model --features cuda --tests
bash scripts/check_model_family_boundaries.sh
```

## Execution notes

- BF16 has one maintained layer composition. Hardware and tensor contracts
  select the native kernels; no experimental layout switch is required.
- Direct prompts pad to 3387 with trailing zero token IDs and attention masks.
  Action attention excludes padding. Longer prompts are not truncated;
  reasoning uses variable-length execution. Logical length is part of prepared
  plan compatibility, alongside geometry, image-token positions and step count.
- Whole-model preparation owns its input buffers, weights, KV/state and workspace.
  Tactic changes invalidate the plan. Shape-only preparation lacks the required
  geometry; callers use sample-based `prepare_for`.
- Pillow 12.3.0-compatible geometry uses native bicubic resize, normalization
  and patch packing. Other supported inputs retain their preprocessing contract.
- `model_runner/prepare.rs` owns preparation and tests; `model/blocks/bf16.rs`
  owns BF16 computation and direct inputs; `blocks/mod.rs` declares execution
  callbacks. Follow the [model architecture](model-layer-architecture.md).
- This result qualifies direct planning on SM110. It does not establish the
  same latency or downstream accuracy for reasoning, Orin or RTX 4090.

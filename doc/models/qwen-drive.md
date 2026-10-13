# Qwen-Drive (planning VLA)

ApxInf runs Qwen-Drive 1.0-4B BF16 with direct planning behind the shared
public path (`AutoPolicy` → `QwenDrivePolicy` → `apxinf_py.ModelRunner` →
`AutoModel` → `VlaRuntime`). Family layout follows the shared
`model/` / `model_runner/` / `weights/` separation; see
[Model Layer Architecture](../reference/model-layer-architecture.md) and
[Model Organization](../reference/model-organization.md).

- Supported platform: Jetson AGX Thor (SM110), CUDA 13.2, BF16
  `planner-sft`, direct planning.
- End-to-end request latency (Thor2, 2026-09-28): P50 482.60 ms,
  P95 488.78 ms, 2.07 Hz. Full measurement details, NAVSIM PDM evidence
  and the accepted-baseline provenance: [reports/2026-09-28-qwen-drive-benchmark](../reports/2026-09-28-qwen-drive-benchmark.md).
- Procedure (latency and accuracy), acceptance gates, and maintained
  runtime controls are below; they are the authority for re-measuring.

## Build and load

Use Linux AArch64, CUDA 13.2 and the native SM110 operator artifacts. The offline
exporter requires CuTe DSL 4.7.0 and pinned upstream sources; normal engine builds
consume the generated artifacts. Follow the
[AOT export instructions](../../crates/apxinf-cuda/aot/README.md), then build:

```sh
APXINF_PLATFORM=thor \
APXINF_CUDA_AOT_MANIFEST=/path/to/artifacts/manifest.json \
  just build-py
```

The preset (`configs/platforms/thor.env`) supplies `APXINF_CUDA_ARCH=sm_110`.
For `--compatibility linux` wheels invoke `maturin` directly — see
doc/guides/building.md. The checkpoint, planner, tokenizer and test fixtures
are external assets.

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
cargo test --release -p apxinf-cuda --test aot_bundle --test cuda_arch
cargo test --release -p apxinf-model --features cuda --lib capture_rejects_changed_phase_extent_or_count
bash scripts/check_model_family_boundaries.sh
```

### Final review acceptance (2026-09-28)

- Python with the native extension: 33 passed; isolated CPU environment:
  30 passed, 3 native-only checks skipped.
- Cargo AOT bundle/geometry checks: 8 passed; architecture checks: 5 passed.
- GDN policy/operator checks: 9 passed. BF16/PTX chunk-state relative L1 error
  against the FP64 reference: 0.001419 (limit 0.01).
- Convolution boundary, AdaLN fusion and Pillow axis rejection: all passed.
- Owned RGB snapshot/consume-once tests: 2 passed.
- Whole-model graph/eager, input replacement, RNG and lifetime test, plus arena
  layout mismatch rejection: both passed on Thor3.
- Native invalid-mask and alternating-length contracts passed. Four-scene,
  twelve-case outputs and all 242 NAVSIM trajectories remain bit-identical.
- Final release wheel built incrementally and was reused on Thor2 for the
  locked-clock/fan ABBA measurement above.

## Execution notes

- BF16 has one maintained layer composition. Hardware and tensor contracts
  select the native kernels; no experimental layout switch is required.
- Direct prompts of logical length 3383–3387 pad to 3387 with trailing zero
  token IDs and a policy mask. Shorter prompts keep their original length.
  The native boundary validates this mask and derives the real prefix length;
  it does not pass the mask array to the attention kernel. Causal backbone
  attention and per-row norms preserve the real prefix, and action KV assembly
  overwrites the padded suffix before applying its logical key bound.
  Longer prompts are not truncated;
  reasoning uses variable-length execution. Logical length is part of prepared
  plan compatibility, alongside geometry, image-token positions and step count.
- Whole-model preparation owns its input buffers, weights, KV/state and workspace.
  Tactic changes invalidate the plan. Shape-only preparation lacks the required
  geometry; callers use sample-based `prepare_for`.
- Pillow 12.3.0-compatible geometry uses native bicubic resize, normalization
  and patch packing. Other supported inputs retain their preprocessing contract.
- `model_runner/prepare.rs` owns preparation and tests; `model/blocks/bf16.rs`
  owns BF16 computation and direct inputs; `blocks/mod.rs` declares execution
  callbacks. Follow the [model architecture](../reference/model-layer-architecture.md).
- This result qualifies direct planning on SM110. It does not establish the
  same latency or downstream accuracy for reasoning, Orin or RTX 4090.

### Native regression commands

Run GPU tests explicitly on an idle Thor with the same AOT build inputs above;
`--include-ignored` enables the tests marked as requiring CUDA. Serialize GPU
work with the shared device lock.

```sh
cargo test --release -p apxinf-cuda --lib gdn_ -- --include-ignored --test-threads=1
cargo test --release -p apxinf-cuda --lib causal_conv_tiling -- --ignored
cargo test --release -p apxinf-cuda --lib weighted_adaln_residual -- --ignored
cargo test --release -p apxinf-cuda --lib pillow_axis_rejects -- --ignored
cargo test --release -p apxinf-py --features cuda --lib packed_rgb
```

The BF16 chunk-state test uploads BF16 operands and compares the handwritten
PTX path with an independent FP64 recurrence, including the deliberate BF16
state/value rounding. The legacy recurrence tests likewise upload V/T/VT/KCD
with their current BF16 storage contracts. Their former FP32 test buffers
produced invalid comparisons and have been corrected. Value-split coverage
selects distinct legacy launch policies directly, without environment mutation.

The whole-model test uses a checkpoint plus one canonical fixture directory:
`meta.json` contains `grids` and `pixels_shape`; `tokens.bin` is u32,
`attention_mask.bin` is u8, and `pixels.bin`, `conditioning.bin`, `noise.bin`
contain f32 values in native little-endian order. The fixture retains the
physical padded tokens and the real-prefix mask. Generate it from the same
policy inputs as the benchmark, then run:

```sh
APXINF_QWEN_DRIVE_TEST_MODEL=/path/to/Qwen-Drive-1.0-4B \
APXINF_QWEN_DRIVE_TEST_INPUTS=/path/to/canonical-fixture \
  cargo test --release -p apxinf-model --features cuda --lib \
  whole_direct_graph_rebinds_inputs_and_owns_its_lifetime -- --ignored
```

This checks explicit eager versus required-graph execution, changed image/text/
conditioning/noise inputs, generated-noise keys, incompatible requests and
resource lifetime after dropping the runner. `Action` may alias prepared output
storage; transfer/copy its data before the next call when retaining results.

### Maintained runtime controls and ownership

Qwen-Drive has no production `APXINF_QWEN_*` or GDN experiment environment
switches. Direct planning uses its maintained prepared path; reasoning retains
eager execution. Diagnostic branches that changed fusion/capture were removed.
`OnceLock` remains valid for immutable data caches such as the position table.
The two checkpoint/fixture variables above are test inputs; the AOT manifest
variable is a build input. Generic device/tactic infrastructure is shared with
other model families.

Canonical conditioning packs, in order: flattened historical pose (excluding
the final/current entry), flattened historical velocity, flattened historical
acceleration, current velocity and acceleration plus the one-hot command,
and the scalar navigation command. The native boundary validates dimensions
against the checkpoint before any execution.

Tactic-store reuse is scoped to the toolkit/device/kernel identity recorded by
the tuning infrastructure. AOT identity includes artifact and exporter/contract
hashes; use a compatible explicit store, or the empty vendor store used here.
Do not carry an unrelated measured tactic file across toolkit or kernel changes.

The model architecture owns composition and preparation. The
[CUDA adapter boundary](../../crates/apxinf-cuda/adapters/README.md) owns exported
module lifetime, tensor descriptors and launch ABI. NVTX markers use the existing
profiling feature and do not select different numerical implementations.

The `Qwen-Drive contracts` GitHub workflow runs the portable Python checks and
model-family boundary check for relevant PR changes. Native-binding checks skip
when the extension is unavailable. CUDA numerical tests, full model preparation
and NAVSIM quality/performance acceptance require the external Thor assets and
are run explicitly; the hosted CPU workflow does not qualify GPU performance.

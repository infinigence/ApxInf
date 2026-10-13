# GR00T N1.7

ApxInf serves the NVIDIA GR00T N1.7 model core through the same public path as
the other VLA families:

```text
AutoPolicy -> Gr00tPolicy -> apxinf_py.ModelRunner -> AutoModel -> VlaRuntime
```

`Gr00tPolicy` deliberately keeps NVIDIA's official checkpoint processor around
the native model core. The processor owns raw RGB/state/prompt conversion,
normalization, embodiment selection, token and patch-grid construction, and
action decode. Rust owns validated tensor contracts, weights, CUDA execution,
graph lifetime, and the flow schedule.

## Supported deployments

| Platform | Precision |
| --- | --- |
| Jetson AGX Thor (SM110) | BF16, calibrated FP8 |
| Jetson AGX Orin (SM87) | BF16, INT8/W8A8 |

`precision="auto"` selects BF16 on both platforms. Unsupported combinations
fail at load time; they never silently fall back to another precision.

The loader selects the precision exactly once and constructs a concrete
runtime/executor pair:

```text
BF16 -> bf16_runtime -> bf16_executor -> BF16 device weights
FP8  -> fp8_runtime  -> fp8_executor  -> calibrated FP8 device weights
INT8 -> int8_runtime -> int8_executor -> mixed BF16/W8A8 typed device weights
```

The three executors reuse a generic GR00T topology, but Rust monomorphizes that
topology for their distinct weight types. There is no per-linear precision
enum or BF16/FP8/INT8 dispatch in the inference hot path. The INT8 execution
plan explicitly keeps attention, adapters, encoders, and decoders in BF16 and
uses W8A8 only for the validated FFN and eligible fused self-QKV matrices.

## Loading

Load GR00T from one local `model_dir`. The checkpoint already packages the
backbone and action-head weights together; all inference weight tensors come
from that directory. Configuration and official processor resources live beside
them in `model_dir/assets/cosmos/`.

Prepare those resources once from a compatible local Cosmos-Reason2-2B snapshot:

```python
from apxinf import Gr00tPolicy

Gr00tPolicy.prepare_assets(
    "/models/GR00T-N1.7-LIBERO/libero_10",
    "/models/nvidia/Cosmos-Reason2-2B",
)
```

The method copies configuration, tokenizer, image/video processor and
chat-template resources and writes an `apxinf_assets.json` manifest in the
asset directory. It does not copy Cosmos weight shards or download files.
After preparation, normal loads do not require the source snapshot path.

| Location | Required local assets |
| --- | --- |
| `model_dir` | GR00T `config.json` and SafeTensors weights (including the index and referenced shards for a sharded checkpoint); NVIDIA `processor_config.json` and `statistics.json`, with optional `embodiment_id.json`. The processor files may be at the checkpoint root or together in its `processor/` subdirectory. |
| `model_dir/assets/cosmos/` | Matching Cosmos-Reason2-2B `config.json`, Qwen3-VL tokenizer, chat template, image/video processor resources, and the generated `apxinf_assets.json` manifest. No Cosmos inference weights are needed here. |

The Cosmos resources commonly include `tokenizer.json`, `tokenizer_config.json`,
`chat_template.json`, `preprocessor_config.json`, and
`video_preprocessor_config.json`, together with the tokenizer's vocabulary and
merges files. Rust reads the Cosmos architecture config; NVIDIA's processor
loads its own resources from the same directory. Install the compatible
Isaac-GR00T and Transformers environment before loading a policy. Processor
loading uses `local_files_only=True`; the official processor and inference
arithmetic are unchanged.

```python
from apxinf import AutoPolicy

policy = AutoPolicy.from_pretrained(
    "/models/GR00T-N1.7-LIBERO/libero_10",
    precision="bf16",
)

result = policy.infer({
    "observation/image": base_rgb,
    "observation/wrist_image": wrist_rgb,
    "observation/state": {
        "x": eef_position[0:1],
        "y": eef_position[1:2],
        "z": eef_position[2:3],
        "roll": eef_axis_angle[0:1],
        "pitch": eef_axis_angle[1:2],
        "yaw": eef_axis_angle[2:3],
        "gripper": gripper_qpos,  # both mirrored finger joints
    },
    "prompt": "put the moka pot on the stove",
})
actions = result["actions"]
```

Python `AutoPolicy` and Rust `AutoModel` both resolve the prepared directory by
default. Python's `backbone=` and Rust's `LoadOptions.assets["backbone"]` remain
optional local overrides for existing scripts. Missing or invalid default
resources produce an error with preparation instructions. Loading does not
search environment variables, `/tmp/fakehub`, paths from checkpoint metadata,
or the network. The checkpoint's `model_name` may refer to its training/export
machine and is not used to guess a local snapshot.

FP8 requires a calibration identity matching the weights and selected resources,
including GR00T's own model configuration and processor metadata. The versioned
`gr00t-processor-v1` identity applies to both prepared bundles and explicit
`backbone=` snapshots. Profiles created before this metadata binding was added
are rejected for both layouts and require recalibration. An explicit full
Cosmos snapshot still contributes its weight-shard hashes to the identity;
those Cosmos shards are not used as inference weights. Pass the calibration
JSON explicitly with `calibration=` and a device-specific tactics database with
`tactics=` when needed.

The generic command-line example uses the same default resource directory:

```bash
python python/apxinf/examples/autopolicy_infer.py \
  --model-dir /models/GR00T-N1.7-LIBERO/libero_10 --precision bf16
```

For LIBERO, GR00T's official state contract has eight values: XYZ, axis-angle
rotation, and both mirrored gripper joint positions. Do not pass the seven-value
OpenPI state convention, which intentionally keeps only one gripper value.
The shared evaluator also applies NVIDIA's official decoded-gripper conversion
(`0=closed, 1=open` in the dataset to `+1=closed, -1=open` in robosuite) only
for the GR00T in-process backend; existing policy and websocket conventions are
unchanged.

Importing `apxinf` does not import Torch, Transformers, Isaac-GR00T, or the CUDA
binding. Those optional dependencies are loaded lazily by
`AutoPolicy.from_pretrained` when it dispatches to GR00T, or by the equivalent
model-specific `Gr00tPolicy.from_pretrained` entry point.

## FP8 calibration

GR00T uses the project-wide `CalibrationRunner`; it does not define a private
calibration format or scan module names. `Gr00tPolicy.fp8_execution_plan()`
publishes the actual quantized linear consumers, `calibration_plan()` maps each
consumer to its stable `<consumer>.input` capture site, and
`collect_calibration(observation, context)` runs the normal official processor
before collecting BF16 activation maxima. A calibration job therefore follows
the same model-neutral workflow as a new VLA:

```python
from apxinf import CalibrationRunner, Gr00tPolicy

plan = policy.calibration_plan()
profile = CalibrationRunner(
    policy,
    plan,
    checkpoint=Gr00tPolicy.checkpoint_identity(checkpoint),
    data_identity=representative_dataset_identity,
    source_revision=source_revision,
    device={"requested": "cuda:0", "host": host_identity},
    margin=1.1,
    seed=0,
).run(public_observations)
```

The identity hashes the GR00T checkpoint weights, the selected Cosmos resources,
and versioned GR00T metadata. The metadata component uses
`apxinf.gr00t-processor.v1` and binds the checkpoint-root `config.json`, plus
`config.json`, `processor_config.json`, `statistics.json`, and optional
`embodiment_id.json` in the directory actually selected for NVIDIA's processor.
The checkpoint root wins when it has `processor_config.json`; otherwise an
existing `processor/` directory is selected. Unselected copies do not affect
the identity. File content is hashed after following valid metadata symlinks;
broken links and non-files are rejected. Missing files have explicit identity
markers for native callers supplying preprocessed tensors, while the public
Python processor still requires its configuration and statistics files.

Recalibrate using the selected files whenever their content changes or when
upgrading from a profile without this metadata component. Preparing resources
or editing an old profile's `model.checkpoint` does not perform calibration;
there is no fallback to the previous identity for explicit snapshots. The
shared artifact format remains `apxinf.fp8-calibration.v1`. Runtime loading rejects
the wrong model family, checkpoint identity, scale formula, missing or unknown
consumer/site, incomplete provenance, and non-production data labels. The
profile is an external deployment artifact passed through `calibration=`; it is
not copied into the read-only checkpoint and is not committed to this tree.

## Measured performance and accuracy

Best measured latency (Thor BF16/FP8, Orin BF16/W8A8) and the LIBERO-10
accuracy table are published as a dated report:

[reports/2026-09-28-gr00t-n1.7](../reports/2026-09-28-gr00t-n1.7.md)

Summary (Thor, BF16, 2-view P50): 54.2 ms; LIBERO-10 Thor BF16 94.0%.

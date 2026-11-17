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
| MTT M1000 (MP22) | BF16 (preprocessed two-view); FP8/INT8 rejected at load |

Not validated: S5000 (MP31), single-camera input, and sequence lengths other
than the measured 145-token profile.

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

## Performance optimizations

- Optimize Thor BF16/FP8 and Orin BF16/W8A8 inference.
- Fuse quantization, normalization, RoPE, attention, and projection epilogues to reduce kernel launches and intermediate tensors.
- GR00T explicitly selects optimized operations; shared default paths and interfaces remain compatible.

## Best measured performance

| Device | Precision | 1-view P50 | 2-view P50 |
| --- | --- | ---: | ---: |
| Thor | BF16 | 51.834 ms | 54.216 ms |
| Thor | FP8 | 32.557 ms | 35.436 ms |
| Orin | BF16 | 75.778 ms | 84.864 ms |
| Orin | W8A8 | 56.711 ms | 64.924 ms |

## LIBERO-10 task accuracy

| Device | Precision | Episodes | Successes | Success rate |
| --- | --- | ---: | ---: | ---: |
| Thor | BF16 | 100 | 94 | 94.0% |
| Thor | FP8 | 100 | 92 | 92.0% |
| Orin | BF16 | 100 | 93 | 93.0% |
| Orin | W8A8 | 100 | 93 | 93.0% |

Two views, 10 episodes for each of the 10 tasks, a 720-step limit, and eight
executed actions per chunk.

## Fixed-input benchmark

The maintained runner uses tensors produced by the official NVIDIA processor.
Its timing boundary starts at those host tensors and ends after model-core
action D2H; it excludes simulator and raw-observation preprocessing time.

```bash
python scripts/bench_gr00t.py \
  --checkpoint /models/GR00T-N1.7-LIBERO/libero_10 \
  --backbone /models/GR00T-N1.7-LIBERO/libero_10/assets/cosmos \
  --fixture devlocal/gr00t-n1d7/fixtures/libero-two-view \
  --precision bf16 \
  --warmup 10 \
  --iterations 50
```

Use `--calibration` for FP8, `--tactics` for an explicit tactic database, and
`--reference` for a numerical parity gate. When the installed CUDA/cuBLAS or
kernel implementation version does not match a stored database, pass an
explicit new `--tactics` path together with `--autotune`; never rewrite the
bundled database merely to bypass provenance validation. The report records
P50/P95 and the complete normalized model-core output. Generated fixtures,
reference dumps, logs, tactic databases, and result JSON belong under
`devlocal/gr00t-n1d7/` and are not committed.

### MUSA BF16

The MUSA implementation is a self-contained backend-native port under
`crates/apxinf-model/src/musa_gr00t/`; it does not import or modify the CUDA
`gr00t/` implementation. The maintained MUSA scope includes BF16, batch 1, two
camera grids, explicit processor-owned initial noise, and preprocessed tensors
from NVIDIA's official processor. MP31 also supports explicit selective FP8
GEMM through muDNN, with BF16 output and BF16 fallback for shapes outside the
model's benefit policy. MP22 FP8 and MUSA GR00T INT8 remain unsupported.

For S5000, build with `APXINF_MUSA_ARCH=mp_31` and use `--precision fp8
--calibration path/to/calibration.json` with either MUSA benchmark script.
The activation calibration is static; runtime amax is not part of this path.
Weights are quantized at load time, and NN/NT layout is selected in Rust.
`auto` and `bf16` retain the existing execution. The initial policy selects
17 shape classes; six variable-text rules use `128 < M <= 256`. Per-shape
microbenchmark benefit is not an end-to-end speed or model-quality guarantee.

MP31 Norm producers feeding admitted FP8 GEMMs can emit E4M3 directly. They
retain FP32 statistics and the existing BF16 rounding before static-scale
conversion; fused residual producers still emit BF16 hidden states. Vision
block/merger norms, text RMSNorms, VL norms and eligible DiT norms use this
path. Two additional consumer-scoped rules admit direct FP8 Norm inputs at
`action_head.dit_blocks.<even-index>.attention.query` (`41x1536x1536`) and
`action_head.output_projection` (`41x1024x1536`), both using muDNN NT. These
rules do not enable standalone activation quantization: BF16 inputs at these
shapes still fall back to BF16 GEMM. The same-shaped attention output
projections remain BF16. The existing M=512 rule for N=1024/K=1536 is preserved.
Text Q/K/V and gate/up reuse one quantized Norm output only when their
calibrated scales agree.
`APXINF_GR00T_DISABLE_FP8_NORM=1` at model load disables this producer fusion
and sharing for A/B diagnostics; the two fusion-only consumers then return to
BF16, while the original standalone FP8 GEMM shape rules remain unchanged.

Compare BF16 and FP8 using the same newly built executable, fixture, GPU,
execution policy and BF16 tactics. Native source changes invalidate old
tactics by build fingerprint: regenerate them with a BF16 `--autotune` run
before comparing. Report native model-core P50 and `1000/P50_ms` Hz first;
this boundary includes preprocessed-input handling and synchronized action
D2H, but excludes the external processor and action postprocessing. Keep
Python-wheel API timings separate because they include NumPy/PyO3 conversion;
v4 wheel reports provide Rust-internal model timing with matching BF16 host inputs.
Fixed-input benchmark calibration requires the explicitly experimental
`APXINF_GR00T_ALLOW_BENCHMARK_CALIBRATION=1`; do not treat it as representative
production calibration or evidence of task accuracy.

Build the native runner for the target MUSA generation, then use the Python
script only as a launcher and report/parity wrapper:

```bash
APXINF_MUSA_ARCH=mp_22 \
cargo build --release -p apxinf-model --features musa \
  --example gr00t_bench_musa

python scripts/gr00t/bench_apxinf_musa_rust.py \
  --precision bf16 \
  --binary target/release/examples/gr00t_bench_musa \
  --checkpoint path_to/GR00T-N1.7-LIBERO/libero_10 \
  --backbone path_to/Cosmos-Reason2-2B \
  --fixture path_to/libero-two-view \
  --reference path_to/orin_gold/debug.npz \
  --execution require-graph \
  --warmup 10 \
  --iterations 50 \
  --output path_to/gr00t-musa-bf16.json
```

The benchmark defaults to strict prepared Graph execution (`require-graph`) and
requires the runtime to report `musa-graph`; capture failure is an error rather
than a silent eager fallback. Use `--execution eager` only for an explicit
NoGraph comparison, or `--execution prefer-graph` when an observable eager
fallback is acceptable. The timing boundary starts with the preprocessed host
tensors and ends after action D2H, matching the native CUDA fixed-input
benchmark.

Library callers default to `prefer-graph`: a capture failure prints a warning
to stderr and falls back to prepared eager execution. Benchmark entry points
request `require-graph` explicitly. Setting `APXINF_GR00T_PREPARED_ARENA=0`
makes the prepared path eager without a warning. The Python policy path uses
the library default and does not expose an `execution` argument.

Vision attention batching is enabled only for two `[256,256]` grids on MP22;
single-camera and other grid shapes use the segmented path. No single-camera
A/B measurement is included.

MUSA fixed-profile plans use a single-active-plan contract per model instance.
Building a replacement through `infer` or `prepare_for` synchronizes and
invalidates the previous plan before allocating the replacement arena.
Externally held old handles report `PreparationStatus::Invalidated` and reject
`run`; retaining a handle does not retain its Graph or 4 GiB arena.
`clear_prepared` evicts the implicit cached plan only; explicitly held
fixed-profile plans remain valid and runnable.
If replacement construction fails after retirement, no active plan remains;
prepare again or retry inference. Input validation failures before retirement
leave the existing plan intact. This contract covers prepared Graph and Eager
execution. Calibration and legacy runtime-managed inference (including handles
returned by shape-only `prepare`) also synchronize and invalidate the active
fixed-profile plan before entering the executor, so cache replacement cannot
leave a replayable graph referencing retired buffers. This retirement is
conservative even when the request would reuse the same caches. A subsequent
implicit `infer` rebuilds its invalidated plan; explicit fixed-profile handles
must be prepared again. If synchronization fails, execution stops before
retiring the plan or modifying executor caches.

GPU lifecycle regression tests live in the `gr00t_bench_musa` example and are
ignored by default because they load the real checkpoint. Set
`APXINF_TEST_CHECKPOINT`, `APXINF_TEST_BACKBONE`, and `APXINF_TEST_FIXTURE` to the
local assets; `APXINF_TEST_DEVICE` defaults to `0`.
`APXINF_TEST_OTHER_EMBODIMENT` defaults to `0` and must be a valid checkpoint
embodiment different from the fixture's ID. The tests deliberately switch
embodiments to replace executor-owned device weights, retain the obsolete
Graph handle, and assert that replay is rejected. They also compare rebuilt
Graph outputs exactly against the original fixed-input Graph output.

Run these in separate processes because the implicit-arena setting is cached:

```bash
APXINF_GR00T_PREPARED_ARENA=1 cargo test --release -p apxinf-model --features musa \
  --example gr00t_bench_musa gpu_lifecycle_tests::graph_calibration_and_runtime_managed_gpu \
  -- --ignored --exact --nocapture --test-threads=1
APXINF_GR00T_PREPARED_ARENA=0 cargo test --release -p apxinf-model --features musa \
  --example gr00t_bench_musa gpu_lifecycle_tests::graph_direct_eager_gpu \
  -- --ignored --exact --nocapture --test-threads=1
```

The first test covers calibration coverage, explicit-plan retirement, implicit
Graph rebuilding, execution through an old shape-only `prepare` handle, and
replacement with a prepared Eager plan. The second covers direct eager `infer`
with the implicit arena disabled. These are lifecycle checks, not latency
benchmarks or an independent accuracy comparison against Orin gold. Calibration
only collects BF16 activation maxima; it does not quantize or change weights.

Graph input uploads also refresh the executor's token-cache key, matching the
CUDA cache-consistency policy. The key is committed only after input upload
succeeds; an upload failure clears the cache so eager/calibration cannot trust
potentially stale device bytes. The host token vector reuses its allocation
through `clone_from`; replay adds no extra device upload or synchronization.
The existing eager cache-hit optimization remains enabled.

`gpu_lifecycle_tests::graph_token_cache_gpu` specifically exercises Graph(A) →
Graph(B) on the same plan → eager(A), then repeats the transition into
calibration(A). It swaps ordinary text tokens without changing topology or
embodiment, verifies B changes the action output, and compares eager actions
and calibration maxima against clean A baselines. Run it with the same test
environment and `APXINF_GR00T_PREPARED_ARENA=1`:

```bash
cargo test --release -p apxinf-model --features musa --example gr00t_bench_musa \
  gpu_lifecycle_tests::graph_token_cache_gpu \
  -- --ignored --exact --nocapture --test-threads=1
```

#### MUSA Python wheel

The existing Python bindings also support GR00T preprocessed inputs. Build
and install the MUSA wheel in the intended Python environment:

```bash
APXINF_MUSA_ARCH=mp_22 maturin build --release \
  --manifest-path crates/apxinf-py/Cargo.toml --features musa \
  --interpreter python --compatibility linux --out path_to/wheels
python -m pip install path_to/wheels/apxinf_py-0.1.0-cp310-cp310-linux_aarch64.whl

python scripts/gr00t/bench_apxinf_musa_wheel.py \
  --precision bf16 \
  --checkpoint path_to/GR00T-N1.7-LIBERO/libero_10 \
  --backbone path_to/Cosmos-Reason2-2B \
  --fixture path_to/libero-two-view \
  --reference path_to/orin_gold/debug.npz \
  --execution require-graph --warmup 10 --iterations 50 \
  --output path_to/gr00t-musa-wheel.json
```

Use the wheel filename produced for your Python version and architecture.
This script calls `apxinf_py.ModelRunner` in process, without a Rust executable.
It preserves the fixture's BF16 values via raw uint16 NumPy bit patterns.
The v4 report's `latency_ms` times `infer_host_f32` inside Rust, after binding
input conversion and before NumPy output allocation. It includes synchronized
inference and action D2H; `model_hz` is `1000 / latency_ms.p50`.
`api_latency_ms` separately includes NumPy/PyO3 conversion and output wrapping.
Official processor work and action decoding are excluded from both. Historical
v2 wheel reports measured the entire API call instead. Intermediate v3 wheel
reports passed F32 host tensors, still charging model-internal F32-to-BF16
conversion to model time. The v4 script passes exact BF16 fixture bits as uint16
arrays, and the binding builds BF16 host tensors before timing, as the native
runner does. Rebuild the wheel for the additive `_infer_preprocessed_timed`
entry, which always returns `(actions, model_ms)` and accepts BF16 uint16 bits.
The existing `_infer_preprocessed` and `_calibrate_preprocessed` signatures,
return types, and policy-level timing boundaries remain unchanged.
Both MUSA scripts accept `--prompt` to produce a verified, separate fixture
before timing; see [prompt replacement](../scripts/gr00t/README.md#changing-the-prompt).
The report records the installed extension path, all timing samples,
repeatability and optional reference comparisons.

The kernel predicates cover 129/141/145/255/256 tokens; end-to-end validation
is limited to the measured 145-token profile. Shapes outside an exact
optimization predicate remain on the model-neutral GEMM, attention,
normalization, and RoPE fallbacks; 145 is a measured profile, not a required
constant. MUSA-specific assembly images are selected inside the backend
dispatcher and never by model-layer kernel names.

The repository's shared LIBERO evaluator selects the GR00T state adapter while
leaving the existing OpenPI state wire format unchanged for other policies:

```bash
python scripts/eval_libero.py \
  --backend in-process \
  --model-dir /models/GR00T-N1.7-LIBERO/libero_10 \
  --precision bf16 \
  --suite libero_10 \
  --trials-per-task 10 \
  --max-steps 720 \
  --replan-steps 8 \
  --results-jsonl devlocal/gr00t-n1d7/results/libero-bf16.jsonl \
  --summary-json devlocal/gr00t-n1d7/results/libero-bf16-summary.json
```

## Validation contract

Correctness comparison uses the same official processor output, embodiment ID,
and initial noise on both implementations. The native model-core output is
`[40, 132]`; the processor decodes and trims the LIBERO action to `[16, 7]`.
The closed-loop evaluator executes eight actions before replanning.

Minimum release gates are:

- BF16: cosine similarity at least `0.999`, relative L2 at most `0.05`.
- FP8: cosine similarity at least `0.997`, relative L2 at most `0.10`.
- INT8: cosine similarity at least `0.995`, relative L2 at most `0.10`.
- Each precision also requires maximum absolute error at most `0.125`.
- The reference noise must be supplied and match bit-for-bit
  (`--reference-noise`, for example the Orin `inputs.npz`);
  `--allow-missing-noise` is an explicit exemption.
- Every output must be finite and have the exact expected shape.

The current Orin gold parity on the 145-token fixture is cosine
`0.9999962154`, relative L2 `0.0027513`, and maximum absolute error
`0.01953125`.

One-view fixtures are used only for fixed-input numerical accuracy and
performance; they are never used for LIBERO closed-loop task evaluation. The
two-view release campaign uses 10 episodes for each of the 10 LIBERO-10 tasks
and each supported platform/precision pair. GR00T LIBERO rollouts explicitly
use the NVIDIA N1.7 evaluation protocol of 720 maximum simulator steps and 8
executed actions per predicted chunk; the evaluator's 520/5 defaults remain
unchanged for existing PI0.5 and WallOSS callers.

PI0.5 regression acceptance uses a same-host upstream-versus-candidate A/B
with the same checkpoint, evaluator, trials, frozen noise, and calibration.
Unmatched historical aggregate success rates are useful context, but are not
used to attribute a regression to this change.

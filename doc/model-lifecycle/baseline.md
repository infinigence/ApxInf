# Stage 1: baseline protocol and current coverage

Status: source inventory and local checks complete; GPU qualification pending.
This is the maintained baseline protocol for the refactor, not a new performance
result. Generated logs, manifests, hashes and probes live in the active worktree's
ignored `devlocal/model-lifecycle-refactor/` directory.

## Revisions and scope

- Selected upstream baseline: `ee42185f9b851c7f20b221973c95019a2e4fcacb`.
- Baseline merge into `refactor/model-lifecycle`: `c5268b7ff794e0e800b589a9db570c20f9f512ed`.
- Production sources match upstream at that merge; branch-only changes are
  architecture documentation and the previously requested native execution skill.
- Initial migration targets: PI0.5, then WallOSS. GR00T is inventoried read-only
  and deferred due to concurrent development. LLM/VLM runtime migration is later.
- Local host: macOS arm64, no nvidia-smi/CUDA device. Configured SSH aliases do
  not establish an authorized/available benchmark target or checkpoint identity.

## Current interface inventory

| Family | Input and entry | Preparation and retained state | Output / reset |
| --- | --- | --- | --- |
| PI0.5 | Policy input/output pipelines; Model.infer_rgb or patches; VlaRuntime | Explicit prepare allocates/captures or eager-falls-back; infer additionally autotunes a real request; single cached spec plus tuning generation | Device Action; host copy explicit; generic VLA reset absent; binding has sampling reset |
| WallOSS | Policy tokenizer/image processor, optional custom callable; RGB or patches, tokens, action mask | prepare allocates 12 GiB workspace; first run initializes/captures; cache uses public spec, private vision/noise constraints also apply | Device Action; host copy explicit; generic VLA reset absent |
| GR00T | Generic Model and VlaRuntime; typed private input; metadata, state and provided noise | prepare wrapper shares engine; private graph key; graph/workspace owned by engine; fallback supported | CPU Action despite generic device-result comment; deferred migration |
| Llama | LlmTrait with token IDs; shared native generation driver | KV/workspace retained in model; prewarm attempts capture; single capacity-bound decode graph in current code | Device logits; host token events; reset clears KV |
| Qwen3-VL | Same generation interface plus pixels/grid in prefill | KV and mRoPE state; lazy power-of-two bucket capture; default no-op prewarm | Device logits; host token events; reset clears KV and rope delta |

Public VLA InferenceSpec has only token_count and image_layout. It does not fully
represent all private graph compatibility conditions. Execution mode reporting
also differs; do not infer graph readiness from prepare returning successfully.

Source anchors: vla/mod.rs, llm_trait.rs, pi05/vla_runtime.rs,
walloss/bf16_runtime.rs, gr00t/vla_runtime.rs and executor.rs,
llama/general.rs and decode_graph.rs, qwen3vl/general.rs and decode_graph.rs,
and the corresponding Python policies. All paths are in the selected baseline.

## Execution and asset matrix

"Implemented" below is a source claim, not GPU qualification in this task.

| Model / precision | Target and assets | Stage-1 status |
| --- | --- | --- |
| PI0.5 BF16 | Thor SM110 and Orin SM87 in existing regression protocol; exact local checkpoint/path pending | GPU not run |
| PI0.5 static FP8 | Thor SM110; matching checkpoint and calibration required | GPU not run |
| PI0.5 W8A8 | Orin SM87; current per-channel weight/per-row activation quantization | GPU not run; preserve documented accuracy limitation |
| WallOSS BF16 | CUDA path implemented; actual device, checkpoint and raw fixture pending | GPU not run |
| WallOSS dynamic FP8 | CUDA path implemented; validate target kernel support; static calibration rejected | GPU not run |
| WallOSS W8A8 | Loader rejects this precision | Unsupported, not a pending test |
| GR00T BF16/FP8/W8A8 | Existing implementation; specific qualification tuple deferred | Read-only inventory |
| Llama / Qwen3-VL | Local contract tests; checkpoint-backed GPU tuple deferred to Stage 4 | CPU checks only in this stage |

The inherited PI0.5 performance protocol has 16 cells: four device/precision
paths times two views (2/3) times two token lengths (10/21), H=10 and 10 flow
steps. Three views duplicates the wrist fixture and is not a real third LIBERO
camera. H=50, two views and 500 episodes are the separate formal accuracy
protocol. See [PI0.5 regression](../pi05-cuda-regression.md); do not relabel
historical 100-episode runs as new formal qualification.

## Numerical gates already present

These are inherited from pi05_bench.rs, not newly selected tolerances:

| Precision | Eager/graph max_abs | Eager/graph cosine | Reference cosine | Reference relative L2 | Reference max_abs |
| --- | --- | --- | --- | --- | --- |
| BF16 | <= 0.01 | >= 0.999999 | >= 0.999 | <= 0.05 | Not set |
| FP8 | <= 0.001 | >= 0.999999 | >= 0.997 | <= 0.10 | Not set |
| W8A8 | <= 0.01 | >= 0.999999 | >= 0.995 | <= 0.10 | <= 0.125 |

The example validates its zero-input reference schema separately; these gates
do not certify a raw-observation fixture automatically. pi05_auto_smoke requires
identical repeated outputs for an identical RNG key and cosine >= 0.9999 against
its CPU-generated latent reference. It is a BF16 smoke path, not all-precision
or trained-reference acceptance.

For pure structural changes, compare the same dtype/checkpoint/input before and
after, keeping quantization error versus a higher-precision model separate. Exact
output is the preferred unchanged-operation expectation, but is not a newly
approved universal release gate. Record measured baseline repeatability before
agreeing any additional refactor tolerance.

WallOSS whole-model action tolerances and performance/memory budgets were not
found in maintained documentation during this audit. Processor golden tests and
local preprocessing graph tests are not substitutes. These thresholds remain
pending; do not invent them or inherit PI0.5 static-FP8 thresholds blindly.

LLM/VLM historical fixture rules are documented in tests/qwen3vl_reference/README.md:
embedding exact; hidden max_abs < 0.05 and mean_abs < 0.005; final-logit argmax
exact and top-5 overlap; first ten greedy tokens exact. Verify checkpoint/revision
identity before using those dumps for any newly selected model size.

## Performance and memory gates

Reuse the matching baseline and review policy in pi05-cuda-regression.md. Keep
shape, checkpoint, precision, tactics, calibration, device/power state, input
representation and measurement boundary identical. No new percentage-regression
allowance has been approved. Report cold load, cold prepare, first execution,
steady graph replay, input-update-plus-graph and Python end-to-end separately.

Important reproduction gaps:

1. The current pi05_bench real-checkpoint path uses native config (often H=50)
   and rejects architecture overrides. The documented historical performance
   table is H=10. Neither a real H=50 run nor a random H=10 run is automatically
   a reproduction of that historical checkpoint/fixture baseline. Locate the
   exact compatible workload/runner before making a regression claim.
2. The two first-replan NPZ fixtures named by the regression doc are not tracked
   in this checkout. The doc pins their hashes, but actual asset paths remain
   pending. Do not manufacture equivalent-looking inputs and claim equivalence.
3. Python bench L0 uses zero patches, whereas L1 uses RGB preprocessing. L2 invokes
   the full policy and may have its own noise behavior. Their latency differences
   alone do not demonstrate numerical equivalence of the three paths.
4. Current timing tools do not cover every desired cold-prepare and peak-memory
   boundary. Add private instrumentation or a reusable benchmark extension only
   after the actual target/fixture is fixed; avoid claiming unmeasured metrics.

A 4/12 GiB arena reservation is not a measured memory budget for future code.
Record reserved, peak live and replacement-peak memory plus free device memory.
GPU timing requires an otherwise suitable device; concurrent development loads
must be recorded and avoided for formal timing.

## Reproducible existing commands

Run in the selected source checkout. TASK_DIR is a task-specific variable, not a
replacement for HOME or CODEX_HOME. Capture dependency versions and commands with
results. Checkpoints, calibration and fixtures stay in their existing locations.

```bash
TASK_DIR=devlocal/model-lifecycle-refactor
mkdir -p "$TASK_DIR/logs" "$TASK_DIR/results"
bash scripts/check_model_family_boundaries.sh
cargo test -p apxinf-model --offline
python -m pytest python/apxinf/tests -q -p no:cacheprovider
```

GPU commands below are templates, not reported executions. Set PI05_CHECKPOINT,
WALLOSS_CHECKPOINT and any calibration/tactics variables to verified assets.

```bash
# Public PI0.5 BF16 prepare/infer/cache/RNG smoke; needs a real checkpoint.
cargo run --release -p apxinf-model --features cuda \
  --example pi05_auto_smoke -- "$PI05_CHECKPOINT" 21

# Real checkpoint native-shape eager/graph comparison and timing.
# Does not reproduce H10 historical performance unless the assets/config match.
cargo run --release -p apxinf-model --features cuda \
  --example pi05_bench -- "$PI05_CHECKPOINT" \
  --dtype bf16 --token-count 10 --iterations 30

# Public Policy timing with deterministic synthetic images, not a golden fixture.
python scripts/bench_pi05.py --model-dir "$PI05_CHECKPOINT" \
  --precision bf16 --layer l1,l2 --warmup 10 --samples 30 \
  --out "$TASK_DIR/results/pi05-policy-bf16.json"

# WallOSS supports L2 only in this Python benchmark; no lower-level guesswork.
python scripts/bench_pi05.py --model-dir "$WALLOSS_CHECKPOINT" \
  --model-type walloss --precision bf16 --layer l2 --warmup 10 --samples 30 \
  --out "$TASK_DIR/results/walloss-policy-bf16.json"

# Local preprocessing graph parity/update test, not whole WallOSS qualification.
cargo test -p apxinf-model --features cuda \
  walloss::bf16_runtime::tests::native_rgb_preprocess_cuda_graph_replays_and_observes_updates
```

For native FP8 real-checkpoint runs pass the matching calibration and tactic
assets to pi05_bench explicitly. A uniform-scale fallback is not a calibrated
baseline. Raw fixture flags exist (--images-u8, --token-ids-u32le,
--noise-bf16-u16le), but shape/dtype/hash validation is required first.

## Weight consumer inventory

The private source-inventory.json records every exact Rust symbol occurrence,
file and line at the baseline. This is an occurrence inventory, not a full Rust
semantic dependency graph. It found:

| Type group | Consumers and migration consequence |
| --- | --- |
| PI0.5 linear representations | Used by resident weight construction and precision executors; BF16 packing currently reuses a helper from device_weights.rs |
| PI0.5 Static* model trees | Used by runtimes and low-level examples; preserve or deliberately migrate those public example consumers |
| WallOSS DynamicFp8LinearWeights | Model-local device representation feeding its FP8 weight tree; static PI0.5 scales do not match |
| GR00T DeviceLinearWeights | Private contract used by its generic executor and precision implementations; defer changes |
| Backend FP8/W8A8 weight views | Already have multiple model/backend consumers; preserve physical layout/scale contracts |

This supports Block-local precision implementations and cautious shared matrix
storage extraction. It does not justify merging whole model weight trees across
families.

## Normative document reconciliation

The merged upstream doc/model-lifecycle.md and model-layer-architecture.md are the
current shared responsibilities. This directory adds target refactor contracts,
not a claim that public prepare semantics have already changed.

Two points require explicit treatment in implementation reviews:

- Upstream calls checkpoint-fixed RGB-to-tensor canonicalization runtime-owned;
  the target Processor discussion names its semantic origin. Preserve the current
  native seam and GPU execution. Do not move these operations to Python merely
  to satisfy terminology; record their exact formula owner and execution owner.
- The retained model-port skill has a stronger release requirement for native VLA
  graph coverage (whole graph or the specified Vision/Language/Action partition).
  EagerReady/fallback describes runtime behavior and diagnostics, not automatic
  waiver of that port release gate. A future policy change needs an explicit
  reviewed amendment; Stage 1 preserves the existing requirement.

## Execution status

- Source merge, production-diff check, fixture hash inventory and weight symbol
  inventory: completed.
- Model-family boundary check: passed.
- `cargo test -p apxinf-model --offline`: 94 unit tests and 13 integration tests
  passed (107 total); no CUDA feature. This does not execute GPU lifecycle tests.
- Python Policy/Processor suite: 182 passed, 6 skipped, 5 subtests passed.
  Skips: two native-binding tests (apxinf_py absent), one checkpoint-backed
  layering test (APXINF_PI05_MODEL_DIR absent), and three tokenizer tests
  (tokenizer asset absent). These are qualification gaps, not passed cases.
- Reproduction logs: cargo-model-baseline.log, python-policy-baseline.log and
  python-policy-skip-audit.log under the private logs directory. Exact Python
  dependencies and Rust tool versions are in reports/local-test-environment.json.
  Initial missing-pytest attempts are retained separately; successful runs use
  the task-local venv and leave system Python unchanged.
- GPU runs: not performed; target host and checkpoint/fixture paths pending.
- Stage 1 completion: pending GPU evidence and missing acceptance inputs. Do not
  begin the production refactor or mark the matrix qualified on source inspection.

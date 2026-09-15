# Baseline protocol and qualification record

Status: PI0.5 Stage 2 primary Thor qualification is complete: lifecycle, public
entry points, fixed-input parity and performance/resource comparisons passed.
Supplementary Orin matrix validation is in progress.
The selected-main inventory and historical slices are retained below; the final
PI0.5 qualification section is the current candidate record. Generated logs, manifests, hashes and probes live in the active worktree's
ignored `devlocal/model-lifecycle-refactor/` directory.

## Revisions and scope

- Selected upstream baseline: `ee42185f9b851c7f20b221973c95019a2e4fcacb`.
- Baseline merge into `refactor/model-lifecycle`: `c5268b7ff794e0e800b589a9db570c20f9f512ed`.
- Production sources match upstream at that merge; branch-only changes are
  architecture documentation and the previously requested native execution skill.
- Initial migration targets: PI0.5, then WallOSS. GR00T is inventoried read-only
  and deferred due to concurrent development. LLM/VLM runtime migration is later.
- Local host: macOS arm64, no CUDA device. Thor has now been identified and
  selected for parallel validation: NVIDIA Thor, driver 580.00, CUDA 13.0.48.
  Host and asset paths remain in the private Thor manifest.

## Baseline interface inventory

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
| PI0.5 BF16 | Thor SM110 selected and checkpoint hash verified; Orin SM87 remains a later protocol target | Four two-view H10/H50 × T10/T21 parity cases passed; full matrix pending |
| PI0.5 static FP8 | Thor SM110; checkpoint/calibration identity verified | Slice B public Session smoke and H50 T10/T21 parity passed; full matrix pending |
| PI0.5 W8A8 | Orin SM87 deployment target; current per-channel weight/per-row activation quantization | Extra Thor vendor-path smoke/H10 T21 parity passed; Orin not run; preserve documented accuracy limitation |
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
   in this checkout. They have now been located on Thor and both hashes match
   the maintained pins; checkpoint and tokenizer hashes match too. Independent
   reconstruction verifies derived image/token/noise bytes. H10 uses original
   NPZ noise; H50 uses separately seeded Gaussian noise, not an H10 golden.
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
- Thor: immutable baseline `c5268b7` and candidate `1fc151b` both linked native
  CUDA examples and passed real-checkpoint `pi05_auto_smoke`. Public AutoModel
  prepare/run, cached infer and sampling behavior were exercised.
- PI0.5 checkpoint, tokenizer and task04/task08 fixture hashes match maintained
  pins. Exact paths and derivation checks are in the private Thor reports.
- Matching partial operator cache supplied 19 of 23 objects; only four missing
  FA2 translation units were compiled serially in task-local storage. Candidate
  reused all 23 objects with zero further CUDA compilation. Shared caches and
  production operator/build source were unchanged. Private build instrumentation
  reused objects while retaining the actual kernel build-ID computation.
- Both use kernel build ID `kb1-ebd6446d93b8fc684531cb9c73f56b9e` and archive SHA256
  `700eaa20456bb9bba2e5a729689c1003b4a45f30ca62c18692136ab7a8d20ec2`.
- Other GPU workloads were observed; formal uncontended timing is not claimed.
- Stage 1 completion: pending remaining matrix cells and acceptance inputs. Do not
  mark the matrix qualified on source inspection. The user has authorized Stage 2
  candidate work in parallel with Stage 1 Thor validation; retain isolated baseline
  and candidate sources, and require GPU evidence before accepting a migration.

## Thor BF16 slice A numerical result (2026-09-15)

Two views, real checkpoint, ten flow steps; baseline versus candidate in both
execution modes and each variant's eager versus graph output are elementwise
identical in every case:

| Horizon H | Tokens T | Elements per output | max_abs | Relative L2 | Cosine |
| --- | --- | --- | --- | --- | --- |
| 50 | 21 | 1600 | 0 | 0 | 1 |
| 50 | 10 | 1600 | 0 | 0 | 1 |
| 10 | 21 | 320 | 0 | 0 | 1 |
| 10 | 10 | 320 | 0 | 0 | 1 |

The H50 profile changes only num_views to 2; H10 additionally sets chunk_size to
10. Original checkpoint assets are unchanged. Identical private runner
instrumentation only dumps already-read output arrays after graph execution.
Full outputs and statistics are retained at
`devlocal/model-lifecycle-refactor/thor-baseline/results/all-numerical-summary.json`,
with asset, operator and instrumentation manifests in the sibling reports directory.

This qualifies numerical parity for this BF16 structural slice. It does not
qualify reference-policy accuracy, 500 episodes, FP8/W8A8, WallOSS, Orin or the
remaining shape matrix. Timings are descriptive on the shared GPU; arena
capacity/used bytes are not total or peak GPU memory. These gates remain open.

## Thor Session slice B evidence (2026-09-15)

Library candidate `f81d02d`; strengthened public smoke `3884257`; immutable
baseline `c5268b7`. The Session change applies to BF16, static FP8 and W8A8.
All three precisions pass the strengthened public Session smoke. All seven
fixed-input comparisons are complete. W8A8 uses the Thor native cuBLAS INT8
route; this is additional path coverage, not Orin deployment qualification.

| Precision | Two-view fixed-input workloads | Baseline/candidate and eager/graph |
| --- | --- | --- |
| BF16 | H10/H50 × T10/T21 | All four elementwise identical; max_abs=0, relative L2=0 |
| Static FP8 | H50 × T10/T21 | Both elementwise identical; max_abs=0, relative L2=0 |
| W8A8 (Thor vendor path) | H10, T21 | Elementwise identical; max_abs=0, relative L2=0 |

The final BF16/FP8/W8A8 public smoke tests Eager and RequireGraph, actual Ready mode,
invalid-spec rejection followed by reuse of the same plan, populated implicit
cache eviction to unprepared while an explicit plan survives, generated/provided
noise and deterministic RNG streams. It also exercises raw RGB through public
Session eager/graph paths on diagnostic zero images. Fixed-image numerical
fixtures remain separate low-level comparisons, not a claim of closed-loop
policy success or full processor/action-decoding qualification.

The FP8 profile's full runtime checkpoint identity matches the selected weights;
256 calibration sites and scale metadata passed preflight. Representative-sample
quality is inherited from that profile, not newly calibrated in this run.
Task-local profiles reference existing weights/calibration/tactics; originals
remain unchanged. Kernel build ID and archive SHA256 remain the slice A values.
All 23 operator objects were reused, with zero CUDA translation-unit compilation
in this slice. Host Rust code and test runners were relinked.

Local CPU tests: 107 passed. CUDA-feature examples/tests typecheck passes on
macOS, but native linking there is unavailable without CUDA. Thor native Session
unit tests: 3 passed; nested/thread-local/unwind-safe autotune guard test: 1 passed.
The native smoke uses configured inference tactics; it does not establish full
online tuning/invalidation or forced capture-failure recovery coverage.

Private evidence lives under
`devlocal/model-lifecycle-refactor/thor-baseline/session-b/`: public smoke logs,
`reports/validation.md`, source/build manifests and
`results/numerical-summary.json`, `results/w8a8-summary.json` and
`results/final-public-smoke-summary.json`. Source fingerprints distinguish the original
policy smoke from the strengthened populated-cache/raw-RGB smoke. Shared-host
latencies remain descriptive; complete performance/peak-memory and Orin gates
remain open. Stage 2 computation/Blocks work is tracked in migration.md.


## Stage 2 final PI0.5 qualification

Candidate source `7937440`, immutable baseline `c5268b7`; selected upstream source
`ee42185f9b851c7f20b221973c95019a2e4fcacb` already includes GR00T. This record
supersedes the slice-specific PI0.5 pending items above; it does not qualify
unmodified WallOSS/GR00T or the entire Stage 1 matrix.

### Native lifecycle and public entry points

Thor passed the two shared Network order tests, CUDA error/unwind capture cleanup,
three Session policy/cache tests, tuning suppression, and the explicitly enabled
real-checkpoint preparation-failure/store-invalidation test. Capture recovery was
verified by executing the model after the failure and then successfully capturing
again. Plans retained fixed assets after Session was dropped. Replacing a tuning
store with another store at the same generation invalidated both eager and graph
plans. Repeated prepared eager execution in AutoTune mode did not change tactics.

An initial native failure showed CUDA error 901 leaking from canceled capture to
the next eager kernel check. The failure log is retained; the repaired scope ends
and discards capture and clears the known capture last-error state. The enhanced
unit test now checks that slot directly as well as replaying a valid graph.

All three final public smoke runs passed on H50/T21 with two views:

| Precision | Eager/graph max_abs | Raw RGB eager/graph max_abs | Device/CPU RNG max_abs | Cache eviction and retained plan |
| --- | --- | --- | --- | --- |
| BF16 | 0 | 0 | 0 | Passed |
| Static FP8 | 0 | 0 | 0 | Passed |
| W8A8, Thor vendor path | 0 | 0 | 0 | Passed |

These smoke inputs are diagnostic; fixed real-image comparisons are recorded
separately. Local checks: 107 Rust CPU tests, CUDA-feature examples/tests
typecheck, family boundary check and formatting passed. Python suite: 182 passed,
6 skipped, 5 subtests passed; the local skips require a native binding/checkpoint
or tokenizer and are not counted as GPU evidence. After whitespace/path
normalization, 34 moved Block method bodies match their pre-move implementations.

Thor reuses all 23 existing operator objects; no CUDA translation unit was
compiled for this final candidate. Archive SHA256 remains
`700eaa20456bb9bba2e5a729689c1003b4a45f30ca62c18692136ab7a8d20ec2`,
with kernel build ID `kb1-ebd6446d93b8fc684531cb9c73f56b9e`.
Source manifests, native logs, public smoke summary and retained initial failure
live in `devlocal/model-lifecycle-refactor/thor-baseline/session-c/`.


### Final exact-input comparison

All seven candidate runs passed against the pinned baseline. Baseline/candidate
comparison and each source's eager/graph comparison are elementwise identical:

| Precision | Views | Horizon | Tokens | Cases | max_abs / relative L2 |
| --- | --- | --- | --- | --- | --- |
| BF16 | 2 | 10, 50 | 10, 21 | 4 | 0 / 0 |
| Static FP8 | 2 | 50 | 10, 21 | 2 | 0 / 0 |
| W8A8 (Thor) | 2 | 10 | 21 | 1 | 0 / 0 |

The pinned checkpoint, calibration and fixture derivations above are unchanged.
H50 uses independently seeded noise of the correct shape, not a reshaped H10
golden output. Full arrays, hashes and comparisons are retained under
`thor-baseline/session-c/results/`, especially `numerical-summary.json`.
These are structural parity results against ApxInf's baseline; they are not a new
model-quality calibration or a closed-loop policy success-rate evaluation.


### Orin W8A8 compatibility finding

The unmodified baseline and candidate both failed before graph capture at the
patch projection (M=512, N=1152, K=588). This was not an incompatible tactic DB:
the default W8A8 entry selected CUTLASS on SM87 before checking K/N alignment,
so provider validation failed before the existing vendor fallback could run.
The quantized-activation entry already checked those alignment conditions.

A separate host-side dispatch correction applies the same K%16/N%8 eligibility
to the ordinary W8A8 entry. It changes neither CUDA translation units nor tactic
assets/acceptance thresholds. Native regression tests use the public resolver
with unaligned K and N and independently known integer-dot outputs; the older
forced-preference tests bypassed the failing resolver path.

Orin W8A8 structural comparison uses **c5268b7 plus this identical dispatch fix**
as its baseline, not a claim that unmodified c5268b7 can execute that profile.
The original failure is retained as
`thor-baseline/session-c/logs/orin-w8a8-baseline-reproduction.log`.


### Thor performance matrix

All eight H10 / 10-flow-step profiles completed with 10 warmup replays and 30
measured samples per boundary. Baseline/candidate processes were interleaved by
profile. Graph time includes launch plus synchronization; input-plus-graph also
includes already resized RGB/tokens/noise updates and device preprocessing. It
excludes Python decode/resize and checkpoint loading. Three views use base/wrist/
wrist fixture derivation, not a new real camera. Candidate: `7937440`.

| Precision | Views | T | Baseline graph P50 (ms) | Candidate graph P50 (ms) | Graph delta | Input + graph delta |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 | 2 | 10 | 71.944 | 71.991 | +0.07% | +0.05% |
| BF16 | 2 | 21 | 78.990 | 79.048 | +0.07% | +0.25% |
| BF16 | 3 | 10 | 89.595 | 89.035 | -0.63% | +0.10% |
| BF16 | 3 | 21 | 92.632 | 92.928 | +0.32% | +0.37% |
| FP8 | 2 | 10 | 41.411 | 41.281 | -0.31% | +0.22% |
| FP8 | 2 | 21 | 42.118 | 42.054 | -0.15% | -0.12% |
| FP8 | 3 | 10 | 53.556 | 53.494 | -0.11% | +0.16% |
| FP8 | 3 | 21 | 55.004 | 54.687 | -0.58% | -0.27% |

All eight profiles also have exact baseline/candidate outputs and identical
workspace reserved/used bytes. Graph P50 deltas span -0.63% to +0.32%; graph P95
-1.17% to +0.84%. Input-plus-graph P50 spans -0.27% to +0.37%, P95 -0.19% to
+0.39%. These paired observations show no material regression in this run; they
are not an approved deployment percentage budget or a claimed speedup.

The host retained four other resident GPU services. Load/power/temperature and
sampled per-process memory are retained alongside each command. This task's
compilation was paused during timing, and other services were not modified.
Observed idle conditions do not guarantee exclusive ownership of a shared GPU.
Workspace counters are exact arena counters, not total or allocator-peak memory;
external sampling can miss brief peaks. Full distributions and raw samples live
under `thor-baseline/session-c/performance/`.

The final `7aece37` adds only W8A8 default-dispatch eligibility and its test to
`7937440`; BF16/FP8 execution is unchanged. Their above results remain tied to the
actual tested source rather than being relabeled as a different binary.


### First preparation, resource retention and Python Policy

A private identical-source probe uses the legacy public `model.prepare(spec)`
interface on both revisions: fresh process, load, synchronize, first prepare(T10),
first execution, then prepare(T21). One mode drops the first plan; the other keeps
it. Both drop the first Action before that branch so output ownership does not
confound the comparison. Source: baseline `c5268b7`, candidate `7937440`.

| Scenario | First prepare, baseline → candidate (ms) | Next prepare T21 (ms) | Sampled process GPU peak, baseline → candidate (MiB) |
| --- | --- | --- | --- |
| Drop first plan | 220.9 → 226.5 | 201.1 → 206.8 | 15696 → 15728 |
| Retain first plan | 222.1 → 222.2 | 190.4 → 191.0 | 19700 → 19656 |

Outputs are identical. These are single observations per scenario, with a warm
filesystem cache; they do not establish a cold-start latency distribution.
Process peaks use 200 ms sampling. Phase `cudaMemGetInfo` values are device-global
and include unrelated processes/system effects on this integrated GPU. Neither
is an allocator live/reserved peak trace. Exact graph arena counters and ownership
checks provide complementary evidence; no new absolute memory SLO is asserted.

On Thor, six tokenizer tests plus the real-checkpoint AutoPolicy layering test
passed: **7 passed, 0 skipped**. The latter checks RGB-to-normalized actions,
output unnormalization and `calibrate_observation`, using the actual candidate
native binding (`7937440`). The JUnit skip count was checked explicitly because
that test otherwise permits an environment/load failure to become a skip.
The final `7aece37` W8A8 native alignment regression and public smoke also passed
on Thor. Probe sources/manifests, phase values, logs and JUnit live in
`thor-baseline/session-c/extra/`; W8A8 follow-up evidence is recorded separately.

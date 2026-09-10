# GR00T N1.7 validation

This document defines the reproducible acceptance checks for the GR00T N1.7
integration. Performance, numerical parity and closed-loop task accuracy are
separate results and must not be reported as one metric.

## Supported matrix

| Device | Precision | Public name | Internal runtime |
| --- | --- | --- | --- |
| Jetson AGX Thor | BF16 | `bf16` | BF16 |
| Jetson AGX Thor | FP8 | `fp8` | calibrated E4M3 FP8 |
| Jetson AGX Orin | BF16 | `bf16` | BF16 |
| Jetson AGX Orin | INT8 | `int8` | W8A8 |

`W8A8` is an implementation name, not a second user-facing precision.

## 1. Model-Core performance

The canonical runner invokes the Rust `gr00t_fixture_bench` binary. Inputs are
fixed tensors produced by NVIDIA's processor; timing covers the whole ApxInf
Model-Core CUDA Graph through action D2H. It excludes image loading, processor,
tokenizer and simulator time.

```bash
export APXINF_GR00T_CHECKPOINT=/models/GR00T-N1.7-LIBERO/libero_10
export APXINF_GR00T_COSMOS=/models/Cosmos-Reason2-2B
export APXINF_GR00T_FIXTURE=/data/gr00t/libero-two-view-fixture
export APXINF_GR00T_BF16_TACTICS_FILE=/data/gr00t/bf16-tactics.json
# FP8 only:
export APXINF_GR00T_CALIBRATION=/data/gr00t/fp8-calibration.json
export APXINF_GR00T_TACTICS=/data/gr00t/fp8-fallback-tactics.json

scripts/benchmark_gr00t_n1d7_apxinf.sh PRECISION VIEWS OUTPUT_JSON 10 50
```

Examples:

```bash
scripts/benchmark_gr00t_n1d7_apxinf.sh fp8 2 results/thor-fp8-2v.json 10 50
scripts/benchmark_gr00t_n1d7_apxinf.sh bf16 2 results/orin-bf16-2v.json 10 50
scripts/benchmark_gr00t_n1d7_apxinf.sh int8 2 results/orin-int8-2v.json 10 50
```

The device-specific binary, fixture, tactics and calibration paths are selected
from the runner's documented environment variables. A formal report must store
their hashes and the source revision; machine-local defaults alone are not
sufficient provenance.

## 2. Numerical parity

The parity harness uses identical processor tensors, embodiment and initial
noise for the reference and candidate. It checks:

- exact output shape and finite values;
- cosine similarity `>= 0.997`;
- relative L2 `<= 0.10`;
- maximum and mean absolute error as diagnostics.

```bash
python3 scripts/validate_gr00t_n1d7_e2e.py \
  --checkpoint "$GR00T_LIBERO_CHECKPOINT" \
  --backbone "$GR00T_BACKBONE" \
  --source-dir "$ISAAC_GR00T_SOURCE" \
  --image "$BASE_IMAGE" \
  --wrist-image "$WRIST_IMAGE" \
  --views 2 \
  --prompt "$PROMPT" \
  --engine "$GR00T_BENCH_BIN" \
  --output-dir "$PARITY_OUTPUT" \
  --warmup 5 --iterations 20 --run-nvidia-reference
```

FP8 and INT8 are accepted only after their same-input comparison against BF16
passes. Synthetic zero fixtures are smoke tests, not release parity evidence.

## 3. LIBERO-10 task accuracy

Task accuracy uses the Python `Gr00tPolicy` in the closed-loop LIBERO simulator;
the policy calls the same Rust/CUDA runtime used by deployment. The formal
topology is two physical cameras (`image`, `wrist_image`), ten tasks and 50
episodes per task.

```bash
# Run the shipped policy once per LIBERO-10 task/precision, e.g. by looping
# eval_gr00t_n1d7_libero.py over the ten task ids (see calibrate_gr00t_n1d7_libero.py
# for the task list and per-task invocation shape).
python3 scripts/eval_gr00t_n1d7_libero.py \
  --python "$GR00T_PYTHON" \
  --source-dir "$ISAAC_GR00T_SOURCE" \
  --checkpoint "$GR00T_LIBERO_CHECKPOINT" \
  --backbone "$GR00T_BACKBONE" \
  --tactics "$GR00T_TACTICS" \
  --episodes-per-task 50 --n-envs 10 --seed 7
```

The Orin campaign uses `bf16 int8` as its public precision names and does not
require an FP8 calibration. The runner writes one receipt per task/precision;
the auditor rejects missing or duplicate pairs, mixed views, inconsistent
provenance, malformed success counts and non-finite actions. A formal campaign
also fails before the first episode when either the ApxInf checkout or the
pinned Isaac-GR00T checkout is dirty; the manifest records both Git revisions.

## Recorded task-accuracy evidence

| Device | Precision | Episodes | Successes | Success rate | Candidate status |
| --- | --- | ---: | ---: | ---: | --- |
| Thor | BF16 | 100 | 96 | 96.0% | completed 10-task paired run |
| Thor | FP8 | 100 | 97 | 97.0% | completed 10-task paired run |
| Orin | BF16 | 500 | 462 | 92.4% | valid stage evidence; final scoped build must be rerun |
| Orin | INT8 | 500 | 463 | 92.6% | valid stage evidence; final scoped build must be rerun |

The completed Thor paired run covers all ten LIBERO-10 tasks with 10 episodes
per task, two views and seed 7. All 20 receipts record the same Python extension
SHA-256 (`84d9537307b71c1fade6d737c56b062a96d299d542ed23b13022f96c17869676`).
The audit passed all ten BF16/FP8 same-input pairs: minimum cosine
`0.999916888`, maximum relative L2 `0.012892931`, maximum absolute error
`0.14453125`, and maximum mean absolute error `0.008266874`. The Isaac-GR00T
checkout was clean at revision `51d4c89f72fda44cbf77285c6a8114b52676b8a1`.
The ApxInf checkout was dirty at base revision
`16d9eab8b274c31d4e5d7d797bda7018d907608f`, so these results are complete
10-task evidence but are not attributed to a clean release commit.

The completed Orin paired run used two views, seed 7 and 50 episodes for each
of the 10 tasks. Its same-input first-action parity passed across all task pairs:
minimum cosine `0.999961814`, maximum relative L2 `0.008739077`, maximum
absolute error `0.1171875`, and maximum mean absolute error `0.004672672`.
The processor checkout was clean at revision
`51d4c89f72fda44cbf77285c6a8114b52676b8a1`; checkpoint config SHA-256 was
`c14275c783bc6a1317d6a5ef54b3a44452a9e3620d077e8945d4c51b934394ea` and
the Python extension SHA-256 was
`723809ce749a76baf942be4673ecfb70738e337f4a0b6b6aa2deae764ad13c15`.

The Orin INT8 parity numbers above are retained as historical engineering
evidence only: that run predates the PR#42 branch (it was taken on a dirty
checkout at revision `16d9eab`, which does not contain the PR#42 commits).
The final acceptance gate below is run on the exact Thor PR#42 candidate
(clean tree, commits through the vision-FA2 and dual-geglu-test fixes), which
is the sm_110 target this PR ships for.

Before merge, replace every pending or stage-evidence row with a result from the
final unchanged candidate and record the campaign manifest, audit report,
binary, calibration and tactics SHA-256 values. Results from a different source
revision or runtime binary must not be silently combined.

## Candidate verification status

The PR#42 candidate passes these checks on the clean Thor sm_110 tree
(HEAD through the vision-FA2 and dual-geglu-test-fix commits):

- `cargo check -p apxinf-py` (non-CUDA build);
- `cargo check -p apxinf-py --features cuda`;
- `cargo check -p apxinf-model --features cuda --example gr00t_fixture_bench`;
- `cargo test -p apxinf-model --features cuda --lib`: 108/108;
- model-family boundary and unified-input integration tests: 13/13;
- `cargo test -p apxinf-py --features cuda --lib`: 4/4;
- `cargo test -p apxinf-cuda --lib -- --test-threads=1`: 138/138 (this target
  previously failed to compile: two orphan tests referenced a dual-geglu API
  that da9cae0 refactored away, silently disabling the whole gate; the tests
  are removed and the surviving validate_fp8_dual_geglu_record path retains the
  coverage);
- GR00T Python policy/audit tests: 23 passed, 2 optional real-checkpoint tests
  skipped when checkpoint/native-smoke environment variables are absent;
- real-checkpoint Orin Rust fixture smoke: BF16 whole-graph and INT8
  whole-graph each produced two deterministic finite actions; same-input INT8
  versus BF16 cosine was `0.9995748826` and relative L2 was `0.0296116949`.

The opt-in native smoke covers the complete user-facing call boundary—official
processor, Python policy, PyO3 binding, Rust/CUDA Model Core, action decode and
explicit resource release:

```bash
APXINF_GR00T_CHECKPOINT=/models/GR00T-N1.7-LIBERO/libero_10 \
APXINF_GR00T_BACKBONE=/models/Cosmos-Reason2-2B \
APXINF_GR00T_NATIVE_SMOKE=1 \
APXINF_GR00T_PRECISION=bf16 \
PYTHONPATH=python/apxinf pytest -q \
  python/apxinf/tests/test_gr00t_policy.py -k real_libero
```

Set `APXINF_GR00T_PRECISION=int8` for the Orin INT8 path. FP8 additionally
requires `APXINF_GR00T_CALIBRATION`. The native test performs two calls on the
same policy instance and requires bit-identical normalized and decoded actions,
thereby covering both initial graph capture and steady-state replay.

The CUDA crate's default parallel test run is not yet an acceptance command:
graph-capture tests can overlap tests that use CUDA's legacy stream and produce
CUDA error 906. Every CUDA test passes when the GPU test binary is serialized;
the test-isolation issue must either be fixed or encoded in the GPU CI command.
Thor still needs the same final-tree compile/test run.

## Existing performance evidence

The latency rows in the user guide are retained as pre-scope-reduction
engineering evidence. They are not yet final candidate results. In particular,
the Thor two-view fixture uses real LIBERO RGB images and prompt, but a zero
state and fixed-zero initial noise; this is deterministic Model-Core input, not
a complete closed-loop episode. The reports also lack an embedded ApxInf source
revision/dirty-state record, so they must be reproduced from the final tree.

Recorded Thor report SHA-256 values:

| Precision/views | Report SHA-256 |
| --- | --- |
| BF16/1 | `f488094dfa011cce6aefce2b749cdd41e57d6c9993d9ac9654494d7bd4c7b13f` |
| BF16/2 | `4e4c08bbd30241a07924ebefdc964bc5de80ac555cd5ae3cdba2ad4987b5adca` |
| FP8/1 | `fe88db99614aca71f801e15690fba3aa6859341c06305345daef16735bb6ecbe` |
| FP8/2 | `4220286a99f363198fc25325fd1151db1088bbf870a131b3d7f8a428148fe4ca` |

The corresponding Orin report SHA-256 values are:

| Precision/views | Report SHA-256 |
| --- | --- |
| BF16/1 | `3f85861badb18b4ecace6bffdc51275c96adb10fea737b0cf555df4d8e21b205` |
| BF16/2 | `72c9c7b58d4dbac0edc15257b66f4b595802b1bf6ed6c6a2bab93cfee51199bf` |
| INT8/1 | `4a1129b18422c430f9604d79aca1bdcbb55f25f117b9aa613fbb86d1b9131e50` |
| INT8/2 | `2c4dd6adbaf8e463f2c34731d0df369c2fbe1bcc88f049a10cdb3eed3c06e52c` |

The original Orin delivery binary was rerun read-only on 2026-09-04 using the
same LIBERO checkpoint, two-view fixture, 30-record BF16 tactic database,
batch size one, four flow steps, 10 warmups and 50 whole-graph replays.  The
binary SHA-256 was
`5254dac1517fc612cd1b21c46931c61bca18a28cc7dd09c4f537711e8bc0fc27`.
BF16 reproduced at P50 `87.474436 ms` with action sum
`-17.202308654785156`; INT8 reproduced at P50 `70.229443 ms` with action sum
`-17.022119522094727`.  Every replay within each run produced the same action
sum, and both sums exactly match their corresponding original delivery report.

The archived delivery parity reports compare those outputs with the NVIDIA
BF16 reference on identical processor tensors and initial noise.  BF16 passes
with cosine `0.999825140` and relative L2 `0.018713091`; INT8 passes with
cosine `0.999605250` and relative L2 `0.028381493`.  Therefore the original
Orin performance rows remain valid engineering measurements.

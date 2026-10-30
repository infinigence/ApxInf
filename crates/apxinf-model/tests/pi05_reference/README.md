# PI0.5 reference regression tests

This suite compares a pinned PI0.5 checkpoint on CPU FP32 and CUDA, using exactly
the same RGB, tokens and provided noise. It tests both eager and graph outputs.
Run it directly on Thor/Orin over SSH, or let a registered GitHub Actions runner
execute the same command. Direct execution does **not** require GitHub runner
registration or a device address in the repository.

## Test layout and portable checks

- `../pi05_reference_gate.rs`: Cargo integration test; three portable metric/input
  tests plus an explicitly ignored full-checkpoint hardware test.
- `run.py`: builds and selects the hardware test, prepares fixed inputs and
  qualified recipes, then checks its report.
- `gate.py`, `policy.json`, `test_gate.py`: per-board cosine floors and tests for
  the rule that existing thresholds cannot decrease.
- `../fixtures/pi05_cpu/`: tiny independent CPU oracle used by the CPU tests.
- `../fixtures/pi05_gpu/`: full-size public inputs and qualified CUDA recipe banks.
- `prepare.py`: helper for preparing external NPZ requests for investigations;
  these requests do not replace the fixed CI corpus.

From the repository root, without a GPU or external checkpoint:

```sh
cargo test --locked -p apxinf-model --test cpu_pi05
cargo test --locked -p apxinf-model --lib cpu_pi05::
cargo test --locked -p apxinf-model --test pi05_reference_gate
python3 -m unittest discover -s crates/apxinf-model/tests/pi05_reference -p test_gate.py
python3 crates/apxinf-model/tests/pi05_reference/gate.py
```

The full-checkpoint test is reported as **ignored**, not passed, in this mode.
The GPU runner below explicitly selects it and treats execution failure as an error.

## Run directly on Thor or Orin

### 1. Connect and prepare the checkout

SSH to the target board using your usual host/address, then enter a Git checkout
containing the revision you want to test:

```sh
ssh <thor-host-or-orin-host>
cd /path/to/ApxInf
git rev-parse HEAD
git status --short
```

Build/run prerequisites are Python 3, Rust/Cargo, CUDA, and multithreaded OpenBLAS,
plus sufficient memory and disk for the full checkpoint and native build. The
[qualified recipe banks](../fixtures/pi05_gpu/recipes/README.md) document the
CUDA/library/device identities used for the baseline. Requalify after environment
changes; do not assume a recipe for another GPU/library will be used.

Set paths on that machine. The checkpoint and FP8 profile must match the SHA256
identities in [the manifest](../fixtures/pi05_gpu/manifest.json); files remain
external and are not downloaded by this suite.

```sh
export PI05_CHECKPOINT=/models/pi05/model.safetensors
export OPENBLAS_NUM_THREADS=4
suite=crates/apxinf-model/tests/pi05_reference
run_id=$(date +%Y%m%d-%H%M%S)
```

For a nonstandard OpenBLAS installation, also set `OPENBLAS_LIB_DIR` to its library
directory and configure the runtime library search path, for example
`LD_LIBRARY_PATH` on Linux. On disk-constrained machines, lowering
`APXINF_NVCC_PARALLELISM` reduces concurrent compiler temporary files.

### 2. Run the lanes for this board

On **Thor**, run BF16:

```sh
python3 "$suite/run.py" --board thor --variant bf16 \
  --checkpoint "$PI05_CHECKPOINT" \
  --output "devlocal/pi05-cpu-reference/manual-${run_id}-thor-bf16"
```

On **Thor**, run FP8 with the matching calibration profile:

```sh
export PI05_CALIBRATION=/models/pi05/calibration.json
python3 "$suite/run.py" --board thor --variant fp8_static \
  --checkpoint "$PI05_CHECKPOINT" --calibration "$PI05_CALIBRATION" \
  --output "devlocal/pi05-cpu-reference/manual-${run_id}-thor-fp8"
```

On **Orin**, run BF16:

```sh
python3 "$suite/run.py" --board orin --variant bf16 \
  --checkpoint "$PI05_CHECKPOINT" \
  --output "devlocal/pi05-cpu-reference/manual-${run_id}-orin-bf16"
```

On **Orin**, optionally run the advisory INT8 lane:

```sh
python3 "$suite/run.py" --board orin --variant int8_dynamic \
  --checkpoint "$PI05_CHECKPOINT" \
  --output "devlocal/pi05-cpu-reference/manual-${run_id}-orin-int8"
```

Use a new output directory for each run; existing directories are rejected to
preserve evidence. The runner compiles the integration test in release mode,
computes a fresh CPU reference, executes the GPU test with fixed recipes, and
applies [policy.json](policy.json). No separate input-generation step is needed.
`scripts/run_pi05_reference_gate.py` remains a thin compatibility launcher for
older commands; it contains no test logic.

### 3. Read the results

The runner prints its final result and saves these files under `--output`:

| File | Meaning |
| --- | --- |
| `result.json` | Board/precision, source identity, `required`, `passed`, and failures |
| `report.json` | Per-case eager/CPU, graph/CPU and eager/graph metrics and repeat stability |
| `execution.log` | Cargo test executable's output and execution errors |
| `source.json`, `build.jsonl` | Revision/dirty state, binary/recipe/policy digests and Cargo build records |
| `report.json.cpu.json` | CPU outputs bound to the exact input identity |
| `inputs/`, `recipes/` | Materialized requests and exact recipes used for the run |

For Thor BF16/FP8 and Orin BF16, a numerical failure returns a nonzero exit code.
Orin INT8 numerical failures are recorded as `passed: false`, `required: false`
but return zero. Setup/build/execution errors return nonzero even for a local
INT8 invocation; the GitHub workflow marks that entire lane non-blocking.
If execution stops early, `result.json` may not exist: inspect terminal output
and `execution.log`; missing reports do not mean the test passed.

Each required case must meet its cosine floor in **both** eager and graph mode.
Eager/graph outputs must agree exactly and repeated graph outputs must be bitwise
equal. CPU/GPU relative L2 and max-abs are reported for diagnosis. This is a
numerical regression gate, not a robot task-success evaluation.

CI disables autotuning and uses the tracked recipe bank. To qualify a deliberate
recipe change, supply `--tuning PATH --autotune`; PATH may be a JSON bank or a
directory (an empty directory retunes all operations). Replay without autotuning
and verify the unchanged floors before promoting a bank. Raise floors after
verified accuracy improvements; do not lower them or substitute easier inputs.

## Trigger through GitHub Actions

The [GPU workflow](../../../../.github/workflows/pi05-reference-gpu.yml) only has
`workflow_dispatch`; opening or updating a PR does **not** start GPU jobs.
The [portable workflow](../../../../.github/workflows/pi05-reference.yml) runs
CPU tests and the monotonic-policy check automatically for PRs.

For GPU dispatch, an administrator must first:

1. Register a self-hosted runner on each board through repository/organization
   **Settings → Actions → Runners** and keep the runner service online.
2. Give both runners the `self-hosted`, `Linux`, `ARM64`, `pi05-ci` labels and the
   appropriate `thor` or `orin` label. The runner connects outbound to GitHub and
   executes jobs locally; GitHub does not SSH into the board and no device IP is
   stored in the workflow.
3. Install the dependencies above and provision the pinned model/profile files.
   Configure repository variables `PI05_CHECKPOINT` and `PI05_CALIBRATION` with
   runner-local paths. The shared checkpoint variable must resolve on both boards;
   calibration is used only by Thor FP8.
4. Merge the workflow into the repository's default branch so it is available
   for manual dispatch.

Then select **Actions → PI05 GPU reference → Run workflow**, choose a reviewed
branch in that repository, and start the run. Or use:

```sh
gh workflow run pi05-reference-gpu.yml \
  --repo infinigence/ApxInf --ref <branch-in-that-repository>
```

GitHub assigns the four lanes to matching runners and uploads their output
directories as artifacts. Orin INT8 has `continue-on-error`; the other three
lanes must pass. For a fork PR whose branch is not in the workflow's repository,
use the direct SSH procedure with that PR checked out, or prepare a reviewed
branch in the workflow's repository.

Without a matching online runner, manually dispatched jobs remain queued rather
than getting a GitHub-provided Thor/Orin. This workflow does not register runners,
configure repository variables or change branch protection. GPU dispatch is not
a PR merge requirement unless an administrator explicitly makes it one.

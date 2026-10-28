# Model comparison

Use `just parity` for deterministic checks over recorded model evidence. This
tool separates full-output agreement, actual shape/mode coverage, persisted
tuning growth and source/build binding. It does not decide whether a model is
generally correct, certify a GPU run or infer performance from tuning time.
Run `just parity --help` and each subcommand's help for arguments.

## Freeze the comparison

Keep a JSON contract under `devlocal/<feature>/` before running either model:

```json
{
  "version": 1,
  "comparison": {"kind": "byte"},
  "cases": [{
    "id": "original",
    "input_names": ["pixels", "tokens", "state", "noise"],
    "grid": [[1, 18, 18], [1, 18, 18]],
    "output_shape": [10, 26],
    "mode": "eager",
    "domain": "normalized"
  }],
  "roundtrips": []
}
```

This is a WallOSS-shaped example, not a universal model contract. Use the
requested model's actual tensors, shape and output domain. Masks and position
IDs belong in `input_names` when consumed; use an empty `grid` list for a
nonvisual model. Reference comparisons can select
`{"kind": "tolerance", "min_cosine": 0.999, "max_relative_l2": 0.01}`;
those illustrative values must be replaced by the task's accepted thresholds.
Both metrics apply to the complete tensor, with shape and finite-value checks.
The reference is the first side of relative L2. A nonzero output against a
zero reference fails; two all-zero outputs agree. Byte comparison includes
dtype, shape and all bytes, including signed zero.

For A → B → A coverage, declare three distinct case ids with the original
grid repeated in the last case, and put the first/last ids in `roundtrips`.
Both inputs and full outputs must return byte-identically in each build.
The number of distinct grids is not the number of observations.

## Record at the consuming boundary

The Python frontend already requires NumPy. Import `scripts.model_parity` from
the checkout root in a model-specific harness. This example shows where to
place capture around an existing call; names such as `consumed` and
`observed_mode` must come from the real adapter, not the test's requested values:

```python
from scripts.model_parity import Capture

capture = Capture(run_directory, contract_path, case_id,
                  inputs=consumed, loaded_artifact=native_module.__file__)
actions = run_model(consumed)
capture.finish(actions, effective_grid=observed_grid,
               actual_mode=observed_mode, output_domain="normalized",
               board=board, arch=arch, checkpoint=checkpoint_digest,
               entry_point="Policy.infer")
```

Place each capture in `<side>/<case-id>/`; directories cannot be overwritten.
The tool copies inputs before execution and saves full arrays without pickle,
binds them to the contract hash, and hashes the loaded file before/after the
call. Verify in the adapter that these are the consumed tensors: if a reference
ignores a noise argument, capture inside its forward call instead. Passing the
same random seed to two RNGs does not establish identical noise. Inputs must
match in dtype, shape and bytes; choose a common comparison boundary when the
implementations preprocess differently. Different precision in outputs is
allowed only with a tolerance comparison.

Run `just parity compare` with the contract and the two side directories.
Missing required captures produce UNVERIFIED; shape, input, mode, domain,
integrity or numerical mismatches fail. Contract edits invalidate captures.
The optional `just parity shapes` accepts ordered observation objects with
`id`, `grid`, `output_shape`, `mode`, and `domain` to check shape coverage alone.
It cannot claim output parity. Its observation list must cover every case.

## Bind a build to loaded bytes

Use `just parity build` in each source checkout to run Cargo with a new,
nonexistent evidence directory under `devlocal/`. Supply the expected artifact
relative to its fresh target directory and the Cargo build arguments (include
the required CUDA feature). The wrapper chooses the target directory, records
the command/log, fingerprints source before and after, and hashes the resulting
artifact. It rejects failed builds, source changes and missing artifacts.
Use `just parity identity` for an artifact-only audit; without a fresh-build
receipt it returns UNVERIFIED rather than inferring provenance from cache data.

Load that artifact in the tested process, or install a copy with identical
bytes. Give the resulting `build.json` to `just parity compare` for that side.
The check requires matching built/loaded bytes, intact build log/artifact and
unchanged source. Timestamp/cache metadata by itself never passes this check.
Keep each source checkout and build directory available during verification.

This is a local observed-build record, not cryptographic compilation
attestation: the command, build scripts, Cargo configuration, external toolchain,
ignored dependencies and submodule contents still need review. Source
fingerprinting follows the [acceptance tool](agent-acceptance.md), including its
limits. CUDA feature selection alone does not prove a toolkit or kernels exist.
The wrapper builds an artifact; it does not install a wheel or provide a generic
model adapter. Missing archived build data remains UNVERIFIED; preserve useful
historical output checks without inventing a receipt after the event.

## Check tuning across persistence

`just parity tuning` reads the actual native ApxInf tuning JSON databases,
canonicalizes complete keys and checks device/library/build identity. It rejects
duplicate keys and malformed tactic identities/timings. The backend remains the
authority for whether a particular candidate is legal on a particular device.
Run it on before/after snapshots for **each** stage: empty DB →
first input, then saved/reloaded DB → new shape. A stage passes key growth only
if it adds keys and retains previous keys; changing old records alone fails.
The empty snapshot must have the same native DB header, with an empty `records`
list; a missing file is unresolved evidence. Record changes are displayed
separately. Full output checks and evidence of
actual tuning execution remain separate requirements. Handwritten snapshots
or a printed `AutoTune` flag do not establish that execution occurred.

## Interpret and retain results

Exit codes are 0 for declared checks passing, 1 for failure or invalid evidence,
and 2 for missing evidence. Output agreement can pass while build identity is
UNVERIFIED; the combined comparison then exits 2. Record checks in the existing
[acceptance plan](agent-acceptance.md), pinning contract, harness, snapshots and
build receipts in its `inputs` so edits invalidate evidence. A tuning-only or
shape-only pass covers only that named claim.

Adapters are trusted measurement code, not an independent oracle. Review the
consumer boundary, checkpoint identity, actual execution mode and original
fixture; this tool cannot detect fabricated observations. GPU, Graph replay and
latency acceptance still require their dedicated real-device measurements.

# PI0.5 FP8 calibration

Static FP8 needs representative activation scales in
`<model-dir>/calibration.json`. The calibration input uses the same public
Observation fields as inference; do not provide preprocessed tensors such as
`rgb`, `token_ids`, or `noise`.

## Observation manifest

The portable input is a JSONL file with one Observation per line. Image values
are paths relative to the manifest, or absolute paths. State is optional unless
the checkpoint's input configuration requires it.

```json
{"observation/image":"frames/000-base.png","observation/wrist_image":"frames/000-wrist.png","prompt":"pick up the block","observation/state":[0.1,0.2,0.3]}
{"observation/image":"frames/001-base.png","observation/wrist_image":"frames/001-wrist.png","prompt":"open the drawer","observation/state":[0.2,0.1,0.4]}
```

Generate the calibration file:

```bash
python3 scripts/calibrate_pi05.py \
  --model-dir <path-to-model> \
  --manifest <path-to-observations.jsonl>
```

By default this writes `<path-to-model>/calibration.json`. Use `--output` for
another location. Existing files are not overwritten unless `--force` is
passed.

Choose observations that represent deployment cameras, prompts, robot state,
lighting, scenes, and object poses. Every observation must contain exactly the
camera views expected by the checkpoint.

## Observation directory

The second input is a directory of `*.npz` files, one Observation per file, with
array values stored under the wire keys. `--input-dir` globs `*.npz` and sorts
the names, so zero-padded filenames keep the data identity stable between runs.

```bash
python3 scripts/calibrate_pi05.py \
  --model-dir <path-to-model> \
  --input-dir <path-to-observations>
```

This is the format for frames that were rendered rather than photographed: the
producer writes them once, and the calibration input becomes a reviewable,
re-runnable artifact instead of a side effect of a rollout.

## Native LIBERO task observations

For the PI0.5 LIBERO checkpoint, generate a calibration file from observations
rendered by the actual LIBERO10 task suite:

```bash
python3 scripts/calibrate_pi05.py \
  --model-dir <path-to-model> \
  --libero-suite libero_10
```

This path uses LIBERO's BDDL tasks, language instructions, initial states, and
off-screen simulator cameras. It applies the same camera orientation and
robot-state conversion as `scripts/eval_libero.py` — the resize is the policy's,
so the calibration frames are the ones inference will actually see. Sampling is
deterministic and task-balanced. By default it captures one settled initial
state from every task; `--samples N` selects more initial states while retaining
balanced task coverage.

This command needs the same LIBERO and MuJoCo dependencies as the repository's
LIBERO evaluation command. It exists so a kernel change or a new checkpoint can
be recalibrated against ApxInf's own published LIBERO protocol without checking
out a repository that takes ApxInf as a submodule.

### Capturing out of process

[apxinf-robo](https://github.com/team-mz/APXinf-robo) does the same capture from
the other side of the seam, writing the NPZ directory `--input-dir` reads:

```bash
apxinf-robo capture-libero --suite libero_10 --output-dir /tmp/libero-calib
python3 scripts/calibrate_pi05.py \
  --model-dir <path-to-model> \
  --input-dir /tmp/libero-calib
```

The two produce the same frames — they share `scripts/libero_observation.py` and
its apxinf-robo mirror — and differ in where MuJoCo runs. Prefer this form for a
robot deployment: it keeps the simulator out of the engine host, and it makes the
calibration input a reviewable, re-runnable artifact rather than a side effect of
whoever happened to run the calibrator. `capture-libero` names the NPZ fields
after the robot it was given, so the captured dialect is by construction the one
the checkpoint is served under.

For another simulator or a deployment source, export its public Observations
through the manifest or directory interface instead.

## Loading the calibration file

`AutoPolicy` automatically uses `<model-dir>/calibration.json`:

```python
policy = AutoPolicy.from_pretrained("<path-to-model>", precision="fp8")
```

Pass `calibration=` only when the calibration file is stored elsewhere:

```python
policy = AutoPolicy.from_pretrained(
    "<path-to-model>",
    precision="fp8",
    calibration="/path/to/calibration.json",
)
```

At FP8 startup, ApxInf reads the checkpoint shards to compare their SHA-256
identity with the calibration file. This can add startup time on slower storage.
An identity mismatch emits a warning and continues; malformed calibration files,
missing or invalid scales, and incompatible execution plans still fail because
the runtime cannot use them safely.

`tactics.json` is optional. If it is absent, the runtime warns and continues
with compatible kernel fallbacks; calibration is unaffected, though performance
may be lower than with tuned tactics.

Calibration files using an older schema are not migrated automatically.
Regenerate `calibration.json` with the ApxInf version being deployed.

Run `python3 scripts/calibrate_pi05.py --help` for checkpoint-specific input
overrides and reproducibility options.

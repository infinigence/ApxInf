# π0-FAST FP8 calibration

Static FP8 needs representative activation scales in
`<model-dir>/calibration.json`. Unlike a flow-matching policy, π0-FAST has no
error budget to spend: it selects every action token with a greedy argmax, so a
scale error that only perturbs a continuous action can instead flip a token. The
failure is not graceful — with a uniform scale the decode never emits the FAST
`|` terminator, runs to its `max_action_tokens` budget, and returns a token
stream every rollout discards.

FP8 π0-FAST therefore **requires** a profile. There is no implicit fallback: a
missing `calibration.json` is a load error naming the ways to supply one.

## Observation manifest

The portable input is a JSONL file with one Observation per line. Image values
are paths relative to the manifest, or absolute paths. State is optional unless
the checkpoint's input configuration requires it.

```json
{"observation/image":"frames/000-base.png","observation/wrist_image":"frames/000-wrist.png","prompt":"pick up the block","observation/state":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}
{"observation/image":"frames/001-base.png","observation/wrist_image":"frames/001-wrist.png","prompt":"open the drawer","observation/state":[0.2,0.1,0.4,0.3,0.1,0.2,0.8,0.6]}
```

Generate the calibration file:

```bash
python3 scripts/calibrate_pi0fast.py \
  --model-dir <path-to-model> \
  --manifest <path-to-observations.jsonl>
```

By default this writes `<path-to-model>/calibration.json`. Use `--output` for
another location. Existing files are not overwritten unless `--force` is passed.

Choose observations that represent deployment cameras, prompts, robot state,
lighting, scenes, and object poses. Every observation must contain exactly the
camera views expected by the checkpoint.

## Observation directory

The second input is a directory of `*.npz` files, one Observation per file, with
array values stored under the wire keys. `--input-dir` globs `*.npz` and sorts
the names, so zero-padded filenames keep the data identity stable between runs.

```bash
python3 scripts/calibrate_pi0fast.py \
  --model-dir <path-to-model> \
  --input-dir <path-to-observations>
```

This is the format for frames that were rendered rather than photographed: the
producer writes them once, and the calibration input becomes a reviewable,
re-runnable artifact instead of a side effect of a rollout.

## Native LIBERO task observations

For a π0-FAST LIBERO checkpoint, generate a calibration file from observations
rendered by the actual LIBERO10 task suite:

```bash
python3 scripts/calibrate_pi0fast.py \
  --model-dir <path-to-model> \
  --libero-suite libero_10
```

This path uses LIBERO's BDDL tasks, language instructions, initial states, and
off-screen simulator cameras. It applies the same camera orientation and the
same robot-state convention as `scripts/eval_libero.py` — including the state
width, which is a checkpoint property: π0-FAST's `observation.state` statistics
are 8 wide, so the capture keeps both of LIBERO's mirrored finger joints. The
resize is the policy's, so the calibration frames are the ones inference will
actually see. Sampling is deterministic and task-balanced. By default it
captures one settled initial state from every task; `--samples N` selects more
initial states while retaining balanced task coverage.

The capture runs a complete BF16 inference per observation — prefix and
autoregressive decode — and stops the decode at the FAST terminator, because a
profile is only valid for the activations deployment quantizes. Free-running the
decode past that point records activations from token streams no rollout
reaches, and one such maximum is enough to coarsen a whole layer's scale.

This command needs the same LIBERO and MuJoCo dependencies as the repository's
LIBERO evaluation command. It exists so a kernel change or a new checkpoint can
be recalibrated against ApxInf's own published LIBERO protocol.

### Capturing on another machine

The two halves have disjoint requirements: rendering needs LIBERO and MuJoCo but
no GPU and no checkpoint, and calibrating needs the checkpoint and a GPU but no
simulator. Splitting them at the NPZ directory lets each run where it belongs —
the simulator stays off the engine host — and makes the calibration input a
reviewable, re-runnable artifact rather than a side effect of whoever happened to
run the calibrator.

Whatever writes those files decides the field names, and they have to be the wire
keys the checkpoint is served under. When they are not the checkpoint's own,
name them:

```bash
python3 scripts/calibrate_pi0fast.py \
  --model-dir <path-to-model> \
  --image-key observation/image --image-key observation/wrist_image \
  --input-dir <path-to-observations>
```

For another simulator or a deployment source, export its public Observations
through the manifest or directory interface instead.

## Loading the calibration file

`AutoPolicy` (and therefore `Pi0FastPolicy`) automatically uses
`<model-dir>/calibration.json`:

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

The same two paths reach the native loader; `scripts/eval_libero.py
--precision fp8` needs no extra flag when the profile sits beside the
checkpoint. Run the calibration on the BF16 runtime: an FP8 capture would
measure already-quantized activations, and each regeneration would tighten the
scales further.

At FP8 startup, ApxInf reads the checkpoint shards to compare their SHA-256
identity with the calibration file. This can add startup time on slower storage.
An identity mismatch emits a warning and continues; malformed calibration files,
missing or invalid scales, wrong site coverage, and profiles for another model
family still fail, because the runtime cannot use them safely.

## Measuring the speed ceiling without a profile

`APXINF_PI0FAST_FP8_ACTIVATION_SCALE=<scale>` loads FP8 with one uniform
activation scale for every site. It prints a warning and is not a deployment
configuration: measured profiles are the only ones that preserve the token
stream. Use it to reproduce a speed number, never to ship.

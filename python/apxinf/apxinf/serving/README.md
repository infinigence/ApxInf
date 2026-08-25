# Serving pi05 over the OpenPI websocket protocol

Run a pi05 checkpoint on the ApxInf engine and expose it through an
**OpenPI-compatible websocket**, so any unmodified upstream `openpi_client` can
connect. This guide covers starting the server, calling it, running the
LIBERO-10 accuracy eval, and porting an OpenPI config + processors for a new
robot.

Paths below are relative to the repo root. The engine binding (`apxinf_py`) is
built per GPU architecture — see the root [README](../../../../README.md) for the
Rust/CUDA build (`APXINF_CUDA_ARCH=sm_87` for Orin, `sm_110` for Thor).

## 1. Environment

```bash
# engine binding (architecture-specific wheel built from crates/apxinf-py)
pip install apxinf_py-*.whl
# numpy frontend: processors + L2 policy + serving
pip install -e python/apxinf
# transport + processor deps
pip install -r scripts/requirements-pi05-websocket.txt
# upstream client (for §3 / §4)
pip install openpi-client
```

Self-check the engine loads (`num_views` / `action` shape must match the
checkpoint):

```bash
python -c "import apxinf_py; print(apxinf_py.Model.load('pi05','<ckpt>/model.safetensors',device='cuda:0',precision='bf16'))"
# Model(device=cuda:0, action=[50, 32], views=3, image=224, patch=14)
```

> The engine `.so` is compiled per GPU arch — a wheel built for one arch will not
> load on another. BF16 is the supported serving precision.

## 2. Start the server

```bash
python scripts/pi05_openpi_websocket_server.py \
    --model-dir <ckpt> \
    --robot franka_libero \  # embodiment preset: wire keys + pre/post steps + action width
    --precision bf16 \
    --device cuda:0 \
    --host 0.0.0.0 --port 8000
```

- `--robot` is the one flag that decides **which keys the client must send**. It
  is openpi's `serve_policy.py --policy.config <TrainConfig>` equivalent: the
  embodiment is fixed at startup and the client cannot negotiate it. A checkpoint
  fine-tuned for another robot **must** name it — the wire keys, the state
  routing, and the action encoding all differ, and a mismatch degrades silently
  (plausible-looking actions from the wrong cameras) rather than failing.
  `--robot` also sets `--action-dim` and `--discrete-state`; pass those flags only
  to override the preset. `python scripts/pi05_openpi_websocket_server.py --help`
  lists every preset with its slot→key mapping.
- Presets are named `<arm>_<key convention>`, because the arm alone does not fix
  the contract: LIBERO and DROID are both Franka Panda with different keys and
  action spaces. `--robot libero` still resolves to `franka_libero`.
- `--image-keys` / `--state-key` override individual keys, for a deployed client
  that already speaks a fixed dialect. `--image-keys` order is significant: key
  *i* fills model view slot *i* (`base_0_rgb`, `left_wrist_0_rgb`,
  `right_wrist_0_rgb`).
- `--num-views` serves a checkpoint with **fewer cameras than it declares** — a
  3-view checkpoint on a 2-camera robot. The trailing view slots are dropped at
  load time, which is numerically identical to what openpi does (it zero-pads the
  absent view and masks it; a masked view is excluded from attention, occupies no
  RoPE position, and the vision tower has no per-slot parameters) while skipping
  that view's 256 patch tokens every step. It must equal the number of image
  keys, and it is deliberately required rather than inferred, so a *forgotten*
  camera key is an error instead of a quiet accuracy loss.
- `--host 0.0.0.0` accepts remote clients (split deployment); `127.0.0.1` for a
  local-only test.
- Health check: `curl http://<host>:8000/healthz` → `OK`.
- The startup log prints the served contract; the same fields are pushed to every
  client on connect (see §3):

  ```
  serving robot=franka_libero H=10 x D=7 image_keys=['observation/image', 'observation/wrist_image'] state=observation/state discrete_state=False
  ```

## 3. Call it (stock `openpi_client`, unmodified)

```python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy("<host>", 8000)
meta = client.get_server_metadata()
# {'robot': 'franka_libero', 'image_keys': ['observation/image', 'observation/wrist_image'],
#  'state_key': 'observation/state', 'discrete_state': False, 'prompt_key': 'prompt',
#  'num_views': 2, 'action_horizon': 10, 'action_dim': 7, 'precision': 'bf16', ...}

result = client.infer({
    "observation/image":       base_rgb_uint8,     # HWC uint8 RGB; the server resizes
    "observation/wrist_image": wrist_rgb_uint8,
    "observation/state":       state_f32,          # dropped unless discrete_state; see §5
    "prompt":                  "put both moka pots on the stove",
})
result["actions"]         # float32 [H, action_dim]; H is the checkpoint's native horizon
result["policy_timing"]   # {'infer_ms': bare model, 'policy_ms': full policy}
```

The metadata **is** the wire contract — assert against `meta["image_keys"]` /
`meta["state_key"]` instead of hardcoding keys, so a server/client mismatch shows
up as a failed assertion at startup rather than as degraded accuracy in the
field. Sending a key the server does not serve raises a `KeyError` naming both
sides; sending an *extra* key is silently ignored (matching openpi).

**Images are RGB.** Neither this server nor openpi converts colour: an `H×W×3`
uint8 array is taken as RGB as-is. A client reading frames with OpenCV must
`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` first — BGR frames run fine and score
badly. Resizing *is* done server-side (aspect-preserving pad to the model's
edge), so any input resolution is fine.

Reference client: [python/apxinf/examples/openpi_client.py](../../examples/openpi_client.py).

## 4. LIBERO-10 accuracy eval

The simulation side (client) can run two ways; the server always hosts the
engine. LIBERO needs a pinned sim stack (`robosuite==1.4.0` / `mujoco==2.3.2` /
`bddl==1.0.1`) and `MUJOCO_GL=egl` on headless boxes, plus a LIBERO checkout on
`PYTHONPATH`.

### Form A — same machine

```bash
# terminal 1: server (see §2)
python scripts/pi05_openpi_websocket_server.py --model-dir <ckpt> --precision bf16 --port 8000 &

# terminal 2: eval against the local server
export MUJOCO_GL=egl
export PYTHONPATH=<path/to/LIBERO>
python scripts/eval_pi05_libero_openpi.py \
    --host 127.0.0.1 --port 8000 --precision bf16 \
    --task-ids 0,1,2,3,4,5,6,7,8,9 --trials-per-task 10 \
    --results-jsonl out/libero_bf16.jsonl \
    --summary-json  out/libero_bf16.summary.json
```

### Form B — split: client on an x86 sim box, server remote

Decouple via websocket: keep the fragile sim stack on a machine that already runs
LIBERO, and let the engine host be inference-only — the production-shaped call
pattern (robot/sim on one end, engine on the other).

```bash
# server host: start with --host 0.0.0.0 (see §2)
# x86 client (its own LIBERO env):
python scripts/eval_pi05_libero_openpi.py \
    --host <server-host> --port 8000 --precision bf16 \
    --task-ids 0,1,2,3,4,5,6,7,8,9 --trials-per-task 10 \
    --results-jsonl out/libero_bf16.jsonl \
    --summary-json  out/libero_bf16.summary.json
```

The evaluator is resumable (fsync'd JSONL ledger, skips completed trials on
rerun); every trial records success / steps / replans and a four-segment timing
split; the summary rolls up `success_rate` and timing. Key `summary.json`
fields:

```jsonc
{
  "success_rate": 0.87,                 // LIBERO-10 task-level success (accuracy)
  "per_task": { "0": {"success_rate": ...}, ... },
  "timing": {
    "total_inference_calls": 431,       // = total replans
    "per_call_ms": {                    // mean split per websocket inference call
      "model_ms": 213.4,                //   bare model (engine)
      "server_processor_ms": 6.1,       //   server pre/post pipeline
      "websocket_transport_ms": 4.8,    //   transport + serialization
      "preprocess_ms": 2.3,             //   client resize+state (not in round-trip)
      "inference_ms": 224.5             //   round-trip wall clock ≈ sum of the first three
    }
  }
}
```

> LIBERO is a **Franka Panda 7-DoF** benchmark (7-dim action = 6 EEF deltas + 1
> gripper), so eval runs the default `--robot franka_libero` preset, which
> already sets the 2 camera keys and the 7-dim action width. It is a different
> embodiment from the robots in §5 (e.g. G1) and unrelated to them.

## 5. Port an OpenPI config + processors into apxinf (for your own robot)

Given an **OpenPI fine-tune config + a set of `DataTransformFn` pre/post
transforms** (e.g. `pi05_UnitreeG1`: 3 cameras, 16-DoF state, `action_dim=32`,
delta joint actions, with `unitreeG1Inputs` / `unitreeG1Outputs`), you do **not**
pull any external framework into apxinf. You port those `dict→dict` transforms,
by semantics, into apxinf `ProcessorStep`s and splice them into the pre/post
`Pipeline` of `Pi05Policy`.

A working G1 example ships in-tree — copy and adapt it:

```
python/apxinf/apxinf/processors/robots/unitree_g1.py   # robot-specific ProcessorSteps
python/apxinf/apxinf/robots/unitree_g1.py              # build_unitree_g1_policy factory
```

### 5.1 Two landing spots

| Layer | Where | What |
|---|---|---|
| **robot-specific step** (pure numpy, per-embodiment, model-agnostic) | `python/apxinf/apxinf/processors/robots/<robot>.py` | `ProcessorStep` subclasses: decode-state / delta→absolute / encode-actions |
| **assembly factory** (wires the steps onto `Pi05Policy`) | `python/apxinf/apxinf/robots/<robot>.py` | a `build_<robot>_policy(...)` that rewrites the pre/post pipelines after `from_pretrained` |

### 5.2 OpenPI transform → apxinf equivalent (G1 as the sample)

| OpenPI transform | apxinf equivalent | Note |
|---|---|---|
| camera rename + CHW→HWC + float→uint8 | `image_keys` config + existing `ParseImage` | **no new code** — `ParseImage` already does CHW→HWC / float→uint8 |
| `_decode_state` (joint flip + gripper→angle) | `UnitreeG1DecodeState` (**input** step, before `tokenize`) | so both discretized state and delta→absolute see decoded state |
| 32-dim unnormalize | existing `Unnormalize` (**full model width**) | full-width so delta→absolute sees the complete action |
| `AbsoluteActions` (delta→absolute, needs state) | `UnitreeG1AbsoluteActions` (**output** step) | adds current state on masked joint dims; gripper dims pass through |
| `unitreeG1Outputs` 32→16 + flip + gripper | `UnitreeG1EncodeActions` (**output** step) | trim to robot dims, apply flip, invert gripper map |

> **Not ported**: training-time data-cleaning variants (`_NoLeftCam` /
> `fixed_hand`) are on the training-data path, not the serving path.

### 5.3 Write a ProcessorStep

Narrow contract: `__call__(data) -> data`, mutating the `data` dict in place;
observation lives under `OBSERVATION`, actions under `ACTIONS` (see
`apxinf/processors/transforms.py`). List tunable knobs in `PARAMS` (for
`with_overrides` to copy-and-tweak). Skeleton:

```python
import numpy as np
from ..base import ProcessorStep
from ..transforms import ACTIONS, OBSERVATION

class MyRobotEncodeActions(ProcessorStep):
    """Map absolute pi actions back to robot space: 32->N, flip, gripper."""
    def __call__(self, data):
        actions = np.asarray(data[ACTIONS], dtype=np.float32)[:, :ROBOT_DIM]
        # ...embodiment-specific flip / gripper inverse-map...
        data[ACTIONS] = np.ascontiguousarray(actions)
        return data
```

An input step is analogous: read state from `data[OBSERVATION][state_key]`,
decode, write back (work on a **shallow copy** — don't mutate the caller's dict).
Port **placeholder calibration** faithfully as a hook (in G1 the flip mask is all
`1`, gripper `clip(0,1)` — currently pass-through); fill in real calibration here.

### 5.4 Assemble the pipeline in the factory

Default pi05 pipelines: input `[image_stack, tokenize]`, output
`[trim, unnormalize]`. Noise is generated inside the runtime unless the caller
passes it explicitly; a custom host sampler can still be inserted as a pipeline
step. `Pipeline` offers
`insert_before/insert_after/replace/override/remove/reorder` (each returns a new
pipeline). The factory does three things: load full-width → insert decode on the
input → rewrite the output pipeline:

```python
from ..policies.impls.pi05 import Pi05Policy
from ..processors import Pipeline
from ..processors.robots.my_robot import (
    MyRobotDecodeState, MyRobotAbsoluteActions, MyRobotEncodeActions,
    ROBOT_CAMERAS, ROBOT_DIM,
)

def build_my_robot_policy(model_dir, *, use_delta_joint_actions=True, adapt_to_pi=True,
                          state_key="observation/state", image_keys=ROBOT_CAMERAS, **kw):
    base = Pi05Policy.from_pretrained(
        model_dir,
        image_keys=tuple(image_keys),
        action_dim=None,        # keep full 32 dims; the encode step trims to ROBOT_DIM
        state_key=state_key,
        **kw,
    )
    input_pipeline = base.input_pipeline
    if adapt_to_pi:
        input_pipeline = input_pipeline.insert_before(
            "tokenize", ("decode_state", MyRobotDecodeState(state_key)))

    output_steps = [("unnormalize", base.output_pipeline["unnormalize"])]   # full width
    if use_delta_joint_actions:
        output_steps.append(("absolute", MyRobotAbsoluteActions(state_key)))
    if adapt_to_pi:
        output_steps.append(("encode", MyRobotEncodeActions()))

    return Pi05Policy(
        base.model,
        input_pipeline=input_pipeline,
        output_pipeline=Pipeline(output_steps),
        image_keys=tuple(image_keys),
        state_key=state_key,
        action_dim=ROBOT_DIM,
        metadata={"robot": "my_robot"},
    )
```

Resulting pipelines:

```
input : [image_stack, decode_state, tokenize]
output: [unnormalize, absolute, encode]   # normalized[H,32] -> actions[H,ROBOT_DIM]
```

### 5.5 One general framework hook (built in, nothing to change)

The delta→absolute output step needs to see the **input state**.
`Pi05Policy.infer` already passes the (decoded) observation into the output
pipeline; the stock `trim`/`unnormalize` ignore it, so existing numbers are
unchanged (matching OpenPI's "output transforms can see input state" semantics).

### 5.6 Register a preset (the last step — do not skip it)

A factory alone is not deployable: whoever launches the server still has to know
your robot's wire keys, action width, and state routing, and getting any of them
wrong is silent. Add one entry to
[`python/apxinf/apxinf/robots/presets.py`](../robots/presets.py) and the whole
contract becomes `--robot <name>`:

```python
MY_ROBOT = RobotPreset(
    name="my_robot",                              # <arm>_<key convention> if the arm is shared
    slots=(                                       # (model view slot, wire key), in slot order
        ("base_0_rgb",        "images/cam_high"),
        ("left_wrist_0_rgb",  "images/cam_left_wrist"),
    ),
    state_key="state",
    action_dim=None,                              # None: the encode step trims
    discrete_state=True,                          # False *drops* state entirely
    builder=build_my_robot_policy,
    summary="My robot: 2 cameras, 14-DoF state, delta joint actions",
    builder_kwargs={"use_delta_joint_actions": True, "adapt_to_pi": True},
)

ROBOT_PRESETS = {p.name: p for p in (FRANKA_LIBERO, UNITREE_G1, MY_ROBOT)}
```

Naming each wire key with the **model view slot** it fills is the point: the
tuple is order-significant (entry *i* becomes model view slot *i*), and a wrong
order still stacks, still has the right shape, and silently feeds the wrong
camera to each slot. The pairing is validated — slots must be a prefix of
`base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb` in order, with no duplicate
wire keys.

Wire keys may be written **flat** (`"observation/image"` — the slash is part of
the name, as LIBERO and DROID send it) or as a **nested path**
(`"images/cam_high"` → `obs["images"]["cam_high"]`, as ALOHA and G1 send it). A
flat hit always wins, so both layouts work from one tuple and an unmodified
upstream client needs no changes.

The preset's contract is published in the served metadata (§3) and printed at
startup, so a client can assert it.

### 5.7 Use and verify

```python
from apxinf import build_unitree_g1_policy          # shipped G1 example
policy = build_unitree_g1_policy("<g1-ckpt>", use_delta_joint_actions=True, adapt_to_pi=True)
actions = policy.infer(obs)["actions"]               # [H, 16]
```

Served the same way, once §5.6 is done:

```bash
python scripts/pi05_openpi_websocket_server.py --model-dir <g1-ckpt> --robot unitree_g1
```

Smoke test (plumbing/shape): [scripts/g1_adapter_smoke.py](../../../../scripts/g1_adapter_smoke.py)
feeds a G1-shaped observation (3 cameras uint8 HWC + 16-dim state + prompt)
through the whole chain and asserts the output is `[H,16]`, finite, gripper dims
∈ [0,1]. **Without real weights / norm_stats** the unnormalize degrades to a
full-width identity — that proves the config runs end-to-end through the apxinf
interface (compatibility), not executable numbers. For real values, supply the
robot's `norm_stats` (≥ROBOT_DIM wide) + real gripper limits + checkpoint; the
**same adapter code runs unchanged**.

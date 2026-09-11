# Serving a policy over the OpenPI websocket protocol

Run a checkpoint on the ApxInf engine and expose it through an
**OpenPI-compatible websocket**, so any unmodified upstream `openpi_client` can
connect. This guide covers starting the server, calling it, and the seam an
outer layer uses to splice its own robot steps into the pipelines.

What this guide deliberately does *not* cover: named robots, simulator glue, and
benchmark rollouts. A body's DoF layout and a dataset's wire keys are not
properties of the weights, so they belong one layer up, outside the engine.

Paths below are relative to the repo root. The engine binding (`apxinf_py`) is
built per GPU architecture — see the root [README](../../../../README.md) for the
Rust/CUDA build (`APXINF_CUDA_ARCH=sm_87` for Orin, `sm_110` for Thor).

## 1. Environment

```bash
# engine binding (architecture-specific wheel built from crates/apxinf-py)
pip install apxinf_py-*.whl
# numpy frontend: processors + L2 policy + serving
pip install -e "python/apxinf[tokenizer,serving]"
# upstream client (for §3)
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
python python/apxinf/examples/openpi_server.py \
    --model-dir <ckpt> \
    --image-keys observation/image,observation/wrist_image \
    --state-key observation/state \
    --precision bf16 \
    --device cuda:0 \
    --host 0.0.0.0 --port 8000
```

- `--image-keys` / `--state-key` decide **which keys the client must send**. They
  are the caller's to name: what a client calls its cameras is a property of the
  recording, not of the weights, and one checkpoint architecture is served under
  several dialects. The contract is fixed at startup and the client cannot
  negotiate it.
- `--image-keys` order is significant: key *i* fills model view slot *i*
  (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`). Omit the flag and the
  policy falls back to those slot names themselves — published as
  `apxinf.CANONICAL_IMAGE_KEYS`, a fallback rather than a contract.
- `--state-key` has no fallback. A wrong camera key raises on the first
  inference; a wrong state key is silent, so a policy that reads state refuses to
  be built without one. `apxinf.CANONICAL_STATE_KEY` names the neutral spelling
  (`"state"`) for a caller that wants it, but nothing defaults to it.
- Wire keys may be written **flat** (`"observation/image"` — the slash is part of
  the name, as LIBERO and DROID send it) or as a **nested path**
  (`"images/cam_high"` → `obs["images"]["cam_high"]`, as ALOHA and G1 send it). A
  flat hit always wins, so both layouts work from one tuple and an unmodified
  upstream client needs no changes.
- `--policy-options '{"num_views": 2}'` serves a checkpoint with **fewer cameras
  than the checkpoint declares** — a 3-view checkpoint on a 2-camera robot. The
  trailing view slots are dropped at load time, which is numerically identical to
  what openpi does (it zero-pads the absent view and masks it; a masked view is
  excluded from attention, occupies no RoPE position, and the vision tower has no
  per-slot parameters) while skipping that view's 256 patch tokens every step. It
  must equal the number of image keys. The explicit value prevents a missing
  camera key from silently reducing the loaded view count.
- Remaining policy kwargs — `prompt_key`, `discrete_state`, `norm_stats`,
  `calibration` — go through `--policy-options` as a JSON object.
- `--action-dim` narrows the published action width; omitted, the checkpoint's
  native width is served.
- `--host 0.0.0.0` accepts remote clients (split deployment); `127.0.0.1` for a
  local-only test.
- Health check: `curl http://<host>:8000/healthz` → `OK`.
- The resolved contract — image keys, state key, action horizon and width — is
  pushed to every client on connect (see §3), so it is asserted rather than
  assumed.

An outer layer can wrap this same server and supply the keys and action width
from a named preset instead of the command line.

## 3. Call it (stock `openpi_client`, unmodified)

```python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy("<host>", 8000)
meta = client.get_server_metadata()
# {'image_keys': ['observation/image', 'observation/wrist_image'],
#  'state_key': 'observation/state', 'discrete_state': False, 'prompt_key': 'prompt',
#  'num_views': 2, 'action_horizon': 10, 'action_dim': 7, 'precision': 'bf16', ...}

result = client.infer({
    "observation/image":       base_rgb_uint8,     # HWC uint8 RGB; the server resizes
    "observation/wrist_image": wrist_rgb_uint8,
    "observation/state":       state_f32,          # dropped unless discrete_state
    "prompt":                  "put both moka pots on the stove",
})
result["actions"]         # float32 [H, action_dim]; H is the checkpoint's native horizon
result["policy_timing"]   # {'infer_ms': bare model, 'policy_ms': full policy}
```

The metadata **is** the wire contract — assert against `meta["image_keys"]` /
`meta["state_key"]` instead of hardcoding keys, so a server/client mismatch shows
up as a failed assertion at startup rather than as degraded accuracy in the
field. A `state_key` of `null` is not a gap: it says the served policy drops
state, so there is no key to send one under. Sending a key the server does not
serve raises a `KeyError` naming both sides; sending an *extra* key is silently
ignored (matching openpi).

**Images are RGB.** Neither this server nor openpi converts colour: an `H×W×3`
uint8 array is taken as RGB as-is. A client reading frames with OpenCV must
`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` first — BGR frames run fine and score
badly. Resizing *is* done server-side (aspect-preserving pad to the model's
edge), so any input resolution is fine.

Reference client: [python/apxinf/examples/openpi_client.py](../../examples/openpi_client.py).

## 4. Splice your own robot steps into the pipelines

An OpenPI fine-tune ships a set of `DataTransformFn` pre/post transforms — say a
16-DoF humanoid with `action_dim=32` and delta joint actions, with
`myRobotInputs` / `myRobotOutputs`. Porting those into ApxInf pulls no external
framework in: you translate each `dict→dict` transform, by semantics, into an
ApxInf `ProcessorStep` and splice it onto a policy through
[`ComposablePolicy.with_adapter`](../policies/base.py).

This section documents the **engine seam**. The steps themselves belong to
whoever owns the body, and live outside this repository.

### 4.1 OpenPI transform → ApxInf equivalent

| OpenPI transform | ApxInf equivalent | Note |
|---|---|---|
| camera rename + CHW→HWC + float→uint8 | `image_keys=` config + existing `ParseImage` | **no new code** — `ParseImage` already does CHW→HWC / float→uint8 |
| `_decode_state` (joint flip + gripper→angle) | a `DecodeState` **input** step, before `tokenize` | so both discretized state and delta→absolute see decoded state |
| 32-dim unnormalize | existing `Unnormalize` (**full model width**) | full-width so delta→absolute sees the complete action |
| `AbsoluteActions` (delta→absolute, needs state) | an `AbsoluteActions` **output** step | adds current state on masked joint dims; gripper dims pass through |
| `myRobotOutputs` 32→16 + flip + gripper | an `EncodeActions` **output** step | trim to robot dims, apply flip, invert gripper map |

> **Not ported**: training-time data-cleaning variants are on the training-data
> path, not the serving path.

### 4.2 Write a ProcessorStep

Narrow contract: `__call__(data) -> data`, mutating the `data` dict in place;
observation lives under `OBSERVATION`, actions under `ACTIONS` (see
`apxinf/processors/transforms.py`). List tunable knobs in `PARAMS` (for
`with_overrides` to copy-and-tweak). Skeleton:

```python
import numpy as np
from apxinf import ProcessorStep
from apxinf.processors.transforms import ACTIONS, OBSERVATION

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
Port **placeholder calibration** faithfully as a hook rather than silently
dropping it; fill in real calibration there.

### 4.3 Assemble the pipeline with `with_adapter`

Default pi05 pipelines: input `[image_stack, tokenize]`, output
`[trim, unnormalize]`. Noise is generated inside the runtime unless the caller
passes it explicitly; a custom host sampler can still be inserted as a pipeline
step.

`with_adapter` prepends input steps and appends output steps — strict onion
nesting, so an adapter never has to know a model-specific step name. Load the
full-width policy through `AutoPolicy` and wrap it:

```python
from apxinf import AutoPolicy, ComposablePolicy
from my_package.processors import (
    MyRobotDecodeState, MyRobotAbsoluteActions, MyRobotEncodeActions, ROBOT_DIM,
)

# state_key and image_keys are the deployment's to name — a dataset fact.
def build_my_robot_policy(model_dir, *, state_key, image_keys,
                          use_delta_joint_actions=True, adapt_to_pi=True, **kw):
    base = AutoPolicy.from_pretrained(
        model_dir,
        image_keys=tuple(image_keys),
        action_dim=None,        # keep full 32 dims; the encode step trims to ROBOT_DIM
        state_key=state_key,
        **kw,
    )
    if not isinstance(base, ComposablePolicy):
        raise TypeError(f"{type(base).__name__} has no with_adapter(); ...")

    before = [("decode_state", MyRobotDecodeState(state_key))] if adapt_to_pi else []
    after = []
    if use_delta_joint_actions:
        after.append(("absolute", MyRobotAbsoluteActions(state_key)))
    if adapt_to_pi:
        after.append(("encode", MyRobotEncodeActions()))

    return base.with_adapter(
        before=before,
        after=after,
        # Report the width produced by the appended encode step.
        action_dim=ROBOT_DIM if adapt_to_pi else None,
        metadata={"robot": "my_robot"},
    )
```

Resulting pipelines:

```
input : [decode_state, image_stack, tokenize]
output: [trim, unnormalize, absolute, encode]   # normalized[H,32] -> actions[H,ROBOT_DIM]
```

`trim` is the model's own step at full width (a no-op when `norm_stats` is as
wide as the model), so `absolute` still sees the whole action.

The wrapped policy is still a `Policy`, so it serves through
`WebsocketPolicyServer` unchanged and publishes the adapter's `action_dim` and
`metadata` in the connect-time contract (§3).

### 4.4 One framework hook (built in, nothing to change)

The delta→absolute output step needs to see the **input state**.
`Pi05Policy.infer` already passes the (decoded) observation into the output
pipeline; the stock `trim`/`unnormalize` ignore it, so existing numbers are
unchanged (matching OpenPI's "output transforms can see input state" semantics).

# WallOSS

WallOSS uses the same model-agnostic websocket transport as PI0.5. Install the
model-specific preprocessing dependencies and launch the example:

```bash
pip install 'apxinf[walloss,serving]'
```

The in-process Python API accepts the same observation dict as the server:

```python
from apxinf import AutoPolicy

policy = AutoPolicy.from_pretrained(
    "/path/to/wall-oss-0.5", norm_key="x2_normal", action_dim=7,
    image_keys=("observation/image", "observation/wrist_image"),
    state_key="observation/state",
)
result = policy.infer({
    "observation/image": base_rgb_uint8,
    "observation/wrist_image": wrist_rgb_uint8,
    "observation/state": state_f32,
    "prompt": "pick up the red block",
})
actions = result["actions"]  # float32 [10, 7]
```

To expose the same policy over WebSocket:

```bash
python python/apxinf/examples/openpi_server.py \
  --model-dir /path/to/wall-oss-0.5 \
  --action-dim 7 \
  --image-keys observation/image,observation/wrist_image \
  --state-key observation/state \
  --policy-options '{"norm_key":"x2_normal"}'
```

Add `"tactics":"/path/to/tactics.json"` to `--policy-options` to override a
checkpoint-local tuning database, for example after generating tactics for a
newer kernel build.

Images are RGB `uint8`; the policy owns the Qwen2.5-VL resize/patch/token
preprocessing and returns an `[10, action_dim]` action chunk through the normal
OpenPI-compatible `actions` response.

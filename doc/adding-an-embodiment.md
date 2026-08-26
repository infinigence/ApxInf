# Adding an Embodiment to ApxInf

A guide for serving a new robot through the OpenPI-compatible websocket API, and
for launching a server against an embodiment that already exists. Written after
the Unitree G1 port; every claim here is grounded in code that runs.

The companion doc for the *model* side is
[`doc/adding-a-new-model.md`](adding-a-new-model.md). That one is about weights
and kernels. This one is about the **wire contract**: which keys the client
sends, how state is routed, and what the action vector means when it comes back.

## Why this is a table and not a default

A pi05 checkpoint does not carry its wire contract. The weights fix the view
count and the action width; they say nothing about whether the base camera
arrives as `observation/image` or as `obs["images"]["cam_high"]`, whether the
state vector is discretized into the prompt or dropped, or whether the 32-wide
model output should be truncated to 7 or to 16.

Before presets, `Pi05Policy`'s LIBERO defaults applied to *every* checkpoint. A
G1 checkpoint served with no extra arguments ran LIBERO's keys, dropped state,
and skipped the G1 delta→absolute and 32→16 steps. Nothing failed. Nothing
logged. It looked exactly like a "model accuracy problem."

So the embodiment is a named, explicit launch flag on both sides — the same shape
OpenPI uses, so the two line up one-to-one:

```
openpi:  serve_policy.py --policy.config pi05_UnitreeG1_groundwire
apxinf:  pi05_openpi_websocket_server.py --robot unitree_g1
```

## Three namespaces, kept separate

Most confusion in this area comes from collapsing these. They are different
things and they never need to be equal.

| | what it is | where it lives | on the wire? |
|---|---|---|---|
| **view slots** | `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` | `VIEW_SLOTS` in `robots/presets.py` | never — the *order* is baked into the weights |
| **wire keys** | `observation/image`, `images/cam_high`, … | a preset's `slots` | yes — this is what the client sends |
| **training feature names** | LeRobot `config.json` `input_features` | the checkpoint | no |

A preset pairs each view slot with the wire key that fills it. `image_keys` is
order-significant: entry *i* is stacked into model view slot *i*. A tuple written
in the wrong order still stacks, still has the right shape, and silently feeds
the wrong camera to each slot — pairing every key with its slot makes that
reviewable instead of positional.

## Launching a server for an existing embodiment

`--robot` is the only flag that has to be right. It selects the wire keys, the
state routing, the action width, and the robot pre/post steps together.

```bash
# Franka Panda under LIBERO's key convention (the default)
python3 scripts/pi05_openpi_websocket_server.py \
  --model-dir "$APXINF_MODEL_DIR" \
  --robot franka_libero \
  --precision bf16 --host 0.0.0.0 --port 8000

# Unitree G1: 3 cameras, nested keys, state discretized into the prompt
python3 scripts/pi05_openpi_websocket_server.py \
  --model-dir "$G1_MODEL_DIR" \
  --robot unitree_g1 \
  --precision bf16 --host 0.0.0.0 --port 8000
```

`--robot` defaults to `franka_libero`, so an existing LIBERO launch command needs
no change. `python3 scripts/pi05_openpi_websocket_server.py --help` prints every
registered preset with its slot→key mapping.

### Confirm the contract before trusting the numbers

The server logs what it ended up serving and publishes the same thing in its
connect-time metadata. Read one of them **before** concluding anything about
accuracy:

```
INFO serving robot=unitree_g1 H=50 x D=32
     image_keys=['images/cam_high', 'images/cam_left_wrist', 'images/cam_right_wrist']
     state=state discrete_state=False
```

From a client:

```python
from openpi_client import websocket_client_policy as wcp
c = wcp.WebsocketClientPolicy(host="127.0.0.1", port=8000)
meta = c.get_server_metadata()
assert meta["robot"] == "unitree_g1"
assert meta["image_keys"] == [...]      # the keys you are actually sending
assert meta["discrete_state"] is True   # a joint-space robot needs this on
```

Published keys: `robot`, `robot_slots`, `model_type`, `image_keys`, `state_key`,
`prompt_key`, `discrete_state`, `state_normalized`, `action_horizon`,
`action_dim`, `model_action_dim`, `num_views`, `image_size`, `input_pipeline`,
`output_pipeline`. A key mismatch is invisible on the wire but obvious here.

### Per-field overrides

A deployed client may already speak a fixed dialect that does not match any
preset. Override the individual field rather than editing the preset:

```bash
python3 scripts/pi05_openpi_websocket_server.py \
  --model-dir "$APXINF_MODEL_DIR" --robot franka_libero \
  --image-keys 'observation/exterior_image_1_left,observation/wrist_image_left' \
  --state-key 'observation/joint_position'
```

Nested layouts are written as a slash path: `--image-keys
'images/cam_high,images/cam_left_wrist'`. `--discrete-state` /
`--no-discrete-state` and `--action-dim` override the remaining fields.

If you find yourself passing the same overrides twice, that is the signal to add
a preset.

### Serving fewer cameras than the checkpoint declares

```bash
python3 scripts/pi05_openpi_websocket_server.py \
  --model-dir "$APXINF_MODEL_DIR" --robot franka_libero \
  --image-keys 'observation/image' --num-views 1 \
  --precision bf16 --port 8000
```

This drops the trailing view slots at **load** time. It is numerically equivalent
to what OpenPI does by zero-padding the absent views and masking them out —
masked tokens consume no RoPE position (`positions = cumsum(input_mask) - 1`),
are excluded from attention (`valid_mask = mask[:, None, :] * mask[:, :, None]`),
and `embed_prefix` runs the same `PaliGemma.img` encoder per view with no
per-slot learned embedding. `num_views` only sizes the prefix; nothing
weight-shaped depends on it. The difference is that we skip the ~256 patch tokens
per absent view instead of computing and discarding them (measured 1.38× on a
2-view LIBERO checkpoint served with one camera, on Orin).

**`--num-views` is deliberately required to be explicit.** A short `--image-keys`
list on its own is an error, not an inference. Forgetting a camera key should
fail at startup, not quietly cost accuracy — which is the exact failure mode this
whole mechanism exists to prevent. It is a server-side launch flag, not part of
the wire protocol, so an unmodified openpi client is unaffected.

## Adding a new embodiment

### Step 0 — write down the contract

Before any code, get these seven facts from the OpenPI `TrainConfig` the
checkpoint was fine-tuned under (its `data_transforms` pair is the source of
truth):

1. camera wire keys, **in view-slot order**, and whether they are flat or nested;
2. the state wire key;
3. whether state is discretized into the prompt or dropped;
4. state width and layout;
5. model action width vs. deployable action width;
6. whether actions are deltas needing the current state to resolve;
7. any robot-space convention (joint sign flips, gripper mapping).

Facts 1–5 are just a table row. Facts 6–7 are the only ones that need code.

### Step 1 — decide whether you need robot steps at all

| the checkpoint… | what to write |
|---|---|
| emits absolute actions in robot space, at the deployable width | **nothing** — a preset row is the whole port |
| needs a fixed truncation only | a preset row with `action_dim=N` |
| emits deltas, or needs a sign/gripper convention | processing steps + an adapter (Steps 2–3) |

Take the cheap path when it is available. `franka_libero` is a table row and
nothing else.

### Step 2 — robot processing steps

`python/apxinf/apxinf/processors/robots/<robot>.py`. Each step is a
`ProcessorStep`: `dict -> dict` over the shared data dict, with a name. These
must be **model-agnostic** — they reference no policy symbols and vary only with
the robot body.

The data-dict contract:

* input chain reads `observation` / `prompt`, writes `rgb` (uint8 NHWC),
  `token_ids` (uint32), `noise`;
* output chain reads `normalized_actions`, writes `trimmed` then `actions`
  (unnormalized float32). `observation` is threaded through so an output step can
  read state.

Resolve every wire key through `lookup_key` / `has_key` / `set_key` from
`processors/transforms.py` rather than indexing the observation directly. They
resolve **flat first, then as a nested path**, so one flat key string addresses
either layout: `"observation/image"` hits `data["observation/image"]` if that key
literally exists, and `"images/cam_high"` walks
`data["images"]["cam_high"]` when it does not. `set_key` returns a copy — the
client's dict is never mutated.

Write each step as a no-op when its input is absent. `UnitreeG1DecodeState`
returns `data` unchanged when there is no state, so state-off serving degrades to
"this step does nothing" instead of raising deep in the pipeline.

### Step 3 — the adapter

`python/apxinf/apxinf/robots/<robot>.py`. One factory that loads the checkpoint
through the generic `Pi05Policy.from_pretrained` and then splices the robot steps
into the default pipelines:

```python
def build_<robot>_policy(model_dir, *, state_key=..., image_keys=..., **kwargs):
    base = Pi05Policy.from_pretrained(
        model_dir,
        image_keys=tuple(image_keys),
        action_dim=None,      # keep full model width; the encode step truncates
        state_key=state_key,
        **kwargs,
    )
    input_pipeline = base.input_pipeline.insert_before(
        "tokenize", ("<robot>_decode_state", <Robot>DecodeState(state_key))
    )
    output_pipeline = Pipeline([
        ("unnormalize", base.output_pipeline["unnormalize"]),
        ("<robot>_absolute", <Robot>AbsoluteActions(state_key)),
        ("<robot>_encode", <Robot>EncodeActions()),
    ])
    return Pi05Policy(base.model, input_pipeline=..., output_pipeline=..., ...)
```

Two orderings matter and are easy to get wrong:

* the decode-state step goes **before** `tokenize`, so discretized state (when
  on) and the delta→absolute output step both see the decoded state;
* unnormalize runs at **full model width**, before any truncation, so
  delta→absolute sees the whole action. Pass `action_dim=None` into
  `from_pretrained` and let the encode step truncate.

### Step 4 — register the preset

One entry in `python/apxinf/apxinf/robots/presets.py`. This is the whole
registration step — OpenPI's `training/config.py` equivalent.

```python
MY_ROBOT = RobotPreset(
    name="myarm_mydataset",
    slots=(
        ("base_0_rgb", "observation/image"),
        ("left_wrist_0_rgb", "observation/wrist_image"),
    ),
    state_key="observation/state",
    action_dim=7,               # None keeps full width when an encode step truncates
    discrete_state=False,       # False *drops* state entirely — not "keeps it raw"
    builder=build_my_robot_policy,   # omit for the stock policy
    summary="MyArm, MyDataset keys: 2 cameras, 7-dim action",
    builder_kwargs={},          # constants the builder always receives
)

ROBOT_PRESETS = {p.name: p for p in (FRANKA_LIBERO, UNITREE_G1, MY_ROBOT)}
```

`__post_init__` rejects a `slots` tuple that is not an in-order prefix of
`VIEW_SLOTS`, and rejects duplicate wire keys. A checkpoint fills view slots from
0 up, so there is no way to spell "wrist camera only" other than putting it in
slot 0.

Add an entry to `ROBOT_ALIASES` if a deployment already says something else in
its launch scripts; a rename should not break a running system.

### Naming: `<arm>_<key convention>`

The arm alone does not determine the contract. LIBERO and DROID are both Franka
Panda, yet LIBERO sends `observation/image` with a 7-dim EEF-delta action while
DROID sends `observation/exterior_image_1_left` with a different action space. So
a preset is named for the arm **and** the dataset convention whose keys it
implements: `franka_libero`, not `libero` (a benchmark, not a robot) and not
`franka` (ambiguous). A single-embodiment robot that owns its convention needs no
suffix — `unitree_g1`.

### Step 5 — tests

Add to `tests/test_robot_presets.py`. Four things are worth asserting, and they
run on CPU with a mock model — no GPU, no checkpoint:

1. **the preset matches the openpi transform it mirrors** — keys, order, widths,
   `discrete_state`, spelled out literally rather than derived;
2. **an unmodified openpi client round-trips** — `UnitreeG1ServingTest` starts
   the real `WebsocketPolicyServer` on a mock model and sends the observation the
   integrator's client actually sends;
3. **the wrong dialect fails loudly** — a LIBERO-shaped observation against your
   server must raise, naming the served `image_keys`;
4. **slot order is load-bearing** — reordering `image_keys` reorders the stacked
   views (`ImageSlotOrderTest` asserts this against the tensor, not the metadata).

```bash
python3 tests/test_robot_presets.py          # no GPU needed
```

## Verification recipe

Offline first, then one GPU pass. In order, because each step localizes a
different class of bug:

```bash
# 1. contract + plumbing, CPU only, mock model
python3 tests/test_robot_presets.py

# 2. real checkpoint, native contract
python3 scripts/pi05_openpi_websocket_server.py \
  --model-dir "$CKPT" --robot <preset> --precision bf16 --port 8000
#    -> read the "serving robot=..." line; assert every field

# 3. real transport, unmodified openpi client
#    -> assert actions.shape == (meta["action_horizon"], meta["action_dim"])
#    -> assert np.isfinite(actions).all()

# 4. wrong-dialect rejection: send another preset's keys; it must raise
```

Step 3's shape assertion should read the shape **off the metadata**, not off a
constant. A hard-coded expected shape tests your memory of the checkpoint rather
than the server.

For a transport-only check with no weights on disk, `--random-weights --robot
<preset>` serves the preset's key layout on synthetic weights. Its actions are
numerically meaningless, and it cannot honour `discrete_state=True` (the
synthetic tokenizer never reads state) — it warns at startup when you ask it to.

## Gotchas we hit

1. **`discrete_state=False` drops state; it does not pass it through raw.**
   There is no third state. A joint-space robot served with `discrete_state=False`
   loses its proprioception silently, which also makes any delta→absolute step a
   no-op (a delta cannot be resolved without current joint positions).
2. **`image_keys` order is the view slot order.** Wrong order → right shape,
   wrong cameras, no error. This is why presets pair keys with slot names.
3. **Truncate after unnormalize, not before.** Unnormalize at full model width so
   the delta→absolute step sees the whole action; let the robot encode step do
   the 32→16.
4. **`action_dim` in a preset is the *deployable* width, not the model's.**
   `None` is correct when a robot output step does the truncation itself —
   metadata still reports both (`action_dim` and `model_action_dim`).
5. **Nested keys need `lookup_key`, not `obs[k]`.** Both wire shapes exist in the
   wild and arrive as one string. Indexing directly works for LIBERO and raises
   for G1.
6. **A preset's `num_views` must equal the checkpoint's**, unless you pass
   `--num-views` to load fewer. `Pi05Policy.default_pipelines` rejects a mismatch
   and names the fix in the message.
7. **`--random-weights` cannot preview a `discrete_state=True` contract.** Its
   synthetic tokenizer ignores state, so the published metadata says
   `discrete_state=False` — truthful about that server, misleading as a preview.
   It warns; use a checkpoint for a real contract check.
8. **Metadata is the contract; assert it from the client.** Every silent failure
   in this area is visible in the connect-time metadata one line before it
   becomes an "accuracy problem."

## Rules

1. **The embodiment is explicit, never inferred.** No preset is guessed from the
   checkpoint, and `--num-views` is not inferred from `len(image_keys)`. An
   omission should fail at startup.
2. **Robot steps are model-agnostic; adapters are where they meet a policy.**
   `processors/robots/` imports no policy symbols; `robots/` does the assembly.
3. **Fail loudly with the served contract in the message.** Every error in this
   layer names what the server is actually serving, because the person reading it
   is comparing two dialects.
4. **A preset row is the cheapest port; prefer it.** Only write steps for a
   convention the model output genuinely does not satisfy.
5. **Overrides exist for deployed clients, not for new robots.** If the same
   `--image-keys` shows up twice, add a preset.

## Concrete example

The Unitree G1 port is the reference implementation — it exercises every step,
including the ones `franka_libero` skips:

- `python/apxinf/apxinf/robots/presets.py` — the registry and both presets
- `python/apxinf/apxinf/processors/robots/unitree_g1.py` — decode-state,
  delta→absolute, 32→16 encode
- `python/apxinf/apxinf/robots/unitree_g1.py` — the adapter that splices them in
- `python/apxinf/apxinf/processors/transforms.py` — `lookup_key`/`has_key`/`set_key`
- `tests/test_robot_presets.py` — contract, serving, and slot-order tests

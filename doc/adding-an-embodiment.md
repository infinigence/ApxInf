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

Before presets, `Pi05Policy` defaulted `image_keys` to LIBERO's two, so that
contract applied to *every* checkpoint. A G1 checkpoint served with no extra
arguments ran LIBERO's keys, dropped state, and skipped the G1 delta→absolute and
32→16 steps. Nothing failed. Nothing logged. It looked exactly like a "model
accuracy problem."

The model layer now names no wire keys at all — construct a policy without them
and it falls back to its own view slots, which no client sends, so a mismatch is
a `KeyError` naming both sides. The embodiment is a named, explicit launch flag
on both sides — the same shape OpenPI uses, so the two line up one-to-one:

```
openpi:  serve_policy.py --policy.config pi05_UnitreeG1_groundwire
apxinf:  pi05_openpi_websocket_server.py --robot unitree_g1
```

## Three namespaces, kept separate

Most confusion in this area comes from collapsing these. They are different
things and they never need to be equal.

| | what it is | where it lives | on the wire? |
|---|---|---|---|
| **view slots** | `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` | `VIEW_SLOTS` in `policies/base.py` | never — the *order* is baked into the weights |
| **wire keys** | `observation/image`, `images/cam_high`, … | a `Convention` in `conventions.py` | yes — this is what the client sends |
| **training feature names** | LeRobot `config.json` `input_features` | the checkpoint | no |

A preset pairs each view slot with the wire key that fills it. `image_keys` is
order-significant: entry *i* is stacked into model view slot *i*. A tuple written
in the wrong order still stacks, still has the right shape, and silently feeds
the wrong camera to each slot — pairing every key with its slot makes that
reviewable instead of positional.

The slot names live in `policies/base.py`, not in the preset table: they are
*model* vocabulary, so the robot layer reads them rather than restating them. The
policy layer holds **no** wire keys at all. Construct a policy without naming
cameras and it names them after its own view slots — a client sending
`observation/image` then gets a `KeyError` naming both sides, instead of the old
behaviour where LIBERO's two keys were the built-in default for every checkpoint.

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
INFO serving robot=unitree_g1 robot_steps=True H=50 x D=32
     image_keys=['images/cam_high', 'images/cam_left_wrist', 'images/cam_right_wrist']
     state=state discrete_state=False
```

From a client:

```python
from openpi_client import websocket_client_policy as wcp
c = wcp.WebsocketClientPolicy(host="127.0.0.1", port=8000)
meta = c.get_server_metadata()
assert meta["robot"] == "unitree_g1"
assert meta["robot_steps"] is True      # this robot's arithmetic is actually wired
assert meta["image_keys"] == [...]      # the keys you are actually sending
assert meta["discrete_state"] is True   # a joint-space robot needs this on
```

Published keys: `robot`, `robot_steps`, `robot_slots`, `model_type`,
`image_keys`, `state_key`, `prompt_key`, `discrete_state`, `state_normalized`,
`action_horizon`, `action_dim`, `model_action_dim`, `num_views`, `image_size`,
`input_pipeline`, `output_pipeline`. A key mismatch is invisible on the wire but
obvious here. `robot_steps` is `False` on a server that serves a preset's keys
without running its builder — see `--random-weights` below.

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
through `AutoPolicy` — so `config.json` decides which model it is, not this file
— and then wraps the robot steps *around* whatever chain that policy has:

```python
from ..policies.auto import AutoPolicy
from ..policies.base import ComposablePolicy, Policy

# state_key / image_keys are required and have no defaults — they belong to the
# dataset (Step 4), not to this body. A default here would mean the same arm,
# re-recorded under new keys, silently keeps serving the old ones.
def build_<robot>_policy(model_dir, *, state_key, image_keys, **load_kwargs):
    base = AutoPolicy.from_pretrained(
        model_dir,
        image_keys=tuple(image_keys),
        action_dim=None,      # keep full model width; the encode step truncates
        state_key=state_key,
        **load_kwargs,
    )
    if not isinstance(base, ComposablePolicy):
        raise TypeError(f"{type(base).__name__} has no with_adapter(); ...")
    return base.with_adapter(
        before=[("<robot>_decode_state", <Robot>DecodeState(state_key))],
        after=[("<robot>_absolute", <Robot>AbsoluteActions(state_key)),
               ("<robot>_encode",   <Robot>EncodeActions())],
        action_dim=ROBOT_DIM,             # the width the encode step leaves behind
        metadata={"robot": "<robot>"},
    )
```

The adapter names **no model class and no step inside the model's chain**. That
is deliberate: a robot's requirement is an *ordering* — its steps run outside the
model's, in both directions — not a claim about what those steps are called.
`Pipeline.prepend`/`append` are the only editing verbs that express ordering
without a name, and `with_adapter`
([`ComposablePolicy`](../python/apxinf/apxinf/policies/base.py)) is how a policy
exposes them. Reaching for `insert_before("tokenize", ...)` instead would make
your robot depend on one model's private vocabulary, and on that model class by
import.

Two orderings matter and are easy to get wrong:

* the decode-state step goes **before** the model's whole input chain, so
  discretized state (when on) and the delta→absolute output step both see the
  decoded state;
* unnormalize runs at **full model width**, before any truncation, so
  delta→absolute sees the whole action. Pass `action_dim=None` into the loader
  and let the encode step truncate.

Declare `action_dim=` only for the width a step you appended actually produces.
Without the truncating step there is nothing to claim — inherit the model's own
width instead of advertising one nothing emits.

### Step 4 — register the preset

Two entries: a `Convention` in `python/apxinf/apxinf/conventions.py` and a body +
pairing in `python/apxinf/apxinf/robots/presets.py`. That is the whole
registration step — OpenPI's `training/config.py` equivalent.

A preset is two halves, because the arm and the key convention vary
independently. An `Embodiment` is the **body**: camera count, deployable action
width, which pre/post steps its actions need. It survives a change of dataset. A
`Convention` is a **dataset's recording dialect**: the wire keys and the state
routing that follows from how the data was recorded. It survives a change of
robot, which is why it lives in its own module and not under `robots/` — a
dialect that lived in one robot's file could only ever be that robot's.

```python
# apxinf/conventions.py — the dialect, independent of any arm
MY_DATASET_KEYS = Convention(
    name="mydataset",
    image_keys=("observation/image", "observation/wrist_image"),  # in view-slot order
    state_key="observation/state",
    discrete_state=False,       # False *drops* state entirely — not "keeps it raw"
)

# apxinf/robots/presets.py — the body, and the pairing
MY_ARM = Embodiment(
    name="myarm",
    num_cameras=2,
    action_dim=7,               # None keeps full width when an encode step truncates
    state_dim=8,                # how wide norm_stats["state"] has to be
    action_width=7,             # how wide norm_stats["actions"] has to be
    builder=build_my_robot_policy,   # omit for the stock policy
    builder_kwargs={},          # constants the builder always receives
)

MY_ROBOT = RobotPreset(
    name="myarm_mydataset",
    embodiment=MY_ARM,
    convention=conventions.MY_DATASET_KEYS,
    summary="MyArm, MyDataset keys: 2 cameras, 7-dim action",
)

ROBOT_PRESETS = {p.name: p for p in (FRANKA_LIBERO, UNITREE_G1, MY_ROBOT)}
```

Serving the same arm under a second dataset's keys is then one more `Convention`
and one more pairing — no builder edit, no duplicated action width.

If your robot lives in **your own package**, do not patch either file. Call
`register_convention(...)` and `register_robot_preset(..., aliases=(...))` at
your module scope; importing that module before the server starts is enough to
make it `--robot <name>`. Re-registering an existing name needs an explicit
`replace=True`, and an alias may never shadow a canonical preset name — a silent
overwrite would change what a launch command already in production resolves to.

Only the *pairing* is deployable, so `--robot` stays a single flag over
`ROBOT_PRESETS` rather than becoming `--robot` + `--convention`. Separate flags
would let an operator spell a combination nobody ever recorded, and a body served
under a convention it was not recorded on is exactly the silent mismatch this
mechanism exists to prevent.

Validation runs when the module loads. `Convention.__post_init__` rejects
duplicate wire keys and more keys than there are view slots;
`Embodiment.__post_init__` rejects a camera count outside `1..len(VIEW_SLOTS)`;
`RobotPreset.__post_init__` rejects a convention whose camera count disagrees
with the body's. Slot names are *derived* from `image_keys` in order, so "base +
right wrist" is not expressible at all — a checkpoint fills view slots from 0 up,
and the only way to spell "wrist camera only" is to put it in slot 0.

**Fill in `state_dim` and `action_width`.** They are the only thing that lets
the startup preflight (Step 6) tell your robot's `norm_stats.json` from some
other robot's — leave them `None` and that check silently does nothing. They are
two fields rather than one because a robot's state and action spaces need not
match: LIBERO is 8-dim state / 7-dim EEF-delta action. `action_width` exists
separately from `action_dim` because the two answer different questions —
`action_dim` is a *loading* knob ("trim the model down to this"), `action_width`
is a fact about the body ("this is how many columns its actions have"). A preset
whose encode step owns the 32→N truncation sets `action_dim=None`, and `None` is
not a width: G1 declares `action_dim=None, action_width=16`.

**Set `norm_dtype="float64"` in `builder_kwargs` if the checkpoint came from
openpi.** openpi parses `norm_stats.json` into float64 and never demotes it, so
its whole chain runs in float64 whatever dtype the robot sends. Ours follows the
input. The difference is ~1e-7 and invisible on the output side, but a robot that
discretizes state into its prompt compares the normalized value against bin edges
1/128 apart, and an element near an edge crosses it — different prompt, different
tokens, different rollout. It is a per-body flag rather than a global default so
that presets nobody has re-benchmarked keep their existing numbers exactly.

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

The `Embodiment`/`Convention` split is that naming rule made structural: the two
halves of the name are the two halves of the preset, and `franka_droid` would
reuse `FRANKA` rather than restating its width and builder.

### Step 5 — tests

Add to `tests/test_robot_presets.py`. Five things are worth asserting, and they
run on CPU with a mock model — no GPU, no checkpoint:

1. **the preset matches the openpi transform it mirrors** — keys, order, widths,
   `discrete_state`, spelled out literally rather than derived;
2. **an unmodified openpi client round-trips** — `UnitreeG1ServingTest` starts
   the real `WebsocketPolicyServer` on a mock model and sends the observation the
   integrator's client actually sends;
3. **the wrong dialect fails loudly** — a LIBERO-shaped observation against your
   server must raise, naming the served `image_keys`;
4. **slot order is load-bearing** — reordering `image_keys` reorders the stacked
   views (`ImageSlotOrderTest` asserts this against the tensor, not the metadata);
5. **the contract you publish is the one you serve** — `BuildRobotPolicyTest`
   asserts the resolved keys and `robot_steps` reach the builder and the metadata,
   and `SyntheticContractTest` asserts a checkpoint-free server names what it
   cannot honour instead of passing for the real embodiment.

```bash
python3 tests/test_robot_presets.py          # no GPU needed
```

### Step 6 — check the preflight refuses the wrong checkpoint

A preset and a checkpoint directory are two independent claims about the same
robot, and until the preflight existed nothing compared them. `--robot myarm`
pointed at another robot's checkpoint loads, serves, and returns well-shaped
actions in a plausible numeric range that mean nothing; the only symptom is that
the model "got worse". Run it against your checkpoint before you serve it:

```bash
python3 scripts/openpi_metadata_to_apxinf.py --model-dir "$CKPT" --robot <preset>
```

It reads the checkpoint's `norm_stats.json` widths and tokenizer, and — when the
checkpoint carries openpi's `metadata.pt` — cross-checks your preset field by
field against openpi's own serialized `TrainConfig` (camera wire keys,
`discrete_state_input`, `adapt_to_pi`, `use_delta_joint_actions`,
`action_horizon`, `max_token_len`). Exit status is 1 on any fatal finding, so it
can gate a deployment. The same directory checks run inside
`pi05_openpi_websocket_server.py` **before** the weights load, and refuse to
start; `--skip-preflight` overrides that, and exists only to reproduce a
known-bad configuration deliberately.

Add cases to `tests/test_preflight.py` if your robot has a width rule the generic
checks miss. The check that matters most is the boring one: a *correct*
checkpoint must produce zero fatal findings, or operators will learn to pass
`--skip-preflight` by reflex.

## Verification recipe

Offline first, then one GPU pass. In order, because each step localizes a
different class of bug:

```bash
# 1. contract + plumbing, CPU only, mock model
python3 tests/test_robot_presets.py
python3 -m pytest tests/test_preflight.py

# 2. does the checkpoint agree with the preset? (seconds; no weights loaded)
python3 scripts/openpi_metadata_to_apxinf.py --model-dir "$CKPT" --robot <preset>

# 3. real checkpoint, native contract
python3 scripts/pi05_openpi_websocket_server.py \
  --model-dir "$CKPT" --robot <preset> --precision bf16 --port 8000
#    -> read the "serving robot=..." line; assert every field

# 4. real transport, unmodified openpi client
#    -> assert actions.shape == (meta["action_horizon"], meta["action_dim"])
#    -> assert np.isfinite(actions).all()

# 5. wrong-dialect rejection: send another preset's keys; it must raise
```

Step 4's shape assertion should read the shape **off the metadata**, not off a
constant. A hard-coded expected shape tests your memory of the checkpoint rather
than the server.

For a transport-only check with no weights on disk, `--random-weights --robot
<preset>` serves the preset's key layout and view count on synthetic weights.
That is all it serves: the actions are numerically meaningless, and the preset's
`builder` never runs, so none of its robot pre/post steps is wired. It publishes
`robot_steps=false` and warns once at startup naming every gap — `discrete_state`
(the synthetic tokenizer never reads state) and, for a robot-step preset, the
skipped factory and the action width that came out of the model instead of out of
the encode step.

## Gotchas we hit

1. **`discrete_state=False` drops state; it does not pass it through raw.**
   There is no third state. A joint-space robot served with `discrete_state=False`
   loses its proprioception silently, which also makes any delta→absolute step a
   no-op (a delta cannot be resolved without current joint positions). This is
   also why `state_key` is **required whenever it is read**: a wrong camera key
   raises on the first inference, but a missing state key is silent, so the
   policy refuses to be built with `discrete_state=True` and no key.
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
7. **`--random-weights` previews the wire layer only.** Its synthetic tokenizer
   ignores state, so the published metadata says `discrete_state=False`, and it
   never runs `preset.builder`, so no robot pre/post step is wired and the action
   width is the model's rather than the encode step's. Truthful about that server,
   misleading as a preview — hence `robot_steps=false` in the metadata and a
   startup warning per gap. Use a checkpoint for a real contract check.
8. **Metadata is the contract; assert it from the client.** Every silent failure
   in this area is visible in the connect-time metadata one line before it
   becomes an "accuracy problem."

## Rules

1. **The embodiment is explicit, never inferred.** No preset is guessed from the
   checkpoint, and `--num-views` is not inferred from `len(image_keys)`. An
   omission should fail at startup.
2. **Robot steps are model-agnostic; adapters are where they meet a policy.**
   `processors/robots/` imports no policy symbols; `robots/` does the assembly.
   Neither holds a wire key: those are a dataset's, and live in `conventions.py`.
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
- `python/apxinf/apxinf/conventions.py` — the LIBERO and G1 wire dialects
- `python/apxinf/apxinf/processors/robots/unitree_g1.py` — decode-state,
  delta→absolute, 32→16 encode
- `python/apxinf/apxinf/robots/unitree_g1.py` — the adapter that splices them in
- `python/apxinf/apxinf/processors/transforms.py` — `lookup_key`/`has_key`/`set_key`
- `tests/test_robot_presets.py` — contract, serving, and slot-order tests

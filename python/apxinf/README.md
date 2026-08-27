# apxinf (Python frontend)

Pure-Python processor library + the **L2** policy layer for ApxInf VLA runtimes.
The native PI0.5 bare-model L1 inference binding lives in
the [`apxinf-py`](../../crates/apxinf-py) PyO3 crate; `apxinf` re-exports it as
`apxinf.Model` so you never import `apxinf_py` directly.

## Layout

```
apxinf/
├── processors/   pure-numpy pre/post steps + Pipeline (offline, no GPU/Rust)
├── policies/     the L2 layer — stable machinery + volatile per-model impls
│   ├── base.py       Policy + BareModel contracts (structural Protocols)
│   ├── registry.py   model_type -> policy-class registry
│   ├── auto.py       AutoPolicy: checkpoint -> concrete policy by config type
│   └── impls/        concrete per-model policies (the part that grows)
│       ├── pi05.py       Pi05Policy (native ApxInf runtime)
│       └── dm05.py       Dm05Policy (native Rust VlaRuntime frontend)
├── adapters/     downstream: expose a Policy through a foreign API (lazy imports)
│   └── lerobot.py   ApxInfPolicy — drop-in policy for a lerobot control loop
└── __init__.py   facade: Model (lazy), concrete policies, AutoPolicy, steps
```

**Adding a policy adapter:** after its model runtime is maintained, add
`apxinf/policies/impls/<name>.py` following `pi05.py` (decorate the class with
`@register_policy("<name>")`), then re-export it from
`apxinf/policies/impls/__init__.py`. `AutoPolicy` picks it up automatically. A
complete model port follows [`doc/porting-workflow.md`](../../doc/porting-workflow.md):
new VLA architectures ordinarily start in `crates/apxinf-model/src/<model>/`
and implement Rust `VlaRuntime`; adding only a Python policy is not completion
of that workflow.

## Layers

- **Processor steps** (`apxinf.processors`) — each an independently-callable
  `ProcessorStep`: `ParseImage`, `ResizeWithPad`, `PromptTokenizer`,
  `Normalizer`/`Unnormalizer`, `GaussianNoise`, chained by `Pipeline`. No GPU /
  no Rust dependency; unit-tests run offline. sentencepiece is imported lazily
  by the tokenizer only.
- **L2 policies** (`apxinf.Pi05Policy`, `apxinf.Dm05Policy`, or
  `apxinf.AutoPolicy`) — expose one stable observation-to-actions contract.
  PI0.5 composes a pre pipeline + native L1 handle + post pipeline. DM05 owns
  exact Gemma3 preprocessing/postprocessing around its native L1 handle.
  `import apxinf` stays CUDA-free; only explicit runtime construction imports a
  backend.

## Domains

L1 (the binding) returns **normalized-domain** actions; policies return
the **unnormalized-domain** chunk. `infer` also returns `normalized_actions`,
so the layering invariant `L2 minus unnormalize == L1` is directly checkable.

## Standalone steps

```python
from apxinf.processors import ResizeWithPad, PromptTokenizer, Unnormalizer

img224 = ResizeWithPad(224)(raw_hwc_uint8)
token_ids = PromptTokenizer("tokenizer.model")("pick up the block")
actions = Unnormalizer.from_norm_stats("model_dir", dims=7)(normalized[:, :7])
```

## Adding a processor implementation

Two layers of `ProcessorStep` live here, and they extend differently:

- **Natural-signature steps** (`resize` / `tokenize` / `normalize` / `noise`) —
  each `__call__` takes its natural input (an image, a prompt, an action array,
  nothing). This is where new *implementations* go.
- **dict→dict transforms** (`processors/transforms.py`: `ImageStack` /
  `Tokenize` / `SampleNoise` / `Trim` / `Unnormalize`) — thin adapters that read
  a few data-dict keys, **delegate to an injected natural step**, and write an
  output key. They define a *role* (which key they produce), not an
  implementation.

**A new implementation should not touch `transforms.py`.** The swap seam is
dependency injection at pipeline-assembly time, not the transform classes:

```python
# subclass the natural contract (same signature, e.g. noise: () -> [H, D])
class BetaNoise(ProcessorStep): ...

# inject it — the transform (SampleNoise) is unchanged
input_pipeline, output_pipeline = Pi05Policy.default_pipelines(model, ..., noise=BetaNoise(H, D))
# or insert/swap it on an existing pipeline (copy-on-write):
input_pipeline = input_pipeline.insert_after(
    "tokenize", ("sample_noise", SampleNoise(BetaNoise(H, D)))
)
input_pipeline = input_pipeline.override("sample_noise", ...)   # tweak PARAMS only
```

The default pipeline has no `sample_noise` step: omitting external noise lets
the runtime generate it directly in the stable device buffer.

**Organize growth by implementation family, not by transform key.** When a
category earns a second implementation, promote its file to a package
(`noise.py → noise/`, with `__init__.py` re-exporting so
`from apxinf.processors import GaussianNoise` keeps working) — do **not** create
directories named after the data-dict keys (`rgb` / `token_ids` / `actions`).
Keys are a runtime contract, not a taxonomy: `normalize` alone serves two roles
(action `Unnormalize` **and** state normalization inside `Tokenize`), `Trim` has
no natural-step backing at all, and `ImageStack` wraps a whole `parse → resize`
sub-pipeline — so keys map neither 1:1 to files nor to categories.

**Config-driven selection**, if it is ever needed (e.g. `config.json` naming
`noise.type = "beta"`), should add a per-category registry mirroring
`apxinf.policies.registry` (`@register_noise("gaussian")` / `get_noise(...)`) —
not a key-named directory tree. Like the policy registry, add it only once a
real second implementation and a real need to select it exist; do not abstract
ahead of the second example.

## Policy

Concrete policies and the generic dispatcher all satisfy the `Policy` contract:

```python
from apxinf import AutoPolicy, Dm05Policy, Pi05Policy

# Generic: read config.json's model type and dispatch to the right class.
policy = AutoPolicy.from_pretrained("model_dir", precision="bf16", action_dim=7)

# Concrete: when you need model-specific knobs.
policy = Pi05Policy.from_pretrained("model_dir", precision="bf16", action_dim=7)

result = policy.infer({
    "observation/image": base_rgb,
    "observation/wrist_image": wrist_rgb,
    "observation/state": state,   # currently dropped (see below)
    "prompt": "pick up the block",
})
result["actions"]   # unnormalized float32 [horizon, action_dim]
result["timing"]    # {"model_ms": ..., "total_ms": ...}
```

For `config.json:model_type == "dm05"`, `AutoPolicy` selects `Dm05Policy`.
DM05-libero is a native Rust `VlaRuntime` port: model structure, BF16 weights,
vision/language/action execution, the ten-step denoising schedule, and CUDA
Graph state live under `crates/apxinf-model/src/dm05/`. Python retains the
pinned Gemma3 processor, OpenCV resize, quantile unnormalization, and serial
HTTP wire. The deployment contract is fixed to two ordered images, a required
8D Franka state field, and a 10x7 action chunk. State is validated but not
consumed because the pinned checkpoint uses `add_state=False`. OpenDM is a
private reference, not a runtime dependency or fallback. See the root README's
**DM05-libero native HTTP deployment** section for the immutable checkpoint,
build command, request schema, and qualification status.

For bare-model (L1) use, the binding is reachable as `apxinf.Model`:

```python
from apxinf import Model
model = Model.load("pi05", "model.safetensors", precision="bf16")
model.infer_rgb(rgb_u8, "nhwc", token_ids)          # internal device sampling
model.infer_rgb(rgb_u8, "nhwc", token_ids, noise)   # exact external noise
```

Any default step is replaceable at construction (`image_pipeline=`,
`tokenizer=`, `unnormalizer=`, `noise=`) for a custom high-performance
implementation.

## Policy contract

`apxinf.Policy` is a structural `typing.Protocol` every L2 policy satisfies:
`metadata`, `action_dim`, `action_horizon`, `infer(obs, noise=None) -> dict`, `close()`.
`infer` guarantees the `actions` and `timing` keys across all policies. It's the
anchor point for future models (a `GrootPolicy` satisfies the same contract),
for `AutoPolicy` dispatch, and for a future lerobot adaptor — no inheritance
required, structural typing only.

## lerobot interop

`apxinf.adapters.lerobot.ApxInfPolicy` wraps any `Policy` in the surface a
**hand-written lerobot control loop** calls, so a lerobot user keeps their robot,
cameras, dataset-feature plumbing and action dispatch and swaps only the policy:

```python
from apxinf.adapters.lerobot import ApxInfPolicy

model = ApxInfPolicy.from_pretrained(ckpt_dir, device="cuda:0", precision="bf16")
preprocess, postprocess = model.make_pre_post_processors()

obs   = robot.get_observation()                                   # lerobot's
frame = model.build_inference_frame(obs, ds_features=feats, task=task)
action = postprocess(model.select_action(preprocess(frame)))
robot.send_action(make_robot_action(action, feats))               # lerobot's
```

See `examples/lerobot_loop.py` (runs with `--mock-robot`, no hardware needed).

**Supported:** lerobot `robots`/`cameras`/`teleoperators`, `hw_to_dataset_features`,
`build_dataset_frame`, `make_robot_action`, `LeRobotDataset` recording,
hand-written gym eval loops, chunk-at-a-time via `predict_action_chunk`.

**Not supported:** the `lerobot-eval` / `lerobot-rollout` CLIs and `make_policy`
(they resolve a policy class out of lerobot's registry — that needs a
`lerobot_policy_*` plugin distribution); training / fine-tuning / PEFT
(structurally impossible — the engine has no autograd); RTC inference; lerobot's
async-inference server (`apxinf.serving` covers that need); `torch.compile`.

**Two seams, one default.** lerobot's `build_inference_frame` is
`build_dataset_frame` (numpy `HWC` `uint8`) followed by
`prepare_observation_for_inference` (H2D, `/255`, `CHW`, batch dim). The adapter's
`build_inference_frame` runs **only the first** — that layer is already what
`Policy.infer` eats. A frame that went through the second is also accepted and
undone, but costs a device→host copy per tick, so the numpy seam is the default.

**Whose pre/post runs: ours.** apxinf's `Pipeline` does resize, tokenize, and
unnormalize; the model runtime generates prior noise unless it is supplied
explicitly. lerobot splits the same work differently (resize and prior noise live
*inside* its policy; normalize lives in its processor pipeline), so pipelines from its
`make_pre_post_processors` are **not** interchangeable with ours — feeding their
output here would drop resize and double-normalize. `ApxInfPolicy`'s own
`make_pre_post_processors()` therefore returns pass-throughs that preserve the
call shape and nothing more.

Install with `pip install -e '.[lerobot]'` (adds torch for the tensor boundary;
lerobot itself is not pinned — bring your own version).

## Camera views

pi05 has `num_views` camera slots (3 for the LIBERO checkpoint), but a task may
supply fewer (LIBERO uses base + one wrist). The policy zero-fills the absent
slots — the openpi convention for masked/missing cameras — so 2 cameras drive a
3-slot model. Passing more keys than slots is an error; disable padding with
`pad_missing_views=False`.

## State gap (reserved, not wired)

pi05 expects proprioception injected as a **discretized state** spliced into the
prompt (`build_prompt(..., discrete_state=True)`, aligned with Rust
`pi05_prompt`). The current serving link omits it: `observation/state` is read
but **dropped**. The interface slot exists (construct the tokenizer with
`discrete_state=True`); the default path matches today's behavior.

## Tests

```bash
pip install -e '.[test]'
pytest tests/        # offline; tokenizer + real-model tests skip without a checkpoint
```

Tokenizer-encode tests need a SentencePiece model (`APXINF_TOKENIZER` or
`APXINF_PI05_MODEL_DIR`); the real-model layering test needs a CUDA `apxinf_py`
build plus `APXINF_PI05_MODEL_DIR`.

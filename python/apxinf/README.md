# apxinf (Python frontend)

Pure-Python (numpy/PIL/sentencepiece) processor library + the **L2** policy
layer for the ApxInf VLA runtime. The bare-model L1 inference binding lives in
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
│       └── pi05.py       Pi05Policy (registered as "pi05")
└── __init__.py   facade: Model (lazy), Pi05Policy, AutoPolicy, Policy, steps
```

**Adding a model:** drop `apxinf/policies/impls/<name>.py` following `pi05.py`
(decorate the class with `@register_policy("<name>")`), then re-export it from
`apxinf/policies/impls/__init__.py`. `AutoPolicy` picks it up automatically.

## Layers

- **Processor steps** (`apxinf.processors`) — each an independently-callable
  `ProcessorStep`: `ParseImage`, `ResizeWithPad`, `PromptTokenizer`,
  `Normalizer`/`Unnormalizer`, `GaussianNoise`, chained by `Pipeline`. No GPU /
  no Rust dependency; unit-tests run offline. sentencepiece is imported lazily
  by the tokenizer only.
- **L2 policies** (`apxinf.Pi05Policy` / `apxinf.AutoPolicy`) — compose a pre
  pipeline + a bare-model handle (L1 `infer_rgb` by default) + a post unnormalize
  step into one `infer(obs_dict) -> {actions, timing, ...}` call. `import apxinf`
  stays CUDA-free; only `apxinf.Model` and a policy's `from_pretrained` pull in
  `apxinf_py`.

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
# or swap on an existing pipeline (copy-on-write):
input_pipeline = input_pipeline.replace("sample_noise", SampleNoise(BetaNoise(H, D)))
input_pipeline = input_pipeline.override("sample_noise", ...)   # tweak PARAMS only
```

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

Two entry points, both returning something that satisfies the `Policy` contract:

```python
from apxinf import AutoPolicy, Pi05Policy

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

For bare-model (L1) use, the binding is reachable as `apxinf.Model`:

```python
from apxinf import Model
model = Model.load("pi05", "model.safetensors", precision="bf16")
model.infer_rgb(rgb_u8, "nhwc", token_ids, noise)   # normalized-domain action
```

Any default step is replaceable at construction (`image_pipeline=`,
`tokenizer=`, `unnormalizer=`, `noise=`) for a custom high-performance
implementation.

## Policy contract

`apxinf.Policy` is a structural `typing.Protocol` every L2 policy satisfies:
`metadata`, `action_dim`, `action_horizon`, `infer(obs) -> dict`, `close()`.
`infer` guarantees the `actions` and `timing` keys across all policies. It's the
anchor point for future models (a `GrootPolicy` satisfies the same contract),
for `AutoPolicy` dispatch, and for a future lerobot adaptor — no inheritance
required, structural typing only.

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

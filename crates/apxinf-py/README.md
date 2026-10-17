# apxinf-py

PyO3 bindings for the ApxInf VLA runtime — the **binding layer** of the ApxInf
Python frontend.

`import apxinf_py` gives in-process, numpy-in / numpy-out access to bare-model
inference. The policy-facing tier is L1; tokenize, normalize, and resize belong
to the `apxinf` Python package:

- **PI0.5 L1** `ModelRunner.infer_rgb(...)` — resized RGB in; vision patchification
  runs inside the Rust CUDA graph.
- **WallOSS L1** `ModelRunner.infer_rgb(...)` — resized RGB in; Qwen2.5-VL
  normalization, temporal patchification, and spatial-merge ordering run in
  the captured CUDA path.
- **WallOSS custom-processor bridge** `ModelRunner._infer_patches(...)` — canonical
  patches/tokens/action mask in, preserving user-defined Python processors.

Returns `float32` `[action_horizon, action_dim]` in the **normalized** domain.

> **L0** (pre-computed patches in, equivalent to a Rust `Observation(Patches)`)
> is not an end-user API. It is reachable only under the private name
> `ModelRunner._infer_patches`, for policy implementations and parity tests.

## Build

The VLA runtimes are CUDA-only, so real inference needs the `cuda` feature and a
CUDA machine (e.g. Thor). Without CUDA the module still imports and reports its
shape contract, but model loading errors.

```sh
# In this crate directory. --features cuda is additive to pyproject's
# extension-module feature.
CARGO_TARGET_DIR=../../target/wheel maturin build --release --features cuda --auditwheel skip
pip install --force-reinstall ../../target/wheel/wheels/apxinf_py-*.whl
```

On Thor set the usual environment (see repo memory): `APXINF_CUDA_ARCH=sm_110`,
`CUDA_PATH`, proxy, and checkpoint path.

Compile-check only (no CUDA, no link):

```sh
cargo check -p apxinf-py
```

Device parsing also accepts `metal:N` and its `mlx:N` alias; the canonical
reported name is `metal:N`. Parsing a device does not register an implementation:
loading still requires the MLX feature and a family with a native MLX path.
The existing CUDA-only VLA families do not gain Metal support through this alias.
`ModelRunner` is thread-bound (`#[pyclass(unsendable)]`); use it from the Python
thread that created it, including when its tensors own lazy MLX arrays.

## Usage

```python
import numpy as np
import apxinf_py

model = apxinf_py.ModelRunner.load("pi05", "/path/to/checkpoint", device="cuda:0", model_variant="bf16")

rgb = np.zeros((model.num_views, model.image_size, model.image_size, 3), np.uint8)
tokens = np.zeros(16, np.uint32)
noise = np.random.default_rng(0).standard_normal(
    (model.action_horizon, model.action_dim), dtype=np.float32
)

action = model.infer_rgb(rgb, "nhwc", tokens, noise)  # -> [horizon, dim] float32
```

## Decision requests

`DecisionRunner` supports both encoded-prompt and schema requests on the same
loaded handle. `decide(token_ids, candidate_count)` keeps the D3 output contract:
`{"selected_index": int, "scores": [...]}`.

For a schema-only model such as Clef, build with `--features mlx` and use
`decide_schema(request_json)`:

```python
import json
import apxinf_py

runner = apxinf_py.DecisionRunner.load("/path/to/clef", device="metal:0")
request = {
    "state": {"signal": "red"},
    "questions": {
        "action": {
            "type": "choice",
            "instructions": "Choose an action",
            "criteria": {"go": "Proceed", "stay": "Wait"},
        }
    },
}
result = runner.decide_schema(json.dumps(request))
answer = result["answers"][0]
selected = answer["scores"][answer["selected_index"]]
```

The JSON request uses the shared `DecisionSchema` parser, which rejects duplicate
question IDs and unknown fields. The returned `answers` list preserves question
order. Each answer contains `question_id`, `selected_index`, and `scores` with
`code`, `token_id`, and `probability`. Schema options have `token_id=None`.
Scores retain their option order and per-question probabilities. Invalid requests
raise `ValueError`; model errors raise `RuntimeError`. The loaded family decides
which input forms it supports. Clef does not accept encoded D3 prompts or tuning
options.

## Tests

### CPU binding checks

The Rust unit tests exercise both Python decision methods with a stub
`DecisionRuntime`, without loading a model or using a GPU. They check schema
dispatch, question order, grouped probabilities, errors, and the D3 output shape.
Use a Python interpreter supported by PyO3 0.22 with NumPy installed. Keep
`extension-module` disabled so the test executable links libpython:

```sh
PYO3_PYTHON=/path/to/python3.11 cargo test -p apxinf-py --lib
```

### CUDA smoke tests

Python smoke + L0/L1 consistency tests live in `tests/` and are gated on
environment variables (they skip cleanly without CUDA / a checkpoint):

```sh
pip install pytest numpy
APXINF_PI05_CHECKPOINT=/path/to/checkpoint APXINF_PI05_MODEL_VARIANT=bf16 \
  pytest crates/apxinf-py/tests
```

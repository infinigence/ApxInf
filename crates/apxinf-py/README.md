# apxinf-py

PyO3 bindings for the ApxInf VLA runtime — the **binding layer** of the ApxInf
Python frontend.

`import apxinf_py` gives in-process, numpy-in / numpy-out access to bare-model
inference. The policy-facing tier is L1; tokenize, normalize, and resize belong
to the `apxinf` Python package:

- **PI0.5 L1** `Model.infer_rgb(...)` — resized RGB in; vision patchification
  runs inside the Rust CUDA graph.
- **WallOSS policy bridge** `Model._infer_patches(...)` — Qwen2.5-VL canonical
  patches/tokens/action mask in. Raw-observation preprocessing remains isolated
  in `apxinf.policies.impls.walloss`.

Returns `float32` `[action_horizon, action_dim]` in the **normalized** domain.

> **L0** (pre-computed patches in, equivalent to a Rust `Observation(Patches)`)
> is not an end-user API. It is reachable only under the private name
> `Model._infer_patches`, for policy implementations and parity tests.

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

## Usage

```python
import numpy as np
import apxinf_py

model = apxinf_py.Model.load("pi05", "/path/to/checkpoint", device="cuda:0", precision="bf16")

rgb = np.zeros((model.num_views, model.image_size, model.image_size, 3), np.uint8)
tokens = np.zeros(16, np.uint32)
noise = np.random.default_rng(0).standard_normal(
    (model.action_horizon, model.action_dim), dtype=np.float32
)

action = model.infer_rgb(rgb, "nhwc", tokens, noise)  # -> [horizon, dim] float32
```

## Tests

Python smoke + L0/L1 consistency tests live in `tests/` and are gated on
environment variables (they skip cleanly without CUDA / a checkpoint):

```sh
pip install pytest numpy
APXINF_PI05_CHECKPOINT=/path/to/checkpoint APXINF_PI05_PRECISION=bf16 \
  pytest crates/apxinf-py/tests
```

Rust-side unit tests (device/precision/layout parsing, config fallback) need no
CUDA:

```sh
cargo test -p apxinf-py
```

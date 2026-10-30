# Tiny PI0.5 CPU reference fixture

This tracked fixture contains randomly generated FP32 weights (~28 KB), two
fixed RGB/token/provided-latent requests, and independent PyTorch outputs.
No trained checkpoint, camera recording or user data is included. The model has
two vision/language/action layers, language width 8, action width 6, GQA with two
query heads and one KV head, two 4x4 views, horizon 3 and three flow steps.
Token counts differ (3 and 2). The tests pin this geometry explicitly;
`config.json` exercises family detection, not production-size loading defaults.

`cases.json` records vision output, embedded prefix, K/V for every language layer,
every Euler step, and final actions. The oracle uses raw checkpoint matrices,
PyTorch convolution/SDPA, and unchanged norm/RoPE/time helpers from OpenPI
revision `215abfb217dbac7d5f1273282331b9b1866c0479`. It does not call ApxInf.
This is a small functional oracle, not an execution of the full OpenPI model.
`provenance.json` pins the data hashes and generator environment.

The integration and intermediate-stage tests run without Python, network access,
environment variables or a GPU:

```sh
cargo test -p apxinf-model --test cpu_pi05
cargo test -p apxinf-model --lib cpu_pi05::
```

Both use elementwise `atol=rtol=2e-5`. This tiny-fixture tolerance is distinct
from full-checkpoint CPU/OpenPI (`2e-4`) and GPU precision tolerances.

To regenerate, install PyTorch 2.14.1 and safetensors in a development environment
and use an existing checkout of the pinned OpenPI revision:

```sh
python crates/apxinf-model/tests/fixtures/pi05_cpu/generate.py \
  --openpi /path/to/openpi \
  --output devlocal/pi05-cpu-reference/regenerated-tiny
```

The generator verifies SHA256 of the two OpenPI source files before executing
selected definitions. Compare regenerated tensors/outputs before deliberately
updating golden data; never regenerate expectations from ApxInf outputs to make
a regression pass. SafeTensors header ordering may differ, so file hashes pin the
checked-in artifact rather than promise byte-identical serialization on regeneration.

This small geometry does not exercise production CUDA kernels or calibrate GPU
precision tolerances. Use the full-checkpoint `pi05_reference_gate` example for
that separate check, as described in `doc/pi05-cpu-reference.md`.

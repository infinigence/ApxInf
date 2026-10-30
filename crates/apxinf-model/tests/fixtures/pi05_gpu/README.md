# Deterministic full-size PI0.5 diagnostic requests

`manifest.json` pins the PI0.5 LIBERO checkpoint SHA256, T10/T21 token IDs,
explicit 50x32 FP32 latents and RGB digest. The two views are generated gradients:
`R=(x+41*view)%256`, `G=(3*y+29*view)%256`, `B=(x+2*y)%256`.
No camera data or trained weights are bundled. The latents were drawn with NumPy
`default_rng(198)` and `default_rng(199)` respectively during the original CPU
reference validation; their exact values are stored, so replay does not depend
on an RNG implementation or seed alone.

Materialize the inputs with the standard library:

```sh
python crates/apxinf-model/tests/fixtures/pi05_gpu/materialize.py \
  --output devlocal/pi05-cpu-reference/gpu-diagnostic
```

Use the resulting manifest with `pi05_reference_gate` and the external checkpoint
whose hash is pinned in it. The runner compares actual CPU FP32 and GPU outputs
on identical requests. Run BF16 on Thor/Orin, FP8 on Thor and INT8 on Orin.
FP8 additionally needs an explicit calibration path and its SHA256 recorded as
`calibration_sha256` in the run manifest. Calibration identity is part of the
qualification; a different calibration requires a new baseline.

The empty `limits` map makes the Rust runner a measurement producer. CI applies
`configs/pi05/reference-gate.json` afterward: per-case cosine floors for Thor
BF16/FP8 and Orin BF16; Orin INT8 is advisory. Floors start at measured values
rounded down to six decimal places and can only increase. L2/max-abs remain
reported diagnostics. The policy also requires exact eager/graph agreement and
bitwise repeat stability. The pinned FP8 profile identity cannot be silently
replaced. See `doc/pi05-cpu-reference.md` for workflows and runner setup.

This complements the tracked tiny CPU oracle; it requires a full checkpoint and
GPU and is not part of portable CI or a robot task-success evaluation.

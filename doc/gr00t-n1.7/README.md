# GR00T N1.7

This guide shows how to run the public GR00T N1.7 LIBERO checkpoint through
ApxInf's policy API. NVIDIA's pinned processor owns image/text/state encoding
and action decoding; ApxInf executes the Model Core in Thor BF16/FP8 or Orin
BF16/INT8 mode.

## Quick start

Build the Python binding from the repository checkout and install the Python
package in the same environment as the pinned Isaac-GR00T processor:

```bash
cd crates/apxinf-py
maturin develop --release --features cuda

cd ../../python/apxinf
pip install -e .
```

Keep the GR00T checkpoint and Cosmos backbone outside the repository. Pass
their local paths when constructing the policy:

```python
import numpy as np
from apxinf import Gr00tPolicy

policy = Gr00tPolicy.from_pretrained(
    "/path/to/GR00T-N1.7-LIBERO/libero_10",
    backbone="/path/to/Cosmos-Reason2-2B",
    precision="bf16",  # use "fp8" together with calibration=...
)

result = policy.infer({
    "observation/image": base_rgb,          # uint8 HWC
    "observation/wrist_image": wrist_rgb,  # uint8 HWC
    "observation/state": np.asarray(state, dtype=np.float32),
    "prompt": "put the moka pot on the stove",
})

actions = result["actions"]  # LIBERO checkpoint: float32 [16, 7]
policy.close()
```

Choose the precision supported by the target:

```python
# Thor
precision="bf16"
precision="fp8"   # also pass calibration="/path/to/calibration.json"

# Orin
precision="bf16"
precision="int8"
```

The processor reads the state-field widths and decoded action layout from the
checkpoint. The caller does not need to split the LIBERO state into named
fields. Missing cameras, an invalid state width, unsupported precision, and
non-finite outputs produce explicit errors.

For a complete command-line example, see
[`python/apxinf/examples/gr00tpolicy_infer.py`](../../python/apxinf/examples/gr00tpolicy_infer.py).
The Python package guide contains the detailed API and dependency description:
[`python/apxinf/README.md`](../../python/apxinf/README.md).

The complete CLI example can be run without writing application code:

```bash
python python/apxinf/examples/gr00tpolicy_infer.py \
  --model-dir /models/GR00T-N1.7-LIBERO/libero_10 \
  --backbone /models/Cosmos-Reason2-2B \
  --image /data/base.png \
  --wrist-image /data/wrist.png \
  --state /data/state.npy \
  --prompt "put the moka pot on the stove" \
  --precision bf16
```

## Supported scope

| Item | First PR |
| --- | --- |
| Model | GR00T N1.7 |
| Checkpoint | `GR00T-N1.7-LIBERO/libero_10` |
| Precision | Thor BF16/FP8; Orin BF16/INT8 |
| Input | Two RGB views, robot state, text prompt |
| Output | Decoded LIBERO action chunk `[16, 7]` |
| Model-Core output | Normalized `[40, 132]` |
| Execution | Fixed-shape CUDA Graph |

NVFP4, training, fine-tuning and TensorRT export are outside the first PR.

## Recorded performance boundary

The numbers below are engineering measurements from the pre-scope-reduction
build, not final release gates. They are Model-Core measurements: batch 1, four flow steps,
10 warmups and 50 measured CUDA Graph replays. Timing starts from tensors
already produced by the official processor and ends after action D2H. It is
not raw-camera Full E2E latency.

| Device | Precision | Views | P50 | P90 | P95 | Frequency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Thor | BF16 | 1 | 56.766 ms | 56.934 ms | 56.964 ms | 17.616 Hz |
| Thor | BF16 | 2 | 59.054 ms | 59.227 ms | 59.300 ms | 16.934 Hz |
| Thor | FP8 | 1 | 33.775 ms | 33.854 ms | 33.870 ms | 29.608 Hz |
| Thor | FP8 | 2 | 38.650 ms | 38.748 ms | 38.782 ms | 25.874 Hz |
| Orin | BF16 | 1 | 80.401 ms | 80.457 ms | 80.467 ms | 12.438 Hz |
| Orin | BF16 | 2 | 86.461 ms | 86.525 ms | 86.541 ms | 11.564 Hz |
| Orin | INT8 | 1 | 64.033 ms | 64.070 ms | 64.073 ms | 15.617 Hz |
| Orin | INT8 | 2 | 69.891 ms | 69.934 ms | 69.940 ms | 14.308 Hz |

These rows must be rerun from the exact unchanged PR candidate before they are
promoted to release results. The older reports identify the fixture and binary
outputs, but do not bind the executable to a clean ApxInf source revision.
The original Orin delivery binary was rerun on the same fixture on 2026-09-04:
BF16 reproduced at P50 `87.474 ms` and INT8 at P50 `70.229 ms`, with action
checksums exactly matching the corresponding original reports. See
[`validation.md`](validation.md) for hashes and same-input reference metrics.

## Validation status

- Same-input BF16/FP8 numerical checks cover output shape, finite values,
  cosine similarity, relative L2 and absolute error.
- Release task accuracy is measured over all 10 LIBERO-10 tasks with 50
  episodes per task and two physical camera views.
- Performance, same-input numerical parity and closed-loop task success are
  reported separately; a fast latency result is not treated as task accuracy.

The exact acceptance commands, thresholds and result provenance are recorded
in [`validation.md`](validation.md).

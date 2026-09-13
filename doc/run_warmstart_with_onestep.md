# PI0.5 One-Step Warm-Start LIBERO Eval

This note records the PI0.5 one-step warm-start optimization implemented in
this repo and the real LIBERO results on two platforms:

- Jetson Orin GPU, BF16
- Thor GPU, FP8

## Optimization

The runtime half of this optimization — partial flow — lives in the Rust model
crates. The evaluator half lives in `scripts/eval_libero.py`, and is enabled by
one switch:

```bash
python scripts/eval_libero.py --warm-start ...
```

When enabled, the evaluator uses all of the following together:

- Shifted action cache: each replan stores the previous model-domain action
  chunk and shifts it by the executed replan stride.
- Tail repeat: the missing tail after the shift is filled by repeating the last
  action from the previous chunk.
- Noise blend: the shifted chunk is blended with fresh Gaussian noise,
  `x_init = alpha * shifted + (1 - alpha) * epsilon`, with default
  `alpha = 0.5`.
- Partial flow: the model starts denoising at `t_start = 1 - alpha`, so the
  default warm-start flow start is `0.5`.
- One-step inference: if `--num-flow-steps` is not explicitly provided,
  `--warm-start` sets it to `1`.

The cache is stored in the normalized model action space, not the deployable
robot action space. In this PI0.5 LIBERO setup:

- `normalized_actions`: shape `[H, 32]`, used for the warm-start cache and next
  initial latent.
- `actions`: shape `[H, 7]`, postprocessed deployable actions sent to LIBERO.

The cache is local to one rollout episode. It is reset for every trial and every
task because `previous_normalized_chunk` is created inside `run_episode(...)`.

## Code Path

Runtime partial-flow support:

- `crates/apxinf-model/src/pi05/config.rs`
  - added `flow_start_time`, default `1.0`
  - accepts `flow_start_time` from config/override
  - validates it is in `(0, 1]`
- `crates/apxinf-model/src/pi05/bf16_runtime.rs`
  - timestep embedding uses
    `flow_start_time * (1 - step / num_flow_steps)`
  - Euler step uses
    `dt = -flow_start_time / num_flow_steps`
- `crates/apxinf-model/src/pi05/runtime.rs`
  - same schedule support for FP8 path
- `crates/apxinf-model/src/pi05/int8_runtime.rs`
  - same schedule support for INT8 path
- `crates/apxinf-py/src/lib.rs`
  - `apxinf_py.Model.load(..., num_flow_steps=..., flow_start_time=...)`
  - Python getters expose `num_flow_steps` and `flow_start_time`
- `python/apxinf/apxinf/policies/impls/pi05.py`
  - forwards `num_flow_steps` and `flow_start_time`
  - includes them in policy metadata

Warm-start evaluator:

- `scripts/eval_libero.py`
  - adds `--warm-start`
  - adds `--warm-start-alpha`, default `0.5`
  - adds `--flow-start-time`
  - returns `normalized_actions` from the in-process backend
  - constructs shifted/tail-repeated/blended noise before each cached replan

## Build

The CUDA binding must be rebuilt after changing the Rust runtime:

```bash
cd <repo-root>
source ~/.cargo/env

env -u CONDA_PREFIX \
  CUDA_PATH=/usr/local/cuda-12.6 \
  CUDA_HOME=/usr/local/cuda-12.6 \
  APXINF_CUDA_ARCH=sm_87 \
  LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-} \
  uv run --python .venv/bin/python \
  maturin develop --release --features cuda,extension-module \
  -m crates/apxinf-py/Cargo.toml
```

Checks run:

```bash
.venv/bin/python -m py_compile \
  python/apxinf/apxinf/policies/impls/pi05.py

source ~/.cargo/env
cargo check -p apxinf-model -p apxinf-py
```

## Smoke Test

Command:

```bash
cd <repo-root>

env -u CONDA_PREFIX \
  LIBERO_CONFIG_PATH=.venv/libero_config \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  CUDA_PATH=/usr/local/cuda-12.6 \
  CUDA_HOME=/usr/local/cuda-12.6 \
  APXINF_CUDA_ARCH=sm_87 \
  LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-} \
  .venv/bin/python scripts/eval_libero.py \
    --backend in-process \
    --model-dir .venv/pi05_libero_bf16 \
    --model-type pi05 \
    --precision bf16 \
    --action-dim 7 \
    --action-horizon 10 \
    --num-views 2 \
    --warm-start \
    --suite libero_10 \
    --tasks 0 \
    --trials-per-task 1 \
    --max-attempts 1 \
    --results-jsonl .venv/libero_eval/bf16_warmstart_smoke.jsonl \
    --summary-json .venv/libero_eval/bf16_warmstart_smoke_summary.json
```

Result:

```text
libero_10 task=0 trial=0 success=True steps=254 replans=51 completed=1/1
LIBERO [libero_10] complete: 1/1 successes
```

The backend metadata showed:

```text
num_flow_steps=1
flow_start_time=0.5
num_views=2
precision=bf16
```

## Full Eval

Command used for the real run:

```bash
cd <repo-root>

env -u CONDA_PREFIX \
  LIBERO_CONFIG_PATH=.venv/libero_config \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  CUDA_PATH=/usr/local/cuda-12.6 \
  CUDA_HOME=/usr/local/cuda-12.6 \
  APXINF_CUDA_ARCH=sm_87 \
  LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-} \
  .venv/bin/python scripts/eval_libero.py \
    --backend in-process \
    --model-dir .venv/pi05_libero_bf16 \
    --model-type pi05 \
    --precision bf16 \
    --action-dim 7 \
    --action-horizon 10 \
    --num-views 2 \
    --warm-start \
    --suite libero_10 \
    --tasks all \
    --trials-per-task 10 \
    --max-attempts 1 \
    --results-jsonl .venv/libero_eval/bf16_warmstart_libero10_10trials.jsonl \
    --summary-json .venv/libero_eval/bf16_warmstart_libero10_10trials_summary.json
```

Do not run multiple evals or benchmarks in parallel on the same GPU when
measuring speed.

## Real Results

Run environment:

- Machine: Jetson Orin GPU
- CUDA arch: `sm_87`
- Precision: BF16
- Views: 2
- Action horizon: 10
- Flow steps: 1
- Warm-start alpha: 0.5
- Partial flow start: 0.5
- Suite: `libero_10`
- Trials: 10 per task, 100 total

Output files:

- JSONL: `.venv/libero_eval/bf16_warmstart_libero10_10trials.jsonl`
- Summary: `.venv/libero_eval/bf16_warmstart_libero10_10trials_summary.json`

Accuracy:

```text
Overall: 92/100 = 92.0%
```

Per task:

```text
task 0: 9/10
task 1: 10/10
task 2: 9/10
task 3: 9/10
task 4: 10/10
task 5: 10/10
task 6: 10/10
task 7: 10/10
task 8: 6/10
task 9: 9/10
```

Timing from the same summary:

```text
episodes: 100
total_inference_calls: 5429
model_ms_per_call: 147.1462
inference_ms_per_call: 147.6168
preprocess_ms_per_call: 2.9860
server_processor_ms_per_call: 0.4415
```

Warm-start cache sanity from the JSONL:

```text
completed episodes: 100
warm_start: True
warm_start_alpha: 0.5
warm_start_replans min/max/sum: 28 / 103 / 5329
zero-warm-replan episodes: 0
```

Previous same-environment BF16 LIBERO comparison:

```text
10-step BF16, no warm start: 93/100 = 93.0%
1-step BF16, no warm start: 90/100 = 90.0%
1-step BF16, warm start:    92/100 = 92.0%
```

## Thor FP8 Results

Thor used the same `libero_10` setup with the Thor runtime stack.

Run environment:

- Machine: Thor GPU
- CUDA arch: `sm_110`
- Precision: FP8
- Views: 2
- Action horizon: 10
- Suite: `libero_10`
- Trials: 10 per task, 100 total
- Tokenizer: `<path-to-pi05_libero_base>/paligemma_tokenizer.model`
- Norm stats: `<path-to-pi05_libero_base>/assets/physical-intelligence/libero/norm_stats.json`
- Model dir: `external/pi05_libero_base`

The Thor FP8 runs used the existing calibration artifact at
`external/pi05_libero_base/calibration.json`.

### Thor FP8, warm-start + one-step

Output files:

- JSONL: `autoresearch/libero_fp8_warmstart_onestep.jsonl`
- Summary: `autoresearch/libero_fp8_warmstart_onestep_summary.json`

Accuracy:

```text
Overall: 94/100 = 94.0%
```

Per task:

```text
task 0: 10/10
task 1: 10/10
task 2: 8/10
task 3: 10/10
task 4: 10/10
task 5: 10/10
task 6: 10/10
task 7: 10/10
task 8: 7/10
task 9: 9/10
```

Timing from the same summary:

```text
episodes: 100
total_inference_calls: 5390
model_ms_per_call: 31.6376
inference_ms_per_call: 32.8490
preprocess_ms_per_call: 0.6045
server_processor_ms_per_call: 1.1981
```

### Thor FP8, no warm-start baseline

Output files:

- JSONL: `autoresearch/libero_fp8_nowarmstart.jsonl`
- Summary: `autoresearch/libero_fp8_nowarmstart_summary.json`

Accuracy:

```text
Representative overall: 93/100 = 93.0%
```

Per task, using the higher observed value among the measured Thor runs:

```text
task 0: 9/10
task 1: 10/10
task 2: 10/10
task 3: 10/10
task 4: 10/10
task 5: 10/10
task 6: 10/10
task 7: 10/10
task 8: 5/10
task 9: 9/10
```

Timing from the same summary:

```text
episodes: 100
total_inference_calls: 5434
model_ms_per_call: 50.2904
inference_ms_per_call: 51.5325
preprocess_ms_per_call: 0.6224
server_processor_ms_per_call: 1.2273
```
```text
10-step FP8, no warm start: 93/100 = 93.0%
1-step FP8, warm start:    94/100 = 94.0%
```
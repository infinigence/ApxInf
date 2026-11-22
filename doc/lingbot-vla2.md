# LingBot-VLA 2.0

ApxInf serves Robbyant's LingBot-VLA 2.0 through the same public path as the other VLA families:

```text
AutoPolicy -> LingbotVla2Policy -> apxinf_py.ModelRunner -> AutoModel -> VlaRuntime
```

The native module is `crates/apxinf-model/src/cuda_lingbot_vla2/` (registry name `lingbot_vla2`). The Python policy is
`python/apxinf/apxinf/policies/impls/lingbot_vla2.py`. The module imports no other model family.

## Scope

| Item | Value |
| --- | --- |
| Checkpoints | `robbyant/lingbot-vla-v2-6b` (pre-trained, rev `11c703bf`) and `robbyant/lingbot-vla-v2-6b-robotwin` (RoboTwin post-trained, rev `04518557`); same tensor names |
| Backend | CUDA, `cuda-new`, sm_89 verified (RTX 4060 8 GB) |
| Precision | `model_variant` `auto` or `int8_convrot`: W8A8 with 256-wide Hadamard rotation (ConvRot). BF16 weights are 12.5 GB and do not fit an 8 GB card, so no BF16 variant exists |
| Inputs | 3 camera views, 256x256 after resize (768 ViT patch rows), prompt of up to 72 tokens, one 55-dim state vector, 50x55 noise |
| Output | normalised action chunk `[50, 55]`; the Python policy un-normalises it into the robot's action keys |
| Sampling | 10 flow steps, BF16-rounded schedule (1.0, 0.8984, ... 0.0952, dt -0.100098) |
| Execution | one CUDA Graph over vision, language prefix and all flow steps; one retained plan, keyed by prompt length |

`load` rejects any other `model_variant`, any view count other than 3, and a checkpoint whose tensor shapes differ from the
published architecture, before converting a single weight.

## What stays BF16

Patch embedding, the router, the shared expert, the state/action projections, all norms and all biases. Everything else (ViT
blocks and mergers, the 36 language layers, expert attention, the 32 routed experts) is INT8. The token-embedding table
(742 MiB in BF16) stays in host memory; a request reads at most 90 rows and uploads them as a fixed-address device input, which
keeps the graph free of host access and saves that much GPU memory.

## Mixture of experts without a dispatch kernel

Each action-expert layer has 32 routed experts (top 4, sigmoid scores, renormalised, x4.0) plus one shared expert. The port runs
the routed experts densely: gate and up of all 32 experts are one `[32768, 768]` INT8 GEMM, `moe_swiglu_scaled` multiplies each
expert's 512 columns by its routing weight (zero when not selected), and one K = 16384 GEMM sums the experts. This equals sparse
dispatch. On one measured input, 264 of 360 MoE calls left at least one expert unselected (1.85 on average), so sparse dispatch
would skip about 6% of the routed-expert weight bytes.

Two operators were added to `cuda-new`: `ops::moe::moe_route` and `ops::moe::moe_swiglu_scaled` (see
[cuda-new-api.md](cuda-new-api.md)).

## Memory (RTX 4060, 8188 MiB, 7807 MiB allocatable)

| Stage | MiB (nvidia-smi) |
| --- | ---: |
| After load (weights, KV caches, tables) | 6053 |
| Steady state after the graph is captured | 6561 |
| Highest reading while loading and capturing (250 ms polling) | 6561 |
| Same measurement before the token table moved to host memory | 7302 |

`autotune=True` is required for the graph path on this card. With the default cuBLAS W8A8 fallback the vision tower allocates
about 870 MiB of workspace and capture fails with out of memory; autotune selects CUTLASS W8A8, which needs none.

## Accuracy

All numbers compare the INT8 CUDA runtime with the upstream PyTorch implementation run in BF16 on the CPU, on identical encoded
inputs and noise.

| Input | Cosine | Relative L2 |
| --- | ---: | ---: |
| Synthetic images, 16-token prompt (case a) | 0.99989 | 0.0154 |
| Synthetic images, 8-token prompt, other state and noise (case b) | 0.99978 | 0.0211 |
| 18 real RoboTwin observations (post-trained checkpoint) | 0.99999 | 0.0049 |

Acceptance gates were set before the CUDA run: cosine >= 0.998 and relative L2 <= 0.05 against BF16; relative L2 <= 0.02 against a
CPU simulation of the same INT8 arithmetic.

## Latency (batch 1, 3 views, CUDA Graph, autotune, includes the 4.7 MB patch upload and synchronisation)

| Flow steps | Median ms |
| ---: | ---: |
| 1 | 61.8 |
| 5 | 117.5 |
| 10 | 187.4 |
| 20 | 326.9 |

Latency is linear in steps: 47.8 ms fixed plus 13.96 ms per step. Per-stage kernel time at 10 steps (nsys): vision 13.7 ms,
language prefix 32.7 ms, action expert 140.8 ms. Through the Python policy from raw frames, a request takes about 215 ms with 480x480
frames and 205 ms (median of 82 calls) with the 320x240 frames RoboTwin sends; the difference to the 187 ms graph replay is host-side
preprocessing, which was not broken down further.

## Closed loop (RoboTwin 2.0, one 8 GB card shared by model and simulator)

RoboTwin rev `13c3c47`, `demo_clean`, `instruction_type=unseen`, 50 actions executed per inference, 10 flow steps. The model runs behind a
websocket server that speaks the upstream deploy protocol; the client is upstream's `eval_policy_client_lingbotvla.py`. Twelve tasks, 3
episodes each, seeds from 100000 that RoboTwin's own `collect_data.py` verified with its expert. The noise stream restarts at every episode.

| Policy | Success | Wilson 95% |
| --- | ---: | ---: |
| Post-trained checkpoint, INT8 | 33/36 | 78-97% |
| Pre-trained checkpoint, INT8 | 6/36 (`press_stapler` 3, `turn_switch` 2, `stamp_seal` 1) | 8-32% |
| Hold the current joint state | 0/36 | 0-10% |
| Post-trained checkpoint, upstream PyTorch BF16 on the CPU (control) | 33/36 | 78-97% |

The three post-trained failures are `click_bell` seed 100002 and `turn_switch` seeds 100000 and 100002. Each runs to the step limit. The BF16 control uses the same policy code, seeds and noise stream with the upstream PyTorch model on the CPU; it has the same outcome as INT8 in all 36 episodes, including these three, so they are not caused by INT8 quantisation. Among the 33 episodes both completed, step counts differ by 0 to 8 (mean 1.2). This does not reproduce upstream's published 93.5% average (FP32, 50 tasks); 12 tasks
with 3 episodes each is too small a sample to compare. The simulator side differs from upstream in five ways, all needed to fit the model
(6561 MiB steady) and the ray-traced simulator (about 1.2 GB) on one card whose CUDA-visible memory is 7807 MiB (measured peak with both resident: 7807 MiB, so no spare memory): the ray-tracing denoiser is off, expert-verified
seeds are read from a file instead of re-running the curobo expert before each episode, a gripper-interpolation stand-in replaces the
curobo planner (rollouts use joint-position control), each episode runs in its own simulator process, and the model's CUDA Graph is
captured before the simulator starts.

The first request of an episode is slow (474-604 ms against 205 ms) whenever the instruction tokenises to a different length than the
previous one, because the runtime keeps one plan keyed by prompt length and rebuilds it. In the 36 post-trained episodes that happened in 29.

## Python policy

```python
from apxinf import AutoPolicy

policy = AutoPolicy.from_pretrained(
    model_dir, model_type="lingbot_vla2", robot="robotwin",
    norm_stats="robotwin.json", training_config="lingbotvla_cli.yaml", autotune=True,
)
out = policy.infer({
    "observation.images.cam_high": frame_high,          # uint8 HxWx3
    "observation.images.cam_left_wrist": frame_left,
    "observation.images.cam_right_wrist": frame_right,
    "observation.state": state_14d,                     # float32 [14]
    "prompt": "pick up the red cup and place it on the plate",
})
out["actions"]          # float32 [50, 14]
```

The pre-trained repository ships no normalisation statistics and no training configuration. Use the files from the RoboTwin
post-trained repository (`lingbotvla_cli.yaml`) and the upstream `assets/norm_stats/robotwin.json`. Robot descriptions for
`robotwin` and `agilex_cobot_magic` are built in; any other embodiment needs its own robot config.

## Tests and reference checks

```bash
# operators and model library (needs a CUDA GPU; the cuda-new suite has five failures that also occur on main,
# see below)
cargo test -p apxinf-cuda-new --release --lib moe
cargo test -p apxinf-model --features cuda --release --lib lingbot

# policy contract and host-side processing; no GPU
python -m pytest python/apxinf/tests/test_lingbot_vla2_policy.py python/apxinf/tests/test_lingbot_vla2_parity.py
```

The Python tests always run their contract part with a mock runner. The tests that compare with upstream skip unless these
variables point at reference inputs the repository does not ship:

| Variable | Content |
| --- | --- |
| `LINGBOT_UPSTREAM` | checkout of `robbyant/lingbot-vla-v2`; `assets/norm_stats/robotwin.json` is read from it |
| `LINGBOT_MODEL_DIR` | a checkpoint directory with `tokenizer.json` |
| `LINGBOT_GOLDEN_DIR` | `golden_<tag>.safetensors` and `golden_<tag>.json` from a run of the upstream model |
| `LINGBOT_QWEN3VL_CFG` | the Qwen3-VL `preprocessor_config.json` and `chat_template.json` |

`crates/apxinf-model/examples/lingbot_vla2_parity.rs` runs the checkpoint through `AutoModel` against raw reference files
(layout in its header) and prints cosine, relative L2 and latency. It needs a CUDA GPU; on an 8 GB card pass `autotune`.

`cargo test -p apxinf-cuda-new --release --lib -- --test-threads=1` reports five failures on this tree: a repacked GEGLU graph,
`fused_add_channel_l2_norm_silu`, `rms_rope_matches_unfused`, and two `volume_lowering` tests. They fail the same way on `main` on an
RTX 4060 and none involves the MoE operators.

## Not covered

- Other camera counts, other image sizes, batch > 1, other GPUs, and any non-CUDA backend.
- One `cuBLAS status 13` error occurred once during the first graph capture after enabling autotune and could not be reproduced
  in later runs (bisected by stage and layer count, more than six full reruns).
- Closed-loop success was measured on 12 of the 50 RoboTwin tasks, 3 episodes each (see Closed loop); the other 38 tasks are untested.

# ApxInf

## Description

ApxInf is a reimagined edge inference engine born of the agentic coding era,
combining high performance, reliability, and energy efficiency across devices
with an evolving agentic workflow that radically simplifies custom model development.

- implemented with system language Rust with no other externel dependencies
- embodied AI is highest priority, VLA/WAM models on Jetson/DriveOS Thor/Orin
- Agentically optimized CUDA Kernels

The first version of ApxInf ships with highly optimized PI-0.5 VLA model on Jetson Thor &
Orin devices, and supports BF16, FP8 and INT8 precisions.


## Quick start

Make sure you have ApxInf built and installed, see [Build ApxInf](#build-apxinf) for instructions.

### Quickly benchmarking PI-0.5

Benchmarking PI-0.5 with randomly generated weights.

```bash
python scripts/bench_pi05.py --random-weights --precision bf16 --layer l1 \
  --views 2 --token-count 10 --action-horizon 10 --num-flow-steps 10 \
  --warmup 10 --samples 100 --autotune
```

```bash
python scripts/bench_pi05.py --random-weights --precision fp8 --layer l1 \
  --views 2 --token-count 10 --action-horizon 10 --num-flow-steps 10 --autotune
```

Reported latency is P50 over 30 samples after 10 warm-up iterations
(`--warmup` / `--samples`).

### Run a policy through Python API

```python
import numpy as np
from apxinf import AutoPolicy

policy = AutoPolicy.from_pretrained("<path-to-model>", precision="bf16")

observation = {
    key: np.zeros((256, 256, 3), np.uint8)
    for key in policy.metadata["image_keys"]
}
observation[policy.metadata["prompt_key"]] = "put both moka pots on the stove"
if state_key := policy.metadata.get("state_key"):
    observation[state_key] = np.zeros(policy.metadata["state_dim"], np.float32)

result = policy.infer(observation)

result["actions"]   # (H, policy.action_dim) float32, unnormalized
result["timing"]    # model_ms / total_ms
policy.close()
```

`<path-to-model>` is a checkpoint directory (`model.safetensors`, `config.json`,
and that model's tokenizer/normalizer assets); none ships with this package.
Resize, tokenization, normalization, and the flow sampler all run inside `infer`
— pass raw frames.

### Serve it with OpenPI compatible websocket server

```bash
python python/apxinf/examples/openpi_server.py \
  --model-dir <path-to-model> --precision bf16 --port 8000 \
  --image-keys observation/image,observation/wrist_image \
  --state-key observation/state
```

The wire keys are the caller's to name — see
[OpenPI-compatible serving](#openpi-compatible-serving).

An unmodified `openpi-client` connects to it:

```python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000)
actions = client.infer(observation)["actions"]
```

## Performance

Two views, 224x224 NHWC `uint8`, 10 flow steps, `H=10`, batch 1. Latency is
steady-state CUDA Graph replay P50.

| Hardware | Precision | Latency | Throughput |
|---|---|---:|---:|
| Jetson AGX Thor | BF16 | 72.45 ms | 13.8 Hz |
| Jetson AGX Thor | FP8 | **41.16 ms** | **24.3 Hz** |
| Jetson AGX Orin | BF16 | 165.67 ms | 6.0 Hz |
| RTX 4090 | BF16 | 31.38 ms | 31.9 Hz |
| RTX 4090 | INT8 | 25.99 ms | 38.5 Hz |

LIBERO-10, 10 tasks x 50 episodes, `H=10`, `replan=5`, seed 7. PI0.5 reference
is 92.4%.

| Hardware | Precision | Trials | Success | Rate |
|---|---|---:|---:|---:|
| Jetson AGX Thor | BF16 | 500 | 464 | 92.8% |
| Jetson AGX Thor | FP8 | 500 | 470 | 94.0% |
| Jetson AGX Orin | BF16 | 500 | 460 | 92.0% |


## Port a new model with an agent

`skills/model-port-workflow` drives the whole sequence.

Install it once, from the repository root:

```bash
# Claude Code
mkdir -p .claude/skills && ln -s ../../skills/model-port-workflow .claude/skills/

# Codex
mkdir -p ~/.agents/skills && ln -s "$(pwd)/skills/model-port-workflow" ~/.agents/skills/
```

Then invoke it with the model, the target, and the acceptance bar:

```
/model-port-workflow port GR00T N1.7 from <path-to-reference-implementation> to
ApxInf, Jetson Thor, BF16, parity against the reference within 1e-2
```

It works from the same guides a human would follow:
- [porting workflow](../../doc/porting-workflow.md),
- [adding a new model](../../doc/adding-a-new-model.md),
- [model-layer architecture](../../doc/model-layer-architecture.md),
- [adding new kernels](../../doc/adding-new-kernels.md).

## Build ApxInf

```bash
git clone <repo-url> && cd ApxInf
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
CARGO_TARGET_DIR=target/wheel maturin build --release --features cuda --auditwheel skip -m crates/apxinf-py/Cargo.toml
pip install --force-reinstall target/wheel/wheels/apxinf_py-*.whl
pip install -e "python/apxinf[tokenizer,serving]"
# For WallOSS checkpoints, use its processor extra instead:
# pip install -e "python/apxinf[walloss,serving]"
```

Activate a venv or conda env before installing. The extras keep model-specific processor
dependencies opt-in: PI0.5 uses `tokenizer`, while WallOSS uses `walloss`; both
can add `serving` for msgpack/websockets.

`--features cuda` is a Cargo feature, not a CUDA installation: it compiles the
CUDA backend into the binding, and it is required — the PI0.5 runtime is only
registered on CUDA devices, as is WallOSS.

The build queries the visible GPU for its compute capability and compiles the
kernels for exactly that architecture, so build on the machine you deploy to;
cross-compiling fails unless `APXINF_CUDA_ARCH` names the target (`sm_87` Orin,
`sm_101` Thor-U, `sm_110` Thor).

Confirm the binding imports and reaches the GPU:

```bash
python -c 'import apxinf_py; print(apxinf_py.__version__)'
python scripts/bench_pi05.py --random-weights --precision bf16 --layer l1 --samples 5
```

Needs a Linux host with an NVIDIA driver and a CUDA toolkit plus a stable Rust
toolchain. If `nvcc --version` or `cargo --version` fails, set them up first:

- [NVIDIA build environment](#nvidia-build-environment)
- [Rust toolchain](#rust-toolchain)


## Using ApxInf from Python

Three public layers, each wrapping the one before. Pick the outermost one that
still leaves you the control you need.

### L1 — bare model

You own resize, tokenization, noise, and unnormalization; the model takes
already-resized frames and returns a **normalized-domain** chunk.

```python
from apxinf import Model

model = Model.load("pi05", "<path-to-model>/model.safetensors", precision="bf16")

# rgb: uint8 [views, H, W, 3] at model.image_size; tokens: uint32; noise: float32
actions = model.infer_rgb(rgb, "nhwc", token_ids, noise)   # (H, action_dim)
model.action_horizon, model.num_views, model.image_size    # what it was loaded for
```

### L2 — policy

Adds the pre/post pipelines and reads the checkpoint's tokenizer and
`norm_stats`, so it takes a raw observation dict and returns deployable actions
— the [Run a policy](#run-a-policy) snippet. Beyond that call it exposes the
serving contract, the pipelines, and the layer boundary:

```python
policy = AutoPolicy.from_pretrained(
    "<path-to-model>",
    precision="bf16",
    action_dim=None,        # default: infer the model's full vector from checkpoint weights
)

policy.metadata             # model_type, action_horizon, image_keys, state_key, ...

# Pipelines are ordered named steps — image_stack -> tokenize in, trim -> unnormalize
# out — and every mutation returns a new one, so a custom step drops in as a value.
policy.input_pipeline = policy.input_pipeline.replace("tokenize", MyTokenizeStep())
policy.output_pipeline = policy.output_pipeline.insert_after(
    "unnormalize", ("clip", MyClip())
)

result = policy.infer(observation)
result["normalized_actions"]  # what L1 returned, before trim + unnormalize
```

### L3 — websocket server

Wraps an L2 policy in the OpenPI wire protocol. See
[OpenPI-compatible serving](#openpi-compatible-serving).

## OpenPI-compatible serving

The server speaks OpenPI's websocket protocol, so an existing `openpi-client`
robot stack connects without a code change — swap the endpoint and keep the
observation dict you already send.

```python
from apxinf import AutoPolicy
from apxinf.serving import WebsocketPolicyServer

policy = AutoPolicy.from_pretrained(
    "<path-to-model>",
    precision="bf16",
    image_keys=("observation/image", "observation/wrist_image"),
    state_key="observation/state",
)
WebsocketPolicyServer(policy, "0.0.0.0", 8000).serve_forever()
```

**The wire keys are yours to name.** This engine holds no dataset's dialect: what
a client calls its cameras is a property of the recording, not of the weights, and
one checkpoint architecture is served under several. `image_keys=` / `state_key=`
/ `prompt_key=` say what your client sends. Omit `image_keys` and the policy falls
back to the model's own view-slot names (`base_0_rgb`, ...) — a fallback, not a
contract, published as `apxinf.CANONICAL_IMAGE_KEYS`. `state_key` has no fallback
at all: a wrong camera key raises on the first inference, a wrong state key is
silent, so a policy that reads state refuses to be built without one.

The server keeps the checkpoint's native action width unless the user supplies
`--action-dim`. It publishes the resolved wire contract in connect-time metadata
so the client can assert it rather than guess.

Named robots — `franka_libero`, `unitree_g1` — are the layer *above* this one:
a body (DoF layout, delta mask, gripper convention) paired with a dialect, plus
the simulator glue to drive them. That lives in
[apxinf-robo](https://github.com/team-mz/APXinf-robo), which composes over this
engine and adds nothing to it:

```bash
apxinf-robo serve --robot unitree_g1 --model-dir <path-to-model> --port 8000
```


## Precisions

### BF16

The default, on every supported device. Runs on the checkpoint alone; no
calibration.

```bash
python examples/openpi_server.py \
  --model-dir <path-to-model> --precision bf16 \
  --image-keys observation/image,observation/wrist_image \
  --state-key observation/state \
  --port 8000
```

```python
policy = AutoPolicy.from_pretrained("<path-to-model>", precision="bf16")
```

### FP8

Thor only, where it is the fastest path. Orin has no FP8 Tensor Cores and is not
supported.

FP8 needs per-tensor activation scales. Pass the calibration generated for the
deployment data explicitly; when omitted, ApxInf falls back to
`<path-to-model>/calibration.json`:

```bash
python examples/openpi_server.py \
  --model-dir <path-to-model> --precision fp8 \
  --policy-options '{"calibration":"<path-to-calibration.json>"}' \
  --port 8000
```

```python
policy = AutoPolicy.from_pretrained(
    "<path-to-model>",
    precision="fp8",
    calibration="<path-to-calibration.json>",
)
```

If the checkpoint does not contain `calibration.json`, generate one from
representative Observations:

```bash
python3 scripts/calibrate_pi05.py \
  --model-dir <path-to-model> \
  --manifest <path-to-observations.jsonl>
```

The calibrator reads observations off disk and drives no simulator — MuJoCo and a
task suite are properties of the environment, not of the weights. To calibrate on
LIBERO frames, capture them first with `apxinf-robo capture-libero`, then point
the calibrator at the directory:

```bash
apxinf-robo capture-libero --suite libero_10 --output-dir /tmp/libero-calib
python3 scripts/calibrate_pi05.py \
  --model-dir <path-to-model> \
  --input-dir /tmp/libero-calib
```

See [PI0.5 FP8 calibration](doc/pi05-fp8-calibration.md) for the Observation
format and output options.

### INT8

W8A8, optimized for Orin (SM87) and Ada (SM89). Needs nothing beyond the
checkpoint.

```bash
python examples/openpi_server.py \
  --model-dir <path-to-model> --precision int8 --port 8000
```

```python
policy = AutoPolicy.from_pretrained("<path-to-model>", precision="int8")
```


## LIBERO evaluation

### Get the checkpoint

The published accuracy is `pi05_libero_base`, π0.5 fine-tuned on LIBERO — an
arbitrary π0.5 checkpoint might not reproduce it.

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download lerobot/pi05_libero_base --local-dir <path-to-model>
curl -fL https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  -o <path-to-model>/norm_stats.json
```

The `lerobot/pi05_libero_base` checkpoint lost its normalization statistics
during repository updates. To reproduce the officially reported performance,
download OpenPI's LIBERO `norm_stats.json` separately as shown above and pass it
explicitly with `--norm-stats`.

### Run

The rollout itself lives in [apxinf-robo](https://github.com/team-mz/APXinf-robo),
which owns LIBERO, MuJoCo, the Franka body, and the resumable episode ledger.
This engine holds none of that: a simulator is not a property of the weights.

```bash
pip install "apxinf-robo[libero]"     # plus LIBERO itself; see that repo's README
apxinf-robo eval-libero --backend in-process --model-dir <path-to-model> \
  --norm-stats <path-to-model>/norm_stats.json \
  --precision bf16 --action-horizon 10 \
  --suite libero_10 --tasks all --trials-per-task 50 \
  --results-jsonl <out-dir>/results.jsonl --summary-json <out-dir>/summary.json
```

That is the published protocol: all 10 LIBERO-10 tasks x 50 episodes at seed 7
(the default), 500 episodes in total. `--backend websocket --host <h> --port <p>`
evaluates a running [server](#openpi-compatible-serving) instead — the same
engine, reached over the wire rather than built in-process.


## Benchmark

`scripts/bench_pi05.py` times the concentric serving shells so a regression can
be attributed to the engine, the processors, or the transport.

```bash
python scripts/bench_pi05.py --model-dir <path-to-model> --precision bf16 --layer l1,l2
```

- `--layer` selects any subset of `l1` (bare model), `l2` (full policy), `l3`
  (websocket round trip). L3 attaches to a running server and needs no local
  weights.
- `--model-dir` runs a real checkpoint at its native horizon; `--random-weights`
  runs the engine with no checkpoint on disk, and the shape knobs (`--views`,
  `--image-size`, `--action-horizon`, `--num-flow-steps`, `--token-count`)
  select the synthetic workload.
- `--calibration` is FP8-only and synthetic-only; a checkpoint reads
  `calibration.json` from its own directory.
- `--action-horizon` also applies to a checkpoint — the horizon is a sequence
  length, not a weight dimension — which is what makes a real checkpoint
  comparable to a synthetic run.
- `--warmup` / `--samples` set the sampling protocol (default 10 and 30);
  `--out` writes the report as JSON.

Any registered model type works: `AutoPolicy` dispatches on the checkpoint's
`config.json`, so the same command benchmarks the next model without a flag
change.

For one-step evaluation, please refer to `doc/run_warmstart_with_onestep.md`


## NVIDIA build environment

A complete CUDA toolkit is required: `nvcc`, CUDA headers and runtime, cuBLAS
and cuBLASLt development libraries, and NVTX (`libnvToolsExt` on Jetson,
`libnvtx3interop` on desktop CUDA). Also a C/C++ compiler, linker, `ar`, Git,
`pkg-config`, and Python 3. The CUDA kernels, CUTLASS, and FlashAttention
sources are vendored — no external checkout needed.

Install the driver and toolkit through the JetPack, DRIVE OS, or CUDA
distribution for the machine, then check `nvcc --version`. If CUDA does not live
at `/usr/local/cuda`, point `CUDA_PATH` at it.

| Device | Architecture | Validated toolkit |
|---|---:|---:|
| Jetson AGX Thor | `sm_110` | CUDA 13.0 |
| Thor-U | `sm_101` | CUDA 12.8 |
| Jetson AGX Orin | `sm_87` | CUDA 12.6, 13.2 |
| RTX 4090 | `sm_89` | CUDA 12.8 |


## Rust toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup default stable
```

Built with Rust 1.95 and 1.96; no minimum supported version is declared.


## License

Apache 2.0. Vendored third-party components retain their own licenses.

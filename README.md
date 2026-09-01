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
python -m evaluation.benchmarks.pi05 --random-weights --precision bf16 --interface model \
  --views 2 --token-count 10 --action-horizon 10 --num-flow-steps 10 \
  --warmup 10 --samples 100
```

```bash
python -m evaluation.benchmarks.pi05 --random-weights --precision fp8 --interface model \
  --views 2 --token-count 10 --action-horizon 10 --num-flow-steps 10
```

Reported latency is P50 over 30 samples after 10 warm-up iterations
(`--warmup` / `--samples`).

### Run a policy through Python API

```python
import numpy as np
from apxinf import AutoPolicy

policy = AutoPolicy.from_pretrained("<path-to-model>", precision="bf16", action_dim=7)

result = policy.infer({
    "observation/image":       np.zeros((256, 256, 3), np.uint8),
    "observation/wrist_image": np.zeros((256, 256, 3), np.uint8),
    "observation/state":       np.zeros(8, np.float32),
    "prompt":                  "put both moka pots on the stove",
})

result["actions"]   # (H, 7) float32, unnormalized
result["timing"]    # model_ms / total_ms
policy.close()
```

`<path-to-model>` is a checkpoint directory (`model.safetensors`, `config.json`,
`norm_stats.json`, `*tokenizer.model`); none ships with this package. Resize,
tokenization, normalization, and the flow sampler all run inside `infer` — pass
raw frames.

### Serve it with OpenPI compatible websocket server

```bash
python scripts/pi05_openpi_websocket_server.py \
  --model-dir <path-to-model> --robot franka_libero --precision bf16 --port 8000
```

An unmodified `openpi-client` connects to it:

```python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000)
actions = client.infer(observation)["actions"]
```

Connect to this server to evaluate LIBERO accuracy:

```bash
python -m evaluation.libero.client \
  --host 127.0.0.1 --port 8000 --precision bf16 \
  --suite libero_10 --tasks 0 --trials-per-task 1 \
  --results-jsonl out/libero-smoke.jsonl \
  --summary-json out/libero-smoke.summary.json
```

The inference host does not need LIBERO installed. See
[LIBERO evaluation](#libero-evaluation) for environment setup and the full
10-task protocol.

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
maturin develop --release --features cuda -m crates/apxinf-py/Cargo.toml
pip install -e "python/apxinf[tokenizer,serving]"
```

`maturin develop` compiles the binding into the *active* environment, so a venv
or conda env has to be activated first. The `[tokenizer,serving]` extras pull
the only runtime dependencies — numpy, Pillow, sentencepiece, msgpack,
websockets; there is no separate requirements file to install.

`--features cuda` is a Cargo feature, not a CUDA installation: it compiles the
CUDA backend into the binding, and it is required — the PI0.5 runtime is only
registered on CUDA devices.

The build queries the visible GPU for its compute capability and compiles the
kernels for exactly that architecture, so build on the machine you deploy to;
cross-compiling fails unless `APXINF_CUDA_ARCH` names the target (`sm_87` Orin,
`sm_101` Thor-U, `sm_110` Thor).

To ship a wheel instead of installing in place, build it with
`maturin build --release --features cuda --auditwheel skip`. The skip matters on
Jetson: the default `auditwheel` repair vendors a `libcuda` stub into the wheel,
and the installed binding then fails at runtime with CUDA error 304.

Confirm the binding imports and reaches the GPU:

```bash
python -c 'import apxinf_py; print(apxinf_py.__version__)'
python -m evaluation.benchmarks.pi05 \
  --random-weights --precision bf16 --interface model --samples 5
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
    action_dim=7,           # deployable width; None keeps the model's full vector
    action_horizon=10,      # a sequence length, so it outranks config.json
    discrete_state=False,   # inject state into the prompt, or drop it
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
from apxinf import build_robot_policy
from apxinf.serving import WebsocketPolicyServer

policy = build_robot_policy("unitree_g1", "<path-to-model>", precision="bf16")
WebsocketPolicyServer(policy, "0.0.0.0", 8000).serve_forever()
```

`--robot` selects the wire contract — camera keys, state routing, deployable
action width — the way OpenPI selects a `TrainConfig`. It is not negotiated at
connect time, so a checkpoint fine-tuned for another robot must name its preset;
a mismatch produces wrong actions, not an error.

| Preset | Cameras | State | Action |
|---|---|---|---|
| `franka_libero` | `observation/image`, `observation/wrist_image` | 8-dim, dropped | 7-dim EEF delta |
| `unitree_g1` | 3 views | 16-dim, discretized into the prompt | delta joints, 32→16 encode |

`--help` lists every preset with its slot→key mapping. If your client already
speaks a fixed dialect, match it with `--image-keys`, `--state-key`,
`--action-dim`, `--discrete-state` instead of editing the client. To register a
new robot, see [Adding an embodiment](../../doc/adding-an-embodiment.md).

The server serves the checkpoint's native chunk length unless `--action-horizon`
says otherwise, and publishes the wire contract it settled on in its
connect-time metadata, so a client can assert it rather than guess.
`--random-weights` starts a checkpoint-free server for transport and latency
measurement.


## Precisions

### BF16

The default, on every supported device. Runs on the checkpoint alone; no
calibration.

```bash
python scripts/pi05_openpi_websocket_server.py \
  --model-dir <path-to-model> --robot franka_libero --precision bf16 \
  --port 8000
```

```python
policy = AutoPolicy.from_pretrained("<path-to-model>", precision="bf16")
```

### FP8

Thor only, where it is the fastest path. Orin has no FP8 Tensor Cores and is not
supported.

FP8 is the one precision that needs an artifact beyond the weights: per-tensor
activation scales, and the load fails without them. A `calibration.json` in the
checkpoint directory is picked up automatically, so `--calibration` is only for
pointing elsewhere:

```bash
python scripts/pi05_openpi_websocket_server.py \
  --model-dir <path-to-model> --robot franka_libero --precision fp8 \
  --port 8000
```

```python
policy = AutoPolicy.from_pretrained(
    "<path-to-model>",
    precision="fp8",
    calibration="<path-to-calibration.json>",   # default: <model-dir>/calibration.json
)
```

The calibration is per-tensor activation scales from a calibration sweep and
decides accuracy; `uniform:SCALE` is a flat stand-in for latency work only.

### INT8

W8A8, optimized for Orin (SM87) and Ada (SM89). Needs nothing beyond the
checkpoint.

```bash
python scripts/pi05_openpi_websocket_server.py \
  --model-dir <path-to-model> --robot franka_libero --precision int8 --port 8000
```

```python
policy = AutoPolicy.from_pretrained("<path-to-model>", precision="int8")
```


## LIBERO evaluation

The repository evaluation entry points are documented in
[`evaluation/README.md`](evaluation/README.md).

### Get the checkpoint

The published accuracy is `pi05_libero_base`, π0.5 fine-tuned on LIBERO — an
arbitrary π0.5 checkpoint might not reproduce it.

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download lerobot/pi05_libero_base --local-dir <path-to-model>
```

### Run

The rollout needs LIBERO and MuJoCo:

```bash
python -c 'from libero.libero import benchmark'
```

If that fails, install LIBERO from source:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git <path-to-libero>
pip install -r <path-to-libero>/requirements.txt   # robosuite brings MuJoCo; pins numpy==1.22.4
pip install -e <path-to-libero>
export MUJOCO_GL=egl                               # headless; osmesa if the machine has no EGL
```

That numpy pin is the one thing to watch: it predates Python 3.11, so on a newer
interpreter it has no wheel and builds from source. `--no-deps` skips it.

Client/server evaluation additionally needs `openpi-client`, from an openpi
checkout:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git <path-to-openpi>
pip install -e <path-to-openpi>/packages/openpi-client
```

The recommended setup keeps the inference host free of LIBERO dependencies.
Start the policy server in its inference environment:

```bash
python scripts/pi05_openpi_websocket_server.py \
  --model-dir <path-to-model> --robot franka_libero --precision bf16 --port 8000
```

Then run the simulator and accuracy client from a LIBERO environment:

```bash
python -m evaluation.libero.client \
  --host <server-host> --port 8000 --precision bf16 \
  --suite libero_10 --tasks all --trials-per-task 50 \
  --results-jsonl <out-dir>/results.jsonl --summary-json <out-dir>/summary.json
```

For a single-process development run, the dual-backend evaluator can build the
policy locally:

```bash
python -m evaluation.libero.eval \
  --backend in-process --model-dir <path-to-model> \
  --precision bf16 --action-horizon 10 \
  --suite libero_10 --tasks all --trials-per-task 50 \
  --results-jsonl <out-dir>/results.jsonl --summary-json <out-dir>/summary.json
```

That is the published protocol: all 10 LIBERO-10 tasks x 50 episodes at seed 7
(the default), 500 episodes in total.

### Options

- `--suite` picks the task suite, `--tasks` a comma list within it, and
  `--trials-per-task` the episode count; a smoke run is
  `--tasks 0 --trials-per-task 1`.
- `evaluation.libero.client` accepts only remote server selection; it cannot
  load a checkpoint or initialize CUDA.
- The local model flags — `--model-type`, `--action-horizon`, `--action-dim`,
  `--discrete-state`, FP8 `--calibration` — exist only on
  `evaluation.libero.eval --backend in-process`.
- With a remote server, model flags belong to the server and client
  `--precision` only asserts reported metadata, so a mismatch fails before a
  rollout instead of skewing a run.
- Runs are resumable: completed task/trial rows in the JSONL ledger are skipped,
  and the summary reports success rate alongside per-segment latency.


## Benchmark

`evaluation.benchmarks.pi05` times the concentric serving shells so a regression
can be attributed to the engine, the processors, or the transport.

```bash
python -m evaluation.benchmarks.pi05 \
  --model-dir <path-to-model> --precision bf16 --interface model,policy
```

- `--interface` selects any subset of `model` (`Model.infer_rgb`), `policy`
  (`Pi05Policy.infer`), and `websocket` (round trip). The websocket interface
  attaches to a running server and needs no local weights.
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

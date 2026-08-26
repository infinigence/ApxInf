# ApxInf

ApxInf is a Rust inference engine for autoregressive LLM/VLM generation and
PI0.5 policy inference, with CUDA implementations for PI0.5 in FP8, BF16, and
W8A8 INT8. Run the commands below from the repository root on the NVIDIA
target machine.

The CUDA kernels, CUTLASS, and FlashAttention sources needed by PI0.5 are
vendored in this repository, so no external source checkout is required.

## PI0.5 Performance and Accuracy

ApxInf provides native PI0.5 CUDA paths for Jetson AGX Thor (SM110) and Jetson AGX
Orin (SM87). The performance results below use the primary two-view, T=10 LIBERO
workload with batch 1, 224 x 224 NHWC `uint8` images, 10 flow-matching steps,
`H=10`, 10 warm-up iterations, and 30 measured samples.

**Best published ApxInf PI0.5 performance:** Thor FP8 reaches
**41.159 ms (24.3 Hz)** on the performance workload.

### Performance

Latency is steady-state CUDA Graph replay P50. Throughput is the corresponding
single-stream policy inference rate.

| Hardware | Mode | Latency | Throughput |
|---|---|---:|---:|
| Jetson AGX Thor | BF16 | 72.454 ms | 13.8 Hz |
| Jetson AGX Thor | FP8 | **41.159 ms** | **24.3 Hz** |
| Jetson AGX Orin | BF16 | 165.665 ms | 6.0 Hz |
| RTX 4090 | BF16 | 31.38 ms | 31.9 Hz |
| RTX 4090 | INT8 | 25.99 ms | 38.5 Hz |

### LIBERO accuracy

The formal accuracy protocol uses two views, T=10, `H=50`, 500 episodes, and
`replan=5`. The official PI0.5 reference success rate is 92.4%. ApxInf results
will be filled in after the corresponding 500-episode runs are complete.

| Hardware | Mode | Trials | Success | Rate |
|---|---|---:|---:|---:|
| Jetson AGX Thor | BF16 | 500 | TBD | TBD |
| Jetson AGX Thor | FP8 | 500 | TBD | TBD |
| Jetson AGX Orin | BF16 | 500 | TBD | TBD |

## 1. NVIDIA build environment

Use a Linux host with an NVIDIA driver and a complete CUDA toolkit. The build
needs:

- `nvcc`, CUDA headers, and the CUDA runtime;
- cuBLAS and cuBLASLt development libraries;
- NVTX (`libnvToolsExt` on many Jetson systems or `libnvtx3interop` on desktop
  CUDA installations);
- a C/C++ compiler, linker, `ar`, Git, `pkg-config`, and Python 3.

On Ubuntu, install the non-CUDA tools with:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential curl git pkg-config \
  python3 python3-pip python3-venv
```

Install the NVIDIA driver and CUDA toolkit through the JetPack, DRIVE OS, or
CUDA distribution appropriate for the machine. Known-good configurations are:

| Device | CUDA architecture | Validated toolkit |
|---|---:|---:|
| Jetson Thor | `sm_110` | CUDA 13.0 |
| Thor-U | `sm_101` | CUDA 12.8 |
| Jetson AGX Orin | `sm_87` | CUDA 12.6 and 13.2 |
| RTX 4090 | `sm_89` | CUDA 12.8 |

Set the toolkit path before building. On a native build, ApxInf queries the
CUDA runtime and automatically selects the architecture of the visible GPU:

```bash
export CUDA_PATH=/usr/local/cuda
export PATH="${CUDA_PATH}/bin:$PATH"
unset APXINF_CUDA_ARCH APXINF_CUDA_ARCH_CUTLASS

nvcc --version
test -f "${CUDA_PATH}/include/cuda_runtime.h"
```

The build itself locates `nvcc` through `CUDA_PATH`; the `PATH` entry is what
makes the `nvcc --version` check above work on a shell that has no CUDA on its
default path.

Set `APXINF_CUDA_ARCH` explicitly when cross-compiling, building in a container
without a visible GPU, or producing binaries for a different machine. Prefer a
one-shot override so later native builds still use hardware detection:

```bash
APXINF_CUDA_ARCH=sm_110 cargo build --release --features cuda
# Use sm_101 for Thor-U or sm_87 for Orin.
```

The override always takes precedence over hardware detection. If visible GPUs
have different compute capabilities, restrict the build with
`CUDA_VISIBLE_DEVICES` or set `APXINF_CUDA_ARCH`. The corresponding
architecture-specific CUTLASS target (for example `sm_110a`) is selected
automatically and can be overridden with `APXINF_CUDA_ARCH_CUTLASS`.

If CUDA is installed elsewhere, point `CUDA_PATH` at that directory. The
runtime libraries must also be visible to the system dynamic linker.

## 2. Rust toolchain

Install the current stable Rust toolchain with `rustup`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup toolchain install stable --profile minimal
rustup default stable
rustc --version
cargo --version
```

The current code has been built with Rust 1.95 and 1.96. The repository does
not currently declare an older minimum supported Rust version.

## 3. Build

Build the main ApxInf binary:

```bash
cargo build --release --features cuda
```

Build the unified PI0.5 benchmark (one example drives BF16 / FP8 / INT8 via
`--dtype`):

```bash
cargo build --release -p apxinf-model --features cuda --example pi05_bench
```

The resulting `pi05_bench` executable is under `target/release/examples/`. Set a
separate `CARGO_TARGET_DIR` when keeping builds for multiple GPU architectures
in the same checkout.

### Checkpoint-free quickstart (no download)

`pi05_bench` runs with deterministic **random weights** — graph-replay latency
depends only on tensor shape and dtype, so no checkpoint is needed to measure the
engine. Pass `random` in place of a checkpoint path:

```bash
# BF16 engine floor, 2 views, 10-token prompt
target/release/examples/pi05_bench random --dtype bf16 --views 2 --token-count 10

# FP8 (synthetic uniform activation scale; no calibration/tactics required)
target/release/examples/pi05_bench random --dtype fp8 --views 2 --token-count 10

# INT8 W8A8, 3 views
target/release/examples/pi05_bench random --dtype int8 --views 3 --token-count 21
```

Random mode still runs the eager-vs-graph integrity self-test; it rejects
`--reference` (there are no trained weights to match). It defaults to a reduced
`H=10` action horizon (the shape the published Thor numbers use); sweep the horizon
and other shapes with `--action-horizon/--num-flow-steps/--views/--image-size` —
these are synthetic-only knobs. A **real checkpoint runs its native config instead**
(see below).

## 4. Run LLM and VLM generation

`generate` detects the Hugging Face `model_type` and uses the same
`LlmInput`/`LlmTrait` pipeline for Llama and Qwen3-VL. Text-only generation:

```bash
cargo run --release --features cuda-no-nvtx -- generate \
  --model /path/to/model \
  --prompt "Describe CUDA graphs." \
  --device cuda --dtype bf16 --max-tokens 50
```

By default, `generate` reads model-recommended settings from
`generation_config.json`; missing fields fall back to ApxInf's historical
greedy defaults. Request flags override the model settings. Use `--greedy` to
force greedy decoding or `--sample` to force the backend-native random logits
pipeline; the seed identifies a reproducible counter-based random stream:

```bash
cargo run --release --features cuda-no-nvtx -- generate \
  --model /path/to/model \
  --prompt "Describe CUDA graphs." \
  --device cuda --dtype bf16 --max-tokens 50 \
  --sample --temperature 0.8 --top-k 40 --top-p 0.95 \
  --repetition-penalty 1.1 --seed 42
```

Use `--generation-config apxinf` to ignore the model file, or pass a JSON file
or directory instead of `auto`. Deployment defaults can be layered with
`--override-generation-config '{"temperature":0.7,"top_p":0.9}'`.
Supported JSON fields are `max_new_tokens`, `eos_token_id` (scalar or list),
`do_sample`, `temperature`, `top_k`, `top_p`, and the repetition/frequency/
presence penalties; unrelated Hugging Face fields are ignored.

For Qwen3-VL, add `--image`. The CLI shells out to the Hugging Face processor to
turn the image into `pixel_values` + `image_grid_thw`, so that Python environment
needs:

```bash
python3 -m pip install "transformers>=4.57" torch torchvision Pillow numpy
```

These four packages are needed **only for `--image`**. Text-only `generate` and
everything under [Run PI0.5](#5-run-pi05) call no Python at all — the subprocess
is spawned from the `--image` branch alone.

Qwen3-VL landed in `transformers` 4.57.0, so anything older cannot build its
processor: the run returns no `pixel_values` and fails with
`KeyError: 'pixel_values'`. `torch` and `torchvision` are never called by ApxInf
itself — `AutoProcessor` for Qwen3-VL constructs a `Qwen3VLVideoProcessor`
sub-processor that hard-requires both even for a still image, and in
`transformers` 5.x every image processor is torchvision-backed, so the import
fails before the file is opened. Any build of torch for the machine will do; the
tensors it produces are copied straight out to `.npy` and never touch a GPU.

```bash
cargo run --release --features cuda-no-nvtx -- generate \
  --model /path/to/Qwen3-VL-2B-Instruct \
  --image /path/to/image.jpg \
  --prompt "What is in this image?" \
  --device cuda --dtype bf16 --max-tokens 50
```

`generate` exits non-zero when preprocessing, loading, or generation fails, so it
is safe to chain in a script.

See the [sampling subsystem documentation](doc/20260819-sampling-subsystem/README.md)
for the sampling API and backend design.

## 5. Run PI0.5

### Model and common paths

This repository ships **no checkpoint**. The checkpoint-free quickstart above
needs no download; the commands below (real-weight benchmarks, websocket serving,
LIBERO) need a `model.safetensors` on disk. Websocket serving also needs
`norm_stats.json` and either `tokenizer.model` or `paligemma_tokenizer.model`.

#### Pull a pi05 checkpoint

pi05 weights come from the OpenPI π0.5 release; export them to a
`model.safetensors` in a model directory. One recipe using the Hugging Face CLI:

```bash
python3 -m pip install -U "huggingface_hub[cli]"
export APXINF_MODEL_DIR=/path/to/pi05_libero_base
huggingface-cli download <org/pi05-repo> \
  --local-dir "$APXINF_MODEL_DIR" \
  --include "model.safetensors" "config.json" "norm_stats.json" "*tokenizer.model"
```

Substitute the π0.5 repo you have access to. Each future model registered with
`AutoModel` adds its own pull recipe plus a registry entry; the benchmark and
serving flags stay the same (`--dtype` / `--precision`, `--model`).

Set these paths once for the following commands:

```bash
export APXINF_MODEL_DIR=/path/to/pi05_libero_base
export APXINF_CHECKPOINT="${APXINF_MODEL_DIR}/model.safetensors"
export APXINF_TACTICS=configs/pi05/thor_sm110_cutlass_tactics.json
export APXINF_EXAMPLES=target/release/examples
```

For Thor-U, use `configs/pi05/thor_u_cutlass_tactics.json`. Orin does not have
native FP8 Tensor Cores; its FP8 compatibility path accepts the tactic file but
does not use its GEMM selections.

#### FP8 activation calibration

FP8 needs per-tensor activation scales. `--calibration` takes either a
calibration JSON or `uniform:SCALE`, an unprofiled flat scale:

```bash
# latency/smoke only — a flat scale, not a real activation profile
export APXINF_CALIBRATION=uniform:1.0
```

`uniform:` is enough to measure the FP8 engine and to prove the path runs, and it
is what the FP8 benchmark commands below use. **It is not valid for accuracy**:
use a real profile before reading anything into FP8 rollout success rates.

No calibration file ships with this repository or with a π0.5 checkpoint. Build
one from a BF16 activation sweep over your own checkpoint:

```bash
# 1. record BF16 activation amax (needs the FlashRT torch frontend + a GPU)
python3 scripts/pi05_bf16_zero_reference.py \
  --checkpoint "$APXINF_CHECKPOINT" \
  --output /tmp/pi05-bf16-oracle.json \
  --calibration-output /tmp/pi05-raw-calibration.json

# 2. turn it into a conservative FP8 profile
python3 scripts/pi05_prepare_libero_calibration.py \
  --input /tmp/pi05-raw-calibration.json \
  --output "${APXINF_MODEL_DIR}/calibration.json" --margin 2.35

export APXINF_CALIBRATION="${APXINF_MODEL_DIR}/calibration.json"
```

Step 2 is a bootstrap profile derived from a zero fixture, not a production
calibration; validate it behaviorally against BF16 before you rely on it. If a
`calibration.json` sits in the model directory the loader picks it up
automatically, so `--calibration` is only needed to point elsewhere.

### Benchmark FP8, BF16, and INT8

The following commands benchmark a checkpoint with a representative 21-token
prompt for 30 iterations, selecting the dtype with `--dtype`.
`APXINF_PI05_IMAGE_INPUT=nhwc` includes the captured CUDA path from raw,
already-resized `uint8 [2,224,224,3]` RGB images through normalization,
patchification, and policy inference.

FP8:

```bash
APXINF_PI05_IMAGE_INPUT=nhwc \
"${APXINF_EXAMPLES}/pi05_bench" "$APXINF_CHECKPOINT" --dtype fp8 \
  --calibration "$APXINF_CALIBRATION" --tactics "$APXINF_TACTICS" \
  --token-count 21 --iterations 30
```

BF16:

```bash
APXINF_PI05_IMAGE_INPUT=nhwc \
"${APXINF_EXAMPLES}/pi05_bench" "$APXINF_CHECKPOINT" --dtype bf16 \
  --token-count 21 --iterations 30
```

W8A8 INT8:

```bash
APXINF_PI05_IMAGE_INPUT=nhwc \
"${APXINF_EXAMPLES}/pi05_bench" "$APXINF_CHECKPOINT" --dtype int8 \
  --token-count 21 --iterations 30
```

Use `patches`, `nhwc`, or `nchw` for `APXINF_PI05_IMAGE_INPUT` (or the
`--image-input` flag). Native FP8 is the optimized path on Thor/Thor-U. BF16 and
INT8 are currently optimized primarily for SM87 Orin; FP8 on Orin is a
correctness-oriented decode-to-FP16 compatibility path.

> **Horizon contract.** A checkpoint **defaults** to the native config read from
> `config.json` — `pi05_libero_base` emits `H=50`, the same chunk the LIBERO eval
> and the websocket server run (the rollout then *executes* `replan_steps` of each
> chunk; `H` is what the model *predicts*, not what is executed). The reduced
> `H=10` figures in [the PI0.5 CUDA regression protocol](doc/pi05-cuda-regression.md)
> are the **synthetic** workload, reproduced with
> `pi05_bench random --action-horizon 10`.
>
> `--action-horizon` also applies to a real checkpoint and **outranks
> `config.json`**, on `pi05_bench`, `bench_pi05.py`, `eval_libero.py`, and
> `pi05_openpi_websocket_server.py` alike — the horizon is a sequence length, not
> a weight dimension, so the same weights run at the requested chunk length. That
> is what makes an apples-to-apples `H=10` comparison against the synthetic
> numbers possible. The remaining architecture overrides (`--views`,
> `--image-size`, `--num-flow-steps`, `--max-token-len`) do reshape weights and
> stay rejected on a checkpoint.

PI0.5 accepts an exact caller-supplied initial latent for debugging and parity
checks. When no latent is supplied, it fills the model's stable CUDA latent
buffer from the internal counter-based random stream, avoiding a host noise
allocation and upload. The explicit seeded APIs remain available for exact
replay.

### Python environment

The layered benchmark, the websocket server, and the LIBERO evaluator all load
the model in-process through the `apxinf_py` PyO3 binding, so build and install
it — plus the pure-Python `apxinf` frontend — before running any of them:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r scripts/requirements-pi05-websocket.txt

# `maturin develop` compiles the binding into the active virtualenv
python3 -m pip install maturin
APXINF_CUDA_ARCH=sm_110 CUDA_PATH=/usr/local/cuda \
  maturin develop --release --features cuda -m crates/apxinf-py/Cargo.toml
python3 -m pip install -e python/apxinf
```

`maturin develop` installs into the active virtualenv and does not run
`auditwheel`. If you instead build a redistributable wheel on Jetson, use
`maturin build --release --features cuda --auditwheel skip` — the default
repair vendors a `libcuda` stub into the wheel, which makes the installed
binding fail at runtime with CUDA error 304.

The server itself does not import OpenPI. The smoke test, robot clients, and
LIBERO evaluator use the official `openpi-client`; install it from an OpenPI
checkout:

```bash
python3 -m pip install -e /path/to/openpi/packages/openpi-client
```

Verify the binding imports before going further:

```bash
python3 -c 'import apxinf_py; print(apxinf_py.__version__)'
```

### Layered Python latency (L0–L3)

`scripts/bench_pi05.py` measures the concentric serving shells — L0 (`_infer_patches`,
the engine floor) ⊂ L1 (`infer_rgb`) ⊂ L2 (`Pi05Policy.infer`) ⊂ L3 (websocket
round trip). With no `--model-dir` it runs **checkpoint-free** on synthetic weights and
defaults to L0/L1; add `l2` and it wraps the engine in synthetic processors (a
fixed-length tokenizer + identity unnormalize, so L2's actions are latency-only).
`--model-dir` runs a real checkpoint at its native horizon and defaults to every layer.
L3 attaches to a running server and needs no local weights — start that server with
`--random-weights` for a fully checkpoint-free L3. See [the PI0.5 CUDA regression protocol](doc/pi05-cuda-regression.md) for
the sampling protocol and reference numbers.

```bash
# checkpoint-free engine floor — the zero-config default
python3 scripts/bench_pi05.py --precision bf16 --views 2 --token-count 10

# checkpoint-free L0/L1/L2 (synthetic processors; latency-only actions)
python3 scripts/bench_pi05.py --layer l0,l1,l2 --precision bf16 --views 2 --token-count 10

# full in-process breakdown against a checkpoint (native horizon, e.g. H=50)
python3 scripts/bench_pi05.py --model-dir "$APXINF_MODEL_DIR" --layer l0,l1,l2 \
  --precision bf16 --prompt "put both moka pots on the stove"

# the same checkpoint forced to H=10, comparable to the synthetic numbers
python3 scripts/bench_pi05.py --model-dir "$APXINF_MODEL_DIR" --layer l0,l1,l2 \
  --precision bf16 --action-horizon 10

# L3 against a running websocket server
python3 scripts/bench_pi05.py --layer l3 --precision bf16 \
  --host 127.0.0.1 --port 8000 --prompt "put both moka pots on the stove"
```

### Start an OpenPI-compatible websocket server

The server loads the model in-process through the `apxinf_py` PyO3 binding (no
subprocess), so activate the environment built in
[Python environment](#python-environment) first:

```bash
source .venv/bin/activate
```

`--robot` selects the **embodiment** — the camera wire keys, the state routing,
and the deployable action width — the way OpenPI's `serve_policy.py
--policy.config <TrainConfig>` does. It defaults to `franka_libero`; a checkpoint
fine-tuned for another robot must name its own preset, because a mismatch
degrades silently rather than failing. `--help` lists every registered preset
with its slot→key mapping. See
[Adding an embodiment](doc/adding-an-embodiment.md) for the full launch guide and
for how to register a new robot.

Start FP8 on port 8000:

```bash
python3 scripts/pi05_openpi_websocket_server.py \
  --model-dir "$APXINF_MODEL_DIR" \
  --calibration "$APXINF_CALIBRATION" \
  --tactics "$APXINF_TACTICS" \
  --precision fp8 --host 0.0.0.0 --port 8000
```

BF16 and INT8 do not require calibration or tactics:

```bash
# BF16
python3 scripts/pi05_openpi_websocket_server.py \
  --model-dir "$APXINF_MODEL_DIR" \
  --precision bf16 --host 0.0.0.0 --port 8000

# W8A8 INT8
python3 scripts/pi05_openpi_websocket_server.py \
  --model-dir "$APXINF_MODEL_DIR" \
  --precision int8 --host 0.0.0.0 --port 8000
```

The server logs the shape it ended up serving (`serving H=... x D=...`) and
publishes it in its connect-time metadata. It serves the checkpoint's native
horizon unless `--action-horizon` says otherwise:

```bash
python3 scripts/pi05_openpi_websocket_server.py \
  --model-dir "$APXINF_MODEL_DIR" \
  --precision bf16 --action-horizon 10 --host 0.0.0.0 --port 8000
```

For a **checkpoint-free** server (transport/serving latency with no weights on disk),
pass `--random-weights` instead of `--model-dir`. It serves the engine on synthetic
weights and synthetic processors, so its actions are latency-only; shape knobs
(`--num-views/--image-size/--action-horizon/--num-flow-steps/--token-count`) select the
workload. This is what backs a fully checkpoint-free L3 measurement:

```bash
python3 scripts/pi05_openpi_websocket_server.py \
  --random-weights --precision bf16 --num-views 2 --token-count 10 \
  --host 0.0.0.0 --port 8000
```

Run a smoke test from another terminal, changing the expected precision to
match the server:

```bash
source .venv/bin/activate
python3 scripts/test_pi05_openpi_websocket.py \
  --host 127.0.0.1 --port 8000 \
  --expected-precision fp8 --requests 3
```

The smoke test asserts the action shape the server advertises, so it follows a
checkpoint's native horizon with no extra flags. Pass `--action-horizon` /
`--action-dim` to assert an exact shape instead.

### Evaluate LIBERO-10

Run the evaluator in a Python environment where LIBERO, MuJoCo, its simulator
dependencies and assets, and `openpi-client` are installed. LIBERO and MuJoCo are
not part of `requirements-pi05-websocket.txt` — install them from the LIBERO
distribution you use. Verify the two main imports first:

```bash
python3 -c 'from libero.libero import benchmark; from openpi_client import websocket_client_policy'
```

The evaluator and the server do not have to share a process or even a machine, so
this environment only needs the simulator and the client.

Start the websocket server first, at the precision you are evaluating; `--precision`
here asserts what the server reports. BF16 needs no calibration, so it is the
shortest path to a working run:

```bash
MUJOCO_GL=egl python3 scripts/eval_libero.py --backend websocket \
  --host 127.0.0.1 --port 8000 --precision bf16 \
  --tasks 0 --trials-per-task 1 \
  --results-jsonl /tmp/pi05-bf16-libero-smoke/results.jsonl \
  --summary-json /tmp/pi05-bf16-libero-smoke/summary.json
```

Run the complete LIBERO-10 evaluation (10 tasks, 10 trials each):

```bash
MUJOCO_GL=egl python3 scripts/eval_libero.py --backend websocket \
  --host 127.0.0.1 --port 8000 --precision bf16 \
  --tasks 0,1,2,3,4,5,6,7,8,9 \
  --trials-per-task 10 \
  --results-jsonl /tmp/pi05-bf16-libero10/results.jsonl \
  --summary-json /tmp/pi05-bf16-libero10/summary.json
```

Use `--precision fp8` or `--precision int8` against a server started at that
precision, and use a separate results directory for each. FP8 rollout success
rates are only meaningful with a real activation profile — a `uniform:` scale is
a latency/smoke setting, not a calibration.

Use `--backend in-process --model-dir "$APXINF_MODEL_DIR"` to evaluate without a
running server (the policy is built in-process through `apxinf_py`); the other
flags are unchanged. In that mode `--action-horizon` overrides the checkpoint's
chunk length; with `--backend websocket` the horizon belongs to the server, so
pass it there instead.

The evaluator is resumable: completed task/trial rows in the JSONL ledger are
skipped on the next run. If the evaluator and server are on different machines,
replace `127.0.0.1` with the server's reachable IP address.

## License

ApxInf is licensed under the [Apache License 2.0](LICENSE). Vendored third-party
components retain their respective copyright notices and licenses.

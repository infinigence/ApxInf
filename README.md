# ApxInf

ApxInf is a Rust inference engine for autoregressive LLM/VLM generation and
PI0.5 policy inference, with CUDA implementations for PI0.5 in FP8, BF16, and
W8A8 INT8. Run the commands below from the repository root on the NVIDIA
target machine.

The CUDA kernels, CUTLASS, and FlashAttention sources needed by PI0.5 are
vendored in this repository, so no external source checkout is required.

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

Set the toolkit path and the architecture before building. Do not omit the
architecture on Jetson:

```bash
export CUDA_PATH=/usr/local/cuda
export APXINF_CUDA_ARCH=sm_110  # use sm_101 for Thor-U or sm_87 for Orin

nvcc --version
test -f "${CUDA_PATH}/include/cuda_runtime.h"
```

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

For Qwen3-VL, add `--image`. The CLI image processor currently requires
Python packages `transformers`, `Pillow`, and `numpy`:

```bash
cargo run --release --features cuda-no-nvtx -- generate \
  --model /path/to/Qwen3-VL-2B-Instruct \
  --image /path/to/image.jpg \
  --prompt "What is in this image?" \
  --device cuda --dtype bf16 --max-tokens 50
```

See `doc/20260817-unified-llm-vlm-interface.md` for the runtime architecture
and Rust API.

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
export APXINF_CALIBRATION=doc/20260804-pi05/evidence/libero10-100-m2.35/calibration.json
export APXINF_TACTICS=configs/pi05/thor_sm110_cutlass_tactics.json
export APXINF_EXAMPLES=target/release/examples
```

For Thor-U, use `configs/pi05/thor_u_cutlass_tactics.json`. Orin does not have
native FP8 Tensor Cores; its FP8 compatibility path accepts the tactic file but
does not use its GEMM selections.

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

> **Horizon contract.** A checkpoint is benchmarked at its **native** config read
> from `config.json` — `pi05_libero_base` emits `H=50`, the same chunk the LIBERO
> eval and the websocket server run (the rollout then *executes* `replan_steps` of
> each chunk; `H` is what the model *predicts*, not what is executed). The reduced
> `H=10` figures in [benchmark.md](benchmark.md) are the **synthetic** workload,
> reproduced with `pi05_bench random --action-horizon 10`. Architecture overrides
> are rejected on a checkpoint so it stays faithful to its config.

### Layered Python latency (L0–L3)

`scripts/bench_pi05.py` measures the concentric serving shells — L0 (`_infer_patches`,
the engine floor) ⊂ L1 (`infer_rgb`) ⊂ L2 (`Pi05Policy.infer`) ⊂ L3 (websocket
round trip). With no `--model-dir` it runs **checkpoint-free** on synthetic weights and
defaults to L0/L1; add `l2` and it wraps the engine in synthetic processors (a
fixed-length tokenizer + identity unnormalize, so L2's actions are latency-only).
`--model-dir` runs a real checkpoint at its native horizon and defaults to every layer.
L3 attaches to a running server and needs no local weights — start that server with
`--random-weights` for a fully checkpoint-free L3. See [benchmark.md](benchmark.md) for
the sampling protocol and reference numbers.

```bash
# checkpoint-free engine floor — the zero-config default
python3 scripts/bench_pi05.py --precision bf16 --views 2 --token-count 10

# checkpoint-free L0/L1/L2 (synthetic processors; latency-only actions)
python3 scripts/bench_pi05.py --layer l0,l1,l2 --precision bf16 --views 2 --token-count 10

# full in-process breakdown against a checkpoint (native horizon, e.g. H=50)
python3 scripts/bench_pi05.py --model-dir "$APXINF_MODEL_DIR" --layer l0,l1,l2 \
  --precision bf16 --prompt "put both moka pots on the stove"

# L3 against a running websocket server
python3 scripts/bench_pi05.py --layer l3 --precision bf16 \
  --host 127.0.0.1 --port 8000 --prompt "put both moka pots on the stove"
```

### Start an OpenPI-compatible websocket server

Install the Python transport dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r scripts/requirements-pi05-websocket.txt
```

The server loads the model in-process through the `apxinf_py` PyO3 binding (no
subprocess), so build/install it and the `apxinf` frontend package into the same
environment:

```bash
APXINF_CUDA_ARCH=sm_110 CUDA_PATH=/usr/local/cuda \
  maturin develop --release --features cuda -m crates/apxinf-py/Cargo.toml
python3 -m pip install -e python/apxinf
```

The server itself does not import OpenPI. The smoke test, robot clients, and
LIBERO evaluator use the official `openpi-client`; install it from an OpenPI
checkout:

```bash
python3 -m pip install -e /path/to/openpi/packages/openpi-client
```

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

### Evaluate LIBERO-10

Run the evaluator in a Python environment where LIBERO, MuJoCo, its simulator
dependencies and assets, and `openpi-client` are installed. Verify the two
main imports first:

```bash
python3 -c 'from libero.libero import benchmark; from openpi_client import websocket_client_policy'
```

Start the websocket server first. Then run a one-episode integration check:

```bash
MUJOCO_GL=egl python3 scripts/eval_libero.py --backend websocket \
  --host 127.0.0.1 --port 8000 --precision fp8 \
  --tasks 0 --trials-per-task 1 \
  --results-jsonl /tmp/pi05-fp8-libero-smoke/results.jsonl \
  --summary-json /tmp/pi05-fp8-libero-smoke/summary.json
```

Run the complete LIBERO-10 evaluation (10 tasks, 10 trials each):

```bash
MUJOCO_GL=egl python3 scripts/eval_libero.py --backend websocket \
  --host 127.0.0.1 --port 8000 --precision fp8 \
  --tasks 0,1,2,3,4,5,6,7,8,9 \
  --trials-per-task 10 \
  --results-jsonl /tmp/pi05-fp8-libero10/results.jsonl \
  --summary-json /tmp/pi05-fp8-libero10/summary.json
```

Use `--backend in-process --model-dir "$APXINF_MODEL_DIR"` to evaluate without a
running server (the policy is built in-process through `apxinf_py`); the other
flags are unchanged.

Use `--precision bf16` or `--precision int8` when connected to that server,
and use a separate results directory for each precision. The evaluator is
resumable: completed task/trial rows in the JSONL ledger are skipped on the
next run. If the evaluator and server are on different machines, replace
`127.0.0.1` with the server's reachable IP address.

More PI0.5 implementation, correctness, and performance details are in
[`doc/20260804-pi05/implementation.md`](doc/20260804-pi05/implementation.md)
and
[`doc/20260804-pi05/openpi-websocket.md`](doc/20260804-pi05/openpi-websocket.md).

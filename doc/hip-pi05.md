# PI0.5 on AMD (HIP)

`crates/apxinf-model/src/hip_pi05/` is a second PI0.5 implementation, written
directly against `apxinf_hip::native` the way the CUDA one (`pi05/`) is written
against `apxinf-cuda-new`. It targets **gfx1151** (Radeon 8060S, Strix Halo) and
runs **BF16, eager**. Graph capture and kernel work are the next milestones; this
one establishes a correct baseline reachable from every user entry point.

## Layout: a self-contained port

`hip_pi05` imports no other model module. It owns copies of everything above the
device as well as everything below it:

| File | Owns |
|---|---|
| `config.rs` | checkpoint `config.json` parsing (LeRobot/OpenPI aliases), defaults, validation, time-embedding math |
| `checkpoint.rs` | SafeTensors reading, `[out, in]` → `[in, out]` transposes, Gemma `1 + w`, norm folding, synthetic weights |
| `weights.rs` | BF16 device weights, Q/K/V and gate/up packed into single GEMMs |
| `model.rs` | layer computations and stage methods |
| `runner.rs` | prepared plan and `VlaRuntime` |
| `load.rs` | loader |

The copies are deliberate: a change made for the CUDA implementation (its FP8
schedule, say) cannot alter HIP's results, and the reverse. Agreement between
the two comes from passing the same accuracy fixtures, not from shared code, and
`scripts/check_model_family_boundaries.sh` enforces that neither imports the
other. Frontend overrides arrive in `LoadOptions::config`, typed as
`pi05::Pi05Config` (Python passes view/horizon overrides through it); `AutoModel`
converts that into `hip_pi05::Config` before calling the port, so the conversion
lives outside both families.

The layers follow `pi05/model/blocks/bf16.rs` operator for operator, with the
same rounding points, so the two can be compared stage by stage. Differences
that do not change a value:

- Per-step modulation depends only on the fixed time schedule and is computed
  once at load, not every inference.
- Language layers write K/V straight into the prefix cache (which reserves the
  action rows), instead of copying them into it.
- Activations come from one `HipArena` per prepared plan, reset each run, so
  every run uses the same addresses. Inputs live in fixed buffers and nothing
  synchronizes mid-run — the preconditions graph capture will rely on.

`apxinf_hip::native` holds the operators (GEMM via hipBLAS; the norms and fused
residual norms, GeGLU, Euler step, gathers, QKV split with RoPE into a cache,
and non-causal attention). Each is tested against an F32 host reference at
pi05's shapes in `crates/apxinf-hip/tests/native.rs`.

Attention runs on Composable Kernel's flash-attention, vendored in
`crates/apxinf-hip/kernels/ck/`: CK forward for vision and the prefix (whose 8
query heads share one KV head and are folded into one head of 8x the rows), and
CK split-KV with 4 key splits for the action expert's 10 queries over the
prefix. `native::attention_kernel` makes that choice; apxinf's own kernel remains
the test reference and the fallback for head dims CK does not cover. CK rounds
the attention probabilities to BF16 before the value product; apxinf's kernel
keeps them in F32.

## Use it

Every user path dispatches on the device: `AutoModel::load_model` routes
`pi05` with `Device::Hip` to `hip_pi05::load`. Model variant `auto` or `bf16`
is accepted; anything else is an error.

```sh
# Rust bench on fixture inputs (accuracy gate + eager latency + optional stage dump)
cargo run --release -p apxinf-model --features hip --example pi05_hip_bench -- \
  <checkpoint>/model.safetensors --views 2 --action-horizon 10 --token-count 10 \
  --images-u8 <fixtures>/task_08_first_replan-v2.nhwc-u8.raw \
  --token-ids-u32le <fixtures>/task_08_first_replan.tokens.u32le \
  --noise-bf16-u16le <fixtures>/task_08_first_replan-v2.noise-bf16.raw \
  --reference <fixtures>/task_08_first_replan-v2.reference.json \
  [--dump-stages <dir>]

# Python: build the extension with HIP, then device="hip:0" everywhere
cd crates/apxinf-py && maturin develop --release --features hip,extension-module
python scripts/eval_libero.py --backend in-process --device hip:0 --precision bf16 \
  --model-dir <checkpoint> --norm-stats <checkpoint>/norm_stats.json --action-horizon 10 ...
```

`maturin develop --uv` is needed in a `uv` venv (no pip). The tokenizer crate
builds `sentencepiece-sys` with cmake; `pip install cmake` into the venv if the
host has none. ROCm 7.2's device linker on Ubuntu 26.04 needs the libxml2
workaround in `crates/apxinf-hip/README.md`.

## Comparing against CUDA

`pi05_stage_dump` (CUDA, `--features cuda`) and `pi05_hip_bench --dump-stages`
write the same stages as `.npy`: patches, vision tokens, prefix, every language
layer's K and V, the state after the first flow step, and the final actions.

```sh
python scripts/compare_pi05_stages.py <cuda-dump> <hip-dump>
```

prints cosine / relative L2 / max abs per stage in pipeline order; the first
stage whose agreement drops is where two implementations diverge.

## Status (2026-10-27)

Fixtures: the cross-backend bundle (`cross-backend-accuracy-20260922`), 2 views,
H = 10, 10 flow steps. Gate: `doc/pi05-cuda-regression.md` BF16 — cosine >= 0.999,
relative L2 <= 0.05 against the bundle's `reference.json`.

| | vs reference: cosine / rel. L2 | eager latency P50 |
|---|---|---|
| HIP BF16, T=10 | 0.999991 / 0.0050 | 159 ms |
| HIP BF16, T=21 | 0.999991 / 0.0046 | 161 ms |
| HIP BF16, T=10, apxinf's own attention kernel (2026-10-21) | 0.999992 / 0.0042 | 265 ms |
| CUDA BF16 (Thor), T=10, same fixtures | 0.999994 / 0.0034 | — |

Against the CUDA stage dumps: patches bit-exact; every later stage >= 0.9994
(the worst is a language layer's V, 0.99947). That is slightly further from CUDA
than apxinf's own attention kernel was (0.99968): CK's BF16 rounding of the
probabilities does not match CUDA's attention any more closely. Repeat runs are
bit-identical.

Where the 159 ms goes (rocprofv3, one steady-state run): the GPU is busy 152 ms.
GEMMs take 132 ms of that — the language MLP 55 ms, the action expert's M = 10
GEMMs 58 ms at a third of memory bandwidth, vision 13 ms — and attention 7 ms.
Submitting the run's 2,310 launches takes the host 150 ms, nearly as long as the
GPU runs, so further GPU speed-ups need graph capture to show.

Python: `crates/apxinf-py/tests/test_smoke_pi05.py` passes on `hip:0` (7 tests).
LIBERO: in-process `eval_libero.py --device hip:0` runs through `AutoPolicy`.
LIBERO-10 tasks 0 and 1, 5 trials each (seed 7, H = 10, replan 5): **10/10
successes**, 162 ms model time per call (266 ms with apxinf's own attention
kernel, also 10/10). This is a smoke check, not the published 500-episode
protocol.

## Next

1. **Graph capture**: capture a whole prepared run as one HIP graph; the plan
   already has fixed inputs and arena addresses, and the host now takes nearly
   as long to submit a run as the GPU takes to execute it.
2. **Action-expert GEMMs**: the M = 10 projections read weights at about a third
   of the 243 GB/s the device sustains; per-shape hipBLASLt or a GEMV-style
   kernel.
3. **Language MLP and fusion**: the prefix gate/up and down GEMMs (55 ms), and
   the CUDA path's fused epilogues (GeGLU into the GEMM, bias into residual
   norms).
4. **Full LIBERO-10** (500 episodes) as the regression gate for the above.

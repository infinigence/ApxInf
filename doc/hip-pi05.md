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
and one non-causal attention kernel). Each is tested against an F32 host
reference at pi05's shapes in `crates/apxinf-hip/tests/native.rs`.

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

## Status (2026-10-21)

Fixtures: the cross-backend bundle (`cross-backend-accuracy-20260922`), 2 views,
H = 10, 10 flow steps. Gate: `doc/pi05-cuda-regression.md` BF16 — cosine >= 0.999,
relative L2 <= 0.05 against the bundle's `reference.json`.

| | vs reference: cosine / rel. L2 | eager latency P50 |
|---|---|---|
| HIP BF16, T=10 | 0.999992 / 0.0042 | 265 ms |
| HIP BF16, T=21 | 0.999992 / 0.0042 | 267 ms |
| CUDA BF16 (Thor), T=10, same fixtures | 0.999994 / 0.0034 | — |

Against the CUDA stage dumps: patches bit-exact; vision and prefix cosine
0.99999; all 36 language K/V tensors >= 0.9997, drifting smoothly with depth;
final actions 0.999997. Repeat runs are bit-identical.

Where the 265 ms goes (stages synchronized individually): vision 43 ms, prefix
(18 language layers) 101 ms, ten flow steps 122 ms. The flow steps are about
3x the GEMM-bound estimate — 180 small kernels per step, unfused, plus the
correctness attention kernel — which is what graph capture and fusion target.

Python: `crates/apxinf-py/tests/test_smoke_pi05.py` passes on `hip:0` (7 tests).
LIBERO: in-process `eval_libero.py --device hip:0` runs through `AutoPolicy`.
LIBERO-10 tasks 0 and 1, 5 trials each (seed 7, H = 10, replan 5): **10/10
successes**, 266 ms model time per call. This is a smoke check, not the
published 500-episode protocol, which waits for a faster runtime.

## Next

1. **Graph capture**: capture a whole prepared run as one HIP graph; the plan
   already has fixed inputs and arena addresses.
2. **GEMM selection**: per-shape choice between hipBLAS and hipBLASLt (neither
   wins everywhere on gfx1151), then CK or hand-written kernels for the
   language MLP, which is half the GEMM time.
3. **Attention and fusion**: a tiled attention kernel and the CUDA path's fused
   epilogues (GeGLU into the GEMM, bias into residual norms).
4. **Full LIBERO-10** (500 episodes) once a run is fast enough to make it
   practical, as the regression gate for the above.

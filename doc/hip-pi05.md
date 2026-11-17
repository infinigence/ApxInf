# PI0.5 on AMD (HIP)

`crates/apxinf-model/src/hip_pi05/` is a second PI0.5 implementation, written
directly against `apxinf_hip::native` the way the CUDA one (`pi05/`) is written
against `apxinf-cuda-new`. It targets **gfx1151** (Radeon 8060S, Strix Halo) and
runs **BF16**, eager or captured. It follows the family's module layout:
`model/` (forward math), `model_runner/` (request validation, plan cache,
execution policy and graph capture), `weights/` (checkpoint mapping and the
device weight tree).

## Layout: a self-contained port

`hip_pi05` imports no other model module. It owns copies of everything above the
device as well as everything below it:

| File | Owns |
|---|---|
| `config.rs` | checkpoint `config.json` parsing (LeRobot/OpenPI aliases), defaults, validation |
| `math.rs` | the fixed flow-timestep sinusoidal embedding |
| `weights/host.rs` | SafeTensors reading, `[out, in]` → `[in, out]` transposes, Gemma `1 + w`, norm folding, synthetic weights |
| `weights/bf16.rs` | BF16 device weights, Q/K/V and gate/up packed into single GEMMs |
| `model/model.rs` | layer computations and stage methods |
| `model_runner/runner.rs` | `VlaRuntime` surface and the implicit plan cache |
| `model_runner/prepare.rs` | the prepared plan: fixed inputs, arena, warmup and graph capture |
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
prefix. `native::attention_kernel` makes that choice from the device architecture and
the geometry: the vendored CK dispatchers resolve on `gfx115*` only (split-KV on
`gfx11*`), so every other architecture, and any head dim CK does not cover, runs
apxinf's own kernel — which also remains the test reference. A CK call that
still finds no instance falls back to that kernel at run time instead of
failing. CK rounds
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

## Graph capture (2026-11-16)

A prepared plan under `PreferGraph`/`RequireGraph` warms up once eagerly (the
libraries' first-touch work must not be recorded) and then captures the forward
as one `HipGraph` launch — 2,311 nodes; input upload and `rgb_to_patches` stay
outside the graph. Acceptance is `tests/hip_pi05_graph.rs`: a `RequireGraph`
plan and an eager plan must agree bit for bit under the captured inputs *and*
changed ones, replays are counted, and the implicit `infer` path must report
`graph` and match. The bench interleaves graph and eager runs in the same
window and checks bit-identity.

Measured (same-process interleaved A/B): **no wall improvement** — graph P50
170.4 ms against eager 170.1. The "host-bound" premise behind capture is
falsified: the host's ~160 ms submission fully overlaps the GPU's ~160 ms busy
span, so the wall was GPU-span-bound all along, and replay's own node
scheduling replaces the 6.3 ms of device-side gaps it was meant to remove.
Capture pays once the busy span drops below host submission (the autotune
work below is the path there). The warmup also exposed and fixed a real
correctness hole: it read the `token_ids` buffer before any upload, and the
embedding kernel gathers `table[ids[i] * width]` unchecked, so the ids are
zeroed at allocation.

A machine-level caveat, not a code defect: this gfx1151 stack (amdgpu-dkms
6.16.6 on kernel 6.8.0-138, ROCm 7.1) intermittently drops the device under
dense kernel submission — kernel-mode page faults in runtime-internal memory,
MES queue failures, MODE2 resets, latching `status 719` into the process. The
pre-capture binary hits it too (~1 in 4 runs on a bad day); first-touch bursts
and graph replays raise the odds. `HIP_LAUNCH_BLOCKING=1` never faults and is
the escape hatch for acceptance runs; otherwise retry the run. Building the
dkms module for 6.8.0-149 fails (`pci_resize_resource` signature change), so
GRUB is pinned to 6.8.0-138 until the driver is updated.

## GEMM recipe autotune (2026-11-17)

`APXINF_HIP_GEMM_AUTOTUNE=1` measured every pi05 shape the GR00T-era recipe
missed — all of them, since the recipe held only GR00T shapes — and wrote the
winners back to `configs/tuning/amd/gfx1151/rocm7.1-hipblaslt100100/gemm.recipe`
(now 51 entries). The M = 10 action-expert projections and the M = 1
time-conditioning projections route to hipBLASLt with serialized tactics; the
language and vision GEMMs stay on hipBLAS, which already runs them at the
measured roof.

Same-window interleaved A/B (one binary; the only variable is the routing —
`APXINF_HIP_GEMM_PROVIDER=hipblas` forces the baseline):

| window | recipe P50 | forced-hipBLAS P50 | recovered |
|---|---|---|---|
| 1 | 138.1 ms | 166.9 ms | −28.8 ms |
| 2 | 141.5 ms | 172.5 ms | −31.0 ms |

168 → ~140 ms (−17%). The 2026-11-17 kernel trace puts all of it in the
action expert's M = 10 family: 58.1 → 31.6 ms of GPU time, now 79–86% of
memory bandwidth on `MT64x16` hipBLASLt tiles. (The M = 1 entries are the
time-conditioning projections, which run once at load, not per inference.)
Correctness against the
forced-hipBLAS output on the same fixture: cosine 0.9999995, relative L2
9.9e-4, max-abs 9.8e-4 — the BF16 gate passes with two orders of magnitude of
margin (the difference is tile/reduction order, which is legitimate).
Graph-vs-eager stays bit-identical. The wall is now below the host's eager
submission cost, so graph replay and eager converge at ~140 ms; with the
GEMM floor within reach, the remaining levers are the pointwise/attention
residuals documented in the roofline.

## Pointwise vectorization: geglu (2026-11-17)

The geglu kernel carried the same disease the GR00T campaign cured in
`norm.hip`: scalar 2-byte loads plus a per-element div/mod. A uint4
variant (8 output elements per thread, the per-element F32 expression
unchanged), bit-identical. Shipped behind `inner % 8 == 0` + 16-byte
alignment; the scalar kernel stays as fallback. Interleaved binary A/B
(three clean windows per arm): 136.4 → 134.8 ms P50 (−1.6 ms). In the model
the gain is all on the action side (180 calls, 16.2 → 4.7 µs); the language
call (522 × 16384 out, 51 MB moved) was already at the bandwidth wall and
stays at 223 µs. A shared-frequency-table variant of `split_qkv_rope` probed
*slower* (0.87× at the language shape — the per-block `powf` table build
costs more than it saves at 128 pairs) and was discarded. The other action
pointwise (`ada_gate_residual_rms_norm`) was already vectorized by the
GR00T campaign's norm work.

## Where the time goes (2026-11-17)

rocprofv3, eager, steady state: GPU busy 131.8 ms per forward (GEMM 112.7,
attention 8.9, pointwise 10.2) plus 5.7 ms of device-side gaps; the
per-operator floor is ~115 ms. Language and vision GEMMs and the language
GeGLU run at the roof. The residuals, largest first:

- **Action attention, closed 2026-11-17: the folded flash-decoding pair.**
  The CK split-KV + combine pair it replaces cost 27.5 µs per call × 180
  calls. `kernels/action_attention.hip` folds the 8 query heads (which share
  one KV head) into 80 rows, reads K/V about once, and lands the split
  reduction in a small combine kernel: 19.9 µs per pair warm (probe
  `devlocal/pi05-hip/perf/attention_probe.cpp`), and **−3.0 ms end to end**
  (same-process interleaved A/B, `APXINF_HIP_ATTENTION=ck` as the control:
  132.7–133.6 against 135.7–136.0 P50). F32 scores and probabilities — the
  reference kernel's numerics, unlike CK's BF16-rounded ones. Routed by
  `AttentionKernel::Folded` for one-batch small-GQA geometries on RDNA.
- **Split-K GEMMs, ~4–6 ms, closed at the library level.** `lang_down`
  (K = 16384) runs 80 workgroups at 25.6 TF/s; `act_down`/`act_out` run 16.
  hipBLASLt's tuner was offering zero workspace
  (`HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 0`), so split-K candidates
  were never measured; it now gets a 32 MiB buffer per handle and enumerates
  32 candidates. The 2026-11-17 re-tune of every pi05 shape found the
  workspace-using candidates do exist and all lose — `lang_down`'s best
  split-K runs 1.67x slower than its best plain tactic — and the re-tuned
  winners regress the end-to-end wall by ~1.2 ms against the committed
  tactics (same-process interleaved A/B, three windows), so the recipe keeps
  the tactics recorded before the change. The tuner now measures the full
  candidate set for whatever tunes next.
- **Language GeGLU into the gate/up GEMM, ~3.5 ms.** hipBLASLt has no
  epilogue that multiplies across halves; this needs a custom epilogue.

## Next

1. **Full LIBERO-10** (500 episodes) as the regression gate for the above.

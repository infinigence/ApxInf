# WallOSS

Dual-stream VLA (Qwen2.5-VL 3B vision-language prefix plus a 1024-wide action
expert, 10-step flow) for the `wall-oss-0.5` family. Implementation:
`crates/apxinf-model/src/walloss/`, organized per
[Model Layer Architecture](model-layer-architecture.md); Python
policy: `python/apxinf/apxinf/policies/impls/walloss.py`.

## Support

| Item | Supported |
| --- | --- |
| Device | CUDA; measured on Jetson AGX Thor (sm_110) |
| Precision | BF16 (default), dynamic FP8 (`precision="fp8"`, per-token activation scales) |
| Views | any count; all views of one request resize to one shared size |
| Image size | any size the processor produces (patch grids divisible by the merge size); square and rectangular |
| Execution | one CUDA graph per prepared shape; eager with `ExecutionPolicy::Eager` or `APXINF_WALLOSS_NO_GRAPH` |

Measured latency, stage utilization and accuracy: [Measurements](#measurements-2026-10-16).

## Input contract

The built-in policy resizes each camera like wall-x: when the checkpoint's
training config (`config.yml`) sets `resolution[camera]` (the `data` section
wins over the top level, `-1` means none), the image is first resized so its
longer side equals that value, then Qwen2.5-VL `smart_resize` rounds it to
multiples of 28 within the processor's pixel bounds. The resized views are
sent with `image_grid_thw` through `ModelRunner.infer_rgb`; the runtime
normalizes and patchifies them on the GPU.

So the training config decides the model resolution, not the camera: the
LIBERO fine-tune pins every camera to 256 (grid 18x18), while the published
`wall-oss-0.5` config uses 448 for its face and right-wrist cameras. A copy
without `config.yml` smart-resizes the raw image. Check the grids in
`scripts/bench_walloss.py --model-dir ...` output before comparing shapes.

A custom `processor=` returns patches for the default profile only (two
views, 18x18 grids); other shapes need the built-in RGB route.

## Preparation and shape changes

The runner keeps one prepared plan. A request with a different view count,
image size, token count or input mode prepares a new plan (eager warm-up plus
capture, about 130-180 ms on Thor) and replaces the old one; switching back
pays that cost again. The first request of a new plan is also the one that
`autotune=True` tunes from. Explicit `prepare` serves the default profile.

## Limits

- Views with different resized sizes in one request are rejected; resize them
  to one size first.
- Multi-frame (video) grids are accepted only as patches, not as RGB.
- Tuned GEMM databases are specific to the board and CUDA/cuBLAS version; a
  checkpoint-local `tactics.json` from another CUDA version is ignored.

## Measurements (2026-10-16)

- Code: this change, compared with the pre-change WallOSS runtime.
- Board: Jetson AGX Thor (sm_110), CUDA 13.2, unlocked clocks, one process at
  a time. GEMM tactics: none (the checkpoint's `tactics.json` targets CUDA 13.0
  and is ignored), except where "tuned" is stated.
- Checkpoint: `wall-oss-0.5` without `config.yml` (raw images smart-resize), so
  256, 224 and 448 inputs run at grids 18x18, 16x16 and 32x32.
- Recipe: `python scripts/bench_walloss.py --model-dir <ckpt> --views V --height H --width W
  [--resolution R] [--precision fp8] [--profile]`. Fixed random images, state
  and noise; 20-30 timed calls after warm-up.

### Refactor parity

Before the BF16 padding commit, actions were byte-identical to the previous runtime for
BF16 and FP8, graph and eager, on three inputs (new images, new prompt), and
byte-identical at 3x224 and 3x448 to the build whose reference accuracy is quoted below.
Changing images or prompt changes the output; repeating an input reproduces
it. Interleaved graph p50 (three rounds): BF16 102.33-102.49 ms baseline vs
102.07-102.24 ms; FP8 65.20-65.38 vs 64.92-65.16 ms.

Reference accuracy for the new shapes was measured on that build (identical
numerics, see above; Thor with CUDA 13.0) against the reference implementation in the
normalized action domain: fixture anchor cosine 0.999939, rel-L2 0.011
(BF16); real-grid 3x224 and 3x448 within the input-path band of the 2-view
anchors (active-dimension cosine >= 0.997).

The shared RGB binding change leaves PI0.5 (`pi05_libero_base`, BF16)
byte-identical to the previous runtime through the policy with explicit noise, the
seeded device-noise path and `infer_rgb_seeded`. SmolVLA, PI0-FAST, GR00T and
Qwen-Drive were not run on a GPU for this change (CPU tests only).

The padding commit changes BF16 rounding only: action cosine >= 0.99998,
rel-L2 <= 0.006 versus the unpadded build at 2x252 and 3x448.

### Latency (graph, policy call p50)

| Shape (grid) | Vision rows | Prefix tokens | BF16 before pad | BF16 | BF16 tuned | FP8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2x256 (18x18) | 648 | 243 | 102.3 ms | 94.1 ms | 92.6 ms | 65.2 ms |
| 3x224 (16x16) | 768 | 280 | 104.5 ms | 94.8 ms | | 66.9 ms |
| 2x480x640, resolution 448 (24x32) | 1536 | 467 | 140.5 ms | 122.5 ms | | 89.2 ms |
| 3x448 (32x32) | 3072 | 852 | 214.9 ms | 183.3 ms | 179.8 ms | 129.3 ms |

"Tuned": `--autotune` on the first request of each shape tuned 28 GEMM keys
into a board-local database; reused without autotune, it gains 1.8% (2x256)
and 2.3% (3x448).

A shape change re-prepares and re-captures: the first call costs 230-300 ms
instead of 95-123 ms; switching back reproduces the earlier actions bitwise.

### Stage utilization (BF16 after padding)

Stage times are CUDA-event GPU times of one eager pass (`--profile`); denoise
is the graph total minus the eager vision and prefix times. FLOPs and bytes
are analytic (GEMMs only; attention adds under 3%). References: 137 TFLOP/s
for the best BF16 GEMM measured in the same runs (3072x6840x1280), 259 GB/s
DRAM bandwidth previously measured on this board.

| Shape | Vision ms / TFLOP/s | Prefix ms / TFLOP/s | Denoise ms / GB/s |
| --- | --- | --- | --- |
| 2x256 | 17.95 / 47 | 31.71 / 43 | 42.4 / 186 |
| 3x224 | 18.69 / 53 | 31.26 / 50 | 44.0 / 180 |
| 2x480x640 @448 | 34.42 / 58 | 37.79 / 69 | 43.9 / 180 |
| 3x448 | 75.76 / 53 | 59.29 / 80 | 46.3 / 171 |

Vision costs 1.30 GFLOP per patch row, the prefix 5.55 GFLOP per token
(weights 5.55 GB), and each denoise step reads 0.79 GB of action-expert
weights. Utilization does not degrade at 448: vision and prefix throughput
rise with the larger GEMMs (the prefix moves from bandwidth-bound to
compute-bound), and the denoise loop stays bandwidth-bound at 66-72% of DRAM
bandwidth, slightly lower at 448 because each step attends over a 3.5x
longer prefix.

### Remaining time at 3x448 (BF16, eager kernel summary)

Vision GEMMs take about 30 of its 76 ms. The rest is elementwise and
attention work, in order of size: the separate gate/up bias pass before
SwiGLU (~12 ms, fusable into SwiGLU or a GEMM epilogue), SwiGLU (~9 ms),
window attention issued as one FA2 launch per window with head dim 80 padded
to 96 (~10 ms), QKV bias plus RoPE (~8 ms), and the residual/RMSNorm epilogues.
These need new fused kernels, which belong in `apxinf-cuda-new`, and are not
part of this change.

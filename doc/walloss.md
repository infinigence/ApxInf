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
seeded device-noise path and `infer_rgb_seeded`. At that measurement snapshot, SmolVLA, PI0-FAST, GR00T and Qwen-Drive
had only CPU coverage. See the final-source verification below for later GPU runs.

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


## Preparation contract regression checks (PR #138 follow-up)

The public `VlaRuntime`, `InferenceSpec` and metadata contracts are unchanged.
The fixes below are confined to WallOSS:

- `RequireGraph` fails before allocation when `APXINF_WALLOSS_NO_GRAPH` is set.
  `PreferGraph` records that opt-out as its eager fallback reason; `Eager`
  continues without attempting capture. Explicit preparation also rejects
  token counts outside the model's supported range before allocating buffers.
- An implicit cache miss validates host geometry, token/image correspondence,
  RGB/patch lengths, latent and mask shapes before releasing the old plan.
  Invalid requests preserve the old plan. A valid replacement still releases
  it before allocating a new workspace, so two workspaces do not coexist.
  Device allocation/capture failures after that point can still empty the cache.
- Workspace capacity derives from checkpoint dimensions: all layers' persistent
  KV, one layer's MLP/attention scratch, vision and action scratch, then allocator
  headroom. Checked arithmetic rejects overflow. This replaces coefficients
  fitted only to short prompts; it does not guarantee that the maximum context
  fits available device RAM. Capacity is deliberately conservative and increases
  relative to the short-prompt estimate.

CPU tests check the live KV + SwiGLU lower bound at 243, 32768 and near-maximum
prefix lengths, changed checkpoint widths/depth, and arithmetic overflow.
The CUDA test suite includes execution-policy selection and two opt-in checkpoint
regressions (missing checkpoints fail rather than silently skip when selected):

```bash
APXINF_WALLOSS_TEST_MODEL=/path/to/wall-oss-0.5 \
  cargo test --release -p apxinf-model --features cuda --lib \
  walloss::model_runner::runner::tests -- --ignored --test-threads=1
```

These checks prepare a 32768-token prefix and verify that rejected short-token,
truncated-RGB, wrong-latent and wrong-mask requests preserve both the cached plan identity
and the complete output tensor. Run the checkpoint tests once normally and once
with `APXINF_WALLOSS_NO_GRAPH=1` to exercise graph and eager operation.


The benchmark also checks effective execution mode via the profiled public
entry point, even without `--profile`. It rejects a graph-labelled run that
actually fell back to eager. This extra validation runs outside timed samples;
`tests/test_bench_walloss.py` covers the CLI behavior without requiring a GPU.

Follow-up verification on Jetson AGX Thor (sm_110, CUDA 13.2):

- WallOSS CUDA library suite: 20 passed; both opt-in checkpoint tests passed
  separately with graphs enabled and disabled (2 + 2). The long-prefix check
  runs BF16 eager preparation with 32768 prefix tokens.
- Final CUDA Python extension: 102 passed, no skips, one inherited PI0-FAST
  fixture failure (`test_from_pretrained_rejects_options_that_cannot_mean_anything_here`
  omits the required `config.json`). This is not an all-green Python suite.
- Public `WallossPolicy` through `scripts/bench_walloss.py`: BF16/FP8 ×
  default/rectangular × graph/eager, 8 runs. Effective grids were `[1,18,18]`
  and `[1,24,32]` for both views. Each precision/shape pair's complete normalized
  action arrays matched exactly between graph and eager; all values were finite.
  The real CLI also rejected an environment-forced eager fallback when a graph
  benchmark was requested. Use default arguments or
  `--height 480 --width 640 --resolution 448`, select
  `--precision`, toggle `--eager`, and save full arrays with `--dump-actions`.
- Local model CPU tests: 148 passed, one ignored. Workspace and benchmark
  regression tests fail with the old implementations and pass with the fixes.
  CUDA-feature workspace type checks and family/module boundary guards pass.

These initial follow-up runs are correctness checks, not fresh performance
measurements. The timings earlier in this document retain their original
measurement snapshots. Later final-source reference and regression results
are recorded below.


### Final-source reference comparison (2026-10-19)

The tested runtime is `fec5ee10eb1fb4ac95d5ddb9ee54cb22f1215854`.
The candidate CUDA extension SHA-256 is
`8c1c669a8b298d474b123c2e7ae06dc14a82ac831293270c3dc622897e5501c3`.
The reference was rerun, rather than reusing the earlier output captures:
Mizar's vanilla PyTorch/SDPA path at `ca013e817fead355bfca2f989b71a0d72a641a24`,
with wall-x source `84968afbceca3b8aaa3a7f78aad9d0136438c0ec`.
It uses the existing private pure-Torch `mizar_kernel` stand-in and import
shims, with compilation and quantization disabled. Their hashes are recorded
in the evidence manifest; this is not validation of production Mizar kernels.
Reference execution used Thor/CUDA 13.0, PyTorch `2.14.1+cu130`; candidate
execution used Thor/CUDA 13.2.

Both machines used the same `wall-oss-0.5` weights, SHA-256
`37f8f327d3f9c27f056ed078db43c0e39b8caf3df0dc0163cdb8cf92c167968a`.
Inputs are deterministic RGB arrays, zero 26-dimensional state, prompt
`pick up the block`, and explicit `[10,26]` noise generated with NumPy seed
1234. Training-resolution scaling is disabled for these already aligned
image sizes. Raw images, state, noise, expanded token IDs, grids and broadcast
action masks match between the two paths. Each reference output was repeated
exactly. Normalized actions are captured before reference unnormalization;
all 260 values are compared, including the padded dimensions.

| Views / RGB size | BF16 cosine | BF16 relative L2 | FP8 cosine | FP8 relative L2 |
| --- | ---: | ---: | ---: | ---: |
| 2 × 252×252 | 0.999753 | 0.022762 | 0.999904 | 0.013853 |
| 3 × 224×224 | 0.999977 | 0.007066 | 0.999814 | 0.019926 |
| 3 × 448×448 | 0.999278 | 0.038220 | 0.999485 | 0.032104 |
| 2 × 336×448 | 0.999784 | 0.020797 | 0.999788 | 0.020598 |

Both candidate precisions are compared to the BF16 reference. All eight
comparisons also exceed the earlier first-seven-dimension cosine criterion
of 0.997 (observed minimum 0.999614). Maximum absolute error over the complete
arrays is 0.187413 for BF16 and 0.116814 for FP8; the table is numerical
agreement evidence, not a claim of per-element equality with the reference.

The public `WallossPolicy` was run for all four shapes, both precisions and
both execution modes: **16 runs**, with effective Graph/Eager mode checked
through `infer_rgb_profiled`. Complete normalized outputs are exactly equal
between Graph/Eager for each shape/precision and repeat identically. These
synthetic-input checks do not replace a task-success evaluation or extend the
earlier latency claims.

Private reproduction scripts, full arrays, source/extension/checkpoint hashes
and logs live under `devlocal/walloss-pr138/final-parity/`. They remain ignored;
no checkpoint, reference adapter or generated capture is part of the PR.

### Cross-family GPU regression on the same final runtime

The comparison baseline is PR138's base commit
`da8756eaf112b96dfdf870187d6c58d294a640f3`, rebuilt as a separate CUDA Python
extension. Both arms use identical family policy source (unchanged by this
PR), dependencies, checkpoint files and inputs on the same machine. All rows
below use BF16 and two distinct observations, with repeated inference in each
arm. Comparisons load the complete saved arrays and require exact equality;
matching hashes alone are not the acceptance check.

| Family / checkpoint | Exercised interface | Cross-build result |
| --- | --- | --- |
| PI0.5 / `pi05_libero_base` | `AutoPolicy.infer` with explicit noise, plus `infer_rgb_seeded` | 2/2 exact: `[50,7]` actions, `[50,32]` normalized and seeded outputs |
| SmolVLA / `lerobot/smolvla_libero` | `AutoPolicy.infer`, three cameras, explicit noise | 2/2 exact: `[50,7]` actions and normalized outputs |
| GR00T / prepared N1.7 LIBERO checkpoint | `AutoPolicy.infer` with NVIDIA preprocessing and explicit noise | 2/2 exact: `[16,7]` actions and `[40,132]` normalized outputs |
| Qwen-Drive / `Qwen-Drive-1.0-4B`, `planner-sft` | `AutoPolicy.infer`, two existing recorded driving scenes, direct planning, ten steps | 2/2 exact: complete `[50,3]` trajectories |
| PI0-FAST / local `pi0fast` checkpoint (LeRobot layout) | `ModelRunner.infer_action_tokens_rgb`, supplied token IDs | 2/2 exact: all 256 generated action tokens per case |

PI0.5 and PI0-FAST also reject a truncated RGB buffer; the next valid request
reproduces the prior complete output. GR00T captures successfully in both
arms. SmolVLA exhausts the same 96 MiB graph workspace in both arms and falls
back to eager; its result is an **eager regression pass**, not a successful
Graph test or a fix for that existing capacity limitation. PI0.5 reports
missing normalization statistics for some transforms in this local checkpoint;
its cross-build identity does not establish embodiment-level reference parity.

PI0.5/Qwen-Drive ran on Thor2 with CUDA 13.2, SmolVLA/PI0-FAST on Thor3 with
CUDA 13.2, and GR00T on Thor with CUDA 13.0. The same candidate and baseline
extension binaries were used across these hosts. Native extension identities,
checkpoint hashes, fixed-scene file hashes and full outputs are retained with
the private evidence. The 54 checked runtime, binding and Python source files
match the reviewed checkout byte-for-byte.

**Remaining policy-level blocker:** the PI0-FAST checkpoint is available, but
its declared `google/paligemma-3b-pt-224/tokenizer.json` is absent from the
inspected local assets. Hugging Face returns HTTP 403 with the available
credentials. The original tokenizer is needed to finish `AutoPolicy.infer`
coverage, including prompt tokenization and FAST action decoding. The native
RGB/token regression above does not substitute for that missing policy run.
No claim is made for every precision, every checkpoint or every deployment
profile of these families.

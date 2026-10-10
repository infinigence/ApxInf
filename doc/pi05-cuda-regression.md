# PI0.5 CUDA Regression Test for Every Pull Request

> Status: This document defines the canonical benchmark workload, tuning rules, accuracy protocol, and result format.
> As part of the manual CI/CD process, rerun the applicable tests after every code change and attach the results to the pull request.

## 1. Goals

This benchmark answers two questions:

1. What is the actual PI0.5 inference latency of ApxInf on Thor SM110 and Orin SM87?
2. Which tactic should be selected for each operator shape at real LIBERO language lengths and reserved lengths?

The primary result uses an official LIBERO instruction with 10 tokens. The extended real-world result uses an official longest LIBERO instruction with 21 tokens.

## 2. Fixed workload

| Parameter | Fixed value |
|---|---:|
| Batch size | 1 |
| Camera views | 2 / 3 views |
| Images | 224 x 224 RGB, NHWC `uint8` |
| Action horizon `H` | 10 for performance; 50 for accuracy |
| Action dimension | 32 |
| Flow-matching steps | 10 |
| Token execution mode | Exact length; do not pad to 200 |
| Real benchmark token counts `T` | 10 / 21 |
| Autotune-only token shapes | 50 / 200 |
| Warm-up | 10 iterations |
| Measured samples | 30 iterations |
| Timing statistics | P50 / P95 / min / max / mean / standard deviation |

`H` denotes the action horizon. Use `H=10` for performance tests and `H=50` for accuracy tests. `T` denotes the language token count.

## 3. Token dataset

| `T` | Source | Role |
|---:|---|---|
| 10 | Official LIBERO; 10 instructions have a PaliGemma token length of exactly 10 | **Primary LIBERO** |
| 21 | Official LIBERO; 2 instructions share the maximum length | LIBERO worst-case language |

The primary 10-token result may use this official LIBERO instruction:

```text
put the bowl on top of the cabinet
```

The extended 21-token result may use:

```text
pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
```

Before a formal T=10 or T=21 run, pin the text, token IDs, tokenizer hash, and simulation fixture hash together.

### 3.1 Fixed baseline fixtures

The baseline uses the real LIBERO first-replan fixtures already stored in the repository. The prompts and token IDs below are the actual baseline inputs. They replace the optional example prompts above and must not be interchanged when reproducing the baseline.

| `T` | Fixture | Prompt | PaliGemma token IDs |
|---:|---|---|---|
| 10 | `task_08_first_replan.npz` | `put both moka pots on the stove` | `2,1065,2145,705,1161,37801,611,573,37932,108` |
| 21 | `task_04_first_replan.npz` | `put the white mug on the left plate and put the yellow and white mug on the right plate` | `2,1065,573,2674,24464,611,573,2731,8811,578,2507,573,8123,578,2674,24464,611,573,1833,8811,108` |

Pinned artifact SHA256 values:

| Artifact | SHA256 |
|---|---|
| PaliGemma `tokenizer.model` | `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` |
| PI0.5 checkpoint `model.safetensors` | `21b8711787c4a75861b02cff6aa81675a3a943d32b435a68262ac4461e476ba4` |
| Raw T=10 NPZ fixture | `97f9d8b112605a67277cca65e4cadc06f7fd4ccd5e21f339a215670ea9e56473` |
| Raw T=21 NPZ fixture | `2663c33a3b801a7bf67bdefdea1526fdd9acad8564a0ede5ec98ee10f03381d6` |

Deterministically reconstruct 224 x 224 NHWC `uint8` images from the normalized patches in these fixtures, then pass them through ApxInf CUDA preprocessing. For the third view, reuse the wrist image. Label every three-view result as **duplicated wrist fixture**; it is not a real third LIBERO camera.

## 4. Meaning of views

| Views | Meaning |
|---:|---|
| 2 | Real LIBERO workload: base camera + wrist camera |
| 3 | Three-camera production-shape workload; not an official LIBERO camera configuration |

Use three views with LIBERO weights only for performance or numerical parity tests. Do not treat those runs as LIBERO task-suite accuracy results.

### OpenPI parity suite

`scripts/compare_pi05_openpi.py` checks the normalized `[H, 32]` PI0.5 output
against official OpenPI PyTorch. Both engines read the same `model.safetensors`,
saved images, token IDs, flow noise, horizon, and flow-step count. The seven
bounded cases cover a second scene or noise draw, long language, dark and
bright images, zero and negative noise, float CHW images, and camera ordering.
A suite without `--source-npz` is marked synthetic.

```bash
python scripts/compare_pi05_openpi.py prepare \
  --suite-dir devlocal/pi05-openpi-parity/base-3view \
  --image-keys base,left,right --horizon 10
python scripts/compare_pi05_openpi.py openpi \
  --suite-dir devlocal/pi05-openpi-parity/base-3view \
  --checkpoint-dir /path/to/pi05_base
python scripts/compare_pi05_openpi.py apxinf \
  --suite-dir devlocal/pi05-openpi-parity/base-3view \
  --checkpoint-dir /path/to/pi05_base --precision bf16
python scripts/compare_pi05_openpi.py compare \
  --suite-dir devlocal/pi05-openpi-parity/base-3view
```

For real images, pass one or two `--source-npz` files to `prepare`; each needs
arrays named by `--image-keys`, already resized to 224 × 224 RGB. Change only
the image keys, checkpoint path, horizon, and precision for another configuration.
The report checks input and weight hashes and gives cosine, relative L2, and
maximum absolute error per case. Its initial gate is cosine >= 0.997 and
relative L2 <= 0.10. Keep failed cases visible; a pass on synthetic inputs does
not establish robot-task accuracy.

[`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base/blob/main/config.json)
is a real three-camera checkpoint with PyTorch safetensors. The official OpenPI
[`pi05_base`](https://github.com/Physical-Intelligence/openpi/blob/main/README.md#base-models)
is JAX-format and needs conversion for this comparison. OpenPI mode needs the
official source on `PYTHONPATH`, a compatible CUDA PyTorch build,
`transformers==4.53.2` with OpenPI's `transformers_replace` files,
`safetensors`, and the OpenPI import dependencies (including JAX/Flax,
`augmax`, and `dm-tree`). ApxInf mode needs the CUDA-enabled `apxinf_py`
binding. Run the two inference stages in separate environments. FP8 requires
a calibration profile generated for the **same weights and view count**; pass
its path with `--calibration`. ApxInf's random FP8 benchmark uses
`uniform:1.0` to measure latency, but its internal synthetic weights cannot
be compared to OpenPI.

Thor3 validation on 2026-09-28 used ApxInf based on `upstream/main`
`c36cbbcdd2600240780c922fe8cd49fcc78a00a9` and `lerobot/pi05_base`
revision `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba` (weight SHA256
`0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`),
OpenPI revision `215abfb217dbac7d5f1273282331b9b1866c0479`, `H=10`, ten
flow steps, and seven synthetic cases. The fixed gate was cosine >= 0.997
and relative L2 <= 0.10.

| Active views | ApxInf precision | Passed | Failed case (cosine / relative L2) |
|---:|---|---:|---|
| 1 | BF16 | 6/7 | Dark image + zero noise: 0.97222 / 0.30955 |
| 2 | BF16 | 6/7 | View-order contrast: 0.99113 / 0.18798 |
| 3 | BF16 | 7/7 | None; minimum cosine 0.99927 |
| 3 | FP8 | 4/7 | Dark + zero noise: 0.99127 / 0.18822; bright + negative noise: 0.97100 / 0.42063; view-order contrast: 0.98706 / 0.22440 |

The three-view FP8 profile used seven synthetic observations from the same
suite, so its result diagnoses this calibration and implementation path; it
does not establish production FP8 accuracy. A separate genuine two-view
LIBERO checkpoint passed 7/7 BF16 cases on Thor2. The one- and two-view
`pi05_base` runs reduce the active camera count of a three-camera checkpoint
and should be read as camera-mask corner tests. Full reports and inputs are
kept under ignored `devlocal/pi05-openpi-parity/`.

## 5. Execution paths

| Device | Precision path | Purpose |
|---|---|---|
| Thor SM110 | BF16 | Thor high-precision baseline |
| Thor SM110 | FP8 native | Thor native FP8 quantized path |
| Orin SM87 | BF16 | Orin high-precision baseline |
| Orin SM87 | INT8 (W8A8) | Orin native INT8 quantized path |

The real benchmark matrix contains:

```text
4 device/precision paths x 2 view counts x 2 real token lengths = 16 cells
```

There are also 16 T=50/200 view/device/precision autotune-only profiles. They do not count as end-to-end benchmark results.

NVFP4 is outside the scope of the current ApxInf benchmark. ApxInf currently has no PI0.5 NVFP4 executor, calibration, tactic, or validated result.

## 6. Current best performance results

Use the current best results as the performance baseline. A faster validated result is a candidate for an explicit baseline update. Any result that does not meet the current matching baseline requires human review.

All baseline cells below used 10 warm-up iterations and 30 measured samples. Each latency cell is **P50 / P95** in milliseconds.

### 6.1 Thor SM110 baseline

| Path | Views | `T` | Graph replay P50 / P95 | Input update + graph P50 / P95 |
|---|---:|---:|---:|---:|
| Thor SM110 BF16 | 2 | 10 | **72.454 / 72.673** | 73.392 / 73.773 |
| Thor SM110 BF16 | 2 | 21 | **79.654 / 80.036** | 79.673 / 79.969 |
| Thor SM110 BF16 | 3 | 10 | **89.757 / 90.038** | 89.769 / 90.043 |
| Thor SM110 BF16 | 3 | 21 | **93.649 / 93.908** | 93.206 / 93.502 |
| Thor SM110 FP8 native | 2 | 10 | **41.159 / 41.312** | 41.193 / 41.288 |
| Thor SM110 FP8 native | 2 | 21 | **42.079 / 42.204** | 42.015 / 42.105 |
| Thor SM110 FP8 native | 3 | 10 | **53.636 / 53.903** | 53.588 / 53.763 |
| Thor SM110 FP8 native | 3 | 21 | **55.506 / 55.616** | 55.276 / 55.403 |

### 6.2 Orin SM87 baseline

Orin uses the same fixed LIBERO fixtures, token IDs, NHWC `uint8` images, and BF16 noise as Thor.

| Path | Views | `T` | Graph replay P50 / P95 | Input update + graph P50 / P95 |
|---|---:|---:|---:|---:|
| Orin SM87 BF16 | 2 | 10 | **165.665 / 165.845** | 165.747 / 165.870 |
| Orin SM87 BF16 | 2 | 21 | **166.606 / 166.776** | 166.728 / 166.882 |
| Orin SM87 BF16 | 3 | 10 | **205.269 / 205.739** | 205.565 / 206.132 |
| Orin SM87 BF16 | 3 | 21 | **204.319 / 205.121** | 204.425 / 205.273 |
| Orin SM87 INT8 W8A8 | 2 | 10 | **124.250 / 124.293** | 124.306 / 124.352 |
| Orin SM87 INT8 W8A8 | 2 | 21 | **124.808 / 124.901** | 124.888 / 124.933 |
| Orin SM87 INT8 W8A8 | 3 | 10 | **165.693 / 165.753** | 165.755 / 165.792 |
| Orin SM87 INT8 W8A8 | 3 | 21 | **166.285 / 166.358** | 166.335 / 166.405 |

#### Orin INT8 accuracy limitation

The Orin INT8 CUDA Graph and eager outputs match element by element. Replacing the SM87 CUTLASS W8A8 GEMM with cuBLAS also produces elementwise-identical final outputs across all eight mixed-precision combinations (`max_abs=0`). The accuracy issue comes from the current naive PTQ W8A8 quantization algorithm: weights use per-output-channel absmax scales, activations use dynamic per-token-row absmax scales, and the algorithm has no calibration, SmoothQuant, outlier handling, or QAT. Improving the quantization algorithm remains a TODO.

## 7. Accuracy standard and current reference results

The formal pull-request accuracy standard is:

| Parameter | Required value |
|---|---:|
| Action horizon `H` | 50 |
| Episodes | 500 |
| Replan interval | 5 |
| Views | 2 |
| Token length | T=10 |

A material task-success-rate regression requires human review. New formal results must report the completed count out of 500 and the corresponding percentage.

The tables below are historical 100-episode reference runs. They are useful comparison points, but they do **not** satisfy the current 500-episode pull-request accuracy standard and must not be reported as new formal PR accuracy results.

### 7.1 Thor T=10, 2 views: historical 100-episode reference

| Platform | Precision | Input | LIBERO-10 task success | Official reference |
|---|---|---|---:|---:|
| Thor | BF16 | 2 views / T=10 | **93/100 (93%)** | 92.4% |
| Thor | FP8 | 2 views / T=10 | **94/100 (94%)** | 92.4% |

### 7.2 Orin T=10, 2 views: historical 100-episode reference

The naive W8A8 PTQ scaling strategy used by language QKV causes accuracy loss on the INT8 path.

| Platform | Precision | Input | LIBERO-10 task success | Official reference |
|---|---|---|---:|---:|
| Orin | BF16 | 2 views / T=10 | **93/100 (93%)** | 92.4% |
| Orin | INT8 W8A8 | 2 views / T=10 | **91/100 (91%)** | 92.4% |

## 8. Timing boundaries

Every result cell must report both timing boundaries:

```text
Graph replay
  = steady-state CUDA Graph launch + synchronize

Input update + graph
  = update already-resized uint8 images, tokens, and noise
    + CUDA preprocessing
    + graph replay
    + synchronize
```

Image decoding, camera rotation, and CPU resize are outside both boundaries by default. If Python/client end-to-end latency is also measured, report it in a separate table.

Use Nsight Systems and Nsight Compute only to locate bottlenecks. Profilers change timing behavior, so profiler-instrumented measurements must not replace uninstrumented formal benchmark results.

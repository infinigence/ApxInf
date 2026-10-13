# PI0.5 CUDA regression measured baselines

- Report date: 2026-10-13 (carried on main fde5feb3).
- Boards: Jetson AGX Thor (SM110), Jetson AGX Orin (SM87).
- Procedure and workload definitions: [models/pi05](../models/pi05.md).
- Protocol: 10 warm-up iterations, 30 measured samples; each latency cell is
  P50 / P95 in milliseconds.

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


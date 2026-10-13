# Qwen-Drive BF16 benchmark and validation results

- Report date: 2026-09-28 (accepted in review 2026-09-28, PR #94)
- Board: Jetson AGX Thor (SM110, Thor2), CUDA 13.2, CPU/GPU/EMC locked
  (min=max 2.601 / 1.575 / 4.266 GHz), MAXN, fan PWM 255.
- Recipe: bench + eval against the fixed NAVSIM subset; see
  [models/qwen-drive](../models/qwen-drive.md) for the current procedure.
- Baseline commit: 7f37642d (perf(qwen-drive): integrate native AOT kernels
  and padded BF16 direct planning, #94).

## Results

Jetson AGX Thor SM110 (20 SMs), CUDA 13.2, BF16 `planner-sft`, direct
planning, batch one and ten flow steps. Request latency includes policy
preprocessing, native execution and the materialized `[50, 3]` host trajectory;
images are already decoded and model loading is excluded. This is an end-to-end request latency
measurement, whereas the PI0.5 README table measures CUDA Graph replay.

| Measurement | Result |
| --- | ---: |
| Request P50 / P95 | 482.60 / 488.78 ms |
| Request throughput | 2.07 Hz |
| Fixed NAVSIM scenes | 242 |
| PDM score, 0–100 | 85.6786 |
| Trajectories identical to accepted padded implementation | 242 / 242 |

The fixed subset's official-code SFT direct reference scores **84.6974**.
The accepted unpadded optimized version scores **85.6723**. Padding changes 125
of its trajectories and leaves 117 exact, without changing any discrete PDM
component. The current implementation matches the accepted padded version
exactly. The final review build repeats all 242 predictions with zero position
or heading difference; its 85.6786 score is carried forward from the already
scored identical trajectories, rather than a newly repeated scorer run.

These are fixed-subset results, not the published full-navtest scores. The
[official release](https://github.com/QwenLM/Qwen-Drive-1.0#planning) reports SFT/RL
88.2/90.7, or 89.3/91.4 with best-of-six selection. Checkpoint, scene selection,
image profile and sampling protocol must match before comparing scores. The
local gain over official code includes three binary metric flips and does not
establish a systematic quality improvement.

### Recorded timing conditions

Measured on Thor2 on 2026-09-28 with CPU min=max=2.601 GHz, GPU min=max=1.575 GHz,
EMC min=max=4.266 GHz, MAXN and fan PWM 255 with automatic fan control disabled.
The GPU lock covered the entire run; configuration was restored afterwards.
The table uses the pooled 60 measured requests from both candidate arms.

| ABBA arm | Request P50 | Request P95 |
| --- | ---: | ---: |
| Pre-review maintained version, first | 481.35 ms | 485.83 ms |
| Final candidate, first | 482.67 ms | 486.40 ms |
| Final candidate, second | 481.99 ms | 488.78 ms |
| Pre-review maintained version, last | 485.08 ms | 490.65 ms |

Reference pooled P50/P95 is 483.04/489.38 ms. The final implementation preserves
performance within this run's variation. Its two arm medians average 482.33 ms;
the historical 482.98 ms result used that average-of-arm-medians convention.
The table above reports the pooled P50 instead, retaining both conventions
explicitly rather than conflating them.

An earlier run measured 498.78 ms. Repeating the same binary on the same Thor3
later gave 484.90 ms P50, so the earlier slowdown was not evidence of a lost
code optimization or a fixed difference between the two boards. That run lacks
continuous resource telemetry; its exact transient cause remains unproven.


# Qwen-Drive across three devices

One model implementation, three boards. This is the map: what each board gets,
where its constants live, and what has and has not been re-measured since the
three lines were brought together.

## The fixed workload

Four public VQA scenes from WOD_E2E, batch 1, BF16, exactly 64 generated
tokens, one warmup and three measured requests per scene, warm median. Each
scene is three camera views (front, front-left, front-right) of four frames at
0.5 s spacing — twelve images — plus a 16-point history, ego velocity and
acceleration, and the driving and navigation commands. The four differ in ego
state: stationary, just-braked, 2.67 m/s, 6.99 m/s.

`control/verify_perf_orin.py` runs it; `control/verify_gpu_orin.py` is the
four-mode correctness gate over the same inputs.

## Where each board stands

| | sm | per scene | against its own baseline |
|---|---|---:|---|
| RTX 4090 | sm_89 | 3.580 s | 1.02x over the pre-pilot snapshot |
| Jetson AGX Orin | sm_87 | 6.014 s | 1.09x, from 6.609 s |
| Jetson AGX Thor | sm_110 | **3.505 s** | **1.47x**, from 5.196 s |

The Thor column is the one this consolidation re-measured; see
[RESULTS.md](RESULTS.md) for its decomposition and
[THOR-ROOFLINE.md](THOR-ROOFLINE.md) for the device limits it is scored
against. [jetson-roofline.md](jetson-roofline.md) is the same measurement for
sm_87 and sm_101, taken with `scripts/bench_device_roofline.cu`, and its last
section reconciles the three boards -- including the two measurement traps that
were hit twice independently: a Jetson's DVFS ramp, which makes a cold sweep
read as a dependency cliff that is not there, and `cudaDevAttrMemoryClockRate`,
which does not report the LPDDR rate on Tegra. The 4090 numbers come from
[qwen-drive-performance.md](qwen-drive-performance.md) and the integration
record is [kersor-qwen-drive-4090.md](kersor-qwen-drive-4090.md); the Orin
numbers are #72's.

## Where the per-device decisions live

Three layers, and a change belongs to exactly one of them.

**Compile time — which kernels exist.** `build_support/cuda_arch.rs` selects
the architecture, and the family predicates in `build.rs`
(`is_fa2_sm80_family`, `is_fa2_bf16_arch`, `is_cutlass_sm100_family`,
`is_cutlass_sm89_family`) turn that into `cargo:rustc-cfg=apxinf_*`. Adding a
board that shares a family with one already supported is usually one arm of one
of those predicates — #60 added GB10 that way, and the head-64 and head-256
FlashAttention-2 dispatch adapters reached Thor the same way here.

**Run time — which compiled path runs, with what constants.**
`CudaDeviceCaps` carries the hardware facts and classifies the family;
`kernels/gdn_policy.rs` carries the policy that follows from them. The GDN
kernels take their tile widths, block width, recurrence split and tensor-core
form from that one table rather than deciding for themselves, and each field
has an environment override so a new board can be re-swept without a rebuild.

**Data — which tactic each shape uses.**
`configs/tuning/<vendor>/<family>-sm<N>/tactics.json`, resolved by
`TuningPaths::for_cuda`. Four stores ship: `rtx4090-sm89`, `orin-sm87`,
`thor-sm101`, `thor-sm110`. Recording one is a run of the autotuner, not a code
change — but read the note in RESULTS.md first: on Thor all 116 shipped records
were rejected at load because they were recorded under CUDA 13.0 and the board
runs 13.2, and the autotuner's rewrite replaces rather than merges across
toolkit versions.

## What the constants actually are

`GdnLaunchPolicy::defaults_for` is the whole table. The point of collecting it
is that none of these transfers:

| | sm80 family | sm100 family |
|---|---:|---:|
| chunk-state tile | 8 | 4 |
| chunk-gemm tile | 32 | 4 |
| recurrence split | 1 | 4 |
| chunk-state scan | scalar fp32 | two BF16 terms on tensor cores |

The chunk-state curve does not merely shift between the two: a tile of 16 is
8% off the optimum on Orin and 26% off on Thor.

## What has been verified since the consolidation

On Thor: builds, the four-mode gate in its declared state, `perf-thor.sh` at
1.4708x geomean, `precision_probe.py` unchanged on all three runnable modes,
and all four fp64 operator oracles returning the same numbers as before the
merge.

**Not re-verified on Orin or the 4090.** Both keep the sm80-family row of the
policy table, which holds the constants those two lines measured, and no
tensor-core path is enabled for them — so the intent is no behaviour change on
either. That intent is untested. Run the gate and `verify_perf` on both before
merging.

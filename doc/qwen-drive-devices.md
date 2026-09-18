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

Both Orin and the 4090 have since been run; the section at the end of this
document has their numbers. The 4090 came out 1.78x faster and Orin 5.4%
slower, and the Orin result is not yet explained.

---

# What the three boards measured, on this branch

Run after the consolidation, same four scenes, same script, same two
environment flags (`APXINF_CUDA_ALLOC_CACHE=1`,
`APXINF_CUDA_SKIP_OUTPUT_ZERO=1`) on every board, so the numbers are
comparable to each other and not to any earlier record taken without them.

## Correctness: the gate reproduces, and two boards agree bit for bit

| | VQA first difference | direct |
|---|---|---|
| Orin sm_87 | scene 0, index 121, token 357 | 0.08056783676147461 |
| RTX 4090 sm_89 | scenes 0 and 1 pass; scene 2 at index 487 | 0.08056783676147461 |
| Thor sm_110 | scene 0, index 106, token 5459 | 0.08056640625 |

Orin's row is what #72 and #74 recorded, digit for digit, which is the
evidence that the consolidation did not disturb that line.

The 4090 and Orin produce the *same seventeen digits* for the direct
trajectory error. They are different silicon on different toolkits, and they
take the same code path -- both are sm80-family, so both get the scalar GDN
kernels and the FA2 head-256 dispatch. Thor differs in the last digits, and it
is the one board whose policy row puts the chunk-state scan on tensor cores.
That is the shape the divergence should have if it is BF16 rounding amplified
through the GDN recurrence, as `jetson-roofline.md` argues, and not a defect in
one kernel.

## Performance, measured on each board against its own previous branch

Alternating rounds on one machine, so a slow patch cannot land on one side only.

**RTX 4090, against #73** — 2.5831 / 2.6048 / 2.5942 / 2.5676 s per scene
becomes 1.4513 / 1.4536 / 1.4531 / 1.4536 and 1.4467 / 1.4536 / 1.4537 /
1.4542. **1.78x.**

That is not "no behaviour change", which is what was expected before measuring.
#73 is the 4090's own older line: it has neither #72's Orin work nor the twelve
commits `main` has taken since. Consolidating hands the 4090 both at once. The
policy table's sm80 row is unchanged for it; the speed comes from the rest of
the branch.

Accuracy over the same probe, #73 → this branch: direct trajectory
mean/rms/p99 0.012886/0.043744/0.322266 → 0.011401/0.037494/0.161133, the p99
halving; reasoning rms 0.030624 → 0.029038 with its mean 6.7% worse; VQA token
agreement 0.5813 → 0.5796, unchanged within its own scatter.

**Orin, against #72 at `41066eb`** — 6.4726 / 6.4634 / 6.4721 / 6.4616 s
becomes 6.8259 / 6.8002 / 6.8153 / 6.8070. **5.4% slower**, reproducible over
two alternating rounds with 0.1% spread.

This one is unresolved and should be resolved before merging.

What it is **not**: the policy table. Sweeping its Orin row back to the values
`41066eb` shipped changes nothing — `chunk_gemm_tile` 16 gives 6.82,
`chunk_state_threads` 512 gives 6.83, both together 6.83, `chunk_state_tile` 4
the same, against 6.81 for the defaults. Nor is it accuracy-related: the gate
reproduces exactly.

What it could be: the span from `41066eb` to this branch carries three sets of
commits, and only one of them is this work. Upstream added eleven to #72's own
branch, `main` has moved twelve, and this branch adds its own. The third data
point that separates them is #72's current head `80b0ecc` built on the same
board; that build was in progress when both Jetsons went off the network, and
it is the next thing to run.

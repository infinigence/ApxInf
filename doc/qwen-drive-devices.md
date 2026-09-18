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
`TuningPaths::resolve_for_cuda`. Four stores ship: `rtx4090-sm89`, `orin-sm87`,
`thor-sm101`, `thor-sm110`. Recording one is a run of the autotuner, not a code
change.

A tactic is only valid for the libraries that measured it, and all three boards
were running a different toolkit from the one their store was recorded on, so
every record was rejected at load and the GEMMs quietly fell back to the
untuned heuristic. The resolver therefore prefers
`<family>-sm<N>/cuda<major>.<minor>-cublas<major>.<minor>/tactics.json` and uses
the unqualified file only when its header matches the running libraries — the
subdirectory is named by the same truncation the loader compares, so a store
found under it is one this toolkit can use in full. Nothing relocates: a board
running the toolkit its store was recorded on resolves to exactly the file it
resolved to before. A second toolkit now writes beside the first rather than
over it. See `doc/jetson-roofline.md` for what regenerating is worth — 12.6% on
Orin, 0.05% on Thor after the accuracy filter, nothing at all on the 4090.

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

# Where the 4090's scene actually goes

`ncu` is refused on this box too (`ERR_NVGPUCTRPERM`), but `nsys` is not, and
between a kernel trace, an NVTX range around `policy.infer`, and the same scene
run at one token and at 64, the whole 1.45 s is accounted for.

## The device's own roofline

| | measured | spec |
|---|---:|---:|
| DRAM read, 2 GB | **954.5 GB/s** | 1008 GB/s |
| copy (r+w) | 886.0 GB/s | |
| FP32 FMA, 32 chains | 80.2 TFLOP/s | |
| BF16 GEMM, 8192³ | 168.8 TFLOP/s | |

FP32 on CUDA cores is **80.2 TFLOP/s against Thor's 5.43 and Orin's 3.42**,
which is why the fp32 GDN kernels that dominate a Jetson's prefill are a much
smaller share here, and why the tensor-core forms that pay on Thor are not
obviously worth their extra passes on this board.

## The split, and the same 76% on two very different boards

| | fixed cost | decode |
|---|---:|---:|
| per scene, four scenes | 0.7142 s | 11.572 ms/token |

The byte budget is 8.41 GB per decode token, so the floor at 954.5 GB/s is
8.81 ms and the measured decode is **76.1% of it**. Thor measured 76.3% against
its own floor. Two boards a factor of 3.7 apart in bandwidth and 15 apart in
fp32 throughput sit at the same fraction of their own roofs, which says the
remaining decode gap is a property of the decode path and not of either device.

## A fifth of the scene is not GPU work at all

An NVTX range around `policy.infer` against the kernel trace:

| | |
|---|---:|
| `infer` call | 727.3 ms |
| host before the first kernel launches | **283.0 ms** |
| GPU span | 435.6 ms |
| host after the last kernel | 8.7 ms |

At 64 tokens the GPU span grows to 1167.9 ms and the host prologue does not
move, so it is 283 ms of a 1471 ms scene — **19%**, larger than any kernel, and
entirely invisible to a GPU profiler. Inside the GPU span the device is busy
94.5% of the time in prefill and 96.4% in decode, over 494 launches per decode
token; the gaps are 0.42 ms/token, so there is no launch-overhead story here.

Timing the policy's `_patchify` stage by stage over the twelve frames of a
scene found it:

| stage | ms, twelve frames |
|---|---:|
| PIL bicubic resize to `target_size` | 125.6 |
| PIL bicubic resize onto the patch grid | 52.8 |
| block-ordered permutation and copy | 19.9 |
| `Image.fromarray` | 10.3 |
| normalisation | 9.5 |
| `asarray` to float32 | 4.8 |
| **total** | **222.9** |

Four fifths of it is two bicubic resizes. Collapsing them into one would change
pixel values and the reference performs both, so that is not available. What is
available is that the twelve frames are independent, and both PIL's resampling
and numpy's copies release the GIL. Eight worker threads take 231 ms to 58.4 ms
— 3.96x — and the concatenated patch tensors hash identically to the serial
ones.

Measured end to end, three alternating rounds:

| | scene 0 | scene 1 | scene 2 | scene 3 |
|---|---:|---:|---:|---:|
| serial | 1.4553 | 1.4534 | 1.4556 | 1.4554 |
| eight threads | 1.2903 | 1.2934 | 1.2826 | 1.2837 |

**Geomean 1.1300x**, 167 ms a scene, for a change that hands the model the same
bytes. `APXINF_QWEN_PREPROC_THREADS` sets the worker count; 1 restores the loop.

## The chunk-state scan runs on a quarter of the device

`gdn_chunk_state_kernel<8>` is 131.2 ms of the 410 ms prefill, 24 launches of
5.47 ms, and the trace gives its shape: **grid 32, block 1024, 80 KB of shared
memory**. Thirty-two blocks is one per value head, and this board has 128
multiprocessors, so three quarters of it are idle for the duration. On a 16-SM
Orin the same launch is two full waves, which is why nothing about it looked
wrong until now.

The scan is sequential over chunks, so chunks cannot be spread. The value
dimension can: the decay scales rows of the state, every accumulation runs over
the key dimension, and an output column reads only its own column of the state,
so nothing in the kernel crosses it. Splitting a head across four blocks of 32
columns is the same arithmetic in the same order on a quarter of the columns
each, and `gdn_chunk_state_v_split_is_bit_exact` checks that as bits rather
than against a tolerance: 98304 output words and the whole carried state, zero
differing at both two-way and four-way. The width comes from the device's
multiprocessor count, so Orin and Thor keep one block per head.

Three alternating rounds, on top of the parallel preprocessing:

| blocks per head | scene 0 | scene 1 | scene 2 | scene 3 | geomean |
|---|---:|---:|---:|---:|---:|
| 1 | 1.2921 | 1.2873 | 1.2925 | 1.2861 | — |
| 2 | 1.2564 | 1.2565 | 1.2477 | 1.2553 | 1.0283x |
| 4 | 1.2453 | 1.2475 | 1.2371 | 1.2389 | **1.0381x** |

47 ms a scene. Four is the most this shape allows — the value dimension is 128
and a slice below 32 columns stops being a whole warp of them — and the curve
is already flattening, so the kernel has stopped being short of blocks and
started being short of something else. Together with the preprocessing that is
**1.173x** on the 4090 over the consolidated branch, both halves bit-exact.

One thing this uncovered: carrying the column offset as a runtime value cost
enough registers to push the 1024-thread width past the per-block budget, and
the unsplit launch came back `CUDA 701` on a board that had been running it
for weeks. The split is therefore a template parameter, and one block per head
compiles to what it compiled to before.

## What the parallel preprocessing is worth on the other board

Orin measures the same preprocessing at 241.7 ms serial and 49.4 ms on twelve
workers — 4.89x, against the 4090's 3.96x on eight — and the concatenated
patch tensors hash to the same digest on both boards. End to end, three
alternating rounds on a machine that was not idle (an nvcc from an earlier
build held one of the twelve cores throughout, which is what alternating is
for):

| | scene 0 | scene 1 | scene 2 | scene 3 |
|---|---:|---:|---:|---:|
| serial | 6.8672 | 6.8686 | 6.8636 | 6.8572 |
| eight threads | 6.6622 | 6.6688 | 6.6710 | 6.6758 |

**1.0292x**, 195 ms a scene. The win is a fixed number of milliseconds, so it is
worth 13% on the fastest board and 3% on the slowest.

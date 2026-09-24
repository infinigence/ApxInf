# Qwen-Drive on more than one device

One model implementation, three boards -- Jetson AGX Orin (sm_87), Jetson AGX
Thor (sm_110) and the RTX 4090 (sm_89). This is what to know before changing
anything that behaves differently on one of them: where each kind of
per-device decision belongs, and what the constants currently are.

Planning loads through the shared `AutoModel` registry as a CUDA VLA runtime and
the planner checkpoint is required; see
[Qwen-Drive planning runtime](qwen-drive-planning.md) for the loading, request
and validation contract. Projection layout is
selected once during model construction and retained with the device weights,
so weight packing and execution use the same physical representation.

## The fixed workload

Four public VQA scenes from WOD_E2E, batch 1, BF16, exactly 64 generated
tokens, one warmup and three measured requests per scene, warm median. Each
scene is three camera views (front, front-left, front-right) of four frames at
0.5 s spacing — twelve images — plus a 16-point history, ego velocity and
acceleration, and the driving and navigation commands. The four differ in ego
state: stationary, just-braked, 2.67 m/s, 6.99 m/s.

This workload was recorded against the VQA generation path, which the
planning-only runtime no longer exposes. The constants below are kernel-level
and still apply — reasoning planning drives the same GDN and attention kernels —
but reproducing these exact numbers now needs an equivalent token budget through
`mode="reasoning_planning"` rather than the VQA entry point.

The probe that runs it, and the four-mode correctness gate over the same inputs,
are private workflow artifacts under `devlocal/qwen-drive/` per `AGENTS.md`;
they are not part of the maintained tree.

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
over it. Regenerating a store is worth 12.6% on Orin, 0.05% on Thor after the
accuracy filter, and nothing at all on the 4090.

## What the constants actually are

`GdnLaunchPolicy::defaults_for` is the whole table, and the point of collecting
it in one place is that not one of these values transfers between the two
families:

| | sm80 family | sm100 family |
|---|---:|---:|
| chunk-state tile | 8 | 4 |
| chunk-state block | 512 | 1024 |
| chunk-gemm tile | 8 | 4 |
| recurrence split | 1 | 4 |
| chunk-state scan | scalar fp32 | two BF16 terms on tensor cores |
| blocks per head in the scan | from the multiprocessor count | from the multiprocessor count |

The curves are not merely shifted, either: a chunk-state tile of 16 is 8% off
the optimum on Orin and 26% off on Thor, and a chunk-gemm tile of 32 -- which
was the sm80 default until the kernel's two products were fused -- is a 10%
regression rather than a mild one.

Every field has an environment override (`APXINF_GDN_*`, named in
`gdn_policy.rs`) so a new board can be re-swept without a rebuild. Re-sweep
after changing any of these kernels: the chunk-state block width has moved
twice already, and the chunk-gemm width once.

## When a measurement is needed

Three things decide correctness questions on this model, in this order, and the
four-mode gate is not one of them -- it reports scene 0's maximum trajectory
error, which on this checkpoint only ever takes the two adjacent BF16 output
ULPs 0.0403 and 0.0806, so it flips on changes that move nothing:

1. `cargo test -p apxinf-cuda gdn_ -- --nocapture` runs the fp64 operator
   oracles for the GDN kernels and the bit-exactness test for the value split.
   A kernel change that claims to preserve the arithmetic has to say so here.
2. The precision probe under `devlocal/qwen-drive/` reports token agreement and
   error distributions over every scene of every runnable mode, which move
   continuously where the gate does not.
3. The Orin performance probe for speed, always as alternating A/B rounds on
   one machine.

A recorded store belongs to the toolkit that measured it, so it is committed
at the path that says so:

```
configs/tuning/nvidia/thor-sm110/tactics.json                       CUDA 13.0, 116 records
configs/tuning/nvidia/thor-sm110/cuda13.2-cublas13.4/tactics.json   CUDA 13.2,   7 records
```

Both boards work. The autotuner rewrites rather than merges, so recording on
13.2 into the unqualified file would have left a Thor on 13.0 with seven
records for a toolkit it is not running. An autotune report is an artifact,
not a deliverable -- the engine only ever appends to it -- so it does not
belong in a commit either.

Measurement evidence for the three boards -- rooflines, kernel tables, the
per-device sweeps behind the table above -- is kept out of the review diff
under `devlocal/qwen-drive/reports/` per `AGENTS.md`, along with the probes
that produced it and the two subsystems this revision does not reach: the
packed-weight GEMV and the perception scaffold.

One measurement note that costs hours if it is learned the hard way: on a
Jetson the same build measures several percent apart over an afternoon, and
6.29 s/scene at 14:20 became 6.75 s at 17:40 with nothing changed. Any
comparison has to be alternating rounds of both sides on one machine within
one run. Two numbers an hour apart are not a comparison.

## Whole-model direct planning on Thor (sm_110)

Measured on Thor with clocks and fan pinned (`nvpmodel` MAXN,
`jetson_clocks --fan`, GPU min=max=1575 MHz, `nvfancontrol` stopped), scene 0
of the fixed workload, BF16, ten flow steps, batch 1, ten warm-up and thirty
timed calls per arm. Both arms are the same checkout's Python package and the
same empty tactic store, so the native library is the only variable. Eight
arms interleaved ABBA then BAAB, because of the drift note above.

| arm | request p50 | model p50 |
| --- | --- | --- |
| before, four arms | 2122.801 ms | 2092.855 ms |
| after, four arms | 778.346 ms | 743.799 ms |

That is 1344.455 ms, a 2.73x speedup and 63.3% off the request. Arm spread was
42.772 ms before and 39.668 ms after, so the gain is two orders of magnitude
clear of the variation the board contributes.

Equivalence is stronger than a tolerance here. Every trajectory is repeatable
within an arm, and across the whole 242-scene fixed-profile NAVSIM set all 242
trajectories are **bit-identical** between the two arms -- not close, equal.
Any PDM score computed from them is therefore identical by construction, so no
scoring run can distinguish the two. The 128-key/64-token GDN specializations
that produce the speedup keep each output's accumulation order, and the BF16
Q/K storage only makes physical a rounding their producers had already applied.


# π0-FAST

Physical Intelligence's π0-FAST, ported into ApxInf as a native-BF16 CUDA
runtime. It shares the PaliGemma backbone with π0.5 (SigLIP So400m/14 vision +
Gemma-2B text) but replaces flow-matching action generation with **autoregressive
FAST action-token decoding**: the same LM head emits discrete tokens one at a
time, and a FAST tokenizer turns them back into a continuous action chunk.

The two families are deliberately independent. This module carries its own
architecture code, config parser, weight loader and executors, and reaches the
accelerator only through model-neutral kernels; nothing here is shared with
`pi05`/`walloss` except those kernels. Decode runs in native BF16, or in FP8
E4M3 with `precision="fp8"`; see [Precisions](#precisions).

## Checkpoint

`lerobot/pi0fast-libero-v044`. Facts that shape the code, all read from the
checkpoint's own `config.json`:

| | |
|---|---|
| language | Gemma-2B: width 2048, 18 layers, 8 query heads, 1 KV head, head dim 256, MLP 16384 |
| vision | SigLIP So400m/14: 27 layers, width 1152, MLP 4304, 16 heads, head dim 72 |
| images | 224x224, patch 14 -> 256 patches per view |
| views | 2 real + 1 padded (`empty_cameras: 1`) -> 512 patch tokens |
| vocabulary | 257152, with `fast_skip_tokens: 128` reserved for FAST ids |
| LM head | tied token embedding, pruned to 2112 of 257152 columns (see below) |
| actions | 7-dim, chunk 10, `max_action_tokens: 256` |

The third camera is padding, not a sensor. LeRobot appends an all-`-1` view with
an all-zero pad mask, and the reference positions tokens with
`cumsum(pad_mask) - 1`; dropping the view is therefore exact, not an
approximation, and this port consumes two views. See
`devlocal/pi0-fast/reports/08-views-and-empty-camera.md` for the argument.

## Module layout

| file | role |
|---|---|
| `config.rs` | checkpoint + execution config, parsed from the checkpoint JSON |
| `weights.rs` | typed checkpoint weights; transposes `[out, in]` -> `[in, out]`, folds the RMSNorm offsets |
| `device_weights.rs` | host-side packing (QKV and gate/up concatenated along the output dim) |
| `bf16_weights.rs`, `static_bf16_weights.rs` | device-resident BF16 weights |
| `bf16_executor.rs` | transformer-layer execution; owns the decode projection seam |
| `bf16_runtime.rs` | fixed-shape inference: prefix pass + autoregressive loop |
| `vla_runtime.rs` | owning VLA frontend (`VlaRuntime`), loads the checkpoint |
| `backend.rs` | compile-time accelerator seam |

## Execution

One `infer` call is two phases.

1. **Prefix.** The SigLIP tower and its projector produce image embeddings, the
   prompt ids are looked up and scaled by `sqrt(width)`, the two are
   concatenated, and the Gemma stack runs once over the whole prefix with
   bidirectional attention. Every layer parks its K/V in a cache sized for the
   prefix plus the full token budget.
2. **Decode.** One token per step: embed the previous id, run it through all 18
   layers against the growing cache, project through the tied LM head, argmax on
   device, and feed the id straight back in. The loop stops when the `|`
   terminator is emitted, which on LIBERO frames lands at token 13-30.

`max_action_tokens` (256) is a **loop bound, not the workload**. A terminated
call costs `prefix + ~23 steps`; the same call without a stop token costs 8.4 s
instead of 0.86 s. The whole traversal runs inside one persistent
`GraphWorkspace` (bump arena) sized from the config and reused across calls, so
no operator allocates: this is what took `cudaMalloc`/`cudaFree` per call from
~6051 pairs to ~25.

## Precisions

`precision="bf16"` is the default, and what `auto` resolves to. `precision="fp8"`
runs every projection as E4M3 with per-tensor weight scales and per-site measured
activation scales from a calibration profile. An autoregressive argmax decoder has
no error budget to spend on a guessed scale, so FP8 **requires** a profile
(`<model-dir>/calibration.json`, or an explicit `calibration=`) and refuses to
load without one; `APXINF_PI0FAST_FP8_ACTIVATION_SCALE` overrides that for
bring-up and prints a warning. See `doc/pi0fast-fp8-calibration.md`.

## Accuracy: LIBERO-10

`in_process` backend, BF16, `--action-dim 7`, one trial per task, seed 7,
replanning every 5 actions. **7/10**, measured before the decode-GEMV tuning
described below (that build measured 6/10 on the same protocol; see there).

| task | instruction | result | steps | replans |
|---|---|---|---|---|
| 0 | put both the alphabet soup and the tomato sauce in the basket | pass | 387 | 78 |
| 1 | put both the cream cheese box and the butter in the basket | fail (step cap) | 520 | 104 |
| 2 | turn on the stove and put the moka pot on it | pass | 415 | 83 |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | pass | 332 | 67 |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | pass | 315 | 63 |
| 5 | pick up the book and place it in the back compartment of the caddy | pass | 195 | 39 |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | fail (step cap) | 520 | 104 |
| 7 | put both the alphabet soup and the cream cheese box in the basket | pass | 264 | 53 |
| 8 | put both moka pots on the stove | pass | 473 | 95 |
| 9 | put the yellow and white mug in the microwave and close it | fail (step cap) | 520 | 104 |

All three failures run to the 520-step cap rather than stopping early. Model time
inside the rollout is 1.2-1.8 s per call (mean 1.5 s), one call per 5 actions -
higher than the 1.29 s the standalone benchmark reported for that build, because
a rollout also contends with the MuJoCo simulator and runs longer token streams.

LeRobot's π0-FAST documentation reports 60% for this suite, so 7/10 is in the
expected range. Do **not** compare it against π0.5's LIBERO-10 number (93%): the
two checkpoints are different models with different action heads.

On the same protocol and build, **libero_spatial scores 7/10** (failures at the
520-step cap on tasks 3, 4 and 5) and **libero_goal 1/1** with the suite
abandoned after the first task; `libero_object` has not been run. Both suites
also exist for the tuned build in `devlocal/pi0-fast/results/` (spatial 8/10).

Two caveats on reading this table:

- One trial per task gives 10 samples. Task 8 alone took 442 steps on one build
  and 473 on another whose model outputs are bit-identical, which is enough to
  move a task between pass and fail. Treat ±1-2 tasks as the noise band, and do
  not use this protocol to adjudicate small changes - use multiple trials.
- A hand-written decode GEMV that differed from cuBLAS by at most one ulp on
  99.99% of elements scored 3/10 on the same protocol. The gate is much sharper
  than its apparent statistical power.

### Jetson AGX Thor

Same protocol on Thor (sm_110, 20 SMs), one trial per task, seed 7:

| precision | LIBERO-10 | mean model time per call |
|---|---|---|
| BF16 | 6/10 | 504 ms |
| FP8 E4M3, calibrated | 4/10 | 313 ms |

FP8 **without** a profile scored 0/6 before that run was abandoned: the decode
never emits its `|` terminator, so every rollout spends the full step budget on
tokens the policy throws away. The calibrated profile recovers the terminating
stream but not the last two tasks, so treat FP8 as a speed path with its own
accuracy budget rather than as a drop-in for BF16.

## Latency

Orin AGX 64GB (sm87), BF16, 2 views / 512 patch tokens, 20 held-out LIBERO
frames, `scripts/bench_pi0_fast.py --mode ar`: a least-squares fit over the
distinct terminator positions the frames actually produce, `r2 = 1.0000`. The
same command against the pre-tuning tactic database gave 52.30 ms/step.

| | |
|---|---|
| prefix (vision tower + prompt prefill + first argmax) | **122 ms** |
| per autoregressive token | **32 ms** |
| typical call, 23 tokens | 0.86 s |
| full 256-token budget | 8.4 s |

`--mode all` over the same 20 frames reports L1 p50 898 ms and L2 p50 927 ms
(the median frame costs 24 tokens). The decode term is 87% of a call. Each step
streams ~5.01 GB of weights (18 layers x 220.2 MB + the 1.05 GB LM head) and
therefore runs at ~156 GB/s, inside the 141-165 GB/s this board sustains on a
pure streaming read: the decode step is at the memory roofline. `nsys` already
shows 97.9% GPU busy, so no launch overhead remains to remove, and the only
headroom left is reading *fewer* bytes.

### Jetson AGX Thor

Thor (sm_110, 20 SMs), the shipped `configs/tuning/nvidia/thor-sm110/tactics.json`,
pruned LM head. Step numbers are a least-squares fit from
`devlocal/pi0-fast/scripts/step_sweep.py` (2 frames x 3 repeats, terminator forced
through the binding, `r2 = 1.0000`); the call numbers are
`bench_pi0_fast.py --mode latency` over 10 synthetic frames at the full
256-token budget, and the same policy inside a LIBERO-10 rollout.

| | BF16 | FP8 E4M3 |
|---|---|---|
| prefix (vision tower + prefill + first argmax) | 38.1 ms | 34.7 ms |
| per autoregressive token | **18.7 ms** | **10.6 ms** |
| full 256-token call, L1 p50 | 4860 ms | 3071 ms |
| LIBERO-10 rollout, mean model time per call | 504 ms | 313 ms |

## The decode GEMV: tuned cuBLASLt tactics

Essentially the whole decode step is a matrix-vector product: 73 batch-1 GEMMs
(four per layer - `qkv`, `o_proj`, `gate_up`, `down` - plus the LM head), and
attention, elementwise and argmax together are ~2 ms of the step.

**cuBLAS' default algorithm did not saturate the memory bandwidth** - the step
ran at ~96 GB/s where this board sustains 141-165 GB/s - but that is a choice of
algorithm, not a bad kernel. Three of the five m=1 shapes are now pinned to a
cuBLASLt tactic in the hardware database
(`configs/tuning/nvidia/orin-sm87/tactics.json`): `qkv` (n=2560, k=2048) and
`gate_up` (n=32768, k=2048) to algorithm 4, `down` (n=2048, k=16384) to
algorithm 6.

No model code changed: `auto.rs` resolves the device database on `Model.load`,
and `bf16_executor::projection()` (via `language_layer_cached_decode_bf16`)
consults it per GEMM. `o_proj` and the LM head are deliberately **not** in the
database. They have no bandwidth headroom, and `o_proj` is the one m=1 shape
whose result is nowhere near bit-identical to cuBLAS (74.42% of elements
identical, up to 127 ulp - a transposed operand sends cuBLAS down a precision-
losing K-split). Dropping them costs nothing: tuning all five shapes measured
32.33 ms/step against 32.34 ms/step for these three.

| | cuBLAS default | three tuned shapes |
|---|---|---|
| per decode step | 52.3 ms | **32.2 ms** |
| fixed term (prefix + first argmax) | 102.5 ms | 121.0 ms |
| 24-token call | 1358 ms | **898 ms** |

Measured by interleaving the two arms A/B/A/B in one session (`--mode ar`, all
four runs `r2 = 1.0000`): the default gives 52.30/52.32 ms per step and
102.4/102.5 ms fixed, the tuned database 32.16/32.21 and 121.6/120.3. The
per-step win is a reproducible 1.62x. The **fixed term really does grow by ~19 ms
and that is unexplained**: a different benchmark in a different session put the
same delta at only +6 ms, so this term needs its own profile before it is
understood. The net effect is still large - a 24-token call drops from 1358 ms to
898 ms - but do not quote "1.62x on a call": at 23 tokens it is 1.51x.

**This is not bit-exact.** A different algorithm sums K in a different order, and
on LIBERO frames only 7 of 20 decoded token streams come out identical to the
cuBLAS default. Paired over 20 LIBERO tasks the two arms score 14/20 each
(libero_10 7 -> 6, libero_spatial 7 -> 8): four tasks flip, two in each
direction. That is evidence of no *systematic* regression at that sample size,
not a proof of equivalence - ten samples per suite cannot resolve a few points,
and this sits awkwardly against the 3/10 hand-kernel result in the caveats
above. The one observable difference between the two episodes is that the hand
kernel replaced every projection including `o_proj`, while this change leaves
`o_proj` and the LM head on cuBLAS. Evidence:
`devlocal/pi0-fast/reports/18-decode-gemv-dispatch.md` and
`20-spatial-and-gemv3-accuracy.md`.

## The pruned LM head

The head is the largest read in a decode step (1.05 GB of 5.01 GB in BF16, 21%)
and only its argmax matters, so it keeps the columns the decode can emit
(`Pi0FastConfig::action_head_columns()`): the 2048-wide action window at the tail
of the vocabulary, plus `action_head_extra_tokens` - `"Action: "` (4022, 235292,
235248) and the `"|"` terminator (235371). The extras are not optional: they are
nowhere near the tail, so a tail-only window would change the first three steps
and make the stop condition unreachable. That is 2112 of 257152 columns, padded
to a multiple of 64 because cuBLASLt rejects an unaligned E4M3 leading dimension;
the padding repeats the last column and remaps to the same id.

The argmax writes `remap[argmax]`, so the decode loop, the tied embedding lookup
and `stop_token` all keep speaking global token ids. The pruned head is the
full-vocabulary argmax restricted to the retained set, which makes it identical
whenever the unpruned winner is in that set - a property of the FAST protocol
rather than of the arithmetic. Widen `action_head_extra_tokens` (or
`action_vocab_size`) if a deployment ever emits an id outside it.

The head read drops from 1.05 GB to 8.6 MB per step in BF16, and from 526 MB to
4.3 MB in FP8.

## Reproducing

```sh
# LIBERO-10 accuracy
MUJOCO_GL=egl python scripts/eval_libero.py --backend in-process \
  --model-dir <checkpoint> --precision bf16 \
  --action-dim 7 --suite libero_10 --trials-per-task 1 \
  --results-jsonl out/libero_10.jsonl --summary-json out/libero_10.summary.json

# FP8: calibrate on the same suite first. The profile lands in
# <checkpoint>/calibration.json and the evaluator picks it up by itself.
MUJOCO_GL=egl python scripts/calibrate_pi0fast.py \
  --model-dir <checkpoint> --libero-suite libero_10
MUJOCO_GL=egl python scripts/eval_libero.py --backend in-process \
  --model-dir <checkpoint> --precision fp8 \
  --action-dim 7 --suite libero_10 --trials-per-task 1 \
  --results-jsonl out/libero_10_fp8.jsonl --summary-json out/libero_10_fp8.summary.json

# latency, and the fixed / per-step split
python scripts/bench_pi0_fast.py --model-dir <checkpoint> \
  --state-key observation/state --mode all \
  --frames devlocal/pi0-fast/results/raw/libero_frames.npz \
  --out devlocal/pi0-fast/results/bench_pi0_fast.json

# what the tactic database is worth on this board: same script, same frames,
# once against the pre-tuning database and once against the shipped one
PI0FAST_TACTICS=<db> python devlocal/pi0-fast/scripts/bench_default_db.py  # omit to use the shipped db
```

Tactics are device-specific: the algorithm ids in
`configs/tuning/<vendor>/<family>-sm<version>/tactics.json` are only valid for
that compatibility domain and are not portable between Orin and, say, RTX 4090.
A new board stays on the cuBLAS default until it runs its own autotune
(`Model.load(..., tactics=<db>, autotune=True)`, or
`devlocal/pi0-fast/scripts/autotune_decode_gemv.py --phase tune`).

The FP8-to-BF16 projections have their own records (`op=fp8_bf16`) in the same
database and likewise ship only a Thor entry: on another board FP8 keeps the
cuBLASLt default until that board runs its own autotune.

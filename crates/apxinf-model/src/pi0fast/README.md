# π0-FAST

Physical Intelligence's π0-FAST, ported into ApxInf as a native-BF16 CUDA
runtime. It shares the PaliGemma backbone with π0.5 (SigLIP So400m/14 vision +
Gemma-2B text) but replaces flow-matching action generation with **autoregressive
FAST action-token decoding**: the same LM head emits discrete tokens one at a
time, and a FAST tokenizer turns them back into a continuous action chunk.

The two families are deliberately independent. This module carries its own
architecture code, config parser, weight loader and executors, and reaches the
accelerator only through model-neutral kernels; nothing here is shared with
`pi05`/`walloss` except those kernels. Because the whole model is validated in
BF16, that is the only precision this module implements.

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
call costs `prefix + ~23 steps`; the same call without a stop token costs 13.6 s
instead of 1.3 s. The whole traversal runs inside one persistent
`GraphWorkspace` (bump arena) sized from the config and reused across calls, so
no operator allocates: this is what took `cudaMalloc`/`cudaFree` per call from
~6051 pairs to ~25.

## Accuracy: LIBERO-10

`in_process` backend, BF16, `--action-dim 7`, one trial per task, seed 7,
replanning every 5 actions. **7/10.**

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
higher than the 1.29 s the standalone benchmark reports, because a rollout also
contends with the MuJoCo simulator and runs longer token streams.

LeRobot's π0-FAST documentation reports 60% for this suite, so 7/10 is in the
expected range. Do **not** compare it against π0.5's LIBERO-10 number (93%): the
two checkpoints are different models with different action heads.

Two caveats on reading this table:

- One trial per task gives 10 samples. Task 8 alone took 442 steps on one build
  and 473 on another whose model outputs are bit-identical, which is enough to
  move a task between pass and fail. Treat ±1-2 tasks as the noise band, and do
  not use this protocol to adjudicate small changes - use multiple trials.
- A hand-written decode GEMV that differed from cuBLAS by at most one ulp on
  99.99% of elements scored 3/10 on the same protocol. The gate is much sharper
  than its apparent statistical power.

## Latency

Orin AGX 64GB (sm87), BF16, 2 views / 512 patch tokens, 20 held-out LIBERO
frames. Measured with `scripts/bench_pi0_fast.py`; the fixed/per-step split is a
least-squares fit over 13 distinct terminator positions (`r2 = 0.99998`) and was
confirmed directly by re-running with `max_action_tokens` rewritten to 1..8
(`r2 = 0.99995`).

| | |
|---|---|
| prefix (vision tower + prompt prefill + first argmax) | **96 ms** |
| per autoregressive token | **52 ms** |
| typical call, 23 tokens | 1.29 s |
| full 256-token budget | 13.6 s |

The decode term is 92% of a call. Each step streams ~5.02 GB of weights (18
layers x 220.2 MB + the 1.05 GB LM head) and therefore runs at ~97 GB/s, against
the 141-165 GB/s this board sustains on a pure streaming read. `nsys` already
shows 97.9% GPU busy, so the gap is kernel efficiency, not launch overhead, and
CUDA graph capture/replay - which this family does not implement yet - has
almost nothing to win.

## Next step: the decode GEMV

Essentially the whole decode step is a matrix-vector product. The step's only
real work is streaming ~5.02 GB of weights (18 layers x 220.2 MB + the 1.05 GB LM
head), and every byte of it is read by a batch-1 GEMV; attention, elementwise
and argmax together are ~2 ms of the 52 ms.

**cuBLAS does not saturate the memory bandwidth.** On the layer projections it
reads at 100-150 GB/s, where this board sustains 141-165 GB/s on a pure streaming
read (the LM head on its own reaches 178). Closing that gap is the next step, and
it is worth roughly **1.5x on the decode term** - ~17 ms per token, ~0.4 s on a
typical call.

`bf16_executor::projection()` is the seam it belongs behind, and
`language_layer_cached_decode_bf16` is the layer built on it. A hand-written GEMV
was already tried there, reached 150-177 GB/s, and had to be removed again: it
sums K in a different order than cuBLAS, and argmax over 257152 logits turns a
one-ulp difference into a different action token (LIBERO-10 went 7/10 -> 3/10).
Any replacement therefore has to be bit-exact with the cuBLAS kernel it replaces,
or be validated end to end far beyond this protocol's noise band. The
measurements are in `devlocal/pi0-fast/reports/18-decode-gemv-dispatch.md`.

## Reproducing

```sh
# LIBERO-10 accuracy
MUJOCO_GL=egl python scripts/eval_libero.py --backend in-process \
  --model-dir <checkpoint> --precision bf16 \
  --action-dim 7 --suite libero_10 --trials-per-task 1 \
  --results-jsonl out/libero_10.jsonl --summary-json out/libero_10.summary.json

# latency, and the fixed / per-step split
python scripts/bench_pi0_fast.py --model-dir <checkpoint> \
  --state-key observation/state --mode all \
  --frames devlocal/pi0-fast/results/raw/libero_frames.npz \
  --out devlocal/pi0-fast/results/bench_pi0_fast.json
```

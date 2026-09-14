# Qwen-Drive on Jetson Orin (sm_87): optimization brief

Written for a cloud agent with no GPU and no access to the Orin or 4090
machines. Its output is code candidates; a human verifies them on hardware.

Revised after a first measurement pass on a Jetson AGX Orin Developer Kit
(JetPack R36.5, CUDA 12.6, sm_87, BF16). The ranking changed: what looked like
three separate sm_87 gaps is mostly one gap, and that one is not
architecture-specific at all.

## Standing constraints

The 4090 implementation this branch carries is **not qualified**. See
`experiment/direct-takeover/RESULTS.md`: VQA scene 2 still diverges at decode
token 87, direct and reasoning fail the trajectory gate, and perception raises
a pending-implementation error. Do not describe any of this as a working
deployment, and do not build work on the assumption that it is correct.

Never claim a speedup. Nothing in the cloud sandbox can measure one. Report
what a change *should* do and what measurement would confirm it.

Never relax an accuracy gate, change sampling tie-breaking, or widen a
tolerance to make an example pass. The gates are fixed: exact token equality
for vqa/reasoning, and trajectory error `<= 0.02 + 0.005*|expected|` with
axis-2 max `<= 0.02`.

## What measurement on this device is worth

Two properties of the measured system invalidate naive comparisons, and both
were learned the hard way:

**The model is not run-to-run deterministic.** Three runs of one unmodified
binary gave direct-mode trajectory errors of 0.04028463363647461, then
0.04028606414794922 twice. A difference at that scale is never evidence of a
source change. The stable invariants are coarser: the VQA divergence index and
the token at it, the reasoning error 0.08056640625, the perception error.

**The first scene of a run is not comparable.** The harness's single warmup
iteration does not settle this device: scene 0 measured 12.97s cold against
19.01s for the same arm minutes later, while scenes 1-3 matched to within
1.3%. An arm compared against its own earlier record still reported a geomean
of 0.913, entirely from scene 0. Compare scenes 1-3, or add warmup.

## Architecture gates that matter on sm_87

`crates/apxinf-cuda/build.rs` decides which fast paths exist:

| cfg | sm_89 (4090) | sm_87 (Orin) |
|---|---|---|
| FA2 BF16 SM80 kernels (`is_fa2_sm80_family`) | on | **on** |
| `apxinf_cutlass_int8_sm80` | on | **on** |
| `apxinf_cutlass_bf16_sm89` (`is_cutlass_sm89_family`) | on | **off** |
| `apxinf_cutlass_gemm` / `apxinf_cutlass_fmha` (sm100 only) | off | off |

Worth knowing, but see Target A: qwen-drive does not reach the tuned or fused
GEMM paths on any device, so these cfgs are not what is holding it back.

## Target A -- qwen-drive is not on the tuned GEMM path (highest priority)

This is the finding that reorganized the list, and it is not sm_87-specific:
it holds on the 4090 too.

| | pi05 | qwen-drive |
|---|---|---|
| linear layers | `gemm::bf16` (17 sites) | `gemm::bf16_bias` (10 sites) |
| GeGLU | `gemm::bf16_geglu_fused` | `gemm::matmul` + separate activation |
| resolves a tactic | yes | **no** |

`gemm::bf16_bias` dispatches straight to `apxinf_static_bf16_gemm_bias` and
never consults the tuning session, so:

- `configs/tuning/nvidia/*/tactics.json` holds only pi05 shapes on every
  device, because qwen-drive's path cannot record any. Its `k` values never
  include 2560, which is qwen-drive's hidden size.
- An AutoTune pass over the four-scene VQA workload wrote **zero** records and
  ran at inference speed, confirming the path is never reached.
- The fused GeGLU is unreachable for the same reason -- this is not a missing
  sm_87 kernel, it is a call site asking for the unfused composite.

The work is to route qwen-drive's linear layers onto the tuned API. Note that
`gemm::bf16` takes no bias, which is presumably why `bf16_bias` was used;
`Epilogue::Bias` and `TacticBackend::CublasLtCustomBias` already exist, but
`CublasLtCustomBias` is registered for `GemmOp::Fp8F16`, not BF16. So a BF16
bias operator has to be registered for tuning before the call sites can move.

Do this incrementally and keep each step separately verifiable. One converted
call site that resolves a tactic is worth more than a broad rewrite nobody can
validate.

Why it matters: decode measures ~19.3s per 64-token scene, about 300ms/token.
The VLM weights are roughly 8GB and this board has on the order of 200GB/s, so
a weight-bound floor is nearer 40ms/token. The gap is wide enough that GEMM
selection is worth attacking.

## Target B -- the sm_89-shaped gate on the linear-attention fast path

`crates/apxinf-cuda/src/kernels/linear_attention.rs` around line 1205 gates a
fast path on `compute_major == 8 && compute_minor == 9 && q_shape[2] == 256`,
choosing `fa2_attention_splitkv` when `q_shape[0] <= 64` and
`composed_gqa_bf16` otherwise. That test names a **device model**, not a
capability. `apxinf_fa2_sm80` is enabled on sm_87, so the kernel is already
compiled into the Orin binary and is merely fenced off; sm_87 falls back to
the generic `apxinf_static_fa2_bf16`.

Qwen-Drive uses GDN (see `forward_gdn` in
`crates/apxinf-model/src/qwen_drive/general.rs`), so it takes this path.

Make the gate capability-based so sm_87 can use it. sm_89 behaviour must stay
byte-identical. Comment why the original limit existed and what evidence says
sm_87 also holds. If you find a genuine sm_89 dependency (shared-memory
capacity, specific PTX, wgmma), do not force it -- write down why not.

## Already done, do not redo

`build(cuda): let FA2 drop the feature axes the adapter never reaches` adds
`APXINF_FA2_TRIM_UNUSED`, defining the four FlashAttention-2 trim macros the
adapter provably never needs. Opt-in. Measured at sm_87: an untrimmed build
ran **15h33m without finishing** (9h23m of that inside one `ptxas` invocation
for `flash_fwd_split_hdim256_bf16_sm80`), while the trimmed build completed in
**18m02s**. `UNEVEN_K` is deliberately left enabled. Making the default on is
a reasonable follow-up once a trimmed build has cleared the accuracy gate on a
target device.

`perf(qwen-drive): gate the TEMP-DIAG probes behind APXINF_QWEN_DIAG` and
`fix(qwen-drive): install the GEMM tactic store on the direct load path` are
both in. Neither changes measured throughput: the first measured at parity on
scenes 1-3 with the same binary under both settings, and the second is inert
until Target A moves the call sites. Do not re-litigate either; build on them.

## How to work

1. `git fetch --all` and read `git log --oneline -15` first. Check for existing
   `codex/orin-sm87-*` branches and open PRs -- **do not redo finished work**.
2. Pick the highest-priority target that is not already done.
3. One branch per candidate, named `codex/orin-sm87-<slug>`.
4. `cargo check` is the most you can validate; the CUDA feature will not build
   without a toolkit. Say plainly what you could and could not compile.
5. Open a PR whose body states: what changed, why it should help, why sm_89 is
   unaffected where that applies, and **the exact command a human should run on
   the Orin to verify** -- accuracy gate first, then performance on scenes 1-3.

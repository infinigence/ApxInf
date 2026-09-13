# Qwen-Drive on Jetson Orin (sm_87): optimization brief

Written for a cloud agent with no GPU and no access to the Orin or 4090
machines. Its output is code candidates; a human verifies them on hardware.

## Standing constraints

The 4090 implementation this branch carries is **not qualified**. See
`experiment/direct-takeover/RESULTS.md`: VQA scene 2 still diverges at decode
token 87, direct and reasoning fail the trajectory gate, and perception raises
a pending-implementation error. Do not describe any of this as a working
deployment, and do not build work on the assumption that it is correct.

Never claim a speedup. Nothing in the cloud sandbox can measure one. Report
what a change *should* do and what measurement would confirm it.

Never relax an accuracy gate, change sampling tie-breaking, or widen a
tolerance to make an example pass. The gates are fixed:
exact token equality for vqa/reasoning, and trajectory error
`<= 0.02 + 0.005*|expected|` with axis-2 max `<= 0.02`.

## Architecture gates that matter on sm_87

`crates/apxinf-cuda/build.rs` decides which fast paths exist:

| cfg | sm_89 (4090) | sm_87 (Orin) |
|---|---|---|
| FA2 BF16 SM80 kernels (`is_fa2_sm80_family`) | on | **on** |
| `apxinf_cutlass_int8_sm80` | on | **on** |
| `apxinf_cutlass_bf16_sm89` (`is_cutlass_sm89_family`) | on | **off** |
| `apxinf_cutlass_gemm` / `apxinf_cutlass_fmha` (sm100 only) | off | off |

## Target 3 — highest priority

`crates/apxinf-cuda/src/kernels/linear_attention.rs` ~line 1205 gates a fast
path on `compute_major == 8 && compute_minor == 9 && q_shape[2] == 256`,
choosing `fa2_attention_splitkv` when `q_shape[0] <= 64` and
`composed_gqa_bf16` otherwise. That test names a **device model**, not a
capability. `apxinf_fa2_sm80` is enabled on sm_87, so the kernel is already
compiled into the Orin binary and is merely fenced off; sm_87 falls back to the
generic `apxinf_static_fa2_bf16`.

Qwen-Drive uses GDN (see `forward_gdn` in
`crates/apxinf-model/src/qwen_drive/general.rs`), so it takes this path.

Make the gate capability-based so sm_87 can use it. sm_89 behaviour must stay
byte-identical. Comment why the original limit existed and what evidence says
sm_87 also holds. If you find a genuine sm_89 dependency (shared-memory
capacity, specific PTX, wgmma), do not force it — write down why not.

## Target 1

`configs/tuning/nvidia/orin-sm87/tactics.json` holds 52 records covering
exactly the same shapes as `rtx4090-sm89`, and those are PI0.5 shapes — `k`
never takes 2560. Qwen-Drive's GEMM shapes are in no tactic cache on any
device, so both boards fall back to cuBLASLt runtime heuristics.

Read the autotune machinery under `crates/apxinf-cuda/src/tuning/` and produce
a reproducible script plus documentation for collecting Qwen-Drive's real
shapes on an Orin and writing them into that file. **Invent no `milliseconds`
values** — the file is a measurement record.

## Target 2

`default_bf16_geglu_tactic` in `crates/apxinf-cuda/src/kernels/gemm/bf16.rs`
(~line 678) returns the unfused `GemmThenGeGlu` composite whenever a plain
weight exists. The fused variants — `CutlassBf16GeGluSm89` and
`CutlassBf16DualGeGluM522/M533` — are reachable only at `sm == 89` and
`sm == 110`. Orin therefore runs GeGLU unfused.

This is the largest piece of work and the least likely to land in one pass.
Prefer a written design over a half-finished kernel.

## How to work

1. `git fetch --all` and read `git log --oneline -15` first. Check for existing
   `codex/orin-sm87-*` branches and open PRs — **do not redo finished work**.
2. Pick the highest-priority target that is not already done.
3. One branch per candidate, named `codex/orin-sm87-<slug>`.
4. `cargo check` is the most you can validate; the CUDA feature will not build
   without a toolkit. Say plainly what you could and could not compile.
5. Open a PR whose body states: what changed, why it should help on sm_87, why
   sm_89 is unaffected, and **the exact command a human should run on the Orin
   to verify** — accuracy gate first, then performance.

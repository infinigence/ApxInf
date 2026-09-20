# Vendored: Flash-Attention 2 forward kernels

ApxInf compiles the BF16 head-dimension 96 and 256 non-causal forward
instantiations for SM80/SM86/SM87/SM89. The ApxInf-owned sibling translation
unit `../fa2_hdim64_bf16.cu` additionally instantiates exact head-dimension 64
tiles for an SM87-only dispatch; it does not modify this vendored tree. On
SM100-family builds ApxInf also compiles the FP16 head-dimension 96 and 256
instantiations; the Thor PI0.5 FP8 path uses FP16 Gemma MQA (`head_dim=256`).
These paths use the repository-local raw-pointer wrapper in
`../fa2_bf16_sm80.cu`. Only the listed forward instantiations and the required
head-dimension 256 split-KV instantiation are compiled.

The split-KV count is selected outside the vendored tree by ApxInf's Rust
attention layer. Its occupancy policy is an independent Rust implementation of
the BSD-3-Clause `num_splits_heuristic` in upstream FA2's `flash_api.cpp`; the
raw-pointer CUDA wrapper receives the resulting count and only marshals kernel
parameters.

## Sources

### flash_attn/
- Upstream:  https://github.com/Dao-AILab/flash-attention
- Tag:       v2.7.4.post1
- License:   BSD-3-Clause (see `flash_attn/LICENSE`)
- Subset:    `csrc/flash_attn/src/` headers + BF16/FP16 forward kernel
             instantiations. bwd, CK (AMD) path, Hopper FA3,
             alibi/rotary/dropout runtime paths (headers kept for
             compile; p_dropout=0 in inference means inert), and the
             torch-coupled `flash_api.cpp` pybind wrapper are excluded
             — ApxInf ships its own `fa2_bf16_sm80.cu` wrapper.

### cutlass/
- Upstream:  https://github.com/NVIDIA/cutlass
- Commit:    c506e16788cb08416a4a57e11a9067beeee29420  (2025-01-08,
             between v3.7.0 and v3.8.0)
- License:   BSD-3-Clause (see `cutlass/LICENSE`)
- Subset:    `include/` subtree only. This is the submodule pin of
             Flash-Attention 2 v2.7.4.post1 — FA2 was authored
             against this CUTLASS snapshot, so we pin the same commit.

We keep a dedicated CUTLASS 3.x tree here separate from
`third_party/cutlass/` (which is v4.4.2, used by our FP4/FP8 kernels)
because CUTLASS 4.x has breaking CuTe layout-algebra changes that FA2
2.7.x does not support.

## PyTorch-free build boundary

`flash.h`, `philox_unpack.cuh`, and `flash_fwd_launch_template.h` are kept at
their upstream `v2.7.4.post1` contents. ApxInf does not depend on libtorch, so
the narrow ATen/C10 surface referenced by those files is implemented under
the sibling `../fa2_compat/` include root. Keeping the compatibility code out
of this directory makes the upstream provenance auditable with a byte-for-byte
comparison.

The compatibility layer is inference-only: FA2 is compiled with dropout,
ALiBi, soft-cap, and local attention disabled. It supplies a Philox state
carrier for template completeness and native CUDA error handling for launch
code. It is not intended to emulate PyTorch outside this build boundary.

The direct-E4M3 output path remains an ApxInf extension in
`flash_fwd_kernel.h`; it is unrelated to the PyTorch-free compatibility layer
and is selected only by `APXINF_FA2_DIRECT_E4M3`.

## Backporting upstream bugfixes

FA2 main line is in maintenance (each FA2 release is 5–50 LoC of
toolchain fixes, no algorithmic changes; big work goes to FA3 which
is SM90+ only).

Procedure when we want a fix from upstream:

```bash
# In /tmp, fetch upstream + diff
git clone https://github.com/Dao-AILab/flash-attention.git /tmp/fa-upstream
cd /tmp/fa-upstream
git log v2.7.4.post1..HEAD -- csrc/flash_attn/src/

# For each relevant commit, generate a patch and apply here
git format-patch -1 <SHA> --stdout > /tmp/fa-fix.patch
cd <apxinf>/crates/apxinf-cuda/kernels/cutlass/fa2/flash_attn
patch -p4 < /tmp/fa-fix.patch   # strip leading csrc/flash_attn/src/
# verify: rebuild + cos test
```

Record each applied fix in the commit log on this path. Do NOT edit
the CUTLASS submodule without bumping the pin commit above.

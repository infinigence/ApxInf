# Vendored: Composable Kernel (CK) FMHA forward, ck_tile subset

CK's tile-based flash-attention forward kernels, for the attention shapes pi05
runs on gfx1151. This directory holds only upstream material: headers copied
byte for byte and the output of CK's own code generator. The apxinf-owned C
ABI wrapper that marshals apxinf's tensors into CK's arguments belongs outside
this directory (`../ck_attention.hip`), and choosing between CK and apxinf's own
kernel belongs to `apxinf_hip::native::attention`. Until those land, nothing
compiles this directory.

## Source

- Upstream:  https://github.com/ROCm/composable_kernel
  (a mirror of `projects/composablekernel` in https://github.com/ROCm/rocm-libraries)
- Commit:    `f176bf5b118309cb475b20841daf3f17e5231017`
  (2026-10-25, "[rocm-libraries] ROCm/rocm-libraries#14360 (commit 326cd44)")
- License:   MIT, see `LICENSE` (copied from the same commit)

The ck_tile shipped with ROCm 7.2.1 (`/opt/rocm/include/ck_tile`) does not have
the gfx11 / gfx115 FMHA forward pipelines this needs, which is why apxinf
carries its own copy rather than using the system headers.

## What is here

| Path | Contents | Origin |
|---|---|---|
| `include/ck_tile/` | 301 headers | byte-identical copies from the commit above |
| `example/ck_tile/01_fmha/` | `fmha_fwd.hpp`, `bias.hpp`, `mask.hpp`, `quant.hpp`, `rotary.hpp` | byte-identical copies from the commit above |
| `generated/` | 27 kernel instances and API dispatchers | output of CK's `example/ck_tile/01_fmha/generate.py` at the commit above |
| `LICENSE` | CK's license | byte-identical copy |

The header subset is exactly the transitive include set of `generated/`, as
reported by `hipcc -MM`; nothing else from CK is carried. Upstream provenance is
auditable by comparing any file under `include/` or `example/` byte for byte
against the commit. `generated/` is deterministic: the same generator command
produces identical bytes on macOS and Linux.

Generated instances (all BF16, batch mode, row-major V, no bias, LSE output,
dropout or sink, target `gfx1151`):

- `fmha_fwd`: head dim 64, 128 and 256, each in both pipelines (`qr`, and
  `qr_hpad` for a head dim the tile has to pad -- pi05's 72 and GR00T's 48) and
  both mask settings. Twelve instances. The generator also offers 32 and 192;
  neither model has a head dim of 32 or in (128, 192], so the filter excludes
  them.

  The 64 tile is there because a head dim of 64 or 48 on the 128 tile pays
  twice: the head dimension is padded to 128, and the tile's 128-row M extent
  sits two-thirds empty at the DiT's 41 query rows. Measured, it takes GR00T's
  vision attention from 23% of this device's compute ceiling to 57% and its VL
  attention from 13% to 36%, and the model's attention from 3.79 ms to 2.44 ms
  per inference.
- `fmha_fwd_splitkv`: head dim 256 split-KV kernels (action-expert attention)
  and every head dim 256 combine kernel the split-KV dispatcher references.

The masked instances cover both causal alignments from one kernel:
`mask_top_left` and `mask_bottom_right` differ in the `mask_type` argument, not
in generated code (see `mask.hpp`'s `decode`, where both set
`left = -1, right = 0`). They exist for GR00T's text backbone, whose 16 layers
are the only causal attention either model issues; without them those layers
fall back to apxinf's own kernel at 190 us each against CK's 20 us.

The mask variants are generated under `logits` rather than `nlogits` -- the
logits soft-cap is compiled in and disabled at runtime through
`fmha_fwd_traits::has_logits_soft_cap`. One `fnmatch` pattern has to cover both
settings, because `generate.py` splits `-f` on `@` only for `fwd_splitkv` and
`bwd`; for `fwd` it passes the whole string to a single match
(`codegen/ops/fmha_fwd.py`). Two passes would not work either: each emits its
own `fmha_fwd_api.cpp`, and a dispatcher only knows its own pass's instances.

## Regenerating, or moving to a newer CK

```sh
git clone https://github.com/ROCm/composable_kernel.git
git -C composable_kernel checkout <new commit>
# update PIN in scripts/gen_ck_fmha.sh and the commit above
scripts/gen_ck_fmha.sh composable_kernel   # on a ROCm host: also refreshes the header subset
```

The script refuses a checkout at any commit other than its pin, and refuses to
replace `generated/` with a set that is missing any instance class the crate
needs. That second check exists because the script removes `generated/` before
copying: a filter that matches nothing leaves the generator exiting 0, the
dispatcher emitted with no instances behind it, and the working set already
gone. It happened once -- a stray `@` in the fwd pattern took out all four fwd
instances while splitkv survived, and the crate still built, because
`ck_attention.hip` is designed to fall back.

A shallow fetch of just the pinned commit is enough (`git init`,
`git fetch --depth 1 origin <pin>`); the checkout is only a source for the
generator and can be discarded afterwards. What makes the result auditable is
the pin, not the checkout.

After a bump, rebuild, run `cargo test -p apxinf-hip` on a gfx1151 device and
the pi05 accuracy gates (`doc/hip-pi05.md`), and update this file.

## Build flags

`crates/apxinf-hip/build.rs` compiles this directory, plus apxinf's
`kernels/ck_attention.hip` wrapper, with the flags below and nothing else with
them -- `CK_FLAGS` there is this list. `-enable-post-misched=0` in particular
is a different code generator, and the kernels outside this directory are tuned
and measured without it.

The CK translation units must be compiled with the flags CK's own CMake uses for
these kernels on ROCm >= 6.2, applied to those units only. These are the flags
the instances were benchmarked and validated with:

`build.rs` adds `-fPIC` to these: the objects are archived into a static
library that links into a Rust `cdylib`, which CK's own CMake has no reason to
care about. Everything else below is CK's.

```
-O3 -std=c++20 -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero
-fno-offload-uniform-block -mllvm --lsr-drop-solution=1 -mllvm -enable-post-misched=0
-fbracket-depth=1024
-DCK_TILE_FMHA_FWD_SPLITKV_API=1 -DCK_TILE_FMHA_FWD_APPENDKV_API=0 -DCK_TILE_FMHA_FWD_PAGEDKV_API=0
```

## Local modifications

None.

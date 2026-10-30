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
| `generated/` | 19 kernel instances and API dispatchers | output of CK's `example/ck_tile/01_fmha/generate.py` at the commit above |
| `LICENSE` | CK's license | byte-identical copy |

The header subset is exactly the transitive include set of `generated/`, as
reported by `hipcc -MM`; nothing else from CK is carried. Upstream provenance is
auditable by comparing any file under `include/` or `example/` byte for byte
against the commit. `generated/` is deterministic: the same generator command
produces identical bytes on macOS and Linux.

Generated instances (all BF16, batch mode, row-major V, no mask, bias, LSE
output, dropout or sink, target `gfx1151`):

- `fmha_fwd`: head dim 128 (pipelines `qr` and `qr_hpad`; vision attention's
  head dim 72 runs on `qr_hpad`) and head dim 256 (`qr`, `qr_hpad`; prefix
  attention).
- `fmha_fwd_splitkv`: head dim 256 split-KV kernels (action-expert attention)
  and every head dim 256 combine kernel the split-KV dispatcher references.

## Regenerating, or moving to a newer CK

```sh
git clone https://github.com/ROCm/composable_kernel.git
git -C composable_kernel checkout <new commit>
# update PIN in scripts/gen_ck_fmha.sh and the commit above
scripts/gen_ck_fmha.sh composable_kernel   # on a ROCm host: also refreshes the header subset
```

The script refuses a checkout at any commit other than its pin. After a bump,
rebuild, run `cargo test -p apxinf-hip` on a gfx1151 device and the pi05
accuracy gates (`doc/hip-pi05.md`), and update this file.

## Build flags

The CK translation units must be compiled with the flags CK's own CMake uses for
these kernels on ROCm >= 6.2, applied to those units only. These are the flags
the instances were benchmarked and validated with:

```
-O3 -std=c++20 -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero
-fno-offload-uniform-block -mllvm --lsr-drop-solution=1 -mllvm -enable-post-misched=0
-fbracket-depth=1024
-DCK_TILE_FMHA_FWD_SPLITKV_API=1 -DCK_TILE_FMHA_FWD_APPENDKV_API=0 -DCK_TILE_FMHA_FWD_PAGEDKV_API=0
```

## Local modifications

None.

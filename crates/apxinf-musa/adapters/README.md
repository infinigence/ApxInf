# MUSA host adapters

This directory is the MUSA/C++ host boundary consumed by Rust FFI. It mirrors
`crates/apxinf-cuda/adapters/` in structure and responsibility; the toolchain
underneath is `mcc` instead of `nvcc`.

MP22 decode and selected NN encode GEMMs are native: `asm_gemm_adapter.mu` loads the vendored NN/NT
images built into `mubin_images.h`. Gate/up retains KN layout, other decode
weights are transposed once by the same adapter. Eager and capture use the
same tile selection and engine stream, without Python or a MATE library.
Registered language KN weights additionally use measured M=522 encode tiles;
other encode shapes retain vendor dispatch. Grid order is stored per image.
See `../kernels/asm/README.md` for provenance and the full epilogue ABI.

- `mublas_adapter.mu` owns the mublas handle and the stable C ABI for the BF16
  GEMM operator. It is the one adapter that documents why it uses the API it
  does: `mublasLtMatmul` paired every launch with a `copyMatrixColMajorTiled`
  kernel, and `mublasGemmEx` does not.
- `mubin_fa_adapter.mu` owns the driver-API side of the MP22 attention assembly:
  loading the three `muModuleLoadData` images `build.rs` produces from
  `kernels/asm/`, the device metadata the kernels take, and the split-KV
  workspace. It is also where the split-KV merge kernel lives, because that
  kernel is device code the assembly does not contain but whose constants are
  tied to the assembly's layout.
- `mutlass_fa_adapter.mu` is the same boundary for the vendored mp_31 FMHA in
  `kernels/mutlass_fmha/`: the two `cu_seqlens` arrays those kernels take, the
  LSE buffer their epilogue writes, and a capture guard, because the first touch
  of a shape allocates. It and `mubin_fa_adapter.mu` keep separate metadata
  tables on purpose — different ABIs, mutually exclusive targets — and each file
  says so rather than leaving the duplication to look like an oversight.
- `adapter_common.h` is what those three do share, because it is about no ABI in
  particular: the per-thread failure message the `..._last_error()` accessors
  return, the capture guard they ask before their first allocation for a shape,
  and the two device questions — which card a stream belongs to, and how to
  select one before allocating on it — that every cache in this directory is
  keyed by. `custom_kernels.mu` includes it for that last pair. The metadata
  tables stay separate; these three do not.

Both FMHA adapters carry the same capture guard, for the same reason: each
allocates its metadata on the first touch of a shape, and an allocation inside a
capture either fails with no attribution or bakes a pointer the graph will
outlive. `mubin_fa_adapter.mu` went without one until the asymmetry was noticed
— mp_22 is the target where capture already ran, so it was the live risk.

`rope_frequency` in `custom_kernels.mu` has the same hazard and answers it the
other way: it allocates a frequency table on the first call for a
`(theta, half_dim)` pair, and instead of refusing during capture it **never
frees a superseded table**. Refusing would need a message surface this adapter
does not have, and retaining is affordable here in a way it is not for the two
FMHA adapters — their buffers are per shape, this one is per model config. The
function carries the reasoning.

Rust safe kernel contracts call the symbols defined here directly through the
private `src/ffi/` declarations. Operator implementation files do not export C
ABI symbols.

Unlike CUDA's build script, this crate's does not carry an explicit list of
adapter files: `build.rs` walks `KERNEL_INPUT_TREES` and compiles each file
whose extension its tree names, so a new adapter is picked up by being placed
here. Those trees are not all read the same way:

| tree | hashed / watched | `mcc` compiles |
|---|---|---|
| `adapters` | top level | `*.mu` |
| `kernels/custom` | top level | nothing — `.cuh` headers, pulled in by `#include` |
| `kernels/asm` | top level | nothing — `llvm-mc` and `lld` take it |
| `kernels/mutlass_fmha` | **recursive** | `*.mu`, with the vendor's flags |

The last row is the one exception to this crate's own build rules. Its sources
are four levels deep, which is why that tree alone is walked recursively for the
hash, the rerun watch and the compile; and its translation units are compiled
with **MATE's** device flags rather than the repo's, because the frozen kernels
were rendered and measured with them. `VENDOR_FMHA_FLAGS` in `build.rs` holds the
list and the reasons.

A file stem must still be unique across the trees that are compiled, because
objects are named after it.

Compilation of `mutlass_fa_adapter.mu` is tied to the vendored FMHA being built
at all — `only_with_vendor_fmha` in the tree table — so an mp_22 build does not
carry an object whose only content is a reference to an mp_31 kernel.

Every adapter is compiled with an explicit `-x musa` rather than leaving the
language to the extension. Host-only translation units are the normal case for
an adapter, and they take the same flag.

`qint8_decode_adapter.mu` uses the GEMM ABI with INT8 operands and padded FP16
row and column scales. Inside a W8A8 decode scope it selects NN for a prepared
KN gate/up copy and NT for NK projections; encode and shapes outside
`1 <= M <= 32`, `N % 128 == 0`, `K % 64 == 0` keep `qint8_gemm_adapter.mu`'s
convolution tiles. The quantized scope is separate from the BF16 decode scope,
so an INT8 model's BF16 fallbacks keep their existing dispatch.

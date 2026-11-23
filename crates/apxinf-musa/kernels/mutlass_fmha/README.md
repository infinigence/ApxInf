# mp_31 D256 flash attention, vendored from MATE

Two frozen instantiations of MATE's MUTLASS FMHA, plus the header subtrees they
compile against. They are the whole of `attention::mqa_bf16` on mp_31, and they
are what replaced the torch eager bridge on that target.

`build.rs` compiles each `.mu` with **MATE's own device flags**, not this repo's;
`adapters/mutlass_fa_adapter.mu` supplies the metadata and the LSE buffer and
launches them. No MATE install, no `tvm_ffi`, no JIT at build or run time.

## Why there is no small artifact here

`kernels/asm/` could vendor three rendered `.asm` files and be done, because
MP22 has an assembly path. mp_31 does not, for this shape:

* MATE's published mubin map carries 16 attention kernels and all of them are
  head dimension 128 or 192x128. `_check_valid_asm_input` refuses 256 outright
  (`headdim_qk == headdim_v and headdim_qk <= 128`, or `192`/`128`), so nothing
  is fetched and nothing is refused for being absent.
* The only d256 route MATE has on this target is the MUTLASS template in
  `mate/jit/attention/fmha/fmha_fwd.py`, which renders a `.mu` and compiles it
  with `mcc` + ninja on first call. Its `CONFIG_TABLE` lists `(256, 256)`.

So "vendor the used part" means the render and its include closure, not a
binary. The closure is small in the part that is MATE's and large in the part
that is its GEMM template library: 22 MATE headers, 196 MUTLASS headers.

## Provenance

| | |
|---|---|
| MATE | public tag `v0.2.7` = `e5d73e914fcb30aab915a12fae146d9d26357908`, `include/mate/{attention/fmha,common}` |
| MUTLASS | public commit `78b349514831ed54a797b717f2fc0534646acff2`, `include/{mutlass,mute}` — the `3rdparty/mutlass` gitlink MATE `v0.2.7` pins |
| license | MATE Apache-2.0, MUTLASS (Moore Threads) BSD-3-Clause — `LICENSES/`, both byte-identical to the public repositories' own |
| toolchain | mcc 5.2.0, `--offload-arch=mp_31` |

All 218 headers are byte-identical to those two public trees. `PINS.md` states
the revision-pinned recipe and the aggregate fingerprint; `tools/verify_pins.sh`
re-derives both from a pair of public checkouts, so "these came from a public
tag" is a check rather than a claim.

The headers were originally lifted from the MATE 0.2.7 **wheel**
(the MATE release wheel, build commit
`f4342ab7…`), and the wheel's `mate/data/` holds the same content as the public
tag. Either reproduces this tree; the tag is the one a reader without a wheel
can audit. The two `.mu` renders below still come from the wheel, because they
are JIT output and no checkout carries them.

Pin the tag, not a branch. Neither MATE 0.2.5 nor 0.3.0 vendors the same
headers as the files here, and public HEAD does not reproduce this tree either:
at `v0.2.8` with mutlass `435707c2`, five MATE FMHA headers differ
(`epilogue_fwd.hpp` and `fmha_combine.hpp` gain `HasMaxLogits`, `metadata.hpp`
loses the merge-path helpers, `mp31_fmha_fwd_tme_warpspecialized.hpp` changes
its max-logits and swizzle arguments, `fmha_fwd_kernel.hpp` passes the new
tuple to the epilogue) and `mute/pointer_swizzle.hpp` changes
`iterator_traits` to `element_type`.

**Re-vendoring must `git add -f` the `builders/` directories.** Seven of the 196
MUTLASS headers live under `.../mutlass/{gemm,epilogue}/collective/builders/`,
and a common global gitignore rule — `**/build*/`, which is line 5 of this
checkout's `core.excludesFile` — matches any directory named `builders`. `git add`
skips them silently, and the tree then looks complete while the two
`collective_builder.hpp` files still `#include` them: a fresh clone cannot build
this FMHA at all. The checkout this tree landed from carried those seven only as
ignored files on disk, so the defect survived into the branch until it was found
by comparing the tracked count against the wheel.

Verify after any re-vendor: `git ls-files …/kernels/mutlass_fmha | wc -l` should
equal the wheel's `mate/data` count (226 for 0.2.7 as checked in), and
`git check-ignore -v` on one of the `.inl` files should print nothing.

| checked in | render sha256 | file sha256 | MATE config |
|---|---|---|---|
| `fmha_fwd/mp31_decode_packed_tile128.mu` | `e016a4cf11909883a83cf16088bc2c0cb60b22782fa7c0d78f4cb731aa5c42bb` | `466886471796fde6752a09cf11dc7acf4fb8f3b01be562120a2c538543c978e4` | `fmha_fwd_4386cd8f…` |
| `fmha_fwd/mp31_general_unpacked_tile192.mu` | `946c354ecb826ee8cba7660f38d347f7c30618a93215c23827f4253c02525c6f` | `41ea5a8739c9500a942359840ce36af0850b94ed50784c7d797c9d4e2bc61051` | `fmha_fwd_be3f5142…` |

The rendered originals are the two files under `~/.cache/mate/0.2.7/mp31/generated/`.
`tools/freeze_kernel.py` produced the checked-in files from them and is the only
transform; run it again rather than editing a `.mu` by hand.

## The two instantiations

| | `TileM` | consumers | `IsPackGQA` | PI0.5 shape | where |
|---|---:|---:|---|---|---|
| `mp31_decode_packed_tile128` | 128 | 2 | true | decode `10/532` | `crates/apxinf-model/src/pi05/model/blocks/bf16.rs` decode |
| `mp31_general_unpacked_tile192` | 192 | 3 | false | prefill `522/522` | same file, prefix pass |

MATE's tile rule (`fmha_utils.py::_get_fwd_kernel_config`, bf16, head dimension
256/256, `head_ratio` 8) packs GQA up to 256 query rows, buckets the packed row
count onto a tile of 16/32/64/128/192/256, and caps at 192. Two facts follow and
both are load-bearing:

* Every query length from 257 up lands on the **same** config, so the frozen
  prefill kernel covers any prompt length the model can produce.
* Only two of the instantiations that rule can express are frozen, so
  `kernels::attention::mp31_fa_variant` substitutes the unpacked tile for the
  packed one between 17 and 256 rows. PI0.5 never reaches that range; the
  substitution is licensed by `mutlass_tile_choice_does_not_move_the_bits`,
  which runs both variants on the same input and requires bit-identical output.

## What was changed, and what was not

`tools/fmha_fwd_host.cpp.in` is the host entry, and it is the only thing that
differs from the render:

1. The six `<tvm/...>` includes and `"op_utils.hpp"` are dropped, and
   `using namespace tvm;` with them. `op_utils.hpp` was their only consumer on
   this path: the FMHA headers include neither it nor `mate_utils.hpp`, and use
   no `MATE_*` or `TVM_FFI_*` symbol. That is what removes `mate/data/csrc` and
   `tvm_ffi` from the closure.
2. The 36-argument `ffi::Array<ffi::Any> fmha_fwd_<hash>(...)` entry becomes an
   eleven-argument `extern "C"` launcher taking the four tensors, the LSE
   buffer, the two `cu_seqlens` arrays, the two lengths and the stream.
3. Everything the render derived from `ffi::TensorView` accessors
   (`q.size(-2)`, `q.stride(-3)`, `cu_seqlens_q.value().size(0) - 1`, …) becomes
   a constant of the contract; the derivation is spelled out in the header
   comment of the launcher.
4. Dropped: `ffi::MUSADeviceGuard`, `query_device_multiprocessor_count()` and
   `mp_margin`, the `o_accum`/`lse_accum` `FromEnvAlloc`s, the `dry_run` branch,
   `b_rounded`, the metadata pointers (this instantiation's `HasMetadata` is
   false) and `TVMFFIEnvGetStream`.
5. `MATE_CHECK_MUSA_KERNEL_LAUNCH()` becomes a returned message, so the caller
   reads the reason from the return value rather than a second call.

**Everything else is the render's, byte for byte** — in particular the whole
`constexpr` config block from `// FMHA config` to `TileScheduler>;`, the four
`using` aliases, and the shape of the scheduler, epilogue and mainloop
arguments. `tools/freeze_kernel.py` copies that block verbatim; comparing it back
against a render is a manual step, because the render is a file inside a MATE
wheel that this repository does not carry. Keep the wheel you rendered from if
you intend to check it.

What *is* checked automatically is narrower and aimed at the failure that
actually happened here: `crates/apxinf-musa/tests/vendor_tree.rs` asserts that no
file under this tree is matched by a gitignore rule. `bbc254e` had to add seven
`builders/` headers a global `**/build*/` rule had dropped, and nobody noticed
for two commits — a missing header is not a missing file until something tries
to `#include` it, and the error lands somewhere else.

Two of the drops are worth their own note:

* **`o_accum`/`lse_accum` were dead.** They exist for the split-KV
  `fmha_fwd_combine` step, and with `HasMetadata` false the epilogue's every
  `ptr_O_partial` path is inside `if constexpr (!SplitKV)`. Deleting them is
  what makes the launch allocate nothing, which is what makes it capturable.
* **The LSE buffer is not optional.** `epilogue_fwd.hpp` writes LSE from
  registers when the attention is *not* split, so `ptr_LSE` has to be a real
  `[8, tokens]` f32 buffer. The bridge discarded MATE's LSE and copied only the
  output, so nothing reads it — but a null there is an out-of-bounds write.
  `adapters/mutlass_fa_adapter.mu` owns it.

## Contract

The launcher is `tools/fmha_fwd_args.hpp` and
`adapters/mutlass_fa_adapter.mu`; this section is what they assume.

BF16, inference only, non-causal, no mask.

| | Q / out | K / V | `lse` | `cu_seqlens_q`, `cu_seqlens_k` |
|---|---|---|---|---|
| shape | bf16 `[tokens, 8, 256]` contiguous | bf16 `[key_tokens, 256]` contiguous | f32 `[8, tokens]` | device `uint32[2]` = `{0, n}` |

K and V are contiguous `[key_tokens, 256]`, which is how the model holds them; a
`[key_tokens, 1, 256]` view gives the same row and head stride, which is why the
bridge's reshape is not reproduced. `key_tokens` bounds the rows that take part
and the kernel reads no row past it, so a reserved prefix — NaNs included —
never reaches the softmax.

The scale is `1/sqrt(256)`. It is not a parameter: the tile shape and head
dimension are compile-time constants of the instantiation, and 256 is the only
head dimension these two kernels have.

Grid, block and shared memory all come from the instantiation:
`SingleTileScheduler::get_grid_shape` (a pure function of the scheduler
arguments — it takes the SM count and does not read it), `Kernel::MaxThreadsPerBlock`,
`Kernel::SharedStorageSize`. Nothing is passed from Rust except the tensors, the
metadata, the two lengths and the stream.

## Compilation

MATE's flags, from the `cuda_cflags` of the `build.ninja` MATE wrote:

```
-Od3 -O2 -DNDEBUG -fno-strict-aliasing -fno-signed-zeros
-fmusa-flush-denormals-to-zero
-mllvm -mtgpu-load-cluster-mutation=1 -mllvm --num-dwords-of-load-in-mutation=64
```

`build.rs` holds them as `VENDOR_FMHA_FLAGS` and applies them **only** to this
tree. The reasons are worth keeping: this repo has measured that recompiling a
translation unit moves an unrelated kernel's bits, so applying device-codegen
switches repo-wide would invalidate every recorded replay result at once; `-Od3`
conflicts with the repo's `-O3`; and the two `-mllvm` switches are validated for
this template and nothing else.

Each `.mu` is its own translation unit, as in MATE's build, which compiles one
`.mu` per shared object.

The tree is compiled only when the target is mp_31 **and** the toolkit is at
least 5.2 — the version MATE produced the render with, and below which it passes
different device flags. Outside that, `attention::mqa_bf16` reports a kernel gap
by name.

## Regeneration

```bash
tools/freeze_kernel.py <rendered.mu> <launcher-name> <output.mu>
```

To freeze a config that is not here, render it first with the installed MATE
(the `.mu` appears under `~/.cache/mate/<version>/mp31/generated/` on the first
call that needs it), add its name to `tools/fmha_fwd_args.hpp`, its entry to
`kVariants` in `adapters/mutlass_fa_adapter.mu`, and its name to the enum in
`src/ffi/mutlass_fa.rs`. A new tile also needs a replay: which kernel ran is not
visible in the output.

Re-vendor with `cp -a` into these directories and audit with `diff -r`. Two
things keep that audit readable, and both live inside this crate: the file
`crates/apxinf-musa/.gitattributes` disables whitespace normalisation for
`kernels/mutlass_fmha/**` (so `git diff` shows no changes nobody made — it used
to be a line in the repository root's `.gitattributes`), and
`crates/apxinf-musa/tests/vendor_tree.rs` fails if any file here is matched by a
gitignore rule, which is how `bbc254e` lost seven `builders/` headers.

## Pruning

`mate/data/include/mate/{attention/fmha,common}` and `mutlass/include` are kept
**whole by subsystem** — the prune is by directory, not by reachability. A
config change re-renders the same template and resolves inside these
directories; a config that reached into `flat`, `gemm`, `mega_moe`, `msa` or
`flash_mla` would be a different vendor deliverable, not a new tile.

Dropped, with what it costs: `mate/data/csrc` (only `op_utils.hpp` was in the
closure, and it is gone), `tvm_ffi/include` (27 files), `mate/data/{aot,templates}`,
the other `mate/include` subsystems, and `mutlass/{tools,python,experimental}`.

The guard against a header that was needed and pruned is that a missing header
is a hard `mcc` error naming it. The guard against a needed header that was
*kept* but not re-vendored is that re-vendoring is `cp -a` of three directories
and `diff -r` is the audit.

## Measured

The device payload of each checked-in `.mu` is **bit-identical** to the object
MATE itself compiled — `.mt_fatbin` 68321 bytes `2a19cea5…` and 51945 bytes
`3ac6039f…` respectively.

| | |
|---|---|
| object size | 105184 / 88320 bytes (MATE's own: 751536 / 768408 — the difference is `tvm_ffi` and the `MATE_CHECK` error machinery) |
| device payload | identical to MATE's, both variants |

# MP22 D256 flash attention, vendored assembly

Three hand-written MP22 assembly kernels for PI0.5's `[total,8,256]` /
`[B,capacity,1,256]` BF16 non-causal attention. They are the whole of
`attention::mqa_bf16` **on an MP22 target**; on mp_31 the operator is served by
the vendored MUTLASS FMHA in `kernels/mutlass_fmha/`, because this assembly is
MP22 ISA and no other generation can assemble it.

`build.rs` assembles them with the toolkit's own `llvm-mc`/`lld` and embeds the
result in the binary; `adapters/mubin_fa_adapter.mu` loads and launches them.
**No muDNN checkout and no renderer are needed at build or run time** — these
files are the renderer's output, checked in.

## Provenance

Taken verbatim from MATE's `mate/flash_attention/asm/` as it stands in an
internal working tree, MATE 0.2.5. **That directory is not tracked there.**
The commit the working tree was on when the files were taken, `825890dd`
(*fa: add support for dense k/v, non-casual gqa8*, 2026-09-22), does not contain
it, and the three images were rendered on 2026-09-28/29 — after it.
So a commit hash is not a handle on these bytes; the SHA256 table below is, and
it is also what the upstream `asm/README.md` publishes. Nothing newer exists:
the directory is absent from every ref of that repository, and MATE's mubin asm
route refuses head dimension 256 outright, so there is no released variant to
upgrade to.

| file | SHA256 | rendered from |
|---|---|---|
| `fa_d256_mp22.asm` | `2d4b5818bc0e9d6f84a322fd3883056fb89f9c13abc0431bae873dacdd51395a` | `fa_d256_mp22.s`, `packed_decode=False split_kv=False` |
| `fa_d256_mp22_decode.asm` | `0a5285a6624075b9681d34f77686619d83f7dfb8fb7330f5c4fb3ae47064498c` | same, `packed_decode=True split_kv=False` |
| `fa_d256_mp22_splitkv.asm` | `8d13605463cd9dd624ae1dfeb32a45e1b84419a99bc178bbc1291cd8cd24b17a` | same, `packed_decode=True split_kv=True` |

The upstream renderer is muDNN's `ComputeAsmKern` at `6be23bf2` with
`asm_render` at `aa1fdfa5`, driven by `scripts/render_fa_d256_mp22.py` in the
same checkout.
Re-rendering is only needed to change the kernel, and it is not part of this
build.

The upstream `asm/README.md` still carries a `567ffa04…` hash for the decode
variant. That was accurate when it was written; a later scheduling change
("Decode V prefetch", 2026-09-28) superseded it without updating the table. The
hashes above are of the files actually in this directory.

## Contract

The launcher is `adapters/mubin_fa_adapter.mu`; this section is what it assumes.

Eleven parameters, in order, all passed by address:

```
out, q, k, v, cu_seqlens_q, seqused_k, max_seqlen_q, kv_capacity, heads, kv_heads, scale_log2
```

The first six are **robust pointers** — a `{uint64_t address, uint64_t bytes}`
pair, not a bare pointer. `heads` is 8 and `kv_heads` is 1; the kernel does not
read them, but they are part of the ABI and must still be passed. `scale_log2`
is `softmax_scale * log2(e)`, because the kernel exponentiates with `exp2`.

| | Q / out | K / V | `cu_seqlens_q` | `seqused_k` |
|---|---|---|---|---|
| shape | bf16 `[total_q, 8, 256]` | bf16 `[batch, capacity, 1, 256]` | int32 `[batch+1]` | int32 `[batch]` |

`seqused_k` is the number of leading rows of each KV sequence that participate.
Rows past it are excluded from both softmax and PV, so a reserved prefix can be
passed whole — which is what `cache::reserve_prefix_bf16` leaves behind. Invalid
tails, NaNs included, never reach the softmax.

All three variants export the same symbol, `mate_fa_d256_mp22`; **which kernel
runs is decided by which file was loaded**, not by the name looked up. They
differ in more than grid shape:

| variant | grid | block | when |
|---|---|---|---|
| general | `(ceil(total_q/32), 8, batch)` | 128 | everything else |
| decode | `(ceil(total_q*8/32), 4, batch)` | 128 | `batch==1 && total_q == max_seqlen_q == 10` |
| split-KV | `(ceil(total_q*8/32), 4, batch)` + merge | 128 | the above, and `512 <= capacity <= 640` |

The decode layout flattens the contiguous token and head dimensions into
`total_q * 8` MMA rows and splits D=256 into four 64-column output partitions.
Split-KV additionally cuts the KV axis into four and writes unnormalized
partials plus per-split `(max, denominator)` into a `330240`-byte FP32
workspace — `4*80*256` floats of partial output followed by `4*80*2` of
statistics — which a second kernel then merges. **That merge kernel is not in
these files**: it is device code the launcher owns, and the launcher is where it
must stay consistent with the layout above. Its `80` is the packed row count,
so it is another place the `total_q == 10` bound is baked in.

Both decode variants are built for `total_q == 10`. Any other query length is
served correctly by the general variant, which costs 2.3x more on that shape
(0.1775 ms against 0.07824 ms) — `kernels/attention.rs` logs when it falls back,
because a slower kernel a profile cannot attribute to anything is exactly the
kind of thing that costs an afternoon later.

## Measured

M1000 / MP22, BF16, non-causal, preallocated output, device-event medians from
the upstream `asm/README.md` (20 warmups / 100 launches):

| shape | general | decode | split-KV |
|---|---:|---:|---:|
| encoder `522/522/522` | 0.725 ms | — | — |
| decode `10/532/532` | 0.196 ms | 0.1247 ms | **0.07824 ms** |

This backend measured its own three variants when the port landed: all three variants reproduce the recorded bridge tensors within
**1.000 bf16 ulp at full scale**, cosine ≥ 0.9999978, with a negative control
separating by ≥412x. The split-KV variant is the closest of the three to the
reference (cosine 0.999999762).

## BF16 decode GEMM

The twelve `asm_bf16bf16gemm_{nt,nn}_*.asm` images are vendored verbatim
from gemm-mate, base revision
`3b2909364fc48450f8f004dde3902b1cf67aadbf` plus its local NN additions.
License: `LICENSE.gemm-mate` (Apache-2.0). No MATE package, torch extension,
renderer or external checkout is required to build or run these images.
`asm_gemm_adapter.mu` adapts the upstream selector and full 34-argument
muDNN epilogue ABI to driver modules, retaining the upstream default tile
selection for decode. Registered NN encode weights use the measured specializations
below (without an environment override).

NT consumes `[M,K] @ [N,K]^T`; NN consumes `[M,K] @ [K,N]`. Both write
BF16 with FP32 accumulation, alpha=1, beta=gamma=0, batch=1, K divisible
by 32, and signed int32 byte bounds. Grid is `(ceil(N/tile_n),ceil(M/tile_m),1)`.
The seven pointer/int32-size/int32-zero triples precede M,N,K,lda,ldb,ldc,
alpha,beta,gamma,batch and three batch strides. `ldb` is K for NT, N for NN.
Modules are cached per host thread and driver context, retained for graph
lifetime, loaded only outside capture. Eager, warmup and capture use the
same selector and engine stream. Immutable NT weights are transposed once
by the native adapter; NN retains original KN storage.

| file | SHA256 |
|---|---|
| `asm_bf16bf16gemm_nt_tc_128_128x128_epilogue.asm` | `2cceaac9e977bf99d30b51d84e7a3db0aea05479565ba23441b517e882938275` |
| `asm_bf16bf16gemm_nt_tc_128_128x64_epilogue.asm` | `8d80a14638bdd8c266342ae31bdb99997b1094ba0a435baee3b6af4c43505942` |
| `asm_bf16bf16gemm_nt_tc_128_32x128_epilogue.asm` | `eeec1257320f0a180690b74d70458212e2febeb25843780109fb56c4bad388ad` |
| `asm_bf16bf16gemm_nt_tc_256_256x128_epilogue.asm` | `288bc8922c09f6cc36dd6db5802a2b6cc1e8f13f8dabf65a646d36a6a33846c4` |
| `asm_bf16bf16gemm_nt_tc_256_256x64_epilogue.asm` | `909857723ebd2c94582f05b0572d49beb1526c3713775358591b9fcf863ef830` |
| `asm_bf16bf16gemm_nt_tc_512_256x256_epilogue.asm` | `01b0952df05132c917625102391bc4c38827b1c322e90d87ecbb057909cb71be` |
| `asm_bf16bf16gemm_nn_tc_128_128x128_epilogue.asm` | `2d1857817dc452f03881177a307de793be85868f3569c7938a4fface3cb0bf2b` |
| `asm_bf16bf16gemm_nn_tc_128_128x64_epilogue.asm` | `a34df023492cf9fc0e10fd870acc1b68a49192cd4bc18f711e6e1fb96671f3f2` |
| `asm_bf16bf16gemm_nn_tc_128_32x128_epilogue.asm` | `2999179d090390196a98cd37d061cd6c7e111654fa9f6af73ff7a95b6d3f62b3` |
| `asm_bf16bf16gemm_nn_tc_256_256x128_epilogue.asm` | `7f393e3af2bd3d6e0e9e2c752290c0388514c27a53724a0ce9bb24042dd95e4e` |
| `asm_bf16bf16gemm_nn_tc_256_256x64_epilogue.asm` | `fd2bb57845679c5073cd01ba347c36e29c00f56442aebdf16b28fa2d5ffa10e8` |
| `asm_bf16bf16gemm_nn_tc_512_256x256_epilogue.asm` | `1964498a9d55d3b87170204fa269d214a80766c4b25a15be79de9571c94ac782` |

## BF16 NN encode GEMM

Four additional `asm_bf16bf16gemm_nn_tc_*_epilogue_{normal,wide}_{mfast,nfast}.asm` images specialize registered immutable NN
weights at M=522. All use BK=32, BF16 inputs/output, FP32 accumulation and
MMA.323216. Unsupported shapes, unregistered weights and non-MP22 devices
retain the vendor route. No change is made to decode registration/layouts.

| N | K | tile | loading | grid order | image |
|---:|---:|---|---|---|---|
| 2560 | 2048 | 128×128 | wide | M-fast | `asm_bf16bf16gemm_nn_tc_128_128x128_epilogue_wide_mfast` |
| 2048 | 2048 | 128×256 | wide | M-fast | `asm_bf16bf16gemm_nn_tc_256_128x256_epilogue_wide_mfast` |
| 16384 or 32768 | 2048 | 128×128 | normal | M-fast | `asm_bf16bf16gemm_nn_tc_128_128x128_epilogue_normal_mfast` |
| 2048 | 16384 | 256×128 | wide | N-fast | `asm_bf16bf16gemm_nn_tc_256_256x128_epilogue_wide_nfast` |

Names follow the decode template convention:
`asm_{output_dtype}{mma_dtype}gemm_{layout}_tc_{threads}_{tile_m}x{tile_n}_epilogue`,
with `_{normal,wide}_{mfast,nfast}` appended to distinguish encode variants.
Here `bf16bf16` denotes BF16 output and MMA input, `nn` denotes both operand
layouts, and the number after `tc` is the thread count (not the tile size).
The partitioned 128×256 kernel has 256 threads despite its 128-thread base
partition. File stems, ELF symbols and adapter lookup strings are identical.
Offline generation must set `<KERNEL_NAME>` before rendering, rather than
replace the rendered symbol with an experiment label. The naming correction
changes symbols only; instruction text and launch configuration are unchanged.

M-fast uses `(ceil(M/tile_m),ceil(N/tile_n),1)` and swaps the block-ID
registers in the assembly. The adapter carries grid order per image; an
M-fast image must not be launched with the old N-fast grid. The 128×256
image uses two independent 128-thread N partitions, private 16KiB shared
regions, local thread IDs and per-partition N origins. It duplicates A
loads rather than sharing A across partitions. These kernels use single
buffering; the rejected BK64/288/double-buffer experiments are not vendored.

The sources are rendered from muDNN ComputeAsmKern's
`musa_hgemm_128x128.s` and `musa_hgemm_256x128.s` (muDNN checkout
`706364dd5269aa753ad523370a162ba1904a90f6`), with trans_a=false,
trans_b=false, BF16 A/B/C, offline_reorder=false, online_reorder=false.
The 128×256 partition transformation and block-ID swaps are local changes.
Build/runtime needs only these rendered sources and the MUSA assembler,
not the external checkout or Python renderer.

PI0.5 registers the four language projections at load time using their
existing KN allocations; the context retains them without another weight
copy. `gemm::bf16` tries this path only for a measured shape and a registered
NN weight. Module loading still happens during eager warmup before capture.
The existing GeGLU composition and BF16 intermediate rounding are unchanged.

Two-seed M1000 microbenchmarks measured approximately 0.720ms QKV,
0.577ms attention output and 9.224ms combined gate/up (NN weights prepared
before timing). These are kernel measurements, not an end-to-end promise.
The maintained device regression test is
`encode_nn_reference_and_graph_replay`; it covers every dispatch shape,
registration fallback and changed-input graph replay against eager/vendor.

| file | SHA256 |
|---|---|
| `asm_bf16bf16gemm_nn_tc_128_128x128_epilogue_wide_mfast.asm` | `facb6ddfdd580ac7c965f5f63c8ee1b0bc04dd84490b3b6fb535887fa719e4bc` |
| `asm_bf16bf16gemm_nn_tc_256_128x256_epilogue_wide_mfast.asm` | `10a85540b35d8a29e6f359fd6daceba9d9c48014c9534559bf26a2bef6a4b02c` |
| `asm_bf16bf16gemm_nn_tc_128_128x128_epilogue_normal_mfast.asm` | `946d3b2072afd9a770348417d03fd54b79a2a20466c53259bf58e1816b74500b` |
| `asm_bf16bf16gemm_nn_tc_256_256x128_epilogue_wide_nfast.asm` | `be094113f5f0b0cfa87879d2ff2eb1a69ff37133f2bd85abe7b8b6a916863784` |

# INT8 (W8A8) GEMM tiles, vendored assembly

Six muDNN QY2 `s8mma` convolution tiles configured as a 1x1 GEMM: the whole of
`gemm::w8a8` on an MP22 target. `adapters/qint8_gemm_adapter.mu` loads and
launches them and owns the contract (67-argument convolution ABI in the
H=W=1 degenerate case, `A[m,k]` INT8, `B[n,k]` INT8 output-major, per-row and
per-column **FP16** scales with a 512-entry zero pad, `D[m,n]` **BF16**);
`src/kernels/gemm/qint8_tiles.rs` picks the tile. As with the two sections
above, `build.rs` assembles them with the toolkit's `llvm-mc`/`lld` and embeds
the result, and **no muDNN checkout and no renderer are needed at build or run
time**.

## Provenance

Rendered from upstream muDNN's `ComputeAsmKern` `s8mma` templates. The
renderer and its template tree live in the muDNN checkout the images were
rendered from (`kernels/qint8_mma_tiles/` there, not here); what is checked in
is the rendered output. Rendered with `io_dtype=int8`, `KRSC=true`,
`linear_act=true`, `ldma_ol_trans=true` and per-tile `online_reorder`, in the NN
layout. The templates carry a **local patch** the upstream does not: the
epilogue narrows the f32 accumulator to BF16 (the upstream's
`output_dtype=half` configuration packed FP16, which a second, memory-bound
pass then widened to BF16). The scales stay FP16 either way; an MP22 `FOP.MUL`
with a BF16 destination rejects an FP16 scale source, so the dequant is two
instructions -- an f32 in-place multiply, then an f32→bf16 narrowing move.
Re-rendering the upstream configuration reproduces the old FP16 files bitwise,
which is what pins the patch's blast radius to the epilogue.

| file | threads | tile | shared | SHA256 |
|---|---:|---|---:|---|
| `qint8_256x128.asm` | 256 | 256 × 128 × 64 | 24704 | `223a1cf05ae582bdf5b8696d8911b73c976225b43e8273e5783b7c48d6b8da0b` |
| `qint8_256x256.asm` | 512 | 256 × 256 × 64 | 65664 | `2569a94df6d935cb879b55f2409e4fea8b77c88f2474f7a56706640c96611e5e` |
| `qint8_128x128.asm` | 128 | 128 × 128 × 64 | 16512 | `2aa0fc064f3766b8b191a83ae5f56501f369575570570dd93a4684b8cac46319` |
| `qint8_256x64.asm` | 256 | 256 × 64 × 64 | 20608 | `bfee7d90b39ca6777b4e59f809eb7ad8b074433520b224f2600c2344bc3cee4e` |
| `qint8_32x128.asm` | 128 | 32 × 128 × 64 | 10368 | `deb90df04050744c582a8690be24ea899f08da128988e818a17a0f0e3e7375c4` |
| `qint8_32x64.asm` | 128 | 32 × 64 × 64 | 6272 | `10949d198711683db638df1d464154499d0c7ef7e904ad4cabcaf8547f474310` |

Re-rendering is: render the tile with `OUTPUT_DTYPE=bfloat16`, then apply the
two bf16-output patch passes to the rendered source. It is only needed to change
a tile, and it is not part of this build.

## INT8 decode NN/NT GEMM

`qint8_gemm_nn_32x128.asm` and `qint8_gemm_nt_32x128.asm` implement the same
layout choice the BF16 decode path makes: the concatenated gate/up uses KN
weights (NN), and QKV, attention output and down use NK weights (NT). Both
images use 128 threads, a 32x128x64 tile and 10240 bytes of shared memory;
their symbols are `apxinf_bf16s8gemm_nn_tc_128_32x128_epilogue` and
`apxinf_bf16s8gemm_nt_tc_128_32x128_epilogue`.

These are GEMM templates with INT8 operands, not the convolution templates the
`qint8_<tile>.asm` files above came from — hence the `gemm` in their names.
They are derived from ComputeAsmKern's `src/asm_kernel/musa_hgemm_32x128.s` at
revision `6be23bf22a711c86b173b52bca87b3da62e526e5`, source SHA256
`3ca68d1eb983024d460f161f05f8a79abbe5b87240257bad1ae91bdf13381732`. The
maintained exporter `../../aot/export_qint8_decode.py` applies four changes:

- use S32 accumulators for the template's INT8 MMA instructions;
- use eight 16-byte segments per row for INT8 NN shared-memory stores;
- convert accumulators to FP32 and multiply by the FP16 column and row scales;
- select BF16 narrowing from the output type, keeping a single rounding.

The ABI is the BF16 GEMM ABI documented above, with INT8 byte counts for A and B
and live FP16 scale pointer triples. Both scale vectors keep the 512-element zero
pad. Alpha is one, beta and gamma are zero, and the bias and C pointers are
null. `adapters/qint8_decode_adapter.mu` validates `1 <= M <= 32`,
`N % 128 == 0` and `K % 64 == 0`, loads modules before capture and caches
functions per thread and driver context. Unsupported shapes keep the NK
convolution tiles above. No transpose, scale conversion or module
initialization happens inside capture.

| file | SHA256 |
|---|---|
| `qint8_gemm_nn_32x128.asm` | `b2e43313181bd0aa6a7f0261980067d4b1efc771ca18f224fa21c0a2d026dd77` |
| `qint8_gemm_nt_32x128.asm` | `eebb1090683dd79cf38037a19d1d8015f560c36401738f15554cf7b98b570c46` |

Re-rendering needs the renderer's Python dependencies and the MUSA toolkit:

```bash
python crates/apxinf-musa/aot/export_qint8_decode.py \
  --renderer <path-to-ComputeAsmKern> \
  --out <path-to-output-dir>
```

The renderer reproduces the checked-in files byte for byte. What exercises the
images here is the whole model: `model_variant=int8_dynamic` on the reference
fixture scores the same cosine with and without the decode scope, which is the
end-to-end claim these tiles have to keep.

# Native MLX backend

This crate adapts the MLX **0.32.3** C++ API to owned Rust arrays and the ApxInf
public tensor/sampling boundary. Metal models use its native array API directly. It contains no Python interpreter or external model provider.
The `native` feature requires Apple Silicon macOS; the default empty feature set
does not discover or link an MLX SDK.

Set `MLX_ROOT` to one matching SDK distribution containing `include/mlx/` and
`lib/libmlx.dylib`. The bridge checks the header version at build time and the
linked runtime version when constructing a stream. A Python wheel may supply
these native assets; Python is not called by this crate. Keep `libjaccl.dylib`
and `mlx.metallib` with the same pinned distribution. Preserve Apple's license
when redistributing those assets.

```sh
export MLX_ROOT=/path/to/mlx
cargo test -p apxinf-mlx --features native -- --test-threads=1
```

These tests explicitly select MLX CPU streams. They verify ownership, view and
dtype behavior, pure compiled callbacks, changed state/index propagation,
callback failures, quantized primitives and greedy invalid-logit behavior.
They do not qualify Metal kernels or a complete model.

When a command-line tools installation selects an incompatible SDK, set
`SDKROOT` to a compatible SDK before building. Keep task-specific build artifacts
under `devlocal/<feat-name>/` according to the repository's artifact policy.
The bridge links directly instead of using a
wheel's CMake export, which can contain the wheel builder's absolute SDK paths.
Final executables must have a usable rpath to the pinned native libraries;
when integrating another binary, use its build/link configuration or an
explicit `DYLD_LIBRARY_PATH`, not an ambient unrelated MLX installation.

## Ownership and execution

`Array`, `Stream` and `Compiled` are intentionally not `Send` or `Sync`. A
logical stream owner uses MLX's thread-local default device stream so repeated
model construction does not create an unbounded series of global native queues.
Arrays from separate logical owners cannot be mixed without explicit transfer.
MLX owns native graph and buffer dependencies; Rust retains each handle and its
logical execution owner. No raw CUDA pointer representation is reused.

`Compiled::new` owns a Rust tracing callback through the C ABI. Its result is a
fixed-length list of arrays; all changing state must appear in its inputs and
outputs. `call` constructs lazy results. `call_and_eval`, or `Stream::eval` on
the complete output/state set, completes them before a model publishes new
request state. `synchronize` alone does not evaluate a lazy expression.
`Compiled::prepare` evaluates one profile twice and rejects a replay that invokes
the trace callback again. Family preparation can use it to verify compiler
availability and exact-profile reuse. It does not observe Metal pipeline JIT.
Constructing a compiled callable rejects any present `MLX_DISABLE_COMPILE`
environment variable, even `MLX_DISABLE_COMPILE=0`, matching MLX's presence rule.
The bridge never changes the process-wide compile mode to force availability.
Callback errors and panics become Rust errors, and C++ exceptions are contained
inside the bridge. A model still owns transaction boundaries and invalidation.
The pinned [public C++ compile overload](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/compile.cpp#L1237-L1264)
owns the captured callable through a shared pointer whose deleter erases its
entry from the compiler cache of the thread that created it. Dropping a Rust
`Compiled` uses that ownership path;
the bridge does not clear other functions' compiler caches.

`Stream::counters()` snapshots cumulative trace callbacks, explicit upload and
download bytes, and evaluation boundaries (including downloads). Cloned streams
share counters; separate logical streams do not. Byte counters report bridge
host IO, not driver transfers. `memory_stats`, `reset_peak_memory` and
`clear_cache` observe or change the process-wide MLX allocator. Synchronize before
comparing release observations and run memory qualification in an isolated
process. The ignored `metal_memory` integration executable checks repeated
compiled-instance release and pool reuse when run under the shared Metal lock.

Native `Array` operations use MLX semantics, including broadcasting. Metal
models compose them directly and own their rounding, fusion and compilation
recipes. `MlxBackend` is an adapter for the existing public `Backend`/sampling
interface; it does not introduce a shared multi-backend operator contract.
Its legacy pointwise/norm and cached-attention methods retain explicit F32
intermediates and output casts. They are separate from the optimized models'
native attention and fused kernels.

`MlxBackend::array` checks storage owner, device, dtype, element extent and stream,
then reconciles a public `Tensor::reshape` with the native array's geometry.
`from_array` guarantees a dense row-major public tensor. Internal MLX index and
mask dtypes remain private to arrays. Primitive results do not update aliases.

The categorical sampler currently supports greedy selection without penalties
or requested log-probability. It normalizes NaN/infinity according to the core
contract, selects on device, and reads one token scalar. Other sampling options,
normal generation and CUDA-style capture explicitly return unsupported.

## Checkpoint arrays

`load_safetensors` reads the tensors that a name filter keeps from one or more
SafeTensors files (for example the list from
`apxinf_loader::safetensors::safetensors_files`) into evaluated arrays. Other
tensors are never read. MLX reads on its CPU device; there is no host staging
copy.
Native checkpoint loading requires the pinned MLX 0.32.3 runtime. Its CPU lazy
reader propagates payload-read exceptions; 0.31.2 can report successful
evaluation after a read fails. The native CPU regression truncates a fixture
after header validation and requires a filename-labelled load error without
publishing arrays or recording successful upload bytes.

This covers MLX-format affine checkpoints, whose packed weights are U32 and
have no core `DType`. Duplicate names across files, tensors without a bridge
dtype and truncated files fail with the file and tensor name. Bytes read count
as uploads. The model family still validates every key, shape, dtype and
quantization parameter before use.

## Device architecture

A Metal `Stream` reads `MetalDeviceCaps` once from MLX's `gpu::device_info`:
the device name and the `architecture` string that MLX's own Metal dispatch
reads, for example `applegpu_g13s` on an M1 Pro and `applegpu_g16g` on an M4.
`generation` and `device_class` are its two digits and final letter; any other
string leaves them unset. MLX sizes command buffers and GEMM tiles by that
letter, which is not a product tier: the M1 Pro reports `s`. `MLX_METAL_GPU_ARCH`
overrides the string for MLX and therefore here; use it only in tests.
`Stream::metal_caps()` returns `None` for a CPU stream. `MlxBackend::new` prints
the device and architecture to stderr, as `CudaBackend::new` prints its device.
The type holds hardware facts only; see the
[kernel policy rules](../../doc/mlx-backend.md#device-architecture-and-kernel-policy).

## Custom kernels

`MetalKernel` is the private-source building primitive. Its call is unsafe
because arbitrary source can perform out-of-bounds access. Public fusions belong
in this crate with complete shape/dtype/index guards and provenance. A source
kernel and compile success alone do not establish numerical or performance
acceptance; model and hardware qualification remain separate.

Kernel templates take dtype, integer and boolean arguments (`TemplateArg`).
An integer template is a specialization fact: each distinct value builds and
caches a separate Metal pipeline the first time it is used.

The Clef-derived fusions are `Q4TransposeDecode`, `PadRowsF16`,
`SplitF16Gemm`, `GatedDeltaRecurrence` and `HalfSplitRope` (g20), and
`Q4RowDecodeF16`, `F16Gemm`, `F16GateSwiglu`, `CausalDepthwiseConv`, the
wm1/wn2 tile of `SplitF16Gemm` and the `GatedDeltaLayout::EightLanes`
recurrence (C14). They decode affine Q4 group-64 weights to FP16 for one call
only; they keep no decoded weight resident. `F16GateSwiglu` includes MLX
0.32.3's Steel loader, MMA and transform headers as installed (MIT). The two
recurrence layouts change only the reduction order, not the operation.
Their tests compare each kernel with MLX operators or a host reference on
small shapes. They do not qualify the Clef model.

# Native MLX backend

This crate adapts the MLX **0.31.2** C++ API to owned Rust arrays and the ApxInf
backend contracts. It contains no Python interpreter or external model provider.
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
The pinned [public C++ compile overload](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/compile.cpp#L1131-L1157)
owns the captured callable through a shared pointer whose deleter erases its
compiler-cache entry. Dropping a Rust `Compiled` uses that ownership path;
the bridge does not clear other functions' compiler caches.

`Stream::counters()` snapshots cumulative trace callbacks, explicit upload and
download bytes, and evaluation boundaries (including downloads). Cloned streams
share counters; separate logical streams do not. Byte counters report bridge
host IO, not driver transfers. `memory_stats`, `reset_peak_memory` and
`clear_cache` observe or change the process-wide MLX allocator. Synchronize before
comparing release observations and run memory qualification in an isolated
process. The ignored `metal_memory` integration executable checks repeated
compiled-instance release and pool reuse when run under the shared Metal lock.

Low-level array operations use MLX semantics, including normal broadcasting.
The `MlxBackend` interface applies ApxInf portable precision contracts instead:
F32 pointwise/norm calculations followed by the declared output cast and explicit
attention intermediate rounding. Portable attention handles GQA, causal offsets,
additive masks and all-masked rows. Family implementations may compose ordered
low-level casts for a different documented checkpoint rounding recipe.

`MlxBackend::array` checks storage owner, device, dtype, element extent and stream,
then reconciles a public `Tensor::reshape` with the native array's geometry.
`from_array` guarantees a dense row-major public tensor. Internal MLX index and
mask dtypes remain private to arrays. Primitive results do not update aliases.

The categorical sampler currently supports greedy selection without penalties
or requested log-probability. It normalizes NaN/infinity according to the core
contract, selects on device, and reads one token scalar. Other sampling options,
normal generation and CUDA-style capture explicitly return unsupported.

## Custom kernels

`MetalKernel` is the private-source building primitive. Its call is unsafe
because arbitrary source can perform out-of-bounds access. Public fusions belong
in this crate with complete shape/dtype/index guards and provenance. A source
kernel and compile success alone do not establish numerical or performance
acceptance; model and hardware qualification remain separate.

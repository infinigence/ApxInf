# apxinf-hip

AMD ROCm/HIP implementation of `apxinf_core::Backend`, built beside `apxinf-cuda`
and `apxinf-cuda-new` and independent of both.

## Status

| | |
|---|---|
| Implemented | Every method `Backend` requires, plus sampling: `rms_norm`, `silu`, `add`, `mul`, `scale`, `matmul`, `rope`, `embedding`, `sdpa_decode`, `sdpa_prefill`, KV cache, transfers, synchronization |
| Not yet | The seven methods only Qwen3VL calls (`layer_norm`, `gelu_tanh`, `add_bias`, `rope_mrope`, `rope_vision_2d`, `vision_sdpa`, `concat_2d`); graph capture |
| Dtypes | F32 and BF16 storage, F32 arithmetic, one rounding at the store |
| Verified on | Radeon 8060S (`gfx1151`, wave32), ROCm 7.2.1 |
| Performance | Not optimized: eager, unfused, host-side sampling |

Numerics follow `apxinf-cuda`. Two deliberate differences:

- Where `apxinf-cuda` has undefined behaviour, this backend returns an error:
  `add`/`mul` on operands of different sizes, out-of-range embedding ids, and
  attention over cache positions nothing was written to.
- Attention keeps scores and probabilities in F32 and rounds the output once.
  `apxinf-cuda`'s materialized path rounds both to BF16, so this is the more
  precise of the two.

## Building

Without ROCm the crate still builds; `HipBackend::new` then reports that the
backend was built without ROCm. With ROCm:

```sh
export ROCM_PATH=/opt/rocm             # default
export APXINF_HIP_ARCH=gfx1151         # default: the first GPU rocm_agent_enumerator reports
cargo test -p apxinf-hip               # needs device hip:0
cargo build --features hip             # the workspace, with Device::Hip routed here
```

Use `device="hip:0"` from Python, or `Device::Hip(0)` from Rust.

Kernels are compiled for exactly one architecture. A device of a different one
is refused at `HipBackend::new` instead of failing later with "invalid device
function".

### Host notes

- **libxml2.** ROCm 7.2's device linker loads `libxml2.so.2`. Distributions
  that ship a newer libxml2 (Ubuntu 26.04, on infplane) fail inside `hipcc` with
  `lld: error while loading shared libraries: libxml2.so.2`. Put a compatible
  copy on `LD_LIBRARY_PATH` before building. On infplane:
  `export LD_LIBRARY_PATH=/home/wwxq/workspace/rocm72-libxml2-compat-noble-2.12.7/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`
- **cmake.** Not needed by this crate, but the workspace's tokenizer
  (`sentencepiece-sys`) builds with it, so `cargo build --workspace` fails
  without it whether or not `hip` is enabled.

## Layout

| Path | Contents |
|---|---|
| `kernels/apxinf_hip.hip` | All device code and the C ABI Rust calls. One translation unit |
| `build.rs` | `hipcc` → static library, or the stub build when ROCm is absent |
| `src/ffi.rs` | The C ABI, declared once for both builds |
| `src/runtime.rs` | Context, stream, allocations, `Tensor` ↔ device pointer |
| `src/ops.rs` | Operators: validation, then one launch |
| `src/kv_cache.rs` | KV cache and cached attention |
| `src/sampling.rs` | Token sampling and normal generation, delegated to the host implementations |
| `tests/ops.rs` | Every operator against `CpuBackend`, with tolerances fixed up front |

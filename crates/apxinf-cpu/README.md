# apxinf-cpu

Synchronous CPU backend. This crate owns CPU operators, concrete KV cache
storage, categorical sampling, normal generation and optional OpenBLAS linking.
It depends on `apxinf-core` for tensors and backend contracts, and does not depend
on model code. PI0.5 composition remains in `apxinf-model/src/cpu_pi05/`.

```rust
use apxinf_core::{Backend, Tensor};
use apxinf_cpu::CpuBackend;

let a = Tensor::from_f32(vec![1, 2], &[1.0, 2.0])?;
let b = Tensor::from_f32(vec![2, 1], &[3.0, 4.0])?;
let product = CpuBackend.matmul(&a, &b)?;
```

`CpuBackend` and `CpuKVCache` are exported here rather than from core.
Use `Backend::matmul` instead of the former `Tensor::matmul_cpu` method.
Matrix multiplication accepts matching batch dimensions and produces FP32.

## Build

Default builds use serial Rust GEMM without external libraries.
`cargo test -p apxinf-cpu --features openblas` uses an installed OpenBLAS library
on Linux or macOS. Set `OPENBLAS_LIB_DIR` when it is outside the linker's search
path; configure the platform runtime library search path for dynamic builds.
The root package, model crate and Python extension forward `openblas` to here.

OpenBLAS owns GEMM threading. Pthreads builds use `OPENBLAS_NUM_THREADS`; OpenMP
builds use `OMP_NUM_THREADS`. ApxInf adds no OpenMP directives or outer parallel
loop. Dense attention uses GEMM for QK-transpose and probability-times-V with
Rust softmax. Other CPU operators and host sampling use Rust loops.

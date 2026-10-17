//! FP16 gate GEMM with a SwiGLU epilogue, on MLX 0.32.3 Steel `BlockLoader`
//! and `BlockMMA`. Adapted from EngineTailor C14 `a13` (MIT); the Steel
//! headers are MLX's (MIT, Apple). See NOTICE.
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

/// `silu(x W^T) * up` for FP16 `x [1, P, K]`, `W [N, K]` and `up [1, P, N]`,
/// FP16 output `[1, P, N]`. The gate accumulates in FP32 over Steel's K order
/// (BK 32, 64 x 32 tiles, 2 x 2 SIMD groups) and is rounded to FP16 once;
/// `sigmoid`, `silu` and the product with `up` are evaluated in FP16.
/// `P` is a multiple of 64; every row, padding included, is computed.
pub struct F16GateSwiglu {
    kernel: MetalKernel,
}

impl F16GateSwiglu {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_f16_gate_swiglu_steel",
                &["x", "weight", "up", "iterations"],
                &["out"],
                include_str!("../native/f16_gate_swiglu.metal"),
                include_str!("../native/steel_gate_swiglu.h"),
            )?,
        })
    }

    pub fn call(&self, x: &Array, weight: &Array, up: &Array) -> Result<Array> {
        let [1, padded, k] = *x.shape() else {
            return Err(Error::Contract("gate SwiGLU input must be [1, P, K]"));
        };
        let [n, wk] = *weight.shape() else {
            return Err(Error::Contract("gate SwiGLU weight must be [N, K]"));
        };
        if wk != k
            || up.shape() != [1, padded, n]
            || [x, weight, up].iter().any(|a| a.dtype() != MlxDType::F16)
            || padded == 0
            || padded % 64 != 0
            || k == 0
            || k % 32 != 0
            || n == 0
            || n % 32 != 0
            || [(padded, k), (n, k), (padded, n)].iter().any(|&(r, c)| {
                r.checked_mul(c)
                    .is_none_or(|extent| extent > i32::MAX as usize)
            })
        {
            return Err(Error::Contract(
                "gate SwiGLU requires FP16 [1, P, K], [N, K] and [1, P, N] with P a multiple \
                 of 64, K and N multiples of 32 and extents within i32",
            ));
        }
        let iterations = Array::from_i32(x.stream(), &[1], &[(k / 32) as i32])?;
        // SAFETY: each threadgroup of 4 SIMD groups owns the 64 x 32 output
        // tile at rows `64 * group.y < P` and columns `32 * group.x < N`. The
        // A loader reads rows `c_row .. c_row + 64` of x and the B loader rows
        // `c_col .. c_col + 32` of the weight, K / 32 tiles of 32 columns each,
        // so every load is inside the guarded extents. The epilogue reads `up`
        // and writes `out` at the same `[64, 32]` tile with row stride N; tiles
        // do not overlap. `iterations` is a one-element I32 array equal to K / 32.
        let out = unsafe {
            self.kernel.call(
                &[x.clone(), weight.clone(), up.clone(), iterations],
                &[(&[1, padded, n], MlxDType::F16)],
                [n / 32 * 32, padded / 64 * 4, 1],
                [32, 4, 1],
                &[
                    ("K", TemplateArg::Int(k as i32)),
                    ("N", TemplateArg::Int(n as i32)),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

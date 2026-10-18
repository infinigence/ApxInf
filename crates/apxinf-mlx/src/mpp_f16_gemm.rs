//! FP16 projection GEMMs on Metal 4 tensors and MetalPerformancePrimitives
//! `matmul2d` (EngineTailor M4 frontier m4a012's projections, from m4a005).
//! Adapted from EngineTailor (MIT); see NOTICE.
use crate::q4_f16::SplitF16Gemm;
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

/// Threads per 64 x 64 tile: four SIMD groups share one `matmul2d`.
const THREADS: usize = 128;

/// The shared shape check plus `matmul2d`'s 64-wide K slices.
fn check(x: &Array, tokens: usize, weight: &Array) -> Result<(usize, usize, usize, usize)> {
    let shapes = SplitF16Gemm::check(x, tokens, weight)?;
    if shapes.1 % 64 != 0 {
        return Err(Error::Contract("MPP projection K must be a multiple of 64"));
    }
    Ok(shapes)
}

/// FP16 `[1, P, K] x [K, N]` GEMM with FP32 accumulation and BF16 output, in
/// two column halves of width `NW`, as [`SplitF16Gemm`] but with one 64 x 64 x
/// 64 `matmul2d` per tile. [`low`](Self::low) computes columns `[0, NW)`;
/// [`high`](Self::high) computes `[NW, 2 NW)` and copies the low result into
/// one `[1, T, 2 NW]` output. Rows at and after `T` are padding and are never
/// stored. MLX compiles the kernels at their first evaluation, so a device or
/// OS without Metal 4 tensor support fails there, not in `new`.
pub struct SplitMppF16Gemm {
    low: MetalKernel,
    high: MetalKernel,
}

impl SplitMppF16Gemm {
    pub fn new(stream: &Stream) -> Result<Self> {
        let header = include_str!("../native/metal_tensor_mpp.h");
        let body = include_str!("../native/mpp_f16_gemm_bf16.metal");
        Ok(Self {
            low: MetalKernel::new(
                stream,
                "apxinf_mpp_f16_gemm_bf16_low",
                &["x", "weight"],
                &["out"],
                &format!("    constexpr int WS = N, OFF = 0;\n{body}"),
                header,
            )?,
            high: MetalKernel::new(
                stream,
                "apxinf_mpp_f16_gemm_bf16_high",
                &["weight", "lo", "x"],
                &["out"],
                &format!(
                    "{}    constexpr int WS = NW, OFF = NW;\n{body}",
                    include_str!("../native/mpp_f16_gemm_bf16_copy_low.metal")
                ),
                header,
            )?,
        })
    }

    pub fn low(&self, x: &Array, tokens: usize, weight: &Array) -> Result<Array> {
        let (padded, k, half, _) = check(x, tokens, weight)?;
        // SAFETY: each 128-thread group owns the 64 x 64 output tile at rows
        // `64 * group.y < P` and columns `64 * group.x < NW`. `matmul2d`
        // reads rows < P of x and rows < K, columns < NW of the weight in
        // 64-wide K slices, K a multiple of 64. Stores are limited to rows < T
        // and columns < NW of `[1, T, NW]`; tiles do not overlap.
        let out = unsafe {
            self.low.call(
                &[x.clone(), weight.clone()],
                &[(&[1, tokens, half], MlxDType::BF16)],
                [half / 64 * THREADS, padded / 64, 1],
                [THREADS, 1, 1],
                &[
                    ("T", TemplateArg::Int(tokens as i32)),
                    ("K", TemplateArg::Int(k as i32)),
                    ("N", TemplateArg::Int(half as i32)),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }

    pub fn high(&self, weight: &Array, low: &Array, x: &Array, tokens: usize) -> Result<Array> {
        let (padded, k, half, full) = check(x, tokens, weight)?;
        if low.shape() != [1, tokens, half] || low.dtype() != MlxDType::BF16 {
            return Err(Error::Contract(
                "MPP projection high half requires the BF16 [1, T, NW] low result",
            ));
        }
        // SAFETY: as in `low`, with weight stride NW and stores at columns
        // `NW + col < 2 NW`. Each group also copies 16-byte chunks of rows < T
        // of its 64 low columns into columns `[0, NW)` (8 threads per row, 16
        // rows per step); copy and GEMM stores write disjoint columns, and
        // `lo` is only read.
        let out = unsafe {
            self.high.call(
                &[weight.clone(), low.clone(), x.clone()],
                &[(&[1, tokens, full], MlxDType::BF16)],
                [half / 64 * THREADS, padded / 64, 1],
                [THREADS, 1, 1],
                &[
                    ("T", TemplateArg::Int(tokens as i32)),
                    ("K", TemplateArg::Int(k as i32)),
                    ("NW", TemplateArg::Int(half as i32)),
                    ("N", TemplateArg::Int(full as i32)),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

/// FP16 `[1, P, K] x [K, N]` GEMM with FP32 accumulation and BF16 output
/// `[1, T, N]` in one pass of 64 x 64 x 64 `matmul2d` tiles (m4a012's k and v
/// projections). Rows at and after `T` are padding and are never stored.
pub struct MppF16Gemm {
    kernel: MetalKernel,
}

impl MppF16Gemm {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_mpp_f16_gemm_bf16",
                &["x", "weight"],
                &["out"],
                &format!(
                    "    constexpr int WS = N, OFF = 0;\n{}",
                    include_str!("../native/mpp_f16_gemm_bf16.metal")
                ),
                include_str!("../native/metal_tensor_mpp.h"),
            )?,
        })
    }

    pub fn call(&self, x: &Array, tokens: usize, weight: &Array) -> Result<Array> {
        let (padded, k, n, _) = check(x, tokens, weight)?;
        // SAFETY: as in `SplitMppF16Gemm::low` with N columns; the shared
        // check also bounds `[T, 2 N]`.
        let out = unsafe {
            self.kernel.call(
                &[x.clone(), weight.clone()],
                &[(&[1, tokens, n], MlxDType::BF16)],
                [n / 64 * THREADS, padded / 64, 1],
                [THREADS, 1, 1],
                &[
                    ("T", TemplateArg::Int(tokens as i32)),
                    ("K", TemplateArg::Int(k as i32)),
                    ("N", TemplateArg::Int(n as i32)),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

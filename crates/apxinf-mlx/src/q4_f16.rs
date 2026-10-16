//! Per-call FP16 decode of affine Q4 weights and the FP16 GEMM that consumes it.
//! Adapted from EngineTailor g20 (MIT); see NOTICE. The GEMM fragment layout
//! follows MLX Steel `mma.h` (MIT, Apple).
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

fn int(value: usize, what: &'static str) -> Result<TemplateArg> {
    i32::try_from(value)
        .map(TemplateArg::Int)
        .map_err(|_| Error::Contract(what))
}

/// Where the affine expression `scale * q + bias` is evaluated before the
/// result is stored as FP16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q4DecodeArithmetic {
    /// Scale and bias are rounded to FP16 first.
    F16,
    /// The expression is evaluated in BF16 and widened to FP16.
    Bf16,
}

/// Decodes rows `[row0, row0 + rows)` of an affine Q4 group-64 weight
/// `[N, K]` (packed U32 `[N, K / 8]`, BF16 scales and biases `[N, K / 64]`)
/// into a transposed FP16 matrix `[K, rows]`. The result lives only as long
/// as its consumer; nothing is cached.
pub struct Q4TransposeDecode {
    kernel: MetalKernel,
}

impl Q4TransposeDecode {
    pub fn new(stream: &Stream, arithmetic: Q4DecodeArithmetic) -> Result<Self> {
        let (name, source) = match arithmetic {
            Q4DecodeArithmetic::F16 => (
                "apxinf_q4g64_transpose_decode_f16",
                include_str!("../native/q4_transpose_decode_f16.metal"),
            ),
            Q4DecodeArithmetic::Bf16 => (
                "apxinf_q4g64_transpose_decode_bf16",
                include_str!("../native/q4_transpose_decode_bf16.metal"),
            ),
        };
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                name,
                &["weight", "scales", "biases"],
                &["out"],
                source,
                "",
            )?,
        })
    }

    pub fn call(
        &self,
        weight: &Array,
        scales: &Array,
        biases: &Array,
        row0: usize,
        rows: usize,
    ) -> Result<Array> {
        let [total, words] = *weight.shape() else {
            return Err(Error::Contract("Q4 decode weight must be rank 2"));
        };
        let k = words * 8;
        if weight.dtype() != MlxDType::U32
            || k == 0
            || k % 64 != 0
            || [scales, biases]
                .iter()
                .any(|a| a.shape() != [total, k / 64] || a.dtype() != MlxDType::BF16)
            || rows == 0
            || rows % 32 != 0
            || row0.checked_add(rows).map_or(true, |end| end > total)
        {
            return Err(Error::Contract(
                "Q4 decode requires U32 [N, K/8] with BF16 [N, K/64] scales and biases, \
                 K a multiple of 64, and a row range of whole 32-row blocks",
            ));
        }
        // SAFETY: each 256-thread group decodes a 32-row by 64-column block:
        // thread `tid` reads packed word `tid % 8` of row `R0 + n0 + tid / 8`,
        // with n0 < rows and k0 < K checked above, so every weight, scale and
        // bias read is inside `[total, ...]`. Each thread writes two FP16
        // vectors at `(k0 + col) * rows + n0 + 4 * (tid % 8)`, inside `[K, rows]`
        // because rows is a multiple of 32. Stores do not overlap. The native
        // facility makes inputs row-contiguous and checks stream ownership.
        let out = unsafe {
            self.kernel.call(
                &[weight.clone(), scales.clone(), biases.clone()],
                &[(&[k, rows], MlxDType::F16)],
                [k / 64 * 256, rows / 32, 1],
                [256, 1, 1],
                &[
                    ("N", int(rows, "Q4 decode rows exceed i32")?),
                    ("K", int(k, "Q4 decode width exceeds i32")?),
                    ("R0", int(row0, "Q4 decode row offset exceeds i32")?),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

/// Number of rows after padding `tokens` up to a multiple of 64.
pub fn padded_rows(tokens: usize) -> usize {
    tokens.div_ceil(64) * 64
}

/// Casts BF16 `[1, T, D]` to FP16 `[1, P, D]`, `P = padded_rows(T)`, with
/// zero rows after `T`.
pub struct PadRowsF16 {
    kernel: MetalKernel,
}

impl PadRowsF16 {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_pad_rows_f16",
                &["h"],
                &["out"],
                include_str!("../native/pad_rows_f16.metal"),
                "",
            )?,
        })
    }

    pub fn call(&self, h: &Array) -> Result<Array> {
        let [1, tokens, width] = *h.shape() else {
            return Err(Error::Contract("row padding requires one [1, T, D] batch"));
        };
        let padded = padded_rows(tokens);
        let total = padded
            .checked_mul(width)
            .filter(|&n| n > 0 && u32::try_from(n).is_ok())
            .ok_or(Error::Contract(
                "row padding extent is empty or exceeds u32",
            ))?;
        if h.dtype() != MlxDType::BF16 {
            return Err(Error::Contract("row padding requires BF16 input"));
        }
        // SAFETY: one thread per output element `i < P * D`; it reads `h[i]`
        // only when `i < T * D`, the input's exact extent, and writes `out[i]`.
        let out = unsafe {
            self.kernel.call(
                &[h.clone()],
                &[(&[1, padded, width], MlxDType::F16)],
                [total, 1, 1],
                [256, 1, 1],
                &[
                    ("T", int(tokens, "row padding tokens exceed i32")?),
                    ("D", int(width, "row padding width exceeds i32")?),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

/// FP16 `[1, P, K] x [K, N]` GEMM with FP32 accumulation and BF16 output, in
/// two column halves of width `NW`. [`low`](Self::low) computes columns
/// `[0, NW)`; [`high`](Self::high) computes `[NW, 2 NW)` and copies the low
/// result into one `[1, T, 2 NW]` output. Rows at and after `T` are padding
/// and are never stored. Callers that rely on MLX dispatch order build the
/// high-half weight after calling `low`.
pub struct SplitF16Gemm {
    low: MetalKernel,
    high: MetalKernel,
}

impl SplitF16Gemm {
    pub fn new(stream: &Stream) -> Result<Self> {
        let header = include_str!("../native/simdgroup_mma.h");
        Ok(Self {
            low: MetalKernel::new(
                stream,
                "apxinf_f16_gemm_bf16_tile64",
                &["x", "weight"],
                &["out"],
                include_str!("../native/f16_gemm_bf16.metal"),
                header,
            )?,
            high: MetalKernel::new(
                stream,
                "apxinf_f16_gemm_bf16_tile64_high",
                &["weight", "lo", "x"],
                &["out"],
                include_str!("../native/f16_gemm_bf16_high.metal"),
                header,
            )?,
        })
    }

    fn check(x: &Array, tokens: usize, weight: &Array) -> Result<(usize, usize, usize)> {
        let [1, padded, k] = *x.shape() else {
            return Err(Error::Contract("split GEMM input must be [1, P, K]"));
        };
        let [wk, half] = *weight.shape() else {
            return Err(Error::Contract("split GEMM weight must be [K, NW]"));
        };
        if x.dtype() != MlxDType::F16
            || weight.dtype() != MlxDType::F16
            || padded == 0
            || padded % 64 != 0
            || tokens == 0
            || tokens > padded
            || padded - tokens >= 64
            || k == 0
            || k % 16 != 0
            || wk != k
            || half == 0
            || half % 64 != 0
            || [padded, k, half * 2]
                .iter()
                .any(|&v| i32::try_from(v).is_err())
        {
            return Err(Error::Contract(
                "split GEMM requires FP16 [1, P, K] with P = T rounded up to 64, \
                 K a multiple of 16 and an FP16 [K, NW] weight with NW a multiple of 64",
            ));
        }
        Ok((padded, k, half))
    }

    pub fn low(&self, x: &Array, tokens: usize, weight: &Array) -> Result<Array> {
        let (padded, k, half) = Self::check(x, tokens, weight)?;
        // SAFETY: each 128-thread group owns a 64 x 64 output tile at rows
        // `64 * group.y < P` and columns `64 * group.x < NW`. A, B loads index
        // rows < P of x and rows < K, columns < NW of the weight, with K a
        // multiple of 16 so every half4 load stays inside a row. Stores are
        // limited to rows < T and columns < NW of `[1, T, NW]`; tiles do not
        // overlap.
        let out = unsafe {
            self.low.call(
                &[x.clone(), weight.clone()],
                &[(&[1, tokens, half], MlxDType::BF16)],
                [half / 64 * 128, padded / 64, 1],
                [128, 1, 1],
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
        let (padded, k, half) = Self::check(x, tokens, weight)?;
        if low.shape() != [1, tokens, half] || low.dtype() != MlxDType::BF16 {
            return Err(Error::Contract(
                "split GEMM high half requires the BF16 [1, T, NW] low result",
            ));
        }
        // SAFETY: as in `low`, with weight stride NW and stores at columns
        // `NW + col < 2 NW`. Each group also copies 16-byte chunks of rows < T
        // of its 64 low columns into columns `[0, NW)`; copy and GEMM stores
        // write disjoint columns, and `lo` is only read.
        let out = unsafe {
            self.high.call(
                &[weight.clone(), low.clone(), x.clone()],
                &[(&[1, tokens, half * 2], MlxDType::BF16)],
                [half / 64 * 128, padded / 64, 1],
                [128, 1, 1],
                &[
                    ("T", TemplateArg::Int(tokens as i32)),
                    ("K", TemplateArg::Int(k as i32)),
                    ("NW", TemplateArg::Int(half as i32)),
                    ("N", TemplateArg::Int((half * 2) as i32)),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

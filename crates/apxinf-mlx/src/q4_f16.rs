//! Per-call FP16 decode of affine Q4 weights and the FP16 GEMMs that consume it.
//! Adapted from EngineTailor g20 and C14 (MIT); see NOTICE. The GEMM fragment
//! layout follows MLX Steel `mma.h` (MIT, Apple).
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

fn int(value: usize, what: &'static str) -> Result<TemplateArg> {
    i32::try_from(value)
        .map(TemplateArg::Int)
        .map_err(|_| Error::Contract(what))
}

fn check_index_extent(
    rows: usize,
    columns: usize,
    max_index: usize,
    what: &'static str,
) -> Result<()> {
    rows.checked_mul(columns)
        .and_then(|extent| extent.checked_sub(1))
        .filter(|&last| last <= max_index)
        .ok_or(Error::Contract(what))?;
    Ok(())
}

fn check_q4_decode(total: usize, words: usize, row0: usize, rows: usize) -> Result<usize> {
    let width = words
        .checked_mul(8)
        .ok_or(Error::Contract("Q4 decode width overflows usize"))?;
    if width == 0 || width % 64 != 0 || rows == 0 || rows % 32 != 0 {
        return Err(Error::Contract(
            "Q4 decode requires K a multiple of 64 and a row range of whole 32-row blocks",
        ));
    }
    let end = row0
        .checked_add(rows)
        .filter(|&end| end <= total)
        .ok_or(Error::Contract("Q4 decode row range exceeds the weight"))?;
    int(rows, "Q4 decode rows exceed i32")?;
    int(width, "Q4 decode width exceeds i32")?;
    int(row0, "Q4 decode row offset exceeds i32")?;
    check_index_extent(
        end,
        words,
        u32::MAX as usize,
        "Q4 decode packed weight index exceeds u32",
    )?;
    check_index_extent(
        end,
        width / 64,
        u32::MAX as usize,
        "Q4 decode scale/bias index exceeds u32",
    )?;
    check_index_extent(
        width,
        rows,
        u32::MAX as usize,
        "Q4 decode output index before vector division exceeds u32",
    )?;
    Ok(width)
}

fn check_split_gemm(padded: usize, tokens: usize, width: usize, half: usize) -> Result<usize> {
    let full = half
        .checked_mul(2)
        .ok_or(Error::Contract("split GEMM output width overflows usize"))?;
    if padded == 0
        || padded % 64 != 0
        || tokens == 0
        || tokens > padded
        || padded - tokens >= 64
        || width == 0
        || width % 16 != 0
        || half == 0
        || half % 64 != 0
        || [padded, width, full]
            .iter()
            .any(|&value| i32::try_from(value).is_err())
    {
        return Err(Error::Contract(
            "split GEMM requires P = T rounded up to 64, K a multiple of 16 \
             and NW a multiple of 64",
        ));
    }
    for (rows, columns, what) in [
        (padded, width, "split GEMM input index exceeds i32"),
        (width, half, "split GEMM weight index exceeds i32"),
        (tokens, half, "split GEMM low output index exceeds i32"),
        (tokens, full, "split GEMM high output index exceeds i32"),
    ] {
        check_index_extent(rows, columns, i32::MAX as usize, what)?;
    }
    Ok(full)
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
        let k = check_q4_decode(total, words, row0, rows)?;
        if weight.dtype() != MlxDType::U32
            || [scales, biases]
                .iter()
                .any(|a| a.shape() != [total, k / 64] || a.dtype() != MlxDType::BF16)
        {
            return Err(Error::Contract(
                "Q4 decode requires U32 [N, K/8] with BF16 [N, K/64] scales and biases",
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
                    ("N", TemplateArg::Int(rows as i32)),
                    ("K", TemplateArg::Int(k as i32)),
                    ("R0", TemplateArg::Int(row0 as i32)),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

/// Decodes an affine Q4 group-64 weight `[N, K]` (packed U32 `[N, K / 8]`,
/// BF16 scales and biases `[N, K / 64]`) into row-major FP16 `[N, K]`, one
/// packed word per thread: `scale * q + bias` in FP16 with the scale and bias
/// rounded to FP16 first, the expression of MLX `dequantize` with FP16 scales.
pub struct Q4RowDecodeF16 {
    kernel: MetalKernel,
}

impl Q4RowDecodeF16 {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_q4g64_row_decode_f16",
                &["weight", "scales", "biases"],
                &["out"],
                include_str!("../native/q4_row_decode_f16.metal"),
                "",
            )?,
        })
    }

    pub fn call(&self, weight: &Array, scales: &Array, biases: &Array) -> Result<Array> {
        let [rows, words] = *weight.shape() else {
            return Err(Error::Contract("Q4 row decode weight must be rank 2"));
        };
        let k = check_q4_decode(rows, words, 0, rows)?;
        if weight.dtype() != MlxDType::U32
            || [scales, biases]
                .iter()
                .any(|a| a.shape() != [rows, k / 64] || a.dtype() != MlxDType::BF16)
        {
            return Err(Error::Contract(
                "Q4 row decode requires U32 [N, K/8] with BF16 [N, K/64] scales and biases",
            ));
        }
        check_index_extent(
            rows,
            k,
            u32::MAX as usize,
            "Q4 row decode output index exceeds u32",
        )?;
        // SAFETY: one thread per packed word `i < N * K / 8`; it reads word i
        // and group `i / 8`, both inside the guarded `[N, K/8]` and `[N, K/64]`
        // inputs (eight words per 64-wide group), and writes two FP16 vectors
        // at `2 i` and `2 i + 1`, which cover output elements `8 i .. 8 i + 8`.
        let out = unsafe {
            self.kernel.call(
                &[weight.clone(), scales.clone(), biases.clone()],
                &[(&[rows, k], MlxDType::F16)],
                [rows * words, 1, 1],
                [256, 1, 1],
                &[],
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

/// How a 64 x 64 output tile of an FP16 GEMM is split over SIMD groups and
/// how its K loop is scheduled. Each keeps one FP32 accumulator per output
/// and rounds it once to BF16; the K reduction order differs between them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F16GemmTile {
    /// Four SIMD groups of 32 x 32 (128 threads) with the next K tile loaded
    /// ahead (EngineTailor g20).
    Prefetch,
    /// Two SIMD groups of 64 x 32 (64 threads), MLX's class-g FP16 tile
    /// (`bm64 bn64 bk16 wm1 wn2`), with Steel's unpipelined K loop (C14 `s28`).
    Wm1Wn2,
}

impl F16GemmTile {
    fn threads(self) -> usize {
        match self {
            F16GemmTile::Prefetch => 128,
            F16GemmTile::Wm1Wn2 => 64,
        }
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
    tile: F16GemmTile,
}

impl SplitF16Gemm {
    pub fn new(stream: &Stream) -> Result<Self> {
        Self::with_tile(stream, F16GemmTile::Prefetch)
    }

    pub fn with_tile(stream: &Stream, tile: F16GemmTile) -> Result<Self> {
        let header = include_str!("../native/simdgroup_mma.h");
        let (low, high) = match tile {
            F16GemmTile::Prefetch => (
                MetalKernel::new(
                    stream,
                    "apxinf_f16_gemm_bf16_tile64",
                    &["x", "weight"],
                    &["out"],
                    include_str!("../native/f16_gemm_bf16.metal"),
                    header,
                )?,
                MetalKernel::new(
                    stream,
                    "apxinf_f16_gemm_bf16_tile64_high",
                    &["weight", "lo", "x"],
                    &["out"],
                    include_str!("../native/f16_gemm_bf16_high.metal"),
                    header,
                )?,
            ),
            F16GemmTile::Wm1Wn2 => {
                let body = include_str!("../native/f16_gemm_bf16_wn2.metal");
                (
                    MetalKernel::new(
                        stream,
                        "apxinf_f16_gemm_bf16_wn2",
                        &["x", "weight"],
                        &["out"],
                        &format!("    constexpr int WS = N, OFF = 0;\n{body}"),
                        header,
                    )?,
                    MetalKernel::new(
                        stream,
                        "apxinf_f16_gemm_bf16_wn2_high",
                        &["weight", "lo", "x"],
                        &["out"],
                        &format!(
                            "{}    constexpr int WS = NW, OFF = NW;\n{body}",
                            include_str!("../native/f16_gemm_bf16_wn2_copy_low.metal")
                        ),
                        header,
                    )?,
                )
            }
        };
        Ok(Self { low, high, tile })
    }

    fn check(x: &Array, tokens: usize, weight: &Array) -> Result<(usize, usize, usize, usize)> {
        let [1, padded, k] = *x.shape() else {
            return Err(Error::Contract("split GEMM input must be [1, P, K]"));
        };
        let [wk, half] = *weight.shape() else {
            return Err(Error::Contract("split GEMM weight must be [K, NW]"));
        };
        if x.dtype() != MlxDType::F16 || weight.dtype() != MlxDType::F16 || wk != k {
            return Err(Error::Contract(
                "split GEMM requires FP16 [1, P, K] input and FP16 [K, NW] weight",
            ));
        }
        let full = check_split_gemm(padded, tokens, k, half)?;
        Ok((padded, k, half, full))
    }

    pub fn low(&self, x: &Array, tokens: usize, weight: &Array) -> Result<Array> {
        let (padded, k, half, _) = Self::check(x, tokens, weight)?;
        let threads = self.tile.threads();
        // SAFETY: each threadgroup owns a 64 x 64 output tile at rows
        // `64 * group.y < P` and columns `64 * group.x < NW`. A, B loads index
        // rows < P of x and rows < K, columns < NW of the weight, with K a
        // multiple of 16 so every half4 load stays inside a row. Stores are
        // limited to rows < T and columns < NW of `[1, T, NW]`; tiles do not
        // overlap. Both tiles cover their 64 x 64 block with their thread count.
        let out = unsafe {
            self.low.call(
                &[x.clone(), weight.clone()],
                &[(&[1, tokens, half], MlxDType::BF16)],
                [half / 64 * threads, padded / 64, 1],
                [threads, 1, 1],
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
        let (padded, k, half, full) = Self::check(x, tokens, weight)?;
        if low.shape() != [1, tokens, half] || low.dtype() != MlxDType::BF16 {
            return Err(Error::Contract(
                "split GEMM high half requires the BF16 [1, T, NW] low result",
            ));
        }
        let threads = self.tile.threads();
        // SAFETY: as in `low`, with weight stride NW and stores at columns
        // `NW + col < 2 NW`. Each group also copies 16-byte chunks of rows < T
        // of its 64 low columns into columns `[0, NW)` (8 threads per row,
        // rows stepped by the thread count / 8); copy and GEMM stores write
        // disjoint columns, and `lo` is only read.
        let out = unsafe {
            self.high.call(
                &[weight.clone(), low.clone(), x.clone()],
                &[(&[1, tokens, full], MlxDType::BF16)],
                [half / 64 * threads, padded / 64, 1],
                [threads, 1, 1],
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
/// `[1, T, N]`, in one pass: four SIMD groups of 32 x 32 per 64 x 64 tile
/// with Steel's unpipelined K loop (C14 `s13`). Rows at and after `T` are
/// padding and are never stored.
pub struct F16Gemm {
    kernel: MetalKernel,
}

impl F16Gemm {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_f16_gemm_bf16_steel_loop",
                &["x", "weight"],
                &["out"],
                include_str!("../native/f16_gemm_bf16_steel.metal"),
                include_str!("../native/simdgroup_mma.h"),
            )?,
        })
    }

    pub fn call(&self, x: &Array, tokens: usize, weight: &Array) -> Result<Array> {
        let (padded, k, n, _) = SplitF16Gemm::check(x, tokens, weight)?;
        // SAFETY: as in `SplitF16Gemm::low`: each 128-thread group owns a
        // 64 x 64 tile at rows `64 * group.y < P` and columns `64 * group.x < N`;
        // loads stay inside rows < P of x and the `[K, N]` weight, stores
        // inside rows < T of `[1, T, N]`. The check also bounds `[T, 2 N]`.
        let out = unsafe {
            self.kernel.call(
                &[x.clone(), weight.clone()],
                &[(&[1, tokens, n], MlxDType::BF16)],
                [n / 64 * 128, padded / 64, 1],
                [128, 1, 1],
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

#[cfg(test)]
mod tests {
    use super::{check_index_extent, check_q4_decode, check_split_gemm};
    use apxinf_core::{Error, Result};

    fn assert_contract_error(result: Result<usize>) {
        assert!(matches!(result, Err(Error::Contract(_))), "{result:?}");
    }

    #[test]
    fn index_extent_checks_products_and_inclusive_last_index() {
        assert!(check_index_extent(2, 4, 7, "extent").is_ok());
        for (rows, columns, limit) in [(2, 4, 6), (0, 4, 7), (4, 0, 7), (usize::MAX, 2, usize::MAX)]
        {
            assert!(matches!(
                check_index_extent(rows, columns, limit, "extent"),
                Err(Error::Contract("extent"))
            ));
        }
    }

    #[test]
    fn q4_decode_rejects_reported_output_wrap_before_vector_division() {
        assert_contract_error(check_q4_decode(65_536, 8_200, 0, 65_536));
    }

    #[test]
    fn q4_decode_rejects_source_wrap_with_nonzero_row_offset() {
        assert_contract_error(check_q4_decode(536_870_913, 8, 536_870_881, 32));
        assert_contract_error(check_q4_decode(1_073_741_856, 64, 1_073_741_824, 32));
    }

    #[test]
    fn q4_decode_checks_rust_arithmetic() {
        for words in [usize::MAX / 8 + 1, usize::MAX / 8 + 9] {
            assert_contract_error(check_q4_decode(32, words, 0, 32));
        }
        assert_contract_error(check_q4_decode(usize::MAX, 8, usize::MAX - 15, 32));
    }

    #[test]
    fn q4_decode_accepts_unsigned_index_boundaries() {
        assert_eq!(check_q4_decode(65_536, 8_192, 0, 65_536).unwrap(), 65_536);
        assert_eq!(
            check_q4_decode(536_870_912, 8, 536_870_880, 32).unwrap(),
            64
        );
    }

    #[test]
    fn q4_decode_rejects_invalid_geometry() {
        for (total, words, row0, rows) in [
            (32, 0, 0, 32),
            (32, 1, 0, 32),
            (32, 8, 0, 0),
            (64, 8, 0, 48),
            (64, 8, 33, 32),
            (32, 1 << 28, 0, 32),
            (1 << 31, 8, 0, 1 << 31),
            ((1 << 31) + 32, 8, 1 << 31, 32),
        ] {
            assert_contract_error(check_q4_decode(total, words, row0, rows));
        }
    }

    #[test]
    fn q4_decode_accepts_normal_and_clef_shapes() {
        for (total, width) in [(96, 128), (8_192, 4_096), (4_096, 4_096), (4_096, 12_288)] {
            assert_eq!(check_q4_decode(total, width / 8, 0, total).unwrap(), width);
            for row0 in [0, total / 2] {
                let rows = if total == 96 { 32 } else { total / 2 };
                assert_eq!(
                    check_q4_decode(total, width / 8, row0, rows).unwrap(),
                    width
                );
            }
        }
        assert_eq!(check_q4_decode(33, 8, 1, 32).unwrap(), 64);
    }

    #[test]
    fn split_gemm_rejects_reported_high_output_wrap() {
        assert_contract_error(check_split_gemm(32_832, 32_769, 16, 32_768));
    }

    #[test]
    fn split_gemm_rejects_input_wrap() {
        assert_contract_error(check_split_gemm(128, 65, 16_777_232, 64));
    }

    #[test]
    fn split_gemm_rejects_weight_wrap() {
        assert_contract_error(check_split_gemm(64, 1, 16_777_232, 128));
    }

    #[test]
    fn split_gemm_rejects_low_output_wrap() {
        assert_contract_error(check_split_gemm(65_600, 65_537, 16, 32_768));
    }

    #[test]
    fn split_gemm_checks_rust_arithmetic() {
        for half in [usize::MAX / 2 + 1, usize::MAX / 2 + 65] {
            assert_contract_error(check_split_gemm(64, 1, 16, half));
        }
    }

    #[test]
    fn split_gemm_accepts_signed_index_boundaries() {
        for (padded, tokens, width, half) in [
            (128, 65, 16_777_216, 64),
            (64, 1, 16_777_216, 128),
            (32_768, 32_768, 16, 32_768),
        ] {
            assert_eq!(
                check_split_gemm(padded, tokens, width, half).unwrap(),
                half * 2
            );
        }
    }

    #[test]
    fn split_gemm_rejects_invalid_geometry() {
        for (padded, tokens, width, half) in [
            (0, 0, 16, 64),
            (32, 1, 16, 64),
            (64, 0, 16, 64),
            (64, 65, 16, 64),
            (128, 64, 16, 64),
            (64, 1, 0, 64),
            (64, 1, 8, 64),
            (64, 1, 16, 0),
            (64, 1, 16, 32),
            (1 << 31, 1 << 31, 16, 64),
            (64, 1, 1 << 31, 64),
            (64, 1, 16, 1 << 30),
        ] {
            assert_contract_error(check_split_gemm(padded, tokens, width, half));
        }
    }

    #[test]
    fn split_gemm_accepts_normal_and_clef_shapes() {
        for (width, half) in [(16, 64), (4_096, 4_096), (4_096, 2_048), (12_288, 2_048)] {
            for tokens in 1usize..=4_096 {
                let padded = tokens.div_ceil(64) * 64;
                assert_eq!(
                    check_split_gemm(padded, tokens, width, half).unwrap(),
                    half * 2
                );
            }
        }
    }
}

//! BF16 NT GEMMs on Metal 4 tensors and MetalPerformancePrimitives `matmul2d`
//! (FP32 accumulation, `[1, T, K] x [N, K]^T`) whose epilogues consume the
//! 64 x 64 (or whole-head 64 x 128) accumulator tile directly. Each kernel
//! documents its roundings. Portions adapted from EngineTailor (MIT); see NOTICE.
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

const HEADER: &str = include_str!("../native/metal_tensor_mpp.h");
/// Width of one attention head in the head-major epilogues.
pub const HEAD_DIM: usize = 128;
/// Q4 words per decoded row: 4096 values, eight per U32 word.
const Q4_ROW_WORDS: usize = 512;

fn int(value: usize) -> Result<TemplateArg> {
    i32::try_from(value)
        .map(TemplateArg::Int)
        .map_err(|_| Error::Contract("fused GEMM extent exceeds i32"))
}

/// The last flattened index of an `elements` extent must fit i32.
fn fits_i32(elements: Option<usize>) -> Result<()> {
    elements
        .and_then(|n| n.checked_sub(1))
        .filter(|&last| last <= i32::MAX as usize)
        .map(|_| ())
        .ok_or(Error::Contract("fused GEMM index exceeds i32"))
}

/// BF16 `[1, T, K]` and `[N, K]` with T, K and N positive multiples of 64;
/// both operands and a `[T, N]` result are i32-indexable. Returns `(T, K, N)`.
fn operands(a: &Array, w: &Array) -> Result<(usize, usize, usize)> {
    let [1, t, k] = *a.shape() else {
        return Err(Error::Contract("fused GEMM input must be [1, T, K]"));
    };
    let [n, wk] = *w.shape() else {
        return Err(Error::Contract("fused GEMM weight must be [N, K]"));
    };
    if a.dtype() != MlxDType::BF16
        || w.dtype() != MlxDType::BF16
        || wk != k
        || [t, k, n].iter().any(|&v| v == 0 || v % 64 != 0)
    {
        return Err(Error::Contract(
            "fused GEMM requires BF16 [1, T, K] and [N, K] with T, K and N multiples of 64",
        ));
    }
    for (rows, columns) in [(t, k), (n, k), (t, n)] {
        fits_i32(rows.checked_mul(columns))?;
    }
    Ok((t, k, n))
}

/// The GEMM's 128-thread groups, one per 64 x 64 output tile.
fn tile_grid(t: usize, n: usize) -> ([usize; 3], [usize; 3]) {
    ([128 * (n / 64), t / 64, 1], [128, 1, 1])
}

fn bf16_shape(a: &Array, shape: &[usize]) -> bool {
    a.dtype() == MlxDType::BF16 && a.shape() == shape
}

/// `[1, T, N]` BF16 GEMM, as `MppBf16GemmNt`, whose groups afterwards decode
/// affine Q4 (group 64) rows of a second weight to BF16, as MLX `dequantize`
/// computes `scale * q + bias` natively in BF16: rows `q` U32 `[R, 512]` with
/// BF16 `scales` and `biases` `[R, 64]` become `[R, 4096]`. Group `g` decodes
/// rows `g, g + G, ...` for `G` output tiles, so the decode shares the GEMM's
/// dispatch. Returns `(gemm, decoded)`.
pub struct MppGemmDecodeQ4Rows {
    kernel: MetalKernel,
}

impl MppGemmDecodeQ4Rows {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_mpp_bf16_gemm_decode_q4_rows",
                &["a", "w", "gq", "gs", "gb"],
                &["out", "wgp"],
                concat!(
                    include_str!("../native/mpp_bf16_gemm_nt.metal"),
                    include_str!("../native/mpp_bf16_gemm_decode_q4_rows.metal")
                ),
                HEADER,
            )?,
        })
    }

    pub fn call(
        &self,
        a: &Array,
        w: &Array,
        q: &Array,
        scales: &Array,
        biases: &Array,
    ) -> Result<(Array, Array)> {
        let (t, k, n) = operands(a, w)?;
        let [rows, Q4_ROW_WORDS] = *q.shape() else {
            return Err(Error::Contract("decoded Q4 rows must be U32 [R, 512]"));
        };
        if q.dtype() != MlxDType::U32
            || rows == 0
            || !bf16_shape(scales, &[rows, 64])
            || !bf16_shape(biases, &[rows, 64])
        {
            return Err(Error::Contract(
                "decoded Q4 rows need U32 [R, 512] and BF16 scales and biases [R, 64]",
            ));
        }
        fits_i32(rows.checked_mul(Q4_ROW_WORDS * 8))?;
        let (grid, group) = tile_grid(t, n);
        // SAFETY: operands are row-contiguous BF16 `[1, T, K]`, `[N, K]` with
        // T, K, N multiples of 64 and i32-indexable (checked). Each 128-thread
        // group stores only its own 64 x 64 tile of `[1, T, N]`. Group `g` of
        // `G = (N / 64) (T / 64)` then decodes rows `r = g + G j < R`: thread
        // `tid < 128` reads words `tid + 128 j' < 512` of row `r` and their
        // scale/bias entries `< 64`, and writes the eight decoded values of each
        // word once; rows and words are partitioned, so writes do not overlap.
        let mut out = unsafe {
            self.kernel.call(
                &[
                    a.clone(),
                    w.clone(),
                    q.clone(),
                    scales.clone(),
                    biases.clone(),
                ],
                &[
                    (&[1, t, n], MlxDType::BF16),
                    (&[rows, Q4_ROW_WORDS * 8], MlxDType::BF16),
                ],
                grid,
                group,
                &[("K", int(k)?), ("N", int(n)?), ("GR", int(rows)?)],
            )?
        }
        .into_iter();
        Ok((out.next().unwrap(), out.next().unwrap()))
    }
}

/// `res + tanh(gate) * gemm` for the `[1, T, N]` GEMM with `gate` the columns
/// `N..2N` of `modulation`: the GEMM, `tanh(gate)` and the product are each
/// rounded to BF16 before the residual add, which is rounded once. `res` and
/// `modulation` are read through their strides, so broadcast views are not
/// copied.
pub struct MppGemmGatedResidual {
    kernel: MetalKernel,
}

impl MppGemmGatedResidual {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new_strided(
                stream,
                "apxinf_mpp_bf16_gemm_gated_residual",
                &["a", "w", "res", "mod"],
                &["out"],
                include_str!("../native/mpp_bf16_gemm_gated_residual.metal"),
                HEADER,
            )?,
        })
    }

    pub fn call(&self, a: &Array, w: &Array, res: &Array, modulation: &Array) -> Result<Array> {
        let (t, k, n) = operands(a, w)?;
        let [1, mt, width] = *modulation.shape() else {
            return Err(Error::Contract(
                "gated residual modulation must be [1, T, C]",
            ));
        };
        if !bf16_shape(res, &[1, t, n])
            || modulation.dtype() != MlxDType::BF16
            || mt != t
            || width < 2 * n
        {
            return Err(Error::Contract(
                "gated residual needs BF16 res [1, T, N] and modulation [1, T, >= 2N]",
            ));
        }
        let (a, w) = (a.contiguous()?, w.contiguous()?);
        let (grid, group) = tile_grid(t, n);
        // SAFETY: as `MppGemmDecodeQ4Rows` for the GEMM; `a` and `w` are made
        // row-contiguous for `matmul2d`. The epilogue of a tile at rows `< T`
        // and columns `< N` reads `res[row, col]` and `modulation[row, N + col]`
        // with `N + col < 2N <= C` through their int64 strides and stores
        // `out[row, col]` of `[1, T, N]` once; tiles do not overlap.
        let out = unsafe {
            self.kernel.call(
                &[a, w, res.clone(), modulation.clone()],
                &[(&[1, t, n], MlxDType::BF16)],
                grid,
                group,
                &[("K", int(k)?), ("N", int(n)?)],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

/// SwiGLU over interleaved rows: `w` `[N, K]` holds 32-row pairs (rows
/// `64 j .. 64 j + 31` gate rows `32 j ..`, the next 32 the matching up rows),
/// and `out[:, 32 j + c] = silu(gate) * up` `[1, T, N / 2]`, computed in FP32
/// from the unrounded accumulators and rounded once to BF16.
pub struct MppGemmSwigluPairs {
    kernel: MetalKernel,
}

impl MppGemmSwigluPairs {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_mpp_bf16_gemm_swiglu_pairs",
                &["a", "w"],
                &["out"],
                include_str!("../native/mpp_bf16_gemm_swiglu_pairs.metal"),
                HEADER,
            )?,
        })
    }

    pub fn call(&self, a: &Array, w: &Array) -> Result<Array> {
        let (t, k, n) = operands(a, w)?;
        let (grid, group) = tile_grid(t, n);
        // SAFETY: as `MppGemmDecodeQ4Rows` for the GEMM, with N a multiple of
        // 64. The group of tile `(x, y)` reads its own staged 64 x 64 tile and
        // stores rows `64 y + row < T`, columns `32 x + col < N / 2` of
        // `[1, T, N / 2]` once; tiles map to disjoint output blocks.
        let out = unsafe {
            self.kernel.call(
                &[a.clone(), w.clone()],
                &[(&[1, t, n / 2], MlxDType::BF16)],
                grid,
                group,
                &[("K", int(k)?), ("N", int(n)?), ("NO", int(n / 2)?)],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

fn head_operands(
    a: &Array,
    w: &Array,
    prefix: &Array,
) -> Result<(usize, usize, usize, usize, usize)> {
    let (t, k, n) = operands(a, w)?;
    let [1, heads, text, HEAD_DIM] = *prefix.shape() else {
        return Err(Error::Contract("head prefix must be [1, H, P, 128]"));
    };
    if n != heads * HEAD_DIM || text == 0 || prefix.dtype() != MlxDType::BF16 {
        return Err(Error::Contract(
            "head epilogues need N = 128 H and a nonempty BF16 prefix [1, H, P, 128]",
        ));
    }
    fits_i32(
        heads
            .checked_mul(text + t)
            .and_then(|rows| rows.checked_mul(HEAD_DIM)),
    )?;
    Ok((t, k, n, heads, text))
}

/// Head-major `[1, H, P + T, 128]` result: rows `0..P` of each head copy the
/// prefix `[1, H, P, 128]` and rows `P..` hold the BF16 GEMM of the `T` tokens
/// (`N = 128 H`), so no concatenation follows.
pub struct MppGemmHeadsWithPrefix {
    kernel: MetalKernel,
}

impl MppGemmHeadsWithPrefix {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_mpp_bf16_gemm_heads_with_prefix",
                &["a", "w", "vt"],
                &["out"],
                include_str!("../native/mpp_bf16_gemm_heads_with_prefix.metal"),
                HEADER,
            )?,
        })
    }

    pub fn call(&self, a: &Array, w: &Array, prefix: &Array) -> Result<Array> {
        let (t, k, n, heads, text) = head_operands(a, w, prefix)?;
        let (grid, group) = tile_grid(t, n);
        // SAFETY: as `MppGemmDecodeQ4Rows` for the GEMM. Tile `(x, y)` covers
        // head `64 x / 128`, dims `64 x % 128 .. + 64`, and stores token rows
        // `P + 64 y + row` of that head once; the `y = 0` tiles also copy prefix
        // rows `< P` of those 64 dims. All indices are below `H (P + T) 128`,
        // which fits i32 (checked), and the tiles' outputs are disjoint.
        let out = unsafe {
            self.kernel.call(
                &[a.clone(), w.clone(), prefix.clone()],
                &[(&[1, heads, text + t, HEAD_DIM], MlxDType::BF16)],
                grid,
                group,
                &[("K", int(k)?), ("N", int(n)?)],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

/// Per-head RMSNorm (eps 1e-6, BF16-rounded GEMM values, FP32 statistics)
/// times `norm` `[128]`, then interleaved-pair RoPE from FP32 `cos`/`sin`
/// `[T, 64]`, rounded once to BF16, in the epilogue of one 64 x 128 tile per
/// head. The result is head-major `[1, H, T, 128]`, or `[1, H, P + T, 128]`
/// with the prefix rows copied first when `copy_prefix`.
pub struct MppGemmHeadRmsRope {
    kernel: MetalKernel,
}

impl MppGemmHeadRmsRope {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_mpp_bf16_gemm_head_rms_rope",
                &["a", "w", "nw", "rc", "rs", "kt"],
                &["out"],
                include_str!("../native/mpp_bf16_gemm_head_rms_rope.metal"),
                HEADER,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &self,
        a: &Array,
        w: &Array,
        norm: &Array,
        cos: &Array,
        sin: &Array,
        prefix: &Array,
        copy_prefix: bool,
    ) -> Result<Array> {
        let (t, k, n, heads, text) = head_operands(a, w, prefix)?;
        if norm.shape() != [HEAD_DIM]
            || !matches!(norm.dtype(), MlxDType::BF16 | MlxDType::F32)
            || [cos, sin]
                .iter()
                .any(|table| table.dtype() != MlxDType::F32 || table.shape() != [t, HEAD_DIM / 2])
        {
            return Err(Error::Contract(
                "head RMSNorm/RoPE needs a [128] norm weight and FP32 cos/sin [T, 64]",
            ));
        }
        let rows = if copy_prefix { text + t } else { t };
        // SAFETY: the group of head `x < H` and token tile `y < T / 64` runs
        // the 64 x 128 GEMM tile of rows `< T`, columns `128 x .. 128 x + 127 < N`
        // (8 SIMD groups, 256 threads). Each SIMD group normalizes rows
        // `64 y + row < T`, lane `l` owning dims `4 l .. 4 l + 3 < 128`, reads
        // `cos`/`sin[row, 2 l .. 2 l + 1]` of `[T, 64]` and stores those four
        // values once at row `P' + 64 y + row` of head `x`, `P' = P` only with the
        // prefix. The `y = 0` groups copy that head's prefix rows when it is
        // present. Indices stay below `H (P + T) 128` (i32, checked).
        let out = unsafe {
            self.kernel.call(
                &[
                    a.clone(),
                    w.clone(),
                    norm.clone(),
                    cos.clone(),
                    sin.clone(),
                    prefix.clone(),
                ],
                &[(&[1, heads, rows, HEAD_DIM], MlxDType::BF16)],
                [256 * heads, t / 64, 1],
                [256, 1, 1],
                &[
                    ("K", int(k)?),
                    ("N", int(n)?),
                    ("WITH_TEXT", TemplateArg::Bool(copy_prefix)),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(stream: &Stream, shape: &[usize]) -> Array {
        Array::zeros(stream, shape, MlxDType::BF16).unwrap()
    }

    #[test]
    fn operand_checks_need_no_device() {
        let s = Stream::cpu().unwrap();
        assert_eq!(
            operands(&bf16(&s, &[1, 128, 64]), &bf16(&s, &[192, 64])).unwrap(),
            (128, 64, 192)
        );
        for (a, w) in [
            (vec![1, 100, 64], vec![64, 64]),
            (vec![2, 64, 64], vec![64, 64]),
            (vec![1, 64, 64], vec![64, 128]),
            (vec![1, 0, 64], vec![64, 64]),
        ] {
            assert!(
                operands(&bf16(&s, &a), &bf16(&s, &w)).is_err(),
                "{a:?} {w:?}"
            );
        }
        let f32_input = Array::zeros(&s, &[1, 64, 64], MlxDType::F32).unwrap();
        assert!(operands(&f32_input, &bf16(&s, &[64, 64])).is_err());
        assert!(fits_i32(Some(1 << 31)).is_ok() && fits_i32(Some((1 << 31) + 1)).is_err());
        assert!(fits_i32(None).is_err());
    }

    #[test]
    fn head_prefix_must_match_the_heads() {
        let s = Stream::cpu().unwrap();
        let (a, w) = (bf16(&s, &[1, 64, 64]), bf16(&s, &[256, 64]));
        assert_eq!(
            head_operands(&a, &w, &bf16(&s, &[1, 2, 5, 128])).unwrap(),
            (64, 64, 256, 2, 5)
        );
        for prefix in [[1, 3, 5, 128], [1, 2, 0, 128], [1, 2, 5, 64]] {
            assert!(
                head_operands(&a, &w, &bf16(&s, &prefix)).is_err(),
                "{prefix:?}"
            );
        }
    }
}

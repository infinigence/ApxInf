use apxinf_core::{Error, Result};

use super::contracts::{normalize, GemmArgs, Semantic};
use super::gemm_execution as execution;
use crate::CudaContext;

const PAIR_WIDTH: usize = 256;

/// Convert checkpoint `[2N,K]` gate/up halves into the semantic's canonical
/// `[K,2N]` 256-column pair-interleaved encoding.
pub fn pack_bf16_swiglu_weight_host(
    source: &[f32],
    full_n: usize,
    k: usize,
) -> Result<Vec<half::bf16>> {
    if full_n == 0 || k == 0 || full_n % (2 * PAIR_WIDTH) != 0 || source.len() != full_n * k {
        return Err(Error::Other(format!(
            "BF16 SwiGLU weight pack expects {full_n} x {k} values with {full_n} divisible by {}, got {}",
            2 * PAIR_WIDTH,
            source.len()
        )));
    }
    let half_n = full_n / 2;
    let mut packed = Vec::with_capacity(source.len());
    for row_k in 0..k {
        for tile in 0..half_n / PAIR_WIDTH {
            for half in [0, half_n] {
                let source_row = half + tile * PAIR_WIDTH;
                for within in 0..PAIR_WIDTH {
                    packed.push(half::bf16::from_f32(
                        source[(source_row + within) * k + row_k],
                    ));
                }
            }
        }
    }
    Ok(packed)
}

/// Rounded BF16 SwiGLU over a fused gate/up projection.
///
/// `gemm.b` is `[K,2N]` with 256 gate columns followed immediately by their
/// matching 256 up columns. The encoding is shared by every candidate,
/// including the cuBLAS fallback. The public output is always exactly `[M,N]`;
/// provider tile padding is never visible to callers.
pub struct GemmSwiGluArgs<'a> {
    pub gemm: GemmArgs<'a>,
}

pub fn gemm_swiglu_pair_interleaved_256(ctx: &CudaContext, args: GemmSwiGluArgs<'_>) -> Result<()> {
    execution::execute(
        ctx,
        normalize(ctx, args.gemm, Semantic::GemmSwiGluPairInterleaved256, None)?,
    )
}

#[cfg(test)]
mod tests {
    use super::{pack_bf16_swiglu_weight_host, PAIR_WIDTH};

    #[test]
    fn checkpoint_pack_interleaves_each_gate_up_pair() {
        let full_n = 4 * PAIR_WIDTH;
        let half_n = full_n / 2;
        let k = 2;
        let source: Vec<f32> = (0..full_n * k).map(|value| value as f32).collect();
        let packed = pack_bf16_swiglu_weight_host(&source, full_n, k).unwrap();

        for row_k in 0..k {
            let row = &packed[row_k * full_n..(row_k + 1) * full_n];
            for tile in 0..half_n / PAIR_WIDTH {
                for (pair_half, source_base) in [0, half_n].into_iter().enumerate() {
                    for within in 0..PAIR_WIDTH {
                        let packed_col = tile * 2 * PAIR_WIDTH + pair_half * PAIR_WIDTH + within;
                        let source_n = source_base + tile * PAIR_WIDTH + within;
                        let expected = half::bf16::from_f32(source[source_n * k + row_k]);
                        assert_eq!(row[packed_col].to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }
}

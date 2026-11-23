use apxinf_core::{Error, Result, Tensor};

use super::contracts::{normalize, GemmArgs, GemmWeightLayout, Semantic};
use super::gemm_execution as execution;
use crate::CudaContext;

const PAIR_WIDTH: usize = 256;

/// Convert checkpoint `[2N,K]` gate/up halves into the semantic's canonical
/// `[K,2N]` 256-column pair-interleaved encoding.
pub fn pack_swiglu_weight_host(
    source: &[f32],
    full_n: usize,
    k: usize,
    layout: GemmWeightLayout,
) -> Result<Vec<half::bf16>> {
    let pair_width = match layout {
        GemmWeightLayout::GateUpPairs { pair_width } => pair_width,
        _ => {
            return Err(Error::Other(
                "SwiGLU weight packing requires a gate/up-pair layout".into(),
            ));
        }
    };
    if pair_width != PAIR_WIDTH {
        return Err(Error::Other(format!(
            "SwiGLU weight packing does not support gate/up pair width {pair_width}"
        )));
    }
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
pub struct BlockQuantizedSwiGluArgs<'a> {
    pub activation: &'a Tensor,
    pub activation_scales: &'a Tensor,
    pub weight: &'a Tensor,
    pub weight_scales: &'a Tensor,
    pub output: &'a Tensor,
    pub output_scales: &'a Tensor,
    pub alpha: &'a Tensor,
    pub input_global_scale: &'a Tensor,
    pub output_inverse_global_scale: &'a Tensor,
}

pub enum GemmSwiGluArgs<'a> {
    Gemm(GemmArgs<'a>),
    BlockQuantized(BlockQuantizedSwiGluArgs<'a>),
}

impl<'a> GemmSwiGluArgs<'a> {
    pub fn new(gemm: GemmArgs<'a>) -> Self {
        Self::Gemm(gemm)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn block_quantized(
        activation: &'a Tensor,
        activation_scales: &'a Tensor,
        weight: &'a Tensor,
        weight_scales: &'a Tensor,
        output: &'a Tensor,
        output_scales: &'a Tensor,
        alpha: &'a Tensor,
        input_global_scale: &'a Tensor,
        output_inverse_global_scale: &'a Tensor,
    ) -> Self {
        Self::BlockQuantized(BlockQuantizedSwiGluArgs {
            activation,
            activation_scales,
            weight,
            weight_scales,
            output,
            output_scales,
            alpha,
            input_global_scale,
            output_inverse_global_scale,
        })
    }
}

pub fn gemm_swiglu(ctx: &CudaContext, args: GemmSwiGluArgs<'_>) -> Result<()> {
    match args {
        GemmSwiGluArgs::Gemm(gemm) => {
            super::observer::observe(&gemm)?;
            execution::execute(ctx, normalize(ctx, gemm, Semantic::GemmSwiGlu, None)?)
        }
        GemmSwiGluArgs::BlockQuantized(args) => super::nvfp4_scales::nvfp4_gemm_swiglu(
            ctx,
            args.activation,
            args.activation_scales,
            args.weight,
            args.weight_scales,
            args.output,
            args.output_scales,
            args.alpha,
            args.input_global_scale,
            args.output_inverse_global_scale,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{pack_swiglu_weight_host, GemmWeightLayout, PAIR_WIDTH};

    #[test]
    fn checkpoint_pack_interleaves_each_gate_up_pair() {
        let full_n = 4 * PAIR_WIDTH;
        let half_n = full_n / 2;
        let k = 2;
        let source: Vec<f32> = (0..full_n * k).map(|value| value as f32).collect();
        let packed = pack_swiglu_weight_host(
            &source,
            full_n,
            k,
            GemmWeightLayout::GateUpPairs {
                pair_width: PAIR_WIDTH,
            },
        )
        .unwrap();

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

//! Fixed Q/K normalization and RoPE fusion from EngineTailor (MIT).
//! Copyright 2026 Haiyan Qin. Source: candidate_metal_qk_norm_rope.py.
use crate::{Array, MetalKernel, MlxDType, Stream};
use apxinf_core::{Error, Result};

/// One BF16 decode row, 16 query heads, 8 key heads, width 128, epsilon 1e-6.
/// Computes normalization (including weight) in FP32, rounds to BF16, then
/// separately rounds both RoPE products and their sum as in the source kernel.
pub struct QkNormRope {
    kernel: MetalKernel,
}
impl QkNormRope {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_qk_norm_rope_tg256_hd128",
                &[
                    "q_input", "k_input", "q_weight", "k_weight", "cos_row", "sin_row",
                ],
                &["q_output", "k_output"],
                include_str!("../native/qk_norm_rope.metal"),
                "",
            )?,
        })
    }
    pub fn call(
        &self,
        q: &Array,
        k: &Array,
        q_weight: &Array,
        k_weight: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<(Array, Array)> {
        if q.shape() != [1, 1, 16, 128]
            || k.shape() != [1, 1, 8, 128]
            || [q, k].iter().any(|a| a.dtype() != MlxDType::BF16)
            || [q_weight, k_weight]
                .iter()
                .any(|a| a.shape() != [128] || a.dtype() != MlxDType::F32)
            || [cos, sin]
                .iter()
                .any(|a| a.numel() != 128 || a.dtype() != MlxDType::BF16)
        {
            return Err(Error::Contract(
                "Q/K norm-RoPE requires BF16 B1/T1/H16+8/D128 and F32 weights",
            ));
        }
        // SAFETY: 24 threadgroups address 16 Q and 8 K rows respectively.
        // Guards guarantee all rows contain 128 elements and all weights/tables
        // contain 128. Only lanes <128 write, one unique element each; scratch
        // reductions are separated by threadgroup barriers. Native MLX forces
        // row-contiguous inputs and validates all logical stream owners.
        let mut out = unsafe {
            self.kernel.call(
                &[
                    q.clone(),
                    k.clone(),
                    q_weight.clone(),
                    k_weight.clone(),
                    cos.clone(),
                    sin.clone(),
                ],
                &[
                    (&[1, 16, 1, 128], MlxDType::BF16),
                    (&[1, 8, 1, 128], MlxDType::BF16),
                ],
                [24 * 256, 1, 1],
                [256, 1, 1],
                Some(MlxDType::BF16),
            )?
        };
        let k = out.pop().unwrap();
        Ok((out.pop().unwrap(), k))
    }
}

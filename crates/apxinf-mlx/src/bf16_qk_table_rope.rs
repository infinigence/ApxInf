use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

/// Half-split Q/K rotation for one BF16 token, using precomputed BF16 tables.
/// Negation, each product and the sum round separately to BF16.
pub struct Bf16QkTableRope {
    kernel: MetalKernel,
}

impl Bf16QkTableRope {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_bf16_qk_table_rope",
                &["q", "k", "cos_row", "sin_row"],
                &["q_out", "k_out"],
                include_str!("../native/bf16_qk_table_rope.metal"),
                "",
            )?,
        })
    }

    /// Accepts Q `[1, Hq, 1, D]`, K `[1, Hk, 1, D]` and tables `[1, 1, 1, D]`.
    /// All arrays must be BF16, heads positive, and D positive and even.
    /// Outputs preserve Q/K shapes. Table halves may differ.
    pub fn call(&self, q: &Array, k: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
        let [1, q_heads, 1, dim] = *q.shape() else {
            return Err(Error::Contract("table RoPE Q must be [1, Hq, 1, D]"));
        };
        let [1, k_heads, 1, k_dim] = *k.shape() else {
            return Err(Error::Contract("table RoPE K must be [1, Hk, 1, D]"));
        };
        if q_heads == 0
            || k_heads == 0
            || dim == 0
            || dim % 2 != 0
            || k_dim != dim
            || [q, k, cos, sin].iter().any(|a| a.dtype() != MlxDType::BF16)
            || [cos, sin].iter().any(|a| a.shape() != [1, 1, 1, dim])
        {
            return Err(Error::Contract(
                "table RoPE requires BF16 arrays, positive heads, even D and [1,1,1,D] tables",
            ));
        }
        let elements = q_heads
            .checked_add(k_heads)
            .and_then(|heads| heads.checked_mul(dim))
            .filter(|&n| i32::try_from(n).is_ok())
            .ok_or(Error::Contract("table RoPE element count exceeds i32"))?;
        // SAFETY: positive heads and even D give (Hq + Hk) * D / 2 pairs.
        // Each guarded thread selects one Q or K head, reads columns c and
        // c + D/2 within that head and the same columns from both tables, then
        // writes those two unique columns. The checked total element count
        // bounds every offset, template and grid value by i32::MAX. MetalKernel
        // supplies row-contiguous inputs and validates stream ownership.
        let mut output = unsafe {
            self.kernel.call(
                &[q.clone(), k.clone(), cos.clone(), sin.clone()],
                &[(q.shape(), MlxDType::BF16), (k.shape(), MlxDType::BF16)],
                [elements / 2, 1, 1],
                [256, 1, 1],
                &[
                    ("T", TemplateArg::DType(MlxDType::BF16)),
                    ("Q_HEADS", TemplateArg::Int(q_heads as i32)),
                    ("K_HEADS", TemplateArg::Int(k_heads as i32)),
                    ("DIM", TemplateArg::Int(dim as i32)),
                ],
            )?
        };
        let k = output.pop().unwrap();
        Ok((output.pop().unwrap(), k))
    }
}

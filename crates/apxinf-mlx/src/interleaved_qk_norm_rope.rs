//! Q/K RMSNorm with interleaved-pair RoPE and a head-major layout change.
//! Adapted from mflux (MIT); see NOTICE.
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

/// For `q`, `k` `[B, L, H * D]`: weighted RMSNorm over each head's `D`
/// features (epsilon 1e-6, rounding like `mx.fast.rms_norm`), then RoPE on
/// pairs `(2i, 2i + 1)` with FP32 `cos`/`sin` tables `[L, D / 2]`, each product
/// rounded before its add or subtract. Returns `[B, H, L, D]` for Q and K.
pub struct InterleavedQkNormRope {
    kernel: MetalKernel,
}

impl InterleavedQkNormRope {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_interleaved_qk_norm_rope",
                &["q", "k", "wq", "wk", "cos_t", "sin_t"],
                &["out_q", "out_k"],
                include_str!("../native/interleaved_qk_norm_rope.metal"),
                "",
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &self,
        q: &Array,
        k: &Array,
        q_weight: &Array,
        k_weight: &Array,
        cos: &Array,
        sin: &Array,
        heads: usize,
    ) -> Result<(Array, Array)> {
        let [batch, length, width] = *q.shape() else {
            return Err(Error::Contract("Q/K norm-RoPE input must be [B, L, H * D]"));
        };
        let dtype = q.dtype();
        let dim = width.checked_div(heads).unwrap_or(0);
        let rows = [batch, length, heads, 32]
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .filter(|&n| n > 0 && i32::try_from(n).is_ok());
        if heads == 0
            || dim * heads != width
            || dim % 64 != 0
            || dim > 512
            || rows.is_none()
            || k.shape() != q.shape()
            || !matches!(dtype, MlxDType::BF16 | MlxDType::F16 | MlxDType::F32)
            || [k, q_weight, k_weight].iter().any(|a| a.dtype() != dtype)
            || [q_weight, k_weight].iter().any(|a| a.shape() != [dim])
            || [cos, sin]
                .iter()
                .any(|a| a.shape() != [length, dim / 2] || a.dtype() != MlxDType::F32)
        {
            return Err(Error::Contract(
                "Q/K norm-RoPE requires same-dtype Q/K and norm weights, D a multiple of 64 \
                 up to 512, and FP32 [L, D / 2] rotary tables",
            ));
        }
        let int = |v: usize| TemplateArg::Int(v as i32);
        let shape = [batch, heads, length, dim];
        // SAFETY: one SIMD group per (b, l, h) row; the guard above bounds the
        // grid to B * L * H rows. Each lane reads features `block * 128 + lane
        // * 4 + j < D` of its row through the row-contiguous input strides,
        // weight features < D and table entries `[l, c / 2]` with c < D.
        // It writes pairs (c, c + 1) < D of its own output row; rows and
        // pairs do not overlap.
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
                &[(&shape, dtype), (&shape, dtype)],
                [rows.unwrap(), 1, 1],
                [256, 1, 1],
                &[
                    ("TQ", TemplateArg::DType(dtype)),
                    ("TK", TemplateArg::DType(dtype)),
                    ("B", int(batch)),
                    ("L", int(length)),
                    ("H", int(heads)),
                    ("D", int(dim)),
                ],
            )?
        };
        let k = out.pop().unwrap();
        Ok((out.pop().unwrap(), k))
    }
}

//! Text-encoder layer composition (Qwen3, INT8 ConvRot, BF16 residual stream). One call runs one streamed layer.
use super::refiner::attend_gqa;
use super::util::{alloc, int8_linear};
use crate::cuda_minimax_h3_image::backend::ops::{self, GemmPolicy};
use crate::cuda_minimax_h3_image::backend::{AttentionPolicy, CudaBuffer, CudaContext};
use crate::cuda_minimax_h3_image::config::*;
use crate::cuda_minimax_h3_image::model::host::text_rope_table;
use crate::cuda_minimax_h3_image::weights::TextLayer;
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use std::rc::Rc;

pub(in crate::cuda_minimax_h3_image) struct TextBlocks {
    ctx: Rc<CudaContext>,
    rows: usize,
    rope: Tensor,
}

impl TextBlocks {
    /// `rows` is the padded row count (`TE_ROW_BUCKET` multiple) the layers will run at.
    pub fn new(ctx: Rc<CudaContext>, rows: usize) -> Result<Self> {
        let bytes: Vec<u8> = text_rope_table(rows)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Cuda)?;
        buffer
            .copy_from_host_on(&bytes, ctx.stream())
            .map_err(Error::Cuda)?;
        let rope = buffer
            .as_tensor(Shape::new(vec![rows, 2, TE_HEAD_DIM / 2]), DType::F32)
            .map_err(Error::Cuda)?;
        Ok(Self { ctx, rows, rope })
    }

    /// Upload `embeddings` (BF16, row-major `[real_rows, TE_DIM]`) into a zero-padded `[rows, TE_DIM]` stream.
    pub fn embed(&self, embeddings: &[half::bf16], real_rows: usize) -> Result<Tensor> {
        if real_rows == 0 || real_rows > self.rows || embeddings.len() != real_rows * TE_DIM {
            return Err(Error::Other(format!(
                "text stream of {real_rows} rows does not fit {} padded rows",
                self.rows
            )));
        }
        let mut bytes = vec![0u8; self.rows * TE_DIM * 2];
        for (dst, v) in bytes.chunks_exact_mut(2).zip(embeddings) {
            dst.copy_from_slice(&v.to_le_bytes());
        }
        let buffer = CudaBuffer::alloc(bytes.len(), self.ctx.device_id()).map_err(Error::Cuda)?;
        buffer
            .copy_from_host_on(&bytes, self.ctx.stream())
            .map_err(Error::Cuda)?;
        buffer
            .as_tensor(Shape::new(vec![self.rows, TE_DIM]), DType::BF16)
            .map_err(Error::Cuda)
    }

    /// One decoder layer, in place on the residual stream `x` (`[rows, TE_DIM]` BF16).
    pub fn layer(
        &self,
        layer: &TextLayer,
        x: &Tensor,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
    ) -> Result<()> {
        let ctx = &*self.ctx;
        let rows = self.rows;
        // self-attention
        let h = rms_eps(ctx, x, &layer.input_norm)?;
        let rotated = ops::quantization::rotate_quantize_rows_i8(ctx, &h)?;
        let q = int8_linear(ctx, &rotated, &layer.q, gemm)?;
        let k = int8_linear(ctx, &rotated, &layer.k, gemm)?;
        let v = int8_linear(ctx, &rotated, &layer.v, gemm)?;
        for (t, heads, norm) in [
            (&q, TE_HEADS, &layer.q_norm),
            (&k, TE_KV_HEADS, &layer.k_norm),
        ] {
            let per_head = t.reshape(vec![rows, heads, TE_HEAD_DIM])?;
            ops::rms_norm_apply_precomputed(ctx, &per_head, norm, &self.rope, TE_EPS, &per_head)?;
        }
        let attended = attend_gqa(ctx, &q, &k, &v, rows, attention)?;
        let rotated = ops::quantization::rotate_quantize_rows_i8(ctx, &attended)?;
        let branch = int8_linear(ctx, &rotated, &layer.o, gemm)?;
        ops::add_into(ctx, &branch, x)?;
        // gated SiLU feed-forward
        let h = rms_eps(ctx, x, &layer.post_norm)?;
        let rotated = ops::quantization::rotate_quantize_rows_i8(ctx, &h)?;
        let gate = int8_linear(ctx, &rotated, &layer.gate, gemm)?;
        let up = int8_linear(ctx, &rotated, &layer.up, gemm)?;
        let activated = ops::pointwise::silu_mul(ctx, &gate, &up)?;
        let rotated = ops::quantization::rotate_quantize_rows_i8(ctx, &activated)?;
        let branch = int8_linear(ctx, &rotated, &layer.down, gemm)?;
        ops::add_into(ctx, &branch, x)
    }
}

fn rms_eps(ctx: &CudaContext, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let out = alloc(ctx, x.shape().dims(), DType::BF16)?;
    ops::norm::norm(ctx, ops::NormArgs::rms(x, weight, &out, TE_EPS))?;
    Ok(out)
}

//! Video VAE decoder composition: FP32 latent embedding, 36 pre-norm blocks with per-head RMS-normalised, partially rotated attention
//! and scaled residuals, and the pixel-patch head. One call decodes a batch of tiles of one temporal chunk, which never attend to each
//! other. The block weights are not resident: they stream through the ring of `VideoVaeWeights` while the previous block computes.
use super::refiner::rms;
use super::util::{alloc, dense_linear_bf16, dense_linear_streamed, rows};
use crate::cuda_minimax_h3::backend::ops::{self, GemmArgs, GemmBiasArgs, GemmPolicy};
use crate::cuda_minimax_h3::backend::{AttentionPolicy, CudaBuffer, CudaContext};
use crate::cuda_minimax_h3::config::*;
use crate::cuda_minimax_h3::model::vae_host::{tile_tokens, vae_rope_table};
use crate::cuda_minimax_h3::weights::VideoVaeWeights;
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use std::rc::Rc;

pub(in crate::cuda_minimax_h3) struct VideoVaeBlocks {
    ctx: Rc<CudaContext>,
    weights: Rc<VideoVaeWeights>,
}

fn upload_f32(ctx: &CudaContext, values: &[f32], shape: Vec<usize>) -> Result<Tensor> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Cuda)?;
    buffer
        .copy_from_host_on(&bytes, ctx.stream())
        .map_err(Error::Cuda)?;
    buffer
        .as_tensor(Shape::new(shape), DType::F32)
        .map_err(Error::Cuda)
}

/// `x += scale * branch` with one rounding, in place (`style` is `[0; 0; scale]`).
fn scaled_residual(ctx: &CudaContext, x: &Tensor, branch: &Tensor, style: &Tensor) -> Result<()> {
    ops::residual_add(
        ctx,
        ops::ResidualAddArgs {
            input: branch,
            bias: None,
            residual: x,
            gate: Some(ops::ResidualGate::PackedStyle(style)),
            hidden: x,
            bias_order: ops::BiasOrder::SingleRound,
        },
    )
}

impl VideoVaeBlocks {
    pub fn new(ctx: Rc<CudaContext>, weights: Rc<VideoVaeWeights>) -> Self {
        Self { ctx, weights }
    }

    /// Decode `tiles` tiles of `t` latent steps each. `latents` is FP32 `[tiles * t * VV_TILE_LATENT^2, LATENT_CHANNELS]` normalised
    /// latent tokens (time, row, column order inside a tile). Returns BF16 `[tiles * tile_tokens(t), VV_OUT_DIM]`; the first
    /// `t * VV_TILE_LATENT^2` rows of every tile are pixel patches, the rest belong to the suffix tokens.
    pub fn decode(
        &self,
        latents: &Tensor,
        tiles: usize,
        t: usize,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
    ) -> Result<Tensor> {
        let rope = self.rope_table(tiles, t)?;
        self.decode_with(latents, &rope, tiles, t, gemm, attention)
    }

    /// The rotary table of `tiles` tiles of `t` steps, FP32 `[tiles * tile_tokens(t), 2, VV_HEAD_DIM / 2]`. It depends only on the
    /// shape, so a prepared plan builds it once and keeps its address.
    pub fn rope_table(&self, tiles: usize, t: usize) -> Result<Tensor> {
        let one = vae_rope_table(t);
        let table: Vec<f32> = (0..tiles).flat_map(|_| one.iter().copied()).collect();
        upload_f32(
            &self.ctx,
            &table,
            vec![tiles * tile_tokens(t), 2, VV_HEAD_DIM / 2],
        )
    }

    /// As [`Self::decode`] with a caller-owned rotary table from [`Self::rope_table`].
    pub fn decode_with(
        &self,
        latents: &Tensor,
        rope: &Tensor,
        tiles: usize,
        t: usize,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
    ) -> Result<Tensor> {
        let x = self.residual_stream_with(latents, rope, tiles, t, gemm, attention, VV_BLOCKS)?;
        let ctx = &*self.ctx;
        let resident = &self.weights.resident;
        let normed = alloc(ctx, &[x.shape().dims()[0], VV_DIM], DType::BF16)?;
        ops::norm::norm(
            ctx,
            ops::NormArgs::layer(
                &x,
                &resident.norm_out_weight,
                &resident.norm_out_bias,
                &normed,
                NORM_EPS,
            ),
        )?;
        dense_linear_bf16(
            ctx,
            &normed,
            &resident.proj_out_weight,
            Some(&resident.proj_out_bias),
            gemm,
        )
    }

    /// The residual stream `[tiles * tile_tokens(t), VV_DIM]` after the embedding and the first `blocks` blocks (`0` is the embedding
    /// alone). Blocks past `blocks` are streamed but not computed. Parity tests read intermediate streams through this.
    #[cfg(test)]
    pub fn residual_stream(
        &self,
        latents: &Tensor,
        tiles: usize,
        t: usize,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
        blocks: usize,
    ) -> Result<Tensor> {
        let rope = self.rope_table(tiles, t)?;
        self.residual_stream_with(latents, &rope, tiles, t, gemm, attention, blocks)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn residual_stream_with(
        &self,
        latents: &Tensor,
        rope: &Tensor,
        tiles: usize,
        t: usize,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
        blocks: usize,
    ) -> Result<Tensor> {
        if tiles == 0 || t == 0 {
            return Err(Error::Other(
                "video VAE decode needs at least one tile and one latent step".into(),
            ));
        }
        let ctx = &*self.ctx;
        let resident = &self.weights.resident;
        let pixels = t * VV_TILE_LATENT * VV_TILE_LATENT;
        let per_tile = tile_tokens(t);
        let total = tiles * per_tile;

        // folded post-quant + denormalise + embed, FP32, then rounded to the decoder dtype
        let mut embedded = alloc(ctx, &[tiles * pixels, VV_DIM], DType::F32)?;
        let mut args = GemmArgs::new(latents, &resident.embed_weight, &mut embedded);
        args.policy = gemm.clone();
        args.weight_version = Some(ops::WeightVersion::new(0));
        ops::gemm_bias(
            ctx,
            GemmBiasArgs {
                gemm: args,
                bias: &resident.embed_bias,
            },
        )?;
        let embedded_bf16 = alloc(ctx, &[tiles * pixels, VV_DIM], DType::BF16)?;
        ops::quantization::cast(ctx, &embedded, &embedded_bf16)?;

        // tile layout: [latent tokens | 4 register tokens | 1 zero token]
        let x = alloc(ctx, &[total, VV_DIM], DType::BF16)?;
        for tile in 0..tiles {
            let part =
                |first: usize, count: usize| rows(&x, tile * per_tile + first, count, VV_DIM);
            ops::copy_tensor(
                ctx,
                &rows(&embedded_bf16, tile * pixels, pixels, VV_DIM)?,
                &part(0, pixels)?,
            )?;
            ops::copy_tensor(ctx, &resident.registers, &part(pixels, VV_REGISTERS)?)?;
            CudaBuffer::from_tensor(&part(pixels + VV_REGISTERS, 1)?)
                .map_err(Error::Cuda)?
                .zero_async(ctx.stream())
                .map_err(Error::Cuda)?;
        }
        self.weights.run_blocks(|index, block| {
            if index >= blocks {
                return Ok(());
            }
            // attention
            let h = rms(ctx, &x, &block.norm1)?;
            let q = dense_linear_streamed(ctx, &h, &block.q.weight, Some(&block.q.bias), gemm)?;
            let k = dense_linear_streamed(ctx, &h, &block.k.weight, Some(&block.k.bias), gemm)?;
            let v = dense_linear_streamed(ctx, &h, &block.v.weight, Some(&block.v.bias), gemm)?;
            for tensor in [&q, &k] {
                // Q/K RMSNorm has no learned scale here (unit weight), then the half-split rotation, in place
                let heads = tensor.reshape(vec![total, VV_HEADS, VV_HEAD_DIM])?;
                ops::rms_norm_apply_precomputed(
                    ctx,
                    &heads,
                    &resident.ones,
                    rope,
                    NORM_EPS,
                    &heads,
                )?;
            }
            let shape = vec![tiles, per_tile, VV_HEADS, VV_HEAD_DIM];
            let (q4, k4, v4) = (
                q.reshape(shape.clone())?,
                k.reshape(shape.clone())?,
                v.reshape(shape.clone())?,
            );
            let mut attended = alloc(ctx, &shape, DType::BF16)?;
            let mut attn_args = ops::AttentionArgs::new(&q4, &k4, &v4, &mut attended);
            attn_args.policy = attention.clone();
            ops::attention(ctx, attn_args)?;
            let attended = attended.reshape(vec![total, VV_DIM])?;
            let branch = dense_linear_streamed(
                ctx,
                &attended,
                &block.out.weight,
                Some(&block.out.bias),
                gemm,
            )?;
            scaled_residual(ctx, &x, &branch, &block.style1)?;

            // gated SiLU feed-forward
            let h = rms(ctx, &x, &block.norm2)?;
            let gate =
                dense_linear_streamed(ctx, &h, &block.gate.weight, Some(&block.gate.bias), gemm)?;
            let up = dense_linear_streamed(ctx, &h, &block.up.weight, Some(&block.up.bias), gemm)?;
            let activated = ops::pointwise::silu_mul(ctx, &gate, &up)?;
            let branch = dense_linear_streamed(
                ctx,
                &activated,
                &block.down.weight,
                Some(&block.down.bias),
                gemm,
            )?;
            scaled_residual(ctx, &x, &branch, &block.style2)
        })?;
        Ok(x)
    }
}

//! ViT VAE decoder composition: FP32 latent embedding, 24 pre-norm blocks with per-head RMS-normalised, partially rotated
//! attention and scaled residuals, and the pixel-patch head. A batch is a set of 256-pixel tiles that never attend to each other.
use super::refiner::rms;
use super::util::{alloc, dense_linear_bf16};
use crate::cuda_minimax_h3_image::backend::ops::{self, GemmArgs, GemmBiasArgs, GemmPolicy};
use crate::cuda_minimax_h3_image::backend::{AttentionPolicy, CudaBuffer, CudaContext};
use crate::cuda_minimax_h3_image::config::*;
use crate::cuda_minimax_h3_image::model::vae_host::{vae_rope_table, TILE_TOKENS};
use crate::cuda_minimax_h3_image::weights::VaeWeights;
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use std::rc::Rc;

/// The decoder's computation over a fixed batch of tiles. Persistent buffers, graph capture and the tile loop belong to
/// the runner's `PreparedVae`.
pub(in crate::cuda_minimax_h3_image) struct VaeBlocks {
    ctx: Rc<CudaContext>,
    weights: Rc<VaeWeights>,
    /// FP32 `[max_tiles * TILE_TOKENS, 2, VAE_HEAD_DIM / 2]`: the one-tile table repeated for every tile of a batch.
    rope: Tensor,
    max_tiles: usize,
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

impl VaeBlocks {
    /// `max_tiles` is the fixed batch every decode runs at.
    pub fn new(ctx: Rc<CudaContext>, weights: Rc<VaeWeights>, max_tiles: usize) -> Result<Self> {
        let one = vae_rope_table();
        let table: Vec<f32> = (0..max_tiles).flat_map(|_| one.iter().copied()).collect();
        let rope = upload_f32(
            &ctx,
            &table,
            vec![max_tiles * TILE_TOKENS, 2, VAE_HEAD_DIM / 2],
        )?;
        Ok(Self {
            ctx,
            weights,
            rope,
            max_tiles,
        })
    }

    pub fn max_tiles(&self) -> usize {
        self.max_tiles
    }

    /// Arena bytes for one batch: the live set is the feed-forward (`[rows, 8192]` gate and up plus the activation) beside
    /// the residual stream, about 70 KiB per token; the rest is GEMM scratch headroom.
    pub fn arena_bytes(tiles: usize) -> usize {
        tiles * TILE_TOKENS * 96 * 1024 + 128 * 1024 * 1024
    }

    /// Decode `tiles` tiles. `latents` is FP32 `[tiles * VAE_TILE_LATENT^2, LATENT_CHANNELS]` normalised latent tokens
    /// (raster order inside a tile). Returns BF16 `[tiles * TILE_TOKENS, VAE_OUT_DIM]`; the first `VAE_TILE_LATENT^2`
    /// rows of every tile are pixel patches, the rest belong to the suffix tokens.
    pub fn decode(
        &self,
        latents: &Tensor,
        tiles: usize,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
    ) -> Result<Tensor> {
        if tiles == 0 || tiles > self.max_tiles {
            return Err(Error::Other(format!(
                "VAE batch of {tiles} tiles is outside 1..={}",
                self.max_tiles
            )));
        }
        let ctx = &*self.ctx;
        let w = &*self.weights;
        let pixels = VAE_TILE_LATENT * VAE_TILE_LATENT;
        let total = tiles * TILE_TOKENS;

        // folded post-quant + denormalise + embed, FP32
        let mut embedded = alloc(ctx, &[tiles * pixels, VAE_DIM], DType::F32)?;
        let mut args = GemmArgs::new(latents, &w.embed.weight, &mut embedded);
        args.policy = gemm.clone();
        args.weight_version = Some(ops::WeightVersion::new(0));
        ops::gemm_bias(
            ctx,
            GemmBiasArgs {
                gemm: args,
                bias: &w.embed.bias,
            },
        )?;
        let embedded_bf16 = alloc(ctx, &[tiles * pixels, VAE_DIM], DType::BF16)?;
        ops::quantization::cast(ctx, &embedded, &embedded_bf16)?;

        // tile layout: [pixel tokens | 4 register tokens | 1 zero token]
        let x = alloc(ctx, &[total, VAE_DIM], DType::BF16)?;
        for t in 0..tiles {
            let tile = |first: usize, rows: usize| {
                super::util::rows(&x, t * TILE_TOKENS + first, rows, VAE_DIM)
            };
            ops::copy_tensor(
                ctx,
                &super::util::rows(&embedded_bf16, t * pixels, pixels, VAE_DIM)?,
                &tile(0, pixels)?,
            )?;
            ops::copy_tensor(ctx, &w.registers, &tile(pixels, VAE_REGISTERS)?)?;
            CudaBuffer::from_tensor(&tile(pixels + VAE_REGISTERS, 1)?)
                .map_err(Error::Cuda)?
                .zero_async(ctx.stream())
                .map_err(Error::Cuda)?;
        }
        let rope = super::util::byte_view(
            &self.rope,
            0,
            total * VAE_HEAD_DIM,
            vec![total, 2, VAE_HEAD_DIM / 2],
        )?;

        for block in &w.blocks {
            // attention
            let h = rms(ctx, &x, &block.norm1)?;
            let q = dense_linear_bf16(ctx, &h, &block.q.weight, block.q.bias.as_ref(), gemm)?;
            let k = dense_linear_bf16(ctx, &h, &block.k.weight, block.k.bias.as_ref(), gemm)?;
            let v = dense_linear_bf16(ctx, &h, &block.v.weight, block.v.bias.as_ref(), gemm)?;
            for t in [&q, &k] {
                // Q/K RMSNorm has no learned scale here (unit weight), then the half-split rotation, in place
                let heads = t.reshape(vec![total, VAE_HEADS, VAE_HEAD_DIM])?;
                ops::rms_norm_apply_precomputed(ctx, &heads, &w.ones, &rope, NORM_EPS, &heads)?;
            }
            let shape = vec![tiles, TILE_TOKENS, VAE_HEADS, VAE_HEAD_DIM];
            let (q4, k4, v4) = (
                q.reshape(shape.clone())?,
                k.reshape(shape.clone())?,
                v.reshape(shape.clone())?,
            );
            let mut attended = alloc(ctx, &shape, DType::BF16)?;
            let mut attn_args = ops::AttentionArgs::new(&q4, &k4, &v4, &mut attended);
            attn_args.policy = attention.clone();
            ops::attention(ctx, attn_args)?;
            let attended = attended.reshape(vec![total, VAE_DIM])?;
            let branch = dense_linear_bf16(
                ctx,
                &attended,
                &block.out.weight,
                block.out.bias.as_ref(),
                gemm,
            )?;
            scaled_residual(ctx, &x, &branch, &block.style1)?;

            // gated SiLU feed-forward
            let h = rms(ctx, &x, &block.norm2)?;
            let gate =
                dense_linear_bf16(ctx, &h, &block.gate.weight, block.gate.bias.as_ref(), gemm)?;
            let up = dense_linear_bf16(ctx, &h, &block.up.weight, block.up.bias.as_ref(), gemm)?;
            let activated = ops::pointwise::silu_mul(ctx, &gate, &up)?;
            let branch = dense_linear_bf16(
                ctx,
                &activated,
                &block.down.weight,
                block.down.bias.as_ref(),
                gemm,
            )?;
            scaled_residual(ctx, &x, &branch, &block.style2)?;
        }

        let normed = alloc(ctx, &[total, VAE_DIM], DType::BF16)?;
        ops::norm::norm(
            ctx,
            ops::NormArgs::layer(&x, &w.norm_out_weight, &w.norm_out_bias, &normed, NORM_EPS),
        )?;
        dense_linear_bf16(
            ctx,
            &normed,
            &w.proj_out.weight,
            w.proj_out.bias.as_ref(),
            gemm,
        )
    }
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

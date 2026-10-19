//! Wan video prefill and action denoising. Weights and cached text projections
//! are family local; backend calls express only mathematical operations.
use super::backend::binary;
use super::backend::{self, ops, Context};
use super::weights::dit::{Expert, Projections};
use apxinf_core::{DType, Result, Shape, Tensor};

// Preserve residual state across blocks without repeated BF16 rounding. Branch
// projections and gates retain their checkpoint dtype and round before the add.
fn residual(ctx: &Context, x: &Tensor, y: &Tensor) -> Result<Tensor> {
    backend::add(ctx, x, &backend::cast(ctx, y, DType::F32)?)
}
fn norm(ctx: &Context, x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
    ops::layer_norm(
        ctx,
        ops::LayerNormArgs::new(x, weight, bias, &mut out, 1e-6),
    )?;
    backend::cast(ctx, &out, DType::BF16)
}
fn rms(ctx: &Context, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
    ops::rms_norm_rounded(ctx, ops::RmsNormArgs::new(x, weight, &mut out, 1e-6))?;
    Ok(out)
}
fn attend(
    ctx: &Context,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    policy: &ops::AttentionPolicy,
) -> Result<Tensor> {
    let q = backend::cast(ctx, q, DType::F32)?;
    let k = backend::cast(ctx, k, DType::F32)?;
    let v = backend::cast(ctx, v, DType::F32)?;
    let (t, s) = (q.shape().dims()[0], k.shape().dims()[0]);
    let q = q.reshape(vec![1, t, 24, 128])?;
    let k = k.reshape(vec![1, s, 24, 128])?;
    let v = v.reshape(vec![1, s, 24, 128])?;
    let mut out = ctx.allocate_output(q.shape().clone(), DType::F32)?;
    let mut args = ops::AttentionArgs::new(&q, &k, &v, &mut out);
    args.policy = policy.clone();
    ops::attention(ctx, args)?;
    out.reshape(vec![t, 3072])
}

impl Projections {
    fn kv(&self, ctx: &Context, x: &Tensor, policy: &ops::GemmPolicy) -> Result<(Tensor, Tensor)> {
        Ok((
            rms(ctx, &self.k.forward(ctx, x, policy)?, &self.nk)?,
            self.v.forward(ctx, x, policy)?,
        ))
    }
}

pub(super) struct Kv {
    key: Tensor,
    value: Tensor,
}
pub(super) struct ContextCache {
    kv: Vec<Kv>,
}

impl Expert {
    /// Conservative sum of transient tensors for this fixed expert profile.
    /// Repeated action-step budgets also reserve context space, a safe overestimate.
    /// Native execution-owned scratch is separate from the tensor arena.
    pub(super) fn workspace_bytes(&self) -> usize {
        let tokens = if self.video { 98 } else { 32 };
        let prefix = if self.video { 0 } else { 98 };
        let hidden = self.input.weight.shape().dims()[1];
        let attention_width = 24 * 128;
        let context = 129;
        // Text projection/activation and per-layer K, normalized K and V.
        let context_bytes =
            context * hidden * 6 + self.blocks.len() * (context * attention_width * 6 + 3 * 256);
        let layer_bytes: usize = self
            .blocks
            .iter()
            .map(|b| {
                let ff = b.ff0.weight.shape().dims()[1];
                // Residual/normalization/modulation outputs: <= 16 BF16 hidden
                // tensors; FFN projection and activation: two BF16 FF tensors.
                let hidden_bytes = tokens * (32 * hidden + 4 * ff);
                // Q/K/V, rotary, attention outputs and BF16/F32 conversions.
                // Includes concatenated video prefix and context FP32 K/V.
                let attention_bytes =
                    attention_width * (tokens * 48 + (tokens + prefix) * 12 + context * 8);
                // Reserve F32 residual/normalization storage and dtype conversions.
                2 * (hidden_bytes + attention_bytes + 48 * 256)
            })
            .sum();
        let input_bytes = tokens * hidden * self.input.weight.dtype().size_in_bytes()
            + 392 * 48 * 2
            + 98 * 192 * 2
            + if self.video {
                tokens * hidden * 2 + 256
            } else {
                0
            }; // convolution bias add
        let head_bytes = self.head.as_ref().map_or(0, |h| {
            tokens * h.weight.shape().dims()[1] * h.weight.dtype().size_in_bytes()
        });
        context_bytes + layer_bytes + input_bytes + head_bytes + 16 * 256
    }

    pub(super) fn context(
        &self,
        ctx: &Context,
        context: &Tensor,
        policy: &ops::GemmPolicy,
    ) -> Result<ContextCache> {
        let x = self.text0.forward(ctx, context, policy)?;
        let x = backend::activation(ctx, &x, ops::PointwiseActivation::Gelu)?;
        let x = self.text2.forward(ctx, &x, policy)?;
        let kv = self
            .blocks
            .iter()
            .map(|b| {
                b.ca.kv(ctx, &x, policy)
                    .map(|(key, value)| Kv { key, value })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ContextCache { kv })
    }
    fn rotate(&self, ctx: &Context, x: &Tensor) -> Result<Tensor> {
        let mut out = ctx.allocate_output(
            x.shape().clone(),
            if self.video { DType::F32 } else { DType::BF16 },
        )?;
        ops::pairwise_rope(
            ctx,
            ops::PairwiseRopeArgs {
                input: x,
                table: &self.rope,
                output: &mut out,
                heads: 24,
            },
        )?;
        Ok(out)
    }
    pub(super) fn forward(
        &self,
        ctx: &Context,
        input: &Tensor,
        context: &ContextCache,
        video_cache: Option<&[Kv]>,
        step: usize,
        gemm: &ops::GemmPolicy,
        attention: &ops::AttentionPolicy,
    ) -> Result<(Tensor, Vec<Kv>)> {
        let input = if self.video {
            let input = backend::cast(ctx, input, DType::BF16)?;
            let mut patches = ctx.allocate_output(Shape::new(vec![98, 192]), DType::BF16)?;
            ops::image_im2col(ctx, &input, &mut patches, 14, 28, 2, 2, 0, 7, 14)?;
            patches
        } else {
            input.clone()
        };
        let x = if self.video {
            self.input.forward_conv(ctx, &input, gemm)?
        } else {
            self.input.forward(ctx, &input, gemm)?
        };
        let mut x = backend::cast(ctx, &x, DType::F32)?;
        let mut cache = Vec::new();
        for (i, b) in self.blocks.iter().enumerate() {
            let m = &self.modulations[step][i];
            let u = norm(ctx, &x, &self.one, &self.zero)?;
            let u = binary(ctx, &u, &m[1], ops::PointwiseSemantic::BroadcastMultiply)?;
            let u = binary(ctx, &u, &m[0], ops::PointwiseSemantic::BroadcastAdd)?;
            let q = self.rotate(ctx, &rms(ctx, &b.sa.q.forward(ctx, &u, gemm)?, &b.sa.nq)?)?;
            let k = self.rotate(ctx, &rms(ctx, &b.sa.k.forward(ctx, &u, gemm)?, &b.sa.nk)?)?;
            let v = b.sa.v.forward(ctx, &u, gemm)?;
            let (k, v) = if let Some(prefix) = video_cache {
                (
                    ops::concat_rows(ctx, &prefix[i].key, &backend::cast(ctx, &k, DType::F32)?)?,
                    ops::concat_rows(ctx, &prefix[i].value, &v)?,
                )
            } else {
                cache.push(Kv {
                    key: k.clone(),
                    value: v.clone(),
                });
                (k, v)
            };
            let y = attend(ctx, &q, &k, &v, attention)?;
            let y = b.sa.o.forward(ctx, &y, gemm)?;
            let y = binary(ctx, &y, &m[2], ops::PointwiseSemantic::BroadcastMultiply)?;
            x = residual(ctx, &x, &y)?;
            let u = norm(ctx, &x, &b.norm3w, &b.norm3b)?;
            let q = rms(ctx, &b.ca.q.forward(ctx, &u, gemm)?, &b.ca.nq)?;
            let y = attend(ctx, &q, &context.kv[i].key, &context.kv[i].value, attention)?;
            x = residual(ctx, &x, &b.ca.o.forward(ctx, &y, gemm)?)?;
            let u = norm(ctx, &x, &self.one, &self.zero)?;
            let u = binary(ctx, &u, &m[4], ops::PointwiseSemantic::BroadcastMultiply)?;
            let u = binary(ctx, &u, &m[3], ops::PointwiseSemantic::BroadcastAdd)?;
            let u = b.ff0.forward(ctx, &u, gemm)?;
            let u = backend::activation(ctx, &u, ops::PointwiseActivation::Gelu)?;
            let u = b.ff2.forward(ctx, &u, gemm)?;
            let u = binary(ctx, &u, &m[5], ops::PointwiseSemantic::BroadcastMultiply)?;
            x = residual(ctx, &x, &u)?;
        }
        if let Some(head) = &self.head {
            x = head.forward(ctx, &x, gemm)?;
        }
        Ok((x, cache))
    }
}
pub(super) fn sigma(step: usize) -> f64 {
    let u = 1. - step as f64 / 10.;
    5. * u / (1. + 4. * u)
}

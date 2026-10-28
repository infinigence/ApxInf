//! Immutable checkpoint mapping and prepared weight data.
use super::cast_weight;
use super::{host_tensor, validate_tensor, Bias, LinearWeights, Store};
use super::{load_linear, upload_tensor};
use crate::cuda_fastwam::backend::{self, ops, Context};
use apxinf_core::{DType, Error, Result, Tensor};
pub(in crate::cuda_fastwam) struct ProjectionWeights {
    pub(in crate::cuda_fastwam) q: LinearWeights,
    pub(in crate::cuda_fastwam) k: LinearWeights,
    pub(in crate::cuda_fastwam) v: LinearWeights,
    pub(in crate::cuda_fastwam) o: LinearWeights,
    pub(in crate::cuda_fastwam) nq: Tensor,
    pub(in crate::cuda_fastwam) nk: Tensor,
}

impl ProjectionWeights {
    fn load(s: &mut Store, ctx: &Context, p: &str) -> Result<Self> {
        Ok(Self {
            q: load_linear(&mut *s, ctx, &format!("{p}.q"), Bias::Required)?,
            k: load_linear(&mut *s, ctx, &format!("{p}.k"), Bias::Required)?,
            v: load_linear(&mut *s, ctx, &format!("{p}.v"), Bias::Required)?,
            o: load_linear(&mut *s, ctx, &format!("{p}.o"), Bias::Required)?,
            nq: upload_tensor(&mut *s, ctx, &format!("{p}.norm_q.weight"))?,
            nk: upload_tensor(&mut *s, ctx, &format!("{p}.norm_k.weight"))?,
        })
    }
}

pub(in crate::cuda_fastwam) struct BlockWeights {
    pub(in crate::cuda_fastwam) sa: ProjectionWeights,
    pub(in crate::cuda_fastwam) ca: ProjectionWeights,
    pub(in crate::cuda_fastwam) norm3w: Tensor,
    pub(in crate::cuda_fastwam) norm3b: Tensor,
    pub(in crate::cuda_fastwam) ff0: LinearWeights,
    pub(in crate::cuda_fastwam) ff2: LinearWeights,
    pub(in crate::cuda_fastwam) modulation: Tensor,
}

pub(in crate::cuda_fastwam) struct ExpertWeights {
    pub(in crate::cuda_fastwam) blocks: Vec<BlockWeights>,
    pub(in crate::cuda_fastwam) text0: LinearWeights,
    pub(in crate::cuda_fastwam) text2: LinearWeights,
    pub(in crate::cuda_fastwam) input: LinearWeights,
    pub(in crate::cuda_fastwam) head: Option<LinearWeights>,
    pub(in crate::cuda_fastwam) one: Tensor,
    pub(in crate::cuda_fastwam) zero: Tensor,
    pub(in crate::cuda_fastwam) rope: ops::PairwiseRopeTable,
    pub(in crate::cuda_fastwam) video: bool,
}

impl ExpertWeights {
    pub(in crate::cuda_fastwam) fn load(
        s: &mut Store,
        ctx: &Context,
        video: bool,
    ) -> Result<(Self, TimeWeights)> {
        let name = if video { "video" } else { "action" };
        let p = format!("model.mot.mixtures.{name}");
        let width = if video { 3072 } else { 1024 };
        let mut blocks = Vec::new();
        for i in 0..30 {
            let b = format!("{p}.blocks.{i}");
            blocks.push(BlockWeights {
                sa: ProjectionWeights::load(s, ctx, &format!("{b}.self_attn"))?,
                ca: ProjectionWeights::load(s, ctx, &format!("{b}.cross_attn"))?,
                norm3w: upload_tensor(&mut *s, ctx, &format!("{b}.norm3.weight"))?,
                norm3b: upload_tensor(&mut *s, ctx, &format!("{b}.norm3.bias"))?,
                ff0: load_linear(&mut *s, ctx, &format!("{b}.ffn.0"), Bias::Required)?,
                ff2: load_linear(&mut *s, ctx, &format!("{b}.ffn.2"), Bias::Required)?,
                modulation: upload_tensor(&mut *s, ctx, &format!("{b}.modulation"))?
                    .reshape(vec![1, 6 * width])?,
            });
        }
        for b in &blocks {
            for projections in [&b.sa, &b.ca] {
                for projection in [&projections.q, &projections.k, &projections.v] {
                    projection.validate("Wan QKV", width, 3072)?;
                }
                projections
                    .o
                    .validate("Wan attention output", 3072, width)?;
                validate_tensor("Wan Q norm", &projections.nq, &[3072], DType::BF16)?;
                validate_tensor("Wan K norm", &projections.nk, &[3072], DType::BF16)?;
            }
            let ff = if video { 14336 } else { 4096 };
            b.ff0.validate("Wan FFN up", width, ff)?;
            b.ff2.validate("Wan FFN down", ff, width)?;
            validate_tensor("Wan cross norm", &b.norm3w, &[width], DType::BF16)?;
            validate_tensor("Wan cross norm bias", &b.norm3b, &[width], DType::BF16)?;
        }
        // Keep immutable affine data in the residual state's dtype. This avoids
        // repeating weight conversions inside every captured denoising step.
        for b in &mut blocks {
            b.norm3w = cast_weight(ctx, &b.norm3w, DType::F32)?;
            b.norm3b = cast_weight(ctx, &b.norm3b, DType::F32)?;
        }
        let mut input = if video {
            let raw = s.take(&format!("{p}.patch_embedding.weight"))?;
            if raw.shape().dims() != [3072, 48, 1, 2, 2] {
                return Err(Error::Other(
                    "Fast-WAM video patch weight shape mismatch".into(),
                ));
            }
            let raw = raw.to_f32_vec()?;
            let mut transposed = vec![0.; raw.len()];
            for n in 0..width {
                for c in 0..48 {
                    for y in 0..2 {
                        for x in 0..2 {
                            transposed[((y * 2 + x) * 48 + c) * width + n] =
                                raw[((n * 48 + c) * 2 + y) * 2 + x];
                        }
                    }
                }
            }
            LinearWeights {
                weight: backend::upload(
                    ctx,
                    &host_tensor(vec![192, width], &transposed, DType::BF16)?,
                )?,
                bias: Some(upload_tensor(
                    &mut *s,
                    ctx,
                    &format!("{p}.patch_embedding.bias"),
                )?),
            }
        } else {
            load_linear(&mut *s, ctx, &format!("{p}.action_encoder"), Bias::Required)?
        };
        let mut time0 = load_linear(
            &mut *s,
            ctx,
            &format!("{p}.time_embedding.0"),
            Bias::Required,
        )?;
        let mut time2 = load_linear(
            &mut *s,
            ctx,
            &format!("{p}.time_embedding.2"),
            Bias::Required,
        )?;
        let mut timeproj = load_linear(
            &mut *s,
            ctx,
            &format!("{p}.time_projection.1"),
            Bias::Required,
        )?;
        input.validate("Wan input", if video { 192 } else { 7 }, width)?;
        if !video {
            // Keep Euler state in F32 through its small input projection.
            // Validate the original BF16 checkpoint before exact promotion.
            input.weight = cast_weight(ctx, &input.weight, DType::F32)?;
            input.bias = input
                .bias
                .as_ref()
                .map(|bias| cast_weight(ctx, bias, DType::F32))
                .transpose()?;
        }
        time0.validate("Wan time input", 256, width)?;
        time2.validate("Wan time hidden", width, width)?;
        timeproj.validate("Wan time modulation", width, 6 * width)?;
        if video {
            for projection in [&mut time0, &mut time2, &mut timeproj] {
                projection.weight = cast_weight(ctx, &projection.weight, DType::F32)?;
                projection.bias = projection
                    .bias
                    .as_ref()
                    .map(|b| cast_weight(ctx, b, DType::F32))
                    .transpose()?;
            }
        }
        let one = vector(ctx, &vec![1.; width], DType::F32)?;
        let zero = vector(ctx, &vec![0.; width], DType::F32)?;
        let tokens = if video { 98 } else { 32 };
        let mut coefficients = Vec::new();
        for pos in 0..tokens {
            if video {
                for (coordinate, pairs) in [(0, 22), (pos / 14, 21), (pos % 14, 21)] {
                    for i in 0..pairs {
                        let phase = coordinate as f64 * 10000f64.powf(-(i as f64) / pairs as f64);
                        coefficients.push((phase.cos(), phase.sin()));
                    }
                }
            } else {
                for i in 0..64 {
                    let phase = pos as f64 * 10000f64.powf(-(i as f64) / 64.);
                    coefficients.push((phase.cos(), phase.sin()));
                }
            }
        }
        let text0 = load_linear(
            &mut *s,
            ctx,
            &format!("{p}.text_embedding.0"),
            Bias::Required,
        )?;
        let text2 = load_linear(
            &mut *s,
            ctx,
            &format!("{p}.text_embedding.2"),
            Bias::Required,
        )?;
        text0.validate("Wan text input", 4096, width)?;
        text2.validate("Wan text hidden", width, width)?;
        let mut head = if video {
            None
        } else {
            Some(load_linear(
                &mut *s,
                ctx,
                &format!("{p}.head"),
                Bias::Required,
            )?)
        };
        if let Some(head) = &mut head {
            head.validate("Wan action output", width, 7)?;
            // Preserve the F32 residual through the small velocity projection.
            // Checkpoint values are BF16; their F32 promotion is exact.
            head.weight = cast_weight(ctx, &head.weight, DType::F32)?;
            head.bias = head
                .bias
                .as_ref()
                .map(|bias| cast_weight(ctx, bias, DType::F32))
                .transpose()?;
        }
        Ok((
            Self {
                blocks,
                text0,
                text2,
                input,
                head,
                one,
                zero,
                rope: ops::PairwiseRopeTable::new(ctx, tokens, 128, &coefficients)?,
                video,
            },
            TimeWeights {
                time0,
                time2,
                timeproj,
            },
        ))
    }
}

fn vector(ctx: &Context, values: &[f32], dtype: DType) -> Result<Tensor> {
    backend::upload(ctx, &host_tensor(vec![values.len()], values, dtype)?)
}

pub(in crate::cuda_fastwam) struct TimeWeights {
    pub time0: LinearWeights,
    pub time2: LinearWeights,
    pub timeproj: LinearWeights,
}

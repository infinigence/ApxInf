//! Immutable checkpoint mapping and prepared weight data.
use super::{host_tensor, validate_tensor, Bias, Linear, Store};
use crate::fastwam::backend::{self, binary, ops, Context};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
pub(in crate::fastwam) struct Projections {
    pub(in crate::fastwam) q: Linear,
    pub(in crate::fastwam) k: Linear,
    pub(in crate::fastwam) v: Linear,
    pub(in crate::fastwam) o: Linear,
    pub(in crate::fastwam) nq: Tensor,
    pub(in crate::fastwam) nk: Tensor,
}

impl Projections {
    fn load(s: &mut Store, ctx: &Context, p: &str) -> Result<Self> {
        Ok(Self {
            q: s.linear(ctx, &format!("{p}.q"), Bias::Required)?,
            k: s.linear(ctx, &format!("{p}.k"), Bias::Required)?,
            v: s.linear(ctx, &format!("{p}.v"), Bias::Required)?,
            o: s.linear(ctx, &format!("{p}.o"), Bias::Required)?,
            nq: s.upload(ctx, &format!("{p}.norm_q.weight"))?,
            nk: s.upload(ctx, &format!("{p}.norm_k.weight"))?,
        })
    }
}

pub(in crate::fastwam) struct Block {
    pub(in crate::fastwam) sa: Projections,
    pub(in crate::fastwam) ca: Projections,
    pub(in crate::fastwam) norm3w: Tensor,
    pub(in crate::fastwam) norm3b: Tensor,
    pub(in crate::fastwam) ff0: Linear,
    pub(in crate::fastwam) ff2: Linear,
    pub(in crate::fastwam) modulation: Tensor,
}

pub(in crate::fastwam) struct Expert {
    pub(in crate::fastwam) blocks: Vec<Block>,
    pub(in crate::fastwam) text0: Linear,
    pub(in crate::fastwam) text2: Linear,
    pub(in crate::fastwam) input: Linear,
    pub(in crate::fastwam) head: Option<Linear>,
    pub(in crate::fastwam) one: Tensor,
    pub(in crate::fastwam) zero: Tensor,
    pub(in crate::fastwam) rope: ops::PairwiseRopeTable,
    pub(in crate::fastwam) video: bool,
    /// [step][block][shift/scale/gate]. Immutable schedule prepared at load.
    pub(in crate::fastwam) modulations: Vec<Vec<Vec<Tensor>>>,
}

impl Expert {
    pub(in crate::fastwam) fn load(
        s: &mut Store,
        ctx: &Context,
        video: bool,
        gemm: &ops::GemmPolicy,
        timesteps: &[f32],
    ) -> Result<Self> {
        let name = if video { "video" } else { "action" };
        let p = format!("model.mot.mixtures.{name}");
        let width = if video { 3072 } else { 1024 };
        let mut blocks = Vec::new();
        for i in 0..30 {
            let b = format!("{p}.blocks.{i}");
            blocks.push(Block {
                sa: Projections::load(s, ctx, &format!("{b}.self_attn"))?,
                ca: Projections::load(s, ctx, &format!("{b}.cross_attn"))?,
                norm3w: s.upload(ctx, &format!("{b}.norm3.weight"))?,
                norm3b: s.upload(ctx, &format!("{b}.norm3.bias"))?,
                ff0: s.linear(ctx, &format!("{b}.ffn.0"), Bias::Required)?,
                ff2: s.linear(ctx, &format!("{b}.ffn.2"), Bias::Required)?,
                modulation: s
                    .upload(ctx, &format!("{b}.modulation"))?
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
            b.norm3w = backend::cast(ctx, &b.norm3w, DType::F32)?;
            b.norm3b = backend::cast(ctx, &b.norm3b, DType::F32)?;
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
            Linear {
                weight: backend::upload(
                    ctx,
                    &host_tensor(vec![192, width], &transposed, DType::BF16)?,
                )?,
                bias: Some(s.upload(ctx, &format!("{p}.patch_embedding.bias"))?),
            }
        } else {
            s.linear(ctx, &format!("{p}.action_encoder"), Bias::Required)?
        };
        let mut time0 = s.linear(ctx, &format!("{p}.time_embedding.0"), Bias::Required)?;
        let mut time2 = s.linear(ctx, &format!("{p}.time_embedding.2"), Bias::Required)?;
        let mut timeproj = s.linear(ctx, &format!("{p}.time_projection.1"), Bias::Required)?;
        input.validate("Wan input", if video { 192 } else { 7 }, width)?;
        if !video {
            // Keep Euler state in F32 through its small input projection.
            // Validate the original BF16 checkpoint before exact promotion.
            input.weight = backend::cast(ctx, &input.weight, DType::F32)?;
            input.bias = input
                .bias
                .as_ref()
                .map(|bias| backend::cast(ctx, bias, DType::F32))
                .transpose()?;
        }
        time0.validate("Wan time input", 256, width)?;
        time2.validate("Wan time hidden", width, width)?;
        timeproj.validate("Wan time modulation", width, 6 * width)?;
        if video {
            for projection in [&mut time0, &mut time2, &mut timeproj] {
                projection.weight = backend::cast(ctx, &projection.weight, DType::F32)?;
                projection.bias = projection
                    .bias
                    .as_ref()
                    .map(|b| backend::cast(ctx, b, DType::F32))
                    .transpose()?;
            }
        }
        let one = vector(ctx, &vec![1.; width], DType::F32)?;
        let zero = vector(ctx, &vec![0.; width], DType::F32)?;
        let modulation_one = vector(ctx, &vec![1.; width], DType::BF16)?;
        let mut modulations = Vec::new();
        for step in 0..if video { 1 } else { 10 } {
            let t = if video {
                0.
            } else {
                half::bf16::from_f32(timesteps[step]).to_f32() as f64
            };
            let mut embedding = Vec::with_capacity(256);
            for trig in 0..2 {
                for i in 0..128 {
                    let phase = t * 10000f64.powf(-(i as f64) / 128.);
                    embedding.push(if trig == 0 {
                        phase.cos() as f32
                    } else {
                        phase.sin() as f32
                    });
                }
            }
            let emb = backend::upload(
                ctx,
                &host_tensor(vec![1, 256], &embedding, time0.weight.dtype())?,
            )?;
            let t = time0.forward(ctx, &emb, gemm)?;
            let t = backend::activation(ctx, &t, ops::PointwiseActivation::Silu)?;
            let t = time2.forward(ctx, &t, gemm)?;
            let t = backend::activation(ctx, &t, ops::PointwiseActivation::Silu)?;
            let t = backend::cast(ctx, &timeproj.forward(ctx, &t, gemm)?, DType::BF16)?;
            let mut layers = Vec::new();
            for b in &blocks {
                let sum = binary(ctx, &b.modulation, &t, ops::PointwiseSemantic::Add)?;
                let mut parts = Vec::new();
                for j in 0..6 {
                    let part = slice(&sum, j * width, width)?;
                    parts.push(if j == 1 || j == 4 {
                        binary(
                            ctx,
                            &part.reshape(vec![1, width])?,
                            &modulation_one,
                            ops::PointwiseSemantic::BroadcastAdd,
                        )?
                        .reshape(vec![width])?
                    } else {
                        part
                    });
                }
                layers.push(parts);
            }
            modulations.push(layers);
        }
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
        let text0 = s.linear(ctx, &format!("{p}.text_embedding.0"), Bias::Required)?;
        let text2 = s.linear(ctx, &format!("{p}.text_embedding.2"), Bias::Required)?;
        text0.validate("Wan text input", 4096, width)?;
        text2.validate("Wan text hidden", width, width)?;
        let mut head = if video {
            None
        } else {
            Some(s.linear(ctx, &format!("{p}.head"), Bias::Required)?)
        };
        if let Some(head) = &mut head {
            head.validate("Wan action output", width, 7)?;
            // Preserve the F32 residual through the small velocity projection.
            // Checkpoint values are BF16; their F32 promotion is exact.
            head.weight = backend::cast(ctx, &head.weight, DType::F32)?;
            head.bias = head
                .bias
                .as_ref()
                .map(|bias| backend::cast(ctx, bias, DType::F32))
                .transpose()?;
        }
        Ok(Self {
            blocks,
            text0,
            text2,
            input,
            head,
            one,
            zero,
            rope: ops::PairwiseRopeTable::new(ctx, tokens, 128, &coefficients)?,
            video,
            modulations,
        })
    }
}

fn vector(ctx: &Context, values: &[f32], dtype: DType) -> Result<Tensor> {
    backend::upload(ctx, &host_tensor(vec![values.len()], values, dtype)?)
}

fn slice(x: &Tensor, start: usize, count: usize) -> Result<Tensor> {
    backend::DeviceBuffer::from_tensor(x)
        .and_then(|b| {
            b.view(
                start * x.dtype().size_in_bytes(),
                count * x.dtype().size_in_bytes(),
            )
        })
        .and_then(|b| b.as_tensor(Shape::new(vec![count]), x.dtype()))
        .map_err(Error::Cuda)
}

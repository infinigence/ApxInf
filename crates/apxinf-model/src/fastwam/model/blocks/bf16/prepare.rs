//! Prepare fixed time modulation through the same BF16 computation helpers.
use super::{
    dit::Expert,
    ops::{self as compute, binary},
};
use crate::fastwam::{
    backend::{self, ops, Context},
    weights::{
        cast,
        dit::{Expert as ExpertWeights, TimeWeights},
        host_tensor,
    },
};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
impl Expert {
    pub(in crate::fastwam) fn new(
        ctx: &Context,
        loaded: (ExpertWeights, TimeWeights),
        gemm: &ops::GemmPolicy,
        timesteps: &[f32],
    ) -> Result<Self> {
        let (
            weights,
            TimeWeights {
                time0,
                time2,
                timeproj,
            },
        ) = loaded;
        let video = weights.video;
        let width = if video { 3072 } else { 1024 };
        let blocks = &weights.blocks;
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
            let t = compute::activation(ctx, &t, ops::PointwiseActivation::Silu)?;
            let t = time2.forward(ctx, &t, gemm)?;
            let t = compute::activation(ctx, &t, ops::PointwiseActivation::Silu)?;
            let t = cast(ctx, &timeproj.forward(ctx, &t, gemm)?, DType::BF16)?;
            let mut layers = Vec::new();
            for b in blocks {
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
        Ok(Self {
            weights,
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

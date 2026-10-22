//! Full joint denoising: both modalities traverse the same DiT on every step.
use super::super::{
    backend::{allocate, binary, ops, transfers, Context, Policies},
    schedule::Schedule,
    weights::dit::DitWeights,
};
use super::{
    dit::{DitContext, DitWorkspace, VideoCache},
    dit_block::rows,
    unipc::UniPc,
};
use apxinf_core::{DType, Error, Result, Shape, Tensor};

const VIDEO: usize = 16 * 2 * 44 * 80;

struct TimeInputs {
    modulation: Tensor,
    action: Tensor,
}
fn times(ctx: &Context, video: &Schedule, action: &Schedule) -> Result<Vec<TimeInputs>> {
    if video.steps.len() != action.steps.len() {
        return Err(Error::Other(
            "joint schedules must have the same number of evaluations".into(),
        ));
    }
    // The reference action encoder materializes FP32 exp/multiply/sin/cos on
    // CUDA. Host libm differs at BF16 rounding boundaries, so prepare these
    // fixed constants once on the device with the same operation boundaries.
    let exponents: Vec<_> = (0..2560)
        .map(|i| -(i as f32) * (10000f32.ln() / 2560.))
        .collect();
    let exponent = transfers::to_cuda(
        &Tensor::from_f32(vec![1, 2560], &exponents)?,
        ctx.device_id(),
    )?;
    let float = || ctx.allocate_output(Shape::new(vec![1, 2560]), DType::F32);
    let mut frequency = float()?;
    let mut angles = float()?;
    let mut activation = float()?;
    let mut rounded = allocate(ctx, &[1, 2560])?;
    ops::pointwise(
        ctx,
        ops::PointwiseArgs::new(ops::PointwiseSemantic::Exp, &exponent, &mut frequency),
    )?;
    let mut result = Vec::with_capacity(video.steps.len());
    for (v, a) in video.steps.iter().zip(&action.steps) {
        let mut modulation = Vec::with_capacity(512);
        for t in [v.timestep, a.timestep] {
            let angles: Vec<_> = (0..128)
                .map(|i| t as f64 * 10000f64.powf(-(i as f64) / 128.))
                .collect();
            modulation.extend(angles.iter().map(|x| half::bf16::from_f32(x.cos() as f32)));
            modulation.extend(angles.iter().map(|x| half::bf16::from_f32(x.sin() as f32)));
        }
        let mut scale =
            ops::PointwiseArgs::new(ops::PointwiseSemantic::Scale, &frequency, &mut angles);
        scale.dt = a.timestep as f32;
        ops::pointwise(ctx, scale)?;
        let tau = allocate(ctx, &[48, 2560])?;
        for (offset, semantic) in [ops::PointwiseSemantic::Sin, ops::PointwiseSemantic::Cos]
            .into_iter()
            .enumerate()
        {
            ops::pointwise(
                ctx,
                ops::PointwiseArgs::new(semantic, &angles, &mut activation),
            )?;
            ops::quantization(
                ctx,
                ops::QuantizationArgs::new(
                    ops::QuantizationSemantic::CastF32ToBf16,
                    &activation,
                    &mut rounded,
                ),
            )?;
            for row in 0..24 {
                transfers::copy_cuda_to_cuda(ctx, &rounded, &rows(&tau, row * 2 + offset, 1)?)?;
            }
        }
        result.push(TimeInputs {
            modulation: transfers::to_cuda(
                &Tensor::from_bf16(vec![2, 256], &modulation)?,
                ctx.device_id(),
            )?,
            action: tau.reshape(vec![24, 5120])?,
        });
    }
    Ok(result)
}

pub(in crate::dreamzero) struct JointWorkspace {
    pub joint_noise: Tensor,
    pub state: Tensor,
    pub video: UniPc,
    pub action: UniPc,
    start_frame: usize,
    dit: DitWorkspace,
    time: Vec<TimeInputs>,
    conditional_video: Tensor,
    conditional_action: Tensor,
    delta: Tensor,
    scaled: Tensor,
    guided: Tensor,
}
impl JointWorkspace {
    pub fn new(
        ctx: &Context,
        start_frame: usize,
        history_rows: usize,
        video: Schedule,
        action: Schedule,
    ) -> Result<Self> {
        let time = times(ctx, &video, &action)?;
        let image = || allocate(ctx, &[1, VIDEO]);
        Ok(Self {
            joint_noise: allocate(ctx, &[VIDEO + 24 * 32])?,
            state: allocate(ctx, &[1, 64])?,
            video: UniPc::new(ctx, [1, VIDEO], video)?,
            action: UniPc::new(ctx, [24, 32], action)?,
            start_frame,
            dit: DitWorkspace::new(ctx, 2, start_frame, history_rows, true)?,
            time,
            conditional_video: image()?,
            conditional_action: allocate(ctx, &[24, 32])?,
            delta: image()?,
            scaled: image()?,
            guided: image()?,
        })
    }
    pub fn sample(
        &mut self,
        ctx: &Context,
        p: &Policies,
        w: &DitWeights,
        positive: &DitContext,
        negative: &DitContext,
        history_positive: &VideoCache,
        history_negative: &VideoCache,
        conditioning: &Tensor,
    ) -> Result<()> {
        if conditioning.shape().dims() != [20, 9, 44, 80] {
            return Err(Error::Other(
                "DreamZero initial conditioning must be [20,9,44,80]".into(),
            ));
        }
        let noise = self.joint_noise.reshape(vec![VIDEO + 768, 1])?;
        transfers::copy_cuda_to_cuda(
            ctx,
            &rows(&noise, 0, VIDEO)?.reshape(vec![1, VIDEO])?,
            &self.video.sample,
        )?;
        transfers::copy_cuda_to_cuda(
            ctx,
            &rows(&noise, VIDEO, 768)?.reshape(vec![24, 32])?,
            &self.action.sample,
        )?;
        transfers::copy_cuda_to_cuda(ctx, &self.state, &self.dit.state)?;
        let source = conditioning.reshape(vec![20 * 9, 44 * 80])?;
        let target = self.dit.conditioning.reshape(vec![20 * 2, 44 * 80])?;
        for c in 0..20 {
            transfers::copy_cuda_to_cuda(
                ctx,
                &rows(&source, c * 9 + self.start_frame, 2)?,
                &rows(&target, c * 2, 2)?,
            )?;
        }
        for (i, t) in self.time.iter().enumerate() {
            transfers::copy_cuda_to_cuda(
                ctx,
                &self.video.sample.reshape(vec![16, 2, 44, 80])?,
                &self.dit.video,
            )?;
            transfers::copy_cuda_to_cuda(ctx, &self.action.sample, &self.dit.action)?;
            transfers::copy_cuda_to_cuda(ctx, &t.modulation, &self.dit.time_sinusoidal)?;
            transfers::copy_cuda_to_cuda(ctx, &t.action, &self.dit.action_sinusoidal)?;
            self.dit
                .forward(ctx, p, w, positive, Some(history_positive), None)?;
            transfers::copy_cuda_to_cuda(
                ctx,
                &self.dit.video_velocity.reshape(vec![1, VIDEO])?,
                &self.conditional_video,
            )?;
            transfers::copy_cuda_to_cuda(ctx, &self.dit.action_velocity, &self.conditional_action)?;
            self.dit
                .forward(ctx, p, w, negative, Some(history_negative), None)?;
            let unconditioned = self.dit.video_velocity.reshape(vec![1, VIDEO])?;
            binary(
                ctx,
                ops::PointwiseSemantic::Subtract,
                &self.conditional_video,
                &unconditioned,
                &mut self.delta,
            )?;
            let mut scale = ops::PointwiseArgs::new(
                ops::PointwiseSemantic::Scale,
                &self.delta,
                &mut self.scaled,
            );
            scale.dt = 5.;
            ops::pointwise(ctx, scale)?;
            binary(
                ctx,
                ops::PointwiseSemantic::Add,
                &unconditioned,
                &self.scaled,
                &mut self.guided,
            )?;
            self.video.step(ctx, &self.guided, i)?;
            self.action.step(ctx, &self.conditional_action, i)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires independently captured official CUDA time embeddings"]
    fn checkpoint_time_inputs_match_cuda_reference() {
        let path = std::env::var("APXINF_DREAMZERO_TIME_FIXTURE").unwrap();
        let fixture = apxinf_loader::safetensors::load_native(std::path::Path::new(&path))
            .unwrap()
            .0;
        let ctx = Context::new_with_autotune(0, false).unwrap();
        let schedule = Schedule::standard().unwrap();
        let inputs = times(&ctx, &schedule, &schedule).unwrap();
        for (i, input) in inputs.iter().enumerate() {
            for (kind, tensor) in [("modulation", &input.modulation), ("action", &input.action)] {
                ctx.synchronize().unwrap();
                let actual = transfers::to_cpu(tensor).unwrap().to_f32_vec().unwrap();
                let expected = fixture[&format!("{kind}.{i}")].to_f32_vec().unwrap();
                let error: f64 = actual
                    .iter()
                    .zip(&expected)
                    .map(|(&a, &b)| f64::from(a - b).powi(2))
                    .sum();
                let energy: f64 = expected.iter().map(|&x| f64::from(x).powi(2)).sum();
                let relative = (error / energy).sqrt();
                eprintln!("time {kind}.{i} relative_l2={relative}");
                assert_eq!(
                    actual, expected,
                    "{kind}.{i}: fixed time inputs differ from the pinned CUDA reference"
                );
            }
        }
    }
}

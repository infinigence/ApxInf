use super::super::{
    backend::{allocate, binary, ops, transfers, Context},
    schedule::{Schedule, Update},
};
#[cfg(test)]
use crate::dreamzero::backend;
use apxinf_core::{Error, Result, Tensor};

fn scale(ctx: &Context, x: &Tensor, k: f32, out: &mut Tensor) -> Result<()> {
    let mut args = ops::PointwiseArgs::new(ops::PointwiseSemantic::Scale, x, out);
    // The official schedule stores coefficients as zero-dimensional CUDA
    // FP32 tensors. TensorIterator converts those operands to the BF16 common
    // dtype before multiplication (unlike a wrapped Python float scalar).
    args.dt = half::bf16::from_f32(k).to_f32();
    ops::pointwise(ctx, args)
}

#[cfg(test)]
#[path = "unipc_fixture.rs"]
mod golden;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires all 32 fixed official full-model video/action UniPC captures"]
    fn checkpoint_unipc_full_velocity_trajectory() {
        let root =
            std::path::PathBuf::from(std::env::var("APXINF_DREAMZERO_STEP_REFERENCE").unwrap());
        let ctx = Context::new_with_autotune(0, false).unwrap();
        let read = |index: usize, suffix: &str| {
            apxinf_loader::safetensors::load_native(
                &root.join(format!("step.{index}.{suffix}.safetensors")),
            )
            .unwrap()
            .0
        };
        for modality in 0..2 {
            let initial = read(modality, "input");
            let count = initial["value.kwargs.sample"].shape().numel();
            assert_eq!(count, if modality == 0 { 112640 } else { 768 });
            let shape = vec![1, count];
            let mut sampler = UniPc::new(&ctx, [1, count], Schedule::standard().unwrap()).unwrap();
            transfers::copy_cpu_to_cuda(
                &initial["value.kwargs.sample"]
                    .reshape(shape.clone())
                    .unwrap(),
                &sampler.sample,
            )
            .unwrap();
            for step in 0..16 {
                let input = read(step * 2 + modality, "input");
                let output = read(step * 2 + modality, "output");
                let velocity = transfers::to_cuda(
                    &input["value.kwargs.model_output"]
                        .reshape(shape.clone())
                        .unwrap(),
                    0,
                )
                .unwrap();
                sampler.step(&ctx, &velocity, step).unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(
                    transfers::to_cpu(&sampler.sample)
                        .unwrap()
                        .to_f32_vec()
                        .unwrap(),
                    output["value.0"].to_f32_vec().unwrap(),
                    "modality {modality}, step {step}"
                );
            }
        }
    }

    fn host(values: &[f32]) -> Tensor {
        Tensor::from_bf16(
            vec![2, 8],
            &values
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }
    #[test]
    fn unipc_matches_official_every_step_and_captured_reset() {
        let ctx = Context::new(0).unwrap();
        let schedule = Schedule::standard().unwrap();
        for (i, s) in schedule.steps.iter().enumerate() {
            assert_eq!(s.timestep, golden::TIMES[i]);
            assert_eq!(s.sigma, golden::SIGMAS[i]);
        }
        let mut sampler = UniPc::new(&ctx, [2, 8], schedule).unwrap();
        let velocities: Vec<_> = golden::VELOCITIES
            .iter()
            .map(|v| transfers::to_cuda(&host(v), 0).unwrap())
            .collect();
        transfers::copy_cpu_to_cuda(&host(&golden::INITIAL), &sampler.sample).unwrap();
        for (i, v) in velocities.iter().enumerate() {
            sampler.step(&ctx, v, i).unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                transfers::to_cpu(&sampler.sample)
                    .unwrap()
                    .to_f32_vec()
                    .unwrap(),
                golden::OUTPUTS[i],
                "UniPC step {i}"
            );
        }
        let graph = backend::capture(&ctx, || {
            for (i, v) in velocities.iter().enumerate() {
                sampler.step(&ctx, v, i)?;
            }
            Ok(())
        })
        .unwrap();
        ctx.synchronize().unwrap();
        for _ in 0..3 {
            transfers::copy_cpu_to_cuda(&host(&golden::INITIAL), &sampler.sample).unwrap();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                transfers::to_cpu(&sampler.sample)
                    .unwrap()
                    .to_f32_vec()
                    .unwrap(),
                golden::OUTPUTS[15]
            );
        }
        transfers::copy_cpu_to_cuda(&host(&golden::INITIAL.map(|v| -v)), &sampler.sample).unwrap();
        for (v, values) in velocities.iter().zip(golden::VELOCITIES) {
            transfers::copy_cpu_to_cuda(&host(&values.map(|v| -v)), v).unwrap();
        }
        let output = sampler.sample.clone();
        drop(sampler);
        drop(velocities);
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(
            transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap(),
            golden::OUTPUTS[15].map(|v| -v)
        );
    }
}
fn subtract(ctx: &Context, x: &Tensor, y: &Tensor, out: &mut Tensor) -> Result<()> {
    binary(ctx, ops::PointwiseSemantic::Subtract, x, y, out)
}
fn divide(ctx: &Context, x: &Tensor, k: f32, out: &mut Tensor) -> Result<()> {
    let mut args = ops::PointwiseArgs::new(ops::PointwiseSemantic::ScalarDivide, x, out);
    args.dt = half::bf16::from_f32(k).to_f32();
    ops::pointwise(ctx, args)
}

/// Reusable BF16 state. Every arithmetic materialization is explicit; neither
/// scheduler history nor tensor computation crosses to the host during replay.
pub(in crate::dreamzero) struct UniPc {
    pub sample: Tensor,
    schedule: Schedule,
    converted: Tensor,
    previous: Tensor,
    older: Tensor,
    last_sample: Tensor,
    corrected: Tensor,
    next: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d: Tensor,
    e: Tensor,
    f: Tensor,
}
impl UniPc {
    pub fn new(ctx: &Context, shape: [usize; 2], schedule: Schedule) -> Result<Self> {
        let alloc = || allocate(ctx, &shape);
        Ok(Self {
            sample: alloc()?,
            schedule,
            converted: alloc()?,
            previous: alloc()?,
            older: alloc()?,
            last_sample: alloc()?,
            corrected: alloc()?,
            next: alloc()?,
            a: alloc()?,
            b: alloc()?,
            c: alloc()?,
            d: alloc()?,
            e: alloc()?,
            f: alloc()?,
        })
    }
    fn base(
        ctx: &Context,
        x: &Tensor,
        m: &Tensor,
        u: Update,
        a: &mut Tensor,
        b: &mut Tensor,
        c: &mut Tensor,
    ) -> Result<()> {
        scale(ctx, x, u.ratio, a)?;
        scale(ctx, m, u.weight, b)?;
        subtract(ctx, a, b, c)
    }
    /// Steps must be emitted in order when preparing/eagerly running a full
    /// sampling call. Step zero overwrites all subsequently read history.
    pub fn step(&mut self, ctx: &Context, velocity: &Tensor, index: usize) -> Result<()> {
        let s = *self
            .schedule
            .steps
            .get(index)
            .ok_or_else(|| Error::Other("DreamZero UniPC step out of range".into()))?;
        if velocity.shape() != self.sample.shape() {
            return Err(Error::Other(
                "DreamZero UniPC velocity shape mismatch".into(),
            ));
        }
        scale(ctx, velocity, s.sigma, &mut self.a)?;
        subtract(ctx, &self.sample, &self.a, &mut self.converted)?;
        if let Some(u) = s.corrector {
            Self::base(
                ctx,
                &self.last_sample,
                &self.previous,
                u,
                &mut self.a,
                &mut self.b,
                &mut self.c,
            )?;
            subtract(ctx, &self.converted, &self.previous, &mut self.d)?;
            scale(ctx, &self.d, u.rho_current, &mut self.e)?;
            if let Some(divisor) = u.divisor {
                subtract(ctx, &self.older, &self.previous, &mut self.a)?;
                divide(ctx, &self.a, divisor, &mut self.b)?;
                scale(ctx, &self.b, u.rho_old, &mut self.f)?;
                binary(
                    ctx,
                    ops::PointwiseSemantic::Add,
                    &self.f,
                    &self.e,
                    &mut self.d,
                )?;
                scale(ctx, &self.d, u.weight, &mut self.a)?;
            } else {
                scale(ctx, &self.e, u.weight, &mut self.a)?;
            }
            subtract(ctx, &self.c, &self.a, &mut self.corrected)?;
        } else {
            transfers::copy_cuda_to_cuda(ctx, &self.sample, &self.corrected)?;
        }
        if index > 0 {
            transfers::copy_cuda_to_cuda(ctx, &self.previous, &self.older)?;
        }
        transfers::copy_cuda_to_cuda(ctx, &self.converted, &self.previous)?;
        transfers::copy_cuda_to_cuda(ctx, &self.corrected, &self.last_sample)?;
        let u = s.predictor;
        Self::base(
            ctx,
            &self.last_sample,
            &self.previous,
            u,
            &mut self.a,
            &mut self.b,
            &mut self.c,
        )?;
        if let Some(divisor) = u.divisor {
            subtract(ctx, &self.older, &self.previous, &mut self.a)?;
            divide(ctx, &self.a, divisor, &mut self.b)?;
            scale(ctx, &self.b, u.rho_old, &mut self.d)?;
            scale(ctx, &self.d, u.weight, &mut self.e)?;
            subtract(ctx, &self.c, &self.e, &mut self.next)?;
            transfers::copy_cuda_to_cuda(ctx, &self.next, &self.sample)
        } else {
            transfers::copy_cuda_to_cuda(ctx, &self.c, &self.sample)
        }
    }
}

use super::super::weights::{upload, Weights};
use super::{dit::Dit, tensor_ops::*, vae::VaeChunk};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use apxinf_cuda_new::ops::GemmPolicy;
use apxinf_cuda_new::{
    AttentionPolicy, CudaBuffer, CudaContext, F64RecurrenceState, GatherArgs, GatherPatchGeometry,
    GatherSemantic, PreparedIndexedMean, PreparedRecurrence, RecurrenceCoefficients,
};
use std::rc::Rc;
const PIXELS: usize = 224 * 224;
const LATENT: usize = 16 * 9 * 28 * 28;
const FRAME: usize = 16 * 28 * 28;

fn slice(t: &Tensor, start: usize, len: usize, shape: Vec<usize>) -> Result<Tensor> {
    CudaBuffer::from_tensor(t)
        .map_err(Error::Cuda)?
        .view(
            start * t.dtype().size_in_bytes(),
            len * t.dtype().size_in_bytes(),
        )
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), t.dtype())
        .map_err(Error::Cuda)
}
fn constant_f32(ctx: &CudaContext, values: &[f32]) -> Result<Tensor> {
    upload(
        &Tensor::from_f32(vec![1, values.len()], values)?,
        ctx.device_id(),
    )
}
fn copy(ctx: &CudaContext, a: &Tensor, b: &Tensor) -> Result<()> {
    apxinf_cuda_new::transfers::copy_cuda_to_cuda(
        ctx,
        &a.reshape(vec![a.numel()])?,
        &b.reshape(vec![b.numel()])?,
    )
}

struct Step {
    input_scale: Tensor,
    output_scale: Tensor,
    time: Tensor,
    update: Option<PreparedRecurrence>,
}

/// Fixed LIBERO direct-policy computation and stable device storage. The runner
/// owns preparation/capture and request binding; this module only computes.
pub(in crate::cosmos_policy) struct Model {
    pub ctx: Rc<CudaContext>,
    weights: Rc<Weights>,
    pub rgb: CudaBuffer,
    pub proprio: Tensor,
    pub task: Tensor,
    pub noise: Tensor,
    pub actions: Tensor,
    pub joint: Tensor,
    images: Tensor,
    video: Tensor,
    first: VaeChunk,
    rest: VaeChunk,
    condition_source: Tensor,
    condition_map: Tensor,
    condition_bf16: Tensor,
    condition: Tensor,
    latent_input: Tensor,
    latent_view: Tensor,
    patch_map: Tensor,
    unpatch_map: Tensor,
    network_output: Tensor,
    text: Tensor,
    dit: Dit,
    state: Tensor,
    scratch: Tensor,
    scaled_noise: Tensor,
    eighty: Tensor,
    reset: PreparedRecurrence,
    steps: Vec<Step>,
    readout: PreparedIndexedMean,
}
impl Model {
    pub fn new(ctx: Rc<CudaContext>, weights: Rc<Weights>) -> Result<Self> {
        let rgb = CudaBuffer::alloc(2 * PIXELS * 3, ctx.device_id()).map_err(Error::Cuda)?;
        let images = bf(&ctx, 2 * PIXELS, 3)?;
        let video = upload(
            &Tensor::from_bf16(
                vec![17, 224, 224, 3],
                &vec![half::bf16::NEG_ONE; 17 * PIXELS * 3],
            )?,
            ctx.device_id(),
        )?;
        let first_input = slice(&video, 0, PIXELS * 3, vec![1, 224, 224, 3])?;
        let rest_input = slice(&video, PIXELS * 3, 16 * PIXELS * 3, vec![16, 224, 224, 3])?;
        let first = VaeChunk::prepare(&ctx, &weights.vae, &first_input, None)?;
        let rest = VaeChunk::prepare(&ctx, &weights.vae, &rest_input, Some(&first))?;
        let proprio = alloc(&ctx, vec![1, 9], DType::F32)?;
        let task = alloc(&ctx, vec![1], DType::I32)?;
        let noise = alloc(&ctx, vec![1, LATENT], DType::F32)?;
        // Last element is the zero used for every unconditioned slot.
        let condition_source = upload(
            &Tensor::from_bf16(
                vec![5 * FRAME + 10, 1],
                &vec![half::bf16::ZERO; 5 * FRAME + 10],
            )?,
            ctx.device_id(),
        )?;
        let mut cond_map = Vec::with_capacity(LATENT);
        for c in 0..16 {
            for t in 0..9 {
                for p in 0..784 {
                    cond_map.push(if t == 1 {
                        (5 * FRAME + (c * 784 + p) % 9) as i32
                    } else if t < 4 {
                        ((t * 784 + p) * 16 + c) as i32
                    } else {
                        (5 * FRAME + 9) as i32
                    });
                }
            }
        }
        let condition_map = indices(&ctx, &cond_map)?;
        let condition_bf16 = bf(&ctx, 1, LATENT)?;
        let condition = alloc(&ctx, vec![1, LATENT], DType::F32)?;
        let mut latent = vec![half::bf16::ZERO; 18 * 9 * 784];
        latent[16 * 9 * 784..16 * 9 * 784 + 4 * 784].fill(half::bf16::ONE);
        let latent_input = upload(
            &Tensor::from_bf16(vec![1, latent.len()], &latent)?,
            ctx.device_id(),
        )?;
        let latent_view = slice(&latent_input, 0, LATENT, vec![1, LATENT])?;
        let mut patch = Vec::with_capacity(1764 * 72);
        for t in 0..9 {
            for y in 0..14 {
                for x in 0..14 {
                    for c in 0..18 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                patch.push(
                                    (c * 9 * 784 + t * 784 + (y * 2 + dy) * 28 + x * 2 + dx) as i32,
                                );
                            }
                        }
                    }
                }
            }
        }
        let patch_map = indices(&ctx, &patch)?;
        let mut unpatch = Vec::with_capacity(LATENT);
        for c in 0..16 {
            for t in 0..9 {
                for y in 0..28 {
                    for x in 0..28 {
                        unpatch.push(
                            ((t * 196 + y / 2 * 14 + x / 2) * 64 + (y % 2 * 2 + x % 2) * 16 + c)
                                as i32,
                        );
                    }
                }
            }
        }
        let unpatch_map = indices(&ctx, &unpatch)?;
        let network_output = bf(&ctx, 1, LATENT)?;
        let text = bf(&ctx, 512, 1024)?;
        // Pinned LIBERO runtime: spatial extrapolation 3, temporal 1.
        let spatial_ntk = 3f64.powf(42. / 40.) as f32;
        let dit = Dit::prepare(&ctx, &weights.dit, [1., spatial_ntk, spatial_ntk])?;
        let state = alloc(&ctx, vec![1, LATENT], DType::F32)?;
        let scratch = alloc(&ctx, vec![1, LATENT], DType::F32)?;
        let joint = alloc(&ctx, vec![1, LATENT], DType::F32)?;
        let scaled_noise = alloc(&ctx, vec![1, LATENT], DType::F32)?;
        let eighty = constant_f32(&ctx, &vec![80.; LATENT])?;
        let recurrence = F64RecurrenceState::new(&ctx, LATENT)?;
        let reset = recurrence.prepare_reset(&ctx, &scaled_noise, &state)?;
        let sigmas: [f64; 5] = std::array::from_fn(|i| {
            (80f64.powf(1. / 7.) + (i as f64) / 4. * (4f64.powf(1. / 7.) - 80f64.powf(1. / 7.)))
                .powi(7)
        });
        let mut steps = Vec::new();
        for (i, &sigma) in sigmas.iter().enumerate() {
            let sigma = sigma as f32;
            let t = sigma / (sigma + 1.);
            let clean = 1. - t;
            let mut input_scale = Vec::with_capacity(LATENT);
            let mut output_scale = Vec::with_capacity(LATENT);
            for _ in 0..16 {
                for frame in 0..9 {
                    for _ in 0..784 {
                        input_scale.push(if frame < 4 { 0. } else { clean });
                        output_scale.push(if frame < 4 { 0. } else { -t });
                    }
                }
            }
            let mut time = Vec::with_capacity(9 * 2048);
            for frame in 0..9 {
                let time_value = if frame < 4 {
                    0.
                } else {
                    half::bf16::from_f32(t).to_f32()
                };
                let angles: Vec<_> = (0..1024)
                    .map(|d| time_value * (-(10000f32).ln() * (d as f32) / 1024.).exp())
                    .collect();
                time.extend(angles.iter().map(|a| half::bf16::from_f32(a.cos())));
                time.extend(angles.iter().map(|a| half::bf16::from_f32(a.sin())));
            }
            let update = if i == 4 {
                None
            } else {
                let (s, t) = (sigmas[i], sigmas[i + 1]);
                let coefficients = if i == 0 {
                    RecurrenceCoefficients {
                        a: t / s,
                        b: (s - t) / s,
                        c: 0.,
                        d: 1.,
                    }
                } else {
                    let (s, t, m) = (-s.ln(), -t.ln(), -sigmas[i - 1].ln());
                    let dt = t - s;
                    let c2 = (m - s) / dt;
                    let phi1 = (-dt).exp_m1() / (-dt);
                    let phi2 = (phi1 - 1.) / (-dt);
                    RecurrenceCoefficients {
                        a: (-dt).exp(),
                        b: phi1 - 1. / c2 * phi2,
                        c: 1. / c2 * phi2,
                        d: dt,
                    }
                };
                Some(recurrence.prepare_update(&ctx, &joint, &state, coefficients)?)
            };
            steps.push(Step {
                input_scale: constant_f32(&ctx, &input_scale)?,
                output_scale: constant_f32(&ctx, &output_scale)?,
                time: upload(&Tensor::from_bf16(vec![9, 2048], &time)?, ctx.device_id())?,
                update,
            });
        }
        let actions = alloc(&ctx, vec![16, 7], DType::F32)?;
        let action_map: Vec<u32> = (0..FRAME)
            .map(|i| ((i / 784) * 9 * 784 + 4 * 784 + i % 784) as u32)
            .collect();
        let readout = PreparedIndexedMean::new(&ctx, &joint, &actions, &action_map)?;
        Ok(Self {
            ctx,
            weights,
            rgb,
            proprio,
            task,
            noise,
            actions,
            joint,
            images,
            video,
            first,
            rest,
            condition_source,
            condition_map,
            condition_bf16,
            condition,
            latent_input,
            latent_view,
            patch_map,
            unpatch_map,
            network_output,
            text,
            dit,
            state,
            scratch,
            scaled_noise,
            eighty,
            reset,
            steps,
            readout,
        })
    }
    pub fn forward(&mut self, gemm: &GemmPolicy, attention: &AttentionPolicy) -> Result<()> {
        let ctx = &self.ctx;
        let mut images = self.images.clone();
        let mut args = GatherArgs::rgb_to_patches(
            &self.rgb,
            &mut images,
            GatherPatchGeometry {
                views: 2,
                image_size: 224,
                patch_size: 1,
                nhwc: true,
            },
        );
        args.semantic = GatherSemantic::RgbToPatchesStaged;
        apxinf_cuda_new::gather(ctx, args)?;
        for (view, start) in [(0, 5), (1, 9)] {
            let source = slice(&self.images, view * PIXELS * 3, PIXELS * 3, vec![PIXELS, 3])?;
            for frame in start..start + 4 {
                copy(
                    ctx,
                    &source,
                    &slice(&self.video, frame * PIXELS * 3, PIXELS * 3, vec![PIXELS, 3])?,
                )?;
            }
        }
        self.first
            .forward(ctx, &self.weights.vae, gemm, attention)?;
        self.rest.forward(ctx, &self.weights.vae, gemm, attention)?;
        copy(
            ctx,
            &self.first.output,
            &slice(&self.condition_source, 0, FRAME, vec![FRAME, 1])?,
        )?;
        copy(
            ctx,
            &self.rest.output,
            &slice(&self.condition_source, FRAME, 4 * FRAME, vec![4 * FRAME, 1])?,
        )?;
        cast(
            ctx,
            &self.proprio,
            &slice(&self.condition_source, 5 * FRAME, 9, vec![1, 9])?,
        )?;
        gather(
            ctx,
            &self.condition_source,
            &self.condition_map,
            1,
            &self.condition_bf16,
        )?;
        cast(ctx, &self.condition_bf16, &self.condition)?;
        gather(ctx, &self.weights.text, &self.task, 512 * 1024, &self.text)?;
        multiply(ctx, &self.noise, &self.eighty, &self.scaled_noise)?;
        self.reset.enqueue()?;
        for step in &self.steps {
            multiply(ctx, &self.state, &step.input_scale, &self.scratch)?;
            add(ctx, &self.scratch, &self.condition, &self.scratch)?;
            cast(ctx, &self.scratch, &self.latent_view)?;
            gather(
                ctx,
                &self.latent_input,
                &self.patch_map,
                1,
                &self.dit.patches,
            )?;
            self.dit.forward(
                ctx,
                &self.weights.dit,
                &step.time,
                &self.text,
                gemm,
                attention,
            )?;
            gather(
                ctx,
                &self.dit.output,
                &self.unpatch_map,
                1,
                &self.network_output,
            )?;
            cast(ctx, &self.network_output, &self.scratch)?;
            multiply(ctx, &self.scratch, &step.output_scale, &self.scratch)?;
            multiply(ctx, &self.state, &step.input_scale, &self.joint)?;
            add(ctx, &self.joint, &self.scratch, &self.joint)?;
            add(ctx, &self.joint, &self.condition, &self.joint)?;
            if let Some(update) = &step.update {
                update.enqueue()?;
            }
        }
        self.readout.enqueue()
    }
}

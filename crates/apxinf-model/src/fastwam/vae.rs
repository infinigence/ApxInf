//! Wan 2.2 encoder for a single observed frame. Temporal causal convolutions
//! use their final temporal tap because the cache is empty for every request.
//! This module never keeps a video cache or invokes a reference framework.
use super::backend::{self, ops, Context};
use super::weights::vae::{Attention, Conv, DownBlock, Norm, Residual, Vae};
use apxinf_core::{DType, Error, Result, Shape, Tensor};

// Each arena allocation can consume at most 255 leading alignment bytes.
// Keeping 256 per allocation also covers an aligned final capacity boundary.
fn allocation_bytes(elements: usize, dtype: DType) -> usize {
    elements * dtype.size_in_bytes() + 256
}

impl Conv {
    fn workspace_bytes(&self, h: usize, w: usize) -> usize {
        let rows = (h / self.stride) * (w / self.stride);
        let weight = self.projection.weight.shape().dims();
        let projection = allocation_bytes(rows * weight[1], self.projection.weight.dtype());
        2 * projection // dtype-rounded GEMM result plus separately rounded bias add
            + if self.kernel == 1 && self.stride == 1 {
                0
            } else {
                allocation_bytes(rows * weight[0], self.projection.weight.dtype())
            }
    }
    fn forward(
        &self,
        ctx: &Context,
        x: &Tensor,
        h: usize,
        w: usize,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        if self.kernel == 1 && self.stride == 1 {
            return self.projection.forward_conv(ctx, x, policy);
        }
        let (oh, ow) = (h / self.stride, w / self.stride);
        let mut columns = ctx.allocate_output(
            Shape::new(vec![oh * ow, self.projection.weight.shape().dims()[0]]),
            x.dtype(),
        )?;
        ops::image_im2col(
            ctx,
            x,
            &mut columns,
            h,
            w,
            self.kernel,
            self.stride,
            self.padding,
            oh,
            ow,
        )?;
        self.projection.forward_conv(ctx, &columns, policy)
    }
}

impl Norm {
    fn forward(&self, ctx: &Context, x: &Tensor) -> Result<Tensor> {
        let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
        ops::channel_l2_normalize(
            ctx,
            x,
            &self.gamma,
            &mut out,
            (x.shape().dims()[1] as f32).sqrt(),
            1e-12,
        )?;
        Ok(out)
    }
    fn activated(&self, ctx: &Context, x: &Tensor) -> Result<Tensor> {
        backend::activation(ctx, &self.forward(ctx, x)?, ops::PointwiseActivation::Silu)
    }
}

impl Residual {
    fn workspace_bytes(&self, h: usize, w: usize) -> usize {
        let input_channels = self.conv1.projection.weight.shape().dims()[0]
            / (self.conv1.kernel * self.conv1.kernel);
        let output_channels = self.conv1.projection.weight.shape().dims()[1];
        let dtype = self.conv1.projection.weight.dtype();
        self.shortcut
            .as_ref()
            .map_or(0, |c| c.workspace_bytes(h, w))
            + 2 * allocation_bytes(h * w * input_channels, dtype)
            + self.conv1.workspace_bytes(h, w)
            + 2 * allocation_bytes(h * w * output_channels, dtype)
            + self.conv2.workspace_bytes(h, w)
            + allocation_bytes(h * w * output_channels, dtype)
    }
    fn forward(
        &self,
        ctx: &Context,
        x: &Tensor,
        h: usize,
        w: usize,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        let residual = match &self.shortcut {
            Some(conv) => conv.forward(ctx, x, h, w, policy)?,
            None => x.clone(),
        };
        let x = self
            .conv1
            .forward(ctx, &self.norm1.activated(ctx, x)?, h, w, policy)?;
        let x = self
            .conv2
            .forward(ctx, &self.norm2.activated(ctx, &x)?, h, w, policy)?;
        backend::add(ctx, &x, &residual)
    }
}

impl DownBlock {
    fn forward(
        &self,
        ctx: &Context,
        x: &Tensor,
        h: usize,
        w: usize,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        let mut y = x.clone();
        for block in &self.residuals {
            y = block.forward(ctx, &y, h, w, policy)?;
        }
        let fs = if self.downsample.is_some() { 2 } else { 1 };
        if let Some(conv) = &self.downsample {
            y = conv.forward(ctx, &y, h, w, policy)?;
        }
        let mut residual = ctx.allocate_output(y.shape().clone(), y.dtype())?;
        ops::image_grouped_average(
            ctx,
            x,
            &mut residual,
            h,
            w,
            fs,
            self.temporal_factor,
            y.shape().dims()[1],
        )?;
        backend::add(ctx, &y, &residual)
    }
}

impl Attention {
    fn forward(
        &self,
        ctx: &Context,
        x: &Tensor,
        h: usize,
        w: usize,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        let normalized = self.norm.forward(ctx, x)?;
        let channels = x.shape().dims()[1];
        let q = backend::reshape(
            &self.q.forward_conv(ctx, &normalized, policy)?,
            vec![1, h * w, 1, channels],
        )?;
        let k = backend::reshape(
            &self.k.forward_conv(ctx, &normalized, policy)?,
            vec![1, h * w, 1, channels],
        )?;
        let v = backend::reshape(
            &self.v.forward_conv(ctx, &normalized, policy)?,
            vec![1, h * w, 1, channels],
        )?;
        let mut output = ctx.allocate_output(q.shape().clone(), q.dtype())?;
        let mut args = ops::AttentionArgs::new(&q, &k, &v, &mut output);
        args.policy.online_tune = policy.online_tune;
        args.policy.cache_dir = policy.cache_dir.clone();
        ops::attention(ctx, args)?;
        let output = backend::reshape(&output, vec![h * w, channels])?;
        backend::add(
            ctx,
            &self.projection.forward(ctx, &output, h, w, policy)?,
            x,
        )
    }
}

impl Vae {
    /// Conservative arena capacity derived from this encoder's allocation
    /// schedule. Native provider workspaces are separately owned by executions.
    pub(super) fn workspace_bytes(&self) -> usize {
        let mut bytes = allocation_bytes(224*448*3,self.dtype) // possible input cast
            +allocation_bytes(112*224*12,self.dtype); // width-first patchify
        let (mut h, mut w) = (112, 224);
        bytes += self.input.workspace_bytes(h, w);
        for block in &self.down {
            for residual in &block.residuals {
                bytes += residual.workspace_bytes(h, w);
            }
            if let Some(conv) = &block.downsample {
                bytes += conv.workspace_bytes(h, w);
                h /= 2;
                w /= 2;
            }
            let channels = block
                .residuals
                .last()
                .expect("validated two-residual VAE block")
                .conv2
                .projection
                .weight
                .shape()
                .dims()[1];
            bytes += 2 * allocation_bytes(h * w * channels, self.dtype); // average shortcut + add
        }
        bytes += self.mid0.workspace_bytes(h, w);
        let channels = self.attention.q.weight.shape().dims()[1];
        bytes += 11 * allocation_bytes(h * w * channels, self.dtype); // norm,Q/K/V GEMM+bias,attention,projection GEMM+bias,add
        bytes += self.mid1.workspace_bytes(h, w);
        bytes += 2 * allocation_bytes(h * w * channels, self.dtype); // final norm + SiLU
        bytes += self.output.workspace_bytes(h, w) + self.quant.workspace_bytes(h, w);
        bytes + allocation_bytes(h * w * 48, DType::F32)
    }
    /// One stitched HWC RGB frame in [-1,1], flattened `[224*448,3]`.
    /// Returns standardized posterior mean `[14*28,48]` in F32.
    pub(super) fn forward(
        &self,
        ctx: &Context,
        input: &Tensor,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        if input.shape().dims() != [224 * 448, 3] {
            return Err(Error::Other(
                "Wan VAE requires one 224x448 RGB frame".into(),
            ));
        }
        let input = backend::cast(ctx, input, self.dtype)?;
        let mut packed = ctx.allocate_output(Shape::new(vec![112 * 224, 12]), self.dtype)?;
        ops::image_space_to_depth(ctx, &input, &mut packed, 224, 448, 2)?;
        let (mut h, mut w) = (112, 224);
        let mut x = self.input.forward(ctx, &packed, h, w, policy)?;
        for block in &self.down {
            x = block.forward(ctx, &x, h, w, policy)?;
            if block.downsample.is_some() {
                h /= 2;
                w /= 2;
            }
        }
        x = self.mid0.forward(ctx, &x, h, w, policy)?;
        x = self.attention.forward(ctx, &x, h, w, policy)?;
        x = self.mid1.forward(ctx, &x, h, w, policy)?;
        x = self
            .output
            .forward(ctx, &self.norm.activated(ctx, &x)?, h, w, policy)?;
        x = self.quant.forward(ctx, &x, h, w, policy)?;
        let mut out = ctx.allocate_output(Shape::new(vec![h * w, 48]), DType::F32)?;
        ops::channel_standardize(ctx, &x, &self.mean, &self.std, &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::Path;

    fn compare(name: &str, actual: &Tensor, fixtures: &HashMap<String, Tensor>, dtype: DType) {
        let a = backend::transfers::to_cpu(actual)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let reference = fixtures.get(name).unwrap();
        assert_eq!(actual.shape(), reference.shape(), "{name}");
        let b = reference.to_f32_vec().unwrap();
        let max_abs = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let relative_l2 = (a
            .iter()
            .zip(&b)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / b.iter()
                .map(|b| f64::from(*b).powi(2))
                .sum::<f64>()
                .max(1e-30))
        .sqrt();
        eprintln!("VAE {dtype:?} {name}: max_abs={max_abs:.9} relative_l2={relative_l2:.9}");
        assert!(a.iter().all(|x| x.is_finite()), "nonfinite {name}");
        // Component parity is stricter than one BF16 ULP in relative norm;
        // full-model acceptance separately checks the FP32 quality floor.
        let tolerance = if dtype == DType::F32 { 1e-4 } else { 0.008 };
        assert!(
            relative_l2 <= tolerance,
            "{name} relative error {relative_l2} exceeds {tolerance}"
        );
    }

    #[test]
    #[ignore = "requires Wan VAE assets and independent reference captures on CUDA"]
    fn fastwam_vae_checkpoint_precision() {
        let assets = std::env::var("APXINF_FASTWAM_VAE")
            .expect("APXINF_FASTWAM_VAE points to vae directory");
        let captures = std::env::var("APXINF_FASTWAM_VAE_FIXTURES")
            .expect("APXINF_FASTWAM_VAE_FIXTURES points to capture directory");
        let ctx = Context::new_with_autotune(0, false).unwrap();
        let policy = ops::GemmPolicy {
            online_tune: false,
            ..Default::default()
        };
        for (dtype, name) in [(DType::F32, "float32-strict"), (DType::BF16, "bfloat16")] {
            let (fixtures, _) = apxinf_loader::safetensors::load_native_path(
                &Path::new(&captures).join(format!("{name}.safetensors")),
            )
            .unwrap();
            let model = Vae::load(&ctx, Path::new(&assets), dtype).unwrap();
            let input = backend::upload(&ctx, fixtures.get("input").unwrap()).unwrap();
            let input = backend::cast(&ctx, &input, dtype).unwrap();
            let mut packed = ctx
                .allocate_output(Shape::new(vec![112 * 224, 12]), dtype)
                .unwrap();
            ops::image_space_to_depth(&ctx, &input, &mut packed, 224, 448, 2).unwrap();
            let (mut h, mut w) = (112, 224);
            let mut x = model.input.forward(&ctx, &packed, h, w, &policy).unwrap();
            compare("encoder.conv_in", &x, &fixtures, dtype);
            for (i, block) in model.down.iter().enumerate() {
                x = block.forward(&ctx, &x, h, w, &policy).unwrap();
                if block.downsample.is_some() {
                    h /= 2;
                    w /= 2;
                }
                compare(&format!("encoder.down_blocks.{i}"), &x, &fixtures, dtype);
            }
            x = model.mid0.forward(&ctx, &x, h, w, &policy).unwrap();
            compare("encoder.mid_block.resnets.0", &x, &fixtures, dtype);
            x = model.attention.forward(&ctx, &x, h, w, &policy).unwrap();
            compare("encoder.mid_block.attentions.0", &x, &fixtures, dtype);
            x = model.mid1.forward(&ctx, &x, h, w, &policy).unwrap();
            compare("encoder.mid_block.resnets.1", &x, &fixtures, dtype);
            x = model
                .output
                .forward(
                    &ctx,
                    &model.norm.activated(&ctx, &x).unwrap(),
                    h,
                    w,
                    &policy,
                )
                .unwrap();
            compare("encoder.conv_out", &x, &fixtures, dtype);
            x = model.quant.forward(&ctx, &x, h, w, &policy).unwrap();
            compare("quant_conv", &x, &fixtures, dtype);
            let output = model.forward(&ctx, &input, &policy).unwrap();
            compare("output", &output, &fixtures, dtype);
            if dtype == DType::BF16 {
                let mut captured = None;
                let phase = apxinf_cuda_new::PreparedPhase::prepare_and_capture(
                    &ctx,
                    ops::ExecutionSession::with_capacity(model.workspace_bytes(), ctx.device_id())
                        .unwrap(),
                    || {
                        captured = Some(model.forward(&ctx, &input, &policy)?);
                        Ok(())
                    },
                )
                .unwrap();
                phase.replay().unwrap();
                ctx.synchronize().unwrap();
                compare("output", captured.as_ref().unwrap(), &fixtures, dtype);
                let baseline = backend::transfers::to_cpu(captured.as_ref().unwrap())
                    .unwrap()
                    .to_f32_vec()
                    .unwrap();
                for _ in 0..3 {
                    phase.replay().unwrap();
                }
                ctx.synchronize().unwrap();
                assert_eq!(
                    baseline,
                    backend::transfers::to_cpu(captured.as_ref().unwrap())
                        .unwrap()
                        .to_f32_vec()
                        .unwrap()
                );
                let input_buffer = backend::DeviceBuffer::from_tensor(&input).unwrap();
                input_buffer
                    .copy_from_host(&vec![0; input.shape().numel() * 2])
                    .unwrap();
                phase
                    .run(|| {
                        captured = Some(model.forward(&ctx, &input, &policy)?);
                        Ok(())
                    })
                    .unwrap();
                let changed = backend::transfers::to_cpu(captured.as_ref().unwrap())
                    .unwrap()
                    .to_f32_vec()
                    .unwrap();
                assert!(
                    changed
                        .iter()
                        .zip(&baseline)
                        .any(|(x, y)| (x - y).abs() > 0.01),
                    "changed frame did not propagate"
                );
                drop(model);
                drop(input);
                drop(input_buffer);
                phase.replay().unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(
                    changed,
                    backend::transfers::to_cpu(captured.as_ref().unwrap())
                        .unwrap()
                        .to_f32_vec()
                        .unwrap()
                );
                let started = std::time::Instant::now();
                for _ in 0..10 {
                    phase.replay().unwrap();
                }
                ctx.synchronize().unwrap();
                eprintln!(
                    "VAE BF16 captured latency_ms={} workspace_used={}",
                    started.elapsed().as_secs_f64() * 100.,
                    phase.workspace_used()
                );
            }
        }
    }
}

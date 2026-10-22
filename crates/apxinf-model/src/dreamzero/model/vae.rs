//! Complete causal encoder; temporal downsampling preserves the first frame.
//! Activations are contiguous NTHWC for channel normalization and unfolded
//! GEMM; cuDNN convolutions use bounded temporary NCDHW input/output storage.
use super::super::{
    backend::{allocate, binary, ops, transfers, Context, Policies},
    weights::vae::{Conv, Down, Residual, VaeWeights},
};
use crate::dreamzero::backend;
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use ops::PointwiseSemantic as S;

const TILE: usize = 8192;
fn view(tensor: &Tensor, offset: usize, shape: &[usize]) -> Result<Tensor> {
    let elements = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| Error::Other("VAE view overflow".into()))?;
    let buffer = backend::CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    buffer
        .view(
            offset * tensor.dtype().size_in_bytes(),
            elements * tensor.dtype().size_in_bytes(),
        )
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape.to_vec()), tensor.dtype())
        .map_err(Error::Cuda)
}
fn matrix(x: &Tensor) -> Result<Tensor> {
    let cols = *x
        .shape()
        .dims()
        .last()
        .ok_or_else(|| Error::Other("VAE tensor has no channels".into()))?;
    x.reshape(vec![x.numel() / cols, cols])
}

pub(in crate::dreamzero) struct VaeWorkspace {
    pub input: Tensor,
    pub output: Tensor,
    promoted_norm: Option<[Tensor; 3]>,
    frames: usize,
    height: usize,
    width: usize,
    slots: Vec<Tensor>,
    columns: Tensor,
    projected_tile: Tensor,
    biased_tile: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention: Tensor,
    convolution_input: Tensor,
    convolution_output: Tensor,
    convolution_layout: Tensor,
    convolution_workspace: Tensor,
}
impl VaeWorkspace {
    pub fn new(
        ctx: &Context,
        frames: usize,
        height: usize,
        width: usize,
        promote_norm: bool,
    ) -> Result<Self> {
        if ![1, 9, 33].contains(&frames) || height != 352 || width != 640 {
            return Err(Error::Other(
                "DreamZero VAE profile requires 1/9/33 frames at 352x640".into(),
            ));
        }
        let capacity = frames * height * width * 96;
        let latent_frames = (frames - 1) / 4 + 1;
        let latent_pixels = latent_frames * (height / 8) * (width / 8);
        Ok(Self {
            input: allocate(ctx, &[1, frames, height, width, 3])?,
            output: allocate(ctx, &[1, latent_frames, height / 8, width / 8, 16])?,
            promoted_norm: if promote_norm {
                Some([
                    ctx.allocate_output(Shape::new(vec![capacity]), DType::F32)?,
                    ctx.allocate_output(Shape::new(vec![capacity]), DType::F32)?,
                    ctx.allocate_output(Shape::new(vec![384]), DType::F32)?,
                ])
            } else {
                None
            },
            frames,
            height,
            width,
            slots: (0..7)
                .map(|i| allocate(ctx, &[if promote_norm && i == 4 { 1 } else { capacity }]))
                .collect::<Result<_>>()?,
            columns: allocate(ctx, &[TILE * 27 * 384])?,
            projected_tile: allocate(ctx, &[TILE * 384])?,
            biased_tile: allocate(ctx, &[TILE * 384])?,
            q: allocate(ctx, &[latent_pixels * 384])?,
            k: allocate(ctx, &[latent_pixels * 384])?,
            v: allocate(ctx, &[latent_pixels * 384])?,
            attention: allocate(ctx, &[latent_pixels * 384])?,
            convolution_input: allocate(ctx, &[4 * 90 * 162 * 384])?,
            convolution_output: allocate(ctx, &[4 * 352 * 640 * 96])?,
            convolution_layout: allocate(ctx, &[4 * 352 * 640 * 96])?,
            convolution_workspace: ctx.allocate_output(Shape::new(vec![256 << 20]), DType::I8)?,
        })
    }
    fn slot(&self, index: usize, shape: &[usize]) -> Result<Tensor> {
        view(&self.slots[index], 0, shape)
    }
}
fn geometry(w: &Conv) -> ops::Unfold3dGeometry {
    ops::Unfold3dGeometry {
        kernel: w.kernel,
        stride: [1; 3],
        dilation: [1; 3],
        padding_before: [w.kernel[0] - 1, w.kernel[1] / 2, w.kernel[2] / 2],
        padding_after: [0, w.kernel[1] / 2, w.kernel[2] / 2],
    }
}
/// Tiled columns and projection addresses are reused across all tiles; only
/// the unfold row offset and destination copy range vary. Prepared GEMMs do
/// not multiply native workspaces by the number of spatial tiles.
fn convolution(
    ctx: &Context,
    p: &Policies,
    s: &VaeWorkspace,
    x: &Tensor,
    w: &Conv,
    g: ops::Unfold3dGeometry,
    out: &Tensor,
) -> Result<()> {
    let shape = x.shape().dims();
    let dst = out.shape().dims();
    if shape.len() != 5
        || dst.len() != 5
        || shape[0] != 1
        || dst[0] != 1
        || shape[4] != w.input
        || dst[4] != w.output
        || g.kernel != w.kernel
        || dst[1..4] != g.output_spatial([shape[1], shape[2], shape[3]])?
    {
        return Err(Error::Other(
            "DreamZero VAE convolution shape mismatch".into(),
        ));
    }
    if let Some(weight) = &w.native_weight {
        native_convolution(ctx, s, x, w, weight, out)?;
        return Ok(());
    }
    let rows = dst[1] * dst[2] * dst[3];
    let k = w.padded_input * w.kernel.iter().product::<usize>();
    for start in (0..rows).step_by(TILE) {
        let n = (rows - start).min(TILE);
        let mut columns = view(&s.columns, 0, &[n, k])?;
        let mut projected = view(&s.projected_tile, 0, &[n, w.output])?;
        let mut biased = view(&s.biased_tile, 0, &[n, w.output])?;
        ops::unfold3d_blocked(ctx, x, g, start, w.channel_block, &mut columns)?;
        let mut args = ops::GemmArgs::new(&columns, &w.weight, &mut projected);
        args.policy = p.gemm.clone();
        args.policy.workspace_limit = 0;
        ops::gemm(ctx, args)?;
        // cuDNN convolution rounds its output before adding the BF16 bias.
        binary(ctx, S::BroadcastAdd, &projected, &w.bias, &mut biased)?;
        transfers::copy_cuda_to_cuda(ctx, &biased, &view(out, start * w.output, &[n, w.output])?)?;
    }
    Ok(())
}
/// The official encoder consumes one initial frame, then four RGB frames
/// or two frames at the 88x160 resolution per chunk. Keep those convolution shapes/reduction plans
/// while reading the complete causal activation from the shared workspace.
fn native_convolution(
    ctx: &Context,
    s: &VaeWorkspace,
    x: &Tensor,
    w: &Conv,
    weight: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let shape = x.shape().dims();
    let (height, width, chunk_frames) = match (shape[2], shape[3], w.input) {
        (352, 640, 3) => (352, 640, 4),
        (88, 160, 192 | 384) => (88, 160, 2),
        _ => {
            return Err(Error::Other(
                "DreamZero native convolution profile mismatch".into(),
            ))
        }
    };
    if w.kernel != [3; 3] {
        return Err(Error::Other(
            "DreamZero native convolution profile mismatch".into(),
        ));
    }
    let frame = height * width * w.input;
    let mut start = 0;
    while start < shape[1] {
        let count = if start == 0 {
            1
        } else {
            (shape[1] - start).min(chunk_frames)
        };
        let begin = start.saturating_sub(2);
        let padding = 2 - (start - begin);
        let source = view(
            x,
            begin * frame,
            &[1, start + count - begin, height, width, w.input],
        )?;
        let padded_shape = [1, count + 2, height + 2, width + 2, w.input];
        let mut padded = view(
            &s.columns,
            0,
            &[(count + 2) * (height + 2) * (width + 2), w.input],
        )?;
        ops::unfold3d(
            ctx,
            &source,
            ops::Unfold3dGeometry {
                kernel: [1; 3],
                stride: [1; 3],
                dilation: [1; 3],
                padding_before: [padding, 1, 1],
                padding_after: [0, 1, 1],
            },
            0,
            &mut padded,
        )?;
        let mut input = view(
            &s.convolution_input,
            0,
            &[1, w.input, count + 2, height + 2, width + 2],
        )?;
        ops::permute(
            ctx,
            &padded.reshape(padded_shape.to_vec())?,
            &[0, 4, 1, 2, 3],
            &mut input,
        )?;
        let mut projected = view(
            &s.convolution_output,
            0,
            &[1, w.output, count, height, width],
        )?;
        ops::convolution3d(
            ctx,
            &input,
            weight,
            [1; 3],
            &mut projected,
            &s.convolution_workspace,
        )?;
        let mut layout = view(
            &s.convolution_layout,
            0,
            &[1, count, height, width, w.output],
        )?;
        ops::permute(ctx, &projected, &[0, 2, 3, 4, 1], &mut layout)?;
        let mut destination = view(
            out,
            start * height * width * w.output,
            &[count * height * width, w.output],
        )?;
        binary(
            ctx,
            S::BroadcastAdd,
            &matrix(&layout)?,
            &w.bias,
            &mut destination,
        )?;
        start += count;
    }
    Ok(())
}

fn promoted_normalization(
    ctx: &Context,
    s: &VaeWorkspace,
    x: &Tensor,
    gamma: &Tensor,
    silu: bool,
) -> Result<Tensor> {
    let scratch = s
        .promoted_norm
        .as_ref()
        .expect("promoted normalization workspace");
    let input = matrix(x)?;
    let shape = input.shape().dims();
    let mut a = view(&scratch[0], 0, shape)?;
    let mut b = view(&scratch[1], 0, shape)?;
    let mut weight = view(&scratch[2], 0, &[1, shape[1]])?;
    ops::quantization(
        ctx,
        ops::QuantizationArgs::new(ops::QuantizationSemantic::CastBf16ToF32, &input, &mut a),
    )?;
    ops::quantization(
        ctx,
        ops::QuantizationArgs::new(ops::QuantizationSemantic::CastBf16ToF32, gamma, &mut weight),
    )?;
    ops::l2_norm_rounded(ctx, &a, &mut b, 1e-12)?;
    let mut scale = ops::PointwiseArgs::new(S::Scale, &b, &mut a);
    scale.dt = (shape[1] as f32).sqrt();
    ops::pointwise(ctx, scale)?;
    binary(ctx, S::BroadcastMultiply, &a, &weight, &mut b)?;
    let mut out = s.slot(2, shape)?;
    if silu {
        let mut args = ops::PointwiseArgs::new(S::BiasActivation, &b, &mut a);
        args.activation = ops::PointwiseActivation::Silu;
        ops::pointwise(ctx, args)?;
        ops::quantization(
            ctx,
            ops::QuantizationArgs::new(ops::QuantizationSemantic::CastF32ToBf16, &a, &mut out),
        )?;
    } else {
        ops::quantization(
            ctx,
            ops::QuantizationArgs::new(ops::QuantizationSemantic::CastF32ToBf16, &b, &mut out),
        )?;
    }
    out.reshape(x.shape().dims().to_vec())
}
fn normalized_silu(ctx: &Context, s: &VaeWorkspace, x: &Tensor, gamma: &Tensor) -> Result<Tensor> {
    if s.promoted_norm.is_some() {
        return promoted_normalization(ctx, s, x, gamma, true);
    }
    let input = matrix(x)?;
    let shape = input.shape().dims();
    let mut norm = s.slot(2, shape)?;
    let mut scaled = s.slot(3, shape)?;
    let mut weighted = s.slot(4, shape)?;
    ops::l2_norm_rounded(ctx, &input, &mut norm, 1e-12)?;
    let mut args = ops::PointwiseArgs::new(S::Scale, &norm, &mut scaled);
    args.dt = (shape[1] as f32).sqrt();
    ops::pointwise(ctx, args)?;
    binary(ctx, S::BroadcastMultiply, &scaled, gamma, &mut weighted)?;
    let mut activation = ops::PointwiseArgs::new(S::BiasActivation, &weighted, &mut norm);
    activation.activation = ops::PointwiseActivation::Silu;
    ops::pointwise(ctx, activation)?;
    norm.reshape(x.shape().dims().to_vec())
}
fn residual(
    ctx: &Context,
    p: &Policies,
    s: &VaeWorkspace,
    x: &Tensor,
    w: &Residual,
    out: &Tensor,
) -> Result<()> {
    let activated = normalized_silu(ctx, s, x, &w.norm1)?;
    let branch = s.slot(5, out.shape().dims())?;
    convolution(ctx, p, s, &activated, &w.conv1, geometry(&w.conv1), &branch)?;
    let activated = normalized_silu(ctx, s, &branch, &w.norm2)?;
    let projected = s.slot(6, out.shape().dims())?;
    convolution(
        ctx,
        p,
        s,
        &activated,
        &w.conv2,
        geometry(&w.conv2),
        &projected,
    )?;
    let shortcut = if let Some(w) = &w.shortcut {
        // The normalization scratch is dead before shortcut computation.
        let target = s.slot(3, out.shape().dims())?;
        convolution(ctx, p, s, x, w, geometry(w), &target)?;
        target
    } else {
        x.clone()
    };
    binary(
        ctx,
        S::Add,
        &matrix(&projected)?,
        &matrix(&shortcut)?,
        &mut matrix(out)?,
    )
}

pub(in crate::dreamzero) fn encode(
    ctx: &Context,
    p: &Policies,
    w: &VaeWeights,
    s: &VaeWorkspace,
) -> Result<()> {
    let mut shape = [1, s.frames, s.height, s.width, 96];
    let mut current = s.slot(0, &shape)?;
    convolution(ctx, p, s, &s.input, &w.input, geometry(&w.input), &current)?;
    let mut index = 0;
    for layer in &w.down {
        let next_index = 1 - index;
        match layer {
            Down::Residual(block) => {
                shape[4] = block.conv2.output;
                let out = s.slot(next_index, &shape)?;
                residual(ctx, p, s, &current, block, &out)?;
                current = out;
            }
            Down::Spatial { conv, temporal } => {
                shape[2] /= 2;
                shape[3] /= 2;
                let out = s.slot(next_index, &shape)?;
                let g = ops::Unfold3dGeometry {
                    kernel: [1, 3, 3],
                    stride: [1, 2, 2],
                    dilation: [1; 3],
                    padding_before: [0; 3],
                    padding_after: [0, 1, 1],
                };
                convolution(ctx, p, s, &current, conv, g, &out)?;
                if let Some(time) = temporal {
                    let target_shape = [1, (shape[1] - 1) / 2 + 1, shape[2], shape[3], shape[4]];
                    // Original cached encoder skips temporal convolution on the
                    // first frame, then uses [previous,last four] with stride2.
                    let target = s.slot(index, &target_shape)?;
                    let frame = shape[2] * shape[3] * shape[4];
                    transfers::copy_cuda_to_cuda(
                        ctx,
                        &view(&out, 0, &[frame])?,
                        &view(&target, 0, &[frame])?,
                    )?;
                    if shape[1] > 1 {
                        let tail_shape = [1, (shape[1] - 1) / 2, shape[2], shape[3], shape[4]];
                        let tail = view(&target, frame, &tail_shape)?;
                        let g = ops::Unfold3dGeometry {
                            kernel: [3, 1, 1],
                            stride: [2, 1, 1],
                            dilation: [1; 3],
                            padding_before: [0; 3],
                            padding_after: [0; 3],
                        };
                        convolution(ctx, p, s, &out, time, g, &tail)?;
                    }
                    shape = target_shape;
                    current = target;
                    // Result returned to the current ping-pong slot.
                    continue;
                }
                current = out;
            }
        }
        index = next_index;
    }
    let other = s.slot(1 - index, &shape)?;
    residual(ctx, p, s, &current, &w.middle1, &other)?;
    current = other;
    index = 1 - index;
    // Spatial attention is independent for each encoded frame, one head of384.
    let normalized = if s.promoted_norm.is_some() {
        promoted_normalization(ctx, s, &current, &w.attention.norm, false)?
    } else {
        let input = matrix(&current)?;
        let mut norm = s.slot(2, input.shape().dims())?;
        let mut scaled = s.slot(3, input.shape().dims())?;
        let mut normalized = s.slot(4, input.shape().dims())?;
        ops::l2_norm_rounded(ctx, &input, &mut norm, 1e-12)?;
        let mut scale = ops::PointwiseArgs::new(S::Scale, &norm, &mut scaled);
        scale.dt = 384f32.sqrt();
        ops::pointwise(ctx, scale)?;
        binary(
            ctx,
            S::BroadcastMultiply,
            &scaled,
            &w.attention.norm,
            &mut normalized,
        )?;
        normalized.reshape(shape.to_vec())?
    };
    let q = view(&s.q, 0, &shape)?;
    let k = view(&s.k, 0, &shape)?;
    let v = view(&s.v, 0, &shape)?;
    convolution(
        ctx,
        p,
        s,
        &normalized,
        &w.attention.q,
        geometry(&w.attention.q),
        &q,
    )?;
    convolution(
        ctx,
        p,
        s,
        &normalized,
        &w.attention.k,
        geometry(&w.attention.k),
        &k,
    )?;
    convolution(
        ctx,
        p,
        s,
        &normalized,
        &w.attention.v,
        geometry(&w.attention.v),
        &v,
    )?;
    let att_shape = vec![shape[1], shape[2] * shape[3], 1, 384];
    let q = q.reshape(att_shape.clone())?;
    let k = k.reshape(att_shape.clone())?;
    let v = v.reshape(att_shape.clone())?;
    let mut attention = view(&s.attention, 0, &att_shape)?;
    let mut args = ops::AttentionArgs::new(&q, &k, &v, &mut attention);
    args.scale = (1.0 / 384f64.sqrt()) as f32;
    args.policy = p.attention.clone();
    args.policy.workspace_limit = 512 << 20;
    ops::attention(ctx, args)?;
    let projected = s.slot(5, &shape)?;
    convolution(
        ctx,
        p,
        s,
        &attention.reshape(shape.to_vec())?,
        &w.attention.out,
        geometry(&w.attention.out),
        &projected,
    )?;
    let other = s.slot(1 - index, &shape)?;
    binary(
        ctx,
        S::Add,
        &matrix(&current)?,
        &matrix(&projected)?,
        &mut matrix(&other)?,
    )?;
    current = other;
    index = 1 - index;
    let other = s.slot(1 - index, &shape)?;
    residual(ctx, p, s, &current, &w.middle2, &other)?;
    current = other;
    let activated = normalized_silu(ctx, s, &current, &w.final_norm)?;
    shape[4] = 32;
    let projected = s.slot(5, &shape)?;
    convolution(
        ctx,
        p,
        s,
        &activated,
        &w.final_conv,
        geometry(&w.final_conv),
        &projected,
    )?;
    let moments = s.slot(6, &shape)?;
    convolution(
        ctx,
        p,
        s,
        &projected,
        &w.moments,
        geometry(&w.moments),
        &moments,
    )?;
    let rows = shape[1] * shape[2] * shape[3];
    let mut mu = s.slot(2, &[rows, 16])?;
    ops::quantization(
        ctx,
        ops::QuantizationArgs::new(
            ops::QuantizationSemantic::SliceColumnsBf16,
            &matrix(&moments)?,
            &mut mu,
        ),
    )?;
    let mut mean = s.slot(3, &[rows, 16])?;
    // Keep the mean subtraction as its own rounding boundary. BroadcastAdd
    // consumes a load-time negated mean; original constants stay attributable.
    let mut negative = mean_scratch(s)?;
    let mut neg = ops::PointwiseArgs::new(S::Scale, &w.mean, &mut negative);
    neg.dt = -1.;
    ops::pointwise(ctx, neg)?;
    binary(ctx, S::BroadcastAdd, &mu, &negative, &mut mean)?;
    binary(
        ctx,
        S::BroadcastMultiply,
        &mean,
        &w.inverse_std,
        &mut matrix(&s.output)?,
    )
}
fn mean_scratch(s: &VaeWorkspace) -> Result<Tensor> {
    view(&s.biased_tile, 0, &[1, 16])
}

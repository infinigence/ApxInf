//! The same joint network serves independently specified video/action times.
use super::super::{
    backend::{allocate, binary, linear, ops, transfers, Context, Policies},
    weights::dit::{DitWeights, Linear, DIM},
};
use super::dit_block::{self, projection, rows, BlockWorkspace};
use apxinf_core::{Error, Result, Tensor};

fn activate(
    ctx: &Context,
    x: &Tensor,
    kind: ops::PointwiseActivation,
    out: &mut Tensor,
) -> Result<()> {
    let mut args = ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, x, out);
    args.activation = kind;
    ops::pointwise(ctx, args)
}
// CategorySpecificLinear explicitly materializes bmm before adding its bias.
fn category(
    ctx: &Context,
    p: &Policies,
    x: &Tensor,
    w: &Linear,
    scratch: &mut Tensor,
    out: &mut Tensor,
) -> Result<()> {
    linear(ctx, p, x, &w.weight, scratch)?;
    binary(
        ctx,
        ops::PointwiseSemantic::BroadcastAdd,
        scratch,
        &w.bias.reshape(vec![1, w.bias.shape().numel()])?,
        out,
    )
}

pub(in crate::dreamzero) struct DitContext {
    pub text: Tensor,
    pub image: Tensor,
    text_linear: Tensor,
    text_activation: Tensor,
    image_norm: Tensor,
    image_linear: Tensor,
    image_activation: Tensor,
    image_projected: Tensor,
}
impl DitContext {
    pub fn new(ctx: &Context) -> Result<Self> {
        Ok(Self {
            text: allocate(ctx, &[512, DIM])?,
            image: allocate(ctx, &[257, DIM])?,
            text_linear: allocate(ctx, &[512, DIM])?,
            text_activation: allocate(ctx, &[512, DIM])?,
            image_norm: allocate(ctx, &[257, 1280])?,
            image_linear: allocate(ctx, &[257, 1280])?,
            image_activation: allocate(ctx, &[257, 1280])?,
            image_projected: allocate(ctx, &[257, DIM])?,
        })
    }
    pub fn encode(
        &mut self,
        ctx: &Context,
        p: &Policies,
        w: &DitWeights,
        text: &Tensor,
        clip: &Tensor,
    ) -> Result<()> {
        projection(ctx, p, text, &w.text[0], &mut self.text_linear)?;
        activate(
            ctx,
            &self.text_linear,
            ops::PointwiseActivation::Gelu,
            &mut self.text_activation,
        )?;
        projection(ctx, p, &self.text_activation, &w.text[1], &mut self.text)?;
        ops::norm(
            ctx,
            ops::NormArgs::layer(
                clip,
                &w.image_norm1.weight,
                &w.image_norm1.bias,
                &mut self.image_norm,
                1e-5,
            ),
        )?;
        projection(
            ctx,
            p,
            &self.image_norm,
            &w.image[0],
            &mut self.image_linear,
        )?;
        activate(
            ctx,
            &self.image_linear,
            ops::PointwiseActivation::GeluExact,
            &mut self.image_activation,
        )?;
        projection(
            ctx,
            p,
            &self.image_activation,
            &w.image[1],
            &mut self.image_projected,
        )?;
        ops::norm(
            ctx,
            ops::NormArgs::layer(
                &self.image_projected,
                &w.image_norm2.weight,
                &w.image_norm2.bias,
                &mut self.image,
                1e-5,
            ),
        )
    }
}

/// Video cache outputs contain no action/state rows. Their lifetime and
/// publication belong to the family runner, independently of prepared plans.
#[derive(Clone)]
pub(in crate::dreamzero) struct VideoCache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
    pub rows: usize,
}
impl VideoCache {
    pub fn new(ctx: &Context, rows: usize) -> Result<Self> {
        if rows == 0 || rows > 7920 || rows % 880 != 0 {
            return Err(Error::Other(
                "invalid DreamZero observed cache capacity".into(),
            ));
        }
        let mut keys = Vec::with_capacity(40);
        let mut values = Vec::with_capacity(40);
        for _ in 0..40 {
            keys.push(allocate(ctx, &[rows, DIM])?);
            values.push(allocate(ctx, &[rows, DIM])?);
        }
        Ok(Self { keys, values, rows })
    }
}

pub(super) fn rotary_coefficients(
    frames: usize,
    start_frame: usize,
    with_action: bool,
) -> Vec<f64> {
    let n = frames * 880;
    let mut coefficients = Vec::with_capacity((n + if with_action { 25 } else { 0 }) * 128);
    let mut position = |pos: usize, dims: usize| {
        for pair in 0..dims / 2 {
            let inverse_frequency = 1. / 10000f64.powf(2. * pair as f64 / dims as f64);
            let angle = pos as f64 * inverse_frequency;
            coefficients.extend([angle.cos(), angle.sin()]);
        }
    };
    for f in start_frame..start_frame + frames {
        for h in 0..22 {
            for w in 0..40 {
                position(f, 44);
                position(h, 42);
                position(w, 42);
            }
        }
    }
    if with_action {
        let index = (start_frame - 1) / 2;
        for a in index * 24..(index + 1) * 24 {
            position(a, 128);
        }
        position(index, 128);
    }
    coefficients
}

pub(in crate::dreamzero) struct DitWorkspace {
    frames: usize,
    with_action: bool,
    history_rows: usize,
    pub video: Tensor,
    pub action: Tensor,
    pub state: Tensor,
    /// Fixed per-step sinusoidal inputs: video and action/state, respectively.
    pub time_sinusoidal: Tensor,
    pub action_sinusoidal: Tensor,
    pub conditioning: Tensor,
    pub video_velocity: Tensor,
    pub action_velocity: Tensor,
    packed: Tensor,
    channels_last: Tensor,
    patches: Tensor,
    patch_linear: Tensor,
    patch_biased: Tensor,
    block: BlockWorkspace,
    time_linear: Tensor,
    time_activation: Tensor,
    time_embedded: Tensor,
    time_project_activation: Tensor,
    time_projected: Tensor,
    action_linear: Tensor,
    action_encoded: Tensor,
    action_concat: Tensor,
    action_hidden: Tensor,
    action_swish: Tensor,
    action_final: Tensor,
    state_linear: Tensor,
    state_biased: Tensor,
    state_relu: Tensor,
    state_final_linear: Tensor,
    state_final: Tensor,
    decoded_linear: Tensor,
    decoded_biased: Tensor,
    decoded_relu: Tensor,
    decoded_final_linear: Tensor,
    head_normalized: Tensor,
    head_modulation: Tensor,
    head_scale: Tensor,
    head_scaled: Tensor,
    head_shifted: Tensor,
    head_patches: Tensor,
    unit_weight: Tensor,
    zero_bias: Tensor,
    rotary: ops::ComplexRotaryTable,
}
impl DitWorkspace {
    pub fn new(
        ctx: &Context,
        frames: usize,
        start_frame: usize,
        history_rows: usize,
        with_action: bool,
    ) -> Result<Self> {
        if ![1, 2].contains(&frames)
            || start_frame + frames > 9
            || (with_action && (frames != 2 || start_frame == 0 || start_frame % 2 == 0))
        {
            return Err(Error::Other(
                "unsupported DreamZero temporal position/profile".into(),
            ));
        }
        let n = frames * 880;
        let coefficients = rotary_coefficients(frames, start_frame, with_action);
        let h = || allocate(ctx, &[24, DIM]);
        let state = || allocate(ctx, &[1, 1024]);
        let time = || allocate(ctx, &[2, DIM]);
        Ok(Self {
            frames,
            with_action,
            history_rows,
            video: allocate(ctx, &[16, frames, 44, 80])?,
            action: allocate(ctx, &[24, 32])?,
            state: allocate(ctx, &[1, 64])?,
            conditioning: allocate(ctx, &[20, frames, 44, 80])?,
            time_sinusoidal: allocate(ctx, &[2, 256])?,
            action_sinusoidal: h()?,
            video_velocity: allocate(ctx, &[16, frames, 44, 80])?,
            action_velocity: allocate(ctx, &[24, 32])?,
            packed: allocate(ctx, &[36, frames, 44, 80])?,
            channels_last: allocate(ctx, &[1, frames, 44, 80, 36])?,
            patches: allocate(ctx, &[n, 4 * 48])?,
            patch_linear: allocate(ctx, &[n, DIM])?,
            patch_biased: allocate(ctx, &[n, DIM])?,
            block: BlockWorkspace::new(ctx, n, if with_action { 25 } else { 0 }, history_rows)?,
            time_linear: time()?,
            time_activation: time()?,
            time_embedded: time()?,
            time_project_activation: time()?,
            time_projected: allocate(ctx, &[2, 6 * DIM])?,
            action_linear: h()?,
            action_encoded: h()?,
            action_concat: allocate(ctx, &[24, 2 * DIM])?,
            action_hidden: h()?,
            action_swish: h()?,
            action_final: h()?,
            state_linear: state()?,
            state_biased: state()?,
            state_relu: state()?,
            state_final_linear: allocate(ctx, &[1, DIM])?,
            state_final: allocate(ctx, &[1, DIM])?,
            decoded_linear: allocate(ctx, &[24, 1024])?,
            decoded_biased: allocate(ctx, &[24, 1024])?,
            decoded_relu: allocate(ctx, &[24, 1024])?,
            decoded_final_linear: allocate(ctx, &[24, 32])?,
            head_normalized: allocate(ctx, &[n, DIM])?,
            head_modulation: allocate(ctx, &[2, DIM])?,
            head_scale: allocate(ctx, &[1, DIM])?,
            head_scaled: allocate(ctx, &[n, DIM])?,
            head_shifted: allocate(ctx, &[n, DIM])?,
            head_patches: allocate(ctx, &[n, 64])?,
            unit_weight: transfers::to_cuda(
                &Tensor::from_bf16(vec![DIM], &vec![half::bf16::ONE; DIM])?,
                ctx.device_id(),
            )?,
            zero_bias: transfers::to_cuda(
                &Tensor::from_bf16(vec![DIM], &vec![half::bf16::ZERO; DIM])?,
                ctx.device_id(),
            )?,
            rotary: ops::ComplexRotaryTable::new(
                ctx,
                n + if with_action { 25 } else { 0 },
                128,
                &coefficients,
            )?,
        })
    }
    pub fn forward(
        &mut self,
        ctx: &Context,
        p: &Policies,
        w: &DitWeights,
        context: &DitContext,
        history: Option<&VideoCache>,
        next: Option<&VideoCache>,
    ) -> Result<()> {
        if history.map_or(0, |h| h.rows) != self.history_rows
            || (!self.with_action && next.is_none())
            || (self.with_action && next.is_some())
            || next.is_some_and(|n| {
                n.rows != self.history_rows + self.frames * 880
                    || n.keys.len() != 40
                    || n.values.len() != 40
            })
            || w.blocks.len() != 40
            || history.is_some_and(|h| h.keys.len() != 40 || h.values.len() != 40)
        {
            return Err(Error::Other("DreamZero DiT cache/profile mismatch".into()));
        }
        let pixels = self.frames * 44 * 80;
        let packed = self.packed.reshape(vec![36, pixels])?;
        transfers::copy_cuda_to_cuda(
            ctx,
            &self.video.reshape(vec![16, pixels])?,
            &rows(&packed, 0, 16)?,
        )?;
        transfers::copy_cuda_to_cuda(
            ctx,
            &self.conditioning.reshape(vec![20, pixels])?,
            &rows(&packed, 16, 20)?,
        )?;
        ops::permute(
            ctx,
            &self.packed,
            &[1, 2, 3, 0],
            &mut self.channels_last.reshape(vec![self.frames, 44, 80, 36])?,
        )?;
        ops::unfold3d(
            ctx,
            &self.channels_last,
            ops::Unfold3dGeometry {
                kernel: [1, 2, 2],
                stride: [1, 2, 2],
                dilation: [1, 1, 1],
                padding_before: [0; 3],
                padding_after: [0; 3],
            },
            0,
            &mut self.patches,
        )?;
        // Conv3D accumulation and bias are independently checked against the
        // pinned cuDNN reference during component acceptance.
        category(
            ctx,
            p,
            &self.patches,
            &w.patch,
            &mut self.patch_linear,
            &mut self.patch_biased,
        )?;
        let n = self.frames * 880;
        transfers::copy_cuda_to_cuda(ctx, &self.patch_biased, &rows(&self.block.hidden, 0, n)?)?;
        if self.with_action {
            category(
                ctx,
                p,
                &self.action,
                &w.action[0],
                &mut self.action_linear,
                &mut self.action_encoded,
            )?;
            let concatenated = self.action_concat.reshape(vec![48, DIM])?;
            for r in 0..24 {
                transfers::copy_cuda_to_cuda(
                    ctx,
                    &rows(&self.action_encoded, r, 1)?,
                    &rows(&concatenated, 2 * r, 1)?,
                )?;
                transfers::copy_cuda_to_cuda(
                    ctx,
                    &rows(&self.action_sinusoidal, r, 1)?,
                    &rows(&concatenated, 2 * r + 1, 1)?,
                )?;
            }
            category(
                ctx,
                p,
                &self.action_concat,
                &w.action[1],
                &mut self.action_linear,
                &mut self.action_hidden,
            )?;
            activate(
                ctx,
                &self.action_hidden,
                ops::PointwiseActivation::SwishRounded,
                &mut self.action_swish,
            )?;
            category(
                ctx,
                p,
                &self.action_swish,
                &w.action[2],
                &mut self.action_linear,
                &mut self.action_final,
            )?;
            category(
                ctx,
                p,
                &self.state,
                &w.state[0],
                &mut self.state_linear,
                &mut self.state_biased,
            )?;
            activate(
                ctx,
                &self.state_biased,
                ops::PointwiseActivation::Relu,
                &mut self.state_relu,
            )?;
            category(
                ctx,
                p,
                &self.state_relu,
                &w.state[1],
                &mut self.state_final_linear,
                &mut self.state_final,
            )?;
            transfers::copy_cuda_to_cuda(
                ctx,
                &self.action_final,
                &rows(&self.block.hidden, n, 24)?,
            )?;
            transfers::copy_cuda_to_cuda(
                ctx,
                &self.state_final,
                &rows(&self.block.hidden, n + 24, 1)?,
            )?;
        }
        projection(
            ctx,
            p,
            &self.time_sinusoidal,
            &w.time[0],
            &mut self.time_linear,
        )?;
        activate(
            ctx,
            &self.time_linear,
            ops::PointwiseActivation::Silu,
            &mut self.time_activation,
        )?;
        projection(
            ctx,
            p,
            &self.time_activation,
            &w.time[1],
            &mut self.time_embedded,
        )?;
        activate(
            ctx,
            &self.time_embedded,
            ops::PointwiseActivation::Silu,
            &mut self.time_project_activation,
        )?;
        projection(
            ctx,
            p,
            &self.time_project_activation,
            &w.time_projection,
            &mut self.time_projected,
        )?;
        for (i, b) in w.blocks.iter().enumerate() {
            dit_block::block(
                ctx,
                p,
                b,
                &mut self.block,
                &self.time_projected.reshape(vec![12, DIM])?,
                &context.text,
                &context.image,
                &self.rotary,
                history.map(|h| &h.keys[i]),
                history.map(|h| &h.values[i]),
                next.map(|n| (&n.keys[i], &n.values[i])),
            )?;
        }
        if self.with_action {
            category(
                ctx,
                p,
                &rows(&self.block.hidden, n, 24)?,
                &w.action_decoder[0],
                &mut self.decoded_linear,
                &mut self.decoded_biased,
            )?;
            activate(
                ctx,
                &self.decoded_biased,
                ops::PointwiseActivation::Relu,
                &mut self.decoded_relu,
            )?;
            category(
                ctx,
                p,
                &self.decoded_relu,
                &w.action_decoder[1],
                &mut self.decoded_final_linear,
                &mut self.action_velocity,
            )?;
        }
        ops::norm(
            ctx,
            ops::NormArgs::layer(
                &rows(&self.block.hidden, 0, n)?,
                &self.unit_weight,
                &self.zero_bias,
                &mut self.head_normalized,
                1e-6,
            ),
        )?;
        binary(
            ctx,
            ops::PointwiseSemantic::BroadcastAdd,
            &w.head_modulation,
            &rows(&self.time_embedded, 0, 1)?,
            &mut self.head_modulation,
        )?;
        let scale = rows(&self.head_modulation, 1, 1)?;
        let mut args = ops::PointwiseArgs::new(
            ops::PointwiseSemantic::ScalarAdd,
            &scale,
            &mut self.head_scale,
        );
        args.dt = 1.;
        ops::pointwise(ctx, args)?;
        binary(
            ctx,
            ops::PointwiseSemantic::BroadcastMultiply,
            &self.head_normalized,
            &self.head_scale,
            &mut self.head_scaled,
        )?;
        binary(
            ctx,
            ops::PointwiseSemantic::BroadcastAdd,
            &self.head_scaled,
            &rows(&self.head_modulation, 0, 1)?,
            &mut self.head_shifted,
        )?;
        projection(ctx, p, &self.head_shifted, &w.head, &mut self.head_patches)?;
        ops::permute(
            ctx,
            &self
                .head_patches
                .reshape(vec![self.frames, 22, 40, 1, 2, 2, 16])?,
            &[6, 0, 3, 1, 4, 2, 5],
            &mut self
                .video_velocity
                .reshape(vec![16, self.frames, 1, 22, 2, 40, 2])?,
        )
    }
}

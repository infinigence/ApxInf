//! Device vision stages from the policy's normalized RGB montage rows.
use super::super::{
    backend::{allocate, ops, transfers, Context, Policies},
    weights::{clip::ClipWeights, vae::VaeWeights},
};
use super::{
    clip::{self, ClipWorkspace},
    dit_block::rows,
    vae::{self, VaeWorkspace},
};
use crate::dreamzero::backend;
use apxinf_core::{DType, Error, Result, Shape, Tensor};

const PIXELS: usize = 352 * 640;
const CLIP_PIXELS: usize = 224 * 224;

fn scalar(
    ctx: &Context,
    x: &Tensor,
    semantic: ops::PointwiseSemantic,
    value: f32,
    out: &mut Tensor,
) -> Result<()> {
    let mut args = ops::PointwiseArgs::new(semantic, x, out);
    args.dt = value;
    ops::pointwise(ctx, args)
}
fn zero(ctx: &Context, x: &Tensor) -> Result<()> {
    backend::CudaBuffer::from_tensor(x)
        .map_err(Error::Cuda)?
        .zero_async(ctx.stream())
        .map_err(Error::Cuda)
}

pub(in crate::dreamzero) struct VisionWorkspace {
    initial: bool,
    input_frames: usize,
    /// BF16 rows [frames*352*640,3], after policy normalization.
    pub images: Tensor,
    /// Initial-only CLIP feature; otherwise retained by the episode owner.
    pub clip: ClipWorkspace,
    pub vae: VaeWorkspace,
    /// Initial mask + all nine conditioning latents, in channel-first order.
    pub conditioning: Tensor,
    /// Current observed latent frames (one initial, two subsequent).
    pub observed: Tensor,
    input_float: Tensor,
    resized: Tensor,
    planar: Tensor,
    scaled: Tensor,
    shifted: Tensor,
    centered: Tensor,
    normalized: Tensor,
    normalized_bf16: Tensor,
    normalized_hwc: Tensor,
    latent_planar: Tensor,
    mask_first: Tensor,
}
impl VisionWorkspace {
    pub fn new(ctx: &Context, input_frames: usize, initial: bool) -> Result<Self> {
        if ![1, 4, 9].contains(&input_frames) || (!initial && input_frames == 1) {
            return Err(Error::Other(
                "DreamZero vision needs 1 initial or 4/9 history frames".into(),
            ));
        }
        let clip = || ctx.allocate_output(Shape::new(vec![3, CLIP_PIXELS]), DType::F32);
        let latent_frames = if initial { 9 } else { 3 };
        Ok(Self {
            initial,
            input_frames,
            images: allocate(ctx, &[input_frames * PIXELS, 3])?,
            clip: ClipWorkspace::new(ctx)?,
            vae: VaeWorkspace::new(ctx, if initial { 33 } else { 9 }, 352, 640, initial)?,
            conditioning: allocate(ctx, &[20, 9, 44, 80])?,
            observed: allocate(ctx, &[16, if initial { 1 } else { 2 }, 44, 80])?,
            input_float: ctx.allocate_output(Shape::new(vec![PIXELS, 3]), DType::F32)?,
            resized: ctx.allocate_output(Shape::new(vec![1, 224, 224, 3]), DType::F32)?,
            planar: clip()?,
            scaled: clip()?,
            shifted: clip()?,
            centered: clip()?,
            normalized: clip()?,
            normalized_bf16: allocate(ctx, &[3, CLIP_PIXELS])?,
            normalized_hwc: allocate(ctx, &[1, 1, 224, 224, 3])?,
            latent_planar: allocate(ctx, &[16, latent_frames, 44, 80])?,
            mask_first: transfers::to_cuda(
                &Tensor::from_bf16(vec![1, 44 * 80], &vec![half::bf16::ONE; 44 * 80])?,
                ctx.device_id(),
            )?,
        })
    }
    pub fn forward(
        &mut self,
        ctx: &Context,
        p: &Policies,
        clip: &ClipWeights,
        vae: &VaeWeights,
    ) -> Result<()> {
        let vae_rows = self.vae.input.reshape(vec![
            if self.initial {
                33 * PIXELS
            } else {
                9 * PIXELS
            },
            3,
        ])?;
        if self.initial {
            zero(ctx, &self.vae.input)?;
            let selected = if self.input_frames == 1 {
                0
            } else {
                self.input_frames - 1
            };
            let image = rows(&self.images, selected * PIXELS, PIXELS)?;
            transfers::copy_cuda_to_cuda(ctx, &image, &rows(&vae_rows, 0, PIXELS)?)?;
            // The official encode_image autocast scope promotes bicubic
            // interpolation and its subsequent normalization to FP32. Cast
            // back only at the learned CLIP encoder boundary.
            ops::quantization(
                ctx,
                ops::QuantizationArgs::new(
                    ops::QuantizationSemantic::CastBf16ToF32,
                    &image,
                    &mut self.input_float,
                ),
            )?;
            ops::resize_bicubic(
                ctx,
                &self.input_float.reshape(vec![1, 352, 640, 3])?,
                &mut self.resized,
            )?;
            ops::permute(
                ctx,
                &self.resized.reshape(vec![224, 224, 3])?,
                &[2, 0, 1],
                &mut self.planar.reshape(vec![3, 224, 224])?,
            )?;
            scalar(
                ctx,
                &self.planar,
                ops::PointwiseSemantic::Scale,
                0.5,
                &mut self.scaled,
            )?;
            scalar(
                ctx,
                &self.scaled,
                ops::PointwiseSemantic::ScalarAdd,
                0.5,
                &mut self.shifted,
            )?;
            for (c, (mean, std)) in [0.48145466f32, 0.4578275, 0.40821073]
                .into_iter()
                .zip([0.26862954f32, 0.26130258, 0.27577711])
                .enumerate()
            {
                scalar(
                    ctx,
                    &rows(&self.shifted, c, 1)?,
                    ops::PointwiseSemantic::ScalarAdd,
                    -mean,
                    &mut rows(&self.centered, c, 1)?,
                )?;
                scalar(
                    ctx,
                    &rows(&self.centered, c, 1)?,
                    ops::PointwiseSemantic::ScalarDivide,
                    std,
                    &mut rows(&self.normalized, c, 1)?,
                )?;
            }
            ops::quantization(
                ctx,
                ops::QuantizationArgs::new(
                    ops::QuantizationSemantic::CastF32ToBf16,
                    &self.normalized,
                    &mut self.normalized_bf16,
                ),
            )?;
            ops::permute(
                ctx,
                &self.normalized_bf16.reshape(vec![3, 224, 224])?,
                &[1, 2, 0],
                &mut self.normalized_hwc.reshape(vec![224, 224, 3])?,
            )?;
            ops::unfold3d(
                ctx,
                &self.normalized_hwc,
                ops::Unfold3dGeometry {
                    kernel: [1, 14, 14],
                    stride: [1, 14, 14],
                    dilation: [1; 3],
                    padding_before: [0; 3],
                    padding_after: [0; 3],
                },
                0,
                &mut self.clip.patches,
            )?;
            clip::encode_patches(ctx, p, clip, &mut self.clip)?;
        } else if self.input_frames == 9 {
            transfers::copy_cuda_to_cuda(ctx, &self.images, &vae_rows)?;
        } else {
            // Four observed frames are repeated twice, then the first frame
            // is prepended: [0,0,0,1,1,2,2,3,3]. This is an observation commit.
            for (dst, src) in [0, 0, 0, 1, 1, 2, 2, 3, 3].into_iter().enumerate() {
                transfers::copy_cuda_to_cuda(
                    ctx,
                    &rows(&self.images, src * PIXELS, PIXELS)?,
                    &rows(&vae_rows, dst * PIXELS, PIXELS)?,
                )?;
            }
        }
        vae::encode(ctx, p, vae, &self.vae)?;
        let frames = if self.initial { 9 } else { 3 };
        ops::permute(
            ctx,
            &self.vae.output.reshape(vec![frames, 44, 80, 16])?,
            &[3, 0, 1, 2],
            &mut self.latent_planar,
        )?;
        let planar = self.latent_planar.reshape(vec![16 * frames, 44 * 80])?;
        if self.initial {
            zero(ctx, &self.conditioning)?;
            let condition = self.conditioning.reshape(vec![20 * 9, 44 * 80])?;
            for channel in 0..4 {
                transfers::copy_cuda_to_cuda(
                    ctx,
                    &self.mask_first,
                    &rows(&condition, channel * 9, 1)?,
                )?;
            }
            transfers::copy_cuda_to_cuda(ctx, &planar, &rows(&condition, 4 * 9, 16 * 9)?)?;
        }
        let count = if self.initial { 1 } else { 2 };
        let observed = self.observed.reshape(vec![16 * count, 44 * 80])?;
        for c in 0..16 {
            transfers::copy_cuda_to_cuda(
                ctx,
                &rows(
                    &planar,
                    c * frames + if self.initial { 0 } else { 1 },
                    count,
                )?,
                &rows(&observed, c * count, count)?,
            )?;
        }
        Ok(())
    }
}

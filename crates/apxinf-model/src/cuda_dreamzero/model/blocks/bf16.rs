// BF16 Blocks: physical tensors, fused operators, modality subgraphs and storage.
use crate::cuda_dreamzero::{
    backend::{ops, transfers, Context},
    config::Policies,
};
use apxinf_core::{DType, Result, Shape, Tensor};
pub(in crate::cuda_dreamzero) fn allocate(ctx: &Context, dims: &[usize]) -> Result<Tensor> {
    ctx.allocate_output(Shape::new(dims.to_vec()), DType::BF16)
}

pub(in crate::cuda_dreamzero) fn linear(
    ctx: &Context,
    policy: &Policies,
    x: &Tensor,
    w: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let mut args = ops::GemmArgs::new(x, w, out);
    args.policy = policy.gemm.clone();
    ops::gemm(ctx, args)
}

pub(in crate::cuda_dreamzero) fn binary(
    ctx: &Context,
    semantic: ops::PointwiseSemantic,
    x: &Tensor,
    y: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let mut args = ops::PointwiseArgs::new(semantic, x, out);
    args.secondary = Some(y);
    ops::pointwise(ctx, args)
}

fn layer(ctx: &Context, mut args: ops::NormArgs<'_>) -> Result<()> {
    args.numerics = ops::NormNumerics::Welford;
    ops::norm(ctx, args)
}
pub(in crate::cuda_dreamzero) fn staged_rms(
    ctx: &Context,
    mut args: ops::NormArgs<'_>,
) -> Result<()> {
    args.numerics = ops::NormNumerics::Staged;
    ops::norm(ctx, args)
}
pub(in crate::cuda_dreamzero) fn staged_l2(
    ctx: &Context,
    input: &Tensor,
    output: &mut Tensor,
    eps: f32,
) -> Result<()> {
    ops::norm(
        ctx,
        ops::NormArgs {
            input,
            output: Some(output),
            kind: ops::NormKind::L2,
            affine: ops::Affine::None,
            quantized: None,
            numerics: ops::NormNumerics::Staged,
            eps,
        },
    )
}

mod text {
    use super::{allocate, binary, linear};
    use crate::cuda_dreamzero::{
        backend::{ops, Context},
        config::Policies,
        weights::{TextBlockWeights, TextWeights},
        DreamZeroConfig as C,
    };
    use apxinf_core::{Result, Tensor};
    use ops::{PointwiseActivation as A, PointwiseSemantic as S};

    /// Explicit reusable storage avoids retaining an activation arena for every
    /// text layer. Allocate outside preparation/capture; all forward writes are
    /// device-only and graph addresses remain stable.
    pub(in crate::cuda_dreamzero::model) struct TextWorkspace {
        pub ids: Tensor,
        /// `[1,512]`: zero for valid keys, BF16 minimum for padding keys.
        pub mask: Tensor,
        pub output: Tensor,
        pub(super) hidden: Tensor,
        norm: Tensor,
        residual: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention: Tensor,
        projection: Tensor,
        gate: Tensor,
        activated: Tensor,
        up: Tensor,
        product: Tensor,
        bias: Tensor,
    }
    impl TextWorkspace {
        pub fn new(ctx: &Context) -> Result<Self> {
            let hidden = || allocate(ctx, &[C::TEXT_LEN, C::TEXT_DIM]);
            let ffn = || allocate(ctx, &[C::TEXT_LEN, C::TEXT_FFN]);
            Ok(Self {
                ids: ctx.allocate_output(
                    apxinf_core::Shape::new(vec![C::TEXT_LEN]),
                    apxinf_core::DType::I32,
                )?,
                mask: allocate(ctx, &[1, C::TEXT_LEN])?,
                output: hidden()?,
                hidden: hidden()?,
                norm: hidden()?,
                residual: hidden()?,
                q: hidden()?,
                k: hidden()?,
                v: hidden()?,
                attention: hidden()?,
                projection: hidden()?,
                gate: ffn()?,
                activated: ffn()?,
                up: ffn()?,
                product: ffn()?,
                bias: allocate(ctx, &[C::TEXT_HEADS * C::TEXT_LEN, C::TEXT_LEN])?,
            })
        }
    }

    /// Exact BF16 UMT5 computation, including each FFN and attention rounding.
    pub(in crate::cuda_dreamzero::model) fn encode(
        ctx: &Context,
        policies: &Policies,
        weights: &TextWeights,
        work: &mut TextWorkspace,
    ) -> Result<()> {
        ops::embedding_gather(ctx, &weights.embedding, &work.ids, &work.hidden)?;
        for block in &weights.blocks {
            encode_block(ctx, policies, block, work)?;
        }
        crate::cuda_dreamzero::model::blocks::staged_rms(
            ctx,
            ops::NormArgs::rms(&work.hidden, &weights.norm, &mut work.output, 1e-6),
        )
    }

    pub(super) fn encode_block(
        ctx: &Context,
        policies: &Policies,
        block: &TextBlockWeights,
        work: &mut TextWorkspace,
    ) -> Result<()> {
        let n = C::TEXT_LEN;
        crate::cuda_dreamzero::model::blocks::staged_rms(
            ctx,
            ops::NormArgs::rms(&work.hidden, &block.norm1, &mut work.norm, 1e-6),
        )?;
        linear(ctx, policies, &work.norm, &block.q, &mut work.q)?;
        linear(ctx, policies, &work.norm, &block.k, &mut work.k)?;
        linear(ctx, policies, &work.norm, &block.v, &mut work.v)?;
        binary(
            ctx,
            S::BroadcastAdd,
            &block.position.reshape(vec![C::TEXT_HEADS * n, n])?,
            &work.mask,
            &mut work.bias,
        )?;
        let shape = vec![1, n, C::TEXT_HEADS, C::TEXT_DIM / C::TEXT_HEADS];
        let q = work.q.reshape(shape.clone())?;
        let k = work.k.reshape(shape.clone())?;
        let v = work.v.reshape(shape.clone())?;
        let mut out = work.attention.reshape(shape)?;
        let bias = work.bias.reshape(vec![1, C::TEXT_HEADS, n, n])?;
        let mut attention = ops::AttentionArgs::new(&q, &k, &v, &mut out);
        attention.scale = 1.;
        attention.policy = policies.attention.clone();
        ops::rounded_additive_attention(
            ctx,
            ops::RoundedAttentionBiasArgs {
                attention,
                bias: &bias,
            },
        )?;
        linear(
            ctx,
            policies,
            &work.attention,
            &block.o,
            &mut work.projection,
        )?;
        binary(
            ctx,
            S::Add,
            &work.hidden,
            &work.projection,
            &mut work.residual,
        )?;
        crate::cuda_dreamzero::model::blocks::staged_rms(
            ctx,
            ops::NormArgs::rms(&work.residual, &block.norm2, &mut work.norm, 1e-6),
        )?;
        linear(ctx, policies, &work.norm, &block.gate, &mut work.gate)?;
        let mut activation =
            ops::PointwiseArgs::new(S::BiasActivation, &work.gate, &mut work.activated);
        activation.activation = A::GeluRounded;
        ops::pointwise(ctx, activation)?;
        linear(ctx, policies, &work.norm, &block.up, &mut work.up)?;
        binary(
            ctx,
            S::Multiply,
            &work.up,
            &work.activated,
            &mut work.product,
        )?;
        linear(
            ctx,
            policies,
            &work.product,
            &block.down,
            &mut work.projection,
        )?;
        binary(
            ctx,
            S::Add,
            &work.residual,
            &work.projection,
            &mut work.hidden,
        )?;
        Ok(())
    }
}
mod clip {
    use super::layer;
    use super::{allocate, binary, linear};
    use crate::cuda_dreamzero::backend;
    use crate::cuda_dreamzero::{
        backend::{ops, transfers, Context},
        config::Policies,
        weights::clip::{ClipBlock, ClipWeights, Linear, Norm, DIM, HEADS, TOKENS},
    };
    use apxinf_core::{Error, Result, Shape, Tensor};

    pub(in crate::cuda_dreamzero::model) struct ClipWorkspace {
        pub patches: Tensor,
        padded_patches: Tensor,
        pub output: Tensor,
        pub(super) projected: Tensor,
        embedded: Tensor,
        position: Tensor,
        pub(super) hidden: Tensor,
        pub(super) normalized: Tensor,
        pub(super) residual: Tensor,
        qkv: Tensor,
        qkv_split: Tensor,
        pub(super) q: Tensor,
        pub(super) k: Tensor,
        pub(super) v: Tensor,
        pub(super) attention: Tensor,
        pub(super) projected_attention: Tensor,
        pub(super) up: Tensor,
        pub(super) activated: Tensor,
    }
    impl ClipWorkspace {
        pub fn new(ctx: &Context) -> Result<Self> {
            let hidden = || allocate(ctx, &[TOKENS, DIM]);
            let qkv_split = allocate(ctx, &[3, TOKENS, DIM])?;
            let flat = qkv_split.reshape(vec![3 * TOKENS, DIM])?;
            let q = rows(&flat, 0, TOKENS)?;
            let k = rows(&flat, TOKENS, TOKENS)?;
            let v = rows(&flat, 2 * TOKENS, TOKENS)?;
            Ok(Self {
                patches: allocate(ctx, &[256, 14 * 14 * 3])?,
                padded_patches: allocate(ctx, &[256, 14 * 14 * 16])?,
                output: hidden()?,
                projected: allocate(ctx, &[256, DIM])?,
                embedded: hidden()?,
                position: hidden()?,
                hidden: hidden()?,
                normalized: hidden()?,
                residual: hidden()?,
                qkv: allocate(ctx, &[TOKENS, 3 * DIM])?,
                qkv_split,
                q,
                k,
                v,
                attention: hidden()?,
                projected_attention: hidden()?,
                up: allocate(ctx, &[TOKENS, 4 * DIM])?,
                activated: allocate(ctx, &[TOKENS, 4 * DIM])?,
            })
        }
    }
    fn norm(ctx: &Context, x: &Tensor, w: &Norm, y: &mut Tensor) -> Result<()> {
        layer(ctx, ops::NormArgs::layer(x, &w.weight, &w.bias, y, 1e-5))
    }
    fn projection(
        ctx: &Context,
        p: &Policies,
        x: &Tensor,
        w: &Linear,
        y: &mut Tensor,
    ) -> Result<()> {
        let mut gemm = ops::GemmArgs::new(x, &w.weight, y);
        gemm.policy = p.gemm.clone();
        // Preserve the fixed BF16 reference's Lt reduction/workspace profile.
        gemm.policy.workspace_limit = gemm.policy.workspace_limit.min(1 << 20);
        gemm.weight_layout = ops::GemmWeightLayout::OutputMajor;
        gemm.policy.numerics = ops::GemmNumerics::StagedBias;
        ops::gemm_bias(
            ctx,
            ops::GemmBiasArgs {
                gemm,
                bias: &w.bias,
            },
        )
    }
    fn rows(x: &Tensor, offset: usize, count: usize) -> Result<Tensor> {
        let buffer = backend::CudaBuffer::from_tensor(x).map_err(Error::Cuda)?;
        let view = buffer
            .view(offset * DIM * 2, count * DIM * 2)
            .map_err(Error::Cuda)?;
        view.as_tensor(Shape::new(vec![count, DIM]), x.dtype())
            .map_err(Error::Cuda)
    }
    /// CLIP visual encoder from device-resident normalized 14×14 RGB patches.
    /// Image resizing/normalization populates `patches` before this computation.
    pub(in crate::cuda_dreamzero::model) fn encode_patches(
        ctx: &Context,
        p: &Policies,
        w: &ClipWeights,
        s: &mut ClipWorkspace,
    ) -> Result<()> {
        encode_prefix(ctx, p, w, s)?;
        for b in &w.blocks {
            encode_block(ctx, p, b, s)?;
        }
        transfers::copy_cuda_to_cuda(ctx, &s.hidden, &s.output)
    }
    pub(super) fn encode_prefix(
        ctx: &Context,
        p: &Policies,
        w: &ClipWeights,
        s: &mut ClipWorkspace,
    ) -> Result<()> {
        // Match the reference convolution's channel-aligned tensor-core reduction.
        // Zero extension changes the reduction layout, not the convolution values.
        ops::unfold3d(
            ctx,
            &s.patches.reshape(vec![256, 1, 14, 14, 3])?,
            ops::Unfold3dGeometry {
                kernel: [1, 14, 14],
                stride: [1, 1, 1],
                dilation: [1, 1, 1],
                padding_before: [0; 3],
                padding_after: [0; 3],
            },
            0,
            &mut s.padded_patches,
        )?;
        linear(ctx, p, &s.padded_patches, &w.patch, &mut s.projected)?;
        transfers::copy_cuda_to_cuda(ctx, &w.class, &rows(&s.embedded, 0, 1)?)?;
        transfers::copy_cuda_to_cuda(ctx, &s.projected, &rows(&s.embedded, 1, 256)?)?;
        binary(
            ctx,
            ops::PointwiseSemantic::Add,
            &s.embedded,
            &w.position,
            &mut s.position,
        )?;
        norm(ctx, &s.position, &w.pre_norm, &mut s.hidden)?;
        Ok(())
    }
    pub(super) fn encode_block(
        ctx: &Context,
        p: &Policies,
        b: &ClipBlock,
        s: &mut ClipWorkspace,
    ) -> Result<()> {
        norm(ctx, &s.hidden, &b.norm1, &mut s.normalized)?;
        projection(ctx, p, &s.normalized, &b.qkv, &mut s.qkv)?;
        ops::permute(
            ctx,
            &s.qkv.reshape(vec![TOKENS, 3, DIM])?,
            &[1, 0, 2],
            &mut s.qkv_split,
        )?;
        let shape = vec![1, TOKENS, HEADS, DIM / HEADS];
        let q = s.q.reshape(shape.clone())?;
        let k = s.k.reshape(shape.clone())?;
        let v = s.v.reshape(shape.clone())?;
        let mut out = s.attention.reshape(shape)?;
        let mut args = ops::AttentionArgs::new(&q, &k, &v, &mut out);
        args.policy = p.attention.clone();
        // Keep all 31 prepared layers on the existing linear-memory FA2 path.
        args.policy.workspace_limit = 2 << 20;
        ops::attention(ctx, args)?;
        projection(ctx, p, &s.attention, &b.o, &mut s.projected_attention)?;
        binary(
            ctx,
            ops::PointwiseSemantic::Add,
            &s.hidden,
            &s.projected_attention,
            &mut s.residual,
        )?;
        norm(ctx, &s.residual, &b.norm2, &mut s.normalized)?;
        projection(ctx, p, &s.normalized, &b.up, &mut s.up)?;
        let mut activation = ops::PointwiseArgs::new(
            ops::PointwiseSemantic::BiasActivation,
            &s.up,
            &mut s.activated,
        );
        activation.activation = ops::PointwiseActivation::GeluExact;
        ops::pointwise(ctx, activation)?;
        projection(ctx, p, &s.activated, &b.down, &mut s.projected_attention)?;
        binary(
            ctx,
            ops::PointwiseSemantic::Add,
            &s.residual,
            &s.projected_attention,
            &mut s.hidden,
        )?;
        Ok(())
    }
}
mod dit {
    use super::layer;
    use super::{allocate, binary, linear};
    // The same joint network serves independently specified video/action times.
    use super::dit_block::{self, projection, rows, BlockWorkspace, CrossAttentionContext};
    use crate::cuda_dreamzero::{
        backend::{ops, transfers, Context},
        config::Policies,
        weights::dit::{DitWeights, Linear, DIM},
    };
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

    pub(in crate::cuda_dreamzero::model) struct DitContext {
        pub text: Tensor,
        pub image: Tensor,
        cross_attention: Vec<CrossAttentionContext>,
        text_k_raw: Tensor,
        image_k_raw: Tensor,
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
                cross_attention: (0..40)
                    .map(|_| CrossAttentionContext::new(ctx))
                    .collect::<Result<Vec<_>>>()?,
                text_k_raw: allocate(ctx, &[512, DIM])?,
                image_k_raw: allocate(ctx, &[257, DIM])?,
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
            layer(
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
            layer(
                ctx,
                ops::NormArgs::layer(
                    &self.image_projected,
                    &w.image_norm2.weight,
                    &w.image_norm2.bias,
                    &mut self.image,
                    1e-5,
                ),
            )?;
            // Context does not depend on diffusion time or observed history.
            // Rebuild all layer K/V in the Language graph on every request,
            // including padded rows after the biased context projections.
            for (context, block) in self.cross_attention.iter_mut().zip(&w.blocks) {
                context.encode(
                    ctx,
                    p,
                    block,
                    &self.text,
                    &self.image,
                    &mut self.text_k_raw,
                    &mut self.image_k_raw,
                )?;
            }
            Ok(())
        }
    }

    /// Video cache outputs contain no action/state rows. Their lifetime and
    /// publication belong to the family runner, independently of prepared plans.
    #[derive(Clone)]
    pub(in crate::cuda_dreamzero::model) struct VideoCache {
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

    pub(in crate::cuda_dreamzero::model) struct DitWorkspace {
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
            transfers::copy_cuda_to_cuda(
                ctx,
                &self.patch_biased,
                &rows(&self.block.hidden, 0, n)?,
            )?;
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
                    &context.cross_attention[i],
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
            layer(
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
}
mod dit_block {
    use super::layer;
    // Causal joint transformer block. Only observed video K/V is eligible for
    // publication; generated video and action tokens remain call-local outputs.
    use crate::cuda_dreamzero::backend;
    use crate::cuda_dreamzero::{
        backend::{ops, transfers, Context},
        config::Policies,
        model::blocks::{allocate, binary},
        weights::dit::{Block, Linear, Norm, DIM, FFN, HEADS},
    };
    use apxinf_core::{Error, Result, Shape, Tensor};

    pub(super) fn rows(x: &Tensor, start: usize, count: usize) -> Result<Tensor> {
        let dims = x.shape().dims();
        if dims.len() != 2 || start.checked_add(count).is_none_or(|end| end > dims[0]) {
            return Err(Error::Other(
                "DreamZero matrix row view out of bounds".into(),
            ));
        }
        let width = dims[1];
        backend::CudaBuffer::from_tensor(x)
            .map_err(Error::Cuda)?
            .view(
                start * width * x.dtype().size_in_bytes(),
                count * width * x.dtype().size_in_bytes(),
            )
            .map_err(Error::Cuda)?
            .as_tensor(Shape::new(vec![count, width]), x.dtype())
            .map_err(Error::Cuda)
    }
    pub(super) fn projection(
        ctx: &Context,
        p: &Policies,
        x: &Tensor,
        w: &Linear,
        out: &mut Tensor,
    ) -> Result<()> {
        let mut gemm = ops::GemmArgs::new(x, &w.weight, out);
        gemm.policy = p.gemm.clone();
        // Match the fixed BF16 reference's Lt workspace budget. Its narrow output
        // head uses F32 split-K; a different budget changes reduction grouping.
        gemm.policy.workspace_limit = gemm.policy.workspace_limit.min(1 << 20);
        gemm.weight_layout = ops::GemmWeightLayout::OutputMajor;
        gemm.policy.numerics = ops::GemmNumerics::StagedBias;
        ops::gemm_bias(
            ctx,
            ops::GemmBiasArgs {
                gemm,
                bias: &w.bias,
            },
        )?;
        Ok(())
    }
    fn norm(ctx: &Context, x: &Tensor, w: &Norm, out: &mut Tensor) -> Result<()> {
        layer(ctx, ops::NormArgs::layer(x, &w.weight, &w.bias, out, 1e-6))?;
        Ok(())
    }
    fn rms(ctx: &Context, x: &Tensor, weight: &Tensor, out: &mut Tensor) -> Result<()> {
        crate::cuda_dreamzero::model::blocks::staged_rms(
            ctx,
            ops::NormArgs::rms(x, weight, out, 1e-6),
        )?;
        Ok(())
    }
    fn attend(
        ctx: &Context,
        p: &Policies,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &mut Tensor,
    ) -> Result<()> {
        let shape = |x: &Tensor| vec![1, x.shape().dims()[0], HEADS, DIM / HEADS];
        let q = q.reshape(shape(q))?;
        let k = k.reshape(shape(k))?;
        let v = v.reshape(shape(v))?;
        let mut out = out.reshape(shape(out))?;
        let mut args = ops::AttentionArgs::new(&q, &k, &v, &mut out);
        args.policy = p.attention.clone();
        // Keep self, text and image attention on the same linear-memory cold
        // path, including short cross-attention shapes that fit dense scratch.
        args.policy.workspace_limit = 2 << 20;
        ops::attention(ctx, args)?;
        Ok(())
    }

    fn block_policies(p: &Policies) -> Policies {
        // Keep bulk transformer reductions independent of encoder/output heads.
        let mut policies = p.clone();
        policies.gemm.algorithm = ops::GemmAlgorithmPolicy::Any;
        policies
    }

    /// Per-request, per-layer cross-attention K/V. These tensors are produced
    /// by Language and read by every observed/joint transformer evaluation.
    /// This is prepared workspace, never a cache shared between requests.
    pub(in crate::cuda_dreamzero::model) struct CrossAttentionContext {
        text_k: Tensor,
        text_v: Tensor,
        image_k: Tensor,
        image_v: Tensor,
    }
    impl CrossAttentionContext {
        pub fn new(ctx: &Context) -> Result<Self> {
            Ok(Self {
                text_k: allocate(ctx, &[512, DIM])?,
                text_v: allocate(ctx, &[512, DIM])?,
                image_k: allocate(ctx, &[257, DIM])?,
                image_v: allocate(ctx, &[257, DIM])?,
            })
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode(
            &mut self,
            ctx: &Context,
            p: &Policies,
            w: &Block,
            text: &Tensor,
            image: &Tensor,
            text_k_raw: &mut Tensor,
            image_k_raw: &mut Tensor,
        ) -> Result<()> {
            let p = &block_policies(p);
            let a = &w.cross_attention;
            projection(ctx, p, text, &a.k, text_k_raw)?;
            rms(ctx, text_k_raw, &a.norm_k, &mut self.text_k)?;
            projection(ctx, p, text, &a.v, &mut self.text_v)?;
            projection(ctx, p, image, &w.image_k, image_k_raw)?;
            rms(ctx, image_k_raw, &w.image_norm_k, &mut self.image_k)?;
            projection(ctx, p, image, &w.image_v, &mut self.image_v)
        }
    }

    pub(in crate::cuda_dreamzero::model) struct BlockWorkspace {
        rows: usize,
        video_rows: usize,
        history_rows: usize,
        pub hidden: Tensor,
        normed: Tensor,
        shifted: Tensor,
        q_raw: Tensor,
        q: Tensor,
        attended: Tensor,
        projected: Tensor,
        residual: Tensor,
        up: Tensor,
        activated: Tensor,
        kv_k: Tensor,
        kv_v: Tensor,
        image_attended: Tensor,
        modulation: Tensor,
        unit_norm: Norm,
    }
    impl BlockWorkspace {
        pub fn new(
            ctx: &Context,
            video_rows: usize,
            action_rows: usize,
            history_rows: usize,
        ) -> Result<Self> {
            if ![880, 1760].contains(&video_rows)
                || ![0, 25].contains(&action_rows)
                || history_rows % 880 != 0
                || history_rows + video_rows > 7920
            {
                return Err(Error::Other("unsupported DreamZero block geometry".into()));
            }
            let n = video_rows + action_rows;
            let h = || allocate(ctx, &[n, DIM]);
            Ok(Self {
                rows: n,
                video_rows,
                history_rows,
                hidden: h()?,
                normed: h()?,
                shifted: h()?,
                q_raw: h()?,
                q: h()?,
                attended: h()?,
                projected: h()?,
                residual: h()?,
                up: allocate(ctx, &[n, FFN])?,
                activated: allocate(ctx, &[n, FFN])?,
                kv_k: allocate(ctx, &[history_rows + n, DIM])?,
                kv_v: allocate(ctx, &[history_rows + n, DIM])?,
                image_attended: h()?,
                modulation: allocate(ctx, &[12, DIM])?,
                unit_norm: Norm {
                    weight: transfers::to_cuda(
                        &Tensor::from_bf16(vec![DIM], &vec![half::bf16::ONE; DIM])?,
                        ctx.device_id(),
                    )?,
                    bias: transfers::to_cuda(
                        &Tensor::from_bf16(vec![DIM], &vec![half::bf16::ZERO; DIM])?,
                        ctx.device_id(),
                    )?,
                },
            })
        }
        /// Time projections contain one six-vector group for video and one for
        /// action/state. This represents independent modality schedules directly.
        fn bind_modulation(&mut self, ctx: &Context, w: &Tensor, time: &Tensor) -> Result<()> {
            if time.shape().dims() != [12, DIM] {
                return Err(Error::Other(
                    "DreamZero time modulation must be [12,5120]".into(),
                ));
            }
            for group in 0..2 {
                binary(
                    ctx,
                    ops::PointwiseSemantic::Add,
                    &rows(time, group * 6, 6)?,
                    w,
                    &mut rows(&self.modulation, group * 6, 6)?,
                )?;
            }
            Ok(())
        }
        fn modulate(&mut self, ctx: &Context, shift: usize, scale: usize) -> Result<()> {
            for (group, start, count) in [
                (0, 0, self.video_rows),
                (1, self.video_rows, self.rows - self.video_rows),
            ] {
                if count == 0 {
                    continue;
                }
                let scale = rows(&self.modulation, group * 6 + scale, 1)?;
                let shift = rows(&self.modulation, group * 6 + shift, 1)?.reshape(vec![DIM])?;
                let input = rows(&self.normed, start, count)?;
                let mut output = rows(&self.shifted, start, count)?;
                let mut args = ops::PointwiseArgs::new(
                    ops::PointwiseSemantic::StagedModulation,
                    &input,
                    &mut output,
                );
                args.secondary = Some(&scale);
                args.bias = Some(&shift);
                ops::pointwise(ctx, args)?;
            }
            Ok(())
        }
        fn gate_residual(&self, ctx: &Context, gate: usize, output: &Tensor) -> Result<()> {
            for (group, start, count) in [
                (0, 0, self.video_rows),
                (1, self.video_rows, self.rows - self.video_rows),
            ] {
                if count == 0 {
                    continue;
                }
                let input = rows(&self.projected, start, count)?;
                let residual = rows(&self.hidden, start, count)?;
                let gate = rows(&self.modulation, group * 6 + gate, 1)?.reshape(vec![DIM])?;
                let mut out = rows(output, start, count)?;
                let mut args = ops::PointwiseArgs::new(
                    ops::PointwiseSemantic::StagedGatedResidual,
                    &input,
                    &mut out,
                );
                args.secondary = Some(&residual);
                args.bias = Some(&gate);
                ops::pointwise(ctx, args)?;
            }
            Ok(())
        }
    }

    /// `next_k/v` receive the candidate observed-video cache. The caller decides
    /// whether to commit it; running a denoising step never mutates `history_k/v`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::cuda_dreamzero::model) fn block(
        ctx: &Context,
        p: &Policies,
        w: &Block,
        s: &mut BlockWorkspace,
        time: &Tensor,
        context: &CrossAttentionContext,
        rotary: &ops::ComplexRotaryTable,
        history_k: Option<&Tensor>,
        history_v: Option<&Tensor>,
        observed_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<()> {
        // Tune bulk transformer projections independently of the qualified
        // encoder and output-head reduction orders. Preparation owns tuning;
        // replay uses the retained executions and the same numerical contract.
        let p = &block_policies(p);
        if observed_cache.is_some_and(|(k, v)| {
            k.shape().dims() != [s.history_rows + s.video_rows, DIM] || v.shape() != k.shape()
        }) {
            return Err(Error::Other(
                "DreamZero block context/cache shape mismatch".into(),
            ));
        }
        match (history_k, history_v) {
            (None, None) if s.history_rows == 0 => (),
            (Some(k), Some(v))
                if k.shape().dims() == [s.history_rows, DIM] && v.shape() == k.shape() =>
            {
                transfers::copy_cuda_to_cuda(ctx, k, &rows(&s.kv_k, 0, s.history_rows)?)?;
                transfers::copy_cuda_to_cuda(ctx, v, &rows(&s.kv_v, 0, s.history_rows)?)?;
            }
            _ => {
                return Err(Error::Other(
                    "DreamZero history cache shape mismatch".into(),
                ))
            }
        }
        s.bind_modulation(ctx, &w.modulation, time)?;
        norm(ctx, &s.hidden, &s.unit_norm, &mut s.normed)?;
        s.modulate(ctx, 0, 1)?;
        let a = &w.self_attention;
        projection(ctx, p, &s.shifted, &a.q, &mut s.q_raw)?;
        // The history prefix was populated above. Project current tokens into
        // the final contiguous attention buffers, then normalize/rotate only K's
        // current-token view. No current K/V staging copy or scratch is needed.
        let mut current_k = rows(&s.kv_k, s.history_rows, s.rows)?;
        let mut current_v = rows(&s.kv_v, s.history_rows, s.rows)?;
        projection(ctx, p, &s.shifted, &a.k, &mut current_k)?;
        projection(ctx, p, &s.shifted, &a.v, &mut current_v)?;
        let shape = vec![s.rows, HEADS, DIM / HEADS];
        for (input, weight) in [(&s.q_raw, &a.norm_q), (&current_k, &a.norm_k)] {
            ops::rms_complex_rotary(
                ctx,
                rotary,
                ops::RmsComplexRotaryArgs {
                    input: &input.reshape(shape.clone())?,
                    weight,
                    output: &mut input.reshape(shape.clone())?,
                    eps: 1e-6,
                    numerics: ops::NormNumerics::Staged,
                },
            )?;
        }
        if let Some((next_k, next_v)) = observed_cache {
            let observed_rows = s.history_rows + s.video_rows;
            transfers::copy_cuda_to_cuda(ctx, &rows(&s.kv_k, 0, observed_rows)?, next_k)?;
            transfers::copy_cuda_to_cuda(ctx, &rows(&s.kv_v, 0, observed_rows)?, next_v)?;
        }
        attend(ctx, p, &s.q_raw, &s.kv_k, &s.kv_v, &mut s.attended)?;
        projection(ctx, p, &s.attended, &a.o, &mut s.projected)?;
        s.gate_residual(ctx, 2, &s.residual)?;

        norm(ctx, &s.residual, &w.norm3, &mut s.normed)?;
        let a = &w.cross_attention;
        projection(ctx, p, &s.normed, &a.q, &mut s.q_raw)?;
        rms(ctx, &s.q_raw, &a.norm_q, &mut s.q)?;
        attend(
            ctx,
            p,
            &s.q,
            &context.text_k,
            &context.text_v,
            &mut s.attended,
        )?;
        attend(
            ctx,
            p,
            &s.q,
            &context.image_k,
            &context.image_v,
            &mut s.image_attended,
        )?;
        binary(
            ctx,
            ops::PointwiseSemantic::Add,
            &s.attended,
            &s.image_attended,
            &mut s.shifted,
        )?;
        projection(ctx, p, &s.shifted, &a.o, &mut s.projected)?;
        binary(
            ctx,
            ops::PointwiseSemantic::Add,
            &s.residual,
            &s.projected,
            &mut s.hidden,
        )?;

        norm(ctx, &s.hidden, &s.unit_norm, &mut s.normed)?;
        s.modulate(ctx, 3, 4)?;
        projection(ctx, p, &s.shifted, &w.up, &mut s.up)?;
        let mut activation = ops::PointwiseArgs::new(
            ops::PointwiseSemantic::BiasActivation,
            &s.up,
            &mut s.activated,
        );
        activation.activation = ops::PointwiseActivation::Gelu;
        ops::pointwise(ctx, activation)?;
        projection(ctx, p, &s.activated, &w.down, &mut s.projected)?;
        s.gate_residual(ctx, 5, &s.hidden)
    }
}
mod vae {
    // Complete causal encoder; temporal downsampling preserves the first frame.
    // Activations are contiguous NTHWC for channel normalization and unfolded
    // GEMM; cuDNN convolutions use bounded temporary NCDHW input/output storage.
    use crate::cuda_dreamzero::backend;
    use crate::cuda_dreamzero::{
        backend::{ops, transfers, Context},
        config::Policies,
        model::blocks::{allocate, binary},
        weights::vae::{Conv, Down, Residual, VaeWeights},
    };
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

    pub(in crate::cuda_dreamzero::model) struct VaeWorkspace {
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
                convolution_workspace: ctx
                    .allocate_output(Shape::new(vec![256 << 20]), DType::I8)?,
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
            args.policy.numerics = ops::GemmNumerics::F32Reduction;
            ops::gemm(ctx, args)?;
            // cuDNN convolution rounds its output before adding the BF16 bias.
            binary(ctx, S::BroadcastAdd, &projected, &w.bias, &mut biased)?;
            transfers::copy_cuda_to_cuda(
                ctx,
                &biased,
                &view(out, start * w.output, &[n, w.output])?,
            )?;
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
            ops::QuantizationArgs::new(
                ops::QuantizationSemantic::CastBf16ToF32,
                gamma,
                &mut weight,
            ),
        )?;
        crate::cuda_dreamzero::model::blocks::staged_l2(ctx, &a, &mut b, 1e-12)?;
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
    fn normalized_silu(
        ctx: &Context,
        s: &VaeWorkspace,
        x: &Tensor,
        gamma: &Tensor,
    ) -> Result<Tensor> {
        if s.promoted_norm.is_some() {
            return promoted_normalization(ctx, s, x, gamma, true);
        }
        let input = matrix(x)?;
        let shape = input.shape().dims();
        let mut norm = s.slot(2, shape)?;
        let mut scaled = s.slot(3, shape)?;
        let mut weighted = s.slot(4, shape)?;
        crate::cuda_dreamzero::model::blocks::staged_l2(ctx, &input, &mut norm, 1e-12)?;
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

    pub(in crate::cuda_dreamzero::model) fn encode(
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
                        let target_shape =
                            [1, (shape[1] - 1) / 2 + 1, shape[2], shape[3], shape[4]];
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
            crate::cuda_dreamzero::model::blocks::staged_l2(ctx, &input, &mut norm, 1e-12)?;
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
}
mod vision {
    // Device vision stages from the policy's normalized RGB montage rows.
    use super::{
        clip::{self, ClipWorkspace},
        dit_block::rows,
        vae::{self, VaeWorkspace},
    };
    use crate::cuda_dreamzero::backend;
    use crate::cuda_dreamzero::{
        backend::{ops, transfers, Context},
        config::Policies,
        model::blocks::allocate,
        weights::{clip::ClipWeights, vae::VaeWeights},
    };
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

    pub(in crate::cuda_dreamzero::model) struct VisionWorkspace {
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
}
mod unipc {
    #[cfg(test)]
    use crate::cuda_dreamzero::backend;
    use crate::cuda_dreamzero::{
        backend::{ops, transfers, Context},
        model::blocks::{allocate, binary},
        schedule::{Schedule, Update},
    };
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
    mod golden {
        // Pinned DreamZero ab790c198fbce33503358efbbd4187ce9a89adf3, eager UniPC.
        // Torch 2.11.0+cu130, BF16 on NVIDIA Thor; synthetic fixed velocity field.
        pub const INITIAL: [f32; 16] = [
            -1.75, -0.875, 0.0, 0.875, 1.75, -1.0, -0.125, 0.75, 1.625, -1.125, -0.25, 0.625, 1.5,
            -1.25, -0.375, 0.5,
        ];
        pub const SIGMAS: [f32; 17] = [
            0.9997998476028442,
            0.9866341948509216,
            0.9720060229301453,
            0.9556572437286377,
            0.9372654557228088,
            0.9164220690727234,
            0.8926018476486206,
            0.865118145942688,
            0.8330553770065308,
            0.7951651215553284,
            0.7496998310089111,
            0.6941356062889099,
            0.6246873736381531,
            0.5354080200195312,
            0.4163888096809387,
            0.2497999668121338,
            0.0,
        ];
        pub const TIMES: [i64; 16] = [
            999, 986, 972, 955, 937, 916, 892, 865, 833, 795, 749, 694, 624, 535, 416, 249,
        ];
        pub const VELOCITIES: [[f32; 16]; 16] = [
            [
                -0.5625, -0.21875, 0.125, 0.46875, -0.34375, 0.0, 0.34375, -0.46875, -0.125,
                0.21875, 0.5625, -0.25, 0.09375, 0.4375, -0.375, -0.03125,
            ],
            [
                -0.46875, -0.125, 0.21875, 0.5625, -0.25, 0.09375, 0.4375, -0.375, -0.03125,
                0.3125, -0.5, -0.15625, 0.1875, 0.53125, -0.28125, 0.0625,
            ],
            [
                -0.375, -0.03125, 0.3125, -0.5, -0.15625, 0.1875, 0.53125, -0.28125, 0.0625,
                0.40625, -0.40625, -0.0625, 0.28125, -0.53125, -0.1875, 0.15625,
            ],
            [
                -0.28125, 0.0625, 0.40625, -0.40625, -0.0625, 0.28125, -0.53125, -0.1875, 0.15625,
                0.5, -0.3125, 0.03125, 0.375, -0.4375, -0.09375, 0.25,
            ],
            [
                -0.1875, 0.15625, 0.5, -0.3125, 0.03125, 0.375, -0.4375, -0.09375, 0.25, -0.5625,
                -0.21875, 0.125, 0.46875, -0.34375, 0.0, 0.34375,
            ],
            [
                -0.09375, 0.25, -0.5625, -0.21875, 0.125, 0.46875, -0.34375, 0.0, 0.34375,
                -0.46875, -0.125, 0.21875, 0.5625, -0.25, 0.09375, 0.4375,
            ],
            [
                0.0, 0.34375, -0.46875, -0.125, 0.21875, 0.5625, -0.25, 0.09375, 0.4375, -0.375,
                -0.03125, 0.3125, -0.5, -0.15625, 0.1875, 0.53125,
            ],
            [
                0.09375, 0.4375, -0.375, -0.03125, 0.3125, -0.5, -0.15625, 0.1875, 0.53125,
                -0.28125, 0.0625, 0.40625, -0.40625, -0.0625, 0.28125, -0.53125,
            ],
            [
                0.1875, 0.53125, -0.28125, 0.0625, 0.40625, -0.40625, -0.0625, 0.28125, -0.53125,
                -0.1875, 0.15625, 0.5, -0.3125, 0.03125, 0.375, -0.4375,
            ],
            [
                0.28125, -0.53125, -0.1875, 0.15625, 0.5, -0.3125, 0.03125, 0.375, -0.4375,
                -0.09375, 0.25, -0.5625, -0.21875, 0.125, 0.46875, -0.34375,
            ],
            [
                0.375, -0.4375, -0.09375, 0.25, -0.5625, -0.21875, 0.125, 0.46875, -0.34375, 0.0,
                0.34375, -0.46875, -0.125, 0.21875, 0.5625, -0.25,
            ],
            [
                0.46875, -0.34375, 0.0, 0.34375, -0.46875, -0.125, 0.21875, 0.5625, -0.25, 0.09375,
                0.4375, -0.375, -0.03125, 0.3125, -0.5, -0.15625,
            ],
            [
                0.5625, -0.25, 0.09375, 0.4375, -0.375, -0.03125, 0.3125, -0.5, -0.15625, 0.1875,
                0.53125, -0.28125, 0.0625, 0.40625, -0.40625, -0.0625,
            ],
            [
                -0.5, -0.15625, 0.1875, 0.53125, -0.28125, 0.0625, 0.40625, -0.40625, -0.0625,
                0.28125, -0.53125, -0.1875, 0.15625, 0.5, -0.3125, 0.03125,
            ],
            [
                -0.40625, -0.0625, 0.28125, -0.53125, -0.1875, 0.15625, 0.5, -0.3125, 0.03125,
                0.375, -0.4375, -0.09375, 0.25, -0.5625, -0.21875, 0.125,
            ],
            [
                -0.3125, 0.03125, 0.375, -0.4375, -0.09375, 0.25, -0.5625, -0.21875, 0.125,
                0.46875, -0.34375, 0.0, 0.34375, -0.46875, -0.125, 0.21875,
            ],
        ];
        pub const OUTPUTS: [[f32; 16]; 16] = [
            [
                -1.7421875,
                -0.87109375,
                -0.00164794921875,
                0.8671875,
                1.7578125,
                -1.0,
                -0.1298828125,
                0.7578125,
                1.6328125,
                -1.125,
                -0.2578125,
                0.62890625,
                1.5,
                -1.2578125,
                -0.37109375,
                0.5,
            ],
            [
                -1.734375,
                -0.87109375,
                -0.005584716796875,
                0.859375,
                1.7578125,
                -1.0,
                -0.1376953125,
                0.76171875,
                1.6328125,
                -1.1328125,
                -0.2412109375,
                0.62890625,
                1.4921875,
                -1.2578125,
                -0.3671875,
                0.498046875,
            ],
            [
                -1.734375,
                -0.875,
                -0.0118408203125,
                0.8828125,
                1.7578125,
                -1.0078125,
                -0.1474609375,
                0.765625,
                1.6328125,
                -1.140625,
                -0.236328125,
                0.62890625,
                1.4921875,
                -1.2421875,
                -0.365234375,
                0.49609375,
            ],
            [
                -1.734375,
                -0.87890625,
                -0.020263671875,
                0.88671875,
                1.765625,
                -1.015625,
                -0.12255859375,
                0.76953125,
                1.6328125,
                -1.1484375,
                -0.232421875,
                0.62890625,
                1.484375,
                -1.2421875,
                -0.36328125,
                0.4921875,
            ],
            [
                -1.7265625,
                -0.8828125,
                -0.03173828125,
                0.89453125,
                1.765625,
                -1.0234375,
                -0.12109375,
                0.76953125,
                1.625,
                -1.125,
                -0.228515625,
                0.625,
                1.4765625,
                -1.234375,
                -0.36328125,
                0.484375,
            ],
            [
                -1.71875,
                -0.88671875,
                0.0032958984375,
                0.8984375,
                1.765625,
                -1.03125,
                -0.1142578125,
                0.76953125,
                1.609375,
                -1.1171875,
                -0.2265625,
                0.62109375,
                1.4609375,
                -1.2265625,
                -0.369140625,
                0.46875,
            ],
            [
                -1.71875,
                -0.89453125,
                0.00543212890625,
                0.90234375,
                1.7578125,
                -1.046875,
                -0.10888671875,
                0.765625,
                1.6015625,
                -1.109375,
                -0.2265625,
                0.61328125,
                1.5,
                -1.21875,
                -0.376953125,
                0.451171875,
            ],
            [
                -1.7265625,
                -0.91015625,
                0.0152587890625,
                0.90625,
                1.75,
                -1.0,
                -0.10546875,
                0.76171875,
                1.5859375,
                -1.1015625,
                -0.2314453125,
                0.59765625,
                1.5,
                -1.2265625,
                -0.388671875,
                0.498046875,
            ],
            [
                -1.734375,
                -0.9296875,
                0.024169921875,
                0.90234375,
                1.734375,
                -1.0,
                -0.10498046875,
                0.75,
                1.640625,
                -1.09375,
                -0.2392578125,
                0.578125,
                1.5078125,
                -1.2265625,
                -0.404296875,
                0.498046875,
            ],
            [
                -1.7421875,
                -0.86328125,
                0.030517578125,
                0.88671875,
                1.7109375,
                -0.98828125,
                -0.1083984375,
                0.73046875,
                1.640625,
                -1.0859375,
                -0.251953125,
                0.640625,
                1.515625,
                -1.234375,
                -0.427734375,
                0.5078125,
            ],
            [
                -1.7578125,
                -0.85546875,
                0.03271484375,
                0.8671875,
                1.7890625,
                -0.98046875,
                -0.11865234375,
                0.69921875,
                1.65625,
                -1.0859375,
                -0.2734375,
                0.6484375,
                1.5234375,
                -1.2421875,
                -0.462890625,
                0.51953125,
            ],
            [
                -1.7890625,
                -0.83984375,
                0.0291748046875,
                0.8359375,
                1.796875,
                -0.9765625,
                -0.1376953125,
                0.65625,
                1.65625,
                -1.09375,
                -0.30859375,
                0.6640625,
                1.5234375,
                -1.265625,
                -0.361328125,
                0.52734375,
            ],
            [
                -1.84375,
                -0.8203125,
                0.015869140625,
                0.79296875,
                1.8203125,
                -0.9765625,
                -0.1689453125,
                0.78125,
                1.65625,
                -1.1171875,
                -0.361328125,
                0.6796875,
                1.5078125,
                -1.3125,
                -0.353515625,
                0.52734375,
            ],
            [
                -1.6796875,
                -0.80859375,
                -0.0140380859375,
                0.72265625,
                1.84375,
                -0.9921875,
                -0.224609375,
                0.80078125,
                1.65625,
                -1.15625,
                -0.18359375,
                0.6953125,
                1.4765625,
                -1.3828125,
                -0.330078125,
                0.515625,
            ],
            [
                -1.6328125,
                -0.8125,
                -0.07421875,
                0.984375,
                1.859375,
                -1.03125,
                -0.32421875,
                0.82421875,
                1.640625,
                -1.234375,
                -0.1240234375,
                0.6953125,
                1.421875,
                -1.125,
                -0.310546875,
                0.48046875,
            ],
            [
                -1.5546875,
                -0.8203125,
                -0.16796875,
                1.09375,
                1.8828125,
                -1.09375,
                -0.18359375,
                0.87890625,
                1.609375,
                -1.3515625,
                -0.0380859375,
                0.6953125,
                1.3359375,
                -1.0078125,
                -0.279296875,
                0.42578125,
            ],
        ];
    }
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
                let mut sampler =
                    UniPc::new(&ctx, [1, count], Schedule::standard().unwrap()).unwrap();
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
            transfers::copy_cpu_to_cuda(&host(&golden::INITIAL.map(|v| -v)), &sampler.sample)
                .unwrap();
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
    pub(in crate::cuda_dreamzero::model) struct UniPc {
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
}
mod sample {
    // Full joint denoising: both modalities traverse the same DiT on every step.
    use super::{
        dit::{DitContext, DitWorkspace, VideoCache},
        dit_block::rows,
        unipc::UniPc,
    };
    use crate::cuda_dreamzero::{
        backend::{ops, transfers, Context},
        config::Policies,
        model::blocks::{allocate, binary},
        schedule::Schedule,
        weights::dit::DitWeights,
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

    pub(in crate::cuda_dreamzero::model) struct JointWorkspace {
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
        pub fn initialize(&mut self, ctx: &Context, conditioning: &Tensor) -> Result<()> {
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
            Ok(())
        }
        pub fn step(
            &mut self,
            ctx: &Context,
            p: &Policies,
            w: &DitWeights,
            positive: &DitContext,
            negative: &DitContext,
            history_positive: &VideoCache,
            history_negative: &VideoCache,
            i: usize,
        ) -> Result<()> {
            let t = &self.time[i];
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
                for (kind, tensor) in [("modulation", &input.modulation), ("action", &input.action)]
                {
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
}
use super::super::Profile;
// BF16 stage composition and storage.
use self::{
    dit::{DitContext, DitWorkspace, VideoCache},
    dit_block::rows,
    sample::JointWorkspace,
    text::TextWorkspace,
    vision::VisionWorkspace,
};
use crate::cuda_dreamzero::{schedule::Schedule, weights::Bf16Weights};
use apxinf_core::Error;
use std::rc::Rc;

#[derive(Clone)]
pub(in crate::cuda_dreamzero) struct History {
    positive: VideoCache,
    negative: VideoCache,
    clip: Tensor,
    conditioning: Tensor,
}
pub(in crate::cuda_dreamzero::model) struct Workspace {
    pub profile: Profile,
    pub ids: [Tensor; 2],
    pub masks: [Tensor; 2],
    pub valid_rows: [Tensor; 2],
    pub vision: VisionWorkspace,
    pub joint: JointWorkspace,
    text: TextWorkspace,
    masked_text: Tensor,
    contexts: [DitContext; 2],
    clip: Tensor,
    conditioning: Tensor,
    observed: DitWorkspace,
    prior: Option<[VideoCache; 2]>,
    candidate: [VideoCache; 2],
    zero_time: Tensor,
}
pub(in crate::cuda_dreamzero::model) struct Bf16Blocks {
    pub ctx: Rc<Context>,
    weights: Bf16Weights,
    policies: Policies,
}
impl Bf16Blocks {
    pub fn new(ctx: Rc<Context>, weights: Bf16Weights, policies: Policies) -> Self {
        Self {
            ctx,
            weights,
            policies,
        }
    }
    fn execution_policies(&self, tune: bool) -> Policies {
        let mut p = self.policies.clone();
        p.gemm.online_tune &= tune;
        p.attention.online_tune &= tune;
        p
    }
    /// Every activation has explicit model-owned storage; prepared native
    /// GEMM/attention executions own their opaque provider workspaces. The
    /// session arena is unused but the backend requires nonzero capacity.
    pub fn phase_scratch_bytes(&self) -> [usize; 3] {
        [256; 3]
    }
    pub fn workspace(&self, profile: Profile) -> Result<Workspace> {
        profile.validate()?;
        let ctx = &self.ctx;
        let previous = profile.previous_rows();
        let observed_frames = if profile.initial() { 1 } else { 2 };
        let observed_start = if profile.initial() {
            0
        } else {
            profile.start_frame - 2
        };
        let mut zero_time = vec![half::bf16::ZERO; 512];
        zero_time[..128].fill(half::bf16::ONE);
        zero_time[256..384].fill(half::bf16::ONE);
        let ids = || ctx.allocate_output(Shape::new(vec![512]), DType::I32);
        Ok(Workspace {
            profile,
            ids: [ids()?, ids()?],
            masks: [allocate(ctx, &[1, 512])?, allocate(ctx, &[1, 512])?],
            valid_rows: [allocate(ctx, &[512, 1])?, allocate(ctx, &[512, 1])?],
            vision: VisionWorkspace::new(ctx, profile.input_frames, profile.initial())?,
            joint: JointWorkspace::new(
                ctx,
                profile.start_frame,
                profile.start_frame * 880,
                Schedule::standard()?,
                Schedule::standard()?,
            )?,
            text: TextWorkspace::new(ctx)?,
            masked_text: allocate(ctx, &[512, 4096])?,
            contexts: [DitContext::new(ctx)?, DitContext::new(ctx)?],
            clip: allocate(ctx, &[257, 1280])?,
            conditioning: allocate(ctx, &[20, 9, 44, 80])?,
            observed: DitWorkspace::new(ctx, observed_frames, observed_start, previous, false)?,
            prior: if previous == 0 {
                None
            } else {
                Some([
                    VideoCache::new(ctx, previous)?,
                    VideoCache::new(ctx, previous)?,
                ])
            },
            candidate: [
                VideoCache::new(ctx, profile.start_frame * 880)?,
                VideoCache::new(ctx, profile.start_frame * 880)?,
            ],
            zero_time: transfers::to_cuda(
                &Tensor::from_bf16(vec![2, 256], &zero_time)?,
                ctx.device_id(),
            )?,
        })
    }
    /// Bind an immutable episode snapshot into stable graph input storage.
    pub fn bind_history(&self, w: &Workspace, history: Option<&History>) -> Result<()> {
        match (&w.prior, history) {
            (None, None) => Ok(()),
            (Some(prior), Some(history)) => {
                copy_cache(&self.ctx, &history.positive, &prior[0])?;
                copy_cache(&self.ctx, &history.negative, &prior[1])?;
                transfers::copy_cuda_to_cuda(&self.ctx, &history.clip, &w.clip)?;
                transfers::copy_cuda_to_cuda(&self.ctx, &history.conditioning, &w.conditioning)
            }
            _ => Err(Error::Other(
                "DreamZero history does not match workspace profile".into(),
            )),
        }
    }
    pub fn vision(&self, w: &mut Workspace, tune: bool) -> Result<()> {
        let p = self.execution_policies(tune);
        w.vision
            .forward(&self.ctx, &p, &self.weights.clip, &self.weights.vae)?;
        if w.profile.initial() {
            transfers::copy_cuda_to_cuda(&self.ctx, &w.vision.clip.output, &w.clip)?;
            transfers::copy_cuda_to_cuda(&self.ctx, &w.vision.conditioning, &w.conditioning)?;
        }
        Ok(())
    }
    pub fn language(&self, w: &mut Workspace, tune: bool) -> Result<()> {
        let p = self.execution_policies(tune);
        for i in 0..2 {
            transfers::copy_cuda_to_cuda(&self.ctx, &w.ids[i], &w.text.ids)?;
            transfers::copy_cuda_to_cuda(&self.ctx, &w.masks[i], &w.text.mask)?;
            text::encode(&self.ctx, &p, &self.weights.text, &mut w.text)?;
            // Official encode_prompt zeros padded output rows before the DiT MLP.
            binary(
                &self.ctx,
                ops::PointwiseSemantic::RowMultiply,
                &w.text.output,
                &w.valid_rows[i],
                &mut w.masked_text,
            )?;
            w.contexts[i].encode(&self.ctx, &p, &self.weights.dit, &w.masked_text, &w.clip)?;
        }
        Ok(())
    }
    pub fn action(&self, w: &mut Workspace, tune: bool) -> Result<()> {
        let p = self.execution_policies(tune);
        transfers::copy_cuda_to_cuda(&self.ctx, &w.vision.observed, &w.observed.video)?;
        transfers::copy_cuda_to_cuda(&self.ctx, &w.zero_time, &w.observed.time_sinusoidal)?;
        let count = if w.profile.initial() { 1 } else { 2 };
        let start = if w.profile.initial() {
            0
        } else {
            w.profile.start_frame - 2
        };
        let source = w.conditioning.reshape(vec![20 * 9, 44 * 80])?;
        let target = w.observed.conditioning.reshape(vec![20 * count, 44 * 80])?;
        for c in 0..20 {
            transfers::copy_cuda_to_cuda(
                &self.ctx,
                &rows(&source, c * 9 + start, count)?,
                &rows(&target, c * count, count)?,
            )?;
        }
        for i in 0..2 {
            w.observed.forward(
                &self.ctx,
                &p,
                &self.weights.dit,
                &w.contexts[i],
                w.prior.as_ref().map(|p| &p[i]),
                Some(&w.candidate[i]),
            )?;
        }
        w.joint.initialize(&self.ctx, &w.conditioning)
    }
    pub fn denoise_step(&self, w: &mut Workspace, step: usize, tune: bool) -> Result<()> {
        let p = self.execution_policies(tune);
        w.joint.step(
            &self.ctx,
            &p,
            &self.weights.dit,
            &w.contexts[0],
            &w.contexts[1],
            &w.candidate[0],
            &w.candidate[1],
            step,
        )
    }

    /// Independent storage prevents a later warmup or failed inference from
    /// overwriting committed history through a prepared workspace alias.
    pub fn snapshot(&self, w: &Workspace) -> Result<History> {
        let h = History {
            positive: VideoCache::new(&self.ctx, w.candidate[0].rows)?,
            negative: VideoCache::new(&self.ctx, w.candidate[1].rows)?,
            clip: allocate(&self.ctx, &[257, 1280])?,
            conditioning: allocate(&self.ctx, &[20, 9, 44, 80])?,
        };
        copy_cache(&self.ctx, &w.candidate[0], &h.positive)?;
        copy_cache(&self.ctx, &w.candidate[1], &h.negative)?;
        transfers::copy_cuda_to_cuda(&self.ctx, &w.clip, &h.clip)?;
        transfers::copy_cuda_to_cuda(&self.ctx, &w.conditioning, &h.conditioning)?;
        Ok(h)
    }
}
fn copy_cache(ctx: &Context, source: &VideoCache, target: &VideoCache) -> Result<()> {
    if source.rows != target.rows {
        return Err(Error::Other(
            "DreamZero committed history has the wrong length".into(),
        ));
    }
    for ((sk, sv), (tk, tv)) in source
        .keys
        .iter()
        .zip(&source.values)
        .zip(target.keys.iter().zip(&target.values))
    {
        transfers::copy_cuda_to_cuda(ctx, sk, tk)?;
        transfers::copy_cuda_to_cuda(ctx, sv, tv)?;
    }
    Ok(())
}

#[cfg(test)]
mod acceptance {
    // Opt-in checkpoint regressions against externally captured official tensors.
    // Fixture files and run logs live outside the maintained source tree.
    use super::{clip, vae};
    use crate::cuda_dreamzero::backend;
    use crate::cuda_dreamzero::{
        backend::{transfers, Context},
        config::Policies,
        weights,
    };
    use apxinf_core::Tensor;
    use std::{collections::HashMap, path::PathBuf};

    fn inputs() -> (HashMap<String, Tensor>, HashMap<String, Tensor>) {
        let shard = PathBuf::from(
            std::env::var("APXINF_DREAMZERO_COMPONENT_SHARD")
                .expect("set the verified official shard containing CLIP/VAE"),
        );
        let fixture = PathBuf::from(
            std::env::var("APXINF_DREAMZERO_COMPONENT_FIXTURE")
                .expect("set the pinned official component fixture safetensors"),
        );
        (
            apxinf_loader::safetensors::load_native(&shard).unwrap().0,
            apxinf_loader::safetensors::load_native(&fixture).unwrap().0,
        )
    }
    fn host(ctx: &Context, x: &Tensor) -> Vec<f32> {
        ctx.synchronize().unwrap();
        transfers::to_cpu(x).unwrap().to_f32_vec().unwrap()
    }
    fn compare(name: &str, actual: &[f32], expected: &Tensor) {
        compare_with_limit(name, actual, expected, 0.01)
    }
    fn compare_with_limit(name: &str, actual: &[f32], expected: &Tensor, limit: f64) {
        let expected = expected.to_f32_vec().unwrap();
        assert_eq!(actual.len(), expected.len(), "{name}");
        assert!(actual.iter().all(|x| x.is_finite()), "{name} nonfinite");
        let mut squared = 0f64;
        let mut energy = 0f64;
        let mut maximum = 0f64;
        for (&a, &b) in actual.iter().zip(&expected) {
            let d = (a as f64 - b as f64).abs();
            squared += d * d;
            energy += (b as f64) * (b as f64);
            maximum = maximum.max(d);
        }
        let relative = (squared / energy.max(1e-30)).sqrt();
        eprintln!(
            "{}",
            serde_json::json!({"checkpoint":name,"relative_l2":relative,
        "rmse":(squared/actual.len() as f64).sqrt(),"max_abs":maximum})
        );
        assert!(
            relative <= limit,
            "{name} relative L2 {relative} exceeds {limit} tolerance"
        );
    }

    #[test]
    #[ignore = "requires official DROID shard and independent CLIP captures"]
    fn checkpoint_clip_eager_capture_changed_input_and_lifetime() {
        let (mut map, fixture) = inputs();
        map.retain(|key, _| key.starts_with("action_head.image_encoder."));
        let ctx =
            Context::new_with_autotune(0, std::env::var_os("APXINF_DREAMZERO_AUTOTUNE").is_some())
                .unwrap();
        let weights = weights::clip::ClipWeights::from_map(&ctx, &mut map).unwrap();
        assert!(map.is_empty());
        let mut w = clip::ClipWorkspace::new(&ctx).unwrap();
        let p = Policies::default();
        transfers::copy_cpu_to_cuda(&fixture["clip.input"], &w.patches).unwrap();
        let session = backend::ExecutionSession::with_capacity(256, 0).unwrap();
        session
            .prepare(|| clip::encode_patches(&ctx, &p, &weights, &mut w))
            .unwrap();
        let first = host(&ctx, &w.output);
        let graph = session
            .run(|| backend::capture(&ctx, || clip::encode_patches(&ctx, &p, &weights, &mut w)))
            .unwrap();
        for _ in 0..3 {
            graph.replay().unwrap();
            assert_eq!(host(&ctx, &w.output), first);
        }
        transfers::copy_cpu_to_cuda(&fixture["clip.changed.input"], &w.patches).unwrap();
        session
            .run(|| clip::encode_patches(&ctx, &p, &weights, &mut w))
            .unwrap();
        let changed = host(&ctx, &w.output);
        assert_ne!(changed, first);
        let output = w.output.clone();
        drop(w);
        drop(weights);
        drop(session);
        graph.replay().unwrap();
        assert_eq!(host(&ctx, &output), changed);
        eprintln!("CLIP eager/captured, changed-input and retained-resource replay passed");
        compare_with_limit(
            "clip.changed",
            &changed,
            &fixture["clip.changed.output"],
            0.01,
        );
        compare_with_limit("clip", &first, &fixture["clip.output"], 0.01);
    }

    #[test]
    #[ignore = "requires official DROID shard and independent temporal VAE captures"]
    fn checkpoint_vae_temporal_profiles_eager_capture_and_lifetime() {
        let (mut map, fixture) = inputs();
        map.retain(|key, _| key.starts_with("action_head.vae."));
        let ctx =
            Context::new_with_autotune(0, std::env::var_os("APXINF_DREAMZERO_AUTOTUNE").is_some())
                .unwrap();
        let weights = weights::vae::VaeWeights::from_map(&ctx, &mut map).unwrap();
        drop(map); // decoder is validated by the complete checkpoint loader.
        let p = Policies::default();
        for frames in [1, 9, 33] {
            let name = format!("vae{frames}");
            let w = vae::VaeWorkspace::new(&ctx, frames, 352, 640, frames == 33).unwrap();
            transfers::copy_cpu_to_cuda(&fixture[&format!("{name}.input")], &w.input).unwrap();
            let session = backend::ExecutionSession::with_capacity(256, 0).unwrap();
            session
                .prepare(|| vae::encode(&ctx, &p, &weights, &w))
                .unwrap();
            let first = host(&ctx, &w.output);
            compare(&name, &first, &fixture[&format!("{name}.output")]);
            let graph = session
                .run(|| backend::capture(&ctx, || vae::encode(&ctx, &p, &weights, &w)))
                .unwrap();
            graph.replay().unwrap();
            assert_eq!(host(&ctx, &w.output), first, "{name} captured parity");
            transfers::copy_cpu_to_cuda(&fixture[&format!("{name}.changed.input")], &w.input)
                .unwrap();
            session.run(|| vae::encode(&ctx, &p, &weights, &w)).unwrap();
            let changed = host(&ctx, &w.output);
            compare(
                &format!("{name}.changed"),
                &changed,
                &fixture[&format!("{name}.changed.output")],
            );
            assert_ne!(first, changed);
            let output = w.output.clone();
            drop(w);
            drop(session);
            graph.replay().unwrap();
            assert_eq!(
                host(&ctx, &output),
                changed,
                "{name} retained graph storage"
            );
        }
    }

    #[test]
    #[ignore = "requires official CLIP intermediate captures"]
    fn checkpoint_clip_layer_diagnostics() {
        let (mut map, fixture) = inputs();
        let path = std::env::var("APXINF_DREAMZERO_CLIP_LAYERS").unwrap();
        let layers = apxinf_loader::safetensors::load_native(std::path::Path::new(&path))
            .unwrap()
            .0;
        map.retain(|key, _| key.starts_with("action_head.image_encoder."));
        let ctx =
            Context::new_with_autotune(0, std::env::var_os("APXINF_DREAMZERO_AUTOTUNE").is_some())
                .unwrap();
        let weights = weights::clip::ClipWeights::from_map(&ctx, &mut map).unwrap();
        let mut w = clip::ClipWorkspace::new(&ctx).unwrap();
        let p = Policies::default();
        transfers::copy_cpu_to_cuda(&fixture["clip.input"], &w.patches).unwrap();
        let session = backend::ExecutionSession::with_capacity(256, 0).unwrap();
        session
            .prepare(|| clip::encode_prefix(&ctx, &p, &weights, &mut w))
            .unwrap();
        compare("clip.pre_norm", &host(&ctx, &w.hidden), &layers["pre_norm"]);
        for (i, b) in weights.blocks.iter().enumerate() {
            let input = if i == 0 {
                &layers["pre_norm"]
            } else {
                &layers[&format!("transformer.{}", i - 1)]
            };
            transfers::copy_cpu_to_cuda(&input.reshape(vec![257, 1280]).unwrap(), &w.hidden)
                .unwrap();
            session
                .prepare(|| clip::encode_block(&ctx, &p, b, &mut w))
                .unwrap();
            if i == 0 {
                compare(
                    "clip.block0.up",
                    &host(&ctx, &w.up),
                    &layers["transformer.0.mlp.0"],
                );
                compare(
                    "clip.block0.activation",
                    &host(&ctx, &w.activated),
                    &layers["transformer.0.mlp.1"],
                );
            }
            compare_with_limit(
                &format!("clip.block{i}"),
                &host(&ctx, &w.hidden),
                &layers[&format!("transformer.{i}")],
                0.005,
            );
        }
    }

    #[test]
    #[ignore = "requires official DROID text shards and independently captured prompts"]
    fn checkpoint_text_both_prompts_eager_capture() {
        let root = PathBuf::from(std::env::var("APXINF_DREAMZERO_CHECKPOINT").unwrap());
        let path = PathBuf::from(std::env::var("APXINF_DREAMZERO_TEXT_FIXTURE").unwrap());
        let fixture = apxinf_loader::safetensors::load_native(&path).unwrap().0;
        let mut map = HashMap::new();
        for i in 1..=3 {
            let mut shard = apxinf_loader::safetensors::load_native(
                &root.join(format!("model-{i:05}-of-00010.safetensors")),
            )
            .unwrap()
            .0;
            shard.retain(|key, _| key.starts_with("action_head.text_encoder."));
            map.extend(shard);
        }
        let ctx =
            Context::new_with_autotune(0, std::env::var_os("APXINF_DREAMZERO_AUTOTUNE").is_some())
                .unwrap();
        let weights = weights::TextWeights::from_map(&ctx, &mut map).unwrap();
        assert!(map.is_empty());
        let mut w = super::text::TextWorkspace::new(&ctx).unwrap();
        let p = Policies::default();
        let session = backend::ExecutionSession::with_capacity(256, 0).unwrap();
        let bind = |name: &str, w: &super::text::TextWorkspace| {
            let ids = fixture[&format!("{name}.ids")]
                .to_f32_vec()
                .unwrap()
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>();
            let mask = fixture[&format!("{name}.mask")]
                .to_f32_vec()
                .unwrap()
                .iter()
                .map(|&x| {
                    if x == 1. {
                        half::bf16::ZERO
                    } else {
                        half::bf16::MIN
                    }
                })
                .collect::<Vec<_>>();
            transfers::copy_cpu_to_cuda(&Tensor::from_i32(vec![512], &ids).unwrap(), &w.ids)
                .unwrap();
            transfers::copy_cpu_to_cuda(&Tensor::from_bf16(vec![1, 512], &mask).unwrap(), &w.mask)
                .unwrap();
        };
        bind("positive", &w);
        session
            .prepare(|| super::text::encode(&ctx, &p, &weights, &mut w))
            .unwrap();
        let positive = host(&ctx, &w.output);
        for name in ["positive", "negative"] {
            bind(name, &w);
            crate::cuda_dreamzero::backend::ops::embedding_gather(
                &ctx,
                &weights.embedding,
                &w.ids,
                &w.hidden,
            )
            .unwrap();
            for (i, block) in weights.blocks.iter().enumerate() {
                if i > 0 {
                    transfers::copy_cpu_to_cuda(
                        &fixture[&format!("{name}.block{}", i - 1)]
                            .reshape(vec![512, 4096])
                            .unwrap(),
                        &w.hidden,
                    )
                    .unwrap();
                }
                session
                    .prepare(|| super::text::encode_block(&ctx, &p, block, &mut w))
                    .unwrap();
                compare_with_limit(
                    &format!("text.{name}.block{i}.isolated"),
                    &host(&ctx, &w.hidden),
                    &fixture[&format!("{name}.block{i}")],
                    0.005,
                );
            }
        }
        bind("positive", &w);
        session
            .prepare(|| super::text::encode(&ctx, &p, &weights, &mut w))
            .unwrap();
        let graph = session
            .run(|| backend::capture(&ctx, || super::text::encode(&ctx, &p, &weights, &mut w)))
            .unwrap();
        graph.replay().unwrap();
        assert_eq!(positive, host(&ctx, &w.output));
        bind("negative", &w);
        session
            .run(|| super::text::encode(&ctx, &p, &weights, &mut w))
            .unwrap();
        let negative = host(&ctx, &w.output);
        let output = w.output.clone();
        drop(w);
        drop(weights);
        drop(session);
        graph.replay().unwrap();
        assert_eq!(negative, host(&ctx, &output));
        eprintln!("UMT5 eager/captured, changed-prompt and retained-resource replay passed");
        compare_with_limit(
            "text.negative",
            &negative,
            &fixture["negative.output"],
            0.01,
        );
        compare_with_limit(
            "text.positive",
            &positive,
            &fixture["positive.output"],
            0.01,
        );
        for (name, actual) in [("positive", &positive), ("negative", &negative)] {
            let count = fixture[&format!("{name}.mask")]
                .to_f32_vec()
                .unwrap()
                .iter()
                .filter(|&&x| x == 1.)
                .count();
            let expected = fixture[&format!("{name}.output")].to_f32_vec().unwrap();
            compare_with_limit(
                &format!("text.{name}.valid"),
                &actual[..count * 4096],
                &Tensor::from_f32(vec![count, 4096], &expected[..count * 4096]).unwrap(),
                0.01,
            );
        }
    }

    #[test]
    #[ignore = "requires official observed/joint DiT block captures and complete checkpoint"]
    fn checkpoint_dit_observed_and_joint_blocks_eager_capture() {
        use super::dit_block::{block, BlockWorkspace, CrossAttentionContext};
        let root = PathBuf::from(std::env::var("APXINF_DREAMZERO_CHECKPOINT").unwrap());
        let fixtures = PathBuf::from(std::env::var("APXINF_DREAMZERO_BLOCK_FIXTURES").unwrap());
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("model.safetensors.index.json")).unwrap(),
        )
        .unwrap();
        let ctx =
            Context::new_with_autotune(0, std::env::var_os("APXINF_DREAMZERO_AUTOTUNE").is_some())
                .unwrap();
        for layer in [0, 19, 39] {
            let prefix = format!("action_head.model.blocks.{layer}.");
            let shards: std::collections::BTreeSet<_> = index["weight_map"]
                .as_object()
                .unwrap()
                .iter()
                .filter(|(k, _)| k.starts_with(&prefix))
                .map(|(_, v)| v.as_str().unwrap())
                .collect();
            let mut map = HashMap::new();
            for shard in shards {
                let mut data = apxinf_loader::safetensors::load_native(&root.join(shard))
                    .unwrap()
                    .0;
                data.retain(|k, _| k.starts_with(&prefix));
                map.extend(data);
            }
            let weights = weights::dit::Block::from_map(&ctx, &mut map, layer).unwrap();
            assert!(map.is_empty());
            for call in 0..4 {
                let name = format!("dit.block{layer}.{call}");
                let input = apxinf_loader::safetensors::load_native(
                    &fixtures.join(format!("{name}.safetensors")),
                )
                .unwrap()
                .0;
                let joint = call >= 2;
                let n = if joint { 1760 } else { 880 };
                let h = if joint { 880 } else { 0 };
                let mut work = BlockWorkspace::new(&ctx, n, if joint { 25 } else { 0 }, h).unwrap();
                transfers::copy_cpu_to_cuda(&input["hidden"], &work.hidden).unwrap();
                let time = transfers::to_cuda(&input["time"], 0).unwrap();
                let text = transfers::to_cuda(&input["text"], 0).unwrap();
                let image = transfers::to_cuda(&input["image"], 0).unwrap();
                let history = if joint {
                    Some((
                        transfers::to_cuda(&input["k"], 0).unwrap(),
                        transfers::to_cuda(&input["v"], 0).unwrap(),
                    ))
                } else {
                    None
                };
                let next = if joint {
                    None
                } else {
                    Some((
                        crate::cuda_dreamzero::model::blocks::allocate(&ctx, &[880, 5120]).unwrap(),
                        crate::cuda_dreamzero::model::blocks::allocate(&ctx, &[880, 5120]).unwrap(),
                    ))
                };
                let bytes = std::fs::read(fixtures.join(format!("{name}.rotary.f64"))).unwrap();
                assert_eq!(bytes.len() % 8, 0);
                let coefficients: Vec<f64> = bytes
                    .chunks_exact(8)
                    .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let generated = super::dit::rotary_coefficients(
                    if joint { 2 } else { 1 },
                    if joint { 1 } else { 0 },
                    joint,
                );
                assert_eq!(generated.len(), coefficients.len());
                for (actual, expected) in generated.iter().zip(&coefficients) {
                    assert!(
                        (actual - expected).abs() <= 1e-12,
                        "native rotary coefficient mismatch"
                    );
                }
                let rotary = crate::cuda_dreamzero::backend::ops::ComplexRotaryTable::new(
                    &ctx,
                    n + if joint { 25 } else { 0 },
                    128,
                    &generated,
                )
                .unwrap();
                let policies = Policies::default();
                let mut cross = CrossAttentionContext::new(&ctx).unwrap();
                let mut text_k_raw = super::allocate(&ctx, &[512, 5120]).unwrap();
                let mut image_k_raw = super::allocate(&ctx, &[257, 5120]).unwrap();
                cross
                    .encode(
                        &ctx,
                        &policies,
                        &weights,
                        &text,
                        &image,
                        &mut text_k_raw,
                        &mut image_k_raw,
                    )
                    .unwrap();
                let invoke = |work: &mut BlockWorkspace| {
                    block(
                        &ctx,
                        &policies,
                        &weights,
                        work,
                        &time,
                        &cross,
                        &rotary,
                        history.as_ref().map(|x| &x.0),
                        history.as_ref().map(|x| &x.1),
                        next.as_ref().map(|x| (&x.0, &x.1)),
                    )
                };
                let session = backend::ExecutionSession::with_capacity(256, 0).unwrap();
                session.prepare(|| invoke(&mut work)).unwrap();
                let actual = host(&ctx, &work.hidden);
                compare_with_limit(&name, &actual, &input["output"], 0.005);
                if let Some((k, v)) = &next {
                    let expected = apxinf_loader::safetensors::load_native(
                        &fixtures.join(format!("dit.block{layer}.{}.safetensors", call + 2)),
                    )
                    .unwrap()
                    .0;
                    compare_with_limit(
                        &format!("{name}.committed-k"),
                        &host(&ctx, k),
                        &expected["k"],
                        0.005,
                    );
                    compare_with_limit(
                        &format!("{name}.committed-v"),
                        &host(&ctx, v),
                        &expected["v"],
                        0.005,
                    );
                }
                // Block residual writes in place, so rebind its original input.
                transfers::copy_cpu_to_cuda(&input["hidden"], &work.hidden).unwrap();
                let graph = session
                    .run(|| backend::capture(&ctx, || invoke(&mut work)))
                    .unwrap();
                graph.replay().unwrap();
                assert_eq!(actual, host(&ctx, &work.hidden));
                transfers::copy_cpu_to_cuda(&input["hidden"], &work.hidden).unwrap();
                let output = work.hidden.clone();
                drop(work);
                drop(session);
                drop(rotary);
                drop(history);
                drop(next);
                drop(time);
                drop(text);
                drop(image);
                graph.replay().unwrap();
                assert_eq!(actual, host(&ctx, &output));
            }
        }
    }

    #[test]
    #[ignore = "requires official DROID component fixture including initial 33-frame conditioning"]
    fn checkpoint_initial_vision_boundary_and_graph() {
        let (mut map, fixture) = inputs();
        map.retain(|k, _| {
            k.starts_with("action_head.image_encoder.") || k.starts_with("action_head.vae.")
        });
        let ctx =
            Context::new_with_autotune(0, std::env::var_os("APXINF_DREAMZERO_AUTOTUNE").is_some())
                .unwrap();
        let clip = weights::clip::ClipWeights::from_map(&ctx, &mut map).unwrap();
        let vae = weights::vae::VaeWeights::from_map(&ctx, &mut map).unwrap();
        drop(map);
        let mut work = super::vision::VisionWorkspace::new(&ctx, 1, true).unwrap();
        let p = Policies::default();
        let bind = |name: &str, work: &super::vision::VisionWorkspace| {
            let data = fixture[name].as_bf16().unwrap();
            let first = Tensor::from_bf16(vec![352 * 640, 3], &data[..352 * 640 * 3]).unwrap();
            transfers::copy_cpu_to_cuda(&first, &work.images).unwrap();
        };
        bind("vae33.input", &work);
        let session = backend::ExecutionSession::with_capacity(256, 0).unwrap();
        session
            .prepare(|| work.forward(&ctx, &p, &clip, &vae))
            .unwrap();
        assert_eq!(
            host(&ctx, &work.clip.patches),
            fixture["clip.input"].to_f32_vec().unwrap(),
            "checkpoint-specific CLIP preprocessing"
        );
        compare_with_limit(
            "vision.clip",
            &host(&ctx, &work.clip.output),
            &fixture["clip.output"],
            0.01,
        );
        compare(
            "vision.vae33",
            &host(&ctx, &work.vae.output),
            &fixture["vae33.output"],
        );
        let observed = host(&ctx, &work.observed);
        let condition = host(&ctx, &work.conditioning);
        let graph = session
            .run(|| backend::capture(&ctx, || work.forward(&ctx, &p, &clip, &vae)))
            .unwrap();
        graph.replay().unwrap();
        assert_eq!(observed, host(&ctx, &work.observed));
        assert_eq!(condition, host(&ctx, &work.conditioning));
        bind("vae33.changed.input", &work);
        session.run(|| work.forward(&ctx, &p, &clip, &vae)).unwrap();
        assert_eq!(
            host(&ctx, &work.clip.patches),
            fixture["clip.changed.input"].to_f32_vec().unwrap()
        );
        compare_with_limit(
            "vision.changed.clip",
            &host(&ctx, &work.clip.output),
            &fixture["clip.changed.output"],
            0.01,
        );
        compare(
            "vision.changed.vae33",
            &host(&ctx, &work.vae.output),
            &fixture["vae33.changed.output"],
        );
        let changed = host(&ctx, &work.conditioning);
        assert_ne!(changed, condition);
        let output = work.conditioning.clone();
        drop(work);
        drop(session);
        drop(clip);
        drop(vae);
        graph.replay().unwrap();
        assert_eq!(changed, host(&ctx, &output));
    }
}

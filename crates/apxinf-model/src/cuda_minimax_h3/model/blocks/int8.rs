//! INT8 ConvRot DiT of the video-and-audio model: per-step modulation tables for three modalities, the 50 transformer blocks over the
//! packed `text | audio | video` sequence and the two final heads.
use super::refiner::attend;
use super::util::{alloc, byte_view, dense_linear, int8_linear, rows};
use crate::cuda_minimax_h3::backend::ops::{self, GemmArgs, GemmPolicy, NormArgs};
use crate::cuda_minimax_h3::backend::{AttentionPolicy, CudaBuffer, CudaContext};
use crate::cuda_minimax_h3::config::*;
use crate::cuda_minimax_h3::model::host;
use crate::cuda_minimax_h3::model::layout::{Kind, Layout};
use crate::cuda_minimax_h3::weights::{DitBlock, DitWeights};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use std::rc::Rc;

/// What one denoising step needs from the schedule: the normalised times the video and audio streams are at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::cuda_minimax_h3) struct StepTimes {
    /// `sigma` of the video stream; the audio stream's follows from the two flow shifts.
    pub sigma_video: f32,
}

impl StepTimes {
    pub fn sigma_audio(&self) -> f32 {
        host::time_shift_sigma(
            self.sigma_video.max(1e-6),
            SIGMA_SHIFT_VIDEO,
            SIGMA_SHIFT_AUDIO,
        )
    }
}

/// Everything of one denoising step that depends on its timestep and nothing else, uploaded once per request.
struct StepTables {
    /// BF16 `[BLOCKS, MODALITIES, 6, HIDDEN]`: per block the modulation rows of visual (0), text (1) and audio (2) tokens, each
    /// evaluated at the timestep of the stream it belongs to.
    modulation: Tensor,
    /// Folded heads of this step: FP32 `[HIDDEN, PATCH_DIM]` + `[PATCH_DIM]` (video) and `[HIDDEN, AUDIO_CHANNELS]` + `[AUDIO_CHANNELS]`.
    video_weight: Tensor,
    video_bias: Tensor,
    audio_weight: Tensor,
    audio_bias: Tensor,
}

/// The tables the forward pass reads. A captured graph bakes in these addresses, so a step is selected by copying its tables
/// here, never by pointing the forward pass somewhere else.
struct Selected {
    modulation: Tensor,
    video_weight: Tensor,
    video_bias: Tensor,
    audio_weight: Tensor,
    audio_bias: Tensor,
}

fn upload(ctx: &CudaContext, bytes: &[u8], shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Cuda)?;
    buffer
        .copy_from_host_on(bytes, ctx.stream())
        .map_err(Error::Cuda)?;
    buffer
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

fn bf16_bytes(values: &[half::bf16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub(in crate::cuda_minimax_h3) struct Int8Blocks {
    ctx: Rc<CudaContext>,
    weights: Rc<DitWeights>,
    layout: Layout,
    /// FP32 `[total_rows, 2, HEAD_DIM / 2]` cosine/sine rows for the permuted half-split rotation.
    rope: Tensor,
    steps: Vec<StepTables>,
    selected: Selected,
}

impl Int8Blocks {
    /// Build the per-step tables for `schedule` (one entry per forward) and the rotary table of the packed layout. Pure host
    /// arithmetic plus uploads; no forward is run.
    pub fn new(
        ctx: Rc<CudaContext>,
        weights: Rc<DitWeights>,
        layout: Layout,
        schedule: &[StepTimes],
    ) -> Result<Self> {
        if schedule.is_empty() {
            return Err(Error::Other(
                "a MiniMax-H3 request needs at least one denoising step".into(),
            ));
        }
        let table = host::rope_table(&layout.positions(), &weights.rope_inv_freq);
        let rope = upload(
            &ctx,
            &f32_bytes(&table),
            vec![layout.total_rows(), 2, HEAD_DIM / 2],
            DType::F32,
        )?;
        let mut steps = Vec::with_capacity(schedule.len());
        for step in schedule {
            let (t_video, t_audio) = (1.0 - step.sigma_video, 1.0 - step.sigma_audio());
            let video = host::curve_embedding(&weights.adaln_table, t_video);
            let audio = host::curve_embedding(&weights.adaln_table, t_audio);
            let mut modulation = Vec::with_capacity(BLOCKS * MODALITIES * 6 * HIDDEN);
            for block in &weights.blocks {
                // text and visual rows follow the video stream; audio rows follow the audio stream
                let per_video = host::block_modulation(&block.adaln, &video);
                let per_audio = host::block_modulation(&block.adaln, &audio);
                let row = 6 * HIDDEN;
                modulation.extend_from_slice(
                    &per_video[MODALITY_VISUAL * row..(MODALITY_VISUAL + 1) * row],
                );
                modulation
                    .extend_from_slice(&per_video[MODALITY_TEXT * row..(MODALITY_TEXT + 1) * row]);
                modulation.extend_from_slice(
                    &per_audio[MODALITY_AUDIO * row..(MODALITY_AUDIO + 1) * row],
                );
            }
            // the final layer has a single modulation row set per stream, evaluated at that stream's timestep
            let (v_shift, v_scale) = host::final_modulation(&weights.final_layer.adaln, &video);
            let (a_shift, a_scale) = host::final_modulation(&weights.final_layer.adaln, &audio);
            let (vw, vb) =
                host::fold_head(&weights.final_layer.video, PATCH_DIM, &v_shift, &v_scale);
            let (aw, ab) = host::fold_head(
                &weights.final_layer.audio,
                AUDIO_CHANNELS,
                &a_shift,
                &a_scale,
            );
            steps.push(StepTables {
                modulation: upload(
                    &ctx,
                    &bf16_bytes(&modulation),
                    vec![BLOCKS, MODALITIES, 6, HIDDEN],
                    DType::BF16,
                )?,
                video_weight: upload(&ctx, &f32_bytes(&vw), vec![HIDDEN, PATCH_DIM], DType::F32)?,
                video_bias: upload(&ctx, &f32_bytes(&vb), vec![PATCH_DIM], DType::F32)?,
                audio_weight: upload(
                    &ctx,
                    &f32_bytes(&aw),
                    vec![HIDDEN, AUDIO_CHANNELS],
                    DType::F32,
                )?,
                audio_bias: upload(&ctx, &f32_bytes(&ab), vec![AUDIO_CHANNELS], DType::F32)?,
            });
        }
        let selected = Selected {
            modulation: alloc(&ctx, &[BLOCKS, MODALITIES, 6, HIDDEN], DType::BF16)?,
            video_weight: alloc(&ctx, &[HIDDEN, PATCH_DIM], DType::F32)?,
            video_bias: alloc(&ctx, &[PATCH_DIM], DType::F32)?,
            audio_weight: alloc(&ctx, &[HIDDEN, AUDIO_CHANNELS], DType::F32)?,
            audio_bias: alloc(&ctx, &[AUDIO_CHANNELS], DType::F32)?,
        };
        Ok(Self {
            ctx,
            weights,
            layout,
            rope,
            steps,
            selected,
        })
    }

    /// Embed the packed sequence: refined text rows, then `audio_patch_proj(audio rows)`, then `video_patch_proj(video rows)`.
    /// Both latents are FP32 and are rounded to BF16 first, as the reference model receives them.
    fn embed(
        &self,
        video: &Tensor,
        audio: &Tensor,
        text: &Tensor,
        gemm: &GemmPolicy,
    ) -> Result<Tensor> {
        let ctx = &*self.ctx;
        let total = self.layout.total_rows();
        let x = alloc(ctx, &[total, HIDDEN], DType::BF16)?;
        ops::copy_tensor(ctx, text, &rows(&x, 0, self.layout.text_rows, HIDDEN)?)?;
        let [_, audio_seg, video_seg] = self.layout.segments();
        for (latent, weight, seg, width) in [
            (audio, &self.weights.audio_patch, audio_seg, AUDIO_CHANNELS),
            (video, &self.weights.patch, video_seg, PATCH_DIM),
        ] {
            let rounded = alloc(ctx, &[seg.rows(), width], DType::BF16)?;
            ops::quantization::cast(ctx, latent, &rounded)?;
            let widened = alloc(ctx, &[seg.rows(), width], DType::F32)?;
            ops::quantization::cast(ctx, &rounded, &widened)?;
            let projected = dense_linear(ctx, &widened, weight, gemm)?;
            ops::quantization::cast(ctx, &projected, &rows(&x, seg.start, seg.rows(), HIDDEN)?)?;
        }
        Ok(x)
    }

    /// RMSNorm with the block's learned weight, then `* (1 + scale) + shift` using each segment's modality row.
    fn modulated_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        modulation: &Tensor,
        part: usize,
    ) -> Result<Tensor> {
        let ctx = &*self.ctx;
        let out = alloc(ctx, x.shape().dims(), DType::BF16)?;
        for seg in self.layout.segments() {
            let row = |p: usize| {
                byte_view(
                    modulation,
                    (seg.kind.modality() * 6 + p) * HIDDEN,
                    HIDDEN,
                    vec![HIDDEN],
                )
            };
            let (shift, scale) = (row(part)?, row(part + 1)?);
            ops::norm::norm(
                ctx,
                NormArgs {
                    input: &rows(x, seg.start, seg.rows(), HIDDEN)?,
                    output: Some(&rows(&out, seg.start, seg.rows(), HIDDEN)?),
                    kind: ops::NormKind::Rms,
                    affine: ops::Affine::WeightedAdaptive {
                        weight,
                        scale: &scale,
                        shift: &shift,
                    },
                    quantized: None,
                    numerics: ops::NormNumerics::Flexible,
                    eps: NORM_EPS,
                },
            )?;
        }
        Ok(out)
    }

    /// `x += gate * branch`, one rounding, with each segment's modality gate (`part` is the shift row of its group).
    fn gated_residual(
        &self,
        x: &Tensor,
        branch: &Tensor,
        modulation: &Tensor,
        part: usize,
    ) -> Result<()> {
        let ctx = &*self.ctx;
        for seg in self.layout.segments() {
            let style = byte_view(
                modulation,
                (seg.kind.modality() * 6 + part) * HIDDEN,
                3 * HIDDEN,
                vec![3 * HIDDEN],
            )?;
            let target = rows(x, seg.start, seg.rows(), HIDDEN)?;
            ops::residual_add(
                ctx,
                ops::ResidualAddArgs {
                    input: &rows(branch, seg.start, seg.rows(), HIDDEN)?,
                    bias: None,
                    residual: &target,
                    gate: Some(ops::ResidualGate::PackedStyle(&style)),
                    hidden: &target,
                    bias_order: ops::BiasOrder::SingleRound,
                },
            )?;
        }
        Ok(())
    }

    fn attention_branch(
        &self,
        block: &DitBlock,
        x: &Tensor,
        modulation: &Tensor,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
    ) -> Result<()> {
        let ctx = &*self.ctx;
        let total = self.layout.total_rows();
        let h = self.modulated_norm(x, &block.norm1, modulation, 0)?;
        let rotated = ops::quantization::rotate_quantize_rows_i8(ctx, &h)?;
        let a = &block.attention;
        let q = int8_linear(ctx, &rotated, &a.q, gemm)?;
        let k = int8_linear(ctx, &rotated, &a.k, gemm)?;
        let v = int8_linear(ctx, &rotated, &a.v, gemm)?;
        // per-head RMSNorm and the half-split rotation over the permuted channels, in place
        for (t, norm) in [(&q, &a.q_norm), (&k, &a.k_norm)] {
            let heads = t.reshape(vec![total, HEADS, HEAD_DIM])?;
            ops::rms_norm_apply_precomputed(ctx, &heads, norm, &self.rope, NORM_EPS, &heads)?;
        }
        let attended = attend(ctx, &q, &k, &v, attention)?;
        let rotated = ops::quantization::rotate_quantize_rows_i8(ctx, &attended)?;
        let branch = int8_linear(ctx, &rotated, &a.out, gemm)?;
        self.gated_residual(x, &branch, modulation, 0)
    }

    fn mlp_branch(
        &self,
        block: &DitBlock,
        x: &Tensor,
        modulation: &Tensor,
        gemm: &GemmPolicy,
    ) -> Result<()> {
        let ctx = &*self.ctx;
        let h = self.modulated_norm(x, &block.norm2, modulation, 3)?;
        let rotated = ops::quantization::rotate_quantize_rows_i8(ctx, &h)?;
        let gate = int8_linear(ctx, &rotated, &block.mlp.gate, gemm)?;
        let up = int8_linear(ctx, &rotated, &block.mlp.up, gemm)?;
        let activated = ops::pointwise::silu_mul(ctx, &gate, &up)?;
        let rotated = ops::quantization::rotate_quantize_rows_i8(ctx, &activated)?;
        let branch = int8_linear(ctx, &rotated, &block.mlp.fc2, gemm)?;
        self.gated_residual(x, &branch, modulation, 3)
    }

    /// Bytes the session arena needs for one forward at this layout: the live set of the MLP branch is the largest, about
    /// 124 KiB per row, plus the residual stream and GEMM scratch.
    pub fn arena_bytes(layout: Layout) -> usize {
        layout.total_rows() * 160 * 1024 + 384 * 1024 * 1024
    }

    /// Copy the tables of `step` into the buffers the forward pass reads. Eager and stream-ordered, so it runs between graph
    /// replays; the forward pass itself never changes address.
    pub fn select_step(&self, step: usize) -> Result<()> {
        let tables = self
            .steps
            .get(step)
            .ok_or_else(|| Error::Other(format!("step {step} is outside the schedule")))?;
        let ctx = &*self.ctx;
        ops::copy_tensor(ctx, &tables.modulation, &self.selected.modulation)?;
        ops::copy_tensor(ctx, &tables.video_weight, &self.selected.video_weight)?;
        ops::copy_tensor(ctx, &tables.video_bias, &self.selected.video_bias)?;
        ops::copy_tensor(ctx, &tables.audio_weight, &self.selected.audio_weight)?;
        ops::copy_tensor(ctx, &tables.audio_bias, &self.selected.audio_bias)
    }

    /// One head over a target segment: final RMSNorm on its rows, then the step's folded FP32 head.
    fn head(
        &self,
        x: &Tensor,
        kind: Kind,
        weight: &Tensor,
        bias: &Tensor,
        out: &Tensor,
        gemm: &GemmPolicy,
    ) -> Result<()> {
        let ctx = &*self.ctx;
        let seg = self
            .layout
            .segments()
            .into_iter()
            .find(|s| s.kind == kind)
            .expect("layout has every kind");
        let part = rows(x, seg.start, seg.rows(), HIDDEN)?;
        let normed = alloc(ctx, &[seg.rows(), HIDDEN], DType::BF16)?;
        ops::norm::norm(
            ctx,
            NormArgs::rms(&part, &self.weights.final_layer.norm, &normed, NORM_EPS),
        )?;
        let widened = alloc(ctx, &[seg.rows(), HIDDEN], DType::F32)?;
        ops::quantization::cast(ctx, &normed, &widened)?;
        let mut target = out.clone();
        let mut args = GemmArgs::new(&widened, weight, &mut target);
        args.policy = gemm.clone();
        args.weight_version = Some(ops::WeightVersion::new(0));
        ops::gemm_bias(ctx, ops::GemmBiasArgs { gemm: args, bias })
    }

    /// Run the DiT on the selected step. `video` is FP32 `[video_rows, PATCH_DIM]`, `audio` FP32 `[audio_rows, AUDIO_CHANNELS]`,
    /// `text` the refined BF16 `[text_rows, HIDDEN]`. The FP32 head outputs are written to `video_out` and `audio_out`; the model's
    /// returned velocity is the negation of the head output and is handled by the sampler. Every input and output is caller-owned and
    /// keeps its address, so this is capturable.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_into(
        &self,
        video: &Tensor,
        audio: &Tensor,
        text: &Tensor,
        video_out: &Tensor,
        audio_out: &Tensor,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
    ) -> Result<()> {
        self.forward_tapped(
            video,
            audio,
            text,
            video_out,
            audio_out,
            gemm,
            attention,
            &mut |_, _| Ok(()),
        )
    }

    /// As [`Self::forward_into`], calling `tap(index, x)` with the residual stream after the embedding (`index == usize::MAX`) and after
    /// every block. The tap only reads `x`; the parity tests use it to find the block where a difference starts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tapped(
        &self,
        video: &Tensor,
        audio: &Tensor,
        text: &Tensor,
        video_out: &Tensor,
        audio_out: &Tensor,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
        tap: &mut dyn FnMut(usize, &Tensor) -> Result<()>,
    ) -> Result<()> {
        let x = self.embed(video, audio, text, gemm)?;
        tap(usize::MAX, &x)?;
        for (index, block) in self.weights.blocks.iter().enumerate() {
            let modulation = byte_view(
                &self.selected.modulation,
                index * MODALITIES * 6 * HIDDEN,
                MODALITIES * 6 * HIDDEN,
                vec![MODALITIES, 6, HIDDEN],
            )?;
            self.attention_branch(block, &x, &modulation, gemm, attention)?;
            self.mlp_branch(block, &x, &modulation, gemm)?;
            tap(index, &x)?;
        }
        self.head(
            &x,
            Kind::Video,
            &self.selected.video_weight,
            &self.selected.video_bias,
            video_out,
            gemm,
        )?;
        self.head(
            &x,
            Kind::Audio,
            &self.selected.audio_weight,
            &self.selected.audio_bias,
            audio_out,
            gemm,
        )
    }
}

//! INT8 ConvRot DiT: per-step modulation tables, the 50 transformer blocks and the final PDD head.
use super::refiner::attend;
use super::util::{alloc, byte_view, dense_linear, int8_linear, rows};
use crate::cuda_minimax_h3_image::backend::ops::{self, GemmArgs, GemmPolicy, NormArgs};
use crate::cuda_minimax_h3_image::backend::{AttentionPolicy, CudaBuffer, CudaContext};
use crate::cuda_minimax_h3_image::config::*;
use crate::cuda_minimax_h3_image::model::host;
use crate::cuda_minimax_h3_image::weights::{DitBlock, DitWeights};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use std::rc::Rc;

/// Everything of one denoising step that depends on its timestep and nothing else, uploaded once per request.
struct StepTables {
    /// BF16 `[BLOCKS, MODALITIES_USED, 6, HIDDEN]`: per block the modulation rows of image (0) and text (1).
    modulation: Tensor,
    /// Folded PDD head of this step: FP32 `[HIDDEN, PATCH_DIM]` and `[PATCH_DIM]`.
    head_weight: Tensor,
    head_bias: Tensor,
}

/// The tables the forward pass reads. A captured graph bakes in these addresses, so a step is selected by copying its
/// tables here, never by pointing the forward pass somewhere else.
struct Selected {
    modulation: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
}

/// Modality rows kept per block (image and text; audio is video-only).
const MODALITIES_USED: usize = MODALITY_TEXT + 1;

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

pub(in crate::cuda_minimax_h3_image) struct Int8Blocks {
    ctx: Rc<CudaContext>,
    weights: Rc<DitWeights>,
    layout: Layout,
    /// FP32 `[total_rows, 2, HEAD_DIM / 2]` cosine/sine rows for the permuted half-split rotation.
    rope: Tensor,
    steps: Vec<StepTables>,
    selected: Selected,
}

impl Int8Blocks {
    /// Build the per-step tables for `sigmas` (`sigmas[i]` is the noise level the model sees at step `i`) and
    /// the rotary table of the packed layout. Pure host arithmetic plus uploads; no forward is run.
    pub fn new(
        ctx: Rc<CudaContext>,
        weights: Rc<DitWeights>,
        layout: Layout,
        canvas: Canvas,
        sigmas: &[f32],
    ) -> Result<Self> {
        if sigmas.len() != PDD_STEPS {
            return Err(Error::Other(format!(
                "expected {PDD_STEPS} step sigmas, got {}",
                sigmas.len()
            )));
        }
        let table = host::rope_table(&host::positions(layout, canvas), &weights.rope_inv_freq);
        let rope = upload(
            &ctx,
            &f32_bytes(&table),
            vec![layout.total_rows(), 2, HEAD_DIM / 2],
            DType::F32,
        )?;
        let mut steps = Vec::with_capacity(PDD_STEPS);
        for (step, &sigma) in sigmas.iter().enumerate() {
            let embedding = host::curve_embedding(&weights.adaln_table, 1.0 - sigma);
            let mut modulation = Vec::with_capacity(BLOCKS * MODALITIES_USED * 6 * HIDDEN);
            for block in &weights.blocks {
                modulation.extend(host::block_modulation(&block.adaln, &embedding));
            }
            let (shift, scale) = host::final_modulation(&weights.final_layer.adaln, &embedding);
            let (head_w, head_b) =
                host::fold_head(&weights.final_layer.heads[step], &shift, &scale);
            steps.push(StepTables {
                modulation: upload(
                    &ctx,
                    &bf16_bytes(&modulation),
                    vec![BLOCKS, MODALITIES_USED, 6, HIDDEN],
                    DType::BF16,
                )?,
                head_weight: upload(
                    &ctx,
                    &f32_bytes(&head_w),
                    vec![HIDDEN, PATCH_DIM],
                    DType::F32,
                )?,
                head_bias: upload(&ctx, &f32_bytes(&head_b), vec![PATCH_DIM], DType::F32)?,
            });
        }
        let selected = Selected {
            modulation: alloc(&ctx, &[BLOCKS, MODALITIES_USED, 6, HIDDEN], DType::BF16)?,
            head_weight: alloc(&ctx, &[HIDDEN, PATCH_DIM], DType::F32)?,
            head_bias: alloc(&ctx, &[PATCH_DIM], DType::F32)?,
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

    /// Embed the packed sequence: refined text rows, then the image rows `video_patch_proj(latent)`.
    /// `latent` is FP32 `[image_rows, PATCH_DIM]`; it is rounded to BF16 first, as the reference model receives it.
    fn embed(&self, latent: &Tensor, text: &Tensor, gemm: &GemmPolicy) -> Result<Tensor> {
        let ctx = &*self.ctx;
        let total = self.layout.total_rows();
        let x = alloc(ctx, &[total, HIDDEN], DType::BF16)?;
        ops::copy_tensor(ctx, text, &rows(&x, 0, self.layout.text_rows, HIDDEN)?)?;
        let rounded = alloc(ctx, &[self.layout.image_rows, PATCH_DIM], DType::BF16)?;
        ops::quantization::cast(ctx, latent, &rounded)?;
        let widened = alloc(ctx, &[self.layout.image_rows, PATCH_DIM], DType::F32)?;
        ops::quantization::cast(ctx, &rounded, &widened)?;
        let projected = dense_linear(ctx, &widened, &self.weights.patch, gemm)?;
        let image = rows(&x, self.layout.text_rows, self.layout.image_rows, HIDDEN)?;
        ops::quantization::cast(ctx, &projected, &image)?;
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
        for (first, count, modality) in self.segments() {
            let row = |p: usize| {
                byte_view(
                    modulation,
                    (modality * 6 + p) * HIDDEN,
                    HIDDEN,
                    vec![HIDDEN],
                )
            };
            let (shift, scale) = (row(part)?, row(part + 1)?);
            ops::norm::norm(
                ctx,
                NormArgs {
                    input: &rows(x, first, count, HIDDEN)?,
                    output: Some(&rows(&out, first, count, HIDDEN)?),
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
        for (first, count, modality) in self.segments() {
            let style = byte_view(
                modulation,
                (modality * 6 + part) * HIDDEN,
                3 * HIDDEN,
                vec![3 * HIDDEN],
            )?;
            let target = rows(x, first, count, HIDDEN)?;
            ops::residual_add(
                ctx,
                ops::ResidualAddArgs {
                    input: &rows(branch, first, count, HIDDEN)?,
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

    /// `(first_row, row_count, modality)` of the text and image segments.
    fn segments(&self) -> [(usize, usize, usize); 2] {
        [
            (0, self.layout.text_rows, MODALITY_TEXT),
            (
                self.layout.text_rows,
                self.layout.image_rows,
                MODALITY_IMAGE,
            ),
        ]
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

    /// Copy the tables of `step` into the buffers the forward pass reads. Eager and stream-ordered, so it runs between
    /// graph replays; the forward pass itself never changes address.
    pub fn select_step(&self, step: usize) -> Result<()> {
        let tables = self
            .steps
            .get(step)
            .ok_or_else(|| Error::Other(format!("step {step} is outside the schedule")))?;
        let ctx = &*self.ctx;
        ops::copy_tensor(ctx, &tables.modulation, &self.selected.modulation)?;
        ops::copy_tensor(ctx, &tables.head_weight, &self.selected.head_weight)?;
        ops::copy_tensor(ctx, &tables.head_bias, &self.selected.head_bias)
    }

    /// Run the DiT on the selected step. `latent` is FP32 `[image_rows, PATCH_DIM]`, `text` the refined BF16
    /// `[text_rows, HIDDEN]`; the FP32 velocity head output for the image rows, `[image_rows, PATCH_DIM]`, is written
    /// to `out`. Every input and the output are caller-owned and keep their address, so this is capturable.
    pub fn forward_into(
        &self,
        latent: &Tensor,
        text: &Tensor,
        out: &Tensor,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
    ) -> Result<()> {
        let ctx = &*self.ctx;
        let x = self.embed(latent, text, gemm)?;
        for (index, block) in self.weights.blocks.iter().enumerate() {
            let modulation = byte_view(
                &self.selected.modulation,
                index * MODALITIES_USED * 6 * HIDDEN,
                MODALITIES_USED * 6 * HIDDEN,
                vec![MODALITIES_USED, 6, HIDDEN],
            )?;
            self.attention_branch(block, &x, &modulation, gemm, attention)?;
            self.mlp_branch(block, &x, &modulation, gemm)?;
        }
        // final norm on the image rows only; its scale and shift are folded into the step's FP32 head
        let image = rows(&x, self.layout.text_rows, self.layout.image_rows, HIDDEN)?;
        let normed = alloc(ctx, &[self.layout.image_rows, HIDDEN], DType::BF16)?;
        ops::norm::norm(
            ctx,
            NormArgs::rms(&image, &self.weights.final_layer.norm, &normed, NORM_EPS),
        )?;
        let widened = alloc(ctx, &[self.layout.image_rows, HIDDEN], DType::F32)?;
        ops::quantization::cast(ctx, &normed, &widened)?;
        let mut velocity = out.clone();
        let mut args = GemmArgs::new(&widened, &self.selected.head_weight, &mut velocity);
        args.policy = gemm.clone();
        args.weight_version = Some(ops::WeightVersion::new(0));
        ops::gemm_bias(
            ctx,
            ops::GemmBiasArgs {
                gemm: args,
                bias: &self.selected.head_bias,
            },
        )
    }
}

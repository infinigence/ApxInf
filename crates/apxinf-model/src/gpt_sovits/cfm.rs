//! V5 conditional flow-matching decoder (DiT).

use apxinf_core::{Error, Result, RngKey, Tensor};
use apxinf_cuda_new::CudaContext as Context;

use super::config::{
    CFM_DEPTH, CFM_DIM, CFM_HEADS, CFM_HEAD_DIM, CFM_MEL, CFM_TEXT, CFM_TEXT_BLOCKS,
};
use super::ops;
use super::weights::{fp16_tensor, Linear, Weights};

const MAX_POSITIONS: usize = 4096;
const NOISE_TEMPERATURE: f32 = 0.875;
const REFERENCE_FRAMES: usize = 500;
const MAX_TARGET_CHUNK: usize = 640;
const TOTAL_CHUNK: usize = 1000;
const ROLLING_TAIL: usize = 32;

struct TextBlock {
    dw_w: Tensor,
    dw_b: Option<Tensor>,
    ln_w: Tensor,
    ln_b: Tensor,
    pw1: Linear,
    grn_g: Tensor,
    grn_b: Tensor,
    pw2: Linear,
}

struct Block {
    ada: Linear,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ff1: Linear,
    ff2: Linear,
}

pub struct Cfm {
    step_modulations: Vec<StepModulation>,
    text_blocks: Vec<TextBlock>,
    text_pe: Tensor,
    noise_scale: Tensor,
    norm_ones: Tensor,
    norm_zeros: Tensor,
    w_x: Tensor,
    w_text: Tensor,
    bias: Tensor,
    convpos0_w: Tensor,
    convpos0_b: Option<Tensor>,
    convpos2_w: Tensor,
    convpos2_b: Option<Tensor>,
    blocks: Vec<Block>,
    proj_out: Linear,
    rotary_cos: Tensor,
    rotary_sin: Tensor,
}

struct BlockModulation {
    shift_msa: Tensor,
    scale_msa: Tensor,
    gate_msa: Tensor,
    shift_mlp: Tensor,
    scale_mlp: Tensor,
    gate_mlp: Tensor,
}

pub(crate) struct StepModulation {
    blocks: Vec<BlockModulation>,
    final_scale: Tensor,
    final_shift: Tensor,
}

fn sinus_tensor(ctx: &Context) -> Result<Tensor> {
    let times = (0..32).map(|i| i as f32 / 32.).collect::<Vec<_>>();
    let times = fp16_tensor(ctx, &[32], &times)?;
    let output = ops::alloc(ctx, &[32, 256])?;
    apxinf_cuda_new::ops::sinusoidal_embedding(ctx, &times, &output, 1000., 10000.)?;
    Ok(output)
}

fn rotary_tables(inv_freq: &[f32]) -> (Vec<f32>, Vec<f32>) {
    // The reference applies rotary embeddings before reshaping Q/K heads.
    // Its table therefore covers only head zero.  Expand to model width with
    // identity rotations for the remaining heads so the standard interleaved
    // kernel reproduces the reference behavior in one pass.
    let width = CFM_DIM;
    let rope_width = 2 * inv_freq.len();
    let mut cos_table = vec![1.0f32; MAX_POSITIONS * width];
    let mut sin_table = vec![0.0f32; MAX_POSITIONS * width];
    for position in 0..MAX_POSITIONS {
        for (index, frequency) in inv_freq.iter().enumerate() {
            // RotaryEmbedding follows the model's FP16 dtype in the reference.
            // Match its rounding of the position, frequency, product and trig
            // outputs before the half-precision rotary multiply/add.
            let position_value = half::f16::from_f32(position as f32).to_f32();
            let frequency = half::f16::from_f32(*frequency).to_f32();
            let angle = half::f16::from_f32(position_value * frequency).to_f32();
            let cosine = half::f16::from_f32(angle.cos()).to_f32();
            let sine = half::f16::from_f32(angle.sin()).to_f32();
            let offset = position * width;
            cos_table[offset + 2 * index] = cosine;
            cos_table[offset + 2 * index + 1] = cosine;
            sin_table[offset + 2 * index] = sine;
            sin_table[offset + 2 * index + 1] = sine;
        }
        debug_assert_eq!(rope_width, CFM_HEAD_DIM);
    }
    (cos_table, sin_table)
}

impl Cfm {
    pub fn load(weights: &mut Weights, ctx: &Context) -> Result<Self> {
        let time0 = Linear::load(weights, ctx, "cfm.time0")?;
        let time2 = Linear::load(weights, ctx, "cfm.time2")?;
        let mut text_blocks = Vec::with_capacity(CFM_TEXT_BLOCKS);
        for index in 0..CFM_TEXT_BLOCKS {
            let prefix = format!("cfm.text{index}");
            text_blocks.push(TextBlock {
                dw_w: weights.get(ctx, &format!("{prefix}.dw.w"))?,
                dw_b: weights.get_optional(ctx, &format!("{prefix}.dw.b"))?,
                ln_w: weights.get(ctx, &format!("{prefix}.ln.w"))?,
                ln_b: weights.get(ctx, &format!("{prefix}.ln.b"))?,
                pw1: Linear::load(weights, ctx, &format!("{prefix}.pw1"))?,
                grn_g: weights.get(ctx, &format!("{prefix}.grn.g"))?,
                grn_b: weights.get(ctx, &format!("{prefix}.grn.b"))?,
                pw2: Linear::load(weights, ctx, &format!("{prefix}.pw2"))?,
            });
        }
        let host_proj = weights.host("cfm.input_proj.w")?;
        let values = host_proj.to_f32_vec()?;
        let shape = host_proj.shape().dims();
        let (rows, width) = match &shape[..] {
            [rows, width] => (*rows, *width),
            _ => {
                return Err(Error::Other(format!(
                    "cfm.input_proj weight shape mismatch: got {shape:?}"
                )))
            }
        };
        let expected_rows = CFM_MEL * 2 + CFM_TEXT;
        if rows != expected_rows || rows.checked_mul(width) != Some(values.len()) {
            return Err(Error::Other(format!(
                "cfm.input_proj weight shape mismatch: got {:?}",
                shape
            )));
        }
        // The converted native checkpoint stores this projection as
        // `[input, output]`: the first mel inputs feed `w_x`, and the
        // remaining static-conditioning inputs feed `w_text`.
        let x_values = values[..CFM_MEL * width].to_vec();
        let text_values = values[CFM_MEL * width..].to_vec();
        let w_x = fp16_tensor(ctx, &[CFM_MEL, width], &x_values)?;
        let w_text = fp16_tensor(ctx, &[CFM_MEL + CFM_TEXT, width], &text_values)?;
        let bias = weights.get(ctx, "cfm.input_proj.b")?;

        let mut blocks = Vec::with_capacity(CFM_DEPTH);
        for index in 0..CFM_DEPTH {
            let prefix = format!("cfm.blocks.{index}");
            blocks.push(Block {
                ada: Self::packed_linear(
                    weights,
                    ctx,
                    &[
                        format!("{prefix}.ada.shift_msa"),
                        format!("{prefix}.ada.scale_msa"),
                        format!("{prefix}.ada.gate_msa"),
                        format!("{prefix}.ada.shift_mlp"),
                        format!("{prefix}.ada.scale_mlp"),
                        format!("{prefix}.ada.gate_mlp"),
                    ],
                )?,
                q: Linear::load(weights, ctx, &format!("{prefix}.q"))?,
                k: Linear::load(weights, ctx, &format!("{prefix}.k"))?,
                v: Linear::load(weights, ctx, &format!("{prefix}.v"))?,
                o: Linear::load(weights, ctx, &format!("{prefix}.o"))?,
                ff1: Linear::load(weights, ctx, &format!("{prefix}.ff1"))?,
                ff2: Linear::load(weights, ctx, &format!("{prefix}.ff2"))?,
            });
        }
        let inv_freq = weights.host("cfm.rotary.inv_freq")?.to_f32_vec()?;
        let (rotary_cos, rotary_sin) = rotary_tables(&inv_freq);
        let final_ada = Self::packed_linear(
            weights,
            ctx,
            &["cfm.final.scale".into(), "cfm.final.shift".into()],
        )?;
        let sinus = sinus_tensor(ctx)?;
        let norm_ones = fp16_tensor(ctx, &[CFM_DIM], &vec![1.0; CFM_DIM])?;
        let mut step_modulations = Vec::with_capacity(32);
        // The admitted schedule indexes a fixed 32-row sinusoidal table.
        // Keep every projection's original [1,K] geometry and FP16 rounding.
        // These values depend only on immutable checkpoint weights and time;
        // moving them to loading removes repeated GEMMs from every chunk.
        for step_index in 0..32 {
            let row = ops::slice_rows(ctx, &sinus, step_index, 1)?;
            let time = time0.apply_accumulator(ctx, &row)?;
            let time = ops::silu(ctx, &time)?;
            let time = time2.apply_accumulator(ctx, &time)?;
            let ada_time = ops::silu(ctx, &time)?;
            let mut modulations = Vec::with_capacity(blocks.len());
            for block in &blocks {
                modulations.push(Self::block_modulation(
                    ctx, &block.ada, &ada_time, &norm_ones,
                )?);
            }
            let final_values = final_ada
                .apply_accumulator(ctx, &ada_time)?
                .reshape(apxinf_core::Shape::new(vec![2, CFM_DIM]))?;
            let final_scale = ops::slice_rows(ctx, &final_values, 0, 1)?
                .reshape(apxinf_core::Shape::new(vec![CFM_DIM]))?;
            let final_shift = ops::slice_rows(ctx, &final_values, 1, 1)?
                .reshape(apxinf_core::Shape::new(vec![CFM_DIM]))?;
            step_modulations.push(StepModulation {
                blocks: modulations,
                final_scale: ops::add_bias(ctx, &final_scale, &norm_ones)?,
                final_shift,
            });
        }
        Ok(Self {
            step_modulations,
            text_blocks,
            // The converter preserves the reference nonpersistent positional buffer.
            text_pe: {
                let table = weights.get(ctx, "cfm.text_pe")?;
                if table.shape().dims() != [MAX_POSITIONS, CFM_TEXT] {
                    return Err(Error::Other(
                        "cfm.text_pe must have shape [4096,512]".into(),
                    ));
                }
                table
            },
            noise_scale: fp16_tensor(ctx, &[CFM_MEL], &vec![NOISE_TEMPERATURE; CFM_MEL])?,
            norm_ones,
            norm_zeros: fp16_tensor(ctx, &[CFM_DIM], &vec![0.0; CFM_DIM])?,
            w_x,
            w_text,
            bias,
            convpos0_w: weights.get(ctx, "cfm.convpos0.w")?,
            convpos0_b: weights.get_optional(ctx, "cfm.convpos0.b")?,
            convpos2_w: weights.get(ctx, "cfm.convpos2.w")?,
            convpos2_b: weights.get_optional(ctx, "cfm.convpos2.b")?,
            blocks,
            proj_out: Linear::load(weights, ctx, "cfm.proj_out")?,
            rotary_cos: fp16_tensor(ctx, &[MAX_POSITIONS, CFM_DIM], &rotary_cos)?,
            rotary_sin: fp16_tensor(ctx, &[MAX_POSITIONS, CFM_DIM], &rotary_sin)?,
        })
    }

    #[cfg(test)]
    fn validate_step_modulations(&self, ctx: &Context, weights: &mut Weights) -> Result<()> {
        let time0 = Linear::load(weights, ctx, "cfm.time0")?;
        let time2 = Linear::load(weights, ctx, "cfm.time2")?;
        let sinus = sinus_tensor(ctx)?;
        let final_ada = Self::packed_linear(
            weights,
            ctx,
            &["cfm.final.scale".into(), "cfm.final.shift".into()],
        )?;
        let exact = |left: &Tensor, right: &Tensor| -> Result<()> {
            ctx.synchronize().map_err(Error::Cuda)?;
            let left = apxinf_cuda_new::transfers::to_cpu(left)?;
            let right = apxinf_cuda_new::transfers::to_cpu(right)?;
            assert_eq!(left.shape(), right.shape());
            assert_eq!(
                left.storage().as_cpu().unwrap(),
                right.storage().as_cpu().unwrap()
            );
            Ok(())
        };
        for index in 0..32 {
            let row = ops::slice_rows(ctx, &sinus, index, 1)?;
            let time = time0.apply_accumulator(ctx, &row)?;
            let time = time2.apply_accumulator(ctx, &ops::silu(ctx, &time)?)?;
            let ada = ops::silu(ctx, &time)?;
            for (block, stored) in self.blocks.iter().zip(&self.step_modulations[index].blocks) {
                let expected = Self::block_modulation(ctx, &block.ada, &ada, &self.norm_ones)?;
                exact(&expected.shift_msa, &stored.shift_msa)?;
                exact(&expected.scale_msa, &stored.scale_msa)?;
                exact(&expected.gate_msa, &stored.gate_msa)?;
                exact(&expected.shift_mlp, &stored.shift_mlp)?;
                exact(&expected.scale_mlp, &stored.scale_mlp)?;
                exact(&expected.gate_mlp, &stored.gate_mlp)?;
            }
            let final_values = final_ada
                .apply_accumulator(ctx, &ada)?
                .reshape(apxinf_core::Shape::new(vec![2, CFM_DIM]))?;
            let scale = ops::slice_rows(ctx, &final_values, 0, 1)?
                .reshape(apxinf_core::Shape::new(vec![CFM_DIM]))?;
            let shift = ops::slice_rows(ctx, &final_values, 1, 1)?
                .reshape(apxinf_core::Shape::new(vec![CFM_DIM]))?;
            exact(
                &ops::add_bias(ctx, &scale, &self.norm_ones)?,
                &self.step_modulations[index].final_scale,
            )?;
            exact(&shift, &self.step_modulations[index].final_shift)?;
        }
        Ok(())
    }

    // Checkpoint conversion split the output channels. Restore the original
    // linear geometry once at loading so FP16 GEMM sees the reference shape.
    fn packed_linear(weights: &mut Weights, ctx: &Context, prefixes: &[String]) -> Result<Linear> {
        let mut parts = prefixes.iter();
        let first = parts
            .next()
            .ok_or_else(|| Error::Other("empty CFM projection".into()))?;
        let first = Linear::load(weights, ctx, first)?;
        let mut w = first.w;
        let mut b = first
            .b
            .ok_or_else(|| Error::Other("CFM modulation bias missing".into()))?
            .reshape(apxinf_core::Shape::new(vec![1, CFM_DIM]))?;
        for prefix in parts {
            let part = Linear::load(weights, ctx, prefix)?;
            w = ops::concat_columns(ctx, &w, &part.w)?;
            let bias = part
                .b
                .ok_or_else(|| Error::Other("CFM modulation bias missing".into()))?
                .reshape(apxinf_core::Shape::new(vec![1, CFM_DIM]))?;
            b = ops::concat_columns(ctx, &b, &bias)?;
        }
        let width = b.numel();
        Ok(Linear {
            w,
            b: Some(b.reshape(apxinf_core::Shape::new(vec![width]))?),
        })
    }

    fn block_modulation(
        ctx: &Context,
        linear: &Linear,
        time: &Tensor,
        ones: &Tensor,
    ) -> Result<BlockModulation> {
        let projected = linear
            .apply_accumulator(ctx, time)?
            .reshape(apxinf_core::Shape::new(vec![6, CFM_DIM]))?;
        let vector = |part| {
            ops::slice_rows(ctx, &projected, part, 1)?
                .reshape(apxinf_core::Shape::new(vec![CFM_DIM]))
        };
        Ok(BlockModulation {
            shift_msa: vector(0)?,
            scale_msa: ops::add_bias(ctx, &vector(1)?, ones)?,
            gate_msa: vector(2)?,
            shift_mlp: vector(3)?,
            scale_mlp: ops::add_bias(ctx, &vector(4)?, ones)?,
            gate_mlp: vector(5)?,
        })
    }

    /// Rotate only head zero, matching the reference Q/K projection order.
    fn rope_first_head(
        ctx: &Context,
        tensor: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let dims = tensor.shape().dims();
        let (length, width) = match dims[..] {
            [length, width] => (length, width),
            _ => {
                return Err(Error::Other(format!(
                    "CFM rope expects [length, width], got {dims:?}"
                )))
            }
        };
        if width != CFM_DIM {
            return Err(Error::Other(format!(
                "CFM rope width mismatch: got {width}, expected {CFM_DIM}"
            )));
        }
        let flat = tensor.reshape(apxinf_core::Shape::new(vec![1, length, 1, width]))?;
        let rotated = ops::rope_apply(ctx, &flat, cos, sin)?;
        rotated.reshape(apxinf_core::Shape::new(vec![
            1,
            length,
            CFM_HEADS,
            CFM_HEAD_DIM,
        ]))
    }

    fn text_embed(&self, ctx: &Context, text: &Tensor) -> Result<Tensor> {
        let len = text.shape().dims()[0];
        let pe = ops::slice_rows(ctx, &self.text_pe, 0, len)?;
        let mut x = ops::residual_add(ctx, text, &pe)?;

        for block in &self.text_blocks {
            let residual = x.clone();
            let x3 = x.reshape(apxinf_core::Shape::new(vec![1, len, CFM_TEXT]))?;
            let mut hidden = ops::conv1d(
                ctx,
                &x3,
                &block.dw_w,
                block.dw_b.as_ref(),
                1,
                3,
                1,
                CFM_TEXT,
            )?;
            hidden = hidden.reshape(apxinf_core::Shape::new(vec![len, CFM_TEXT]))?;
            hidden = ops::layer_norm(ctx, &hidden, &block.ln_w, &block.ln_b, 1e-6)?;
            hidden = block.pw1.apply_accumulator(ctx, &hidden)?;
            // ConvNeXt text blocks use the exact GELU; the DiT FFN below
            // uses the separate tanh approximation.
            hidden = ops::gelu_exact(ctx, &hidden)?;
            hidden = ops::grn(ctx, &hidden, &block.grn_g, &block.grn_b)?;
            hidden = block.pw2.apply_accumulator(ctx, &hidden)?;
            x = ops::residual_add(ctx, &hidden, &residual)?;
        }
        Ok(x)
    }

    // PyTorch eager rounds the product before adding shift/residual.
    // Preserve that tensor boundary using existing device primitives.
    fn staged_affine(
        &self,
        ctx: &Context,
        input: &Tensor,
        scale: &Tensor,
        shift: &Tensor,
    ) -> Result<Tensor> {
        let product = ops::affine(ctx, input, scale, &self.norm_zeros)?;
        ops::add_bias(ctx, &product, shift)
    }

    fn staged_gate_residual(
        &self,
        ctx: &Context,
        input: &Tensor,
        residual: &Tensor,
        gate: &Tensor,
    ) -> Result<Tensor> {
        let product = ops::affine(ctx, input, gate, &self.norm_zeros)?;
        ops::residual_add(ctx, &product, residual)
    }

    #[allow(clippy::too_many_arguments)]
    fn estimator(
        &self,
        ctx: &Context,
        x: &Tensor,
        _prompt_x: &Tensor,
        _text_embedding: &Tensor,
        static_condition: &Tensor,
        step_index: usize,
    ) -> Result<Tensor> {
        let modulation = self.step_modulations.get(step_index).ok_or_else(|| {
            Error::Other(format!("CFM modulation index {step_index} exceeds 32 rows"))
        })?;
        self.estimator_with_modulation(ctx, x, static_condition, modulation)
    }

    fn estimator_with_modulation(
        &self,
        ctx: &Context,
        x: &Tensor,
        static_condition: &Tensor,
        modulation: &StepModulation,
    ) -> Result<Tensor> {
        let len = x.shape().dims()[0];
        let x_part = ops::linear(ctx, x, &self.w_x, None)?;
        let mut hidden = ops::residual_add(ctx, &x_part, static_condition)?;
        let conv = ops::conv1d(
            ctx,
            &hidden.reshape(apxinf_core::Shape::new(vec![1, len, 1024]))?,
            &self.convpos0_w,
            self.convpos0_b.as_ref(),
            1,
            15,
            1,
            16,
        )?;
        let conv = ops::mish(ctx, &conv)?;
        let conv = ops::conv1d(
            ctx,
            &conv,
            &self.convpos2_w,
            self.convpos2_b.as_ref(),
            1,
            15,
            1,
            16,
        )?;
        let conv = ops::mish(ctx, &conv)?;
        let conv = conv.reshape(apxinf_core::Shape::new(vec![len, 1024]))?;
        hidden = ops::residual_add(ctx, &conv, &hidden)?;

        let ones = &self.norm_ones;
        let zeros = &self.norm_zeros;

        for (block_index, block) in self.blocks.iter().enumerate() {
            let step = &modulation.blocks[block_index];
            let shift_msa = step.shift_msa.clone();
            let scale_msa = step.scale_msa.clone();
            let gate_msa = step.gate_msa.clone();
            let shift_mlp = step.shift_mlp.clone();
            let scale_mlp = step.scale_mlp.clone();
            let gate_mlp = step.gate_mlp.clone();

            let normed = ops::layer_norm(ctx, &hidden, ones, zeros, 1e-6)?;
            let modulated = self.staged_affine(ctx, &normed, &scale_msa, &shift_msa)?;

            let q = block.q.apply_accumulator(ctx, &modulated)?;
            let k = block.k.apply_accumulator(ctx, &modulated)?;
            let v = block.v.apply_accumulator(ctx, &modulated)?;
            let q = q.reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                CFM_HEADS,
                CFM_HEAD_DIM,
            ]))?;
            let k = k.reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                CFM_HEADS,
                CFM_HEAD_DIM,
            ]))?;
            let v = v.reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                CFM_HEADS,
                CFM_HEAD_DIM,
            ]))?;
            let q_flat = q.reshape(apxinf_core::Shape::new(vec![len, 1024]))?;
            let k_flat = k.reshape(apxinf_core::Shape::new(vec![len, 1024]))?;
            let q = Self::rope_first_head(ctx, &q_flat, &self.rotary_cos, &self.rotary_sin)?;
            let k = Self::rope_first_head(ctx, &k_flat, &self.rotary_cos, &self.rotary_sin)?;
            let attended = ops::attention(ctx, &q, &k, &v, CFM_HEADS)?;
            let attended = attended.reshape(apxinf_core::Shape::new(vec![len, 1024]))?;
            let attended = block.o.apply_accumulator(ctx, &attended)?;
            hidden = self.staged_gate_residual(ctx, &attended, &hidden, &gate_msa)?;

            let normed = ops::layer_norm(ctx, &hidden, ones, zeros, 1e-6)?;
            let modulated = self.staged_affine(ctx, &normed, &scale_mlp, &shift_mlp)?;
            let ff = block.ff1.apply_accumulator(ctx, &modulated)?;
            let ff = ops::gelu_tanh(ctx, &ff)?;
            let ff = block.ff2.apply_accumulator(ctx, &ff)?;
            hidden = self.staged_gate_residual(ctx, &ff, &hidden, &gate_mlp)?;
        }

        let normed = ops::layer_norm(ctx, &hidden, ones, zeros, 1e-6)?;
        let modulated = self.staged_affine(
            ctx,
            &normed,
            &modulation.final_scale,
            &modulation.final_shift,
        )?;
        let velocity = self.proj_out.apply_accumulator(ctx, &modulated)?;
        Ok(velocity)
    }

    pub(crate) fn allocate_step_modulation(&self, ctx: &Context) -> Result<StepModulation> {
        let source = &self.step_modulations[0];
        let allocate = |tensor: &Tensor| ops::alloc(ctx, tensor.shape().dims());
        let mut blocks = Vec::with_capacity(source.blocks.len());
        for block in &source.blocks {
            blocks.push(BlockModulation {
                shift_msa: allocate(&block.shift_msa)?,
                scale_msa: allocate(&block.scale_msa)?,
                gate_msa: allocate(&block.gate_msa)?,
                shift_mlp: allocate(&block.shift_mlp)?,
                scale_mlp: allocate(&block.scale_mlp)?,
                gate_mlp: allocate(&block.gate_mlp)?,
            });
        }
        Ok(StepModulation {
            blocks,
            final_scale: allocate(&source.final_scale)?,
            final_shift: allocate(&source.final_shift)?,
        })
    }

    pub(crate) fn bind_step_modulation(
        &self,
        ctx: &Context,
        index: usize,
        destination: &StepModulation,
    ) -> Result<()> {
        let source = self
            .step_modulations
            .get(index)
            .ok_or_else(|| Error::Other(format!("CFM modulation index {index} exceeds 32 rows")))?;
        for (source, destination) in source.blocks.iter().zip(&destination.blocks) {
            for (source, destination) in [
                (&source.shift_msa, &destination.shift_msa),
                (&source.scale_msa, &destination.scale_msa),
                (&source.gate_msa, &destination.gate_msa),
                (&source.shift_mlp, &destination.shift_mlp),
                (&source.scale_mlp, &destination.scale_mlp),
                (&source.gate_mlp, &destination.gate_mlp),
            ] {
                apxinf_cuda_new::transfers::copy_cuda_to_cuda(ctx, source, destination)?;
            }
        }
        apxinf_cuda_new::transfers::copy_cuda_to_cuda(
            ctx,
            &source.final_scale,
            &destination.final_scale,
        )?;
        apxinf_cuda_new::transfers::copy_cuda_to_cuda(
            ctx,
            &source.final_shift,
            &destination.final_shift,
        )
    }

    pub(crate) fn prepared_velocity(
        &self,
        ctx: &Context,
        x: &Tensor,
        static_condition: &Tensor,
        modulation: &StepModulation,
    ) -> Result<Tensor> {
        self.estimator_with_modulation(ctx, x, static_condition, modulation)
    }

    fn chunk_conditioning(
        &self,
        ctx: &Context,
        features: &Tensor,
        target: &Tensor,
        prompt_mel: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let target_len = target.shape().dims()[0];
        let mu = ops::concat_rows(ctx, features, target)?;
        let text_embedding = self.text_embed(ctx, &mu)?;
        let zeros = ops::alloc(ctx, &[target_len, CFM_MEL])?;
        ops::fill_zero(ctx, &zeros)?;
        let prompt_x = ops::concat_rows(ctx, prompt_mel, &zeros)?;
        let concatenated = ops::concat_columns(ctx, &prompt_x, &text_embedding)?;
        let static_condition =
            ops::linear_bias_accumulator(ctx, &concatenated, &self.w_text, Some(&self.bias))?;
        Ok((static_condition, prompt_x, text_embedding))
    }

    fn suppress_reference(
        ctx: &Context,
        value: &Tensor,
        prompt_len: usize,
        target_len: usize,
    ) -> Result<Tensor> {
        let zeros = ops::alloc(ctx, &[prompt_len, CFM_MEL])?;
        ops::fill_zero(ctx, &zeros)?;
        ops::concat_rows(
            ctx,
            &zeros,
            &ops::slice_rows(ctx, value, prompt_len, target_len)?,
        )
    }

    pub(crate) fn prepared_chunk_inputs(
        &self,
        ctx: &Context,
        features: &Tensor,
        target: &Tensor,
        prompt_mel: &Tensor,
        noise: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (condition, _, _) = self.chunk_conditioning(ctx, features, target, prompt_mel)?;
        let zeros = ops::alloc(ctx, &[CFM_MEL])?;
        ops::fill_zero(ctx, &zeros)?;
        let x = ops::affine(ctx, noise, &self.noise_scale, &zeros)?;
        let x = Self::suppress_reference(
            ctx,
            &x,
            prompt_mel.shape().dims()[0],
            target.shape().dims()[0],
        )?;
        Ok((condition, x))
    }

    pub(crate) fn prepared_solver_step(
        &self,
        ctx: &Context,
        x: &Tensor,
        static_condition: &Tensor,
        modulation: &StepModulation,
        steps: usize,
        prompt_len: usize,
        target_len: usize,
    ) -> Result<Tensor> {
        let velocity = self.prepared_velocity(ctx, x, static_condition, modulation)?;
        let updated = ops::euler_update(ctx, x, &velocity, 1.0f32 / steps as f32)?;
        Self::suppress_reference(ctx, &updated, prompt_len, target_len)
    }

    fn run_chunk(
        &self,
        ctx: &Context,
        features: &Tensor,
        target: &Tensor,
        prompt_mel: &Tensor,
        steps: usize,
        seed: u64,
    ) -> Result<Tensor> {
        let prompt_len = prompt_mel.shape().dims()[0];
        let target_len = target.shape().dims()[0];
        let noise = ops::alloc(ctx, &[prompt_len + target_len, CFM_MEL])?;
        {
            let mut generator =
                apxinf_cuda_new::sampling::create_normal_generator(ctx, noise.clone())?;
            generator.generate(RngKey::new(seed, 0, 1))?;
        }
        self.run_chunk_with_noise(ctx, features, target, prompt_mel, steps, &noise)
    }

    fn run_chunk_with_noise(
        &self,
        ctx: &Context,
        features: &Tensor,
        target: &Tensor,
        prompt_mel: &Tensor,
        steps: usize,
        noise: &Tensor,
    ) -> Result<Tensor> {
        let prompt_len = prompt_mel.shape().dims()[0];
        let target_len = target.shape().dims()[0];
        let total = prompt_len + target_len;
        if noise.shape().dims() != [total, CFM_MEL] {
            return Err(Error::Other(format!(
                "cfm noise shape mismatch: got {:?}, expected {:?}",
                noise.shape().dims(),
                [total, CFM_MEL]
            )));
        }
        let (static_condition, prompt_x, text_embedding) =
            self.chunk_conditioning(ctx, features, target, prompt_mel)?;

        let zeros = ops::alloc(ctx, &[CFM_MEL])?;
        ops::fill_zero(ctx, &zeros)?;
        let mut x = ops::affine(ctx, noise, &self.noise_scale, &zeros)?;
        x = ops::concat_rows(
            ctx,
            &{
                let zeros = ops::alloc(ctx, &[prompt_len, CFM_MEL])?;
                ops::fill_zero(ctx, &zeros)?;
                zeros
            },
            &ops::slice_rows(ctx, &x, prompt_len, target_len)?,
        )?;

        let step = 1.0f32 / steps as f32;
        for index in 0..steps {
            let velocity = self.estimator(
                ctx,
                &x,
                &prompt_x,
                &text_embedding,
                &static_condition,
                index * 32 / steps,
            )?;
            x = ops::euler_update(ctx, &x, &velocity, step)?;
            x = ops::concat_rows(
                ctx,
                &{
                    let zeros = ops::alloc(ctx, &[prompt_len, CFM_MEL])?;
                    ops::fill_zero(ctx, &zeros)?;
                    zeros
                },
                &ops::slice_rows(ctx, &x, prompt_len, target_len)?,
            )?;
        }
        ops::slice_rows(ctx, &x, prompt_len, target_len)
    }

    /// `reference_features` and `target_features` are `[frames, 512]`;
    /// `reference_mel` is normalized `[frames, 100]`.
    pub fn synthesize(
        &self,
        ctx: &Context,
        reference_features: &Tensor,
        target_features: &Tensor,
        reference_mel: &Tensor,
        steps: usize,
        cfg_rate: f32,
        seed: u64,
    ) -> Result<Tensor> {
        if cfg_rate > 1e-5 {
            return Err(Error::Other(
                "gpt_sovits v5turbo does not support classifier-free guidance".into(),
            ));
        }
        let reference_frames = reference_mel.shape().dims()[0]
            .min(reference_features.shape().dims()[0])
            .min(REFERENCE_FRAMES);
        let prompt_mel = ops::slice_rows(
            ctx,
            reference_mel,
            reference_mel.shape().dims()[0] - reference_frames,
            reference_frames,
        )?;
        let prompt_features = ops::slice_rows(
            ctx,
            reference_features,
            reference_features.shape().dims()[0] - reference_frames,
            reference_frames,
        )?;
        let target_len = target_features.shape().dims()[0];
        let chunk = (TOTAL_CHUNK - reference_frames).min(MAX_TARGET_CHUNK);
        let original_mel = prompt_mel;
        let original_features = prompt_features;
        let mut rolling_mel = original_mel.clone();
        let mut rolling_features = original_features.clone();
        let mut results: Vec<Tensor> = Vec::new();
        let mut start = 0usize;
        let mut chunk_index = 0usize;
        while start < target_len {
            let length = chunk.min(target_len - start);
            let target = ops::slice_rows(ctx, target_features, start, length)?;
            let (prompt, features) = if chunk_index == 0 {
                (original_mel.clone(), original_features.clone())
            } else {
                let tail = ROLLING_TAIL
                    .min(reference_frames)
                    .min(rolling_mel.shape().dims()[0])
                    .min(rolling_features.shape().dims()[0]);
                let prefix = reference_frames - tail;
                let prompt = ops::concat_rows(
                    ctx,
                    &ops::slice_rows(ctx, &original_mel, 0, prefix)?,
                    &ops::slice_rows(
                        ctx,
                        &rolling_mel,
                        rolling_mel.shape().dims()[0] - tail,
                        tail,
                    )?,
                )?;
                let features = ops::concat_rows(
                    ctx,
                    &ops::slice_rows(ctx, &original_features, 0, prefix)?,
                    &ops::slice_rows(
                        ctx,
                        &rolling_features,
                        rolling_features.shape().dims()[0] - tail,
                        tail,
                    )?,
                )?;
                (prompt, features)
            };
            let generated = self.run_chunk(
                ctx,
                &features,
                &target,
                &prompt,
                steps,
                seed.wrapping_add(chunk_index as u64),
            )?;
            let generated_frames = generated.shape().dims()[0];
            rolling_mel = ops::slice_rows(
                ctx,
                &generated,
                generated_frames.saturating_sub(reference_frames),
                reference_frames.min(generated_frames),
            )?;
            rolling_features = ops::slice_rows(
                ctx,
                &target,
                length.saturating_sub(reference_frames),
                reference_frames.min(length),
            )?;
            results.push(generated);
            start += length;
            chunk_index += 1;
        }
        let mut chunks = results.into_iter();
        let mut combined = chunks.next().ok_or_else(|| {
            Error::Other("cfm synthesis requires at least one target frame".into())
        })?;
        for tensor in chunks {
            combined = ops::concat_rows(ctx, &combined, &tensor)?;
        }
        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires APXINF_GPT_CHECKPOINT and CUDA"]
    fn cached_time_modulations_match_original_projection_rounding() {
        let directory = std::path::PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let ctx = Context::new_with_autotune(0, false).unwrap();
        let mut weights = Weights::load(&directory).unwrap();
        let cfm = Cfm::load(&mut weights, &ctx).unwrap();
        cfm.validate_step_modulations(&ctx, &mut weights).unwrap();
        println!("all cached block/final modulations match original bytes");
    }
}

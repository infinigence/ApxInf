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
    time_embeddings: Vec<Tensor>,
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

pub(crate) struct CfmStepTrace {
    pub x0: Tensor,
    pub velocity: Tensor,
    pub static_condition: Tensor,
    pub text0: Tensor,
    pub estimator: Option<CfmEstimatorStageTrace>,
}

pub(crate) struct CfmEstimatorStageTrace {
    pub block_outputs: Vec<Tensor>,
    pub input_after_static: Option<Tensor>,
    pub convpos_output: Option<Tensor>,
    pub time_embedding: Option<Tensor>,
    pub q: Option<Tensor>,
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
    pub q_rope: Option<Tensor>,
    pub k_rope: Option<Tensor>,
    pub attention_raw: Option<Tensor>,
    pub attention_output: Option<Tensor>,
    pub block0_normed: Option<Tensor>,
    pub block0_shift_msa: Option<Tensor>,
    pub block0_scale_msa: Option<Tensor>,
    pub block0_gate_msa: Option<Tensor>,
    pub block0_modulated: Option<Tensor>,
    pub block0_attention: Option<Tensor>,
    pub block0_ff: Option<Tensor>,
    pub block0_output: Option<Tensor>,
    pub final_modulated: Option<Tensor>,
    pub velocity: Option<Tensor>,
}

pub(crate) struct CfmTextBlock0Trace {
    pub dwconv: Tensor,
    pub norm: Tensor,
    pub pwconv1: Tensor,
    pub gelu: Tensor,
    pub grn: Tensor,
    pub pwconv2: Tensor,
    pub residual: Tensor,
}

pub(crate) struct CfmTextStageTrace {
    pub chunk_index: usize,
    pub position_added: Tensor,
    pub block_outputs: Vec<Tensor>,
    pub block0: CfmTextBlock0Trace,
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
        let mut time_embeddings = Vec::with_capacity(32);
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
            time_embeddings.push(time);
        }
        Ok(Self {
            time_embeddings,
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
            exact(&time, &self.time_embeddings[index])?;
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

    fn time_embedding(&self, _ctx: &Context, step_index: usize) -> Result<Tensor> {
        self.time_embeddings
            .get(step_index)
            .cloned()
            .ok_or_else(|| Error::Other(format!("CFM time index {step_index} exceeds 32 rows")))
    }

    fn text_embed(&self, ctx: &Context, text: &Tensor) -> Result<Tensor> {
        self.text_embed_impl(ctx, text, false)
            .map(|(output, _)| output)
    }

    fn text_embed_with_trace(
        &self,
        ctx: &Context,
        text: &Tensor,
    ) -> Result<(Tensor, CfmTextStageTrace)> {
        let (output, trace) = self.text_embed_impl(ctx, text, true)?;
        let trace = trace.ok_or_else(|| Error::Other("CFM text stage trace missing".into()))?;
        Ok((output, trace))
    }

    pub(crate) fn text_embed_debug(
        &self,
        ctx: &Context,
        text: &Tensor,
    ) -> Result<(Tensor, CfmTextStageTrace)> {
        self.text_embed_with_trace(ctx, text)
    }

    fn text_embed_impl(
        &self,
        ctx: &Context,
        text: &Tensor,
        trace_text: bool,
    ) -> Result<(Tensor, Option<CfmTextStageTrace>)> {
        let len = text.shape().dims()[0];
        let pe = ops::slice_rows(ctx, &self.text_pe, 0, len)?;
        let mut x = ops::residual_add(ctx, text, &pe)?;
        let position_added = trace_text.then(|| x.clone());
        let mut block_outputs = Vec::with_capacity(self.text_blocks.len());
        let mut block0_dwconv = None;
        let mut block0_norm = None;
        let mut block0_pwconv1 = None;
        let mut block0_gelu = None;
        let mut block0_grn = None;
        let mut block0_pwconv2 = None;
        let mut block0_residual = None;

        for (block_index, block) in self.text_blocks.iter().enumerate() {
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
            if trace_text && block_index == 0 {
                block0_dwconv = Some(hidden.clone());
            }
            hidden = ops::layer_norm(ctx, &hidden, &block.ln_w, &block.ln_b, 1e-6)?;
            if trace_text && block_index == 0 {
                block0_norm = Some(hidden.clone());
            }
            hidden = block.pw1.apply_accumulator(ctx, &hidden)?;
            if trace_text && block_index == 0 {
                block0_pwconv1 = Some(hidden.clone());
            }
            // ConvNeXt text blocks use the exact GELU; the DiT FFN below
            // uses the separate tanh approximation.
            hidden = ops::gelu_exact(ctx, &hidden)?;
            if trace_text && block_index == 0 {
                block0_gelu = Some(hidden.clone());
            }
            hidden = ops::grn(ctx, &hidden, &block.grn_g, &block.grn_b)?;
            if trace_text && block_index == 0 {
                block0_grn = Some(hidden.clone());
            }
            hidden = block.pw2.apply_accumulator(ctx, &hidden)?;
            if trace_text && block_index == 0 {
                block0_pwconv2 = Some(hidden.clone());
            }
            x = ops::residual_add(ctx, &hidden, &residual)?;
            if trace_text {
                block_outputs.push(x.clone());
                if block_index == 0 {
                    block0_residual = Some(x.clone());
                }
            }
        }

        let trace = if trace_text {
            Some(CfmTextStageTrace {
                chunk_index: 0,
                position_added: position_added
                    .ok_or_else(|| Error::Other("CFM text position trace missing".into()))?,
                block_outputs,
                block0: CfmTextBlock0Trace {
                    dwconv: block0_dwconv
                        .ok_or_else(|| Error::Other("CFM text dwconv trace missing".into()))?,
                    norm: block0_norm
                        .ok_or_else(|| Error::Other("CFM text norm trace missing".into()))?,
                    pwconv1: block0_pwconv1
                        .ok_or_else(|| Error::Other("CFM text pwconv1 trace missing".into()))?,
                    gelu: block0_gelu
                        .ok_or_else(|| Error::Other("CFM text gelu trace missing".into()))?,
                    grn: block0_grn
                        .ok_or_else(|| Error::Other("CFM text grn trace missing".into()))?,
                    pwconv2: block0_pwconv2
                        .ok_or_else(|| Error::Other("CFM text pwconv2 trace missing".into()))?,
                    residual: block0_residual
                        .ok_or_else(|| Error::Other("CFM text residual trace missing".into()))?,
                },
            })
        } else {
            None
        };
        Ok((x, trace))
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
        trace_stages: bool,
    ) -> Result<(Tensor, Option<CfmEstimatorStageTrace>)> {
        let modulation = self.step_modulations.get(step_index).ok_or_else(|| {
            Error::Other(format!("CFM modulation index {step_index} exceeds 32 rows"))
        })?;
        self.estimator_with_modulation(
            ctx,
            x,
            static_condition,
            step_index,
            modulation,
            trace_stages,
        )
    }

    fn estimator_with_modulation(
        &self,
        ctx: &Context,
        x: &Tensor,
        static_condition: &Tensor,
        step_index: usize,
        modulation: &StepModulation,
        trace_stages: bool,
    ) -> Result<(Tensor, Option<CfmEstimatorStageTrace>)> {
        let mut debug: Option<CfmEstimatorStageTrace> =
            trace_stages.then(|| CfmEstimatorStageTrace {
                block_outputs: Vec::new(),
                input_after_static: None,
                convpos_output: None,
                time_embedding: None,
                q: None,
                k: None,
                v: None,
                q_rope: None,
                k_rope: None,
                attention_raw: None,
                attention_output: None,
                block0_normed: None,
                block0_shift_msa: None,
                block0_scale_msa: None,
                block0_gate_msa: None,
                block0_modulated: None,
                block0_attention: None,
                block0_ff: None,
                block0_output: None,
                final_modulated: None,
                velocity: None,
            });
        let len = x.shape().dims()[0];
        let x_part = ops::linear(ctx, x, &self.w_x, None)?;
        let mut hidden = ops::residual_add(ctx, &x_part, static_condition)?;
        if let Some(trace) = debug.as_mut() {
            trace.input_after_static = Some(hidden.clone());
        }
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
        if let Some(trace) = debug.as_mut() {
            trace.convpos_output = Some(conv.clone());
        }

        let time_embedding = self.time_embedding(ctx, step_index)?;
        if let Some(trace) = debug.as_mut() {
            trace.time_embedding = Some(time_embedding.clone());
        }
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
            if block_index == 0 {
                if let Some(trace) = debug.as_mut() {
                    trace.block0_normed = Some(normed.clone());
                    trace.block0_shift_msa = Some(shift_msa.clone());
                    let projected = block
                        .ada
                        .apply_accumulator(ctx, &ops::silu(ctx, &time_embedding)?)?
                        .reshape(apxinf_core::Shape::new(vec![6, CFM_DIM]))?;
                    trace.block0_scale_msa = Some(ops::slice_rows(ctx, &projected, 1, 1)?);
                    trace.block0_gate_msa = Some(gate_msa.clone());
                    trace.block0_modulated = Some(modulated.clone());
                }
            }

            let q = block.q.apply_accumulator(ctx, &modulated)?;
            let k = block.k.apply_accumulator(ctx, &modulated)?;
            let v = block.v.apply_accumulator(ctx, &modulated)?;
            if block_index == 0 {
                if let Some(trace) = debug.as_mut() {
                    trace.q = Some(q.clone());
                    trace.k = Some(k.clone());
                    trace.v = Some(v.clone());
                }
            }
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
            if block_index == 0 {
                if let Some(trace) = debug.as_mut() {
                    trace.q_rope = Some(q.clone());
                    trace.k_rope = Some(k.clone());
                }
            }
            let attended = ops::attention(ctx, &q, &k, &v, CFM_HEADS)?;
            if block_index == 0 {
                if let Some(trace) = debug.as_mut() {
                    trace.attention_raw = Some(attended.clone());
                }
            }
            let attended = attended.reshape(apxinf_core::Shape::new(vec![len, 1024]))?;
            let attended = block.o.apply_accumulator(ctx, &attended)?;
            if block_index == 0 {
                if let Some(trace) = debug.as_mut() {
                    trace.attention_output = Some(attended.clone());
                    trace.block0_attention = Some(attended.clone());
                }
            }
            hidden = self.staged_gate_residual(ctx, &attended, &hidden, &gate_msa)?;

            let normed = ops::layer_norm(ctx, &hidden, ones, zeros, 1e-6)?;
            let modulated = self.staged_affine(ctx, &normed, &scale_mlp, &shift_mlp)?;
            let ff = block.ff1.apply_accumulator(ctx, &modulated)?;
            let ff = ops::gelu_tanh(ctx, &ff)?;
            #[cfg(test)]
            let ff = if let Some(limit) = std::env::var_os("APXINF_PRIVATE_CFM_FF2_WORKSPACE") {
                ops::linear_bias_accumulator_with_workspace(
                    ctx,
                    &ff,
                    &block.ff2.w,
                    block.ff2.b.as_ref().unwrap(),
                    Some(limit.to_str().unwrap().parse().unwrap()),
                )?
            } else {
                block.ff2.apply_accumulator(ctx, &ff)?
            };
            #[cfg(not(test))]
            let ff = block.ff2.apply_accumulator(ctx, &ff)?;
            if block_index == 0 {
                if let Some(trace) = debug.as_mut() {
                    trace.block0_ff = Some(ff.clone());
                }
            }
            hidden = self.staged_gate_residual(ctx, &ff, &hidden, &gate_mlp)?;
            if let Some(trace) = debug.as_mut() {
                trace.block_outputs.push(hidden.clone());
                if block_index == 0 {
                    trace.block0_output = Some(hidden.clone());
                }
            }
        }

        let normed = ops::layer_norm(ctx, &hidden, ones, zeros, 1e-6)?;
        let modulated = self.staged_affine(
            ctx,
            &normed,
            &modulation.final_scale,
            &modulation.final_shift,
        )?;
        if let Some(trace) = debug.as_mut() {
            trace.final_modulated = Some(modulated.clone());
        }
        let velocity = self.proj_out.apply_accumulator(ctx, &modulated)?;
        if let Some(trace) = debug.as_mut() {
            trace.velocity = Some(velocity.clone());
        }
        Ok((velocity, debug))
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
        self.estimator_with_modulation(ctx, x, static_condition, 0, modulation, false)
            .map(|(velocity, _)| velocity)
    }

    fn chunk_conditioning(
        &self,
        ctx: &Context,
        features: &Tensor,
        target: &Tensor,
        prompt_mel: &Tensor,
        trace_steps: bool,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Option<CfmTextStageTrace>)> {
        let target_len = target.shape().dims()[0];
        let mu = ops::concat_rows(ctx, features, target)?;
        let (text_embedding, text_stage) = if trace_steps {
            let (embedding, stage) = self.text_embed_with_trace(ctx, &mu)?;
            (embedding, Some(stage))
        } else {
            (self.text_embed(ctx, &mu)?, None)
        };
        let zeros = ops::alloc(ctx, &[target_len, CFM_MEL])?;
        ops::fill_zero(ctx, &zeros)?;
        let prompt_x = ops::concat_rows(ctx, prompt_mel, &zeros)?;
        let concatenated = ops::concat_columns(ctx, &prompt_x, &text_embedding)?;
        let static_condition =
            ops::linear_bias_accumulator(ctx, &concatenated, &self.w_text, Some(&self.bias))?;
        Ok((static_condition, mu, prompt_x, text_embedding, text_stage))
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
        let (condition, _, _, _, _) =
            self.chunk_conditioning(ctx, features, target, prompt_mel, false)?;
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

    /// Convert one captured noise chunk from comparison-harness layout
    /// `[1, 100, frames]` (row-major) to the model's contiguous `[frames, 100]`
    /// layout. The values are the pre-temperature normal samples.
    fn captured_noise(ctx: &Context, values: &[f32]) -> Result<Tensor> {
        if values.is_empty() || values.len() % CFM_MEL != 0 {
            return Err(Error::Other(format!(
                "cfm noise must contain a positive multiple of {CFM_MEL} values, got {}",
                values.len()
            )));
        }
        let frames = values.len() / CFM_MEL;
        let mut transposed = vec![0.0f32; values.len()];
        for channel in 0..CFM_MEL {
            for frame in 0..frames {
                transposed[frame * CFM_MEL + channel] = values[channel * frames + frame];
            }
        }
        super::weights::fp16_tensor(ctx, &[frames, CFM_MEL], &transposed)
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
        self.run_chunk_with_noise(ctx, features, target, prompt_mel, steps, &noise, 0, false)
            .map(|(generated, _, _)| generated)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_chunk_with_noise(
        &self,
        ctx: &Context,
        features: &Tensor,
        target: &Tensor,
        prompt_mel: &Tensor,
        steps: usize,
        noise: &Tensor,
        chunk_index: usize,
        trace_steps: bool,
    ) -> Result<(Tensor, Vec<CfmStepTrace>, Vec<CfmTextStageTrace>)> {
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
        let (static_condition, mu, prompt_x, text_embedding, text_stage) =
            self.chunk_conditioning(ctx, features, target, prompt_mel, trace_steps)?;

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

        let mut trace = Vec::new();
        let step = 1.0f32 / steps as f32;
        for index in 0..steps {
            let (velocity, estimator_trace) = self.estimator(
                ctx,
                &x,
                &prompt_x,
                &text_embedding,
                &static_condition,
                index * 32 / steps,
                trace_steps && index == 0,
            )?;
            if trace_steps {
                trace.push(CfmStepTrace {
                    x0: x.clone(),
                    velocity: velocity.clone(),
                    static_condition: static_condition.clone(),
                    text0: mu.clone(),
                    estimator: estimator_trace,
                });
            }
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
        let text_trace = text_stage
            .map(|mut stage| {
                stage.chunk_index = chunk_index;
                stage
            })
            .into_iter()
            .collect();
        Ok((
            ops::slice_rows(ctx, &x, prompt_len, target_len)?,
            trace,
            text_trace,
        ))
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
        self.synthesize_impl(
            ctx,
            reference_features,
            target_features,
            reference_mel,
            steps,
            cfg_rate,
            CfmNoiseSource::Seed(seed),
        )
        .map(|(mel, _, _)| mel)
    }

    pub(crate) fn synthesize_with_captured_noise(
        &self,
        ctx: &Context,
        reference_features: &Tensor,
        target_features: &Tensor,
        reference_mel: &Tensor,
        steps: usize,
        cfg_rate: f32,
        noise_chunks: &[&[f32]],
    ) -> Result<(Tensor, Vec<CfmStepTrace>, Vec<CfmTextStageTrace>)> {
        self.synthesize_impl(
            ctx,
            reference_features,
            target_features,
            reference_mel,
            steps,
            cfg_rate,
            CfmNoiseSource::Captured(noise_chunks),
        )
    }

    /// Deterministic captured-noise path without diagnostic tensor retention.
    pub(crate) fn synthesize_with_captured_noise_untraced(
        &self,
        ctx: &Context,
        reference_features: &Tensor,
        target_features: &Tensor,
        reference_mel: &Tensor,
        steps: usize,
        cfg_rate: f32,
        noise_chunks: &[&[f32]],
    ) -> Result<Tensor> {
        self.synthesize_impl(
            ctx,
            reference_features,
            target_features,
            reference_mel,
            steps,
            cfg_rate,
            CfmNoiseSource::CapturedUntraced(noise_chunks),
        )
        .map(|(mel, _, _)| mel)
    }

    #[allow(clippy::too_many_arguments)]
    fn synthesize_impl(
        &self,
        ctx: &Context,
        reference_features: &Tensor,
        target_features: &Tensor,
        reference_mel: &Tensor,
        steps: usize,
        cfg_rate: f32,
        noise_source: CfmNoiseSource<'_>,
    ) -> Result<(Tensor, Vec<CfmStepTrace>, Vec<CfmTextStageTrace>)> {
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
        let mut original_mel = prompt_mel;
        let mut original_features = prompt_features;
        let mut rolling_mel = original_mel.clone();
        let mut rolling_features = original_features.clone();
        let mut results: Vec<Tensor> = Vec::new();
        let mut step_trace = Vec::new();
        let mut text_trace = Vec::new();
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
            let (generated, chunk_trace, chunk_text_trace) = match noise_source {
                CfmNoiseSource::Seed(seed) => (
                    self.run_chunk(
                        ctx,
                        &features,
                        &target,
                        &prompt,
                        steps,
                        seed + chunk_index as u64,
                    )?,
                    Vec::new(),
                    Vec::new(),
                ),
                CfmNoiseSource::Captured(chunks) | CfmNoiseSource::CapturedUntraced(chunks) => {
                    let values = chunks.get(chunk_index).ok_or_else(|| {
                        Error::Other(format!(
                            "missing captured cfm noise chunk {chunk_index}; got {} chunks",
                            chunks.len()
                        ))
                    })?;
                    let noise = Self::captured_noise(ctx, values)?;
                    let trace_steps = matches!(noise_source, CfmNoiseSource::Captured(_));
                    self.run_chunk_with_noise(
                        ctx,
                        &features,
                        &target,
                        &prompt,
                        steps,
                        &noise,
                        chunk_index,
                        trace_steps,
                    )?
                }
            };
            step_trace.extend(chunk_trace);
            text_trace.extend(chunk_text_trace);
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
        let captured_chunks = match noise_source {
            CfmNoiseSource::Captured(chunks) | CfmNoiseSource::CapturedUntraced(chunks) => {
                Some(chunks)
            }
            CfmNoiseSource::Seed(_) => None,
        };
        if let Some(chunks) = captured_chunks {
            if chunk_index != chunks.len() {
                return Err(Error::Other(format!(
                    "captured cfm noise chunk count mismatch: used {chunk_index}, got {}",
                    chunks.len()
                )));
            }
        }
        let mut combined = results.remove(0);
        for tensor in results {
            combined = ops::concat_rows(ctx, &combined, &tensor)?;
        }
        let _ = (original_mel, original_features);
        Ok((combined, step_trace, text_trace))
    }
}

#[derive(Clone, Copy)]
enum CfmNoiseSource<'a> {
    Seed(u64),
    Captured(&'a [&'a [f32]]),
    CapturedUntraced(&'a [&'a [f32]]),
}

#[cfg(test)]
mod capture_tests {
    use super::*;
    #[test]
    #[ignore = "private same-input official CFM operator diagnostics"]
    fn official_input_operator_isolation() {
        use std::{fs, path::PathBuf};
        let root = PathBuf::from(std::env::var("APXINF_CFM_OPERATION_ROOT").unwrap());
        let output = std::env::var_os("APXINF_CFM_OPERATION_OUTPUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                root.join(if std::env::var_os("APXINF_ALIGNMENT_AUTOTUNE").is_some() {
                    "native-tuned"
                } else {
                    "native-fallback"
                })
            });
        fs::create_dir_all(&output).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
        let ctx =
            Context::new_with_autotune(0, std::env::var_os("APXINF_ALIGNMENT_AUTOTUNE").is_some())
                .unwrap();
        let mut weights = Weights::load(&PathBuf::from(
            std::env::var("APXINF_GPT_CHECKPOINT").unwrap(),
        ))
        .unwrap();
        let read = |name: &str| {
            let shape = manifest[name]["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            let data = fs::read(root.join(format!("{name}.f32")))
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            fp16_tensor(&ctx, &shape, &data).unwrap()
        };
        let write = |name: &str, tensor: &Tensor| {
            let cpu = apxinf_cuda_new::transfers::to_cpu(tensor)
                .unwrap()
                .to_f32_vec()
                .unwrap();
            fs::write(
                output.join(format!("{name}.f32")),
                cpu.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            )
            .unwrap();
            println!("operation/{name}: exported {} elements", cpu.len());
        };
        let time0 = Linear::load(&mut weights, &ctx, "cfm.time0").unwrap();
        let time2 = Linear::load(&mut weights, &ctx, "cfm.time2").unwrap();
        write(
            "time0",
            &time0.apply_accumulator(&ctx, &read("time-sinus")).unwrap(),
        );
        write("time-silu", &ops::silu(&ctx, &read("time0")).unwrap());
        write(
            "time2",
            &time2.apply_accumulator(&ctx, &read("time-silu")).unwrap(),
        );
        write(
            "ada-input",
            &ops::silu(&ctx, &read("official-time")).unwrap(),
        );
        for part in ["shift_msa", "scale_msa", "gate_msa"] {
            let linear =
                Linear::load(&mut weights, &ctx, &format!("cfm.blocks.0.ada.{part}")).unwrap();
            write(
                &format!("ada-{part}"),
                &linear.apply_accumulator(&ctx, &read("ada-input")).unwrap(),
            );
        }
        for part in ["q", "k", "v", "o"] {
            let linear = Linear::load(&mut weights, &ctx, &format!("cfm.blocks.0.{part}")).unwrap();
            let input = read(if part == "o" { "o-input" } else { "qkv-input" });
            write(part, &linear.apply_accumulator(&ctx, &input).unwrap());
            write(
                &format!("{part}-staged"),
                &linear.apply(&ctx, &input).unwrap(),
            );
        }
        let cfm = Cfm::load(&mut weights, &ctx).unwrap();
        write(
            "ada-packed",
            &cfm.blocks[0]
                .ada
                .apply_accumulator(&ctx, &read("ada-input"))
                .unwrap(),
        );
        write(
            "rope-cos",
            &ops::slice_rows(&ctx, &cfm.rotary_cos, 0, 1000).unwrap(),
        );
        write(
            "rope-sin",
            &ops::slice_rows(&ctx, &cfm.rotary_sin, 0, 1000).unwrap(),
        );
        write(
            "norm",
            &ops::layer_norm(
                &ctx,
                &read("norm-input"),
                &cfm.norm_ones,
                &cfm.norm_zeros,
                1e-6,
            )
            .unwrap(),
        );
        write(
            "modulation",
            &cfm.staged_affine(
                &ctx,
                &read("modulation-norm"),
                &read("modulation-scale"),
                &read("modulation-shift"),
            )
            .unwrap(),
        );
        write(
            "rope-q",
            &Cfm::rope_first_head(
                &ctx,
                &read("rope-input-q"),
                &cfm.rotary_cos,
                &cfm.rotary_sin,
            )
            .unwrap(),
        );
        write(
            "rope-k",
            &Cfm::rope_first_head(
                &ctx,
                &read("rope-input-k"),
                &cfm.rotary_cos,
                &cfm.rotary_sin,
            )
            .unwrap(),
        );
        write(
            "attention",
            &ops::attention(
                &ctx,
                &read("attention-q"),
                &read("attention-k"),
                &read("attention-v"),
                CFM_HEADS,
            )
            .unwrap(),
        );
    }

    #[test]
    #[ignore = "private unmodified official CFM FFN and modulation diagnostic"]
    fn official_ffn_operator_isolation() {
        use std::{fs, path::PathBuf};
        let root = PathBuf::from(std::env::var("APXINF_CFM_DETAILED_ROOT").unwrap());
        let output = PathBuf::from(std::env::var("APXINF_CFM_OPERATION_OUTPUT").unwrap());
        fs::create_dir_all(&output).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
        let ctx =
            Context::new_with_autotune(0, std::env::var_os("APXINF_ALIGNMENT_AUTOTUNE").is_some())
                .unwrap();
        let mut weights = Weights::load(&PathBuf::from(
            std::env::var("APXINF_GPT_CHECKPOINT").unwrap(),
        ))
        .unwrap();
        let read = |name: &str| {
            let entry = manifest["entries"]
                .as_array()
                .unwrap()
                .iter()
                .find(|e| e["name"] == name)
                .unwrap();
            let data = fs::read(root.join(entry["path"].as_str().unwrap()))
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let width = entry["shape"]
                .as_array()
                .unwrap()
                .last()
                .unwrap()
                .as_u64()
                .unwrap() as usize;
            fp16_tensor(&ctx, &[data.len() / width, width], &data).unwrap()
        };
        let write = |name: &str, tensor: &Tensor| {
            let data = apxinf_cuda_new::transfers::to_cpu(tensor)
                .unwrap()
                .to_f32_vec()
                .unwrap();
            fs::write(
                output.join(format!("{name}.f32")),
                data.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        };
        let cfm = Cfm::load(&mut weights, &ctx).unwrap();
        let read_ncl = |name: &str| {
            let entry = manifest["entries"]
                .as_array()
                .unwrap()
                .iter()
                .find(|e| e["name"] == name)
                .unwrap();
            let shape = entry["shape"].as_array().unwrap();
            let channels = shape[1].as_u64().unwrap() as usize;
            let frames = shape[2].as_u64().unwrap() as usize;
            let data = fs::read(root.join(entry["path"].as_str().unwrap()))
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let nlc = (0..frames)
                .flat_map(|t| {
                    let data = &data;
                    (0..channels).map(move |c| data[c * frames + t])
                })
                .collect::<Vec<_>>();
            fp16_tensor(&ctx, &[1, frames, channels], &nlc).unwrap()
        };
        for (i, weight, bias) in [
            (0, &cfm.convpos0_w, cfm.convpos0_b.as_ref()),
            (2, &cfm.convpos2_w, cfm.convpos2_b.as_ref()),
        ] {
            let x = read_ncl(&format!("convpos{i}.input"));
            write(
                &format!("convpos{i}.output"),
                &ops::conv1d(&ctx, &x, weight, bias, 1, 15, 1, 16).unwrap(),
            );
            write(
                &format!("convpos{i}.mish"),
                &ops::mish(&ctx, &read_ncl(&format!("convpos{i}.output"))).unwrap(),
            );
        }
        for i in 0..CFM_DEPTH {
            let hidden = if i == 0 {
                ops::residual_add(&ctx, &read("input_after_static"), &read("convpos_output"))
                    .unwrap()
            } else {
                read(&format!("block{}.output", i - 1))
            };
            write(
                &format!("block{i}.normed"),
                &ops::layer_norm(&ctx, &hidden, &cfm.norm_ones, &cfm.norm_zeros, 1e-6).unwrap(),
            );
        }
        for (i, tensor) in cfm.time_embeddings.iter().enumerate() {
            write(&format!("time_embedding_step{i}"), tensor);
        }
        let time = ops::silu(&ctx, &read("time_embedding")).unwrap();
        for (i, block) in cfm.blocks.iter().enumerate() {
            write(
                &format!("block{i}.ada"),
                &block.ada.apply_accumulator(&ctx, &time).unwrap(),
            );
            let x = read(&format!("block{i}.ff0.input"));
            write(
                &format!("block{i}.ff0.output"),
                &block.ff1.apply_accumulator(&ctx, &x).unwrap(),
            );
            let x = read(&format!("block{i}.ff0.output"));
            write(
                &format!("block{i}.ff.gelu"),
                &ops::gelu_tanh(&ctx, &x).unwrap(),
            );
            let x = read(&format!("block{i}.ff2.input"));
            write(
                &format!("block{i}.ff2.output"),
                &block.ff2.apply_accumulator(&ctx, &x).unwrap(),
            );
        }
        write(
            "final_ada",
            &Cfm::packed_linear(
                &mut weights,
                &ctx,
                &["cfm.final.scale".into(), "cfm.final.shift".into()],
            )
            .unwrap()
            .apply_accumulator(&ctx, &time)
            .unwrap(),
        );
        let hidden = read("block21.output");
        let normed = ops::layer_norm(&ctx, &hidden, &cfm.norm_ones, &cfm.norm_zeros, 1e-6).unwrap();
        write("final_norm", &normed);
        let official_ada = read("final_ada")
            .reshape(apxinf_core::Shape::new(vec![2, CFM_DIM]))
            .unwrap();
        let scale = ops::slice_rows(&ctx, &official_ada, 0, 1)
            .unwrap()
            .reshape(apxinf_core::Shape::new(vec![CFM_DIM]))
            .unwrap();
        let shift = ops::slice_rows(&ctx, &official_ada, 1, 1)
            .unwrap()
            .reshape(apxinf_core::Shape::new(vec![CFM_DIM]))
            .unwrap();
        let scale = ops::add_bias(&ctx, &scale, &cfm.norm_ones).unwrap();
        write(
            "final_modulated_official_ada",
            &cfm.staged_affine(&ctx, &normed, &scale, &shift).unwrap(),
        );
        write(
            "final_modulated_native_ada",
            &cfm.staged_affine(
                &ctx,
                &normed,
                &cfm.step_modulations[0].final_scale,
                &cfm.step_modulations[0].final_shift,
            )
            .unwrap(),
        );
        write(
            "final_projection",
            &cfm.proj_out
                .apply_accumulator(&ctx, &read("final_projection_input"))
                .unwrap(),
        );
        println!(
            "22 packed modulations, 22 FFN projections/activations, final projection exported"
        );
    }

    #[test]
    #[ignore = "requires captured official fixtures and CUDA checkpoint"]
    fn isolated_estimator_stage_metrics() {
        use std::{fs, path::PathBuf};
        let checkpoint = PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let root = PathBuf::from(std::env::var("APXINF_GPT_CAPTURE_ROOT").unwrap());
        let stages = PathBuf::from(
            std::env::var("APXINF_CFM_PRISTINE_STAGES")
                .expect("unmodified official stage capture required"),
        );
        let raw = root.join("pair1/raw");
        let ctx = Context::new_with_autotune(0, false).unwrap();
        let mut weights = Weights::load(&checkpoint).unwrap();
        let cfm = Cfm::load(&mut weights, &ctx).unwrap();
        let values = |directory: &std::path::Path, name: &str| {
            let manifest: serde_json::Value =
                serde_json::from_slice(&fs::read(directory.join("manifest.json")).unwrap())
                    .unwrap();
            let entry = manifest["entries"]
                .as_array()
                .unwrap()
                .iter()
                .find(|entry| entry["name"] == name)
                .unwrap();
            let dims = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            let data = fs::read(directory.join(entry["path"].as_str().unwrap()))
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            (dims, data)
        };
        let nlc = |name: &str| {
            let (dims, data) = values(&raw, name);
            let (channels, frames) = (dims[1], dims[2]);
            let data = (0..frames)
                .flat_map(|t| {
                    let data = &data;
                    (0..channels).map(move |c| data[c * frames + t])
                })
                .collect::<Vec<_>>();
            fp16_tensor(&ctx, &[frames, channels], &data).unwrap()
        };
        let x = nlc("cfm_steps.0.x0");
        let cond = nlc("cfm_steps.0.cond0");
        let text = nlc("cfm_steps.0.text0");
        let (_, static_values) = values(&raw, "cfm_steps.0.static_cache.condition");
        let static_condition = fp16_tensor(&ctx, &[1000, 1024], &static_values).unwrap();
        let (_, trace) = cfm
            .estimator(&ctx, &x, &cond, &text, &static_condition, 0, true)
            .unwrap();
        let trace = trace.unwrap();
        if let Some(directory) = std::env::var_os("APXINF_CFM_STAGE_CAPTURE") {
            let directory = PathBuf::from(directory);
            fs::create_dir_all(&directory).unwrap();
            for (i, tensor) in trace.block_outputs.iter().enumerate() {
                let data = apxinf_cuda_new::transfers::to_cpu(tensor)
                    .unwrap()
                    .to_f32_vec()
                    .unwrap();
                fs::write(
                    directory.join(format!("block{i}.output.f32")),
                    data.iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            }
        }
        for (label, tensor) in [
            ("input_after_static", trace.input_after_static),
            ("convpos_output", trace.convpos_output),
            ("time_embedding", trace.time_embedding),
            ("block0.normed", trace.block0_normed),
            ("block0.modulated", trace.block0_modulated),
            ("block0.q", trace.q),
            ("block0.k", trace.k),
            ("block0.v", trace.v),
            ("q_rope", trace.q_rope),
            ("k_rope", trace.k_rope),
            ("attention_raw", trace.attention_raw),
            ("block0.attention_output", trace.attention_output),
            ("block0.ff", trace.block0_ff),
            ("block0.output", trace.block0_output),
            ("final_norm_modulated", trace.final_modulated),
            ("velocity", trace.velocity),
        ] {
            let actual = apxinf_cuda_new::transfers::to_cpu(&tensor.unwrap())
                .unwrap()
                .to_f32_vec()
                .unwrap();
            let (dims, mut expected) = values(&stages, label);
            if label == "attention_raw" {
                let original = expected.clone();
                for t in 0..dims[2] {
                    for h in 0..dims[1] {
                        for d in 0..dims[3] {
                            expected[(t * dims[1] + h) * dims[3] + d] =
                                original[(h * dims[2] + t) * dims[3] + d];
                        }
                    }
                }
            }
            assert_eq!(actual.len(), expected.len());
            let mut power = 0f64;
            let mut err = 0f64;
            let mut max = 0f32;
            let mut mismatch = 0usize;
            for (&a, &b) in actual.iter().zip(&expected) {
                power += f64::from(b).powi(2);
                err += f64::from(a - b).powi(2);
                max = max.max((a - b).abs());
                mismatch += usize::from(a != b);
            }
            println!(
                "isolated/{label}: snr_db={} max_abs={max} mismatches={mismatch}",
                10. * (power / err).log10()
            );
            if let Some(directory) = std::env::var_os("APXINF_CFM_STAGE_CAPTURE") {
                let directory = PathBuf::from(directory);
                fs::create_dir_all(&directory).unwrap();
                fs::write(
                    directory.join(format!("{label}.f32")),
                    actual
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            }
        }
    }

    #[test]
    #[ignore = "requires APXINF_GPT_CHECKPOINT and CUDA"]
    fn cached_time_modulations_match_original_projection_rounding() {
        let directory = std::path::PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let ctx = Context::new_with_autotune(0, false).unwrap();
        let mut weights = Weights::load(&directory).unwrap();
        let cfm = Cfm::load(&mut weights, &ctx).unwrap();
        cfm.validate_step_modulations(&ctx, &mut weights).unwrap();
        println!("32 time embeddings and all 22 block/final modulations match original bytes");
    }
}

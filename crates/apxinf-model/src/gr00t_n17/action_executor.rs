use std::sync::Arc;

use apxinf_core::{Backend, Error, Result, Tensor};
use apxinf_cuda::kernels::{activation, attention, elementwise, gemm, norm};
use apxinf_cuda::{CudaBackend, CudaContext};
use half::bf16;

use super::{
    AttentionWeights, CategoryMlpWeights, FeedForwardWeights, GrootN17ActionWeights,
    GrootN17Config, LayerNormWeights, LinearWeights,
};

struct StepConditioning {
    action_time: Tensor,
    block_styles: Vec<Tensor>,
    final_style: Tensor,
}

pub struct GrootN17ActionExecutor {
    config: Arc<GrootN17Config>,
    weights: Arc<GrootN17ActionWeights>,
    conditions: Vec<StepConditioning>,
    unit_norm: LayerNormWeights,
    action_rows: elementwise::RowIndices,
}

impl GrootN17ActionExecutor {
    pub fn new(
        config: Arc<GrootN17Config>,
        weights: Arc<GrootN17ActionWeights>,
        backend: &dyn Backend,
        cuda: &CudaBackend,
    ) -> Result<Self> {
        let width = config.action_width();
        let unit_norm = LayerNormWeights {
            weight: backend.to_device(&Tensor::from_bf16(
                vec![width],
                &vec![bf16::from_f32(1.0); width],
            )?)?,
            bias: backend.to_device(&Tensor::from_bf16(
                vec![width],
                &vec![bf16::from_f32(0.0); width],
            )?)?,
        };
        let action_rows = elementwise::RowIndices::new(
            cuda.device_id(),
            &(1..=config.action_horizon as u32).collect::<Vec<_>>(),
        )?;
        let mut conditions = Vec::with_capacity(config.num_inference_timesteps);
        for timestep in config.timestep_values() {
            let action_time = backend.to_device(&action_time_embedding(
                timestep,
                config.action_horizon,
                width,
            )?)?;
            let timestep_input = backend.to_device(&dit_time_embedding(timestep, 256)?)?;
            let temb = linear(cuda.context(), &timestep_input, &weights.timestep_in)?;
            let temb = activation::bias_silu_bf16(
                cuda.context(),
                &temb,
                weights.timestep_in.bias.as_ref(),
            )?;
            let temb = linear_bias(cuda.context(), &temb, &weights.timestep_out)?;
            let activated_temb = activation::silu(cuda.context(), &temb)?;
            let mut block_styles = Vec::with_capacity(weights.dit_blocks.len());
            for block in &weights.dit_blocks {
                block_styles.push(linear_bias(
                    cuda.context(),
                    &activated_temb,
                    &block.ada_norm,
                )?);
            }
            let final_style =
                linear_bias(cuda.context(), &activated_temb, &weights.final_condition)?;
            conditions.push(StepConditioning {
                action_time,
                block_styles,
                final_style,
            });
        }
        backend.synchronize()?;
        Ok(Self {
            config,
            weights,
            conditions,
            unit_norm,
            action_rows,
        })
    }

    pub fn refine_backbone(&self, ctx: &CudaContext, features: &Tensor) -> Result<Tensor> {
        let mut hidden = norm::layer_bf16(
            ctx,
            features,
            &self.weights.vlln.weight,
            &self.weights.vlln.bias,
            1e-5,
        )?;
        for block in &self.weights.vl_blocks {
            hidden = self.vl_block(ctx, &hidden, block)?;
        }
        Ok(hidden)
    }

    pub fn infer(
        &self,
        ctx: &CudaContext,
        refined_backbone: &Tensor,
        state: &Tensor,
        initial_noise: &Tensor,
        text_rows: &elementwise::RowIndices,
        image_rows: &elementwise::RowIndices,
        backend: &dyn Backend,
    ) -> Result<Tensor> {
        let state_features = category_mlp(ctx, state, &self.weights.state_encoder)?;
        let text_features = elementwise::gather_rows_bf16(ctx, refined_backbone, text_rows)?;
        let image_features = elementwise::gather_rows_bf16(ctx, refined_backbone, image_rows)?;

        let mut cross_kv = Vec::with_capacity(self.weights.dit_blocks.len());
        for (index, block) in self.weights.dit_blocks.iter().enumerate() {
            if index % 2 == 1 {
                cross_kv.push(None);
                continue;
            }
            let encoder = if index % 4 == 0 {
                &text_features
            } else {
                &image_features
            };
            let key = linear_bias(ctx, encoder, &block.attention.k)?.reshape(vec![
                encoder.shape().dims()[0],
                self.config.diffusion_model_cfg.num_attention_heads,
                self.config.diffusion_model_cfg.attention_head_dim,
            ])?;
            let value = linear_bias(ctx, encoder, &block.attention.v)?.reshape(vec![
                encoder.shape().dims()[0],
                self.config.diffusion_model_cfg.num_attention_heads,
                self.config.diffusion_model_cfg.attention_head_dim,
            ])?;
            cross_kv.push(Some((key, value)));
        }

        let mut actions = initial_noise.clone();
        for condition in &self.conditions {
            let action_base = linear_bias(ctx, &actions, &self.weights.action_encoder.w1)?;
            let encoded = backend.concat_2d(&[&action_base, &condition.action_time])?;
            let encoded = linear(ctx, &encoded, &self.weights.action_encoder.w2)?;
            let encoded = activation::bias_silu_bf16(
                ctx,
                &encoded,
                self.weights.action_encoder.w2.bias.as_ref(),
            )?;
            let action_features = linear_bias(ctx, &encoded, &self.weights.action_encoder.w3)?;
            let action_features =
                backend.add(&action_features, &self.weights.position_embedding)?;
            let mut hidden = elementwise::concat_rows_bf16(ctx, &state_features, &action_features)?;

            for (index, block) in self.weights.dit_blocks.iter().enumerate() {
                let normalized = norm::adaptive_layer_bf16(
                    ctx,
                    &hidden,
                    &condition.block_styles[index],
                    1e-5,
                    false,
                )?;
                let query = linear_bias(ctx, &normalized, &block.attention.q)?.reshape(vec![
                    hidden.shape().dims()[0],
                    self.config.diffusion_model_cfg.num_attention_heads,
                    self.config.diffusion_model_cfg.attention_head_dim,
                ])?;
                let attended = if let Some((key, value)) = &cross_kv[index] {
                    attention::cross_mha_bf16(ctx, &query, key, value)?
                } else {
                    let key = linear_bias(ctx, &normalized, &block.attention.k)?.reshape(vec![
                        hidden.shape().dims()[0],
                        self.config.diffusion_model_cfg.num_attention_heads,
                        self.config.diffusion_model_cfg.attention_head_dim,
                    ])?;
                    let value =
                        linear_bias(ctx, &normalized, &block.attention.v)?.reshape(vec![
                            hidden.shape().dims()[0],
                            self.config.diffusion_model_cfg.num_attention_heads,
                            self.config.diffusion_model_cfg.attention_head_dim,
                        ])?;
                    attention::mha_bf16(ctx, &query, &key, &value, hidden.shape().dims()[0])?
                };
                let attended =
                    attended.reshape(vec![hidden.shape().dims()[0], self.config.action_width()])?;
                let projected = linear_bias(ctx, &attended, &block.attention.output)?;
                hidden = backend.add(&hidden, &projected)?;
                let normalized = norm::layer_bf16(
                    ctx,
                    &hidden,
                    &self.unit_norm.weight,
                    &self.unit_norm.bias,
                    1e-5,
                )?;
                let ff = linear(ctx, &normalized, &block.feed_forward.input)?;
                let ff =
                    activation::bias_gelu_bf16(ctx, &ff, block.feed_forward.input.bias.as_ref())?;
                let ff = linear_bias(ctx, &ff, &block.feed_forward.output)?;
                hidden = backend.add(&hidden, &ff)?;
            }

            hidden = norm::adaptive_layer_bf16(ctx, &hidden, &condition.final_style, 1e-6, true)?;
            let decoded_input = linear_bias(ctx, &hidden, &self.weights.final_projection)?;
            let decoded_input =
                elementwise::gather_rows_bf16(ctx, &decoded_input, &self.action_rows)?;
            let velocity = category_mlp(ctx, &decoded_input, &self.weights.action_decoder)?;
            actions = elementwise::euler_update_bf16(
                ctx,
                &actions,
                &velocity,
                1.0 / self.config.num_inference_timesteps as f32,
            )?;
        }
        Ok(actions)
    }

    fn vl_block(
        &self,
        ctx: &CudaContext,
        hidden: &Tensor,
        block: &super::VlBlockWeights,
    ) -> Result<Tensor> {
        let normalized =
            norm::layer_bf16(ctx, hidden, &block.norm1.weight, &block.norm1.bias, 1e-5)?;
        let attended = self_attention(
            ctx,
            &normalized,
            &block.attention,
            self.config.vl_self_attention_cfg.num_attention_heads,
            self.config.vl_self_attention_cfg.attention_head_dim,
        )?;
        let mut hidden = apxinf_cuda::kernels::elementwise::add(ctx, hidden, &attended)?;
        let normalized =
            norm::layer_bf16(ctx, &hidden, &block.norm3.weight, &block.norm3.bias, 1e-5)?;
        let ff = feed_forward(ctx, &normalized, &block.feed_forward)?;
        hidden = apxinf_cuda::kernels::elementwise::add(ctx, &hidden, &ff)?;
        Ok(hidden)
    }
}

fn linear(ctx: &CudaContext, input: &Tensor, weights: &LinearWeights) -> Result<Tensor> {
    gemm::matmul(ctx, input, &weights.weight)
}

fn linear_bias(ctx: &CudaContext, input: &Tensor, weights: &LinearWeights) -> Result<Tensor> {
    let output = linear(ctx, input, weights)?;
    elementwise::bias_bf16(ctx, &output, weights.bias.as_ref())
}

fn category_mlp(ctx: &CudaContext, input: &Tensor, weights: &CategoryMlpWeights) -> Result<Tensor> {
    let hidden = linear(ctx, input, &weights.layer1)?;
    let hidden = activation::bias_relu_bf16(ctx, &hidden, weights.layer1.bias.as_ref())?;
    linear_bias(ctx, &hidden, &weights.layer2)
}

fn feed_forward(ctx: &CudaContext, input: &Tensor, weights: &FeedForwardWeights) -> Result<Tensor> {
    let hidden = linear(ctx, input, &weights.input)?;
    let hidden = activation::bias_gelu_bf16(ctx, &hidden, weights.input.bias.as_ref())?;
    linear_bias(ctx, &hidden, &weights.output)
}

fn self_attention(
    ctx: &CudaContext,
    input: &Tensor,
    weights: &AttentionWeights,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let tokens = input.shape().dims()[0];
    let q = linear_bias(ctx, input, &weights.q)?.reshape(vec![tokens, heads, head_dim])?;
    let k = linear_bias(ctx, input, &weights.k)?.reshape(vec![tokens, heads, head_dim])?;
    let v = linear_bias(ctx, input, &weights.v)?.reshape(vec![tokens, heads, head_dim])?;
    let attended = attention::mha_bf16(ctx, &q, &k, &v, tokens)?;
    let attended = attended.reshape(vec![tokens, heads * head_dim])?;
    linear_bias(ctx, &attended, &weights.output)
}

fn action_time_embedding(timestep: u32, rows: usize, width: usize) -> Result<Tensor> {
    let half = width / 2;
    let mut row = Vec::with_capacity(width);
    for index in 0..half {
        let frequency = (-((index as f32) * 10000.0f32.ln() / half as f32)).exp();
        row.push(bf16::from_f32((timestep as f32 * frequency).sin()));
    }
    for index in 0..half {
        let frequency = (-((index as f32) * 10000.0f32.ln() / half as f32)).exp();
        row.push(bf16::from_f32((timestep as f32 * frequency).cos()));
    }
    let values = row.repeat(rows);
    Tensor::from_bf16(vec![rows, width], &values)
}

fn dit_time_embedding(timestep: u32, width: usize) -> Result<Tensor> {
    if width % 2 != 0 || width < 4 {
        return Err(Error::Other("invalid DiT timestep embedding width".into()));
    }
    let half = width / 2;
    let denominator = (half - 1) as f32;
    let mut cosine = Vec::with_capacity(half);
    let mut sine = Vec::with_capacity(half);
    for index in 0..half {
        let frequency = (-(10000.0f32.ln()) * index as f32 / denominator).exp();
        let phase = timestep as f32 * frequency;
        cosine.push(bf16::from_f32(phase.cos()));
        sine.push(bf16::from_f32(phase.sin()));
    }
    cosine.extend(sine);
    Tensor::from_bf16(vec![1, width], &cosine)
}

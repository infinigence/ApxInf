use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, Result, Shape, Tensor};
use half::{bf16, f16};

use super::backend::{downcast_arc, ops, Context, CudaEventTimer, DeviceBuffer, RuntimeBackend};
use crate::vla::{ImageLayout, Observation, VisionObservation};

use super::{SmolVlaConfig, SmolVlaWeights};

fn embedding_lookup(
    ctx: &Context,
    table: &Tensor,
    ids: &DeviceBuffer,
    tokens: usize,
    scale: ops::EmbeddingScale,
) -> Result<Tensor> {
    let dims = table.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(
            "SmolVLA embedding table must be rank 2".into(),
        ));
    }
    let mut output = ctx.allocate_output(Shape::new(vec![tokens, dims[1]]), table.dtype())?;
    let mut args = ops::GatherArgs::new(ops::GatherSemantic::EmbeddingLookup, table, &mut output);
    args.ids = Some(ids);
    args.vocab_size = dims[0];
    args.embedding_scale = scale;
    ops::gather(ctx, args)?;
    Ok(output)
}

fn add_position(
    ctx: &Context,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    let dims = projection.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(
            "SmolVLA position projection must be rank 2".into(),
        ));
    }
    let position = match position.shape().dims() {
        [1, tokens, cols] if *tokens == tokens_per_view && *cols == dims[1] => {
            position.reshape(vec![tokens_per_view, dims[1]])?
        }
        [tokens, cols] if *tokens == tokens_per_view && *cols == dims[1] => position.clone(),
        _ => {
            return Err(Error::Other(
                "SmolVLA position embedding shape mismatch".into(),
            ))
        }
    };
    let mut output = ctx.allocate_output(projection.shape().clone(), projection.dtype())?;
    let mut args = ops::GatherArgs::new(ops::GatherSemantic::BiasPosition, projection, &mut output);
    args.bias = bias;
    args.position = Some(&position);
    args.tokens_per_view = tokens_per_view;
    ops::gather(ctx, args)?;
    Ok(output)
}

pub struct SmolVlaModel {
    backend: Arc<RuntimeBackend>,
    config: Arc<SmolVlaConfig>,
    weights: Arc<SmolVlaWeights>,
    fp16_gemm: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SmolVlaInferenceTiming {
    pub preprocess_ms: f64,
    pub prefix_embedding_ms: f64,
    pub vlm_ms: f64,
    pub action_expert_ms: f64,
    pub output_ms: f64,
}

impl SmolVlaInferenceTiming {
    pub fn as_map(&self) -> BTreeMap<String, f64> {
        let vlm_prefix_ms = self.prefix_embedding_ms + self.vlm_ms;
        BTreeMap::from([
            ("preprocess_ms".to_string(), self.preprocess_ms),
            ("prefix_embedding_ms".to_string(), self.prefix_embedding_ms),
            ("vlm_ms".to_string(), self.vlm_ms),
            ("vlm_prefix_ms".to_string(), vlm_prefix_ms),
            ("action_expert_ms".to_string(), self.action_expert_ms),
            ("output_ms".to_string(), self.output_ms),
        ])
    }
}

fn attention_rank3_shape(tensor: &Tensor, what: &str) -> Result<[usize; 3]> {
    let shape = tensor.shape().dims();
    if shape.len() != 3 || shape.contains(&0) {
        return Err(Error::Other(format!(
            "{what} requires non-empty [tokens,heads,head_dim] storage"
        )));
    }
    Ok([shape[0], shape[1], shape[2]])
}

fn split_qkv_bias(
    ctx: &Context,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = qkv.shape().dims();
    if dims.len() != 2
        || dims[0] == 0
        || heads == 0
        || head_dim == 0
        || dims[1] != 3 * heads * head_dim
    {
        return Err(Error::Other(
            "SmolVLA packed QKV split shape mismatch".into(),
        ));
    }
    let shape = Shape::new(vec![dims[0], heads, head_dim]);
    let mut query = ctx.allocate_output(shape.clone(), qkv.dtype())?;
    let mut key = ctx.allocate_output(shape.clone(), qkv.dtype())?;
    let mut value = ctx.allocate_output(shape, qkv.dtype())?;
    ops::rope(
        ctx,
        ops::RopeArgs {
            semantic: ops::RopeSemantic::SplitQkvBias,
            qkv,
            bias,
            q: &mut query,
            k: &mut key,
            v: &mut value,
            q_heads: heads,
            kv_heads: heads,
            head_dim,
            theta: 0.0,
            position_offset: 0,
            kv_output_offset: 0,
        },
    )?;
    Ok((query, key, value))
}

#[allow(clippy::too_many_arguments)]
fn split_qkv_rope(
    ctx: &Context,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let tokens = qkv
        .shape()
        .dims()
        .first()
        .copied()
        .ok_or_else(|| Error::Other("SmolVLA packed QKV must have a token dimension".into()))?;
    let mut query =
        ctx.allocate_output(Shape::new(vec![tokens, q_heads, head_dim]), qkv.dtype())?;
    let kv_shape = Shape::new(vec![tokens, kv_heads, head_dim]);
    let mut key = ctx.allocate_output(kv_shape.clone(), qkv.dtype())?;
    let mut value = ctx.allocate_output(kv_shape, qkv.dtype())?;
    ops::rope(
        ctx,
        ops::RopeArgs {
            semantic: ops::RopeSemantic::SplitQkvRope,
            qkv,
            bias,
            q: &mut query,
            k: &mut key,
            v: &mut value,
            q_heads,
            kv_heads,
            head_dim,
            theta,
            position_offset,
            kv_output_offset: 0,
        },
    )?;
    Ok((query, key, value))
}

fn dense_attention(
    ctx: &Context,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    causal: bool,
    numerics: ops::AttentionNumerics,
) -> Result<Tensor> {
    let [query_tokens, query_heads, head_dim] =
        attention_rank3_shape(query, "SmolVLA Attention query")?;
    let [key_tokens, kv_heads, key_head_dim] = attention_rank3_shape(key, "SmolVLA Attention key")?;
    if value.shape() != key.shape()
        || query.dtype() != key.dtype()
        || value.dtype() != key.dtype()
        || head_dim != key_head_dim
        || query_heads % kv_heads != 0
        || (causal && key_tokens < query_tokens)
    {
        return Err(Error::Other("SmolVLA Attention contract mismatch".into()));
    }
    let query = query.reshape(vec![1, query_tokens, query_heads, head_dim])?;
    let key = key.reshape(vec![1, key_tokens, kv_heads, head_dim])?;
    let value = value.reshape(vec![1, key_tokens, kv_heads, head_dim])?;
    let mut output = ctx.allocate_output(query.shape().clone(), query.dtype())?;
    let mut args = ops::AttentionArgs::new(&query, &key, &value, &mut output);
    if causal {
        args.mask = ops::AttentionMask::Causal;
    }
    args.numerics = numerics;
    ops::attention(ctx, args)?;
    output.reshape(vec![query_tokens, query_heads, head_dim])
}

fn cached_attention(
    ctx: &Context,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    valid_key_tokens: usize,
    causal: bool,
) -> Result<Tensor> {
    let [query_tokens, query_heads, head_dim] =
        attention_rank3_shape(query, "SmolVLA cached Attention query")?;
    let [key_capacity, kv_heads, key_head_dim] =
        attention_rank3_shape(key, "SmolVLA cached Attention key")?;
    if value.shape() != key.shape()
        || query.dtype() != key.dtype()
        || value.dtype() != key.dtype()
        || head_dim != key_head_dim
        || query_heads % kv_heads != 0
        || valid_key_tokens == 0
        || valid_key_tokens > key_capacity
        || (causal && valid_key_tokens < query_tokens)
    {
        return Err(Error::Other(
            "SmolVLA cached Attention contract mismatch".into(),
        ));
    }
    let query = query.reshape(vec![1, query_tokens, query_heads, head_dim])?;
    let key = key.reshape(vec![1, key_capacity, kv_heads, head_dim])?;
    let value = value.reshape(vec![1, key_capacity, kv_heads, head_dim])?;
    let mut output = ctx.allocate_output(query.shape().clone(), query.dtype())?;
    let mut args = ops::KvCacheAttentionArgs::new(&query, &key, &value, &mut output);
    args.valid_key_tokens = valid_key_tokens;
    if causal {
        args.mask = ops::AttentionMask::Causal;
        args.query_start = valid_key_tokens - query_tokens;
    } else {
        args.mask = ops::AttentionMask::None;
        args.query_start = 0;
    }
    args.numerics = ops::AttentionNumerics::Flexible;
    ops::kv_cache_attention(ctx, args)?;
    output.reshape(vec![query_tokens, query_heads, head_dim])
}

fn attention_row_view(ctx: &Context, input: &Tensor, start: usize, end: usize) -> Result<Tensor> {
    let [rows, heads, head_dim] = attention_rank3_shape(input, "SmolVLA Attention row view")?;
    if start >= end || end > rows {
        return Err(Error::Other("SmolVLA Attention row range mismatch".into()));
    }
    let row_bytes = heads
        .checked_mul(head_dim)
        .and_then(|elements| elements.checked_mul(input.dtype().size_in_bytes()))
        .ok_or_else(|| Error::Other("SmolVLA Attention row size overflow".into()))?;
    DeviceBuffer::from_tensor(input)
        .and_then(|buffer| buffer.view(start * row_bytes, (end - start) * row_bytes))
        .and_then(|buffer| {
            buffer.as_tensor(
                Shape::new(vec![end - start, heads, head_dim]),
                input.dtype(),
            )
        })
        .map_err(Error::Cuda)
        .and_then(|tensor| {
            if tensor.device() == Device::Cuda(ctx.device_id()) {
                Ok(tensor)
            } else {
                Err(Error::Other(
                    "SmolVLA Attention row view device mismatch".into(),
                ))
            }
        })
}

fn full_gqa(ctx: &Context, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
    match query.dtype() {
        DType::BF16 => cached_attention(ctx, query, key, value, key.shape().dims()[0], false),
        DType::F16 => dense_attention(
            ctx,
            query,
            key,
            value,
            false,
            ops::AttentionNumerics::Stable,
        ),
        dtype => Err(Error::Other(format!(
            "SmolVLA full GQA does not support {dtype}"
        ))),
    }
}

fn suffix_causal_gqa(
    ctx: &Context,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
) -> Result<Tensor> {
    match query.dtype() {
        DType::BF16 => cached_attention(ctx, query, key, value, key.shape().dims()[0], true),
        DType::F16 => dense_attention(ctx, query, key, value, true, ops::AttentionNumerics::Stable),
        dtype => Err(Error::Other(format!(
            "SmolVLA causal GQA does not support {dtype}"
        ))),
    }
}

fn prefix_gqa(
    ctx: &Context,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    shared_tokens: usize,
) -> Result<Tensor> {
    let [tokens, heads, head_dim] = attention_rank3_shape(query, "SmolVLA prefix GQA")?;
    if key.shape().dims().first().copied() != Some(tokens)
        || value.shape() != key.shape()
        || shared_tokens == 0
        || shared_tokens >= tokens
    {
        return Err(Error::Other("SmolVLA prefix GQA contract mismatch".into()));
    }
    let shared_query = attention_row_view(ctx, query, 0, shared_tokens)?;
    let shared_key = attention_row_view(ctx, key, 0, shared_tokens)?;
    let shared_value = attention_row_view(ctx, value, 0, shared_tokens)?;
    let state_query = attention_row_view(ctx, query, shared_tokens, tokens)?;
    let shared = full_gqa(ctx, &shared_query, &shared_key, &shared_value)?
        .reshape(vec![shared_tokens, heads * head_dim])?;
    let state = full_gqa(ctx, &state_query, key, value)?
        .reshape(vec![tokens - shared_tokens, heads * head_dim])?;
    ops::elementwise::concat_rows(ctx, &shared, &state)?.reshape(vec![tokens, heads, head_dim])
}

fn vision_mha(
    ctx: &Context,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    let [tokens, heads, head_dim] = attention_rank3_shape(query, "SmolVLA vision MHA")?;
    if key.shape() != query.shape()
        || value.shape() != query.shape()
        || tokens_per_view == 0
        || tokens % tokens_per_view != 0
    {
        return Err(Error::Other("SmolVLA vision MHA contract mismatch".into()));
    }
    let views = tokens / tokens_per_view;
    if query.dtype() == DType::BF16 {
        let query = query.reshape(vec![views, tokens_per_view, heads, head_dim])?;
        let key = key.reshape(vec![views, tokens_per_view, heads, head_dim])?;
        let value = value.reshape(vec![views, tokens_per_view, heads, head_dim])?;
        let mut output = ctx.allocate_output(query.shape().clone(), DType::BF16)?;
        let mut args = ops::AttentionArgs::new(&query, &key, &value, &mut output);
        args.numerics = ops::AttentionNumerics::Stable;
        ops::attention(ctx, args)?;
        return output.reshape(vec![tokens, heads, head_dim]);
    }
    if query.dtype() != DType::F16 {
        return Err(Error::Other(format!(
            "SmolVLA vision MHA does not support {}",
            query.dtype()
        )));
    }
    let mut output = None;
    for view in 0..views {
        let start = view * tokens_per_view;
        let end = start + tokens_per_view;
        let view_output = dense_attention(
            ctx,
            &attention_row_view(ctx, query, start, end)?,
            &attention_row_view(ctx, key, start, end)?,
            &attention_row_view(ctx, value, start, end)?,
            false,
            ops::AttentionNumerics::Stable,
        )?
        .reshape(vec![tokens_per_view, heads * head_dim])?;
        output = Some(match output {
            Some(rows) => ops::elementwise::concat_rows(ctx, &rows, &view_output)?,
            None => view_output,
        });
    }
    output
        .ok_or_else(|| Error::Other("SmolVLA vision MHA requires at least one view".into()))?
        .reshape(vec![tokens, heads, head_dim])
}

impl SmolVlaModel {
    pub fn new(
        backend: Arc<RuntimeBackend>,
        config: Arc<SmolVlaConfig>,
        weights: Arc<SmolVlaWeights>,
    ) -> Result<Self> {
        config.validate()?;
        if weights.vision_layers.len() != config.vision_depth
            || weights.text_layers.len() != config.language_depth
            || weights.expert_layers.len() != config.language_depth
        {
            return Err(Error::Other("SmolVLA device weight depth mismatch".into()));
        }
        let fp16_gemm = weights.patch_embedding.weight.dtype() == DType::F16;
        if fp16_gemm && weights.action_out.weight.dtype() != DType::F16 {
            return Err(Error::Other(
                "SmolVLA FP16 GEMM weights are incomplete".into(),
            ));
        }
        Ok(Self {
            backend,
            config,
            weights,
            fp16_gemm,
        })
    }

    pub(in crate::smolvla) fn backend(&self) -> &Arc<RuntimeBackend> {
        &self.backend
    }

    pub fn uses_fp16_gemm(&self) -> bool {
        self.fp16_gemm
    }

    pub(in crate::smolvla) fn dtype(&self) -> DType {
        if self.fp16_gemm {
            DType::F16
        } else {
            DType::BF16
        }
    }

    pub(in crate::smolvla) fn config(&self) -> &Arc<SmolVlaConfig> {
        &self.config
    }

    fn add_bias(&self, ctx: &Context, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
        self.bias_activation(ctx, input, Some(bias), ops::PointwiseActivation::None)
    }

    fn bias_activation(
        &self,
        ctx: &Context,
        input: &Tensor,
        bias: Option<&Tensor>,
        activation: ops::PointwiseActivation,
    ) -> Result<Tensor> {
        let mut output = ctx.allocate_output(input.shape().clone(), input.dtype())?;
        let mut args =
            ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, input, &mut output);
        args.bias = bias;
        args.activation = activation;
        ops::pointwise(ctx, args)?;
        Ok(output)
    }

    fn activation(
        &self,
        ctx: &Context,
        input: &Tensor,
        activation: ops::ElementwiseActivation,
    ) -> Result<Tensor> {
        let output = ctx.allocate_output(input.shape().clone(), input.dtype())?;
        ops::elementwise_activation(ctx, input, &output, activation)?;
        Ok(output)
    }

    fn euler_update(
        &self,
        ctx: &Context,
        state: &Tensor,
        velocity: &Tensor,
        dt: f32,
    ) -> Result<Tensor> {
        let mut output = ctx.allocate_output(state.shape().clone(), state.dtype())?;
        let mut args =
            ops::PointwiseArgs::new(ops::PointwiseSemantic::EulerUpdate, state, &mut output);
        args.secondary = Some(velocity);
        args.dt = dt;
        ops::pointwise(ctx, args)?;
        Ok(output)
    }

    fn gemm(&self, ctx: &Context, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
        if !self.fp16_gemm {
            let a = activation.shape().dims();
            let b = weight.shape().dims();
            if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
                return Err(Error::Other(format!(
                    "SmolVLA GEMM expects [M,K] @ [K,N], got {a:?} and {b:?}"
                )));
            }
            let mut output = ctx.allocate_output(Shape::new(vec![a[0], b[1]]), DType::BF16)?;
            ops::gemm::gemm(
                ctx,
                ops::gemm::GemmArgs::new(activation, weight, &mut output),
            )?;
            return Ok(output);
        }
        if activation.dtype() != DType::F16 || weight.dtype() != DType::F16 {
            return Err(Error::Other(
                format!(
                    "SmolVLA FP16 GEMM expects FP16 activation and weight, got activation {} and weight {}",
                    activation.dtype(),
                    weight.dtype()
                ),
            ));
        }
        ops::gemm::matmul(ctx, activation, weight)
    }

    pub fn infer(
        &self,
        observation: &Observation,
        noise: &Tensor,
        token_ids: &DeviceBuffer,
    ) -> Result<Tensor> {
        self.infer_with_timing(observation, noise, token_ids)
            .map(|(output, _)| output)
    }

    pub fn infer_with_timing(
        &self,
        observation: &Observation,
        noise: &Tensor,
        token_ids: &DeviceBuffer,
    ) -> Result<(Tensor, SmolVlaInferenceTiming)> {
        let ctx = self.backend.context();
        let preprocess_timer = CudaEventTimer::new()?;
        preprocess_timer.start(ctx)?;
        let patches = match &observation.vision {
            VisionObservation::Patches(patches) => {
                let expected = [
                    self.config.num_views * self.config.patches_per_view(),
                    3 * self.config.patch_size * self.config.patch_size,
                ];
                if patches.dtype() != self.dtype() || patches.shape().dims() != expected {
                    return Err(Error::Other("SmolVLA patch input mismatch".into()));
                }
                self.backend.to_device(patches)?
            }
            VisionObservation::RgbU8 { bytes, layout } => {
                let expected = self.config.num_views * 3 * self.config.image_size.pow(2);
                if bytes.len() != expected {
                    return Err(Error::Other(format!(
                        "SmolVLA expected {expected} image bytes, got {}",
                        bytes.len()
                    )));
                }
                let raw = DeviceBuffer::alloc(expected, ctx.device_id()).map_err(Error::Cuda)?;
                raw.copy_from_host(bytes).map_err(Error::Cuda)?;
                let patches = self.backend.to_device(&Tensor::zeros(
                    vec![
                        self.config.num_views * self.config.patches_per_view(),
                        3 * self.config.patch_size * self.config.patch_size,
                    ],
                    self.dtype(),
                ))?;
                ops::preprocess::preprocess(
                    ctx,
                    ops::preprocess::PreprocessArgs {
                        images: &raw,
                        patches: &patches,
                        lookup_table: None,
                        spec: ops::preprocess::PreprocessSpec {
                            layout: kernel_layout(*layout),
                            geometry: ops::preprocess::PatchGeometry::Uniform {
                                views: self.config.num_views,
                                image_height: self.config.image_size,
                                image_width: self.config.image_size,
                                patch_size: self.config.patch_size,
                                temporal_patch_size: 1,
                                merge_size: 1,
                            },
                            normalization: ops::preprocess::PatchNormalization::UnitRange,
                        },
                    },
                )?;
                patches
            }
        };
        preprocess_timer.stop(ctx)?;
        let token_count = observation.token_ids.len();
        if token_count == 0 || token_count > self.config.max_token_len {
            return Err(Error::Other(format!(
                "SmolVLA token count must be in 1..={}, got {token_count}",
                self.config.max_token_len
            )));
        }
        if noise.dtype() != self.dtype()
            || noise.shape().dims() != [self.config.action_horizon, self.config.max_action_dim]
        {
            return Err(Error::Other("SmolVLA initial latent shape mismatch".into()));
        }
        let noise = if noise.device() == Device::Cpu {
            self.backend.to_device(noise)?
        } else {
            noise.clone()
        };
        let prefix_timer = CudaEventTimer::new()?;
        prefix_timer.start(ctx)?;
        let state = self.backend.to_device(&self.prepare_state(observation)?)?;
        let prefix = self.embed_prefix(&patches, token_ids, token_count, &state)?;
        prefix_timer.stop(ctx)?;
        let vlm_timer = CudaEventTimer::new()?;
        vlm_timer.start(ctx)?;
        let prefix_states = self.encode_prefix(prefix)?;
        vlm_timer.stop(ctx)?;
        let mut state = noise;
        let dt = -1.0 / self.config.num_flow_steps as f32;
        let expert_timer = CudaEventTimer::new()?;
        expert_timer.start(ctx)?;
        let time_embeddings = self.time_embeddings()?;
        for step in 0..self.config.num_flow_steps {
            let velocity =
                self.denoise_step_with_embedding(&state, &time_embeddings[step], &prefix_states)?;
            if step == 0 {}
            state = self.euler_update(ctx, &state, &velocity, dt)?;
        }
        expert_timer.stop(ctx)?;
        let output_timer = CudaEventTimer::new()?;
        output_timer.start(ctx)?;
        let output = ops::quantization::slice_columns(ctx, &state, self.config.action_dim)?;
        output_timer.stop(ctx)?;

        self.backend.synchronize()?;
        let timing = SmolVlaInferenceTiming {
            preprocess_ms: preprocess_timer.elapsed_ms()?,
            prefix_embedding_ms: prefix_timer.elapsed_ms()?,
            vlm_ms: vlm_timer.elapsed_ms()?,
            action_expert_ms: expert_timer.elapsed_ms()?,
            output_ms: output_timer.elapsed_ms()?,
        };
        Ok((output, timing))
    }

    pub(in crate::smolvla) fn time_embeddings(&self) -> Result<Vec<Tensor>> {
        let dt = -1.0 / self.config.num_flow_steps as f32;
        (0..self.config.num_flow_steps)
            .map(|step| self.time_embedding(1.0 + step as f32 * dt))
            .collect()
    }

    pub(in crate::smolvla) fn infer_fixed(
        &self,
        raw_images: Option<&DeviceBuffer>,
        patches: &Tensor,
        state: &Tensor,
        noise: &Tensor,
        token_ids: &DeviceBuffer,
        token_count: usize,
        layout: ImageLayout,
        time_embeddings: &[Tensor],
    ) -> Result<Tensor> {
        if let Some(raw_images) = raw_images {
            ops::preprocess::preprocess(
                self.backend.context(),
                ops::preprocess::PreprocessArgs {
                    images: raw_images,
                    patches,
                    lookup_table: None,
                    spec: ops::preprocess::PreprocessSpec {
                        layout: kernel_layout(layout),
                        geometry: ops::preprocess::PatchGeometry::Uniform {
                            views: self.config.num_views,
                            image_height: self.config.image_size,
                            image_width: self.config.image_size,
                            patch_size: self.config.patch_size,
                            temporal_patch_size: 1,
                            merge_size: 1,
                        },
                        normalization: ops::preprocess::PatchNormalization::UnitRange,
                    },
                },
            )?;
        }
        self.infer_parts(
            patches,
            state,
            noise,
            token_ids,
            token_count,
            time_embeddings,
        )
    }

    fn infer_parts(
        &self,
        patches: &Tensor,
        state: &Tensor,
        noise: &Tensor,
        token_ids: &DeviceBuffer,
        token_count: usize,
        time_embeddings: &[Tensor],
    ) -> Result<Tensor> {
        if token_count == 0 || token_count > self.config.max_token_len {
            return Err(Error::Other(format!(
                "SmolVLA token count must be in 1..={}, got {token_count}",
                self.config.max_token_len
            )));
        }
        if noise.dtype() != self.dtype()
            || noise.shape().dims() != [self.config.action_horizon, self.config.max_action_dim]
        {
            return Err(Error::Other("SmolVLA initial latent shape mismatch".into()));
        }
        let ctx = self.backend.context();
        let prefix = self.embed_prefix(patches, token_ids, token_count, state)?;
        let prefix_states = self.encode_prefix(prefix)?;
        let mut state = noise.clone();
        let dt = -1.0 / self.config.num_flow_steps as f32;
        for embedding in time_embeddings {
            let velocity = self.denoise_step_with_embedding(&state, embedding, &prefix_states)?;
            state = self.euler_update(ctx, &state, &velocity, dt)?;
        }
        ops::quantization::slice_columns(ctx, &state, self.config.action_dim)
    }

    fn embed_prefix(
        &self,
        patches: &Tensor,
        token_ids: &DeviceBuffer,
        token_count: usize,
        state: &Tensor,
    ) -> Result<Tensor> {
        let ctx = self.backend.context();
        let vision = self.encode_vision(ctx, patches)?;
        let language = embedding_lookup(
            ctx,
            &self.weights.token_embedding,
            token_ids,
            token_count,
            ops::EmbeddingScale::SqrtWidth,
        )?;
        let state = self.embed_state(state)?;
        let concat_rows = ops::elementwise::concat_rows;
        let prefix = concat_rows(ctx, &vision, &language)?;
        concat_rows(ctx, &prefix, &state)
    }

    fn encode_vision(&self, ctx: &Context, patches: &Tensor) -> Result<Tensor> {
        let projection = self.gemm(ctx, patches, &self.weights.patch_embedding.weight)?;
        let mut hidden = add_position(
            ctx,
            &projection,
            self.weights.patch_embedding.bias.as_ref(),
            &self.weights.position_embedding,
            self.config.patches_per_view(),
        )?;
        for layer in &self.weights.vision_layers {
            hidden = self.vision_layer(ctx, layer, &hidden)?;
        }
        let normalized = ctx.allocate_output(hidden.shape().clone(), hidden.dtype())?;
        ops::norm::norm(
            ctx,
            ops::NormArgs::layer(
                &hidden,
                &self.weights.vision_post_norm.weight,
                self.weights.vision_post_norm.bias.as_ref().unwrap(),
                &normalized,
                self.config.layer_norm_eps,
            ),
        )?;
        let shuffled =
            ops::preprocess::pixel_shuffle_4(ctx, &normalized, self.config.patches_per_view())?;
        let projected = self.gemm(ctx, &shuffled, &self.weights.connector.weight)?;
        let scaled = ctx.allocate_output(projected.shape().clone(), projected.dtype())?;
        ops::elementwise::scale(
            ctx,
            &projected,
            &scaled,
            (self.config.language_width as f32).sqrt(),
        )?;
        Ok(scaled)
    }

    fn vision_layer(
        &self,
        ctx: &Context,
        weights: &super::weights::VisionLayer,
        input: &Tensor,
    ) -> Result<Tensor> {
        let normalized = ctx.allocate_output(input.shape().clone(), input.dtype())?;
        ops::norm::norm(
            ctx,
            ops::NormArgs::layer(
                input,
                &weights.norm1.weight,
                weights.norm1.bias.as_ref().unwrap(),
                &normalized,
                self.config.layer_norm_eps,
            ),
        )?;
        let qkv = self.gemm(ctx, &normalized, &weights.qkv.weight)?;
        let (query, key, value) = split_qkv_bias(
            ctx,
            &qkv,
            weights.qkv.bias.as_ref(),
            self.config.vision_heads,
            self.config.vision_width / self.config.vision_heads,
        )?;
        let attention = vision_mha(ctx, &query, &key, &value, self.config.patches_per_view())?
            .reshape(vec![input.shape().dims()[0], self.config.vision_width])?;
        let projected = self.gemm(ctx, &attention, &weights.attention_output.weight)?;
        let hidden = if self.fp16_gemm {
            let projected = self.add_bias(
                ctx,
                &projected,
                weights.attention_output.bias.as_ref().unwrap(),
            )?;
            self.backend.add(&projected, input)?
        } else {
            let hidden = ctx.allocate_output(projected.shape().clone(), DType::BF16)?;
            ops::residual_add(
                ctx,
                ops::ResidualAddArgs {
                    input: &projected,
                    bias: Some(weights.attention_output.bias.as_ref().unwrap()),
                    residual: input,
                    gate: None,
                    hidden: &hidden,
                    bias_order: ops::BiasOrder::BiasThenResidual,
                },
            )?;
            hidden
        };
        let normalized = ctx.allocate_output(hidden.shape().clone(), hidden.dtype())?;
        ops::norm::norm(
            ctx,
            ops::NormArgs::layer(
                &hidden,
                &weights.norm2.weight,
                weights.norm2.bias.as_ref().unwrap(),
                &normalized,
                self.config.layer_norm_eps,
            ),
        )?;
        let activated = self.gemm(ctx, &normalized, &weights.fc1.weight)?;
        let activated = self.bias_activation(
            ctx,
            &activated,
            weights.fc1.bias.as_ref(),
            ops::pointwise::PointwiseActivation::Gelu,
        )?;
        let projected = self.gemm(ctx, &activated, &weights.fc2.weight)?;
        if self.fp16_gemm {
            let projected = self.add_bias(ctx, &projected, weights.fc2.bias.as_ref().unwrap())?;
            self.backend.add(&projected, &hidden)
        } else {
            let output = ctx.allocate_output(projected.shape().clone(), DType::BF16)?;
            ops::residual_add(
                ctx,
                ops::ResidualAddArgs {
                    input: &projected,
                    bias: Some(weights.fc2.bias.as_ref().unwrap()),
                    residual: &hidden,
                    gate: None,
                    hidden: &output,
                    bias_order: ops::BiasOrder::BiasThenResidual,
                },
            )?;
            Ok(output)
        }
    }

    pub(in crate::smolvla) fn prepare_state(&self, observation: &Observation) -> Result<Tensor> {
        let state = observation.state.as_ref().ok_or_else(|| {
            Error::Other("SmolVLA observation requires proprioceptive state".into())
        })?;
        if state.dtype() != DType::F32 && state.dtype() != DType::BF16 {
            return Err(Error::Other("SmolVLA state must be F32 or BF16".into()));
        }
        if state.device() != Device::Cpu {
            return Err(Error::Other("SmolVLA state must be a CPU tensor".into()));
        }
        let values = state.to_f32_vec()?;
        if values.len() > self.config.max_state_dim {
            return Err(Error::Other(
                "SmolVLA state exceeds maximum dimension".into(),
            ));
        }
        let mut padded = vec![0.0; self.config.max_state_dim];
        padded[..values.len()].copy_from_slice(&values);
        let host = if self.fp16_gemm {
            Tensor::from_f16(
                vec![1, self.config.max_state_dim],
                &padded.into_iter().map(f16::from_f32).collect::<Vec<_>>(),
            )?
        } else {
            Tensor::from_bf16(
                vec![1, self.config.max_state_dim],
                &padded.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
            )?
        };
        Ok(host)
    }

    fn embed_state(&self, state: &Tensor) -> Result<Tensor> {
        if state.device() != Device::Cuda(self.backend.device_id())
            || state.dtype() != self.dtype()
            || state.shape().dims() != [1, self.config.max_state_dim]
        {
            return Err(Error::Other(
                "SmolVLA fixed state input shape or dtype mismatch".into(),
            ));
        }
        let projected = self.gemm(
            self.backend.context(),
            state,
            &self.weights.state_projection.weight,
        )?;
        self.add_bias(
            self.backend.context(),
            &projected,
            self.weights.state_projection.bias.as_ref().unwrap(),
        )
    }

    fn encode_prefix(&self, prefix: Tensor) -> Result<PrefixStates> {
        let ctx = self.backend.context();
        let mut hidden = prefix;
        let mut keys = Vec::with_capacity(self.config.language_depth);
        let mut values = Vec::with_capacity(self.config.language_depth);
        for layer in &self.weights.text_layers {
            let normalized = ctx.allocate_output(hidden.shape().clone(), hidden.dtype())?;
            ops::norm::norm(
                ctx,
                ops::NormArgs::rms(
                    &hidden,
                    &layer.input_norm,
                    &normalized,
                    self.config.rms_norm_eps,
                ),
            )?;
            let qkv = self.gemm(ctx, &normalized, &layer.qkv.weight)?;
            let (query, key, value) = split_qkv_rope(
                ctx,
                &qkv,
                layer.qkv.bias.as_ref(),
                self.config.language_heads,
                self.config.language_kv_heads,
                self.config.language_width / self.config.language_heads,
                self.config.rope_theta,
                0,
            )?;
            let tokens = hidden.shape().dims()[0];
            let kv_width = self.config.language_kv_heads
                * (self.config.language_width / self.config.language_heads);
            keys.push(key.clone().reshape(vec![tokens, kv_width])?);
            values.push(value.clone().reshape(vec![tokens, kv_width])?);
            let attention = prefix_gqa(ctx, &query, &key, &value, tokens - 1)?
                .reshape(vec![tokens, self.config.language_width])?;
            if keys.len() == 1 {}
            let projected = self.gemm(ctx, &attention, &layer.attention_output.weight)?;
            hidden = self.backend.add(&projected, &hidden)?;
            let mlp_output =
                self.mlp(ctx, &hidden, &layer.post_norm, &layer.gate_up, &layer.down)?;
            if keys.len() == 1 {}
            hidden = mlp_output;
        }
        let normalized = ctx.allocate_output(hidden.shape().clone(), hidden.dtype())?;
        ops::norm::norm(
            ctx,
            ops::NormArgs::rms(
                &hidden,
                &self.weights.text_norm,
                &normalized,
                self.config.rms_norm_eps,
            ),
        )?;
        let mut cross_keys = Vec::new();
        let mut cross_values = Vec::new();
        for (layer_index, layer) in self.weights.expert_layers.iter().enumerate() {
            let weights = match layer {
                super::weights::ExpertLayer::CrossAttention(weights) => weights,
                super::weights::ExpertLayer::SelfAttention(_) => continue,
            };
            let prefix_tokens = keys[layer_index].shape().dims()[0];
            let key = self.gemm(
                ctx,
                &keys[layer_index].reshape(vec![prefix_tokens, 320])?,
                &weights.key.weight,
            )?;
            let value = self.gemm(
                ctx,
                &values[layer_index].reshape(vec![prefix_tokens, 320])?,
                &weights.value.weight,
            )?;
            let kv_heads = self.config.language_kv_heads;
            cross_keys.push(key.reshape(vec![prefix_tokens, kv_heads, 64])?);
            cross_values.push(value.reshape(vec![prefix_tokens, kv_heads, 64])?);
        }
        Ok(PrefixStates {
            keys,
            values,
            cross_keys,
            cross_values,
            hidden: normalized,
        })
    }

    fn mlp(
        &self,
        ctx: &Context,
        input: &Tensor,
        norm: &Tensor,
        gate_up: &super::weights::Linear,
        down: &super::weights::Linear,
    ) -> Result<Tensor> {
        let normalized = ctx.allocate_output(input.shape().clone(), input.dtype())?;
        ops::norm::norm(
            ctx,
            ops::NormArgs::rms(input, norm, &normalized, self.config.rms_norm_eps),
        )?;
        let activated = self.gemm(ctx, &normalized, &gate_up.weight)?;
        let activated = ops::pointwise::swiglu(
            ctx,
            &activated,
            ops::pointwise::SwiGluRounding::FullPrecision,
        )?;
        let projected = self.gemm(ctx, &activated, &down.weight)?;
        self.backend.add(&projected, input)
    }

    fn denoise_step(&self, state: &Tensor, time: f32, prefix: &PrefixStates) -> Result<Tensor> {
        let embedding = self.time_embedding(time)?;
        self.denoise_step_with_embedding(state, &embedding, prefix)
    }

    fn denoise_step_with_embedding(
        &self,
        state: &Tensor,
        embedding: &Tensor,
        prefix: &PrefixStates,
    ) -> Result<Tensor> {
        let ctx = self.backend.context();
        let action = self.gemm(ctx, state, &self.weights.action_in.weight)?;
        let action = self.add_bias(ctx, &action, self.weights.action_in.bias.as_ref().unwrap())?;
        let action_time = ops::elementwise::concat_columns(ctx, &[&action, embedding])?;
        let fused = self.gemm(ctx, &action_time, &self.weights.action_time_in.weight)?;
        let fused = self.add_bias(
            ctx,
            &fused,
            self.weights.action_time_in.bias.as_ref().unwrap(),
        )?;
        let fused = if self.fp16_gemm {
            self.bias_activation(ctx, &fused, None, ops::pointwise::PointwiseActivation::Silu)?
        } else {
            self.activation(ctx, &fused, ops::ElementwiseActivation::Silu)?
        };
        let mut hidden = self.gemm(ctx, &fused, &self.weights.action_time_out.weight)?;
        hidden = self.add_bias(
            ctx,
            &hidden,
            self.weights.action_time_out.bias.as_ref().unwrap(),
        )?;
        let mut cross_layer_index = 0;
        for (layer_index, layer) in self.weights.expert_layers.iter().enumerate() {
            hidden = match layer {
                super::weights::ExpertLayer::SelfAttention(weights) => self.expert_self_layer(
                    ctx,
                    weights,
                    &hidden,
                    &prefix.keys[layer_index],
                    &prefix.values[layer_index],
                )?,
                super::weights::ExpertLayer::CrossAttention(weights) => {
                    let key = &prefix.cross_keys[cross_layer_index];
                    let value = &prefix.cross_values[cross_layer_index];
                    cross_layer_index += 1;
                    self.expert_cross_layer(ctx, weights, &hidden, key, value)?
                }
            };
        }
        let normalized = ctx.allocate_output(hidden.shape().clone(), hidden.dtype())?;
        ops::norm::norm(
            ctx,
            ops::NormArgs::rms(
                &hidden,
                &self.weights.expert_norm,
                &normalized,
                self.config.rms_norm_eps,
            ),
        )?;
        let velocity = self.gemm(ctx, &normalized, &self.weights.action_out.weight)?;
        self.add_bias(
            ctx,
            &velocity,
            self.weights.action_out.bias.as_ref().unwrap(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn expert_self_layer(
        &self,
        ctx: &Context,
        weights: &super::weights::ExpertSelfLayer,
        input: &Tensor,
        prefix_key: &Tensor,
        prefix_value: &Tensor,
    ) -> Result<Tensor> {
        let normalized = ctx.allocate_output(input.shape().clone(), input.dtype())?;
        ops::norm::norm(
            ctx,
            ops::NormArgs::rms(
                input,
                &weights.input_norm,
                &normalized,
                self.config.rms_norm_eps,
            ),
        )?;
        let qkv = self.gemm(ctx, &normalized, &weights.qkv.weight)?;
        let prefix_tokens = prefix_key.shape().dims()[0];
        let (query, key, value) = split_qkv_rope(
            ctx,
            &qkv,
            weights.qkv.bias.as_ref(),
            self.config.language_heads,
            self.config.language_kv_heads,
            64,
            self.config.rope_theta,
            prefix_tokens,
        )?;
        let input_tokens = input.shape().dims()[0];
        let kv_width = self.config.language_kv_heads * 64;
        let concat_rows = ops::elementwise::concat_rows;
        let key = concat_rows(ctx, prefix_key, &key.reshape(vec![input_tokens, kv_width])?)?;
        let value = concat_rows(
            ctx,
            prefix_value,
            &value.reshape(vec![input_tokens, kv_width])?,
        )?;
        let key = key.reshape(vec![
            key.shape().dims()[0],
            self.config.language_kv_heads,
            64,
        ])?;
        let value = value.reshape(vec![
            value.shape().dims()[0],
            self.config.language_kv_heads,
            64,
        ])?;
        let attention = suffix_causal_gqa(ctx, &query, &key, &value)?
            .reshape(vec![input_tokens, self.config.language_width])?;
        let projected = self.gemm(ctx, &attention, &weights.attention_output.weight)?;
        let hidden = self.backend.add(&projected, input)?;
        self.mlp(
            ctx,
            &hidden,
            &weights.post_norm,
            &weights.gate_up,
            &weights.down,
        )
    }

    fn expert_cross_layer(
        &self,
        ctx: &Context,
        weights: &super::weights::ExpertCrossLayer,
        input: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<Tensor> {
        let normalized = ctx.allocate_output(input.shape().clone(), input.dtype())?;
        ops::norm::norm(
            ctx,
            ops::NormArgs::rms(
                input,
                &weights.input_norm,
                &normalized,
                self.config.rms_norm_eps,
            ),
        )?;
        let query = self.gemm(ctx, &normalized, &weights.query.weight)?;
        let query = query.reshape(vec![
            input.shape().dims()[0],
            self.config.language_heads,
            64,
        ])?;
        let query = if self.fp16_gemm {
            ops::rope::apply_half_split(
                ctx,
                &query,
                self.config.language_heads,
                64,
                self.config.rope_theta,
                0,
            )?
        } else {
            ops::rope::execute_batched(
                ctx,
                &query,
                self.config.language_heads,
                64,
                self.config.rope_theta,
                0,
            )?
        };
        let attention = full_gqa(ctx, &query, &key, &value)?
            .reshape(vec![input.shape().dims()[0], self.config.language_width])?;
        let projected = self.gemm(ctx, &attention, &weights.attention_output.weight)?;
        let hidden = self.backend.add(&projected, input)?;
        self.mlp(
            ctx,
            &hidden,
            &weights.post_norm,
            &weights.gate_up,
            &weights.down,
        )
    }

    fn time_embedding(&self, time: f32) -> Result<Tensor> {
        let base = sinusoidal_time_embedding(
            time,
            self.config.expert_width,
            self.config.time_min_period,
            self.config.time_max_period,
        );
        let mut values = Vec::with_capacity(self.config.action_horizon * self.config.expert_width);
        for _ in 0..self.config.action_horizon {
            values.extend_from_slice(&base);
        }
        let host = if self.fp16_gemm {
            let row = values
                .iter()
                .map(|value| f16::from_f32(*value))
                .collect::<Vec<_>>();
            Tensor::from_f16(
                vec![self.config.action_horizon, self.config.expert_width],
                &row,
            )?
        } else {
            let row = values
                .iter()
                .map(|value| bf16::from_f32(*value))
                .collect::<Vec<_>>();
            Tensor::from_bf16(
                vec![self.config.action_horizon, self.config.expert_width],
                &row,
            )?
        };
        self.backend.to_device(&host)
    }
}

fn sinusoidal_time_embedding(
    time: f32,
    dimension: usize,
    min_period: f32,
    max_period: f32,
) -> Vec<f32> {
    assert_eq!(dimension % 2, 0);
    let half = dimension / 2;
    let mut sin = Vec::with_capacity(half);
    let mut cos = Vec::with_capacity(half);
    for index in 0..half {
        let fraction = if half == 1 {
            0.0
        } else {
            index as f64 / (half - 1) as f64
        };
        let period = min_period as f64 * (max_period as f64 / min_period as f64).powf(fraction);
        let phase = time as f64 * TAU / period;
        sin.push(phase.sin() as f32);
        cos.push(phase.cos() as f32);
    }
    sin.extend(cos);
    sin
}

pub struct PrefixStates {
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    cross_keys: Vec<Tensor>,
    cross_values: Vec<Tensor>,
    hidden: Tensor,
}

fn kernel_layout(layout: crate::vla::ImageLayout) -> ops::preprocess::ImageLayout {
    match layout {
        crate::vla::ImageLayout::Nhwc => ops::preprocess::ImageLayout::Nhwc,
        crate::vla::ImageLayout::Nchw => ops::preprocess::ImageLayout::Nchw,
    }
}

pub fn downcast_backend(backend: Arc<dyn Backend>) -> Result<Arc<RuntimeBackend>> {
    downcast_arc(backend).ok_or_else(|| Error::Other("SmolVLA is only registered for CUDA".into()))
}

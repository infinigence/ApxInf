use std::path::Path;

use apxinf_core::{Error, Result};

pub const DM05_IMAGE_TOKEN_ID: u32 = 262_144;
pub const DM05_HISTORY_PAD_TOKEN_ID: u32 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dm05LayerType {
    Sliding,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dm05RopeConfig {
    pub theta: f32,
    pub linear_factor: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dm05TextConfig {
    pub width: usize,
    pub depth: usize,
    pub mlp_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub query_pre_attn_scalar: f32,
    pub rms_norm_eps: f32,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub layer_types: Vec<Dm05LayerType>,
    pub sliding_rope: Dm05RopeConfig,
    pub full_rope: Dm05RopeConfig,
}

impl Dm05TextConfig {
    pub fn rope_for_layer(&self, layer: usize) -> Result<Dm05RopeConfig> {
        match self.layer_types.get(layer) {
            Some(Dm05LayerType::Sliding) => Ok(self.sliding_rope),
            Some(Dm05LayerType::Full) => Ok(self.full_rope),
            None => Err(Error::Other(format!(
                "DM05 layer index {layer} is outside depth {}",
                self.depth
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dm05VisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub width: usize,
    pub depth: usize,
    pub mlp_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub layer_norm_eps: f32,
    pub pooled_tokens_per_image: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dm05Config {
    pub action_dim: usize,
    pub deploy_action_dim: usize,
    pub action_horizon: usize,
    pub diffusion_steps: usize,
    pub num_views: usize,
    pub max_prefix_len: usize,
    pub image_token_id: u32,
    pub pad_token_id: u32,
    pub history_pad_token_id: u32,
    pub vision: Dm05VisionConfig,
    pub language: Dm05TextConfig,
    pub action_expert: Dm05TextConfig,
}

impl Dm05Config {
    pub const SUPPORTED_NUM_VIEWS: usize = 2;
    pub const SUPPORTED_DEPLOY_ACTION_DIM: usize = 7;
    pub const SUPPORTED_DIFFUSION_STEPS: usize = 10;
    /// OpenDM e41 LIBERO `ChatTokenization(max_length=768)` contract.
    pub const SUPPORTED_MAX_PREFIX_LEN: usize = 768;

    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("DM05 config JSON: {error}")))?;
        Self::from_value(&value)
    }

    fn from_value(value: &serde_json::Value) -> Result<Self> {
        require_string(value, "model_type", "dm05")?;
        require_string(value, "dtype", "bfloat16")?;
        let architectures = value
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| Error::Other("DM05 config.architectures must be an array".into()))?;
        if architectures.as_slice()
            != [serde_json::Value::String(
                "DM05ForConditionalGeneration".into(),
            )]
        {
            return Err(Error::Other(format!(
                "DM05 architecture mismatch: {architectures:?}"
            )));
        }

        let vlm = object(value, "vlm_config")?;
        let vision_value = object(vlm, "vision_config")?;
        let language_value = object(vlm, "text_config")?;
        let action_value = object(value, "action_config")?;

        let vision_heads = usize_field(vision_value, "num_attention_heads")?;
        let vision_width = usize_field(vision_value, "hidden_size")?;
        require_string(vision_value, "hidden_act", "gelu_pytorch_tanh")?;
        require_bool(vision_value, "vision_use_head", false)?;
        require_f32(vision_value, "attention_dropout", 0.0)?;
        if usize_field(vision_value, "num_channels")? != 3 {
            return Err(Error::Other(
                "DM05 vision_config.num_channels must be 3".into(),
            ));
        }
        let vision = Dm05VisionConfig {
            image_size: usize_field(vision_value, "image_size")?,
            patch_size: usize_field(vision_value, "patch_size")?,
            width: vision_width,
            depth: usize_field(vision_value, "num_hidden_layers")?,
            mlp_dim: usize_field(vision_value, "intermediate_size")?,
            num_heads: vision_heads,
            head_dim: vision_width
                .checked_div(vision_heads)
                .ok_or_else(|| Error::Other("DM05 vision head count must be non-zero".into()))?,
            layer_norm_eps: f32_field(vision_value, "layer_norm_eps")?,
            pooled_tokens_per_image: usize_field(vlm, "mm_tokens_per_image")?,
        };

        let config = Self {
            action_dim: usize_field(value, "action_dim")?,
            deploy_action_dim: Self::SUPPORTED_DEPLOY_ACTION_DIM,
            action_horizon: usize_field(value, "chunk_size")?,
            diffusion_steps: Self::SUPPORTED_DIFFUSION_STEPS,
            num_views: Self::SUPPORTED_NUM_VIEWS,
            max_prefix_len: Self::SUPPORTED_MAX_PREFIX_LEN,
            image_token_id: u32::try_from(usize_field(vlm, "image_token_index")?)
                .map_err(|_| Error::Other("DM05 image token ID exceeds u32".into()))?,
            pad_token_id: u32::try_from(usize_field(value, "pad_token_id")?)
                .map_err(|_| Error::Other("DM05 pad token ID exceeds u32".into()))?,
            history_pad_token_id: DM05_HISTORY_PAD_TOKEN_ID,
            vision,
            language: parse_text_config(language_value)?,
            action_expert: parse_text_config(action_value)?,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_action_horizon(mut self, action_horizon: usize) -> Result<Self> {
        self.action_horizon = action_horizon;
        self.validate()?;
        Ok(self)
    }

    pub fn patches_per_side(&self) -> usize {
        self.vision.image_size / self.vision.patch_size
    }

    pub fn patches_per_view(&self) -> usize {
        let side = self.patches_per_side();
        side * side
    }

    pub fn pooled_side(&self) -> usize {
        integer_square_root(self.vision.pooled_tokens_per_image)
    }

    pub fn pool_kernel(&self) -> usize {
        self.patches_per_side() / self.pooled_side()
    }

    pub fn image_tokens(&self) -> usize {
        self.num_views * self.vision.pooled_tokens_per_image
    }

    /// Conservative device arena size for one allocation-free native-BF16
    /// inference capture at a fixed prefix length.
    ///
    /// The graph workspace intentionally gives every intermediate a stable
    /// address. The accounting below mirrors the shipping DM05 schedule:
    /// vision, multimodal merge, all language layers, and all ten action-flow
    /// steps. Prefix attention reserves for all five legal attention segments;
    /// suffix attention additionally reserves the SM89 FA2 split-KV
    /// accumulators. This is a conservative ownership bound, not a liveness
    /// estimate: later buffer reuse may reduce it without changing graph
    /// semantics.
    pub fn cuda_graph_workspace_bytes_bf16(&self, prefix_tokens: usize) -> Result<usize> {
        self.validate()?;
        if prefix_tokens == 0 || prefix_tokens > self.max_prefix_len {
            return Err(Error::Other(format!(
                "DM05 prefix length must be in 1..={}, got {prefix_tokens}",
                self.max_prefix_len
            )));
        }

        let mut budget = CudaGraphWorkspaceBudget::default();
        let patches = self
            .num_views
            .checked_mul(self.patches_per_view())
            .ok_or_else(|| Error::Other("DM05 graph patch count overflow".into()))?;
        let image_tokens = self.image_tokens();
        let vision = self.vision.width;

        // Patch projection (GEMM + bias) and positional embedding.
        budget.bf16(&[patches, vision], 3)?;
        for _ in 0..self.vision.depth {
            // Norms, Q/K/V projections, attention output, output projection,
            // residuals, and the MLP down projection all use vision width.
            budget.bf16(&[patches, vision], 15)?;
            // FC1 projection + bias, GELU, FC2 input projection pair.
            budget.bf16(&[patches, self.vision.mlp_dim], 3)?;
            // FA2 retains one FP32 log-sum-exp value per query/head.
            budget.f32(&[patches, self.vision.num_heads], 1)?;
        }
        budget.bf16(&[patches, vision], 1)?; // post-vision norm
        budget.bf16(&[image_tokens, vision], 2)?; // pool output + projector RMS
        budget.bf16(&[image_tokens, self.language.width], 1)?; // projector

        // Token embedding plus at most four cumulative concatenation outputs
        // for the five [text, image, text, image, text] pieces.
        budget.bf16(&[prefix_tokens, self.language.width], 5)?;

        let language_query = self
            .language
            .num_heads
            .checked_mul(self.language.head_dim)
            .ok_or_else(|| Error::Other("DM05 graph language query width overflow".into()))?;
        let language_kv = self
            .language
            .num_kv_heads
            .checked_mul(self.language.head_dim)
            .ok_or_else(|| Error::Other("DM05 graph language KV width overflow".into()))?;
        for layer in 0..self.language.depth {
            budget.bf16(&[prefix_tokens, self.language.width], 1)?; // input RMS
            budget.bf16(&[prefix_tokens, language_query], 4)?; // Q proj/bias + norm/RoPE
            budget.bf16(&[prefix_tokens, language_kv], 6)?; // K/V proj/bias + K norm/RoPE
            if layer + 1 < self.language.depth {
                // Segment outputs plus four conservative cumulative concats.
                budget.bf16(&[prefix_tokens, language_query], 5)?;
                budget.f32(&[prefix_tokens, self.language.num_heads], 1)?;
                // Attention output projection, four RMS/residual tensors, and
                // MLP down projection/bias.
                budget.bf16(&[prefix_tokens, self.language.width], 9)?;
                // Gate projection/bias + GELU, up projection/bias, and product.
                budget.bf16(&[prefix_tokens, self.language.mlp_dim], 6)?;
            }
        }

        let horizon = self.action_horizon;
        let action_query = self
            .action_expert
            .num_heads
            .checked_mul(self.action_expert.head_dim)
            .ok_or_else(|| Error::Other("DM05 graph action query width overflow".into()))?;
        let action_kv = self
            .action_expert
            .num_kv_heads
            .checked_mul(self.action_expert.head_dim)
            .ok_or_else(|| Error::Other("DM05 graph action KV width overflow".into()))?;
        let suffix_key_tokens = prefix_tokens
            .checked_add(horizon)
            .ok_or_else(|| Error::Other("DM05 graph suffix key length overflow".into()))?;
        // SM89 FA2 uses 64-token blocks for head_dim=256. Reserving split-KV
        // scratch remains safe when the runtime override disables split-KV.
        let suffix_splits = suffix_key_tokens.div_ceil(64).min(128);
        for _ in 0..self.diffusion_steps {
            // Action mask, velocity projection/bias, and two-stage Euler.
            budget.bf16(&[horizon, self.action_dim], 5)?;
            // Action input projection/bias and final adaptive RMS.
            budget.bf16(&[horizon, self.action_expert.width], 3)?;
            for _ in 0..self.action_expert.depth {
                budget.bf16(&[horizon, self.action_expert.width], 12)?;
                budget.bf16(&[horizon, action_query], 5)?;
                budget.bf16(&[horizon, action_kv], 6)?;
                // Stable combined prefix+suffix K/V buffers.
                budget.bf16(&[suffix_key_tokens, action_kv], 2)?;
                budget.bf16(&[horizon, self.action_expert.mlp_dim], 6)?;
                budget.f32(&[horizon, self.action_expert.num_heads], 1)?;
                budget.f32(&[suffix_splits, horizon, self.action_expert.num_heads], 1)?;
                budget.f32(&[suffix_splits, horizon, action_query], 1)?;
            }
        }

        // Covers alignment/accounting drift in model-neutral wrappers while
        // keeping allocation failure explicit on the fixed 24-GiB target.
        budget.bytes(64 * 1024 * 1024, 1)?;
        budget.finish()
    }

    pub fn validate_prefix_tokens(&self, token_ids: &[u32]) -> Result<[(usize, usize); 2]> {
        if token_ids.is_empty() || token_ids.len() > self.max_prefix_len {
            return Err(Error::Other(format!(
                "DM05 prefix token count must be in 1..={}, got {}",
                self.max_prefix_len,
                token_ids.len()
            )));
        }
        if token_ids.contains(&self.pad_token_id) || token_ids.contains(&self.history_pad_token_id)
        {
            return Err(Error::Other(
                "DM05 native LIBERO profile does not support padding or history tokens".into(),
            ));
        }
        if let Some(token) = token_ids
            .iter()
            .find(|token| (**token as usize) >= self.language.vocab_size)
        {
            return Err(Error::Other(format!(
                "DM05 token ID {token} is outside vocabulary {}",
                self.language.vocab_size
            )));
        }

        let mut runs = Vec::new();
        let mut index = 0;
        while index < token_ids.len() {
            if token_ids[index] != self.image_token_id {
                index += 1;
                continue;
            }
            let start = index;
            while index < token_ids.len() && token_ids[index] == self.image_token_id {
                index += 1;
            }
            runs.push((start, index));
        }
        if runs.len() != self.num_views
            || runs
                .iter()
                .any(|(start, end)| end - start != self.vision.pooled_tokens_per_image)
        {
            return Err(Error::Other(format!(
                "DM05 requires exactly {} contiguous image-token runs of length {}, got {runs:?}",
                self.num_views, self.vision.pooled_tokens_per_image
            )));
        }
        Ok([runs[0], runs[1]])
    }

    pub fn validate(&self) -> Result<()> {
        if self.action_dim != 32 {
            return Err(Error::Other(format!(
                "DM05 internal action_dim must be 32, got {}",
                self.action_dim
            )));
        }
        if self.action_horizon == 0 || self.action_horizon > 50 {
            return Err(Error::Other(format!(
                "DM05 action_horizon must be in 1..=50, got {}",
                self.action_horizon
            )));
        }
        if self.diffusion_steps != 10 || self.num_views != 2 {
            return Err(Error::Other(
                "DM05 native profile requires two views and ten diffusion steps".into(),
            ));
        }
        let vision = self.vision;
        if vision.image_size != 448
            || vision.patch_size != 14
            || vision.width != 1152
            || vision.depth != 27
            || vision.mlp_dim != 4304
            || vision.num_heads != 16
            || vision.head_dim != 72
            || vision.pooled_tokens_per_image != 256
            || vision.layer_norm_eps != 1e-6
        {
            return Err(Error::Other(format!(
                "unsupported DM05 vision contract: {vision:?}"
            )));
        }
        if self.patches_per_side() != 32 || self.pooled_side() != 16 || self.pool_kernel() != 2 {
            return Err(Error::Other(
                "DM05 projector requires an exact 32x32 to 16x16 pool".into(),
            ));
        }
        validate_text_config("language", &self.language, 2560, 10_240)?;
        validate_text_config("action", &self.action_expert, 1024, 4096)?;
        if self.language.depth != self.action_expert.depth
            || self.language.num_heads != self.action_expert.num_heads
            || self.language.num_kv_heads != self.action_expert.num_kv_heads
            || self.language.head_dim != self.action_expert.head_dim
            || self.language.layer_types != self.action_expert.layer_types
            || self.language.sliding_rope != self.action_expert.sliding_rope
            || self.language.full_rope != self.action_expert.full_rope
        {
            return Err(Error::Other(
                "DM05 language/action cache geometry and RoPE must match".into(),
            ));
        }
        if self.image_token_id != DM05_IMAGE_TOKEN_ID
            || self.language.vocab_size != 262_208
            || self.action_expert.vocab_size != 262_208
        {
            return Err(Error::Other(
                "unsupported DM05 vocabulary or image token identity".into(),
            ));
        }
        Ok(())
    }
}

const CUDA_GRAPH_WORKSPACE_ALIGNMENT: u128 = 256;

#[derive(Default)]
struct CudaGraphWorkspaceBudget {
    bytes: u128,
}

impl CudaGraphWorkspaceBudget {
    fn bf16(&mut self, dimensions: &[usize], allocations: usize) -> Result<()> {
        self.tensor(dimensions, 2, allocations)
    }

    fn f32(&mut self, dimensions: &[usize], allocations: usize) -> Result<()> {
        self.tensor(dimensions, 4, allocations)
    }

    fn tensor(
        &mut self,
        dimensions: &[usize],
        element_bytes: usize,
        allocations: usize,
    ) -> Result<()> {
        let elements = dimensions.iter().try_fold(1u128, |product, dimension| {
            product
                .checked_mul(*dimension as u128)
                .ok_or_else(|| Error::Other("DM05 graph tensor size overflow".into()))
        })?;
        let bytes = elements
            .checked_mul(element_bytes as u128)
            .ok_or_else(|| Error::Other("DM05 graph tensor byte size overflow".into()))?;
        self.bytes(bytes, allocations)
    }

    fn bytes(&mut self, bytes: u128, allocations: usize) -> Result<()> {
        for _ in 0..allocations {
            self.bytes = self
                .bytes
                .checked_add(CUDA_GRAPH_WORKSPACE_ALIGNMENT - 1)
                .ok_or_else(|| Error::Other("DM05 graph workspace alignment overflow".into()))?
                & !(CUDA_GRAPH_WORKSPACE_ALIGNMENT - 1);
            self.bytes = self
                .bytes
                .checked_add(bytes)
                .ok_or_else(|| Error::Other("DM05 graph workspace size overflow".into()))?;
        }
        Ok(())
    }

    fn finish(self) -> Result<usize> {
        usize::try_from(self.bytes)
            .map_err(|_| Error::Other("DM05 graph workspace exceeds address space".into()))
    }
}

fn parse_text_config(value: &serde_json::Value) -> Result<Dm05TextConfig> {
    require_string(value, "hidden_activation", "gelu_pytorch_tanh")?;
    require_bool(value, "attention_bias", false)?;
    require_bool(value, "use_bidirectional_attention", false)?;
    require_f32(value, "attention_dropout", 0.0)?;
    let layer_types = value
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::Other("DM05 layer_types must be an array".into()))?
        .iter()
        .map(|item| match item.as_str() {
            Some("sliding_attention") => Ok(Dm05LayerType::Sliding),
            Some("full_attention") => Ok(Dm05LayerType::Full),
            other => Err(Error::Other(format!(
                "unsupported DM05 layer type {other:?}"
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    let ropes = object(value, "rope_parameters")?;
    let sliding = object(ropes, "sliding_attention")?;
    let full = object(ropes, "full_attention")?;
    require_string(sliding, "rope_type", "default")?;
    require_string(full, "rope_type", "linear")?;
    Ok(Dm05TextConfig {
        width: usize_field(value, "hidden_size")?,
        depth: usize_field(value, "num_hidden_layers")?,
        mlp_dim: usize_field(value, "intermediate_size")?,
        num_heads: usize_field(value, "num_attention_heads")?,
        num_kv_heads: usize_field(value, "num_key_value_heads")?,
        head_dim: usize_field(value, "head_dim")?,
        query_pre_attn_scalar: f32_field(value, "query_pre_attn_scalar")?,
        rms_norm_eps: f32_field(value, "rms_norm_eps")?,
        sliding_window: usize_field(value, "sliding_window")?,
        max_position_embeddings: usize_field(value, "max_position_embeddings")?,
        vocab_size: usize_field(value, "vocab_size")?,
        layer_types,
        sliding_rope: Dm05RopeConfig {
            theta: f32_field(sliding, "rope_theta")?,
            linear_factor: 1.0,
        },
        full_rope: Dm05RopeConfig {
            theta: f32_field(full, "rope_theta")?,
            linear_factor: f32_field(full, "factor")?,
        },
    })
}

fn validate_text_config(
    label: &str,
    config: &Dm05TextConfig,
    width: usize,
    mlp_dim: usize,
) -> Result<()> {
    let expected_full = [5usize, 11, 17, 23, 29];
    let observed_full = config
        .layer_types
        .iter()
        .enumerate()
        .filter_map(|(index, kind)| (*kind == Dm05LayerType::Full).then_some(index))
        .collect::<Vec<_>>();
    if config.width != width
        || config.depth != 34
        || config.mlp_dim != mlp_dim
        || config.num_heads != 8
        || config.num_kv_heads != 4
        || config.head_dim != 256
        || config.query_pre_attn_scalar != 256.0
        || config.rms_norm_eps != 1e-6
        || config.sliding_window != 4096
        || config.max_position_embeddings != 131_072
        || config.layer_types.len() != config.depth
        || observed_full != expected_full
        || config.sliding_rope
            != (Dm05RopeConfig {
                theta: 10_000.0,
                linear_factor: 1.0,
            })
        || config.full_rope
            != (Dm05RopeConfig {
                theta: 1_000_000.0,
                linear_factor: 8.0,
            })
    {
        return Err(Error::Other(format!(
            "unsupported DM05 {label} config: {config:?}"
        )));
    }
    Ok(())
}

fn object<'a>(value: &'a serde_json::Value, name: &str) -> Result<&'a serde_json::Value> {
    let field = value
        .get(name)
        .ok_or_else(|| Error::Other(format!("DM05 config is missing {name}")))?;
    if !field.is_object() {
        return Err(Error::Other(format!(
            "DM05 config.{name} must be an object"
        )));
    }
    Ok(field)
}

fn usize_field(value: &serde_json::Value, name: &str) -> Result<usize> {
    value
        .get(name)
        .and_then(serde_json::Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| Error::Other(format!("DM05 config.{name} must be an unsigned integer")))
}

fn f32_field(value: &serde_json::Value, name: &str) -> Result<f32> {
    value
        .get(name)
        .and_then(serde_json::Value::as_f64)
        .map(|number| number as f32)
        .filter(|number| number.is_finite())
        .ok_or_else(|| Error::Other(format!("DM05 config.{name} must be finite")))
}

fn require_string(value: &serde_json::Value, name: &str, expected: &str) -> Result<()> {
    let actual = value.get(name).and_then(serde_json::Value::as_str);
    if actual != Some(expected) {
        return Err(Error::Other(format!(
            "DM05 config.{name} must be {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn require_bool(value: &serde_json::Value, name: &str, expected: bool) -> Result<()> {
    let actual = value.get(name).and_then(serde_json::Value::as_bool);
    if actual != Some(expected) {
        return Err(Error::Other(format!(
            "DM05 config.{name} must be {expected}, got {actual:?}"
        )));
    }
    Ok(())
}

fn require_f32(value: &serde_json::Value, name: &str, expected: f32) -> Result<()> {
    let actual = value.get(name).and_then(serde_json::Value::as_f64);
    if actual.map(|number| number as f32) != Some(expected) {
        return Err(Error::Other(format!(
            "DM05 config.{name} must be {expected}, got {actual:?}"
        )));
    }
    Ok(())
}

fn integer_square_root(value: usize) -> usize {
    (value as f64).sqrt() as usize
}

#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;
    use serde_json::json;

    fn layer_types() -> Vec<&'static str> {
        (0..34)
            .map(|index| {
                if [5, 11, 17, 23, 29].contains(&index) {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect()
    }

    fn text(width: usize, mlp: usize) -> serde_json::Value {
        json!({
            "hidden_size": width,
            "num_hidden_layers": 34,
            "intermediate_size": mlp,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "query_pre_attn_scalar": 256,
            "rms_norm_eps": 1e-6,
            "hidden_activation":"gelu_pytorch_tanh",
            "attention_bias":false,
            "attention_dropout":0,
            "use_bidirectional_attention":false,
            "sliding_window": 4096,
            "max_position_embeddings": 131072,
            "vocab_size": 262208,
            "layer_types": layer_types(),
            "rope_parameters": {
                "sliding_attention": {"rope_type":"default", "rope_theta":10000},
                "full_attention": {"rope_type":"linear", "rope_theta":1000000, "factor":8}
            }
        })
    }

    fn config() -> serde_json::Value {
        json!({
            "model_type":"dm05",
            "dtype":"bfloat16",
            "architectures":["DM05ForConditionalGeneration"],
            "action_dim":32,
            "chunk_size":50,
            "pad_token_id":0,
            "vlm_config": {
                "image_token_index":262144,
                "mm_tokens_per_image":256,
                "vision_config": {
                    "image_size":448,
                    "patch_size":14,
                    "hidden_size":1152,
                    "num_hidden_layers":27,
                    "intermediate_size":4304,
                    "num_attention_heads":16,
                    "layer_norm_eps":1e-6,
                    "num_channels":3,
                    "hidden_act":"gelu_pytorch_tanh",
                    "attention_dropout":0,
                    "vision_use_head":false
                },
                "text_config":text(2560,10240)
            },
            "action_config":text(1024,4096)
        })
    }

    pub(crate) fn exact_config() -> Dm05Config {
        Dm05Config::from_value(&config()).unwrap()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parses_exact_checkpoint_and_applies_horizon_override() {
            let parsed = Dm05Config::from_value(&config()).unwrap();
            assert_eq!(parsed.action_horizon, 50);
            assert_eq!(parsed.patches_per_view(), 1024);
            assert_eq!(parsed.image_tokens(), 512);
            assert_eq!(parsed.pool_kernel(), 2);
            assert_eq!(parsed.max_prefix_len, 768);
            assert_eq!(parsed.with_action_horizon(10).unwrap().action_horizon, 10);
        }

        #[test]
        fn validates_two_exact_image_runs_and_rejects_padding() {
            let parsed = Dm05Config::from_value(&config()).unwrap();
            let mut ids = vec![2; 564];
            ids[37..293].fill(DM05_IMAGE_TOKEN_ID);
            ids[301..557].fill(DM05_IMAGE_TOKEN_ID);
            assert_eq!(
                parsed.validate_prefix_tokens(&ids).unwrap(),
                [(37, 293), (301, 557)]
            );
            ids[0] = 0;
            assert!(parsed.validate_prefix_tokens(&ids).is_err());

            ids[0] = parsed.language.vocab_size as u32;
            assert!(parsed.validate_prefix_tokens(&ids).is_err());
        }

        #[test]
        fn same_length_prefixes_can_have_distinct_graph_layouts() {
            let parsed = exact_config();
            let mut first = vec![2; 564];
            first[37..293].fill(DM05_IMAGE_TOKEN_ID);
            first[301..557].fill(DM05_IMAGE_TOKEN_ID);
            let mut shifted = vec![2; 564];
            shifted[36..292].fill(DM05_IMAGE_TOKEN_ID);
            shifted[300..556].fill(DM05_IMAGE_TOKEN_ID);

            assert_ne!(
                parsed.validate_prefix_tokens(&first).unwrap(),
                parsed.validate_prefix_tokens(&shifted).unwrap()
            );
        }

        #[test]
        fn rejects_wrong_gqa_or_rope_contract() {
            let mut value = config();
            value["action_config"]["num_key_value_heads"] = json!(1);
            assert!(Dm05Config::from_value(&value).is_err());

            let mut value = config();
            value["vlm_config"]["text_config"]["rope_parameters"]["full_attention"]["factor"] =
                json!(4);
            assert!(Dm05Config::from_value(&value).is_err());
        }

        #[test]
        fn graph_workspace_is_checked_and_monotonic_for_qualification_shapes() {
            let config = exact_config().with_action_horizon(10).unwrap();
            assert!(config.cuda_graph_workspace_bytes_bf16(0).is_err());
            assert!(config
                .cuda_graph_workspace_bytes_bf16(config.max_prefix_len + 1)
                .is_err());

            let frame_zero = config.cuda_graph_workspace_bytes_bf16(557).unwrap();
            let frozen = config.cuda_graph_workspace_bytes_bf16(564).unwrap();
            let maximum = config
                .cuda_graph_workspace_bytes_bf16(config.max_prefix_len)
                .unwrap();
            assert!(frame_zero > 64 * 1024 * 1024);
            assert!(frozen >= frame_zero);
            assert!(maximum >= frozen);
            // The fixed 4090 profile must leave room for the pinned BF16
            // checkpoint and CUDA runtime inside 24 GiB.
            assert!(maximum < 12 * 1024 * 1024 * 1024usize);
        }
    }
}

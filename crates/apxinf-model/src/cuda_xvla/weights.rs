//! Device-resident DaViT weights for the X-VLA vision tower.
//!
//! Every linear matrix is stored row-major `[input, output]` (checkpoint
//! `[output, input]` transposed at load time), dense convolution weights are
//! flattened in im2col tap order `[in_channels * kh * kw, out_channels]`, and
//! depthwise weights are `[channels, 9]` with `ky * 3 + kx` tap order.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::cuda_xvla::backend::kernels::xvla_ops::RowGatherPlan;
use crate::cuda_xvla::backend::DeviceBuffer;

#[derive(Debug)]
pub struct LayerNormWeights {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct LinearWeights {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct DepthwiseWeights {
    /// `[channels, 9]` in `ky * 3 + kx` tap order.
    pub weight: Tensor,
    pub bias: Tensor,
}

/// One DaViT block. Spatial and channel stages share this layout; the stage
/// plan decides which attention kernel runs.
#[derive(Debug)]
pub struct VisionBlockWeights {
    pub conv1: DepthwiseWeights,
    pub norm_attn: LayerNormWeights,
    pub qkv: LinearWeights,
    pub proj: LinearWeights,
    pub conv2: DepthwiseWeights,
    pub norm_ffn: LayerNormWeights,
    pub fc1: LinearWeights,
    pub fc2: LinearWeights,
}

/// Host-side window partition tables (device index buffers) for one stage.
pub struct StagePlan {
    pub height: usize,
    pub width: usize,
    pub window: usize,
    pub windows: usize,
    /// Padded window partition: `windows * window²` rows gathered from a
    /// `tokens + 1` matrix whose trailing row is all zeros.
    pub partition: RowGatherPlan,
    /// Inverse permutation back to `tokens` rows.
    pub reverse: RowGatherPlan,
}

pub struct VisionStageWeights {
    pub pre_norm: bool,
    pub channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub heads: usize,
    pub groups: usize,
    /// Dense patch-embedding convolution `[channels_in * k * k, channels]`.
    pub conv: LinearWeights,
    /// `pre_norm ? LayerNorm(channels_in) : LayerNorm(channels)`.
    pub norm: LayerNormWeights,
    /// `(spatial, channel)` block pairs in checkpoint order.
    pub blocks: Vec<(VisionBlockWeights, VisionBlockWeights)>,
    pub plan: StagePlan,
    /// `[1, channels]` zeros; padding sentinel for window gathers.
    pub zero_row: Tensor,
}

pub struct VisionWeights {
    /// Vision-tower input side (224 for the validated profile).
    pub input_side: usize,
    pub stages: Vec<VisionStageWeights>,
    /// `[49, 2048]`: column ‖ row embeddings plus the folded temporal constant.
    pub image_position_table: Tensor,
    /// `[2048, 1024]` stored as checkpoint `[in, out]` (no transpose).
    pub image_projection: Tensor,
    pub image_proj_norm: LayerNormWeights,
    /// `[1, 49]` of ones for the spatial average.
    pub spatial_mean_row: Tensor,
}

/// One post-norm BART encoder layer.
#[derive(Debug)]
pub struct TextLayerWeights {
    pub q_proj: LinearWeights,
    pub k_proj: LinearWeights,
    pub v_proj: LinearWeights,
    pub out_proj: LinearWeights,
    pub attn_norm: LayerNormWeights,
    pub fc1: LinearWeights,
    pub fc2: LinearWeights,
    pub ffn_norm: LayerNormWeights,
}

/// Florence-2 BART text encoder (`vlm.language_model.model.encoder`).
pub struct TextWeights {
    /// Tied token embedding `[vocab, d_model]` (`model.shared.weight`).
    pub embedding: Tensor,
    /// Pre-gathered absolute positions `[tokens, d_model]` (offset 2).
    pub positions: Tensor,
    pub embed_norm: LayerNormWeights,
    pub layers: Vec<TextLayerWeights>,
    pub heads: usize,
    pub head_dim: usize,
}

/// Everything needed for `forward_vlm` (vision + text encoder).
pub struct VlmWeights {
    pub vision: VisionWeights,
    pub text: TextWeights,
}

/// Domain-sliced linear layer (`DomainAwareLinear`): one `[input, output]`
/// matrix and one `[output]` bias per domain, stacked along the checkpoint
/// embedding axis.
#[derive(Debug)]
pub struct DomainLinearWeights {
    /// `[domains * input, output]` row-major; each checkpoint row is already
    /// `[input, output]` flattened (`W.view(B, input, output)` in the reference).
    pub weight: Tensor,
    /// `[domains, output]`.
    pub bias: Tensor,
    pub domains: usize,
    pub input: usize,
    pub output: usize,
}

impl DomainLinearWeights {
    /// Zero-copy `[input, output]` view for one domain.
    pub fn weight_view(&self, domain: usize) -> Result<Tensor> {
        if domain >= self.domains {
            return Err(Error::Other(format!(
                "X-VLA domain {domain} is out of range (domains = {})",
                self.domains
            )));
        }
        domain_view(
            &self.weight,
            domain,
            self.input * self.output,
            &[self.input, self.output],
        )
    }

    /// Zero-copy `[output]` bias view for one domain.
    pub fn bias_view(&self, domain: usize) -> Result<Tensor> {
        if domain >= self.domains {
            return Err(Error::Other(format!(
                "X-VLA domain {domain} is out of range (domains = {})",
                self.domains
            )));
        }
        domain_view(&self.bias, domain, self.output, &[self.output])
    }
}

fn domain_view(tensor: &Tensor, domain: usize, elements: usize, dims: &[usize]) -> Result<Tensor> {
    let buffer = DeviceBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let element_bytes = DType::BF16.size_in_bytes();
    let view = buffer
        .view(domain * elements * element_bytes, elements * element_bytes)
        .map_err(Error::Cuda)?;
    view.as_tensor(Shape::new(dims.to_vec()), DType::BF16)
        .map_err(Error::Cuda)
}

/// One pre-norm transformer block of the action head.
#[derive(Debug)]
pub struct ActionBlockWeights {
    pub norm1: LayerNormWeights,
    pub qkv: LinearWeights,
    pub proj: LinearWeights,
    pub norm2: LayerNormWeights,
    pub fc1: LinearWeights,
    pub fc2: LinearWeights,
}

/// SoftPromptedTransformer weights (24 pre-norm blocks + domain projections).
#[derive(Debug)]
pub struct ActionHeadWeights {
    pub vlm_proj: LinearWeights,
    pub aux_visual_proj: LinearWeights,
    /// `[action_horizon + merged_tokens + aux_tokens, hidden]` prefix of the
    /// checkpoint `pos_emb` table (soft prompts receive no positions).
    pub pos_emb: Tensor,
    pub norm: LayerNormWeights,
    pub action_encoder: DomainLinearWeights,
    pub action_decoder: DomainLinearWeights,
    /// `[domains * len_soft_prompts, hidden]` flattened soft prompt table.
    pub soft_prompt: Tensor,
    pub blocks: Vec<ActionBlockWeights>,
    /// Precomputed timestep embeddings `[flow_steps, dim_time]`, row `k` for
    /// `t = (k + 1) / flow_steps`.
    pub time_tokens: Tensor,
}

impl ActionHeadWeights {
    /// Load the action head from an X-VLA checkpoint directory or file.
    pub fn from_checkpoint(
        config: &crate::cuda_xvla::config::XvlaConfig,
        path: &std::path::Path,
        ctx: &crate::cuda_xvla::backend::Context,
    ) -> Result<Self> {
        let (mut tensors, _) =
            apxinf_loader::safetensors::load_native_path(path).map_err(Error::Other)?;
        crate::cuda_xvla::load::load_action(&mut tensors, config, ctx)
    }
}

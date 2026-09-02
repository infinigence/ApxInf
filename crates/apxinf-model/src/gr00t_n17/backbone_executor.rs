//! Fixed-shape, GPU-resident Cosmos/Qwen3-VL backbone for GR00T N1.7 LIBERO.

use apxinf_core::{Error, Result, Tensor};
use apxinf_cuda::buffer::CudaBuffer;
use apxinf_cuda::kernels::{activation, attention, elementwise, embedding, gemm, norm, rope};
use apxinf_cuda::CudaContext;
use std::sync::Arc;

use super::backbone_weights::{GrootLanguageLayer, GrootVisionMerger};
use super::{GrootN17BackboneWeights, LayerNormWeights, LinearWeights};

pub struct GrootN17BackboneExecutor {
    weights: Arc<GrootN17BackboneWeights>,
    token_count: usize,
    view_count: usize,
    patch_rows: usize,
    token_ids: CudaBuffer,
    language_positions: CudaBuffer,
    vision_positions: CudaBuffer,
    image_rows: elementwise::RowIndices,
}

impl GrootN17BackboneExecutor {
    pub fn new(
        weights: Arc<GrootN17BackboneWeights>,
        token_ids: &[u32],
        device: usize,
    ) -> Result<Self> {
        let view_count = match token_ids.len() {
            76 => 1,
            142 => 2,
            count => {
                return Err(Error::Other(format!(
                    "GR00T profile requires 76 or 142 tokens, got {count}"
                )))
            }
        };
        let image_positions = token_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &token)| (token == 151655).then_some(i as u32))
            .collect::<Vec<_>>();
        if image_positions.len() != view_count * 64 {
            return Err(Error::Other(format!(
                "GR00T {view_count}-view profile requires {} image tokens, got {}",
                view_count * 64,
                image_positions.len()
            )));
        }
        Ok(Self {
            weights,
            token_count: token_ids.len(),
            view_count,
            patch_rows: view_count * 256,
            token_ids: upload_u32(device, token_ids)?,
            language_positions: upload_u32(device, &mrope_positions(token_ids, view_count)?)?,
            vision_positions: upload_u32(device, &vision_positions(view_count))?,
            image_rows: elementwise::RowIndices::new(device, &image_positions)?,
        })
    }

    pub fn forward(&self, ctx: &CudaContext, patches: &Tensor) -> Result<Tensor> {
        if patches.shape().dims() != [self.patch_rows, 1536] {
            return Err(Error::Other(format!(
                "GR00T patches must be [{},1536], got {:?}",
                self.patch_rows,
                patches.shape().dims()
            )));
        }
        let vision = self
            .forward_vision(ctx, patches)
            .map_err(|error| Error::Other(format!("GR00T vision tower: {error}")))?;
        let mut hidden = embedding::lookup_unscaled_bf16(
            ctx,
            &self.weights.embedding,
            &self.token_ids,
            self.token_count,
        )
        .map_err(|error| Error::Other(format!("GR00T token embedding: {error}")))?;
        hidden = elementwise::scatter_rows_bf16(ctx, &hidden, &vision.0, &self.image_rows, false)
            .map_err(|error| {
            Error::Other(format!("GR00T primary vision injection: {error}"))
        })?;
        for (index, layer) in self.weights.language_layers.iter().enumerate() {
            hidden = language_layer(
                ctx,
                &hidden,
                layer,
                &self.language_positions,
                self.token_count,
            )
            .map_err(|error| Error::Other(format!("GR00T language layer {index}: {error}")))?;
            if index < vision.1.len() {
                hidden = elementwise::scatter_rows_bf16(
                    ctx,
                    &hidden,
                    &vision.1[index],
                    &self.image_rows,
                    true,
                )
                .map_err(|error| {
                    Error::Other(format!("GR00T deepstack injection {index}: {error}"))
                })?;
            }
        }
        // GR00T selects `outputs.hidden_states[-1]` from the truncated Qwen
        // backbone. Transformers records that boundary before the language
        // model's final RMSNorm; the action head's own VLLN is the next norm.
        Ok(hidden)
    }

    pub fn update_token_ids(&self, token_ids: &[u32]) -> Result<()> {
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "GR00T prepared graph requires {} token IDs",
                self.token_count
            )));
        }
        let image_positions = token_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &token)| (token == 151655).then_some(i))
            .collect::<Vec<_>>();
        let expected = if self.view_count == 1 {
            (4..68).collect::<Vec<_>>()
        } else {
            (4..68).chain(70..134).collect::<Vec<_>>()
        };
        if image_positions != expected {
            return Err(Error::Other(
                "GR00T token image spans do not match the prepared graph".into(),
            ));
        }
        let bytes = token_ids
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        self.token_ids.copy_from_host(&bytes).map_err(Error::Cuda)
    }

    fn forward_vision(&self, ctx: &CudaContext, patches: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let w = &self.weights.vision;
        let mut hidden = linear_bias(ctx, patches, &w.patch)
            .map_err(|error| Error::Other(format!("patch projection: {error}")))?;
        let position = if self.view_count == 1 {
            &w.position_one_view
        } else {
            &w.position_two_view
        };
        hidden = elementwise::add(ctx, &hidden, position)
            .map_err(|error| Error::Other(format!("vision position add: {error}")))?;
        let mut deep = Vec::with_capacity(3);
        for (index, block) in w.blocks.iter().enumerate() {
            hidden = (|| -> Result<Tensor> {
                let normalized = layer_norm(ctx, &hidden, &block.norm1, 1e-6)?;
                let q = linear_bias(ctx, &normalized, &block.q)?.reshape(vec![
                    self.patch_rows,
                    16,
                    64,
                ])?;
                let k = linear_bias(ctx, &normalized, &block.k)?.reshape(vec![
                    self.patch_rows,
                    16,
                    64,
                ])?;
                let v = linear_bias(ctx, &normalized, &block.v)?.reshape(vec![
                    self.patch_rows,
                    16,
                    64,
                ])?;
                let q = rope::apply_vision_2d(ctx, &q, 16, 64, 10000.0, &self.vision_positions)?;
                let k = rope::apply_vision_2d(ctx, &k, 16, 64, 10000.0, &self.vision_positions)?;
                let attended = attention::mha_bf16(ctx, &q, &k, &v, 256)?
                    .reshape(vec![self.patch_rows, 1024])?;
                let output =
                    elementwise::add(ctx, &hidden, &linear_bias(ctx, &attended, &block.output)?)?;
                let normalized = layer_norm(ctx, &output, &block.norm2, 1e-6)?;
                let ff = linear(ctx, &normalized, &block.fc1)?;
                let ff = activation::bias_gelu_bf16(ctx, &ff, block.fc1.bias.as_ref())?;
                elementwise::add(ctx, &output, &linear_bias(ctx, &ff, &block.fc2)?)
            })()
            .map_err(|error| Error::Other(format!("vision block {index}: {error}")))?;
            if matches!(index, 5 | 11 | 17) {
                deep.push(hidden.clone());
            }
        }
        let merged_rows = self.view_count * 64;
        let primary = merge_primary(ctx, &hidden, &w.primary_merger, merged_rows)
            .map_err(|error| Error::Other(format!("primary merger: {error}")))?;
        let deep = deep
            .iter()
            .zip(&w.deepstack_mergers)
            .enumerate()
            .map(|(index, (value, merger))| {
                merge_deep(ctx, value, merger, merged_rows)
                    .map_err(|error| Error::Other(format!("deepstack merger {index}: {error}")))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((primary, deep))
    }
}

fn language_layer(
    ctx: &CudaContext,
    input: &Tensor,
    w: &GrootLanguageLayer,
    positions: &CudaBuffer,
    tokens: usize,
) -> Result<Tensor> {
    let normalized = norm::rms_bf16(ctx, input, &w.input_norm, 1e-6)?;
    let q = linear(ctx, &normalized, &w.q)?.reshape(vec![tokens * 16, 128])?;
    let k = linear(ctx, &normalized, &w.k)?.reshape(vec![tokens * 8, 128])?;
    let q = norm::rms_bf16(ctx, &q, &w.q_norm, 1e-6)?.reshape(vec![tokens, 16, 128])?;
    let k = norm::rms_bf16(ctx, &k, &w.k_norm, 1e-6)?.reshape(vec![tokens, 8, 128])?;
    let v = linear(ctx, &normalized, &w.v)?.reshape(vec![tokens, 8, 128])?;
    let q = rope::apply_mrope(ctx, &q, 16, 128, 5_000_000.0, [24, 20, 20], positions)?;
    let k = rope::apply_mrope(ctx, &k, 8, 128, 5_000_000.0, [24, 20, 20], positions)?;
    let attended = attention::causal_gqa_bf16(ctx, &q, &k, &v)?.reshape(vec![tokens, 2048])?;
    let hidden = elementwise::add(ctx, input, &linear(ctx, &attended, &w.output)?)?;
    let normalized = norm::rms_bf16(ctx, &hidden, &w.post_norm, 1e-6)?;
    let gate = activation::silu(ctx, &linear(ctx, &normalized, &w.gate)?)?;
    let up = linear(ctx, &normalized, &w.up)?;
    let ff = elementwise::mul(ctx, &gate, &up)?;
    elementwise::add(ctx, &hidden, &linear(ctx, &ff, &w.down)?)
}

fn merge_primary(
    ctx: &CudaContext,
    input: &Tensor,
    w: &GrootVisionMerger,
    rows: usize,
) -> Result<Tensor> {
    let normalized = layer_norm(ctx, input, &w.norm, 1e-6)?.reshape(vec![rows, 4096])?;
    merger_mlp(ctx, &normalized, w)
}

fn merge_deep(
    ctx: &CudaContext,
    input: &Tensor,
    w: &GrootVisionMerger,
    rows: usize,
) -> Result<Tensor> {
    let merged = input.reshape(vec![rows, 4096])?;
    let normalized = layer_norm(ctx, &merged, &w.norm, 1e-6)?;
    merger_mlp(ctx, &normalized, w)
}

fn merger_mlp(ctx: &CudaContext, input: &Tensor, w: &GrootVisionMerger) -> Result<Tensor> {
    let hidden = linear(ctx, input, &w.fc1)?;
    let hidden = activation::bias_gelu_bf16(ctx, &hidden, w.fc1.bias.as_ref())?;
    linear_bias(ctx, &hidden, &w.fc2)
}

fn linear(ctx: &CudaContext, input: &Tensor, w: &LinearWeights) -> Result<Tensor> {
    gemm::matmul(ctx, input, &w.weight)
}
fn linear_bias(ctx: &CudaContext, input: &Tensor, w: &LinearWeights) -> Result<Tensor> {
    elementwise::bias_bf16(ctx, &linear(ctx, input, w)?, w.bias.as_ref())
}
fn layer_norm(ctx: &CudaContext, input: &Tensor, w: &LayerNormWeights, eps: f32) -> Result<Tensor> {
    norm::layer_bf16(ctx, input, &w.weight, &w.bias, eps)
}

fn upload_u32(device: usize, values: &[u32]) -> Result<CudaBuffer> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let output = CudaBuffer::alloc(bytes.len(), device).map_err(Error::Cuda)?;
    output.copy_from_host(&bytes).map_err(Error::Cuda)?;
    Ok(output)
}

fn vision_positions(views: usize) -> Vec<u32> {
    let mut output = Vec::with_capacity(views * 512);
    for _ in 0..views {
        for mr in 0..8 {
            for mc in 0..8 {
                for ir in 0..2 {
                    for ic in 0..2 {
                        output.extend_from_slice(&[(mr * 2 + ir) as u32, (mc * 2 + ic) as u32]);
                    }
                }
            }
        }
    }
    output
}

fn mrope_positions(tokens: &[u32], expected_images: usize) -> Result<Vec<u32>> {
    let mut output = Vec::with_capacity(tokens.len() * 3);
    let mut cursor = 0usize;
    let mut next = 0u32;
    let mut images = 0;
    while cursor < tokens.len() {
        if tokens[cursor] != 151655 {
            output.extend_from_slice(&[next, next, next]);
            next += 1;
            cursor += 1;
            continue;
        }
        images += 1;
        for row in 0..8 {
            for col in 0..8 {
                output.extend_from_slice(&[next, next + row, next + col]);
            }
        }
        next += 8;
        cursor += 64;
    }
    if images != expected_images {
        return Err(Error::Other(format!(
            "expected {expected_images} image spans, got {images}"
        )));
    }
    Ok(output)
}

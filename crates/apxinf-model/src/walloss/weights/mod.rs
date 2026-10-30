//! Checkpoint mapping for the Walloss dual-stream execution graph.

#[cfg(feature = "cuda")]
mod device;
#[cfg(feature = "cuda")]
mod fp8;

#[cfg(feature = "cuda")]
pub use device::DynamicFp8LinearWeights;
#[cfg(feature = "cuda")]
pub use fp8::{
    WallossDynamicFp8LayerWeights, WallossDynamicFp8VisionBlockWeights,
    WallossDynamicFp8VisionWeights, WallossDynamicFp8Weights,
};

use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{Backend, DType, Error, Result, Tensor};

use crate::walloss::WallossConfig;

pub struct WallossWeights {
    pub token_embedding: Tensor,
    pub language_layers: Vec<WallossLayerWeights>,
    pub action_layers: Vec<WallossLayerWeights>,
    pub action_norm: Tensor,
    pub vision: WallossVisionWeights,
    pub action: WallossActionWeights,
}

pub struct WallossLayerWeights {
    pub input_norm: Tensor,
    pub post_attention_norm: Tensor,
    /// Matmul-ready `[input, q + k + v]` projection.
    pub qkv: Tensor,
    pub qkv_bias: Tensor,
    /// Matmul-ready `[attention_width, output]` projection.
    pub output: Tensor,
    /// Matmul-ready `[hidden, 2 * intermediate]` fused gate/up projection.
    pub gate_up: Tensor,
    /// Matmul-ready `[intermediate, hidden]` projection.
    pub down: Tensor,
}

pub struct WallossVisionWeights {
    /// Flattened and transposed patch projection.
    pub patch_projection: Tensor,
    pub blocks: Vec<WallossVisionBlockWeights>,
    pub merger_norm: Tensor,
    pub merger_hidden: Tensor,
    pub merger_hidden_bias: Tensor,
    pub merger_output: Tensor,
    pub merger_output_bias: Tensor,
}

pub struct WallossVisionBlockWeights {
    pub input_norm: Tensor,
    pub qkv: Tensor,
    pub qkv_bias: Tensor,
    pub output: Tensor,
    pub output_bias: Tensor,
    pub post_attention_norm: Tensor,
    pub gate_up: Tensor,
    pub gate_up_bias: Tensor,
    pub down: Tensor,
    pub down_bias: Tensor,
}

pub struct WallossActionWeights {
    pub noisy_action_projection: Tensor,
    pub dof_projection: Tensor,
    pub action_projection: Tensor,
    pub time_projection: Tensor,
    pub action_embedding_projection: Tensor,
    pub velocity_projection: Tensor,
}

impl WallossWeights {
    pub fn from_safetensors(config: &mut WallossConfig, path: &Path) -> Result<Self> {
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|error| Error::Other(format!("load walloss checkpoint: {error}")))?;
        Self::from_map(config, tensors)
    }

    pub fn from_map(
        config: &mut WallossConfig,
        mut tensors: HashMap<String, Tensor>,
    ) -> Result<Self> {
        let token_embedding = take(&mut tensors, "model.embed_tokens.weight")?;
        expect_rank(&token_embedding, 2, "model.embed_tokens.weight")?;
        let embedding_shape = token_embedding.shape().dims();
        if embedding_shape[1] != config.text.hidden_size {
            return Err(Error::Other(format!(
                "walloss checkpoint: embedding width {} != configured {}",
                embedding_shape[1], config.text.hidden_size
            )));
        }
        // Added control tokens make the tensor shape authoritative.
        config.text.vocab_size = embedding_shape[0];
        // Published WallOSS checkpoints do not consistently serialize the
        // native action width in config.json. The output projection is the
        // architecture boundary: [action_dim, action_hidden_size]. Validate
        // the paired action+DoF input projection and make those weight shapes
        // authoritative before any action buffers are allocated.
        let action_dim = checkpoint_action_dim(&tensors, config.action.hidden_size)?;
        config.action.action_dim = action_dim;
        config.action.proprio_dim = action_dim;

        let q_width = config.text.num_attention_heads * config.text.head_dim;
        let kv_width = config.text.num_kv_heads * config.text.head_dim;
        let qkv_width = q_width + 2 * kv_width;
        let mut language_layers = Vec::with_capacity(config.text.num_layers);
        let mut action_layers = Vec::with_capacity(config.text.num_layers);
        for layer in 0..config.text.num_layers {
            language_layers.push(load_layer(
                &mut tensors,
                layer,
                0,
                config.text.hidden_size,
                config.text.intermediate_size,
                qkv_width,
                q_width,
            )?);
            action_layers.push(load_layer(
                &mut tensors,
                layer,
                1,
                config.action.hidden_size,
                config.action.intermediate_size,
                qkv_width,
                q_width,
            )?);
        }

        // The V/L prefill contributes only its per-layer KV cache to action
        // denoising. Its final hidden state is discarded, so the reference
        // model's terminal V/L norm has no numerical consumer here. Still
        // require and validate the tensor to reject incomplete checkpoints.
        let language_norm = take(&mut tensors, "model.norms.0.weight")?;
        let action_norm = take(&mut tensors, "model.norms.1.weight")?;
        expect_shape(
            &language_norm,
            &[config.text.hidden_size],
            "language final norm",
        )?;
        expect_shape(
            &action_norm,
            &[config.action.hidden_size],
            "action final norm",
        )?;

        Ok(Self {
            token_embedding,
            language_layers,
            action_layers,
            action_norm,
            vision: load_vision(config, &mut tensors)?,
            action: load_action(config, &mut tensors)?,
        })
    }

    pub fn to_bf16_device(&self, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            token_embedding: bf16_to_device(&self.token_embedding, backend)?,
            language_layers: self
                .language_layers
                .iter()
                .map(|layer| layer.to_bf16_device(backend))
                .collect::<Result<_>>()?,
            action_layers: self
                .action_layers
                .iter()
                .map(|layer| layer.to_bf16_device(backend))
                .collect::<Result<_>>()?,
            action_norm: bf16_to_device(&self.action_norm, backend)?,
            vision: self.vision.to_bf16_device(backend)?,
            action: self.action.to_bf16_device(backend)?,
        })
    }
}

impl WallossLayerWeights {
    fn to_bf16_device(&self, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            input_norm: bf16_to_device(&self.input_norm, backend)?,
            post_attention_norm: bf16_to_device(&self.post_attention_norm, backend)?,
            qkv: bf16_to_device(&self.qkv, backend)?,
            qkv_bias: bf16_to_device(&self.qkv_bias, backend)?,
            output: bf16_to_device(&self.output, backend)?,
            gate_up: bf16_to_device(&self.gate_up, backend)?,
            down: bf16_to_device(&self.down, backend)?,
        })
    }
}

impl WallossVisionWeights {
    fn to_bf16_device(&self, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            patch_projection: bf16_to_device(&self.patch_projection, backend)?,
            blocks: self
                .blocks
                .iter()
                .map(|block| block.to_bf16_device(backend))
                .collect::<Result<_>>()?,
            merger_norm: bf16_to_device(&self.merger_norm, backend)?,
            merger_hidden: bf16_to_device(&self.merger_hidden, backend)?,
            merger_hidden_bias: bf16_to_device(&self.merger_hidden_bias, backend)?,
            merger_output: bf16_to_device(&self.merger_output, backend)?,
            merger_output_bias: bf16_to_device(&self.merger_output_bias, backend)?,
        })
    }
}

impl WallossVisionBlockWeights {
    fn to_bf16_device(&self, backend: &dyn Backend) -> Result<Self> {
        let (gate_up, gate_up_bias, down) =
            pad_swiglu_inner(&self.gate_up, &self.gate_up_bias, &self.down, 64)?;
        Ok(Self {
            input_norm: bf16_to_device(&self.input_norm, backend)?,
            qkv: bf16_to_device(&self.qkv, backend)?,
            qkv_bias: bf16_to_device(&self.qkv_bias, backend)?,
            output: bf16_to_device(&self.output, backend)?,
            output_bias: bf16_to_device(&self.output_bias, backend)?,
            post_attention_norm: bf16_to_device(&self.post_attention_norm, backend)?,
            gate_up: bf16_to_device(&gate_up, backend)?,
            gate_up_bias: bf16_to_device(&gate_up_bias, backend)?,
            down: bf16_to_device(&down, backend)?,
            down_bias: bf16_to_device(&self.down_bias, backend)?,
        })
    }
}

/// Zero-pad the SwiGLU inner width of `gate_up` `[hidden, 2*inner]`, its bias
/// and `down` `[inner, out]` to a multiple of `align`. Padded gate/up columns
/// and down rows are zero, so the block output is unchanged; the aligned
/// reduction dimension lets the down projection use full-width GEMM tiles
/// (the vision inner width 3420 is not a multiple of 16 elements).
fn pad_swiglu_inner(
    gate_up: &Tensor,
    gate_up_bias: &Tensor,
    down: &Tensor,
    align: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let [hidden, twice_inner] = gate_up.shape().dims() else {
        return Err(Error::Other("walloss gate/up weight must be rank 2".into()));
    };
    let (hidden, inner) = (*hidden, *twice_inner / 2);
    let out = down.shape().dims()[1];
    let padded = inner.div_ceil(align) * align;
    let pad_halves = |values: &[f32], rows: usize| {
        let mut output = vec![0.0; rows * 2 * padded];
        for row in 0..rows {
            let (source, target) = (&values[row * 2 * inner..], &mut output[row * 2 * padded..]);
            target[..inner].copy_from_slice(&source[..inner]);
            target[padded..padded + inner].copy_from_slice(&source[inner..2 * inner]);
        }
        output
    };
    let mut down_values = down.to_f32_vec()?;
    down_values.resize(padded * out, 0.0);
    Ok((
        Tensor::from_f32(vec![hidden, 2 * padded], &pad_halves(&gate_up.to_f32_vec()?, hidden))?,
        Tensor::from_f32(vec![2 * padded], &pad_halves(&gate_up_bias.to_f32_vec()?, 1))?,
        Tensor::from_f32(vec![padded, out], &down_values)?,
    ))
}

impl WallossActionWeights {
    pub(super) fn to_bf16_device(&self, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            noisy_action_projection: bf16_to_device(&self.noisy_action_projection, backend)?,
            dof_projection: bf16_to_device(&self.dof_projection, backend)?,
            action_projection: bf16_to_device(&self.action_projection, backend)?,
            time_projection: bf16_to_device(&self.time_projection, backend)?,
            action_embedding_projection: bf16_to_device(
                &self.action_embedding_projection,
                backend,
            )?,
            velocity_projection: bf16_to_device(&self.velocity_projection, backend)?,
        })
    }
}

pub(super) fn bf16_to_device(tensor: &Tensor, backend: &dyn Backend) -> Result<Tensor> {
    if tensor.dtype() == DType::F8E4M3 {
        return Err(Error::Other(
            "walloss BF16 upload cannot decode scale-less FP8 data".into(),
        ));
    }
    let values = tensor
        .to_f32_vec()?
        .into_iter()
        .map(half::bf16::from_f32)
        .collect::<Vec<_>>();
    backend.to_device(&Tensor::from_bf16(tensor.shape().dims().to_vec(), &values)?)
}

fn load_layer(
    tensors: &mut HashMap<String, Tensor>,
    layer: usize,
    branch: usize,
    hidden: usize,
    intermediate: usize,
    qkv_width: usize,
    attention_width: usize,
) -> Result<WallossLayerWeights> {
    let prefix = format!("model.layers.{layer}");
    let input_norm = take(
        tensors,
        &format!("{prefix}.input_layernorms.{branch}.weight"),
    )?;
    let post_attention_norm = take(
        tensors,
        &format!("{prefix}.post_attention_layernorms.{branch}.weight"),
    )?;
    let qkv_source = take(
        tensors,
        &format!("{prefix}.self_attn.qkv_proj_experts.{branch}.weight"),
    )?;
    let qkv_bias = take(
        tensors,
        &format!("{prefix}.self_attn.qkv_proj_experts.{branch}.bias"),
    )?;
    let output_source = take(
        tensors,
        &format!("{prefix}.self_attn.o_proj_experts.{branch}.weight"),
    )?;
    let gate_up_source = take(
        tensors,
        &format!("{prefix}.moe.experts.{branch}.gate_up_proj.weight"),
    )?;
    let down_source = take(
        tensors,
        &format!("{prefix}.moe.experts.{branch}.down_proj.weight"),
    )?;

    expect_shape(&input_norm, &[hidden], "input norm")?;
    expect_shape(&post_attention_norm, &[hidden], "post-attention norm")?;
    expect_shape(&qkv_source, &[qkv_width, hidden], "QKV projection")?;
    expect_shape(&qkv_bias, &[qkv_width], "QKV bias")?;
    expect_shape(
        &output_source,
        &[hidden, attention_width],
        "output projection",
    )?;
    expect_shape(
        &gate_up_source,
        &[2 * intermediate, hidden],
        "gate/up projection",
    )?;
    expect_shape(&down_source, &[hidden, intermediate], "down projection")?;

    Ok(WallossLayerWeights {
        input_norm,
        post_attention_norm,
        qkv: transpose_2d(&qkv_source)?,
        qkv_bias,
        output: transpose_2d(&output_source)?,
        gate_up: transpose_2d(&gate_up_source)?,
        down: transpose_2d(&down_source)?,
    })
}

fn load_vision(
    config: &WallossConfig,
    tensors: &mut HashMap<String, Tensor>,
) -> Result<WallossVisionWeights> {
    let vision = &config.vision;
    if vision.hidden_size % vision.num_heads != 0 {
        return Err(Error::Other(
            "walloss checkpoint: invalid vision head geometry".into(),
        ));
    }
    let mut blocks = Vec::with_capacity(vision.depth);
    for layer in 0..vision.depth {
        let prefix = format!("visual.blocks.{layer}");
        blocks.push(WallossVisionBlockWeights {
            input_norm: take(tensors, &format!("{prefix}.norm1.weight"))?,
            qkv: transpose_2d(&take(tensors, &format!("{prefix}.attn.qkv.weight"))?)?,
            qkv_bias: take(tensors, &format!("{prefix}.attn.qkv.bias"))?,
            output: transpose_2d(&take(tensors, &format!("{prefix}.attn.proj.weight"))?)?,
            output_bias: take(tensors, &format!("{prefix}.attn.proj.bias"))?,
            post_attention_norm: take(tensors, &format!("{prefix}.norm2.weight"))?,
            gate_up: transpose_2d(&take(
                tensors,
                &format!("{prefix}.mlp.gate_up_proj.weight"),
            )?)?,
            gate_up_bias: take(tensors, &format!("{prefix}.mlp.gate_up_proj.bias"))?,
            down: transpose_2d(&take(tensors, &format!("{prefix}.mlp.down_proj.weight"))?)?,
            down_bias: take(tensors, &format!("{prefix}.mlp.down_proj.bias"))?,
        });
    }

    let patch = take(tensors, "visual.patch_embed.proj.weight")?;
    let patch_columns = 3 * vision.temporal_patch_size * vision.patch_size * vision.patch_size;
    let patch_projection = flatten_and_transpose(&patch, vision.hidden_size, patch_columns)?;

    Ok(WallossVisionWeights {
        patch_projection,
        blocks,
        merger_norm: take(tensors, "visual.merger.ln_q.weight")?,
        merger_hidden: transpose_2d(&take(tensors, "visual.merger.mlp.0.weight")?)?,
        merger_hidden_bias: take(tensors, "visual.merger.mlp.0.bias")?,
        merger_output: transpose_2d(&take(tensors, "visual.merger.mlp.2.weight")?)?,
        merger_output_bias: take(tensors, "visual.merger.mlp.2.bias")?,
    })
}

fn load_action(
    config: &WallossConfig,
    tensors: &mut HashMap<String, Tensor>,
) -> Result<WallossActionWeights> {
    let width = config.action.hidden_size;
    let dim = config.action.action_dim;
    let noisy_action_projection = take(tensors, "action_preprocessor.w1.weight")?;
    let action_time_projection = take(tensors, "action_preprocessor.w2.weight")?;
    let action_embedding_projection = take(tensors, "action_preprocessor.w3.weight")?;
    let velocity_projection = take(tensors, "action_preprocessor.action_proj_back.weight")?;
    expect_shape(
        &noisy_action_projection,
        &[width, 2 * dim],
        "noisy action projection",
    )?;
    expect_shape(
        &action_time_projection,
        &[width, 2 * width],
        "action/time projection",
    )?;
    expect_shape(
        &action_embedding_projection,
        &[width, width],
        "action embedding projection",
    )?;
    expect_shape(&velocity_projection, &[dim, width], "velocity projection")?;
    let (noisy_action_projection, dof_projection) = split_columns(&noisy_action_projection, dim)?;
    let (action_projection, time_projection) = split_columns(&action_time_projection, width)?;
    Ok(WallossActionWeights {
        noisy_action_projection: to_bf16(&transpose_2d(&noisy_action_projection)?)?,
        dof_projection: to_bf16(&transpose_2d(&dof_projection)?)?,
        action_projection: to_bf16(&transpose_2d(&action_projection)?)?,
        time_projection: to_bf16(&transpose_2d(&time_projection)?)?,
        action_embedding_projection: to_bf16(&transpose_2d(&action_embedding_projection)?)?,
        velocity_projection: to_bf16(&transpose_2d(&velocity_projection)?)?,
    })
}

fn checkpoint_action_dim(
    tensors: &HashMap<String, Tensor>,
    action_hidden_size: usize,
) -> Result<usize> {
    let output = tensors
        .get("action_preprocessor.action_proj_back.weight")
        .ok_or_else(|| {
            Error::Other(
                "walloss checkpoint: missing action_preprocessor.action_proj_back.weight".into(),
            )
        })?;
    expect_rank(output, 2, "velocity projection")?;
    let output_shape = output.shape().dims();
    let action_dim = output_shape[0];
    if action_dim == 0 || output_shape[1] != action_hidden_size {
        return Err(Error::Other(format!(
            "walloss checkpoint: velocity projection has shape {output_shape:?}, expected [action_dim, {action_hidden_size}]"
        )));
    }

    let input = tensors
        .get("action_preprocessor.w1.weight")
        .ok_or_else(|| {
            Error::Other("walloss checkpoint: missing action_preprocessor.w1.weight".into())
        })?;
    expect_shape(
        input,
        &[action_hidden_size, 2 * action_dim],
        "noisy action projection",
    )?;
    Ok(action_dim)
}

fn take(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| Error::Other(format!("walloss checkpoint: missing {name}")))
}

fn expect_rank(tensor: &Tensor, rank: usize, name: &str) -> Result<()> {
    if tensor.shape().dims().len() != rank {
        return Err(Error::Other(format!(
            "walloss checkpoint: {name} has shape {:?}, expected rank {rank}",
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn expect_shape(tensor: &Tensor, expected: &[usize], name: &str) -> Result<()> {
    if tensor.shape().dims() != expected {
        return Err(Error::Other(format!(
            "walloss checkpoint: {name} has shape {:?}, expected {expected:?}",
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn flatten_and_transpose(tensor: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
    if tensor.shape().numel() != rows * cols {
        return Err(Error::Other(format!(
            "walloss checkpoint: patch projection has {} elements, expected {}",
            tensor.shape().numel(),
            rows * cols
        )));
    }
    match tensor.dtype() {
        DType::F32 => Tensor::from_f32(vec![rows, cols], tensor.as_f32()?)
            .and_then(|flat| transpose_2d(&flat)),
        DType::BF16 => Tensor::from_bf16(vec![rows, cols], tensor.as_bf16()?)
            .and_then(|flat| transpose_2d(&flat)),
        dtype => Err(Error::Other(format!(
            "walloss checkpoint: unsupported patch weight dtype {dtype}"
        ))),
    }
}

fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    expect_rank(tensor, 2, "linear weight")?;
    let rows = tensor.shape().dims()[0];
    let cols = tensor.shape().dims()[1];
    match tensor.dtype() {
        DType::F32 => {
            let source = tensor.as_f32()?;
            let mut output = vec![0.0; rows * cols];
            for row in 0..rows {
                for col in 0..cols {
                    output[col * rows + row] = source[row * cols + col];
                }
            }
            Tensor::from_f32(vec![cols, rows], &output)
        }
        DType::BF16 => {
            let source = tensor.as_bf16()?;
            let mut output = vec![half::bf16::ZERO; rows * cols];
            for row in 0..rows {
                for col in 0..cols {
                    output[col * rows + row] = source[row * cols + col];
                }
            }
            Tensor::from_bf16(vec![cols, rows], &output)
        }
        dtype => Err(Error::Other(format!(
            "walloss checkpoint: unsupported linear weight dtype {dtype}"
        ))),
    }
}

fn split_columns(tensor: &Tensor, first_columns: usize) -> Result<(Tensor, Tensor)> {
    expect_rank(tensor, 2, "split weight")?;
    let rows = tensor.shape().dims()[0];
    let columns = tensor.shape().dims()[1];
    if first_columns == 0 || first_columns >= columns {
        return Err(Error::Other(format!(
            "walloss checkpoint: invalid column split {first_columns} for {columns} columns"
        )));
    }
    let second_columns = columns - first_columns;
    match tensor.dtype() {
        DType::F32 => {
            let source = tensor.as_f32()?;
            let mut first = Vec::with_capacity(rows * first_columns);
            let mut second = Vec::with_capacity(rows * second_columns);
            for row in 0..rows {
                let start = row * columns;
                first.extend_from_slice(&source[start..start + first_columns]);
                second.extend_from_slice(&source[start + first_columns..start + columns]);
            }
            Ok((
                Tensor::from_f32(vec![rows, first_columns], &first)?,
                Tensor::from_f32(vec![rows, second_columns], &second)?,
            ))
        }
        DType::BF16 => {
            let source = tensor.as_bf16()?;
            let mut first = Vec::with_capacity(rows * first_columns);
            let mut second = Vec::with_capacity(rows * second_columns);
            for row in 0..rows {
                let start = row * columns;
                first.extend_from_slice(&source[start..start + first_columns]);
                second.extend_from_slice(&source[start + first_columns..start + columns]);
            }
            Ok((
                Tensor::from_bf16(vec![rows, first_columns], &first)?,
                Tensor::from_bf16(vec![rows, second_columns], &second)?,
            ))
        }
        dtype => Err(Error::Other(format!(
            "walloss checkpoint: unsupported split weight dtype {dtype}"
        ))),
    }
}

fn to_bf16(tensor: &Tensor) -> Result<Tensor> {
    match tensor.dtype() {
        DType::BF16 => Ok(tensor.clone()),
        DType::F32 => {
            let values = tensor
                .as_f32()?
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>();
            Tensor::from_bf16(tensor.shape().clone(), &values)
        }
        dtype => Err(Error::Other(format!(
            "walloss checkpoint: cannot convert {dtype} action weight to BF16"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action_tensors(action_dim: usize, hidden_size: usize) -> HashMap<String, Tensor> {
        HashMap::from([
            (
                "action_preprocessor.action_proj_back.weight".to_owned(),
                Tensor::zeros((action_dim, hidden_size), DType::BF16),
            ),
            (
                "action_preprocessor.w1.weight".to_owned(),
                Tensor::zeros((hidden_size, 2 * action_dim), DType::BF16),
            ),
        ])
    }

    #[test]
    fn derives_native_action_width_from_checkpoint_projections() {
        assert_eq!(
            checkpoint_action_dim(&action_tensors(7, 1024), 1024).unwrap(),
            7
        );
        assert_eq!(
            checkpoint_action_dim(&action_tensors(26, 1024), 1024).unwrap(),
            26
        );
    }

    #[test]
    fn rejects_inconsistent_action_and_mask_projection_width() {
        let mut tensors = action_tensors(7, 1024);
        tensors.insert(
            "action_preprocessor.w1.weight".to_owned(),
            Tensor::zeros((1024, 15), DType::BF16),
        );
        assert!(checkpoint_action_dim(&tensors, 1024).is_err());
    }

    #[test]
    fn swiglu_padding_keeps_gate_and_up_halves_and_zero_fills() {
        // hidden 2, inner 3 -> padded 4; gate columns g*, up columns u*.
        let gate_up = Tensor::from_f32(
            vec![2, 6],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let bias = Tensor::from_f32(vec![6], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let down = Tensor::from_f32(vec![3, 1], &[1.0, 2.0, 3.0]).unwrap();
        let (gate_up, bias, down) = pad_swiglu_inner(&gate_up, &bias, &down, 4).unwrap();
        assert_eq!(gate_up.shape().dims(), [2, 8]);
        assert_eq!(
            gate_up.to_f32_vec().unwrap(),
            [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0, 7.0, 8.0, 9.0, 0.0, 10.0, 11.0, 12.0, 0.0]
        );
        assert_eq!(
            bias.to_f32_vec().unwrap(),
            [0.1, 0.2, 0.3, 0.0, 0.4, 0.5, 0.6, 0.0]
        );
        assert_eq!(down.to_f32_vec().unwrap(), [1.0, 2.0, 3.0, 0.0]);
    }
}

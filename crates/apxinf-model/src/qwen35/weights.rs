use std::collections::HashMap;

use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_loader::compressed_tensors::{W4KernelLayout, W4LayoutMetadata};

use super::{LayerKind, Qwen35Config};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedLinearWeight {
    pub prefix: String,
    pub rows: usize,
    pub cols: usize,
    pub packed_shape: Vec<usize>,
    pub scale_shape: Vec<usize>,
    pub zero_point_shape: Vec<usize>,
    pub kernel_layout: W4LayoutMetadata,
}

#[derive(Clone, Debug)]
pub struct Qwen35WeightsManifest {
    pub token_embedding: String,
    pub lm_head: String,
    pub compressed_linears: Vec<CompressedLinearWeight>,
}

impl Qwen35WeightsManifest {
    /// Validate the Qwen3.5 compressed-tensors naming scheme present in a loaded
    /// SafeTensors map.  This does not materialize dequantized weights.
    pub fn from_tensors(config: &Qwen35Config, tensors: &HashMap<String, Tensor>) -> Result<Self> {
        expect_tensor(tensors, "model.language_model.embed_tokens.weight", DType::BF16)?;
        expect_tensor(tensors, "lm_head.weight", DType::BF16)?;

        let mut compressed_linears = Vec::new();
        for layer in 0..config.text.n_layers {
            let layer_prefix = format!("model.language_model.layers.{layer}");
            let layer_linears: Vec<&str> = match config.text.layer_types[layer] {
                LayerKind::LinearAttention => vec![
                    "linear_attn.in_proj_qkv",
                    "linear_attn.in_proj_z",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                ],
                LayerKind::FullAttention => vec![
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                ],
            };
            for name in layer_linears {
                let prefix = format!("{layer_prefix}.{name}");
                compressed_linears.push(read_compressed_linear(tensors, &prefix)?);
            }
            if matches!(config.text.layer_types[layer], LayerKind::LinearAttention) {
                let dense_out = format!("{layer_prefix}.linear_attn.out_proj.weight");
                if tensors.contains_key(&dense_out) {
                    expect_tensor(tensors, &dense_out, DType::BF16)?;
                } else {
                    compressed_linears.push(read_compressed_linear(
                        tensors,
                        &format!("{layer_prefix}.linear_attn.out_proj"),
                    )?);
                }
            }

            expect_tensor(tensors, &format!("{layer_prefix}.input_layernorm.weight"), DType::BF16)?;
            expect_tensor(tensors, &format!("{layer_prefix}.post_attention_layernorm.weight"), DType::BF16)?;
            match config.text.layer_types[layer] {
                LayerKind::LinearAttention => {
                    expect_tensor(tensors, &format!("{layer_prefix}.linear_attn.in_proj_a.weight"), DType::BF16)?;
                    expect_tensor(tensors, &format!("{layer_prefix}.linear_attn.in_proj_b.weight"), DType::BF16)?;
                    expect_tensor(tensors, &format!("{layer_prefix}.linear_attn.conv1d.weight"), DType::BF16)?;
                    expect_tensor(tensors, &format!("{layer_prefix}.linear_attn.A_log"), DType::BF16)?;
                    expect_tensor(tensors, &format!("{layer_prefix}.linear_attn.dt_bias"), DType::BF16)?;
                    expect_tensor(tensors, &format!("{layer_prefix}.linear_attn.norm.weight"), DType::BF16)?;
                }
                LayerKind::FullAttention => {
                    expect_tensor(tensors, &format!("{layer_prefix}.self_attn.q_norm.weight"), DType::BF16)?;
                    expect_tensor(tensors, &format!("{layer_prefix}.self_attn.k_norm.weight"), DType::BF16)?;
                }
            }
        }
        expect_tensor(tensors, "model.language_model.norm.weight", DType::BF16)?;

        Ok(Self {
            token_embedding: "model.language_model.embed_tokens.weight".into(),
            lm_head: "lm_head.weight".into(),
            compressed_linears,
        })
    }
}

fn read_compressed_linear(
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
) -> Result<CompressedLinearWeight> {
    let packed = expect_tensor(tensors, &format!("{prefix}.weight_packed"), DType::I32)?;
    let scale = expect_tensor(tensors, &format!("{prefix}.weight_scale"), DType::BF16)?;
    let zero_point = expect_tensor(tensors, &format!("{prefix}.weight_zero_point"), DType::I32)?;
    let weight_shape = expect_tensor(tensors, &format!("{prefix}.weight_shape"), DType::I64)?;
    let shape = weight_shape.as_i64()?;
    if shape.len() != 2 || shape.iter().any(|&dim| dim <= 0) {
        return Err(Error::Other(format!(
            "{prefix}.weight_shape must be [rows, cols], got {shape:?}"
        )));
    }
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    let packed_shape = packed.shape().dims().to_vec();
    let scale_shape = scale.shape().dims().to_vec();
    let zero_point_shape = zero_point.shape().dims().to_vec();
    if packed_shape != [rows, cols.div_ceil(8)] {
        return Err(Error::Other(format!(
            "{prefix}.weight_packed shape {packed_shape:?} does not match logical [{rows}, {cols}]"
        )));
    }
    if scale_shape.len() != 2 || scale_shape[0] != rows {
        return Err(Error::Other(format!(
            "{prefix}.weight_scale shape {scale_shape:?} does not start with rows {rows}"
        )));
    }
    if zero_point_shape.len() != 2 || zero_point_shape[1] != scale_shape[1] {
        return Err(Error::Other(format!(
            "{prefix}.weight_zero_point shape {zero_point_shape:?} does not match scale groups {scale_shape:?}"
        )));
    }
    let padded_rows = rows.div_ceil(64) * 64;
    let group_size = if scale_shape[1] == 0 { 0 } else { cols / scale_shape[1] };
    let eligible = scale_shape[1] != 0
        && cols % scale_shape[1] == 0
        && group_size == 32
        && cols % 128 == 0;
    let padded_rows = if eligible { padded_rows } else { rows };
    let packed_bytes = padded_rows * cols.div_ceil(8) * 4;
    let scale_bytes = padded_rows * scale_shape[1] * 2;
    let zero_point_bytes = padded_rows.div_ceil(8) * scale_shape[1] * 4;
    let source_total_bytes = packed.size_in_bytes() + scale.size_in_bytes()
        + zero_point.size_in_bytes();
    Ok(CompressedLinearWeight {
        prefix: prefix.to_owned(),
        rows,
        cols,
        packed_shape,
        scale_shape: scale_shape.clone(),
        zero_point_shape,
        kernel_layout: W4LayoutMetadata {
            layout: if eligible {
                W4KernelLayout::RepackedN64K16V1
            } else {
                W4KernelLayout::RawCompressedTensors
            },
            version: u32::from(eligible),
            logical_rows: rows,
            logical_cols: cols,
            padded_rows,
            padded_cols: cols,
            group_size,
            groups: scale_shape[1],
            packed_bytes,
            scale_bytes,
            zero_point_bytes,
            total_bytes: packed_bytes + scale_bytes + zero_point_bytes,
            source_total_bytes,
        },
    })
}

fn expect_tensor<'a>(
    tensors: &'a HashMap<String, Tensor>,
    name: &str,
    dtype: DType,
) -> Result<&'a Tensor> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| Error::Other(format!("qwen3_5 weights: missing {name}")))?;
    if tensor.dtype() != dtype {
        return Err(Error::Other(format!(
            "qwen3_5 weights: {name} must be {dtype}, got {}",
            tensor.dtype()
        )));
    }
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::{Device, Shape};
    use half::bf16;

    fn config() -> Qwen35Config {
        Qwen35Config::from_json_str(
            r#"{
                "model_type": "qwen3_5",
                "text_config": {
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 8,
                    "vocab_size": 64,
                    "max_position_embeddings": 128,
                    "rms_norm_eps": 1e-6,
                    "partial_rotary_factor": 0.25,
                    "full_attention_interval": 2,
                    "attn_output_gate": true,
                    "output_gate_type": "swish",
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 8,
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                    "linear_value_head_dim": 8,
                    "layer_types": ["linear_attention", "full_attention"],
                    "rope_parameters": {"mrope_interleaved": true, "mrope_section": [1, 1, 0], "rope_theta": 10000}
                }
            }"#,
        )
        .unwrap()
    }

    fn raw_i32(shape: &[usize], values: &[i32]) -> Tensor {
        Tensor::from_raw(
            Shape::from(shape.to_vec()),
            DType::I32,
            Device::Cpu,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap()
    }

    fn raw_i64(shape: &[usize], values: &[i64]) -> Tensor {
        Tensor::from_raw(
            Shape::from(shape.to_vec()),
            DType::I64,
            Device::Cpu,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap()
    }

    fn add_bf16(tensors: &mut HashMap<String, Tensor>, name: impl Into<String>, shape: Vec<usize>) {
        let n = shape.iter().product();
        tensors.insert(
            name.into(),
            Tensor::from_bf16(shape, &vec![bf16::ZERO; n]).unwrap(),
        );
    }

    fn add_linear(tensors: &mut HashMap<String, Tensor>, prefix: &str) {
        tensors.insert(format!("{prefix}.weight_packed"), raw_i32(&[16, 2], &[0; 32]));
        add_bf16(tensors, format!("{prefix}.weight_scale"), vec![16, 1]);
        tensors.insert(format!("{prefix}.weight_zero_point"), raw_i32(&[2, 1], &[0; 2]));
        tensors.insert(format!("{prefix}.weight_shape"), raw_i64(&[2], &[16, 16]));
    }

    #[test]
    fn validates_hybrid_compressed_weight_map() {
        let config = config();
        let mut tensors = HashMap::new();
        add_bf16(&mut tensors, "model.language_model.embed_tokens.weight", vec![64, 16]);
        add_bf16(&mut tensors, "lm_head.weight", vec![64, 16]);
        add_bf16(&mut tensors, "model.language_model.norm.weight", vec![16]);
        for layer in 0..2 {
            let p = format!("model.language_model.layers.{layer}");
            add_bf16(&mut tensors, format!("{p}.input_layernorm.weight"), vec![16]);
            add_bf16(&mut tensors, format!("{p}.post_attention_layernorm.weight"), vec![16]);
        }
        add_bf16(&mut tensors, "model.language_model.layers.0.linear_attn.norm.weight", vec![16]);
        for name in [
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            add_linear(&mut tensors, &format!("model.language_model.layers.0.{name}"));
        }
        for name in [
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.out_proj.weight",
            "linear_attn.conv1d.weight",
            "linear_attn.A_log",
            "linear_attn.dt_bias",
        ] {
            add_bf16(&mut tensors, format!("model.language_model.layers.0.{name}"), vec![16]);
        }
        add_bf16(&mut tensors, "model.language_model.layers.1.self_attn.q_norm.weight", vec![16]);
        add_bf16(&mut tensors, "model.language_model.layers.1.self_attn.k_norm.weight", vec![16]);
        for name in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            add_linear(&mut tensors, &format!("model.language_model.layers.1.{name}"));
        }

        let manifest = Qwen35WeightsManifest::from_tensors(&config, &tensors).unwrap();
        assert_eq!(manifest.compressed_linears.len(), 12);
        assert!(manifest
            .compressed_linears
            .iter()
            .any(|linear| linear.prefix.ends_with("linear_attn.in_proj_qkv")));
        assert!(manifest
            .compressed_linears
            .iter()
            .any(|linear| linear.prefix.ends_with("linear_attn.in_proj_z")));
        assert!(manifest.compressed_linears.iter().all(|linear| {
            linear.kernel_layout.layout == W4KernelLayout::RawCompressedTensors
                && linear.kernel_layout.additional_device_bytes() == 0
        }));
    }
}

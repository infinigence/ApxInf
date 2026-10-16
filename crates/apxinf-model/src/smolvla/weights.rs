use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{Backend, DType, Device, Error, Result, Tensor};
use half::{bf16, f16};

use super::SmolVlaConfig;

#[derive(Debug)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

#[derive(Debug)]
pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

#[derive(Debug)]
pub struct VisionLayer {
    pub norm1: LayerNorm,
    pub qkv: Linear,
    pub attention_output: Linear,
    pub norm2: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

#[derive(Debug)]
pub struct TextLayer {
    pub input_norm: Tensor,
    pub qkv: Linear,
    pub attention_output: Linear,
    pub post_norm: Tensor,
    pub gate_up: Linear,
    pub down: Linear,
}

#[derive(Debug)]
pub struct ExpertSelfLayer {
    pub input_norm: Tensor,
    pub qkv: Linear,
    pub attention_output: Linear,
    pub post_norm: Tensor,
    pub gate_up: Linear,
    pub down: Linear,
}

#[derive(Debug)]
pub struct ExpertCrossLayer {
    pub input_norm: Tensor,
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub attention_output: Linear,
    pub post_norm: Tensor,
    pub gate_up: Linear,
    pub down: Linear,
}

#[derive(Debug)]
pub struct SmolVlaWeights {
    pub patch_embedding: Linear,
    pub position_embedding: Tensor,
    pub vision_layers: Vec<VisionLayer>,
    pub vision_post_norm: LayerNorm,
    pub connector: Linear,
    pub token_embedding: Tensor,
    pub text_layers: Vec<TextLayer>,
    pub text_norm: Tensor,
    pub expert_layers: Vec<ExpertLayer>,
    pub expert_norm: Tensor,
    pub state_projection: Linear,
    pub action_in: Linear,
    pub action_time_in: Linear,
    pub action_time_out: Linear,
    pub action_out: Linear,
}

#[derive(Debug)]
pub enum ExpertLayer {
    SelfAttention(ExpertSelfLayer),
    CrossAttention(ExpertCrossLayer),
}

impl SmolVlaWeights {
    pub fn from_safetensors(config: &SmolVlaConfig, path: &Path) -> Result<Self> {
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|error| Error::Other(format!("load SmolVLA safetensors: {error}")))?;
        let prefix = "model.vlm_with_expert";
        let vision_prefix = format!("{prefix}.vlm.model.vision_model");
        let text_prefix = format!("{prefix}.vlm.model.text_model");
        let expert_prefix = format!("{prefix}.lm_expert");

        let patch = tensor(&tensors, &format!("{vision_prefix}.embeddings.patch_embedding.weight"))?;
        let patch_bias = tensor(&tensors, &format!("{vision_prefix}.embeddings.patch_embedding.bias"))?;
        let patch_weight = patch.reshape(vec![config.vision_width, 768])?;
        let position_embedding = tensor(&tensors, &format!("{vision_prefix}.embeddings.position_embedding.weight"))?;
        if position_embedding.shape().dims() != [config.patches_per_view(), config.vision_width] {
            return Err(Error::Other("SmolVLA vision position embedding mismatch".into()));
        }

        let mut vision_layers = Vec::with_capacity(config.vision_depth);
        for index in 0..config.vision_depth {
            let root = format!("{vision_prefix}.encoder.layers.{index}");
            let q = tensor(&tensors, &format!("{root}.self_attn.q_proj.weight"))?;
            let k = tensor(&tensors, &format!("{root}.self_attn.k_proj.weight"))?;
            let v = tensor(&tensors, &format!("{root}.self_attn.v_proj.weight"))?;
            let q_bias = tensor(&tensors, &format!("{root}.self_attn.q_proj.bias"))?;
            let k_bias = tensor(&tensors, &format!("{root}.self_attn.k_proj.bias"))?;
            let v_bias = tensor(&tensors, &format!("{root}.self_attn.v_proj.bias"))?;
            vision_layers.push(VisionLayer {
                norm1: LayerNorm::new(
                    tensor(&tensors, &format!("{root}.layer_norm1.weight"))?,
                    Some(tensor(&tensors, &format!("{root}.layer_norm1.bias"))?),
                ),
                qkv: Linear::pack(&[q, k, v], &[q_bias, k_bias, v_bias])?,
                attention_output: Linear::new(
                    tensor(&tensors, &format!("{root}.self_attn.out_proj.weight"))?,
                    Some(tensor(&tensors, &format!("{root}.self_attn.out_proj.bias"))?),
                )?,
                norm2: LayerNorm::new(
                    tensor(&tensors, &format!("{root}.layer_norm2.weight"))?,
                    Some(tensor(&tensors, &format!("{root}.layer_norm2.bias"))?),
                ),
                fc1: Linear::new(
                    tensor(&tensors, &format!("{root}.mlp.fc1.weight"))?,
                    Some(tensor(&tensors, &format!("{root}.mlp.fc1.bias"))?),
                )?,
                fc2: Linear::new(
                    tensor(&tensors, &format!("{root}.mlp.fc2.weight"))?,
                    Some(tensor(&tensors, &format!("{root}.mlp.fc2.bias"))?),
                )?,
            });
        }

        let mut text_layers = Vec::with_capacity(config.language_depth);
        let mut expert_layers = Vec::with_capacity(config.language_depth);
        for index in 0..config.language_depth {
            let root = format!("{text_prefix}.layers.{index}");
            let expert_root = format!("{expert_prefix}.layers.{index}");
            let q = tensor(&tensors, &format!("{root}.self_attn.q_proj.weight"))?;
            let k = tensor(&tensors, &format!("{root}.self_attn.k_proj.weight"))?;
            let v = tensor(&tensors, &format!("{root}.self_attn.v_proj.weight"))?;
            let gate = tensor(&tensors, &format!("{root}.mlp.gate_proj.weight"))?;
            let up = tensor(&tensors, &format!("{root}.mlp.up_proj.weight"))?;
            text_layers.push(TextLayer {
                input_norm: tensor(&tensors, &format!("{root}.input_layernorm.weight"))?,
                qkv: Linear::pack(&[q, k, v], &[])?,
                attention_output: Linear::new(
                    tensor(&tensors, &format!("{root}.self_attn.o_proj.weight"))?,
                    None,
                )?,
                post_norm: tensor(&tensors, &format!("{root}.post_attention_layernorm.weight"))?,
                gate_up: Linear::pack(&[gate, up], &[])?,
                down: Linear::new(
                    tensor(&tensors, &format!("{root}.mlp.down_proj.weight"))?,
                    None,
                )?,
            });

            let expert_gate = tensor(&tensors, &format!("{expert_root}.mlp.gate_proj.weight"))?;
            let expert_up = tensor(&tensors, &format!("{expert_root}.mlp.up_proj.weight"))?;
            let common = ExpertCommon {
                input_norm: tensor(&tensors, &format!("{expert_root}.input_layernorm.weight"))?,
                attention_output: Linear::new(
                    tensor(&tensors, &format!("{expert_root}.self_attn.o_proj.weight"))?,
                    None,
                )?,
                post_norm: tensor(&tensors, &format!("{expert_root}.post_attention_layernorm.weight"))?,
                gate_up: Linear::pack(&[expert_gate, expert_up], &[])?,
                down: Linear::new(
                    tensor(&tensors, &format!("{expert_root}.mlp.down_proj.weight"))?,
                    None,
                )?,
            };
            if index % 2 == 0 {
                let q = tensor(&tensors, &format!("{expert_root}.self_attn.q_proj.weight"))?;
                let k = tensor(&tensors, &format!("{expert_root}.self_attn.k_proj.weight"))?;
                let v = tensor(&tensors, &format!("{expert_root}.self_attn.v_proj.weight"))?;
                expert_layers.push(ExpertLayer::SelfAttention(ExpertSelfLayer {
                    input_norm: common.input_norm,
                    qkv: Linear::pack(&[q, k, v], &[])?,
                    attention_output: common.attention_output,
                    post_norm: common.post_norm,
                    gate_up: common.gate_up,
                    down: common.down,
                }));
            } else {
                let k = tensor(&tensors, &format!("{expert_root}.self_attn.k_proj.weight"))?;
                let v = tensor(&tensors, &format!("{expert_root}.self_attn.v_proj.weight"))?;
                expert_layers.push(ExpertLayer::CrossAttention(ExpertCrossLayer {
                    input_norm: common.input_norm,
                    query: Linear::new(
                        tensor(&tensors, &format!("{expert_root}.self_attn.q_proj.weight"))?,
                        None,
                    )?,
                    key: Linear::new(k, None)?,
                    value: Linear::new(v, None)?,
                    attention_output: common.attention_output,
                    post_norm: common.post_norm,
                    gate_up: common.gate_up,
                    down: common.down,
                }));
            }
        }

        Ok(Self {
            patch_embedding: Linear::new(patch_weight, Some(patch_bias))?,
            position_embedding,
            vision_layers,
            vision_post_norm: LayerNorm::new(
                tensor(&tensors, &format!("{vision_prefix}.post_layernorm.weight"))?,
                Some(tensor(&tensors, &format!("{vision_prefix}.post_layernorm.bias"))?),
            ),
            connector: Linear::new(
                tensor(&tensors, &format!("{prefix}.vlm.model.connector.modality_projection.proj.weight"))?,
                None,
            )?,
            token_embedding: tensor(&tensors, &format!("{text_prefix}.embed_tokens.weight"))?,
            text_layers,
            text_norm: tensor(&tensors, &format!("{text_prefix}.norm.weight"))?,
            expert_layers,
            expert_norm: tensor(&tensors, &format!("{expert_prefix}.norm.weight"))?,
            state_projection: Linear::new(
                tensor(&tensors, "model.state_proj.weight")?,
                Some(tensor(&tensors, "model.state_proj.bias")?),
            )?,
            action_in: Linear::new(
                tensor(&tensors, "model.action_in_proj.weight")?,
                Some(tensor(&tensors, "model.action_in_proj.bias")?),
            )?,
            action_time_in: Linear::new(
                tensor(&tensors, "model.action_time_mlp_in.weight")?,
                Some(tensor(&tensors, "model.action_time_mlp_in.bias")?),
            )?,
            action_time_out: Linear::new(
                tensor(&tensors, "model.action_time_mlp_out.weight")?,
                Some(tensor(&tensors, "model.action_time_mlp_out.bias")?),
            )?,
            action_out: Linear::new(
                tensor(&tensors, "model.action_out_proj.weight")?,
                Some(tensor(&tensors, "model.action_out_proj.bias")?),
            )?,
        })
    }

    pub fn upload(
        self,
        backend: &dyn Backend,
        _config: &SmolVlaConfig,
        use_fp16_gemm: bool,
    ) -> Result<Self> {
        Ok(Self {
            patch_embedding: self.patch_embedding.upload(backend, use_fp16_gemm)?,
            position_embedding: upload_field(self.position_embedding, backend, use_fp16_gemm)?,
            vision_layers: self
                .vision_layers
                .into_iter()
                .map(|layer| layer.upload(backend, use_fp16_gemm))
                .collect::<Result<Vec<_>>>()?,
            vision_post_norm: self.vision_post_norm.upload(backend, use_fp16_gemm)?,
            connector: self.connector.upload(backend, use_fp16_gemm)?,
            token_embedding: upload_field(self.token_embedding, backend, use_fp16_gemm)?,
            text_layers: self
                .text_layers
                .into_iter()
                .map(|layer| layer.upload(backend, use_fp16_gemm))
                .collect::<Result<Vec<_>>>()?,
            text_norm: upload_field(self.text_norm, backend, use_fp16_gemm)?,
            expert_layers: self
                .expert_layers
                .into_iter()
                .map(|layer| layer.upload(backend, use_fp16_gemm))
                .collect::<Result<Vec<_>>>()?,
            expert_norm: upload_field(self.expert_norm, backend, use_fp16_gemm)?,
            state_projection: self.state_projection.upload(backend, use_fp16_gemm)?,
            action_in: self.action_in.upload(backend, use_fp16_gemm)?,
            action_time_in: self.action_time_in.upload(backend, use_fp16_gemm)?,
            action_time_out: self.action_time_out.upload(backend, use_fp16_gemm)?,
            action_out: self.action_out.upload(backend, use_fp16_gemm)?,
        })
    }
}

#[derive(Debug)]
struct ExpertCommon {
    input_norm: Tensor,
    attention_output: Linear,
    post_norm: Tensor,
    gate_up: Linear,
    down: Linear,
}

impl Linear {
    fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self {
            weight: transpose_bf16(weight)?,
            bias,
        })
    }

    fn pack(weights: &[Tensor], biases: &[Tensor]) -> Result<Self> {
        let packed = concat_columns(weights)?;
        let bias = if biases.is_empty() {
            None
        } else {
            Some(concat_rows(biases)?)
        };
        Self::new(packed, bias)
    }

    fn upload(self, backend: &dyn Backend, use_fp16_gemm: bool) -> Result<Self> {
        Ok(Self {
            weight: if use_fp16_gemm {
                upload_f16(self.weight, backend)?
            } else {
                upload(self.weight, backend)?
            },
            bias: self
                .bias
                .map(|bias| upload_field(bias, backend, use_fp16_gemm))
                .transpose()?,
        })
    }
}

impl LayerNorm {
    fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    fn upload(self, backend: &dyn Backend, use_fp16_gemm: bool) -> Result<Self> {
        Ok(Self {
            weight: upload_field(self.weight, backend, use_fp16_gemm)?,
            bias: self
                .bias
                .map(|bias| upload_field(bias, backend, use_fp16_gemm))
                .transpose()?,
        })
    }
}

macro_rules! upload_impl {
    ($name:ident { $($field:ident),+ }) => {
        impl $name {
            fn upload(self, backend: &dyn Backend, use_fp16_gemm: bool) -> Result<Self> {
                Ok(Self { $($field: upload_field(self.$field, backend, use_fp16_gemm)?),+ })
            }
        }
    };
}

trait UploadField {
    fn upload_field(
        self,
        backend: &dyn Backend,
        use_fp16_gemm: bool,
    ) -> Result<Self>
    where
        Self: Sized;
}

impl UploadField for Tensor {
    fn upload_field(
        self,
        backend: &dyn Backend,
        use_fp16_gemm: bool,
    ) -> Result<Self> {
        if use_fp16_gemm {
            upload_f16(self, backend)
        } else {
            upload(self, backend)
        }
    }
}

impl UploadField for Linear {
    fn upload_field(
        self,
        backend: &dyn Backend,
        use_fp16_gemm: bool,
    ) -> Result<Self> {
        self.upload(backend, use_fp16_gemm)
    }
}

impl UploadField for LayerNorm {
    fn upload_field(
        self,
        backend: &dyn Backend,
        use_fp16_gemm: bool,
    ) -> Result<Self> {
        self.upload(backend, use_fp16_gemm)
    }
}

fn upload_field<T: UploadField>(
    field: T,
    backend: &dyn Backend,
    use_fp16_gemm: bool,
) -> Result<T> {
    field.upload_field(backend, use_fp16_gemm)
}

upload_impl!(VisionLayer { norm1, qkv, attention_output, norm2, fc1, fc2 });
upload_impl!(TextLayer { input_norm, qkv, attention_output, post_norm, gate_up, down });
upload_impl!(ExpertSelfLayer { input_norm, qkv, attention_output, post_norm, gate_up, down });
upload_impl!(ExpertCrossLayer { input_norm, query, key, value, attention_output, post_norm, gate_up, down });

impl ExpertLayer {
    fn upload(self, backend: &dyn Backend, use_fp16_gemm: bool) -> Result<Self> {
        match self {
            Self::SelfAttention(layer) => {
                Ok(Self::SelfAttention(layer.upload(backend, use_fp16_gemm)?))
            }
            Self::CrossAttention(layer) => {
                Ok(Self::CrossAttention(layer.upload(backend, use_fp16_gemm)?))
            }
        }
    }
}

fn tensor(tensors: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .get(name)
        .cloned()
        .ok_or_else(|| Error::Other(format!("SmolVLA checkpoint is missing tensor `{name}`")))
}

fn upload(tensor: Tensor, backend: &dyn Backend) -> Result<Tensor> {
    let tensor = to_bf16(tensor)?;
    if tensor.device() == Device::Cpu {
        backend.to_device(&tensor)
    } else {
        Ok(tensor)
    }
}

fn to_bf16(tensor: Tensor) -> Result<Tensor> {
    if tensor.dtype() == DType::BF16 {
        return Ok(tensor);
    }
    if tensor.dtype() != DType::F32 {
        return Err(Error::Other(format!(
            "SmolVLA expects BF16 or F32 weights, got {}",
            tensor.dtype()
        )));
    }
    let values = tensor
        .to_f32_vec()?
        .into_iter()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    Tensor::from_bf16(tensor.shape().dims(), &values)
}

fn upload_f16(tensor: Tensor, backend: &dyn Backend) -> Result<Tensor> {
    let tensor = to_bf16(tensor)?;
    let values = tensor
        .as_bf16()?
        .iter()
        .map(|value| f16::from_f32(value.to_f32()))
        .collect::<Vec<_>>();
    let tensor = Tensor::from_f16(tensor.shape().dims(), &values)?;
    if tensor.device() == Device::Cpu {
        backend.to_device(&tensor)
    } else {
        Ok(tensor)
    }
}

fn transpose_bf16(tensor: Tensor) -> Result<Tensor> {
    let tensor = to_bf16(tensor)?;
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "SmolVLA linear weight must be 2D, got {dims:?}"
        )));
    }
    if tensor.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "SmolVLA linear weight must be BF16, got {}",
            tensor.dtype()
        )));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let source = tensor.as_bf16()?;
    let mut target = vec![bf16::from_f32(0.0); source.len()];
    for row in 0..rows {
        for col in 0..cols {
            target[col * rows + row] = source[row * cols + col];
        }
    }
    Tensor::from_bf16(vec![cols, rows], &target)
}

fn concat_columns(tensors: &[Tensor]) -> Result<Tensor> {
    concat(tensors, false)
}

fn concat_rows(tensors: &[Tensor]) -> Result<Tensor> {
    concat(tensors, true)
}

fn concat(tensors: &[Tensor], rows: bool) -> Result<Tensor> {
    let first = tensors.first().ok_or_else(|| Error::Other("empty tensor concat".into()))?;
    let first_dims = first.shape().dims();
    let values = tensors
        .iter()
        .map(|tensor| {
            let tensor = to_bf16(tensor.clone())?;
            let dims = tensor.shape().dims();
            let compatible = if rows {
                dims.len() == 1
            } else {
                dims.len() == 2 && dims[1] == first_dims[1]
            };
            if tensor.dtype() != DType::BF16 || !compatible {
                return Err(Error::Other("incompatible BF16 tensor concat".into()));
            }
            Ok(tensor.as_bf16()?.to_vec())
        })
        .collect::<Result<Vec<Vec<_>>>>()?;
    let flat = values.concat();
    let shape = if rows {
        vec![flat.len()]
    } else {
        vec![flat.len() / first_dims[1], first_dims[1]]
    };
    Tensor::from_bf16(shape, &flat)
}

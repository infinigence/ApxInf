//! Source-key validation and one-time device packing for Cosmos3-Edge.
//!
//! Linear matrices become `[in, out]`; convolution columns are kernel T/H/W/C
//! to match native `Unfold3d` lowering. The transformer is a Mixture-of-
//! Transformers: layer `i` carries a UND tower and a GEN tower.
use super::host::Store;
use std::path::Path;

use crate::cuda_cosmos3_edge::backend::CudaBuffer;
use apxinf_core::{Device, Error, Result, Shape, Tensor};

const WIDTH: usize = 2048;
const KV_WIDTH: usize = 1024;
const MLP: usize = 9216;
const HEAD_DIM: usize = 128;
const LAYERS: usize = 28;
const PATCH_LATENT: usize = 192;

pub(in crate::cuda_cosmos3_edge) struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}
pub(in crate::cuda_cosmos3_edge) struct Conv {
    pub linear: Linear,
    pub kernel: [usize; 3],
    pub input_channels: usize,
    pub output_channels: usize,
}

pub(in crate::cuda_cosmos3_edge) struct UndLayer {
    pub input_norm: Tensor,
    pub post_norm: Tensor,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub mlp_up: Linear,
    pub mlp_down: Linear,
    pub k_norm_for_gen: Tensor,
}
pub(in crate::cuda_cosmos3_edge) struct GenLayer {
    pub input_norm: Tensor,
    pub post_norm: Tensor,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub mlp_up: Linear,
    pub mlp_down: Linear,
    pub norm_q: Tensor,
    pub norm_k: Tensor,
}
pub(in crate::cuda_cosmos3_edge) struct TimeEmbedder {
    pub l1: Linear,
    pub l2: Linear,
}
pub(in crate::cuda_cosmos3_edge) struct TransformerWeights {
    pub norm: Tensor,
    pub norm_gen: Tensor,
    pub proj_in: Linear,
    pub proj_out: Linear,
    pub time: TimeEmbedder,
    pub action_in: Linear,
    pub action_out: Linear,
    pub action_modality_embed: Tensor,
    pub und: Vec<UndLayer>,
    pub gen: Vec<GenLayer>,
}

pub(in crate::cuda_cosmos3_edge) struct VaeResidual {
    pub norm1: Tensor,
    pub conv1: Conv,
    pub norm2: Tensor,
    pub conv2: Conv,
    pub shortcut: Option<Conv>,
}
pub(in crate::cuda_cosmos3_edge) struct VaeDownsample {
    pub spatial: Conv,
    pub temporal: Option<Conv>,
}
pub(in crate::cuda_cosmos3_edge) struct VaeWeights {
    pub input: Conv,
    pub down_blocks: Vec<[VaeResidual; 2]>,
    pub downsample: Vec<VaeDownsample>,
    pub avg_shortcut: Vec<Conv>,
    pub middle_before: VaeResidual,
    pub attention_norm: Tensor,
    pub attention_qkv: Conv,
    pub attention_output: Conv,
    pub middle_after: VaeResidual,
    pub output_norm: Tensor,
    pub output: Conv,
    pub mean_projection: Conv,
}

struct Reader {
    host: Store,
    device: usize,
}
impl Reader {
    fn new(path: &Path, device: usize) -> Result<Self> {
        Ok(Self {
            host: Store::new(path)?,
            device,
        })
    }
    fn vector(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let tensor = self.host.take(name, shape)?;
        upload(&tensor.reshape(vec![tensor.numel()])?, self.device)
    }
    fn linear(&mut self, prefix: &str, output: usize, input: usize) -> Result<Linear> {
        let tensor = self
            .host
            .take(&format!("{prefix}.weight"), &[output, input])?;
        Ok(Linear {
            weight: upload(&transpose_bf16(&tensor)?, self.device)?,
            bias: None,
        })
    }
    fn linear_bias(&mut self, prefix: &str, output: usize, input: usize) -> Result<Linear> {
        let tensor = self
            .host
            .take(&format!("{prefix}.weight"), &[output, input])?;
        Ok(Linear {
            weight: upload(&transpose_bf16(&tensor)?, self.device)?,
            bias: Some(self.vector(&format!("{prefix}.bias"), &[output])?),
        })
    }
    fn conv(
        &mut self,
        prefix: &str,
        output: usize,
        input: usize,
        kernel: &[usize],
    ) -> Result<Conv> {
        let shape: Vec<_> = [output, input]
            .into_iter()
            .chain(kernel.iter().copied())
            .collect();
        let tensor = self.host.take(&format!("{prefix}.weight"), &shape)?;
        let packed = pack_conv(&tensor)?;
        let mut k3 = [1, 1, 1];
        k3[3 - kernel.len()..].copy_from_slice(kernel);
        Ok(Conv {
            linear: Linear {
                weight: upload(&packed, self.device)?,
                bias: Some(self.vector(&format!("{prefix}.bias"), &[output])?),
            },
            kernel: k3,
            input_channels: input,
            output_channels: output,
        })
    }
    /// Wan2.2 residual block: `{prefix}.norm1.gamma`, `conv1`, ...
    fn w22_residual(&mut self, prefix: &str, input: usize, output: usize) -> Result<VaeResidual> {
        Ok(VaeResidual {
            norm1: self.vector(&format!("{prefix}.norm1.gamma"), &[input, 1, 1, 1])?,
            conv1: self.conv(&format!("{prefix}.conv1"), output, input, &[3, 3, 3])?,
            norm2: self.vector(&format!("{prefix}.norm2.gamma"), &[output, 1, 1, 1])?,
            conv2: self.conv(&format!("{prefix}.conv2"), output, output, &[3, 3, 3])?,
            shortcut: if input == output {
                None
            } else {
                Some(self.conv(
                    &format!("{prefix}.conv_shortcut"),
                    output,
                    input,
                    &[1, 1, 1],
                )?)
            },
        })
    }
}

impl TransformerWeights {
    pub fn load(path: &Path, device: usize) -> Result<Self> {
        let mut r = Reader::new(path, device)?;
        let mut und = Vec::with_capacity(LAYERS);
        let mut gen = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("layers.{i}");
            und.push(UndLayer {
                input_norm: r.vector(&format!("{p}.input_layernorm.weight"), &[WIDTH])?,
                post_norm: r.vector(&format!("{p}.post_attention_layernorm.weight"), &[WIDTH])?,
                q: r.linear(&format!("{p}.self_attn.to_q"), WIDTH, WIDTH)?,
                k: r.linear(&format!("{p}.self_attn.to_k"), KV_WIDTH, WIDTH)?,
                v: r.linear(&format!("{p}.self_attn.to_v"), KV_WIDTH, WIDTH)?,
                o: r.linear(&format!("{p}.self_attn.to_out"), WIDTH, WIDTH)?,
                mlp_up: r.linear(&format!("{p}.mlp.up_proj"), MLP, WIDTH)?,
                mlp_down: r.linear(&format!("{p}.mlp.down_proj"), WIDTH, MLP)?,
                k_norm_for_gen: r.vector(
                    &format!("{p}.self_attn.k_norm_und_for_gen.weight"),
                    &[HEAD_DIM],
                )?,
            });
            gen.push(GenLayer {
                input_norm: r.vector(&format!("{p}.input_layernorm_moe_gen.weight"), &[WIDTH])?,
                post_norm: r.vector(
                    &format!("{p}.post_attention_layernorm_moe_gen.weight"),
                    &[WIDTH],
                )?,
                q: r.linear(&format!("{p}.self_attn.add_q_proj"), WIDTH, WIDTH)?,
                k: r.linear(&format!("{p}.self_attn.add_k_proj"), KV_WIDTH, WIDTH)?,
                v: r.linear(&format!("{p}.self_attn.add_v_proj"), KV_WIDTH, WIDTH)?,
                o: r.linear(&format!("{p}.self_attn.to_add_out"), WIDTH, WIDTH)?,
                mlp_up: r.linear(&format!("{p}.mlp_moe_gen.up_proj"), MLP, WIDTH)?,
                mlp_down: r.linear(&format!("{p}.mlp_moe_gen.down_proj"), WIDTH, MLP)?,
                norm_q: r.vector(&format!("{p}.self_attn.norm_added_q.weight"), &[HEAD_DIM])?,
                norm_k: r.vector(&format!("{p}.self_attn.norm_added_k.weight"), &[HEAD_DIM])?,
            });
        }
        let result = Self {
            norm: r.vector("norm.weight", &[WIDTH])?,
            norm_gen: r.vector("norm_moe_gen.weight", &[WIDTH])?,
            proj_in: r.linear_bias("proj_in", WIDTH, PATCH_LATENT)?,
            proj_out: r.linear_bias("proj_out", PATCH_LATENT, WIDTH)?,
            time: TimeEmbedder {
                l1: r.linear_bias("time_embedder.linear_1", WIDTH, 256)?,
                l2: r.linear_bias("time_embedder.linear_2", WIDTH, WIDTH)?,
            },
            action_in: r.linear_bias("action_in", WIDTH, 64)?,
            action_out: r.linear_bias("action_out", 64, WIDTH)?,
            action_modality_embed: r.vector("action_modality_embed", &[WIDTH])?,
            und,
            gen,
        };
        r.host.finish()?;
        Ok(result)
    }
}

impl VaeWeights {
    pub fn load(path: &Path, device: usize) -> Result<Self> {
        let mut r = Reader::new(path, device)?;
        let mut down_blocks = Vec::new();
        let mut downsample = Vec::new();
        let outs = [160usize, 320, 640, 640];
        let ins = [160usize, 160, 320, 640];
        for level in 0..4 {
            let first = r.w22_residual(
                &format!("encoder.down_blocks.{level}.resnets.0"),
                ins[level],
                outs[level],
            )?;
            let second = r.w22_residual(
                &format!("encoder.down_blocks.{level}.resnets.1"),
                outs[level],
                outs[level],
            )?;
            down_blocks.push([first, second]);
            if level < 3 {
                downsample.push(VaeDownsample {
                    spatial: r.conv(
                        &format!("encoder.down_blocks.{level}.downsampler.resample.1"),
                        outs[level],
                        outs[level],
                        &[3, 3],
                    )?,
                    temporal: if level == 0 {
                        None
                    } else {
                        Some(r.conv(
                            &format!("encoder.down_blocks.{level}.downsampler.time_conv"),
                            outs[level],
                            outs[level],
                            &[3, 1, 1],
                        )?)
                    },
                });
            }
        }
        let mut avg_shortcut = Vec::new();
        for (in_c, out_c, ft, fs) in [
            (160usize, 160usize, 1usize, 2usize),
            (160, 320, 2, 2),
            (320, 640, 2, 2),
        ] {
            avg_shortcut.push(synthesize_avg_shortcut(in_c, out_c, ft, fs, device)?);
        }
        let result = Self {
            input: r.conv("encoder.conv_in", 160, 12, &[3, 3, 3])?,
            down_blocks,
            downsample,
            avg_shortcut,
            middle_before: r.w22_residual("encoder.mid_block.resnets.0", 640, 640)?,
            attention_norm: r.vector("encoder.mid_block.attentions.0.norm.gamma", &[640, 1, 1])?,
            attention_qkv: r.conv("encoder.mid_block.attentions.0.to_qkv", 1920, 640, &[1, 1])?,
            attention_output: r.conv("encoder.mid_block.attentions.0.proj", 640, 640, &[1, 1])?,
            middle_after: r.w22_residual("encoder.mid_block.resnets.1", 640, 640)?,
            output_norm: r.vector("encoder.norm_out.gamma", &[640, 1, 1, 1])?,
            output: r.conv("encoder.conv_out", 96, 640, &[3, 3, 3])?,
            mean_projection: r.conv("quant_conv", 96, 96, &[1, 1, 1])?,
        };
        r.host.finish()?;
        Ok(result)
    }
}

pub(in crate::cuda_cosmos3_edge) fn upload(tensor: &Tensor, device: usize) -> Result<Tensor> {
    if tensor.device() != Device::Cpu {
        return Err(Error::Other("Cosmos3 upload requires host weights".into()));
    }
    let buffer = CudaBuffer::alloc(tensor.size_in_bytes(), device).map_err(Error::Cuda)?;
    let bytes = tensor
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::Other("missing host weight storage".into()))?;
    buffer.copy_from_host(bytes).map_err(Error::Cuda)?;
    buffer
        .as_tensor(tensor.shape().clone(), tensor.dtype())
        .map_err(Error::Cuda)
}

fn pack_conv(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if !(dims.len() == 4 || dims.len() == 5) || dims.contains(&0) {
        return Err(Error::Other(
            "convolution weights must be OIHW or OITHW".into(),
        ));
    }
    let (o, i, volume) = (dims[0], dims[1], dims[2..].iter().product::<usize>());
    let src = tensor.as_bf16()?;
    let mut packed = vec![half::bf16::ZERO; src.len()];
    for output in 0..o {
        for input in 0..i {
            for k in 0..volume {
                packed[(k * i + input) * o + output] = src[(output * i + input) * volume + k];
            }
        }
    }
    Tensor::from_bf16(Shape::new(vec![volume * i, o]), &packed)
}

/// Parameter-free Wan 2.2 VAE `_AvgDown3d` shortcut: a strided space-to-channel
/// average-pool expressed as a fixed-weight convolution (stride = kernel =
/// `[ft, fs, fs]`). Each output channel averages `group_size = in*ft*fs*fs/out`
/// of the unshuffled `(c, t_offset, h_offset, w_offset)` channels.
fn synthesize_avg_shortcut(
    in_c: usize,
    out_c: usize,
    ft: usize,
    fs: usize,
    device: usize,
) -> Result<Conv> {
    let volume = ft * fs * fs;
    let group_size = in_c * volume / out_c;
    let inv = half::bf16::from_f32(1.0 / group_size as f32);
    let mut packed = vec![half::bf16::ZERO; volume * in_c * out_c];
    for c in 0..in_c {
        for kt in 0..ft {
            for kh in 0..fs {
                for kw in 0..fs {
                    let flat = c * volume + kt * fs * fs + kh * fs + kw;
                    let o = flat / group_size;
                    let k = kt * fs * fs + kh * fs + kw;
                    packed[(k * in_c + c) * out_c + o] = inv;
                }
            }
        }
    }
    Ok(Conv {
        linear: Linear {
            weight: upload(
                &Tensor::from_bf16(vec![volume * in_c, out_c], &packed)?,
                device,
            )?,
            bias: None,
        },
        kernel: [ft, fs, fs],
        input_channels: in_c,
        output_channels: out_c,
    })
}

fn transpose_bf16(tensor: &Tensor) -> Result<Tensor> {
    let [rows, cols] = *tensor.shape().dims() else {
        return Err(Error::Other("linear weights must have rank two".into()));
    };
    let source = tensor.as_bf16()?;
    let mut output = vec![half::bf16::ZERO; source.len()];
    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = source[row * cols + col];
        }
    }
    Tensor::from_bf16(vec![cols, rows], &output)
}

pub(in crate::cuda_cosmos3_edge) struct Bf16Weights {
    pub transformer: TransformerWeights,
    pub vae: VaeWeights,
    pub text: Tensor,
    pub num_tasks: usize,
    pub text_tokens: usize,
}
impl Bf16Weights {
    pub fn load(path: &Path, device: usize, num_tasks: usize, text_tokens: usize) -> Result<Self> {
        let transformer = TransformerWeights::load(&path.join("model.safetensors"), device)?;
        let vae = VaeWeights::load(&path.join("vae_encoder.safetensors"), device)?;
        let mut r = Reader::new(&path.join("text.safetensors"), device)?;
        let text = upload(
            &r.host
                .take("embeddings", &[num_tasks, text_tokens, WIDTH])?
                .reshape(vec![num_tasks, text_tokens * WIDTH])?,
            device,
        )?;
        r.host.finish()?;
        Ok(Self {
            transformer,
            vae,
            text,
            num_tasks,
            text_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn linear_packing_preserves_bf16_bits() {
        let bits = [0x3f80, 0x8000, 0x3f81, 0xbf80, 0x0001, 0x4000];
        let values: Vec<_> = bits.into_iter().map(half::bf16::from_bits).collect();
        let source = Tensor::from_bf16(vec![2, 3], &values).unwrap();
        let output = transpose_bf16(&source).unwrap();
        let actual: Vec<_> = output
            .as_bf16()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(
            actual,
            [bits[0], bits[3], bits[1], bits[4], bits[2], bits[5]]
        );
    }
}

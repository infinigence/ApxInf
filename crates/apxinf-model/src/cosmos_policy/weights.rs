//! Source-key validation and one-time device packing. Linear matrices become
//! [in,out]; convolution columns are kernel T/H/W/C to match native lowering.
use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_cuda_new::CudaBuffer;

pub(super) struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

pub(super) struct Conv {
    pub linear: Linear,
    pub kernel: [usize; 3],
    pub input_channels: usize,
    pub output_channels: usize,
}

pub(super) struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub output: Linear,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

pub(super) struct Modulation {
    pub down: Linear,
    pub up: Linear,
}

pub(super) struct DitBlock {
    pub self_attention: Attention,
    pub cross_attention: Attention,
    pub mlp_up: Linear,
    pub mlp_down: Linear,
    pub self_modulation: Modulation,
    pub cross_modulation: Modulation,
    pub mlp_modulation: Modulation,
}

pub(super) struct DitWeights {
    pub patch: Linear,
    pub time_in: Linear,
    pub time_out: Linear,
    pub time_norm: Tensor,
    pub blocks: Vec<DitBlock>,
    pub final_modulation: Modulation,
    pub final_projection: Linear,
    /// Checkpoint buffers defining the exact positional frequencies.
    pub position_seq: Tensor,
    pub spatial_range: Tensor,
    pub temporal_range: Tensor,
}

struct Reader {
    tensors: HashMap<String, Tensor>,
    device: usize,
}

impl Reader {
    fn new(path: &Path, device: usize) -> Result<Self> {
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|e| Error::Other(format!("load {}: {e}", path.display())))?;
        Ok(Self { tensors, device })
    }

    fn take(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let tensor = self
            .tensors
            .remove(name)
            .ok_or_else(|| Error::Other(format!("missing Cosmos Policy weight {name}")))?;
        if tensor.dtype() != DType::BF16 || tensor.shape().dims() != shape {
            return Err(Error::Other(format!(
                "Cosmos Policy weight {name}: expected BF16 {shape:?}, got {} {:?}",
                tensor.dtype(),
                tensor.shape().dims()
            )));
        }
        if tensor.as_bf16()?.iter().any(|x| !x.to_f32().is_finite()) {
            return Err(Error::Other(format!(
                "nonfinite Cosmos Policy weight {name}"
            )));
        }
        Ok(tensor)
    }

    fn vector(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let tensor = self.take(name, shape)?;
        upload(&tensor.reshape(vec![tensor.numel()])?, self.device)
    }

    fn linear(&mut self, prefix: &str, output: usize, input: usize) -> Result<Linear> {
        let tensor = self.take(&format!("{prefix}.weight"), &[output, input])?;
        Ok(Linear {
            weight: upload(&transpose_bf16(&tensor)?, self.device)?,
            bias: None,
        })
    }

    fn modulation(&mut self, prefix: &str, outputs: usize) -> Result<Modulation> {
        Ok(Modulation {
            down: self.linear(&format!("{prefix}.1"), 256, 2048)?,
            up: self.linear(&format!("{prefix}.2"), outputs, 256)?,
        })
    }

    fn attention(&mut self, prefix: &str, context_width: usize) -> Result<Attention> {
        Ok(Attention {
            q: self.linear(&format!("{prefix}.q_proj"), 2048, 2048)?,
            k: self.linear(&format!("{prefix}.k_proj"), 2048, context_width)?,
            v: self.linear(&format!("{prefix}.v_proj"), 2048, context_width)?,
            output: self.linear(&format!("{prefix}.output_proj"), 2048, 2048)?,
            q_norm: self.vector(&format!("{prefix}.q_norm.weight"), &[128])?,
            k_norm: self.vector(&format!("{prefix}.k_norm.weight"), &[128])?,
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
        let tensor = self.take(&format!("{prefix}.weight"), &shape)?;
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

    fn residual(&mut self, prefix: &str, input: usize, output: usize) -> Result<VaeResidual> {
        Ok(VaeResidual {
            norm1: self.vector(&format!("{prefix}.residual.0.gamma"), &[input, 1, 1, 1])?,
            conv1: self.conv(&format!("{prefix}.residual.2"), output, input, &[3, 3, 3])?,
            norm2: self.vector(&format!("{prefix}.residual.3.gamma"), &[output, 1, 1, 1])?,
            conv2: self.conv(&format!("{prefix}.residual.6"), output, output, &[3, 3, 3])?,
            shortcut: if input == output {
                None
            } else {
                Some(self.conv(&format!("{prefix}.shortcut"), output, input, &[1, 1, 1])?)
            },
        })
    }

    fn finish(self) -> Result<()> {
        if !self.tensors.is_empty() {
            let mut names: Vec<_> = self.tensors.into_keys().collect();
            names.sort();
            return Err(Error::Other(format!(
                "unexpected Cosmos Policy weights: {names:?}"
            )));
        }
        Ok(())
    }
}

impl DitWeights {
    pub fn load(path: &Path, device: usize) -> Result<Self> {
        let mut r = Reader::new(path, device)?;
        let mut blocks = Vec::with_capacity(28);
        for i in 0..28 {
            let p = format!("blocks.{i}");
            blocks.push(DitBlock {
                self_attention: r.attention(&format!("{p}.self_attn"), 2048)?,
                cross_attention: r.attention(&format!("{p}.cross_attn"), 1024)?,
                mlp_up: r.linear(&format!("{p}.mlp.layer1"), 8192, 2048)?,
                mlp_down: r.linear(&format!("{p}.mlp.layer2"), 2048, 8192)?,
                self_modulation: r.modulation(&format!("{p}.adaln_modulation_self_attn"), 6144)?,
                cross_modulation: r
                    .modulation(&format!("{p}.adaln_modulation_cross_attn"), 6144)?,
                mlp_modulation: r.modulation(&format!("{p}.adaln_modulation_mlp"), 6144)?,
            });
        }
        let result = Self {
            patch: r.linear("x_embedder.proj.1", 2048, 72)?,
            time_in: r.linear("t_embedder.1.linear_1", 2048, 2048)?,
            time_out: r.linear("t_embedder.1.linear_2", 6144, 2048)?,
            time_norm: r.vector("t_embedding_norm.weight", &[2048])?,
            blocks,
            final_modulation: r.modulation("final_layer.adaln_modulation", 4096)?,
            final_projection: r.linear("final_layer.linear", 64, 2048)?,
            position_seq: r.take("pos_embedder.seq", &[128])?,
            spatial_range: r.take("pos_embedder.dim_spatial_range", &[21])?,
            temporal_range: r.take("pos_embedder.dim_temporal_range", &[22])?,
        };
        r.finish()?;
        Ok(result)
    }
}

pub(super) struct VaeResidual {
    pub norm1: Tensor,
    pub conv1: Conv,
    pub norm2: Tensor,
    pub conv2: Conv,
    pub shortcut: Option<Conv>,
}

pub(super) struct VaeDownsample {
    pub spatial: Conv,
    pub temporal: Option<Conv>,
}

pub(super) struct VaeWeights {
    pub input: Conv,
    pub down_blocks: Vec<[VaeResidual; 2]>,
    pub downsample: Vec<VaeDownsample>,
    pub middle_before: VaeResidual,
    pub attention_norm: Tensor,
    pub attention_qkv: Conv,
    pub attention_output: Conv,
    pub middle_after: VaeResidual,
    pub output_norm: Tensor,
    pub output: Conv,
    pub mean_projection: Conv,
}

impl VaeWeights {
    pub fn load(path: &Path, device: usize) -> Result<Self> {
        let mut r = Reader::new(path, device)?;
        let mut down_blocks = Vec::new();
        let mut downsample = Vec::new();
        let mut channels = 96;
        for (level, out) in [96, 192, 384, 384].into_iter().enumerate() {
            let first = r.residual(&format!("encoder.downsamples.{}", level * 3), channels, out)?;
            let second = r.residual(&format!("encoder.downsamples.{}", level * 3 + 1), out, out)?;
            down_blocks.push([first, second]);
            if level < 3 {
                downsample.push(VaeDownsample {
                    spatial: r.conv(
                        &format!("encoder.downsamples.{}.resample.1", level * 3 + 2),
                        out,
                        out,
                        &[3, 3],
                    )?,
                    temporal: if level == 0 {
                        None
                    } else {
                        Some(r.conv(
                            &format!("encoder.downsamples.{}.time_conv", level * 3 + 2),
                            out,
                            out,
                            &[3, 1, 1],
                        )?)
                    },
                });
            }
            channels = out;
        }
        let result = Self {
            input: r.conv("encoder.conv1", 96, 3, &[3, 3, 3])?,
            down_blocks,
            downsample,
            middle_before: r.residual("encoder.middle.0", 384, 384)?,
            attention_norm: r.vector("encoder.middle.1.norm.gamma", &[384, 1, 1])?,
            attention_qkv: r.conv("encoder.middle.1.to_qkv", 1152, 384, &[1, 1])?,
            attention_output: r.conv("encoder.middle.1.proj", 384, 384, &[1, 1])?,
            middle_after: r.residual("encoder.middle.2", 384, 384)?,
            output_norm: r.vector("encoder.head.0.gamma", &[384, 1, 1, 1])?,
            output: r.conv("encoder.head.2", 32, 384, &[3, 3, 3])?,
            mean_projection: r.conv("conv1", 32, 32, &[1, 1, 1])?,
        };
        r.finish()?;
        Ok(result)
    }
}

pub(super) fn upload(tensor: &Tensor, device: usize) -> Result<Tensor> {
    if tensor.device() != Device::Cpu {
        return Err(Error::Other(
            "Cosmos Policy upload requires host weights".into(),
        ));
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn linear_packing_preserves_bf16_bits() {
        let bits = [0x3f80, 0x8000, 0x3f81, 0xbf80, 0x0001, 0x4000];
        let values: Vec<_> = bits.into_iter().map(half::bf16::from_bits).collect();
        let source = Tensor::from_bf16(vec![2, 3], &values).unwrap();
        let output = transpose_bf16(&source).unwrap();
        assert_eq!(output.dtype(), DType::BF16);
        assert_eq!(output.shape().dims(), [3, 2]);
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
    #[test]
    fn convolution_packing_preserves_source_axes() {
        let source: Vec<_> = (0..72).map(|i| half::bf16::from_f32(i as f32)).collect();
        let tensor = Tensor::from_bf16(vec![2, 3, 2, 2, 3], &source).unwrap();
        let packed = pack_conv(&tensor).unwrap();
        assert_eq!(packed.shape().dims(), [36, 2]);
        for t in 0..2 {
            for y in 0..2 {
                for x in 0..3 {
                    for c in 0..3 {
                        for o in 0..2 {
                            let actual =
                                packed.as_bf16().unwrap()[(((t * 2 + y) * 3 + x) * 3 + c) * 2 + o];
                            assert_eq!(actual, source[(((o * 3 + c) * 2 + t) * 2 + y) * 3 + x]);
                        }
                    }
                }
            }
        }
    }
}

pub(super) struct Weights {
    pub dit: DitWeights,
    pub vae: VaeWeights,
    pub text: Tensor,
}
impl Weights {
    pub fn load(path: &Path, device: usize) -> Result<Self> {
        let dit = DitWeights::load(&path.join("model.safetensors"), device)?;
        let vae = VaeWeights::load(&path.join("vae_encoder.safetensors"), device)?;
        let mut r = Reader::new(&path.join("text.safetensors"), device)?;
        let text = upload(
            &r.take("embeddings", &[40, 512, 1024])?
                .reshape(vec![40, 512 * 1024])?,
            device,
        )?;
        r.finish()?;
        Ok(Self { dit, vae, text })
    }
}

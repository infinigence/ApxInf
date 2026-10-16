//! Checkpoint loading for the mflux Qwen-Image-2.1 export: affine Q4 DiT, F32
//! VAE decoder and the BF16 text encoder (read one layer at a time).
use apxinf_core::{Error, Result};
use apxinf_mlx::{Array, MlxDType, Stream};
use std::collections::{BTreeSet, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};

pub const DIM: usize = 4096;
pub const HEADS: usize = 32;
pub const HEAD_DIM: usize = 128;
pub const BLOCKS: usize = 32;
pub const MLP: usize = 12288;
pub const LATENT: usize = 64;
pub const TEXT_LAYERS: usize = 36;
pub const TEXT_KV_HEADS: usize = 8;
pub const TEXT_VOCAB: usize = 151_936;

/// One affine Q4 group-64 linear layer without bias, kept as stored.
pub struct Q4 {
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
}

impl Q4 {
    /// `nn.QuantizedLinear` without bias.
    pub fn apply(&self, x: &Array) -> Result<Array> {
        x.quantized_matmul(&self.weight, &self.scales, &self.biases, true, 64, 4)
    }
}

pub struct Block {
    pub norm_q: Array,
    pub norm_k: Array,
    pub to_q: Q4,
    pub to_k: Q4,
    pub to_v: Q4,
    pub to_out: Q4,
    pub gate: Q4,
    pub proj: Q4,
    pub out: Q4,
}

pub struct Transformer {
    pub img_in: Q4,
    pub txt_norm: Array,
    pub txt_in: Q4,
    pub txt_out: Q4,
    pub time_1: Q4,
    pub time_2: Q4,
    pub modulation: Q4,
    pub blocks: Vec<Block>,
    pub norm_out: Q4,
    pub proj_out: Q4,
}

/// `nn.Conv2d` weights `[O, KH, KW, I]` and bias `[O]`, F32.
pub struct Conv {
    pub weight: Array,
    pub bias: Array,
}

pub struct Residual {
    pub norm1: Array,
    pub conv1: Conv,
    pub norm2: Array,
    pub conv2: Conv,
    pub shortcut: Option<Conv>,
}

pub struct Up {
    pub resnets: Vec<Residual>,
    pub upsampler: Option<Conv>,
    pub in_dim: usize,
    pub out_dim: usize,
}

pub struct Decoder {
    pub post_quant: Conv,
    pub conv_in: Conv,
    pub mid: [Residual; 2],
    pub attn_norm: Array,
    pub attn_qkv: Conv,
    pub attn_proj: Conv,
    pub up: Vec<Up>,
    pub norm_out: Array,
    pub conv_out: Conv,
}

pub struct TextLayer {
    pub input_norm: Array,
    pub q: Array,
    pub k: Array,
    pub v: Array,
    pub o: Array,
    pub q_norm: Array,
    pub k_norm: Array,
    pub post_norm: Array,
    pub gate: Array,
    pub up: Array,
    pub down: Array,
}

struct Take {
    map: HashMap<String, Array>,
    what: &'static str,
}

impl Take {
    fn array(&mut self, name: &str, dtype: MlxDType, shape: &[usize]) -> Result<Array> {
        let array = self
            .map
            .remove(name)
            .ok_or_else(|| Error::Other(format!("{} is missing `{name}`", self.what)))?;
        if array.dtype() != dtype || array.shape() != shape {
            return Err(Error::Other(format!(
                "{} `{name}`: expected {dtype:?} {shape:?}, got {:?} {:?}",
                self.what,
                array.dtype(),
                array.shape()
            )));
        }
        Ok(array)
    }

    fn q4(&mut self, name: &str, out: usize, input: usize) -> Result<Q4> {
        Ok(Q4 {
            weight: self.array(&format!("{name}.weight"), MlxDType::U32, &[out, input / 8])?,
            scales: self.array(
                &format!("{name}.scales"),
                MlxDType::BF16,
                &[out, input / 64],
            )?,
            biases: self.array(
                &format!("{name}.biases"),
                MlxDType::BF16,
                &[out, input / 64],
            )?,
        })
    }

    fn conv(&mut self, name: &str, out: usize, kernel: usize, input: usize) -> Result<Conv> {
        Ok(Conv {
            weight: self.array(
                &format!("{name}.weight"),
                MlxDType::F32,
                &[out, kernel, kernel, input],
            )?,
            bias: self.array(&format!("{name}.bias"), MlxDType::F32, &[out])?,
        })
    }

    fn finish(self) -> Result<()> {
        let mut extra: Vec<_> = self.map.into_keys().collect();
        if extra.is_empty() {
            return Ok(());
        }
        extra.sort();
        extra.truncate(8);
        Err(Error::Other(format!(
            "{} has unrecognized tensors: {}",
            self.what,
            extra.join(", ")
        )))
    }
}

fn files(dir: &Path) -> Result<Vec<PathBuf>> {
    apxinf_loader::safetensors::safetensors_files(dir).map_err(Error::Other)
}

/// Deterministic buffers older exports store; mflux skips and recomputes them.
fn recomputed_buffer(name: &str) -> bool {
    name == "time_text_embed.time_proj.freqs" || name.starts_with("pos_embed.")
}

impl Transformer {
    pub fn load(stream: &Stream, dir: &Path) -> Result<Self> {
        let map = apxinf_mlx::load_safetensors(stream, &files(dir)?, |n| !recomputed_buffer(n))?;
        let mut t = Take {
            map,
            what: "Qwen-Image transformer",
        };
        let mut blocks = Vec::with_capacity(BLOCKS);
        for i in 0..BLOCKS {
            let p = format!("transformer_blocks.{i}");
            blocks.push(Block {
                norm_q: t.array(
                    &format!("{p}.attn.norm_q.weight"),
                    MlxDType::BF16,
                    &[HEAD_DIM],
                )?,
                norm_k: t.array(
                    &format!("{p}.attn.norm_k.weight"),
                    MlxDType::BF16,
                    &[HEAD_DIM],
                )?,
                to_q: t.q4(&format!("{p}.attn.to_q"), DIM, DIM)?,
                to_k: t.q4(&format!("{p}.attn.to_k"), DIM, DIM)?,
                to_v: t.q4(&format!("{p}.attn.to_v"), DIM, DIM)?,
                to_out: t.q4(&format!("{p}.attn.to_out.0"), DIM, DIM)?,
                gate: t.q4(&format!("{p}.img_mlp.gate_layer"), MLP, DIM)?,
                proj: t.q4(&format!("{p}.img_mlp.proj"), MLP, DIM)?,
                out: t.q4(&format!("{p}.img_mlp.out"), DIM, MLP)?,
            });
        }
        let transformer = Self {
            img_in: t.q4("img_in", DIM, LATENT)?,
            txt_norm: t.array("txt_in.text_norm.weight", MlxDType::BF16, &[DIM])?,
            txt_in: t.q4("txt_in.in_layer", DIM, DIM)?,
            txt_out: t.q4("txt_in.out_layer", DIM, DIM)?,
            time_1: t.q4("time_text_embed.timestep_embedder.linear_1", DIM, 256)?,
            time_2: t.q4("time_text_embed.timestep_embedder.linear_2", DIM, DIM)?,
            modulation: t.q4("modulation.layers.1", 4 * DIM, DIM)?,
            blocks,
            norm_out: t.q4("norm_out.linear", DIM, DIM)?,
            proj_out: t.q4("proj_out", LATENT, DIM)?,
        };
        t.finish()?;
        Ok(transformer)
    }
}

impl Decoder {
    /// Decoder channel widths: `144 * [8, 8, 8, 4, 2, 1]`.
    pub const DIMS: [usize; 6] = [1152, 1152, 1152, 576, 288, 144];

    pub fn load(stream: &Stream, dir: &Path) -> Result<Self> {
        let map = apxinf_mlx::load_safetensors(stream, &files(dir)?, |n| {
            n.starts_with("decoder.") || n.starts_with("post_quant_conv.")
        })?;
        let mut t = Take {
            map,
            what: "Qwen-Image VAE decoder",
        };
        let residual = |t: &mut Take, p: &str, input: usize, out: usize| -> Result<Residual> {
            Ok(Residual {
                norm1: t.array(&format!("{p}.norm1.weight"), MlxDType::F32, &[input])?,
                conv1: t.conv(&format!("{p}.conv1.conv"), out, 3, input)?,
                norm2: t.array(&format!("{p}.norm2.weight"), MlxDType::F32, &[out])?,
                conv2: t.conv(&format!("{p}.conv2.conv"), out, 3, out)?,
                shortcut: if input == out {
                    None
                } else {
                    Some(t.conv(&format!("{p}.conv_shortcut.conv"), out, 1, input)?)
                },
            })
        };
        let top = Self::DIMS[0];
        let post_quant = t.conv("post_quant_conv.conv", LATENT, 1, LATENT)?;
        let conv_in = t.conv("decoder.conv_in.conv", top, 3, LATENT)?;
        let mid = [
            residual(&mut t, "decoder.mid_block.resnets.0", top, top)?,
            residual(&mut t, "decoder.mid_block.resnets.1", top, top)?,
        ];
        let a = "decoder.mid_block.attentions.0";
        let attn_norm = t.array(&format!("{a}.norm.weight"), MlxDType::F32, &[top])?;
        let attn_qkv = t.conv(&format!("{a}.to_qkv"), 3 * top, 1, top)?;
        let attn_proj = t.conv(&format!("{a}.proj"), top, 1, top)?;
        let mut up = Vec::new();
        for (i, pair) in Self::DIMS.windows(2).enumerate() {
            let (input, out) = (pair[0], pair[1]);
            let p = format!("decoder.up_blocks.{i}");
            let resnets = (0..3)
                .map(|j| {
                    let from = if j == 0 { input } else { out };
                    residual(&mut t, &format!("{p}.resnets.{j}"), from, out)
                })
                .collect::<Result<Vec<_>>>()?;
            let upsampler = if i < Self::DIMS.len() - 2 {
                Some(t.conv(&format!("{p}.upsampler.conv"), out, 3, out)?)
            } else {
                None
            };
            up.push(Up {
                resnets,
                upsampler,
                in_dim: input,
                out_dim: out,
            });
        }
        let last = Self::DIMS[Self::DIMS.len() - 1];
        let decoder = Self {
            post_quant,
            conv_in,
            mid,
            attn_norm,
            attn_qkv,
            attn_proj,
            up,
            norm_out: t.array("decoder.norm_out.weight", MlxDType::F32, &[last])?,
            conv_out: t.conv("decoder.conv_out.conv", 4, 3, last)?,
        };
        t.finish()?;
        Ok(decoder)
    }
}

/// The stored dtype and shape of each text-encoder tensor; `None` for names
/// the family does not use.
fn text_tensor(name: &str) -> Option<(MlxDType, Vec<usize>)> {
    let bf16 = |shape: &[usize]| Some((MlxDType::BF16, shape.to_vec()));
    let kv = TEXT_KV_HEADS * HEAD_DIM;
    match name {
        "embed_tokens.weight" => bf16(&[TEXT_VOCAB, DIM]),
        "norm.weight" => bf16(&[DIM]),
        // mflux loads the stored rotary frequencies instead of recomputing them.
        "rotary_emb.inv_freq" => Some((MlxDType::F32, vec![HEAD_DIM / 2])),
        _ => {
            let (layer, rest) = name.strip_prefix("layers.")?.split_once('.')?;
            let index: usize = layer.parse().ok()?;
            if index >= TEXT_LAYERS || index.to_string() != layer {
                return None;
            }
            match rest {
                "input_layernorm.weight" | "post_attention_layernorm.weight" => bf16(&[DIM]),
                "self_attn.q_norm.weight" | "self_attn.k_norm.weight" => bf16(&[HEAD_DIM]),
                "self_attn.q_proj.weight" | "self_attn.o_proj.weight" => bf16(&[DIM, DIM]),
                "self_attn.k_proj.weight" | "self_attn.v_proj.weight" => bf16(&[kv, DIM]),
                "mlp.gate_proj.weight" | "mlp.up_proj.weight" => bf16(&[MLP, DIM]),
                "mlp.down_proj.weight" => bf16(&[DIM, MLP]),
                _ => None,
            }
        }
    }
}

/// SafeTensors dtype, shape and file of one tensor.
type Header = (String, Vec<usize>, PathBuf);

/// Tensor name -> header entry, read from file headers only.
fn headers(files: &[PathBuf]) -> Result<HashMap<String, Header>> {
    let mut out = HashMap::new();
    for file in files {
        let fail = |e: String| Error::Other(format!("read {}: {e}", file.display()));
        let mut reader = std::fs::File::open(file).map_err(|e| fail(e.to_string()))?;
        let mut length = [0; 8];
        reader
            .read_exact(&mut length)
            .map_err(|e| fail(e.to_string()))?;
        let length = u64::from_le_bytes(length);
        if length > 1 << 26 {
            return Err(fail("SafeTensors header exceeds 64 MiB".into()));
        }
        let mut raw = vec![0; length as usize];
        reader
            .read_exact(&mut raw)
            .map_err(|e| fail(e.to_string()))?;
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&raw).map_err(|e| fail(e.to_string()))?;
        for (name, info) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = info.get("dtype").and_then(|v| v.as_str());
            let shape: Option<Vec<usize>> = info
                .get("shape")
                .and_then(|v| serde_json::from_value(v.clone()).ok());
            let (Some(dtype), Some(shape)) = (dtype, shape) else {
                return Err(fail(format!("malformed header entry `{name}`")));
            };
            if out
                .insert(name.clone(), (dtype.to_owned(), shape, file.clone()))
                .is_some()
            {
                return Err(fail(format!("duplicate tensor `{name}`")));
            }
        }
    }
    Ok(out)
}

/// The BF16 text encoder. Opening validates every tensor header; requests
/// then read one decoder layer at a time.
pub struct TextEncoderFiles {
    /// Tensor name -> shard file.
    shards: HashMap<String, PathBuf>,
}

impl TextEncoderFiles {
    pub fn open(dir: &Path) -> Result<Self> {
        let mut shards = HashMap::new();
        for (name, (dtype, shape, file)) in headers(&files(dir)?)? {
            let Some((want_dtype, want_shape)) = text_tensor(&name) else {
                return Err(Error::Other(format!(
                    "Qwen-Image text encoder has an unrecognized tensor `{name}`"
                )));
            };
            let want = match want_dtype {
                MlxDType::F32 => "F32",
                _ => "BF16",
            };
            if dtype != want || shape != want_shape {
                return Err(Error::Other(format!(
                    "Qwen-Image text encoder `{name}`: expected {want} {want_shape:?}, \
                     got {dtype} {shape:?}"
                )));
            }
            shards.insert(name, file);
        }
        // Every name is known and unique, so the count proves completeness.
        if shards.len() != 3 + 11 * TEXT_LAYERS {
            return Err(Error::Other(format!(
                "Qwen-Image text encoder has {} of {} tensors",
                shards.len(),
                3 + 11 * TEXT_LAYERS
            )));
        }
        Ok(Self { shards })
    }

    fn load(&self, stream: &Stream, keep: impl Fn(&str) -> bool) -> Result<Take> {
        let files: BTreeSet<_> = self
            .shards
            .iter()
            .filter(|(name, _)| keep(name))
            .map(|(_, file)| file)
            .collect();
        let files: Vec<_> = files.into_iter().collect();
        Ok(Take {
            map: apxinf_mlx::load_safetensors(stream, &files, keep)?,
            what: "Qwen-Image text encoder",
        })
    }

    fn take(t: &mut Take, name: &str) -> Result<Array> {
        let (dtype, shape) = text_tensor(name).expect("text tensor names are fixed");
        t.array(name, dtype, &shape)
    }

    pub fn tensor(&self, stream: &Stream, name: &str) -> Result<Array> {
        let mut t = self.load(stream, |n| n == name)?;
        let array = Self::take(&mut t, name)?;
        t.finish()?;
        Ok(array)
    }

    pub fn layer(&self, stream: &Stream, i: usize) -> Result<TextLayer> {
        let p = format!("layers.{i}.");
        let mut t = self.load(stream, |n| n.starts_with(&p))?;
        let mut b = |name: &str| Self::take(&mut t, &format!("{p}{name}"));
        let layer = TextLayer {
            input_norm: b("input_layernorm.weight")?,
            q: b("self_attn.q_proj.weight")?,
            k: b("self_attn.k_proj.weight")?,
            v: b("self_attn.v_proj.weight")?,
            o: b("self_attn.o_proj.weight")?,
            q_norm: b("self_attn.q_norm.weight")?,
            k_norm: b("self_attn.k_norm.weight")?,
            post_norm: b("post_attention_layernorm.weight")?,
            gate: b("mlp.gate_proj.weight")?,
            up: b("mlp.up_proj.weight")?,
            down: b("mlp.down_proj.weight")?,
        };
        t.finish()?;
        Ok(layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_tensor_names_are_exact() {
        assert!(text_tensor("layers.35.mlp.down_proj.weight").is_some());
        assert!(text_tensor("layers.36.mlp.down_proj.weight").is_none());
        assert!(text_tensor("layers.01.mlp.down_proj.weight").is_none());
        assert!(text_tensor("layers.0.mlp.down_proj.bias").is_none());
        assert_eq!(
            text_tensor("rotary_emb.inv_freq"),
            Some((MlxDType::F32, vec![64]))
        );
        let names = (0..TEXT_LAYERS)
            .flat_map(|i| {
                [
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "self_attn.q_norm.weight",
                    "self_attn.k_norm.weight",
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                ]
                .map(|n| format!("layers.{i}.{n}"))
            })
            .chain(["embed_tokens.weight", "norm.weight", "rotary_emb.inv_freq"].map(String::from));
        assert_eq!(names.filter(|n| text_tensor(n).is_some()).count(), 399);
    }
}

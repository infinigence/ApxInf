//! Checkpoint loading: affine Q4 backbone, BF16 norms and gates, BF16 joint head.
use super::config::Config;
use apxinf_core::{Error, Result};
use apxinf_mlx::{Array, MlxDType, Stream};
use std::collections::HashMap;
use std::path::Path;

/// One affine Q4 group-64 matrix `[out, in]`: packed U32 `[out, in / 8]`,
/// BF16 scales and biases `[out, in / 64]`, kept exactly as stored.
#[derive(Clone)]
pub struct Q4 {
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
}

impl Q4 {
    pub fn rows(&self) -> usize {
        self.weight.shape()[0]
    }
}

pub struct Mlp {
    pub gate: Q4,
    pub up: Q4,
    pub down: Q4,
}

pub struct GatedDelta {
    pub qkv: Q4,
    pub z: Q4,
    pub b: Q4,
    pub a: Q4,
    pub out: Q4,
    /// Depthwise causal kernel `[8192, 4, 1]`.
    pub conv: Array,
    pub a_log: Array,
    pub dt_bias: Array,
    pub norm: Array,
}

pub struct Attention {
    pub q: Q4,
    pub k: Q4,
    pub v: Q4,
    pub o: Q4,
    pub q_norm: Array,
    pub k_norm: Array,
}

pub enum Mixer {
    GatedDelta(GatedDelta),
    Attention(Attention),
}

pub struct Layer {
    pub input_norm: Array,
    pub post_norm: Array,
    pub mixer: Mixer,
    pub mlp: Mlp,
}

pub struct Backbone {
    pub embedding: Q4,
    pub layers: Vec<Layer>,
    pub norm: Array,
    pub lm_head: Q4,
}

/// `torch.nn.MultiheadAttention` with a packed input projection.
pub struct Mha {
    pub in_weight: Array,
    pub in_bias: Array,
    pub out_weight: Array,
    pub out_bias: Array,
}

pub struct Norm {
    pub weight: Array,
    pub bias: Array,
}

pub struct Linear {
    pub weight: Array,
    pub bias: Array,
}

pub struct EvidenceLayer {
    pub query_norm: Norm,
    pub memory_norm: Norm,
    pub attention: Mha,
    pub feedforward_norm: Norm,
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct DecoderLayer {
    pub self_attn: Mha,
    pub cross_attn: Mha,
    pub linear1: Linear,
    pub linear2: Linear,
    pub norm1: Norm,
    pub norm2: Norm,
    pub norm3: Norm,
}

pub struct Head {
    pub hidden_norm: Norm,
    pub memory_projection: Array,
    pub question_projection: Array,
    pub option_question_projection: Array,
    pub global_projection: Array,
    pub option_context_projection: Array,
    pub option_lexical_projection: Array,
    pub type_embedding: Array,
    pub evidence: Vec<EvidenceLayer>,
    pub option_summary_norm: Norm,
    pub layers: Vec<DecoderLayer>,
    pub field_norm: Norm,
    pub option_norm: Norm,
    pub scorer1: Linear,
    pub scorer2: Linear,
    pub prior_logit_scale: Array,
    pub joint_logit_scale: Array,
    pub residual_gate: Array,
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

    fn bf16(&mut self, name: &str, shape: &[usize]) -> Result<Array> {
        self.array(name, MlxDType::BF16, shape)
    }

    fn q4(&mut self, name: &str, out: usize, input: usize) -> Result<Q4> {
        Ok(Q4 {
            weight: self.array(&format!("{name}.weight"), MlxDType::U32, &[out, input / 8])?,
            scales: self.bf16(&format!("{name}.scales"), &[out, input / 64])?,
            biases: self.bf16(&format!("{name}.biases"), &[out, input / 64])?,
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

impl Backbone {
    /// Loads every `language_model.*` tensor; the vision tower is never read.
    pub fn load(config: &Config, stream: &Stream, checkpoint: &Path) -> Result<Self> {
        let files =
            apxinf_loader::safetensors::safetensors_files(checkpoint).map_err(Error::Other)?;
        let map = apxinf_mlx::load_safetensors(stream, &files, |name| {
            !name.starts_with("vision_tower.")
        })?;
        let mut take = Take {
            map,
            what: "Clef backbone checkpoint",
        };
        let t = &config.text;
        let (hidden, inter) = (t.hidden_size, t.intermediate_size);
        let key_dim = t.linear_num_key_heads * t.linear_key_head_dim;
        let value_dim = t.linear_num_value_heads * t.linear_value_head_dim;
        let conv_dim = 2 * key_dim + value_dim;
        let q_dim = t.num_attention_heads * t.head_dim;
        let kv_dim = t.num_key_value_heads * t.head_dim;
        let embedding = take.q4("language_model.model.embed_tokens", t.vocab_size, hidden)?;
        let mut layers = Vec::with_capacity(t.num_hidden_layers);
        for i in 0..t.num_hidden_layers {
            let p = format!("language_model.model.layers.{i}");
            let mixer = if config.is_attention(i) {
                let a = format!("{p}.self_attn");
                Mixer::Attention(Attention {
                    q: take.q4(&format!("{a}.q_proj"), 2 * q_dim, hidden)?,
                    k: take.q4(&format!("{a}.k_proj"), kv_dim, hidden)?,
                    v: take.q4(&format!("{a}.v_proj"), kv_dim, hidden)?,
                    o: take.q4(&format!("{a}.o_proj"), hidden, q_dim)?,
                    q_norm: take.bf16(&format!("{a}.q_norm.weight"), &[t.head_dim])?,
                    k_norm: take.bf16(&format!("{a}.k_norm.weight"), &[t.head_dim])?,
                })
            } else {
                let g = format!("{p}.linear_attn");
                let heads = t.linear_num_value_heads;
                Mixer::GatedDelta(GatedDelta {
                    qkv: take.q4(&format!("{g}.in_proj_qkv"), conv_dim, hidden)?,
                    z: take.q4(&format!("{g}.in_proj_z"), value_dim, hidden)?,
                    b: take.q4(&format!("{g}.in_proj_b"), heads, hidden)?,
                    a: take.q4(&format!("{g}.in_proj_a"), heads, hidden)?,
                    out: take.q4(&format!("{g}.out_proj"), hidden, value_dim)?,
                    conv: take.bf16(
                        &format!("{g}.conv1d.weight"),
                        &[conv_dim, t.linear_conv_kernel_dim, 1],
                    )?,
                    a_log: take.bf16(&format!("{g}.A_log"), &[heads])?,
                    dt_bias: take.bf16(&format!("{g}.dt_bias"), &[heads])?,
                    norm: take.bf16(&format!("{g}.norm.weight"), &[t.linear_value_head_dim])?,
                })
            };
            let mlp = Mlp {
                gate: take.q4(&format!("{p}.mlp.gate_proj"), inter, hidden)?,
                up: take.q4(&format!("{p}.mlp.up_proj"), inter, hidden)?,
                down: take.q4(&format!("{p}.mlp.down_proj"), hidden, inter)?,
            };
            layers.push(Layer {
                input_norm: take.bf16(&format!("{p}.input_layernorm.weight"), &[hidden])?,
                post_norm: take.bf16(&format!("{p}.post_attention_layernorm.weight"), &[hidden])?,
                mixer,
                mlp,
            });
        }
        let norm = take.bf16("language_model.model.norm.weight", &[hidden])?;
        let lm_head = take.q4("language_model.lm_head", t.vocab_size, hidden)?;
        take.finish()?;
        Ok(Self {
            embedding,
            layers,
            norm,
            lm_head,
        })
    }
}

impl Head {
    /// Loads `joint_head.safetensors` with the reference key names.
    pub fn load(config: &Config, stream: &Stream, path: &Path) -> Result<Self> {
        let map = apxinf_mlx::load_safetensors(stream, &[path], |_| true)?;
        let mut take = Take {
            map,
            what: "Clef joint head",
        };
        let h = &config.head;
        let (hidden, width, ff) = (h.hidden_size, h.width, h.feedforward);
        let norm = |take: &mut Take, name: &str, n: usize| -> Result<Norm> {
            Ok(Norm {
                weight: take.bf16(&format!("{name}.weight"), &[n])?,
                bias: take.bf16(&format!("{name}.bias"), &[n])?,
            })
        };
        let linear = |take: &mut Take, name: &str, out: usize, input: usize| -> Result<Linear> {
            Ok(Linear {
                weight: take.bf16(&format!("{name}.weight"), &[out, input])?,
                bias: take.bf16(&format!("{name}.bias"), &[out])?,
            })
        };
        let mha = |take: &mut Take, name: &str| -> Result<Mha> {
            Ok(Mha {
                in_weight: take.bf16(&format!("{name}.in_proj_weight"), &[3 * width, width])?,
                in_bias: take.bf16(&format!("{name}.in_proj_bias"), &[3 * width])?,
                out_weight: take.bf16(&format!("{name}.out_proj.weight"), &[width, width])?,
                out_bias: take.bf16(&format!("{name}.out_proj.bias"), &[width])?,
            })
        };
        let hidden_norm = norm(&mut take, "hidden_norm", hidden)?;
        let projection =
            |take: &mut Take, name: &str| take.bf16(&format!("{name}.weight"), &[width, hidden]);
        let memory_projection = projection(&mut take, "memory_projection")?;
        let question_projection = projection(&mut take, "question_projection")?;
        let option_question_projection = projection(&mut take, "option_question_projection")?;
        let global_projection = projection(&mut take, "global_projection")?;
        let option_context_projection = projection(&mut take, "option_context_projection")?;
        let option_lexical_projection = projection(&mut take, "option_lexical_projection")?;
        let type_embedding = take.bf16("type_embedding.weight", &[3, width])?;
        let mut evidence = Vec::with_capacity(h.routing_layers);
        for i in 0..h.routing_layers {
            let p = format!("evidence_layers.{i}");
            evidence.push(EvidenceLayer {
                query_norm: norm(&mut take, &format!("{p}.query_norm"), width)?,
                memory_norm: norm(&mut take, &format!("{p}.memory_norm"), width)?,
                attention: mha(&mut take, &format!("{p}.attention"))?,
                feedforward_norm: norm(&mut take, &format!("{p}.feedforward_norm"), width)?,
                fc1: linear(&mut take, &format!("{p}.feedforward.0"), ff, width)?,
                fc2: linear(&mut take, &format!("{p}.feedforward.3"), width, ff)?,
            });
        }
        let option_summary_norm = norm(&mut take, "option_summary_norm", width)?;
        let mut layers = Vec::with_capacity(h.layers);
        for i in 0..h.layers {
            let p = format!("layers.{i}");
            layers.push(DecoderLayer {
                self_attn: mha(&mut take, &format!("{p}.self_attn"))?,
                cross_attn: mha(&mut take, &format!("{p}.multihead_attn"))?,
                linear1: linear(&mut take, &format!("{p}.linear1"), ff, width)?,
                linear2: linear(&mut take, &format!("{p}.linear2"), width, ff)?,
                norm1: norm(&mut take, &format!("{p}.norm1"), width)?,
                norm2: norm(&mut take, &format!("{p}.norm2"), width)?,
                norm3: norm(&mut take, &format!("{p}.norm3"), width)?,
            });
        }
        let head = Self {
            hidden_norm,
            memory_projection,
            question_projection,
            option_question_projection,
            global_projection,
            option_context_projection,
            option_lexical_projection,
            type_embedding,
            evidence,
            option_summary_norm,
            layers,
            field_norm: norm(&mut take, "field_norm", width)?,
            option_norm: norm(&mut take, "option_norm", width)?,
            scorer1: linear(&mut take, "residual_scorer.0", width, 4 * width)?,
            scorer2: linear(&mut take, "residual_scorer.3", 1, width)?,
            prior_logit_scale: take.bf16("prior_logit_scale", &[])?,
            joint_logit_scale: take.bf16("joint_logit_scale", &[])?,
            residual_gate: take.bf16("residual_gate", &[])?,
        };
        take.finish()?;
        Ok(head)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn take(entries: &[(&str, MlxDType, &[usize])]) -> Take {
        let s = Stream::cpu().unwrap();
        let map = entries
            .iter()
            .map(|&(name, dtype, shape)| (name.to_owned(), Array::zeros(&s, shape, dtype).unwrap()))
            .collect();
        Take {
            map,
            what: "test checkpoint",
        }
    }

    #[test]
    fn every_tensor_is_checked_and_none_is_left_over() {
        let q4 = [
            ("p.weight", MlxDType::U32, &[64, 16][..]),
            ("p.scales", MlxDType::BF16, &[64, 2][..]),
            ("p.biases", MlxDType::BF16, &[64, 2][..]),
        ];
        let mut ok = take(&q4);
        assert_eq!(ok.q4("p", 64, 128).unwrap().rows(), 64);
        ok.finish().unwrap();

        let error =
            |mut t: Take, f: fn(&mut Take) -> Result<Q4>| f(&mut t).err().unwrap().to_string();
        assert!(error(take(&q4[..2]), |t| t.q4("p", 64, 128)).contains("missing `p.biases`"));
        let mut wrong = q4;
        wrong[0].1 = MlxDType::I32;
        assert!(error(take(&wrong), |t| t.q4("p", 64, 128)).contains("`p.weight`"));
        let mut wrong = q4;
        wrong[1].2 = &[64, 4];
        assert!(error(take(&wrong), |t| t.q4("p", 64, 128)).contains("`p.scales`"));
        // Group 32 or 8-bit packing changes the stored shapes and is refused.
        assert!(error(take(&q4), |t| t.q4("p", 64, 256)).contains("`p.weight`"));

        let mut extra = take(&[q4[0], q4[1], q4[2], ("q.weight", MlxDType::BF16, &[1])]);
        extra.q4("p", 64, 128).unwrap();
        assert!(extra.finish().unwrap_err().to_string().contains("q.weight"));
    }

    /// Run with `--ignored --exact metal_clef::weights::tests::recorded_joint_head_loads`.
    /// APXINF_CLEF_CHECKPOINT names the checkpoint directory. Reads only the
    /// 243 MB BF16 joint head, on a CPU stream.
    #[test]
    #[ignore = "requires the Clef checkpoint"]
    fn recorded_joint_head_loads() {
        let root = std::path::PathBuf::from(std::env::var_os("APXINF_CLEF_CHECKPOINT").unwrap());
        let read = |name: &str| std::fs::read_to_string(root.join(name)).unwrap();
        let config =
            Config::from_json(&read("config.json"), &read("joint_head_config.json")).unwrap();
        let s = Stream::cpu().unwrap();
        let head = Head::load(&config, &s, &root.join("joint_head.safetensors")).unwrap();
        assert_eq!((head.evidence.len(), head.layers.len()), (2, 4));
        assert_eq!(head.scorer2.weight.shape(), [1, 1024]);
        assert_eq!(head.type_embedding.shape(), [3, 1024]);
    }
}

//! Weight loading for Qwen3.8-27B-AWQ-INT4.
//!
//! Quantized `Linear` weights stay in compressed-tensors `pack-quantized`
//! form (int32 packed + bf16 scales + packed zero points) for later GPU
//! upload; the dequant step only materialises bf16 on demand.

use std::collections::HashMap;
use std::path::Path;

use half::bf16;

use crate::qwen35::config::Qwen35TextConfig;
use crate::qwen35::dequant::dequantize_w4a16;
use crate::qwen35::safetensors_raw::{load_checkpoint, RawTensor};

/// A W4A16 packed linear weight exactly as stored on disk.
pub struct Q4Linear {
    pub out: usize,
    pub inp: usize,
    pub packed: Vec<i32>, // [out, ceil(inp/8)]
    pub scale: Vec<bf16>, // [out, inp/32]
    pub zp: Vec<i32>,     // [ceil(out/8), inp/32], nibbles packed along rows
}

impl Q4Linear {
    fn from_tensors(base: &str, map: &HashMap<String, RawTensor>) -> Result<Self, String> {
        let key = |suffix: &str| format!("{base}.{suffix}");
        let packed = map
            .get(&key("weight_packed"))
            .ok_or_else(|| format!("missing {}.weight_packed", base))?
            .as_i32()?;
        let scale = map
            .get(&key("weight_scale"))
            .ok_or_else(|| format!("missing {}.weight_scale", base))?
            .as_bf16()?;
        let zp = map
            .get(&key("weight_zero_point"))
            .ok_or_else(|| format!("missing {}.weight_zero_point", base))?
            .as_i32()?;
        let shape = map
            .get(&key("weight_shape"))
            .ok_or_else(|| format!("missing {}.weight_shape", base))?
            .as_shape()?;
        if shape.len() != 2 {
            return Err(format!("{base}.weight_shape must be length 2"));
        }
        Ok(Self {
            out: shape[0],
            inp: shape[1],
            packed,
            scale,
            zp,
        })
    }

    pub fn dequant_into(&self, dst: &mut Vec<bf16>) -> Result<(), String> {
        dequantize_w4a16(self.out, self.inp, &self.packed, &self.scale, &self.zp, dst)
    }
}

/// Dense bf16 matrix stored row-major (`data[rows][cols]`), matching the
/// on-disk `[rows, cols]` layout.
/// Either a dense bf16 weight or a W4A16 packed weight (the checkpoint mixes
/// both for `linear_attn.out_proj` depending on the layer).
pub enum MatKind {
    Dense(Bf16Mat),
    Q4(Q4Linear),
}

impl MatKind {
    pub fn from_tensors(
        base: &str,
        map: &HashMap<String, RawTensor>,
        expect_out: usize,
        expect_in: usize,
    ) -> Result<Self, String> {
        if map.contains_key(&format!("{base}.weight_packed")) {
            let q = Q4Linear::from_tensors(base, map)?;
            if q.out != expect_out || q.inp != expect_in {
                return Err(format!(
                    "{base}: expected [{expect_out}, {expect_in}], got [{}, {}]",
                    q.out, q.inp
                ));
            }
            Ok(MatKind::Q4(q))
        } else {
            Ok(MatKind::Dense(bf16_mat(
                &format!("{base}.weight"),
                map,
                expect_out,
                expect_in,
            )?))
        }
    }
}

pub struct Bf16Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<bf16>,
}

fn bf16_mat(name: &str, map: &HashMap<String, RawTensor>, rows: usize, cols: usize) -> Result<Bf16Mat, String> {
    let t = map.get(name).ok_or_else(|| format!("missing {name}"))?;
    let data = t.as_bf16()?;
    if t.shape.len() != 2 || t.shape[0] != rows || t.shape[1] != cols {
        return Err(format!(
            "{name}: expected [{rows}, {cols}], got {:?}",
            t.shape
        ));
    }
    Ok(Bf16Mat { rows, cols, data })
}

fn bf16_vec(name: &str, map: &HashMap<String, RawTensor>, len: usize) -> Result<Vec<bf16>, String> {
    let t = map.get(name).ok_or_else(|| format!("missing {name}"))?;
    let data = t.as_bf16()?;
    if t.shape.len() != 1 || t.shape[0] != len {
        return Err(format!("{name}: expected [{len}], got {:?}", t.shape));
    }
    Ok(data)
}

fn f32_from_bf16(name: &str, map: &HashMap<String, RawTensor>, len: usize) -> Result<Vec<f32>, String> {
    Ok(bf16_vec(name, map, len)?
        .into_iter()
        .map(|v| v.to_f32())
        .collect())
}

/// Per-layer weights; the two layer types share the same MLP + norms.
pub enum LayerWeights {
    Full {
        q: Q4Linear,
        k: Q4Linear,
        v: Q4Linear,
        o: Q4Linear,
        q_norm: Vec<bf16>,
        k_norm: Vec<bf16>,
        gate: Q4Linear,
        up: Q4Linear,
        down: Q4Linear,
        in_norm: Vec<bf16>,
        post_norm: Vec<bf16>,
    },
    Linear {
        in_qkv: Q4Linear,
        in_z: Q4Linear,
        in_a: Bf16Mat,
        in_b: Bf16Mat,
        conv: Bf16Mat,
        a_log: Vec<f32>,
        dt_bias: Vec<f32>,
        norm: Vec<bf16>,
        out_proj: MatKind,
        gate: Q4Linear,
        up: Q4Linear,
        down: Q4Linear,
        in_norm: Vec<bf16>,
        post_norm: Vec<bf16>,
    },
}

pub struct Qwen35Weights {
    pub config: Qwen35TextConfig,
    pub embed: Bf16Mat,
    pub lm_head: Bf16Mat,
    pub final_norm: Vec<bf16>,
    pub layers: Vec<LayerWeights>,
}

impl Qwen35Weights {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let config = Qwen35TextConfig::from_json_file(&model_dir.join("config.json"))?;
        let map = load_checkpoint(model_dir)?;

        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let nk = config.linear_num_key_heads;
        let nv = config.linear_num_value_heads;
        let k_dim = nk * config.linear_key_head_dim;
        let v_dim = nv * config.linear_value_head_dim;

        let embed = bf16_mat(
            "model.language_model.embed_tokens.weight",
            &map,
            config.vocab_size,
            hidden,
        )?;
        let lm_head = bf16_mat("lm_head.weight", &map, config.vocab_size, hidden)?;
        let final_norm = bf16_vec("model.language_model.norm.weight", &map, hidden)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{layer_idx}");
            let in_norm = bf16_vec(&format!("{prefix}.input_layernorm.weight"), &map, hidden)?;
            let post_norm = bf16_vec(
                &format!("{prefix}.post_attention_layernorm.weight"),
                &map,
                hidden,
            )?;
            let gate = Q4Linear::from_tensors(&format!("{prefix}.mlp.gate_proj"), &map)?;
            let up = Q4Linear::from_tensors(&format!("{prefix}.mlp.up_proj"), &map)?;
            let down = Q4Linear::from_tensors(&format!("{prefix}.mlp.down_proj"), &map)?;
            assert_eq!((gate.out, gate.inp), (intermediate, hidden));
            assert_eq!((up.out, up.inp), (intermediate, hidden));
            assert_eq!((down.out, down.inp), (hidden, intermediate));

            layers.push(if config.is_full_attention(layer_idx) {
                let base = format!("{prefix}.self_attn");
                let q = Q4Linear::from_tensors(&format!("{base}.q_proj"), &map)?;
                let k = Q4Linear::from_tensors(&format!("{base}.k_proj"), &map)?;
                let v = Q4Linear::from_tensors(&format!("{base}.v_proj"), &map)?;
                let o = Q4Linear::from_tensors(&format!("{base}.o_proj"), &map)?;
                assert_eq!((q.out, q.inp), (heads * head_dim * 2, hidden));
                assert_eq!((k.out, k.inp), (kv_heads * head_dim, hidden));
                assert_eq!((v.out, v.inp), (kv_heads * head_dim, hidden));
                assert_eq!((o.out, o.inp), (hidden, heads * head_dim));
                let q_norm = bf16_vec(&format!("{base}.q_norm.weight"), &map, head_dim)?;
                let k_norm = bf16_vec(&format!("{base}.k_norm.weight"), &map, head_dim)?;
                LayerWeights::Full {
                    q, k, v, o,
                    q_norm, k_norm,
                    gate, up, down,
                    in_norm, post_norm,
                }
            } else {
                let base = format!("{prefix}.linear_attn");
                let in_qkv = Q4Linear::from_tensors(&format!("{base}.in_proj_qkv"), &map)?;
                let in_z = Q4Linear::from_tensors(&format!("{base}.in_proj_z"), &map)?;
                assert_eq!((in_qkv.out, in_qkv.inp), (k_dim * 2 + v_dim, hidden));
                assert_eq!((in_z.out, in_z.inp), (v_dim, hidden));
                let in_a = bf16_mat(&format!("{base}.in_proj_a.weight"), &map, nv, hidden)?;
                let in_b = bf16_mat(&format!("{base}.in_proj_b.weight"), &map, nv, hidden)?;
                let conv_raw = map
                    .get(&format!("{base}.conv1d.weight"))
                    .ok_or_else(|| format!("missing {base}.conv1d.weight"))?;
                let conv = {
                    // [conv_dim, 1, kernel] -> [conv_dim, kernel]
                    let data = conv_raw.as_bf16()?;
                    let kernel = config.linear_conv_kernel_dim;
                    let conv_dim = k_dim * 2 + v_dim;
                    if conv_raw.shape.len() != 3
                        || conv_raw.shape[0] != conv_dim
                        || conv_raw.shape[2] != kernel
                    {
                        return Err(format!(
                            "{base}.conv1d.weight: unexpected shape {:?}",
                            conv_raw.shape
                        ));
                    }
                    Bf16Mat {
                        rows: conv_dim,
                        cols: kernel,
                        data,
                    }
                };
                let a_log = f32_from_bf16(&format!("{base}.A_log"), &map, nv)?;
                let dt_bias = f32_from_bf16(&format!("{base}.dt_bias"), &map, nv)?;
                let norm = bf16_vec(&format!("{base}.norm.weight"), &map, config.linear_value_head_dim)?;
                let out_proj = MatKind::from_tensors(&format!("{base}.out_proj"), &map, hidden, v_dim)?;
                LayerWeights::Linear {
                    in_qkv,
                    in_z,
                    in_a,
                    in_b,
                    conv,
                    a_log,
                    dt_bias,
                    norm,
                    out_proj,
                    gate,
                    up,
                    down,
                    in_norm,
                    post_norm,
                }
            });
        }

        Ok(Self {
            config,
            embed,
            lm_head,
            final_norm,
            layers,
        })
    }
}

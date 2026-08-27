use std::collections::HashMap;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, Result, Tensor};
use apxinf_loader::compressed_tensors::{
    quantize_bf16_marlin_awq_u4_g32_v1, repack_w4_marlin_awq_u4_g32_v1,
    repack_w4_marlin_awq_u4_g32_v1_concat, repack_w4_n64_k16_v1,
    W4_MARLIN_AWQ_U4_G32_V1_SUFFIX, W4_TRANSFORM_CACHE_SUFFIX,
};
use apxinf_loader::ModelConfig;

use crate::llm_trait::{DraftBlockResult, LlmCapabilities, LlmTrait};

use super::{LayerKind, Qwen35Config};

pub struct GeneralQwen35 {
    pub config: Qwen35Config,
    tensors: HashMap<String, Tensor>,
    #[allow(dead_code)]
    backend: Arc<dyn Backend>,
    #[allow(dead_code)]
    device_tensors: HashMap<String, Tensor>,
    linear_states: Vec<Option<LinearAttentionState>>,
    full_k: Vec<Vec<f32>>,
    full_v: Vec<Vec<f32>>,
    #[cfg(feature = "cuda")]
    cuda: Option<crate::qwen35::cuda::Qwen35Cuda>,
}

struct LinearAttentionState {
    conv: Vec<f32>,
    recurrent: Vec<f32>,
}

#[cfg(feature = "cuda")]
fn marlin_projection_role(prefix: &str) -> Option<&'static str> {
    // Fixed APXINF_MARLIN_ROLES names: qkv, q, k, v, o, out, gate, up, down.
    let layer = prefix.strip_prefix("model.language_model.layers.")?;
    let (layer_index, projection) = layer.split_once('.')?;
    layer_index.parse::<usize>().ok()?;
    match projection {
        "linear_attn.in_proj_qkv" => Some("qkv"),
        "self_attn.q_proj" => Some("q"),
        "self_attn.k_proj" => Some("k"),
        "self_attn.v_proj" => Some("v"),
        "self_attn.o_proj" => Some("o"),
        "linear_attn.out_proj" => Some("out"),
        "mlp.gate_proj" => Some("gate"),
        "mlp.up_proj" => Some("up"),
        "mlp.down_proj" => Some("down"),
        _ => None,
    }
}

impl GeneralQwen35 {
    pub fn new(config: Qwen35Config) -> Self {
        let backend = crate::accelerator::create_backend(Device::Cpu)
            .expect("CPU backend creation must not fail");
        Self::from_weights_with_backend(config, HashMap::new(), backend)
            .expect("empty qwen3_5 metadata model must construct")
    }

    pub fn from_weights_with_backend(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let n_layers = config.text.n_layers;
        let device_tensors = Self::prepare_device_tensors(&tensors, backend.as_ref())?;
        #[cfg(feature = "cuda")]
        let cuda = if backend.device().is_gpu() && !tensors.is_empty() {
            Some(crate::qwen35::cuda::Qwen35Cuda::new(
                backend.clone(),
                &config,
                &tensors,
                &device_tensors,
            )?)
        } else {
            None
        };
        Ok(Self {
            config,
            tensors,
            backend,
            device_tensors,
            linear_states: (0..n_layers).map(|_| None).collect(),
            full_k: vec![Vec::new(); n_layers],
            full_v: vec![Vec::new(); n_layers],
            #[cfg(feature = "cuda")]
            cuda,
        })
    }


    #[cfg(feature = "cuda")]
    fn prepare_device_tensors(
        tensors: &HashMap<String, Tensor>,
        backend: &dyn Backend,
    ) -> Result<HashMap<String, Tensor>> {
        if !backend.device().is_gpu() {
            return Ok(HashMap::new());
        }
        let quantized_lm_head = std::env::var("APXINF_LM_HEAD_W4")
            .map_or(true, |value| value != "0");
        let mut device_tensors = HashMap::new();
        for (name, tensor) in tensors {
            let compressed = name.ends_with(".weight_packed")
                || name.ends_with(".weight_scale")
                || name.ends_with(".weight_zero_point");
            let cache = name == "model.language_model.embed_tokens.weight"
                || (name == "lm_head.weight" && !quantized_lm_head)
                || (name != "lm_head.weight"
                    && name.ends_with(".weight")
                    && tensor.shape().dims().len() == 2);
            if cache && !compressed && matches!(tensor.dtype(), DType::BF16 | DType::I32) {
                device_tensors.insert(name.clone(), backend.to_device(tensor)?);
            }
        }
        if quantized_lm_head {
            let dense = tensors.get("lm_head.weight")
                .ok_or_else(|| Error::Other("qwen3_5 weights: missing lm_head.weight".into()))?;
            let quantized = quantize_bf16_marlin_awq_u4_g32_v1(dense)
                .map_err(Error::Other)?
                .ok_or_else(|| Error::Other("qwen3_5 LM head W4 requires a 2-D BF16 group-32-compatible matrix".into()))?;
            let suffix = W4_MARLIN_AWQ_U4_G32_V1_SUFFIX;
            device_tensors.insert(
                format!("lm_head.weight_packed.{suffix}"),
                backend.to_device(&quantized.packed)?,
            );
            device_tensors.insert(
                format!("lm_head.weight_scale.{suffix}"),
                backend.to_device(&quantized.scale)?,
            );
            device_tensors.insert(
                format!("lm_head.weight_zero_point.{suffix}"),
                backend.to_device(&quantized.zero_point)?,
            );
        }
        let fused_raw_mlp = std::env::var_os("APXINF_MLP_FUSED_RAW")
            .is_some_and(|value| value == "1");
        let marlin = std::env::var("APXINF_MARLIN").map_or(true, |value| value != "0");
        let marlin_roles = match std::env::var("APXINF_MARLIN_ROLES") {
            Ok(value) if value.trim() == "all" => None,
            Ok(value) => Some(
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|role| !role.is_empty())
                    .map(str::to_owned)
                    .collect::<std::collections::HashSet<_>>(),
            ),
            Err(_) => Some(
                ["q", "k", "v", "o", "out", "gate", "up", "down"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
            ),
        };
        let transform_cache = std::env::var_os("APXINF_W4_TRANSFORM_CACHE")
            .is_some_and(|value| value == "1");
        let prefixes: Vec<String> = tensors
            .keys()
            .filter(|name| name.ends_with(".weight_packed"))
            .map(|name| name.trim_end_matches(".weight_packed").to_owned())
            .collect();
        // Same-activation Marlin groups use one physical N-concatenated
        // allocation. Members are deliberately skipped below: cloned tensors
        // share the backend allocation, while Gemm records its logical slice.
        let mut combined_members = std::collections::HashSet::new();
        if marlin && std::env::var_os("APXINF_MARLIN_CONCAT").is_some_and(|value| value == "1") {
            let prefix_set = prefixes.iter().map(String::as_str)
                .collect::<std::collections::HashSet<_>>();
            if std::env::var("APXINF_MARLIN_CONCAT_QKV").map_or(true, |value| value != "0") {
                for q in prefixes.iter().filter(|prefix| prefix.ends_with(".self_attn.q_proj")) {
                    let base = q.trim_end_matches("q_proj");
                    let k = format!("{base}k_proj");
                    let v = format!("{base}v_proj");
                    let selected = [q.as_str(), k.as_str(), v.as_str()].iter().all(|prefix| {
                        prefix_set.contains(prefix)
                            && marlin_roles.as_ref().is_none_or(|roles| {
                                marlin_projection_role(prefix).is_some_and(|role| roles.contains(role))
                            })
                    });
                    if selected {
                        if let Some((repacked, _)) = repack_w4_marlin_awq_u4_g32_v1_concat(
                            tensors, &[q, &k, &v],
                        ).map_err(Error::Other)? {
                            let combined = format!("{base}qkv_concat");
                            let suffix = W4_MARLIN_AWQ_U4_G32_V1_SUFFIX;
                            device_tensors.insert(format!("{combined}.weight_packed.{suffix}"), backend.to_device(&repacked.packed)?);
                            device_tensors.insert(format!("{combined}.weight_scale.{suffix}"), backend.to_device(&repacked.scale)?);
                            device_tensors.insert(format!("{combined}.weight_zero_point.{suffix}"), backend.to_device(&repacked.zero_point)?);
                            combined_members.extend([q.clone(), k, v]);
                        }
                    }
                }
            }
            if !fused_raw_mlp
                && std::env::var("APXINF_MARLIN_CONCAT_GATE_UP").map_or(true, |value| value != "0")
            {
                for gate in prefixes.iter().filter(|prefix| prefix.ends_with(".mlp.gate_proj")) {
                    let base = gate.trim_end_matches("gate_proj");
                    let up = format!("{base}up_proj");
                    let selected = [gate.as_str(), up.as_str()].iter().all(|prefix| {
                        prefix_set.contains(prefix)
                            && marlin_roles.as_ref().is_none_or(|roles| {
                                marlin_projection_role(prefix).is_some_and(|role| roles.contains(role))
                            })
                    });
                    if selected {
                        if let Some((repacked, _)) = repack_w4_marlin_awq_u4_g32_v1_concat(
                            tensors, &[gate, &up],
                        ).map_err(Error::Other)? {
                            let combined = format!("{base}gate_up_concat");
                            let suffix = W4_MARLIN_AWQ_U4_G32_V1_SUFFIX;
                            device_tensors.insert(format!("{combined}.weight_packed.{suffix}"), backend.to_device(&repacked.packed)?);
                            device_tensors.insert(format!("{combined}.weight_scale.{suffix}"), backend.to_device(&repacked.scale)?);
                            device_tensors.insert(format!("{combined}.weight_zero_point.{suffix}"), backend.to_device(&repacked.zero_point)?);
                            combined_members.extend([gate.clone(), up]);
                        }
                    }
                }
            }
        }
        for prefix in prefixes {
            if combined_members.contains(&prefix) {
                continue;
            }
            let raw_names = [
                format!("{prefix}.weight_packed"),
                format!("{prefix}.weight_scale"),
                format!("{prefix}.weight_zero_point"),
            ];
            let raw = raw_names
                .iter()
                .map(|name| tensors.get(name))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    Error::Other(format!("qwen3_5 W4 fallback missing tensors for {prefix}"))
                })?;
            let role = marlin_projection_role(&prefix);
            let role_selected = (!fused_raw_mlp || !matches!(role, Some("gate" | "up")))
                && marlin_roles.as_ref().is_none_or(|roles| {
                    role.is_some_and(|role| roles.contains(role))
                });
            if marlin && role_selected {
                if let Some(repacked) = repack_w4_marlin_awq_u4_g32_v1(tensors, &prefix)
                    .map_err(Error::Other)?
                    .filter(|weight| weight.metadata.padded_cols % 128 == 0
                        && weight.metadata.padded_rows % 64 == 0)
                {
                    let suffix = W4_MARLIN_AWQ_U4_G32_V1_SUFFIX;
                    device_tensors.insert(
                        format!("{prefix}.weight_packed.{suffix}"),
                        backend.to_device(&repacked.packed)?,
                    );
                    device_tensors.insert(
                        format!("{prefix}.weight_scale.{suffix}"),
                        backend.to_device(&repacked.scale)?,
                    );
                    device_tensors.insert(
                        format!("{prefix}.weight_zero_point.{suffix}"),
                        backend.to_device(&repacked.zero_point)?,
                    );
                    continue;
                }
            }
            if !marlin && transform_cache {
                if let Some(repacked) = repack_w4_n64_k16_v1(tensors, &prefix)
                    .map_err(Error::Other)?
                {
                    let suffix = W4_TRANSFORM_CACHE_SUFFIX;
                    device_tensors.insert(
                        format!("{prefix}.weight_packed.{suffix}"),
                        backend.to_device(&repacked.packed)?,
                    );
                    device_tensors.insert(
                        format!("{prefix}.weight_scale.{suffix}"),
                        backend.to_device(&repacked.scale)?,
                    );
                    device_tensors.insert(
                        format!("{prefix}.weight_zero_point.{suffix}"),
                        backend.to_device(&repacked.zero_point)?,
                    );
                    continue;
                }
            }
            for (name, tensor) in raw_names.iter().zip(raw.iter()) {
                device_tensors.insert(name.clone(), backend.to_device(tensor)?);
            }
        }
        Ok(device_tensors)
    }

    #[cfg(not(feature = "cuda"))]
    fn prepare_device_tensors(
        _tensors: &HashMap<String, Tensor>,
        _backend: &dyn Backend,
    ) -> Result<HashMap<String, Tensor>> {
        Ok(HashMap::new())
    }

    fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::Other(format!("qwen3_5 weights: missing {name}")))
    }

    fn embedding(&self, token_ids: &[u32]) -> Result<Tensor> {
        let table = self.tensor("model.language_model.embed_tokens.weight")?;
        let dims = table.shape().dims();
        if dims.len() != 2 {
            return Err(Error::Other("qwen3_5 embedding table must be 2D".into()));
        }
        let vocab = dims[0];
        let hidden = dims[1];
        for &token in token_ids {
            if token as usize >= vocab {
                return Err(Error::Other(format!(
                    "token id {token} out of vocabulary {vocab}"
                )));
            }
        }
        #[cfg(feature = "cuda")]
        if self.backend.device().is_gpu() {
            if let Some(table_gpu) = self.device_tensors.get("model.language_model.embed_tokens.weight") {
                let out_gpu = self.backend.embedding(table_gpu, token_ids)?;
                let out_cpu = self.backend.to_cpu(&out_gpu)?;
                return Tensor::from_f32(
                    vec![token_ids.len(), hidden],
                    &tensor_to_f32(&out_cpu)?,
                );
            }
        }
        let storage = table.storage().as_cpu().ok_or_else(|| Error::Other("embedding table must be CPU-backed".into()))?;
        let mut out = vec![0.0; token_ids.len() * hidden];
        match table.dtype() {
            DType::BF16 => {
                let values = table.as_bf16()?;
                for (row, &token) in token_ids.iter().enumerate() {
                    let src = token as usize * hidden;
                    for col in 0..hidden {
                        out[row * hidden + col] = values[src + col].to_f32();
                    }
                }
            }
            DType::F32 => {
                let values = table.as_f32()?;
                for (row, &token) in token_ids.iter().enumerate() {
                    let src = token as usize * hidden;
                    out[row * hidden..(row + 1) * hidden]
                        .copy_from_slice(&values[src..src + hidden]);
                }
            }
            dtype => return Err(Error::Other(format!("embedding does not support {dtype}"))),
        }
        let _ = storage;
        Tensor::from_f32(vec![token_ids.len(), hidden], &out)
    }

    fn forward_layer(&mut self, x: &Tensor, layer: usize, start_pos: u32) -> Result<Tensor> {
        match self.config.text.layer_types[layer] {
            LayerKind::LinearAttention => self.forward_linear_layer(x, layer),
            LayerKind::FullAttention => self.forward_full_layer(x, layer, start_pos),
        }
    }

    fn forward_full_layer(&mut self, x: &Tensor, layer: usize, start_pos: u32) -> Result<Tensor> {
        let prefix = format!("model.language_model.layers.{layer}");
        let tc = &self.config.text;
        let seq = x.shape().dims()[0];
        let hidden = tc.hidden_size;
        let head_dim = tc.head_dim;
        let n_heads = tc.n_heads;
        let n_kv = tc.n_kv_heads;

        let normed = rms_norm_plus_one(x, self.tensor(&format!("{prefix}.input_layernorm.weight"))?, tc.rms_norm_eps)?;
        if layer == 3 {
            trace_f32("f3_normed", normed.as_f32()?);
        }
        let q_gate = self.linear_packed(&normed, &format!("{prefix}.self_attn.q_proj"))?;
        if layer == 3 {
            trace_f32("f3_qgate", q_gate.as_f32()?);
        }
        let mut k = self.linear_packed(&normed, &format!("{prefix}.self_attn.k_proj"))?;
        let v = self.linear_packed(&normed, &format!("{prefix}.self_attn.v_proj"))?;
        let qg = q_gate.as_f32()?;
        let vd = v.as_f32()?;

        let mut q = vec![0.0f32; seq * n_heads * head_dim];
        let mut gate = vec![0.0f32; seq * n_heads * head_dim];
        for s in 0..seq {
            for h in 0..n_heads {
                let src = s * n_heads * head_dim * 2 + h * head_dim * 2;
                let dst = s * n_heads * head_dim + h * head_dim;
                q[dst..dst + head_dim].copy_from_slice(&qg[src..src + head_dim]);
                gate[dst..dst + head_dim].copy_from_slice(&qg[src + head_dim..src + 2 * head_dim]);
            }
        }
        let q = Tensor::from_f32(vec![seq * n_heads, head_dim], &q)?;
        let k_2d = k.reshape(vec![seq * n_kv, head_dim])?;
        let q = rms_norm_plus_one(&q, self.tensor(&format!("{prefix}.self_attn.q_norm.weight"))?, tc.rms_norm_eps)?;
        k = rms_norm_plus_one(&k_2d, self.tensor(&format!("{prefix}.self_attn.k_norm.weight"))?, tc.rms_norm_eps)?;
        let mut q = q.as_f32()?.to_vec();
        let mut k = k.as_f32()?.to_vec();
        apply_partial_rope(&mut q, seq, n_heads, head_dim, tc.partial_rotary_factor, tc.rope_theta, start_pos);
        apply_partial_rope(&mut k, seq, n_kv, head_dim, tc.partial_rotary_factor, tc.rope_theta, start_pos);
        if layer == 3 {
            trace_f32("f3_q", &q);
            trace_f32("f3_k", &k);
            trace_f32("f3_v", vd);
            trace_f32("f3_gate", &gate);
        }

        self.full_k[layer].extend_from_slice(&k);
        self.full_v[layer].extend_from_slice(vd);
        let total = self.full_v[layer].len() / (n_kv * head_dim);
        let mut attn = vec![0.0f32; seq * n_heads * head_dim];
        for s in 0..seq {
            let visible = start_pos as usize + s + 1;
            let visible = visible.min(total);
            for h in 0..n_heads {
                let kv_h = h * n_kv / n_heads;
                let q_base = s * n_heads * head_dim + h * head_dim;
                let mut scores = vec![0.0f32; visible];
                for t in 0..visible {
                    let k_base = t * n_kv * head_dim + kv_h * head_dim;
                    let mut dot = 0.0;
                    for d in 0..head_dim {
                        dot += q[q_base + d] * self.full_k[layer][k_base + d];
                    }
                    scores[t] = dot / (head_dim as f32).sqrt();
                }
                softmax_in_place(&mut scores);
                let out_base = q_base;
                for t in 0..visible {
                    let v_base = t * n_kv * head_dim + kv_h * head_dim;
                    let w = scores[t];
                    for d in 0..head_dim {
                        attn[out_base + d] += w * self.full_v[layer][v_base + d];
                    }
                }
            }
        }
        if layer == 3 {
            trace_f32("f3_attn_pre", &attn);
        }
        for (value, gate_value) in attn.iter_mut().zip(gate.iter()) {
            *value *= sigmoid(*gate_value);
        }
        if layer == 3 {
            trace_f32("f3_attn_post", &attn);
        }
        let attn = Tensor::from_f32(vec![seq, n_heads * head_dim], &attn)?;
        let attn = self.linear_packed(&attn, &format!("{prefix}.self_attn.o_proj"))?;
        if layer == 3 {
            trace_f32("f3_attn2", attn.as_f32()?);
        }
        let x = add_tensors(x, &attn)?;
        if layer == 3 {
            trace_f32("f3_x", x.as_f32()?);
        }
        self.forward_mlp(&x, layer, hidden)
    }

    fn forward_linear_layer(&mut self, x: &Tensor, layer: usize) -> Result<Tensor> {
        let prefix = format!("model.language_model.layers.{layer}");
        let attn_prefix = format!("{prefix}.linear_attn");
        let tc = &self.config.text;
        let seq = x.shape().dims()[0];
        let k_heads = tc.linear_num_key_heads;
        let v_heads = tc.linear_num_value_heads;
        let kdim = tc.linear_key_head_dim;
        let vdim = tc.linear_value_head_dim;
        let key_dim = k_heads * kdim;
        let value_dim = v_heads * vdim;
        let conv_dim = key_dim * 2 + value_dim;
        let kernel = tc.linear_conv_kernel_dim;

        if layer == 0 {
            trace_f32("cpu_x_pre", x.as_f32()?);
        }
        let normed = rms_norm_plus_one(x, self.tensor(&format!("{prefix}.input_layernorm.weight"))?, tc.rms_norm_eps)?;
        if layer == 0 {
            trace_f32("cpu_normed", normed.as_f32()?);
        }
        let qkv = self.linear_packed(&normed, &format!("{attn_prefix}.in_proj_qkv"))?;
        if layer == 0 {
            trace_f32("cpu_qkv_pre", qkv.as_f32()?);
        }
        let z = self.linear_packed(&normed, &format!("{attn_prefix}.in_proj_z"))?;
        let a = self.linear_dense(&normed, &format!("{attn_prefix}.in_proj_a.weight"))?;
        let b = self.linear_dense(&normed, &format!("{attn_prefix}.in_proj_b.weight"))?;
        let conv_weight = self.tensor(&format!("{attn_prefix}.conv1d.weight"))?.clone();
        let qkv = depthwise_causal_conv_silu(
            qkv.as_f32()?,
            &conv_weight,
            seq,
            conv_dim,
            kernel,
            self.linear_states[layer]
                .get_or_insert_with(|| LinearAttentionState {
                    conv: vec![0.0; conv_dim * kernel.saturating_sub(1)],
                    recurrent: vec![0.0; v_heads * kdim * vdim],
                }),
        )?;
        if layer == 0 {
            trace_f32("cpu_qkv_post", &qkv);
        }
        let a = a.as_f32()?;
        let b = b.as_f32()?;
        let a_log = tensor_to_f32(self.tensor(&format!("{attn_prefix}.A_log"))?)?;
        let dt_bias = tensor_to_f32(self.tensor(&format!("{attn_prefix}.dt_bias"))?)?;

        let state = self.linear_states[layer]
            .as_mut()
            .ok_or_else(|| Error::Other("linear state missing after conv".into()))?;
        let mut out = vec![0.0f32; seq * value_dim];
        let repeat = v_heads / k_heads;
        let scale = 1.0 / (kdim as f32).sqrt();
        for s in 0..seq {
            for vh in 0..v_heads {
                let kh = vh / repeat;
                let q_base = s * conv_dim + kh * kdim;
                let k_base = s * conv_dim + key_dim + kh * kdim;
                let v_base = s * conv_dim + key_dim * 2 + vh * vdim;
                let beta = sigmoid(b[s * v_heads + vh]);
                let decay = (-(a_log[vh].exp()) * softplus(a[s * v_heads + vh] + dt_bias[vh])).exp();
                let state_base = vh * kdim * vdim;
                let mut q_vec = vec![0.0f32; kdim];
                let mut k_vec = vec![0.0f32; kdim];
                q_vec.copy_from_slice(&qkv[q_base..q_base + kdim]);
                k_vec.copy_from_slice(&qkv[k_base..k_base + kdim]);
                l2_normalize(&mut q_vec);
                l2_normalize(&mut k_vec);
                for value in &mut state.recurrent[state_base..state_base + kdim * vdim] {
                    *value *= decay;
                }
                let mut mem = vec![0.0f32; vdim];
                for kd in 0..kdim {
                    let kval = k_vec[kd];
                    for vd in 0..vdim {
                        mem[vd] += state.recurrent[state_base + kd * vdim + vd] * kval;
                    }
                }
                for vd in 0..vdim {
                    let delta = (qkv[v_base + vd] - mem[vd]) * beta;
                    for kd in 0..kdim {
                        state.recurrent[state_base + kd * vdim + vd] += k_vec[kd] * delta;
                    }
                }
                let out_base = s * value_dim + vh * vdim;
                for vd in 0..vdim {
                    let mut acc = 0.0;
                    for kd in 0..kdim {
                        acc += state.recurrent[state_base + kd * vdim + vd] * q_vec[kd] * scale;
                    }
                    out[out_base + vd] = acc;
                }
            }
        }
        if layer == 0 {
            trace_f32("cpu_delta_out", &out);
        }
        let out = gated_rms_norm(
            &Tensor::from_f32(vec![seq * v_heads, vdim], &out)?,
            &z.reshape(vec![seq * v_heads, vdim])?,
            self.tensor(&format!("{attn_prefix}.norm.weight"))?,
            tc.rms_norm_eps,
        )?;
        if layer == 0 {
            trace_f32("cpu_gated", out.as_f32()?);
        }
        let out = out.reshape(vec![seq, value_dim])?;
        let attn = self.linear_maybe_packed(&out, &format!("{attn_prefix}.out_proj"))?;
        if layer == 0 {
            trace_f32("cpu_attn", attn.as_f32()?);
        }
        let x = add_tensors(x, &attn)?;
        if layer == 0 {
            trace_f32("cpu_x", x.as_f32()?);
        }
        self.forward_mlp(&x, layer, tc.hidden_size)
    }

    fn forward_mlp(&self, x: &Tensor, layer: usize, hidden: usize) -> Result<Tensor> {
        let prefix = format!("model.language_model.layers.{layer}");
        let normed = rms_norm_plus_one(
            x,
            self.tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
            self.config.text.rms_norm_eps,
        )?;
        let gate = silu_tensor(&self.linear_packed(&normed, &format!("{prefix}.mlp.gate_proj"))?)?;
        let up = self.linear_packed(&normed, &format!("{prefix}.mlp.up_proj"))?;
        let hidden_act = mul_tensors(&gate, &up)?;
        let out = self.linear_packed(&hidden_act, &format!("{prefix}.mlp.down_proj"))?;
        debug_assert_eq!(out.shape().dims(), &[x.shape().dims()[0], hidden]);
        add_tensors(x, &out)
    }

    fn linear_maybe_packed(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let dense_name = format!("{prefix}.weight");
        if self.tensors.contains_key(&dense_name) {
            self.linear_dense(x, &dense_name)
        } else {
            self.linear_packed(x, prefix)
        }
    }


    #[cfg(feature = "cuda")]
    fn linear_dense_cuda(&self, x: &Tensor, weight_name: &str, weight_gpu: &Tensor) -> Result<Tensor> {
        let x_shape = x.shape().dims();
        if x_shape.len() != 2 {
            return Err(Error::Other(format!(
                "{weight_name}: dense CUDA expects 2D activation, got {x_shape:?}"
            )));
        }
        let rows = x_shape[0];
        let x_bf16_values = x
            .as_f32()?
            .iter()
            .map(|&value| half::bf16::from_f32(value))
            .collect::<Vec<_>>();
        let x_bf16 = Tensor::from_bf16(x_shape.to_vec(), &x_bf16_values)?;
        let x_gpu = self.backend.to_device(&x_bf16)?;
        let cb = self
            .backend
            .as_any()
            .downcast_ref::<apxinf_cuda::CudaBackend>()
            .ok_or_else(|| Error::Other("qwen3_5 CUDA backend downcast failed".into()))?;
        let y_gpu = apxinf_cuda::kernels::quantization::matmul_bf16_transposed(
            cb.context(),
            &x_gpu,
            weight_gpu,
        )?;
        let out_cols = y_gpu.shape().dims()[1];
        let y_cpu = self.backend.to_cpu(&y_gpu)?;
        Tensor::from_f32(vec![rows, out_cols], &tensor_to_f32(&y_cpu)?)
    }

    fn linear_dense(&self, x: &Tensor, weight_name: &str) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.backend.device().is_gpu() {
            if let Some(weight_gpu) = self.device_tensors.get(weight_name) {
                return self.linear_dense_cuda(x, weight_name, weight_gpu);
            }
        }
        let x_data = x.as_f32()?;
        let x_shape = x.shape().dims();
        let rows = x_shape[0];
        let in_cols = x_shape[1];
        let weight = self.tensor(weight_name)?;
        let w_shape = weight.shape().dims();
        if w_shape.len() != 2 || w_shape[1] != in_cols {
            return Err(Error::Other(format!(
                "{weight_name} shape {w_shape:?} incompatible with activation {x_shape:?}"
            )));
        }
        let out_cols = w_shape[0];
        let weight = tensor_to_f32(weight)?;
        let mut out = vec![0.0f32; rows * out_cols];
        for r in 0..rows {
            for o in 0..out_cols {
                let mut acc = 0.0;
                for c in 0..in_cols {
                    acc += x_data[r * in_cols + c] * weight[o * in_cols + c];
                }
                out[r * out_cols + o] = acc;
            }
        }
        Tensor::from_f32(vec![rows, out_cols], &out)
    }

    #[cfg(feature = "cuda")]
    fn linear_packed_cuda(
        &self,
        x: &Tensor,
        prefix: &str,
        packed: &Tensor,
        scales: &Tensor,
        zero_points: &Tensor,
        shape: &[i64],
    ) -> Result<Tensor> {
        let x_shape = x.shape().dims();
        if x_shape.len() != 2 || shape.len() != 2 || shape[1] as usize != x_shape[1] {
            return Err(Error::Other(format!(
                "{prefix}.weight_shape {shape:?} incompatible with activation {x_shape:?}"
            )));
        }
        let rows = x_shape[0];
        let in_cols = x_shape[1];
        let out_cols = shape[0] as usize;
        let groups = scales.shape().dims()[1];
        let cb = self
            .backend
            .as_any()
            .downcast_ref::<apxinf_cuda::CudaBackend>()
            .ok_or_else(|| Error::Other("qwen3_5 CUDA backend downcast failed".into()))?;
        let x_bf16_values = x
            .as_f32()?
            .iter()
            .map(|&value| half::bf16::from_f32(value))
            .collect::<Vec<_>>();
        let x_bf16 = Tensor::from_bf16(x_shape.to_vec(), &x_bf16_values)?;
        let x_gpu = self.backend.to_device(&x_bf16)?;
        let packed_name = format!("{prefix}.weight_packed");
        let scales_name = format!("{prefix}.weight_scale");
        let zero_points_name = format!("{prefix}.weight_zero_point");
        let packed_owned;
        let scales_owned;
        let zero_points_owned;
        let packed_gpu = if let Some(tensor) = self.device_tensors.get(&packed_name) {
            tensor
        } else {
            packed_owned = self.backend.to_device(packed)?;
            &packed_owned
        };
        let scales_gpu = if let Some(tensor) = self.device_tensors.get(&scales_name) {
            tensor
        } else {
            scales_owned = self.backend.to_device(scales)?;
            &scales_owned
        };
        let zero_points_gpu = if let Some(tensor) = self.device_tensors.get(&zero_points_name) {
            tensor
        } else {
            zero_points_owned = self.backend.to_device(zero_points)?;
            &zero_points_owned
        };
        let y_gpu = apxinf_cuda::kernels::quantization::matmul_bf16_w4a16_asym(
            cb.context(),
            &x_gpu,
            packed_gpu,
            scales_gpu,
            zero_points_gpu,
            out_cols,
            in_cols,
            groups,
        )?;
        let y_cpu = self.backend.to_cpu(&y_gpu)?;
        let y = y_cpu
            .as_bf16()?
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        Tensor::from_f32(vec![rows, out_cols], &y)
    }

    fn linear_packed(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let x_data = x.as_f32()?;
        let x_shape = x.shape().dims();
        let rows = x_shape[0];
        let in_cols = x_shape[1];
        let packed = self.tensor(&format!("{prefix}.weight_packed"))?;
        let scales = self.tensor(&format!("{prefix}.weight_scale"))?;
        let zero_points = self.tensor(&format!("{prefix}.weight_zero_point"))?;
        let shape = self.tensor(&format!("{prefix}.weight_shape"))?.as_i64()?;
        #[cfg(feature = "cuda")]
        if self.backend.device().is_gpu() {
            return self.linear_packed_cuda(x, prefix, packed, scales, zero_points, &shape);
        }
        if shape.len() != 2 || shape[1] as usize != in_cols {
            return Err(Error::Other(format!(
                "{prefix}.weight_shape {shape:?} incompatible with activation {x_shape:?}"
            )));
        }
        let out_cols = shape[0] as usize;
        let packed_cols = in_cols.div_ceil(8);
        let group_count = scales.shape().dims()[1];
        let group_size = in_cols.div_ceil(group_count);
        let packed = packed.as_i32()?;
        let scales = scales.as_bf16()?;
        let zero_points = zero_points.as_i32()?;
        let mut out = vec![0.0f32; rows * out_cols];
        for r in 0..rows {
            for o in 0..out_cols {
                let zp_row = o / 8;
                let zp_shift = (o % 8) * 4;
                let mut acc = 0.0f32;
                for c in 0..in_cols {
                    let group = c / group_size;
                    let word = packed[o * packed_cols + c / 8] as u32;
                    let q = ((word >> ((c % 8) * 4)) & 0xF) as i32;
                    let zp_word = zero_points[zp_row * group_count + group] as u32;
                    let zp = ((zp_word >> zp_shift) & 0xF) as i32;
                    let w = (q - zp) as f32 * scales[o * group_count + group].to_f32();
                    acc += x_data[r * in_cols + c] * w;
                }
                out[r * out_cols + o] = acc;
            }
        }
        Tensor::from_f32(vec![rows, out_cols], &out)
    }
    /// Forward pass that reports the hidden state after every decoder layer.
    /// Debugging entry point: `hook(layer_index, hidden_state)` is invoked with
    /// the layer output for layers `0..n_layers`, then the final norm and
    /// `lm_head` produce the returned logits exactly as in [`Self::forward`].
    pub fn forward_with_hook<F: FnMut(usize, &Tensor)>(
        &mut self,
        token_ids: &[u32],
        start_pos: u32,
        mut hook: F,
    ) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(Error::Other("qwen3_5 forward: empty token_ids".into()));
        }
        let mut x = self.embedding(token_ids)?;
        for layer in 0..self.config.text.n_layers {
            x = self.forward_layer(&x, layer, start_pos)?;
            hook(layer, &x);
        }
        let x = rms_norm_plus_one(
            &x,
            self.tensor("model.language_model.norm.weight")?,
            self.config.text.rms_norm_eps,
        )?;
        self.linear_dense(&x, "lm_head.weight")
    }

    /// Debug helper: run the GPU path and dump per-layer hidden states.
    #[cfg(feature = "cuda")]
    pub fn forward_dump_gpu(
        &mut self,
        token_ids: &[u32],
        start_pos: u32,
        dir: &std::path::Path,
    ) -> Result<Tensor> {
        match &mut self.cuda {
            Some(cuda) => cuda.forward_dump(token_ids, start_pos, dir),
            None => Err(Error::Other("no CUDA state".into())),
        }
    }
}
impl LlmTrait for GeneralQwen35 {
    fn load(_config: ModelConfig, _weights: HashMap<String, Tensor>, _device: Device) -> Result<Self>
    where
        Self: Sized,
    {
        Err(Error::Other(
            "qwen3_5 must be loaded from its Hugging Face config.json, not generic ModelConfig".into(),
        ))
    }

    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &mut self.cuda {
            return cuda.forward(token_ids, start_pos);
        }
        self.forward_with_hook(token_ids, start_pos, |_, _| {})
    }

    fn prefill_token(&mut self, input: crate::llm_trait::LlmInput<'_>) -> Option<Result<u32>> {
        if input.image.is_some() {
            return None;
        }
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &mut self.cuda {
            return Some(cuda.prefill_token(input.token_ids));
        }
        None
    }

    fn decode_token(&mut self, token: u32, pos: u32) -> Option<Result<u32>> {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &mut self.cuda {
            return Some(cuda.decode_token(token, pos));
        }
        let _ = (token, pos);
        None
    }

    fn decode_draft_block(
        &mut self,
        current_token: u32,
        start_pos: u32,
        draft: &[u32],
        verified: &mut Vec<u32>,
    ) -> Option<Result<DraftBlockResult>> {
        // Default-off until measured. Unsupported configurations return None
        // without touching state, preserving the ordinary token loop fallback.
        if std::env::var_os("APXINF_PROMPT_LOOKUP").is_none_or(|value| value != "1") {
            return None;
        }
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &mut self.cuda {
            if !cuda.draft_block_supported() {
                return None;
            }
            return Some(cuda.decode_draft_block(current_token, start_pos, draft, verified));
        }
        let _ = (current_token, start_pos, draft, verified);
        None
    }


    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::TEXT_ONLY
    }

    fn reset(&mut self) {
        for state in &mut self.linear_states {
            *state = None;
        }
        for cache in &mut self.full_k {
            cache.clear();
        }
        for cache in &mut self.full_v {
            cache.clear();
        }
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &mut self.cuda {
            let _ = cuda.reset();
        }
    }

    fn vocab_size(&self) -> usize {
        self.config.text.vocab_size
    }
}

/// Debug: write an f32 slice to /tmp/qwen35_trace/<name>.f32 when APXINF_TRACE is set.
fn trace_f32(name: &str, data: &[f32]) {
    if std::env::var_os("APXINF_TRACE").is_none() {
        return;
    }
    let dir = std::path::Path::new("/tmp/qwen35_trace");
    let _ = std::fs::create_dir_all(dir);
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let _ = std::fs::write(dir.join(format!("{name}.f32")), bytes);
}

fn tensor_to_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    match tensor.dtype() {
        apxinf_core::DType::F32 => tensor.as_f32().map(|v| v.to_vec()),
        apxinf_core::DType::BF16 => Ok(tensor.as_bf16()?.iter().map(|v| v.to_f32()).collect()),
        apxinf_core::DType::F16 => Ok(tensor.as_f16()?.iter().map(|v| v.to_f32()).collect()),
        dtype => Err(Error::Other(format!("cannot convert {dtype} tensor to f32"))),
    }
}

fn rms_norm_plus_one(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    rms_norm_impl(input, weight, eps, true)
}

fn gated_rms_norm(input: &Tensor, gate: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let mut out = rms_norm_impl(input, weight, eps, false)?.as_f32()?.to_vec();
    let gate = gate.as_f32()?;
    for (value, gate_value) in out.iter_mut().zip(gate.iter()) {
        *value *= silu(*gate_value);
    }
    Tensor::from_f32(input.shape().dims().to_vec(), &out)
}

fn rms_norm_impl(input: &Tensor, weight: &Tensor, eps: f32, plus_one: bool) -> Result<Tensor> {
    let data = input.as_f32()?;
    let w = tensor_to_f32(weight)?;
    let dims = input.shape().dims();
    let rows = dims[0];
    let cols = dims[1];
    if w.len() != cols {
        return Err(Error::Other(format!(
            "RMSNorm weight len {} != cols {cols}",
            w.len()
        )));
    }
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for c in 0..cols {
            let scale = if plus_one { 1.0 + w[c] } else { w[c] };
            out[r * cols + c] = row[c] * inv * scale;
        }
    }
    Tensor::from_f32(dims.to_vec(), &out)
}

fn add_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(Error::Other(format!(
            "tensor shape mismatch {:?} != {:?}",
            a.shape().dims(),
            b.shape().dims()
        )));
    }
    let ad = a.as_f32()?;
    let bd = b.as_f32()?;
    let out: Vec<f32> = ad.iter().zip(bd.iter()).map(|(x, y)| x + y).collect();
    Tensor::from_f32(a.shape().dims().to_vec(), &out)
}

fn mul_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let ad = a.as_f32()?;
    let bd = b.as_f32()?;
    if ad.len() != bd.len() {
        return Err(Error::Other(format!("tensor size mismatch {} != {}", ad.len(), bd.len())));
    }
    let out: Vec<f32> = ad.iter().zip(bd.iter()).map(|(x, y)| x * y).collect();
    Tensor::from_f32(a.shape().dims().to_vec(), &out)
}

fn silu_tensor(x: &Tensor) -> Result<Tensor> {
    let out: Vec<f32> = x.as_f32()?.iter().map(|&v| silu(v)).collect();
    Tensor::from_f32(x.shape().dims().to_vec(), &out)
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum != 0.0 {
        for value in values.iter_mut() {
            *value /= sum;
        }
    }
}

fn l2_normalize(values: &mut [f32]) {
    let norm = (values.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
    for value in values {
        *value /= norm;
    }
}

fn apply_partial_rope(
    data: &mut [f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
    factor: f32,
    theta: f32,
    start_pos: u32,
) {
    let rotary_dim = ((head_dim as f32 * factor) as usize).min(head_dim) & !1usize;
    if rotary_dim == 0 { return; }
    let half = rotary_dim / 2;
    for s in 0..seq {
        let pos = start_pos as usize + s;
        for h in 0..heads {
            let base = s * heads * head_dim + h * head_dim;
            for i in 0..half {
                let inv = 1.0 / theta.powf((2 * i) as f32 / rotary_dim as f32);
                let angle = pos as f32 * inv;
                let (sin, cos) = angle.sin_cos();
                let x1 = data[base + i];
                let x2 = data[base + half + i];
                data[base + i] = x1 * cos - x2 * sin;
                data[base + half + i] = x2 * cos + x1 * sin;
            }
        }
    }
}

fn depthwise_causal_conv_silu(
    input: &[f32],
    weight: &Tensor,
    seq: usize,
    channels: usize,
    kernel: usize,
    state: &mut LinearAttentionState,
) -> Result<Vec<f32>> {
    let weight = tensor_to_f32(weight)?;
    if weight.len() != channels * kernel {
        return Err(Error::Other(format!(
            "linear_attn conv weight len {} != channels {channels} * kernel {kernel}",
            weight.len()
        )));
    }
    let mut out = vec![0.0f32; seq * channels];
    for s in 0..seq {
        for c in 0..channels {
            let mut acc = 0.0;
            for k in 0..kernel {
                let source_offset = k as isize - (kernel as isize - 1);
                let value = if source_offset + s as isize >= 0 {
                    input[(s as isize + source_offset) as usize * channels + c]
                } else {
                    let state_t = (kernel as isize - 1 + source_offset + s as isize) as usize;
                    state.conv[state_t * channels + c]
                };
                acc += value * weight[c * kernel + k];
            }
            out[s * channels + c] = silu(acc);
        }
    }
    if kernel > 1 {
        if seq >= kernel - 1 {
            // Enough new inputs: the state becomes the last (kernel-1) of them.
            for t in 0..kernel - 1 {
                let src_s = seq - (kernel - 1) + t;
                let dst = t * channels;
                state.conv[dst..dst + channels]
                    .copy_from_slice(&input[src_s * channels..(src_s + 1) * channels]);
            }
        } else {
            // Decode / short prefill: shift the old state left and append the
            // new inputs at the back, keeping [oldest .. newest] order.
            for t in 0..kernel - 1 - seq {
                let src = (t + seq) * channels;
                let dst = t * channels;
                state.conv.copy_within(src..src + channels, dst);
            }
            for t in 0..seq {
                let src = t * channels;
                let dst = (kernel - 1 - seq + t) * channels;
                state.conv[dst..dst + channels].copy_from_slice(&input[src..src + channels]);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_vocab_size_and_text_only_capability() {
        let config = Qwen35Config::from_json_str(
            r#"{
                "model_type": "qwen3_5",
                "text_config": {
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 8,
                    "vocab_size": 64,
                    "max_position_embeddings": 128,
                    "rms_norm_eps": 1e-6,
                    "partial_rotary_factor": 0.25,
                    "full_attention_interval": 1,
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 8,
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                    "linear_value_head_dim": 8,
                    "layer_types": ["full_attention"],
                    "rope_parameters": {"mrope_interleaved": true, "mrope_section": [1, 1, 0], "rope_theta": 10000}
                }
            }"#,
        )
        .unwrap();
        let model = GeneralQwen35::new(config);
        assert_eq!(model.vocab_size(), 64);
        assert_eq!(model.capabilities(), LlmCapabilities::TEXT_ONLY);
    }

    fn raw_i32(shape: &[usize], values: &[i32]) -> Tensor {
        Tensor::from_raw(
            apxinf_core::Shape::from(shape.to_vec()),
            apxinf_core::DType::I32,
            apxinf_core::Device::Cpu,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap()
    }

    fn raw_i64(shape: &[usize], values: &[i64]) -> Tensor {
        Tensor::from_raw(
            apxinf_core::Shape::from(shape.to_vec()),
            apxinf_core::DType::I64,
            apxinf_core::Device::Cpu,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap()
    }

    fn pack(values: &[u8]) -> i32 {
        values.iter().enumerate().fold(0u32, |word, (i, value)| {
            word | (((value & 0xF) as u32) << (i * 4))
        }) as i32
    }

    /// Regression test for the decode conv-state bug: the carry state must
    /// shift by one input per decode step, not overwrite the oldest slot with
    /// the newest input (which made every decode token after the first see a
    /// corrupted input window).
    #[test]
    fn conv_state_shifts_correctly_across_decode_steps() {
        let kernel = 4usize;
        let channels = 2usize;
        // Depthwise identity-ish kernel: w[c] = [0, 0, 0, 1] → y[t] = x[t].
        let weight = Tensor::from_f32(
            vec![channels, kernel],
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let mut state = LinearAttentionState {
            conv: vec![0.0; channels * (kernel - 1)],
            recurrent: vec![0.0; 0],
        };
        // Prefill of three tokens [a, b, c] with distinct values per channel.
        let prefill: Vec<f32> = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let out = depthwise_causal_conv_silu(&prefill, &weight, 3, channels, kernel, &mut state)
            .unwrap();
        // The output passes through SiLU; the state holds the raw inputs.
        let silu = |x: f32| x / (1.0 + (-x).exp());
        assert_eq!(
            out,
            vec![silu(1.0), silu(10.0), silu(2.0), silu(20.0), silu(3.0), silu(30.0)]
        );
        assert_eq!(state.conv, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
        // Decode steps: each must see its own input (tap k=3 weight 1).
        for (token, _expected) in [(4.0f32, 40.0f32), (5.0f32, 50.0f32)] {
            let out = depthwise_causal_conv_silu(
                &[token, token * 10.0],
                &weight,
                1,
                channels,
                kernel,
                &mut state,
            )
            .unwrap();
            assert_eq!(out, vec![silu(token), silu(token * 10.0)]);
        }
        assert_eq!(state.conv, vec![3.0, 30.0, 4.0, 40.0, 5.0, 50.0]);
    }
    #[test]
    fn packed_linear_matches_reference() {
        let config = Qwen35Config::from_json_str(
            r#"{
                "model_type": "qwen3_5",
                "text_config": {
                    "hidden_size": 8,
                    "intermediate_size": 8,
                    "num_hidden_layers": 0,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "head_dim": 8,
                    "vocab_size": 8,
                    "max_position_embeddings": 16,
                    "rms_norm_eps": 1e-6,
                    "partial_rotary_factor": 0.25,
                    "full_attention_interval": 1,
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 8,
                    "linear_num_key_heads": 1,
                    "linear_num_value_heads": 1,
                    "linear_value_head_dim": 8,
                    "layer_types": [],
                    "rope_parameters": {"mrope_interleaved": true, "mrope_section": [1, 1, 0], "rope_theta": 10000}
                }
            }"#,
        )
        .unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("w.weight_packed".into(), raw_i32(&[2, 1], &[pack(&[1, 2, 3, 4, 5, 6, 7, 8]), pack(&[8, 7, 6, 5, 4, 3, 2, 1])]));
        tensors.insert(
            "w.weight_scale".into(),
            Tensor::from_bf16(vec![2, 1], &[half::bf16::from_f32(1.0); 2]).unwrap(),
        );
        tensors.insert("w.weight_zero_point".into(), raw_i32(&[1, 1], &[0]));
        tensors.insert("w.weight_shape".into(), raw_i64(&[2], &[2, 8]));
        let backend = crate::accelerator::create_backend(Device::Cpu).unwrap();
        let model = GeneralQwen35::from_weights_with_backend(config, tensors, backend).unwrap();
        let x = Tensor::from_f32(vec![1, 8], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let y = model.linear_packed(&x, "w").unwrap();
        assert_eq!(y.as_f32().unwrap(), &[36.0, 36.0]);
    }


    #[test]
    fn forward_zero_layer_smoke() {
        let config = Qwen35Config::from_json_str(
            r#"{
                "model_type": "qwen3_5",
                "text_config": {
                    "hidden_size": 4,
                    "intermediate_size": 4,
                    "num_hidden_layers": 0,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "vocab_size": 4,
                    "max_position_embeddings": 16,
                    "rms_norm_eps": 1e-6,
                    "partial_rotary_factor": 0.5,
                    "full_attention_interval": 1,
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 4,
                    "linear_num_key_heads": 1,
                    "linear_num_value_heads": 1,
                    "linear_value_head_dim": 4,
                    "layer_types": [],
                    "rope_parameters": {"mrope_interleaved": true, "mrope_section": [1, 1, 0], "rope_theta": 10000}
                }
            }"#,
        )
        .unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.language_model.embed_tokens.weight".into(),
            Tensor::from_bf16(vec![4, 4], &[
                half::bf16::from_f32(1.0), half::bf16::from_f32(0.0), half::bf16::from_f32(0.0), half::bf16::from_f32(0.0),
                half::bf16::from_f32(0.0), half::bf16::from_f32(1.0), half::bf16::from_f32(0.0), half::bf16::from_f32(0.0),
                half::bf16::from_f32(0.0), half::bf16::from_f32(0.0), half::bf16::from_f32(1.0), half::bf16::from_f32(0.0),
                half::bf16::from_f32(0.0), half::bf16::from_f32(0.0), half::bf16::from_f32(0.0), half::bf16::from_f32(1.0),
            ]).unwrap(),
        );
        tensors.insert("model.language_model.norm.weight".into(), Tensor::from_bf16(vec![4], &[half::bf16::ZERO; 4]).unwrap());
        tensors.insert(
            "lm_head.weight".into(),
            Tensor::from_bf16(vec![4, 4], &[half::bf16::from_f32(1.0); 16]).unwrap(),
        );
        let backend = crate::accelerator::create_backend(Device::Cpu).unwrap();
        let mut model = GeneralQwen35::from_weights_with_backend(config, tensors, backend).unwrap();
        let logits = model.forward(&[0, 1], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[2, 4]);
    }

}

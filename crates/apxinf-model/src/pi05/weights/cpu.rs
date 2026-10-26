//! CPU materialization retains canonical host layouts and converts once to FP32.
use super::host::*;
use crate::pi05::Pi05Config;
use apxinf_core::{DType, Error, Result, Tensor};
use std::path::Path;

fn f32_tensor(t: &mut Tensor) -> Result<()> {
    if !matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        return Err(Error::Other(
            "PI0.5 CPU reference requires floating point, unquantized weights".into(),
        ));
    }
    if t.dtype() != DType::F32 {
        *t = Tensor::from_f32(t.shape().clone(), &t.to_f32_vec()?)?;
    }
    Ok(())
}
fn linear(w: &mut LinearWeights) -> Result<()> {
    f32_tensor(&mut w.weight)?;
    if let Some(b) = &mut w.bias {
        f32_tensor(b)?;
    }
    Ok(())
}
fn norm(w: &mut LayerNormWeights) -> Result<()> {
    f32_tensor(&mut w.weight)?;
    f32_tensor(&mut w.bias)
}
fn attention(w: &mut GemmaAttentionWeights) -> Result<()> {
    linear(&mut w.q)?;
    linear(&mut w.k)?;
    linear(&mut w.v)?;
    linear(&mut w.output)
}
fn mlp(w: &mut GemmaMlpWeights) -> Result<()> {
    linear(&mut w.gate)?;
    linear(&mut w.up)?;
    linear(&mut w.down)
}
impl Pi05Weights {
    /// Convert before checkpoint transforms, including Gemma's learned norm offset.
    pub(in crate::pi05) fn from_safetensors_cpu(config: &Pi05Config, path: &Path) -> Result<Self> {
        let (mut tensors, _) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|e| Error::Other(format!("load PI0.5 CPU weights: {e}")))?;
        for t in tensors.values_mut() {
            f32_tensor(t)?;
        }
        Self::from_map(config, tensors)
    }
    pub(in crate::pi05) fn into_cpu_f32(mut self, config: &Pi05Config) -> Result<Self> {
        self.validate_cpu_shapes(config)?;
        let v = &mut self.vision;
        linear(&mut v.patch_embedding)?;
        f32_tensor(&mut v.position_embedding)?;
        for b in &mut v.blocks {
            norm(&mut b.norm1)?;
            norm(&mut b.norm2)?;
            for w in [
                &mut b.q,
                &mut b.k,
                &mut b.v,
                &mut b.output,
                &mut b.fc1,
                &mut b.fc2,
            ] {
                linear(w)?;
            }
        }
        norm(&mut v.post_layer_norm)?;
        linear(&mut v.multimodal_projector)?;
        f32_tensor(&mut v.token_embedding)?;
        for l in &mut self.language_layers {
            f32_tensor(&mut l.input_norm_scale)?;
            f32_tensor(&mut l.post_attention_norm_scale)?;
            attention(&mut l.attention)?;
            mlp(&mut l.mlp)?;
        }
        f32_tensor(&mut self.language_final_norm_scale)?;
        for l in &mut self.action_layers {
            linear(&mut l.input_norm.modulation)?;
            linear(&mut l.post_attention_norm.modulation)?;
            attention(&mut l.attention)?;
            mlp(&mut l.mlp)?;
        }
        for w in [
            &mut self.action_final_norm.modulation,
            &mut self.action_in,
            &mut self.action_out,
            &mut self.time_mlp_in,
            &mut self.time_mlp_out,
        ] {
            linear(w)?;
        }
        Ok(self)
    }
}

impl Pi05Weights {
    fn validate_cpu_shapes(&self, c: &Pi05Config) -> Result<()> {
        super::host::validate_shapes(c, self)?;
        let shape = super::host::expect_shape;
        let linear = super::host::expect_linear;
        let norm = |name: &str, w: &LayerNormWeights, width| -> Result<()> {
            shape(name, &w.weight, &[width])?;
            shape(name, &w.bias, &[width])
        };
        let v = &self.vision;
        if v.blocks.len() != c.vision_depth {
            return Err(Error::Other("PI0.5 CPU vision depth mismatch".into()));
        }
        linear(
            "patch_embedding",
            &v.patch_embedding,
            3 * c.patch_size * c.patch_size,
            c.vision_width,
        )?;
        shape(
            "position_embedding",
            &v.position_embedding,
            &[c.patches_per_view(), c.vision_width],
        )?;
        norm("vision.post_norm", &v.post_layer_norm, c.vision_width)?;
        for b in &v.blocks {
            norm("vision.norm1", &b.norm1, c.vision_width)?;
            norm("vision.norm2", &b.norm2, c.vision_width)?;
            for w in [&b.q, &b.k, &b.v, &b.output] {
                linear("vision.attention", w, c.vision_width, c.vision_width)?;
            }
            linear("vision.fc1", &b.fc1, c.vision_width, c.vision_mlp_dim)?;
            linear("vision.fc2", &b.fc2, c.vision_mlp_dim, c.vision_width)?;
        }
        for l in &self.language_layers {
            shape(
                "language.input_norm",
                &l.input_norm_scale,
                &[c.language.width],
            )?;
            shape(
                "language.post_attention_norm",
                &l.post_attention_norm_scale,
                &[c.language.width],
            )?;
        }
        for l in &self.action_layers {
            linear(
                "action.input_modulation",
                &l.input_norm.modulation,
                c.action_expert.width,
                3 * c.action_expert.width,
            )?;
            linear(
                "action.post_modulation",
                &l.post_attention_norm.modulation,
                c.action_expert.width,
                3 * c.action_expert.width,
            )?;
        }
        linear(
            "action.final_modulation",
            &self.action_final_norm.modulation,
            c.action_expert.width,
            3 * c.action_expert.width,
        )
    }
}

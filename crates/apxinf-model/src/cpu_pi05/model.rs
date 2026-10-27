//! Unfused FP32 PI0.5 reference blocks. BLAS owns projection/attention parallelism.
use crate::cpu_pi05::weights::{
    GemmaAttentionWeights, GemmaMlpWeights, LayerNormWeights, LinearWeights, Weights,
};
use crate::cpu_pi05::{Config, GemmaConfig};
use apxinf_core::{Backend, CpuBackend, Error, Result, Tensor};

pub(in crate::cpu_pi05) struct CpuPi05Model {
    config: Config,
    weights: Weights,
}
pub struct PrefixKvCache {
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    tokens: usize,
}
pub struct StepModulation {
    attention: Vec<Tensor>,
    mlp: Vec<Tensor>,
    final_norm: Tensor,
}

fn linear(x: &Tensor, w: &LinearWeights) -> Result<Tensor> {
    let out = CpuBackend.matmul(x, &w.weight)?;
    match &w.bias {
        Some(b) => CpuBackend.add_bias(&out, b),
        None => Ok(out),
    }
}
fn norm(x: &Tensor, w: &LayerNormWeights, eps: f32) -> Result<Tensor> {
    CpuBackend.layer_norm(x, &w.weight, &w.bias, eps)
}
fn mlp(x: &Tensor, w: &GemmaMlpWeights) -> Result<Tensor> {
    let gate = CpuBackend.gelu_tanh(&linear(x, &w.gate)?)?;
    linear(&CpuBackend.mul(&gate, &linear(x, &w.up)?)?, &w.down)
}
fn concat(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.ndim() != b.ndim() || a.shape().dims()[1..] != b.shape().dims()[1..] {
        return Err(Error::Other("PI0.5 CPU concat shape mismatch".into()));
    }
    let mut shape = a.shape().dims().to_vec();
    shape[0] += b.shape().dims()[0];
    let mut values = a.as_f32()?.to_vec();
    values.extend_from_slice(b.as_f32()?);
    Tensor::from_f32(shape, &values)
}
fn qkv(
    x: &Tensor,
    w: &GemmaAttentionWeights,
    c: GemmaConfig,
    offset: usize,
    theta: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let n = x.shape().dims()[0];
    let q = linear(x, &w.q)?.reshape(vec![n, c.num_heads, c.head_dim])?;
    let k = linear(x, &w.k)?.reshape(vec![n, c.num_kv_heads, c.head_dim])?;
    let v = linear(x, &w.v)?.reshape(vec![n, c.num_kv_heads, c.head_dim])?;
    Ok((
        CpuBackend.rope(&q, c.num_heads, c.head_dim, theta, offset as u32)?,
        CpuBackend.rope(&k, c.num_kv_heads, c.head_dim, theta, offset as u32)?,
        v,
    ))
}
// The projection is [scale, shift, gate]; the final norm ignores gate.
fn adaptive(x: &Tensor, modulation: &Tensor, eps: f32) -> Result<Tensor> {
    let width = x.shape().dims()[1];
    let m = modulation.as_f32()?;
    if m.len() != 3 * width {
        return Err(Error::Other("PI0.5 CPU modulation width mismatch".into()));
    }
    let ones = Tensor::from_f32(vec![width], &vec![1.; width])?;
    let normalized = CpuBackend.rms_norm(x, &ones, eps)?;
    let out: Vec<_> = normalized
        .as_f32()?
        .iter()
        .enumerate()
        .map(|(i, v)| v * (1. + m[i % width]) + m[width + i % width])
        .collect();
    Tensor::from_f32(x.shape().clone(), &out)
}
fn gated_residual(x: &Tensor, update: &Tensor, modulation: &Tensor) -> Result<Tensor> {
    let width = x.shape().dims()[1];
    let m = modulation.as_f32()?;
    if x.shape() != update.shape() || m.len() != 3 * width {
        return Err(Error::Other(
            "PI0.5 CPU gated residual shape mismatch".into(),
        ));
    }
    let out: Vec<_> = x
        .as_f32()?
        .iter()
        .zip(update.as_f32()?)
        .enumerate()
        .map(|(i, (x, u))| x + u * m[2 * width + i % width])
        .collect();
    Tensor::from_f32(x.shape().clone(), &out)
}
impl CpuPi05Model {
    pub(in crate::cpu_pi05) fn new(config: Config, weights: Weights) -> Result<Self> {
        config.validate()?;
        let weights = weights.into_f32(&config)?;
        Ok(Self { config, weights })
    }
    fn modulation(&self, time: &Tensor) -> Result<StepModulation> {
        let w = &self.weights;
        let cond = CpuBackend.silu(&linear(time, &w.time_mlp_in)?)?;
        let cond = CpuBackend.silu(&linear(&cond, &w.time_mlp_out)?)?;
        Ok(StepModulation {
            attention: w
                .action_layers
                .iter()
                .map(|l| linear(&cond, &l.input_norm.modulation))
                .collect::<Result<_>>()?,
            mlp: w
                .action_layers
                .iter()
                .map(|l| linear(&cond, &l.post_attention_norm.modulation))
                .collect::<Result<_>>()?,
            final_norm: linear(&cond, &w.action_final_norm.modulation)?,
        })
    }
}
impl CpuPi05Model {
    fn vision(&self, patches: &Tensor) -> Result<Tensor> {
        let c = &self.config;
        let w = &self.weights.vision;
        let n = c.patches_per_view();
        let width = c.vision_width;
        let projected = linear(patches, &w.patch_embedding)?;
        let pos = w.position_embedding.as_f32()?;
        if pos.len() != n * width {
            return Err(Error::Other(
                "PI0.5 position embedding shape mismatch".into(),
            ));
        }
        let data: Vec<_> = projected
            .as_f32()?
            .iter()
            .enumerate()
            .map(|(i, v)| v + pos[i % (n * width)])
            .collect();
        let mut hidden = Tensor::from_f32(projected.shape().clone(), &data)?;
        for b in &w.blocks {
            let x = norm(&hidden, &b.norm1, c.layer_norm_eps)?;
            let q = linear(&x, &b.q)?;
            let k = linear(&x, &b.k)?;
            let v = linear(&x, &b.v)?;
            let stride = n * c.vision_heads * c.vision_head_dim;
            let mut values = Vec::with_capacity(c.num_views * stride);
            for view in 0..c.num_views {
                let head = |t: &Tensor| {
                    Tensor::from_f32(
                        vec![n, c.vision_heads, c.vision_head_dim],
                        &t.as_f32()?[view * stride..(view + 1) * stride],
                    )
                };
                values.extend_from_slice(
                    CpuBackend
                        .attention(&head(&q)?, &head(&k)?, &head(&v)?)?
                        .as_f32()?,
                );
            }
            let attn = Tensor::from_f32(
                vec![c.num_views * n, c.vision_heads * c.vision_head_dim],
                &values,
            )?;
            hidden = CpuBackend.add(&hidden, &linear(&attn, &b.output)?)?;
            let x = norm(&hidden, &b.norm2, c.layer_norm_eps)?;
            let x = CpuBackend.gelu_tanh(&linear(&x, &b.fc1)?)?;
            hidden = CpuBackend.add(&hidden, &linear(&x, &b.fc2)?)?;
        }
        linear(
            &norm(&hidden, &w.post_layer_norm, c.layer_norm_eps)?,
            &w.multimodal_projector,
        )
    }
    fn embed_prefix(&self, vision: &Tensor, ids: &[u32], count: usize) -> Result<Tensor> {
        if ids.len() != count
            || count == 0
            || count > self.config.max_token_len
            || ids.iter().any(|&i| i as usize >= self.config.vocab_size)
        {
            return Err(Error::Other("PI0.5 CPU invalid token IDs/count".into()));
        }
        let lang = CpuBackend.embedding(&self.weights.vision.token_embedding, ids)?;
        concat(
            vision,
            &CpuBackend.scale(&lang, (self.config.language.width as f32).sqrt())?,
        )
    }
    fn prefix(&self, input: &Tensor) -> Result<PrefixKvCache> {
        let c = &self.config;
        let mut hidden = input.clone();
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for (i, l) in self.weights.language_layers.iter().enumerate() {
            let x = CpuBackend.rms_norm(&hidden, &l.input_norm_scale, c.rms_norm_eps)?;
            let (q, k, v) = qkv(&x, &l.attention, c.language, 0, c.rope_theta)?;
            if i + 1 < c.language.depth {
                let a = CpuBackend.attention(&q, &k, &v)?.reshape(vec![
                    input.shape().dims()[0],
                    c.language.num_heads * c.language.head_dim,
                ])?;
                hidden = CpuBackend.add(&hidden, &linear(&a, &l.attention.output)?)?;
                let x =
                    CpuBackend.rms_norm(&hidden, &l.post_attention_norm_scale, c.rms_norm_eps)?;
                hidden = CpuBackend.add(&hidden, &mlp(&x, &l.mlp)?)?;
            }
            keys.push(k);
            values.push(v);
        }
        Ok(PrefixKvCache {
            keys,
            values,
            tokens: input.shape().dims()[0],
        })
    }
    pub(super) fn prepare_all_modulation(
        &self,
        embeddings: &[Tensor],
    ) -> Result<Vec<StepModulation>> {
        if embeddings.len() != self.config.num_flow_steps {
            return Err(Error::Other("PI0.5 CPU timestep count mismatch".into()));
        }
        embeddings.iter().map(|e| self.modulation(e)).collect()
    }
    #[cfg(test)]
    fn step(
        &self,
        state: &Tensor,
        time: &Tensor,
        prefix: &PrefixKvCache,
        dt: f32,
    ) -> Result<Tensor> {
        self.step_with_modulation(state, &self.modulation(time)?, prefix, dt)
    }
    fn step_with_modulation(
        &self,
        state: &Tensor,
        m: &StepModulation,
        prefix: &PrefixKvCache,
        dt: f32,
    ) -> Result<Tensor> {
        let c = &self.config;
        let w = &self.weights;
        let mut hidden = linear(state, &w.action_in)?;
        for (i, l) in w.action_layers.iter().enumerate() {
            let x = adaptive(&hidden, &m.attention[i], c.rms_norm_eps)?;
            let (q, k, v) = qkv(
                &x,
                &l.attention,
                c.action_expert,
                prefix.tokens,
                c.rope_theta,
            )?;
            let k = concat(&prefix.keys[i], &k)?;
            let v = concat(&prefix.values[i], &v)?;
            let a = CpuBackend.attention(&q, &k, &v)?.reshape(vec![
                c.action_horizon,
                c.action_expert.num_heads * c.action_expert.head_dim,
            ])?;
            hidden = gated_residual(&hidden, &linear(&a, &l.attention.output)?, &m.attention[i])?;
            let x = adaptive(&hidden, &m.mlp[i], c.rms_norm_eps)?;
            hidden = gated_residual(&hidden, &mlp(&x, &l.mlp)?, &m.mlp[i])?;
        }
        let velocity = linear(
            &adaptive(&hidden, &m.final_norm, c.rms_norm_eps)?,
            &w.action_out,
        )?;
        CpuBackend.add(state, &CpuBackend.scale(&velocity, dt)?)
    }
}

impl CpuPi05Model {
    /// CPU-owned forward order and Euler schedule. No CUDA model dependency.
    pub(super) fn infer_with_modulation(
        &self,
        patches: &Tensor,
        ids: &[u32],
        noise: &Tensor,
        modulation: &[StepModulation],
    ) -> Result<Tensor> {
        if modulation.len() != self.config.num_flow_steps {
            return Err(Error::Other("PI0.5 CPU modulation count mismatch".into()));
        }
        let vision = self.vision(patches)?;
        let prefix = self.embed_prefix(&vision, ids, ids.len())?;
        let prefix = self.prefix(&prefix)?;
        let dt = -self.config.flow_start_time / self.config.num_flow_steps as f32;
        let mut state = noise.clone();
        for m in modulation {
            state = self.step_with_modulation(&state, m, &prefix, dt)?;
        }
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires APXINF_PI05_CPU_REFERENCE fixture directory"]
    fn intermediate_stages_match_independent_reference() {
        let dir = std::path::PathBuf::from(
            std::env::var("APXINF_PI05_CPU_REFERENCE").expect("fixture directory"),
        );
        let lang = GemmaConfig {
            width: 8,
            depth: 2,
            mlp_dim: 12,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
        };
        let c = Config {
            action_dim: 3,
            action_horizon: 3,
            max_token_len: 4,
            num_flow_steps: 3,
            num_views: 2,
            image_size: 4,
            patch_size: 2,
            vision_width: 8,
            vision_depth: 2,
            vision_mlp_dim: 12,
            vision_heads: 2,
            vision_head_dim: 4,
            vocab_size: 16,
            language: lang,
            action_expert: GemmaConfig { width: 6, ..lang },
            ..Config::default()
        };
        let b = CpuPi05Model::new(c.clone(), Weights::from_safetensors(&c, &dir).unwrap()).unwrap();
        let cases: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("cases.json")).unwrap())
                .unwrap();
        let check = |actual: &Tensor, expected: &serde_json::Value| {
            let e: Vec<f32> = serde_json::from_value(expected.clone()).unwrap();
            let a = actual.as_f32().unwrap();
            assert_eq!(a.len(), e.len());
            let max = a
                .iter()
                .zip(&e)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("CPU stage max_abs={max:e}");
            for (a, e) in a.iter().zip(e) {
                assert!((a - e).abs() < 2e-5 + 2e-5 * e.abs(), "{a} != {e}");
            }
        };
        for case in cases.as_array().unwrap() {
            let rgb: Vec<u8> = serde_json::from_value(case["rgb"].clone()).unwrap();
            let mut patches = Vec::new();
            for view in 0..2 {
                for py in 0..2 {
                    for px in 0..2 {
                        for ch in 0..3 {
                            for dy in 0..2 {
                                for dx in 0..2 {
                                    patches.push(
                                        rgb[((view * 4 + py * 2 + dy) * 4 + px * 2 + dx) * 3 + ch]
                                            as f32
                                            * (2. / 255.)
                                            - 1.,
                                    );
                                }
                            }
                        }
                    }
                }
            }
            let vision = b
                .vision(&Tensor::from_f32(vec![8, 12], &patches).unwrap())
                .unwrap();
            check(&vision, &case["vision"]);
            let ids: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
            let prefix = b.embed_prefix(&vision, &ids, ids.len()).unwrap();
            check(&prefix, &case["prefix"]);
            let cache = b.prefix(&prefix).unwrap();
            for i in 0..2 {
                check(&cache.keys[i], &case["keys"][i]);
                check(&cache.values[i], &case["values"][i]);
            }
            let noise: Vec<f32> = serde_json::from_value(case["noise"].clone()).unwrap();
            let mut state = Tensor::from_f32(vec![3, 3], &noise).unwrap();
            for step in 0..3 {
                let time = crate::cpu_pi05::config::sinusoidal_time_embedding(
                    1. - step as f32 / 3.,
                    6,
                    c.time_min_period,
                    c.time_max_period,
                );
                state = b
                    .step(
                        &state,
                        &Tensor::from_f32(vec![1, 6], &time).unwrap(),
                        &cache,
                        -1. / 3.,
                    )
                    .unwrap();
                check(&state, &case["steps"][step]);
            }
        }
    }
}

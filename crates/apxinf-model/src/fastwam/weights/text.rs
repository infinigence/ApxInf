//! Immutable checkpoint mapping and prepared weight data.
use super::{host_tensor, validate_tensor, Bias, Linear, Store};
use crate::fastwam::backend::{self, Context};
use apxinf_core::{DType, Error, Result, Tensor};
use std::path::Path;
pub(in crate::fastwam) struct Block {
    pub(in crate::fastwam) norm0: Tensor,
    pub(in crate::fastwam) q: Linear,
    pub(in crate::fastwam) k: Linear,
    pub(in crate::fastwam) v: Linear,
    pub(in crate::fastwam) o: Linear,
    pub(in crate::fastwam) bias: Tensor,
    pub(in crate::fastwam) norm1: Tensor,
    pub(in crate::fastwam) wi0: Linear,
    pub(in crate::fastwam) wi1: Linear,
    pub(in crate::fastwam) wo: Linear,
}

pub(in crate::fastwam) struct TextEncoder {
    pub(in crate::fastwam) embedding: Tensor,
    pub(in crate::fastwam) blocks: Vec<Block>,
    pub(in crate::fastwam) norm: Tensor,
}

impl TextEncoder {
    pub(in crate::fastwam) fn load(ctx: &Context, path: &Path) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.join("config.json")).map_err(|e| Error::Other(e.to_string()))?,
        )
        .map_err(|e| Error::Other(e.to_string()))?;
        for (key, value) in [
            ("d_model", 4096),
            ("d_ff", 10240),
            ("d_kv", 64),
            ("num_heads", 64),
            ("num_layers", 24),
            ("relative_attention_num_buckets", 32),
            ("relative_attention_max_distance", 128),
            ("vocab_size", 256384),
        ] {
            if v[key].as_u64() != Some(value) {
                return Err(Error::Other(format!("unsupported Fast-WAM UMT5 {key}")));
            }
        }
        if v["dense_act_fn"] != "gelu_new"
            || v["feed_forward_proj"] != "gated-gelu"
            || v["layer_norm_epsilon"].as_f64() != Some(1e-6)
        {
            return Err(Error::Other(
                "unsupported UMT5 activation or normalization".into(),
            ));
        }
        let mut store = Store::load(path)?;
        let embedding = store.upload(ctx, "shared.weight")?;
        validate_tensor("UMT5 embedding", &embedding, &[256384, 4096], DType::BF16)?;
        let mut blocks = Vec::new();
        for i in 0..24 {
            let p = format!("encoder.block.{i}");
            let a = format!("{p}.layer.0.SelfAttention");
            let f = format!("{p}.layer.1.DenseReluDense");
            let relative = store.take(&format!("{a}.relative_attention_bias.weight"))?;
            if relative.shape().dims() != [32, 64] {
                return Err(Error::Other("UMT5 relative bias must be [32,64]".into()));
            }
            let relative = relative.to_f32_vec()?;
            let mut expanded = vec![0.; 64 * 128 * 128];
            for h in 0..64 {
                for q in 0..128 {
                    for k in 0..128 {
                        let distance = (k as i32 - q as i32).unsigned_abs() as usize;
                        let bucket = if distance < 8 {
                            distance
                        } else {
                            8 + ((distance as f32 / 8.).ln() / 16f32.ln() * 8.) as usize
                        }
                        .min(15)
                            + if k > q { 16 } else { 0 };
                        expanded[(h * 128 + q) * 128 + k] = relative[bucket * 64 + h];
                    }
                }
            }
            blocks.push(Block {
                norm0: store.upload(ctx, &format!("{p}.layer.0.layer_norm.weight"))?,
                q: store.linear(ctx, &format!("{a}.q"), Bias::Forbidden)?,
                k: store.linear(ctx, &format!("{a}.k"), Bias::Forbidden)?,
                v: store.linear(ctx, &format!("{a}.v"), Bias::Forbidden)?,
                o: store.linear(ctx, &format!("{a}.o"), Bias::Forbidden)?,
                bias: backend::upload(
                    ctx,
                    &host_tensor(vec![64 * 128, 128], &expanded, DType::F32)?,
                )?,
                norm1: store.upload(ctx, &format!("{p}.layer.1.layer_norm.weight"))?,
                wi0: store.linear(ctx, &format!("{f}.wi_0"), Bias::Forbidden)?,
                wi1: store.linear(ctx, &format!("{f}.wi_1"), Bias::Forbidden)?,
                wo: store.linear(ctx, &format!("{f}.wo"), Bias::Forbidden)?,
            });
        }
        let norm = store.upload(ctx, "encoder.final_layer_norm.weight")?;
        validate_tensor("UMT5 final norm", &norm, &[4096], DType::BF16)?;
        for b in &blocks {
            for projection in [&b.q, &b.k, &b.v, &b.o] {
                projection.validate("UMT5 attention projection", 4096, 4096)?;
            }
            b.wi0.validate("UMT5 gate", 4096, 10240)?;
            b.wi1.validate("UMT5 up", 4096, 10240)?;
            b.wo.validate("UMT5 down", 10240, 4096)?;
            validate_tensor("UMT5 norm", &b.norm0, &[4096], DType::BF16)?;
            validate_tensor("UMT5 norm", &b.norm1, &[4096], DType::BF16)?;
        }
        Ok(Self {
            embedding,
            blocks,
            norm,
        })
    }
}

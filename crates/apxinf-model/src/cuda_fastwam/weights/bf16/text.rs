//! Immutable checkpoint mapping and prepared weight data.
use super::{host_tensor, validate_tensor, Bias, LinearWeights, Store};
use super::{load_linear, upload_tensor};
use crate::cuda_fastwam::backend::{self, Context};
use apxinf_core::{DType, Error, Result, Tensor};
use std::path::Path;
pub(in crate::cuda_fastwam) struct BlockWeights {
    pub(in crate::cuda_fastwam) norm0: Tensor,
    pub(in crate::cuda_fastwam) q: LinearWeights,
    pub(in crate::cuda_fastwam) k: LinearWeights,
    pub(in crate::cuda_fastwam) v: LinearWeights,
    pub(in crate::cuda_fastwam) o: LinearWeights,
    pub(in crate::cuda_fastwam) bias: Tensor,
    pub(in crate::cuda_fastwam) norm1: Tensor,
    pub(in crate::cuda_fastwam) wi0: LinearWeights,
    pub(in crate::cuda_fastwam) wi1: LinearWeights,
    pub(in crate::cuda_fastwam) wo: LinearWeights,
}

pub(in crate::cuda_fastwam) struct TextEncoderWeights {
    pub(in crate::cuda_fastwam) embedding: Tensor,
    pub(in crate::cuda_fastwam) blocks: Vec<BlockWeights>,
    pub(in crate::cuda_fastwam) norm: Tensor,
}

impl TextEncoderWeights {
    pub(in crate::cuda_fastwam) fn load(ctx: &Context, path: &Path) -> Result<Self> {
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
        let embedding = upload_tensor(&mut store, ctx, "shared.weight")?;
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
            blocks.push(BlockWeights {
                norm0: upload_tensor(&mut store, ctx, &format!("{p}.layer.0.layer_norm.weight"))?,
                q: load_linear(&mut store, ctx, &format!("{a}.q"), Bias::Forbidden)?,
                k: load_linear(&mut store, ctx, &format!("{a}.k"), Bias::Forbidden)?,
                v: load_linear(&mut store, ctx, &format!("{a}.v"), Bias::Forbidden)?,
                o: load_linear(&mut store, ctx, &format!("{a}.o"), Bias::Forbidden)?,
                bias: backend::upload(
                    ctx,
                    &host_tensor(vec![64 * 128, 128], &expanded, DType::F32)?,
                )?,
                norm1: upload_tensor(&mut store, ctx, &format!("{p}.layer.1.layer_norm.weight"))?,
                wi0: load_linear(&mut store, ctx, &format!("{f}.wi_0"), Bias::Forbidden)?,
                wi1: load_linear(&mut store, ctx, &format!("{f}.wi_1"), Bias::Forbidden)?,
                wo: load_linear(&mut store, ctx, &format!("{f}.wo"), Bias::Forbidden)?,
            });
        }
        let norm = upload_tensor(&mut store, ctx, "encoder.final_layer_norm.weight")?;
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

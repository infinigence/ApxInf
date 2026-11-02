//! Dense attention contracts.

use apxinf_core::{Error, Result, Tensor};

use super::require_unit_output_scale;
use crate::kernels::attention as kernels;
use crate::kernels::not_ported;
use crate::MusaContext;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionMask {
    None,
    Causal,
}

/// What a model wants from dense attention, declared at load.
///
/// **Carried, not consumed**, for the reason [`GemmPolicy`] documents: the
/// session `configure_tuning` builds is what decides the tactic, so these
/// fields are accepted and ignored here.
///
/// [`GemmPolicy`]: super::gemm::GemmPolicy
#[derive(Clone, Debug)]
pub struct AttentionPolicy {
    pub workspace_limit: usize,
    pub online_tune: bool,
    pub allow_fallback: bool,
    pub graph_safe: bool,
    pub deterministic: bool,
    /// None uses the context's tuning directory; Some("") disables disk caching.
    pub cache_dir: Option<String>,
}

impl Default for AttentionPolicy {
    fn default() -> Self {
        Self {
            workspace_limit: 256 * 1024 * 1024,
            online_tune: true,
            allow_fallback: true,
            graph_safe: true,
            deterministic: false,
            cache_dir: None,
        }
    }
}

/// Canonical dense attention contract. Tensors are contiguous row-major:
/// Q `[batch, query_tokens, query_heads, head_dim]`, K/V
/// `[batch, key_tokens, kv_heads, head_dim]`, and the output has Q's shape.
pub struct AttentionArgs<'a> {
    pub query: &'a Tensor,
    pub key: &'a Tensor,
    pub value: &'a Tensor,
    pub out: &'a mut Tensor,
    pub mask: AttentionMask,
    pub scale: f32,
    pub output_scale: f32,
    pub policy: AttentionPolicy,
}

impl<'a> AttentionArgs<'a> {
    pub fn new(query: &'a Tensor, key: &'a Tensor, value: &'a Tensor, out: &'a mut Tensor) -> Self {
        let head_dim = query.shape().dims().get(3).copied().unwrap_or(1);
        Self {
            query,
            key,
            value,
            out,
            mask: AttentionMask::None,
            scale: 1.0 / (head_dim as f32).sqrt(),
            output_scale: 1.0,
            policy: AttentionPolicy::default(),
        }
    }
}

pub fn attention(ctx: &MusaContext, args: AttentionArgs<'_>) -> Result<()> {
    require_unit_output_scale("attention", args.output_scale)?;
    if args.mask == AttentionMask::Causal {
        return Err(not_ported("causal attention"));
    }
    let query_dims = args.query.shape().dims();
    let [batch, tokens, heads, head_dim] = query_dims else {
        return Err(Error::Other(format!(
            "MUSA attention expects rank-4 q [batch, tokens, heads, head_dim], got {query_dims:?}"
        )));
    };
    let (batch, tokens, heads, head_dim) = (*batch, *tokens, *heads, *head_dim);
    if args.out.shape().dims() != query_dims {
        return Err(Error::Other(format!(
            "MUSA attention out must have q's shape {query_dims:?}, got {:?}",
            args.out.shape().dims()
        )));
    }
    let expected_scale = 1.0 / (head_dim as f32).sqrt();
    if (args.scale - expected_scale).abs() > 1e-6 {
        return Err(not_ported("attention with a non-default scale"));
    }

    let mut query = args
        .query
        .reshape(vec![batch * tokens, heads, head_dim])?;
    let mut out = args.out.reshape(vec![batch * tokens, heads, head_dim])?;
    if args.key.shape().dims() == query_dims {
        let [key_batch, key_tokens, key_heads, key_dim] = args.key.shape().dims() else {
            unreachable!("shape compared above");
        };
        if (*key_batch, *key_tokens, *key_heads, *key_dim) != (batch, tokens, heads, head_dim) {
            return Err(not_ported("attention whose k/v differ from q in shape"));
        }
        let mut key = args.key.reshape(vec![batch * tokens, heads, head_dim])?;
        let mut value = args.value.reshape(vec![batch * tokens, heads, head_dim])?;
        // The vision kernel folds the batch into the token axis; it reads
        // `tokens_per_batch` to keep the per-image spans apart.
        kernels::mha_bf16_into(ctx, &mut query, &mut key, &mut value, &mut out, tokens)
    } else {
        if batch != 1 {
            return Err(not_ported("batched decode attention"));
        }
        let key_tokens = key_rows(args.key, head_dim, "key")?;
        if key_rows(args.value, head_dim, "value")? != key_tokens {
            return Err(Error::Other(
                "MUSA attention k and v caches hold different row counts".into(),
            ));
        }
        // The kernel reads the cache as `key_tokens` rows of `head_dim`; every
        // accepted rank describes the same bytes of one contiguous buffer.
        let key = args.key.reshape(vec![key_tokens, head_dim])?;
        let value = args.value.reshape(vec![key_tokens, head_dim])?;
        kernels::mqa_bf16_into(ctx, &mut query, &key, &value, &mut out, key_tokens)
    }
}

/// Rows in a KV cache: 2-D `[rows, head_dim]`, 3-D `[rows, 1, head_dim]`, or
/// the 4-D `[1, rows, 1, head_dim]` the prefix layer builds.
fn key_rows(tensor: &Tensor, head_dim: usize, name: &str) -> Result<usize> {
    let dims = tensor.shape().dims();
    match dims {
        [rows, width] if *width == head_dim => Ok(*rows),
        [rows, 1, width] if *width == head_dim => Ok(*rows),
        [1, rows, 1, width] if *width == head_dim => Ok(*rows),
        _ => Err(Error::Other(format!(
            "MUSA attention expects a {name} cache of [rows, {head_dim}], got {dims:?}"
        ))),
    }
}

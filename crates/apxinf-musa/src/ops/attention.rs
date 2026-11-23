//! Dense attention contracts.

use apxinf_core::{Error, Result, Tensor};

use super::{require_deterministic, require_unit_output_scale, require_workspace};
use crate::kernels::attention as kernels;
use crate::kernels::{not_ported, TacticDispatch};
use crate::MusaContext;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionMask {
    None,
    Causal,
}

/// What a model wants from dense attention, declared at load.
///
/// Every field is either enforced here or refused by name. `graph_safe` is the
/// exception in form rather than in effect: MUSA's routes are capture-safe
/// unconditionally, so `true` (the requirement) and `false` (a weaker request)
/// are both satisfied.
#[derive(Clone, Debug)]
pub struct AttentionPolicy {
    /// Bounds the bytes the operator allocates for itself. Attention writes only
    /// the caller's output and allocates no intermediate, so any limit —
    /// including zero — is satisfied. Vendor-library internal workspaces are
    /// outside this backend's accounting.
    pub workspace_limit: usize,
    /// `false` requires the measured packed decode route for a decode-shaped
    /// call, per build: the assembly build's packed decode kernel, or the
    /// MUTLASS build's packed tile 128 (9..=16 query rows). The general D256
    /// assembly, and the MUTLASS unpacked tile 192 for 17..=256 rows, are
    /// substitutes and are refused. A prefix pass uses the general route by
    /// construction and is not a fallback.
    pub allow_fallback: bool,
    /// Both values are accepted; see the type-level note.
    pub graph_safe: bool,
    /// Bitwise determinism is not guaranteed across the available routes, so
    /// `true` is refused rather than accepted without the guarantee.
    pub deterministic: bool,
}

impl Default for AttentionPolicy {
    fn default() -> Self {
        Self {
            workspace_limit: 256 * 1024 * 1024,
            allow_fallback: true,
            graph_safe: true,
            deterministic: false,
        }
    }
}

/// The constraints this backend carries into one attention call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdmittedAttentionPolicy {
    tactic: TacticDispatch,
}

impl AttentionPolicy {
    /// Validate every field and keep only what execution must honour.
    fn admit(&self, operation: &str) -> Result<AdmittedAttentionPolicy> {
        require_deterministic(operation, self.deterministic)?;
        // Attention allocates no workspace of its own.
        require_workspace(operation, 0, self.workspace_limit)?;
        Ok(AdmittedAttentionPolicy {
            tactic: if self.allow_fallback {
                TacticDispatch::AllowFallback
            } else {
                TacticDispatch::RequireExact
            },
        })
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
    let admitted = args.policy.admit("attention")?;
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
        kernels::mha_bf16_into_with(
            ctx,
            &mut query,
            &mut key,
            &mut value,
            &mut out,
            tokens,
            admitted.tactic,
        )
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
        kernels::mqa_bf16_into_with(
            ctx,
            &mut query,
            &key,
            &value,
            &mut out,
            key_tokens,
            admitted.tactic,
        )
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

#[cfg(test)]
mod policy_tests {
    use super::*;

    fn policy() -> AttentionPolicy {
        AttentionPolicy::default()
    }

    #[test]
    fn the_default_policy_is_admitted() {
        let admitted = policy().admit("attention").expect("the default policy");
        assert_eq!(admitted.tactic, TacticDispatch::AllowFallback);
    }

    #[test]
    fn deterministic_is_refused() {
        let error = AttentionPolicy {
            deterministic: true,
            ..policy()
        }
        .admit("attention")
        .expect_err("there is no bitwise-determinism contract to honour");
        assert!(error.to_string().contains("deterministic"), "{error}");
    }

    #[test]
    fn allow_fallback_false_requires_an_exact_tactic() {
        let admitted = AttentionPolicy {
            allow_fallback: false,
            ..policy()
        }
        .admit("attention")
        .expect("a valid policy");
        assert_eq!(admitted.tactic, TacticDispatch::RequireExact);
    }

    #[test]
    fn a_zero_workspace_limit_is_satisfied() {
        assert!(AttentionPolicy {
            workspace_limit: 0,
            ..policy()
        }
        .admit("attention")
        .is_ok());
    }

    #[test]
    fn graph_safe_is_accepted_either_way() {
        assert!(AttentionPolicy {
            graph_safe: true,
            ..policy()
        }
        .admit("attention")
        .is_ok());
        assert!(AttentionPolicy {
            graph_safe: false,
            ..policy()
        }
        .admit("attention")
        .is_ok());
    }
}

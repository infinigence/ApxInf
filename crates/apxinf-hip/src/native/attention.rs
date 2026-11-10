//! Attention, through Composable Kernel's FMHA where it applies and apxinf's
//! own kernel everywhere else.
//!
//! Two providers, for the same reason the GEMM has two: measured on GR00T's
//! four attention shapes, CK runs 6.9x to 17.3x faster than the kernel in
//! `kernels/native_attention.hip`, which reaches under 3% of this device's
//! measured compute ceiling because it launches one block per query row and
//! computes Q.K with a scalar loop. Unlike the GEMM, there is no shape here
//! where the slower provider wins, so CK is used wherever it can be rather
//! than from a measured table.
//!
//! CK cannot serve everything. The vendored instances are generated without a
//! mask, so a causal call has no instance; `apxinf_hip_ck_attention` reports
//! that and the call continues on apxinf's kernel, which covers every
//! geometry. `APXINF_HIP_ATTENTION=apxinf` forces that path for A/B
//! measurement.
//!
//! See `crates/apxinf-hip/kernels/ck_attention.hip` for what CK costs in
//! precision and why it does not show end to end.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use apxinf_core::{Error, Result, Tensor};

use super::{bf16_ptr, int, long, require_numel};
use crate::ffi::{self, check, DTYPE_BF16};
use crate::runtime::HipContext;

/// Which provider served each attention call, as `(apxinf, CK)`.
///
/// The fallback is silent by design -- a causal call, or a build without the
/// vendored CK, lands on apxinf's kernel and returns the right answer -- so
/// without a count an A/B that never reached CK is indistinguishable from one
/// that found no speedup. The same reasoning as the GEMM's dispatch counters,
/// and the same hazard: a test asserting "CK matches the reference" passes
/// most convincingly when CK never ran.
static DISPATCHES: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];

/// `(apxinf, CK)` attention dispatch counts since [`reset_attention_counts`].
pub fn attention_counts() -> (usize, usize) {
    (
        DISPATCHES[0].load(Ordering::Relaxed),
        DISPATCHES[1].load(Ordering::Relaxed),
    )
}

pub fn reset_attention_counts() {
    DISPATCHES[0].store(0, Ordering::Relaxed);
    DISPATCHES[1].store(0, Ordering::Relaxed);
}

/// apxinf's own kernel, bypassing provider selection.
///
/// For tests that need both outputs in one process: the provider is resolved
/// once from the environment, so a test cannot get CK and the fallback by
/// setting a variable between calls.
#[doc(hidden)]
pub fn attention_apxinf_for_test(
    ctx: &HipContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    geometry: &AttentionGeometry,
    out: &Tensor,
) -> Result<()> {
    attention_inner(ctx, q, k, v, geometry, out, false)
}

/// Whether CK may be used at all, from `APXINF_HIP_ATTENTION`.
fn ck_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        match std::env::var("APXINF_HIP_ATTENTION")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "apxinf" | "native" => false,
            "" | "ck" | "auto" => true,
            other => {
                eprintln!(
                    "apxinf-hip: APXINF_HIP_ATTENTION={other} is not ck or apxinf; using ck"
                );
                true
            }
        }
    })
}

/// Extents of one attention call. Query head `h` reads KV head
/// `h / (q_heads / kv_heads)`.
#[derive(Clone, Copy, Debug)]
pub struct AttentionGeometry {
    pub batch: usize,
    pub q_tokens: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Keys every query attends to, from row 0.
    pub kv_tokens: usize,
    /// `None` attends every key (pi05's masks are all non-causal). `Some(s)`
    /// masks causally: query row `i` of `q_tokens` sees keys `[0, s + i]`,
    /// so `s = kv_tokens - q_tokens` is a prefill over a cache that already
    /// holds the earlier rows.
    pub causal_query_start: Option<usize>,
}

/// `out = softmax(q kᵀ / sqrt(head_dim)) v`, optionally causally masked.
///
/// `q` and `out` are `[batch, q_tokens, q_heads, head_dim]`. `k` and `v` are
/// `[batch, rows, kv_heads, head_dim]` with `rows >= kv_tokens`: a cache may
/// hold more rows than the call attends, and only the first `kv_tokens` of
/// each batch are read. Any leading-dimension grouping of these element
/// counts is accepted.
pub fn attention(
    ctx: &HipContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    geometry: &AttentionGeometry,
    out: &Tensor,
) -> Result<()> {
    attention_inner(ctx, q, k, v, geometry, out, ck_enabled())
}

fn attention_inner(
    ctx: &HipContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    geometry: &AttentionGeometry,
    out: &Tensor,
    allow_ck: bool,
) -> Result<()> {
    let AttentionGeometry {
        batch,
        q_tokens,
        q_heads,
        kv_heads,
        head_dim,
        kv_tokens,
        causal_query_start,
    } = *geometry;
    if batch == 0 || kv_heads == 0 || q_heads % kv_heads != 0 || head_dim == 0 || kv_tokens == 0 {
        return Err(Error::Contract("apxinf-hip attention: invalid geometry"));
    }
    let q_elements = batch * q_tokens * q_heads * head_dim;
    for t in [q, out] {
        require_numel(t, q_elements, || {
            format!("[{batch}, {q_tokens}, {q_heads}, {head_dim}]")
        })?;
    }
    let kv_row = kv_heads * head_dim;
    let per_batch = k.numel() / batch;
    if k.shape() != v.shape()
        || !k.numel().is_multiple_of(batch)
        || !per_batch.is_multiple_of(kv_row)
    {
        return Err(Error::ShapeMismatch {
            expected: format!("K and V of [{batch}, rows, {kv_heads}, {head_dim}]"),
            got: format!("{} and {}", k.shape(), v.shape()),
        });
    }
    if per_batch / kv_row < kv_tokens {
        return Err(Error::Other(format!(
            "apxinf-hip attention: {kv_tokens} keys requested, the cache holds {}",
            per_batch / kv_row
        )));
    }
    let (pq, pk, pv, po) = (
        bf16_ptr(ctx, q)?,
        bf16_ptr(ctx, k)?,
        bf16_ptr(ctx, v)?,
        bf16_ptr(ctx, out)?,
    );
    let query_start = match causal_query_start {
        None => -1,
        Some(start) => int(start)?,
    };
    let scale = 1.0 / (head_dim as f32).sqrt();

    if allow_ck
        && unsafe {
            ffi::apxinf_hip_ck_attention_supported(
                DTYPE_BF16,
                int(head_dim)?,
                query_start,
                int(q_tokens)?,
                int(kv_tokens)?,
            )
        } == 1
    {
        // `INVALID_ARGUMENT` here means no vendored instance covers the call,
        // which is a routing outcome: it falls through to apxinf's kernel.
        // Anything else is a real failure and is reported.
        let status = unsafe {
            ffi::apxinf_hip_ck_attention(
                DTYPE_BF16,
                pq,
                pk,
                pv,
                po,
                int(batch)?,
                int(q_tokens)?,
                int(q_heads)?,
                int(kv_heads)?,
                int(head_dim)?,
                int(kv_tokens)?,
                long(per_batch)?,
                long(kv_row)?,
                scale,
                query_start,
                ctx.stream(),
            )
        };
        if status == 0 {
            DISPATCHES[1].fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        if status != ffi::INVALID_ARGUMENT {
            return check("attention (CK)", status);
        }
    }

    DISPATCHES[0].fetch_add(1, Ordering::Relaxed);
    check("attention", unsafe {
        ffi::apxinf_hip_native_attention(
            pq,
            pk,
            pv,
            po,
            int(batch)?,
            int(q_tokens)?,
            int(q_heads)?,
            int(kv_heads)?,
            int(head_dim)?,
            int(kv_tokens)?,
            long(per_batch)?,
            long(kv_row)?,
            scale,
            query_start,
            ctx.stream(),
        )
    })
}

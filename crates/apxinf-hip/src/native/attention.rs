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
//! [`AttentionKernel`] is the caller's lever over that choice:
//! [`AttentionKernel::Reference`] runs apxinf's kernel, `Ck` and `CkSplitKv`
//! run CK's flash-attention forward -- split-KV when few query rows face many
//! keys, which one 16-row query tile per head cannot fill. [`attention_kernel`]
//! picks the default for a geometry; callers may force any kernel, which is
//! how tests compare them on the same inputs. Whatever the choice, a call the
//! vendored instances cannot cover falls back at run time to apxinf's kernel,
//! which covers every geometry, and `APXINF_HIP_ATTENTION=apxinf` forces that
//! path outright for A/B measurement.
//!
//! What CK cannot cover: the vendored forward instances take no mask and the
//! two causal alignments of `causal_query_start` (offsets 0 and `kv - q`); an
//! offset in between has no instance. The split-KV instances are unmasked.
//! Either way the C entry reports a routing miss and the call continues on
//! apxinf's kernel.
//!
//! See `crates/apxinf-hip/kernels/ck_attention.hip` for what CK costs in
//! precision and why it does not show end to end.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use apxinf_core::{DType, Error, Result, Tensor};

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
    let bound = validate_and_bind(ctx, q, k, v, geometry, out)?;
    launch_native(ctx, geometry, &bound, geometry.q_heads * geometry.head_dim)
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
    /// `None` attends every key. `Some(s)` masks causally: query row `i` of
    /// `q_tokens` sees keys `[0, s + i]`, so `s = kv_tokens - q_tokens` is a
    /// prefill over a cache that already holds the earlier rows.
    pub causal_query_start: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKernel {
    /// apxinf's own kernel.
    Reference,
    /// CK flash-attention forward.
    Ck,
    /// CK split-KV forward: keys split across `splits` workgroups per query
    /// tile, then combined. Needs [`attention_scratch_elements`] F32 scratch.
    CkSplitKv { splits: usize },
}

/// Key splits for CK split-KV. 4 measured fastest for pi05's action expert
/// (10 queries x 8 heads over 532 keys) on gfx1151: 24 us, against 57 us
/// unsplit; 2 and 8 splits were within 15%.
const SPLITKV_SPLITS: usize = 4;

/// The default kernel for a geometry on a device.
///
/// The vendored CK dispatchers gate on the device name at run time — the
/// forward instances on `gfx115*`, split-KV on `gfx11*` — so any other
/// architecture (CDNA, gfx12) gets apxinf's own kernel rather than a call
/// that cannot resolve. `arch` is `HipDeviceCaps::base_arch()`.
///
/// Within CK's architectures: CK covers head dims up to 256 in multiples of 8
/// (head dim 72 runs on the padded 128 instances); the vendored split-KV
/// instances are head dim 256 only. Split-KV pays off when few query rows face
/// many keys: one 16-row query tile per head cannot fill the GPU. Its
/// instances are unmasked, so a causal geometry takes the fused path instead.
pub fn attention_kernel(arch: &str, geometry: &AttentionGeometry) -> AttentionKernel {
    let g = geometry;
    if g.head_dim > 256 || !g.head_dim.is_multiple_of(8) {
        return AttentionKernel::Reference;
    }
    if g.causal_query_start.is_none()
        && g.head_dim == 256
        && g.q_tokens <= 16
        && g.kv_tokens >= 256
        && arch.starts_with("gfx11")
    {
        return AttentionKernel::CkSplitKv {
            splits: SPLITKV_SPLITS,
        };
    }
    if arch.starts_with("gfx115") {
        return AttentionKernel::Ck;
    }
    AttentionKernel::Reference
}

/// F32 scratch elements `kernel` needs for `geometry`: split-KV's partial
/// log-sum-exps and outputs; zero for the other kernels. The log-sum-exp block
/// is padded to 64 elements so the output block starts 256-byte aligned.
pub fn attention_scratch_elements(geometry: &AttentionGeometry, kernel: AttentionKernel) -> usize {
    match kernel {
        AttentionKernel::CkSplitKv { splits } => {
            let rows = geometry.batch * geometry.q_heads * splits * geometry.q_tokens;
            rows.next_multiple_of(64) + rows * geometry.head_dim
        }
        _ => 0,
    }
}

/// `out = softmax(q kᵀ / sqrt(head_dim)) v`, optionally causally masked,
/// computed by `kernel`.
///
/// `q` and `out` are `[batch, q_tokens, q_heads, head_dim]`. `k` and `v` are
/// `[batch, rows, kv_heads, head_dim]` with `rows >= kv_tokens`: a cache may
/// hold more rows than the call attends, and only the first `kv_tokens` of
/// each batch are read. Any leading-dimension grouping of these element
/// counts is accepted. `scratch` is an F32 tensor of at least
/// [`attention_scratch_elements`] elements, or `None` when that is zero.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    ctx: &HipContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    geometry: &AttentionGeometry,
    kernel: AttentionKernel,
    scratch: Option<&Tensor>,
    out: &Tensor,
) -> Result<()> {
    let bound = validate_and_bind(ctx, q, k, v, geometry, out)?;
    if !ck_enabled() || kernel == AttentionKernel::Reference {
        return launch_native(ctx, geometry, &bound, geometry.q_heads * geometry.head_dim);
    }
    match kernel {
        AttentionKernel::Reference => unreachable!(),
        AttentionKernel::Ck => {
            // With one KV head and no mask, the query heads are independent rows
            // over the same keys: [tokens, heads, d] is [tokens * heads, d] in
            // memory. One head of more rows tiles better (pi05's prefix: 106 us
            // against 124 us on gfx1151). A causal mask gives rows of the same
            // token different key windows, so folding is valid only unmasked.
            let (rows, heads, stride) =
                if geometry.kv_heads == 1 && geometry.causal_query_start.is_none() {
                    (geometry.q_tokens * geometry.q_heads, 1, geometry.head_dim)
                } else {
                    (geometry.q_tokens, geometry.q_heads, geometry.q_heads * geometry.head_dim)
                };
            if launch_ck_fused(ctx, geometry, &bound, rows, heads, stride)? {
                return Ok(());
            }
            launch_native(ctx, geometry, &bound, geometry.q_heads * geometry.head_dim)
        }
        AttentionKernel::CkSplitKv { splits } => {
            if launch_ck_splitkv(ctx, geometry, &bound, splits, scratch)? {
                return Ok(());
            }
            launch_native(ctx, geometry, &bound, geometry.q_heads * geometry.head_dim)
        }
    }
}

/// `out = softmax(q kᵀ / sqrt(head_dim)) v` where q, k and v are the column
/// sections of one packed projection, `packed = [batch * q_tokens,
/// q_width + 2 * kv_width]`: q at column 0, k at `q_width`, v at
/// `q_width + kv_width`, as the fused q/k/v GEMM writes them.
///
/// This is how the fused projection pays off: attention reads the sections in
/// place instead of a split kernel copying them out, so the block spends one
/// GEMM and one attention kernel instead of three, one and a split. The same
/// values are read in the same order as [`attention`] on the split tensors,
/// so the two agree bit for bit -- which is the acceptance contract between
/// them.
///
/// `q_width` must be `q_heads * head_dim` and `kv_width` `kv_heads * head_dim`;
/// every batch's rows are contiguous in `packed`, so the strides are derived.
/// `kv_tokens` may not exceed `q_tokens`: the packed buffer holds exactly one
/// row per query token, so unlike [`attention`] it cannot read a longer cache.
///
/// The kernel is chosen the same way [`attention_kernel`] would: CK where a
/// vendored instance covers the geometry, apxinf's kernel otherwise, with the
/// dispatch counters on both paths.
pub fn attention_qkv_packed(
    ctx: &HipContext,
    packed: &Tensor,
    q_width: usize,
    kv_width: usize,
    geometry: &AttentionGeometry,
    out: &Tensor,
) -> Result<()> {
    let AttentionGeometry {
        batch,
        q_tokens,
        q_heads,
        kv_heads,
        head_dim,
        kv_tokens,
        ..
    } = *geometry;
    if batch == 0 || q_heads == 0 || head_dim == 0 {
        return Err(Error::Contract("apxinf-hip attention: invalid geometry"));
    }
    if q_heads * head_dim != q_width || kv_heads * head_dim != kv_width {
        return Err(Error::ShapeMismatch {
            expected: format!(
                "sections of q width {q_heads} x {head_dim} and kv width {kv_heads} x {head_dim}"
            ),
            got: format!("q width {q_width}, kv width {kv_width}"),
        });
    }
    if kv_tokens > q_tokens {
        return Err(Error::Other(format!(
            "apxinf-hip attention: {kv_tokens} keys requested, the packed projection \
             holds {q_tokens}"
        )));
    }
    let row = q_width + 2 * kv_width;
    let dims = packed.shape().dims();
    if dims.len() != 2 || dims[0] != batch * q_tokens || dims[1] != row {
        return Err(Error::ShapeMismatch {
            expected: format!("[{}, {row}]", batch * q_tokens),
            got: packed.shape().to_string(),
        });
    }
    require_numel(out, batch * q_tokens * q_heads * head_dim, || {
        format!("[{batch}, {q_tokens}, {q_heads}, {head_dim}]")
    })?;
    let base = bf16_ptr(ctx, packed)?;
    let element = DType::BF16.size_in_bytes();
    // SAFETY: both offsets stay inside `packed`, whose shape was just checked
    // against `q_width + 2 * kv_width` columns.
    let k = unsafe { (base as *mut u8).add(q_width * element) as *const c_void };
    let v = unsafe { (base as *mut u8).add((q_width + kv_width) * element) as *const c_void };
    let bound = Bound {
        q: base,
        k,
        v,
        out: bf16_ptr(ctx, out)?,
        kv_batch_stride: q_tokens * row,
        kv_token_stride: row,
    };
    dispatch_attention(ctx, geometry, &bound, row)
}

/// The pointers and KV layout strides of one validated attention call, shared
/// by every entry point.
struct Bound {
    q: *mut c_void,
    k: *const c_void,
    v: *const c_void,
    out: *mut c_void,
    /// Elements between k/v batch entries: `k.numel() / batch`.
    kv_batch_stride: usize,
    /// Elements between k/v rows: `kv_heads * head_dim` for split tensors,
    /// the packed row width for projection sections.
    kv_token_stride: usize,
}

/// The shape and layout checks every attention entry point owes its caller
/// before any launch, culminating in the device pointers.
fn validate_and_bind(
    ctx: &HipContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    geometry: &AttentionGeometry,
    out: &Tensor,
) -> Result<Bound> {
    let AttentionGeometry {
        batch,
        q_tokens,
        q_heads,
        kv_heads,
        head_dim,
        kv_tokens,
        ..
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
    // A causal start at or past the last key is never "mask the rows before
    // it": the apxinf kernel would clamp it and return full attention, and
    // the CK path reads it as an alignment it has no instance for. Both are
    // silent nonsense, so refuse it here.
    if let Some(start) = geometry.causal_query_start {
        if start >= kv_tokens {
            return Err(Error::Other(format!(
                "apxinf-hip attention: causal_query_start {start} is not inside the \
                 {kv_tokens}-key window"
            )));
        }
    }
    Ok(Bound {
        q: bf16_ptr(ctx, q)?,
        k: bf16_ptr(ctx, k)?,
        v: bf16_ptr(ctx, v)?,
        out: bf16_ptr(ctx, out)?,
        kv_batch_stride: per_batch,
        kv_token_stride: kv_row,
    })
}

/// apxinf's own kernel: F32 scores and probabilities, every geometry.
///
/// `q_token_stride` is the address stride between q rows in elements —
/// `q_heads * head_dim` for a contiguous q, wider for a packed section.
fn launch_native(
    ctx: &HipContext,
    geometry: &AttentionGeometry,
    bound: &Bound,
    q_token_stride: usize,
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
    let query_start = match causal_query_start {
        None => -1,
        Some(start) => int(start)?,
    };
    let scale = 1.0 / (head_dim as f32).sqrt();
    DISPATCHES[0].fetch_add(1, Ordering::Relaxed);
    check("attention", unsafe {
        ffi::apxinf_hip_native_attention(
            bound.q,
            bound.k,
            bound.v,
            bound.out,
            int(batch)?,
            int(q_tokens)?,
            int(q_heads)?,
            int(kv_heads)?,
            int(head_dim)?,
            int(kv_tokens)?,
            long(bound.kv_batch_stride)?,
            long(bound.kv_token_stride)?,
            long(q_token_stride)?,
            scale,
            query_start,
            ctx.stream(),
        )
    })
}

/// CK's fused flash-attention forward, `rows` query rows of `heads` heads at
/// `q_row_stride` elements apart. Returns `Ok(false)` when no vendored
/// instance covers the call, so the caller falls back to apxinf's kernel.
fn launch_ck_fused(
    ctx: &HipContext,
    geometry: &AttentionGeometry,
    bound: &Bound,
    rows: usize,
    heads: usize,
    q_row_stride: usize,
) -> Result<bool> {
    let query_start = match geometry.causal_query_start {
        None => -1,
        Some(start) => int(start)?,
    };
    let scale = 1.0 / (geometry.head_dim as f32).sqrt();
    let status = unsafe {
        ffi::apxinf_hip_ck_attention(
            DTYPE_BF16,
            bound.q,
            bound.k,
            bound.v,
            bound.out,
            int(geometry.batch)?,
            int(rows)?,
            int(heads)?,
            int(geometry.kv_heads)?,
            int(geometry.head_dim)?,
            int(geometry.kv_tokens)?,
            long(bound.kv_batch_stride)?,
            long(bound.kv_token_stride)?,
            long(q_row_stride)?,
            scale,
            query_start,
            ctx.stream(),
        )
    };
    if status == ffi::NO_CK_INSTANCE {
        return Ok(false);
    }
    check("attention (CK)", status)?;
    DISPATCHES[1].fetch_add(1, Ordering::Relaxed);
    Ok(true)
}

/// CK's split-KV forward: keys split across `splits` workgroups per query
/// tile, combined through F32 partials in `scratch`. Returns `Ok(false)`
/// when the vendored instances cannot serve the call — they are unmasked —
/// so the caller falls back to apxinf's kernel.
fn launch_ck_splitkv(
    ctx: &HipContext,
    geometry: &AttentionGeometry,
    bound: &Bound,
    splits: usize,
    scratch: Option<&Tensor>,
) -> Result<bool> {
    if geometry.causal_query_start.is_some() {
        return Ok(false);
    }
    let AttentionGeometry {
        batch,
        q_tokens,
        q_heads,
        head_dim,
        ..
    } = *geometry;
    let needed = attention_scratch_elements(
        geometry,
        AttentionKernel::CkSplitKv { splits },
    );
    let scratch = scratch
        .ok_or(Error::Contract("apxinf-hip attention: split-KV needs scratch"))?;
    if scratch.dtype() != DType::F32 || scratch.numel() < needed {
        return Err(Error::ShapeMismatch {
            expected: format!("at least {needed} f32 scratch elements"),
            got: format!("{} {}", scratch.shape(), scratch.dtype()),
        });
    }
    let base = ctx.ptr(scratch)?;
    // The log-sum-exp block is padded to 64 elements so the output block
    // starts 256-byte aligned.
    let lse_elements = (batch * q_heads * splits * q_tokens).next_multiple_of(64);
    // SAFETY: the scratch holds `needed` floats, which is the padded
    // log-sum-exp block plus the output block.
    let o_acc = unsafe { base.cast::<f32>().add(lse_elements) };
    let status = unsafe {
        ffi::apxinf_hip_ck_attention_splitkv(
            DTYPE_BF16,
            bound.q,
            bound.k,
            bound.v,
            bound.out,
            int(batch)?,
            int(q_tokens)?,
            int(q_heads)?,
            int(geometry.kv_heads)?,
            int(head_dim)?,
            int(geometry.kv_tokens)?,
            long(bound.kv_batch_stride)?,
            long(bound.kv_token_stride)?,
            int(splits)?,
            base,
            o_acc.cast(),
            1.0 / (head_dim as f32).sqrt(),
            ctx.stream(),
        )
    };
    if status == ffi::NO_CK_INSTANCE {
        return Ok(false);
    }
    check("ck split-KV attention", status)?;
    DISPATCHES[1].fetch_add(1, Ordering::Relaxed);
    Ok(true)
}

/// The provider routing for the packed entry point: CK where a vendored
/// instance covers the geometry, apxinf's kernel otherwise, with the
/// dispatch counters on both paths.
///
/// `q_token_stride` is the address stride between q rows in elements —
/// `q_heads * head_dim` for a contiguous q, wider for a packed section.
fn dispatch_attention(
    ctx: &HipContext,
    geometry: &AttentionGeometry,
    bound: &Bound,
    q_token_stride: usize,
) -> Result<()> {
    if ck_enabled() {
        let query_start = match geometry.causal_query_start {
            None => -1,
            Some(start) => int(start)?,
        };
        if unsafe {
            ffi::apxinf_hip_ck_attention_supported(
                DTYPE_BF16,
                int(geometry.head_dim)?,
                query_start,
                int(geometry.q_tokens)?,
                int(geometry.kv_tokens)?,
            )
        } == 1
            && launch_ck_fused(
                ctx,
                geometry,
                bound,
                geometry.q_tokens,
                geometry.q_heads,
                q_token_stride,
            )?
        {
            return Ok(());
        }
    }
    launch_native(ctx, geometry, bound, q_token_stride)
}

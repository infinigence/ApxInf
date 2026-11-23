//! Attention operator contracts.

#[cfg(apxinf_musa_available)]
use std::ffi::c_void;

use apxinf_core::{DType, Error, Result, Tensor};

use super::contracts::{
    bf16_tensor, f32_tensor, require_dtype_on, require_optional_vector_bf16,
    require_output_tensor,
};
#[cfg(apxinf_musa_available)]
use super::contracts::{checked_i32, optional_tensor_ptr, tensor_ptr};
use crate::context::MusaContext;
#[cfg(apxinf_musa_available)]
use crate::ffi;
#[cfg(apxinf_musa_available)]
use crate::ffi::mubin::{check_fa_d256, AsmFaVariant};
#[cfg(all(apxinf_musa_available, apxinf_musa_mutlass_fmha))]
use crate::ffi::mutlass_fa::MutlassFaVariant;
#[cfg(apxinf_musa_available)]
use crate::ffi::mublas::check_mublas;
#[cfg(apxinf_musa_available)]
use crate::kernels::check_native;
#[cfg(not(apxinf_musa_available))]
use crate::kernels::native_unavailable;
use crate::kernels::{not_ported, TacticDispatch};

/// The ABI both strided-batch GEMM adapters share: f32 operands, and BF16
/// operands with an f32 result. The argument lists are identical, so which one
/// runs is the caller's choice, made per contraction:
/// `(a, b, c, m, n, k, trans_a, trans_b, lda, ldb, ldc, stride_a, stride_b,
/// stride_c, batch_count, alpha, stream)`.
#[cfg(apxinf_musa_available)]
type BatchedGemm = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i64,
    i64,
    i64,
    i32,
    f32,
    *mut c_void,
) -> ffi::mublas::mublasStatus_t;

/// The three projections a QKV split produces.
///
/// Each is an independent workspace slice, holding `[tokens, heads, head_dim]`.
#[derive(Debug)]
pub struct QkvTensors {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

/// Require a 3-D BF16 tensor and return `(tokens, heads, head_dim)`.
fn require_3d_bf16(
    ctx: &MusaContext,
    tensor: &Tensor,
    operation: &str,
    name: &str,
) -> Result<(usize, usize, usize)> {
    let dims = tensor.shape().dims();
    let [tokens, heads, head_dim] = dims else {
        return Err(Error::Other(format!(
            "MUSA {operation} {name} expects a 3D tensor, got {dims:?}"
        )));
    };
    if tensor.dtype() != DType::BF16
        || tensor.device() != apxinf_core::Device::Musa(ctx.device_id())
    {
        return Err(Error::Other(format!(
            "MUSA {operation} {name} expects BF16 on MUSA{}, got {} on {}",
            ctx.device_id(),
            tensor.dtype(),
            tensor.device()
        )));
    }
    Ok((*tokens, *heads, *head_dim))
}

/// The head geometry the MP22 D256 assembly is built for.
///
/// Fixed by its fragment layouts, not a tuning parameter: the kernel keeps Q in
/// 32 registers per thread and its MMA operands follow a layout written for
/// exactly these dimensions. A different geometry is a different kernel, which
/// this backend does not have.
const ASSEMBLY_HEADS: usize = 8;
const ASSEMBLY_HEAD_DIM: usize = 256;

/// The head dimension the vision attention runs at.
///
/// The vision tower's own parameter -- 16 heads of 72 columns. Nothing under it
/// is built for this width: the permutation is an index map and the GEMMs take
/// the dimension as an argument, so this is the shape the port has evidence for
/// rather than a layout the code assumes.
const VISION_HEAD_DIM: usize = 72;

/// The query length the two packed decode kernels are built for.
///
/// Both flatten the contiguous token and head dimensions into `tokens * 8` MMA
/// rows, and the split-KV merge kernel's row count is compiled in at the same
/// size. The bound comes from the model's action horizon -- ten flow steps --
/// and not from anything in this crate.
const PACKED_DECODE_TOKENS: usize = 10;

/// The KV capacity range the split-KV variant is **measured** to win in.
///
/// Not a bound the kernel needs, and nothing refuses a capacity outside it: the
/// split is over the KV axis at run time, so an out-of-band capacity computes the
/// same answer — it simply has no measurement behind the partitioning, and falls
/// back to the packed-decode variant, which is correct and slower. The band comes
/// from the shapes this port runs (the measured decode shape is `10/532/532`).
///
/// Stated here because it is the one place it is written, and because the
/// adapter's C ABI deliberately has no matching check — see the comment there.
const SPLIT_KV_CAPACITY: std::ops::RangeInclusive<usize> = 512..=640;

/// Pick the assembly variant for a shape.
///
/// A pure function of the shape, mirroring the rule `FA_mate` applies in
/// `mate/jit/mubin/flash_attention/mp22.py`. Anything outside the packed decode
/// kernels' fixed query length still computes correctly on the general kernel;
/// it is only slower, which [`note_general_fallback`] says out loud because a
/// profile cannot attribute a kernel it cannot name.
#[cfg(apxinf_musa_available)]
fn fa_d256_variant(tokens: usize, capacity: usize) -> AsmFaVariant {
    if tokens == PACKED_DECODE_TOKENS {
        if SPLIT_KV_CAPACITY.contains(&capacity) {
            return AsmFaVariant::SplitKv;
        }
        return AsmFaVariant::PackedDecode;
    }
    AsmFaVariant::General
}

/// Note, once, that a decode-shaped call is running the general kernel.
///
/// A query shorter than the KV it reads is the decode shape, and the packed
/// kernels exist for it. The general variant is 2.3x slower there and equally
/// correct, so this is a note rather than an error -- but it is the one thing
/// that distinguishes "the model changed its action horizon" from "the kernel
/// got slower", and no profile can tell them apart afterwards.
#[cfg(apxinf_musa_available)]
fn note_general_fallback(tokens: usize, capacity: usize) {
    if tokens >= capacity {
        // A prefix pass: the query covers the whole KV, which is the shape the
        // general kernel is for. Nothing to report.
        return;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "[apxinf] MUSA attention: {tokens} query rows over {capacity} KV rows ran the \
             general D256 assembly. The packed decode kernels are built for \
             {PACKED_DECODE_TOKENS} rows and are 2.3x faster on this shape; see \
             crates/apxinf-musa/kernels/asm/README.md."
        );
    });
}

/// The rule [`note_general_fallback`] states, made refusable.
///
/// A decode-shaped call (fewer query rows than KV rows) that the packed decode
/// kernels do not serve runs the general assembly: correct, 2.3x slower, and a
/// substitute for the measured route. `RequireExact` turns that substitution
/// into a refusal. A prefix pass (`tokens >= capacity`) uses the general kernel
/// by construction; there is nothing to substitute.
///
/// `tokens != PACKED_DECODE_TOKENS` is `fa_d256_variant`'s `General` rule,
/// restated here so this check has no dependency on the build's feature gates.
#[cfg_attr(not(any(apxinf_musa_available, test)), allow(dead_code))]
fn require_packed_decode(tokens: usize, capacity: usize, tactic: TacticDispatch) -> Result<()> {
    if tactic.requires_exact() && tokens != PACKED_DECODE_TOKENS && tokens < capacity {
        return Err(Error::Other(format!(
            "MUSA mqa {tokens} query rows over {capacity} KV rows: allow_fallback=false \
             requires the packed decode kernel, which is built for {PACKED_DECODE_TOKENS} rows; \
             the general D256 assembly is a substitute"
        )));
    }
    Ok(())
}

/// The MUTLASS build's version of the same rule.
///
/// `mp31_fa_variant` serves 9..=16 query rows from the packed decode tile 128.
/// From 17 to 256 rows the packed count lands in MATE's 192 bucket, but this
/// build freezes only the unpacked tile 192, so those decode-shaped calls run a
/// substitute and `RequireExact` refuses them. From 257 rows the unpacked tile
/// is MATE's own choice, and a prefix pass (`tokens >= capacity`) uses the
/// general kernel by construction.
///
/// `mp31_fa_variant` is the authority; this restates its boundaries so the
/// check compiles without the two feature gates that hide it.
#[cfg_attr(not(any(apxinf_musa_available, test)), allow(dead_code))]
fn require_mutlass_decode(tokens: usize, capacity: usize, tactic: TacticDispatch) -> Result<()> {
    if tactic.requires_exact() && tokens < capacity && (17..=256).contains(&tokens) {
        return Err(Error::Other(format!(
            "MUSA mqa {tokens} query rows over {capacity} KV rows: allow_fallback=false \
             requires the packed decode tile, and this MUTLASS build serves 17..=256 rows \
             only from the unpacked tile 192 substitute"
        )));
    }
    Ok(())
}

/// Multi-query attention over a prefix.
///
/// `key_tokens` bounds how many rows of `k`/`v` participate, which is what
/// lets the caller pass the whole reserved prefix while only the first
/// `key_tokens` rows hold meaningful data — the tail of a reserved prefix is
/// deliberately left as the arena found it.
///
/// `k` and `v` are 2-D `[rows, head_dim]` views of that prefix. The MP22 kernel
/// is built for `kv_heads = 1`, so a per-head 3-D cache is not an input it can
/// read.
///
/// # Numerics
/// The MP22 assembly keeps the online softmax in f32 but rounds the
/// probabilities to bf16 before PV, a rounding the eager bridge this replaced
/// did not have. Measured against the recorded bridge tensors that is 1.000
/// bf16 ulp at full scale, cosine 0.9999978.
pub fn mqa_bf16(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let (tokens, heads, head_dim) = require_3d_bf16(ctx, q, "mqa", "q")?;
    let mut output = bf16_tensor(ctx, &[tokens, heads, head_dim])?;
    mqa_bf16_into(ctx, q, k, v, &mut output, key_tokens)?;
    Ok(output)
}

/// Write single-query attention into a caller-provided output.
pub fn mqa_bf16_into(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output: &mut Tensor,
    key_tokens: usize,
) -> Result<()> {
    mqa_bf16_into_with(
        ctx,
        q,
        k,
        v,
        output,
        key_tokens,
        TacticDispatch::AllowFallback,
    )
}

/// [`mqa_bf16_into`] with the caller's fallback constraint.
///
/// `TacticDispatch::RequireExact` refuses the general D256 assembly for a
/// decode-shaped call (fewer query rows than KV rows), where the packed decode
/// kernels are the measured route. A prefix pass uses the general kernel by
/// construction and is not a fallback.
pub fn mqa_bf16_into_with(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output: &mut Tensor,
    key_tokens: usize,
    tactic: TacticDispatch,
) -> Result<()> {
    let (tokens, heads, head_dim) = require_3d_bf16(ctx, q, "mqa", "q")?;
    if (heads, head_dim) != (ASSEMBLY_HEADS, ASSEMBLY_HEAD_DIM) {
        return Err(Error::Other(format!(
            "MUSA mqa runs the MP22 assembly, which is built for {ASSEMBLY_HEADS} heads of \
             dimension {ASSEMBLY_HEAD_DIM}; got {heads} heads of dimension {head_dim}"
        )));
    }
    // `require_3d_bf16` cannot serve k or v: both may be 2-D `[rows, head_dim]`
    // views of the KV prefix, so the rank it demands is not a contract here.
    // Their dtype and device are — `tensor_ptr` hands the raw pointer to an
    // assembly that reads it as BF16, so an F16 cache, or one on another card,
    // would reach the launch as a wrong answer or a fault rather than a
    // rejected call. `mha_bf16` below checks the same two things on its k/v.
    for (name, tensor) in [("k", k), ("v", v)] {
        require_dtype_on(ctx, tensor, DType::BF16, "mqa", name)?;
    }
    if v.shape().dims() != k.shape().dims() {
        return Err(Error::Other(format!(
            "MUSA mqa needs matching k/v shapes, got {:?} and {:?}",
            k.shape().dims(),
            v.shape().dims()
        )));
    }
    let k_dims = k.shape().dims();
    // Exactly 2-D, not "at least". `available` is the row count of one flat
    // `[rows, head_dim]` cache; a 3-D `[rows, heads, head_dim]` operand would
    // both inflate that bound by `heads` and hand the kernel a per-head layout
    // it does not read, since the MP22 kernel is built for `kv_heads = 1`.
    if k_dims.len() != 2 || k_dims[1] != head_dim {
        return Err(Error::Other(format!(
            "MUSA mqa takes k and v as 2-D `[rows, head_dim]` views of one shared KV cache, \
             got {k_dims:?}"
        )));
    }
    let available = k_dims[0];
    if key_tokens == 0 || key_tokens > available {
        return Err(Error::Other(format!(
            "MUSA mqa needs 0 < key_tokens <= {available}, got {key_tokens}"
        )));
    }
    require_output_tensor(ctx, output, &[tokens, heads, head_dim], "mqa", "out")?;
    crate::workspace::note_shape(crate::workspace::OperationShape::Attention {
        queries: tokens,
        keys: key_tokens,
        heads,
        head_dim,
    })?;
    launch_mqa(ctx, q, k, v, output, tokens, available, key_tokens, tactic)
}

/// The launch, split by whether this build has a MUSA toolkit.
#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_mqa(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output: &Tensor,
    tokens: usize,
    capacity: usize,
    key_tokens: usize,
    tactic: TacticDispatch,
) -> Result<()> {
    // Two kernels, two targets, and the choice is a property of the build
    // rather than of the call: `kernels/asm/` is the assembly generation and
    // `kernels/mutlass_fmha/` is the vendored one, so a build has at most one.
    match d256_route() {
        D256Route::Assembly => {}
        D256Route::Mutlass => {
            require_mutlass_decode(tokens, capacity, tactic)?;
            return launch_mqa_mutlass(ctx, q, k, v, output, tokens, key_tokens);
        }
        D256Route::Unsupported => {
            return Err(Error::Other(format!(
                "MUSA attention::mqa_bf16 has no kernel for {}: the vendored D256 assembly is \
                 {} ISA and the vendored MUTLASS FMHA is {} device code built \
                 for a MUSA toolkit this build did not find. See \
                 crates/apxinf-musa/kernels/mutlass_fmha/README.md",
                crate::MUSA_ARCH,
                crate::ASSEMBLY_ARCH,
                crate::VENDOR_FMHA_ARCH
            )));
        }
    }
    require_packed_decode(tokens, capacity, tactic)?;
    let variant = fa_d256_variant(tokens, capacity);
    if variant == AsmFaVariant::General {
        note_general_fallback(tokens, capacity);
    }
    let tokens = checked_i32("mqa", "tokens", tokens)?;
    let capacity = checked_i32("mqa", "capacity", capacity)?;
    let key_tokens = checked_i32("mqa", "key_tokens", key_tokens)?;
    // SAFETY: the shapes, dtypes and single device are established by the
    // caller; `q`, `k`, `v` and `output` are device allocations that outlive
    // the asynchronous launch on `ctx.stream()`, and the adapter reads no
    // tensor past the row counts it is given.
    unsafe {
        check_fa_d256(
            ffi::mubin::apxinf_musa_fa_d256(
                tensor_ptr(q)?,
                tensor_ptr(k)?,
                tensor_ptr(v)?,
                tensor_ptr(output)?,
                tokens,
                capacity,
                key_tokens,
                variant as i32,
                ctx.stream().handle(),
            ),
            "attention::mqa_bf16",
        )
        .map_err(crate::error::musa_error)
    }
}

/// Which D256 attention kernel this build runs.
///
/// A property of the build, not of the call: the three assembly images are
/// [`crate::ASSEMBLY_ARCH`] ISA and `kernels/mutlass_fmha/` is
/// [`crate::VENDOR_FMHA_ARCH`] device code, so a target that is neither has
/// neither. `Mutlass` also requires the vendored tree to have been compiled at
/// all, which `build.rs` decides from the target *and* the toolkit version the
/// frozen render was measured against.
#[cfg(apxinf_musa_available)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum D256Route {
    Assembly,
    Mutlass,
    Unsupported,
}

/// The route, as `build.rs` decided it.
///
/// Read, not derived: `build_support::musa_arch::d256_route` is the one place
/// the rule is written, and this constant publishes its answer.
/// `tests/arch_gate.rs` recomputes the route from [`crate::ASSEMBLY_ARCH`],
/// [`crate::VENDOR_FMHA_ARCH`] and the vendor cfg and compares, so the rule and
/// the published value cannot drift apart.
///
/// The catch-all is not the silent-fallback hazard it looks like:
/// [`crate::D256_ROUTE`]'s compile-time assertion admits exactly the three names
/// below, so "something else" cannot reach here.
#[cfg(apxinf_musa_available)]
fn d256_route() -> D256Route {
    match crate::D256_ROUTE {
        "assembly" => D256Route::Assembly,
        "mutlass" => D256Route::Mutlass,
        _ => D256Route::Unsupported,
    }
}

/// The mp_31 instantiation to run for `tokens` query rows, or `None` when this
/// build has no tile for that length.
///
/// Mirrors `mate/jit/attention/fmha/fmha_utils.py::_get_fwd_kernel_config` for
/// bf16 at head dimension 256 with `head_ratio` 8, which packs GQA up to 256
/// query rows, buckets the packed row count onto a tile of
/// 16/32/64/128/192/256, and caps the tile at 192. Three query lengths come out
/// of that rule, not two:
///
/// * 9..=16 rows -- `rows * 8` lands in the 128 bucket, so MATE chooses the
///   packed tile 128. Frozen, and returned here.
/// * 17 rows and up -- the packed count lands in the 192 bucket. Frozen, but
///   this build freezes only the *unpacked* tile 192, so these lengths run that
///   instantiation in place of the one MATE would have chosen. The substitution
///   is licensed by `mutlass_tile_choice_does_not_move_the_bits`, which runs
///   both on the same input and requires bit-identical output; 257 and up is
///   MATE's own choice of unpacked 192.
/// * 0..=8 rows -- `rows * 8` buckets below 128, where MATE's rule has no
///   answer this build has frozen. Those lengths get the named error rather
///   than a neighbouring tile.
///
/// PI0.5 reaches only the first two: its query lengths are 10 and 522.
#[cfg(all(apxinf_musa_available, apxinf_musa_mutlass_fmha))]
fn mp31_fa_variant(tokens: usize) -> Option<MutlassFaVariant> {
    match tokens {
        0..=8 => None,
        9..=16 => Some(MutlassFaVariant::PackedDecodeTile128),
        _ => Some(MutlassFaVariant::GeneralTile192),
    }
}

/// The D256 attention on the vendored mp_31 FMHA.
///
/// The contract is the assembly's; the kernel behind it is MATE's render, which
/// MATE produces from a MUTLASS template for every head dimension its assembly
/// path rejects — 256 is one of them. See
/// `crates/apxinf-musa/kernels/mutlass_fmha/README.md`.
#[cfg(all(apxinf_musa_available, apxinf_musa_mutlass_fmha))]
fn launch_mqa_mutlass(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output: &Tensor,
    tokens: usize,
    key_tokens: usize,
) -> Result<()> {
    let Some(variant) = mp31_fa_variant(tokens) else {
        return Err(Error::Other(format!(
            "MUSA attention::mqa_bf16 has no {} kernel for {tokens} query rows: \
             the vendored FMHA is frozen at MATE's tile 128 (9..=16 rows) and tile 192 (17 rows \
             and up). A new tile is a new render and a new replay -- see \
             crates/apxinf-musa/kernels/mutlass_fmha/README.md",
            crate::VENDOR_FMHA_ARCH
        )));
    };
    let tokens = checked_i32("mqa", "tokens", tokens)?;
    let key_tokens = checked_i32("mqa", "key_tokens", key_tokens)?;
    // SAFETY: shapes, dtypes and the single device are established by the
    // caller; `q`, `k`, `v` and `output` are device allocations that outlive the
    // asynchronous launch on `ctx.stream()`, and the kernel reads no KV row past
    // `key_tokens`.
    unsafe {
        ffi::mutlass_fa::check_mutlass_fa(
            ffi::mutlass_fa::apxinf_musa_mutlass_fa_d256(
                tensor_ptr(q)?,
                tensor_ptr(k)?,
                tensor_ptr(v)?,
                tensor_ptr(output)?,
                tokens,
                key_tokens,
                variant as i32,
                ctx.stream().handle(),
            ),
            "attention::mqa_bf16",
        )
        .map_err(Error::Other)
    }
}

/// The mp_31 launcher on a build that has no mp_31 kernel.
///
/// Unreachable: `d256_route` returns `Mutlass` only when
/// `apxinf_musa_mutlass_fmha` is set, which is exactly when the kernel exists.
/// It is here so the route match in `launch_mqa` stays total without the cfg
/// leaking into it, and so a build that somehow reached it would say which
/// kernel is missing rather than fail at link time.
#[cfg(all(apxinf_musa_available, not(apxinf_musa_mutlass_fmha)))]
fn launch_mqa_mutlass(
    _ctx: &MusaContext,
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _output: &Tensor,
    _tokens: usize,
    _key_tokens: usize,
) -> Result<()> {
    Err(Error::Other(format!(
        "MUSA attention::mqa_bf16 has no {} kernel in this build: the vendored \
         FMHA was not compiled. See crates/apxinf-musa/kernels/mutlass_fmha/README.md",
        crate::VENDOR_FMHA_ARCH
    )))
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch_mqa(
    _ctx: &MusaContext,
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _output: &Tensor,
    _tokens: usize,
    _capacity: usize,
    _key_tokens: usize,
    _tactic: TacticDispatch,
) -> Result<()> {
    Err(native_unavailable("attention::mqa_bf16"))
}

/// Multi-head attention over self-attention windows of `tokens_per_batch`.
///
/// Each window attends only within itself: the vision tower's patches are a
/// bidirectional block per view, not one long sequence.
///
/// # Numerics
/// f32 scores, f32 online softmax, f32 accumulation, rounded once on the way
/// out — the same shape as a torch eager bridge. The one place it parts from
/// that shape is Q and K: they stay BF16 into the scores product instead of
/// being widened, so the MMA's intermediate rounding joins the summation order
/// as a difference. Widened, the path measures 0.99991 end-to-end cosine
/// against the reference; narrow, 0.99987.
pub fn mha_bf16(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    tokens_per_batch: usize,
) -> Result<Tensor> {
    let (tokens, heads, head_dim) = require_3d_bf16(ctx, q, "mha", "q")?;
    let mut output = bf16_tensor(ctx, &[tokens, heads, head_dim])?;
    mha_bf16_into(ctx, q, k, v, &mut output, tokens_per_batch)?;
    Ok(output)
}

/// Write multi-head attention into a caller-provided output.
pub fn mha_bf16_into(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output: &mut Tensor,
    tokens_per_batch: usize,
) -> Result<()> {
    mha_bf16_into_with(
        ctx,
        q,
        k,
        v,
        output,
        tokens_per_batch,
        TacticDispatch::AllowFallback,
    )
}

/// [`mha_bf16_into`] with the caller's fallback constraint.
///
/// The vision path has one kernel route — a fixed 72-column-per-head mapping
/// built for the tower's windows — so there is no alternative tactic to fall
/// back to. Both dispatch values are honoured identically; the parameter is
/// carried so the op contract has one shape for both attention paths.
pub fn mha_bf16_into_with(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output: &mut Tensor,
    tokens_per_batch: usize,
    _tactic: TacticDispatch,
) -> Result<()> {
    let (tokens, heads, head_dim) = require_3d_bf16(ctx, q, "mha", "q")?;
    for (name, tensor) in [("k", k), ("v", v)] {
        let dims = tensor.shape().dims();
        if dims != q.shape().dims()
            || tensor.dtype() != DType::BF16
            || tensor.device() != apxinf_core::Device::Musa(ctx.device_id())
        {
            return Err(Error::Other(format!(
                "MUSA mha needs q/k/v to share a shape and dtype, {name} is {} {:?}",
                tensor.dtype(),
                dims
            )));
        }
    }
    if head_dim != VISION_HEAD_DIM {
        return Err(Error::Other(format!(
            "MUSA mha runs the vision kernel, whose column mapping is written for \
             {VISION_HEAD_DIM} columns per head; got {head_dim}"
        )));
    }
    if tokens_per_batch == 0 || tokens % tokens_per_batch != 0 {
        return Err(Error::Other(format!(
            "MUSA mha needs tokens_per_batch to divide {tokens}, got {tokens_per_batch}"
        )));
    }
    require_output_tensor(ctx, output, &[tokens, heads, head_dim], "mha", "out")?;
    crate::workspace::note_shape(crate::workspace::OperationShape::Attention {
        queries: tokens,
        keys: tokens,
        heads,
        head_dim,
    })?;
    launch_mha(
        ctx,
        q,
        k,
        v,
        output,
        tokens,
        heads,
        head_dim,
        tokens_per_batch,
    )
}

/// The launch, split by whether this build has a MUSA toolkit.
///
/// # Why this is three GEMM-shaped steps and not one kernel
/// A hand-written flash-decoding kernel for this shape was measured at
/// 8.4 ms/call against the eager bridge's 2.1, and an ablation put 84% of it in
/// memory traffic: one warp per `(query, head)` reads a whole 73.7 KB window of
/// K/V for its own row, so a pass drags 604 MB through the memory system to do
/// 302M MACs.
/// The two contractions here are GEMM shapes, and the vendor GEMM does not have
/// that problem.
///
/// # Layout
/// The model holds Q/K/V token-major as `[tokens, heads, 72]`, where one head
/// is 72 elements away and one token is 1152 — no single batch stride describes
/// both, so the operands are permuted to `[batch * heads, tokens, 72]` first.
/// `permute_heads_kernel` is that, converting on the way only where the two
/// sides disagree on a type — which is V alone.
///
/// In the permuted layout, with `T = tokens_per_batch` and `D = head_dim`:
///
/// * `S^T = K @ Q^T` — column-major `m=T n=T k=D`, `TRANSA=T` on K because its
///   bytes are `[T,D]` and the transposed (that is, `[D,T]`) operand is what the
///   product needs. The softmax then runs along the contiguous axis, which is
///   the key axis.
/// * `O^T = V^T @ P^T` — column-major `m=D n=T k=T`, no transposes: `V^T` and
///   `O^T` are exactly how `[T,D]` row-major bytes read column-major.
///
/// `alpha` carries the softmax scale into the first GEMM, so `S` is already
/// scaled when the softmax runs.
#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_mha(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output: &Tensor,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    tokens_per_batch: usize,
) -> Result<()> {
    let rows = tokens_per_batch;
    let batches = tokens / rows;
    let batch_heads = batches * heads;
    let stream = ctx.stream().handle();

    // MP22 D72 uses a strided B,H,S,D descriptor over native B,S,H,D
    // storage. No global score matrix or head permutation is materialized.
    static FLASH_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let flash = *FLASH_ENABLED.get_or_init(|| {
        std::env::var("APXINF_MUSA_BF16_MHA").map_or(true, |value| value != "legacy")
    });
    // Which generations have the muDNN flash kernel is the family's answer, not
    // a pair of numbers written here: a list at the dispatch site is a list a
    // new generation falls off silently, keeping the correct-but-slower path
    // with nothing to say so.
    let family = ctx.caps().arch_family;
    if flash && family.has_vision_flash_mha() && heads == 16 && head_dim == 72 && rows == 256 {
        let status = unsafe {
            ffi::custom::apxinf_musa_mudnn_flash_bf16(
                tensor_ptr(q)?,
                tensor_ptr(k)?,
                tensor_ptr(v)?,
                tensor_ptr(output)?,
                checked_i32("flash", "batch", batches)?,
                checked_i32("flash", "heads", heads)?,
                checked_i32("flash", "tokens", rows)?,
                checked_i32("flash", "dim", head_dim)?,
                stream,
            )
        };
        return crate::ffi::custom::check_mudnn(status, || {
            format!("muDNN flash attention {rows}x{head_dim} over {heads} heads")
        })
        .map_err(crate::error::musa_error);
    }

    let as_i32 = |name: &str, value: usize| checked_i32("mha", name, value);
    let tokens_i32 = as_i32("tokens", tokens)?;
    let rows_i32 = as_i32("tokens_per_batch", rows)?;
    let dim_i32 = as_i32("head_dim", head_dim)?;
    let batch_heads_i32 = as_i32("batch*heads", batch_heads)?;

    // Head-major operands, one contiguous `[rows, dim]` matrix per batch. The
    // entry point follows the operand types, which is what says whether the
    // permutation has to convert at all: BF16 on both sides for Q and K, which
    // arrive BF16 and are consumed BF16, and f32 on V's head-major side, which
    // meets the f32 score matrix.
    let permute = |to_head_major: bool, from: &Tensor, to: &Tensor| -> Result<()> {
        let permutation = match (to_head_major, from.dtype(), to.dtype()) {
            (true, DType::BF16, DType::BF16) => ffi::custom::apxinf_musa_permute_heads_bf16,
            (true, DType::BF16, DType::F32) => ffi::custom::apxinf_musa_permute_heads_to_f32,
            (false, DType::F32, DType::BF16) => ffi::custom::apxinf_musa_permute_heads_from_f32,
            (to_head_major, from, to) => {
                return Err(Error::Other(format!(
                    "MUSA mha permute has no entry for {from} -> {to} with \
                     to_head_major={to_head_major}"
                )))
            }
        };
        // SAFETY: both buffers hold `tokens * heads * dim` elements of the type
        // the entry point expects on its side of the permutation, established
        // by the caller and by `bf16_tensor`/`f32_tensor`; both outlive the
        // asynchronous launch on `ctx.stream()`.
        unsafe {
            check_native(
                permutation(
                    tensor_ptr(from)?,
                    tensor_ptr(to)?,
                    tokens_i32,
                    as_i32("heads", heads)?,
                    dim_i32,
                    rows_i32,
                    stream,
                ),
                "attention::mha_bf16 permute",
            )
        }
    };

    let q_heads = bf16_tensor(ctx, &[batch_heads, rows, head_dim])?;
    let k_heads = bf16_tensor(ctx, &[batch_heads, rows, head_dim])?;
    let v_heads = f32_tensor(ctx, &[batch_heads, rows, head_dim])?;
    permute(true, q, &q_heads)?;
    permute(true, k, &k_heads)?;
    permute(true, v, &v_heads)?;

    // `S^T = K @ Q^T`, scaled by 1/sqrt(dim) on the way in.
    let scores = f32_tensor(ctx, &[batch_heads, rows, rows])?;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let batched = |entry: BatchedGemm,
                   a: &Tensor,
                   b: &Tensor,
                   c: &Tensor,
                   m: i32,
                   n: i32,
                   k: i32,
                   trans_a: i32,
                   lda: i32,
                   ldb: i32,
                   ldc: i32,
                   stride_a: i64,
                   stride_b: i64,
                   stride_c: i64,
                   alpha: f32,
                   label: &str|
     -> Result<()> {
        // SAFETY: every operand is a contiguous device allocation whose element
        // type matches what `entry` declares, and whose batch count and leading
        // dimensions are passed explicitly, so mublas reads exactly the
        // elements it is told to. All of them outlive the asynchronous launch
        // on `ctx.stream()`.
        unsafe {
            check_mublas(entry(
                tensor_ptr(a)?,
                tensor_ptr(b)?,
                tensor_ptr(c)?,
                m,
                n,
                k,
                trans_a,
                0,
                lda,
                ldb,
                ldc,
                stride_a,
                stride_b,
                stride_c,
                batch_heads_i32,
                alpha,
                stream,
            ))
            .map_err(|message| {
                crate::error::musa_error(format!("attention::mha_bf16 {label}: {message}"))
            })?
        }
        Ok(())
    };

    batched(
        ffi::mublas::apxinf_musa_bf16in_f32out_gemm_batched_colmajor,
        &k_heads,
        &q_heads,
        &scores,
        rows_i32,
        rows_i32,
        dim_i32,
        1, // TRANSA: K's bytes are [rows, dim], used transposed
        dim_i32,
        dim_i32,
        rows_i32,
        (rows * head_dim) as i64,
        (rows * head_dim) as i64,
        (rows * rows) as i64,
        scale,
        "scores",
    )?;

    // Softmax over the key axis, which is the contiguous one in this layout.
    let score_rows = as_i32("score rows", batch_heads * rows)?;
    // SAFETY: `scores` is a contiguous f32 allocation of
    // `batch_heads * rows * rows` elements and is not aliased.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_row_softmax_f32(
                tensor_ptr(&scores)?,
                score_rows,
                rows_i32,
                stream,
            ),
            "attention::mha_bf16 softmax",
        )?
    }

    // `O^T = V^T @ P^T`, no transposes.
    let out_heads = f32_tensor(ctx, &[batch_heads, rows, head_dim])?;
    batched(
        ffi::mublas::apxinf_musa_f32_gemm_batched_colmajor,
        &v_heads,
        &scores,
        &out_heads,
        dim_i32,
        rows_i32,
        rows_i32,
        0,
        dim_i32,
        rows_i32,
        dim_i32,
        (rows * head_dim) as i64,
        (rows * rows) as i64,
        (rows * head_dim) as i64,
        1.0,
        "values",
    )?;

    // Back out of the head-major layout, and back to BF16: this is the one
    // `permute` call whose destination is the operator's output.
    permute(false, &out_heads, output)
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch_mha(
    _ctx: &MusaContext,
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _output: &Tensor,
    _tokens: usize,
    _heads: usize,
    _head_dim: usize,
    _tokens_per_batch: usize,
) -> Result<()> {
    Err(native_unavailable("attention::mha_bf16"))
}

/// Split a packed QKV projection into `[tokens, heads, head_dim]` parts,
/// adding the fused bias on the way.
pub fn split_qkv_bias_bf16(
    ctx: &MusaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    heads: usize,
    head_dim: usize,
) -> Result<QkvTensors> {
    let dims = qkv.shape().dims();
    let [tokens, _] = dims else {
        return Err(Error::Other(format!(
            "MUSA qkv split expects a 2D tensor, got {dims:?}"
        )));
    };
    let q = bf16_tensor(ctx, &[*tokens, heads, head_dim])?;
    let k = bf16_tensor(ctx, &[*tokens, heads, head_dim])?;
    let v = bf16_tensor(ctx, &[*tokens, heads, head_dim])?;
    let (mut q, mut k, mut v) = (q, k, v);
    split_qkv_bias_bf16_into(ctx, qkv, bias, &mut q, &mut k, &mut v, heads, head_dim)?;
    Ok(QkvTensors { q, k, v })
}

/// Write the QKV split into caller-provided outputs.
pub fn split_qkv_bias_bf16_into(
    ctx: &MusaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q: &mut Tensor,
    k: &mut Tensor,
    v: &mut Tensor,
    heads: usize,
    head_dim: usize,
) -> Result<()> {
    let dims = qkv.shape().dims();
    let [tokens, packed] = dims else {
        return Err(Error::Other(format!(
            "MUSA qkv split expects a 2D tensor, got {dims:?}"
        )));
    };
    let width = 3 * heads * head_dim;
    if qkv.dtype() != DType::BF16
        || qkv.device() != apxinf_core::Device::Musa(ctx.device_id())
        || *packed != width
    {
        return Err(Error::Other(format!(
            "MUSA qkv split expects BF16 [tokens, {width}], got {} {:?}",
            qkv.dtype(),
            dims
        )));
    }
    if *tokens == 0 {
        return Err(Error::Other(
            "MUSA qkv split needs a non-zero token count".into(),
        ));
    }
    require_optional_vector_bf16(ctx, bias, width, "qkv split", "bias")?;
    require_output_tensor(ctx, q, &[*tokens, heads, head_dim], "qkv split", "q")?;
    require_output_tensor(ctx, k, &[*tokens, heads, head_dim], "qkv split", "k")?;
    require_output_tensor(ctx, v, &[*tokens, heads, head_dim], "qkv split", "v")?;
    launch_split(ctx, qkv, bias, q, k, v, *tokens, heads * head_dim)
}

/// The launch, split by whether this build has a MUSA toolkit.
///
/// # Numerics
/// Elementwise: `qkv + bias` in f32, rounded once, and the three segments are
/// chosen by index rather than by any arithmetic. The bridge did the same
/// through a slice and a `_write`, so this is expected to reproduce the
/// recorded tensors **bit for bit** rather than merely within the gate.
#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_split(
    ctx: &MusaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    tokens: usize,
    projection_width: usize,
) -> Result<()> {
    let tokens = checked_i32("qkv split", "tokens", tokens)?;
    let projection_width = checked_i32("qkv split", "projection_width", projection_width)?;
    let bias_pointer = optional_tensor_ptr(bias)?;
    // SAFETY: shapes, dtypes and one device are established above, the three
    // outputs are separate `[tokens, heads, head_dim]` BF16 allocations of
    // exactly `projection_width` elements per token, and every buffer outlives
    // the asynchronous launch on `ctx.stream()`.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_qkv_split_bias_bf16(
                tensor_ptr(qkv)?,
                bias_pointer,
                tensor_ptr(q)?,
                tensor_ptr(k)?,
                tensor_ptr(v)?,
                tokens,
                projection_width,
                ctx.stream().handle(),
            ),
            "qkv split",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch_split(
    _ctx: &MusaContext,
    _qkv: &Tensor,
    _bias: Option<&Tensor>,
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _tokens: usize,
    _projection_width: usize,
) -> Result<()> {
    Err(native_unavailable("attention::split_qkv_bias_bf16"))
}

// ── FP16 attention (not ported) ──────────────────────────────────────────
//
// The FP16 paths serve the `int8_dynamic` and `fp8_static` variants, which are
// outside this port's scope. Signatures mirror
// `apxinf-cuda/src/kernels/attention.rs`; `mqa_cached_f16` in particular is a
// pure mublas path on the CUDA side (no kernel launch), so a port should not
// assume it needs a kernel.

/// Multi-query attention in FP16, key length taken from `k`.
pub fn mqa_f16(ctx: &MusaContext, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let _ = (ctx, q, k, v);
    Err(not_ported("attention::mqa_f16"))
}

/// Multi-head attention in FP16.
pub fn mha_f16(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    tokens_per_batch: usize,
) -> Result<Tensor> {
    let _ = (ctx, q, k, v, tokens_per_batch);
    Err(not_ported("attention::mha_f16"))
}

/// Split a packed QKV projection in FP16, without positional encoding.
pub fn split_qkv_bias_f16(
    ctx: &MusaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    heads: usize,
    head_dim: usize,
) -> Result<QkvTensors> {
    let _ = (ctx, qkv, bias, heads, head_dim);
    Err(not_ported("attention::split_qkv_bias_f16"))
}

/// Packed-QKV multi-head attention in FP16.
pub fn mha_packed_qkv_bias_f16(
    ctx: &MusaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    tokens_per_batch: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let _ = (ctx, qkv, bias, tokens_per_batch, heads, head_dim);
    Err(not_ported("attention::mha_packed_qkv_bias_f16"))
}

/// Multi-query attention over a two-dimensional cache, in FP16.
pub fn mqa_cached_f16(
    ctx: &MusaContext,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let _ = (ctx, q, k_cache, v_cache, key_tokens);
    Err(not_ported("attention::mqa_cached_f16"))
}

/// PI0.5's `[522, 8, 256]` MQA with a direct E4M3 output.
///
/// The CUDA build gates this on an SM100-family FA2 build; the shapes are
/// hard-coded there, which is worth keeping if it is ever ported.
pub fn mqa_f16_e4m3_522(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output_scale: f32,
) -> Result<Tensor> {
    let _ = (ctx, q, k, v, output_scale);
    Err(not_ported("attention::mqa_f16_e4m3_522"))
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    //! Device test for the QKV split.
    //!
    //! The replay harness covers `attention.split_qkv_bias_bf16` at its one
    //! recorded shape -- `[512, 3456]` with a bias -- and it comes out
    //! bit-identical. The **bias-less** branch is not in that recording, because
    //! the vision tower always passes one. This covers it, on a shape whose
    //! columns are not a multiple of the block size so the per-token loop takes
    //! more than one step.
    //!
    //! Needs a MUSA device, so it is `#[ignore]`d:
    //! `cargo test -p apxinf-musa --lib -- --ignored --nocapture`.

    use apxinf_core::{DType, Shape};

    use crate::context::MusaContext;
    use crate::kernels::attention::{mha_bf16, mqa_bf16, mqa_bf16_into_with, split_qkv_bias_bf16};
    use crate::kernels::attention::{ASSEMBLY_HEADS, ASSEMBLY_HEAD_DIM, VISION_HEAD_DIM};
    use crate::kernels::test_support::{
        from_bf16_bits, gate_for, read_back, to_bf16_bits, upload, values, zeros_as,
    };
    use crate::kernels::TacticDispatch;

    fn check(ctx: &MusaContext, tokens: usize, heads: usize, head_dim: usize, bias: bool) {
        let width = heads * head_dim;
        let qkv_values = values(tokens * 3 * width, 0x5eed_1234);
        let bias_values = values(3 * width, 0x0bad_c0de);
        let qkv = upload(ctx, &qkv_values, Shape::new(vec![tokens, 3 * width]));
        let bias_tensor = bias.then(|| upload(ctx, &bias_values, Shape::new(vec![3 * width])));

        let split = split_qkv_bias_bf16(ctx, &qkv, bias_tensor.as_ref(), heads, head_dim)
            .expect("native qkv split");
        ctx.stream().synchronize().expect("synchronise");
        let (q, k, v) = (
            read_back(&split.q),
            read_back(&split.k),
            read_back(&split.v),
        );

        let round = |value: f32| from_bf16_bits(to_bf16_bits(value));
        let tolerance = gate_for(&q);
        for (name, got, offset) in [("q", &q, 0usize), ("k", &k, width), ("v", &v, 2 * width)] {
            let mut worst = 0.0f32;
            for index in 0..tokens * width {
                let token = index / width;
                let col = index % width;
                let mut expected = round(qkv_values[token * 3 * width + offset + col]);
                if bias {
                    expected += round(bias_values[offset + col]);
                }
                let expected = round(expected);
                let error = (got[index] - expected).abs();
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "{name} [{tokens}x{width}] at [{token}][{col}]: {} vs {expected}, \
                     abs {error:.3e} > {tolerance:.3e}",
                    got[index]
                );
            }
            println!("  split_bias {name} bias {bias}  worst abs {worst:.3e}");
        }
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn split_with_and_without_a_bias_matches_a_reference() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        check(&ctx, 3, 2, 6, true);
        // 5 * 72 * 3 = 1080 columns, which is more than four blocks' worth, so
        // the per-token loop runs five times.
        check(&ctx, 2, 5, 72, false);
        // The vision tower's own head count and dimension: 16 * 72 = 1152 is a
        // multiple of eight, so this covers the vectorized branch the tower
        // actually takes, with the bias it always passes.
        check(&ctx, 8, 16, 72, true);
    }

    // ── The MP22 D256 assembly ───────────────────────────────────────────────

    /// A CPU reference for the same arithmetic the assembly computes: f32
    /// scores, f32 softmax, f32 PV.
    ///
    /// It is *not* the same as the assembly, which rounds the probabilities to
    /// bf16 before PV -- that difference is real and shows up as a fraction of
    /// the gate, not as zero. What this checks is that the kernel's shapes,
    /// strides and scaling are right; the size of the probability rounding is
    /// measured separately, against the recorded bridge tensors.
    fn reference(q: &[f32], k: &[f32], v: &[f32], tokens: usize, key_tokens: usize) -> Vec<f32> {
        let scale = 1.0 / (ASSEMBLY_HEAD_DIM as f32).sqrt();
        let mut out = vec![0.0f32; tokens * ASSEMBLY_HEADS * ASSEMBLY_HEAD_DIM];
        for token in 0..tokens {
            for head in 0..ASSEMBLY_HEADS {
                let row = (token * ASSEMBLY_HEADS + head) * ASSEMBLY_HEAD_DIM;
                let mut scores = vec![0.0f32; key_tokens];
                for (key, score) in scores.iter_mut().enumerate() {
                    let key_row = key * ASSEMBLY_HEAD_DIM;
                    let dot: f32 = (0..ASSEMBLY_HEAD_DIM)
                        .map(|d| q[row + d] * k[key_row + d])
                        .sum();
                    *score = dot * scale;
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut weights: Vec<f32> = scores.iter().map(|s| (s - maximum).exp()).collect();
                let total: f32 = weights.iter().sum();
                for weight in weights.iter_mut() {
                    *weight /= total;
                }
                for d in 0..ASSEMBLY_HEAD_DIM {
                    out[row + d] = (0..key_tokens)
                        .map(|key| weights[key] * v[key * ASSEMBLY_HEAD_DIM + d])
                        .sum();
                }
            }
        }
        out
    }

    /// Run one `mqa_bf16` against the reference, over a 2-D KV view.
    fn check_mqa(ctx: &MusaContext, tokens: usize, capacity: usize, key_tokens: usize) {
        let width = ASSEMBLY_HEADS * ASSEMBLY_HEAD_DIM;
        let q = upload(
            ctx,
            &values(tokens * width, 0x9e37_79b9),
            Shape::new(vec![tokens, ASSEMBLY_HEADS, ASSEMBLY_HEAD_DIM]),
        );
        let k_tensor = upload(
            ctx,
            &values(capacity * ASSEMBLY_HEAD_DIM, 0x85eb_ca6b),
            Shape::new(vec![capacity, ASSEMBLY_HEAD_DIM]),
        );
        let v_tensor = upload(
            ctx,
            &values(capacity * ASSEMBLY_HEAD_DIM, 0xc2b2_ae35),
            Shape::new(vec![capacity, ASSEMBLY_HEAD_DIM]),
        );

        // The reference reads the same rounded inputs the kernel does, so the
        // only difference left between them is the arithmetic.
        let expected = reference(
            &read_back(&q),
            &read_back(&k_tensor),
            &read_back(&v_tensor),
            tokens,
            key_tokens,
        );

        let out = mqa_bf16(ctx, &q, &k_tensor, &v_tensor, key_tokens).expect("native mqa");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        let tolerance = gate_for(&expected);
        assert_eq!(got.len(), expected.len());
        let mut worst = 0.0f32;
        for (index, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
            let error = (got - expected).abs();
            worst = worst.max(error);
            assert!(
                error <= tolerance,
                "mqa [{tokens}x{capacity}] key_tokens {key_tokens} at element {index}: \
                 {got} vs {expected}, abs {error:.3e} > {tolerance:.3e}"
            );
        }
        println!(
            "  mqa [{tokens}x{capacity}] key_tokens {key_tokens}  worst abs {worst:.3e} \
             (gate {tolerance:.3e}, {:.1}% of it)",
            100.0 * worst / tolerance
        );
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn mqa_covers_all_three_assembly_variants() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        // General: a query length the packed decode layouts are not built for.
        check_mqa(&ctx, 33, 33, 33);
        // Packed decode: ten rows, with a capacity outside the split-KV range.
        check_mqa(&ctx, 10, 20, 20);
        // Split-KV: ten rows and the capacity the model actually presents.
        check_mqa(&ctx, 10, 532, 532);
        // A key_tokens short of capacity is the reserved-prefix case: the tail
        // is poisoned and must not reach the softmax.
        check_mqa(&ctx, 10, 532, 500);
    }

    // ── The vendored mp_31 FMHA ──────────────────────────────────────────────

    #[cfg(apxinf_musa_mutlass_fmha)]
    mod mutlass_fmha {
        use apxinf_core::{Shape, Tensor};

        use crate::context::MusaContext;
        use crate::ffi::mutlass_fa::{check_mutlass_fa, MutlassFaVariant};
        use crate::kernels::attention::{mqa_bf16, ASSEMBLY_HEADS, ASSEMBLY_HEAD_DIM};
        use crate::kernels::contracts::{bf16_tensor, checked_i32, tensor_ptr};
        use crate::kernels::test_support::{read_back, to_bf16_bits, upload, values};

        /// Deterministic inputs for one shape, uploaded.
        ///
        /// The seeds are `check_mqa`'s, so a shape exercised there and here sees
        /// the same data and the two tests are looking at the same problem.
        fn mqa_inputs(ctx: &MusaContext, tokens: usize, capacity: usize) -> (Tensor, Tensor, Tensor) {
            let width = ASSEMBLY_HEADS * ASSEMBLY_HEAD_DIM;
            (
                upload(
                    ctx,
                    &values(tokens * width, 0x9e37_79b9),
                    Shape::new(vec![tokens, ASSEMBLY_HEADS, ASSEMBLY_HEAD_DIM]),
                ),
                upload(
                    ctx,
                    &values(capacity * ASSEMBLY_HEAD_DIM, 0x85eb_ca6b),
                    Shape::new(vec![capacity, ASSEMBLY_HEAD_DIM]),
                ),
                upload(
                    ctx,
                    &values(capacity * ASSEMBLY_HEAD_DIM, 0xc2b2_ae35),
                    Shape::new(vec![capacity, ASSEMBLY_HEAD_DIM]),
                ),
            )
        }

        /// Run the vendored kernel with an instantiation named by the caller.
        ///
        /// `mqa_bf16` chooses from the query length and the choice is read once
        /// per process, so a test that wants the *other* instantiation cannot go
        /// through it. This calls the same adapter `launch_mqa_mutlass` calls,
        /// with the variant passed in.
        fn mqa_bf16_with_variant(
            ctx: &MusaContext,
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            tokens: usize,
            key_tokens: usize,
            variant: MutlassFaVariant,
        ) -> Tensor {
            let output = bf16_tensor(ctx, &[tokens, ASSEMBLY_HEADS, ASSEMBLY_HEAD_DIM])
                .expect("output tensor");
            // SAFETY: shapes, dtype and device are established by the caller, and
            // every allocation outlives the asynchronous launch on this stream.
            let status = unsafe {
                crate::ffi::mutlass_fa::apxinf_musa_mutlass_fa_d256(
                    tensor_ptr(q).expect("q"),
                    tensor_ptr(k).expect("k"),
                    tensor_ptr(v).expect("v"),
                    tensor_ptr(&output).expect("out"),
                    checked_i32("mqa", "tokens", tokens).expect("tokens"),
                    checked_i32("mqa", "key_tokens", key_tokens).expect("key_tokens"),
                    variant as i32,
                    ctx.stream().handle(),
                )
            };
            check_mutlass_fa(status, "attention::mqa_bf16").expect("launch");
            output
        }

        /// The output's bits, after the stream has drained.
        fn output_bits(ctx: &MusaContext, output: &Tensor) -> Vec<u16> {
            ctx.stream().synchronize().expect("synchronise");
            read_back(output).iter().map(|value| to_bf16_bits(*value)).collect()
        }

        /// Read a little-endian bf16 file as f32. Every one of these values is
        /// exactly representable, so uploading them back is lossless.
        fn read_bf16(path: &std::path::Path) -> Vec<f32> {
            let bytes = std::fs::read(path)
                .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
            assert_eq!(bytes.len() % 2, 0, "{} is not a bf16 file", path.display());
            bytes
                .chunks_exact(2)
                .map(|pair| crate::kernels::test_support::from_bf16_bits(u16::from_le_bytes([pair[0], pair[1]])))
                .collect()
        }

        /// The two vendored instantiations must agree bit for bit.
        ///
        /// They tile the same arithmetic differently — tile 128 packed against
        /// tile 192 unpacked — and `mp31_fa_variant` substitutes between them
        /// for lengths MATE's own rule would have tiled another way. That
        /// substitution is only defensible if the tiling moves no bits at all,
        /// which is what this measures: at both of PI0.5's query lengths, and at
        /// the lengths around the boundary the rule divides on.
        ///
        /// If this fails, `mp31_fa_variant` must shrink to MATE's exact set and
        /// a third instantiation must be frozen. Do not ship the substitution.
        #[test]
        #[ignore = "needs a MUSA device"]
        fn mutlass_tile_choice_does_not_move_the_bits() {
            let ctx = MusaContext::new(0).expect("MUSA device 0");
            // The shapes PI0.5 presents, then one on each side of the boundary
            // between the two kernels: 16 is the last packed length, 17 the
            // first the rule would have packed at 192, and 522 the first it
            // leaves unpacked.
            let shapes = [(10usize, 532usize), (10, 10), (16, 16), (17, 17), (33, 33), (522, 522)];
            let mut first = None;
            for (tokens, key_tokens) in shapes {
                let (q, k, v) = mqa_inputs(&ctx, tokens, key_tokens);
                let mut bits = Vec::new();
                for variant in [MutlassFaVariant::PackedDecodeTile128, MutlassFaVariant::GeneralTile192] {
                    let output = mqa_bf16_with_variant(&ctx, &q, &k, &v, tokens, key_tokens, variant);
                    bits.push(output_bits(&ctx, &output));
                }
                let differing = bits[0].iter().zip(&bits[1]).filter(|(a, b)| a != b).count();
                println!(
                    "  tile agreement [{tokens}x{key_tokens}]  {differing} of {} elements differ",
                    bits[0].len()
                );
                assert_eq!(
                    differing, 0,
                    "the two mp_31 instantiations disagree on {differing} of {} elements at \
                     [{tokens}x{key_tokens}]; mp31_fa_variant may not substitute between them",
                    bits[0].len()
                );
                // A shape whose two variants agree because both wrote the same
                // thing -- zeros, say -- would pass vacuously. Keep the first
                // output to check against a different shape below.
                first.get_or_insert(bits[0].clone());
            }

            // Negative control: two different shapes must produce different
            // bits, or the comparison above is measuring nothing.
            let (q, k, v) = mqa_inputs(&ctx, 17, 17);
            let other = output_bits(
                &ctx,
                &mqa_bf16_with_variant(&ctx, &q, &k, &v, 17, 17, MutlassFaVariant::GeneralTile192),
            );
            let shared = first.as_ref().expect("a first shape").len().min(other.len());
            let shared_differing = first.as_ref().unwrap()[..shared]
                .iter()
                .zip(&other[..shared])
                .filter(|(a, b)| a != b)
                .count();
            let ratio = shared_differing as f64 / shared as f64;
            println!("  negative control: {:.1}% of the shared prefix differs", 100.0 * ratio);
            assert!(
                ratio > 0.5,
                "two different shapes agree on {:.1}% of their shared prefix, so this test \
                 cannot tell agreement from vacuity",
                100.0 * (1.0 - ratio)
            );
        }

        /// Replay the tensors this operator last ran through the torch bridge.
        ///
        /// The two paths are the same kernel source compiled with the same
        /// flags, so the expected result is bit identity and any difference is a
        /// fact about this build rather than about the arithmetic. The reference
        /// is a capture taken through the torch bridge before its entry was
        /// deleted.
        ///
        /// Skips loudly when `APXINF_MUSA_FA_REPLAY_DIR` is unset, rather than
        /// passing without having compared anything.
        #[test]
        #[ignore = "needs a MUSA device and a captured reference"]
        fn mqa_bf16_reproduces_the_mate_reference() {
            let Some(directory) = std::env::var_os("APXINF_MUSA_FA_REPLAY_DIR") else {
                eprintln!(
                    "skipping: set APXINF_MUSA_FA_REPLAY_DIR to a torch-bridge \
                     capture of this operator"
                );
                return;
            };
            let directory = std::path::PathBuf::from(directory);
            let ctx = MusaContext::new(0).expect("MUSA device 0");
            let width = ASSEMBLY_HEADS * ASSEMBLY_HEAD_DIM;

            let mut compared = 0usize;
            for entry in std::fs::read_dir(&directory).expect("replay directory") {
                let shape_dir = entry.expect("entry").path();
                if !shape_dir.is_dir() {
                    continue;
                }
                let name = shape_dir.file_name().unwrap().to_string_lossy().into_owned();
                let Some((tokens, key_tokens)) = name
                    .trim_start_matches("shape-")
                    .split_once('x')
                    .and_then(|(t, k)| Some((t.parse::<usize>().ok()?, k.parse::<usize>().ok()?)))
                else {
                    continue;
                };

                let q = upload(
                    &ctx,
                    &read_bf16(&shape_dir.join("q.bin")),
                    Shape::new(vec![tokens, ASSEMBLY_HEADS, ASSEMBLY_HEAD_DIM]),
                );
                let k = upload(
                    &ctx,
                    &read_bf16(&shape_dir.join("k.bin")),
                    Shape::new(vec![key_tokens, ASSEMBLY_HEAD_DIM]),
                );
                let v = upload(
                    &ctx,
                    &read_bf16(&shape_dir.join("v.bin")),
                    Shape::new(vec![key_tokens, ASSEMBLY_HEAD_DIM]),
                );
                let expected = read_bf16(&shape_dir.join("out.bin"));

                let output = mqa_bf16(&ctx, &q, &k, &v, key_tokens).expect("native mqa");
                let got: Vec<f32> = {
                    ctx.stream().synchronize().expect("synchronise");
                    read_back(&output)
                };

                assert_eq!(got.len(), expected.len(), "{name}: element count");
                assert_eq!(got.len(), tokens * width, "{name}: shape");
                let differing = got
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| to_bf16_bits(**a) != to_bf16_bits(**b))
                    .count();
                let max_abs = got
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let dot: f32 = got.iter().zip(&expected).map(|(a, b)| a * b).sum();
                let norm = |values: &[f32]| values.iter().map(|v| v * v).sum::<f32>().sqrt();
                let cosine = dot / (norm(&got) * norm(&expected)).max(f32::MIN_POSITIVE);

                println!(
                    "  {name}: {differing} of {} elements differ, max_abs {max_abs:.3e}, \
                     cosine {cosine:.9}",
                    got.len()
                );
                assert_eq!(
                    differing, 0,
                    "{name}: {differing} of {} elements differ from MATE's own output, \
                     max_abs {max_abs:.3e}, cosine {cosine:.9}. This is the same kernel source \
                     compiled with the same flags, so a difference here is a fact about this \
                     build",
                    got.len()
                );
                compared += 1;
            }
            assert!(compared > 0, "no shape-<tokens>x<key_tokens> directories in {}", directory.display());
            println!("  replayed {compared} shapes bit for bit");
        }
    }

    // ── The vision kernel ────────────────────────────────────────────────────

    /// A CPU reference for the vision attention: f32 scores, softmax and PV,
    /// each window on its own.
    fn reference_mha(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batches: usize,
        tokens_per_batch: usize,
        heads: usize,
    ) -> Vec<f32> {
        let head_dim = VISION_HEAD_DIM;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let row_of = |batch: usize, token: usize, head: usize| {
            ((batch * tokens_per_batch + token) * heads + head) * head_dim
        };
        let mut out = vec![0.0f32; batches * tokens_per_batch * heads * head_dim];
        for batch in 0..batches {
            for query in 0..tokens_per_batch {
                for head in 0..heads {
                    let row = row_of(batch, query, head);
                    let mut scores = vec![0.0f32; tokens_per_batch];
                    for (key, score) in scores.iter_mut().enumerate() {
                        let key_row = row_of(batch, key, head);
                        let dot: f32 = (0..head_dim).map(|d| q[row + d] * k[key_row + d]).sum();
                        *score = dot * scale;
                    }
                    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut weights: Vec<f32> =
                        scores.iter().map(|s| (s - maximum).exp()).collect();
                    let total: f32 = weights.iter().sum();
                    for weight in weights.iter_mut() {
                        *weight /= total;
                    }
                    for d in 0..head_dim {
                        out[row + d] = (0..tokens_per_batch)
                            .map(|key| weights[key] * v[row_of(batch, key, head) + d])
                            .sum();
                    }
                }
            }
        }
        out
    }

    /// Run one `mha_bf16` against the reference.
    fn check_mha(ctx: &MusaContext, batches: usize, tokens_per_batch: usize, heads: usize) {
        let tokens = batches * tokens_per_batch;
        let shape = Shape::new(vec![tokens, heads, VISION_HEAD_DIM]);
        let q = upload(
            ctx,
            &values(tokens * heads * VISION_HEAD_DIM, 0x27d4_eb2f),
            shape.clone(),
        );
        let k_tensor = upload(
            ctx,
            &values(tokens * heads * VISION_HEAD_DIM, 0x1656_67b1),
            shape.clone(),
        );
        let v_tensor = upload(
            ctx,
            &values(tokens * heads * VISION_HEAD_DIM, 0x9e37_79b9),
            shape,
        );

        let expected = reference_mha(
            &read_back(&q),
            &read_back(&k_tensor),
            &read_back(&v_tensor),
            batches,
            tokens_per_batch,
            heads,
        );

        let out = mha_bf16(ctx, &q, &k_tensor, &v_tensor, tokens_per_batch).expect("native mha");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        let tolerance = gate_for(&expected);
        assert_eq!(got.len(), expected.len());
        let mut worst = 0.0f32;
        for (index, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
            let error = (got - expected).abs();
            worst = worst.max(error);
            assert!(
                error <= tolerance,
                "mha [{batches}x{tokens_per_batch}x{heads}] at element {index}: \
                 {got} vs {expected}, abs {error:.3e} > {tolerance:.3e}"
            );
        }
        println!(
            "  mha [{batches}x{tokens_per_batch}] {heads} heads  worst abs {worst:.3e} \
             (gate {tolerance:.3e}, {:.1}% of it)",
            100.0 * worst / tolerance
        );
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn mha_matches_a_reference_across_window_shapes() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        // The model's shape.
        check_mha(&ctx, 2, 256, 16);
        // Windows that do not divide the warp count, and a single-row window:
        // both leave later warps with an empty key shard.
        check_mha(&ctx, 2, 3, 4);
        check_mha(&ctx, 1, 1, 16);
        // More windows than the key shard count, on a head count that is not
        // the model's.
        check_mha(&ctx, 3, 40, 5);
    }

    /// `mqa_bf16` checked k/v for shape only; a same-shaped F16 pair passed
    /// every test and reached the assembly as a raw pointer read as BF16.
    ///
    /// `kernels/cache.rs::reserve_prefix_f16` produces exactly this shape with
    /// an F16 label, and only `pi05/model/blocks/fp8_static.rs` consumes that
    /// today — a path that never calls this operator. If an F16 KV path ever
    /// does reach here, this refusal is the answer rather than a bug: both D256
    /// kernels are BF16-only.
    #[test]
    #[ignore = "needs a MUSA device"]
    fn mqa_refuses_a_kv_pair_that_is_not_bf16() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let (tokens, head_dim) = (10usize, ASSEMBLY_HEAD_DIM);
        let q = upload(
            &ctx,
            &values(tokens * ASSEMBLY_HEADS * head_dim, 0x5eed_1234),
            Shape::new(vec![tokens, ASSEMBLY_HEADS, head_dim]),
        );
        let rows = tokens * ASSEMBLY_HEADS;

        let f16 = zeros_as(&ctx, Shape::new(vec![rows, head_dim]), DType::F16);
        let message = mqa_bf16(&ctx, &q, &f16, &f16, tokens)
            .unwrap_err()
            .to_string();
        assert!(message.contains("mqa k expects bf16"), "{message}");

        // v is checked on its own rather than only through k's shape.
        let bf16 = zeros_as(&ctx, Shape::new(vec![rows, head_dim]), DType::BF16);
        let message = mqa_bf16(&ctx, &q, &bf16, &f16, tokens)
            .unwrap_err()
            .to_string();
        assert!(message.contains("mqa v expects bf16"), "{message}");
    }

    /// `allow_fallback = false` refuses the general D256 assembly on a
    /// decode-shaped call: the packed decode kernel is the measured route.
    #[test]
    #[ignore = "needs a MUSA device"]
    fn mqa_without_fallback_refuses_a_general_variant_decode_shape() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let (tokens, capacity) = (3usize, 8usize);
        let q = upload(
            &ctx,
            &values(tokens * ASSEMBLY_HEADS * ASSEMBLY_HEAD_DIM, 0x5eed_1234),
            Shape::new(vec![tokens, ASSEMBLY_HEADS, ASSEMBLY_HEAD_DIM]),
        );
        let k = upload(
            &ctx,
            &values(capacity * ASSEMBLY_HEAD_DIM, 0x0bad_c0de),
            Shape::new(vec![capacity, ASSEMBLY_HEAD_DIM]),
        );
        let v = upload(
            &ctx,
            &values(capacity * ASSEMBLY_HEAD_DIM, 0x0bad_c0de),
            Shape::new(vec![capacity, ASSEMBLY_HEAD_DIM]),
        );
        let mut output = ctx
            .allocate_output(
                Shape::new(vec![tokens, ASSEMBLY_HEADS, ASSEMBLY_HEAD_DIM]),
                DType::BF16,
            )
            .expect("an output");

        let error = mqa_bf16_into_with(
            &ctx,
            &q,
            &k,
            &v,
            &mut output,
            capacity,
            TacticDispatch::RequireExact,
        )
        .expect_err("the general D256 assembly is a substitute");
        assert!(error.to_string().contains("packed decode"), "{error}");
    }
}

/// The D256 fallback rule, without a device.
#[cfg(test)]
mod route_tests {
    use super::*;

    #[test]
    fn a_general_decode_shape_is_refused_when_exact_is_required() {
        let error = require_packed_decode(3, 8, TacticDispatch::RequireExact)
            .expect_err("the general assembly is a substitute on a decode shape");
        assert!(error.to_string().contains("packed decode"), "{error}");
    }

    #[test]
    fn the_packed_decode_shape_is_not_a_fallback() {
        assert!(
            require_packed_decode(PACKED_DECODE_TOKENS, 532, TacticDispatch::RequireExact).is_ok()
        );
    }

    #[test]
    fn a_prefix_pass_is_not_a_fallback() {
        assert!(require_packed_decode(522, 522, TacticDispatch::RequireExact).is_ok());
    }

    #[test]
    fn allowing_fallback_accepts_the_general_kernel() {
        assert!(require_packed_decode(3, 8, TacticDispatch::AllowFallback).is_ok());
    }

    #[test]
    fn a_mutlass_packed_decode_shape_is_not_a_fallback() {
        assert!(require_mutlass_decode(10, 532, TacticDispatch::RequireExact).is_ok());
    }

    #[test]
    fn a_mutlass_unpacked_substitute_is_refused_when_exact_is_required() {
        let error = require_mutlass_decode(20, 532, TacticDispatch::RequireExact)
            .expect_err("17..=256 rows run the unpacked tile in place of MATE's packed 192");
        assert!(error.to_string().contains("packed decode"), "{error}");
    }

    #[test]
    fn a_mutlass_prefix_pass_is_not_a_fallback() {
        assert!(require_mutlass_decode(522, 522, TacticDispatch::RequireExact).is_ok());
    }

    #[test]
    fn a_mutlass_mate_choice_is_not_a_fallback() {
        assert!(require_mutlass_decode(300, 532, TacticDispatch::RequireExact).is_ok());
    }

    #[test]
    fn allowing_fallback_accepts_the_mutlass_substitute() {
        assert!(require_mutlass_decode(20, 532, TacticDispatch::AllowFallback).is_ok());
    }
}

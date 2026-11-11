//! Raw bindings for the custom MUSA operators' C ABI.
//!
//! The symbols are defined in `adapters/custom_kernels.mu`, which owns the
//! launch configuration; the device code it includes is
//! `kernels/custom/*.cuh`. As in `ffi::musa`, only the symbols ApxInf calls are
//! declared, by hand.
//!
//! This file is also one of the kernel-build-id inputs (`build.rs`'s
//! `KERNEL_INPUT_FILES`), so a signature change here re-keys the build even
//! when no `.cuh` moved.

use std::ffi::c_void;

use super::musa::musaError_t;

extern "C" {
    pub fn apxinf_musa_capture_anchor(stream: *mut c_void) -> musaError_t;
    pub fn apxinf_musa_mublas_release_stream(stream: *mut c_void);
    pub fn apxinf_musa_mudnn_release_stream(stream: *mut c_void);
    /// muDNN BF16 GEMM. Returns 0 on success; on failure
    /// [`apxinf_musa_mudnn_last_error`] names the reason.
    pub fn apxinf_musa_mudnn_bf16_gemm(
        a: *const c_void,
        w: *const c_void,
        y: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
    /// muDNN flash attention. Same status convention as the GEMM above.
    pub fn apxinf_musa_mudnn_flash_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        y: *mut c_void,
        batch: i32,
        heads: i32,
        tokens: i32,
        dim: i32,
        stream: *mut c_void,
    ) -> i32;
    /// Why the last muDNN call on this thread failed. A static string.
    pub fn apxinf_musa_mudnn_last_error() -> *const std::os::raw::c_char;
    /// `bias + activation(input)`, elementwise over a row-major `[rows, cols]`
    /// BF16 buffer.
    ///
    /// `bias` may be null, in which case it is skipped; when it is not, it
    /// holds one value per column and is broadcast down the rows. `activation`
    /// is the CUDA kernel ABI's code: 0 none, 1 GELU (tanh approximation),
    /// 2 SiLU. Any other value is rejected rather than treated as "none",
    /// because a code this ABI does not know means the facade above it and this
    /// boundary disagree, and computing the identity would look like success.
    ///
    /// The launch is asynchronous on `stream`.
    pub fn apxinf_musa_bias_activation_bf16(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        activation: i32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// Gated residual followed by an adaLN RMSNorm, writing both outputs.
    ///
    /// `projection` and `residual` are `[rows, cols]`; `gate_style` and
    /// `norm_style` are `[3 * cols]`, and only parts of them are read -- the
    /// gate is the third segment of `gate_style`, and the norm's scale and
    /// shift are the first two of `norm_style`. `hidden` is written **rounded
    /// to bf16** and read back before it is normalized, so that rounding is
    /// part of the arithmetic.
    ///
    /// The launch is asynchronous on `stream`.
    pub fn apxinf_musa_ada_gate_residual_rms_norm_bf16(
        projection: *const c_void,
        residual: *const c_void,
        gate_style: *const c_void,
        norm_style: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// RMSNorm over `[rows, cols]` with a `[cols]` weight.
    pub fn apxinf_musa_rms_norm_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// adaLN RMSNorm: `style` is `[3 * cols]` and its first two segments are
    /// the scale and shift; the third belongs to the gated variant.
    pub fn apxinf_musa_ada_rms_norm_bf16(
        input: *const c_void,
        style: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// `projection + bias + residual`, rounded once. `bias` may be null and is
    /// row-broadcast when it is not.
    pub fn apxinf_musa_bias_residual_bf16(
        projection: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// The same followed by an RMSNorm with a weight and no shift. `hidden` is
    /// written rounded to bf16 and read back before the norm.
    pub fn apxinf_musa_bias_residual_rms_norm_bf16(
        projection: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// `first`'s rows then `second`'s, both `cols` wide.
    pub fn apxinf_musa_concat_rows_bf16(
        first: *const c_void,
        second: *const c_void,
        output: *mut c_void,
        first_rows: i32,
        second_rows: i32,
        cols: i32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// One explicit Euler step: `state + dt * velocity`.
    pub fn apxinf_musa_euler_update_bf16(
        state: *const c_void,
        velocity: *const c_void,
        output: *mut c_void,
        count: i64,
        dt: f32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// Per-channel scale: a BF16 `[rows, cols]` matrix times an f32 `[cols]`
    /// vector, rounded once.
    pub fn apxinf_musa_mul_channels_bf16(
        input: *const c_void,
        channel_scale: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// Position embedding: `projection + position[row % tokens_per_view] + bias`,
    /// rounded once. `bias` may be null.
    ///
    /// The position table holds `tokens_per_view` rows of `cols`; `rows` must be
    /// a multiple of `tokens_per_view`, which is what the tiling means.
    pub fn apxinf_musa_add_position_bf16(
        projection: *const c_void,
        bias: *const c_void,
        position: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        tokens_per_view: i32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// Row gather from a BF16 `[vocab, width]` table, scaled by `normalizer`.
    ///
    /// `ids` is a device buffer of `tokens` `u32` values, all of which must be
    /// inside the table — the facade checks that before calling, because the
    /// kernel has nowhere to report a bad id from.
    ///
    /// `normalizer` is `bf16(sqrtf(width))` as an f32. Rounding it is part of
    /// the contract, not an optimization: see `kernels/custom/embedding.cuh`.
    pub fn apxinf_musa_embedding_bf16_scaled(
        table: *const c_void,
        ids: *const c_void,
        output: *mut c_void,
        tokens: i32,
        width: i32,
        normalizer: f32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// `[tokens, heads, dim]` BF16 -> `[batch * heads, tokens_per_batch, dim]`
    /// BF16.
    ///
    /// Q and K take this one. They arrive BF16 from the QKV split and the
    /// scores GEMM takes BF16 operands, so for them the permutation is a pure
    /// move; widening would only add a copy and a rounding. Only the
    /// into-head-major direction exists, because that is the only one they
    /// need.
    pub fn apxinf_musa_permute_heads_bf16(
        input: *const c_void,
        output: *mut c_void,
        tokens: i32,
        heads: i32,
        dim: i32,
        tokens_per_batch: i32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// The same permutation with an f32 head-major side, BF16 -> f32.
    ///
    /// V takes this one: the softmax output it is multiplied by is f32, and
    /// mublas does not mix operand types within one product, so V is the
    /// operand that widens.
    pub fn apxinf_musa_permute_heads_to_f32(
        input: *const c_void,
        output: *mut c_void,
        tokens: i32,
        heads: i32,
        dim: i32,
        tokens_per_batch: i32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// The inverse of [`apxinf_musa_permute_heads_to_f32`], f32 back to BF16.
    pub fn apxinf_musa_permute_heads_from_f32(
        input: *const c_void,
        output: *mut c_void,
        tokens: i32,
        heads: i32,
        dim: i32,
        tokens_per_batch: i32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// In-place softmax over each row of a `[rows, cols]` f32 tensor.
    ///
    /// f32 arithmetic with the max subtracted — the contract `torch.softmax`
    /// had in the eager bridge this replaced.
    pub fn apxinf_musa_row_softmax_f32(
        data: *mut c_void,
        rows: i32,
        cols: i32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// Split a packed `[tokens, 3 * projection_width]` projection into
    /// contiguous q | k | v, adding an optional per-column bias, rounded once.
    pub fn apxinf_musa_qkv_split_bias_bf16(
        qkv: *const c_void,
        bias: *const c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        tokens: i32,
        projection_width: i32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// LayerNorm over `[rows, cols]` with a `[cols]` weight **and** bias.
    pub fn apxinf_musa_layer_norm_bf16(
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// Projection plus an optional bias, added into the residual, then a
    /// LayerNorm with a weight and a shift, writing both outputs.
    ///
    /// `hidden` is written **rounded to bf16** and read back for the mean and
    /// variance, so the rounding is part of the arithmetic.
    pub fn apxinf_musa_bias_residual_layer_norm_bf16(
        projection: *const c_void,
        projection_bias: *const c_void,
        residual: *const c_void,
        norm_weight: *const c_void,
        norm_bias: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// Split a packed QKV projection, rotate Q and K, write V unrotated.
    ///
    /// One symbol for both rope entry points. `k` and `v` point either at fresh
    /// `[tokens, kv_heads, head_dim]` buffers (`kv_output_offset` 0) or at the
    /// caller's 2-D caches, whose rows `[kv_output_offset, kv_output_offset +
    /// tokens)` are the ones written. `bias` may be null.
    ///
    /// The numerical contract is the recorded reference's rounding chain, not
    /// the CUDA kernel's: see `kernels/custom/rope.cuh`.
    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_musa_qkv_rope_bf16(
        qkv: *const c_void,
        bias: *const c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        tokens: i32,
        q_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        theta: f32,
        position_offset: i32,
        kv_output_offset: i32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// BF16-rounded GeGLU, optional BF16-rounded equalization, then INT8 rows
    /// with FP16 scales and 512 trailing zero scales.
    pub fn apxinf_musa_geglu_quantize_rows_bf16_int8(
        input: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        rows: i32,
        cols: i32,
        inv_c: *const c_void,
        stream: *mut c_void,
    ) -> musaError_t;

    /// Gated GELU over `[rows, 2 * inner]`: `gelu_tanh(gate) * up`, rounded once.
    pub fn apxinf_musa_geglu_bf16(
        gate_up: *const c_void,
        output: *mut c_void,
        rows: i32,
        inner: i32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// Raw `uint8` RGB frames -> BF16 patch rows, in one gather.
    ///
    /// `images` holds `views * 3 * image_size^2` bytes with no shape metadata of
    /// its own; `patches` is `[views * (image_size / patch_size)^2,
    /// 3 * patch_size^2]` BF16 and is written in full. `layout` names the input
    /// arrangement only: 0 is NHWC, 1 is NCHW. Any other value is refused
    /// rather than assumed to be NHWC.
    ///
    /// No resize, no interpolation, no patch embedding and no per-channel
    /// statistics -- see `kernels/custom/preprocess.cuh` for why adding one
    /// would be wrong rather than extra. The launch is asynchronous on
    /// `stream`.
    ///
    /// Unlike CUDA's `apxinf_static_rgb_u8_to_patches_bf16`, the name carries
    /// no `static_` infix: CUDA splits that adapter into a translation unit per
    /// precision route, while every MUSA operator lands in one.
    pub fn apxinf_musa_rgb_u8_to_patches_bf16(
        images: *const c_void,
        patches: *mut c_void,
        views: i32,
        image_size: i32,
        patch_size: i32,
        layout: i32,
        stream: *mut c_void,
    ) -> musaError_t;

    /// Row-quantize a BF16 `[rows, cols]` activation to INT8, with FP16 row
    /// scales.
    ///
    /// `output` is `rows * cols` INT8. `scales` is `rows + 512` FP16 entries:
    /// the first `rows` are written and the 512-entry tail is zeroed, because the
    /// `s8mma` tile's guarded scale load can reach into it. One block per row, so
    /// `cols` may be any positive width.
    ///
    /// The launch is asynchronous on `stream`. `inv_c` is either null or `cols`
    /// f32 per-channel equalization factors the row is multiplied by before its
    /// amax is measured (the SmoothQuant fold's activation half).
    pub fn apxinf_musa_quantize_rows_bf16_int8(
        input: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        rows: i32,
        cols: i32,
        inv_c: *const c_void,
        stream: *mut c_void,
    ) -> musaError_t;
}

/// Turn a muDNN status into the adapter's message, naming the operator.
///
/// The counterpart of `ffi::mublas::check_mublas` and `ffi::mubin::check_fa_d256`
/// for the one adapter that returns a bare `int`. Two vocabularies share that
/// channel — the vendor's `md::Status` and the adapter's own `kStatus*` codes —
/// and this does not try to separate them: `muDNN BF16 GEMM {shape}: status 1004`
/// is what the adapter wrote into its buffer, and what it says there is the
/// adapter's business.
///
/// Falls back to the bare number when the buffer holds nothing, which is the
/// case for a status no `record_failure` saw — a value returned before this
/// adapter started writing messages.
fn mudnn_detail(status: i32) -> String {
    // SAFETY: the adapter returns a pointer to a thread-local static buffer that
    // lives until the next call on this thread.
    let text = unsafe { crate::ffi::error_text(apxinf_musa_mudnn_last_error()) };
    text.filter(|text| !text.is_empty())
        .unwrap_or_else(|| format!("the muDNN adapter returned {status}"))
}

/// Turn a status into the adapter's message; `operation` runs only on failure,
/// so a successful launch does not format a string it would throw away.
pub fn check_mudnn(status: i32, operation: impl FnOnce() -> String) -> Result<(), String> {
    crate::ffi::check_status(status, || format!("{}: {}", operation(), mudnn_detail(status)))
}

//! Fused residual and normalization operator contracts.

use apxinf_core::{Result, Tensor};

#[cfg(apxinf_musa_available)]
use super::contracts::{checked_i32, optional_tensor_ptr, tensor_ptr};
use super::contracts::{
    bf16_output, matrix_tensor, require_matching_bf16, require_matrix_bf16,
    require_optional_vector_bf16, require_output_matrix, require_vector_at_least_bf16,
    require_vector_bf16,
};
use crate::context::MusaContext;
#[cfg(apxinf_musa_available)]
use crate::error::musa_error;
#[cfg(apxinf_musa_available)]
use crate::ffi;
use crate::kernels::not_ported;
#[cfg(apxinf_musa_available)]
use crate::kernels::check_native;
#[cfg(not(apxinf_musa_available))]
use crate::kernels::native_unavailable;

/// The two outputs of a fused residual-plus-normalization step.
///
/// Both are independent workspace slices: `hidden` is the residual sum and
/// `normalized` is its normalization. PI0.5 uses `hidden` as the next block's
/// residual and `normalized` as its input, so neither can be a view of the
/// other.
#[derive(Debug)]
pub struct ResidualNormTensors {
    pub hidden: Tensor,
    pub normalized: Tensor,
}

/// Bias plus residual add.
pub fn bias_residual_bf16(
    ctx: &MusaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    let (rows, cols) = require_matrix_bf16(ctx, projection, "bias residual", "projection")?;
    let mut output = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    bias_residual_bf16_into(ctx, projection, bias, residual, &mut output)?;
    Ok(output)
}

/// Write the bias-plus-residual sum into a caller-provided output.
pub fn bias_residual_bf16_into(
    ctx: &MusaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    output: &mut Tensor,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, projection, "bias residual", "projection")?;
    require_matching_bf16(ctx, "bias residual", "residual", projection, residual)?;
    require_optional_vector_bf16(ctx, bias, cols, "bias residual", "bias")?;
    require_output_matrix(ctx, output, rows, cols, "bias residual", "hidden")?;
    launch_bias_residual(ctx, projection, bias, residual, output, rows, cols)
}

#[cfg(apxinf_musa_available)]
fn launch_bias_residual(
    ctx: &MusaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    output: &Tensor,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let rows = checked_i32("bias_residual", "rows", rows)?;
    let cols = checked_i32("bias_residual", "cols", cols)?;
    let bias_pointer = optional_tensor_ptr(bias)?;
    // SAFETY: shapes, dtypes and one device are established above; the null
    // bias is the kernel's documented "no bias" case; every buffer is owned by
    // a caller `Tensor` that outlives the asynchronous launch.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_bias_residual_bf16(
                tensor_ptr(projection)?,
                bias_pointer,
                tensor_ptr(residual)?,
                tensor_ptr(output)?,
                rows,
                cols,
                ctx.stream().handle(),
            ),
            "bias_residual",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch_bias_residual(
    _ctx: &MusaContext,
    _projection: &Tensor,
    _bias: Option<&Tensor>,
    _residual: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
) -> Result<()> {
    Err(native_unavailable("fused::bias_residual_bf16"))
}

/// Bias, residual add, and RMS normalization in one pass.
pub fn bias_residual_rms_bf16(
    ctx: &MusaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let (rows, cols) = require_matrix_bf16(ctx, projection, "bias residual rms", "projection")?;
    let hidden = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    let normalized = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    let (mut hidden, mut normalized) = (hidden, normalized);
    bias_residual_rms_bf16_into(
        ctx,
        projection,
        bias,
        residual,
        weight,
        &mut hidden,
        &mut normalized,
        eps,
    )?;
    Ok(ResidualNormTensors { hidden, normalized })
}

/// Write the residual sum and its RMS normalization into caller-provided
/// outputs.
pub fn bias_residual_rms_bf16_into(
    ctx: &MusaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    hidden: &mut Tensor,
    normalized: &mut Tensor,
    eps: f32,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, projection, "bias residual rms", "projection")?;
    require_matching_bf16(ctx, "bias residual rms", "residual", projection, residual)?;
    require_vector_bf16(ctx, weight, cols, "bias residual rms", "weight")?;
    require_optional_vector_bf16(ctx, bias, cols, "bias residual rms", "bias")?;
    require_output_matrix(ctx, hidden, rows, cols, "bias residual rms", "hidden")?;
    require_output_matrix(ctx, normalized, rows, cols, "bias residual rms", "normalized")?;
    launch_bias_residual_rms(
        ctx, projection, bias, residual, weight, hidden, normalized, rows, cols, eps,
    )
}

/// The launch, split by whether this build has a MUSA toolkit.
///
/// # Numerics
/// The reduction sums in a different order from torch's `mean`, so this is
/// gated rather than expected to be bit-identical. `hidden` is the exception
/// and does come out equal: it is `projection + bias + residual` rounded once,
/// with no reduction in it at all.
#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_bias_residual_rms(
    ctx: &MusaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    hidden: &Tensor,
    normalized: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    let rows = checked_i32("bias_residual_rms", "rows", rows)?;
    let cols = checked_i32("bias_residual_rms", "cols", cols)?;
    let bias_pointer = optional_tensor_ptr(bias)?;
    // SAFETY: as in `launch_bias_residual`.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_bias_residual_rms_norm_bf16(
                tensor_ptr(projection)?,
                bias_pointer,
                tensor_ptr(residual)?,
                tensor_ptr(weight)?,
                tensor_ptr(hidden)?,
                tensor_ptr(normalized)?,
                rows,
                cols,
                eps,
                ctx.stream().handle(),
            ),
            "bias_residual_rms",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch_bias_residual_rms(
    _ctx: &MusaContext,
    _projection: &Tensor,
    _bias: Option<&Tensor>,
    _residual: &Tensor,
    _weight: &Tensor,
    _hidden: &Tensor,
    _normalized: &Tensor,
    _rows: usize,
    _cols: usize,
    _eps: f32,
) -> Result<()> {
    Err(native_unavailable("fused::bias_residual_rms_bf16"))
}

/// Bias, residual add, and layer normalization in one pass.
///
/// Native: `kernels/custom/fused.cuh`'s `bias_residual_layer_norm_bf16_kernel`,
/// one block per row, through `adapters/custom_kernels.mu`.
#[allow(clippy::too_many_arguments)]
pub fn bias_residual_layer_bf16(
    ctx: &MusaContext,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let (rows, cols) = require_matrix_bf16(ctx, projection, "bias residual layer", "projection")?;
    let hidden = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    let normalized = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    let (mut hidden, mut normalized) = (hidden, normalized);
    bias_residual_layer_bf16_into(
        ctx,
        projection,
        projection_bias,
        residual,
        norm_weight,
        norm_bias,
        &mut hidden,
        &mut normalized,
        eps,
    )?;
    Ok(ResidualNormTensors { hidden, normalized })
}

/// Write the residual sum and its layer normalization into caller-provided
/// outputs.
#[allow(clippy::too_many_arguments)]
pub fn bias_residual_layer_bf16_into(
    ctx: &MusaContext,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    hidden: &mut Tensor,
    normalized: &mut Tensor,
    eps: f32,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, projection, "bias residual layer", "projection")?;
    require_matching_bf16(ctx, "bias residual layer", "residual", projection, residual)?;
    require_vector_bf16(ctx, norm_weight, cols, "bias residual layer", "norm_weight")?;
    require_vector_bf16(ctx, norm_bias, cols, "bias residual layer", "norm_bias")?;
    require_optional_vector_bf16(
        ctx,
        projection_bias,
        cols,
        "bias residual layer",
        "projection_bias",
    )?;
    require_output_matrix(ctx, hidden, rows, cols, "bias residual layer", "hidden")?;
    require_output_matrix(
        ctx,
        normalized,
        rows,
        cols,
        "bias residual layer",
        "normalized",
    )?;
    launch_layer(
        ctx,
        projection,
        projection_bias,
        residual,
        norm_weight,
        norm_bias,
        hidden,
        normalized,
        rows,
        cols,
        eps,
    )
}

/// The launch, split by whether this build has a MUSA toolkit.
///
/// # Numerics
/// The kernel rounds `hidden` to bf16 and reads it back for the mean and
/// variance, which is what the reference model does. The eager bridge this
/// replaces writes the same rounded `hidden` but takes its statistics from the
/// unrounded f32 sum it still holds -- so this operator does **not** reproduce
/// the bridge's arithmetic here, deliberately: the reference is the model, and
/// the bridge is only a scaffold standing in for a kernel that was always going
/// to round. The difference is one bf16 rounding inside the statistics, the same
/// one `fused.bias_residual_rms_bf16` measures at ~1.4 bf16 ulp.
#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_layer(
    ctx: &MusaContext,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    hidden: &Tensor,
    normalized: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    let rows = checked_i32("bias_residual_layer", "rows", rows)?;
    let cols = checked_i32("bias_residual_layer", "cols", cols)?;
    let projection_bias = optional_tensor_ptr(projection_bias)?;
    // SAFETY: shapes, dtypes and one device are established above, both outputs
    // are `[rows, cols]` BF16 workspace allocations, and every buffer outlives
    // the asynchronous launch on `ctx.stream()`. The null bias is the kernel's
    // documented "no projection bias" case.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_bias_residual_layer_norm_bf16(
                tensor_ptr(projection)?,
                projection_bias,
                tensor_ptr(residual)?,
                tensor_ptr(norm_weight)?,
                tensor_ptr(norm_bias)?,
                tensor_ptr(hidden)?,
                tensor_ptr(normalized)?,
                rows,
                cols,
                eps,
                ctx.stream().handle(),
            ),
            "bias_residual_layer",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch_layer(
    _ctx: &MusaContext,
    _projection: &Tensor,
    _projection_bias: Option<&Tensor>,
    _residual: &Tensor,
    _norm_weight: &Tensor,
    _norm_bias: &Tensor,
    _hidden: &Tensor,
    _normalized: &Tensor,
    _rows: usize,
    _cols: usize,
    _eps: f32,
) -> Result<()> {
    Err(native_unavailable("fused::bias_residual_layer_bf16"))
}

/// Gated residual add plus adaptive RMS normalization.
///
/// `gate_style` and `norm_style` are each BF16 `[3 * cols]`: gate, shift and
/// scale, as [`super::norm::adaptive_rms_bf16`] takes them.
pub fn adaptive_gate_residual_rms_bf16(
    ctx: &MusaContext,
    projection: &Tensor,
    residual: &Tensor,
    gate_style: &Tensor,
    norm_style: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let (rows, cols) = require_matrix_bf16(ctx, projection, "ada gate residual", "projection")?;
    let hidden = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    let normalized = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    let (mut hidden, mut normalized) = (hidden, normalized);
    adaptive_gate_residual_rms_bf16_into(
        ctx,
        projection,
        residual,
        gate_style,
        norm_style,
        &mut hidden,
        &mut normalized,
        eps,
    )?;
    Ok(ResidualNormTensors { hidden, normalized })
}

/// Write the gated residual sum and its adaptive RMS normalization into
/// caller-provided outputs.
///
/// The kernel reads `norm_style` as scale then shift, so its first `2 * cols`
/// entries are the contract; `gate_style` must carry `3 * cols` because the
/// gate is its third segment.
#[allow(clippy::too_many_arguments)]
pub fn adaptive_gate_residual_rms_bf16_into(
    ctx: &MusaContext,
    projection: &Tensor,
    residual: &Tensor,
    gate_style: &Tensor,
    norm_style: &Tensor,
    hidden: &mut Tensor,
    normalized: &mut Tensor,
    eps: f32,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, projection, "ada gate residual", "projection")?;
    require_matching_bf16(ctx, "ada gate residual", "residual", projection, residual)?;
    require_vector_at_least_bf16(ctx, gate_style, 3 * cols, "ada gate residual", "gate_style")?;
    require_vector_at_least_bf16(ctx, norm_style, 2 * cols, "ada gate residual", "norm_style")?;
    require_output_matrix(ctx, hidden, rows, cols, "ada gate residual", "hidden")?;
    require_output_matrix(ctx, normalized, rows, cols, "ada gate residual", "normalized")?;
    launch(
        ctx, projection, residual, gate_style, norm_style, hidden, normalized, rows, cols, eps,
    )
}

/// The launch, split by whether this build has a MUSA toolkit.
///
/// # Numerics
/// The kernel reproduces the bridge's arithmetic, including the part that is
/// easy to lose: `hidden` is written rounded to bf16 and read back before the
/// RMS is taken, so the rounding is inside the computation rather than a
/// storage detail. The bridge does the same thing explicitly (`_write` then
/// `_f32(t["hidden"])`).
///
/// What the two sides do **not** share is the summation order: torch's `mean`
/// over the last dimension and this kernel's warp/block tree both accumulate
/// f32 squares in a different order. Expect differences at the last f32 digit
/// rather than bit equality, which is why this operator is gated and the
/// elementwise one is not.
#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch(
    ctx: &MusaContext,
    projection: &Tensor,
    residual: &Tensor,
    gate_style: &Tensor,
    norm_style: &Tensor,
    hidden: &Tensor,
    normalized: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    let rows = checked_i32("adaptive_gate_residual_rms", "rows", rows)?;
    let cols = checked_i32("adaptive_gate_residual_rms", "cols", cols)?;
    // SAFETY: the checks above have established shapes, dtypes and one device,
    // and both outputs are `[rows, cols]` BF16 allocations from this context.
    // The buffers the kernel writes are owned by the caller's `Tensor`s, which
    // outlive the asynchronous launch on `ctx.stream()`.
    unsafe {
        ffi::check_runtime(ffi::custom::apxinf_musa_ada_gate_residual_rms_norm_bf16(
            tensor_ptr(projection)?,
            tensor_ptr(residual)?,
            tensor_ptr(gate_style)?,
            tensor_ptr(norm_style)?,
            tensor_ptr(hidden)?,
            tensor_ptr(normalized)?,
            rows,
            cols,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(musa_error)?;
    }
    Ok(())
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch(
    _ctx: &MusaContext,
    _projection: &Tensor,
    _residual: &Tensor,
    _gate_style: &Tensor,
    _norm_style: &Tensor,
    _hidden: &Tensor,
    _normalized: &Tensor,
    _rows: usize,
    _cols: usize,
    _eps: f32,
) -> Result<()> {
    Err(native_unavailable("fused::adaptive_gate_residual_rms_bf16"))
}

// ── FP8 and FP8-output fused operators (not ported) ──────────────────────

/// FP8 GEMM plus an FP16 bias, returning FP16.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bias_fp8(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<Tensor> {
    let _ = (ctx, activation, weight, bias, activation_scale, weight_scale);
    Err(not_ported("fused::gemm_bias_fp8"))
}

/// FP8 GEMM plus FP16 bias and GELU, returning E4M3.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bias_gelu_fp8(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
    output_scale: f32,
) -> Result<Tensor> {
    let _ = (
        ctx,
        activation,
        weight,
        bias,
        activation_scale,
        weight_scale,
        output_scale,
    );
    Err(not_ported("fused::gemm_bias_gelu_fp8"))
}

/// FP8 GEMM plus bias and residual add, returning FP16.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bias_residual_fp8(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<Tensor> {
    let _ = (
        ctx,
        activation,
        weight,
        bias,
        residual,
        activation_scale,
        weight_scale,
    );
    Err(not_ported("fused::gemm_bias_residual_fp8"))
}

/// Bias, residual and RMSNorm with an E4M3 normalized output.
#[allow(clippy::too_many_arguments)]
pub fn bias_residual_rms_quant_f16_e4m3(
    ctx: &MusaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<ResidualNormTensors> {
    let _ = (ctx, projection, bias, residual, weight, eps, scale);
    Err(not_ported("fused::bias_residual_rms_quant_f16_e4m3"))
}

/// Bias, residual and LayerNorm with an E4M3 normalized output.
#[allow(clippy::too_many_arguments)]
pub fn bias_residual_layer_quant_f16_e4m3(
    ctx: &MusaContext,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<ResidualNormTensors> {
    let _ = (
        ctx,
        projection,
        projection_bias,
        residual,
        norm_weight,
        norm_bias,
        eps,
        scale,
    );
    Err(not_ported("fused::bias_residual_layer_quant_f16_e4m3"))
}

/// Gated residual and adaptive RMSNorm with an E4M3 normalized output.
#[allow(clippy::too_many_arguments)]
pub fn adaptive_gate_residual_rms_quant_f16_e4m3(
    ctx: &MusaContext,
    projection: &Tensor,
    residual: &Tensor,
    gate_style: &Tensor,
    norm_style: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<ResidualNormTensors> {
    let _ = (
        ctx,
        projection,
        residual,
        gate_style,
        norm_style,
        eps,
        scale,
    );
    Err(not_ported("fused::adaptive_gate_residual_rms_quant_f16_e4m3"))
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    //! Device tests for the fused adaptive-gate residual RMSNorm.
    //!
    //! This is the port's first kernel with a block reduction, so the thing
    //! worth testing is the reduction itself: `block_sum_parallel_unsafe`
    //! splits each row across 8 software warps, and if MUSA's 32-lane shuffle
    //! groups did not behave as the port measured them, the sum
    //! would be wrong in a way that only shows up as a slightly-off RMS.
    //!
    //! The reference below mirrors the kernel exactly, including the part that
    //! is easy to get wrong: `hidden` is rounded to bf16 before it is
    //! normalized, so the rounding is inside the arithmetic.
    //!
    //! Needs a MUSA device, so it is `#[ignore]`d:
    //! `cargo test -p apxinf-musa --lib -- --ignored --nocapture`.

    use apxinf_core::{DType, Error, Shape, Tensor};

    use crate::context::MusaContext;
    use crate::kernels::fused::{
        adaptive_gate_residual_rms_bf16, bias_residual_bf16, bias_residual_layer_bf16,
        bias_residual_rms_bf16,
    };
    use crate::kernels::test_support::{
        from_bf16_bits, gate_for, read_back, to_bf16_bits, upload, values, zeros_as,
    };

    /// The value the model uses; the recorded signatures carry it as a scalar.
    const EPS: f32 = 1e-6;

    fn check(ctx: &MusaContext, rows: usize, cols: usize) {
        let projection_values = values(rows * cols, 0x5eed_1234);
        let residual_values = values(rows * cols, 0x0bad_c0de);
        let gate_style_values = values(3 * cols, 0x1357_9bdf);
        let norm_style_values = values(3 * cols, 0x2468_ace0);

        let projection = upload(ctx, &projection_values, Shape::new(vec![rows, cols]));
        let residual = upload(ctx, &residual_values, Shape::new(vec![rows, cols]));
        let gate_style = upload(ctx, &gate_style_values, Shape::new(vec![3 * cols]));
        let norm_style = upload(ctx, &norm_style_values, Shape::new(vec![3 * cols]));

        let out = adaptive_gate_residual_rms_bf16(
            ctx,
            &projection,
            &residual,
            &gate_style,
            &norm_style,
            EPS,
        )
        .expect("native fused residual norm");
        ctx.stream().synchronize().expect("synchronise");
        let got_hidden = read_back(&out.hidden);
        let got_normalized = read_back(&out.normalized);

        // The reference consumes the bf16 values the kernel sees.
        let round = |v: f32| from_bf16_bits(to_bf16_bits(v));
        let projection_values: Vec<f32> = projection_values.iter().map(|v| round(*v)).collect();
        let residual_values: Vec<f32> = residual_values.iter().map(|v| round(*v)).collect();
        let gate_style_values: Vec<f32> = gate_style_values.iter().map(|v| round(*v)).collect();
        let norm_style_values: Vec<f32> = norm_style_values.iter().map(|v| round(*v)).collect();

        let gate_tolerance = gate_for(&got_hidden);
        let norm_tolerance = gate_for(&got_normalized);
        let mut worst_gate = 0.0f32;
        let mut worst_norm = 0.0f32;

        for row in 0..rows {
            // Pass one: the gated residual, rounded to bf16.
            let mut hidden = vec![0.0f32; cols];
            for col in 0..cols {
                let index = row * cols + col;
                hidden[col] = round(
                    residual_values[index] + projection_values[index] * gate_style_values[2 * cols + col],
                );
                let error = (got_hidden[index] - hidden[col]).abs();
                worst_gate = worst_gate.max(error);
                assert!(
                    error <= gate_tolerance,
                    "hidden [{rows}x{cols}] at [{row}][{col}]: {} vs {}, abs {error:.3e}",
                    got_hidden[index],
                    hidden[col]
                );
            }
            // Pass two: the RMS over the rounded row, then the adaLN scale/shift.
            let sum: f32 = hidden.iter().map(|v| v * v).sum();
            let inverse_rms = 1.0 / (sum / cols as f32 + EPS).sqrt();
            for (col, value) in hidden.iter().enumerate() {
                let index = row * cols + col;
                let expected = round(
                    value * inverse_rms * (1.0 + norm_style_values[col]) + norm_style_values[cols + col],
                );
                let error = (got_normalized[index] - expected).abs();
                worst_norm = worst_norm.max(error);
                assert!(
                    error <= norm_tolerance,
                    "normalized [{rows}x{cols}] at [{row}][{col}]: {} vs {}, abs {error:.3e}",
                    got_normalized[index],
                    expected
                );
            }
        }
        println!(
            "  [{rows} x {cols}]  hidden worst {worst_gate:.3e} (gate {gate_tolerance:.3e}), \
             normalized worst {worst_norm:.3e} (gate {norm_tolerance:.3e})"
        );
    }

    /// `bias_residual_rms_norm_bf16` with and without a bias.
    ///
    /// The replay harness covers this operator at its one recorded signature,
    /// and that signature has **no bias** -- which means the `bias != nullptr`
    /// half of the kernel has no reference to be compared against, and the null
    /// check in the kernel is exactly the kind of branch that silently does the
    /// wrong thing. Both are checked here.
    fn check_bias_residual_rms(ctx: &MusaContext, rows: usize, cols: usize, with_bias: bool) {
        use crate::kernels::fused::bias_residual_rms_bf16;

        let projection_values = values(rows * cols, 0x5eed_1234);
        let residual_values = values(rows * cols, 0x0bad_c0de);
        let weight_values = values(cols, 0x1357_9bdf);
        let bias_values = values(cols, 0x2468_ace0);

        let projection = upload(ctx, &projection_values, Shape::new(vec![rows, cols]));
        let residual = upload(ctx, &residual_values, Shape::new(vec![rows, cols]));
        let weight = upload(ctx, &weight_values, Shape::new(vec![cols]));
        let bias = with_bias.then(|| upload(ctx, &bias_values, Shape::new(vec![cols])));

        let out = bias_residual_rms_bf16(
            ctx,
            &projection,
            bias.as_ref(),
            &residual,
            &weight,
            EPS,
        )
        .expect("native fused bias residual rms");
        ctx.stream().synchronize().expect("synchronise");
        let got_hidden = read_back(&out.hidden);
        let got_normalized = read_back(&out.normalized);

        let round = |v: f32| from_bf16_bits(to_bf16_bits(v));
        let projection_values: Vec<f32> = projection_values.iter().map(|v| round(*v)).collect();
        let residual_values: Vec<f32> = residual_values.iter().map(|v| round(*v)).collect();
        let weight_values: Vec<f32> = weight_values.iter().map(|v| round(*v)).collect();
        let bias_values: Vec<f32> = bias_values.iter().map(|v| round(*v)).collect();

        let hidden_tolerance = gate_for(&got_hidden);
        let norm_tolerance = gate_for(&got_normalized);
        for row in 0..rows {
            let base = row * cols;
            let mut hidden = vec![0.0f32; cols];
            for col in 0..cols {
                let mut value = projection_values[base + col] + residual_values[base + col];
                if with_bias {
                    value += bias_values[col];
                }
                hidden[col] = round(value);
                let error = (got_hidden[base + col] - hidden[col]).abs();
                assert!(
                    error <= hidden_tolerance,
                    "hidden [{rows}x{cols}] bias={with_bias} at [{row}][{col}]: {} vs {}, abs {error:.3e}",
                    got_hidden[base + col],
                    hidden[col]
                );
            }
            let sum: f32 = hidden.iter().map(|v| v * v).sum();
            let inverse_rms = 1.0 / (sum / cols as f32 + EPS).sqrt();
            for (col, value) in hidden.iter().enumerate() {
                let expected = round(value * inverse_rms * weight_values[col]);
                let error = (got_normalized[base + col] - expected).abs();
                assert!(
                    error <= norm_tolerance,
                    "normalized [{rows}x{cols}] bias={with_bias} at [{row}][{col}]: {} vs {}, abs {error:.3e}",
                    got_normalized[base + col],
                    expected
                );
            }
        }
        println!("  bias_residual_rms [{rows} x {cols}] bias={with_bias}: ok");
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn bias_residual_rms_norm_handles_the_null_bias() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        for with_bias in [false, true] {
            check_bias_residual_rms(&ctx, 3, 5, with_bias);
            check_bias_residual_rms(&ctx, 2, 256, with_bias);
            check_bias_residual_rms(&ctx, 5, 300, with_bias);
        }
    }

    /// `bias_residual_bf16`: projection plus an optional bias, added into the
    /// residual and rounded once.
    ///
    /// The vision tower's fc2 residual takes this at `[512, 1152]`, a width
    /// that is a multiple of eight and so takes the vectorized branch; a ragged
    /// width is checked too, along with the null-bias branch, which is the one
    /// no recording covers.
    fn check_bias_residual(ctx: &MusaContext, rows: usize, cols: usize, with_bias: bool) {
        let projection_values = values(rows * cols, 0x5eed_1234);
        let residual_values = values(rows * cols, 0x0bad_c0de);
        let bias_values = values(cols, 0x2468_ace0);

        let projection = upload(ctx, &projection_values, Shape::new(vec![rows, cols]));
        let residual = upload(ctx, &residual_values, Shape::new(vec![rows, cols]));
        let bias = with_bias.then(|| upload(ctx, &bias_values, Shape::new(vec![cols])));

        let out = bias_residual_bf16(ctx, &projection, bias.as_ref(), &residual)
            .expect("native fused bias residual");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        let round = |v: f32| from_bf16_bits(to_bf16_bits(v));
        let tolerance = gate_for(&got);
        for row in 0..rows {
            for col in 0..cols {
                let index = row * cols + col;
                let mut value = round(projection_values[index]) + round(residual_values[index]);
                if with_bias {
                    value += round(bias_values[col]);
                }
                let expected = round(value);
                let error = (got[index] - expected).abs();
                assert!(
                    error <= tolerance,
                    "[{rows}x{cols}] bias={with_bias} at [{row}][{col}]: {} vs {expected}, \
                     abs {error:.3e} > {tolerance:.3e}",
                    got[index]
                );
            }
        }
        println!("  bias_residual [{rows} x {cols}] bias={with_bias}: ok");
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn bias_residual_matches_a_reference_with_and_without_a_bias() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        for with_bias in [false, true] {
            check_bias_residual(&ctx, 3, 5, with_bias);
            check_bias_residual(&ctx, 4, 300, with_bias);
            check_bias_residual(&ctx, 4, 1152, with_bias);
        }
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn adaptive_gate_residual_rms_matches_a_reference() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        // A width below one block, one that lands exactly on it, and the real
        // shape's width -- 4304 is not a multiple of 256, so the grid-stride
        // loop takes a ragged last step.
        check(&ctx, 3, 5);
        check(&ctx, 2, 256);
        check(&ctx, 1, 4304);
        check(&ctx, 10, 1024);
    }

    /// `bias_residual_layer_bf16`: add, LayerNorm, both outputs.
    ///
    /// The replay covers `[512, 1152]` with a projection bias and it lands
    /// within 0.27 bf16 ulp of the recording. The **null-bias** branch is not
    /// in that recording -- PI0.5's vision tower always passes one -- and it is
    /// a branch the kernel takes, so it is what this covers.
    fn check_layer(ctx: &MusaContext, rows: usize, cols: usize, bias: bool) {
        let projection_values = values(rows * cols, 0x5eed_1234);
        let residual_values = values(rows * cols, 0x0bad_c0de);
        let norm_weight_values = values(cols, 0x1357_9bdf);
        let norm_bias_values = values(cols, 0x2468_ace0);
        let projection_bias_values = values(cols, 0xfeed_face);

        let projection = upload(ctx, &projection_values, Shape::new(vec![rows, cols]));
        let residual = upload(ctx, &residual_values, Shape::new(vec![rows, cols]));
        let norm_weight = upload(ctx, &norm_weight_values, Shape::new(vec![cols]));
        let norm_bias = upload(ctx, &norm_bias_values, Shape::new(vec![cols]));
        let projection_bias =
            bias.then(|| upload(ctx, &projection_bias_values, Shape::new(vec![cols])));

        let out = bias_residual_layer_bf16(
            ctx,
            &projection,
            projection_bias.as_ref(),
            &residual,
            &norm_weight,
            &norm_bias,
            EPS,
        )
        .expect("native fused layer norm");
        ctx.stream().synchronize().expect("synchronise");
        let got_hidden = read_back(&out.hidden);
        let got_normalized = read_back(&out.normalized);

        let round = |v: f32| from_bf16_bits(to_bf16_bits(v));
        let projection_values: Vec<f32> = projection_values.iter().map(|v| round(*v)).collect();
        let residual_values: Vec<f32> = residual_values.iter().map(|v| round(*v)).collect();
        let norm_weight_values: Vec<f32> = norm_weight_values.iter().map(|v| round(*v)).collect();
        let norm_bias_values: Vec<f32> = norm_bias_values.iter().map(|v| round(*v)).collect();
        let projection_bias_values: Vec<f32> =
            projection_bias_values.iter().map(|v| round(*v)).collect();

        let hidden_tolerance = gate_for(&got_hidden);
        let norm_tolerance = gate_for(&got_normalized);
        let (mut worst_hidden, mut worst_normalized) = (0.0f32, 0.0f32);
        for row in 0..rows {
            let base = row * cols;
            // `hidden` is the sum rounded to bf16 -- and the statistics are
            // taken from *that* value, not from the f32 sum, which is the part
            // the bridge did differently.
            let hidden: Vec<f32> = (0..cols)
                .map(|col| {
                    let mut value = projection_values[base + col] + residual_values[base + col];
                    if bias {
                        value += projection_bias_values[col];
                    }
                    round(value)
                })
                .collect();
            let mean = hidden.iter().sum::<f32>() / cols as f32;
            let variance =
                hidden.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
            let inverse_std = 1.0 / (variance + EPS).sqrt();
            for col in 0..cols {
                let hidden_error = (got_hidden[base + col] - hidden[col]).abs();
                worst_hidden = worst_hidden.max(hidden_error);
                assert!(
                    hidden_error <= hidden_tolerance,
                    "hidden [{rows}x{cols}] bias {bias} at [{row}][{col}]: {} vs {}",
                    got_hidden[base + col],
                    hidden[col]
                );
                let expected = round(
                    (hidden[col] - mean) * inverse_std * norm_weight_values[col]
                        + norm_bias_values[col],
                );
                let error = (got_normalized[base + col] - expected).abs();
                worst_normalized = worst_normalized.max(error);
                assert!(
                    error <= norm_tolerance,
                    "normalized [{rows}x{cols}] bias {bias} at [{row}][{col}]: {} vs {expected}, \
                     abs {error:.3e} > {norm_tolerance:.3e}",
                    got_normalized[base + col]
                );
            }
        }
        println!(
            "  bias_residual_layer [{rows:>4} x {cols:>4}] bias {bias}  worst abs \
             hidden {worst_hidden:.3e} normalized {worst_normalized:.3e}"
        );
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn bias_residual_layer_matches_a_reference_with_and_without_a_projection_bias() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        check_layer(&ctx, 3, 5, true);
        check_layer(&ctx, 3, 5, false);
        check_layer(&ctx, 512, 1152, true);
    }

    /// A residual of another dtype, or from another device, must be refused.
    ///
    /// The shape check cannot see either: F16 and BF16 are both two bytes, so a
    /// same-shaped F16 residual passed every rank and extent test and reached
    /// the launch as a raw pointer the kernel reads as BF16 — a wrong answer
    /// with no error. The device half is the same hazard one card over:
    /// `MusaBuffer::from_tensor` accepts any MUSA device, so nothing else
    /// compares the tensor's device with the context's.
    #[test]
    #[ignore = "needs a MUSA device"]
    fn a_residual_that_is_not_bf16_on_this_device_is_refused() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let (rows, cols) = (3usize, 5usize);
        let projection = upload(
            &ctx,
            &values(rows * cols, 0x5eed_1234),
            Shape::new(vec![rows, cols]),
        );
        let weight = upload(&ctx, &values(cols, 0x0bad_c0de), Shape::new(vec![cols]));
        let style = upload(&ctx, &values(3 * cols, 0x1357_9bdf), Shape::new(vec![3 * cols]));

        // Same shape, same device, F16 instead of BF16.
        let f16_residual = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::F16);
        let assert_named = |error: Error| {
            let message = error.to_string();
            assert!(
                message.contains("residual expects bf16"),
                "the dtype was not named: {message}"
            );
        };
        assert_named(bias_residual_bf16(&ctx, &projection, None, &f16_residual).unwrap_err());
        assert_named(
            bias_residual_rms_bf16(&ctx, &projection, None, &f16_residual, &weight, EPS)
                .unwrap_err(),
        );
        assert_named(
            bias_residual_layer_bf16(
                &ctx,
                &projection,
                None,
                &f16_residual,
                &weight,
                &weight,
                EPS,
            )
            .unwrap_err(),
        );
        assert_named(
            adaptive_gate_residual_rms_bf16(&ctx, &projection, &f16_residual, &style, &style, EPS)
                .unwrap_err(),
        );

        // A host tensor separates the two halves: the dtype is right and the
        // device is not, which is the case a same-shaped check cannot see.
        let on_host = Tensor::zeros(Shape::new(vec![rows, cols]), DType::BF16);
        let message = bias_residual_bf16(&ctx, &projection, None, &on_host)
            .unwrap_err()
            .to_string();
        assert!(message.contains("residual is on"), "{message}");
    }
}

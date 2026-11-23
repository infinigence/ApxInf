//! Normalization operator contracts.

use apxinf_core::{Result, Tensor};

#[cfg(apxinf_musa_available)]
use super::contracts::{checked_i32, tensor_ptr};
use super::contracts::{
    bf16_output, matrix_tensor, require_matrix_bf16, require_output_matrix,
    require_vector_at_least_bf16, require_vector_bf16,
};
use crate::context::MusaContext;
#[cfg(apxinf_musa_available)]
use crate::ffi;
use crate::kernels::not_ported;
#[cfg(apxinf_musa_available)]
use crate::kernels::check_native;
#[cfg(not(apxinf_musa_available))]
use crate::kernels::native_unavailable;

/// BF16 RMS normalization over the last axis.
///
/// `input` is 2-D BF16 `[rows, cols]`, `weight` a BF16 `[cols]` vector.
pub fn rms_bf16(ctx: &MusaContext, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "rms_norm", "input")?;
    let mut output = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    rms_bf16_into(ctx, input, weight, &mut output, eps)?;
    Ok(output)
}

/// Write RMS normalization into a caller-provided `[rows, cols]` BF16 output.
pub fn rms_bf16_into(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
    eps: f32,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "rms_norm", "input")?;
    require_vector_bf16(ctx, weight, cols, "rms_norm", "weight")?;
    require_output_matrix(ctx, output, rows, cols, "rms_norm", "normalized")?;
    launch_rms(ctx, input, weight, output, rows, cols, eps)
}

#[cfg(apxinf_musa_available)]
fn launch_rms(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    let rows = checked_i32("rms_norm", "rows", rows)?;
    let cols = checked_i32("rms_norm", "cols", cols)?;
    // SAFETY: the checks above established shapes, dtypes and one device; both
    // outputs are `[rows, cols]` BF16 allocations owned by the caller's
    // `Tensor`, which outlives the asynchronous launch on `ctx.stream()`.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_rms_norm_bf16(
                tensor_ptr(input)?,
                tensor_ptr(weight)?,
                tensor_ptr(output)?,
                rows,
                cols,
                eps,
                ctx.stream().handle(),
            ),
            "rms_norm",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch_rms(
    _ctx: &MusaContext,
    _input: &Tensor,
    _weight: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
    _eps: f32,
) -> Result<()> {
    Err(native_unavailable("norm::rms_bf16"))
}

/// BF16 layer normalization over the last axis.
///
/// Native: `kernels/custom/normalization.cuh`'s three-pass kernel, one block per
/// row, through `adapters/custom_kernels.mu`.
pub fn layer_bf16(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "layer_norm", "input")?;
    let mut output = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    layer_bf16_into(ctx, input, weight, bias, &mut output, eps)?;
    Ok(output)
}

/// Write layer normalization into a caller-provided `[rows, cols]` BF16 output.
pub fn layer_bf16_into(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    output: &mut Tensor,
    eps: f32,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "layer_norm", "input")?;
    require_vector_bf16(ctx, weight, cols, "layer_norm", "weight")?;
    require_vector_bf16(ctx, bias, cols, "layer_norm", "bias")?;
    require_output_matrix(ctx, output, rows, cols, "layer_norm", "normalized")?;
    launch_layer(ctx, input, weight, bias, output, rows, cols, eps)
}

/// The launch, split by whether this build has a MUSA toolkit.
///
/// # Numerics
/// The kernel reproduces the bridge's arithmetic -- mean and variance over the
/// row in f32, then the affine transform, rounded once -- but not its summation
/// order: the bridge left the reduction to torch's `mean`, this one walks the
/// row per thread and combines across a warp/block tree. The two do not add the
/// same numbers in the same sequence, so this operator is gated rather than
/// expected to be bit-identical, as `norm.rms_bf16` is.
#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_layer(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    output: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    let rows = checked_i32("layer_norm", "rows", rows)?;
    let cols = checked_i32("layer_norm", "cols", cols)?;
    // SAFETY: the checks above established shapes, dtypes and one device, and
    // all four buffers are owned by caller `Tensor`s that outlive the
    // asynchronous launch on `ctx.stream()`.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_layer_norm_bf16(
                tensor_ptr(input)?,
                tensor_ptr(weight)?,
                tensor_ptr(bias)?,
                tensor_ptr(output)?,
                rows,
                cols,
                eps,
                ctx.stream().handle(),
            ),
            "layer_norm",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch_layer(
    _ctx: &MusaContext,
    _input: &Tensor,
    _weight: &Tensor,
    _bias: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
    _eps: f32,
) -> Result<()> {
    Err(native_unavailable("norm::layer_bf16"))
}

/// Adaptive (adaLN) BF16 RMS normalization.
///
/// `style` is a single BF16 `[3 * cols]` vector holding gate, shift and scale
/// in that order.
pub fn adaptive_rms_bf16(
    ctx: &MusaContext,
    input: &Tensor,
    style: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "adaptive_rms_norm", "input")?;
    let mut output = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    adaptive_rms_bf16_into(ctx, input, style, &mut output, eps)?;
    Ok(output)
}

/// Write adaptive RMS normalization into a caller-provided output.
///
/// The kernel reads `style` as scale then shift, so the first `2 * cols`
/// entries are the contract; a longer vector is accepted because PI0.5's
/// modulation carries a third gate segment that this operator does not consume.
pub fn adaptive_rms_bf16_into(
    ctx: &MusaContext,
    input: &Tensor,
    style: &Tensor,
    output: &mut Tensor,
    eps: f32,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "adaptive_rms_norm", "input")?;
    require_vector_at_least_bf16(ctx, style, 2 * cols, "adaptive_rms_norm", "style")?;
    require_output_matrix(ctx, output, rows, cols, "adaptive_rms_norm", "normalized")?;
    launch_adaptive_rms(ctx, input, style, output, rows, cols, eps)
}

/// The launch, split by whether this build has a MUSA toolkit.
///
/// # Numerics
/// The kernel sums squares with a per-thread loop plus a two-level shuffle
/// tree, while a torch `mean` over the last axis does not.
/// The two do not accumulate in the same order, so this operator is gated
/// rather than expected to be bit-identical -- unlike the purely elementwise
/// ones. `hidden`-style outputs that skip the reduction do come out equal.
#[cfg(apxinf_musa_available)]
fn launch_adaptive_rms(
    ctx: &MusaContext,
    input: &Tensor,
    style: &Tensor,
    output: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    let rows = checked_i32("adaptive_rms_norm", "rows", rows)?;
    let cols = checked_i32("adaptive_rms_norm", "cols", cols)?;
    // SAFETY: as in `launch_rms` -- shapes, dtypes and one device are already
    // established, and the operands outlive the asynchronous launch.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_ada_rms_norm_bf16(
                tensor_ptr(input)?,
                tensor_ptr(style)?,
                tensor_ptr(output)?,
                rows,
                cols,
                eps,
                ctx.stream().handle(),
            ),
            "adaptive_rms_norm",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch_adaptive_rms(
    _ctx: &MusaContext,
    _input: &Tensor,
    _style: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
    _eps: f32,
) -> Result<()> {
    Err(native_unavailable("norm::adaptive_rms_bf16"))
}

// ── FP8-output normalization (not ported) ────────────────────────────────
//
// The `*_quant_f16_e4m3` family writes E4M3 rather than BF16 and belongs to
// the `fp8_static` variant, outside this port's scope. Signatures mirror
// `apxinf-cuda/src/kernels/norm.rs`; note those CUDA functions do **not**
// validate `eps`/`scale` finiteness (their BF16 siblings do), so a port should
// not add the check and then call itself faithful.

/// RMSNorm whose output is quantized to E4M3.
pub fn rms_quant_f16_e4m3(
    _ctx: &MusaContext,
    _input: &Tensor,
    _weight: &Tensor,
    _eps: f32,
    _scale: f32,
) -> Result<Tensor> {
    Err(not_ported("norm::rms_quant_f16_e4m3"))
}

/// LayerNorm whose output is quantized to E4M3.
pub fn layer_quant_f16_e4m3(
    _ctx: &MusaContext,
    _input: &Tensor,
    _weight: &Tensor,
    _bias: &Tensor,
    _eps: f32,
    _scale: f32,
) -> Result<Tensor> {
    Err(not_ported("norm::layer_quant_f16_e4m3"))
}

/// Adaptive RMSNorm whose output is quantized to E4M3.
pub fn adaptive_rms_quant_f16_e4m3(
    _ctx: &MusaContext,
    _input: &Tensor,
    _style: &Tensor,
    _eps: f32,
    _scale: f32,
) -> Result<Tensor> {
    Err(not_ported("norm::adaptive_rms_quant_f16_e4m3"))
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    //! Device tests for the normalization kernels.
    //!
    //! The replay harness covers `norm.rms_bf16` at its one recorded shape,
    //! `[522, 2048]` -- which takes the **vectorized** reduction, because
    //! `rows >= 16 && cols > 128 && cols % 4 == 0`. The block-reduction branch
    //! is therefore unreached by that evidence, and it is the one the small and
    //! ragged shapes use. These tests cover both, plus `ada_rms_norm`'s scale
    //! and shift.
    //!
    //! Needs a MUSA device, so they are `#[ignore]`d:
    //! `cargo test -p apxinf-musa --lib -- --ignored --nocapture`.

    use apxinf_core::Shape;

    use crate::context::MusaContext;
    use crate::kernels::norm::{adaptive_rms_bf16, layer_bf16, rms_bf16};
    use crate::kernels::test_support::{
        from_bf16_bits, gate_for, read_back, to_bf16_bits, upload, values,
    };

    const EPS: f32 = 1e-6;

    fn check_rms(ctx: &MusaContext, rows: usize, cols: usize) {
        let input_values = values(rows * cols, 0x5eed_1234);
        let weight_values = values(cols, 0x0bad_c0de);
        let input = upload(ctx, &input_values, Shape::new(vec![rows, cols]));
        let weight = upload(ctx, &weight_values, Shape::new(vec![cols]));

        let out = rms_bf16(ctx, &input, &weight, EPS).expect("native rms_norm");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        let round = |v: f32| from_bf16_bits(to_bf16_bits(v));
        let input_values: Vec<f32> = input_values.iter().map(|v| round(*v)).collect();
        let weight_values: Vec<f32> = weight_values.iter().map(|v| round(*v)).collect();

        let tolerance = gate_for(&got);
        let mut worst = 0.0f32;
        for row in 0..rows {
            let base = row * cols;
            let sum: f32 = (0..cols).map(|c| input_values[base + c] * input_values[base + c]).sum();
            let inverse_rms = 1.0 / (sum / cols as f32 + EPS).sqrt();
            for col in 0..cols {
                let expected = round(input_values[base + col] * inverse_rms * weight_values[col]);
                let error = (got[base + col] - expected).abs();
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "rms [{rows}x{cols}] at [{row}][{col}]: {} vs {expected}, abs {error:.3e} > {tolerance:.3e}",
                    got[base + col]
                );
            }
        }
        println!("  rms_norm [{rows:>4} x {cols:>4}]  worst abs {worst:.3e} (gate {tolerance:.3e})");
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn rms_norm_matches_a_reference_across_shapes() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        // One block reduction now covers every shape: the single-warp
        // `rms_vector_square_mean_bf16` path was 89% of this kernel's runtime
        // and 6.4x slower than the same reduction over all threads, and the
        // two summed in different orders only because CUDA's dispatch did.
        // `[522, 2048]` is the recorded shape; the ragged ones pin the
        // widths where the old dispatch switched.
        for (rows, cols) in [(3, 5), (16, 130), (16, 256), (522, 2048), (7, 2047)] {
            check_rms(&ctx, rows, cols);
        }
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn adaptive_rms_norm_applies_scale_and_shift() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        check_adaptive_rms(&ctx, 7, 300);
        // `[10, 1024]` is the action expert's shape and a multiple of the
        // block, so it takes the register-cached branch; 300 above does not.
        check_adaptive_rms(&ctx, 10, 1024);
    }

    fn check_adaptive_rms(ctx: &MusaContext, rows: usize, cols: usize) {
        let input_values = values(rows * cols, 0x2468_ace0);
        let style_values = values(3 * cols, 0x1357_9bdf);
        let input = upload(ctx, &input_values, Shape::new(vec![rows, cols]));
        let style = upload(ctx, &style_values, Shape::new(vec![3 * cols]));

        let out = adaptive_rms_bf16(ctx, &input, &style, EPS).expect("native ada_rms_norm");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        let round = |v: f32| from_bf16_bits(to_bf16_bits(v));
        let input_values: Vec<f32> = input_values.iter().map(|v| round(*v)).collect();
        let style_values: Vec<f32> = style_values.iter().map(|v| round(*v)).collect();

        let tolerance = gate_for(&got);
        let mut worst = 0.0f32;
        for row in 0..rows {
            let base = row * cols;
            let sum: f32 = (0..cols).map(|c| input_values[base + c] * input_values[base + c]).sum();
            let inverse_rms = 1.0 / (sum / cols as f32 + EPS).sqrt();
            for col in 0..cols {
                let normalized = input_values[base + col] * inverse_rms;
                let expected = round(normalized * (1.0 + style_values[col]) + style_values[cols + col]);
                let error = (got[base + col] - expected).abs();
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "ada_rms [{rows}x{cols}] at [{row}][{col}]: {} vs {expected}, abs {error:.3e}",
                    got[base + col]
                );
            }
        }
        println!("  ada_rms_norm [{rows} x {cols}]  worst abs {worst:.3e} (gate {tolerance:.3e})");
    }

    /// `layer_norm` over a row, with a weight and a shift.
    ///
    /// The replay covers `[512, 1152]`; this adds a ragged shape small enough
    /// that most threads of a block find no column at all, which is where the
    /// three-pass structure (sum, then squared deviations, then the transform)
    /// has to keep its barriers straight.
    fn check_layer(ctx: &MusaContext, rows: usize, cols: usize) {
        let input_values = values(rows * cols, 0x2468_ace0);
        let weight_values = values(cols, 0x0bad_c0de);
        let bias_values = values(cols, 0x1357_9bdf);
        let input = upload(ctx, &input_values, Shape::new(vec![rows, cols]));
        let weight = upload(ctx, &weight_values, Shape::new(vec![cols]));
        let bias = upload(ctx, &bias_values, Shape::new(vec![cols]));

        let out = layer_bf16(ctx, &input, &weight, &bias, EPS).expect("native layer_norm");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        let round = |v: f32| from_bf16_bits(to_bf16_bits(v));
        let input_values: Vec<f32> = input_values.iter().map(|v| round(*v)).collect();
        let weight_values: Vec<f32> = weight_values.iter().map(|v| round(*v)).collect();
        let bias_values: Vec<f32> = bias_values.iter().map(|v| round(*v)).collect();

        let tolerance = gate_for(&got);
        let mut worst = 0.0f32;
        for row in 0..rows {
            let base = row * cols;
            let row_values = &input_values[base..base + cols];
            let mean = row_values.iter().sum::<f32>() / cols as f32;
            let variance =
                row_values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
            let inverse_std = 1.0 / (variance + EPS).sqrt();
            for col in 0..cols {
                let expected =
                    round((row_values[col] - mean) * inverse_std * weight_values[col] + bias_values[col]);
                let error = (got[base + col] - expected).abs();
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "layer [{rows}x{cols}] at [{row}][{col}]: {} vs {expected}, \
                     abs {error:.3e} > {tolerance:.3e}",
                    got[base + col]
                );
            }
        }
        // `[512, 1152]` runs at a noticeable fraction of the gate, and it is
        // the reference's fault rather than the kernel's: the sum above is
        // sequential while the kernel splits the row across 256 threads and
        // combines a warp tree, so the mean and variance differ in the last f32
        // bits and occasionally that moves a bf16 rounding. The replay against
        // the *recorded* output -- torch's own reduction, not this one -- is
        // what bounds the kernel itself, and it lands at 7.6e-6 against a 9.7e-2
        // gate.
        println!("  layer_norm [{rows:>4} x {cols:>4}]  worst abs {worst:.3e} (gate {tolerance:.3e})");
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn layer_norm_matches_a_reference() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        check_layer(&ctx, 3, 5);
        check_layer(&ctx, 16, 130);
        check_layer(&ctx, 512, 1152);
    }
}

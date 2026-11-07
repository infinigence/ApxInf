//! Activation operator contracts.

use apxinf_core::{Error, Result, Tensor};

use super::contracts::{
    bf16_output, matrix_tensor, require_matrix_bf16, require_optional_vector_bf16,
    require_output_matrix,
};
#[cfg(apxinf_musa_available)]
use super::contracts::{checked_i32, optional_tensor_ptr, tensor_ptr};
use crate::context::MusaContext;
#[cfg(apxinf_musa_available)]
use crate::error::musa_error;
#[cfg(apxinf_musa_available)]
use crate::ffi;
#[cfg(apxinf_musa_available)]
use crate::kernels::check_native;
#[cfg(not(apxinf_musa_available))]
use crate::kernels::native_unavailable;
use crate::kernels::not_ported;

/// Activation codes shared with the CUDA backend's kernel ABI: the same
/// integer selects the same nonlinearity, so a tactic or a diagnosable name
/// never has to be renumbered across backends.
pub(super) const ACTIVATION_NONE: i32 = 0;
pub(super) const ACTIVATION_GELU: i32 = 1;
pub(super) const ACTIVATION_SILU: i32 = 2;

/// Bias plus the activation selected by `activation`.
///
/// Native: one elementwise kernel on the device, `kernels/custom/activation.cuh`
/// through `adapters/custom_kernels.mu`. All three of `bias_gelu_bf16`,
/// `bias_silu_bf16` and `elementwise::bias_bf16` land here, because they are one
/// CUDA entry point three ways — the bridge recorded them as three `kind`
/// branches of a single signature for the same reason.
pub(super) fn bias_activation(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
    activation: i32,
) -> Result<Tensor> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "bias activation", "input")?;
    let mut output = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    bias_activation_into(ctx, input, value, &mut output, activation)?;
    Ok(output)
}

pub(super) fn bias_activation_into(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
    output: &mut Tensor,
    activation: i32,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "bias activation", "input")?;
    require_optional_vector_bf16(ctx, value, cols, "bias activation", "bias")?;
    require_output_matrix(ctx, output, rows, cols, "bias activation", "out")?;
    launch(ctx, input, value, output, rows, cols, activation)
}

/// The launch, split by whether this build has a MUSA toolkit.
///
/// # Numerics
/// The kernel computes in f32 and rounds once on the way out, which is what the
/// eager bridge did — it upcast both operands, added and activated in f32, and
/// let the write convert back. GELU is the tanh approximation rather than erf,
/// matching both `_gelu_tanh` in the bridge and `gelu_tanh` in
/// `kernels/custom/math.cuh`; the three spellings have to move together or the
/// recorded reference tensors stop describing this kernel.
///
/// # Stream
/// `ctx.stream()`, as everywhere else in this backend. That is ordered against
/// the torch operators around it — see the streaming note in the eager bridge
/// that recorded the reference (it is not part of this tree) — so no
/// synchronisation is needed here.
#[cfg(apxinf_musa_available)]
fn launch(
    ctx: &MusaContext,
    input: &Tensor,
    bias: Option<&Tensor>,
    output: &Tensor,
    rows: usize,
    cols: usize,
    activation: i32,
) -> Result<()> {
    let rows = checked_i32("bias_activation", "rows", rows)?;
    let cols = checked_i32("bias_activation", "cols", cols)?;
    let bias_pointer = optional_tensor_ptr(bias)?;
    // SAFETY: `require_matrix_bf16` and `require_vector_bf16` have established
    // that the input and the optional bias are contiguous BF16 tensors on this
    // context's device, and `matrix_tensor` that the output is a `[rows, cols]`
    // BF16 allocation there. `tensor_ptr` borrows storage the caller's `Tensor`s
    // keep alive across the call, and the null bias is the kernel's documented
    // "no bias" case. The launch is asynchronous on `ctx.stream()`.
    unsafe {
        ffi::check_runtime(ffi::custom::apxinf_musa_bias_activation_bf16(
            tensor_ptr(input)?,
            bias_pointer,
            tensor_ptr(output)?,
            rows,
            cols,
            activation,
            ctx.stream().handle(),
        ))
        .map_err(musa_error)?;
    }
    Ok(())
}

#[cfg(not(apxinf_musa_available))]
fn launch(
    _ctx: &MusaContext,
    _input: &Tensor,
    _bias: Option<&Tensor>,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
    _activation: i32,
) -> Result<()> {
    Err(native_unavailable("activation::bias_activation_bf16"))
}

/// Bias plus GELU.
pub fn bias_gelu_bf16(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
) -> Result<Tensor> {
    bias_activation(ctx, input, value, ACTIVATION_GELU)
}

/// Write bias plus GELU into a caller-provided output.
pub fn bias_gelu_bf16_into(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    bias_activation_into(ctx, input, value, output, ACTIVATION_GELU)
}

/// Bias plus SiLU.
pub fn bias_silu_bf16(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
) -> Result<Tensor> {
    bias_activation(ctx, input, value, ACTIVATION_SILU)
}

/// Write bias plus SiLU into a caller-provided output.
pub fn bias_silu_bf16_into(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    bias_activation_into(ctx, input, value, output, ACTIVATION_SILU)
}

/// Gated GELU: splits `[rows, 2 * half]` into gate and up halves.
///
/// Native: `kernels/custom/activation.cuh`'s grid-stride kernel, through
/// `adapters/custom_kernels.mu`. PI0.5 never calls this on its own -- its gate/up
/// projection is fused into the GEMM -- but the fused operator is served by
/// calling this one, so it is on the model's path through
/// [`crate::kernels::gemm::bf16_geglu_fused`].
pub fn geglu_bf16(ctx: &MusaContext, gate_up: &Tensor) -> Result<Tensor> {
    let (rows, cols) = require_matrix_bf16(ctx, gate_up, "geglu", "gate_up")?;
    if cols % 2 != 0 {
        return Err(Error::Other(format!(
            "MUSA geglu expects an even inner dimension, got {cols}"
        )));
    }
    let half = cols / 2;
    let mut output = matrix_tensor(ctx, rows, half, bf16_output(ctx, rows, half)?);
    geglu_bf16_into(ctx, gate_up, &mut output)?;
    Ok(output)
}

/// Write gated GELU into a caller-provided `[rows, cols / 2]` output.
pub fn geglu_bf16_into(
    ctx: &MusaContext,
    gate_up: &Tensor,
    output: &mut Tensor,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, gate_up, "geglu", "gate_up")?;
    if cols % 2 != 0 {
        return Err(Error::Other(format!(
            "MUSA geglu expects an even inner dimension, got {cols}"
        )));
    }
    let half = cols / 2;
    require_output_matrix(ctx, output, rows, half, "geglu", "out")?;
    launch_geglu(ctx, gate_up, output, rows, half)
}

/// The launch, split by whether this build has a MUSA toolkit.
#[cfg(apxinf_musa_available)]
fn launch_geglu(
    ctx: &MusaContext,
    gate_up: &Tensor,
    output: &Tensor,
    rows: usize,
    inner: usize,
) -> Result<()> {
    let rows = checked_i32("geglu", "rows", rows)?;
    let inner = checked_i32("geglu", "inner", inner)?;
    // SAFETY: `require_matrix_bf16` established `gate_up` as `[rows, 2 * inner]`
    // BF16 on this device, `matrix_tensor` that the output is `[rows, inner]`
    // there, and both outlive the asynchronous launch on `ctx.stream()`.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_geglu_bf16(
                tensor_ptr(gate_up)?,
                tensor_ptr(output)?,
                rows,
                inner,
                ctx.stream().handle(),
            ),
            "geglu",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch_geglu(
    _ctx: &MusaContext,
    _gate_up: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _inner: usize,
) -> Result<()> {
    Err(native_unavailable("activation::geglu_bf16"))
}

// ── FP16 and FP8-output activation (not ported) ──────────────────────────

/// Bias plus SiLU in FP16.
pub fn bias_silu_f16(
    _ctx: &MusaContext,
    _input: &Tensor,
    _bias: Option<&Tensor>,
) -> Result<Tensor> {
    Err(not_ported("activation::bias_silu_f16"))
}

/// Bias plus SiLU, output quantized to E4M3.
pub fn bias_silu_quant_f16_e4m3(
    _ctx: &MusaContext,
    _input: &Tensor,
    _bias: Option<&Tensor>,
    _scale: f32,
) -> Result<Tensor> {
    Err(not_ported("activation::bias_silu_quant_f16_e4m3"))
}

/// Gated GELU whose output is quantized to E4M3.
pub fn geglu_quant_f16_e4m3(
    _ctx: &MusaContext,
    _gate_up: &Tensor,
    _scale: f32,
) -> Result<Tensor> {
    Err(not_ported("activation::geglu_quant_f16_e4m3"))
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    //! Device tests for the bias/activation kernel.
    //!
    //! The replay harness already compares this
    //! operator against recorded reference tensors on six real shapes covering
    //! all three activation codes, and it is the stronger evidence. What it
    //! cannot reach is the **bias-less** path: every recorded call carried a
    //! bias, because that is what the model does, so `bias == nullptr` — a real
    //! branch in the kernel, reached by `elementwise::bias_bf16(ctx, x, None)`
    //! — has no reference to be compared against. That is what this checks.
    //!
    //! Needs a MUSA device, so it is `#[ignore]`d:
    //! `cargo test -p apxinf-musa --lib -- --ignored --nocapture`.

    use apxinf_core::Shape;

    use crate::context::MusaContext;
    use crate::kernels::activation::{bias_gelu_bf16, bias_silu_bf16, geglu_bf16};
    use crate::kernels::elementwise::bias_bf16;
    use crate::kernels::test_support::{from_bf16_bits, gate_for, read_back, to_bf16_bits, upload, values};

    /// The kernel's own spellings, in f32, so the reference differs only where
    /// `tanhf`/`expf` differ between the host libm and the device.
    fn reference(kind: i32, input: &[f32], bias: Option<f32>) -> Vec<f32> {
        input
            .iter()
            .map(|value| {
                let mut x = *value;
                if let Some(bias) = bias {
                    x += bias;
                }
                match kind {
                    1 => {
                        let alpha = 0.7978845608028654f32;
                        0.5 * x * (1.0 + (alpha * (x + 0.044715 * x * x * x)).tanh())
                    }
                    2 => x / (1.0 + (-x).exp()),
                    _ => x,
                }
            })
            .map(|v| from_bf16_bits(to_bf16_bits(v)))
            .collect()
    }

    fn check(ctx: &MusaContext, rows: usize, cols: usize, kind: i32) {
        let input_values = values(rows * cols, 0x5eed_1234);
        let bias_values = values(cols, 0x0bad_c0de);
        let input = upload(ctx, &input_values, Shape::new(vec![rows, cols]));
        let bias = upload(ctx, &bias_values, Shape::new(vec![cols]));

        let out = match kind {
            0 => bias_bf16(ctx, &input, Some(&bias)),
            1 => bias_gelu_bf16(ctx, &input, Some(&bias)),
            _ => bias_silu_bf16(ctx, &input, Some(&bias)),
        }
        .expect("native activation");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        // The reference consumes the bf16-rounded inputs the kernel sees.
        let rounded_input: Vec<f32> = input_values
            .iter()
            .map(|v| from_bf16_bits(to_bf16_bits(*v)))
            .collect();
        let rounded_bias: Vec<f32> = bias_values
            .iter()
            .map(|v| from_bf16_bits(to_bf16_bits(*v)))
            .collect();

        let tolerance = gate_for(&got);
        let mut worst = 0.0f32;
        for row in 0..rows {
            for col in 0..cols {
                let index = row * cols + col;
                // The bias goes in *before* the activation, which is what the
                // kernel does; getting this order wrong is a mistake worth
                // having caught here rather than in a diff downstream.
                let expected =
                    reference(kind, &rounded_input[index..index + 1], Some(rounded_bias[col]))[0];
                let error = (got[index] - expected).abs();
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "kind {kind} [{rows}x{cols}] at [{row}][{col}]: {} vs {expected}, \
                     abs {error:.3e} > {tolerance:.3e}",
                    got[index]
                );
            }
        }
        println!("  kind {kind}  [{rows} x {cols}]  worst abs {worst:.3e} (gate {tolerance:.3e})");
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn bias_activation_matches_a_reference_for_every_kind() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        for kind in [0, 1, 2] {
            check(&ctx, 3, 5, kind);
        }
        // A row count above one and a column count that is not a multiple of
        // the block size, so the grid-stride loop takes more than one step.
        check(&ctx, 257, 300, 1);
        // A width that is a multiple of eight takes the vectorized branch,
        // which is the one the vision MLP's `[512, 4304]` GELU runs on.
        for kind in [0, 1, 2] {
            check(&ctx, 3, 320, kind);
        }
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn bias_activation_without_a_bias_copies_the_input() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let input_values = values(4 * 32, 0x1357_9bdf);
        let input = upload(&ctx, &input_values, Shape::new(vec![4, 32]));

        let out = bias_bf16(&ctx, &input, None).expect("bespoke bias-less activation");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        // Code 0 with a null bias is the identity on bf16 values, which makes
        // this the one case in the whole operator where bit equality is the
        // right assertion rather than a gate.
        let expected: Vec<f32> = input_values
            .iter()
            .map(|v| from_bf16_bits(to_bf16_bits(*v)))
            .collect();
        assert_eq!(got, expected, "bias-less activation must copy the input");
        println!("  bias-less identity over {} elements: bit-identical", got.len());
    }

    /// `geglu_bf16`: `gelu_tanh(gate) * up`, gate from the first half.
    ///
    /// The replay covers the two shapes the split gate/up projection produces
    /// (`[522, 32768]` and `[10, 8192]`) and both come out **bit-identical**,
    /// which is stronger evidence than this. What it does not reach is a shape
    /// where the row/column split of a flat index does not line up with the
    /// block, or an odd inner width, which the facade refuses.
    fn check_geglu(ctx: &MusaContext, rows: usize, cols: usize) {
        let inner = cols / 2;
        let gate_up_values = values(rows * cols, 0x5eed_1234);
        let gate_up = upload(ctx, &gate_up_values, Shape::new(vec![rows, cols]));

        let out = geglu_bf16(ctx, &gate_up).expect("native geglu");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        let round = |v: f32| from_bf16_bits(to_bf16_bits(v));
        let alpha = 0.7978845608028654f32;
        let tolerance = gate_for(&got);
        let mut worst = 0.0f32;
        for row in 0..rows {
            for col in 0..inner {
                let index = row * cols + col;
                let gate = round(gate_up_values[index]);
                let up = round(gate_up_values[index + inner]);
                let gelu = 0.5 * gate * (1.0 + (alpha * (gate + 0.044715 * gate * gate * gate)).tanh());
                let expected = round(gelu * up);
                let error = (got[row * inner + col] - expected).abs();
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "geglu [{rows}x{cols}] at [{row}][{col}]: {} vs {expected}, \
                     abs {error:.3e} > {tolerance:.3e}",
                    got[row * inner + col]
                );
            }
        }
        println!("  geglu [{rows:>4} x {cols:>5}]  worst abs {worst:.3e} (gate {tolerance:.3e})");
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn geglu_matches_a_reference() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        // An inner width of 5 is not a multiple of the block size and neither
        // is 151, so both take a ragged last grid-stride step.
        check_geglu(&ctx, 3, 10);
        check_geglu(&ctx, 7, 302);
        check_geglu(&ctx, 522, 32768);
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn geglu_refuses_an_odd_inner_dimension() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let gate_up = upload(&ctx, &values(3 * 5, 0x1357_9bdf), Shape::new(vec![3, 5]));
        let error = geglu_bf16(&ctx, &gate_up).expect_err("an odd width has no gate/up split");
        assert!(error.to_string().contains("even inner dimension"), "{error}");
        println!("  geglu refused: {error}");
    }
}

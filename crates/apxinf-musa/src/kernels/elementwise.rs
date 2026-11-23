//! Elementwise and row-manipulation operator contracts.

use apxinf_core::{DType, Error, Result, Tensor};

use super::activation::ACTIVATION_NONE;
use super::contracts::{
    bf16_output, matrix_tensor, output_tensor, require_dtype_on, require_matching_bf16,
    require_matrix_bf16, require_output_matrix, require_output_tensor,
};
#[cfg(apxinf_musa_available)]
use super::contracts::{checked_i32, checked_i64, tensor_ptr};
use crate::context::MusaContext;
#[cfg(apxinf_musa_available)]
use crate::ffi;
use crate::kernels::not_ported;
#[cfg(apxinf_musa_available)]
use crate::kernels::check_native;
#[cfg(not(apxinf_musa_available))]
use crate::kernels::native_unavailable;

/// Add a bias vector to every row (or copy the input when there is none).
///
/// Shares the activation kernel with [`super::activation`], selected by the
/// `none` code, exactly as the CUDA facade does.
pub fn bias_bf16(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
) -> Result<Tensor> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "bias", "input")?;
    let mut output = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    bias_bf16_into(ctx, input, value, &mut output)?;
    Ok(output)
}

/// Write the bias add into a caller-provided output.
pub fn bias_bf16_into(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    super::activation::bias_activation_into(ctx, input, value, output, ACTIVATION_NONE)
}

/// Concatenate two 2-D BF16 tensors along rows.
pub fn concat_rows_bf16(ctx: &MusaContext, first: &Tensor, second: &Tensor) -> Result<Tensor> {
    let (first_rows, first_cols) = require_matrix_bf16(ctx, first, "concat_rows", "first")?;
    let (second_rows, second_cols) = require_matrix_bf16(ctx, second, "concat_rows", "second")?;
    if first_cols != second_cols {
        return Err(Error::Other(format!(
            "MUSA concat_rows needs matching inner dimensions, got {first_cols} and {second_cols}"
        )));
    }
    let rows = first_rows
        .checked_add(second_rows)
        .ok_or_else(|| Error::Other("MUSA concat_rows row count overflow".into()))?;
    let output = matrix_tensor(ctx, rows, first_cols, bf16_output(ctx, rows, first_cols)?);
    launch_concat(ctx, first, second, &output, first_rows, second_rows, first_cols)?;
    Ok(output)
}

#[cfg(apxinf_musa_available)]
fn launch_concat(
    ctx: &MusaContext,
    first: &Tensor,
    second: &Tensor,
    output: &Tensor,
    first_rows: usize,
    second_rows: usize,
    cols: usize,
) -> Result<()> {
    // SAFETY: shapes, dtypes and one device are established above, and every
    // buffer is owned by a caller `Tensor` that outlives the launch.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_concat_rows_bf16(
                tensor_ptr(first)?,
                tensor_ptr(second)?,
                tensor_ptr(output)?,
                checked_i32("concat_rows", "first_rows", first_rows)?,
                checked_i32("concat_rows", "second_rows", second_rows)?,
                checked_i32("concat_rows", "cols", cols)?,
                ctx.stream().handle(),
            ),
            "concat_rows",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch_concat(
    _ctx: &MusaContext,
    _first: &Tensor,
    _second: &Tensor,
    _output: &Tensor,
    _first_rows: usize,
    _second_rows: usize,
    _cols: usize,
) -> Result<()> {
    Err(native_unavailable("elementwise::concat_rows_bf16"))
}

/// One Euler step of the flow sampler: `state + dt * velocity`.
///
/// The result is a **new buffer**, never written over `state`: the denoising
/// loop keeps every step's state alive in the graph, which is part of why the
/// workspace is sized the way it is.
pub fn euler_update_bf16(
    ctx: &MusaContext,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    // The kernel is element-count based and rank-agnostic; the output keeps the
    // input's shape.
    let mut output = output_tensor(
        ctx,
        state.shape().clone(),
        DType::BF16,
        bf16_output(ctx, state.shape().numel(), 1)?,
    );
    euler_update_bf16_into(ctx, state, velocity, &mut output, dt)?;
    Ok(output)
}

/// Write one Euler step into a caller-provided output of the state's shape.
pub fn euler_update_bf16_into(
    ctx: &MusaContext,
    state: &Tensor,
    velocity: &Tensor,
    output: &mut Tensor,
    dt: f32,
) -> Result<()> {
    require_dtype_on(ctx, state, DType::BF16, "euler_update", "state")?;
    require_matching_bf16(ctx, "euler_update", "velocity", state, velocity)?;
    require_output_tensor(ctx, output, state.shape().dims(), "euler_update", "out")?;
    let count = state.shape().numel();
    launch_euler(ctx, state, velocity, output, count, dt)
}

/// The launch, split by whether this build has a MUSA toolkit.
///
/// # Numerics
/// The kernel writes `state + dt * velocity` with `__fmul_rn`/`__fadd_rn` rather
/// than letting the compiler contract the pair into an FMA. The bridge computed
/// the same expression in torch, which does not fuse it, so a contracted kernel
/// would differ in the last bit of every element.
#[cfg(apxinf_musa_available)]
fn launch_euler(
    ctx: &MusaContext,
    state: &Tensor,
    velocity: &Tensor,
    output: &Tensor,
    count: usize,
    dt: f32,
) -> Result<()> {
    let count = checked_i64("euler_update", "count", count)?;
    // SAFETY: the facade has already checked dtypes and shapes, `output` is a
    // BF16 allocation of the same element count on the same device, and all
    // three outlive the asynchronous launch.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_euler_update_bf16(
                tensor_ptr(state)?,
                tensor_ptr(velocity)?,
                tensor_ptr(output)?,
                count,
                dt,
                ctx.stream().handle(),
            ),
            "euler_update",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch_euler(
    _ctx: &MusaContext,
    _state: &Tensor,
    _velocity: &Tensor,
    _output: &Tensor,
    _count: usize,
    _dt: f32,
) -> Result<()> {
    Err(native_unavailable("elementwise::euler_update_bf16"))
}

/// Multiply a BF16 matrix by a per-channel f32 scale vector.
///
/// The activation half of the W8A8 per-channel equalization: the projection's
/// weight was multiplied by `c` at pack time, so the activation is divided by
/// it here and the product is unchanged while the per-row amax escapes the
/// outlier channels.
pub fn mul_channels_bf16(
    ctx: &MusaContext,
    input: &Tensor,
    channel_scale: &Tensor,
) -> Result<Tensor> {
    let (rows, cols) = require_mul_channels(ctx, input, channel_scale)?;
    let output = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    launch_mul_channels(ctx, input, channel_scale, &output, rows, cols)?;
    Ok(output)
}

/// [`mul_channels_bf16`] into a caller-provided output.
pub fn mul_channels_bf16_into(
    ctx: &MusaContext,
    input: &Tensor,
    channel_scale: &Tensor,
    output: &mut Tensor,
) -> Result<()> {
    let (rows, cols) = require_mul_channels(ctx, input, channel_scale)?;
    require_output_matrix(ctx, output, rows, cols, "mul_channels", "output")?;
    launch_mul_channels(ctx, input, channel_scale, output, rows, cols)
}

fn require_mul_channels(
    ctx: &MusaContext,
    input: &Tensor,
    channel_scale: &Tensor,
) -> Result<(usize, usize)> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "mul_channels", "input")?;
    require_dtype_on(ctx, channel_scale, DType::F32, "mul_channels", "channel_scale")?;
    if channel_scale.shape().dims() != [cols] {
        return Err(Error::Other(format!(
            "MUSA mul_channels needs a [{cols}] channel scale, got {:?}",
            channel_scale.shape().dims()
        )));
    }
    Ok((rows, cols))
}

#[cfg(apxinf_musa_available)]
fn launch_mul_channels(
    ctx: &MusaContext,
    input: &Tensor,
    channel_scale: &Tensor,
    output: &Tensor,
    rows: usize,
    cols: usize,
) -> Result<()> {
    // SAFETY: shapes, dtypes and one device are established above, and every
    // buffer is owned by a caller `Tensor` that outlives the launch.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_mul_channels_bf16(
                tensor_ptr(input)?,
                tensor_ptr(channel_scale)?,
                tensor_ptr(output)?,
                checked_i32("mul_channels", "rows", rows)?,
                checked_i32("mul_channels", "cols", cols)?,
                ctx.stream().handle(),
            ),
            "mul_channels",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch_mul_channels(
    _ctx: &MusaContext,
    _input: &Tensor,
    _channel_scale: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
) -> Result<()> {
    Err(native_unavailable("elementwise::mul_channels_bf16"))
}

// ── FP16 elementwise (not ported) ────────────────────────────────────────

/// Add a bias vector to every row in FP16.
pub fn bias_f16(ctx: &MusaContext, input: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let _ = (ctx, input, bias);
    Err(not_ported("elementwise::bias_f16"))
}

/// Concatenate two FP16 tensors along rows.
pub fn concat_rows_f16(ctx: &MusaContext, first: &Tensor, second: &Tensor) -> Result<Tensor> {
    let _ = (ctx, first, second);
    Err(not_ported("elementwise::concat_rows_f16"))
}

/// One Euler step in FP16.
pub fn euler_update_f16(
    ctx: &MusaContext,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    let _ = (ctx, state, velocity, dt);
    Err(not_ported("elementwise::euler_update_f16"))
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    //! Device tests for the elementwise kernels.
    //!
    //! The position-embedding case: the replay harness covers
    //! `embedding.add_position_bf16` at its one recorded shape (`[512, 1152]`,
    //! two views of 256 positions, with a bias) and it comes out bit-identical.
    //! What it does not reach is the **bias-less** branch, which is a real one
    //! in the kernel and a real one in the facade: `bias == nullptr` skips the
    //! add. This test covers it, and a row count that is three views rather than
    //! two. The per-channel scale is covered at the end of the module, for the
    //! reason stated there.
    //!
    //! Needs a MUSA device, so it is `#[ignore]`d:
    //! `cargo test -p apxinf-musa --lib -- --ignored --nocapture`.

    use apxinf_core::{DType, Shape, Tensor};

    use crate::buffer::MusaBuffer;
    use crate::context::MusaContext;
    use crate::kernels::embedding::add_position_bf16;
    use crate::kernels::test_support::{
        from_bf16_bits, gate_for, read_back, to_bf16_bits, upload, values,
    };

    use super::mul_channels_bf16;

    fn check(ctx: &MusaContext, views: usize, tokens_per_view: usize, cols: usize, bias: bool) {
        let rows = views * tokens_per_view;
        let projection_values = values(rows * cols, 0x5eed_1234);
        let position_values = values(tokens_per_view * cols, 0x0bad_c0de);
        let bias_values = values(cols, 0x1357_9bdf);

        let projection = upload(ctx, &projection_values, Shape::new(vec![rows, cols]));
        let position = upload(ctx, &position_values, Shape::new(vec![tokens_per_view, cols]));
        let bias_tensor = bias.then(|| upload(ctx, &bias_values, Shape::new(vec![cols])));

        let out = add_position_bf16(
            ctx,
            &projection,
            bias_tensor.as_ref(),
            &position,
            tokens_per_view,
        )
        .expect("native position embedding");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        let round = |v: f32| from_bf16_bits(to_bf16_bits(v));
        let tolerance = gate_for(&got);
        let mut worst = 0.0f32;
        for row in 0..rows {
            let token = row % tokens_per_view;
            for col in 0..cols {
                let index = row * cols + col;
                // The bias-less case is the interesting one: code 0 with a null
                // bias has to *skip* the add rather than add zero, and the two
                // differ only if the kernel reads a pointer it was not given.
                let mut expected =
                    round(projection_values[index]) + round(position_values[token * cols + col]);
                if bias {
                    expected += round(bias_values[col]);
                }
                let expected = round(expected);
                let error = (got[index] - expected).abs();
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "position [{rows}x{cols}] at [{row}][{col}]: {} vs {expected}, \
                     abs {error:.3e} > {tolerance:.3e}",
                    got[index]
                );
            }
        }
        println!(
            "  add_position [{rows:>4} x {cols:>4}] views {views} bias {bias}  \
             worst abs {worst:.3e} (gate {tolerance:.3e})"
        );
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn position_embedding_matches_a_reference_with_and_without_a_bias() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        check(&ctx, 2, 256, 1152, true);
        check(&ctx, 3, 5, 7, true);
        check(&ctx, 3, 5, 7, false);
    }

    /// An f32 `[cols]` vector: `mul_channels_bf16` takes its factors in f32,
    /// and [`upload`] renders BF16.
    fn upload_f32(ctx: &MusaContext, values: &[f32], shape: Shape) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        assert_eq!(bytes.len(), shape.numel() * 4);
        let buffer = MusaBuffer::alloc_on(ctx, bytes.len()).expect("allocate f32 operand");
        buffer.copy_from_host(&bytes).expect("upload f32 operand");
        buffer.as_tensor(shape, DType::F32).expect("view as f32")
    }

    /// The per-channel scale, column for column.
    ///
    /// The model reaches this kernel only through a projection's BF16 fallback
    /// that still carries an equalization fold, which no site it loads does
    /// today -- so nothing else exercises it. What it has to do is plain: one
    /// multiply by that column's factor, rounded once.
    fn check_mul_channels(ctx: &MusaContext, rows: usize, cols: usize) {
        let input_values = values(rows * cols, 0x5eed_1234);
        let factor_values: Vec<f32> = (0..cols)
            .map(|column| 0.25 + (column % 11) as f32 * 0.37)
            .collect();
        let input = upload(ctx, &input_values, Shape::new(vec![rows, cols]));
        let factors = upload_f32(ctx, &factor_values, Shape::new(vec![cols]));

        let out = mul_channels_bf16(ctx, &input, &factors).expect("native per-channel scale");
        ctx.stream().synchronize().expect("synchronise");
        assert_eq!(out.shape().dims(), &[rows, cols]);
        let got = read_back(&out);

        let round = |value: f32| from_bf16_bits(to_bf16_bits(value));
        let tolerance = gate_for(&got);
        let mut worst = 0.0f32;
        for row in 0..rows {
            for column in 0..cols {
                let index = row * cols + column;
                let expected = round(round(input_values[index]) * factor_values[column]);
                let error = (got[index] - expected).abs();
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "mul_channels [{rows}x{cols}] at [{row}][{column}]: {} vs {expected}, \
                     abs {error:.3e} > {tolerance:.3e}",
                    got[index]
                );
            }
        }
        println!(
            "  mul_channels [{rows:>4} x {cols:>5}]  worst abs {worst:.3e} (gate {tolerance:.3e})"
        );
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn per_channel_scale_multiplies_each_column_by_its_factor() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        // The down projection's width, where the fold would run if its INT8
        // route were ever unavailable, and an odd width whose row does not
        // divide the launch grid.
        check_mul_channels(&ctx, 522, 16384);
        check_mul_channels(&ctx, 10, 7);
    }
}

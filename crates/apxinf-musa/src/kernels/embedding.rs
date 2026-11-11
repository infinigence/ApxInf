//! Embedding operator contracts.

use apxinf_core::{Error, Result, Tensor};

use super::contracts::{
    bf16_output, matrix_tensor, require_matrix_bf16, require_optional_vector_bf16,
    require_output_matrix};
#[cfg(apxinf_musa_available)]
use super::contracts::{checked_i32, optional_tensor_ptr, tensor_ptr};
use crate::buffer::MusaBuffer;
use crate::context::MusaContext;
#[cfg(apxinf_musa_available)]
use crate::ffi;
#[cfg(apxinf_musa_available)]
use crate::kernels::check_native;
#[cfg(not(apxinf_musa_available))]
use crate::kernels::native_unavailable;
use crate::kernels::not_ported;

/// Gather rows from a BF16 embedding table by device-resident ids.
///
/// `ids` is a raw device buffer of `u32` token ids, not a typed tensor: the
/// runtime keeps token ids in a mapped buffer so a caller can publish a new
/// token without a copy, and that address is what reaches here.
///
/// Native: `kernels/custom/embedding.cuh`'s scaled gather, through
/// `adapters/custom_kernels.mu`.
pub fn lookup_bf16(
    ctx: &MusaContext,
    table: &Tensor,
    ids: &MusaBuffer,
    tokens: usize,
) -> Result<Tensor> {
    let (_, width) = require_matrix_bf16(ctx, table, "embedding lookup", "table")?;
    let mut output = matrix_tensor(ctx, tokens, width, bf16_output(ctx, tokens, width)?);
    lookup_bf16_into(ctx, table, ids, &mut output, tokens)?;
    Ok(output)
}

/// Write a scaled token-embedding gather into a caller-provided output.
pub fn lookup_bf16_into(
    ctx: &MusaContext,
    table: &Tensor,
    ids: &MusaBuffer,
    output: &mut Tensor,
    tokens: usize,
) -> Result<()> {
    let (vocab, width) = require_matrix_bf16(ctx, table, "embedding lookup", "table")?;
    if tokens == 0 {
        return Err(Error::Other(
            "MUSA embedding lookup needs a non-zero token count".into(),
        ));
    }
    if ids.device() != ctx.device_id() {
        return Err(Error::DeviceMismatch {
            expected: apxinf_core::Device::Musa(ctx.device_id()),
            got: apxinf_core::Device::Musa(ids.device()),
        });
    }
    let required = tokens
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| Error::Other("MUSA embedding lookup id size overflow".into()))?;
    if ids.len() < required {
        return Err(Error::Other(format!(
            "MUSA embedding lookup needs {required} bytes of ids, has {}",
            ids.len()
        )));
    }
    require_output_matrix(ctx, output, tokens, width, "embedding lookup", "out")?;
    launch_lookup(ctx, table, ids, output, tokens, width, vocab)
}

/// The bf16 value `sqrtf(width)` rounds to, as an f32.
///
/// Part of the operator's contract rather than a detail of it: the reference
/// multiplies the looked-up row by `bf16(sqrt(width))`, and for width 2048 that
/// is 45.25 rather than `sqrtf`'s 45.254833. Rounding the scalar before the
/// multiply is what makes the result independent of operand promotion -- see
/// `_lookup_bf16` in the torch eager bridge that recorded the reference (it is
/// not part of this tree), which carries the measurement.
fn scaled_sqrt_bf16(width: usize) -> f32 {
    let root = (width as f32).sqrt();
    let bits = root.to_bits();
    // Round-to-nearest-even into bf16 and back, which is exact in f32.
    let round = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(round) & 0xffff_0000)
}

/// The launch, split by whether this build has a MUSA toolkit.
///
/// # Out-of-range ids
/// The bridge raised on them (`index_select` checks its indices), and the CUDA
/// kernel writes zeros for them. This follows the bridge: the ids are read back
/// and checked first, and a bad one is reported with its position and the
/// table's size. The read is `tokens` integers -- once per inference, on the
/// prefill path, next to a gather over a gigabyte of table -- so it costs
/// nothing worth weighing against a silent zero that would look like a valid
/// embedding row downstream.
///
/// # Why the check is conditional
/// It is a **host round trip inside the model's execution**, which
/// `doc/model-execution-wiring.md`'s residency gate forbids between the public
/// inputs and the outputs, and a synchronous device-to-host copy cannot appear
/// inside a capture region at all. So it runs on every traversal **outside**
/// one -- which includes the prepare pass `ExecutionSession::prepare` runs before
/// `musaStreamBeginCapture`, on the same stream and the same buffers, so a
/// prepared plan does validate the ids its buffers hold at preparation time.
///
/// What it cannot do is re-validate at replay: a graph owns no host code, so a
/// caller that overwrites the id buffer afterwards and replays is unguarded
/// here. That is why the model layer checks ids where it publishes them
/// (`pi05/model_runner/runner.rs::validate_token_ids`), on every request, and
/// that check -- not this one -- is what keeps a replayed plan safe.
/// `graph::is_capturing()` is the seam between the two: it is set only between
/// `musaStreamBeginCapture` and `musaStreamEndCapture`.
#[cfg(apxinf_musa_available)]
fn launch_lookup(
    ctx: &MusaContext,
    table: &Tensor,
    ids: &MusaBuffer,
    output: &Tensor,
    tokens: usize,
    width: usize,
    vocab: usize,
) -> Result<()> {
    if !crate::workspace::is_capturing() {
        let mut raw = vec![0u8; tokens * std::mem::size_of::<u32>()];
        ids.copy_to_host(&mut raw)
            .map_err(|message| Error::Other(format!("MUSA embedding lookup ids: {message}")))?;
        for (position, chunk) in raw.chunks_exact(4).enumerate() {
            let id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if id as usize >= vocab {
                return Err(Error::Other(format!(
                "MUSA embedding lookup id {id} at {position} is outside the table's {vocab} rows"
            )));
            }
        }
    }
    let normalizer = scaled_sqrt_bf16(width);
    let tokens = checked_i32("embedding lookup", "tokens", tokens)?;
    let width = checked_i32("embedding lookup", "width", width)?;
    // SAFETY: the table and the output are BF16 allocations on this device of
    // the shapes `require_matrix_bf16` and `matrix_tensor` established, the ids
    // are a device buffer of at least `tokens` u32 (checked above) whose every
    // value is inside the table, and both outlive the asynchronous launch.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_embedding_bf16_scaled(
                tensor_ptr(table)?,
                ids.ptr(),
                tensor_ptr(output)?,
                tokens,
                width,
                normalizer,
                ctx.stream().handle(),
            ),
            "embedding lookup",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch_lookup(
    _ctx: &MusaContext,
    _table: &Tensor,
    _ids: &MusaBuffer,
    _output: &Tensor,
    _tokens: usize,
    _width: usize,
    _vocab: usize,
) -> Result<()> {
    Err(native_unavailable("embedding::lookup_bf16"))
}

/// Add a per-position bias to a projection, tiled across views.
///
/// `position` is BF16 `[tokens_per_view, cols]` or `[1, tokens_per_view, cols]`
/// and `projection` has a row count that is a multiple of `tokens_per_view`.
pub fn add_position_bf16(
    ctx: &MusaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    let (rows, cols) = require_matrix_bf16(ctx, projection, "position embedding", "projection")?;
    let mut output = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    add_position_bf16_into(ctx, projection, bias, position, &mut output, tokens_per_view)?;
    Ok(output)
}

/// Write the position-added projection into a caller-provided output.
pub fn add_position_bf16_into(
    ctx: &MusaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    output: &mut Tensor,
    tokens_per_view: usize,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, projection, "position embedding", "projection")?;
    if tokens_per_view == 0 || rows % tokens_per_view != 0 {
        return Err(Error::Other(format!(
            "MUSA position embedding needs tokens_per_view dividing {rows} rows, got {tokens_per_view}"
        )));
    }
    let position_dims = position.shape().dims();
    let position_ok = match position_dims {
        [tokens, width] => *tokens == tokens_per_view && *width == cols,
        [one, tokens, width] => *one == 1 && *tokens == tokens_per_view && *width == cols,
        _ => false,
    };
    // The device is checked here rather than by `require_matrix_bf16`, because
    // this operand is allowed two ranks: a same-shaped tensor living on another
    // device would pass a shape check and then be dereferenced by the kernel.
    if position.dtype() != apxinf_core::DType::BF16
        || position.device() != apxinf_core::Device::Musa(ctx.device_id())
        || !position_ok
    {
        return Err(Error::Other(format!(
            "MUSA position embedding expects a BF16 [{tokens_per_view}, {cols}] or \
             [1, {tokens_per_view}, {cols}] position table on MUSA{}, got {} {:?} on {}",
            ctx.device_id(),
            position.dtype(),
            position_dims,
            position.device()
        )));
    }
    require_optional_vector_bf16(ctx, bias, cols, "position embedding", "bias")?;
    require_output_matrix(ctx, output, rows, cols, "position embedding", "out")?;
    launch_position(ctx, projection, bias, position, output, rows, cols, tokens_per_view)
}

/// The launch, split by whether this build has a MUSA toolkit.
#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_position(
    ctx: &MusaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    output: &Tensor,
    rows: usize,
    cols: usize,
    tokens_per_view: usize,
) -> Result<()> {
    let rows = checked_i32("position embedding", "rows", rows)?;
    let cols = checked_i32("position embedding", "cols", cols)?;
    let tokens_per_view = checked_i32("position embedding", "tokens_per_view", tokens_per_view)?;
    let bias_pointer = optional_tensor_ptr(bias)?;
    // SAFETY: shapes, dtypes and one device are established above; `position`
    // holds exactly `tokens_per_view * cols` contiguous elements of the same
    // width as the projection (checked by shape), and every buffer outlives the
    // asynchronous launch on `ctx.stream()`.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_add_position_bf16(
                tensor_ptr(projection)?,
                bias_pointer,
                tensor_ptr(position)?,
                tensor_ptr(output)?,
                rows,
                cols,
                tokens_per_view,
                ctx.stream().handle(),
            ),
            "position embedding",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch_position(
    _ctx: &MusaContext,
    _projection: &Tensor,
    _bias: Option<&Tensor>,
    _position: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
    _tokens_per_view: usize,
) -> Result<()> {
    Err(native_unavailable("embedding::add_position_bf16"))
}

// ── FP16 embedding (not ported) ──────────────────────────────────────────

/// Gather rows from an FP16 embedding table by device-resident ids.
pub fn lookup_f16(
    ctx: &MusaContext,
    table: &Tensor,
    ids: &MusaBuffer,
    tokens: usize,
) -> Result<Tensor> {
    let _ = (ctx, table, ids, tokens);
    Err(not_ported("embedding::lookup_f16"))
}

/// Add a per-position bias to an FP16 projection.
pub fn add_position_f16(
    ctx: &MusaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    let _ = (ctx, projection, bias, position, tokens_per_view);
    Err(not_ported("embedding::add_position_f16"))
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    //! Device tests for the embedding kernels.
    //!
    //! The replay harness covers `embedding.lookup_bf16` on the real 1.05 GB
    //! table with the recorded ids, and it comes out bit-identical -- which is
    //! the strongest statement available about the multiplier, since a
    //! `sqrtf(width)` instead of `bf16(sqrt(width))` would move 542 of those
    //! 20480 elements. What it cannot reach is the **refusal**: the recorded
    //! ids are all inside the table, so the branch that rejects an out-of-range
    //! one has no reference to be compared against. That is what this checks,
    //! along with the gather itself on a table small enough to write down.
    //!
    //! `add_position_bf16` gets the same treatment from the other side: its
    //! `position` operand is checked by hand (two ranks, so `require_matrix_bf16`
    //! does not reach it) and a tensor from another device is the failure that
    //! check exists to prevent.
    //!
    //! Needs a MUSA device, so they are `#[ignore]`d:
    //! `cargo test -p apxinf-musa --lib -- --ignored --nocapture`.

    use apxinf_core::{DType, Shape, Tensor};

    use crate::buffer::MusaBuffer;
    use crate::context::MusaContext;
    use crate::kernels::embedding::{add_position_bf16, lookup_bf16, scaled_sqrt_bf16};
    use crate::kernels::test_support::{from_bf16_bits, read_back, to_bf16_bits, upload, values};

    fn ids_buffer(ctx: &MusaContext, ids: &[u32]) -> MusaBuffer {
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let buffer = MusaBuffer::alloc_on(ctx, bytes.len()).expect("allocate ids");
        buffer.copy_from_host(&bytes).expect("upload ids");
        buffer
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn lookup_gathers_the_selected_rows_scaled_once() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let (vocab, width) = (16usize, 8usize);
        let table_values = values(vocab * width, 0x5eed_1234);
        let table = upload(&ctx, &table_values, Shape::new(vec![vocab, width]));
        let ids = [3u32, 0, 15, 3];

        let out = lookup_bf16(&ctx, &table, &ids_buffer(&ctx, &ids), ids.len())
            .expect("native embedding lookup");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        // A gather and one rounded multiply: no reduction, so nothing to
        // accumulate and nothing to gate. Bit equality is the right assertion,
        // and it is what a wrong row or a dropped scale fails.
        let scale = scaled_sqrt_bf16(width);
        let mut expected = Vec::new();
        for id in ids {
            for col in 0..width {
                let value = from_bf16_bits(to_bf16_bits(table_values[id as usize * width + col]));
                expected.push(from_bf16_bits(to_bf16_bits(value * scale)));
            }
        }
        assert_eq!(got, expected, "lookup must gather exactly the named rows");
        println!(
            "  lookup {vocab}x{width}, ids {ids:?}, scale {scale}: bit-identical over {} elements",
            got.len()
        );
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn lookup_refuses_an_id_outside_the_table() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let (vocab, width) = (4usize, 8usize);
        let table = upload(
            &ctx,
            &values(vocab * width, 0x0bad_c0de),
            Shape::new(vec![vocab, width]),
        );
        let ids = [1u32, 4];

        // The bridge raised on this through `index_select`; the CUDA kernel
        // writes zeros. Following the bridge is only defensible if the refusal
        // actually happens, and names what was wrong -- which is what this pins
        // down. A silent zero would look like a valid embedding row downstream.
        let error = lookup_bf16(&ctx, &table, &ids_buffer(&ctx, &ids), ids.len())
            .expect_err("an id past the end of the table must be refused");
        let message = error.to_string();
        assert!(
            message.contains("id 4") && message.contains("at 1") && message.contains("4 rows"),
            "unhelpful refusal: {message}"
        );
        println!("  lookup refused: {message}");
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn position_embedding_refuses_a_position_table_on_another_device() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let projection = upload(&ctx, &values(15 * 7, 0x5eed_1234), Shape::new(vec![15, 7]));

        // The kernel dereferences `position` through a raw pointer, so an
        // operand of the right shape on the wrong device is a fault rather than
        // a wrong answer. `position` is checked by hand for exactly this reason
        // -- it is the one operand `require_matrix_bf16` cannot reach, because
        // it accepts two ranks.
        let position = Tensor::zeros(Shape::new(vec![5, 7]), DType::BF16);
        let error = add_position_bf16(&ctx, &projection, None, &position, 5)
            .expect_err("a host operand must be refused");
        assert!(
            error.to_string().contains("position table on MUSA0"),
            "unhelpful refusal: {error}"
        );
        println!("  position embedding refused: {error}");
    }
}

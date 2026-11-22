//! Device weights. Every large linear is ConvRot-rotated and INT8-quantised on the host during load;
//! everything that does not depend on the request (the 16 learned query tokens, the AdaRMS
//! scale/shift of all ten flow steps, the ViT position table) is computed once here.

use super::host::Checkpoint;
use super::quant::{quantize_rows, QuantLinear};
use crate::cuda_lingbot_vla2::backend::{ops::RepackedWeight, Context, DeviceBuffer};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use half::bf16;

/// W8A8 linear: INT8 weight (logical `[K, N]`, physical layout owned by `layout`) with one FP32
/// scale per output channel. The rotation is already folded into the weight bytes.
pub(crate) struct Int8Linear {
    pub weight: Tensor,
    pub layout: RepackedWeight,
    pub scales: Tensor,
    pub k: usize,
    pub n: usize,
}

/// BF16 linear stored `[K, N]` (transposed from the checkpoint's `[N, K]`).
pub(crate) struct DenseLinear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(format!("LingBot-VLA 2.0 weights: {}", message.into()))
}

fn upload(ctx: &Context, bytes: &[u8], shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    let buffer = DeviceBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Cuda)?;
    buffer
        .copy_from_host_on(bytes, ctx.stream())
        .map_err(Error::Cuda)?;
    buffer
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

pub(crate) fn bf16_tensor(ctx: &Context, values: &[bf16], shape: Vec<usize>) -> Result<Tensor> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    upload(ctx, &bytes, shape, DType::BF16)
}

pub(crate) fn f32_tensor(ctx: &Context, values: &[f32], shape: Vec<usize>) -> Result<Tensor> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    upload(ctx, &bytes, shape, DType::F32)
}

pub(crate) fn u32_tensor(ctx: &Context, values: &[u32], shape: Vec<usize>) -> Result<Tensor> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    upload(ctx, &bytes, shape, DType::I32)
}

/// Upload one quantised `[n, k]` linear. When the device has an output-major W8A8 candidate the
/// checkpoint-order bytes go up unchanged; otherwise they are transposed first (handled inside).
pub(crate) fn upload_int8(ctx: &Context, q: &QuantLinear) -> Result<Int8Linear> {
    let (weight, layout) =
        RepackedWeight::upload_row_channel_output_major(ctx, &q.weight, q.n, q.k)?;
    let scales = f32_tensor(ctx, &q.scales, vec![q.n])?;
    Ok(Int8Linear {
        weight,
        layout,
        scales,
        k: q.k,
        n: q.n,
    })
}

/// Read an F32 `[n, k]` matrix from the checkpoint, rotate+quantise it and upload it.
/// Matrices with many rows are processed in blocks so host memory stays bounded.
pub(crate) fn int8_from_checkpoint(
    ctx: &Context,
    ck: &Checkpoint,
    name: &str,
    n: usize,
    k: usize,
) -> Result<Int8Linear> {
    ck.expect(name, &[n, k])?;
    const BLOCK_ROWS: usize = 2048;
    let mut weight = Vec::with_capacity(n * k);
    let mut scales = Vec::with_capacity(n);
    let mut row = 0;
    while row < n {
        let rows = BLOCK_ROWS.min(n - row);
        let part = quantize_rows(&ck.f32_rows(name, row, rows)?, rows, k);
        weight.extend_from_slice(&part.weight);
        scales.extend_from_slice(&part.scales);
        row += rows;
    }
    upload_int8(
        ctx,
        &QuantLinear {
            weight,
            scales,
            n,
            k,
        },
    )
}

/// BF16 linear transposed to `[in, out]`, with an optional BF16 bias.
pub(crate) fn dense_from_checkpoint(
    ctx: &Context,
    ck: &Checkpoint,
    name: &str,
    out_features: usize,
    in_features: usize,
    bias: Option<&str>,
) -> Result<DenseLinear> {
    let w = ck.bf16(name, &[out_features, in_features])?;
    let mut transposed = vec![bf16::ZERO; w.len()];
    for o in 0..out_features {
        for i in 0..in_features {
            transposed[i * out_features + o] = w[o * in_features + i];
        }
    }
    let bias = match bias {
        Some(b) => Some(bf16_tensor(
            ctx,
            &ck.bf16(b, &[out_features])?,
            vec![out_features],
        )?),
        None => None,
    };
    Ok(DenseLinear {
        weight: bf16_tensor(ctx, &transposed, vec![in_features, out_features])?,
        bias,
    })
}

/// Bias of a W8A8 linear is applied after the GEMM, so keep it as its own BF16 vector.
pub(crate) fn bias_bf16(ctx: &Context, ck: &Checkpoint, name: &str, n: usize) -> Result<Tensor> {
    bf16_tensor(ctx, &ck.bf16(name, &[n])?, vec![n])
}

pub(crate) fn vector_bf16(ctx: &Context, ck: &Checkpoint, name: &str, n: usize) -> Result<Tensor> {
    bias_bf16(ctx, ck, name, n)
}

pub(crate) fn check_len(name: &str, got: usize, want: usize) -> Result<()> {
    if got != want {
        return Err(invalid(format!("{name}: {got} elements, expected {want}")));
    }
    Ok(())
}

/// Raw device copy of `values`, for operators that take a buffer rather than a tensor.
pub(crate) fn upload_u32_buffer(ctx: &Context, values: &[u32]) -> Result<DeviceBuffer> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let buffer = DeviceBuffer::alloc(bytes.len().max(1), ctx.device_id()).map_err(Error::Cuda)?;
    buffer
        .copy_from_host_on(&bytes, ctx.stream())
        .map_err(Error::Cuda)?;
    Ok(buffer)
}

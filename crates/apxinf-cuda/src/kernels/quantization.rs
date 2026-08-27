//! Quantization operator contracts.

use apxinf_core::{DType, Error, Result, Tensor};

use super::contracts::{check_cuda, gpu_ptr, make_gpu_tensor};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::workspace::output_buffer;
use crate::cublas::CublasTranspose;

/// Quantize an FP16 device tensor to E4M3 using a pre-calibrated scale.
pub fn quantize_f16_e4m3(ctx: &CudaContext, input: &Tensor, scale: f32) -> Result<Tensor> {
    if input.dtype() != DType::F16 {
        return Err(Error::DTypeMismatch {
            expected: DType::F16,
            got: input.dtype(),
        });
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Other(format!("invalid FP8 scale {scale}")));
    }
    let output = output_buffer(ctx, input.numel())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_quantize_f16_e4m3(
            gpu_ptr(input)?,
            output.ptr(),
            input.numel() as i64,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        input.shape().clone(),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    ))
}


/// BF16 activation times asymmetric group-wise packed INT4 weights.
///
/// Activation shape is `[rows, in_cols]`. Packed weights are Hugging Face
/// compressed-tensors W4A16 layout with logical shape `[out_cols, in_cols]`,
/// packed low-nibble-first across input columns and zero-points packed across
/// output rows. The output is BF16 `[rows, out_cols]`.

/// Run the standalone Marlin asymmetric U4 group-32 GEMM.
///
/// `activation_scratch` and `output_scratch` are caller-owned BF16 physical
/// `[rows, padded_k]` and `[rows, padded_n]` buffers. They are used only when
/// the corresponding logical dimension is padded. `workspace` contains one
/// i32 lock per SM; the adapter resets it immediately before launch.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_marlin_awq_u4_g32_v1_into(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    marlin_qweight: &CudaBuffer,
    scales: &CudaBuffer,
    zero_points: &CudaBuffer,
    output: &CudaBuffer,
    activation_scratch: &CudaBuffer,
    output_scratch: &CudaBuffer,
    workspace: &CudaBuffer,
    rows: usize,
    logical_n: usize,
    logical_k: usize,
    padded_n: usize,
    padded_k: usize,
) -> Result<()> {
    let sms = ctx.caps().multiprocessor_count as usize;
    let fits_i32 = [rows, logical_n, logical_k, padded_n, padded_k, sms]
        .into_iter()
        .all(|value| value <= i32::MAX as usize);
    if !fits_i32
        || !(1..=512).contains(&rows)
        || logical_n == 0
        || ctx.caps().sm != 89
        || logical_k == 0
        || logical_n > padded_n
        || logical_k > padded_k
        || padded_n % 64 != 0
        || padded_k % 128 != 0
        || sms == 0
    {
        return Err(Error::Other("Marlin W4 GEMM: invalid dimensions".into()));
    }
    let bf16 = DType::BF16.size_in_bytes();
    let checked_bytes = |m: usize, n: usize| {
        m.checked_mul(n)
            .and_then(|elements| elements.checked_mul(bf16))
            .ok_or_else(|| Error::Other("Marlin W4 GEMM buffer size overflow".into()))
    };
    let activation_bytes = checked_bytes(rows, logical_k)?;
    let output_bytes = checked_bytes(rows, logical_n)?;
    let padded_activation_bytes = checked_bytes(rows, padded_k)?;
    let padded_output_bytes = checked_bytes(rows, padded_n)?;
    let packed_bytes = padded_k
        .checked_mul(padded_n)
        .and_then(|elements| elements.checked_div(2))
        .ok_or_else(|| Error::Other("Marlin W4 packed size overflow".into()))?;
    let groups = padded_k / 32;
    let scales_bytes = padded_n
        .checked_mul(groups)
        .and_then(|elements| elements.checked_mul(bf16))
        .ok_or_else(|| Error::Other("Marlin W4 scale size overflow".into()))?;
    let zero_point_bytes = padded_n
        .checked_mul(groups)
        .and_then(|elements| elements.checked_div(2))
        .ok_or_else(|| Error::Other("Marlin W4 zero-point size overflow".into()))?;
    if activation.len() < activation_bytes
        || output.len() < output_bytes
        || marlin_qweight.len() < packed_bytes
        || scales.len() < scales_bytes
        || zero_points.len() < zero_point_bytes
        || activation_scratch.len() < padded_activation_bytes
        || output_scratch.len() < padded_output_bytes
        || workspace.len() < sms * std::mem::size_of::<i32>()
    {
        return Err(Error::Other("Marlin W4 GEMM buffer too small".into()));
    }
    for buffer in [activation, marlin_qweight, scales, zero_points, output,
        activation_scratch, output_scratch, workspace]
    {
        if buffer.device() != ctx.device_id() {
            return Err(Error::Other("Marlin W4 buffers must share the CUDA context device".into()));
        }
    }

    let launch_activation = if logical_k == padded_k {
        activation
    } else {
        activation_scratch.zero_async(ctx.stream()).map_err(Error::Cuda)?;
        unsafe {
            ffi::check_cuda(ffi::cudaMemcpy2DAsync(
                activation_scratch.ptr(), padded_k * bf16, activation.ptr(), logical_k * bf16,
                logical_k * bf16, rows, ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                ctx.stream().handle(),
            ))
        }.map_err(Error::Cuda)?;
        activation_scratch
    };
    let launch_output = if logical_n == padded_n { output } else { output_scratch };
    unsafe {
        ffi::check_cuda(ffi::apxinf_marlin_awq_u4_g32_v1_gemm_bf16(
            launch_activation.ptr(), marlin_qweight.ptr(), scales.ptr(), zero_points.ptr(),
            launch_output.ptr(), workspace.ptr(), rows as i32, logical_n as i32,
            logical_k as i32, padded_n as i32, padded_k as i32, sms as i32,
            ctx.stream().handle(),
        ))
    }.map_err(Error::Cuda)?;
    if logical_n != padded_n {
        unsafe {
            ffi::check_cuda(ffi::cudaMemcpy2DAsync(
                output.ptr(), logical_n * bf16, output_scratch.ptr(), padded_n * bf16,
                logical_n * bf16, rows, ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                ctx.stream().handle(),
            ))
        }.map_err(Error::Cuda)?;
    }
    Ok(())
}

/// Exact prefill path for persistent Marlin AWQ U4 group-32 weights.
///
/// Each output tile is first inverted into canonical compressed-tensors
/// qweight/scales/zero-points in caller-owned scratch.  The established raw
/// row dequantizer then produces the BF16 bytes consumed by the same row-tiled
/// cuBLAS call shape as raw weights.  No full dense inverse is materialized.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_marlin_awq_u4_g32_v1_prefill_into(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    marlin_qweight: &CudaBuffer,
    scales: &CudaBuffer,
    zero_points: &CudaBuffer,
    scratch: &CudaBuffer,
    output: &CudaBuffer,
    rows: usize,
    logical_n: usize,
    logical_k: usize,
    padded_n: usize,
    padded_k: usize,
    source_n_offset: usize,
) -> Result<()> {
    if rows <= 1
        || logical_n == 0
        || logical_k == 0
        || logical_k % 32 != 0
        || source_n_offset > padded_n
        || logical_n > padded_n - source_n_offset
        || logical_k > padded_k
        || padded_n % 64 != 0
        || padded_k % 128 != 0
        || [rows, logical_n, logical_k, padded_n, padded_k, source_n_offset]
            .into_iter()
            .any(|value| value > i32::MAX as usize)
    {
        return Err(Error::Other("Marlin W4 prefill: invalid dimensions".into()));
    }
    let bf16 = DType::BF16.size_in_bytes();
    let bytes = |m: usize, n: usize| {
        m.checked_mul(n)
            .and_then(|elements| elements.checked_mul(bf16))
            .ok_or_else(|| Error::Other("Marlin W4 prefill buffer size overflow".into()))
    };
    let packed_bytes = padded_k
        .checked_mul(padded_n)
        .and_then(|elements| elements.checked_div(2))
        .ok_or_else(|| Error::Other("Marlin W4 prefill packed size overflow".into()))?;
    let padded_groups = padded_k / 32;
    let scales_bytes = bytes(padded_groups, padded_n)?;
    let zero_point_bytes = padded_groups
        .checked_mul(padded_n)
        .and_then(|elements| elements.checked_div(2))
        .ok_or_else(|| Error::Other("Marlin W4 prefill zero-point size overflow".into()))?;
    if activation.len() < bytes(rows, logical_k)?
        || marlin_qweight.len() < packed_bytes
        || scales.len() < scales_bytes
        || zero_points.len() < zero_point_bytes
        || output.len() < bytes(rows, logical_n)?
    {
        return Err(Error::Other("Marlin W4 prefill buffer too small".into()));
    }
    for buffer in [activation, marlin_qweight, scales, zero_points, scratch, output] {
        if buffer.device() != ctx.device_id() {
            return Err(Error::Other(
                "Marlin W4 prefill buffers must share the CUDA context device".into(),
            ));
        }
    }

    if !std::env::var_os("APXINF_MARLIN_PREFILL_RAW").is_some_and(|value| value == "1") {
        let dense_bytes = bytes(logical_n, logical_k)?;
        if scratch.len() < dense_bytes {
            return Err(Error::Other("Marlin W4 vector prefill scratch too small".into()));
        }
        let dense = scratch.view(0, dense_bytes).map_err(Error::Cuda)?;
        unsafe {
            ffi::check_cuda(ffi::apxinf_marlin_awq_u4_g32_v1_dequant_bf16(
                marlin_qweight.ptr(), scales.ptr(), zero_points.ptr(), dense.ptr(),
                logical_n as i32, logical_k as i32, padded_n as i32,
                padded_k as i32, source_n_offset as i32, ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        let dense_row_bytes = logical_k
            .checked_mul(bf16)
            .ok_or_else(|| Error::Other("Marlin W4 vector prefill row overflow".into()))?;
        let tile_rows = dense.len().min(W4A16_PREFILL_DENSE_TILE_BYTES) / dense_row_bytes;
        if tile_rows == 0 {
            return Err(Error::Other("Marlin W4 vector prefill tile is empty".into()));
        }
        if tile_rows >= logical_n {
            return matmul_bf16_transposed_into(
                ctx, activation, &dense, output, rows, logical_k, logical_n,
            );
        }
        let output_bytes = bytes(rows, logical_n)?;
        for out_start in (0..logical_n).step_by(tile_rows) {
            let out_count = tile_rows.min(logical_n - out_start);
            let dense_offset = out_start
                .checked_mul(dense_row_bytes)
                .ok_or_else(|| Error::Other("Marlin W4 vector dense offset overflow".into()))?;
            let tile_bytes = out_count
                .checked_mul(dense_row_bytes)
                .ok_or_else(|| Error::Other("Marlin W4 vector tile overflow".into()))?;
            let dense_tile = dense.view(dense_offset, tile_bytes).map_err(Error::Cuda)?;
            let output_offset = out_start
                .checked_mul(bf16)
                .ok_or_else(|| Error::Other("Marlin W4 vector output offset overflow".into()))?;
            let output_tile = output
                .view(output_offset, output_bytes - output_offset)
                .map_err(Error::Cuda)?;
            super::gemm::write_ex(
                ctx, DType::BF16, CublasTranspose::None, CublasTranspose::Transpose,
                rows, out_count, logical_k, 1.0, activation, logical_k as i32,
                &dense_tile, logical_k as i32, 0.0, &output_tile, logical_n as i32,
            )?;
        }
        return Ok(());
    }

    let groups = logical_k / 32;
    let dense_row_bytes = logical_k
        .checked_mul(bf16)
        .ok_or_else(|| Error::Other("Marlin W4 prefill dense row overflow".into()))?;
    // Match the raw path's 32 MiB dense tile schedule. Compact inverse storage
    // is placed before that tile in the same persistent allocation.
    let mut tile_capacity = (W4A16_PREFILL_DENSE_TILE_BYTES / dense_row_bytes).min(logical_n);
    let align16 = |value: usize| value.checked_add(15).map(|v| v & !15);
    let layout_bytes = |tile_rows: usize| -> Option<(usize, usize, usize, usize, usize)> {
        let q_bytes = tile_rows.checked_mul(logical_k)?.checked_div(2)?;
        let scale_offset = align16(q_bytes)?;
        let scale_bytes = tile_rows.checked_mul(groups)?.checked_mul(bf16)?;
        let zp_offset = align16(scale_offset.checked_add(scale_bytes)?)?;
        let zp_bytes = tile_rows.div_ceil(8).checked_mul(groups)?.checked_mul(4)?;
        let dense_offset = align16(zp_offset.checked_add(zp_bytes)?)?;
        let dense_bytes = tile_rows.checked_mul(dense_row_bytes)?;
        Some((scale_offset, zp_offset, dense_offset, dense_bytes,
              dense_offset.checked_add(dense_bytes)?))
    };
    while tile_capacity > 0
        && layout_bytes(tile_capacity).is_none_or(|layout| layout.4 > scratch.len())
    {
        tile_capacity -= 1;
    }
    if tile_capacity == 0 {
        return Err(Error::Other(
            "Marlin W4 prefill scratch cannot hold one compact raw/dense tile".into(),
        ));
    }

    let output_bytes = bytes(rows, logical_n)?;
    for out_start in (0..logical_n).step_by(tile_capacity) {
        let out_count = tile_capacity.min(logical_n - out_start);
        let (scale_offset, zp_offset, dense_offset, dense_bytes, _) =
            layout_bytes(out_count)
                .ok_or_else(|| Error::Other("Marlin W4 prefill tile overflow".into()))?;
        let q_bytes = out_count
            .checked_mul(logical_k)
            .and_then(|v| v.checked_div(2))
            .ok_or_else(|| Error::Other("Marlin W4 prefill qweight tile overflow".into()))?;
        let scale_bytes = out_count
            .checked_mul(groups)
            .and_then(|v| v.checked_mul(bf16))
            .ok_or_else(|| Error::Other("Marlin W4 prefill scale tile overflow".into()))?;
        let zp_bytes = out_count
            .div_ceil(8)
            .checked_mul(groups)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| Error::Other("Marlin W4 prefill zero-point tile overflow".into()))?;
        let raw_q = scratch.view(0, q_bytes).map_err(Error::Cuda)?;
        let raw_scales = scratch.view(scale_offset, scale_bytes).map_err(Error::Cuda)?;
        let raw_zp = scratch.view(zp_offset, zp_bytes).map_err(Error::Cuda)?;
        let dense_tile = scratch.view(dense_offset, dense_bytes).map_err(Error::Cuda)?;
        let output_offset = out_start
            .checked_mul(bf16)
            .ok_or_else(|| Error::Other("Marlin W4 prefill output offset overflow".into()))?;
        let output_tile = output
            .view(output_offset, output_bytes - output_offset)
            .map_err(Error::Cuda)?;

        unsafe {
            ffi::check_cuda(ffi::apxinf_marlin_awq_u4_g32_v1_inverse_raw_rows(
                marlin_qweight.ptr(),
                scales.ptr(),
                zero_points.ptr(),
                raw_q.ptr(),
                raw_scales.ptr(),
                raw_zp.ptr(),
                logical_k as i32,
                padded_n as i32,
                padded_k as i32,
                source_n_offset as i32,
                out_start as i32,
                out_count as i32,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::apxinf_qwen35_dequant_w4a16_bf16_rows(
                raw_q.ptr(),
                raw_scales.ptr(),
                raw_zp.ptr(),
                dense_tile.ptr(),
                logical_k as i32,
                out_count as i32,
                groups as i32,
                0,
                out_count as i32,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        super::gemm::write_ex(
            ctx,
            DType::BF16,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            rows,
            out_count,
            logical_k,
            1.0,
            activation,
            logical_k as i32,
            &dense_tile,
            logical_k as i32,
            0.0,
            &output_tile,
            logical_n as i32,
        )?;
    }
    Ok(())
}


/// One independently weighted output in a shared-activation Marlin batch.
pub struct MarlinAwqU4G32V1Projection<'a> {
    pub marlin_qweight: &'a CudaBuffer,
    pub scales: &'a CudaBuffer,
    pub zero_points: &'a CudaBuffer,
    pub output: &'a CudaBuffer,
    pub logical_n: usize,
    pub padded_n: usize,
}

/// Enqueue two to four Marlin projections with one activation preparation,
/// one FFI call, and a sequentially reused output/lock scratch region.
pub fn matmul_bf16_marlin_awq_u4_g32_v1_batch_into(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    projections: &[MarlinAwqU4G32V1Projection<'_>],
    scratch: &CudaBuffer,
    rows: usize,
    logical_k: usize,
    padded_k: usize,
) -> Result<()> {
    let sms = ctx.caps().multiprocessor_count as usize;
    if !(2..=4).contains(&projections.len())
        || !(1..=512).contains(&rows)
        || ctx.caps().sm != 89
        || logical_k == 0
        || logical_k > padded_k
        || padded_k % 128 != 0
        || sms == 0
        || [rows, logical_k, padded_k, sms]
            .into_iter()
            .any(|value| value > i32::MAX as usize)
    {
        return Err(Error::Other("Marlin W4 batch: invalid dimensions".into()));
    }
    let bf16 = DType::BF16.size_in_bytes();
    let bytes = |m: usize, n: usize| {
        m.checked_mul(n)
            .and_then(|elements| elements.checked_mul(bf16))
            .ok_or_else(|| Error::Other("Marlin W4 batch buffer size overflow".into()))
    };
    let activation_bytes = bytes(rows, logical_k)?;
    let padded_activation_bytes = bytes(rows, padded_k)?;
    let max_padded_n = projections
        .iter()
        .map(|projection| projection.padded_n)
        .max()
        .unwrap_or(0);
    let padded_output_bytes = bytes(rows, max_padded_n)?;
    let lock_bytes = sms
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| Error::Other("Marlin W4 batch lock size overflow".into()))?;
    let output_offset = padded_activation_bytes;
    let lock_offset = output_offset
        .checked_add(padded_output_bytes)
        .ok_or_else(|| Error::Other("Marlin W4 batch scratch offset overflow".into()))?;
    let required_scratch = lock_offset
        .checked_add(lock_bytes)
        .ok_or_else(|| Error::Other("Marlin W4 batch scratch size overflow".into()))?;
    if activation.len() < activation_bytes || scratch.len() < required_scratch {
        return Err(Error::Other("Marlin W4 batch buffer too small".into()));
    }
    if activation.device() != ctx.device_id() || scratch.device() != ctx.device_id() {
        return Err(Error::Other("Marlin W4 batch buffers must share the CUDA context device".into()));
    }
    let activation_scratch = scratch
        .view(0, padded_activation_bytes)
        .map_err(Error::Cuda)?;
    let output_scratch = scratch
        .view(output_offset, padded_output_bytes)
        .map_err(Error::Cuda)?;
    let workspace = scratch.view(lock_offset, lock_bytes).map_err(Error::Cuda)?;
    let launch_activation = if logical_k == padded_k {
        activation
    } else {
        activation_scratch.zero_async(ctx.stream()).map_err(Error::Cuda)?;
        unsafe {
            ffi::check_cuda(ffi::cudaMemcpy2DAsync(
                activation_scratch.ptr(),
                padded_k * bf16,
                activation.ptr(),
                logical_k * bf16,
                logical_k * bf16,
                rows,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                ctx.stream().handle(),
            ))
        }
        .map_err(Error::Cuda)?;
        &activation_scratch
    };
    let mut descriptors = Vec::with_capacity(projections.len());
    for projection in projections {
        if projection.logical_n == 0
            || projection.logical_n > projection.padded_n
            || projection.padded_n % 64 != 0
            || projection.logical_n > i32::MAX as usize
            || projection.padded_n > i32::MAX as usize
        {
            return Err(Error::Other("Marlin W4 batch: invalid projection dimensions".into()));
        }
        let packed_bytes = padded_k
            .checked_mul(projection.padded_n)
            .and_then(|elements| elements.checked_div(2))
            .ok_or_else(|| Error::Other("Marlin W4 batch packed size overflow".into()))?;
        let groups = padded_k / 32;
        let scales_bytes = bytes(groups, projection.padded_n)?;
        let zero_point_bytes = groups
            .checked_mul(projection.padded_n)
            .and_then(|elements| elements.checked_div(2))
            .ok_or_else(|| Error::Other("Marlin W4 batch zero-point size overflow".into()))?;
        if projection.marlin_qweight.len() < packed_bytes
            || projection.scales.len() < scales_bytes
            || projection.zero_points.len() < zero_point_bytes
            || projection.output.len() < bytes(rows, projection.logical_n)?
        {
            return Err(Error::Other("Marlin W4 batch projection buffer too small".into()));
        }
        for buffer in [
            projection.marlin_qweight,
            projection.scales,
            projection.zero_points,
            projection.output,
        ] {
            if buffer.device() != ctx.device_id() {
                return Err(Error::Other("Marlin W4 batch buffers must share the CUDA context device".into()));
            }
        }
        descriptors.push(ffi::MarlinAwqU4G32V1Projection {
            marlin_qweight: projection.marlin_qweight.ptr(),
            scales_bf16: projection.scales.ptr(),
            zero_points_u4: projection.zero_points.ptr(),
            output: projection.output.ptr(),
            padded_output: if projection.logical_n == projection.padded_n {
                std::ptr::null_mut()
            } else {
                output_scratch.ptr()
            },
            logical_n: projection.logical_n as i32,
            padded_n: projection.padded_n as i32,
        });
    }
    unsafe {
        ffi::check_cuda(ffi::apxinf_marlin_awq_u4_g32_v1_gemm_batch_bf16(
            launch_activation.ptr(),
            descriptors.as_ptr(),
            descriptors.len() as i32,
            workspace.ptr(),
            rows as i32,
            logical_k as i32,
            padded_k as i32,
            sms as i32,
            ctx.stream().handle(),
        ))
    }
    .map_err(Error::Cuda)
}

/// BF16 activation times a row-major transposed dense BF16 weight.
///
/// Activation shape is `[rows, in_cols]`; weight shape is `[out_cols, in_cols]`.
/// The output is BF16 `[rows, out_cols]`. This matches HuggingFace linear
/// weight layout without materializing a `[in_cols, out_cols]` transpose.

const W4A16_PREFILL_DENSE_TILE_BYTES: usize = 32 * 1024 * 1024;

/// Prefill W4A16 uses the caller-owned persistent dense scratch as its tile
/// budget. The Qwen 3.5 runner sizes this to the largest packed projection
/// (`intermediate * hidden * 2` bytes), so fitting projections take one
/// dequant launch and one full-K cuBLAS call without allocating transient
/// storage. Smaller callers still get bounded row tiles below.

/// W4A16 prefill GEMM using caller-owned dense workspace as a bounded
/// output-column tile. Each weight tile is dequantized immediately before its
/// cuBLAS GEMM, avoiding full-matrix materialization when workspace is smaller
/// than the matrix. cuBLAS retains FP32 accumulation and writes the same BF16
/// output boundary as the full-matrix path.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_prefill_into(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    dense: &CudaBuffer,
    output: &CudaBuffer,
    rows: usize,
    in_cols: usize,
    out_cols: usize,
    groups: usize,
) -> Result<()> {
    let bf16_bytes = DType::BF16.size_in_bytes();
    if rows <= 1
        || in_cols == 0
        || out_cols == 0
        || groups == 0
        || rows > i32::MAX as usize
        || in_cols > i32::MAX as usize
        || out_cols > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("prefill GEMM: invalid dimensions".into()));
    }
    let dense_row_bytes = in_cols
        .checked_mul(bf16_bytes)
        .ok_or_else(|| Error::Other("prefill GEMM dense row overflow".into()))?;
    let activation_bytes = rows
        .checked_mul(dense_row_bytes)
        .ok_or_else(|| Error::Other("prefill GEMM activation overflow".into()))?;
    let output_bytes = rows
        .checked_mul(out_cols)
        .and_then(|v| v.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("prefill GEMM output overflow".into()))?;
    if activation.len() < activation_bytes || output.len() < output_bytes {
        return Err(Error::Other("prefill GEMM: activation/output buffer too small".into()));
    }
    let packed_bytes = out_cols
        .checked_mul(in_cols / 8 + usize::from(in_cols % 8 != 0))
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("prefill GEMM packed weight overflow".into()))?;
    let scale_bytes = out_cols
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("prefill GEMM scale overflow".into()))?;
    let zero_point_bytes = (out_cols / 8 + usize::from(out_cols % 8 != 0))
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("prefill GEMM zero-point overflow".into()))?;
    if weight_packed.len() < packed_bytes
        || weight_scale.len() < scale_bytes
        || weight_zero_point.len() < zero_point_bytes
    {
        return Err(Error::Other("prefill GEMM: quantized weight buffer too small".into()));
    }

    // The vectorized CUDA dequantizer requires complete packed words and
    // group boundaries. Other layouts use the committed full dequant path.
    let tiled_eligible = in_cols % 8 == 0
        && in_cols % groups == 0
        && (in_cols / groups) % 8 == 0;
    let tile_workspace_bytes = dense.len().min(W4A16_PREFILL_DENSE_TILE_BYTES);
    let tile_rows = tile_workspace_bytes / dense_row_bytes;
    if !tiled_eligible || tile_rows == 0 {
        let dense_bytes = out_cols
            .checked_mul(dense_row_bytes)
            .ok_or_else(|| Error::Other("prefill GEMM dense overflow".into()))?;
        if dense.len() < dense_bytes {
            return Err(Error::Other("prefill GEMM: dense scratch too small for fallback".into()));
        }
        unsafe {
            ffi::check_cuda(ffi::apxinf_static_dequantize_w4a16_asym_bf16(
                weight_packed.ptr(),
                weight_scale.ptr(),
                weight_zero_point.ptr(),
                dense.ptr(),
                in_cols as i32,
                out_cols as i32,
                groups as i32,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        return matmul_bf16_transposed_into(
            ctx, activation, dense, output, rows, in_cols, out_cols,
        );
    }
    if tile_rows >= out_cols {
        // A single row-tiled launch is equivalent to full materialization; use
        // the established full path and one cuBLAS call to avoid extra setup.
        unsafe {
            ffi::check_cuda(ffi::apxinf_static_dequantize_w4a16_asym_bf16(
                weight_packed.ptr(),
                weight_scale.ptr(),
                weight_zero_point.ptr(),
                dense.ptr(),
                in_cols as i32,
                out_cols as i32,
                groups as i32,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        return matmul_bf16_transposed_into(
            ctx, activation, dense, output, rows, in_cols, out_cols,
        );
    }
    let tile_capacity = tile_rows.min(out_cols);
    for out_start in (0..out_cols).step_by(tile_capacity) {
        let out_count = tile_capacity.min(out_cols - out_start);
        let tile_bytes = out_count
            .checked_mul(dense_row_bytes)
            .ok_or_else(|| Error::Other("prefill GEMM tile overflow".into()))?;
        let dense_tile = dense.view(0, tile_bytes).map_err(Error::Cuda)?;
        let output_offset = out_start
            .checked_mul(bf16_bytes)
            .ok_or_else(|| Error::Other("prefill GEMM output offset overflow".into()))?;
        let output_tile = output
            .view(output_offset, output_bytes - output_offset)
            .map_err(Error::Cuda)?;
        unsafe {
            ffi::check_cuda(ffi::apxinf_qwen35_dequant_w4a16_bf16_rows(
                weight_packed.ptr(),
                weight_scale.ptr(),
                weight_zero_point.ptr(),
                dense_tile.ptr(),
                in_cols as i32,
                out_cols as i32,
                groups as i32,
                out_start as i32,
                out_count as i32,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        super::gemm::write_ex(
            ctx,
            DType::BF16,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            rows,
            out_count,
            in_cols,
            1.0,
            activation,
            in_cols as i32,
            &dense_tile,
            in_cols as i32,
            0.0,
            &output_tile,
            out_cols as i32,
        )?;
    }
    Ok(())
}

/// Attempts to co-launch raw W4 dequantization for two multi-row projections.
/// The two cuBLAS calls retain their original shapes, order, FP32 accumulation,
/// and BF16 output boundaries. `Ok(false)` is returned before any launch when
/// geometry or caller-owned scratch capacity cannot preserve that path.
#[allow(clippy::too_many_arguments)]
pub fn try_matmul_bf16_w4a16_asym_prefill_pair_into(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    first_weight_packed: &CudaBuffer,
    first_weight_scale: &CudaBuffer,
    first_weight_zero_point: &CudaBuffer,
    first_output: &CudaBuffer,
    first_out_cols: usize,
    second_weight_packed: &CudaBuffer,
    second_weight_scale: &CudaBuffer,
    second_weight_zero_point: &CudaBuffer,
    second_output: &CudaBuffer,
    second_out_cols: usize,
    dense: &CudaBuffer,
    rows: usize,
    in_cols: usize,
    groups: usize,
) -> Result<bool> {
    let bf16_bytes = DType::BF16.size_in_bytes();
    let eligible = rows > 1
        && rows <= i32::MAX as usize
        && in_cols > 0
        && in_cols <= i32::MAX as usize
        && in_cols % 8 == 0
        && groups > 0
        && groups <= i32::MAX as usize
        && in_cols % groups == 0
        && (in_cols / groups) % 8 == 0
        && first_out_cols > 0
        && first_out_cols <= i32::MAX as usize
        && second_out_cols > 0
        && second_out_cols <= i32::MAX as usize;
    if !eligible {
        return Ok(false);
    }
    let dense_row_bytes = in_cols
        .checked_mul(bf16_bytes)
        .ok_or_else(|| Error::Other("prefill pair dense row overflow".into()))?;
    let first_dense_bytes = first_out_cols
        .checked_mul(dense_row_bytes)
        .ok_or_else(|| Error::Other("prefill pair first dense overflow".into()))?;
    let second_dense_bytes = second_out_cols
        .checked_mul(dense_row_bytes)
        .ok_or_else(|| Error::Other("prefill pair second dense overflow".into()))?;
    let total_dense_bytes = first_dense_bytes
        .checked_add(second_dense_bytes)
        .ok_or_else(|| Error::Other("prefill pair dense overflow".into()))?;
    let activation_bytes = rows
        .checked_mul(dense_row_bytes)
        .ok_or_else(|| Error::Other("prefill pair activation overflow".into()))?;
    let first_output_bytes = rows
        .checked_mul(first_out_cols)
        .and_then(|value| value.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("prefill pair first output overflow".into()))?;
    let second_output_bytes = rows
        .checked_mul(second_out_cols)
        .and_then(|value| value.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("prefill pair second output overflow".into()))?;
    let packed_cols = in_cols / 8;
    let first_packed_bytes = first_out_cols
        .checked_mul(packed_cols)
        .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("prefill pair first packed overflow".into()))?;
    let second_packed_bytes = second_out_cols
        .checked_mul(packed_cols)
        .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("prefill pair second packed overflow".into()))?;
    let first_scale_bytes = first_out_cols
        .checked_mul(groups)
        .and_then(|value| value.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("prefill pair first scale overflow".into()))?;
    let second_scale_bytes = second_out_cols
        .checked_mul(groups)
        .and_then(|value| value.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("prefill pair second scale overflow".into()))?;
    let zp_rows = first_out_cols.div_ceil(8);
    let first_zp_bytes = zp_rows
        .checked_mul(groups)
        .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("prefill pair first zero-point overflow".into()))?;
    let second_zp_bytes = second_out_cols
        .div_ceil(8)
        .checked_mul(groups)
        .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("prefill pair second zero-point overflow".into()))?;
    if dense.len() < total_dense_bytes
        || activation.len() < activation_bytes
        || first_output.len() < first_output_bytes
        || second_output.len() < second_output_bytes
        || first_weight_packed.len() < first_packed_bytes
        || second_weight_packed.len() < second_packed_bytes
        || first_weight_scale.len() < first_scale_bytes
        || second_weight_scale.len() < second_scale_bytes
        || first_weight_zero_point.len() < first_zp_bytes
        || second_weight_zero_point.len() < second_zp_bytes
    {
        return Ok(false);
    }
    let first_dense = dense.view(0, first_dense_bytes).map_err(Error::Cuda)?;
    let second_dense = dense
        .view(first_dense_bytes, second_dense_bytes)
        .map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_qwen35_dequant_w4a16_bf16_pair_rows(
            first_weight_packed.ptr(),
            first_weight_scale.ptr(),
            first_weight_zero_point.ptr(),
            first_dense.ptr(),
            first_out_cols as i32,
            second_weight_packed.ptr(),
            second_weight_scale.ptr(),
            second_weight_zero_point.ptr(),
            second_dense.ptr(),
            second_out_cols as i32,
            in_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    matmul_bf16_transposed_into(
        ctx, activation, &first_dense, first_output, rows, in_cols, first_out_cols,
    )?;
    matmul_bf16_transposed_into(
        ctx, activation, &second_dense, second_output, rows, in_cols, second_out_cols,
    )?;
    Ok(true)
}

/// The fused short-prefill kernel materializes only block-local tiles.
/// It does not consume the caller's persistent dense scratch.
pub const W4A16_PREFILL_FAST_WORKSPACE_BYTES: usize = 0;

/// Attempts the opt-in fused W4 prefill path. `Ok(false)` means the physical
/// geometry is unsupported and the caller must use its established baseline.
/// Every geometry and buffer check completes before the first output write.
#[allow(clippy::too_many_arguments)]
pub fn try_matmul_bf16_w4a16_asym_prefill_fast_into(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    rows: usize,
    in_cols: usize,
    out_cols: usize,
    padded_out_cols: usize,
    groups: usize,
    repacked_v1: bool,
) -> Result<bool> {
    let supported = (1..=256).contains(&rows)
        && ctx.caps().compute_major >= 8
        && in_cols != 0
        && out_cols != 0
        && groups != 0
        && in_cols % 32 == 0
        && in_cols % groups == 0
        && in_cols / groups == 32
        && (!repacked_v1
            || (in_cols % 128 == 0
                && padded_out_cols >= out_cols
                && padded_out_cols % 64 == 0
                && padded_out_cols <= i32::MAX as usize))
        && rows <= i32::MAX as usize
        && in_cols <= i32::MAX as usize
        && out_cols <= i32::MAX as usize
        && groups <= i32::MAX as usize;
    if !supported {
        return Ok(false);
    }

    let bf16_bytes = DType::BF16.size_in_bytes();
    let physical_rows = if repacked_v1 { padded_out_cols } else { out_cols };
    let activation_bytes = rows
        .checked_mul(in_cols)
        .and_then(|v| v.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("prefill-fast activation overflow".into()))?;
    let output_bytes = rows
        .checked_mul(out_cols)
        .and_then(|v| v.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("prefill-fast output overflow".into()))?;
    let packed_bytes = physical_rows
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("prefill-fast packed weight overflow".into()))?;
    let scale_bytes = physical_rows
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("prefill-fast scale overflow".into()))?;
    let zp_rows = if repacked_v1 {
        physical_rows / 8
    } else {
        out_cols.div_ceil(8)
    };
    let zero_point_bytes = zp_rows
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("prefill-fast zero-point overflow".into()))?;
    if activation.device() != ctx.device_id()
        || weight_packed.device() != ctx.device_id()
        || weight_scale.device() != ctx.device_id()
        || weight_zero_point.device() != ctx.device_id()
        || output.device() != ctx.device_id()
    {
        return Err(Error::Other(
            "prefill-fast W4 buffers must share the CUDA context device".into(),
        ));
    }
    if activation.len() < activation_bytes
        || output.len() < output_bytes
        || weight_packed.len() < packed_bytes
        || weight_scale.len() < scale_bytes
        || weight_zero_point.len() < zero_point_bytes
    {
        return Err(Error::Other(
            "prefill-fast W4 input/output buffer too small".into(),
        ));
    }

    let status = unsafe {
        if repacked_v1 {
            ffi::apxinf_qwen35_gemm_w4a16_bf16_prefill_fast_repacked_v1(
                activation.ptr(),
                weight_packed.ptr(),
                weight_scale.ptr(),
                weight_zero_point.ptr(),
                output.ptr(),
                rows as i32,
                in_cols as i32,
                out_cols as i32,
                padded_out_cols as i32,
                groups as i32,
                ctx.stream().handle(),
            )
        } else {
            ffi::apxinf_qwen35_gemm_w4a16_bf16_prefill_fast_raw(
                activation.ptr(),
                weight_packed.ptr(),
                weight_scale.ptr(),
                weight_zero_point.ptr(),
                output.ptr(),
                rows as i32,
                in_cols as i32,
                out_cols as i32,
                groups as i32,
                ctx.stream().handle(),
            )
        }
    };
    unsafe { ffi::check_cuda(status) }.map_err(Error::Cuda)?;
    Ok(true)
}

/// Maximum row count accepted by the direct raw W4 prefill ABI.
pub const W4A16_PREFILL_NATIVE_MAX_ROWS: usize = 2048;

/// Attempts the raw-layout native W4 prefill path for up to the model CHUNK
/// bound. Grid tiling keeps per-CTA resources fixed as rows grow. This
/// candidate stays separate from decode and repacked-v1 routing so it cannot
/// change seq=1.
#[allow(clippy::too_many_arguments)]
pub fn try_matmul_bf16_w4a16_asym_prefill_native_into(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    rows: usize,
    in_cols: usize,
    out_cols: usize,
    groups: usize,
) -> Result<bool> {
    let supported = rows > 1
        && rows <= W4A16_PREFILL_NATIVE_MAX_ROWS
        && ctx.caps().compute_major >= 8
        && in_cols != 0
        && out_cols != 0
        && groups != 0
        && in_cols % 32 == 0
        && in_cols % groups == 0
        && in_cols / groups == 32
        && rows <= i32::MAX as usize
        && in_cols <= i32::MAX as usize
        && out_cols <= i32::MAX as usize
        && groups <= i32::MAX as usize;
    if !supported {
        return Ok(false);
    }

    let bf16_bytes = DType::BF16.size_in_bytes();
    let activation_bytes = rows
        .checked_mul(in_cols)
        .and_then(|value| value.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("prefill-native activation overflow".into()))?;
    let output_bytes = rows
        .checked_mul(out_cols)
        .and_then(|value| value.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("prefill-native output overflow".into()))?;
    let packed_bytes = out_cols
        .checked_mul(in_cols / 8)
        .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("prefill-native packed weight overflow".into()))?;
    let scale_bytes = out_cols
        .checked_mul(groups)
        .and_then(|value| value.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("prefill-native scale overflow".into()))?;
    let zero_point_bytes = out_cols
        .div_ceil(8)
        .checked_mul(groups)
        .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("prefill-native zero-point overflow".into()))?;
    if activation.device() != ctx.device_id()
        || weight_packed.device() != ctx.device_id()
        || weight_scale.device() != ctx.device_id()
        || weight_zero_point.device() != ctx.device_id()
        || output.device() != ctx.device_id()
    {
        return Err(Error::Other(
            "prefill-native W4 buffers must share the CUDA context device".into(),
        ));
    }
    if activation.len() < activation_bytes
        || output.len() < output_bytes
        || weight_packed.len() < packed_bytes
        || weight_scale.len() < scale_bytes
        || weight_zero_point.len() < zero_point_bytes
    {
        return Err(Error::Other(
            "prefill-native W4 input/output buffer too small".into(),
        ));
    }

    unsafe {
        ffi::check_cuda(ffi::apxinf_qwen35_gemm_w4a16_bf16_prefill_native(
            activation.ptr(),
            weight_packed.ptr(),
            weight_scale.ptr(),
            weight_zero_point.ptr(),
            output.ptr(),
            rows as i32,
            in_cols as i32,
            out_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(true)
}

/// Attempts the metadata-reusing packed W4 prefill path. It is restricted to
/// multi-row prefill so seq=1 decode never observes this candidate.
#[allow(clippy::too_many_arguments)]
pub fn try_matmul_bf16_w4a16_asym_prefill_packed_into(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    rows: usize,
    in_cols: usize,
    out_cols: usize,
    padded_out_cols: usize,
    groups: usize,
    repacked_v1: bool,
) -> Result<bool> {
    let supported = rows > 1
        && rows <= 256
        && ctx.caps().compute_major >= 8
        && in_cols != 0
        && out_cols != 0
        && groups != 0
        && in_cols % 32 == 0
        && in_cols % groups == 0
        && in_cols / groups == 32
        && (!repacked_v1
            || (in_cols % 128 == 0
                && padded_out_cols >= out_cols
                && padded_out_cols % 64 == 0
                && padded_out_cols <= i32::MAX as usize))
        && rows <= i32::MAX as usize
        && in_cols <= i32::MAX as usize
        && out_cols <= i32::MAX as usize
        && groups <= i32::MAX as usize;
    if !supported {
        return Ok(false);
    }
    let bf16_bytes = DType::BF16.size_in_bytes();
    let physical_rows = if repacked_v1 { padded_out_cols } else { out_cols };
    let activation_bytes = rows
        .checked_mul(in_cols)
        .and_then(|v| v.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("prefill-packed activation overflow".into()))?;
    let output_bytes = rows
        .checked_mul(out_cols)
        .and_then(|v| v.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("prefill-packed output overflow".into()))?;
    let packed_bytes = physical_rows
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("prefill-packed weight overflow".into()))?;
    let scale_bytes = physical_rows
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("prefill-packed scale overflow".into()))?;
    let zp_rows = if repacked_v1 { physical_rows / 8 } else { out_cols.div_ceil(8) };
    let zero_point_bytes = zp_rows
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("prefill-packed zero-point overflow".into()))?;
    if activation.device() != ctx.device_id()
        || weight_packed.device() != ctx.device_id()
        || weight_scale.device() != ctx.device_id()
        || weight_zero_point.device() != ctx.device_id()
        || output.device() != ctx.device_id()
    {
        return Err(Error::Other(
            "prefill-packed W4 buffers must share the CUDA context device".into(),
        ));
    }
    if activation.len() < activation_bytes
        || output.len() < output_bytes
        || weight_packed.len() < packed_bytes
        || weight_scale.len() < scale_bytes
        || weight_zero_point.len() < zero_point_bytes
    {
        return Err(Error::Other(
            "prefill-packed W4 input/output buffer too small".into(),
        ));
    }
    let status = unsafe {
        if repacked_v1 {
            ffi::apxinf_qwen35_gemm_w4a16_bf16_prefill_packed_repacked_v1(
                activation.ptr(), weight_packed.ptr(), weight_scale.ptr(),
                weight_zero_point.ptr(), output.ptr(), rows as i32,
                in_cols as i32, out_cols as i32, padded_out_cols as i32,
                groups as i32, ctx.stream().handle(),
            )
        } else {
            ffi::apxinf_qwen35_gemm_w4a16_bf16_prefill_packed_raw(
                activation.ptr(), weight_packed.ptr(), weight_scale.ptr(),
                weight_zero_point.ptr(), output.ptr(), rows as i32,
                in_cols as i32, out_cols as i32, groups as i32,
                ctx.stream().handle(),
            )
        }
    };
    unsafe { ffi::check_cuda(status) }.map_err(Error::Cuda)?;
    Ok(true)
}

/// W4_REPACKED_N64_K16_V1 prefill GEMM. The exact Qwen chunk geometry
/// (`M=512`, group-32, aligned K/N) uses the packed tensor-core kernel. Other
/// prefill row counts reconstruct bounded dense output-row tiles from the
/// explicit physical layout and use the established FP32-accumulating cuBLAS
/// path. No branch interprets a repacked buffer as canonical HF packing.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_prefill_repacked_v1_into(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_qwords: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    dense: &CudaBuffer,
    output: &CudaBuffer,
    rows: usize,
    in_cols: usize,
    out_cols: usize,
    padded_out_cols: usize,
    groups: usize,
) -> Result<()> {
    let bf16_bytes = DType::BF16.size_in_bytes();
    if rows <= 1
        || in_cols == 0
        || out_cols == 0
        || groups == 0
        || in_cols % 128 != 0
        || padded_out_cols < out_cols
        || padded_out_cols % 64 != 0
        || in_cols / groups != 32
        || rows > i32::MAX as usize
        || in_cols > i32::MAX as usize
        || out_cols > i32::MAX as usize
        || padded_out_cols > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other(
            "repacked-v1 prefill GEMM: invalid physical geometry".into(),
        ));
    }
    let activation_bytes = rows
        .checked_mul(in_cols)
        .and_then(|v| v.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("repacked-v1 prefill activation overflow".into()))?;
    let output_bytes = rows
        .checked_mul(out_cols)
        .and_then(|v| v.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("repacked-v1 prefill output overflow".into()))?;
    let qword_bytes = padded_out_cols
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("repacked-v1 prefill qword overflow".into()))?;
    let scale_bytes = padded_out_cols
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Other("repacked-v1 prefill scale overflow".into()))?;
    let zero_point_bytes = (padded_out_cols / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("repacked-v1 prefill zero-point overflow".into()))?;
    if activation.len() < activation_bytes
        || output.len() < output_bytes
        || weight_qwords.len() < qword_bytes
        || weight_scale.len() < scale_bytes
        || weight_zero_point.len() < zero_point_bytes
    {
        return Err(Error::Other(
            "repacked-v1 prefill GEMM: input/output buffer too small".into(),
        ));
    }

    let exact_qwen_shape = matches!(
        (in_cols, out_cols),
        (5120, 48)
            | (5120, 1024)
            | (5120, 6144)
            | (5120, 10240)
            | (5120, 12288)
            | (5120, 17408)
            | (6144, 5120)
            | (12288, 5120)
            | (17408, 5120)
    );

    if rows == 512 && exact_qwen_shape {
        return unsafe {
            ffi::check_cuda(
                ffi::apxinf_qwen35_gemm_w4a16_bf16_prefill_repacked_v1(
                    activation.ptr(),
                    weight_qwords.ptr(),
                    weight_scale.ptr(),
                    weight_zero_point.ptr(),
                    output.ptr(),
                    rows as i32,
                    in_cols as i32,
                    out_cols as i32,
                    padded_out_cols as i32,
                    groups as i32,
                    ctx.stream().handle(),
                ),
            )
            .map_err(Error::Cuda)
        };
    }

    let dense_row_bytes = in_cols
        .checked_mul(bf16_bytes)
        .ok_or_else(|| Error::Other("repacked-v1 prefill dense row overflow".into()))?;
    let tile_rows = dense.len().min(W4A16_PREFILL_DENSE_TILE_BYTES) / dense_row_bytes;
    if tile_rows == 0 {
        return Err(Error::Other(
            "repacked-v1 prefill GEMM: dense scratch too small for fallback".into(),
        ));
    }
    let tile_capacity = tile_rows.min(out_cols);
    for out_start in (0..out_cols).step_by(tile_capacity) {
        let out_count = tile_capacity.min(out_cols - out_start);
        let tile_bytes = out_count
            .checked_mul(dense_row_bytes)
            .ok_or_else(|| Error::Other("repacked-v1 prefill tile overflow".into()))?;
        let dense_tile = dense.view(0, tile_bytes).map_err(Error::Cuda)?;
        let output_offset = out_start
            .checked_mul(bf16_bytes)
            .ok_or_else(|| Error::Other("repacked-v1 prefill output offset overflow".into()))?;
        let output_tile = output
            .view(output_offset, output_bytes - output_offset)
            .map_err(Error::Cuda)?;
        unsafe {
            ffi::check_cuda(
                ffi::apxinf_qwen35_dequant_w4a16_bf16_rows_repacked_v1(
                    weight_qwords.ptr(),
                    weight_scale.ptr(),
                    weight_zero_point.ptr(),
                    dense_tile.ptr(),
                    in_cols as i32,
                    out_cols as i32,
                    groups as i32,
                    out_start as i32,
                    out_count as i32,
                    padded_out_cols as i32,
                    ctx.stream().handle(),
                ),
            )
            .map_err(Error::Cuda)?;
        }
        super::gemm::write_ex(
            ctx,
            DType::BF16,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            rows,
            out_count,
            in_cols,
            1.0,
            activation,
            in_cols as i32,
            &dense_tile,
            in_cols as i32,
            0.0,
            &output_tile,
            out_cols as i32,
        )?;
    }
    Ok(())
}

/// BF16 activation (row-major `[rows, in_cols]`) times transposed dense BF16
/// weight (`[out_cols, in_cols]`) into caller-owned output storage.
pub fn matmul_bf16_transposed_into(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    rows: usize,
    in_cols: usize,
    out_cols: usize,
) -> Result<()> {
    let out_bytes = rows
        .checked_mul(out_cols)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("matmul_bf16_transposed_into output overflow".into()))?;
    if output.len() < out_bytes {
        return Err(Error::Other("matmul_bf16_transposed_into: buffer too small".into()));
    }
    super::gemm::write_ex(
        ctx,
        DType::BF16,
        CublasTranspose::None,
        CublasTranspose::Transpose,
        rows,
        out_cols,
        in_cols,
        1.0,
        activation,
        in_cols as i32,
        weight,
        in_cols as i32,
        0.0,
        output,
        out_cols as i32,
    )
}

pub fn matmul_bf16_transposed(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
) -> Result<Tensor> {
    if activation.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return Err(Error::Other("matmul_bf16_transposed dtype mismatch".into()));
    }
    let a_dims = activation.shape().dims();
    let w_dims = weight.shape().dims();
    if a_dims.len() != 2 || w_dims.len() != 2 || a_dims[1] != w_dims[1] {
        return Err(Error::Other(format!(
            "matmul_bf16_transposed activation {a_dims:?} incompatible with weight {w_dims:?}"
        )));
    }
    let rows = a_dims[0];
    let in_cols = a_dims[1];
    let out_cols = w_dims[0];
    let bytes = rows
        .checked_mul(out_cols)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("matmul_bf16_transposed output overflow".into()))?;
    let output = CudaBuffer::alloc_zeros(bytes, ctx.device_id()).map_err(Error::Cuda)?;
    let a = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let w = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    super::gemm::write_ex(
        ctx,
        DType::BF16,
        CublasTranspose::None,
        CublasTranspose::Transpose,
        rows,
        out_cols,
        in_cols,
        1.0,
        &a,
        in_cols as i32,
        &w,
        in_cols as i32,
        0.0,
        &output,
        out_cols as i32,
    )?;
    Ok(output.into_tensor(
        apxinf_core::Shape::from(vec![rows, out_cols]),
        DType::BF16,
    ))
}

    // Allocation-free single-row variant: writes straight into caller storage
    // so decode steps never touch cudaMalloc (which degrades after prefill
    // churns large transient buffers).
    #[allow(clippy::too_many_arguments)]
    /// Decode GEMM on tensor cores: [1, in_cols] x W4A16 -> [1, out_cols].
pub fn matmul_bf16_w4a16_asym_tc(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    in_cols: usize,
    out_cols: usize,
    groups: usize,
) -> Result<()> {
    let out_bytes = out_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("TC decode GEMM overflow".into()))?;
    if output.len() < out_bytes {
        return Err(Error::Other("TC decode GEMM: output buffer too small".into()));
    }
    let prof = std::env::var_os("APXINF_GEMM_PROF").is_some();
    let mut e0: ffi::cudaEvent_t = std::ptr::null_mut();
    let mut e1: ffi::cudaEvent_t = std::ptr::null_mut();
    if prof {
        unsafe {
            ffi::check_cuda(ffi::cudaEventCreate(&mut e0)).map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaEventCreate(&mut e1)).map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaEventRecord(e0, ctx.stream().handle()))
                .map_err(Error::Cuda)?;
        }
    }
    let result = check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc(
            activation.ptr(),
            weight_packed.ptr(),
            weight_scale.ptr(),
            weight_zero_point.ptr(),
            output.ptr(),
            in_cols as i32,
            out_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        )
    });
    if prof {
        unsafe {
            ffi::check_cuda(ffi::cudaEventRecord(e1, ctx.stream().handle()))
                .map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaEventSynchronize(e1)).map_err(Error::Cuda)?;
            let mut ms: f32 = 0.0;
            ffi::check_cuda(ffi::cudaEventElapsedTime(&mut ms, e0, e1))
                .map_err(Error::Cuda)?;
            if ms > 0.01 {
                eprintln!("[tc_gemm] {}x{} : {ms:.3} ms", out_cols, in_cols);
            }
            ffi::check_cuda(ffi::cudaEventDestroy(e0)).map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaEventDestroy(e1)).map_err(Error::Cuda)?;
        }
    }
    result
}

/// Opt-in raw-layout decode GEMM using read-only cache hints for packed W4
/// weights, scales, and zero points while preserving the baseline arithmetic.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_cache_hint(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    in_cols: usize,
    out_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols == 0
        || out_cols % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("cache-hint TC decode GEMM invalid geometry".into()));
    }
    let out_bytes = out_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("cache-hint TC decode GEMM overflow".into()))?;
    if output.len() < out_bytes {
        return Err(Error::Other(
            "cache-hint TC decode GEMM output buffer too small".into(),
        ));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_cache_hint(
            activation.ptr(),
            weight_packed.ptr(),
            weight_scale.ptr(),
            weight_zero_point.ptr(),
            output.ptr(),
            in_cols as i32,
            out_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        )
    })
}

/// Opt-in single-row W4A16 decode with exact baseline arithmetic and packed
/// adjacent BF16 output stores.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_store_alt_single(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    in_cols: usize,
    out_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols == 0
        || out_cols % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("store-alt TC decode GEMM invalid geometry".into()));
    }
    let out_bytes = out_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("store-alt TC decode GEMM overflow".into()))?;
    if output.len() < out_bytes {
        return Err(Error::Other("store-alt TC decode GEMM output buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_store_alt_single(
            activation.ptr(),
            weight_packed.ptr(),
            weight_scale.ptr(),
            weight_zero_point.ptr(),
            output.ptr(),
            in_cols as i32,
            out_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        )
    })
}
/// Exact raw-layout single-row W4A16 decode using vectorized BF16 and packed
/// word loads. The CUDA ABI enforces group-32, K-128-aligned geometry.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_vector_mma(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    in_cols: usize,
    out_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols == 0
        || out_cols % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("vector-MMA TC decode GEMM invalid geometry".into()));
    }
    let out_bytes = out_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("vector-MMA TC decode GEMM overflow".into()))?;
    if output.len() < out_bytes {
        return Err(Error::Other(
            "vector-MMA TC decode GEMM: output buffer too small".into(),
        ));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_vector_mma(
            activation.ptr(),
            weight_packed.ptr(),
            weight_scale.ptr(),
            weight_zero_point.ptr(),
            output.ptr(),
            in_cols as i32,
            out_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        )
    })
}


/// Opt-in raw-layout W4A16 decode with scale and zero-point fused into the
/// exact BF16 fragment transform. The MMA and FP32 accumulator order matches
/// the baseline tensor-core kernel.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_scale_epilogue(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    in_cols: usize,
    out_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols == 0
        || out_cols % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("scale-epilogue TC decode GEMM invalid geometry".into()));
    }
    let out_bytes = out_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("scale-epilogue TC decode GEMM overflow".into()))?;
    if output.len() < out_bytes {
        return Err(Error::Other(
            "scale-epilogue TC decode GEMM output buffer too small".into(),
        ));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_scale_epilogue(
            activation.ptr(),
            weight_packed.ptr(),
            weight_scale.ptr(),
            weight_zero_point.ptr(),
            output.ptr(),
            in_cols as i32,
            out_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        )
    })
}

/// Shape-specific exact raw W4 decode for linear-attention in_proj_qkv.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_qkv_10240x5120(
    ctx: &CudaContext, activation: &CudaBuffer, weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer, weight_zero_point: &CudaBuffer,
    output: &CudaBuffer, in_cols: usize, out_cols: usize, groups: usize,
) -> Result<()> {
    if (in_cols, out_cols, groups) != (5120, 10240, 160)
        || output.len() < 10240 * DType::BF16.size_in_bytes()
    {
        return Err(Error::Other("raw qkv decode GEMM invalid geometry".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_qkv_10240x5120(
            activation.ptr(), weight_packed.ptr(), weight_scale.ptr(),
            weight_zero_point.ptr(), output.ptr(), in_cols as i32,
            out_cols as i32, groups as i32, ctx.stream().handle(),
        )
    })
}

/// Iteration-33 exact QKV scheduling matrix. `mode` selects rows/CTA and
/// independent activation/metadata stage counts without changing arithmetic.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_qkv_sched(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    in_cols: usize,
    out_cols: usize,
    groups: usize,
    mode: usize,
) -> Result<()> {
    if (in_cols, out_cols, groups) != (5120, 10240, 160)
        || mode > 7
        || output.len() < 10240 * DType::BF16.size_in_bytes()
    {
        return Err(Error::Other("QKV scheduling candidate invalid geometry".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_qkv_sched(
            activation.ptr(), weight_packed.ptr(), weight_scale.ptr(),
            weight_zero_point.ptr(), output.ptr(), in_cols as i32,
            out_cols as i32, groups as i32, mode as i32,
            ctx.stream().handle(),
        )
    })
}

/// Persistent-device transform-cache W4A16 decode. The caller owns explicit
/// N64/K16 metadata and reuses it across tokens without a per-token copy.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_w4_transform_cache(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    in_cols: usize,
    out_cols: usize,
    padded_out_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0 || in_cols % 128 != 0 || out_cols == 0
        || padded_out_cols < out_cols || padded_out_cols % 64 != 0
        || groups == 0 || in_cols % groups != 0 || in_cols / groups != 32 {
        return Err(Error::Other("transform-cache TC decode GEMM invalid geometry".into()));
    }

    let out_bytes = out_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("transform-cache TC decode GEMM overflow".into()))?;
    let expected_packed = padded_out_cols
        .checked_mul(in_cols / 8).and_then(|v| v.checked_mul(4))
        .ok_or_else(|| Error::Other("transform-cache packed bytes overflow".into()))?;
    let expected_scale = padded_out_cols
        .checked_mul(groups).and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("transform-cache scale bytes overflow".into()))?;
    let expected_zp = (padded_out_cols / 8)
        .checked_mul(groups).and_then(|v| v.checked_mul(4))
        .ok_or_else(|| Error::Other("transform-cache zero-point bytes overflow".into()))?;
    if output.len() < out_bytes || weight_packed.len() != expected_packed
        || weight_scale.len() != expected_scale || weight_zero_point.len() != expected_zp {
        return Err(Error::Other("transform-cache buffer/layout mismatch".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_w4_transform_cache(
            activation.ptr(), weight_packed.ptr(), weight_scale.ptr(),
            weight_zero_point.ptr(), output.ptr(), in_cols as i32, out_cols as i32,
            padded_out_cols as i32, groups as i32, ctx.stream().handle(),
        )
    })
}
/// Exact batch-one gate/up/SILU fusion for raw asymmetric U4 group-32 weights.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_gate_up_silu(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    gate_packed: &CudaBuffer,
    gate_scale: &CudaBuffer,
    gate_zero_point: &CudaBuffer,
    up_packed: &CudaBuffer,
    up_scale: &CudaBuffer,
    up_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    in_cols: usize,
    out_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols != 5120 || out_cols != 17408 || groups != 160 {
        return Err(Error::Other("fused gate/up/SILU unsupported geometry".into()));
    }
    let packed_bytes = out_cols * (in_cols / 8) * 4;
    let scale_bytes = out_cols * groups * 2;
    let zero_point_bytes = (out_cols / 8) * groups * 4;
    if activation.len() < in_cols * 2 || output.len() < out_cols * 2
        || gate_packed.len() != packed_bytes || up_packed.len() != packed_bytes
        || gate_scale.len() != scale_bytes || up_scale.len() != scale_bytes
        || gate_zero_point.len() != zero_point_bytes
        || up_zero_point.len() != zero_point_bytes
    {
        return Err(Error::Other("fused gate/up/SILU buffer mismatch".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_gate_up_silu(
            activation.ptr(), gate_packed.ptr(), gate_scale.ptr(),
            gate_zero_point.ptr(), up_packed.ptr(), up_scale.ptr(),
            up_zero_point.ptr(), output.ptr(), in_cols as i32,
            out_cols as i32, groups as i32, ctx.stream().handle(),
        )
    })
}

/// Opt-in raw-layout W4A16 decode using a 32-output, four-warp CTA.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_tile_alt(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    in_cols: usize,
    out_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols == 0
        || out_cols % 32 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("tile-alt TC decode GEMM invalid geometry".into()));
    }
    let out_bytes = out_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("tile-alt TC decode GEMM overflow".into()))?;
    if output.len() < out_bytes {
        return Err(Error::Other(
            "tile-alt TC decode GEMM: output buffer too small".into(),
        ));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_tile_alt(
            activation.ptr(),
            weight_packed.ptr(),
            weight_scale.ptr(),
            weight_zero_point.ptr(),
            output.ptr(),
            in_cols as i32,
            out_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        )
    })
}

/// Exact tile-alt raw W4 decode launch on an explicitly supplied stream.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_tile_alt_on_stream(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    in_cols: usize,
    out_cols: usize,
    groups: usize,
    stream: &crate::CudaStream,
) -> Result<()> {
    if in_cols == 0 || in_cols % 128 != 0 || out_cols == 0 || out_cols % 32 != 0
        || groups == 0 || in_cols % groups != 0 || in_cols / groups != 32
        || activation.device() != ctx.device_id() || output.device() != ctx.device_id()
    {
        return Err(Error::Other("side-stream tile-alt invalid geometry".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_tile_alt(
            activation.ptr(), weight_packed.ptr(), weight_scale.ptr(),
            weight_zero_point.ptr(), output.ptr(), in_cols as i32,
            out_cols as i32, groups as i32, stream.handle(),
        )
    })
}

/// Raw-layout single-row W4A16 decode with exact per-row metadata staged in
/// shared memory. The group-32 geometry is required by the four-group K tile.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_meta_shared(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    in_cols: usize,
    out_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols == 0
        || out_cols % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("metadata-shared TC decode GEMM invalid geometry".into()));
    }
    let out_bytes = out_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("metadata-shared TC decode GEMM overflow".into()))?;
    if output.len() < out_bytes {
        return Err(Error::Other(
            "metadata-shared TC decode GEMM: output buffer too small".into(),
        ));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_meta_shared(
            activation.ptr(),
            weight_packed.ptr(),
            weight_scale.ptr(),
            weight_zero_point.ptr(),
            output.ptr(),
            in_cols as i32,
            out_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        )
    })
}

/// Raw-layout single-row W4A16 candidate with two output tiles per CTA.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_persistent(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    in_cols: usize,
    out_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols < 128
        || out_cols % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("persistent TC decode GEMM invalid geometry".into()));
    }
    let out_bytes = out_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("persistent TC decode GEMM overflow".into()))?;
    if output.len() < out_bytes {
        return Err(Error::Other(
            "persistent TC decode GEMM: output buffer too small".into(),
        ));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_persistent(
            activation.ptr(),
            weight_packed.ptr(),
            weight_scale.ptr(),
            weight_zero_point.ptr(),
            output.ptr(),
            in_cols as i32,
            out_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        )
    })
}

/// Decode GEMM for `W4_REPACKED_N64_K16_V1`. This is intentionally a
/// distinct ABI: raw compressed-tensors pointers must never reach this launch.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_repacked_v1(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    output: &CudaBuffer,
    in_cols: usize,
    out_cols: usize,
    padded_out_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols == 0
        || padded_out_cols < out_cols
        || padded_out_cols % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols > i32::MAX as usize
        || padded_out_cols > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("repacked TC decode GEMM invalid physical geometry".into()));
    }
    let out_bytes = out_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("repacked TC decode GEMM overflow".into()))?;
    let expected_packed = padded_out_cols
        .checked_mul(in_cols / 8)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| Error::Other("repacked TC packed bytes overflow".into()))?;
    let expected_scale = padded_out_cols
        .checked_mul(groups)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| Error::Other("repacked TC scale bytes overflow".into()))?;
    let expected_zp = (padded_out_cols / 8)
        .checked_mul(groups)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| Error::Other("repacked TC zero-point bytes overflow".into()))?;
    if output.len() < out_bytes
        || weight_packed.len() != expected_packed
        || weight_scale.len() != expected_scale
        || weight_zero_point.len() != expected_zp
    {
        return Err(Error::Other("repacked TC decode GEMM buffer/layout mismatch".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_repacked_v1(
            activation.ptr(),
            weight_packed.ptr(),
            weight_scale.ptr(),
            weight_zero_point.ptr(),
            output.ptr(),
            in_cols as i32,
            out_cols as i32,
            padded_out_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        )
    })
}

/// Runs two independent single-row W4A16 projections in one CUDA launch.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    let out_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("paired TC decode GEMM output overflow".into()))?;
    let out_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("paired TC decode GEMM output overflow".into()))?;
    if output0.len() < out_bytes0 || output1.len() < out_bytes1 {
        return Err(Error::Other(
            "paired TC decode GEMM: output buffer too small".into(),
        ));
    }
    let prof = std::env::var_os("APXINF_GEMM_PROF").is_some();
    let mut e0: ffi::cudaEvent_t = std::ptr::null_mut();
    let mut e1: ffi::cudaEvent_t = std::ptr::null_mut();
    if prof {
        unsafe {
            ffi::check_cuda(ffi::cudaEventCreate(&mut e0)).map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaEventCreate(&mut e1)).map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaEventRecord(e0, ctx.stream().handle()))
                .map_err(Error::Cuda)?;
        }
    }
    let result = check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair(
            activation.ptr(),
            weight_packed0.ptr(),
            weight_scale0.ptr(),
            weight_zero_point0.ptr(),
            output0.ptr(),
            out_cols0 as i32,
            weight_packed1.ptr(),
            weight_scale1.ptr(),
            weight_zero_point1.ptr(),
            output1.ptr(),
            out_cols1 as i32,
            in_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        )
    });
    if prof {
        unsafe {
            ffi::check_cuda(ffi::cudaEventRecord(e1, ctx.stream().handle()))
                .map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaEventSynchronize(e1)).map_err(Error::Cuda)?;
            let mut ms = 0.0;
            ffi::check_cuda(ffi::cudaEventElapsedTime(&mut ms, e0, e1))
                .map_err(Error::Cuda)?;
            if ms > 0.01 {
                eprintln!(
                    "[tc_gemm_pair] {}+{}x{} : {ms:.3} ms",
                    out_cols0, out_cols1, in_cols
                );
            }
            ffi::check_cuda(ffi::cudaEventDestroy(e0)).map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaEventDestroy(e1)).map_err(Error::Cuda)?;
        }
    }
    result
}

/// Opt-in paired W4A16 decode with exact baseline arithmetic and coalesced
/// adjacent BF16 output stores.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_store_alt(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols0 == 0
        || out_cols1 == 0
        || out_cols0 % 64 != 0
        || out_cols1 % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols0 > i32::MAX as usize
        || out_cols1 > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("store-alt paired TC decode GEMM invalid geometry".into()));
    }
    let out_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("store-alt paired output overflow".into()))?;
    let out_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("store-alt paired output overflow".into()))?;
    if output0.len() < out_bytes0 || output1.len() < out_bytes1 {
        return Err(Error::Other("store-alt paired output buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_store_alt(
            activation.ptr(),
            weight_packed0.ptr(),
            weight_scale0.ptr(),
            weight_zero_point0.ptr(),
            output0.ptr(),
            out_cols0 as i32,
            weight_packed1.ptr(),
            weight_scale1.ptr(),
            weight_zero_point1.ptr(),
            output1.ptr(),
            out_cols1 as i32,
            in_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        )
    })
}

/// Opt-in paired raw-layout decode where each CTA computes two adjacent
/// 64-row tiles while sharing the exact activation tile.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair_coarsen(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols0 == 0
        || out_cols1 == 0
        || out_cols0 % 128 != 0
        || out_cols1 % 128 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols0 > (i32::MAX as usize - 127)
        || out_cols1 > (i32::MAX as usize - 127)
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("coarsened paired TC decode GEMM invalid geometry".into()));
    }
    let activation_bytes = in_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("coarsened paired TC activation overflow".into()))?;
    let packed_bytes0 = out_cols0
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("coarsened paired TC packed weight overflow".into()))?;
    let packed_bytes1 = out_cols1
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("coarsened paired TC packed weight overflow".into()))?;
    let scale_bytes0 = out_cols0
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("coarsened paired TC scale overflow".into()))?;
    let scale_bytes1 = out_cols1
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("coarsened paired TC scale overflow".into()))?;
    let zp_bytes0 = (out_cols0 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("coarsened paired TC zero-point overflow".into()))?;
    let zp_bytes1 = (out_cols1 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("coarsened paired TC zero-point overflow".into()))?;
    let output_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("coarsened paired TC output overflow".into()))?;
    let output_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("coarsened paired TC output overflow".into()))?;
    if activation.len() < activation_bytes
        || weight_packed0.len() < packed_bytes0
        || weight_packed1.len() < packed_bytes1
        || weight_scale0.len() < scale_bytes0
        || weight_scale1.len() < scale_bytes1
        || weight_zero_point0.len() < zp_bytes0
        || weight_zero_point1.len() < zp_bytes1
        || output0.len() < output_bytes0
        || output1.len() < output_bytes1
    {
        return Err(Error::Other("coarsened paired TC decode buffer/layout mismatch".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair_coarsen(
            activation.ptr(),
            weight_packed0.ptr(), weight_scale0.ptr(), weight_zero_point0.ptr(),
            output0.ptr(), out_cols0 as i32,
            weight_packed1.ptr(), weight_scale1.ptr(), weight_zero_point1.ptr(),
            output1.ptr(), out_cols1 as i32,
            in_cols as i32, groups as i32, ctx.stream().handle(),
        )
    })
}

/// Opt-in paired decode with dedicated loader warps staging exact B operands
/// for the baseline eight compute warps.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair_warp(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    let supported_in = matches!(in_cols, 5120 | 6144 | 17408);
    let supported_out0 = matches!(out_cols0, 1024 | 5120 | 6144 | 10240 | 12288 | 17408);
    let supported_out1 = matches!(out_cols1, 1024 | 5120 | 6144 | 10240 | 12288 | 17408);
    if !supported_in
        || !supported_out0
        || !supported_out1
        || out_cols0 % 64 != 0
        || out_cols1 % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols0 > i32::MAX as usize
        || out_cols1 > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("warp-specialized paired TC decode GEMM invalid geometry".into()));
    }
    let activation_bytes = in_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("warp-specialized paired TC activation overflow".into()))?;
    let packed_bytes0 = out_cols0
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("warp-specialized paired TC packed weight overflow".into()))?;
    let packed_bytes1 = out_cols1
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("warp-specialized paired TC packed weight overflow".into()))?;
    let scale_bytes0 = out_cols0
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("warp-specialized paired TC scale overflow".into()))?;
    let scale_bytes1 = out_cols1
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("warp-specialized paired TC scale overflow".into()))?;
    let zp_bytes0 = (out_cols0 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("warp-specialized paired TC zero-point overflow".into()))?;
    let zp_bytes1 = (out_cols1 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("warp-specialized paired TC zero-point overflow".into()))?;
    let output_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("warp-specialized paired TC output overflow".into()))?;
    let output_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("warp-specialized paired TC output overflow".into()))?;
    if activation.len() < activation_bytes
        || weight_packed0.len() < packed_bytes0
        || weight_packed1.len() < packed_bytes1
        || weight_scale0.len() < scale_bytes0
        || weight_scale1.len() < scale_bytes1
        || weight_zero_point0.len() < zp_bytes0
        || weight_zero_point1.len() < zp_bytes1
        || output0.len() < output_bytes0
        || output1.len() < output_bytes1
    {
        return Err(Error::Other("warp-specialized paired TC decode buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair_warp(
            activation.ptr(),
            weight_packed0.ptr(), weight_scale0.ptr(), weight_zero_point0.ptr(),
            output0.ptr(), out_cols0 as i32,
            weight_packed1.ptr(), weight_scale1.ptr(), weight_zero_point1.ptr(),
            output1.ptr(), out_cols1 as i32,
            in_cols as i32, groups as i32, ctx.stream().handle(),
        )
    })
}

/// Opt-in baseline paired decode with a source-local compiler resource hint.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair_reg(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols0 == 0
        || out_cols1 == 0
        || out_cols0 % 64 != 0
        || out_cols1 % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols0 > i32::MAX as usize
        || out_cols1 > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("register-tuned paired TC decode GEMM invalid geometry".into()));
    }
    let activation_bytes = in_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("register-tuned paired TC activation overflow".into()))?;
    let packed_bytes0 = out_cols0
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("register-tuned paired TC packed weight overflow".into()))?;
    let packed_bytes1 = out_cols1
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("register-tuned paired TC packed weight overflow".into()))?;
    let scale_bytes0 = out_cols0
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("register-tuned paired TC scale overflow".into()))?;
    let scale_bytes1 = out_cols1
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("register-tuned paired TC scale overflow".into()))?;
    let zp_bytes0 = (out_cols0 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("register-tuned paired TC zero-point overflow".into()))?;
    let zp_bytes1 = (out_cols1 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("register-tuned paired TC zero-point overflow".into()))?;
    let output_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("register-tuned paired TC output overflow".into()))?;
    let output_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("register-tuned paired TC output overflow".into()))?;
    if activation.len() < activation_bytes
        || weight_packed0.len() < packed_bytes0
        || weight_packed1.len() < packed_bytes1
        || weight_scale0.len() < scale_bytes0
        || weight_scale1.len() < scale_bytes1
        || weight_zero_point0.len() < zp_bytes0
        || weight_zero_point1.len() < zp_bytes1
        || output0.len() < output_bytes0
        || output1.len() < output_bytes1
    {
        return Err(Error::Other("register-tuned paired TC quantized buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair_reg(
            activation.ptr(),
            weight_packed0.ptr(), weight_scale0.ptr(), weight_zero_point0.ptr(),
            output0.ptr(), out_cols0 as i32,
            weight_packed1.ptr(), weight_scale1.ptr(), weight_zero_point1.ptr(),
            output1.ptr(), out_cols1 as i32,
            in_cols as i32, groups as i32, ctx.stream().handle(),
        )
    })
}
/// Opt-in paired raw W4A16 decode with a source-local occupancy resource hint.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair_occupancy(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols0 == 0
        || out_cols1 == 0
        || out_cols0 % 64 != 0
        || out_cols1 % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols0 > i32::MAX as usize
        || out_cols1 > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("occupancy-tuned paired TC decode GEMM invalid geometry".into()));
    }
    let activation_bytes = in_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("occupancy paired TC activation overflow".into()))?;
    let packed_bytes0 = out_cols0
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("occupancy paired TC packed weight overflow".into()))?;
    let packed_bytes1 = out_cols1
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("occupancy paired TC packed weight overflow".into()))?;
    let scale_bytes0 = out_cols0
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("occupancy paired TC scale overflow".into()))?;
    let scale_bytes1 = out_cols1
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("occupancy paired TC scale overflow".into()))?;
    let zp_bytes0 = (out_cols0 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("occupancy paired TC zero-point overflow".into()))?;
    let zp_bytes1 = (out_cols1 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("occupancy paired TC zero-point overflow".into()))?;
    let output_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("occupancy paired TC output overflow".into()))?;
    let output_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("occupancy paired TC output overflow".into()))?;
    if activation.len() < activation_bytes
        || weight_packed0.len() < packed_bytes0
        || weight_packed1.len() < packed_bytes1
        || weight_scale0.len() < scale_bytes0
        || weight_scale1.len() < scale_bytes1
        || weight_zero_point0.len() < zp_bytes0
        || weight_zero_point1.len() < zp_bytes1
        || output0.len() < output_bytes0
        || output1.len() < output_bytes1
    {
        return Err(Error::Other("occupancy paired TC quantized buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair_occupancy(
            activation.ptr(),
            weight_packed0.ptr(), weight_scale0.ptr(), weight_zero_point0.ptr(),
            output0.ptr(), out_cols0 as i32,
            weight_packed1.ptr(), weight_scale1.ptr(), weight_zero_point1.ptr(),
            output1.ptr(), out_cols1 as i32,
            in_cols as i32, groups as i32, ctx.stream().handle(),
        )
    })
}

/// Opt-in paired raw-layout W4A16 decode with SM80 activation prefetch.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair_prefetch(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0 || in_cols % 128 != 0 || out_cols0 == 0 || out_cols1 == 0
        || out_cols0 % 64 != 0 || out_cols1 % 64 != 0 || groups == 0
        || in_cols % groups != 0 || in_cols / groups != 32
        || in_cols > i32::MAX as usize || out_cols0 > i32::MAX as usize
        || out_cols1 > i32::MAX as usize || groups > i32::MAX as usize
    {
        return Err(Error::Other("paired prefetch TC decode GEMM invalid geometry".into()));
    }
    let activation_bytes = in_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("paired prefetch activation overflow".into()))?;
    let out_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("paired prefetch output overflow".into()))?;
    let out_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("paired prefetch output overflow".into()))?;
    if activation.len() < activation_bytes || output0.len() < out_bytes0
        || output1.len() < out_bytes1
    {
        return Err(Error::Other("paired prefetch TC buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair_prefetch(
            activation.ptr(), weight_packed0.ptr(), weight_scale0.ptr(),
            weight_zero_point0.ptr(), output0.ptr(), out_cols0 as i32,
            weight_packed1.ptr(), weight_scale1.ptr(), weight_zero_point1.ptr(),
            output1.ptr(), out_cols1 as i32, in_cols as i32, groups as i32,
            ctx.stream().handle(),
        )
    })
}

/// Exact paired raw-layout W4A16 decode using vectorized BF16 and packed-word
/// loads while retaining the baseline paired block routing.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair_vector_mma(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols0 == 0
        || out_cols1 == 0
        || out_cols0 % 64 != 0
        || out_cols1 % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols0 > i32::MAX as usize
        || out_cols1 > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("vector-MMA paired TC decode GEMM invalid geometry".into()));
    }
    let out_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("vector-MMA paired TC decode output overflow".into()))?;
    let out_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("vector-MMA paired TC decode output overflow".into()))?;
    if output0.len() < out_bytes0 || output1.len() < out_bytes1 {
        return Err(Error::Other(
            "vector-MMA paired TC decode GEMM: output buffer too small".into(),
        ));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair_vector_mma(
            activation.ptr(),
            weight_packed0.ptr(),
            weight_scale0.ptr(),
            weight_zero_point0.ptr(),
            output0.ptr(),
            out_cols0 as i32,
            weight_packed1.ptr(),
            weight_scale1.ptr(),
            weight_zero_point1.ptr(),
            output1.ptr(),
            out_cols1 as i32,
            in_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        )
    })
}


/// Opt-in paired raw-layout decode with coalesced shared staging of packed
/// weights and metadata for both projection candidates.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair_weight_stage(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols0 == 0
        || out_cols1 == 0
        || out_cols0 % 64 != 0
        || out_cols1 % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols0 > i32::MAX as usize
        || out_cols1 > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("paired weight-stage TC decode GEMM invalid geometry".into()));
    }
    let activation_bytes = in_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("paired weight-stage activation overflow".into()))?;
    let packed_bytes0 = out_cols0
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("paired weight-stage packed weight overflow".into()))?;
    let packed_bytes1 = out_cols1
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("paired weight-stage packed weight overflow".into()))?;
    let scale_bytes0 = out_cols0
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("paired weight-stage scale overflow".into()))?;
    let scale_bytes1 = out_cols1
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("paired weight-stage scale overflow".into()))?;
    let zp_bytes0 = (out_cols0 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("paired weight-stage zero-point overflow".into()))?;
    let zp_bytes1 = (out_cols1 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("paired weight-stage zero-point overflow".into()))?;
    let output_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("paired weight-stage output overflow".into()))?;
    let output_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("paired weight-stage output overflow".into()))?;
    if activation.len() < activation_bytes
        || weight_packed0.len() < packed_bytes0
        || weight_packed1.len() < packed_bytes1
        || weight_scale0.len() < scale_bytes0
        || weight_scale1.len() < scale_bytes1
        || weight_zero_point0.len() < zp_bytes0
        || weight_zero_point1.len() < zp_bytes1
        || output0.len() < output_bytes0
        || output1.len() < output_bytes1
    {
        return Err(Error::Other("paired weight-stage TC decode buffer/layout mismatch".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair_weight_stage(
            activation.ptr(),
            weight_packed0.ptr(), weight_scale0.ptr(), weight_zero_point0.ptr(),
            output0.ptr(), out_cols0 as i32,
            weight_packed1.ptr(), weight_scale1.ptr(), weight_zero_point1.ptr(),
            output1.ptr(), out_cols1 as i32,
            in_cols as i32, groups as i32, ctx.stream().handle(),
        )
    })
}

/// Raw-layout paired decode with per-CTA metadata staging. This candidate is
/// opt-in at the model dispatch layer and leaves the baseline pair ABI intact.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair_meta(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols0 == 0
        || out_cols1 == 0
        || out_cols0 % 64 != 0
        || out_cols1 % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols0 > i32::MAX as usize
        || out_cols1 > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("paired metadata TC decode GEMM invalid geometry".into()));
    }
    let activation_bytes = in_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("paired metadata TC activation overflow".into()))?;
    let packed_bytes0 = out_cols0
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("paired metadata TC packed weight overflow".into()))?;
    let packed_bytes1 = out_cols1
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("paired metadata TC packed weight overflow".into()))?;
    let scale_bytes0 = out_cols0
        .checked_mul(groups).and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("paired metadata TC scale overflow".into()))?;
    let scale_bytes1 = out_cols1
        .checked_mul(groups).and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("paired metadata TC scale overflow".into()))?;
    let zp_bytes0 = (out_cols0 / 8)
        .checked_mul(groups).and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("paired metadata TC zero-point overflow".into()))?;
    let zp_bytes1 = (out_cols1 / 8)
        .checked_mul(groups).and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("paired metadata TC zero-point overflow".into()))?;
    if activation.len() < activation_bytes
        || weight_packed0.len() < packed_bytes0 || weight_packed1.len() < packed_bytes1
        || weight_scale0.len() < scale_bytes0 || weight_scale1.len() < scale_bytes1
        || weight_zero_point0.len() < zp_bytes0 || weight_zero_point1.len() < zp_bytes1
    {
        return Err(Error::Other("paired metadata TC quantized buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair_meta(
            activation.ptr(),
            weight_packed0.ptr(), weight_scale0.ptr(), weight_zero_point0.ptr(),
            output0.ptr(), out_cols0 as i32,
            weight_packed1.ptr(), weight_scale1.ptr(), weight_zero_point1.ptr(),
            output1.ptr(), out_cols1 as i32,
            in_cols as i32, groups as i32, ctx.stream().handle(),
        )
    })
}

/// Opt-in paired raw-layout decode using 48 output rows and six warps per CTA.
/// Output widths need only preserve the raw layout's eight-row packing; the
/// final CTA of each projection predicates rows beyond its logical extent.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair_6w(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols0 == 0
        || out_cols1 == 0
        || out_cols0 % 8 != 0
        || out_cols1 % 8 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols0 > (i32::MAX as usize - 47)
        || out_cols1 > (i32::MAX as usize - 47)
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("six-warp paired TC decode GEMM invalid geometry".into()));
    }
    let activation_bytes = in_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("six-warp paired TC activation overflow".into()))?;
    let packed_bytes0 = out_cols0
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("six-warp paired TC packed weight overflow".into()))?;
    let packed_bytes1 = out_cols1
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("six-warp paired TC packed weight overflow".into()))?;
    let scale_bytes0 = out_cols0
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("six-warp paired TC scale overflow".into()))?;
    let scale_bytes1 = out_cols1
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("six-warp paired TC scale overflow".into()))?;
    let zp_bytes0 = (out_cols0 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("six-warp paired TC zero-point overflow".into()))?;
    let zp_bytes1 = (out_cols1 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("six-warp paired TC zero-point overflow".into()))?;
    let output_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("six-warp paired TC output overflow".into()))?;
    let output_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("six-warp paired TC output overflow".into()))?;
    if activation.len() < activation_bytes
        || weight_packed0.len() < packed_bytes0
        || weight_packed1.len() < packed_bytes1
        || weight_scale0.len() < scale_bytes0
        || weight_scale1.len() < scale_bytes1
        || weight_zero_point0.len() < zp_bytes0
        || weight_zero_point1.len() < zp_bytes1
        || output0.len() < output_bytes0
        || output1.len() < output_bytes1
    {
        return Err(Error::Other("six-warp paired TC decode GEMM buffer/layout mismatch".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair_6w(
            activation.ptr(),
            weight_packed0.ptr(),
            weight_scale0.ptr(),
            weight_zero_point0.ptr(),
            output0.ptr(),
            out_cols0 as i32,
            weight_packed1.ptr(),
            weight_scale1.ptr(),
            weight_zero_point1.ptr(),
            output1.ptr(),
            out_cols1 as i32,
            in_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        )
    })
}

/// Opt-in paired raw-layout decode with double-buffered activation staging.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair_act(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols0 == 0
        || out_cols1 == 0
        || out_cols0 % 64 != 0
        || out_cols1 % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols0 > i32::MAX as usize
        || out_cols1 > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("paired activation TC decode GEMM invalid geometry".into()));
    }
    let activation_bytes = in_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("paired activation TC activation overflow".into()))?;
    let packed_bytes0 = out_cols0
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("paired activation TC packed weight overflow".into()))?;
    let packed_bytes1 = out_cols1
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("paired activation TC packed weight overflow".into()))?;
    let scale_bytes0 = out_cols0
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("paired activation TC scale overflow".into()))?;
    let scale_bytes1 = out_cols1
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("paired activation TC scale overflow".into()))?;
    let zp_bytes0 = (out_cols0 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("paired activation TC zero-point overflow".into()))?;
    let zp_bytes1 = (out_cols1 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("paired activation TC zero-point overflow".into()))?;
    let output_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("paired activation TC output overflow".into()))?;
    let output_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("paired activation TC output overflow".into()))?;
    if activation.len() < activation_bytes
        || weight_packed0.len() < packed_bytes0
        || weight_packed1.len() < packed_bytes1
        || weight_scale0.len() < scale_bytes0
        || weight_scale1.len() < scale_bytes1
        || weight_zero_point0.len() < zp_bytes0
        || weight_zero_point1.len() < zp_bytes1
        || output0.len() < output_bytes0
        || output1.len() < output_bytes1
    {
        return Err(Error::Other("paired activation TC buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair_act(
            activation.ptr(),
            weight_packed0.ptr(), weight_scale0.ptr(), weight_zero_point0.ptr(),
            output0.ptr(), out_cols0 as i32,
            weight_packed1.ptr(), weight_scale1.ptr(), weight_zero_point1.ptr(),
            output1.ptr(), out_cols1 as i32,
            in_cols as i32, groups as i32, ctx.stream().handle(),
        )
    })
}

/// Opt-in paired raw-layout decode using the accepted 32-row, four-warp tile.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair_alt(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols0 == 0
        || out_cols1 == 0
        || out_cols0 % 32 != 0
        || out_cols1 % 32 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols0 > i32::MAX as usize
        || out_cols1 > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("alternate-tile paired TC decode GEMM invalid geometry".into()));
    }
    let activation_bytes = in_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("alternate-tile paired TC activation overflow".into()))?;
    let packed_bytes0 = out_cols0
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("alternate-tile paired TC packed weight overflow".into()))?;
    let packed_bytes1 = out_cols1
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("alternate-tile paired TC packed weight overflow".into()))?;
    let scale_bytes0 = out_cols0
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("alternate-tile paired TC scale overflow".into()))?;
    let scale_bytes1 = out_cols1
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("alternate-tile paired TC scale overflow".into()))?;
    let zp_bytes0 = (out_cols0 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("alternate-tile paired TC zero-point overflow".into()))?;
    let zp_bytes1 = (out_cols1 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("alternate-tile paired TC zero-point overflow".into()))?;
    let output_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("alternate-tile paired TC output overflow".into()))?;
    let output_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("alternate-tile paired TC output overflow".into()))?;
    if activation.len() < activation_bytes
        || weight_packed0.len() < packed_bytes0
        || weight_packed1.len() < packed_bytes1
        || weight_scale0.len() < scale_bytes0
        || weight_scale1.len() < scale_bytes1
        || weight_zero_point0.len() < zp_bytes0
        || weight_zero_point1.len() < zp_bytes1
        || output0.len() < output_bytes0
        || output1.len() < output_bytes1
    {
        return Err(Error::Other("alternate-tile paired TC decode buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair_alt(
            activation.ptr(),
            weight_packed0.ptr(), weight_scale0.ptr(), weight_zero_point0.ptr(),
            output0.ptr(), out_cols0 as i32,
            weight_packed1.ptr(), weight_scale1.ptr(), weight_zero_point1.ptr(),
            output1.ptr(), out_cols1 as i32,
            in_cols as i32, groups as i32, ctx.stream().handle(),
        )
    })
}

/// Opt-in paired raw-layout decode with two projection-local warp groups
/// sharing one activation tile while preserving exact per-projection MMA order.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair_shared(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols0 == 0
        || out_cols1 == 0
        || out_cols0 % 64 != 0
        || out_cols1 % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols0 > i32::MAX as usize
        || out_cols1 > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("shared-activation paired TC decode GEMM invalid geometry".into()));
    }
    let activation_bytes = in_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("shared-activation paired TC activation overflow".into()))?;
    let packed_bytes0 = out_cols0
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("shared-activation paired TC packed weight overflow".into()))?;
    let packed_bytes1 = out_cols1
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("shared-activation paired TC packed weight overflow".into()))?;
    let scale_bytes0 = out_cols0
        .checked_mul(groups).and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("shared-activation paired TC scale overflow".into()))?;
    let scale_bytes1 = out_cols1
        .checked_mul(groups).and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("shared-activation paired TC scale overflow".into()))?;
    let zp_bytes0 = (out_cols0 / 8)
        .checked_mul(groups).and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("shared-activation paired TC zero-point overflow".into()))?;
    let zp_bytes1 = (out_cols1 / 8)
        .checked_mul(groups).and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("shared-activation paired TC zero-point overflow".into()))?;
    let output_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("shared-activation paired TC output overflow".into()))?;
    let output_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("shared-activation paired TC output overflow".into()))?;
    if activation.len() < activation_bytes
        || weight_packed0.len() < packed_bytes0 || weight_packed1.len() < packed_bytes1
        || weight_scale0.len() < scale_bytes0 || weight_scale1.len() < scale_bytes1
        || weight_zero_point0.len() < zp_bytes0 || weight_zero_point1.len() < zp_bytes1
        || output0.len() < output_bytes0 || output1.len() < output_bytes1
    {
        return Err(Error::Other("shared-activation paired TC decode buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair_shared(
            activation.ptr(),
            weight_packed0.ptr(), weight_scale0.ptr(), weight_zero_point0.ptr(),
            output0.ptr(), out_cols0 as i32,
            weight_packed1.ptr(), weight_scale1.ptr(), weight_zero_point1.ptr(),
            output1.ptr(), out_cols1 as i32,
            in_cols as i32, groups as i32, ctx.stream().handle(),
        )
    })
}

/// Opt-in paired raw-layout decode that stages each activation tile once for
/// both projections while preserving separate weight metadata and MMA order.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair_reuse(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols0 == 0
        || out_cols1 == 0
        || out_cols0 % 64 != 0
        || out_cols1 % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols0 > i32::MAX as usize
        || out_cols1 > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("activation-reuse paired TC decode GEMM invalid geometry".into()));
    }
    let activation_bytes = in_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("activation-reuse paired TC activation overflow".into()))?;
    let packed_bytes0 = out_cols0
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("activation-reuse paired TC packed weight overflow".into()))?;
    let packed_bytes1 = out_cols1
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("activation-reuse paired TC packed weight overflow".into()))?;
    let scale_bytes0 = out_cols0
        .checked_mul(groups).and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("activation-reuse paired TC scale overflow".into()))?;
    let scale_bytes1 = out_cols1
        .checked_mul(groups).and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("activation-reuse paired TC scale overflow".into()))?;
    let zp_bytes0 = (out_cols0 / 8)
        .checked_mul(groups).and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("activation-reuse paired TC zero-point overflow".into()))?;
    let zp_bytes1 = (out_cols1 / 8)
        .checked_mul(groups).and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("activation-reuse paired TC zero-point overflow".into()))?;
    let output_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("activation-reuse paired TC output overflow".into()))?;
    let output_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("activation-reuse paired TC output overflow".into()))?;
    if activation.len() < activation_bytes
        || weight_packed0.len() < packed_bytes0 || weight_packed1.len() < packed_bytes1
        || weight_scale0.len() < scale_bytes0 || weight_scale1.len() < scale_bytes1
        || weight_zero_point0.len() < zp_bytes0 || weight_zero_point1.len() < zp_bytes1
        || output0.len() < output_bytes0 || output1.len() < output_bytes1
    {
        return Err(Error::Other("activation-reuse paired TC decode buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair_reuse(
            activation.ptr(),
            weight_packed0.ptr(), weight_scale0.ptr(), weight_zero_point0.ptr(),
            output0.ptr(), out_cols0 as i32,
            weight_packed1.ptr(), weight_scale1.ptr(), weight_zero_point1.ptr(),
            output1.ptr(), out_cols1 as i32,
            in_cols as i32, groups as i32, ctx.stream().handle(),
        )
    })
}

/// Opt-in paired raw-layout decode using read-only cache loads for immutable
/// activation, packed-weight, and metadata inputs.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair_cache(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols0 == 0
        || out_cols1 == 0
        || out_cols0 % 64 != 0
        || out_cols1 % 64 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols0 > i32::MAX as usize
        || out_cols1 > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("cache-load paired TC decode GEMM invalid geometry".into()));
    }
    let pointers = [
        (activation.ptr() as usize, std::mem::align_of::<u16>()),
        (weight_packed0.ptr() as usize, std::mem::align_of::<i32>()),
        (weight_packed1.ptr() as usize, std::mem::align_of::<i32>()),
        (weight_scale0.ptr() as usize, std::mem::align_of::<u16>()),
        (weight_scale1.ptr() as usize, std::mem::align_of::<u16>()),
        (weight_zero_point0.ptr() as usize, std::mem::align_of::<i32>()),
        (weight_zero_point1.ptr() as usize, std::mem::align_of::<i32>()),
    ];
    if pointers.iter().any(|(ptr, alignment)| ptr % alignment != 0) {
        return Err(Error::Other("cache-load paired TC decode input alignment invalid".into()));
    }
    let activation_bytes = in_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("cache-load paired TC activation overflow".into()))?;
    let packed_bytes0 = out_cols0
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("cache-load paired TC packed weight overflow".into()))?;
    let packed_bytes1 = out_cols1
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("cache-load paired TC packed weight overflow".into()))?;
    let scale_bytes0 = out_cols0
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("cache-load paired TC scale overflow".into()))?;
    let scale_bytes1 = out_cols1
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("cache-load paired TC scale overflow".into()))?;
    let zp_bytes0 = (out_cols0 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("cache-load paired TC zero-point overflow".into()))?;
    let zp_bytes1 = (out_cols1 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("cache-load paired TC zero-point overflow".into()))?;
    let output_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("cache-load paired TC output overflow".into()))?;
    let output_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("cache-load paired TC output overflow".into()))?;
    if activation.len() < activation_bytes
        || weight_packed0.len() < packed_bytes0
        || weight_packed1.len() < packed_bytes1
        || weight_scale0.len() < scale_bytes0
        || weight_scale1.len() < scale_bytes1
        || weight_zero_point0.len() < zp_bytes0
        || weight_zero_point1.len() < zp_bytes1
        || output0.len() < output_bytes0
        || output1.len() < output_bytes1
    {
        return Err(Error::Other("cache-load paired TC decode buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair_cache(
            activation.ptr(),
            weight_packed0.ptr(), weight_scale0.ptr(), weight_zero_point0.ptr(),
            output0.ptr(), out_cols0 as i32,
            weight_packed1.ptr(), weight_scale1.ptr(), weight_zero_point1.ptr(),
            output1.ptr(), out_cols1 as i32,
            in_cols as i32, groups as i32, ctx.stream().handle(),
        )
    })
}

/// Opt-in paired raw-layout decode using 16 output rows and two warps per CTA.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_pair_2w(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed0: &CudaBuffer,
    weight_scale0: &CudaBuffer,
    weight_zero_point0: &CudaBuffer,
    output0: &CudaBuffer,
    out_cols0: usize,
    weight_packed1: &CudaBuffer,
    weight_scale1: &CudaBuffer,
    weight_zero_point1: &CudaBuffer,
    output1: &CudaBuffer,
    out_cols1: usize,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || out_cols0 == 0
        || out_cols1 == 0
        || out_cols0 % 16 != 0
        || out_cols1 % 16 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || out_cols0 > i32::MAX as usize
        || out_cols1 > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("two-warp paired TC decode GEMM invalid geometry".into()));
    }
    let activation_bytes = in_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("two-warp paired TC activation overflow".into()))?;
    let packed_bytes0 = out_cols0
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("two-warp paired TC packed weight overflow".into()))?;
    let packed_bytes1 = out_cols1
        .checked_mul(in_cols / 8)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("two-warp paired TC packed weight overflow".into()))?;
    let scale_bytes0 = out_cols0
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("two-warp paired TC scale overflow".into()))?;
    let scale_bytes1 = out_cols1
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("two-warp paired TC scale overflow".into()))?;
    let zp_bytes0 = (out_cols0 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("two-warp paired TC zero-point overflow".into()))?;
    let zp_bytes1 = (out_cols1 / 8)
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::Other("two-warp paired TC zero-point overflow".into()))?;
    let output_bytes0 = out_cols0
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("two-warp paired TC output overflow".into()))?;
    let output_bytes1 = out_cols1
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("two-warp paired TC output overflow".into()))?;
    if activation.len() < activation_bytes
        || weight_packed0.len() < packed_bytes0
        || weight_packed1.len() < packed_bytes1
        || weight_scale0.len() < scale_bytes0
        || weight_scale1.len() < scale_bytes1
        || weight_zero_point0.len() < zp_bytes0
        || weight_zero_point1.len() < zp_bytes1
        || output0.len() < output_bytes0
        || output1.len() < output_bytes1
    {
        return Err(Error::Other("two-warp paired TC decode buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_pair_2w(
            activation.ptr(),
            weight_packed0.ptr(), weight_scale0.ptr(), weight_zero_point0.ptr(),
            output0.ptr(), out_cols0 as i32,
            weight_packed1.ptr(), weight_scale1.ptr(), weight_zero_point1.ptr(),
            output1.ptr(), out_cols1 as i32,
            in_cols as i32, groups as i32, ctx.stream().handle(),
        )
    })
}

/// Runs three or four independent single-row W4A16 projections in one CUDA launch.
/// Each tuple is `(packed, scale, zero_point, output, out_cols)`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_multi(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    first: (&CudaBuffer, &CudaBuffer, &CudaBuffer, &CudaBuffer, usize),
    second: (&CudaBuffer, &CudaBuffer, &CudaBuffer, &CudaBuffer, usize),
    third: (&CudaBuffer, &CudaBuffer, &CudaBuffer, &CudaBuffer, usize),
    fourth: Option<(&CudaBuffer, &CudaBuffer, &CudaBuffer, &CudaBuffer, usize)>,
    in_cols: usize,
    groups: usize,
) -> Result<()> {
    if in_cols == 0
        || in_cols % 128 != 0
        || groups == 0
        || in_cols % groups != 0
        || in_cols / groups != 32
        || in_cols > i32::MAX as usize
        || groups > i32::MAX as usize
    {
        return Err(Error::Other("staged multi TC decode GEMM invalid geometry".into()));
    }
    let activation_bytes = in_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("staged multi TC activation overflow".into()))?;
    if activation.len() < activation_bytes {
        return Err(Error::Other("staged multi TC activation buffer too small".into()));
    }
    for (packed, scale, zero_point, output, out_cols) in
        [first, second, third].into_iter().chain(fourth)
    {
        if out_cols == 0 || out_cols % 8 != 0 || out_cols > i32::MAX as usize {
            return Err(Error::Other("staged multi TC decode output geometry invalid".into()));
        }
        let packed_bytes = out_cols
            .checked_mul(in_cols / 8)
            .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
            .ok_or_else(|| Error::Other("staged multi TC packed weight overflow".into()))?;
        let scale_bytes = out_cols
            .checked_mul(groups)
            .and_then(|value| value.checked_mul(DType::BF16.size_in_bytes()))
            .ok_or_else(|| Error::Other("staged multi TC scale overflow".into()))?;
        let zero_point_bytes = out_cols
            .div_ceil(8)
            .checked_mul(groups)
            .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
            .ok_or_else(|| Error::Other("staged multi TC zero-point overflow".into()))?;
        let output_bytes = out_cols
            .checked_mul(DType::BF16.size_in_bytes())
            .ok_or_else(|| Error::Other("staged multi TC output overflow".into()))?;
        if packed.len() < packed_bytes
            || scale.len() < scale_bytes
            || zero_point.len() < zero_point_bytes
            || output.len() < output_bytes
        {
            return Err(Error::Other("staged multi TC decode buffer/layout mismatch".into()));
        }
    }
    let null = std::ptr::null();
    let null_mut = std::ptr::null_mut();
    let (w3, s3, z3, o3, n3, count) = fourth.map_or(
        (null, null, null, null_mut, 0, 3),
        |(w, s, z, o, n)| (w.ptr(), s.ptr(), z.ptr(), o.ptr(), n as i32, 4),
    );
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_multi(
            activation.ptr(),
            first.0.ptr(), first.1.ptr(), first.2.ptr(), first.3.ptr(), first.4 as i32,
            second.0.ptr(), second.1.ptr(), second.2.ptr(), second.3.ptr(), second.4 as i32,
            third.0.ptr(), third.1.ptr(), third.2.ptr(), third.3.ptr(), third.4 as i32,
            w3, s3, z3, o3, n3, count, in_cols as i32, groups as i32,
            ctx.stream().handle(),
        )
    })
}

/// Same exact raw multi-projection launch on an explicitly supplied stream.
/// Callers own cross-stream dependencies; no host synchronization is issued.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_tc_multi_on_stream(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    first: (&CudaBuffer, &CudaBuffer, &CudaBuffer, &CudaBuffer, usize),
    second: (&CudaBuffer, &CudaBuffer, &CudaBuffer, &CudaBuffer, usize),
    third: (&CudaBuffer, &CudaBuffer, &CudaBuffer, &CudaBuffer, usize),
    in_cols: usize,
    groups: usize,
    stream: &crate::CudaStream,
) -> Result<()> {
    if in_cols == 0 || in_cols % 128 != 0 || groups == 0
        || in_cols % groups != 0 || in_cols / groups != 32
    {
        return Err(Error::Other("side-stream multi projection invalid geometry".into()));
    }
    let activation_bytes = in_cols * DType::BF16.size_in_bytes();
    if activation.device() != ctx.device_id() || activation.len() < activation_bytes {
        return Err(Error::Other("side-stream activation mismatch".into()));
    }
    for (packed, scale, zero_point, output, out_cols) in [first, second, third] {
        let packed_bytes = out_cols * (in_cols / 8) * std::mem::size_of::<i32>();
        let scale_bytes = out_cols * groups * DType::BF16.size_in_bytes();
        let zp_bytes = out_cols.div_ceil(8) * groups * std::mem::size_of::<i32>();
        let out_bytes = out_cols * DType::BF16.size_in_bytes();
        if out_cols == 0 || out_cols % 8 != 0
            || [packed, scale, zero_point, output].iter().any(|buf| buf.device() != ctx.device_id())
            || packed.len() < packed_bytes || scale.len() < scale_bytes
            || zero_point.len() < zp_bytes || output.len() < out_bytes
        {
            return Err(Error::Other("side-stream projection buffer/layout mismatch".into()));
        }
    }
    let null = std::ptr::null();
    let null_mut = std::ptr::null_mut();
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gemm_w4a16_bf16_tc_multi(
            activation.ptr(),
            first.0.ptr(), first.1.ptr(), first.2.ptr(), first.3.ptr(), first.4 as i32,
            second.0.ptr(), second.1.ptr(), second.2.ptr(), second.3.ptr(), second.4 as i32,
            third.0.ptr(), third.1.ptr(), third.2.ptr(), third.3.ptr(), third.4 as i32,
            null, null, null, null_mut, 0, 3, in_cols as i32, groups as i32,
            stream.handle(),
        )
    })
}

pub fn matmul_bf16_w4a16_asym_into(
        ctx: &CudaContext,
        activation: &CudaBuffer,
        weight_packed: &CudaBuffer,
        weight_scale: &CudaBuffer,
        weight_zero_point: &CudaBuffer,
        output: &CudaBuffer,
        rows: usize,
        in_cols: usize,
        out_cols: usize,
        groups: usize,
    ) -> Result<()> {
        let out_bytes = out_cols
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
            .ok_or_else(|| Error::Other("matmul_bf16_w4a16_asym_into output overflow".into()))?;
        if output.len() < out_bytes {
            return Err(Error::Other("matmul_bf16_w4a16_asym_into: buffer too small".into()));
        }
        if in_cols == 0
            || in_cols % 256 != 0
            || groups == 0
            || in_cols % groups != 0
            || (in_cols / groups) % 8 != 0
            || out_cols % 128 != 0
        {
            return Err(Error::Other(
                "matmul_bf16_w4a16_asym_into: invalid raw W4 geometry".into(),
            ));
        }
        check_cuda(unsafe {
            ffi::apxinf_qwen35_gemm_w4a16_bf16(
                activation.ptr(),
                weight_packed.ptr(),
                weight_scale.ptr(),
                weight_zero_point.ptr(),
                output.ptr(),
                in_cols as i32,
                out_cols as i32,
                groups as i32,
                ctx.stream().handle(),
            )
        })
    }

pub fn matmul_bf16_w4a16_asym(
    ctx: &CudaContext,
    activation: &Tensor,
    weight_packed: &Tensor,
    weight_scale: &Tensor,
    weight_zero_point: &Tensor,
    out_cols: usize,
    in_cols: usize,
    groups: usize,
) -> Result<Tensor> {
    if activation.dtype() != DType::BF16
        || weight_packed.dtype() != DType::I32
        || weight_scale.dtype() != DType::BF16
        || weight_zero_point.dtype() != DType::I32
    {
        return Err(Error::Other("matmul_bf16_w4a16_asym dtype mismatch".into()));
    }
    let dims = activation.shape().dims();
    if dims.len() != 2 || dims[1] != in_cols {
        return Err(Error::Other(format!(
            "matmul_bf16_w4a16_asym activation shape {dims:?} incompatible with in_cols {in_cols}"
        )));
    }
    let _rows = dims[0];
    if weight_packed.shape().dims() != [out_cols, in_cols.div_ceil(8)] {
        return Err(Error::Other("matmul_bf16_w4a16_asym packed shape mismatch".into()));
    }
    if weight_scale.shape().dims() != [out_cols, groups]
        || weight_zero_point.shape().dims() != [out_cols.div_ceil(8), groups]
    {
        return Err(Error::Other("matmul_bf16_w4a16_asym scale/zero-point shape mismatch".into()));
    }
    // Decode (single row) is latency-bound: the static dequant + cublas path
    // would stream the entire weight matrix through HBM twice per call. Use
    // the fused dequant-GEMM kernel, which reads the packed weights once.
    // Prefill keeps the cublas path: batched GEMM amortizes the dequant cost
    // and tensor-core GEMM dominates the fused kernel's scalar loop.
    if dims[0] == 1 {
        let out_bytes = out_cols
            .checked_mul(DType::BF16.size_in_bytes())
            .ok_or_else(|| Error::Other("matmul_bf16_w4a16_asym output overflow".into()))?;
        let output = CudaBuffer::alloc_zeros(out_bytes, ctx.device_id()).map_err(Error::Cuda)?;
        unsafe {
            ffi::check_cuda(ffi::apxinf_static_matmul_bf16_w4a16_asym(
                gpu_ptr(activation)?,
                gpu_ptr(weight_packed)?,
                gpu_ptr(weight_scale)?,
                gpu_ptr(weight_zero_point)?,
                output.ptr(),
                1,
                in_cols as i32,
                out_cols as i32,
                groups as i32,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        return Ok(output.into_tensor(
            apxinf_core::Shape::from(vec![1, out_cols]),
            DType::BF16,
        ));
    }

    let _t0 = std::time::Instant::now();
    let dense_bytes = out_cols
        .checked_mul(in_cols)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("matmul_bf16_w4a16_asym dense weight overflow".into()))?;
    let dense = CudaBuffer::alloc_zeros(dense_bytes, ctx.device_id()).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_dequantize_w4a16_asym_bf16(
            gpu_ptr(weight_packed)?,
            gpu_ptr(weight_scale)?,
            gpu_ptr(weight_zero_point)?,
            dense.ptr(),
            in_cols as i32,
            out_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    let t_dequant = _t0.elapsed().as_secs_f32() * 1000.0;
    let dense = dense.into_tensor(
        apxinf_core::Shape::from(vec![out_cols, in_cols]),
        DType::BF16,
    );
    let result = matmul_bf16_transposed(ctx, activation, &dense);
    if std::env::var_os("APXINF_GEMM_PROF").is_some() {
        let ms = _t0.elapsed().as_secs_f32() * 1000.0;
        if ms > 5.0 {
            eprintln!(
                "[gemm] {}x{} rows={} : {:.1} ms (dequant {:.1} ms)",
                out_cols, in_cols, dims[0], ms, t_dequant
            );
        }
    }
    result
}


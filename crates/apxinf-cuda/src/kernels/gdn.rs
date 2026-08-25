use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;

pub const QWEN35_GDN_HEADS: usize = 48;
pub const QWEN35_GDN_KEY_DIM: usize = 128;
pub const QWEN35_GDN_VALUE_DIM: usize = 128;
pub const QWEN35_GDN_KEY_HEADS: usize = 16;
pub const QWEN35_GDN_CONV_DIM: usize = 10240;
pub const QWEN35_GDN_CONV_KERNEL: usize = 4;

/// Run one Qwen3.5 recurrent gated-delta decode step. `recurrent_state` is
/// FP32 and updated in place; `output` is caller-owned BF16 storage.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_recurrent_write(
    ctx: &CudaContext,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    recurrent_state: &Tensor,
    output: &Tensor,
) -> Result<()> {
    let device = Device::Cuda(ctx.device_id());
    require(
        query,
        "query",
        DType::BF16,
        &[QWEN35_GDN_HEADS, QWEN35_GDN_KEY_DIM],
        device,
    )?;
    require(
        key,
        "key",
        DType::BF16,
        &[QWEN35_GDN_HEADS, QWEN35_GDN_KEY_DIM],
        device,
    )?;
    require(
        value,
        "value",
        DType::BF16,
        &[QWEN35_GDN_HEADS, QWEN35_GDN_VALUE_DIM],
        device,
    )?;
    require(g, "g", DType::F32, &[QWEN35_GDN_HEADS], device)?;
    require(beta, "beta", DType::F32, &[QWEN35_GDN_HEADS], device)?;
    require(
        recurrent_state,
        "recurrent state",
        DType::F32,
        &[QWEN35_GDN_HEADS, QWEN35_GDN_KEY_DIM, QWEN35_GDN_VALUE_DIM],
        device,
    )?;
    require(
        output,
        "output",
        DType::BF16,
        &[QWEN35_GDN_HEADS, QWEN35_GDN_VALUE_DIM],
        device,
    )?;

    let query = CudaBuffer::from_tensor(query).map_err(Error::Cuda)?;
    let key = CudaBuffer::from_tensor(key).map_err(Error::Cuda)?;
    let value = CudaBuffer::from_tensor(value).map_err(Error::Cuda)?;
    let g = CudaBuffer::from_tensor(g).map_err(Error::Cuda)?;
    let beta = CudaBuffer::from_tensor(beta).map_err(Error::Cuda)?;
    let state = CudaBuffer::from_tensor(recurrent_state).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_gdn_recurrent_bf16(
            query.ptr(),
            key.ptr(),
            value.ptr(),
            g.ptr(),
            beta.ptr(),
            state.ptr(),
            output.ptr(),
            QWEN35_GDN_HEADS as i32,
            QWEN35_GDN_KEY_DIM as i32,
            QWEN35_GDN_VALUE_DIM as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

pub fn qwen35_conv4_silu_write(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    output: &Tensor,
) -> Result<()> {
    let device = Device::Cuda(ctx.device_id());
    require(
        input,
        "conv input",
        DType::BF16,
        &[QWEN35_GDN_CONV_DIM],
        device,
    )?;
    require(
        weight,
        "conv weight",
        DType::BF16,
        &[QWEN35_GDN_CONV_DIM, 1, QWEN35_GDN_CONV_KERNEL],
        device,
    )?;
    require(
        conv_state,
        "conv state",
        DType::BF16,
        &[QWEN35_GDN_CONV_DIM, QWEN35_GDN_CONV_KERNEL],
        device,
    )?;
    require(
        output,
        "conv output",
        DType::BF16,
        &[QWEN35_GDN_CONV_DIM],
        device,
    )?;
    let input = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let weight = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let state = CudaBuffer::from_tensor(conv_state).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_gdn_conv4_silu_bf16(
            input.ptr(),
            weight.ptr(),
            state.ptr(),
            output.ptr(),
            QWEN35_GDN_CONV_DIM as i32,
            QWEN35_GDN_CONV_KERNEL as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_prepare_write(
    ctx: &CudaContext,
    convolved_qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<()> {
    let device = Device::Cuda(ctx.device_id());
    require(
        convolved_qkv,
        "prepared QKV input",
        DType::BF16,
        &[QWEN35_GDN_CONV_DIM],
        device,
    )?;
    for (name, tensor) in [("a", a), ("b", b), ("A_log", a_log), ("dt_bias", dt_bias)] {
        require(tensor, name, DType::BF16, &[QWEN35_GDN_HEADS], device)?;
    }
    for (name, tensor) in [("query", query), ("key", key), ("value", value)] {
        require(
            tensor,
            name,
            DType::BF16,
            &[QWEN35_GDN_HEADS, QWEN35_GDN_KEY_DIM],
            device,
        )?;
    }
    require(g, "g", DType::F32, &[QWEN35_GDN_HEADS], device)?;
    require(beta, "beta", DType::F32, &[QWEN35_GDN_HEADS], device)?;
    let input = CudaBuffer::from_tensor(convolved_qkv).map_err(Error::Cuda)?;
    let a = CudaBuffer::from_tensor(a).map_err(Error::Cuda)?;
    let b = CudaBuffer::from_tensor(b).map_err(Error::Cuda)?;
    let a_log = CudaBuffer::from_tensor(a_log).map_err(Error::Cuda)?;
    let dt_bias = CudaBuffer::from_tensor(dt_bias).map_err(Error::Cuda)?;
    let query = CudaBuffer::from_tensor(query).map_err(Error::Cuda)?;
    let key = CudaBuffer::from_tensor(key).map_err(Error::Cuda)?;
    let value = CudaBuffer::from_tensor(value).map_err(Error::Cuda)?;
    let g = CudaBuffer::from_tensor(g).map_err(Error::Cuda)?;
    let beta = CudaBuffer::from_tensor(beta).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_gdn_prepare_bf16(
            input.ptr(),
            a.ptr(),
            b.ptr(),
            a_log.ptr(),
            dt_bias.ptr(),
            query.ptr(),
            key.ptr(),
            value.ptr(),
            g.ptr(),
            beta.ptr(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

pub fn qwen35_gated_rmsnorm_write(
    ctx: &CudaContext,
    input: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    epsilon: f32,
) -> Result<()> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::Other(format!(
            "Qwen3.5 gated RMSNorm epsilon must be positive, got {epsilon}"
        )));
    }
    let device = Device::Cuda(ctx.device_id());
    let shape = [QWEN35_GDN_HEADS, QWEN35_GDN_VALUE_DIM];
    require(input, "gated norm input", DType::BF16, &shape, device)?;
    require(gate, "gated norm gate", DType::BF16, &shape, device)?;
    require(
        weight,
        "gated norm weight",
        DType::BF16,
        &[QWEN35_GDN_VALUE_DIM],
        device,
    )?;
    require(output, "gated norm output", DType::BF16, &shape, device)?;
    let input = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let gate = CudaBuffer::from_tensor(gate).map_err(Error::Cuda)?;
    let weight = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_gdn_gated_rmsnorm_bf16(
            input.ptr(),
            gate.ptr(),
            weight.ptr(),
            output.ptr(),
            QWEN35_GDN_HEADS as i32,
            QWEN35_GDN_VALUE_DIM as i32,
            epsilon,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_conv4_prepare_write(
    ctx: &CudaContext,
    projected_qkv: &Tensor,
    conv_weight: &Tensor,
    conv_state: &Tensor,
    projected_ab: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    a_output: &Tensor,
    b_output: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<()> {
    let device = Device::Cuda(ctx.device_id());
    require(
        projected_qkv,
        "fused QKV input",
        DType::BF16,
        &[QWEN35_GDN_CONV_DIM],
        device,
    )?;
    require(
        conv_weight,
        "fused conv weight",
        DType::BF16,
        &[QWEN35_GDN_CONV_DIM, 1, QWEN35_GDN_CONV_KERNEL],
        device,
    )?;
    require(
        conv_state,
        "fused conv state",
        DType::BF16,
        &[QWEN35_GDN_CONV_DIM, QWEN35_GDN_CONV_KERNEL],
        device,
    )?;
    require(
        projected_ab,
        "fused a/b input",
        DType::BF16,
        &[2 * QWEN35_GDN_HEADS],
        device,
    )?;
    for (name, tensor) in [("A_log", a_log), ("dt_bias", dt_bias)] {
        require(tensor, name, DType::BF16, &[QWEN35_GDN_HEADS], device)?;
    }
    for (name, tensor) in [("a output", a_output), ("b output", b_output)] {
        require(tensor, name, DType::BF16, &[QWEN35_GDN_HEADS], device)?;
    }
    for (name, tensor) in [("query", query), ("key", key), ("value", value)] {
        require(
            tensor,
            name,
            DType::BF16,
            &[QWEN35_GDN_HEADS, QWEN35_GDN_KEY_DIM],
            device,
        )?;
    }
    require(g, "g", DType::F32, &[QWEN35_GDN_HEADS], device)?;
    require(beta, "beta", DType::F32, &[QWEN35_GDN_HEADS], device)?;
    let projected_qkv = CudaBuffer::from_tensor(projected_qkv).map_err(Error::Cuda)?;
    let conv_weight = CudaBuffer::from_tensor(conv_weight).map_err(Error::Cuda)?;
    let conv_state = CudaBuffer::from_tensor(conv_state).map_err(Error::Cuda)?;
    let projected_ab = CudaBuffer::from_tensor(projected_ab).map_err(Error::Cuda)?;
    let a_log = CudaBuffer::from_tensor(a_log).map_err(Error::Cuda)?;
    let dt_bias = CudaBuffer::from_tensor(dt_bias).map_err(Error::Cuda)?;
    let a_output = CudaBuffer::from_tensor(a_output).map_err(Error::Cuda)?;
    let b_output = CudaBuffer::from_tensor(b_output).map_err(Error::Cuda)?;
    let query = CudaBuffer::from_tensor(query).map_err(Error::Cuda)?;
    let key = CudaBuffer::from_tensor(key).map_err(Error::Cuda)?;
    let value = CudaBuffer::from_tensor(value).map_err(Error::Cuda)?;
    let g = CudaBuffer::from_tensor(g).map_err(Error::Cuda)?;
    let beta = CudaBuffer::from_tensor(beta).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_gdn_conv4_prepare_bf16(
            projected_qkv.ptr(),
            conv_weight.ptr(),
            conv_state.ptr(),
            projected_ab.ptr(),
            a_log.ptr(),
            dt_bias.ptr(),
            a_output.ptr(),
            b_output.ptr(),
            query.ptr(),
            key.ptr(),
            value.ptr(),
            g.ptr(),
            beta.ptr(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_conv4_prepare_m8_write(
    ctx: &CudaContext,
    projected_qkv: &Tensor,
    conv_weight: &Tensor,
    conv_state: &Tensor,
    projected_ab: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    a_output: &Tensor,
    b_output: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<()> {
    let device = Device::Cuda(ctx.device_id());
    let tokens = m8_tokens(
        projected_qkv,
        "M8 QKV",
        DType::BF16,
        QWEN35_GDN_CONV_DIM,
        device,
    )?;
    require(
        conv_weight,
        "M8 conv weight",
        DType::BF16,
        &[QWEN35_GDN_CONV_DIM, 1, QWEN35_GDN_CONV_KERNEL],
        device,
    )?;
    require(
        conv_state,
        "M8 conv state",
        DType::BF16,
        &[QWEN35_GDN_CONV_DIM, QWEN35_GDN_CONV_KERNEL],
        device,
    )?;
    require(
        projected_ab,
        "M8 a/b input",
        DType::BF16,
        &[tokens, 2 * QWEN35_GDN_HEADS],
        device,
    )?;
    for (name, tensor) in [("A_log", a_log), ("dt_bias", dt_bias)] {
        require(tensor, name, DType::BF16, &[QWEN35_GDN_HEADS], device)?;
    }
    for (name, tensor) in [("a output", a_output), ("b output", b_output)] {
        require(
            tensor,
            name,
            DType::BF16,
            &[tokens, QWEN35_GDN_HEADS],
            device,
        )?;
    }
    for (name, tensor) in [("query", query), ("key", key), ("value", value)] {
        require(
            tensor,
            name,
            DType::BF16,
            &[tokens, QWEN35_GDN_HEADS, QWEN35_GDN_KEY_DIM],
            device,
        )?;
    }
    require(g, "g", DType::F32, &[tokens, QWEN35_GDN_HEADS], device)?;
    require(
        beta,
        "beta",
        DType::F32,
        &[tokens, QWEN35_GDN_HEADS],
        device,
    )?;
    let buffers = [
        projected_qkv,
        conv_weight,
        conv_state,
        projected_ab,
        a_log,
        dt_bias,
        a_output,
        b_output,
        query,
        key,
        value,
        g,
        beta,
    ]
    .map(|tensor| CudaBuffer::from_tensor(tensor).map_err(Error::Cuda))
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_gdn_conv4_prepare_m8_bf16(
            buffers[0].ptr(),
            buffers[1].ptr(),
            buffers[2].ptr(),
            buffers[3].ptr(),
            buffers[4].ptr(),
            buffers[5].ptr(),
            buffers[6].ptr(),
            buffers[7].ptr(),
            buffers[8].ptr(),
            buffers[9].ptr(),
            buffers[10].ptr(),
            buffers[11].ptr(),
            buffers[12].ptr(),
            tokens as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_recurrent_m8_write(
    ctx: &CudaContext,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    recurrent_state: &Tensor,
    output: &Tensor,
) -> Result<()> {
    let device = Device::Cuda(ctx.device_id());
    let tokens = query.shape().dims().first().copied().unwrap_or(0);
    if !(1..=8).contains(&tokens) {
        return Err(Error::Other(
            "Qwen3.5 recurrent M8 tokens must be 1..=8".into(),
        ));
    }
    for (name, tensor) in [("query", query), ("key", key), ("value", value)] {
        require(
            tensor,
            name,
            DType::BF16,
            &[tokens, QWEN35_GDN_HEADS, QWEN35_GDN_KEY_DIM],
            device,
        )?;
    }
    for (name, tensor) in [("g", g), ("beta", beta)] {
        require(
            tensor,
            name,
            DType::F32,
            &[tokens, QWEN35_GDN_HEADS],
            device,
        )?;
    }
    require(
        recurrent_state,
        "M8 recurrent state",
        DType::F32,
        &[QWEN35_GDN_HEADS, QWEN35_GDN_KEY_DIM, QWEN35_GDN_VALUE_DIM],
        device,
    )?;
    require(
        output,
        "M8 recurrent output",
        DType::BF16,
        &[tokens, QWEN35_GDN_HEADS, QWEN35_GDN_VALUE_DIM],
        device,
    )?;
    let buffers = [query, key, value, g, beta, recurrent_state, output]
        .map(|tensor| CudaBuffer::from_tensor(tensor).map_err(Error::Cuda))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_gdn_recurrent_m8_bf16(
            buffers[0].ptr(),
            buffers[1].ptr(),
            buffers[2].ptr(),
            buffers[3].ptr(),
            buffers[4].ptr(),
            buffers[5].ptr(),
            buffers[6].ptr(),
            tokens as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

pub fn qwen35_gated_rmsnorm_m8_write(
    ctx: &CudaContext,
    input: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    epsilon: f32,
) -> Result<()> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::Other(
            "Qwen3.5 gated RMSNorm M8 epsilon must be positive".into(),
        ));
    }
    let device = Device::Cuda(ctx.device_id());
    let tokens = input.shape().dims().first().copied().unwrap_or(0);
    if !(1..=8).contains(&tokens) {
        return Err(Error::Other(
            "Qwen3.5 gated RMSNorm M8 tokens must be 1..=8".into(),
        ));
    }
    let shape = [tokens, QWEN35_GDN_HEADS, QWEN35_GDN_VALUE_DIM];
    require(input, "M8 gated input", DType::BF16, &shape, device)?;
    require(gate, "M8 gated gate", DType::BF16, &shape, device)?;
    require(output, "M8 gated output", DType::BF16, &shape, device)?;
    require(
        weight,
        "M8 gated weight",
        DType::BF16,
        &[QWEN35_GDN_VALUE_DIM],
        device,
    )?;
    let input = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let gate = CudaBuffer::from_tensor(gate).map_err(Error::Cuda)?;
    let weight = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_gdn_gated_rmsnorm_m8_bf16(
            input.ptr(),
            gate.ptr(),
            weight.ptr(),
            output.ptr(),
            epsilon,
            tokens as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

fn m8_tokens(
    tensor: &Tensor,
    name: &str,
    dtype: DType,
    trailing: usize,
    device: Device,
) -> Result<usize> {
    let dims = tensor.shape().dims();
    if tensor.dtype() != dtype
        || tensor.device() != device
        || dims.len() != 2
        || !(1..=8).contains(&dims[0])
        || dims[1] != trailing
    {
        return Err(Error::Other(format!(
            "Qwen3.5 {name} must be {dtype} [1..=8,{trailing}] on {device}, got {} {:?} on {}",
            tensor.dtype(),
            dims,
            tensor.device()
        )));
    }
    Ok(dims[0])
}

fn require(
    tensor: &Tensor,
    name: &str,
    dtype: DType,
    shape: &[usize],
    device: Device,
) -> Result<()> {
    if tensor.dtype() != dtype || tensor.shape().dims() != shape || tensor.device() != device {
        return Err(Error::Other(format!(
            "Qwen3.5 GDN {name} must be {dtype} {shape:?} on {device}, got {} {:?} on {}",
            tensor.dtype(),
            tensor.shape().dims(),
            tensor.device()
        )));
    }
    Ok(())
}

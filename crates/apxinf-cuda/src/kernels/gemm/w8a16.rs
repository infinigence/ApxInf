use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;

#[derive(Clone, Copy)]
pub struct W8A16WeightView<'a> {
    /// Physical output-major signed INT8 `[output,input]` bytes.
    pub values_i8: &'a CudaBuffer,
    pub scales_f32: &'a Tensor,
    pub input_dim: usize,
    pub output_dim: usize,
}

pub fn gemv_w8a16_write(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: W8A16WeightView<'_>,
    output: &Tensor,
) -> Result<()> {
    let device = Device::Cuda(ctx.device_id());
    if activation.dtype() != DType::BF16
        || activation.device() != device
        || activation.shape().dims() != [1, weight.input_dim]
        || weight.input_dim == 0
        || weight.output_dim == 0
        || weight.input_dim % 8 != 0
        || weight.output_dim % 8 != 0
    {
        return Err(Error::Other(format!(
            "W8A16 activation/shape contract mismatch: activation {} {:?} on {}, weight [{},{}]",
            activation.dtype(),
            activation.shape().dims(),
            activation.device(),
            weight.output_dim,
            weight.input_dim
        )));
    }
    if weight.values_i8.device() != ctx.device_id()
        || weight.values_i8.len() != weight.input_dim * weight.output_dim
        || weight.scales_f32.dtype() != DType::F32
        || weight.scales_f32.device() != device
        || weight.scales_f32.shape().dims() != [weight.output_dim]
        || output.dtype() != DType::BF16
        || output.device() != device
        || output.shape().dims() != [1, weight.output_dim]
    {
        return Err(Error::Other("W8A16 weight/output contract mismatch".into()));
    }
    let activation = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let scales = CudaBuffer::from_tensor(weight.scales_f32).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_w8a16_gemv_bf16(
            activation.ptr(),
            weight.values_i8.ptr(),
            scales.ptr(),
            output.ptr(),
            weight.input_dim as i32,
            weight.output_dim as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

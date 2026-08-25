use apxinf_core::{DType, Device, Error, Result, Tensor};

#[cfg(apxinf_marlin_sm89)]
use crate::ffi;
use crate::{CudaBuffer, CudaContext};

use super::w4a16::{W4A16Layout, W4A16WeightView};

pub struct MarlinW4A16WeightView<'a> {
    pub repacked_i32: &'a Tensor,
    pub scales_bf16: &'a Tensor,
    pub zero_points_i32: &'a Tensor,
    pub input_dim: usize,
    pub output_dim: usize,
}

pub struct MarlinWorkspace {
    reduce: CudaBuffer,
    locks: CudaBuffer,
}

pub struct MarlinPreparedWeight {
    transposed: CudaBuffer,
    repacked: Tensor,
    scales: Tensor,
    zero_points: Tensor,
    input_dim: usize,
    output_dim: usize,
}

impl MarlinPreparedWeight {
    pub fn new(ctx: &CudaContext, input_dim: usize, output_dim: usize) -> Result<Self> {
        if input_dim == 0 || input_dim % 128 != 0 || output_dim == 0 || output_dim % 256 != 0 {
            return Err(Error::Other(
                "Marlin prepared weight requires K%128=0 and N%256=0".into(),
            ));
        }
        let packed_bytes = input_dim
            .checked_mul(output_dim)
            .and_then(|value| value.checked_div(2))
            .ok_or_else(|| Error::Other("Marlin packed weight size overflow".into()))?;
        let groups = input_dim / 32;
        let repacked = CudaBuffer::alloc(packed_bytes, ctx.device_id())
            .map_err(Error::Cuda)?
            .into_tensor(
                apxinf_core::Shape::new(vec![input_dim / 16, 2 * output_dim]),
                DType::I32,
            );
        let scales = CudaBuffer::alloc(
            groups * output_dim * DType::BF16.size_in_bytes(),
            ctx.device_id(),
        )
        .map_err(Error::Cuda)?
        .into_tensor(
            apxinf_core::Shape::new(vec![groups, output_dim]),
            DType::BF16,
        );
        let zero_points = CudaBuffer::alloc(
            groups * (output_dim / 8) * DType::I32.size_in_bytes(),
            ctx.device_id(),
        )
        .map_err(Error::Cuda)?
        .into_tensor(
            apxinf_core::Shape::new(vec![groups, output_dim / 8]),
            DType::I32,
        );
        Ok(Self {
            transposed: CudaBuffer::alloc(packed_bytes, ctx.device_id()).map_err(Error::Cuda)?,
            repacked,
            scales,
            zero_points,
            input_dim,
            output_dim,
        })
    }

    pub fn prepare(&self, ctx: &CudaContext, source: W4A16WeightView<'_>) -> Result<()> {
        if ctx.caps().sm != 89
            || source.input_dim != self.input_dim
            || source.output_dim != self.output_dim
            || source.group_size != 32
            || source.layout != W4A16Layout::CompressedTensorsPackQuantized
        {
            return Err(Error::Other(
                "Marlin source weight contract mismatch".into(),
            ));
        }
        let device = Device::Cuda(ctx.device_id());
        let groups = self.input_dim / 32;
        for (tensor, dtype, shape) in [
            (
                source.packed_i32,
                DType::I32,
                vec![self.output_dim, self.input_dim / 8],
            ),
            (
                source.scales_bf16,
                DType::BF16,
                vec![self.output_dim, groups],
            ),
            (
                source.zero_points_i32,
                DType::I32,
                vec![self.output_dim / 8, groups],
            ),
        ] {
            if tensor.device() != device
                || tensor.dtype() != dtype
                || tensor.shape().dims() != shape
            {
                return Err(Error::Other(
                    "Marlin original-layout tensor contract mismatch".into(),
                ));
            }
        }
        let packed = CudaBuffer::from_tensor(source.packed_i32).map_err(Error::Cuda)?;
        let original_scales = CudaBuffer::from_tensor(source.scales_bf16).map_err(Error::Cuda)?;
        let original_zero = CudaBuffer::from_tensor(source.zero_points_i32).map_err(Error::Cuda)?;
        let repacked = CudaBuffer::from_tensor(&self.repacked).map_err(Error::Cuda)?;
        let scales = CudaBuffer::from_tensor(&self.scales).map_err(Error::Cuda)?;
        let zero = CudaBuffer::from_tensor(&self.zero_points).map_err(Error::Cuda)?;
        #[cfg(apxinf_marlin_sm89)]
        unsafe {
            ffi::check_cuda(ffi::apxinf_static_marlin_repack_u4(
                packed.ptr(),
                self.transposed.ptr(),
                repacked.ptr(),
                self.output_dim as i32,
                self.input_dim as i32,
                ctx.stream().handle(),
            ))
            .and_then(|_| {
                ffi::check_cuda(ffi::apxinf_static_marlin_transform_scales_zero_u4_group32(
                    original_scales.ptr(),
                    original_zero.ptr(),
                    scales.ptr(),
                    zero.ptr(),
                    self.output_dim as i32,
                    self.input_dim as i32,
                    ctx.stream().handle(),
                ))
            })
            .map_err(Error::Cuda)
        }
        #[cfg(not(apxinf_marlin_sm89))]
        {
            let _ = (
                packed,
                original_scales,
                original_zero,
                repacked,
                scales,
                zero,
            );
            Err(Error::Other(
                "Marlin transform was not compiled for this target".into(),
            ))
        }
    }

    pub fn view(&self) -> MarlinW4A16WeightView<'_> {
        MarlinW4A16WeightView {
            repacked_i32: &self.repacked,
            scales_bf16: &self.scales,
            zero_points_i32: &self.zero_points,
            input_dim: self.input_dim,
            output_dim: self.output_dim,
        }
    }
}

impl MarlinWorkspace {
    pub fn new(ctx: &CudaContext) -> Result<Self> {
        let sms = ctx.caps().multiprocessor_count as usize;
        let reduce_bytes = sms
            .checked_mul(64)
            .and_then(|value| value.checked_mul(256))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Other("Marlin reduce workspace size overflow".into()))?;
        let lock_bytes = sms
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| Error::Other("Marlin lock workspace size overflow".into()))?;
        Ok(Self {
            reduce: CudaBuffer::alloc(reduce_bytes, ctx.device_id()).map_err(Error::Cuda)?,
            locks: CudaBuffer::alloc_zeros(lock_bytes, ctx.device_id()).map_err(Error::Cuda)?,
        })
    }
}

pub fn w4a16_marlin_write(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: MarlinW4A16WeightView<'_>,
    output: &Tensor,
    workspace: &MarlinWorkspace,
) -> Result<()> {
    if ctx.caps().sm != 89 {
        return Err(Error::Other(format!(
            "Marlin W4A16 is frozen for SM89, got SM{}",
            ctx.caps().sm
        )));
    }
    let device = Device::Cuda(ctx.device_id());
    let dims = activation.shape().dims();
    let rows = dims.first().copied().unwrap_or(0);
    if activation.dtype() != DType::BF16
        || activation.device() != device
        || dims != [rows, weight.input_dim]
        || !(1..=64).contains(&rows)
        || weight.input_dim == 0
        || weight.output_dim == 0
        || weight.input_dim % 128 != 0
        || weight.output_dim % 256 != 0
        || output.dtype() != DType::BF16
        || output.device() != device
        || output.shape().dims() != [rows, weight.output_dim]
    {
        return Err(Error::Other(
            "Marlin W4A16 activation/output contract mismatch".into(),
        ));
    }
    let groups = weight.input_dim / 32;
    for (tensor, dtype, shape) in [
        (
            weight.repacked_i32,
            DType::I32,
            vec![weight.input_dim / 16, 2 * weight.output_dim],
        ),
        (
            weight.scales_bf16,
            DType::BF16,
            vec![groups, weight.output_dim],
        ),
        (
            weight.zero_points_i32,
            DType::I32,
            vec![groups, weight.output_dim / 8],
        ),
    ] {
        if tensor.dtype() != dtype || tensor.device() != device || tensor.shape().dims() != shape {
            return Err(Error::Other(
                "Marlin W4A16 transformed weight contract mismatch".into(),
            ));
        }
    }
    let activation = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let repacked = CudaBuffer::from_tensor(weight.repacked_i32).map_err(Error::Cuda)?;
    let scales = CudaBuffer::from_tensor(weight.scales_bf16).map_err(Error::Cuda)?;
    let zero = CudaBuffer::from_tensor(weight.zero_points_i32).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;

    #[cfg(apxinf_marlin_sm89)]
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_marlin_bf16_u4_group32(
            activation.ptr(),
            repacked.ptr(),
            scales.ptr(),
            zero.ptr(),
            output.ptr(),
            workspace.reduce.ptr(),
            workspace.locks.ptr(),
            rows as i32,
            weight.output_dim as i32,
            weight.input_dim as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
    #[cfg(not(apxinf_marlin_sm89))]
    {
        let _ = (activation, repacked, scales, zero, output, workspace);
        Err(Error::Other(
            "Marlin W4A16 was not compiled for this target".into(),
        ))
    }
}

//! Sampling — CPU fallback with direct H2D transfer.
//!
//! PI0.5 is a flow-matching model; only the normal generator is used.
//! We deliberately avoid creating an `AscendBackend` here because that would
//! create a second `AscendContext`, which overwrites the global context and
//! gets destroyed when the backend drops — leaving the thread with a stale
//! context pointer (ACL_ERROR_RT_CONTEXT_NULL = 107002).

use apxinf_core::{
    CpuNormalGenerator, CpuTokenSampler, Device, Error, NormalGenerator, Result,
    Storage, Tensor, TokenSampler, TokenSamplingSpec,
};

use crate::l3::context::AscendNewContext;

/// Create a token sampler (CPU fallback).
pub fn create_token_sampler(
    _ctx: &AscendNewContext,
    spec: TokenSamplingSpec,
) -> Result<Box<dyn TokenSampler>> {
    Ok(Box::new(CpuTokenSampler::new(spec)?))
}

/// Create a normal generator (CPU Philox RNG → H2D).
///
/// Does NOT create an `AscendBackend` — that would create a second ACL
/// context and corrupt the global context state.
pub fn create_normal_generator(
    ctx: &AscendNewContext,
    output: Tensor,
) -> Result<Box<dyn NormalGenerator>> {
    ctx.ensure_context();
    if output.device() != Device::Cpu {
        // Copy the device tensor to CPU, create a CPU normal generator,
        // then wrap it so generated noise is H2D-copied back to the device
        // tensor on each `generate` call.
        let dev_ptr = match output.storage() {
            Storage::Gpu { handle, .. } => handle.ptr(),
            _ => return Err(Error::Other("normal_generator: output not on device".into())),
        };
        let num_bytes = output.size_in_bytes();
        let mut cpu_buf = vec![0u8; num_bytes];
        crate::transfers::d2h(
            cpu_buf.as_mut_ptr() as *mut std::ffi::c_void,
            dev_ptr as *const std::ffi::c_void,
            num_bytes,
        )
        .map_err(|e| Error::Other(format!("normal_generator D2H: {e}")))?;
        let cpu_tensor = unsafe {
            Tensor::from_raw_parts_unchecked(
                output.shape().clone(),
                output.dtype(),
                Device::Cpu,
                Storage::cpu_from_bytes(cpu_buf),
            )
        };
        let cpu_gen = CpuNormalGenerator::new(cpu_tensor)?;
        return Ok(Box::new(AscendNormalGenerator {
            device_output: output,
            cpu_gen,
        }));
    }
    Ok(Box::new(CpuNormalGenerator::new(output)?))
}

/// Normal generator that fills a CPU buffer then copies to NPU.
pub struct AscendNormalGenerator {
    device_output: Tensor,
    cpu_gen: CpuNormalGenerator,
}

impl NormalGenerator for AscendNormalGenerator {
    fn output(&self) -> &Tensor {
        &self.device_output
    }

    fn generate(&mut self, rng: apxinf_core::RngKey) -> Result<&Tensor> {
        let cpu_result = self.cpu_gen.generate(rng)?;
        let dev_ptr = match self.device_output.storage() {
            Storage::Gpu { handle, .. } => handle.ptr(),
            _ => return Err(Error::Other("normal_generator: output not on device".into())),
        };
        let num_bytes = cpu_result.size_in_bytes();
        let cpu_bytes = match cpu_result.storage() {
            Storage::Cpu(bytes) => bytes.as_slice(),
            _ => return Err(Error::Other("normal_generator: CPU buffer missing".into())),
        };
        crate::transfers::h2d(
            dev_ptr as *mut std::ffi::c_void,
            cpu_bytes.as_ptr() as *const std::ffi::c_void,
            num_bytes,
        )
        .map_err(|e| Error::Other(format!("normal_generator H2D: {e}")))?;
        Ok(&self.device_output)
    }
}

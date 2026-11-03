//! Ascend sampling — CPU fallback.
//!
//! PI0.5 is a flow-matching model that doesn't use categorical token
//! sampling. The normal generator produces noise on CPU then transfers
//! to device, reusing apxinf-core's deterministic Philox RNG.

use apxinf_core::{
    Backend, CpuNormalGenerator, CpuTokenSampler, Device, Error, NormalGenerator, Result,
    SamplingBackend, Storage, Tensor, TokenSampler, TokenSamplingSpec,
};

use crate::backend::AscendBackend;
use crate::transfers;

impl SamplingBackend for AscendBackend {
    fn create_token_sampler(&self, spec: TokenSamplingSpec) -> Result<Box<dyn TokenSampler>> {
        Ok(Box::new(CpuTokenSampler::new(spec)?))
    }

    fn create_normal_generator(&self, output: Tensor) -> Result<Box<dyn NormalGenerator>> {
        if output.device() != Device::Cpu {
            let cpu_tensor = self.to_cpu(&output)?;
            let cpu_gen = CpuNormalGenerator::new(cpu_tensor)?;
            return Ok(Box::new(AscendNormalGenerator {
                device_output: output,
                cpu_gen,
            }));
        }
        Ok(Box::new(CpuNormalGenerator::new(output)?))
    }
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
        transfers::h2d(
            dev_ptr as *mut std::ffi::c_void,
            cpu_bytes.as_ptr() as *const std::ffi::c_void,
            num_bytes,
        )
        .map_err(|e| Error::Other(format!("normal_generator H2D: {e}")))?;
        Ok(&self.device_output)
    }
}

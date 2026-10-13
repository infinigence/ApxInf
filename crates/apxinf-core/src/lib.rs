mod backend;
mod dtype;
mod error;
mod kv_cache;
mod ops;
mod op_impls;
mod sampling;
mod shape;
pub mod storage;
mod tensor;

pub use backend::{Backend, Graph, RopeKind};
pub use dtype::DType;
pub use error::{Error, Result};
pub use kv_cache::{CpuKVCache, KvCache};
pub use op_impls::cpu::CpuBackend;
pub use sampling::{
    philox4x32_10, standard_normal_f32, uniform_f32, NextTokenLogits,
    NormalGenerator, RngKey, SamplingBackend, TokenPenalties, TokenSample,
    TokenSampler, TokenSamplingInit, TokenSamplingParams, TokenSamplingSpec,
    TokenSelection,
};
pub use shape::Shape;
pub use storage::Storage;
pub use tensor::Tensor;

/// Represents where a tensor lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(usize),
    /// An AMD GPU addressed through ROCm/HIP.
    ///
    /// Separate from `Cuda` so a tensor records which runtime owns its
    /// allocation, and so both backends can exist in one process. The CUDA
    /// crates reject it through the fallback arm they already have for
    /// non-CUDA devices.
    Hip(usize),
}

impl Device {
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Cuda(_) | Device::Hip(_))
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(id) => write!(f, "cuda:{id}"),
            Device::Hip(id) => write!(f, "hip:{id}"),
        }
    }
}

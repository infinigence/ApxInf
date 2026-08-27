mod backend;
mod dtype;
mod error;
mod kv_cache;
mod op_impls;
mod ops;
mod shape;
pub mod storage;
mod tensor;

pub use backend::{Backend, Graph, RopeKind};
pub use dtype::DType;
pub use error::{Error, Result};
pub use kv_cache::{CpuKVCache, KvCache};
pub use op_impls::cpu::CpuBackend;
pub use shape::Shape;
pub use storage::Storage;
pub use tensor::Tensor;

/// Represents where a tensor lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

impl Device {
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(id) => write!(f, "cuda:{id}"),
        }
    }
}

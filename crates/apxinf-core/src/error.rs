use thiserror::Error;

use crate::{DType, Device};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    /// A valid operator contract has no implementation in the selected backend.
    #[error("operator not implemented by this backend: {0}")]
    UnsupportedOp(&'static str),

    /// A portable operator contract was violated by its arguments. The reason is
    /// static so validation never allocates on the error path.
    #[error("portable contract violation: {0}")]
    Contract(&'static str),

    /// A dtype falls outside the set an operator accepts.
    #[error("unsupported dtype {got}: expected one of {allowed}")]
    UnsupportedDType { got: DType, allowed: &'static str },

    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("dtype mismatch: expected {expected}, got {got}")]
    DTypeMismatch { expected: DType, got: DType },

    #[error("device mismatch: expected {expected}, got {got}")]
    DeviceMismatch { expected: Device, got: Device },

    #[error("invalid axis {axis} for tensor with {ndim} dimensions")]
    InvalidAxis { axis: usize, ndim: usize },

    #[error("cannot reshape tensor of {src_numel} elements into shape with {dst_numel} elements")]
    ReshapeError { src_numel: usize, dst_numel: usize },

    #[error("matmul dimension mismatch: [{m}x{k1}] @ [{k2}x{n}]")]
    MatmulDimMismatch { m: usize, k1: usize, k2: usize, n: usize },

    #[error("data length mismatch: expected {expected} bytes, got {got} bytes")]
    DataLengthMismatch { expected: usize, got: usize },

    #[error("operation not supported on device {0}")]
    UnsupportedDevice(Device),

    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

impl Error {
    /// The operator name when this error means "backend lacks an implementation".
    /// Lets a portable path log which op it fell back on without string matching.
    pub fn unsupported_op(&self) -> Option<&'static str> {
        match self {
            Error::UnsupportedOp(name) => Some(name),
            _ => None,
        }
    }
}

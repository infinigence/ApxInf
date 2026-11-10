use thiserror::Error;

use crate::{DType, Device};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
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
    MatmulDimMismatch {
        m: usize,
        k1: usize,
        k2: usize,
        n: usize,
    },

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

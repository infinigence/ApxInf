//! Build-time fallback for CUDA installations without TensorRT development files.

use std::path::Path;

use crate::{CudaBuffer, CudaContext};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorMode {
    Input,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorDType {
    F32,
    F16,
    I8,
    I32,
    Bool,
    U8,
    Fp8,
    BF16,
    I64,
    I4,
    Fp4,
}

impl TensorDType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I64 => 8,
            _ => 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub mode: TensorMode,
    pub dtype: TensorDType,
    pub shape: Vec<i64>,
}

pub struct Engine;

impl Engine {
    pub fn load(path: &Path) -> Result<Self, String> {
        Err(format!(
            "TensorRT support was not compiled (cannot load {}); install NvInfer.h and libnvinfer before building ApxInf",
            path.display()
        ))
    }
    pub fn tensors(&self) -> Result<Vec<TensorInfo>, String> {
        Err("TensorRT support was not compiled".into())
    }
    pub fn mode(&self, _tensor: &str) -> Result<TensorMode, String> {
        Err("TensorRT support was not compiled".into())
    }
    pub fn dtype(&self, _tensor: &str) -> Result<TensorDType, String> {
        Err("TensorRT support was not compiled".into())
    }
    pub fn shape(&self, _tensor: &str) -> Result<Vec<i64>, String> {
        Err("TensorRT support was not compiled".into())
    }
    pub fn set_input_shape(&self, _tensor: &str, _dims: &[i64]) -> Result<(), String> {
        Err("TensorRT support was not compiled".into())
    }
    pub fn set_address(&self, _tensor: &str, _buffer: &CudaBuffer) -> Result<(), String> {
        Err("TensorRT support was not compiled".into())
    }
    pub fn enqueue(&self, _context: &CudaContext) -> Result<(), String> {
        Err("TensorRT support was not compiled".into())
    }
}

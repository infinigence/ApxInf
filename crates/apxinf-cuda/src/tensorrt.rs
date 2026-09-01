//! Thin, owning TensorRT execution wrapper.
//!
//! TensorRT remains an optional platform capability. The C++ ABI boundary is
//! deliberately confined to `adapters/tensorrt_adapter.cpp`; model code sees
//! checked names, shapes, data types, and ApxInf-owned CUDA buffers.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::ptr::NonNull;

use crate::{CudaBuffer, CudaContext};

unsafe extern "C" {
    fn apxinf_trt_last_error() -> *const c_char;
    fn apxinf_trt_load(path: *const c_char) -> *mut c_void;
    fn apxinf_trt_destroy(engine: *mut c_void);
    fn apxinf_trt_num_io(engine: *const c_void) -> i32;
    fn apxinf_trt_tensor_name(engine: *const c_void, index: i32) -> *const c_char;
    fn apxinf_trt_tensor_mode(engine: *const c_void, name: *const c_char) -> i32;
    fn apxinf_trt_tensor_dtype(engine: *const c_void, name: *const c_char) -> i32;
    fn apxinf_trt_tensor_shape(
        engine: *const c_void,
        name: *const c_char,
        dims: *mut i64,
        capacity: i32,
    ) -> i32;
    fn apxinf_trt_set_input_shape(
        engine: *mut c_void,
        name: *const c_char,
        dims: *const i64,
        rank: i32,
    ) -> c_int;
    fn apxinf_trt_set_address(
        engine: *mut c_void,
        name: *const c_char,
        address: *mut c_void,
    ) -> c_int;
    fn apxinf_trt_enqueue(engine: *mut c_void, stream: *mut c_void) -> c_int;
}

fn last_error(operation: &str) -> String {
    let detail = unsafe {
        let value = apxinf_trt_last_error();
        (!value.is_null())
            .then(|| CStr::from_ptr(value).to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    if detail.is_empty() {
        operation.to_owned()
    } else {
        format!("{operation}: {detail}")
    }
}

fn name(value: &str) -> Result<CString, String> {
    CString::new(value).map_err(|_| format!("TensorRT tensor name contains NUL: {value:?}"))
}

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
            Self::I8 | Self::Bool | Self::U8 | Self::Fp8 => 1,
            Self::I4 | Self::Fp4 => 1,
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

pub struct Engine {
    raw: NonNull<c_void>,
}

impl Engine {
    pub fn load(path: &Path) -> Result<Self, String> {
        let path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| format!("TensorRT engine path contains NUL: {}", path.display()))?;
        let raw = NonNull::new(unsafe { apxinf_trt_load(path.as_ptr()) })
            .ok_or_else(|| last_error("load TensorRT engine"))?;
        Ok(Self { raw })
    }

    pub fn tensors(&self) -> Result<Vec<TensorInfo>, String> {
        let count = unsafe { apxinf_trt_num_io(self.raw.as_ptr()) };
        if count < 0 {
            return Err(last_error("query TensorRT I/O count"));
        }
        (0..count)
            .map(|index| {
                let raw_name = unsafe { apxinf_trt_tensor_name(self.raw.as_ptr(), index) };
                if raw_name.is_null() {
                    return Err(format!("TensorRT I/O {index} has no name"));
                }
                let value = unsafe { CStr::from_ptr(raw_name) }
                    .to_string_lossy()
                    .into_owned();
                Ok(TensorInfo {
                    mode: self.mode(&value)?,
                    dtype: self.dtype(&value)?,
                    shape: self.shape(&value)?,
                    name: value,
                })
            })
            .collect()
    }

    pub fn mode(&self, tensor: &str) -> Result<TensorMode, String> {
        match unsafe { apxinf_trt_tensor_mode(self.raw.as_ptr(), name(tensor)?.as_ptr()) } {
            0 => Ok(TensorMode::Input),
            1 => Ok(TensorMode::Output),
            value => Err(format!("unknown TensorRT tensor mode {value} for {tensor}")),
        }
    }

    pub fn dtype(&self, tensor: &str) -> Result<TensorDType, String> {
        match unsafe { apxinf_trt_tensor_dtype(self.raw.as_ptr(), name(tensor)?.as_ptr()) } {
            0 => Ok(TensorDType::F32),
            1 => Ok(TensorDType::F16),
            2 => Ok(TensorDType::I8),
            3 => Ok(TensorDType::I32),
            4 => Ok(TensorDType::Bool),
            5 => Ok(TensorDType::U8),
            6 => Ok(TensorDType::Fp8),
            7 => Ok(TensorDType::BF16),
            8 => Ok(TensorDType::I64),
            9 => Ok(TensorDType::I4),
            10 => Ok(TensorDType::Fp4),
            value => Err(format!("unknown TensorRT dtype {value} for {tensor}")),
        }
    }

    pub fn shape(&self, tensor: &str) -> Result<Vec<i64>, String> {
        let tensor = name(tensor)?;
        let mut dims = [0i64; 16];
        let rank = unsafe {
            apxinf_trt_tensor_shape(
                self.raw.as_ptr(),
                tensor.as_ptr(),
                dims.as_mut_ptr(),
                dims.len() as i32,
            )
        };
        if rank < 0 {
            return Err(last_error("query TensorRT tensor shape"));
        }
        Ok(dims[..rank as usize].to_vec())
    }

    pub fn set_input_shape(&self, tensor: &str, dims: &[i64]) -> Result<(), String> {
        let tensor = name(tensor)?;
        let status = unsafe {
            apxinf_trt_set_input_shape(
                self.raw.as_ptr(),
                tensor.as_ptr(),
                dims.as_ptr(),
                dims.len() as i32,
            )
        };
        (status == 0)
            .then_some(())
            .ok_or_else(|| last_error("set TensorRT input shape"))
    }

    pub fn set_address(&self, tensor: &str, buffer: &CudaBuffer) -> Result<(), String> {
        let tensor = name(tensor)?;
        let status = unsafe {
            apxinf_trt_set_address(self.raw.as_ptr(), tensor.as_ptr(), buffer.address().ptr())
        };
        (status == 0)
            .then_some(())
            .ok_or_else(|| last_error("set TensorRT tensor address"))
    }

    pub fn enqueue(&self, context: &CudaContext) -> Result<(), String> {
        let status = unsafe {
            apxinf_trt_enqueue(self.raw.as_ptr(), context.stream().handle() as *mut c_void)
        };
        (status == 0)
            .then_some(())
            .ok_or_else(|| last_error("enqueue TensorRT engine"))
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { apxinf_trt_destroy(self.raw.as_ptr()) }
    }
}

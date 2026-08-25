use half::{bf16, f16};

use crate::{DType, Device, Error, Result, Shape, Storage};

/// A multi-dimensional array of elements stored contiguously in row-major order.
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Shape,
    dtype: DType,
    device: Device,
    storage: Storage,
}

impl Tensor {
    // ── Constructors ────────────────────────────────────────────────

    /// Create a tensor from raw bytes. Caller must ensure `data` length
    /// matches `shape.numel() * dtype.size_in_bytes()`.
    pub fn from_raw(shape: Shape, dtype: DType, device: Device, data: Vec<u8>) -> Result<Self> {
        let expected = shape.numel() * dtype.size_in_bytes();
        if data.len() != expected {
            return Err(Error::DataLengthMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Self {
            shape,
            dtype,
            device,
            storage: Storage::cpu_from_bytes(data),
        })
    }

    /// Create a tensor directly from components (used by apxinf-cuda for GPU tensors).
    pub fn from_raw_parts(shape: Shape, dtype: DType, device: Device, storage: Storage) -> Self {
        Self {
            shape,
            dtype,
            device,
            storage,
        }
    }

    /// Create a zero-filled tensor on CPU.
    pub fn zeros(shape: impl Into<Shape>, dtype: DType) -> Self {
        let shape = shape.into();
        let num_bytes = shape.numel() * dtype.size_in_bytes();
        Self {
            shape,
            dtype,
            device: Device::Cpu,
            storage: Storage::cpu_zeros(num_bytes),
        }
    }

    /// Create a tensor from an f32 slice.
    pub fn from_f32(shape: impl Into<Shape>, data: &[f32]) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.numel() {
            return Err(Error::DataLengthMismatch {
                expected: shape.numel() * 4,
                got: data.len() * 4,
            });
        }
        let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        Ok(Self {
            shape,
            dtype: DType::F32,
            device: Device::Cpu,
            storage: Storage::cpu_from_bytes(bytes),
        })
    }

    /// Create a tensor from a bf16 slice.
    pub fn from_bf16(shape: impl Into<Shape>, data: &[bf16]) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.numel() {
            return Err(Error::DataLengthMismatch {
                expected: shape.numel() * 2,
                got: data.len() * 2,
            });
        }
        let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        Ok(Self {
            shape,
            dtype: DType::BF16,
            device: Device::Cpu,
            storage: Storage::cpu_from_bytes(bytes),
        })
    }

    /// Create a tensor from an IEEE fp16 slice.
    pub fn from_f16(shape: impl Into<Shape>, data: &[f16]) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.numel() {
            return Err(Error::DataLengthMismatch {
                expected: shape.numel() * 2,
                got: data.len() * 2,
            });
        }
        let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        Ok(Self {
            shape,
            dtype: DType::F16,
            device: Device::Cpu,
            storage: Storage::cpu_from_bytes(bytes),
        })
    }

    /// Create a tensor containing raw CUDA-compatible E4M3 bytes.
    pub fn from_f8_e4m3(shape: impl Into<Shape>, data: &[u8]) -> Result<Self> {
        Self::from_raw(shape.into(), DType::F8E4M3, Device::Cpu, data.to_vec())
    }

    /// Create a tensor from signed 32-bit integers without changing their
    /// bit representation. Packed INT4 checkpoints use this container dtype.
    pub fn from_i32(shape: impl Into<Shape>, data: &[i32]) -> Result<Self> {
        Self::from_raw(
            shape.into(),
            DType::I32,
            Device::Cpu,
            bytemuck::cast_slice(data).to_vec(),
        )
    }

    /// Create a tensor from signed 64-bit integers.
    pub fn from_i64(shape: impl Into<Shape>, data: &[i64]) -> Result<Self> {
        Self::from_raw(
            shape.into(),
            DType::I64,
            Device::Cpu,
            bytemuck::cast_slice(data).to_vec(),
        )
    }

    // ── Accessors ───────────────────────────────────────────────────

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn storage_mut(&mut self) -> &mut Storage {
        &mut self.storage
    }

    /// Total bytes used by this tensor's data.
    pub fn size_in_bytes(&self) -> usize {
        self.numel() * self.dtype.size_in_bytes()
    }

    // ── CPU data access ─────────────────────────────────────────────

    /// View the data as f32 slice (panics if dtype != F32 or not on CPU).
    pub fn as_f32(&self) -> Result<&[f32]> {
        self.ensure_cpu()?;
        self.ensure_dtype(DType::F32)?;
        let bytes = self.storage.as_cpu().unwrap();
        Ok(bytemuck::cast_slice(bytes))
    }

    /// View the data as mutable f32 slice.
    pub fn as_f32_mut(&mut self) -> Result<&mut [f32]> {
        self.ensure_cpu()?;
        self.ensure_dtype(DType::F32)?;
        let bytes = self.storage.as_cpu_mut().unwrap();
        Ok(bytemuck::cast_slice_mut(bytes))
    }

    /// View the data as bf16 slice.
    pub fn as_bf16(&self) -> Result<&[bf16]> {
        self.ensure_cpu()?;
        self.ensure_dtype(DType::BF16)?;
        let bytes = self.storage.as_cpu().unwrap();
        Ok(bytemuck::cast_slice(bytes))
    }

    pub fn as_f16(&self) -> Result<&[f16]> {
        self.ensure_cpu()?;
        self.ensure_dtype(DType::F16)?;
        Ok(bytemuck::cast_slice(self.storage.as_cpu().unwrap()))
    }

    pub fn as_f8_e4m3(&self) -> Result<&[u8]> {
        self.ensure_cpu()?;
        self.ensure_dtype(DType::F8E4M3)?;
        Ok(self.storage.as_cpu().unwrap())
    }

    pub fn as_i32(&self) -> Result<&[i32]> {
        self.ensure_cpu()?;
        self.ensure_dtype(DType::I32)?;
        Ok(bytemuck::cast_slice(self.storage.as_cpu().unwrap()))
    }

    pub fn as_i64(&self) -> Result<&[i64]> {
        self.ensure_cpu()?;
        self.ensure_dtype(DType::I64)?;
        Ok(bytemuck::cast_slice(self.storage.as_cpu().unwrap()))
    }

    /// Convert data to f32 regardless of stored dtype (copies if bf16).
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        self.ensure_cpu()?;
        match self.dtype {
            DType::F32 => Ok(self.as_f32()?.to_vec()),
            DType::F16 => Ok(self.as_f16()?.iter().map(|x| x.to_f32()).collect()),
            DType::BF16 => Ok(self.as_bf16()?.iter().map(|x| x.to_f32()).collect()),
            DType::F8E4M3 => Err(Error::Other(
                "raw E4M3 conversion requires an explicit quantization scale".into(),
            )),
            DType::I32 | DType::I64 => Err(Error::Other(
                "integer tensor conversion requires an explicit semantic operation".into(),
            )),
        }
    }

    // ── Shape operations ────────────────────────────────────────────

    /// Reshape the tensor (must preserve total element count).
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Result<Self> {
        let new_shape = new_shape.into();
        if new_shape.numel() != self.shape.numel() {
            return Err(Error::ReshapeError {
                src_numel: self.shape.numel(),
                dst_numel: new_shape.numel(),
            });
        }
        // For CPU tensors, share the data (clone the bytes).
        // For GPU tensors, share the handle (zero-copy metadata reshape).
        match &self.storage {
            Storage::Cpu(data) => Ok(Self {
                shape: new_shape,
                dtype: self.dtype,
                device: self.device,
                storage: Storage::Cpu(data.clone()),
            }),
            Storage::Gpu { device, handle } => Ok(Self {
                shape: new_shape,
                dtype: self.dtype,
                device: *device,
                storage: Storage::Gpu {
                    device: *device,
                    handle: crate::storage::GpuStorageHandle {
                        ptr: handle.ptr,
                        len: handle.len,
                        _prevent_leak: handle._prevent_leak.clone(),
                    },
                },
            }),
        }
    }

    /// Transpose the last two dimensions (for matmul).
    pub fn transpose_last_two(&self) -> Result<Self> {
        if self.ndim() < 2 {
            return Err(Error::Other(
                "transpose requires at least 2D tensor".to_string(),
            ));
        }
        let mut new_dims = self.shape.dims().to_vec();
        let n = new_dims.len();
        new_dims.swap(n - 1, n - 2);
        // Note: this only updates the shape metadata. For actual transposed
        // access we'd need stride-aware indexing. For now, we do a physical
        // transpose for CPU f32 tensors.
        self.ensure_cpu()?;
        let new_shape = Shape::new(new_dims);
        let src = self.to_f32_vec()?;
        let rows = self.shape.dims()[self.ndim() - 2];
        let cols = self.shape.dims()[self.ndim() - 1];
        let batch_size: usize = self.shape.dims()[..self.ndim() - 2].iter().product();
        let batch_size = if batch_size == 0 { 1 } else { batch_size };

        let mut dst = vec![0.0f32; src.len()];
        for b in 0..batch_size {
            let off = b * rows * cols;
            for r in 0..rows {
                for c in 0..cols {
                    dst[off + c * rows + r] = src[off + r * cols + c];
                }
            }
        }
        Tensor::from_f32(new_shape, &dst)
    }

    // ── CPU math operations ─────────────────────────────────────────

    /// CPU-side matrix multiplication.
    ///
    /// When a BLAS feature (`accelerate` or `openblas`) is enabled, this
    /// dispatches to the vendor BLAS library. Otherwise a naive Rust
    /// fallback is used.
    pub fn matmul_cpu(&self, other: &Tensor) -> Result<Self> {
        self.ensure_cpu()?;
        other.ensure_cpu()?;
        self.ensure_dtype(other.dtype())?;

        let out_shape = self.shape.matmul_shape(other.shape())?;

        let a = self.to_f32_vec()?;
        let b = other.to_f32_vec()?;

        let m = self.shape.dims()[self.ndim() - 2];
        let k = self.shape.dims()[self.ndim() - 1];
        let n = other.shape().dims()[other.ndim() - 1];
        let batch: usize = self.shape.dims()[..self.ndim() - 2]
            .iter()
            .product::<usize>()
            .max(1);

        let mut out = vec![0.0f32; out_shape.numel()];
        for batch_idx in 0..batch {
            let a_off = batch_idx * m * k;
            let b_off = batch_idx * k * n;
            let o_off = batch_idx * m * n;
            crate::ops::sgemm(m, k, n, &a[a_off..], &b[b_off..], &mut out[o_off..]);
        }

        Tensor::from_f32(out_shape, &out)
    }

    // ── Internal helpers ────────────────────────────────────────────

    fn ensure_cpu(&self) -> Result<()> {
        if self.device != Device::Cpu {
            return Err(Error::UnsupportedDevice(self.device));
        }
        Ok(())
    }

    fn ensure_dtype(&self, expected: DType) -> Result<()> {
        if self.dtype != expected {
            return Err(Error::DTypeMismatch {
                expected,
                got: self.dtype,
            });
        }
        Ok(())
    }

    /// Replace the internals (used by cuda module to swap storage after transfer).
    pub fn set_device_and_storage(&mut self, device: Device, storage: Storage) {
        self.device = device;
        self.storage = storage;
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor(shape={}, dtype={}, device={})",
            self.shape, self.dtype, self.device
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(vec![2, 3], DType::F32);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.size_in_bytes(), 24);
        let data = t.as_f32().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_from_f32() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_f32(vec![2, 3], &data).unwrap();
        assert_eq!(t.shape(), &Shape::new(vec![2, 3]));
        assert_eq!(t.as_f32().unwrap(), &data);
    }

    #[test]
    fn test_from_bf16() {
        let data: Vec<bf16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let t = Tensor::from_bf16(vec![2, 2], &data).unwrap();
        assert_eq!(t.dtype(), DType::BF16);
        assert_eq!(t.numel(), 4);
    }

    #[test]
    fn test_to_f32_vec_from_bf16() {
        let data: Vec<bf16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let t = Tensor::from_bf16(vec![3], &data).unwrap();
        let f32_data = t.to_f32_vec().unwrap();
        assert_eq!(f32_data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_reshape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_f32(vec![2, 3], &data).unwrap();
        let t2 = t.reshape(vec![3, 2]).unwrap();
        assert_eq!(t2.shape(), &Shape::new(vec![3, 2]));
        assert_eq!(t2.as_f32().unwrap(), &data);
    }

    #[test]
    fn test_reshape_mismatch() {
        let t = Tensor::zeros(vec![2, 3], DType::F32);
        assert!(t.reshape(vec![2, 4]).is_err());
    }

    #[test]
    fn test_matmul_cpu() {
        // [2, 3] @ [3, 2] = [2, 2]
        let a = Tensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::from_f32(vec![3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c = a.matmul_cpu(&b).unwrap();
        assert_eq!(c.shape(), &Shape::new(vec![2, 2]));
        let data = c.as_f32().unwrap();
        // Row 0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
        // Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
        assert_eq!(data, &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_transpose_2d() {
        let a = Tensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let at = a.transpose_last_two().unwrap();
        assert_eq!(at.shape(), &Shape::new(vec![3, 2]));
        let data = at.as_f32().unwrap();
        assert_eq!(data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_display() {
        let t = Tensor::zeros(vec![2, 3], DType::F32);
        assert_eq!(
            format!("{t}"),
            "Tensor(shape=[2, 3], dtype=f32, device=cpu)"
        );
    }
}

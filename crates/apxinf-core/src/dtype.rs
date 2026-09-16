/// Supported data types for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    /// NVIDIA/CUDA FP8 E4M3 finite-number encoding.
    F8E4M3,
    /// NVIDIA NVFP4 E2M1 data, packed two logical elements per byte.
    #[cfg(feature = "quantized-dtypes")]
    F4E2M1,
    /// Unsigned E4M3 scale-factor encoding used by NVFP4 block scaling.
    #[cfg(feature = "quantized-dtypes")]
    F8UE4M3,
    /// Signed 8-bit integer storage used by pre-quantized GEMM operands.
    #[cfg(feature = "quantized-dtypes")]
    I8,
    /// Signed 32-bit integer accumulation/output storage.
    #[cfg(feature = "quantized-dtypes")]
    I32,
}

impl DType {
    /// Size of the smallest addressable storage unit in bytes.
    ///
    /// Use [`DType::storage_bytes_for`] for tensor allocation because packed
    /// sub-byte formats can store multiple logical elements in one unit.
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F8E4M3 => 1,
            // This is the smallest addressable storage unit. Use
            // `storage_bytes_for` when computing a tensor's allocation size.
            #[cfg(feature = "quantized-dtypes")]
            DType::F4E2M1 | DType::F8UE4M3 => 1,
            #[cfg(feature = "quantized-dtypes")]
            DType::I8 => 1,
            #[cfg(feature = "quantized-dtypes")]
            DType::I32 => 4,
        }
    }

    /// Number of bytes needed for `elements` logical values.
    pub fn storage_bytes_for(self, elements: usize) -> Option<usize> {
        match self {
            #[cfg(feature = "quantized-dtypes")]
            DType::F4E2M1 => elements.checked_add(1).map(|value| value / 2),
            _ => elements.checked_mul(self.size_in_bytes()),
        }
    }

    /// Minimum byte alignment implied by the scalar storage type.
    pub fn alignment_in_bytes(self) -> usize {
        match self {
            #[cfg(feature = "quantized-dtypes")]
            DType::F4E2M1 => 1,
            _ => self.size_in_bytes(),
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::F8E4M3 => write!(f, "f8_e4m3"),
            #[cfg(feature = "quantized-dtypes")]
            DType::F4E2M1 => write!(f, "f4_e2m1"),
            #[cfg(feature = "quantized-dtypes")]
            DType::F8UE4M3 => write!(f, "f8_ue4m3"),
            #[cfg(feature = "quantized-dtypes")]
            DType::I8 => write!(f, "i8"),
            #[cfg(feature = "quantized-dtypes")]
            DType::I32 => write!(f, "i32"),
        }
    }
}

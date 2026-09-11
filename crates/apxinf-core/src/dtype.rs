/// Supported data types for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    /// NVIDIA/CUDA FP8 E4M3 finite-number encoding.
    F8E4M3,
    /// Signed 8-bit integer storage used by pre-quantized GEMM operands.
    #[cfg(feature = "quantized-dtypes")]
    I8,
    /// Signed 32-bit integer accumulation/output storage.
    #[cfg(feature = "quantized-dtypes")]
    I32,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F8E4M3 => 1,
            #[cfg(feature = "quantized-dtypes")]
            DType::I8 => 1,
            #[cfg(feature = "quantized-dtypes")]
            DType::I32 => 4,
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
            DType::I8 => write!(f, "i8"),
            #[cfg(feature = "quantized-dtypes")]
            DType::I32 => write!(f, "i32"),
        }
    }
}

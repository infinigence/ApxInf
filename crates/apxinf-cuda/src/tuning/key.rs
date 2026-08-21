use apxinf_core::DType;

use crate::device_caps::CudaDeviceCaps;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemmOp {
    Bf16,
    W8A8,
    Fp8F16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TuningDType {
    F32,
    F16,
    Bf16,
    F8E4M3,
    I8,
    I32,
    I64,
}

impl From<DType> for TuningDType {
    fn from(value: DType) -> Self {
        match value {
            DType::F32 => Self::F32,
            DType::F16 => Self::F16,
            DType::BF16 => Self::Bf16,
            DType::F8E4M3 => Self::F8E4M3,
            DType::I32 => Self::I32,
            DType::I64 => Self::I64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemmLayout {
    RowMajor,
    WeightOutputMajor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScaleMode {
    None,
    PerTensor,
    DynamicRowPerOutputChannel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Epilogue {
    None,
    Bias,
    BiasGelu,
    BiasResidual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceFingerprint {
    pub sm: u32,
    pub multiprocessor_count: u32,
}

impl From<&CudaDeviceCaps> for DeviceFingerprint {
    fn from(caps: &CudaDeviceCaps) -> Self {
        Self {
            sm: caps.sm,
            multiprocessor_count: caps.multiprocessor_count,
        }
    }
}

/// Complete physical contract used for GEMM tactic selection.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GemmTuningKey {
    pub op: GemmOp,
    pub device: DeviceFingerprint,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub activation_dtype: TuningDType,
    pub weight_dtype: TuningDType,
    pub output_dtype: TuningDType,
    pub layout: GemmLayout,
    pub scale_mode: ScaleMode,
    pub epilogue: Epilogue,
    pub workspace_limit: usize,
}

impl GemmTuningKey {
    pub(crate) fn bucket(&self) -> GemmBucketKey {
        GemmBucketKey {
            op: self.op,
            device: self.device,
            m_bucket: m_bucket(self.m),
            n: self.n,
            k: self.k,
            activation_dtype: self.activation_dtype,
            weight_dtype: self.weight_dtype,
            output_dtype: self.output_dtype,
            layout: self.layout,
            scale_mode: self.scale_mode,
            epilogue: self.epilogue,
            workspace_limit: self.workspace_limit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GemmBucketKey {
    op: GemmOp,
    device: DeviceFingerprint,
    m_bucket: usize,
    n: usize,
    k: usize,
    activation_dtype: TuningDType,
    weight_dtype: TuningDType,
    output_dtype: TuningDType,
    layout: GemmLayout,
    scale_mode: ScaleMode,
    epilogue: Epilogue,
    workspace_limit: usize,
}

fn m_bucket(m: usize) -> usize {
    m.max(1).checked_next_power_of_two().unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m_bucket_is_deterministic() {
        assert_eq!(m_bucket(0), 1);
        assert_eq!(m_bucket(1), 1);
        assert_eq!(m_bucket(10), 16);
        assert_eq!(m_bucket(512), 512);
        assert_eq!(m_bucket(522), 1024);
    }
}

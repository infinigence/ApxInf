//! CUDA-facing seam for the GR00T runtime.

#[cfg(feature = "cuda")]
pub(crate) use crate::accelerator::cuda::{
    downcast_arc, kernels, transfers, DeviceBuffer, RuntimeBackend,
};
#[cfg(feature = "cuda")]
pub(crate) use apxinf_cuda::{tuning::TuningMode, CudaBackend};

/// The model opts into packed8 only for its measured BF16 epilogues.
const fn use_packed8_bias_activation(
    sm: u32,
    rows: usize,
    cols: usize,
    activation: i32,
    legacy: bool,
) -> bool {
    sm == 110
        && !legacy
        && matches!(
            (rows, cols, activation),
            (41, 1536, 0) | (41, 4608, 0) | (41, 6144, 1) | (256, 4096, 1) | (512, 4096, 1)
        )
}

#[cfg(feature = "cuda")]
pub(super) fn try_packed8_bias_activation(
    backend: &RuntimeBackend,
    input: &apxinf_core::Tensor,
    bias: &apxinf_core::Tensor,
    activation: i32,
) -> apxinf_core::Result<Option<apxinf_core::Tensor>> {
    let [rows, cols] = input.shape().dims() else {
        return Err(apxinf_core::Error::Other(
            "GR00T packed8 bias activation expects a matrix".into(),
        ));
    };
    if !use_packed8_bias_activation(
        backend.context().caps().sm,
        *rows,
        *cols,
        activation,
        std::env::var_os("APXINF_GR00T_BF16_LEGACY_PACKED8_BIAS_ACTIVATION").is_some(),
    ) {
        return Ok(None);
    }
    kernels::activation::try_bias_activation_bf16_packed8(
        backend.context(),
        input,
        bias,
        activation,
    )
}

pub(super) const fn qk_rms_mrope_block_threads(
    sm: u32,
    seq_len: usize,
    head_dim: usize,
    query_heads: usize,
    key_heads: usize,
    legacy_threads: bool,
) -> u32 {
    let measured_arch_shape =
        (sm == 110 && (seq_len == 90 || seq_len == 156)) || (sm == 87 && seq_len == 156);
    if !legacy_threads
        && measured_arch_shape
        && head_dim == 128
        && query_heads == 16
        && key_heads == 8
    {
        128
    } else {
        256
    }
}

#[cfg(test)]
mod tests {
    use super::{qk_rms_mrope_block_threads as threads, use_packed8_bias_activation};

    #[test]
    fn qk_rms_mrope_threads_preserve_per_architecture_shapes() {
        for (sm, seq_len, expected) in [
            (110, 90, 128),
            (110, 156, 128),
            (87, 90, 256),
            (87, 156, 128),
            (89, 156, 256),
        ] {
            assert_eq!(threads(sm, seq_len, 128, 16, 8, false), expected);
            assert_eq!(threads(sm, seq_len, 128, 16, 8, true), 256);
        }
        assert_eq!(threads(110, 41, 128, 16, 8, false), 256);
        assert_eq!(threads(110, 90, 64, 16, 8, false), 256);
        assert_eq!(threads(87, 156, 128, 8, 8, false), 256);
    }

    #[test]
    fn packed8_policy_keeps_measured_shapes_and_legacy_opt_out() {
        for (rows, cols, activation) in [
            (41, 1536, 0),
            (41, 4608, 0),
            (41, 6144, 1),
            (256, 4096, 1),
            (512, 4096, 1),
        ] {
            assert!(use_packed8_bias_activation(
                110, rows, cols, activation, false
            ));
            assert!(!use_packed8_bias_activation(
                87, rows, cols, activation, false
            ));
            assert!(!use_packed8_bias_activation(
                110, rows, cols, activation, true
            ));
        }
        for (rows, cols, activation) in [(40, 1536, 0), (41, 1536, 1), (128, 4096, 1)] {
            assert!(!use_packed8_bias_activation(
                110, rows, cols, activation, false
            ));
        }
    }
}

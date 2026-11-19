use std::ffi::c_void;

pub(crate) use super::types::CudaStream;
pub(crate) use super::types::Runtime;

#[repr(C)]
pub(crate) struct FramesBindings {
    pub waveform: *const c_void,
    pub hann: *const c_void,
    pub frames: *mut c_void,
    pub stream: CudaStream,
}

#[repr(C)]
pub(crate) struct PowerBindings {
    pub re: *const c_void,
    pub im: *const c_void,
    pub power: *mut c_void,
    pub stream: CudaStream,
}

#[repr(C)]
pub(crate) struct LogMelBindings {
    pub spec: *const c_void,
    pub out_f16: *mut c_void,
    pub max_scratch: *mut c_void,
    pub stream: CudaStream,
}

unsafe extern "C" {
    pub(crate) fn apxinf_whisper_frames_launch(
        runtime: Runtime,
        n_samples: i64,
        n_fft: i64,
        hop: i64,
        pad: i64,
        frames: i64,
        bindings: *const FramesBindings,
    ) -> i32;

    pub(crate) fn apxinf_whisper_power_launch(
        runtime: Runtime,
        count: i64,
        bindings: *const PowerBindings,
    ) -> i32;

    pub(crate) fn apxinf_whisper_log_mel_launch(
        runtime: Runtime,
        rows: i64,
        cols: i64,
        bindings: *const LogMelBindings,
    ) -> i32;
}

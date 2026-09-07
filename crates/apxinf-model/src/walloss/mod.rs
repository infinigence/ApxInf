//! Walloss vision-language-action runtime.

#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
mod bf16_executor;
#[cfg(feature = "cuda")]
mod bf16_runtime;
#[cfg(any(feature = "cuda", test))]
mod config;
#[cfg(feature = "cuda")]
mod device_weights;
#[cfg(any(feature = "cuda", test))]
mod fp8;
#[cfg(feature = "cuda")]
mod fp8_weights;
#[cfg(any(feature = "cuda", test))]
mod geometry;
#[cfg(any(feature = "cuda", test))]
mod schedule;
#[cfg(any(feature = "cuda", test))]
mod weights;

#[cfg(feature = "cuda")]
pub(crate) use config::WallossImageProcessorConfig;
#[cfg(any(feature = "cuda", test))]
pub(crate) use config::{WallossConfig, WallossTextConfig, WallossVisionConfig};
#[cfg(feature = "cuda")]
pub(crate) use device_weights::DynamicFp8LinearWeights;
#[cfg(all(feature = "cuda", test))]
pub(crate) use fp8::decode_e4m3;
#[cfg(feature = "cuda")]
pub(crate) use fp8::{encode_e4m3, E4M3_MAX};
#[cfg(feature = "cuda")]
pub(crate) use fp8_weights::{
    WallossDynamicFp8LayerWeights, WallossDynamicFp8VisionBlockWeights,
    WallossDynamicFp8VisionWeights, WallossDynamicFp8Weights,
};
#[cfg(feature = "cuda")]
pub(crate) use geometry::DeviceVisionGeometry;
#[cfg(feature = "cuda")]
pub(crate) use geometry::{multimodal_position_ids, VisionGeometry};
#[cfg(feature = "cuda")]
pub(crate) use schedule::{sinusoidal_time_embedding, solver_times};
#[cfg(feature = "cuda")]
pub(crate) use weights::{
    WallossActionWeights, WallossLayerWeights, WallossVisionBlockWeights, WallossVisionWeights,
    WallossWeights,
};

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register("walloss-cuda", bf16_runtime::load_registered);
    crate::registry::register("wall-oss-cuda", bf16_runtime::load_registered);
    crate::registry::register("wall_oss_05-cuda", bf16_runtime::load_registered);
}

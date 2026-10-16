//! LLM model architectures and abstractions.

mod accelerator;
pub mod auto;
pub mod builtin;
pub mod d3;
pub mod debug;
mod generation_config;
pub mod gr00t;
pub mod llama;
pub mod llm_trait;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod metal_minicpm5;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod metal_qwen3;
pub mod pi05;
pub mod pi0fast;
pub mod profiling;
#[cfg(feature = "cuda")]
pub mod qwen38;
pub mod qwen3vl;
pub mod qwen_drive;
pub mod registry;
#[cfg(feature = "cuda")]
pub mod smolvla;
pub mod vla;
mod walloss;

pub use auto::{AutoModel, LoadOptions, LoadedModel, ModelPrecision, SyntheticWeights};
pub use builtin::register_builtin_models;
pub use d3::{
    D3Config, D3DecisionConfig, D3PackageConfig, D3TextConfig, D3VisionConfig, DecisionInput,
    DecisionOutput, DecisionRuntime, DecisionScore, D3_DEFAULT_TEXT_PROFILE_LIMIT,
    D3_DEFAULT_TEXT_PROFILE_TOKEN_BUDGET,
};
pub use debug::{DebugCapture, DebugConfig};
pub use generation_config::{GenerationConfigSource, GenerationOptions, SamplingMode};
#[cfg(feature = "cuda")]
pub use llama::{DecodeGraph, DecodeGraphConfig, DecodeGraphWeights, DecodeLayerWeights};
pub use llama::{GeneralLlama, KVCache, LlamaModel, LlamaWeights, TransformerLayer};
pub use llm_trait::{
    generate_streaming, generate_streaming_with_options, GeneratedToken, GenerationOutput,
    GenerationRequest, ImageInput, LlmCapabilities, LlmInput, LlmTrait, TextCompilationScope,
    TextPreparationState, TextPreparationStatus,
};
pub use pi05::{Pi05Config, Pi05PerformanceProfile};
pub use profiling::GenerationProfile;
pub use qwen3vl::{GeneralQwen3VL, Qwen3VLConfig, Qwen3VLTextWeights};
pub use qwen_drive::QwenDriveConfig;
#[cfg(feature = "cuda")]
pub use qwen_drive::QwenDriveModelRunner;
pub use registry::{get, list, register};
pub use vla::{
    Action, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent, Observation,
    PreparationStatus, PreparedInference, VisionObservation, VlaContract, VlaMetadata, VlaRequest,
    VlaRuntime,
};

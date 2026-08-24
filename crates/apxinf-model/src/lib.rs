//! LLM model architectures and abstractions.

mod accelerator;
pub mod builtin;
pub mod debug;
mod generation_config;
pub mod llama;
pub mod llm_trait;
pub mod registry;
pub mod auto;
pub mod profiling;
pub mod pi05;
pub mod qwen3vl;
pub mod vla;

pub use builtin::register_builtin_models;
pub use debug::{DebugCapture, DebugConfig};
pub use generation_config::{GenerationConfigSource, GenerationOptions, SamplingMode};
pub use llama::{GeneralLlama, LlamaModel, LlamaWeights, TransformerLayer, KVCache};
pub use llm_trait::{
    generate_streaming, generate_streaming_with_options, GeneratedToken, GenerationOutput,
    GenerationRequest, ImageInput, LlmCapabilities, LlmInput, LlmTrait,
};
pub use registry::{register, get, list};
pub use auto::{AutoModel, LoadOptions, LoadedModel, ModelPrecision, SyntheticWeights};
pub use profiling::GenerationProfile;
pub use pi05::{Pi05Config, Pi05PerformanceProfile};
pub use qwen3vl::{GeneralQwen3VL, Qwen3VLConfig, Qwen3VLTextWeights};
pub use vla::{
    Action, ImageLayout, InferenceSpec, InitialLatent, Observation,
    PreparedInference, VisionObservation, VlaRequest, VlaRuntime,
};
#[cfg(feature = "cuda")]
pub use llama::{DecodeGraph, DecodeGraphConfig, DecodeGraphWeights, DecodeLayerWeights};

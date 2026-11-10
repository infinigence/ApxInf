//! LLM model architectures and abstractions.

mod accelerator;
pub mod auto;
pub mod builtin;
pub mod debug;
pub mod llama;
pub mod llm_trait;
pub mod pi05;
pub mod profiling;
pub mod qwen35;
pub mod qwen3vl;
pub mod registry;
pub mod runtime;
pub mod vla;

pub use auto::{AutoModel, LoadOptions, LoadedModel, ModelPrecision, SyntheticWeights};
pub use builtin::register_builtin_models;
pub use debug::{DebugCapture, DebugConfig};
#[cfg(feature = "cuda")]
pub use llama::{DecodeGraph, DecodeGraphConfig, DecodeGraphWeights, DecodeLayerWeights};
pub use llama::{GeneralLlama, KVCache, LlamaModel, LlamaWeights, TransformerLayer};
pub use llm_trait::{generate_streaming, ImageInput, LlmCapabilities, LlmInput, LlmTrait};
pub use pi05::{Pi05Config, Pi05PerformanceProfile};
pub use profiling::GenerationProfile;
pub use qwen35::{
    AdmissionError, LayerType, Qwen35CheckpointInventory, Qwen35Config, Qwen35ConfigError,
    Qwen35LoaderError, Qwen35ModelConfig, IMAGE_TOKEN_ID, MODEL_VOCAB_SIZE,
};
pub use qwen3vl::{GeneralQwen3VL, Qwen3VLConfig, Qwen3VLTextWeights};
pub use registry::{get, list, register};
pub use runtime::{
    AdmissionDecision, CancellationToken, MultimodalPayload, RuntimeCapabilities, RuntimeError,
    RuntimeHandle, RuntimeRequest, RuntimeResult, RuntimeTicket, RuntimeWorker,
};
pub use vla::{
    Action, ImageLayout, InferenceSpec, Observation, PreparedInference, VisionObservation,
    VlaRuntime,
};

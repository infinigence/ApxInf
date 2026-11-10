pub mod admission;
pub mod attention;
pub mod config;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod gdn;
pub mod loader;
pub mod model;
pub mod runtime;
pub mod vision;
pub mod weights;

#[cfg(feature = "cuda")]
pub use cuda::{
    Qwen35AttentionProjectionTensors, Qwen35Bf16Projection, Qwen35CheckpointProjection,
    Qwen35CudaFullAttentionLayer, Qwen35CudaFullAttentionState, Qwen35CudaGdnLayer,
    Qwen35CudaGdnState, Qwen35MixedProjection,
};

pub use admission::{
    validate_input_ids, validate_input_ids_with_vocab, validate_total_budget, AdmissionError,
};
pub use attention::{
    apply_output_gate_f32, apply_partial_rope_f32, causal_gqa_attention_f32, causal_gqa_decode_f32,
    embedding_f32, gqa_expand_f32, lm_head_f32, qk_norm_f32, residual_add_f32, rms_norm_f32,
    swiglu_f32, AttentionError, FullAttentionKv, FullAttentionReferenceConfig,
    FullAttentionReferenceLayer, PackedLinearReference,
};
pub use config::{
    LayerType, Qwen35Config, Qwen35ConfigError, Qwen35ModelConfig, IMAGE_TOKEN_ID, MODEL_VOCAB_SIZE,
};
pub use gdn::{GdnDimensions, GdnError, GdnState};
pub use loader::{
    Bf16TensorPayload, GdnLayerPayload, LinearPayload, PackedLinearPayload,
    Qwen35CheckpointInventory, Qwen35LoaderError,
};
pub use model::{
    greedy_argmax, ExecutorError, GenerationOutput, Qwen35ReferenceExecutor, Qwen35RequestState,
    StopReason,
};
#[cfg(feature = "cuda")]
pub use runtime::{decode_step_batch, Qwen35CudaModel, Qwen35CudaSession};
pub use runtime::{multimodal_enabled, request_resident_bytes, request_state_bytes};
#[cfg(feature = "cuda")]
pub use vision::{Qwen35VisionTower, VisionProbe};
pub use weights::{PackedLinearLayout, WeightLayoutError};

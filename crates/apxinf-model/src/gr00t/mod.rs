//! NVIDIA GR00T model contracts.
//!
//! The N1.7 checkpoint uses a Qwen3-VL backbone and a flow-matching action
//! head.  Keep model-specific configuration and math here; CUDA operators
//! remain model-neutral under `apxinf-cuda`.

mod checkpoint;
mod config;
mod execution;
#[cfg(feature = "cuda")]
mod fp8;
mod input;
mod math;
mod options;
#[cfg(feature = "cuda")]
mod runtime;
mod weights;

pub use checkpoint::{load_backbone_config, Gr00tWeights};
pub use config::{Gr00tConfig, Gr00tDiffusionConfig, Gr00tVlSelfAttentionConfig};
pub use execution::{
    build_backbone_token_groups, dit_attention_source, Gr00tBackboneTokenGroups,
    Gr00tDitAttentionSource,
};
pub use input::{Gr00tInferenceSpec, Gr00tObservation};
pub use math::{
    action_timestep_embedding, dit_timestep_projection, euler_flow_step, flow_schedule,
    Gr00tFlowStep,
};
pub use options::Gr00tLoadOptions;
#[cfg(feature = "cuda")]
pub use runtime::Gr00tVlaRuntime;
pub use weights::{
    Gr00tActionEncoderWeights, Gr00tActionHeadWeights, Gr00tAttentionWeights,
    Gr00tCategoryLinearWeights, Gr00tCategoryMlpWeights, Gr00tDitBlockWeights,
    Gr00tEmbodimentWeights, Gr00tFeedForwardWeights, Gr00tLayerNormWeights, Gr00tLinearWeights,
    Gr00tMlpWeights, Gr00tVlSelfAttentionBlockWeights,
};

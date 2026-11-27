//! OpenAI/Anthropic-compatible, batch-one serving for ApxInf LLMs and VLMs.
//!
//! The HTTP runtime never owns a model. A bounded queue feeds one dedicated
//! worker thread, and that thread constructs and retains the mutable model and
//! device context for their complete lifetime.

pub mod anthropic;
mod api;
mod engine;
mod error;
mod media;
pub mod protocol;
mod responses;
pub mod tool_calling;
mod worker;

pub use api::{router, router_with_limits, serve, serve_blocking, ApiConfig, ServeConfig};
pub use engine::{ApxInfEngineConfig, ChatEngine, EngineOutput, EngineRequest, SamplingDefault};
pub use error::ApiError;
pub use tool_calling::ToolFormat;
pub use worker::{EngineStreamEvent, WorkerClient};

pub use media::MediaLimits;

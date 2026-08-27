//! Model weight loading from SafeTensors and GGUF formats.

pub mod config;
pub mod gguf;
pub mod manifest;
pub mod safetensors;
pub mod w4;

pub use config::ModelConfig;
pub use manifest::{
    LoaderManifest, ManifestDType, PackAxis, QuantizationRole, TensorManifest,
    LOADER_MANIFEST_SCHEMA, QWEN35_MODEL_REVISION,
};

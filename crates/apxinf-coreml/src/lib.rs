//! Core ML ML Program writer and runtime for the Apple Neural Engine. The
//! writer is portable Rust; the runtime needs the `native` feature on Apple
//! Silicon macOS and is absent elsewhere. No Python is used.
pub mod blob;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod placement;
pub mod program;
pub mod proto;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod runtime;

#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub use placement::{
    ComputeDevice, ComputeDeviceKind, DeviceUsage, ExpectedPlacement, FunctionPlacement,
    InputBinding, OperationPlacement, PlacementInput,
};
pub use program::{Operand, Package, ProgramBuilder, Var};
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub use runtime::{compile, ComputeUnits, Model, ProgramCache};

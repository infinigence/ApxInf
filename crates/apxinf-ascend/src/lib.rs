//! Huawei Ascend NPU backend for APXInf.
//!
//! This crate implements `apxinf_core::Backend` for Huawei Ascend NPUs
//! (950PR / 910B / 310B) through the CANN ACL Runtime and aclnn operator
//! library.
//!
//! # Architecture
//!
//! - [`ffi::acl`] — raw FFI bindings to ACL Runtime functions.
//! - [`context`] — `AscendContext`: owns the device, context, and stream.
//! - [`buffer`] — `AscendBuffer`: device memory allocation.
//! - [`stream`] — stream/event wrappers for synchronization and capture.
//! - [`transfers`] — H2D/D2H/D2D `aclrtMemcpy` wrappers.
//! - [`l3`] — L3 operator layer (ops / RI-graph / execution session), merged from the former separate L3 crate.
//! - [`backend`] — `AscendBackend`: implements `Backend` + `SamplingBackend`.
//!
//! # Status
//!
//! Phase 1 skeleton. Every operator reports which operator is missing, so
//! running a model against it names the next thing to implement rather than
//! failing with a generic message. The aclnn operator implementations arrive
//! in Phase 2; the model path that reaches them is already wired, so that
//! work is purely additive.
//!
//! # CANN environment
//!
//! The build script expects `ASCEND_HOME_PATH` to point at the CANN install
//! root (e.g. `/usr/local/Ascend/cann-9.1.0/x86_64-linux`). If unset, it
//! falls back to that standard path. The runtime requires the CANN
//! environment to be sourced (for `LD_LIBRARY_PATH`).

pub mod acl_tensor;
pub mod backend;
pub mod buffer;
pub mod context;
pub mod ffi;
pub mod kv_cache;
pub mod l3;
pub mod sampling;
pub mod stream;
pub mod transfers;

pub use backend::AscendBackend;
pub use buffer::AscendBuffer;
pub use context::AscendContext;
pub use acl_tensor::{
    WorkspacePool, SessionGuard, enter_session, is_in_session, current_workspace_pool,
    reset_diagnostics, print_diagnostics, diag_mem_enabled, note_output_overflow,
    output_overflow_stats, get_mem_info, set_capture_active, capture_active,
};
pub use transfers::{reset_h2d_diagnostics, print_h2d_diagnostics};

//! X-VLA 0.9B: DaViT vision tower + Florence-2 BART encoder + soft-prompted
//! action transformer.
//!
//! Port target: 2toINF/X-VLA @ 6bc2513, checkpoint `X-VLA-Libero` (ee6d action
//! space, 30-step horizon, 10 linear flow steps). See
//! `devlocal/xvla-port/reports/` on the engine host for the semantics inventory,
//! kernel plan and validation gates.
//!
//! The model modules are wired into the policy runner in a later step, so they
//! are allowed to stay unused until then.

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(crate) mod backend;
pub mod config;
pub mod math;

pub use config::XvlaConfig;

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(crate) mod load;

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(crate) mod model;

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(crate) mod runtime;

#[cfg(feature = "cuda")]
pub(crate) use runtime::register_builtin;

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(crate) mod weights;

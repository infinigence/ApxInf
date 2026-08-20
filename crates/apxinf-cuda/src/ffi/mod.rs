//! Private raw foreign-function bindings used by `apxinf-cuda`.
//!
//! Provider-specific declarations live in child modules. The flat re-exports
//! intentionally preserve the existing internal `crate::ffi::<name>` API.

#![allow(non_camel_case_types, dead_code, unused_imports)]

mod cublas;
mod cublaslt;
mod cuda;
mod custom;
mod cutlass;
mod driver;
mod fa2;

pub(crate) use cublas::*;
pub(crate) use cublaslt::*;
pub(crate) use cuda::*;
pub(crate) use custom::*;
pub(crate) use cutlass::*;
pub(crate) use driver::*;
pub(crate) use fa2::*;

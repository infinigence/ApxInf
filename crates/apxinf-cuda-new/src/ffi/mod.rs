//! Private raw foreign-function bindings used by `apxinf-cuda`.
//!
//! Provider-specific declarations live in child modules. The flat re-exports
//! intentionally preserve the existing internal `crate::ffi::<name>` API.

#![allow(non_camel_case_types, dead_code, unused_imports)]

pub(crate) mod raw;

pub(crate) use raw::*;

pub(crate) mod abi;

//! Private foreign-function bindings used by cuda-new.
//!
//! Typed operator ABIs live under `abi`; only CUDA runtime and sampling retain
//! small raw bindings.

#![allow(non_camel_case_types, dead_code, unused_imports)]

pub(crate) mod raw;

pub(crate) use raw::*;

pub(crate) mod abi;

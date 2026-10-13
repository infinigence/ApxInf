//! Test organization and maintenance policy for L3 operators.
//!
//! - `precision`: every new L3 operator must add a Torch-golden candidate test.
//! - `l3_behavior`: every new L3 operator must add a public semantic-contract
//!   test; add Graph replay coverage when it has an independent execution path.
//! - `family_contracts`: the same obligation for the Spec-dispatched families
//!   (`rope/`, `pointwise/`, `gather/`, `quantization/`), which had almost no
//!   coverage.
//! - `norm_fields`: the norm family's field model asserted as a table, because
//!   a wrong field tuple routes an operator to a different kernel and nothing
//!   else in the crate notices.
//! - `framework`: shared selection, cache, safety, and lifetime regressions;
//!   adding an operator normally does not require changing it.

use super::*;

mod family_contracts;
#[path = "backend_framework.rs"]
mod framework;
mod l3_behavior;
mod norm_fields;
#[path = "precision/nvfp4_precision.rs"]
mod nvfp4_precision;
mod operator_doc;
#[path = "precision/precision.rs"]
mod precision;

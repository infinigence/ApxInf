//! Test organization and maintenance policy for L3 operators.
//!
//! - `precision`: every new L3 operator must add a Torch-golden candidate test.
//! - `l3_behavior`: every new L3 operator must add a public semantic-contract
//!   test; add Graph replay coverage when it has an independent execution path.
//! - `framework`: shared selection, cache, safety, and lifetime regressions;
//!   adding an operator normally does not require changing it.

use super::*;

#[path = "backend_framework.rs"]
mod framework;
mod l3_behavior;
#[path = "precision/nvfp4_precision.rs"]
mod nvfp4_precision;
mod operator_doc;
mod pointwise_behavior;
#[path = "precision/precision.rs"]
mod precision;

mod transfer_ordering;

mod bicubic_behavior;
mod complex_rotary_behavior;
mod layout_behavior;

mod layer_norm_behavior;

mod convolution_behavior;

mod efficient_attention_behavior;

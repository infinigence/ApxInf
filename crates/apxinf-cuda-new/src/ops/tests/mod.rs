//! Test organization and maintenance policy for L3 operators.
//!
//! - `precision`: every new L3 operator must add a Torch-golden candidate test.
//! - `l3_behavior`: every new L3 operator must add a public semantic-contract
//!   test; add Graph replay coverage when it has an independent execution path.
//! - `framework`: shared selection, cache, safety, and lifetime regressions;
//!   adding an operator normally does not require changing it.

use super::*;

mod framework;
mod l3_behavior;
#[path = "precision/precision.rs"]
mod precision;

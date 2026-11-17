//! BF16 GR00T execution owned by the MUSA model family.

pub(super) mod calibration;
pub(super) mod executor;
pub(super) mod linear;
pub(super) mod precision;
pub(super) mod fp8;

// Keep the implementation modules shallow while the family remains fully
// self-contained. These aliases are private to `musa_gr00t::model`.
use super::{action_weights, backbone, backend, geometry, math, weights, Gr00tConfig};
use linear as device_weights;
use weights::bf16 as bf16_weights;

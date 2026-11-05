pub(crate) mod core;
mod fixed;
mod inplace;

pub use core::{add, mul, scale};
pub use fixed::*;
pub use inplace::add_into;

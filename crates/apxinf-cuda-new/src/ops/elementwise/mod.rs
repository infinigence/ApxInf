pub(crate) mod core;
mod fixed;
mod inplace;
mod rows;

pub use core::{add, mul, scale};
pub use fixed::*;
pub use inplace::add_into;
pub use rows::{argmax_rows, fill_zero, mean_rows};

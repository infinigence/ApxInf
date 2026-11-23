//! Element-wise contracts.

use apxinf_core::{Error, Result, Tensor};

use super::require_unit_output_scale;
use crate::kernels::activation;
use crate::kernels::elementwise;
use crate::kernels::not_ported;
use crate::MusaContext;

/// Which element-wise operation the arguments describe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointwiseSemantic {
    /// `input [rows, 2 * cols]` -> `output [rows, cols]`.
    Geglu,
    BiasActivation,
    EulerUpdate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointwiseActivation {
    None,
    Gelu,
    Silu,
    GeluExact,
}

/// Element-wise operation over a contiguous `[rows, cols]` activation, where
/// `cols` is the *output* width.
pub struct PointwiseArgs<'a> {
    pub semantic: PointwiseSemantic,
    pub input: &'a Tensor,
    /// Euler-update velocity; unused by the other semantics.
    pub secondary: Option<&'a Tensor>,
    pub bias: Option<&'a Tensor>,
    pub out: &'a mut Tensor,
    pub activation: PointwiseActivation,
    pub dt: f32,
    pub output_scale: f32,
}

impl<'a> PointwiseArgs<'a> {
    pub fn new(semantic: PointwiseSemantic, input: &'a Tensor, out: &'a mut Tensor) -> Self {
        Self {
            semantic,
            input,
            secondary: None,
            bias: None,
            out,
            activation: PointwiseActivation::None,
            dt: 0.0,
            output_scale: 1.0,
        }
    }
}

pub fn pointwise(ctx: &MusaContext, args: PointwiseArgs<'_>) -> Result<()> {
    require_unit_output_scale("pointwise", args.output_scale)?;
    match args.semantic {
        PointwiseSemantic::Geglu => {
            if args.bias.is_some() || args.secondary.is_some() {
                return Err(Error::Other(
                    "MUSA pointwise geglu takes neither a bias nor a velocity".into(),
                ));
            }
            activation::geglu_bf16_into(ctx, args.input, args.out)
        }
        PointwiseSemantic::BiasActivation => {
            if args.secondary.is_some() {
                return Err(Error::Other(
                    "MUSA pointwise bias activation takes no velocity".into(),
                ));
            }
            match args.activation {
                PointwiseActivation::None => {
                    elementwise::bias_bf16_into(ctx, args.input, args.bias, args.out)
                }
                PointwiseActivation::Gelu => {
                    activation::bias_gelu_bf16_into(ctx, args.input, args.bias, args.out)
                }
                PointwiseActivation::Silu => {
                    activation::bias_silu_bf16_into(ctx, args.input, args.bias, args.out)
                }
                PointwiseActivation::GeluExact => Err(not_ported("pointwise gelu_exact")),
            }
        }
        PointwiseSemantic::EulerUpdate => {
            let velocity = args.secondary.ok_or_else(|| {
                Error::Other("MUSA pointwise Euler update requires a velocity tensor".into())
            })?;
            if args.bias.is_some() {
                return Err(Error::Other(
                    "MUSA pointwise Euler update takes no bias".into(),
                ));
            }
            elementwise::euler_update_bf16_into(ctx, args.input, velocity, args.out, args.dt)
        }
    }
}

use apxinf_core::{DType, Error, Result, Tensor};

use crate::context::CudaContext;

use super::contracts::{QuantizationArgs, QuantizationSemantic};
use super::quantization::quantization;

pub fn cast(ctx: &CudaContext, input: &Tensor, output: &Tensor) -> Result<()> {
    let semantic = match (input.dtype(), output.dtype()) {
        (DType::F16, DType::BF16) => QuantizationSemantic::CastF16ToBf16,
        (DType::F32, DType::BF16) => QuantizationSemantic::CastF32ToBf16,
        (DType::BF16, DType::F32) => QuantizationSemantic::CastBf16ToF32,
        _ => Err(Error::Other(
            "cast supports F16->BF16, F32->BF16, and BF16->F32".into(),
        ))?,
    };
    let mut output = output.clone();
    quantization(ctx, QuantizationArgs::new(semantic, input, &mut output))
}

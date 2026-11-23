//! Model-neutral sampling contracts on MUSA.
//!
//! Only the standard-normal generator exists here. The draws are the CUDA
//! backend's draws — `apxinf_core::standard_normal_f32` is the host half of the
//! same counter-based Philox generator the CUDA kernel implements — because the
//! latent is an *input* to the model and the one thing a replay fixture pins
//! exactly. A device-side generator would save a host round trip of
//! `action_horizon * action_dim` floats per request (1.28 KB for PI0.5), not per
//! layer or per solver step, and would risk changing the numbers a seeded
//! caller reproduces.

use apxinf_core::{DType, Device, Error, NormalGenerator, Result, RngKey, Tensor};

use crate::buffer::MusaBuffer;
use crate::context::MusaContext;

/// Fill `output` with standard-normal draws derived from `rng`.
///
/// The draws are f32 and the buffer is not necessarily f32: PI0.5's latent is
/// the model's input dtype, so a BF16 buffer takes the draws rounded once, the
/// same conversion `apxinf_core`'s CPU generator performs. Copying the f32 bytes
/// straight in would overrun the buffer and, where it fit, change the numbers.
struct MusaNormalGenerator {
    output: Tensor,
}

impl MusaNormalGenerator {
    fn new(output: Tensor, device_id: usize) -> Result<Self> {
        if output.device() != Device::Musa(device_id) {
            return Err(Error::DeviceMismatch {
                expected: Device::Musa(device_id),
                got: output.device(),
            });
        }
        if !matches!(output.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::Other(format!(
                "standard-normal generation does not support {}",
                output.dtype()
            )));
        }
        Ok(Self { output })
    }
}

impl NormalGenerator for MusaNormalGenerator {
    fn output(&self) -> &Tensor {
        &self.output
    }

    fn generate(&mut self, rng: RngKey) -> Result<&Tensor> {
        let values = apxinf_core::standard_normal_f32(self.output.shape().numel(), rng);
        let bytes: Vec<u8> = match self.output.dtype() {
            DType::F32 => bytemuck::cast_slice(&values).to_vec(),
            DType::F16 => values
                .iter()
                .flat_map(|value| half::f16::from_f32(*value).to_bits().to_ne_bytes())
                .collect(),
            DType::BF16 => values
                .iter()
                .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_ne_bytes())
                .collect(),
            dtype => {
                return Err(Error::Other(format!(
                    "standard-normal generation does not support {dtype}"
                )))
            }
        };
        MusaBuffer::from_tensor(&self.output)
            .map_err(crate::error::musa_error)?
            .copy_from_host(&bytes)
            .map_err(crate::error::musa_error)?;
        Ok(&self.output)
    }
}

/// Bind a reusable standard-normal generator to `output`.
pub fn create_normal_generator(
    ctx: &MusaContext,
    output: Tensor,
) -> Result<Box<dyn NormalGenerator>> {
    Ok(Box::new(MusaNormalGenerator::new(output, ctx.device_id())?))
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    use super::*;
    use apxinf_core::Shape;

    fn read_bits(ctx: &MusaContext, tensor: &Tensor) -> Vec<u8> {
        ctx.synchronize().unwrap();
        let host = crate::transfers::to_cpu(tensor).unwrap();
        let bytes: Vec<u8> = match host.dtype() {
            DType::F32 => bytemuck::cast_slice(host.as_f32().unwrap()).to_vec(),
            DType::BF16 => host
                .as_bf16()
                .unwrap()
                .iter()
                .flat_map(|value| value.to_bits().to_ne_bytes())
                .collect(),
            dtype => panic!("unexpected host dtype {dtype}"),
        };
        bytes
    }

    #[test]
    fn the_draws_are_the_cpu_backends_draws_rounded_once() {
        let ctx = MusaContext::new(0).unwrap();
        let rng = RngKey::new(7, 1, 2);
        let (horizon, dim) = (10usize, 32usize);
        let draws = apxinf_core::standard_normal_f32(horizon * dim, rng);

        let mut generator =
            create_normal_generator(&ctx, ctx.allocate_output(Shape::new(vec![horizon, dim]), DType::BF16).unwrap())
                .unwrap();
        let produced = generator.output().clone();
        generator.generate(rng).unwrap();
        let expected: Vec<u8> = draws
            .iter()
            .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_ne_bytes())
            .collect();
        assert_eq!(read_bits(&ctx, &produced), expected);

        // f32 takes the draws unchanged, which a dtype-blind implementation
        // would also satisfy; the two cases together separate them.
        let mut generator = create_normal_generator(
            &ctx,
            ctx.allocate_output(Shape::new(vec![horizon, dim]), DType::F32)
                .unwrap(),
        )
        .unwrap();
        let produced = generator.output().clone();
        generator.generate(rng).unwrap();
        assert_eq!(
            read_bits(&ctx, &produced),
            bytemuck::cast_slice::<f32, u8>(&draws)
        );
    }

    #[test]
    fn a_tensor_from_another_device_is_refused() {
        let ctx = MusaContext::new(0).unwrap();
        let host = Tensor::zeros(Shape::new(vec![4]), DType::F32);
        assert!(create_normal_generator(&ctx, host).is_err());
    }
}

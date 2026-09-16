//! bf16 device weights: model aggregates and linear storage.
mod linear {
    //! Device-ready BF16 linear weights for π0.5.

    use apxinf_core::{Backend, DType, Error, Result, Tensor};

    use crate::pi05::weights::packing::concat_host_2d;
    use crate::pi05::LinearWeights;

    #[derive(Debug)]
    pub struct Bf16LinearWeights {
        /// Physical row-major `[input, output]` matrix.
        pub weight: Tensor,
        /// Exact `[gate256,up256]` physical layout. This layout may only be
        /// consumed by a validated BF16 dual-GeGLU backend.
        pub bf16_dual_geglu_interleaved: bool,
        /// Optional auto-mode `[gate256,up256]` matrix. The primary tensor remains
        /// plain for every non-dual route and for the explicit control mode.
        pub bf16_dual_geglu_auto_interleaved: Option<Tensor>,
        /// Optional `[gate,up]` column-pair layout consumed by the SM89 CUTLASS
        /// visitor GeGLU epilogue.
        pub bf16_sm89_geglu_interleaved: Option<Tensor>,
        pub bias: Option<Tensor>,
    }

    impl Bf16LinearWeights {
        pub fn from_host(linear: &LinearWeights, backend: &dyn Backend) -> Result<Self> {
            Self::from_host_parts(&[linear], backend)
        }

        /// Pack projections along the output dimension so QKV and gate/up each
        /// remain one tensor-core GEMM, matching the static FP8 computation schedule.
        pub fn from_host_parts(linears: &[&LinearWeights], backend: &dyn Backend) -> Result<Self> {
            Self::from_host_parts_with_dual_layout(linears, backend, true)
        }

        pub(crate) fn from_host_parts_with_dual_layout(
            linears: &[&LinearWeights],
            backend: &dyn Backend,
            allow_dual_layout: bool,
        ) -> Result<Self> {
            if linears.is_empty() {
                return Err(Error::Other(
                    "cannot pack an empty BF16 linear group".into(),
                ));
            }
            let bf16_dual_geglu_mode = bf16_dual_geglu_mode()?;
            let dual_geglu_exact = allow_dual_layout
                && linears.len() == 2
                && linears
                    .iter()
                    .all(|linear| linear.weight.shape().dims() == [2048, 16384]);
            let bf16_dual_geglu_interleaved =
                dual_geglu_exact && bf16_dual_geglu_mode == Bf16DualGeGluMode::On;
            let plain_weight = concat_host_2d(
                &linears
                    .iter()
                    .map(|linear| &linear.weight)
                    .collect::<Vec<_>>(),
            )?;
            let interleaved_weight =
                if dual_geglu_exact && bf16_dual_geglu_mode != Bf16DualGeGluMode::Off {
                    Some(interleave_gate_up_host(
                        &linears[0].weight,
                        &linears[1].weight,
                        256,
                    )?)
                } else {
                    None
                };
            let weight_host = if bf16_dual_geglu_interleaved {
                interleaved_weight.as_ref().unwrap()
            } else {
                &plain_weight
            };
            let weight = bf16_to_device(weight_host, backend)?;
            let bf16_dual_geglu_auto_interleaved =
                if dual_geglu_exact && bf16_dual_geglu_mode == Bf16DualGeGluMode::Auto {
                    Some(bf16_to_device(
                        interleaved_weight.as_ref().unwrap(),
                        backend,
                    )?)
                } else {
                    None
                };
            let bf16_sm89_geglu_interleaved = if allow_dual_layout
                && linears.len() == 2
                && bf16_dual_geglu_mode != Bf16DualGeGluMode::Off
            {
                Some(bf16_to_device(
                    &interleave_gate_up_host(&linears[0].weight, &linears[1].weight, 1)?,
                    backend,
                )?)
            } else {
                None
            };
            let bias = if linears.iter().all(|linear| linear.bias.is_none()) {
                None
            } else if linears.iter().all(|linear| linear.bias.is_some()) {
                Some(concat_biases_bf16(
                    &linears
                        .iter()
                        .map(|linear| linear.bias.as_ref().unwrap())
                        .collect::<Vec<_>>(),
                    backend,
                )?)
            } else {
                return Err(Error::Other(
                    "cannot pack BF16 projections with mixed bias presence".into(),
                ));
            };
            Ok(Self {
                weight,
                bf16_dual_geglu_interleaved,
                bf16_dual_geglu_auto_interleaved,
                bf16_sm89_geglu_interleaved,
                bias,
            })
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Bf16DualGeGluMode {
        Auto,
        Off,
        On,
    }

    fn parse_bf16_dual_geglu_mode(value: Option<&str>) -> Result<Bf16DualGeGluMode> {
        match value {
            None | Some("auto") => Ok(Bf16DualGeGluMode::Auto),
            Some("0" | "off") => Ok(Bf16DualGeGluMode::Off),
            Some("1" | "on") => Ok(Bf16DualGeGluMode::On),
            Some(value) => Err(Error::Other(format!(
                "APXINF_PI05_BF16_DUAL_GEGLU must be auto, 0/off, or 1/on; got {value}"
            ))),
        }
    }

    fn bf16_dual_geglu_mode() -> Result<Bf16DualGeGluMode> {
        const NAME: &str = "APXINF_PI05_BF16_DUAL_GEGLU";
        match std::env::var(NAME) {
            Err(std::env::VarError::NotPresent) => parse_bf16_dual_geglu_mode(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(Error::Other(format!("{NAME} must be valid Unicode")))
            }
            Ok(value) => parse_bf16_dual_geglu_mode(Some(&value)),
        }
    }

    fn interleave_gate_up_host(gate: &Tensor, up: &Tensor, tile: usize) -> Result<Tensor> {
        let gate_shape = gate.shape().dims();
        let up_shape = up.shape().dims();
        if gate_shape.len() != 2 || gate_shape != up_shape || tile == 0 || gate_shape[1] % tile != 0
        {
            return Err(Error::Other(format!(
            "BF16 dual GeGLU requires equal 2D Gate/Up widths divisible by {tile}, got {gate_shape:?} and {up_shape:?}"
        )));
        }
        let rows = gate_shape[0];
        let width = gate_shape[1];
        let gate_values = gate.to_f32_vec()?;
        let up_values = up.to_f32_vec()?;
        let mut output = vec![0.0f32; rows * width * 2];
        for row in 0..rows {
            for tile_index in 0..width / tile {
                let src = row * width + tile_index * tile;
                let dst = row * width * 2 + tile_index * tile * 2;
                output[dst..dst + tile].copy_from_slice(&gate_values[src..src + tile]);
                output[dst + tile..dst + 2 * tile].copy_from_slice(&up_values[src..src + tile]);
            }
        }
        Tensor::from_f32(vec![rows, width * 2], &output)
    }

    pub fn bf16_to_device(tensor: &Tensor, backend: &dyn Backend) -> Result<Tensor> {
        if tensor.dtype() == DType::F8E4M3 {
            return Err(Error::Other(
                "cannot convert scale-less E4M3 data to BF16".into(),
            ));
        }
        let values = tensor
            .to_f32_vec()?
            .into_iter()
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>();
        backend.to_device(&Tensor::from_bf16(tensor.shape().dims().to_vec(), &values)?)
    }

    fn concat_biases_bf16(tensors: &[&Tensor], backend: &dyn Backend) -> Result<Tensor> {
        let mut values = Vec::new();
        for tensor in tensors {
            if tensor.shape().dims().len() != 1 || tensor.dtype() == DType::F8E4M3 {
                return Err(Error::Other(
                    "packed BF16 biases must be non-FP8 vectors".into(),
                ));
            }
            values.extend(tensor.to_f32_vec()?.into_iter().map(half::bf16::from_f32));
        }
        backend.to_device(&Tensor::from_bf16(vec![values.len()], &values)?)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use apxinf_core::CpuBackend;

        #[test]
        fn packs_qkv_as_native_bf16() {
            let linear = |weight: &[f32], shape: [usize; 2], bias: &[f32]| LinearWeights {
                weight: Tensor::from_f32(shape.to_vec(), weight).unwrap(),
                bias: Some(Tensor::from_f32(vec![bias.len()], bias).unwrap()),
            };
            let q = linear(&[1., 2., 3., 4.], [2, 2], &[1., 2.]);
            let k = linear(&[5., 6.], [2, 1], &[3.]);
            let v = linear(&[7., 8.], [2, 1], &[4.]);
            let packed = Bf16LinearWeights::from_host_parts(&[&q, &k, &v], &CpuBackend).unwrap();
            assert_eq!(packed.weight.shape().dims(), &[2, 4]);
            assert_eq!(packed.weight.dtype(), DType::BF16);
            assert!(!packed.bf16_dual_geglu_interleaved);
            assert!(packed.bf16_dual_geglu_auto_interleaved.is_none());
            assert_eq!(
                packed.bias.unwrap().to_f32_vec().unwrap(),
                vec![1., 2., 3., 4.]
            );
        }

        #[test]
        fn bf16_dual_geglu_mode_parser_is_strict_and_defaults_auto() {
            assert_eq!(
                parse_bf16_dual_geglu_mode(None).unwrap(),
                Bf16DualGeGluMode::Auto
            );
            assert_eq!(
                parse_bf16_dual_geglu_mode(Some("auto")).unwrap(),
                Bf16DualGeGluMode::Auto
            );
            assert_eq!(
                parse_bf16_dual_geglu_mode(Some("0")).unwrap(),
                Bf16DualGeGluMode::Off
            );
            assert_eq!(
                parse_bf16_dual_geglu_mode(Some("off")).unwrap(),
                Bf16DualGeGluMode::Off
            );
            assert_eq!(
                parse_bf16_dual_geglu_mode(Some("1")).unwrap(),
                Bf16DualGeGluMode::On
            );
            assert_eq!(
                parse_bf16_dual_geglu_mode(Some("on")).unwrap(),
                Bf16DualGeGluMode::On
            );
            assert!(parse_bf16_dual_geglu_mode(Some("invalid")).is_err());
        }

        #[test]
        fn bf16_dual_geglu_auto_memory_budget_is_exact() {
            const BYTES_PER_BF16: usize = 2;
            const LAYERS: usize = 18;
            const BYTES_PER_LAYER: usize = 2048 * 32768 * BYTES_PER_BF16;
            assert_eq!(BYTES_PER_LAYER, 128 * 1024 * 1024);
            assert_eq!(LAYERS * BYTES_PER_LAYER, 2_415_919_104);
        }

        #[test]
        fn bf16_dual_geglu_eighteen_layer_interleave_preserves_bytes() {
            const ROWS: usize = 2;
            const WIDTH: usize = 1024;
            const TILE: usize = 256;
            for layer in 0..18usize {
                let gate = (0..ROWS * WIDTH)
                    .map(|index| ((index * 17 + layer * 31) % 1009) as f32 / 127.0 - 4.0)
                    .collect::<Vec<_>>();
                let up = (0..ROWS * WIDTH)
                    .map(|index| ((index * 29 + layer * 43) % 1013) as f32 / 131.0 - 3.5)
                    .collect::<Vec<_>>();
                let gate = Tensor::from_f32(vec![ROWS, WIDTH], &gate).unwrap();
                let up = Tensor::from_f32(vec![ROWS, WIDTH], &up).unwrap();
                let plain = concat_host_2d(&[&gate, &up]).unwrap();
                let interleaved = interleave_gate_up_host(&gate, &up, TILE).unwrap();
                let plain_bits = plain
                    .to_f32_vec()
                    .unwrap()
                    .into_iter()
                    .map(half::bf16::from_f32)
                    .map(half::bf16::to_bits)
                    .collect::<Vec<_>>();
                let interleaved_bits = interleaved
                    .to_f32_vec()
                    .unwrap()
                    .into_iter()
                    .map(half::bf16::from_f32)
                    .map(half::bf16::to_bits)
                    .collect::<Vec<_>>();
                for row in 0..ROWS {
                    for tile_index in 0..WIDTH / TILE {
                        let plain_gate = row * 2 * WIDTH + tile_index * TILE;
                        let plain_up = row * 2 * WIDTH + WIDTH + tile_index * TILE;
                        let packed = row * 2 * WIDTH + tile_index * 2 * TILE;
                        assert_eq!(
                            &interleaved_bits[packed..packed + TILE],
                            &plain_bits[plain_gate..plain_gate + TILE]
                        );
                        assert_eq!(
                            &interleaved_bits[packed + TILE..packed + 2 * TILE],
                            &plain_bits[plain_up..plain_up + TILE]
                        );
                    }
                }
            }
        }
    }
}
pub use linear::*;
// Fully materialized native-BF16 π0.5 weights.

use apxinf_core::{Backend, Result, Tensor};

use crate::pi05::{
    ActionLayerWeights, AdaRmsNormWeights, LanguageLayerWeights, LayerNormWeights, Pi05Weights,
    VisionBlockWeights,
};

#[derive(Debug)]
pub struct Bf16DeviceLayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct Bf16DeviceVisionBlock {
    pub norm1: Bf16DeviceLayerNorm,
    pub qkv: Bf16LinearWeights,
    pub output: Bf16LinearWeights,
    pub norm2: Bf16DeviceLayerNorm,
    pub fc1: Bf16LinearWeights,
    pub fc2: Bf16LinearWeights,
}

#[derive(Debug)]
pub struct Bf16DeviceLanguageLayer {
    pub input_norm_scale: Tensor,
    pub qkv: Bf16LinearWeights,
    pub output: Bf16LinearWeights,
    pub post_attention_norm_scale: Tensor,
    pub gate_up: Bf16LinearWeights,
    pub down: Bf16LinearWeights,
}

#[derive(Debug)]
pub struct Bf16DeviceActionLayer {
    pub input_style: Bf16LinearWeights,
    pub qkv: Bf16LinearWeights,
    pub output: Bf16LinearWeights,
    pub post_attention_style: Bf16LinearWeights,
    pub gate_up: Bf16LinearWeights,
    pub down: Bf16LinearWeights,
}

#[derive(Debug)]
pub struct Bf16Weights {
    pub patch_embedding: Bf16LinearWeights,
    pub position_embedding: Tensor,
    pub vision_layers: Vec<Bf16DeviceVisionBlock>,
    pub vision_post_norm: Bf16DeviceLayerNorm,
    pub multimodal_projector: Bf16LinearWeights,
    pub token_embedding: Tensor,
    pub language_layers: Vec<Bf16DeviceLanguageLayer>,
    pub language_final_norm_scale: Tensor,
    pub action_layers: Vec<Bf16DeviceActionLayer>,
    pub action_final_style: Bf16LinearWeights,
    pub action_in: Bf16LinearWeights,
    pub action_out: Bf16LinearWeights,
    pub time_mlp_in: Bf16LinearWeights,
    pub time_mlp_out: Bf16LinearWeights,
}

impl Bf16Weights {
    pub fn from_host(
        weights: &Pi05Weights,
        backend: &dyn Backend,
        language_dual_layout: bool,
    ) -> Result<Self> {
        Ok(Self {
            patch_embedding: Bf16LinearWeights::from_host(
                &weights.vision.patch_embedding,
                backend,
            )?,
            position_embedding: bf16_to_device(&weights.vision.position_embedding, backend)?,
            vision_layers: weights
                .vision
                .blocks
                .iter()
                .map(|layer| Bf16DeviceVisionBlock::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            vision_post_norm: Bf16DeviceLayerNorm::from_host(
                &weights.vision.post_layer_norm,
                backend,
            )?,
            multimodal_projector: Bf16LinearWeights::from_host(
                &weights.vision.multimodal_projector,
                backend,
            )?,
            token_embedding: bf16_to_device(&weights.vision.token_embedding, backend)?,
            language_layers: weights
                .language_layers
                .iter()
                .map(|layer| {
                    Bf16DeviceLanguageLayer::from_host(layer, backend, language_dual_layout)
                })
                .collect::<Result<Vec<_>>>()?,
            language_final_norm_scale: bf16_to_device(&weights.language_final_norm_scale, backend)?,
            action_layers: weights
                .action_layers
                .iter()
                .map(|layer| Bf16DeviceActionLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            action_final_style: style_to_device(&weights.action_final_norm, backend)?,
            action_in: Bf16LinearWeights::from_host(&weights.action_in, backend)?,
            action_out: Bf16LinearWeights::from_host(&weights.action_out, backend)?,
            time_mlp_in: Bf16LinearWeights::from_host(&weights.time_mlp_in, backend)?,
            time_mlp_out: Bf16LinearWeights::from_host(&weights.time_mlp_out, backend)?,
        })
    }
}

impl Bf16DeviceLayerNorm {
    fn from_host(weights: &LayerNormWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            weight: bf16_to_device(&weights.weight, backend)?,
            bias: bf16_to_device(&weights.bias, backend)?,
        })
    }
}

impl Bf16DeviceVisionBlock {
    fn from_host(weights: &VisionBlockWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            norm1: Bf16DeviceLayerNorm::from_host(&weights.norm1, backend)?,
            qkv: Bf16LinearWeights::from_host_parts(
                &[&weights.q, &weights.k, &weights.v],
                backend,
            )?,
            output: Bf16LinearWeights::from_host(&weights.output, backend)?,
            norm2: Bf16DeviceLayerNorm::from_host(&weights.norm2, backend)?,
            fc1: Bf16LinearWeights::from_host(&weights.fc1, backend)?,
            fc2: Bf16LinearWeights::from_host(&weights.fc2, backend)?,
        })
    }
}

impl Bf16DeviceLanguageLayer {
    fn from_host(
        weights: &LanguageLayerWeights,
        backend: &dyn Backend,
        allow_dual_layout: bool,
    ) -> Result<Self> {
        Ok(Self {
            input_norm_scale: bf16_to_device(&weights.input_norm_scale, backend)?,
            qkv: Bf16LinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Bf16LinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_norm_scale: bf16_to_device(&weights.post_attention_norm_scale, backend)?,
            gate_up: Bf16LinearWeights::from_host_parts_with_dual_layout(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
                allow_dual_layout,
            )?,
            down: Bf16LinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

impl Bf16DeviceActionLayer {
    fn from_host(weights: &ActionLayerWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            input_style: style_to_device(&weights.input_norm, backend)?,
            qkv: Bf16LinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Bf16LinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_style: style_to_device(&weights.post_attention_norm, backend)?,
            gate_up: Bf16LinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Bf16LinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

fn style_to_device(
    weights: &AdaRmsNormWeights,
    backend: &dyn Backend,
) -> Result<Bf16LinearWeights> {
    Bf16LinearWeights::from_host(&weights.style, backend)
}

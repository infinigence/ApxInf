use super::super::action_weights::Gr00tLinearWeights;
use super::super::backend::{kernels, RuntimeBackend};
use super::super::model::linear::DeviceLinearWeights;
use super::bf16::Gr00tBf16LinearWeights;
use super::fp8_policy::{for_consumer, Policy};
use super::fp8_switches::enabled;
use apxinf_core::{DType, Error, Result, Tensor};

fn gemm_enabled(name: &str) -> bool {
    enabled("GEMM")
        && !std::env::var("APXINF_GR00T_FP8_EXCLUDE")
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|prefix| !prefix.is_empty())
            .any(|prefix| name.starts_with(prefix))
}

fn gelu_enabled(name: &str) -> bool {
    enabled("ACTIVATION")
        && enabled("BIAS_GELU")
        && (!name.starts_with("backbone.vision.") || enabled("VISION_GELU"))
}

fn norm_enabled(name: &str) -> bool {
    let group = if name.starts_with("backbone.text.") {
        "RMS_NORM"
    } else if name.starts_with("backbone.vision.") {
        "VISION_NORM"
    } else if name.starts_with("action_head.dit_blocks.") || name == "action_head.output_projection"
    {
        "DIT_NORM"
    } else {
        "VL_NORM"
    };
    enabled("NORM") && enabled(group)
}

#[derive(Debug)]
struct Prepared {
    policy: Policy,
    weight: Tensor,
    activation_scale: f32,
    product_scale: f32,
}

#[derive(Debug)]
pub(crate) struct Gr00tFp8LinearWeights {
    bf16: Gr00tBf16LinearWeights,
    prepared: Option<Prepared>,
    fuse_norm: bool,
    fuse_gelu: bool,
    fuse_silu: bool,
    fusion_only: bool,
}

impl Gr00tFp8LinearWeights {
    pub(crate) fn candidate(weight: &Tensor, name: &str) -> bool {
        let dims = weight.shape().dims();
        gemm_enabled(name) && dims.len() == 2 && for_consumer(name, dims[1], dims[0]).is_some()
    }

    fn prepare(
        weight: &Tensor,
        name: &str,
        activation_scale: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<Prepared>> {
        if !gemm_enabled(name) {
            return Ok(None);
        }
        let dims = weight.shape().dims();
        if dims.len() != 2 {
            return Err(Error::Other("FP8 linear expects a matrix".into()));
        }
        let (inner, cols) = (dims[0], dims[1]);
        let Some(policy) = for_consumer(name, cols, inner) else {
            return Ok(None);
        };
        let values = weight.as_bf16()?;
        if values.iter().any(|value| !value.to_f32().is_finite()) {
            return Err(Error::Other("FP8 weight contains non-finite values".into()));
        }
        let maximum = values
            .iter()
            .map(|value| value.to_f32().abs())
            .fold(0.0f32, f32::max);
        let weight_scale = if maximum == 0.0 {
            1.0
        } else {
            (maximum / 448.0).max(1e-8)
        };
        let packed = match policy.layout() {
            kernels::gemm::Fp8Layout::Nn => weight.clone(),
            kernels::gemm::Fp8Layout::Nt => {
                let mut transposed = vec![half::bf16::ZERO; values.len()];
                for depth in 0..inner {
                    for column in 0..cols {
                        transposed[column * inner + depth] = values[depth * cols + column];
                    }
                }
                Tensor::from_bf16(vec![cols, inner], &transposed)?
            }
        };
        let device = backend.to_device(&packed)?;
        let weight =
            kernels::quantization::quantize_bf16_e4m3(backend.context(), &device, weight_scale)?;
        backend.synchronize()?;
        let product_scale = weight_scale * activation_scale;
        if !product_scale.is_finite() || product_scale <= 0.0 {
            return Err(Error::Other("FP8 combined scale is invalid".into()));
        }
        Ok(Some(Prepared {
            policy,
            weight,
            activation_scale,
            product_scale,
        }))
    }

    pub(crate) fn from_host(
        weights: Gr00tLinearWeights,
        name: &str,
        scale: f32,
        backend: &RuntimeBackend,
    ) -> Result<Self> {
        let prepared = Self::prepare(&weights.weight, name, scale, backend)?;
        Ok(Self {
            bf16: Gr00tBf16LinearWeights::from_host(weights, name, backend)?,
            prepared,
            fuse_norm: norm_enabled(name),
            fuse_gelu: gelu_enabled(name),
            fuse_silu: enabled("ACTIVATION") && enabled("SILU_MUL"),
            fusion_only: enabled("FUSION_ONLY"),
        })
    }

    pub(crate) fn from_device_matrix(
        weight: &Tensor,
        name: &str,
        scale: f32,
        backend: &RuntimeBackend,
    ) -> Result<Self> {
        let host = backend.to_cpu(weight)?;
        let prepared = Self::prepare(&host, name, scale, backend)?;
        let cols = weight.shape().dims()[1];
        let bias = backend.to_device(&Tensor::from_bf16(
            vec![cols],
            &vec![half::bf16::ZERO; cols],
        )?)?;
        let bf16 = Gr00tBf16LinearWeights::from_device(
            Gr00tLinearWeights {
                weight: weight.clone(),
                bias,
            },
            name.into(),
        );
        Ok(Self {
            bf16,
            prepared,
            fuse_norm: norm_enabled(name),
            fuse_gelu: gelu_enabled(name),
            fuse_silu: enabled("ACTIVATION") && enabled("SILU_MUL"),
            fusion_only: enabled("FUSION_ONLY"),
        })
    }
}

impl DeviceLinearWeights for Gr00tFp8LinearWeights {
    type ReusableInput = ();

    fn forward(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Tensor> {
        if input.dtype() == DType::F8E4M3 {
            return self.forward_quantized_tensor(input, backend);
        }
        if let Some(prepared) = &self.prepared {
            let dims = input.shape().dims();
            if dims.len() == 2 && prepared.policy.quantizes_input(dims[0], dims[1]) {
                let quantized = kernels::quantization::quantize_bf16_e4m3(
                    backend.context(),
                    input,
                    prepared.activation_scale,
                )?;
                return kernels::gemm::gemm_fp8(
                    backend.context(),
                    &quantized,
                    &prepared.weight,
                    prepared.policy.layout(),
                    prepared.product_scale,
                );
            }
        }
        self.bf16.forward(input, backend)
    }

    fn fp8_input_scale(&self, rows: usize, inner: usize) -> Option<f32> {
        self.prepared
            .as_ref()
            .filter(|prepared| {
                self.fuse_norm
                    && prepared.policy.accepts_fp8(rows, inner)
                    && (self.fusion_only || prepared.policy.quantizes_input(rows, inner))
            })
            .map(|prepared| prepared.activation_scale)
    }

    fn quantize_tensor_input(
        &self,
        input: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        let Some(prepared) = &self.prepared else {
            return Ok(None);
        };
        let [rows, inner] = input.shape().dims() else {
            return Ok(None);
        };
        if input.dtype() != DType::BF16 || !prepared.policy.quantizes_input(*rows, *inner) {
            return Ok(None);
        }
        kernels::quantization::quantize_bf16_e4m3(
            backend.context(),
            input,
            prepared.activation_scale,
        )
        .map(Some)
    }

    fn forward_quantized_tensor(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Tensor> {
        let dims = input.shape().dims();
        let prepared = self
            .prepared
            .as_ref()
            .filter(|prepared| {
                input.dtype() == DType::F8E4M3
                    && dims.len() == 2
                    && prepared.policy.accepts_fp8(dims[0], dims[1])
            })
            .ok_or_else(|| {
                Error::Other("FP8 input does not match the consumer's shape policy".into())
            })?;
        kernels::gemm::gemm_fp8(
            backend.context(),
            input,
            &prepared.weight,
            prepared.policy.layout(),
            prepared.product_scale,
        )
    }

    fn fused_silu_mul(
        &self,
        gate: &Tensor,
        up: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        if !self.fuse_silu {
            return Ok(None);
        }
        let Some(prepared) = &self.prepared else {
            return Ok(None);
        };
        let [rows, inner] = gate.shape().dims() else {
            return Ok(None);
        };
        if gate.dtype() != DType::BF16
            || up.dtype() != DType::BF16
            || gate.shape() != up.shape()
            || !prepared.policy.accepts_fp8(*rows, *inner)
        {
            return Ok(None);
        }
        let quantized = kernels::activation::silu_mul_quant_bf16_e4m3(
            backend.context(),
            gate,
            up,
            prepared.activation_scale,
        )?;
        Ok(Some(self.forward_quantized_tensor(&quantized, backend)?))
    }

    fn fused_bias_gelu(
        &self,
        input: &Tensor,
        bias: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        self.quantize_bias_gelu_tensor_input(input, bias, backend)?
            .map(|quantized| self.forward_quantized_tensor(&quantized, backend))
            .transpose()
    }

    fn quantize_bias_gelu_tensor_input(
        &self,
        input: &Tensor,
        bias: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        if !self.fuse_gelu {
            return Ok(None);
        }
        let Some(prepared) = &self.prepared else {
            return Ok(None);
        };
        let [rows, inner] = input.shape().dims() else {
            return Ok(None);
        };
        if input.dtype() != DType::BF16
            || bias.dtype() != DType::BF16
            || !prepared.policy.accepts_fp8(*rows, *inner)
        {
            return Ok(None);
        }
        let quantized = kernels::activation::bias_gelu_quant_bf16_e4m3(
            backend.context(),
            input,
            bias,
            prepared.activation_scale,
        )?;
        Ok(Some(quantized))
    }

    fn activation_scale(&self) -> Option<f32> {
        self.prepared
            .as_ref()
            .map(|prepared| prepared.activation_scale)
    }

    fn bias(&self) -> Option<&Tensor> {
        self.bf16.bias()
    }
    fn packed8_bias(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Option<Tensor>> {
        self.bf16.packed8_bias(input, backend)
    }
    fn packed8_bias_gelu(
        &self,
        input: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        self.bf16.packed8_bias_gelu(input, backend)
    }
    fn bias_residual_adaptive_layer_norm(
        &self,
        projection: &Tensor,
        residual: &Tensor,
        modulation: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Tensor)>> {
        self.bf16
            .bias_residual_adaptive_layer_norm(projection, residual, modulation, eps, backend)
    }
    fn bias_residual_layer_norm(
        &self,
        projection: &Tensor,
        residual: &Tensor,
        norm_weight: &Tensor,
        norm_bias: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Tensor)>> {
        self.bf16.bias_residual_layer_norm(
            projection,
            residual,
            norm_weight,
            norm_bias,
            eps,
            backend,
        )
    }
    fn supports_fused_self_qkv(&self) -> bool {
        true
    }
}

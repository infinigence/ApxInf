//! Immutable checkpoint mapping and prepared weight data.
use super::{host_tensor, Linear, Store};
use crate::fastwam::backend::{self, Context};
use apxinf_core::{DType, Error, Result, Tensor};
use std::path::Path;
pub(in crate::fastwam) struct Conv {
    pub(in crate::fastwam) projection: Linear,
    pub(in crate::fastwam) kernel: usize,
    pub(in crate::fastwam) stride: usize,
    pub(in crate::fastwam) padding: usize,
}

impl Conv {
    fn load(
        store: &mut Store,
        ctx: &Context,
        prefix: &str,
        dtype: DType,
        expected: &[usize],
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let raw = store.take(&format!("{prefix}.weight"))?;
        let raw_bias = store.take(&format!("{prefix}.bias"))?;
        validate_conv_shapes(
            prefix,
            raw.shape().dims(),
            raw_bias.shape().dims(),
            expected,
        )?;
        let (n, c, k) = (expected[0], expected[1], expected[expected.len() - 1]);
        let temporal = if expected.len() == 5 { expected[2] } else { 1 };
        let src = raw.to_f32_vec()?;
        let mut dst = vec![0.; k * k * c * n];
        for o in 0..n {
            for i in 0..c {
                for y in 0..k {
                    for x in 0..k {
                        dst[((y * k + x) * c + i) * n + o] =
                            src[(((o * c + i) * temporal + temporal - 1) * k + y) * k + x];
                    }
                }
            }
        }
        let weight = backend::upload(ctx, &host_tensor(vec![k * k * c, n], &dst, dtype)?)?;
        let bias = backend::upload(ctx, &host_tensor(vec![n], &raw_bias.to_f32_vec()?, dtype)?)?;
        Ok(Self {
            projection: Linear {
                weight,
                bias: Some(bias),
            },
            kernel: k,
            stride,
            padding,
        })
    }
}

// The profile fixes each convolution's rank, channels and temporal/spatial
// kernels. A kernel whitelist alone admits shape-preserving architecture changes.
fn validate_conv_shapes(
    prefix: &str,
    weight: &[usize],
    bias: &[usize],
    expected: &[usize],
) -> Result<()> {
    if weight != expected || bias != [expected[0]] {
        return Err(Error::Other(format!(
            "invalid VAE convolution {prefix}: weight {weight:?}, bias {bias:?}; expected {expected:?} and [{}]",
            expected[0]
        )));
    }
    Ok(())
}

pub(in crate::fastwam) struct Norm {
    pub(in crate::fastwam) gamma: Tensor,
}

impl Norm {
    fn load(
        store: &mut Store,
        ctx: &Context,
        prefix: &str,
        dtype: DType,
        expected: &[usize],
    ) -> Result<Self> {
        let raw = store.take(&format!("{prefix}.gamma"))?;
        let shape = raw.shape().dims();
        if shape != expected {
            return Err(Error::Other(format!(
                "invalid VAE norm gamma {prefix}: {shape:?}"
            )));
        }
        Ok(Self {
            gamma: backend::upload(
                ctx,
                &host_tensor(vec![raw.shape().numel()], &raw.to_f32_vec()?, dtype)?,
            )?,
        })
    }
}

pub(in crate::fastwam) struct Residual {
    pub(in crate::fastwam) norm1: Norm,
    pub(in crate::fastwam) conv1: Conv,
    pub(in crate::fastwam) norm2: Norm,
    pub(in crate::fastwam) conv2: Conv,
    pub(in crate::fastwam) shortcut: Option<Conv>,
}

impl Residual {
    fn load(
        store: &mut Store,
        ctx: &Context,
        prefix: &str,
        dtype: DType,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        let has_shortcut = store.contains(&format!("{prefix}.conv_shortcut.weight"));
        if has_shortcut != (input != output)
            || store.contains(&format!("{prefix}.conv_shortcut.bias")) != has_shortcut
        {
            return Err(Error::Other(format!(
                "invalid VAE residual shortcut {prefix}"
            )));
        }
        let shortcut = if has_shortcut {
            Some(Conv::load(
                store,
                ctx,
                &format!("{prefix}.conv_shortcut"),
                dtype,
                &[output, input, 1, 1, 1],
                1,
                0,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: Norm::load(
                store,
                ctx,
                &format!("{prefix}.norm1"),
                dtype,
                &[input, 1, 1, 1],
            )?,
            conv1: Conv::load(
                store,
                ctx,
                &format!("{prefix}.conv1"),
                dtype,
                &[output, input, 3, 3, 3],
                1,
                1,
            )?,
            norm2: Norm::load(
                store,
                ctx,
                &format!("{prefix}.norm2"),
                dtype,
                &[output, 1, 1, 1],
            )?,
            conv2: Conv::load(
                store,
                ctx,
                &format!("{prefix}.conv2"),
                dtype,
                &[output, output, 3, 3, 3],
                1,
                1,
            )?,
            shortcut,
        })
    }
}

pub(in crate::fastwam) struct DownBlock {
    pub(in crate::fastwam) residuals: Vec<Residual>,
    pub(in crate::fastwam) downsample: Option<Conv>,
    pub(in crate::fastwam) temporal_factor: usize,
}

pub(in crate::fastwam) struct Attention {
    pub(in crate::fastwam) norm: Norm,
    pub(in crate::fastwam) q: Linear,
    pub(in crate::fastwam) k: Linear,
    pub(in crate::fastwam) v: Linear,
    pub(in crate::fastwam) projection: Conv,
}

impl Attention {
    fn load(store: &mut Store, ctx: &Context, prefix: &str, dtype: DType) -> Result<Self> {
        let raw = store.take(&format!("{prefix}.to_qkv.weight"))?;
        let raw_bias = store.take(&format!("{prefix}.to_qkv.bias"))?;
        let c = validate_qkv_shapes(raw.shape().dims(), raw_bias.shape().dims())?;
        let values = raw.to_f32_vec()?;
        let bias = raw_bias.to_f32_vec()?;
        let mut parts = Vec::new();
        for part in 0..3 {
            let mut transposed = vec![0.; c * c];
            for n in 0..c {
                for k in 0..c {
                    transposed[k * c + n] = values[(part * c + n) * c + k];
                }
            }
            parts.push(Linear {
                weight: backend::upload(ctx, &host_tensor(vec![c, c], &transposed, dtype)?)?,
                bias: Some(backend::upload(
                    ctx,
                    &host_tensor(vec![c], &bias[part * c..(part + 1) * c], dtype)?,
                )?),
            });
        }
        let mut parts = parts.into_iter();
        Ok(Self {
            norm: Norm::load(store, ctx, &format!("{prefix}.norm"), dtype, &[640, 1, 1])?,
            q: parts.next().unwrap(),
            k: parts.next().unwrap(),
            v: parts.next().unwrap(),
            projection: Conv::load(
                store,
                ctx,
                &format!("{prefix}.proj"),
                dtype,
                &[640, 640, 1, 1],
                1,
                0,
            )?,
        })
    }
}

fn validate_qkv_shapes(weight: &[usize], bias: &[usize]) -> Result<usize> {
    if weight != [1920, 640, 1, 1] || bias != [1920] {
        return Err(Error::Other(format!(
            "invalid VAE attention weight/bias shapes: {weight:?}, {bias:?}"
        )));
    }
    Ok(640)
}

#[cfg(test)]
mod tests {
    use super::{validate_conv_shapes, validate_qkv_shapes};

    #[test]
    fn vae_convolutions_reject_architecture_substitutions() {
        for expected in [
            &[160, 12, 3, 3, 3][..], // encoder input
            &[320, 160, 3, 3, 3],    // channel-changing residual
            &[640, 640, 3, 3, 3],    // constant-width residual
            &[320, 160, 1, 1, 1],    // residual shortcut
            &[320, 320, 3, 3],       // spatial downsampler
            &[640, 640, 1, 1],       // attention output
            &[96, 640, 3, 3, 3],     // encoder output
            &[96, 96, 1, 1, 1],      // latent quantization
        ] {
            let bias = [expected[0]];
            assert!(validate_conv_shapes("test", expected, &bias, expected).is_ok());
            for axis in 0..expected.len() {
                let mut changed = expected.to_vec();
                changed[axis] = if expected[axis] == 1 { 3 } else { 1 };
                assert!(validate_conv_shapes("test", &changed, &bias, expected).is_err());
            }
            assert!(validate_conv_shapes("test", &[], &bias, expected).is_err());
            assert!(validate_conv_shapes("test", &expected[1..], &bias, expected).is_err());
            assert!(validate_conv_shapes("test", expected, &[1], expected).is_err());
            assert!(validate_conv_shapes("test", expected, &[1, expected[0]], expected).is_err());
        }
    }

    #[test]
    fn vae_qkv_rejects_malformed_shapes_before_indexing() {
        assert_eq!(
            validate_qkv_shapes(&[1920, 640, 1, 1], &[1920]).unwrap(),
            640
        );
        for shape in [
            &[][..],
            &[1920][..],
            &[1920, 640][..],
            &[1920, 640, 3, 3][..],
            &[usize::MAX, usize::MAX, 1, 1][..],
        ] {
            assert!(validate_qkv_shapes(shape, &[1920]).is_err());
        }
        assert!(validate_qkv_shapes(&[1920, 640, 1, 1], &[640]).is_err());
        assert!(validate_qkv_shapes(&[1920, 640, 1, 1], &[3, 640]).is_err());
    }
}

pub(in crate::fastwam) struct Vae {
    pub(in crate::fastwam) input: Conv,
    pub(in crate::fastwam) down: Vec<DownBlock>,
    pub(in crate::fastwam) mid0: Residual,
    pub(in crate::fastwam) attention: Attention,
    pub(in crate::fastwam) mid1: Residual,
    pub(in crate::fastwam) norm: Norm,
    pub(in crate::fastwam) output: Conv,
    pub(in crate::fastwam) quant: Conv,
    pub(in crate::fastwam) mean: Tensor,
    pub(in crate::fastwam) std: Tensor,
    pub(in crate::fastwam) dtype: DType,
}

impl Vae {
    pub(in crate::fastwam) fn load(ctx: &Context, path: &Path, dtype: DType) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.join("config.json")).map_err(|e| Error::Other(e.to_string()))?,
        )
        .map_err(|e| Error::Other(e.to_string()))?;
        for (name, expected) in [
            ("base_dim", 160),
            ("z_dim", 48),
            ("patch_size", 2),
            ("num_res_blocks", 2),
            ("in_channels", 12),
        ] {
            if config[name].as_u64() != Some(expected) {
                return Err(Error::Other(format!("unsupported Wan VAE {name}")));
            }
        }
        if config["dim_mult"] != serde_json::json!([1, 2, 4, 4])
            || config["temperal_downsample"] != serde_json::json!([false, true, true])
            || config["is_residual"] != true
        {
            return Err(Error::Other("unsupported Wan VAE block geometry".into()));
        }
        let mut store = Store::load(&path.join("diffusion_pytorch_model.safetensors"))?;
        let mut down = Vec::new();
        let mut channels = 160;
        for (i, output) in [160, 320, 640, 640].into_iter().enumerate() {
            let prefix = format!("encoder.down_blocks.{i}");
            let mut residuals = Vec::new();
            for j in 0..2 {
                residuals.push(Residual::load(
                    &mut store,
                    ctx,
                    &format!("{prefix}.resnets.{j}"),
                    dtype,
                    channels,
                    output,
                )?);
                channels = output;
            }
            let downsample = if i < 3 {
                Some(Conv::load(
                    &mut store,
                    ctx,
                    &format!("{prefix}.downsampler.resample.1"),
                    dtype,
                    &[channels, channels, 3, 3],
                    2,
                    0,
                )?)
            } else {
                None
            };
            down.push(DownBlock {
                residuals,
                downsample,
                temporal_factor: if i == 1 || i == 2 { 2 } else { 1 },
            });
        }
        let stat = |name: &str| -> Result<Tensor> {
            let values = config[name]
                .as_array()
                .ok_or_else(|| Error::Other(format!("VAE missing {name}")))?;
            let values = values
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|f| f as f32)
                        .filter(|f| f.is_finite())
                        .ok_or_else(|| Error::Other("invalid VAE latent statistic".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            if values.len() != 48 || (name == "latents_std" && values.iter().any(|&x| x <= 0.)) {
                return Err(Error::Other("invalid VAE latent statistics".into()));
            }
            // The adapter creates these buffers in F32 after loading the VAE;
            // FastWAM moves them to the device without changing their dtype.
            backend::upload(ctx, &host_tensor(vec![48], &values, DType::F32)?)
        };
        Ok(Self {
            input: Conv::load(
                &mut store,
                ctx,
                "encoder.conv_in",
                dtype,
                &[160, 12, 3, 3, 3],
                1,
                1,
            )?,
            down,
            mid0: Residual::load(
                &mut store,
                ctx,
                "encoder.mid_block.resnets.0",
                dtype,
                640,
                640,
            )?,
            attention: Attention::load(&mut store, ctx, "encoder.mid_block.attentions.0", dtype)?,
            mid1: Residual::load(
                &mut store,
                ctx,
                "encoder.mid_block.resnets.1",
                dtype,
                640,
                640,
            )?,
            norm: Norm::load(&mut store, ctx, "encoder.norm_out", dtype, &[640, 1, 1, 1])?,
            output: Conv::load(
                &mut store,
                ctx,
                "encoder.conv_out",
                dtype,
                &[96, 640, 3, 3, 3],
                1,
                1,
            )?,
            quant: Conv::load(
                &mut store,
                ctx,
                "quant_conv",
                dtype,
                &[96, 96, 1, 1, 1],
                1,
                0,
            )?,
            mean: stat("latents_mean")?,
            std: stat("latents_std")?,
            dtype,
        })
    }
}

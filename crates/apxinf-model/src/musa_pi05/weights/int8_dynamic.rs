//! int8_dynamic device weights: model aggregates and linear storage.
mod linear {
    //! Output-channel-quantized W8A8 linear weights for PI0.5.

    use apxinf_core::{DType, Error, Result, Shape, Tensor};

    use crate::musa_pi05::backend::{ops, Context, DeviceBuffer};
    use crate::musa_pi05::weights::packing::concat_host_2d;
    use crate::musa_pi05::{bf16_to_device, LinearWeights};
    use ops::w8a8::{W8A8Layout, W8A8ScaleMode, W8A8WeightView};

    /// A device allocation for the byte-level operands the tile ABI defines:
    /// the INT8 weight values and the FP16 scale vector. The model never reads
    /// them as tensors, so they stay buffers.
    fn device_buffer(bytes: &[u8], ctx: &Context) -> Result<DeviceBuffer> {
        let buffer = DeviceBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Other)?;
        buffer.copy_from_host(bytes).map_err(Error::Other)?;
        Ok(buffer)
    }

    /// `activation @ weight`, allocating the BF16 output.
    fn bf16_linear(
        ctx: &Context,
        activation: &Tensor,
        weight: &Tensor,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        let rows = activation.shape().dims()[0];
        let cols = weight.shape().dims()[1];
        let mut output = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
        let mut args = ops::GemmArgs::new(activation, weight, &mut output);
        args.policy = policy.clone();
        ops::gemm(ctx, args)?;
        Ok(output)
    }

    /// `activation @ weight` on the W8A8 route, allocating the BF16 output.
    ///
    /// No policy rides along: these tiles are embedded images chosen by shape,
    /// not candidates a tactic session selects between.
    fn w8a8_linear(
        ctx: &Context,
        activation: &Tensor,
        weight: W8A8WeightView<'_>,
    ) -> Result<Tensor> {
        let rows = activation.shape().dims()[0];
        let mut output =
            ctx.allocate_output(Shape::new(vec![rows, weight.output_dim]), DType::BF16)?;
        ops::w8a8::gemm(ctx, activation, weight, &mut output)?;
        Ok(output)
    }

    /// `geglu(gate_up)`: `[rows, 2n] -> [rows, n]`, allocating the output.
    fn geglu(ctx: &Context, gate_up: &Tensor) -> Result<Tensor> {
        let dims = gate_up.shape().dims();
        let mut output = ctx.allocate_output(Shape::new(vec![dims[0], dims[1] / 2]), DType::BF16)?;
        ops::pointwise(
            ctx,
            ops::PointwiseArgs::new(ops::PointwiseSemantic::Geglu, gate_up, &mut output),
        )?;
        Ok(output)
    }

    /// Multiply a BF16 matrix by a per-channel f32 vector, allocating the output.
    fn scale_channels(ctx: &Context, input: &Tensor, channel_scale: &Tensor) -> Result<Tensor> {
        let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
        ops::w8a8::mul_channels(ctx, input, channel_scale, &mut output)?;
        Ok(output)
    }

    pub struct Int8DynamicLinearWeights {
        /// Physical output-major `[output,input]` INT8 bytes. This is also a
        /// zero-copy `[input,output]` column-major view for cuBLAS/CUTLASS.
        pub weight_output_major: DeviceBuffer,
        /// KN copy the short-row decode image reads for gate/up. The NK
        /// original stays for encode and for row counts that image does not
        /// serve; packing this copy once at load keeps a transpose off the GEMM
        /// path.
        weight_decode_nn: Option<DeviceBuffer>,
        /// Dequantization multiplier for each output channel.
        pub weight_scales: Tensor,
        /// FP16 rendering of `weight_scales` with `W8A8_SCALE_PAD` trailing
        /// zeros. The MUSA `s8mma` tiles read their scales as FP16, so packing
        /// converts once here (host round-to-nearest-even, bitwise-identical
        /// to the per-GEMM conversion kernel it replaces) instead of running a
        /// cast kernel on every GEMM. CUDA consumes the f32 vector and ignores
        /// this one.
        pub weight_scales_f16: DeviceBuffer,
        pub bias: Option<Tensor>,
        pub input_dim: usize,
        pub output_dim: usize,
        /// Physical row-major `[input,output]` BF16 matrix, for the shapes no
        /// INT8 tile can serve; `None` when the INT8 route covers this layer.
        ///
        /// The s8mma tiles mask a partial row but neither a partial output
        /// column nor a partial contraction (`ops::w8a8::tile_serves`),
        /// so a projection that misses both keeps the BF16 arithmetic and
        /// `int8_dynamic` stays a runnable variant instead of a load-time gap.
        /// What still lands here is the ViT's `k = 588` patch embedding and the
        /// action expert's `k = 32` input and `n = 32` output. The ViT's MLP
        /// (`k = n = 4304`) does *not*: it is tile-padded to 4352 at pack time
        /// ([`Int8DynamicLinearWeights::from_host_parts_padded`]) and runs INT8.
        ///
        /// The matrix carries the padded shape when it is set, so the padded
        /// INT8 and padded BF16 routes produce the same result.
        bf16_fallback: Option<Tensor>,
        /// Reciprocal of the per-channel equalization vector this projection's
        /// weight was folded with, kept only when the activation cannot be
        /// pre-scaled through a normalization scale (the language down
        /// projection's GeGLU output). The INT8 route hands it to the row
        /// quantizer, which applies it while measuring the amax -- no extra
        /// pass; the BF16 fallback pre-multiplies the activation instead.
        /// With the weight fold the product is unchanged either way.
        equalization_inv_c: Option<Tensor>,
    }

    /// Which projection groups the INT8 route covers.
    ///
    /// Selected by `model_variant` at load
    /// ([`crate::musa_pi05::ModelVariantChoice`]). It is preparation rather than
    /// execution — the choice is baked into the weights' `bf16_fallback` — so
    /// two scopes built from the same checkpoint share every dispatch arm.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Int8Scope {
        /// Every projection an embedded tile can serve.
        All,
        /// Everything but the language (encode) layers' four projections, which
        /// keep BF16 arithmetic. The ViT and the action expert (decode) stay
        /// INT8. The language layers are the ones whose per-row activation scale
        /// collides with the residual stream's outlier channels.
        WithoutEncode,
    }

    impl Int8Scope {
        /// Whether the projection loaded for `group` may take the INT8 route.
        pub fn allows(self, group: &str) -> bool {
            let allowed = match self {
                Self::All => true,
                Self::WithoutEncode => !group.starts_with("lang_"),
            };
            allowed
                && !env_exclusions()
                    .iter()
                    .any(|prefix| group.starts_with(prefix.as_str()))
        }
    }

    /// Diagnostic attribution knob: comma-separated group prefixes the INT8
    /// route refuses, read once from `APXINF_INT8_DYNAMIC_EXCLUDE` (for
    /// example `lang_attn,lang_down`). Excluded projections keep BF16
    /// arithmetic, so per-family cosine/latency A/B runs can isolate which
    /// quantized projection owns an accuracy gap. Load-time only; empty in
    /// production.
    fn env_exclusions() -> &'static [String] {
        static EXCLUSIONS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        EXCLUSIONS.get_or_init(|| {
            let exclusions: Vec<String> = std::env::var("APXINF_INT8_DYNAMIC_EXCLUDE")
                .map(|value| {
                    value
                        .split(',')
                        .map(|item| item.trim().to_string())
                        .filter(|item| !item.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            if !exclusions.is_empty() {
                eprintln!("int8_dynamic: INT8 route excluded for groups {exclusions:?}");
            }
            exclusions
        })
    }

    /// Per-channel activation equalization (SmoothQuant-style), a load-time
    /// accuracy diagnostic for the language encode layers.
    ///
    /// The residual stream those layers share has a handful of outlier
    /// channels 30-1000x the median, and one `amax/127` row scale cannot cover
    /// both them and the typical element. The identity
    /// `x @ W^T == (x ./ c) @ (W .* c)^T` moves that dynamic range onto the
    /// weight, whose per-output-channel scales absorb it: the weight's
    /// contraction row `k` is multiplied by `c[k]` at pack time and the
    /// activation is divided by it before quantization. `c` comes from the
    /// `APXINF_INT8_EQUALIZATION` JSON file — a map from diagnostic site
    /// (`lang.qkv_in`, `lang.gate_up_in`, `lang.down_in`) to one vector per
    /// site, built from the `APXINF_PI05_INT8_STATS` channel maxima
    /// (`scripts/build_pi05_int8_equalization.py`).
    pub struct Equalization {
        /// One multiplier per contraction channel, folded into the weight.
        channels: Vec<f32>,
        /// Whether the caller divided the preceding normalization scale by
        /// `channels` already. When false the GEMM pre-scales the activation
        /// instead, so the reciprocal is kept on the device.
        norm_folded: bool,
    }

    /// The `APXINF_INT8_EQUALIZATION` table, read once; empty in production.
    fn env_equalizations() -> &'static std::collections::HashMap<String, Vec<f32>> {
        static EQUALIZATIONS: std::sync::OnceLock<std::collections::HashMap<String, Vec<f32>>> =
            std::sync::OnceLock::new();
        EQUALIZATIONS.get_or_init(|| {
            let Ok(path) = std::env::var("APXINF_INT8_EQUALIZATION") else {
                return Default::default();
            };
            let parsed = std::fs::read_to_string(&path).and_then(|text| {
                serde_json::from_str::<std::collections::HashMap<String, Vec<f32>>>(&text)
                    .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
            });
            match parsed {
                Ok(map) if !map.is_empty() => {
                    let mut sites: Vec<&str> = map.keys().map(String::as_str).collect();
                    sites.sort_unstable();
                    eprintln!("int8_dynamic: per-channel equalization for sites {sites:?}");
                    map
                }
                Ok(_) => Default::default(),
                Err(error) => {
                    eprintln!("int8_dynamic: ignoring APXINF_INT8_EQUALIZATION={path}: {error}");
                    Default::default()
                }
            }
        })
    }

    /// The diagnostic site whose calibration vector a projection group folds.
    ///
    /// `lang.o_proj_in` is deliberately absent: its activation is an attention
    /// convex combination and already quantizes cleanly (~20 levels), so the
    /// output projection keeps its plain weight.
    fn equalization_site(group: &str) -> Option<&'static str> {
        match group {
            "lang_attn_qkv" => Some("lang.qkv_in"),
            "lang_gate_up" => Some("lang.gate_up_in"),
            "lang_down" => Some("lang.down_in"),
            _ => None,
        }
    }

    /// The equalization a projection loaded for `group` folds, if any.
    ///
    /// Gated on the same `scope.allows` as the INT8 route itself, so an
    /// excluded projection stays bit-identical. The qkv and gate/up inputs
    /// divide by `c` through the preceding RMS norm's scale (`norm_folded`);
    /// the down projection has no norm in front of it and pre-scales in the
    /// GEMM.
    pub fn equalization_for(group: &str, scope: Int8Scope) -> Option<Equalization> {
        if !scope.allows(group) {
            return None;
        }
        let channels = env_equalizations().get(equalization_site(group)?)?.clone();
        Some(Equalization {
            channels,
            norm_folded: group != "lang_down",
        })
    }

    /// Divide a normalization scale by the projection's equalization vector,
    /// so the normalized activation the projection sees already carries
    /// `1/c`. Returns the scale unchanged when the group has no equalization.
    pub fn fold_norm_scale(scale: &Tensor, equalization: Option<&Equalization>) -> Result<Tensor> {
        let Some(equalization) = equalization else {
            return Ok(scale.clone());
        };
        let mut values = scale.to_f32_vec()?;
        if values.len() != equalization.channels.len() {
            return Err(Error::Other(format!(
                "equalization vector is {} wide for a {}-wide norm scale",
                equalization.channels.len(),
                values.len()
            )));
        }
        for (value, channel) in values.iter_mut().zip(&equalization.channels) {
            *value /= channel;
        }
        Tensor::from_f32(vec![values.len()], &values)
    }

    impl Int8DynamicLinearWeights {
        pub fn from_host(
            linear: &LinearWeights,
            backend: &Context,
            group: &str,
            scope: Int8Scope,
        ) -> Result<Self> {
            Self::build(&[linear], backend, group, false, scope)
        }

        /// Pack QKV or gate/up along the output dimension, then independently
        /// quantize every output channel across its complete input row.
        pub fn from_host_parts(
            linears: &[&LinearWeights],
            backend: &Context,
            group: &str,
            scope: Int8Scope,
        ) -> Result<Self> {
            Self::build(linears, backend, group, false, scope)
        }

        /// As [`from_host_parts`](Self::from_host_parts), but first rounds the
        /// packed `[input,output]` shape up to the next multiple of 64 with zero
        /// fill, so a projection whose true shape the tiles cannot divide
        /// becomes INT8-eligible.
        ///
        /// This is what brings the ViT's MLP into INT8: its `k = n = 4304` is a
        /// multiple of neither tile width nor `tile_k`, and rounding to 4352
        /// costs 1.1% more FLOPs. The padding is exact -- the extra output
        /// channels carry a zero weight row and so dequantize to exactly zero,
        /// and the extra contraction rows are zero, so the retained outputs are
        /// unchanged. It only makes sense as a *pair*: the downstream
        /// projection's contraction is this one's output, so both sides have to
        /// round to the same width.
        pub fn from_host_parts_padded(
            linears: &[&LinearWeights],
            backend: &Context,
            group: &str,
            scope: Int8Scope,
        ) -> Result<Self> {
            Self::build(linears, backend, group, true, scope)
        }

        fn build(
            linears: &[&LinearWeights],
            backend: &Context,
            group: &str,
            pad_to_tile: bool,
            scope: Int8Scope,
        ) -> Result<Self> {
            if linears.is_empty() {
                return Err(Error::Other(
                    "cannot pack an empty INT8 linear group".into(),
                ));
            }
            let packed = concat_host_2d(
                &linears
                    .iter()
                    .map(|linear| &linear.weight)
                    .collect::<Vec<_>>(),
            )?;
            let equalization = equalization_for(group, scope);
            let packed = match &equalization {
                Some(equalization) => {
                    fold_equalization_into_weight(&packed, &equalization.channels)?
                }
                None => packed,
            };
            let dims = packed.shape().dims();
            let (padded_input, padded_output) = if pad_to_tile {
                (round_up_to_tile(dims[0]), round_up_to_tile(dims[1]))
            } else {
                (dims[0], dims[1])
            };
            let packed = if padded_input == dims[0] && padded_output == dims[1] {
                packed
            } else {
                zero_pad_matrix(&packed, padded_input, padded_output)?
            };
            let (quantized, scales, input_dim, output_dim) = quantize_output_channels(&packed)?;
            let bytes = quantized
                .into_iter()
                .map(|value| value as u8)
                .collect::<Vec<_>>();
            let weight_output_major = device_buffer(&bytes, backend)?;
            let weight_decode_nn = if group == "action_mlp"
                && linears.len() == 2
                && scope.allows(group)
                && ops::w8a8::tile_serves(output_dim, input_dim)
                && ops::w8a8::decode_supported()
            {
                // A physical weight transformation, completed before warmup
                // and capture. The quantized values and their FP16 scales are
                // preserved bit for bit; only the byte order changes.
                let mut input_major = vec![0u8; bytes.len()];
                for input in 0..input_dim {
                    for output in 0..output_dim {
                        input_major[input * output_dim + output] =
                            bytes[output * input_dim + input];
                    }
                }
                Some(device_buffer(&input_major, backend)?)
            } else {
                None
            };
            let weight_scales = crate::musa_pi05::backend::to_device(
                backend,
                &Tensor::from_f32(vec![output_dim], &scales)?,
            )?;
            let weight_scales_f16 = pack_scales_f16(&scales, backend)?;
            let bf16_fallback =
                if !scope.allows(group) || !ops::w8a8::tile_serves(output_dim, input_dim) {
                    Some(bf16_to_device(&packed, backend)?)
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
                    output_dim,
                    backend,
                )?)
            } else {
                return Err(Error::Other(
                    "cannot pack INT8 projections with mixed bias presence".into(),
                ));
            };
            let equalization_inv_c = match &equalization {
                Some(equalization) if !equalization.norm_folded => {
                    let reciprocals: Vec<f32> = equalization
                        .channels
                        .iter()
                        .map(|channel| 1.0 / channel)
                        .collect();
                    Some(crate::musa_pi05::backend::to_device(
                        backend,
                        &Tensor::from_f32(vec![reciprocals.len()], &reciprocals)?,
                    )?)
                }
                _ => None,
            };
            Ok(Self {
                weight_output_major,
                weight_decode_nn,
                weight_scales,
                weight_scales_f16,
                bias,
                input_dim,
                output_dim,
                bf16_fallback,
                equalization_inv_c,
            })
        }

        pub fn gemm(&self, ctx: &Context, activation: &Tensor) -> Result<Tensor> {
            self.gemm_with_policies(ctx, activation, &ops::GemmPolicy::default())
        }

        pub fn gemm_with_policies(
            &self,
            ctx: &Context,
            activation: &Tensor,
            gemm_policy: &ops::GemmPolicy,
        ) -> Result<Tensor> {
            match &self.bf16_fallback {
                Some(weight) => {
                    // The fallback's BF16 weight carries the same equalization
                    // fold the INT8 one does, so its activation still has to be
                    // pre-scaled -- the fold only fuses into the INT8
                    // quantizer, not into a BF16 GEMM.
                    let scaled;
                    let activation = match &self.equalization_inv_c {
                        Some(inv_c) => {
                            scaled = scale_channels(ctx, activation, inv_c)?;
                            &scaled
                        }
                        None => activation,
                    };
                    bf16_linear(ctx, activation, weight, gemm_policy)
                }
                None => w8a8_linear(ctx, activation, self.as_kernel_view()),
            }
        }

        /// `geglu(gate_up) @ weight`, quantizing the activation as it is produced.
        ///
        /// The fused producer is what keeps the down projection's BF16
        /// intermediate out of the arena. A projection that fell back to BF16
        /// has no quantizer to fuse into, so it materializes the activation and
        /// takes its equalization fold as a separate pass.
        pub fn gemm_geglu(
            &self,
            ctx: &Context,
            gate_up: &Tensor,
            gemm_policy: &ops::GemmPolicy,
        ) -> Result<Tensor> {
            if self.bf16_fallback.is_some() {
                let activated = geglu(ctx, gate_up)?;
                return self.gemm_with_policies(ctx, &activated, gemm_policy);
            }
            let rows = gate_up.shape().dims()[0];
            let mut output =
                ctx.allocate_output(Shape::new(vec![rows, self.output_dim]), DType::BF16)?;
            ops::w8a8::gemm_geglu(ctx, gate_up, self.as_kernel_view(), &mut output)?;
            Ok(output)
        }

        pub fn as_kernel_view(&self) -> W8A8WeightView<'_> {
            W8A8WeightView::new(
                &self.weight_output_major,
                &self.weight_scales,
                &self.weight_scales_f16,
                self.input_dim,
                self.output_dim,
                W8A8ScaleMode::DynamicRowPerOutputChannel,
                W8A8Layout::OutputMajor,
                self.equalization_inv_c.as_ref(),
            )
            .with_decode_nn_weight(self.weight_decode_nn.as_ref())
        }
    }

    /// Round `dim` up to the next multiple of 64.
    ///
    /// 64 is the smallest extent any embedded tile masks by, so a dimension
    /// rounded this way divides some tile's `tile_n`/`tile_k`
    /// (`ops::w8a8::tile_serves`).
    fn round_up_to_tile(dim: usize) -> usize {
        dim.div_ceil(64) * 64
    }

    /// Copy `[rows,cols]` into a zero `[input,output]`, original at the top left.
    fn zero_pad_matrix(tensor: &Tensor, input: usize, output: usize) -> Result<Tensor> {
        let dims = tensor.shape().dims();
        if dims.len() != 2 {
            return Err(Error::Other(format!("expected 2D weight, got {dims:?}")));
        }
        let (rows, cols) = (dims[0], dims[1]);
        if rows > input || cols > output {
            return Err(Error::Other(format!(
                "cannot pad a {rows}x{cols} weight into {input}x{output}"
            )));
        }
        let source = tensor.to_f32_vec()?;
        let mut values = vec![0.0f32; input * output];
        for row in 0..rows {
            values[row * output..row * output + cols]
                .copy_from_slice(&source[row * cols..(row + 1) * cols]);
        }
        Tensor::from_f32(vec![input, output], &values)
    }

    /// Render the f32 output-channel scales as the FP16 vector the MUSA tiles
    /// read, with the `W8A8_SCALE_PAD` trailing zeros their guarded scale load
    /// may over-index into. `half::f16::from_f32` rounds to nearest-even,
    /// bitwise-identical to the `__float2half_rn` pass a per-GEMM conversion
    /// runs. CUDA ignores the buffer; it is packed on either
    /// backend so this code stays backend-neutral.
    fn pack_scales_f16(scales: &[f32], backend: &Context) -> Result<DeviceBuffer> {
        let mut bytes: Vec<u8> = scales
            .iter()
            .flat_map(|value| half::f16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        bytes.resize(bytes.len() + ops::w8a8::W8A8_SCALE_PAD * 2, 0);
        device_buffer(&bytes, backend)
    }

    /// Scale contraction row `k` of the `[input, output]` matrix by
    /// `channels[k]` -- the weight half of the per-channel equalization. Runs
    /// before any tile padding, so `channels` spans the true input width and
    /// the zero rows padding appends stay zero (their activation channels are
    /// exactly zero too, so the fold there is a no-op either way).
    fn fold_equalization_into_weight(tensor: &Tensor, channels: &[f32]) -> Result<Tensor> {
        let dims = tensor.shape().dims();
        if dims.len() != 2 || dims[0] != channels.len() {
            return Err(Error::Other(format!(
                "equalization vector is {} wide for a {dims:?} weight",
                channels.len()
            )));
        }
        let (_, output_dim) = (dims[0], dims[1]);
        let mut values = tensor.to_f32_vec()?;
        for (row, channel) in channels.iter().enumerate() {
            for value in &mut values[row * output_dim..(row + 1) * output_dim] {
                *value *= channel;
            }
        }
        Tensor::from_f32(vec![channels.len(), output_dim], &values)
    }

    /// Convert ApxInf's physical `[input,output]` matrix into the kernel's physical
    /// `[output,input]` INT8 layout, with one `amax/127` scale per output.
    fn quantize_output_channels(tensor: &Tensor) -> Result<(Vec<i8>, Vec<f32>, usize, usize)> {
        if tensor.dtype() == DType::F8E4M3 {
            return Err(Error::Other(
                "cannot quantize a scale-less E4M3 matrix to INT8".into(),
            ));
        }
        let dims = tensor.shape().dims();
        if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
            return Err(Error::Other(format!(
                "INT8 weight must be a non-empty matrix, got {dims:?}"
            )));
        }
        let (input_dim, output_dim) = (dims[0], dims[1]);
        let values = tensor.to_f32_vec()?;
        let mut quantized = vec![0i8; input_dim * output_dim];
        let mut scales = vec![0.0f32; output_dim];
        for output in 0..output_dim {
            let mut maximum = 0.0f32;
            for input in 0..input_dim {
                maximum = maximum.max(values[input * output_dim + output].abs());
            }
            let scale = (maximum / 127.0).max(1.0e-12);
            scales[output] = scale;
            for input in 0..input_dim {
                let value = (values[input * output_dim + output] / scale)
                    .round()
                    .clamp(-128.0, 127.0);
                quantized[output * input_dim + input] = value as i8;
            }
        }
        Ok((quantized, scales, input_dim, output_dim))
    }

    /// Concatenate the packed projections' biases, zero-extended to the (possibly
    /// tile-padded) output width the activations carry.
    fn concat_biases_bf16(
        tensors: &[&Tensor],
        output_dim: usize,
        backend: &Context,
    ) -> Result<Tensor> {
        let mut values = Vec::new();
        for tensor in tensors {
            if tensor.shape().dims().len() != 1 || tensor.dtype() == DType::F8E4M3 {
                return Err(Error::Other(
                    "packed INT8 biases must be non-FP8 vectors".into(),
                ));
            }
            values.extend(tensor.to_f32_vec()?.into_iter().map(half::bf16::from_f32));
        }
        if values.len() > output_dim {
            return Err(Error::Other(format!(
                "packed INT8 biases are {} long, wider than the {output_dim} output",
                values.len()
            )));
        }
        values.resize(output_dim, half::bf16::ZERO);
        crate::musa_pi05::backend::to_device(
            backend,
            &Tensor::from_bf16(vec![values.len()], &values)?,
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn quantizes_each_output_channel_and_transposes_physically() {
            let weight = Tensor::from_f32(vec![3, 2], &[1.0, -10.0, 2.0, 0.0, 3.0, 10.0]).unwrap();
            let (quantized, scales, input, output) = quantize_output_channels(&weight).unwrap();
            assert_eq!((input, output), (3, 2));
            assert_eq!(quantized, vec![42, 85, 127, -127, 0, 127]);
            assert!((scales[0] - 3.0 / 127.0).abs() < 1.0e-7);
            assert!((scales[1] - 10.0 / 127.0).abs() < 1.0e-7);
        }

        #[test]
        fn zero_channel_uses_finite_minimum_scale() {
            let weight = Tensor::from_f32(vec![2, 1], &[0.0, 0.0]).unwrap();
            let (quantized, scales, _, _) = quantize_output_channels(&weight).unwrap();
            assert_eq!(quantized, vec![0, 0]);
            assert_eq!(scales, vec![1.0e-12]);
        }

        #[test]
        fn equalization_scales_weight_rows_and_norm_reciprocally() {
            // The identity the fold rests on: `(x / c) @ (W * c)^T == x @ W^T`,
            // so the weight gains `c` per contraction row and the norm scale
            // loses it.
            let weight = Tensor::from_f32(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
            let folded = fold_equalization_into_weight(&weight, &[2.0, 0.5]).unwrap();
            assert_eq!(folded.to_f32_vec().unwrap(), vec![2.0, 4.0, 1.5, 2.0]);
            assert!(fold_equalization_into_weight(&weight, &[1.0]).is_err());

            let scale = Tensor::from_f32(vec![2], &[2.0, 4.0]).unwrap();
            let equalization = Equalization {
                channels: vec![2.0, 0.5],
                norm_folded: true,
            };
            let folded = fold_norm_scale(&scale, Some(&equalization)).unwrap();
            assert_eq!(folded.to_f32_vec().unwrap(), vec![1.0, 8.0]);
            let untouched = fold_norm_scale(&scale, None).unwrap();
            assert_eq!(untouched.to_f32_vec().unwrap(), vec![2.0, 4.0]);
        }

        #[test]
        fn rounding_reaches_the_next_tile_multiple() {
            assert_eq!(round_up_to_tile(1152), 1152);
            assert_eq!(round_up_to_tile(4304), 4352);
            assert_eq!(round_up_to_tile(588), 640);
            assert_eq!(round_up_to_tile(64), 64);
        }

        #[test]
        fn padding_leaves_the_retained_channels_unchanged() {
            let weight = Tensor::from_f32(vec![2, 2], &[1.0, -8.0, 0.5, 4.0]).unwrap();
            let (before, before_scales, _, _) = quantize_output_channels(&weight).unwrap();

            let padded = zero_pad_matrix(&weight, 4, 4).unwrap();
            let (after, after_scales, input, output) = quantize_output_channels(&padded).unwrap();
            assert_eq!((input, output), (4, 4));

            // The retained channels keep their scale and their codes, now followed
            // by the two zero contraction rows the pad introduced.
            assert_eq!(before_scales.as_slice(), &after_scales[..2]);
            for channel in 0..2 {
                assert_eq!(
                    &after[channel * 4..channel * 4 + 2],
                    &before[channel * 2..channel * 2 + 2]
                );
                assert_eq!(&after[channel * 4 + 2..channel * 4 + 4], &[0, 0]);
            }
            // The padded output channels are dead: minimum scale, zero codes.
            assert_eq!(&after_scales[2..], &[1.0e-12, 1.0e-12]);
            assert!(after[8..].iter().all(|&code| code == 0));
        }

        #[test]
        fn padding_rejects_a_shape_that_would_not_fit() {
            let weight = Tensor::from_f32(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
            assert!(zero_pad_matrix(&weight, 1, 2).is_err());
            assert!(zero_pad_matrix(&weight, 2, 1).is_err());
        }
    }
}
pub use linear::*;
// Fully materialized W8A8 PI0.5 weights.

use apxinf_core::{Result, Tensor};

use crate::musa_pi05::backend::Context;
use crate::musa_pi05::{
    bf16_to_device, ActionLayerWeights, AdaRmsNormWeights, LanguageLayerWeights, LayerNormWeights,
    Pi05Weights, VisionBlockWeights,
};

pub struct Int8DynamicDeviceLayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
}

pub struct Int8DynamicDeviceVisionBlock {
    pub norm1: Int8DynamicDeviceLayerNorm,
    pub qkv: Int8DynamicLinearWeights,
    pub output: Int8DynamicLinearWeights,
    pub norm2: Int8DynamicDeviceLayerNorm,
    pub fc1: Int8DynamicLinearWeights,
    pub fc2: Int8DynamicLinearWeights,
}

pub struct Int8DynamicDeviceLanguageLayer {
    pub input_norm_scale: Tensor,
    pub qkv: Int8DynamicLinearWeights,
    pub output: Int8DynamicLinearWeights,
    pub post_attention_norm_scale: Tensor,
    pub gate_up: Int8DynamicLinearWeights,
    pub down: Int8DynamicLinearWeights,
}

pub struct Int8DynamicDeviceActionLayer {
    pub input_modulation: Int8DynamicLinearWeights,
    pub qkv: Int8DynamicLinearWeights,
    pub output: Int8DynamicLinearWeights,
    pub post_attention_modulation: Int8DynamicLinearWeights,
    pub gate_up: Int8DynamicLinearWeights,
    pub down: Int8DynamicLinearWeights,
}

pub struct Int8DynamicWeights {
    pub patch_embedding: Int8DynamicLinearWeights,
    pub position_embedding: Tensor,
    pub vision_layers: Vec<Int8DynamicDeviceVisionBlock>,
    pub vision_post_norm: Int8DynamicDeviceLayerNorm,
    pub multimodal_projector: Int8DynamicLinearWeights,
    pub token_embedding: Tensor,
    pub language_layers: Vec<Int8DynamicDeviceLanguageLayer>,
    pub language_final_norm_scale: Tensor,
    pub action_layers: Vec<Int8DynamicDeviceActionLayer>,
    pub action_final_modulation: Int8DynamicLinearWeights,
    pub action_in: Int8DynamicLinearWeights,
    pub action_out: Int8DynamicLinearWeights,
    pub time_mlp_in: Int8DynamicLinearWeights,
    pub time_mlp_out: Int8DynamicLinearWeights,
}

impl Int8DynamicWeights {
    pub fn from_host(
        weights: &Pi05Weights,
        backend: &Context,
        scope: Int8Scope,
    ) -> Result<Self> {
        Ok(Self {
            patch_embedding: Int8DynamicLinearWeights::from_host(
                &weights.vision.patch_embedding,
                backend,
                "glue",
                scope,
            )?,
            position_embedding: bf16_to_device(&weights.vision.position_embedding, backend)?,
            vision_layers: weights
                .vision
                .blocks
                .iter()
                .map(|layer| Int8DynamicDeviceVisionBlock::from_host(layer, backend, scope))
                .collect::<Result<Vec<_>>>()?,
            vision_post_norm: Int8DynamicDeviceLayerNorm::from_host(
                &weights.vision.post_layer_norm,
                backend,
            )?,
            multimodal_projector: Int8DynamicLinearWeights::from_host(
                &weights.vision.multimodal_projector,
                backend,
                "glue",
                scope,
            )?,
            token_embedding: bf16_to_device(&weights.vision.token_embedding, backend)?,
            language_layers: weights
                .language_layers
                .iter()
                .map(|layer| Int8DynamicDeviceLanguageLayer::from_host(layer, backend, scope))
                .collect::<Result<Vec<_>>>()?,
            language_final_norm_scale: bf16_to_device(&weights.language_final_norm_scale, backend)?,
            action_layers: weights
                .action_layers
                .iter()
                .map(|layer| Int8DynamicDeviceActionLayer::from_host(layer, backend, scope))
                .collect::<Result<Vec<_>>>()?,
            action_final_modulation: modulation_to_device(
                &weights.action_final_norm,
                backend,
                scope,
            )?,
            action_in: Int8DynamicLinearWeights::from_host(
                &weights.action_in,
                backend,
                "glue",
                scope,
            )?,
            action_out: Int8DynamicLinearWeights::from_host(
                &weights.action_out,
                backend,
                "glue",
                scope,
            )?,
            time_mlp_in: Int8DynamicLinearWeights::from_host(
                &weights.time_mlp_in,
                backend,
                "glue",
                scope,
            )?,
            time_mlp_out: Int8DynamicLinearWeights::from_host(
                &weights.time_mlp_out,
                backend,
                "glue",
                scope,
            )?,
        })
    }
}

impl Int8DynamicDeviceLayerNorm {
    fn from_host(weights: &LayerNormWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            weight: bf16_to_device(&weights.weight, backend)?,
            bias: bf16_to_device(&weights.bias, backend)?,
        })
    }
}

impl Int8DynamicDeviceVisionBlock {
    fn from_host(
        weights: &VisionBlockWeights,
        backend: &Context,
        scope: Int8Scope,
    ) -> Result<Self> {
        Ok(Self {
            norm1: Int8DynamicDeviceLayerNorm::from_host(&weights.norm1, backend)?,
            qkv: Int8DynamicLinearWeights::from_host_parts(
                &[&weights.q, &weights.k, &weights.v],
                backend,
                "vision_attn",
                scope,
            )?,
            output: Int8DynamicLinearWeights::from_host(
                &weights.output,
                backend,
                "vision_attn",
                scope,
            )?,
            norm2: Int8DynamicDeviceLayerNorm::from_host(&weights.norm2, backend)?,
            // The ViT MLP's `k = n = 4304` divides no tile, so both sides are
            // zero-padded to 4352 and the pair stays consistent: FC1's extra
            // output channels come out exactly zero and FC2 ignores them.
            fc1: Int8DynamicLinearWeights::from_host_parts_padded(
                &[&weights.fc1],
                backend,
                "vision_mlp",
                scope,
            )?,
            fc2: Int8DynamicLinearWeights::from_host_parts_padded(
                &[&weights.fc2],
                backend,
                "vision_mlp",
                scope,
            )?,
        })
    }
}

impl Int8DynamicDeviceLanguageLayer {
    fn from_host(
        weights: &LanguageLayerWeights,
        backend: &Context,
        scope: Int8Scope,
    ) -> Result<Self> {
        // The qkv and gate/up equalization vectors divide into the norm scale
        // that produces their activation, so the equalized activation costs no
        // extra pass; `build` folds `c` into the weights themselves. The down
        // projection's GeGLU output has no norm in front of it and is scaled
        // in its `gemm` instead.
        let qkv_equalization = equalization_for("lang_attn_qkv", scope);
        let gate_up_equalization = equalization_for("lang_gate_up", scope);
        Ok(Self {
            input_norm_scale: bf16_to_device(
                &fold_norm_scale(&weights.input_norm_scale, qkv_equalization.as_ref())?,
                backend,
            )?,
            qkv: Int8DynamicLinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
                "lang_attn_qkv",
                scope,
            )?,
            output: Int8DynamicLinearWeights::from_host(
                &weights.attention.output,
                backend,
                "lang_attn_out",
                scope,
            )?,
            post_attention_norm_scale: bf16_to_device(
                &fold_norm_scale(
                    &weights.post_attention_norm_scale,
                    gate_up_equalization.as_ref(),
                )?,
                backend,
            )?,
            gate_up: Int8DynamicLinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
                "lang_gate_up",
                scope,
            )?,
            down: Int8DynamicLinearWeights::from_host(
                &weights.mlp.down,
                backend,
                "lang_down",
                scope,
            )?,
        })
    }
}

impl Int8DynamicDeviceActionLayer {
    fn from_host(
        weights: &ActionLayerWeights,
        backend: &Context,
        scope: Int8Scope,
    ) -> Result<Self> {
        Ok(Self {
            input_modulation: modulation_to_device(&weights.input_norm, backend, scope)?,
            qkv: Int8DynamicLinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
                "action_attn",
                scope,
            )?,
            output: Int8DynamicLinearWeights::from_host(
                &weights.attention.output,
                backend,
                "action_attn",
                scope,
            )?,
            post_attention_modulation: modulation_to_device(
                &weights.post_attention_norm,
                backend,
                scope,
            )?,
            gate_up: Int8DynamicLinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
                "action_mlp",
                scope,
            )?,
            down: Int8DynamicLinearWeights::from_host(
                &weights.mlp.down,
                backend,
                "action_mlp",
                scope,
            )?,
        })
    }
}

fn modulation_to_device(
    weights: &AdaRmsNormWeights,
    backend: &Context,
    scope: Int8Scope,
) -> Result<Int8DynamicLinearWeights> {
    Int8DynamicLinearWeights::from_host(&weights.modulation, backend, "glue", scope)
}

//! Chinese HuBERT feature encoder plus the semantic-token quantizer.

use apxinf_core::{Error, Result, Tensor};
use apxinf_cuda_new::CudaContext as Context;

use super::config::{HUBERT_CONV_DIM, HUBERT_HEADS, HUBERT_HIDDEN, HUBERT_LAYERS};
use super::ops;
use super::weights::{fp16_tensor, Linear, Weights};

const CONV_KERNEL: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
const CONV_STRIDE: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];

struct EncoderLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln_w: Tensor,
    ln_b: Tensor,
    ff1: Linear,
    ff2: Linear,
    out_ln_w: Tensor,
    out_ln_b: Tensor,
}

pub struct Hubert {
    convs: Vec<Tensor>,
    conv_bias: Vec<Option<Tensor>>,
    group_w: Tensor,
    group_b: Tensor,
    proj_w: Tensor,
    proj_b: Option<Tensor>,
    proj_ln_w: Tensor,
    proj_ln_b: Tensor,
    pos_w: Tensor,
    pos_b: Option<Tensor>,
    enc_ln_w: Tensor,
    enc_ln_b: Tensor,
    layers: Vec<EncoderLayer>,
    ssl_proj_w: Tensor,
    ssl_proj_b: Option<Tensor>,
    codebook: Tensor,
    codebook_t: Tensor,
    code_bias: Tensor,
}

#[cfg(test)]
fn capture_feature_stage(ctx: &Context, input_samples: usize, name: &str, value: &Tensor) {
    let Some(root) = std::env::var_os("APXINF_HUBERT_STAGE_CAPTURE") else {
        return;
    };
    let root = std::path::PathBuf::from(root).join(input_samples.to_string());
    std::fs::create_dir_all(&root).unwrap();
    ctx.synchronize().unwrap();
    let value = apxinf_cuda_new::transfers::to_cpu(value).unwrap();
    let bytes = value
        .to_f32_vec()
        .unwrap()
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    std::fs::write(root.join(format!("{name}.f32")), bytes).unwrap();
    std::fs::write(
        root.join(format!("{name}.json")),
        serde_json::to_vec(value.shape().dims()).unwrap(),
    )
    .unwrap();
}

impl Hubert {
    pub fn load(weights: &mut Weights, ctx: &Context) -> Result<Self> {
        let mut convs = Vec::new();
        let mut conv_bias = Vec::new();
        for index in 0..7 {
            convs.push(weights.get(ctx, &format!("hubert.feat.conv{index}.w"))?);
            conv_bias.push(weights.get_optional(ctx, &format!("hubert.feat.conv{index}.b"))?);
        }
        let group_w = weights.get(ctx, "hubert.feat.norm0.w")?;
        let group_b = weights.get(ctx, "hubert.feat.norm0.b")?;
        let proj_w = weights.get(ctx, "hubert.feat_proj.proj.w")?;
        let proj_b = weights.get_optional(ctx, "hubert.feat_proj.proj.b")?;
        let proj_ln_w = weights.get(ctx, "hubert.feat_proj.ln.w")?;
        let proj_ln_b = weights.get(ctx, "hubert.feat_proj.ln.b")?;
        let pos_w = weights.get(ctx, "hubert.posconv.w")?;
        let pos_b = weights.get_optional(ctx, "hubert.posconv.b")?;
        let enc_ln_w = weights.get(ctx, "hubert.enc_ln.w")?;
        let enc_ln_b = weights.get(ctx, "hubert.enc_ln.b")?;
        let mut layers = Vec::with_capacity(HUBERT_LAYERS);
        for index in 0..HUBERT_LAYERS {
            let prefix = format!("hubert.layers.{index}");
            layers.push(EncoderLayer {
                q: Linear::load(weights, ctx, &format!("{prefix}.q"))?,
                k: Linear::load(weights, ctx, &format!("{prefix}.k"))?,
                v: Linear::load(weights, ctx, &format!("{prefix}.v"))?,
                o: Linear::load(weights, ctx, &format!("{prefix}.o"))?,
                ln_w: weights.get(ctx, &format!("{prefix}.ln.w"))?,
                ln_b: weights.get(ctx, &format!("{prefix}.ln.b"))?,
                ff1: Linear::load(weights, ctx, &format!("{prefix}.ff1"))?,
                ff2: Linear::load(weights, ctx, &format!("{prefix}.ff2"))?,
                out_ln_w: weights.get(ctx, &format!("{prefix}.out_ln.w"))?,
                out_ln_b: weights.get(ctx, &format!("{prefix}.out_ln.b"))?,
            });
        }
        let ssl_proj_w = weights.get(ctx, "hubert.ssl_proj.w")?;
        let ssl_proj_b = weights.get_optional(ctx, "hubert.ssl_proj.b")?;

        // The quantizer codebook is stored [1024, 768]. Host-side transposition
        // and the column bias are load-time constants.
        let host_codebook = weights.host_clone("sovits.quant.codebook")?;
        let values = host_codebook.to_f32_vec()?;
        let bins = host_codebook.shape().dims()[0];
        let dim = host_codebook.shape().dims()[1];
        let mut transposed = vec![0.0f32; bins * dim];
        for bin in 0..bins {
            for channel in 0..dim {
                transposed[channel * bins + bin] = values[bin * dim + channel];
            }
        }
        let mut bias = vec![0.0f32; bins];
        for bin in 0..bins {
            let mut sum = 0.0f32;
            for channel in 0..dim {
                let value = values[bin * dim + channel];
                sum += value * value;
            }
            bias[bin] = -0.5 * sum;
        }
        let codebook = weights.to_device(ctx, &host_codebook)?;
        weights.put_device("sovits.quant.codebook", &codebook);
        let codebook_t = fp16_tensor(ctx, &[dim, bins], &transposed)?;
        let code_bias = fp16_tensor(ctx, &[bins], &bias)?;

        Ok(Self {
            convs,
            conv_bias,
            group_w,
            group_b,
            proj_w,
            proj_b,
            proj_ln_w,
            proj_ln_b,
            pos_w,
            pos_b,
            enc_ln_w,
            enc_ln_b,
            layers,
            ssl_proj_w,
            ssl_proj_b,
            codebook,
            codebook_t,
            code_bias,
        })
    }

    /// Encode 16 kHz mono PCM into the `[1, frames, 768]` pre-quantizer
    /// HuBERT representation used by the comparison harness.
    pub fn encode_features(&self, ctx: &Context, audio_16k: &[f32]) -> Result<Tensor> {
        if audio_16k.len() < 400 {
            return Err(Error::Other(
                "reference audio must contain at least 400 samples".into(),
            ));
        }
        let input = fp16_tensor(ctx, &[1, audio_16k.len(), 1], audio_16k)?;
        self.encode_features_device(ctx, &input)
    }

    pub(crate) fn semantic_frames(&self, samples: usize) -> Result<usize> {
        let mut frames = samples;
        for index in 0..7 {
            frames = frames
                .checked_sub(CONV_KERNEL[index])
                .ok_or_else(|| Error::Other("reference audio too short".into()))?
                / CONV_STRIDE[index]
                + 1;
        }
        Ok(frames
            .checked_sub(self.ssl_proj_w.shape().dims()[2])
            .ok_or_else(|| Error::Other("reference has no semantic frames".into()))?
            / 2
            + 1)
    }

    pub(crate) fn encode_features_device(&self, ctx: &Context, input: &Tensor) -> Result<Tensor> {
        let mut hidden = input.clone();
        // Quantizer decisions can change at sub-ULP feature differences. Keep
        // the validated convolution accumulation order for this frontend.
        for index in 0..7 {
            hidden = ops::conv1d(
                ctx,
                &hidden,
                &self.convs[index],
                self.conv_bias[index].as_ref(),
                CONV_STRIDE[index],
                0,
                1,
                1,
            )?;

            #[cfg(test)]
            capture_feature_stage(
                ctx,
                input.shape().dims()[1],
                &format!("conv{index}"),
                &hidden,
            );
            if index == 0 {
                hidden = ops::column_norm(ctx, &hidden, &self.group_w, &self.group_b)?;
                #[cfg(test)]
                capture_feature_stage(ctx, input.shape().dims()[1], "group_norm", &hidden);
            }
            hidden = ops::gelu_exact(ctx, &hidden)?;
            #[cfg(test)]
            capture_feature_stage(
                ctx,
                input.shape().dims()[1],
                &format!("gelu{index}"),
                &hidden,
            );
        }
        let dims = hidden.shape().dims().to_vec();
        let hidden = hidden.reshape(apxinf_core::Shape::new(vec![dims[1], HUBERT_CONV_DIM]))?;
        let normalized = ops::layer_norm(ctx, &hidden, &self.proj_ln_w, &self.proj_ln_b, 1e-5)?;

        #[cfg(test)]
        capture_feature_stage(ctx, input.shape().dims()[1], "feature_ln", &normalized);
        let projected =
            ops::linear_bias_accumulator(ctx, &normalized, &self.proj_w, self.proj_b.as_ref())?;
        #[cfg(test)]
        capture_feature_stage(ctx, input.shape().dims()[1], "feature_proj", &projected);
        let projected_3d =
            projected.reshape(apxinf_core::Shape::new(vec![1, dims[1], HUBERT_HIDDEN]))?;

        // Positional convolution (kernel 128, groups 16, same padding). The
        // even kernel produces one extra frame which HubertSamePadLayer trims.
        let positional = ops::conv1d_ordered(
            ctx,
            &projected_3d,
            &self.pos_w,
            self.pos_b.as_ref(),
            1,
            64,
            1,
            16,
        )?;

        #[cfg(test)]
        capture_feature_stage(ctx, input.shape().dims()[1], "posconv", &positional);
        let positional = ops::gelu_exact(ctx, &positional)?;
        #[cfg(test)]
        capture_feature_stage(ctx, input.shape().dims()[1], "posgelu", &positional);
        let positional_dims = positional.shape().dims().to_vec();
        let positional = if positional_dims[1] == dims[1] + 1 {
            ops::slice_rows(
                ctx,
                &positional.reshape(apxinf_core::Shape::new(vec![
                    positional_dims[1],
                    HUBERT_HIDDEN,
                ]))?,
                0,
                dims[1],
            )?
            .reshape(apxinf_core::Shape::new(vec![1, dims[1], HUBERT_HIDDEN]))?
        } else {
            positional
        };
        let mut hidden = ops::residual_add(ctx, &positional, &projected_3d)?;
        let len = hidden.shape().dims()[1];
        let flat = hidden.reshape(apxinf_core::Shape::new(vec![len, HUBERT_HIDDEN]))?;
        let flat = ops::layer_norm(ctx, &flat, &self.enc_ln_w, &self.enc_ln_b, 1e-5)?;
        hidden = flat.reshape(apxinf_core::Shape::new(vec![1, len, HUBERT_HIDDEN]))?;

        #[cfg(test)]
        capture_feature_stage(ctx, input.shape().dims()[1], "encoder_ln", &hidden);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let len = hidden.shape().dims()[1];
            let flat = hidden.reshape(apxinf_core::Shape::new(vec![len, HUBERT_HIDDEN]))?;
            let q = layer.q.apply(ctx, &flat)?;
            let k = layer.k.apply(ctx, &flat)?;
            let v = layer.v.apply(ctx, &flat)?;
            #[cfg(test)]
            capture_feature_stage(
                ctx,
                input.shape().dims()[1],
                &format!("layer{layer_index}_q"),
                &q,
            );
            #[cfg(test)]
            capture_feature_stage(
                ctx,
                input.shape().dims()[1],
                &format!("layer{layer_index}_k"),
                &k,
            );
            #[cfg(test)]
            capture_feature_stage(
                ctx,
                input.shape().dims()[1],
                &format!("layer{layer_index}_v"),
                &v,
            );
            let head_dim = HUBERT_HIDDEN / HUBERT_HEADS;
            let q = q.reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                HUBERT_HEADS,
                head_dim,
            ]))?;
            let k = k.reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                HUBERT_HEADS,
                head_dim,
            ]))?;
            let v = v.reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                HUBERT_HEADS,
                head_dim,
            ]))?;
            let attended = ops::attention(ctx, &q, &k, &v, HUBERT_HEADS)?;
            let attended = attended.reshape(apxinf_core::Shape::new(vec![len, HUBERT_HIDDEN]))?;

            #[cfg(test)]
            capture_feature_stage(
                ctx,
                input.shape().dims()[1],
                &format!("layer{layer_index}_attention"),
                &attended,
            );
            let projected = layer.o.apply(ctx, &attended)?;
            let summed = ops::residual_add(ctx, &projected, &flat)?;
            let normed = ops::layer_norm(ctx, &summed, &layer.ln_w, &layer.ln_b, 1e-5)?;
            let ff = layer.ff1.apply(ctx, &normed)?;
            let ff = ops::gelu_exact(ctx, &ff)?;
            let ff = layer.ff2.apply(ctx, &ff)?;
            let summed = ops::residual_add(ctx, &ff, &normed)?;
            let normed = ops::layer_norm(ctx, &summed, &layer.out_ln_w, &layer.out_ln_b, 1e-5)?;
            hidden = normed.reshape(apxinf_core::Shape::new(vec![1, len, HUBERT_HIDDEN]))?;
            #[cfg(test)]
            capture_feature_stage(
                ctx,
                input.shape().dims()[1],
                &format!("layer{layer_index}_out"),
                &hidden,
            );
        }
        Ok(hidden)
    }

    pub(crate) fn quantize(&self, ctx: &Context, hidden: &Tensor) -> Result<Tensor> {
        let len = hidden.shape().dims()[1];
        let flat = hidden.reshape(apxinf_core::Shape::new(vec![len, HUBERT_HIDDEN]))?;
        let flat_3d = flat.reshape(apxinf_core::Shape::new(vec![1, len, HUBERT_HIDDEN]))?;
        let ssl = ops::conv1d_ordered(
            ctx,
            &flat_3d,
            &self.ssl_proj_w,
            self.ssl_proj_b.as_ref(),
            2,
            0,
            1,
            1,
        )?;
        let frames = ssl.shape().dims()[1];
        let ssl = ssl.reshape(apxinf_core::Shape::new(vec![frames, HUBERT_HIDDEN]))?;
        let distances = ops::gemm(ctx, &ssl, &self.codebook_t, Some(&self.code_bias))?;
        {
            let indices = ctx.allocate_output(
                apxinf_core::Shape::new(vec![frames]),
                apxinf_core::DType::I32,
            )?;
            apxinf_cuda_new::ops::argmax_rows(ctx, &distances, &indices)?;
            return Ok(indices);
        }
    }

    /// Encode 16 kHz mono PCM into 25 Hz semantic tokens.
    pub fn encode(&self, ctx: &Context, audio_16k: &[f32]) -> Result<Tensor> {
        let features = self.encode_features(ctx, audio_16k)?;
        self.quantize(ctx, &features)
    }

    /// Codebook lookup used by the acoustic decoder.
    pub fn dequantize(&self, ctx: &Context, codes: &Tensor) -> Result<Tensor> {
        ops::embedding(ctx, &self.codebook, codes)
    }
}

#[cfg(test)]
mod capture_tests {
    use super::*;
    /// Teacher-forcing captures live next to the capture root; set
    /// `APXINF_GPT_TEACHER_FORCING` when they are somewhere else.
    fn teacher_forcing_dir(root: &std::path::Path) -> std::path::PathBuf {
        std::env::var_os("APXINF_GPT_TEACHER_FORCING")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                root.parent()
                    .unwrap()
                    .join("results/gpt-teacher-forcing-20261029")
            })
    }
    #[test]
    #[ignore = "requires APXINF_GPT_CAPTURE_ROOT and APXINF_GPT_CHECKPOINT"]
    fn quantizer_on_captured_features_matches_codes() {
        let root = std::path::PathBuf::from(std::env::var("APXINF_GPT_CAPTURE_ROOT").unwrap());
        let checkpoint = std::path::PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let ctx = Context::new(0).unwrap();
        let mut weights = Weights::load(&checkpoint).unwrap();
        let hubert = Hubert::load(&mut weights, &ctx).unwrap();
        for pair in ["pair1", "pair2"] {
            let capture = root.join(pair).join("raw");
            let manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(capture.join("manifest.json")).unwrap())
                    .unwrap();
            let entry = manifest["entries"]
                .as_array()
                .unwrap()
                .iter()
                .find(|e| e["name"] == "hubert_out.0")
                .unwrap();
            let dims = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            let values = std::fs::read(capture.join(entry["path"].as_str().unwrap()))
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let features = fp16_tensor(&ctx, &dims, &values).unwrap();
            let actual = hubert.quantize(&ctx, &features).unwrap();
            let actual = apxinf_cuda_new::transfers::to_cpu(&actual).unwrap();
            let expected =
                std::fs::read(teacher_forcing_dir(&root).join(format!("{pair}-prompt.i32")))
                    .unwrap()
                    .chunks_exact(4)
                    .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>();
            let mismatches = actual
                .as_i32()
                .unwrap()
                .iter()
                .zip(&expected)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, (a, b))| (i, *a, *b))
                .collect::<Vec<_>>();
            println!("{pair}: captured-feature quantizer mismatches={mismatches:?}");
            assert_eq!(actual.as_i32().unwrap(), expected.as_slice());
        }
    }
    #[test]
    #[ignore = "requires APXINF_GPT_CHECKPOINT, APXINF_HUBERT_AUDIO_DIR and APXINF_HUBERT_DUMP_DIR"]
    fn dump_audio_features_and_codes() {
        let checkpoint = std::path::PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let audio_dir = std::path::PathBuf::from(std::env::var("APXINF_HUBERT_AUDIO_DIR").unwrap());
        let dump_dir = std::path::PathBuf::from(std::env::var("APXINF_HUBERT_DUMP_DIR").unwrap());
        std::fs::create_dir_all(&dump_dir).unwrap();
        let ctx = Context::new(0).unwrap();
        let mut weights = Weights::load(&checkpoint).unwrap();
        let hubert = Hubert::load(&mut weights, &ctx).unwrap();
        for entry in std::fs::read_dir(&audio_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("f32") {
                continue;
            }
            let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
            let audio = std::fs::read(&path)
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let features = hubert.encode_features(&ctx, &audio).unwrap();
            let codes = hubert.quantize(&ctx, &features).unwrap();
            let features = apxinf_cuda_new::transfers::to_cpu(&features).unwrap();
            let feature_bytes = features
                .to_f32_vec()
                .unwrap()
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            std::fs::write(dump_dir.join(format!("{stem}.features.f32")), feature_bytes).unwrap();
            std::fs::write(
                dump_dir.join(format!("{stem}.features.json")),
                serde_json::to_vec(features.shape().dims()).unwrap(),
            )
            .unwrap();
            let codes = apxinf_cuda_new::transfers::to_cpu(&codes).unwrap();
            let code_bytes = codes
                .as_i32()
                .unwrap()
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            std::fs::write(dump_dir.join(format!("{stem}.codes.i32")), code_bytes).unwrap();
            println!(
                "{stem}: {} samples -> {} features x {}",
                audio.len(),
                features.shape().dims()[0],
                features.shape().dims()[1]
            );
        }
    }

    #[test]
    #[ignore = "requires APXINF_GPT_CAPTURE_ROOT and APXINF_GPT_CHECKPOINT"]
    fn codes_match_reference() {
        let root = std::path::PathBuf::from(std::env::var("APXINF_GPT_CAPTURE_ROOT").unwrap());
        let checkpoint = std::path::PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let ctx = Context::new(0).unwrap();
        let mut weights = Weights::load(&checkpoint).unwrap();
        let hubert = Hubert::load(&mut weights, &ctx).unwrap();
        let mut failures = Vec::new();
        for pair in ["pair1", "pair2"] {
            let audio = std::fs::read(root.join(pair).join("request/ref_audio.f32"))
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let expected =
                std::fs::read(teacher_forcing_dir(&root).join(format!("{pair}-prompt.i32")))
                    .unwrap()
                    .chunks_exact(4)
                    .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>();
            let features = hubert.encode_features(&ctx, &audio).unwrap();
            if let Ok(directory) = std::env::var("APXINF_HUBERT_FEATURE_CAPTURE") {
                let directory = std::path::PathBuf::from(directory);
                std::fs::create_dir_all(&directory).unwrap();
                let host = apxinf_cuda_new::transfers::to_cpu(&features).unwrap();
                let bytes = host
                    .to_f32_vec()
                    .unwrap()
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>();
                std::fs::write(directory.join(format!("{pair}-features.f32")), bytes).unwrap();
                std::fs::write(
                    directory.join(format!("{pair}-features.json")),
                    serde_json::to_vec(&features.shape().dims()).unwrap(),
                )
                .unwrap();
            }
            let actual = hubert.quantize(&ctx, &features).unwrap();
            let actual = apxinf_cuda_new::transfers::to_cpu(&actual).unwrap();
            let actual = actual.as_i32().unwrap();
            let mismatches = actual
                .iter()
                .zip(&expected)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, (a, b))| (i, *a, *b))
                .collect::<Vec<_>>();
            println!(
                "{pair} codes={} expected={} mismatches={mismatches:?}",
                actual.len(),
                expected.len()
            );
            assert_eq!(actual.len(), expected.len());
            if actual != expected.as_slice() {
                failures.push((pair, mismatches));
            }
        }
        assert!(failures.is_empty(), "HuBERT code mismatches: {failures:?}");
    }
}

#[cfg(test)]
mod stage_isolation_tests {
    use super::*;

    fn load_stage(root: &std::path::Path, samples: usize, name: &str) -> Vec<f32> {
        std::fs::read(root.join(samples.to_string()).join(format!("{name}.f32")))
            .unwrap()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn run_encoder_tail(
        hubert: &Hubert,
        ctx: &Context,
        mut hidden: Tensor,
        first_layer: usize,
    ) -> Tensor {
        for layer in hubert.layers.iter().skip(first_layer) {
            let len = hidden.shape().dims()[1];
            let flat = hidden
                .reshape(apxinf_core::Shape::new(vec![len, HUBERT_HIDDEN]))
                .unwrap();
            let q = layer
                .q
                .apply(ctx, &flat)
                .unwrap()
                .reshape(apxinf_core::Shape::new(vec![
                    1,
                    len,
                    HUBERT_HEADS,
                    HUBERT_HIDDEN / HUBERT_HEADS,
                ]))
                .unwrap();
            let k = layer
                .k
                .apply(ctx, &flat)
                .unwrap()
                .reshape(apxinf_core::Shape::new(vec![
                    1,
                    len,
                    HUBERT_HEADS,
                    HUBERT_HIDDEN / HUBERT_HEADS,
                ]))
                .unwrap();
            let v = layer
                .v
                .apply(ctx, &flat)
                .unwrap()
                .reshape(apxinf_core::Shape::new(vec![
                    1,
                    len,
                    HUBERT_HEADS,
                    HUBERT_HIDDEN / HUBERT_HEADS,
                ]))
                .unwrap();
            let attended = ops::attention(ctx, &q, &k, &v, HUBERT_HEADS)
                .unwrap()
                .reshape(apxinf_core::Shape::new(vec![len, HUBERT_HIDDEN]))
                .unwrap();
            let projected = layer.o.apply(ctx, &attended).unwrap();
            let summed = ops::residual_add(ctx, &projected, &flat).unwrap();
            let normed = ops::layer_norm(ctx, &summed, &layer.ln_w, &layer.ln_b, 1e-5).unwrap();
            let ff = layer.ff1.apply(ctx, &normed).unwrap();
            let ff = ops::gelu_exact(ctx, &ff).unwrap();
            let ff = layer.ff2.apply(ctx, &ff).unwrap();
            let summed = ops::residual_add(ctx, &ff, &normed).unwrap();
            let normed =
                ops::layer_norm(ctx, &summed, &layer.out_ln_w, &layer.out_ln_b, 1e-5).unwrap();
            hidden = normed
                .reshape(apxinf_core::Shape::new(vec![1, len, HUBERT_HIDDEN]))
                .unwrap();
        }
        hidden
    }

    fn stage_metrics(label: &str, actual: &Tensor, expected: &[f32]) {
        let actual = apxinf_cuda_new::transfers::to_cpu(actual)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        assert_eq!(actual.len(), expected.len(), "{label}");
        let mut mismatches = 0usize;
        let mut max_abs = 0.0f32;
        let mut signal = 0.0f64;
        let mut noise = 0.0f64;
        for (&a, &b) in actual.iter().zip(expected) {
            mismatches += usize::from(a != b);
            max_abs = max_abs.max((a - b).abs());
            signal += f64::from(b).powi(2);
            noise += f64::from(a - b).powi(2);
        }
        println!(
            "{label}: mismatches={mismatches} max_abs={max_abs} snr_db={}",
            10.0 * (signal / noise).log10()
        );
    }

    #[test]
    #[ignore = "requires APXINF_HUBERT_REFERENCE_STAGES and APXINF_GPT_CHECKPOINT"]
    fn official_layer0_through_native_layer1_stage_metrics() {
        let root =
            std::path::PathBuf::from(std::env::var("APXINF_HUBERT_REFERENCE_STAGES").unwrap());
        let checkpoint = std::path::PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let ctx = Context::new(0).unwrap();
        let mut weights = Weights::load(&checkpoint).unwrap();
        let hubert = Hubert::load(&mut weights, &ctx).unwrap();
        let samples = 106880usize;
        let values = load_stage(&root, samples, "layer0_out");
        let len = values.len() / HUBERT_HIDDEN;
        let hidden = fp16_tensor(&ctx, &[1, len, HUBERT_HIDDEN], &values).unwrap();
        let flat = hidden
            .reshape(apxinf_core::Shape::new(vec![len, HUBERT_HIDDEN]))
            .unwrap();
        let layer = &hubert.layers[1];
        let q = layer.q.apply(&ctx, &flat).unwrap();
        let k = layer.k.apply(&ctx, &flat).unwrap();
        let v = layer.v.apply(&ctx, &flat).unwrap();
        let q_acc = layer.q.apply_accumulator(&ctx, &flat).unwrap();
        let k_acc = layer.k.apply_accumulator(&ctx, &flat).unwrap();
        let v_acc = layer.v.apply_accumulator(&ctx, &flat).unwrap();
        stage_metrics(
            "layer1_q_accumulator",
            &q_acc,
            &load_stage(&root, samples, "layer1_q"),
        );
        stage_metrics(
            "layer1_k_accumulator",
            &k_acc,
            &load_stage(&root, samples, "layer1_k"),
        );
        stage_metrics(
            "layer1_v_accumulator",
            &v_acc,
            &load_stage(&root, samples, "layer1_v"),
        );
        stage_metrics("layer1_q", &q, &load_stage(&root, samples, "layer1_q"));
        stage_metrics("layer1_k", &k, &load_stage(&root, samples, "layer1_k"));
        stage_metrics("layer1_v", &v, &load_stage(&root, samples, "layer1_v"));
        let q4 = q
            .reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                HUBERT_HEADS,
                HUBERT_HIDDEN / HUBERT_HEADS,
            ]))
            .unwrap();
        let k4 = k
            .reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                HUBERT_HEADS,
                HUBERT_HIDDEN / HUBERT_HEADS,
            ]))
            .unwrap();
        let v4 = v
            .reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                HUBERT_HEADS,
                HUBERT_HIDDEN / HUBERT_HEADS,
            ]))
            .unwrap();
        let attended = ops::attention(&ctx, &q4, &k4, &v4, HUBERT_HEADS).unwrap();
        let attended2 = attended
            .reshape(apxinf_core::Shape::new(vec![len, HUBERT_HIDDEN]))
            .unwrap();
        stage_metrics(
            "layer1_attention",
            &attended2,
            &load_stage(&root, samples, "layer1_attention"),
        );
        let projected = layer.o.apply(&ctx, &attended2).unwrap();
        let summed = ops::residual_add(&ctx, &projected, &flat).unwrap();
        let normed = ops::layer_norm(&ctx, &summed, &layer.ln_w, &layer.ln_b, 1e-5).unwrap();
        let ff = layer.ff1.apply(&ctx, &normed).unwrap();
        let ff = ops::gelu_exact(&ctx, &ff).unwrap();
        let ff = layer.ff2.apply(&ctx, &ff).unwrap();
        let summed = ops::residual_add(&ctx, &ff, &normed).unwrap();
        let output =
            ops::layer_norm(&ctx, &summed, &layer.out_ln_w, &layer.out_ln_b, 1e-5).unwrap();
        stage_metrics(
            "layer1_out",
            &output,
            &load_stage(&root, samples, "layer1_out"),
        );
    }

    #[test]
    #[ignore = "requires APXINF_HUBERT_REFERENCE_STAGES and APXINF_GPT_CHECKPOINT"]
    fn official_encoder_stage_through_native_tail_reports_codes() {
        let root =
            std::path::PathBuf::from(std::env::var("APXINF_HUBERT_REFERENCE_STAGES").unwrap());
        let checkpoint = std::path::PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let expected_path =
            std::path::PathBuf::from(std::env::var("APXINF_HUBERT_EXPECTED_CODES").unwrap());
        let expected = std::fs::read(expected_path)
            .unwrap()
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let ctx = Context::new(0).unwrap();
        let mut weights = Weights::load(&checkpoint).unwrap();
        let hubert = Hubert::load(&mut weights, &ctx).unwrap();
        let samples = 106880usize;
        for stage_index in 0..12usize {
            let name = format!("layer{stage_index}_out");
            let values = load_stage(&root, samples, &name);
            let len = values.len() / HUBERT_HIDDEN;
            let input = fp16_tensor(&ctx, &[1, len, HUBERT_HIDDEN], &values).unwrap();
            let hidden = run_encoder_tail(&hubert, &ctx, input, stage_index + 1);
            let codes = hubert.quantize(&ctx, &hidden).unwrap();
            let codes = apxinf_cuda_new::transfers::to_cpu(&codes).unwrap();
            let codes = codes.as_i32().unwrap();
            let mismatches = codes
                .iter()
                .zip(&expected)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, (a, b))| (i, *a, *b))
                .collect::<Vec<_>>();
            println!("official {name} -> native tail mismatches={mismatches:?}");
        }
    }

    #[test]
    #[ignore = "requires APXINF_HUBERT_REFERENCE_STAGES and APXINF_GPT_CHECKPOINT"]
    fn official_gelu0_through_native_conv1_matches_reference() {
        let root =
            std::path::PathBuf::from(std::env::var("APXINF_HUBERT_REFERENCE_STAGES").unwrap());
        let checkpoint = std::path::PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let ctx = Context::new(0).unwrap();
        let mut weights = Weights::load(&checkpoint).unwrap();
        let hubert = Hubert::load(&mut weights, &ctx).unwrap();
        for samples in [94400usize, 106880] {
            let input_values = load_stage(&root, samples, "gelu0");
            let frames = input_values.len() / HUBERT_CONV_DIM;
            let input = fp16_tensor(&ctx, &[1, frames, HUBERT_CONV_DIM], &input_values).unwrap();
            let actual = ops::conv1d_ordered(
                &ctx,
                &input,
                &hubert.convs[1],
                hubert.conv_bias[1].as_ref(),
                CONV_STRIDE[1],
                0,
                1,
                1,
            )
            .unwrap();
            let actual = apxinf_cuda_new::transfers::to_cpu(&actual)
                .unwrap()
                .to_f32_vec()
                .unwrap();
            let expected = load_stage(&root, samples, "conv1");
            let mut mismatches = 0usize;
            let mut max_abs = 0.0f32;
            let mut signal = 0.0f64;
            let mut noise = 0.0f64;
            for (a, b) in actual.iter().zip(&expected) {
                mismatches += usize::from(a != b);
                max_abs = max_abs.max((a - b).abs());
                signal += f64::from(*b) * f64::from(*b);
                noise += f64::from(*a - *b) * f64::from(*a - *b);
            }
            let snr = 10.0 * (signal / noise).log10();
            println!("{samples}: conv1 official-input mismatches={mismatches} max_abs={max_abs} snr_db={snr}");
        }
    }
}

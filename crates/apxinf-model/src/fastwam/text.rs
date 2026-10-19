//! UMT5 encoder: bidirectional relative bias, rounded RMS normalization and
//! gated GELU. Zero padding is applied after encoding, as in Fast-WAM.
use super::backend::{self, ops, Context};
use super::weights::text::TextEncoder;
use apxinf_core::{DType, Result, Shape, Tensor};

fn rms(ctx: &Context, x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
    ops::rms_norm_rounded(ctx, ops::RmsNormArgs::new(x, w, &mut out, 1e-6))?;
    Ok(out)
}
impl TextEncoder {
    /// Monotonic graph arena: all layer outputs remain allocated until replay ends.
    pub(super) fn workspace_bytes(&self) -> usize {
        let tokens = 128;
        let hidden = self.embedding.shape().dims()[1];
        let layers: usize = self
            .blocks
            .iter()
            .map(|b| {
                let ff = b.wi0.weight.shape().dims()[1];
                // Ten hidden tensors, four FFN tensors and one F32 attention bias.
                tokens * (10 * hidden + 4 * ff) * 2 + 64 * tokens * tokens * 4 + 16 * 256
            })
            .sum();
        // Embedding, final RMS and row mask outputs.
        layers + 3 * (tokens * hidden * 2 + 256)
    }

    pub(super) fn forward(
        &self,
        ctx: &Context,
        ids: &Tensor,
        padding_bias: &Tensor,
        row_mask: &Tensor,
        gemm: &ops::GemmPolicy,
        attention: &ops::AttentionPolicy,
    ) -> Result<Tensor> {
        self.forward_observed(
            ctx,
            ids,
            padding_bias,
            row_mask,
            gemm,
            attention,
            |_, _, _| Ok(()),
        )
    }

    fn forward_observed(
        &self,
        ctx: &Context,
        ids: &Tensor,
        padding_bias: &Tensor,
        row_mask: &Tensor,
        gemm: &ops::GemmPolicy,
        attention: &ops::AttentionPolicy,
        mut observe: impl FnMut(usize, &str, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let mut x = ctx.allocate_output(Shape::new(vec![128, 4096]), DType::BF16)?;
        ops::embedding_gather(ctx, &self.embedding, ids, &x)?;
        for (index, b) in self.blocks.iter().enumerate() {
            observe(index, "layer.0.layer_norm.in", &x)?;
            let u = rms(ctx, &x, &b.norm0)?;
            observe(index, "layer.0.layer_norm.out", &u)?;
            let q = b.q.forward(ctx, &u, gemm)?.reshape(vec![1, 128, 64, 64])?;
            let k = b.k.forward(ctx, &u, gemm)?.reshape(vec![1, 128, 64, 64])?;
            let v = b.v.forward(ctx, &u, gemm)?.reshape(vec![1, 128, 64, 64])?;
            observe(index, "layer.0.SelfAttention.q.out", &q)?;
            observe(index, "layer.0.SelfAttention.k.out", &k)?;
            observe(index, "layer.0.SelfAttention.v.out", &v)?;
            let bias = backend::binary(
                ctx,
                &b.bias,
                padding_bias,
                ops::PointwiseSemantic::BroadcastAdd,
            )?
            .reshape(vec![1, 64, 128, 128])?;
            let mut y = ctx.allocate_output(q.shape().clone(), DType::BF16)?;
            let mut args = ops::AttentionArgs::new(&q, &k, &v, &mut y);
            args.scale = 1.;
            args.policy = attention.clone();
            ops::additive_attention(
                ctx,
                ops::AttentionBiasArgs {
                    attention: args,
                    bias: &bias,
                    precision: ops::AttentionPrecision::Input,
                },
            )?;
            observe(index, "layer.0.SelfAttention.o.in", &y)?;
            let y = b.o.forward(ctx, &y.reshape(vec![128, 4096])?, gemm)?;
            observe(index, "layer.0.SelfAttention.o.out", &y)?;
            x = backend::add(ctx, &x, &y)?;
            observe(index, "layer.1.layer_norm.in", &x)?;
            let u = rms(ctx, &x, &b.norm1)?;
            observe(index, "layer.1.layer_norm.out", &u)?;
            let gate = b.wi0.forward(ctx, &u, gemm)?;
            observe(index, "layer.1.DenseReluDense.act.in", &gate)?;
            let gate = backend::activation(ctx, &gate, ops::PointwiseActivation::GeluRounded)?;
            observe(index, "layer.1.DenseReluDense.act.out", &gate)?;
            let up = b.wi1.forward(ctx, &u, gemm)?;
            observe(index, "layer.1.DenseReluDense.wi_1.out", &up)?;
            let y = backend::binary(ctx, &gate, &up, ops::PointwiseSemantic::Multiply)?;
            observe(index, "layer.1.DenseReluDense.wo.in", &y)?;
            let y = b.wo.forward(ctx, &y, gemm)?;
            observe(index, "layer.1.DenseReluDense.wo.out", &y)?;
            x = backend::add(ctx, &x, &y)?;
        }
        observe(self.blocks.len(), "final_layer_norm.in", &x)?;
        let x = rms(ctx, &x, &self.norm)?;
        observe(self.blocks.len(), "final_layer_norm.out", &x)?;
        backend::binary(ctx, &x, row_mask, ops::PointwiseSemantic::RowMultiply)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastwam::backend::host_tensor;
    use std::collections::HashMap;

    // Use a captured input for each operation so an earlier approximation cannot
    // hide which operation first diverges. These diagnostics do not relax or
    // replace the end-to-end assertion below.
    fn captured_input(ctx: &Context, golden: &HashMap<String, Tensor>, key: &str) -> Tensor {
        let value = &golden[key];
        assert_eq!(value.shape().numel() % 128, 0, "{key}");
        backend::upload(ctx, value)
            .unwrap()
            .reshape(vec![128, value.shape().numel() / 128])
            .unwrap()
    }

    fn print_operator_error(name: &str, actual: &Tensor, expected: &Tensor) {
        let actual = backend::transfers::to_cpu(actual)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let expected = expected.to_f32_vec().unwrap();
        assert_eq!(actual.len(), expected.len(), "{name}");
        let mut square = 0f64;
        let mut norm = 0f64;
        let mut max = 0f32;
        for (&a, &b) in actual.iter().zip(&expected) {
            assert!(a.is_finite() && b.is_finite(), "nonfinite {name}");
            square += (f64::from(a) - f64::from(b)).powi(2);
            norm += f64::from(b).powi(2);
            max = max.max((a - b).abs());
        }
        let relative = (square / norm.max(1e-30)).sqrt();
        eprintln!("UMT5 independent {name}: max_abs={max} relative_l2={relative}");
    }

    fn replay_captured_operators(
        ctx: &Context,
        model: &TextEncoder,
        golden: &HashMap<String, Tensor>,
        padding: &Tensor,
        gemm: &ops::GemmPolicy,
        attention: &ops::AttentionPolicy,
    ) {
        for index in [0, 23] {
            let block = &model.blocks[index];
            let prefix = format!("text.model.encoder.block.{index}");
            let attn_prefix = format!("{prefix}.layer.0.SelfAttention");
            let ff_prefix = format!("{prefix}.layer.1.DenseReluDense");
            for (layer, weight) in [(0, &block.norm0), (1, &block.norm1)] {
                let name = format!("{prefix}.layer.{layer}.layer_norm");
                let input = captured_input(ctx, golden, &format!("{name}.in"));
                let actual = rms(ctx, &input, weight).unwrap();
                print_operator_error(&name, &actual, &golden[&format!("{name}.out")]);
            }
            for (name, linear) in [
                (format!("{attn_prefix}.q"), &block.q),
                (format!("{attn_prefix}.k"), &block.k),
                (format!("{attn_prefix}.v"), &block.v),
                (format!("{attn_prefix}.o"), &block.o),
                (format!("{ff_prefix}.wi_0"), &block.wi0),
                (format!("{ff_prefix}.wi_1"), &block.wi1),
                (format!("{ff_prefix}.wo"), &block.wo),
            ] {
                let input = captured_input(ctx, golden, &format!("{name}.in"));
                let actual = linear.forward(ctx, &input, gemm).unwrap();
                print_operator_error(&name, &actual, &golden[&format!("{name}.out")]);
            }
            let name = format!("{ff_prefix}.act");
            let input = captured_input(ctx, golden, &format!("{name}.in"));
            let actual =
                backend::activation(ctx, &input, ops::PointwiseActivation::GeluRounded).unwrap();
            print_operator_error(&name, &actual, &golden[&format!("{name}.out")]);

            let gate = captured_input(ctx, golden, &format!("{name}.out"));
            let up = captured_input(ctx, golden, &format!("{ff_prefix}.wi_1.out"));
            let actual =
                backend::binary(ctx, &gate, &up, ops::PointwiseSemantic::Multiply).unwrap();
            print_operator_error(
                &format!("{ff_prefix}.gated_multiply"),
                &actual,
                &golden[&format!("{ff_prefix}.wo.in")],
            );

            let qkv = ["q", "k", "v"].map(|projection| {
                captured_input(ctx, golden, &format!("{attn_prefix}.{projection}.out"))
                    .reshape(vec![1, 128, 64, 64])
                    .unwrap()
            });
            let bias = backend::binary(
                ctx,
                &block.bias,
                padding,
                ops::PointwiseSemantic::BroadcastAdd,
            )
            .unwrap()
            .reshape(vec![1, 64, 128, 128])
            .unwrap();
            let mut actual = ctx
                .allocate_output(qkv[0].shape().clone(), DType::BF16)
                .unwrap();
            let mut args = ops::AttentionArgs::new(&qkv[0], &qkv[1], &qkv[2], &mut actual);
            args.scale = 1.;
            args.policy = attention.clone();
            ops::additive_attention(
                ctx,
                ops::AttentionBiasArgs {
                    attention: args,
                    bias: &bias,
                    precision: ops::AttentionPrecision::Input,
                },
            )
            .unwrap();
            print_operator_error(
                &format!("{attn_prefix}.attention"),
                &actual,
                &golden[&format!("{attn_prefix}.o.in")],
            );
        }
    }

    #[test]
    #[ignore = "requires UMT5 checkpoint, pinned Fast-WAM capture and CUDA"]
    fn fastwam_text_checkpoint_precision() {
        let path = std::path::PathBuf::from(std::env::var("APXINF_FASTWAM_TEXT").unwrap());
        let fixture = std::path::PathBuf::from(std::env::var("APXINF_FASTWAM_CAPTURE").unwrap());
        let (inputs, _) =
            apxinf_loader::safetensors::load_native_path(&fixture.join("inputs.safetensors"))
                .unwrap();
        let (golden, _) =
            apxinf_loader::safetensors::load_native_path(&fixture.join("text.safetensors"))
                .unwrap();
        let ctx = Context::new(0).unwrap();
        let model = TextEncoder::load(&ctx, &path).unwrap();
        let ids = Tensor::from_i32(
            vec![128],
            &inputs["inputs.ids"]
                .to_f32_vec()
                .unwrap()
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let ids = backend::upload(&ctx, &ids).unwrap();
        let mask = inputs["inputs.mask"].to_f32_vec().unwrap();
        let padding = backend::upload(
            &ctx,
            &Tensor::from_f32(
                vec![128],
                &mask
                    .iter()
                    .map(|&x| if x == 0. { -3.3895314e38 } else { 0. })
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        )
        .unwrap();
        let rowmask =
            backend::upload(&ctx, &host_tensor(vec![128], &mask, DType::BF16).unwrap()).unwrap();
        let mut gemm = ops::GemmPolicy::default();
        gemm.online_tune = false;
        gemm.cache_dir = Some(String::new());
        let mut attention = ops::AttentionPolicy::default();
        attention.online_tune = false;
        attention.cache_dir = Some(String::new());
        replay_captured_operators(&ctx, &model, &golden, &padding, &gemm, &attention);
        let actual = model
            .forward_observed(
                &ctx,
                &ids,
                &padding,
                &rowmask,
                &gemm,
                &attention,
                |index, suffix, tensor| {
                    let key = if index == 24 {
                        format!("text.model.encoder.{suffix}")
                    } else {
                        format!("text.model.encoder.block.{index}.{suffix}")
                    };
                    if let Some(expected) = golden.get(&key) {
                        print_operator_error(&format!("chain {key}"), tensor, expected);
                    }
                    Ok(())
                },
            )
            .unwrap();
        let actual = backend::transfers::to_cpu(&actual)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let mut expected = golden["text.model.encoder.final_layer_norm.out"]
            .to_f32_vec()
            .unwrap();
        for (row, &keep) in expected.chunks_exact_mut(4096).zip(&mask) {
            if keep == 0. {
                row.fill(0.);
            }
        }
        let mut square = 0f64;
        let mut norm = 0f64;
        let mut max = 0f32;
        for (&a, &b) in actual.iter().zip(&expected) {
            assert!(a.is_finite());
            square += ((a - b) as f64).powi(2);
            norm += (b as f64).powi(2);
            max = max.max((a - b).abs());
        }
        let relative = (square / norm).sqrt();
        eprintln!("UMT5 final max_abs={max} relative_l2={relative}");
        assert!(relative < 0.01, "UMT5 native/reference mismatch");
    }
}

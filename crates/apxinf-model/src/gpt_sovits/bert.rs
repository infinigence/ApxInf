//! RoBERTa-large encoder used for GPT conditioning.

use apxinf_core::{Error, Result, Tensor};
use apxinf_cuda_new::CudaContext as Context;

use super::config::{BERT_HEADS, BERT_HIDDEN, BERT_LAYERS, BERT_OUTPUT_LAYER};
use super::ops;
use super::weights::{ids_tensor, Linear, Weights};

struct Layer {
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

pub struct Bert {
    word_emb: Tensor,
    pos_emb: Tensor,
    type_emb: Tensor,
    emb_ln_w: Tensor,
    emb_ln_b: Tensor,
    layers: Vec<Layer>,
    max_positions: usize,
}

impl Bert {
    pub fn load(weights: &mut Weights, ctx: &Context) -> Result<Self> {
        let word_emb = weights.get(ctx, "roberta.word_emb")?;
        let pos_emb = weights.get(ctx, "roberta.pos_emb")?;
        let type_emb = weights.get(ctx, "roberta.type_emb")?;
        let (emb_ln_w, emb_ln_b) = (
            weights.get(ctx, "roberta.emb_ln.w")?,
            weights.get(ctx, "roberta.emb_ln.b")?,
        );
        let max_positions = pos_emb.shape().dims()[0];
        let mut layers = Vec::with_capacity(BERT_LAYERS);
        for index in 0..BERT_LAYERS {
            let prefix = format!("roberta.layers.{index}");
            layers.push(Layer {
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
        Ok(Self {
            word_emb,
            pos_emb,
            type_emb,
            emb_ln_w,
            emb_ln_b,
            layers,
            max_positions,
        })
    }

    /// Returns the token-level hidden state after `BERT_OUTPUT_LAYER`, with the
    /// leading `[CLS]` and trailing `[SEP]` tokens removed.
    pub fn forward(&self, ctx: &Context, token_ids: &[u32]) -> Result<Tensor> {
        if token_ids.len() < 3 {
            return Err(Error::Other("BERT input requires [CLS] ... [SEP]".into()));
        }
        if token_ids.len() > self.max_positions {
            return Err(Error::Other(format!(
                "BERT input length {} exceeds {} positions",
                token_ids.len(),
                self.max_positions
            )));
        }
        let len = token_ids.len();
        let ids = ids_tensor(ctx, token_ids)?;
        let position_ids = ids_tensor(ctx, &(0..len as u32).collect::<Vec<_>>())?;
        let type_ids = ids_tensor(ctx, &vec![0u32; len])?;
        self.forward_device(ctx, &ids, &position_ids, &type_ids)
    }

    pub(crate) fn forward_device(
        &self,
        ctx: &Context,
        ids: &Tensor,
        position_ids: &Tensor,
        type_ids: &Tensor,
    ) -> Result<Tensor> {
        let len = ids.numel();
        if len < 3
            || len > self.max_positions
            || position_ids.numel() != len
            || type_ids.numel() != len
        {
            return Err(Error::Other("invalid prepared BERT input length".into()));
        }
        #[cfg(test)]
        macro_rules! dump {
            ($label:expr, $tensor:expr) => {
                if let Some(directory) = std::env::var_os("APXINF_BERT_STAGE_CAPTURE") {
                    let directory =
                        std::path::PathBuf::from(directory).join(format!("tokens-{len}"));
                    std::fs::create_dir_all(&directory).unwrap();
                    let values = apxinf_cuda_new::transfers::to_cpu($tensor)
                        .unwrap()
                        .to_f32_vec()
                        .unwrap();
                    std::fs::write(
                        directory.join(format!("{}.f32", $label)),
                        values
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                }
            };
        }
        let word = ops::embedding(ctx, &self.word_emb, &ids)
            .map_err(|e| Error::Other(format!("bert word: {e}")))?;
        let position = ops::embedding(ctx, &self.pos_emb, &position_ids)
            .map_err(|e| Error::Other(format!("bert position: {e}")))?;
        let type_embedding = ops::embedding(ctx, &self.type_emb, &type_ids)
            .map_err(|e| Error::Other(format!("bert type: {e}")))?;
        #[cfg(test)]
        dump!("word", &word);
        #[cfg(test)]
        dump!("type", &type_embedding);
        #[cfg(test)]
        dump!("position", &position);
        // Match BertEmbeddings: word + token_type, then + position.
        // Each addition rounds to FP16, so their order affects conditioning.
        let mut hidden = ops::residual_add(ctx, &word, &type_embedding)
            .map_err(|e| Error::Other(format!("bert word+type: {e}")))?;
        #[cfg(test)]
        dump!("word_type", &hidden);
        hidden = ops::residual_add(ctx, &hidden, &position)
            .map_err(|e| Error::Other(format!("bert +pos: {e}")))?;
        #[cfg(test)]
        dump!("embedding_sum", &hidden);
        hidden = ops::layer_norm(ctx, &hidden, &self.emb_ln_w, &self.emb_ln_b, 1e-12)
            .map_err(|e| Error::Other(format!("bert emb norm: {e}")))?;

        #[cfg(test)]
        dump!("embedding_norm", &hidden);
        for (layer_index, layer) in self.layers.iter().take(BERT_OUTPUT_LAYER + 1).enumerate() {
            let stage = move |name: &str, error: apxinf_core::Error| {
                apxinf_core::Error::Other(format!("bert layer {layer_index} {name}: {error}"))
            };
            let q = layer
                .q
                .apply_accumulator(ctx, &hidden)
                .map_err(|e| stage("q", e))?;
            let k = layer
                .k
                .apply_accumulator(ctx, &hidden)
                .map_err(|e| stage("k", e))?;
            let v = layer
                .v
                .apply_accumulator(ctx, &hidden)
                .map_err(|e| stage("v", e))?;
            #[cfg(test)]
            if layer_index == 0 {
                dump!("layer0.q", &q);
                dump!("layer0.k", &k);
                dump!("layer0.v", &v);
            }
            let q = q.reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                BERT_HEADS,
                BERT_HIDDEN / BERT_HEADS,
            ]))?;
            let k = k.reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                BERT_HEADS,
                BERT_HIDDEN / BERT_HEADS,
            ]))?;
            let v = v.reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                BERT_HEADS,
                BERT_HIDDEN / BERT_HEADS,
            ]))?;
            let attended =
                ops::attention(ctx, &q, &k, &v, BERT_HEADS).map_err(|e| stage("attention", e))?;
            let attended = attended.reshape(apxinf_core::Shape::new(vec![len, BERT_HIDDEN]))?;
            #[cfg(test)]
            if layer_index == 0 {
                dump!("layer0.context", &attended);
            }
            let projected = layer
                .o
                .apply_accumulator(ctx, &attended)
                .map_err(|e| stage("o", e))?;
            #[cfg(test)]
            if layer_index == 0 {
                dump!("layer0.projected", &projected);
            }
            let summed = ops::residual_add(ctx, &projected, &hidden)
                .map_err(|e| stage("attn residual", e))?;
            hidden = ops::layer_norm(ctx, &summed, &layer.ln_w, &layer.ln_b, 1e-12)
                .map_err(|e| stage("attn norm", e))?;

            #[cfg(test)]
            if layer_index == 0 {
                dump!("layer0.attention_norm", &hidden);
            }
            let ff = layer
                .ff1
                .apply_accumulator(ctx, &hidden)
                .map_err(|e| stage("ff1", e))?;
            #[cfg(test)]
            if layer_index == 0 {
                dump!("layer0.ff1", &ff);
            }
            let ff = ops::gelu_exact(ctx, &ff).map_err(|e| stage("gelu", e))?;
            #[cfg(test)]
            if layer_index == 0 {
                dump!("layer0.gelu", &ff);
            }
            let ff = layer
                .ff2
                .apply_accumulator(ctx, &ff)
                .map_err(|e| stage("ff2", e))?;
            #[cfg(test)]
            if layer_index == 0 {
                dump!("layer0.ff2", &ff);
            }
            let summed =
                ops::residual_add(ctx, &ff, &hidden).map_err(|e| stage("ff residual", e))?;
            hidden = ops::layer_norm(ctx, &summed, &layer.out_ln_w, &layer.out_ln_b, 1e-12)
                .map_err(|e| stage("ff norm", e))?;
            #[cfg(test)]
            dump!(format!("layer{layer_index}.output"), &hidden);
        }
        ops::slice_rows(ctx, &hidden, 1, len - 2)
    }
}

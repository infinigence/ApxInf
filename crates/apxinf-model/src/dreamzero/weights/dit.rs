//! Joint video/action transformer weights. Sampler profiles share this network.
use super::super::backend::{transfers, Context};
use apxinf_core::{DType, Error, Result, Tensor};
use std::collections::HashMap;

pub(in crate::dreamzero) const DIM: usize = 5120;
pub(in crate::dreamzero) const FFN: usize = 13824;
pub(in crate::dreamzero) const HEADS: usize = 40;

pub(in crate::dreamzero) struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}
pub(in crate::dreamzero) struct Norm {
    pub weight: Tensor,
    pub bias: Tensor,
}
pub(in crate::dreamzero) struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub norm_q: Tensor,
    pub norm_k: Tensor,
}
pub(in crate::dreamzero) struct Block {
    pub modulation: Tensor,
    pub self_attention: Attention,
    pub cross_attention: Attention,
    pub image_k: Linear,
    pub image_v: Linear,
    pub image_norm_k: Tensor,
    pub norm3: Norm,
    pub up: Linear,
    pub down: Linear,
}
pub(in crate::dreamzero) struct DitWeights {
    pub state: [Linear; 2],
    pub action: [Linear; 3],
    pub action_decoder: [Linear; 2],
    pub patch: Linear,
    pub text: [Linear; 2],
    pub time: [Linear; 2],
    pub time_projection: Linear,
    pub image_norm1: Norm,
    pub image: [Linear; 2],
    pub image_norm2: Norm,
    pub blocks: Vec<Block>,
    pub head_modulation: Tensor,
    pub head: Linear,
}

struct Reader<'a> {
    ctx: &'a Context,
    map: &'a mut HashMap<String, Tensor>,
}
impl Reader<'_> {
    fn take(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let key = format!("action_head.model.{name}");
        let tensor = self
            .map
            .remove(&key)
            .ok_or_else(|| Error::Other(format!("missing DreamZero {key}")))?;
        if tensor.dtype() != DType::BF16 || tensor.shape().dims() != shape {
            return Err(Error::Other(format!(
                "DreamZero {key}: expected BF16 {shape:?}, got {} {:?}",
                tensor.dtype(),
                tensor.shape().dims()
            )));
        }
        Ok(tensor)
    }
    fn upload(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let x = self.take(name, shape)?;
        transfers::to_cuda(&x, self.ctx.device_id())
    }
    fn linear(&mut self, name: &str, input: usize, output: usize) -> Result<Linear> {
        let source = self.take(&format!("{name}.weight"), &[output, input])?;
        Ok(Linear {
            weight: transfers::to_cuda(&source, self.ctx.device_id())?,
            bias: self.upload(&format!("{name}.bias"), &[output])?,
        })
    }
    fn category(&mut self, name: &str, input: usize, output: usize) -> Result<Linear> {
        // The released DROID constructor fixes category count to one. These
        // matrices already use IO order; they are not nn.Linear OI weights.
        Ok(Linear {
            weight: self
                .upload(&format!("{name}.W"), &[1, input, output])?
                .reshape(vec![input, output])?,
            bias: self
                .upload(&format!("{name}.b"), &[1, output])?
                .reshape(vec![output])?,
        })
    }
    fn norm(&mut self, name: &str, dim: usize) -> Result<Norm> {
        Ok(Norm {
            weight: self.upload(&format!("{name}.weight"), &[dim])?,
            bias: self.upload(&format!("{name}.bias"), &[dim])?,
        })
    }
    fn attention(&mut self, name: &str) -> Result<Attention> {
        Ok(Attention {
            q: self.linear(&format!("{name}.q"), DIM, DIM)?,
            k: self.linear(&format!("{name}.k"), DIM, DIM)?,
            v: self.linear(&format!("{name}.v"), DIM, DIM)?,
            o: self.linear(&format!("{name}.o"), DIM, DIM)?,
            norm_q: self.upload(&format!("{name}.norm_q.weight"), &[DIM])?,
            norm_k: self.upload(&format!("{name}.norm_k.weight"), &[DIM])?,
        })
    }
    fn block(&mut self, index: usize) -> Result<Block> {
        let p = format!("blocks.{index}");
        Ok(Block {
            modulation: self
                .upload(&format!("{p}.modulation"), &[1, 6, DIM])?
                .reshape(vec![6, DIM])?,
            self_attention: self.attention(&format!("{p}.self_attn"))?,
            cross_attention: self.attention(&format!("{p}.cross_attn"))?,
            image_k: self.linear(&format!("{p}.cross_attn.k_img"), DIM, DIM)?,
            image_v: self.linear(&format!("{p}.cross_attn.v_img"), DIM, DIM)?,
            image_norm_k: self.upload(&format!("{p}.cross_attn.norm_k_img.weight"), &[DIM])?,
            norm3: self.norm(&format!("{p}.norm3"), DIM)?,
            up: self.linear(&format!("{p}.ffn.0"), DIM, FFN)?,
            down: self.linear(&format!("{p}.ffn.2"), FFN, DIM)?,
        })
    }
}
impl DitWeights {
    pub fn from_map(ctx: &Context, map: &mut HashMap<String, Tensor>) -> Result<Self> {
        let mut r = Reader { ctx, map };
        let kernel = r.take("patch_embedding.weight", &[DIM, 36, 1, 2, 2])?;
        let src = kernel.as_bf16()?;
        let mut values = vec![half::bf16::ZERO; 4 * 48 * DIM];
        // Unfold columns THWC, checkpoint OITHW. Pad channels to 48 so
        // tensor-core reduction grouping matches the reference convolution.
        for o in 0..DIM {
            for c in 0..36 {
                for xy in 0..4 {
                    values[(xy * 48 + c) * DIM + o] = src[(o * 36 + c) * 4 + xy];
                }
            }
        }
        let patch = Linear {
            weight: transfers::to_cuda(
                &Tensor::from_bf16(vec![4 * 48, DIM], &values)?,
                ctx.device_id(),
            )?,
            bias: r.upload("patch_embedding.bias", &[DIM])?,
        };
        let mut blocks = Vec::with_capacity(40);
        for i in 0..40 {
            blocks.push(r.block(i)?);
        }
        Ok(Self {
            state: [
                r.category("state_encoder.layer1", 64, 1024)?,
                r.category("state_encoder.layer2", 1024, DIM)?,
            ],
            action: [
                r.category("action_encoder.W1", 32, DIM)?,
                r.category("action_encoder.W2", 2 * DIM, DIM)?,
                r.category("action_encoder.W3", DIM, DIM)?,
            ],
            action_decoder: [
                r.category("action_decoder.layer1", DIM, 1024)?,
                r.category("action_decoder.layer2", 1024, 32)?,
            ],
            patch,
            text: [
                r.linear("text_embedding.0", 4096, DIM)?,
                r.linear("text_embedding.2", DIM, DIM)?,
            ],
            time: [
                r.linear("time_embedding.0", 256, DIM)?,
                r.linear("time_embedding.2", DIM, DIM)?,
            ],
            time_projection: r.linear("time_projection.1", DIM, 6 * DIM)?,
            image_norm1: r.norm("img_emb.proj.0", 1280)?,
            image: [
                r.linear("img_emb.proj.1", 1280, 1280)?,
                r.linear("img_emb.proj.3", 1280, DIM)?,
            ],
            image_norm2: r.norm("img_emb.proj.4", DIM)?,
            blocks,
            head_modulation: r
                .upload("head.modulation", &[1, 2, DIM])?
                .reshape(vec![2, DIM])?,
            head: r.linear("head.head", DIM, 64)?,
        })
    }
}

#[cfg(test)]
impl Block {
    pub(in crate::dreamzero) fn from_map(
        ctx: &Context,
        map: &mut HashMap<String, Tensor>,
        index: usize,
    ) -> Result<Self> {
        Reader { ctx, map }.block(index)
    }
}

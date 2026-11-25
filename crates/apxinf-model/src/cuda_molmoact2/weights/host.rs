//! Host-side checkpoint reading: one shard at a time, strict names/shapes/dtypes, FP32 -> BF16
//! with round-to-nearest-even (the conversion the official BF16 load performs).
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use half::bf16;
use std::{collections::BTreeMap, path::Path};

fn bad(what: impl std::fmt::Display) -> Error {
    Error::Other(format!("MolmoAct2 weights: {what}"))
}

/// Tensors of the checkpoint this family does not consume (the language-model head: the expert
/// path never decodes tokens).
pub(super) fn is_skipped(name: &str) -> bool {
    name.starts_with("lm_head.")
}

/// Convert a checkpoint tensor (F32 or BF16) to a BF16 host tensor, rejecting non-finite values.
pub(super) fn to_bf16(name: &str, tensor: &Tensor) -> Result<Tensor> {
    let values: Vec<bf16> = match tensor.dtype() {
        DType::BF16 => tensor.as_bf16()?.to_vec(),
        DType::F32 => tensor.as_f32()?.iter().map(|&x| bf16::from_f32(x)).collect(),
        other => return Err(bad(format!("{name}: unsupported checkpoint dtype {other}"))),
    };
    if values.iter().any(|v| !v.to_f32().is_finite()) {
        return Err(bad(format!("{name}: non-finite value")));
    }
    Tensor::from_bf16(tensor.shape().clone(), &values)
}

/// `[rows, cols]` -> `[cols, rows]`, bit-exact. Split across threads by output rows because the
/// checkpoint holds billions of weights and loading must not be single-threaded.
pub(super) fn transpose(tensor: &Tensor) -> Result<Tensor> {
    let [rows, cols] = *tensor.shape().dims() else {
        return Err(bad("transpose needs a rank-2 tensor"));
    };
    let src = tensor.as_bf16()?;
    let mut out = vec![bf16::ZERO; src.len()];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 8);
    let per = cols.div_ceil(threads);
    std::thread::scope(|s| {
        for (t, chunk) in out.chunks_mut(per * rows).enumerate() {
            s.spawn(move || {
                let first = t * per;
                for (local, row) in chunk.chunks_mut(rows).enumerate() {
                    let c = first + local;
                    for (r, slot) in row.iter_mut().enumerate() {
                        *slot = src[r * cols + c];
                    }
                }
            });
        }
    });
    Tensor::from_bf16(vec![cols, rows], &out)
}

/// Re-order the input axis of a `[out, patch*patch*3]` patch-embedding weight from the
/// processor's per-patch order `(dy, dx, c)` to the order the GPU patchify kernel emits,
/// `(c, dy, dx)`. A pure permutation of K: every product is unchanged.
pub(super) fn patch_columns_to_channel_major(tensor: &Tensor, patch: usize) -> Result<Tensor> {
    let [rows, cols] = *tensor.shape().dims() else {
        return Err(bad("patch weight needs a rank-2 tensor"));
    };
    if cols != patch * patch * 3 {
        return Err(bad("patch weight width does not match the patch size"));
    }
    let src = tensor.as_bf16()?;
    let mut out = vec![bf16::ZERO; src.len()];
    for r in 0..rows {
        for dy in 0..patch {
            for dx in 0..patch {
                for c in 0..3 {
                    out[r * cols + c * patch * patch + dy * patch + dx] = src[r * cols + (dy * patch + dx) * 3 + c];
                }
            }
        }
    }
    Tensor::from_bf16(vec![rows, cols], &out)
}

/// Rows `[start, end)` of a rank-2 BF16 host tensor (contiguous, so a plain slice).
pub(super) fn row_slice(tensor: &Tensor, start: usize, end: usize) -> Result<Tensor> {
    let [rows, cols] = *tensor.shape().dims() else {
        return Err(bad("row slice needs a rank-2 tensor"));
    };
    if start >= end || end > rows {
        return Err(bad("row slice out of range"));
    }
    Tensor::from_bf16(vec![end - start, cols], &tensor.as_bf16()?[start * cols..end * cols])
}

/// Elements `[start, end)` of a rank-1 BF16 host tensor.
pub(super) fn slice_1d(tensor: &Tensor, start: usize, end: usize) -> Result<Tensor> {
    let [len] = *tensor.shape().dims() else {
        return Err(bad("slice needs a rank-1 tensor"));
    };
    if start >= end || end > len {
        return Err(bad("slice out of range"));
    }
    Tensor::from_bf16(vec![end - start], &tensor.as_bf16()?[start..end])
}

/// Names of the shards listed by `model.safetensors.index.json`, sorted.
pub(super) fn shard_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index).map_err(|e| bad(format!("read {}: {e}", index.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| bad(format!("index: {e}")))?;
    let map = value
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| bad("index has no weight_map"))?;
    let mut shards: Vec<String> = map.values().filter_map(|v| v.as_str().map(str::to_owned)).collect();
    shards.sort();
    shards.dedup();
    if shards.is_empty() {
        return Err(bad("index lists no shards"));
    }
    for s in &shards {
        let p = Path::new(s);
        if p.is_absolute() || p.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
            return Err(bad(format!("unsafe shard path {s}")));
        }
    }
    Ok(shards.into_iter().map(|s| dir.join(s)).collect())
}

/// Collected, validated BF16 host tensors by checkpoint name, with split tensors already
/// separated. Consumed strictly: `take` removes, `finish` rejects leftovers.
pub(super) struct Collected {
    pub tensors: BTreeMap<String, Tensor>,
    pub inv_freq: Option<Vec<f32>>,
}

impl Collected {
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let t = self.tensors.remove(name).ok_or_else(|| bad(format!("missing {name}")))?;
        if t.shape().dims() != shape {
            return Err(bad(format!("{name}: expected {shape:?}, got {:?}", t.shape().dims())));
        }
        Ok(t)
    }

    pub fn finish(self) -> Result<()> {
        if self.tensors.is_empty() {
            return Ok(());
        }
        let names: Vec<_> = self.tensors.keys().take(8).collect();
        Err(bad(format!("{} unexpected tensors, e.g. {names:?}", self.tensors.len())))
    }
}

/// Concatenate two rank-2 BF16 host tensors along rows.
pub(super) fn concat_rows(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (da, db) = (a.shape().dims(), b.shape().dims());
    if da.len() != 2 || db.len() != 2 || da[1] != db[1] {
        return Err(bad("row concat shape mismatch"));
    }
    let mut v = a.as_bf16()?.to_vec();
    v.extend_from_slice(b.as_bf16()?);
    Tensor::from_bf16(Shape::new(vec![da[0] + db[0], da[1]]), &v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_to_bf16_rounds_to_nearest_even_and_rejects_nonfinite() {
        // 1 + 2^-8 is a tie between 1.0 and 1.0078125 -> even (1.0); 1 + 3*2^-8 ties to the even 1.015625.
        let t = Tensor::from_f32(vec![4], &[1.0 + 2f32.powi(-8), 1.0 + 3.0 * 2f32.powi(-8), -0.0, 3.0e38]).unwrap();
        let b = to_bf16("x", &t).unwrap();
        let v = b.as_bf16().unwrap();
        assert_eq!(v[0].to_f32(), 1.0);
        assert_eq!(v[1].to_f32(), 1.015625);
        assert!(v[2].to_f32().is_sign_negative());
        let inf = Tensor::from_f32(vec![1], &[f32::INFINITY]).unwrap();
        assert!(to_bf16("x", &inf).is_err());
    }

    #[test]
    fn transpose_is_bit_exact_for_ragged_thread_splits() {
        let vals: Vec<bf16> = (0..7 * 13).map(|i| bf16::from_bits(i as u16 * 37)).collect();
        let t = Tensor::from_bf16(vec![7, 13], &vals).unwrap();
        let tt = transpose(&t).unwrap();
        assert_eq!(tt.shape().dims(), [13, 7]);
        let o = tt.as_bf16().unwrap();
        for r in 0..7 {
            for c in 0..13 {
                assert_eq!(o[c * 7 + r].to_bits(), vals[r * 13 + c].to_bits());
            }
        }
    }

    #[test]
    fn patch_permutation_maps_hwc_order_to_channel_major() {
        let p = 2;
        let vals: Vec<bf16> = (0..12).map(|i| bf16::from_f32(i as f32)).collect(); // one row, (dy,dx,c)
        let t = Tensor::from_bf16(vec![1, 12], &vals).unwrap();
        let o = patch_columns_to_channel_major(&t, p).unwrap();
        let o: Vec<f32> = o.as_bf16().unwrap().iter().map(|v| v.to_f32()).collect();
        // channel 0 plane = elements 0,3,6,9 of the HWC order; channel 1 = 1,4,7,10; ...
        assert_eq!(o, vec![0., 3., 6., 9., 1., 4., 7., 10., 2., 5., 8., 11.]);
    }

    #[test]
    fn row_slices_are_contiguous_blocks() {
        let vals: Vec<bf16> = (0..12).map(|i| bf16::from_f32(i as f32)).collect();
        let t = Tensor::from_bf16(vec![4, 3], &vals).unwrap();
        let s = row_slice(&t, 1, 3).unwrap();
        assert_eq!(s.shape().dims(), [2, 3]);
        assert_eq!(s.as_bf16().unwrap()[0].to_f32(), 3.0);
        assert!(row_slice(&t, 3, 5).is_err());
    }

    #[test]
    fn vector_slices_split_a_fused_bias() {
        let vals: Vec<bf16> = (0..6).map(|i| bf16::from_f32(i as f32)).collect();
        let t = Tensor::from_bf16(vec![6], &vals).unwrap();
        let s = slice_1d(&t, 2, 4).unwrap();
        let v: Vec<f32> = s.as_bf16().unwrap().iter().map(|x| x.to_f32()).collect();
        assert_eq!(v, vec![2.0, 3.0]);
        assert!(slice_1d(&t, 4, 7).is_err());
        assert!(slice_1d(&t, 3, 3).is_err());
    }
}

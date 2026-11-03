use super::*;
use crate::{transfers, CudaContext};
use apxinf_core::{DType, Shape, Tensor};

// Two key classes give a closed-form softmax oracle. Q varies across queries,
// K across heads/classes, and V across heads/channels/classes. The reference
// does not share the native reduction, indexing, or row-selection algorithm.
#[test]
fn indexed_attention_all_candidates_two_class_oracle() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    let (qs, heads, dim) = (1785usize, 40usize, 128usize);
    let upload = |shape, values: &[f32]| {
        let host = Tensor::from_bf16(
            shape,
            &values
                .iter()
                .map(|&v| half::bf16::from_f32(v))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        transfers::to_cuda(&host, 0).unwrap()
    };
    for ks in [2665usize, 512, 257] {
        let mut qv = vec![0.; qs * heads * dim];
        let mut kv = vec![0.; ks * heads * dim];
        let mut vv = vec![0.; ks * heads * dim];
        for q in 0..qs {
            for h in 0..heads {
                qv[(q * heads + h) * dim] = (q as i32 % 3 - 1) as f32 / 4.;
            }
        }
        for k in 0..ks {
            let sign = if k % 2 == 0 { 1. } else { -1. };
            for h in 0..heads {
                kv[(k * heads + h) * dim] = sign * (h % 3 + 1) as f32 / 4.;
                for d in 0..dim {
                    vv[(k * heads + h) * dim + d] =
                        0.5 + (h % 5) as f32 / 16. + (d % 7) as f32 / 32. + sign / 4.;
                }
            }
        }
        let q = upload(vec![1, qs, heads, dim], &qv);
        let k = upload(vec![1, ks, heads, dim], &kv);
        let v = upload(vec![1, ks, heads, dim], &vv);
        let mut out = ctx
            .allocate_output(Shape::new(vec![1, qs, heads, dim]), DType::BF16)
            .unwrap();
        let mut args = AttentionArgs::new(&q, &k, &v, &mut out);
        let scale = args.scale as f64;
        args.policy.cache_dir = Some(String::new());
        // The all-candidate oracle also includes the dense probability fallback.
        // The indexed candidate's own smaller bound is checked independently.
        args.policy.workspace_limit = 1 << 30;
        args.policy.deterministic = true;
        let mut expected = vec![0.; qv.len()];
        for q in 0..qs {
            for h in 0..heads {
                let score = (q as i32 % 3 - 1) as f64 / 4. * (h % 3 + 1) as f64 / 4. * scale;
                let positive = ks.div_ceil(2) as f64 * score.exp();
                let negative = (ks / 2) as f64 * (-score).exp();
                let delta = (positive - negative) / (positive + negative) / 4.;
                for d in 0..dim {
                    expected[(q * heads + h) * dim + d] =
                        (0.5 + (h % 5) as f64 / 16. + (d % 7) as f64 / 32. + delta) as f32;
                }
            }
        }
        let normalized = crate::ops::attention::contracts::normalize(&ctx, args).unwrap();
        crate::ops::attention::execution::validate_candidates(&ctx, &normalized, &expected)
            .unwrap();
    }
}

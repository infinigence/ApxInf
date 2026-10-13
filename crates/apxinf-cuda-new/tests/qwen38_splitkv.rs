use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, CudaBuffer, CudaContext};

fn upload(ctx: &CudaContext, values: &[f32], dims: Vec<usize>) -> Tensor {
    let bytes: Vec<u8> = values.iter().flat_map(|value| {
        half::bf16::from_f32(*value).to_bits().to_le_bytes()
    }).collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), DType::BF16).unwrap()
}

#[test]
fn grouped_splitkv_matches_independent_fp64_attention() {
    let ctx = CudaContext::new(0).unwrap();
    let mut random = 47_u32;
    for tokens in [17_usize, 128, 257, 2048, 2176] {
        let mut data = |size| -> Vec<f32> {
            (0..size).map(|_| {
                random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                half::bf16::from_f32(((random >> 16) as i32 - 32768) as f32 / 16384.0).to_f32()
            }).collect()
        };
        let query_host = data(24 * 256);
        let key_host = data(tokens * 4 * 256);
        let value_host = data(tokens * 4 * 256);
        let query = upload(&ctx, &query_host, vec![1, 1, 24, 256]);
        let keys = upload(&ctx, &key_host, vec![1, tokens, 4, 256]);
        let values = upload(&ctx, &value_host, vec![1, tokens, 4, 256]);
        let mut output = upload(&ctx, &vec![f32::NAN; 24 * 256], vec![1, 1, 24, 256]);
        let args = ops::KvCacheAttentionArgs::new(&query, &keys, &values, &mut output);
        ops::kv_cache_attention(&ctx, args).unwrap();
        ctx.synchronize().unwrap();
        let mut bytes = vec![0_u8; 24 * 256 * 2];
        CudaBuffer::from_tensor(&output).unwrap().copy_to_host(&mut bytes).unwrap();
        let actual: Vec<f64> = bytes.chunks_exact(2).map(|value| {
            half::bf16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32() as f64
        }).collect();
        assert_eq!(actual.len(), 6144);
        let mut error_sum = 0.0_f64;
        let mut reference_sum = 0.0_f64;
        let mut max_error = 0.0_f64;
        for head in 0..24 {
            let mut scores: Vec<f64> = (0..tokens).map(|token| {
                (0..256).map(|channel| {
                    query_host[head * 256 + channel] as f64 *
                    key_host[(token * 4 + head / 6) * 256 + channel] as f64
                }).sum::<f64>() / 16.0
            }).collect();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for score in &mut scores { *score = (*score - maximum).exp(); }
            let sum = scores.iter().sum::<f64>();
            for channel in 0..256 {
                let expected = scores.iter().enumerate().map(|(token, score)| {
                    score * value_host[(token * 4 + head / 6) * 256 + channel] as f64
                }).sum::<f64>() / sum;
                let value = actual[head * 256 + channel];
                assert!(value.is_finite());
                let error = value - expected;
                error_sum += error * error;
                reference_sum += expected * expected;
                max_error = max_error.max(error.abs());
            }
        }
        assert!(reference_sum > 0.0);
        let rel_l2 = (error_sum / reference_sum).sqrt();
        assert!(rel_l2 < 0.006 && max_error < 0.02,
                "tokens={tokens} rel_l2={rel_l2} max_error={max_error}");
        println!("PASS tokens={tokens} elements={} rel_l2={rel_l2} max_error={max_error}", actual.len());
    }
}

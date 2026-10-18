#![cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
//! One Core ML batch prediction returns, for every item, the bits that a
//! separate prediction of that item returns.
use apxinf_core::Result;
use apxinf_coreml::{ComputeUnits, ProgramBuilder, ProgramCache};
use half::f16;

#[test]
fn batch_items_match_single_predictions() -> Result<()> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 8, 1, 16]);
    let weight: Vec<f16> = (0..32)
        .map(|i| f16::from_f32((i % 7) as f32 / 8.0 - 0.4))
        .collect();
    let half = b.conv_fp16_scaled(&x, &weight, &[4, 8, 1, 1], f16::from_f32(0.5), "mul_0")?;
    let t = b.tanh(&half);
    let y = b.identity(&t, Some("y"));
    let scratch = std::env::temp_dir().join(format!("apxinf-coreml-batch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    let cache = ProgramCache::new(scratch.join("cache"));
    let model = cache.load(&b.finish(&[&y])?, ComputeUnits::CpuAndNeuralEngine)?;
    let items: Vec<Vec<f16>> = (0..3)
        .map(|k| {
            (0..8 * 16)
                .map(|i| f16::from_f32(((i * 31 + k * 17) % 23) as f32 / 11.0 - 1.0))
                .collect()
        })
        .collect();
    let shape: &[usize] = &[1, 8, 1, 16];
    let outputs: [(&str, &[usize]); 1] = [("y", &[1, 4, 1, 16])];
    let batch: Vec<Vec<&[f16]>> = items.iter().map(|v| vec![&v[..]]).collect();
    let together = model.predict_batch(&[("x", shape)], &batch, &outputs)?;
    assert_eq!(together.len(), items.len());
    for (item, result) in items.iter().zip(&together) {
        let alone = model.predict(&[("x", item, shape)], &outputs)?;
        assert!(result[0]
            .iter()
            .zip(&alone[0])
            .all(|(p, q)| p.to_bits() == q.to_bits()));
        assert!(result[0].iter().any(|v| v.to_f32() != 0.0));
    }
    assert!(model.predict_batch(&[("x", shape)], &[], &outputs).is_err());
    let _ = std::fs::remove_dir_all(&scratch);
    Ok(())
}

#![cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
//! The prediction bridge copies whole rows when the innermost dimension is
//! unit-stride and falls back to element-wise otherwise. Either way the values
//! must come back bit for bit, including rows with padding between them.
use apxinf_core::Result;
use apxinf_coreml::{compile, ComputeUnits, Model, Operand, ProgramBuilder};
use half::f16;

fn round_trip(shape: &[usize]) -> Result<()> {
    let mut builder = ProgramBuilder::new();
    let x = builder.input("x", shape)?;
    // `mul` by one keeps FP16 output; `identity` lets Core ML return a different
    // storage type, which the bridge rejects before it ever reaches the copy loop.
    let y = builder.mul(&x, Operand::Scalar(f16::ONE), None)?;
    let out_name = y.name().to_owned();
    let package = builder.finish(&[&y])?;

    let root = std::env::temp_dir().join(format!("apx-row-copy-{}-{:?}", std::process::id(), shape));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).map_err(|e| apxinf_core::Error::Other(e.to_string()))?;
    let model_dir = root.join("model.mlpackage");
    package.write(&model_dir)?;
    let compiled = root.join("model.mlmodelc");
    compile(&model_dir, &compiled)?;
    let model = Model::load(&compiled, ComputeUnits::All)?;

    let count: usize = shape.iter().product();
    // Distinct values per element so a mis-strided copy cannot pass by accident.
    let input: Vec<f16> = (0..count).map(|i| f16::from_f32((i as f32) * 0.25 - 8.0)).collect();
    let out = model.predict(&[("x", &input, shape)], &[(out_name.as_str(), shape)])?;
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], input, "identity prediction changed values for shape {shape:?}");
    Ok(())
}

#[test]
#[ignore = "requires the Neural Engine"]
fn identity_prediction_returns_the_input_bits() -> Result<()> {
    round_trip(&[1, 2, 3, 8])?;
    round_trip(&[8])?;
    round_trip(&[1, 1, 1, 1])?;
    Ok(())
}

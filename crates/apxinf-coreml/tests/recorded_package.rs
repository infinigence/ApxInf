#![cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
//! Against a recorded coremltools 9.0 package of a Clef int8 MLP program:
//! the writer reproduces its program and weight file, and on this device the
//! two compiled programs predict identical bits. The package is not
//! committed (it holds checkpoint-derived weights); set
//! `APXINF_COREML_REFERENCE` to its `.mlpackage` directory.
use apxinf_core::Result;
use apxinf_coreml::proto::{self, binding, tensor_value, value};
use apxinf_coreml::{blob, compile, ComputeUnits, Model, Operand, ProgramBuilder, ProgramCache};
use half::f16;
use std::path::PathBuf;

fn reference() -> (PathBuf, proto::Model, Vec<u8>) {
    let dir = PathBuf::from(
        std::env::var("APXINF_COREML_REFERENCE").expect("set APXINF_COREML_REFERENCE"),
    );
    let data = dir.join("Data/com.apple.CoreML");
    let model: proto::Model =
        prost::Message::decode(&std::fs::read(data.join("model.mlmodel")).unwrap()[..]).unwrap();
    (
        dir,
        model,
        std::fs::read(data.join("weights/weight.bin")).unwrap(),
    )
}

fn operations(model: &proto::Model) -> &[proto::Operation] {
    &model.ml_program.as_ref().unwrap().functions["main"].block_specializations["CoreML8"]
        .operations
}

fn shape(v: &proto::Value) -> Vec<usize> {
    let t = v.r#type.as_ref().unwrap().tensor_type.as_ref().unwrap();
    t.dimensions
        .iter()
        .map(|d| d.constant.as_ref().unwrap().size as usize)
        .collect()
}

/// INT8 data, its shape, FP16 scale and its shape.
type Quantized = (Vec<i8>, Vec<usize>, Vec<f16>, Vec<usize>);

/// INT8 data and FP16 scale of every blockwise operation, in program order.
fn quantized(model: &proto::Model, weights: &[u8]) -> Vec<Quantized> {
    let blob_of = |op: &proto::Operation, key: &str| {
        let Some(binding::Binding::Value(v)) = &op.inputs[key].arguments[0].binding else {
            panic!()
        };
        let Some(value::Value::BlobFileValue(b)) = &v.value else {
            panic!()
        };
        (blob::read(weights, b.offset).unwrap().1.to_vec(), shape(v))
    };
    operations(model)
        .iter()
        .filter(|op| op.r#type == "constexpr_blockwise_shift_scale")
        .map(|op| {
            let (data, data_shape) = blob_of(op, "data");
            let (scale, scale_shape) = blob_of(op, "scale");
            let scale = scale
                .chunks(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]))
                .collect();
            (
                data.iter().map(|&b| b as i8).collect(),
                data_shape,
                scale,
                scale_shape,
            )
        })
        .collect()
}

/// The FP16 scalar of the const named `name`.
fn scalar(model: &proto::Model, name: &str) -> f16 {
    let op = operations(model)
        .iter()
        .find(|op| op.outputs[0].name == name)
        .unwrap();
    let Some(value::Value::ImmediateValue(v)) = &op.attributes["val"].value else {
        panic!()
    };
    let Some(tensor_value::Value::Bytes(b)) = &v.tensor.as_ref().unwrap().value else {
        panic!()
    };
    f16::from_le_bytes([b.values[0], b.values[1]])
}

/// The int8 MLP program: gate and up int8 convolutions, SiLU in tanh form,
/// normalized SwiGLU factors and K-split int8 down convolutions.
fn int8_mlp_program(model: &proto::Model, weights: &[u8]) -> Result<apxinf_coreml::Package> {
    let q = quantized(model, weights);
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 4096, 1, 512]);
    let w = |b: &mut ProgramBuilder, i: usize| {
        b.constexpr_blockwise_shift_scale(&q[i].0, &q[i].1, &q[i].2, &q[i].3)
    };
    let wg = w(&mut b, 0)?;
    let g = b.conv(&x, &wg)?;
    let wu = w(&mut b, 1)?;
    let u = b.conv(&x, &wu)?;
    let half = b.mul(&g, Operand::Scalar(scalar(model, "mul_0_y_0")), None)?;
    let t = b.tanh(&half);
    let t1 = b.add(&t, Operand::Scalar(scalar(model, "add_0_y_0")), None)?;
    let s = b.mul(&half, Operand::Var(&t1), None)?;
    let sa = b.abs(&s);
    let s_max = b.reduce_max(&sa, &[1], true)?;
    let ua = b.abs(&u);
    let u_max = b.reduce_max(&ua, &[1], true)?;
    let sm = b.mul(&s_max, Operand::Scalar(scalar(model, "mul_2_y_0")), None)?;
    let s_scale = b.maximum(
        &sm,
        Operand::Scalar(scalar(model, "gate_scale_y_0")),
        Some("gate_scale"),
    )?;
    let um = b.mul(&u_max, Operand::Scalar(scalar(model, "mul_3_y_0")), None)?;
    let u_scale = b.maximum(
        &um,
        Operand::Scalar(scalar(model, "up_scale_y_0")),
        Some("up_scale"),
    )?;
    let sn = b.real_div(&s, Operand::Var(&s_scale), None)?;
    let un = b.real_div(&u, Operand::Var(&u_scale), None)?;
    let a = b.mul(&sn, Operand::Var(&un), None)?;
    let n = q[2].1[1];
    let xs = b.split(&a, &[n, n], 1)?;
    let wd0 = w(&mut b, 2)?;
    let y0 = b.conv(&xs[0], &wd0)?;
    let wd1 = w(&mut b, 3)?;
    let y1 = b.conv(&xs[1], &wd1)?;
    let sum = b.add(&y0, Operand::Var(&y1), None)?;
    let y = b.identity(&sum, Some("y"));
    b.finish(&[&y, &s_scale, &u_scale])
}

/// Writer-specific fields excluded from the comparison: coremltools records
/// its version and conversion date there.
fn comparable(mut m: proto::Model) -> proto::Model {
    m.description.as_mut().unwrap().metadata = None;
    m.ml_program.as_mut().unwrap().attributes.clear();
    m
}

#[test]
#[ignore = "requires a recorded coremltools package"]
fn writer_reproduces_the_recorded_program_and_weights() -> Result<()> {
    let (_, model, weights) = reference();
    let package = int8_mlp_program(&model, &weights)?;
    assert_eq!(comparable(package.model.clone()), comparable(model));
    assert!(package.weights == weights, "weight.bin differs");
    Ok(())
}

#[test]
#[ignore = "requires a recorded coremltools package and the Neural Engine"]
fn written_and_recorded_programs_predict_identical_bits() -> Result<()> {
    let (dir, model, weights) = reference();
    let scratch = std::env::temp_dir().join(format!("apxinf-coreml-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let mut cache = ProgramCache::new(scratch.join("cache"));
    let ours = cache.load(
        &int8_mlp_program(&model, &weights)?,
        ComputeUnits::CpuAndNeuralEngine,
    )?;
    cache.seal();
    let mut other = ProgramBuilder::new();
    let x = other.input("x", &[1, 4, 1, 1]);
    let id = other.identity(&x, Some("y"));
    assert!(
        cache
            .load(&other.finish(&[&id])?, ComputeUnits::CpuOnly)
            .is_err(),
        "sealed cache compiled"
    );
    compile(&dir, &scratch.join("reference.mlmodelc"))?;
    let theirs = Model::load(
        &scratch.join("reference.mlmodelc"),
        ComputeUnits::CpuAndNeuralEngine,
    )?;
    let input: Vec<f16> = (0..4096 * 512usize)
        .map(|i| f16::from_f32(((i * 7919) % 401) as f32 / 200.0 - 1.0))
        .collect();
    let outputs: [(&str, &[usize]); 3] = [
        ("y", &[1, 4096, 1, 512]),
        ("gate_scale", &[1, 1, 1, 512]),
        ("up_scale", &[1, 1, 1, 512]),
    ];
    let run = |m: &Model| m.predict(&[("x", &input, &[1, 4096, 1, 512])], &outputs);
    let (a, b) = (run(&ours)?, run(&theirs)?);
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(
            x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
            "output {} differs",
            outputs[i].0
        );
        assert!(
            x.iter().all(|v| v.is_finite()),
            "output {} not finite",
            outputs[i].0
        );
    }
    assert_eq!(run(&ours)?, a, "repeated prediction differs");
    let _ = std::fs::remove_dir_all(&scratch);
    Ok(())
}

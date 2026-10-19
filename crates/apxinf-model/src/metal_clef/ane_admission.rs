use apxinf_core::{Error, Result};
use apxinf_coreml::proto::{self, binding, data_type, tensor_value, value};
use apxinf_coreml::{blob, ComputeDeviceKind, ComputeUnits, ExpectedPlacement, Package};
use std::collections::BTreeSet;

#[derive(Clone, Copy)]
enum RowKind {
    Literal,
    WeightMaterialization,
    Runtime,
}

pub(super) struct Admission<'a> {
    operations: &'a [proto::Operation],
    kinds: Vec<RowKind>,
}

fn check(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Other(format!("Clef Core ML admission: {message}")))
    }
}

fn tensor(value: &proto::ValueType) -> Result<&proto::TensorType> {
    let tensor = value
        .tensor_type
        .as_ref()
        .ok_or(Error::Contract("Clef Core ML needs tensor values"))?;
    check(
        tensor.rank >= 0 && tensor.rank as usize == tensor.dimensions.len(),
        "tensor rank does not match its dimensions",
    )?;
    check(
        tensor.dimensions.iter().all(|dimension| {
            dimension
                .constant
                .as_ref()
                .is_some_and(|size| size.size > 0)
        }),
        "tensor dimensions must be fixed and positive",
    )?;
    Ok(tensor)
}

fn shape_is(value: Option<&proto::ValueType>, expected: &[usize]) -> Result<bool> {
    let tensor = tensor(value.ok_or(Error::Contract("Clef Core ML tensor type is absent"))?)?;
    Ok(tensor.data_type == data_type::FLOAT16
        && tensor.dimensions.len() == expected.len()
        && tensor
            .dimensions
            .iter()
            .zip(expected)
            .all(|(dimension, &size)| dimension.constant.as_ref().unwrap().size == size as u64))
}

fn literal(value: &proto::Value, weights: &[u8]) -> Result<()> {
    let tensor = tensor(
        value
            .r#type
            .as_ref()
            .ok_or(Error::Contract("Clef Core ML literal type is absent"))?,
    )?;
    let elements = tensor
        .dimensions
        .iter()
        .try_fold(1usize, |count, dimension| {
            usize::try_from(dimension.constant.as_ref().unwrap().size)
                .ok()
                .and_then(|size| count.checked_mul(size))
        });
    let elements = elements.ok_or(Error::Contract("Clef Core ML literal extent overflows"))?;
    let byte_count = |width: usize| elements.checked_mul(width);
    let valid = match &value.value {
        Some(value::Value::ImmediateValue(immediate)) => {
            match immediate
                .tensor
                .as_ref()
                .and_then(|tensor| tensor.value.as_ref())
            {
                Some(tensor_value::Value::Ints(values)) => {
                    tensor.data_type == data_type::INT32 && values.values.len() == elements
                }
                Some(tensor_value::Value::Bools(values)) => {
                    tensor.data_type == data_type::BOOL && values.values.len() == elements
                }
                Some(tensor_value::Value::Strings(values)) => {
                    tensor.data_type == data_type::STRING && values.values.len() == elements
                }
                Some(tensor_value::Value::Bytes(values)) => {
                    let width = match tensor.data_type {
                        data_type::FLOAT16 => Some(2),
                        data_type::INT8 => Some(1),
                        _ => None,
                    };
                    width.and_then(byte_count) == Some(values.values.len())
                }
                None => false,
            }
        }
        Some(value::Value::BlobFileValue(reference)) => {
            let expected = match tensor.data_type {
                data_type::FLOAT16 => Some((1, 2)),
                data_type::INT8 => Some((4, 1)),
                _ => None,
            };
            reference.file_name == "@model_path/weights/weight.bin"
                && reference.offset >= 64
                && reference.offset.is_multiple_of(64)
                && weights.get(4..8) == Some(2u32.to_le_bytes().as_slice())
                && blob::read(weights, reference.offset)
                    .zip(expected)
                    .is_some_and(|((dtype, bytes), (wanted, width))| {
                        dtype == wanted && byte_count(width) == Some(bytes.len())
                    })
        }
        None => false,
    };
    check(valid, "literal payload, dtype or blob extent is invalid")
}

impl<'a> Admission<'a> {
    pub(super) fn new(package: &'a Package, shape: [usize; 4]) -> Result<Self> {
        check(
            package.model.specification_version == 9 && !package.model.is_updatable,
            "unsupported or mutable source model",
        )?;
        let description = package
            .model
            .description
            .as_ref()
            .ok_or(Error::Contract("Clef Core ML description is absent"))?;
        let scale_shape = [1, 1, 1, shape[3]];
        check(description.input.len() == 1, "expected one source input")?;
        let input = &description.input[0];
        let feature_matches = |feature: &proto::FeatureDescription, expected: &[usize]| {
            feature.r#type.as_ref().is_some_and(|kind| {
                !kind.is_optional
                    && kind.multi_array_type.as_ref().is_some_and(|array| {
                        array.data_type == proto::ARRAY_FLOAT16
                            && array
                                .shape
                                .iter()
                                .copied()
                                .eq(expected.iter().map(|&size| size as i64))
                    })
            })
        };
        check(
            input.name == "x" && feature_matches(input, &shape),
            "source input profile mismatch",
        )?;
        let outputs = ["y", "gate_scale", "up_scale"];
        check(
            description.output.len() == outputs.len(),
            "source outputs are missing",
        )?;
        for (index, output) in description.output.iter().enumerate() {
            let expected = if index == 0 { &shape } else { &scale_shape };
            check(
                output.name == outputs[index] && feature_matches(output, expected),
                "source output profile mismatch",
            )?;
        }
        let program = package
            .model
            .ml_program
            .as_ref()
            .ok_or(Error::Contract("Clef ML Program is absent"))?;
        check(
            program.version == 1 && program.functions.len() == 1,
            "unsupported source functions",
        )?;
        let function = program
            .functions
            .get("main")
            .ok_or(Error::Contract("Clef main function is absent"))?;
        check(
            function.opset == "CoreML8" && function.block_specializations.len() == 1,
            "unsupported source opset",
        )?;
        check(
            function.inputs.len() == 1
                && function.inputs[0].name == "x"
                && shape_is(function.inputs[0].r#type.as_ref(), &shape)?,
            "function input profile mismatch",
        )?;
        let block = function
            .block_specializations
            .get("CoreML8")
            .ok_or(Error::Contract("Clef source block is absent"))?;
        check(
            block.inputs.is_empty() && block.outputs.iter().map(String::as_str).eq(outputs),
            "source block interface mismatch",
        )?;
        let mut known = BTreeSet::from(["x"]);
        let mut kinds = Vec::with_capacity(block.operations.len());
        for operation in &block.operations {
            for argument in operation.inputs.values() {
                check(!argument.arguments.is_empty(), "empty source argument")?;
                for binding in &argument.arguments {
                    match &binding.binding {
                        Some(binding::Binding::Name(name)) => check(
                            known.contains(name.as_str()),
                            "unresolved source dependency",
                        )?,
                        Some(binding::Binding::Value(value)) => literal(value, &package.weights)?,
                        None => return Err(Error::Contract("Clef source binding is absent")),
                    }
                }
            }
            check(
                !operation.outputs.is_empty(),
                "source operation has no outputs",
            )?;
            for output in &operation.outputs {
                check(
                    !output.name.is_empty() && known.insert(output.name.as_str()),
                    "duplicate source output",
                )?;
                tensor(
                    output
                        .r#type
                        .as_ref()
                        .ok_or(Error::Contract("Clef source output type is absent"))?,
                )?;
            }
            let kind = match operation.r#type.as_str() {
                "const" => {
                    check(
                        operation.inputs.is_empty() && operation.outputs.len() == 1,
                        "constant has runtime inputs",
                    )?;
                    let value = operation
                        .attributes
                        .get("val")
                        .ok_or(Error::Contract("Clef constant value is absent"))?;
                    literal(value, &package.weights)?;
                    check(
                        value.r#type == operation.outputs[0].r#type,
                        "constant output type mismatch",
                    )?;
                    RowKind::Literal
                }
                "constexpr_blockwise_shift_scale" => {
                    check(
                        operation.inputs.len() == 2 && operation.outputs.len() == 1,
                        "unexpected static weight operands",
                    )?;
                    for (name, dtype) in [("data", data_type::INT8), ("scale", data_type::FLOAT16)]
                    {
                        let argument = operation
                            .inputs
                            .get(name)
                            .ok_or(Error::Contract("Clef weight operand is absent"))?;
                        check(
                            argument.arguments.len() == 1,
                            "ambiguous static weight operand",
                        )?;
                        let Some(binding::Binding::Value(value)) = &argument.arguments[0].binding
                        else {
                            return Err(Error::Contract(
                                "Clef static weights must be literal blobs",
                            ));
                        };
                        check(
                            matches!(value.value, Some(value::Value::BlobFileValue(_)))
                                && tensor(value.r#type.as_ref().unwrap())?.data_type == dtype,
                            "static weight dtype or storage mismatch",
                        )?;
                    }
                    RowKind::WeightMaterialization
                }
                _ => RowKind::Runtime,
            };
            kinds.push(kind);
        }
        for (index, name) in outputs.iter().enumerate() {
            let output = block
                .operations
                .iter()
                .flat_map(|operation| &operation.outputs)
                .find(|output| output.name == *name)
                .ok_or(Error::Contract("Clef source result is absent"))?;
            check(
                shape_is(
                    output.r#type.as_ref(),
                    if index == 0 { &shape } else { &scale_shape },
                )?,
                "function output profile mismatch",
            )?;
        }
        Ok(Self {
            operations: &block.operations,
            kinds,
        })
    }

    pub(super) fn check(&self, facts: &ExpectedPlacement) -> Result<()> {
        check(
            facts.compute_units == ComputeUnits::CpuAndNeuralEngine && facts.functions.len() == 1,
            "wrong plan configuration or functions",
        )?;
        let function = &facts.functions[0];
        check(
            function.name == "main"
                && function.root_operation_count == self.operations.len()
                && function.operations.len() == self.operations.len(),
            "unsupported compiled graph rewrite",
        )?;
        for (index, ((source, kind), compiled)) in self
            .operations
            .iter()
            .zip(&self.kinds)
            .zip(&function.operations)
            .enumerate()
        {
            check(
                compiled.path == [index] && compiled.child_operation_counts.is_empty(),
                "unsupported compiled block",
            )?;
            check(
                compiled.operator == source.r#type
                    || compiled.operator.strip_prefix("ios18.") == Some(source.r#type.as_str())
                    || (source.r#type == "reduce_max" && compiled.operator == "ios16.reduce_max"),
                "compiled operator does not match its source",
            )?;
            check(
                compiled
                    .outputs
                    .iter()
                    .map(String::as_str)
                    .eq(source.outputs.iter().map(|output| output.name.as_str())),
                "compiled outputs do not match their source",
            )?;
            check(
                compiled.inputs.len() == source.inputs.len(),
                "compiled ports do not match their source",
            )?;
            let mut ports = BTreeSet::new();
            for input in &compiled.inputs {
                check(
                    ports.insert(input.name.as_str()),
                    "duplicate compiled input port",
                )?;
                let source = source
                    .inputs
                    .get(&input.name)
                    .ok_or(Error::Contract("Clef compiled port has no source"))?;
                check(
                    source.arguments.len() == input.bindings.len(),
                    "compiled binding count mismatch",
                )?;
                for (source, compiled) in source.arguments.iter().zip(&input.bindings) {
                    let matches = match &source.binding {
                        Some(binding::Binding::Name(name)) => {
                            compiled.name.as_ref() == Some(name) && !compiled.literal_present
                        }
                        Some(binding::Binding::Value(_)) => {
                            compiled.name.is_none() && compiled.literal_present
                        }
                        None => false,
                    };
                    check(matches, "compiled dependency or literal binding mismatch")?;
                }
            }
            if matches!(kind, RowKind::Runtime) {
                let usage = compiled
                    .usage
                    .as_ref()
                    .ok_or(Error::Contract("Clef runtime placement is unknown"))?;
                check(
                    usage
                        .preferred
                        .as_ref()
                        .is_some_and(|device| device.kind == ComputeDeviceKind::NeuralEngine)
                        && usage
                            .supported
                            .iter()
                            .any(|device| device.kind == ComputeDeviceKind::NeuralEngine),
                    "runtime operation is not ANE-preferred and supported",
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_coreml::{
        ComputeDevice, DeviceUsage, FunctionPlacement, InputBinding, OperationPlacement,
        PlacementInput, ProgramBuilder,
    };
    use half::f16;

    fn package() -> Package {
        let mut builder = ProgramBuilder::new();
        let input = builder.input("x", &[1, 8, 1, 16]);
        let weight = builder
            .constexpr_blockwise_shift_scale(&[1; 8], &[1, 8, 1, 1], &[f16::ONE], &[1, 1, 1, 1])
            .unwrap();
        let convolution = builder.conv(&input, &weight).unwrap();
        let activation = builder.tanh(&convolution);
        let gate = builder.identity(&activation, Some("gate_scale"));
        let magnitude = builder.abs(&convolution);
        let up = builder.identity(&magnitude, Some("up_scale"));
        let maximum = builder.reduce_max(&input, &[1], true).unwrap();
        let scaled = builder
            .mul(&input, apxinf_coreml::Operand::Var(&gate), None)
            .unwrap();
        let output = builder
            .mul(&scaled, apxinf_coreml::Operand::Var(&maximum), Some("y"))
            .unwrap();
        builder.finish(&[&output, &gate, &up]).unwrap()
    }

    fn device(kind: ComputeDeviceKind) -> ComputeDevice {
        ComputeDevice {
            kind,
            class_name: "test device".to_owned(),
        }
    }

    fn facts(package: &Package) -> ExpectedPlacement {
        let block = &package.model.ml_program.as_ref().unwrap().functions["main"]
            .block_specializations["CoreML8"];
        let operations = block
            .operations
            .iter()
            .enumerate()
            .map(|(index, operation)| OperationPlacement {
                path: vec![index],
                operator: if operation.r#type == "reduce_max" {
                    "ios16.reduce_max".into()
                } else {
                    format!("ios18.{}", operation.r#type)
                },
                inputs: operation
                    .inputs
                    .iter()
                    .map(|(name, argument)| PlacementInput {
                        name: name.clone(),
                        bindings: argument
                            .arguments
                            .iter()
                            .map(|binding| match &binding.binding {
                                Some(binding::Binding::Name(name)) => InputBinding {
                                    name: Some(name.clone()),
                                    literal_present: false,
                                },
                                Some(binding::Binding::Value(_)) => InputBinding {
                                    name: None,
                                    literal_present: true,
                                },
                                None => panic!("invalid fixture"),
                            })
                            .collect(),
                    })
                    .collect(),
                outputs: operation
                    .outputs
                    .iter()
                    .map(|output| output.name.clone())
                    .collect(),
                child_operation_counts: vec![],
                usage: if matches!(
                    operation.r#type.as_str(),
                    "const" | "constexpr_blockwise_shift_scale"
                ) {
                    None
                } else {
                    Some(DeviceUsage {
                        supported: vec![
                            device(ComputeDeviceKind::Cpu),
                            device(ComputeDeviceKind::NeuralEngine),
                        ],
                        preferred: Some(device(ComputeDeviceKind::NeuralEngine)),
                    })
                },
            })
            .collect::<Vec<_>>();
        ExpectedPlacement {
            compiled_path: "/synthetic-only/model.mlmodelc".into(),
            compute_units: ComputeUnits::CpuAndNeuralEngine,
            functions: vec![FunctionPlacement {
                name: "main".to_owned(),
                root_operation_count: operations.len(),
                operations,
            }],
        }
    }

    #[test]
    fn typed_static_operands_and_ane_runtime_match_without_exclusive_execution_claim() {
        let package = package();
        let admission = Admission::new(&package, [1, 8, 1, 16]).unwrap();
        admission.check(&facts(&package)).unwrap();
    }

    #[test]
    fn runtime_cpu_unknown_or_unsupported_placement_is_rejected() {
        let package = package();
        let admission = Admission::new(&package, [1, 8, 1, 16]).unwrap();
        for variant in 0..4 {
            let mut facts = facts(&package);
            let operation = facts.functions[0]
                .operations
                .iter_mut()
                .find(|operation| operation.operator == "ios18.conv")
                .unwrap();
            match variant {
                0 => operation.usage = None,
                1 => {
                    operation.usage.as_mut().unwrap().preferred =
                        Some(device(ComputeDeviceKind::Cpu))
                }
                2 => operation.usage.as_mut().unwrap().supported.clear(),
                _ => {
                    operation.usage.as_mut().unwrap().preferred =
                        Some(device(ComputeDeviceKind::Unknown))
                }
            }
            assert!(
                admission.check(&facts).is_err(),
                "placement variant {variant}"
            );
        }
    }

    #[test]
    fn reduction_namespace_does_not_admit_other_versions_or_operators() {
        let package = package();
        let admission = Admission::new(&package, [1, 8, 1, 16]).unwrap();
        for invalid in ["ios17.reduce_max", "ios160.reduce_max", "ios16.mul"] {
            let mut facts = facts(&package);
            let reduction = facts.functions[0]
                .operations
                .iter_mut()
                .find(|operation| operation.operator == "ios16.reduce_max")
                .unwrap();
            reduction.operator = invalid.into();
            assert!(admission.check(&facts).is_err(), "{invalid}");
        }
    }

    #[test]
    fn literal_flags_names_blocks_and_graph_rewrites_are_not_waived() {
        let package = package();
        let admission = Admission::new(&package, [1, 8, 1, 16]).unwrap();
        for variant in 0..7 {
            let mut facts = facts(&package);
            match variant {
                0 => facts.functions[0].operations[0].inputs[0].bindings[0].literal_present = false,
                1 => facts.functions[0].operations[0].inputs[0].bindings[0].name = Some("x".into()),
                2 => facts.functions[0].operations[0]
                    .child_operation_counts
                    .push(0),
                3 => facts.functions[0].operations[0].outputs[0].push_str("_rewritten"),
                4 => {
                    facts.functions[0].operations.pop();
                }
                5 => {
                    facts.functions[0].operations[0].operator =
                        "future.constexpr_blockwise_shift_scale".into()
                }
                _ => facts.functions[0].operations[0].inputs[0]
                    .bindings
                    .push(InputBinding {
                        name: None,
                        literal_present: true,
                    }),
            }
            assert!(
                admission.check(&facts).is_err(),
                "mapping variant {variant}"
            );
        }
    }

    #[test]
    fn missing_constant_values_and_invalid_blob_sources_are_rejected() {
        for variant in 0..4 {
            let mut package = package();
            if variant == 0 {
                let operations = &mut package
                    .model
                    .ml_program
                    .as_mut()
                    .unwrap()
                    .functions
                    .get_mut("main")
                    .unwrap()
                    .block_specializations
                    .get_mut("CoreML8")
                    .unwrap()
                    .operations;
                operations
                    .iter_mut()
                    .find(|operation| operation.r#type == "const")
                    .unwrap()
                    .attributes
                    .remove("val");
            } else if variant == 1 {
                package.weights.truncate(64);
            } else {
                let operations = &mut package
                    .model
                    .ml_program
                    .as_mut()
                    .unwrap()
                    .functions
                    .get_mut("main")
                    .unwrap()
                    .block_specializations
                    .get_mut("CoreML8")
                    .unwrap()
                    .operations;
                let argument = operations[0].inputs.get_mut("data").unwrap();
                if variant == 2 {
                    argument.arguments[0].binding = Some(binding::Binding::Name("x".into()));
                } else {
                    let Some(binding::Binding::Value(value)) = &mut argument.arguments[0].binding
                    else {
                        panic!()
                    };
                    let Some(value::Value::BlobFileValue(reference)) = &mut value.value else {
                        panic!()
                    };
                    reference.file_name = "@model_path/other.bin".into();
                }
            }
            assert!(
                Admission::new(&package, [1, 8, 1, 16]).is_err(),
                "source variant {variant}"
            );
        }
    }

    #[test]
    #[ignore = "requires native Core ML metadata and a CPU-preferred small-program fixture"]
    fn native_cpu_preference_is_rejected_by_family_admission() -> Result<()> {
        let package = package();
        let admission = Admission::new(&package, [1, 8, 1, 16])?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../devlocal/clef-coreml-admission-tests")
            .join(format!("native-{}-{nonce}", std::process::id()));
        let cache = apxinf_coreml::ProgramCache::new(&root);
        let observed_cpu = std::cell::Cell::new(false);
        let result = cache.load_with_placement(
            &package,
            ComputeUnits::CpuAndNeuralEngine,
            std::time::Duration::from_secs(60),
            |facts| {
                observed_cpu.set(
                    facts
                        .functions
                        .iter()
                        .flat_map(|function| &function.operations)
                        .filter_map(|operation| operation.usage.as_ref())
                        .filter_map(|usage| usage.preferred.as_ref())
                        .any(|device| device.kind == ComputeDeviceKind::Cpu),
                );
                admission.check(facts)
            },
        );
        let error = result.err().map(|error| error.to_string());
        std::fs::remove_dir_all(root)?;
        assert!(
            observed_cpu.get(),
            "fixture did not expose native CPU preference"
        );
        assert!(
            error
                .as_deref()
                .is_some_and(|error| error.contains("not ANE-preferred")),
            "{error:?}"
        );
        Ok(())
    }
}

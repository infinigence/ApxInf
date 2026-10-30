#![cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]

use apxinf_core::{Error, Result};
use apxinf_coreml::{ComputeDeviceKind, ComputeUnits, ProgramBuilder, ProgramCache};
use half::f16;
use std::cell::Cell;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[test]
#[ignore = "requires the native Xcode compiler and macOS/SDK 14.4+"]
fn late_completion_owns_native_state_and_contains_callback_exceptions() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let output = std::env::temp_dir().join(format!(
        "apxinf-plan-lifetime-{}-{nonce}",
        std::process::id()
    ));
    let source =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/placement_lifetime.mm");
    let mut compiler = std::process::Command::new("xcrun");
    compiler
        .args([
            "clang++",
            "-std=c++20",
            "-fobjc-arc",
            "-framework",
            "CoreML",
            "-framework",
            "Foundation",
        ])
        .arg(&source)
        .arg("-o")
        .arg(&output);
    if let Some(sdk) = std::env::var_os("SDKROOT") {
        compiler.arg("-isysroot").arg(sdk);
    }
    assert!(compiler.status().unwrap().success());
    let result = std::process::Command::new(&output).status().unwrap();
    std::fs::remove_file(output).unwrap();
    assert!(result.success(), "native completion probe exited {result}");
}

#[test]
#[ignore = "requires exclusive native Core ML device lease and macOS 14.4+"]
fn native_plan_binds_cpu_configuration_and_rejects_before_publication() -> Result<()> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("apxinf-placement-{}-{nonce}", std::process::id()));
    let mut builder = ProgramBuilder::new();
    let input = builder.input("x", &[1, 8, 1, 16]);
    let other = builder.input("other", &[1, 8, 1, 16]);
    let sum = builder.add(&input, apxinf_coreml::Operand::Var(&other), None)?;
    let output = builder.tanh(&sum);
    let package = builder.finish(&[&output])?;
    let cache = ProgramCache::new(&root);
    let calls = Cell::new(0);
    assert!(cache
        .load_with_placement(&package, ComputeUnits::CpuOnly, Duration::ZERO, |_| {
            calls.set(calls.get() + 1);
            Ok(())
        })
        .is_err());
    assert_eq!(calls.get(), 0);
    assert!(!root.exists());
    let model = cache.load_with_placement(
        &package,
        ComputeUnits::CpuOnly,
        Duration::from_secs(60),
        |facts| {
            calls.set(calls.get() + 1);
            assert_eq!(facts.compute_units, ComputeUnits::CpuOnly);
            assert!(facts.compiled_path.is_absolute() && facts.compiled_path.is_dir());
            let operations: Vec<_> = facts
                .functions
                .iter()
                .flat_map(|function| &function.operations)
                .collect();
            assert!(!operations.is_empty());
            assert!(
                operations
                    .iter()
                    .any(|operation| matches!(operation.operator.as_str(), "tanh" | "ios18.tanh")),
                "{operations:?}"
            );
            assert!(operations
                .iter()
                .flat_map(|operation| &operation.inputs)
                .flat_map(|input| &input.bindings)
                .any(|binding| binding.name.as_deref() == Some("other")));
            for operation in operations {
                if let Some(preferred) = operation
                    .usage
                    .as_ref()
                    .and_then(|usage| usage.preferred.as_ref())
                {
                    assert_eq!(preferred.kind, ComputeDeviceKind::Cpu);
                }
            }
            Ok(())
        },
    )?;
    assert_eq!(calls.get(), 1);
    let result = model.predict(
        &[
            ("x", &vec![f16::ZERO; 128], &[1, 8, 1, 16]),
            ("other", &vec![f16::ZERO; 128], &[1, 8, 1, 16]),
        ],
        &[(output.name(), &[1, 8, 1, 16])],
    )?;
    assert!(result[0].iter().all(|value| *value == f16::ZERO));
    drop(model);
    let refused = cache.load_with_placement(
        &package,
        ComputeUnits::CpuOnly,
        Duration::from_secs(60),
        |_| {
            calls.set(calls.get() + 1);
            Err(Error::Contract("deliberate family placement rejection"))
        },
    );
    assert!(matches!(
        refused,
        Err(Error::Contract("deliberate family placement rejection"))
    ));
    assert_eq!(calls.get(), 2);
    std::fs::remove_dir_all(root).map_err(|error| Error::Other(error.to_string()))?;
    Ok(())
}

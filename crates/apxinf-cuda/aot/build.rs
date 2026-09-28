//! Offline native operator export. Run with `cargo run --example build-aot`.
mod bundle;
use bundle::{digest, Result};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 || args[0] != "--inputs" || args[2] != "--out" {
        return Err("usage: build-aot --inputs <source-paths.json> --out <new-directory>".into());
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("aot");
    let inputs = Path::new(&args[1]).canonicalize()?;
    let paths: BTreeMap<String, String> = serde_json::from_slice(&fs::read(&inputs)?)?;
    let config: Value = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
    let toolkit = Command::new("nvcc").arg("--version").output()?;
    if !toolkit.status.success() || !String::from_utf8(toolkit.stdout)?.contains("release 13.2,") {
        return Err("this operator bundle requires CUDA 13.2".into());
    }
    let out = PathBuf::from(&args[3]);
    fs::create_dir(&out)?;
    let out = out.canonicalize()?;
    let source_path = |key: &str| -> Result<PathBuf> {
        Ok(inputs
            .parent()
            .unwrap()
            .join(
                paths
                    .get(key)
                    .ok_or_else(|| format!("missing source path {key}"))?,
            )
            .canonicalize()?)
    };
    let python = paths.get("python").map(String::as_str).unwrap_or("python3");
    let mut kernels = Vec::new();
    for recipe in config["kernels"].as_array().ok_or("missing recipes")? {
        let id = recipe["id"].as_str().ok_or("missing id")?;
        let symbol = recipe["symbol"].as_str().ok_or("missing symbol")?;
        let exporter = root
            .join("exporters")
            .join(recipe["exporter"].as_str().ok_or("missing exporter")?);
        let exporter_hash = digest(&exporter)?;
        if recipe["exporter_sha256"].as_str() != Some(exporter_hash.as_str()) {
            return Err(format!("{id}: exporter differs from reviewed recipe").into());
        }
        let directory = out.join(id);
        let mut command = Command::new(python);
        command.arg(&exporter);
        for (flag, key) in recipe["inputs"].as_object().ok_or("missing inputs")? {
            command
                .arg(flag)
                .arg(source_path(key.as_str().ok_or("invalid source key")?)?);
        }
        for arg in recipe["args"].as_array().ok_or("missing args")? {
            command.arg(arg.as_str().ok_or("invalid export argument")?);
        }
        command.arg("--out").arg(&directory);
        let log = fs::File::create(out.join(format!("{id}.log")))?;
        command
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
        if !command.status()?.success() {
            return Err(format!("{id} export failed; see {}", out.display()).into());
        }
        let artifact = |extension: &str| -> Result<Value> {
            let relative = PathBuf::from(id).join(format!("{symbol}.{extension}"));
            Ok(json!({"sha256":digest(&out.join(&relative))?,"path":relative}))
        };
        let mut source = recipe["source"].clone();
        source["exporter_sha256"] = json!(exporter_hash);
        source["recipe_sha256"] = json!(digest(&root.join("manifest.json"))?);
        kernels.push(
            json!({"id":id,"symbol":symbol,"contract":recipe["contract"],
            "specialization":recipe["specialization"],
            "source":source,"object":artifact("o")?,"header":artifact("h")?}),
        );
    }
    fs::create_dir(out.join("lib"))?;
    let runtime = source_path("runtime_archive")?;
    let runtime_relative = "lib/libcute_runtime.a";
    fs::copy(runtime, out.join(runtime_relative))?;
    let manifest = json!({"schema":1,"target":config["target"],"sm":config["sm"],
        "cuda":config["cuda"],"cutlass_dsl":config["cutlass_dsl"],"kernels":kernels,
        "runtime":{"path":runtime_relative,"sha256":digest(&out.join(runtime_relative))?,
            "project":"NVIDIA CUTLASS DSL runtime","version":"4.7.0",
            "license":"NVIDIA Software License Agreement"}});
    let destination = out.join("manifest.json");
    fs::write(&destination, serde_json::to_vec_pretty(&manifest)?)?;
    bundle::verify(&destination, "aarch64-unknown-linux-gnu", "sm_110", "13.2")?;
    println!("{}", destination.display());
    Ok(())
}

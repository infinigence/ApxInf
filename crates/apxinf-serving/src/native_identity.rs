//! Native runtime identity shared by the launcher and execution owner.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};

pub fn digest(path: &Path) -> Result<(u64, String), String> {
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let before = file.metadata().map_err(|e| e.to_string())?;
    if !before.is_file() {
        return Err("An identity artifact is not a regular file.".into());
    }
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    let after = file.metadata().map_err(|e| e.to_string())?;
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return Err("An identity artifact changed during hashing.".into());
    }
    Ok((before.len(), format!("{:x}", hash.finalize())))
}

pub fn runtime(executable: &Path, library: &Path) -> Result<Value, String> {
    if library.file_name().and_then(|name| name.to_str()) != Some("libmlx.dylib") {
        return Err("The native runtime requires libmlx.dylib.".into());
    }
    Ok(
        json!({"kind":"native_minicpm5", "apxinf":"0.1.0", "mlx":"0.31.2",
        "executable_sha256":digest(executable)?.1, "metal_library_sha256":digest(library)?.1}),
    )
}

pub fn execution(context: usize, output: usize, memory: u64) -> Value {
    json!({"provider":"metal_minicpm5", "precision":"bf16", "variant":"bf16-compiled",
        "parser":"minicpm5_xml", "enable_thinking":false, "max_context":context,
        "max_output_tokens":output, "memory_limit_bytes":memory})
}

pub fn sampled_execution(context: usize, output: usize, memory: u64) -> Value {
    let mut value = execution(context, output, memory);
    value["sampling"] = json!({"algorithm":"apxinf-cpu-categorical-v1",
        "temperature":"1", "top_p":"0.95", "seed_policy":"request-id-sha256-v1"});
    value
}

pub fn manifest(model: &Path, runtime: Value, execution: Value) -> Result<Value, String> {
    let mut entries = std::fs::read_dir(model)
        .map_err(|e| e.to_string())?
        .map(|e| e.map(|e| e.path()).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort();
    if entries.is_empty() || entries.len() > 4096 {
        return Err("Invalid model artifact count.".into());
    }
    let mut artifacts = Vec::new();
    for path in entries {
        let resolved = std::fs::canonicalize(&path).map_err(|e| e.to_string())?;
        let (size, sha256) = digest(&resolved)?;
        if std::fs::canonicalize(&path).map_err(|e| e.to_string())? != resolved {
            return Err("Model artifact changed.".into());
        }
        artifacts.push(json!({"path":path.file_name().and_then(|v|v.to_str()).ok_or("Invalid artifact name.")?, "size_bytes":size, "sha256":sha256}));
    }
    let adapter_revision = if execution.get("sampling").is_some() {
        "apxinf-native-minicpm5-sampled-v1"
    } else {
        "apxinf-native-minicpm5-v1"
    };
    Ok(
        json!({"artifacts":artifacts, "runtime":runtime, "execution":execution,
        "adapter_revision":adapter_revision}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_identity_adds_only_the_declared_policy() {
        let greedy = execution(16384, 1024, 9000000000);
        let mut sampled = sampled_execution(16384, 1024, 9000000000);
        assert_eq!(
            sampled.as_object_mut().unwrap().remove("sampling"),
            Some(json!({
                "algorithm":"apxinf-cpu-categorical-v1", "temperature":"1", "top_p":"0.95",
                "seed_policy":"request-id-sha256-v1"
            }))
        );
        assert_eq!(sampled, greedy);
        assert_eq!(greedy["enable_thinking"], false);
        assert_eq!(greedy["precision"], "bf16");
        assert_eq!(greedy["variant"], "bf16-compiled");
    }

    #[test]
    fn manifest_revision_and_digest_follow_sampling_identity() {
        let model = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/serving/native-tokenizer"
        ));
        let greedy = manifest(model, json!({}), execution(16384, 1024, 9000000000)).unwrap();
        let sampled =
            manifest(model, json!({}), sampled_execution(16384, 1024, 9000000000)).unwrap();
        assert_eq!(greedy["adapter_revision"], "apxinf-native-minicpm5-v1");
        assert_eq!(
            sampled["adapter_revision"],
            "apxinf-native-minicpm5-sampled-v1"
        );
        assert_eq!(greedy["artifacts"], sampled["artifacts"]);
        assert_ne!(
            crate::contracts::canonical_identity_digest("apxinf-model-identity-v1", &greedy)
                .unwrap(),
            crate::contracts::canonical_identity_digest("apxinf-model-identity-v1", &sampled)
                .unwrap()
        );
    }
}

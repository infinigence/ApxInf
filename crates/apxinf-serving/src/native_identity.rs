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
        json!({"kind":"native_minicpm5", "apxinf":"0.1.0", "mlx":"0.32.3",
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

pub fn thinking_execution(context: usize, output: usize, memory: u64) -> Value {
    let mut value = sampled_execution(context, output, memory);
    value["enable_thinking"] = true.into();
    value["output_policy"] = "minicpm5-initial-thinking-v1".into();
    value
}

pub fn bounded_thinking_execution(context: usize, output: usize, memory: u64) -> Value {
    let mut value = thinking_execution(context, output, memory);
    value["output_policy"] = "minicpm5-initial-thinking-fixed512-v1".into();
    value["initial_reasoning"] = json!({
        "budget_tokens":512, "close_token_id":9, "selection":"cpu-constrained-close-v1"
    });
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
    let adapter_revision = if execution.get("initial_reasoning").is_some() {
        "apxinf-native-minicpm5-bounded-thinking-v1"
    } else if execution.get("output_policy").is_some() {
        "apxinf-native-minicpm5-thinking-v1"
    } else if execution.get("sampling").is_some() {
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
    fn bounded_thinking_identity_preserves_the_underlying_deployment() {
        let corpus: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/serving/native-bounded-thinking-v0.1.json"
        ))
        .unwrap();
        assert_eq!(
            bounded_thinking_execution(16384, 1024, 4194304),
            corpus["templates"]["ready"]["model_manifest"]["execution"]
        );
        let mut bounded = bounded_thinking_execution(16384, 1024, 9000000000);
        assert_eq!(
            bounded.as_object_mut().unwrap().remove("initial_reasoning"),
            Some(
                json!({"budget_tokens":512,"close_token_id":9,"selection":"cpu-constrained-close-v1"})
            )
        );
        assert_eq!(
            bounded["output_policy"],
            "minicpm5-initial-thinking-fixed512-v1"
        );
        bounded["output_policy"] = "minicpm5-initial-thinking-v1".into();
        assert_eq!(bounded, thinking_execution(16384, 1024, 9000000000));
    }

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
    fn thinking_identity_changes_only_the_declared_fields() {
        let sampled = sampled_execution(16384, 1024, 9000000000);
        let mut thinking = thinking_execution(16384, 1024, 9000000000);
        assert_eq!(thinking["enable_thinking"], true);
        assert_eq!(
            thinking.as_object_mut().unwrap().remove("output_policy"),
            Some(json!("minicpm5-initial-thinking-v1"))
        );
        thinking["enable_thinking"] = false.into();
        assert_eq!(thinking, sampled);
        let corpus: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/serving/native-thinking-v0.1.json"
        ))
        .unwrap();
        assert_eq!(
            thinking_execution(16384, 128, 2097152),
            corpus["templates"]["ready"]["model_manifest"]["execution"]
        );
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
        let thinking = manifest(
            model,
            json!({}),
            thinking_execution(16384, 1024, 9000000000),
        )
        .unwrap();
        let bounded = manifest(
            model,
            json!({}),
            bounded_thinking_execution(16384, 1024, 9000000000),
        )
        .unwrap();
        assert_eq!(greedy["adapter_revision"], "apxinf-native-minicpm5-v1");
        assert_eq!(
            sampled["adapter_revision"],
            "apxinf-native-minicpm5-sampled-v1"
        );
        assert_eq!(greedy["artifacts"], sampled["artifacts"]);
        assert_eq!(
            thinking["adapter_revision"],
            "apxinf-native-minicpm5-thinking-v1"
        );
        assert_eq!(thinking["artifacts"], sampled["artifacts"]);
        assert_eq!(
            bounded["adapter_revision"],
            "apxinf-native-minicpm5-bounded-thinking-v1"
        );
        assert_eq!(bounded["artifacts"], thinking["artifacts"]);
        assert_ne!(
            crate::contracts::canonical_identity_digest("apxinf-model-identity-v1", &bounded)
                .unwrap(),
            crate::contracts::canonical_identity_digest("apxinf-model-identity-v1", &thinking)
                .unwrap()
        );
        assert_ne!(
            crate::contracts::canonical_identity_digest("apxinf-model-identity-v1", &thinking)
                .unwrap(),
            crate::contracts::canonical_identity_digest("apxinf-model-identity-v1", &sampled)
                .unwrap()
        );
        assert_ne!(
            crate::contracts::canonical_identity_digest("apxinf-model-identity-v1", &greedy)
                .unwrap(),
            crate::contracts::canonical_identity_digest("apxinf-model-identity-v1", &sampled)
                .unwrap()
        );
    }
}

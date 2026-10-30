use crate::{
    contracts,
    supervisor::{file_digest, Config},
};
use serde_json::{json, Value};
use std::path::Path;

const SOURCES: [&str; 7] = [
    "vision_worker.py",
    "vision_runtime.py",
    "vision_identity.py",
    "vision_preparation.py",
    "text_worker.py",
    "contracts.py",
    "owned_media.py",
];
const LIBRARY_SHA256: &str = "d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6";
const MODULES: [(&str, &str, Option<&str>); 16] = [
    ("mlx_core", "mlx/core.cpython-314-darwin.so", None),
    ("mlx_vlm", "mlx_vlm/__init__.py", None),
    ("transformers", "transformers/__init__.py", None),
    ("numpy", "numpy/__init__.py", None),
    ("pillow", "PIL/__init__.py", None),
    ("torch", "torch/__init__.py", None),
    ("torchvision", "torchvision/__init__.py", None),
    ("tokenizers", "tokenizers/__init__.py", None),
    ("generation", "mlx_vlm/generate/ar.py", None),
    (
        "processor",
        "mlx_vlm/models/minicpmv4_7/processing_minicpmv4_7.py",
        None,
    ),
    (
        "module_state",
        "mlx/nn/layers/base.py",
        Some("cefa99a579f3c2016b70e15c183ae5e8bbef103ce9a0d89cb3789d9aedb23503"),
    ),
    (
        "minicpm_model",
        "mlx_vlm/models/minicpmv4_7/minicpmv4_7.py",
        Some("d52b40283d9dfc62bb0e06df94814b21263d7ff35d90b1bf59741c681e35fe48"),
    ),
    (
        "minicpm_language",
        "mlx_vlm/models/minicpmv4_7/language.py",
        Some("790769ce0dc543e5cc481cde1ac3084b434684fbbc4d948203d4efbfeb417fb5"),
    ),
    (
        "minicpm_language_base",
        "mlx_vlm/models/minicpmv4_6/language.py",
        Some("999aeceece5de4b6bfa386f6a719be898584803af46bf5ac07af68ad75f928ad"),
    ),
    (
        "qwen_language",
        "mlx_vlm/models/qwen3_5/language.py",
        Some("f5821a9d850eef2868e048977b56bf5798a57d36bf2141272661a5f59725000b"),
    ),
    (
        "rotary_state",
        "mlx_vlm/models/rope_utils.py",
        Some("6944f69a03c41cbb9613afbd2f1b54b71c504840aaeb0eade39c6109dbfdd25b"),
    ),
];
const ARTIFACTS: [(&str, u64, &str); 8] = [
    (
        "chat_template.jinja",
        7249,
        "49c37bcc86c7f7f20ed2ebf79e9f872be41ba2b5f9d0d51e6712cc038ef51d6c",
    ),
    (
        "config.json",
        2575,
        "20ed1fe71b174719b88ed1ce37c840ec45e3657f1120391d45f33505be7638a4",
    ),
    (
        "generation_config.json",
        132,
        "4dc190baaab3bc4bccb796ee73deb2c3dd776cd4feae3587b29bc22083e03415",
    ),
    (
        "model.safetensors",
        3109054480,
        "8c4921ac83182c85190ad10c3aeb118d51e385d2d98ec13b88b514bbf6cac002",
    ),
    (
        "preprocessor_config.json",
        148,
        "c64289d04f36e48d9d059e3a0069f54062b2ef6044a807adb448b98449cb54de",
    ),
    (
        "processor_config.json",
        2,
        "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    ),
    (
        "tokenizer.json",
        19992481,
        "33861e37bb955af1e3f3061182b820f347eba2b9c2c1011c82794bf0d6e77b54",
    ),
    (
        "tokenizer_config.json",
        1436,
        "900ecc6ba690a07270dc23935d17bbb7d2ed69a073a90a9cf731e5313e86ebba",
    ),
];

pub(crate) fn verify_manifest(manifest: &Value, config: &Config) -> Result<(), String> {
    verify_profile(manifest, config)?;
    verify_sources(manifest, config)?;
    verify_runtime_files(&manifest["runtime_files"], &config.python)
}

fn verify_profile(manifest: &Value, config: &Config) -> Result<(), String> {
    config.validate_memory()?;
    let runtime =
        json!({"python":"3.14.3","mlx":"0.32.4","mlx_vlm":"0.7.10","transformers":"5.18.0"});
    let packages = json!({"mlx":"0.32.4","mlx-metal":"0.32.4","mlx-vlm":"0.7.10","transformers":"5.18.0", "numpy":"2.5.4","Pillow":"13.0.0","torch":"2.12.1","torchvision":"0.27.1", "tokenizers":"0.23.3","safetensors":"0.8.0","huggingface-hub":"2.2.0"});
    let artifacts: Vec<_> = ARTIFACTS
        .iter()
        .map(|(path, size, digest)| json!({"path":path,"size_bytes":size,"sha256":digest}))
        .collect();
    let processor_artifacts: Vec<_> = artifacts
        .iter()
        .filter(|item| item["path"] != "model.safetensors")
        .collect();
    let processor = json!({
        "class":"mlx_vlm.models.minicpmv4_7.processing_minicpmv4_7.MiniCPMV4_7Processor",
        "runtime":runtime,"artifacts":processor_artifacts,
        "options":{"return_tensors":"np","downsample_mode":"16x","max_slice_nums":1,"padding":true,"add_special_tokens":false},
        "template_options":{"enable_thinking":false,"tokenize":false,"add_generation_prompt":true}
    });
    let execution = json!({
        "provider":"mlx-vlm-minicpmv47","precision":"bf16","enable_thinking":false,"kv_reuse":false,
        "max_context":config.max_context,"max_output_tokens":config.max_output,
        "prefill_step_size":256,"output_batch_tokens":1,"memory_limit_bytes":config.memory_budget,
        "selection":{"algorithm":"mlx-vlm-fixed-controls-v1","top_k":100,"min_p":"0","repetition_penalty":"1.03",
            "repetition_context_size":8192,"presence_penalty":"0","frequency_penalty":"0","seed_policy":"request-id-sha256-v1"}
    });
    let request_state = json!({
        "revision":"minicpmv47-transient-state-v1","max_context":config.max_context,
        "position_fields":["_position_ids","_rope_deltas"],"convolution_cache_field":"_qwen3_5_decode_conv_weight",
        "convolution_module_count":18,"rotary_cache_field":"_compiled_apply","rotary_module_count":6
    });
    for (field, expected) in [
        ("runtime", runtime),
        ("packages", packages),
        ("artifacts", json!(artifacts)),
        ("execution", execution),
        ("request_state", request_state),
        (
            "processor_revision",
            json!(
                contracts::canonical_identity_digest("apxinf-vision-processor-v1", &processor)
                    .map_err(|error| error.to_string())?
            ),
        ),
        ("processor", processor),
        ("platform", json!("macos-metal")),
        ("checkpoint", json!("openbmb/MiniCPM-V-4.7-1B")),
        (
            "checkpoint_revision",
            json!("379f120c2ff3588e6e06559ec6b9ec15cc05114e"),
        ),
        ("adapter_revision", json!("apxinf-mlx-minicpmv47-vision-v1")),
    ] {
        if manifest[field] != expected {
            return Err(format!("The selected vision identity differs: {field}"));
        }
    }
    Ok(())
}

fn exact_fields(value: &Value, fields: &[&str]) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or("The vision identity requires an object.")?;
    if object.len() != fields.len() || fields.iter().any(|field| !object.contains_key(*field)) {
        return Err("The vision identity fields differ from the selected profile.".into());
    }
    Ok(())
}

fn verify_digest(path: &Path, digest: &Value) -> Result<(), String> {
    let resolved = std::fs::canonicalize(path).map_err(|error| error.to_string())?;
    let (_, actual) = file_digest(&resolved)?;
    if digest.as_str() != Some(&actual)
        || std::fs::canonicalize(path).map_err(|error| error.to_string())? != resolved
    {
        return Err(format!(
            "The vision file identity differs: {}",
            path.display()
        ));
    }
    Ok(())
}

fn verify_sources(manifest: &Value, config: &Config) -> Result<(), String> {
    exact_fields(&manifest["sources"], &SOURCES)?;
    for source in SOURCES {
        verify_digest(
            &config.worker.with_file_name(source),
            &manifest["sources"][source],
        )?;
    }
    verify_digest(&config.worker, &manifest["adapter_sha256"])?;
    verify_digest(&config.python, &manifest["python_sha256"])?;
    if manifest["sources"]["vision_worker.py"] != manifest["adapter_sha256"] {
        return Err("The selected worker differs from the vision source identity.".into());
    }
    Ok(())
}

fn verify_runtime_files(files: &Value, python: &Path) -> Result<(), String> {
    exact_fields(files, &["modules", "libraries"])?;
    let keys: Vec<_> = MODULES.iter().map(|(name, _, _)| *name).collect();
    exact_fields(&files["modules"], &keys)?;
    let bin = std::fs::canonicalize(python.parent().ok_or("The Python path has no parent.")?)
        .map_err(|error| error.to_string())?;
    let environment = bin.parent().ok_or("The Python path has no environment.")?;
    if bin.file_name().is_none_or(|name| name != "bin") || !environment.join("pyvenv.cfg").is_file()
    {
        return Err("The selected Python path must belong to a virtual environment.".into());
    }
    let packages = environment.join("lib/python3.14/site-packages");
    for (name, relative, pinned) in MODULES {
        verify_runtime_record(
            &files["modules"][name],
            &packages.join(relative),
            environment,
            pinned,
        )?;
    }
    let libraries = files["libraries"]
        .as_array()
        .ok_or("The vision libraries must be an array.")?;
    if libraries.len() != 1 {
        return Err("The vision profile requires one MLX library.".into());
    }
    verify_runtime_record(
        &libraries[0],
        &packages.join("mlx/lib/libmlx.dylib"),
        environment,
        Some(LIBRARY_SHA256),
    )
}

fn verify_runtime_record(
    record: &Value,
    selected: &Path,
    environment: &Path,
    pinned: Option<&str>,
) -> Result<(), String> {
    exact_fields(record, &["path", "sha256"])?;
    let path = Path::new(
        record["path"]
            .as_str()
            .ok_or("The vision runtime file has no path.")?,
    );
    let resolved = std::fs::canonicalize(selected).map_err(|error| error.to_string())?;
    if path != resolved
        || !resolved.starts_with(environment)
        || pinned.is_some_and(|digest| record["sha256"] != digest)
    {
        return Err("The vision runtime file differs from the selected environment.".into());
    }
    verify_digest(path, &record["sha256"])?;
    if std::fs::canonicalize(selected).map_err(|error| error.to_string())? != resolved {
        return Err("A selected vision runtime path changed during inspection.".into());
    }
    Ok(())
}

pub(crate) fn verify_prepared(
    event: &Value,
    ready: &Value,
    messages: &[Value],
) -> Result<(), String> {
    let media: Vec<_> = messages
        .iter()
        .filter_map(|message| message["content"].as_array())
        .flatten()
        .filter(|part| part["type"] == "image")
        .map(|part| &part["media"])
        .collect();
    let replay = &event["replay"];
    if replay["media"] != json!(media)
        || replay["capability_revision"] != ready["capability_revision"]
        || replay["processor_revision"] != ready["model_manifest"]["processor_revision"]
    {
        return Err(
            "The prepared vision replay differs from the selected input or worker identity.".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests;

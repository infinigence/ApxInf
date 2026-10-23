use super::*;
use crate::owned_media::MediaBatch;
use serde_json::json;

pub(super) fn media(value: &Value) -> Result<()> {
    MediaBatch::from_descriptors(value)
        .map(|_| ())
        .map_err(|error| ContractError(error.to_string()))
}

pub(super) fn content(value: &Value, role: &Value, images: &mut Vec<Value>) -> Result<()> {
    if value.is_string() {
        return Ok(());
    }
    choice(role, &["user", "tool"])?;
    for part in array(value, 1, 256)? {
        match string(&part["type"])? {
            "text" => {
                fields(part, &[], &["type", "text"], &[])?;
                string(&part["text"])?;
            }
            "image" => {
                fields(part, &[], &["type", "media"], &[])?;
                images.push(part["media"].clone());
            }
            _ => return Err(ContractError("unsupported content part".into())),
        }
    }
    Ok(())
}

pub(super) fn ready(frame: &Value) -> Result<()> {
    let runtime =
        json!({"python":"3.14.3","mlx":"0.32.4","mlx_vlm":"0.7.10","transformers":"5.18.0"});
    require(frame["runtime"] == runtime, "unsupported vision runtime")?;
    let manifest = &frame["model_manifest"];
    require(
        manifest["runtime"] == runtime
            && manifest["checkpoint"] == "openbmb/MiniCPM-V-4.7-1B"
            && manifest["checkpoint_revision"] == "379f120c2ff3588e6e06559ec6b9ec15cc05114e"
            && manifest["adapter_revision"] == "apxinf-mlx-minicpmv47-vision-v1",
        "unsupported vision identity",
    )?;
    object(&manifest["processor"])?;
    require(
        manifest["processor_revision"]
            == canonical_identity_digest("apxinf-vision-processor-v1", &manifest["processor"])?,
        "processor revision mismatch",
    )?;
    let artifacts = array(&manifest["artifacts"], 1, MAX_PROMPT_TOKENS as usize)?;
    let mut previous = "";
    let mut weights = false;
    for artifact in artifacts {
        fields(artifact, &[], &["path", "size_bytes", "sha256"], &[])?;
        let path = string(&artifact["path"])?;
        require(
            !path.is_empty()
                && !path.starts_with('/')
                && path.split('/').all(|part| !["", ".", ".."].contains(&part))
                && path > previous,
            "invalid artifact path order",
        )?;
        previous = path;
        integer(&artifact["size_bytes"], 0, MAX_SAFE_INTEGER)?;
        digest(&artifact["sha256"])?;
        if path == "model.safetensors" {
            require(
                artifact["size_bytes"] == 3109054480_u64
                    && artifact["sha256"]
                        == "8c4921ac83182c85190ad10c3aeb118d51e385d2d98ec13b88b514bbf6cac002",
                "vision weight identity mismatch",
            )?;
            weights = true;
        }
    }
    require(weights, "missing vision weight identity")?;
    let execution = &manifest["execution"];
    fields(
        execution,
        &[],
        &[
            "provider",
            "precision",
            "enable_thinking",
            "kv_reuse",
            "max_context",
            "max_output_tokens",
            "prefill_step_size",
            "output_batch_tokens",
            "memory_limit_bytes",
            "selection",
        ],
        &[],
    )?;
    require(
        execution["provider"] == "mlx-vlm-minicpmv47"
            && execution["precision"] == "bf16"
            && execution["enable_thinking"] == false
            && execution["kv_reuse"] == false,
        "unsupported vision execution",
    )?;
    integer(&execution["prefill_step_size"], 256, 256)?;
    integer(&execution["output_batch_tokens"], 1, 1)?;
    integer(&execution["memory_limit_bytes"], 1, MAX_SAFE_INTEGER)?;
    integer(&execution["max_context"], 1, 16384)?;
    integer(&execution["max_output_tokens"], 0, 8192)?;
    require(
        execution["selection"]
            == json!({"algorithm":"mlx-vlm-fixed-controls-v1","top_k":100,"min_p":"0","repetition_penalty":"1.03","repetition_context_size":8192,"presence_penalty":"0","frequency_penalty":"0","seed_policy":"request-id-sha256-v1"}),
        "unsupported vision selection",
    )?;
    let caps = &frame["capabilities"];
    fields(
        caps,
        &[],
        &[
            "text",
            "image",
            "greedy",
            "sampling",
            "token_events",
            "cancel_granularity",
            "max_active_sequences",
            "max_context",
            "max_output_tokens",
            "exact_append",
            "state_schema",
            "media_limits",
        ],
        &[],
    )?;
    for (key, expected) in [
        ("text", true),
        ("image", true),
        ("greedy", true),
        ("sampling", true),
        ("token_events", true),
        ("exact_append", false),
    ] {
        require(
            caps[key].as_bool() == Some(expected),
            "unsupported vision capability",
        )?;
    }
    require(
        caps["cancel_granularity"] == "yield_or_process_exit" && caps["state_schema"] == "none",
        "unsupported vision lifecycle",
    )?;
    integer(&caps["max_active_sequences"], 1, 1)?;
    for key in ["max_context", "max_output_tokens"] {
        require(
            caps[key] == execution[key],
            "vision capability bound mismatch",
        )?;
    }
    require(
        caps["media_limits"]
            == json!({"max_images":4,"max_image_bytes":131072,"max_total_bytes":262144,"max_dimension":1024,"max_host_array_bytes":67108864}),
        "unsupported vision media limits",
    )?;
    require(
        frame["vocab_size"]
            .as_u64()
            .is_some_and(|size| size > 248056),
        "image token outside vocabulary",
    )
}

pub(super) fn replay(frame: &Value) -> Result<()> {
    let replay = &frame["replay"];
    fields(
        replay,
        &[],
        &[
            "worker_epoch",
            "model_revision",
            "capability_revision",
            "preparation_id",
            "rendered_prompt",
            "processor_revision",
            "processing_digest",
            "media",
            "image_bindings",
            "host_array_bytes",
        ],
        &[],
    )?;
    uuid(&replay["worker_epoch"])?;
    digest(&replay["model_revision"])?;
    digest(&replay["capability_revision"])?;
    require(
        replay["worker_epoch"] == frame["worker_epoch"]
            && replay["model_revision"] == frame["model_revision"],
        "replay identity mismatch",
    )?;
    require(
        frame["kind"] != "submit" || replay["capability_revision"] == frame["capability_revision"],
        "replay capability mismatch",
    )?;
    uuid(&replay["preparation_id"])?;
    require(
        if frame["kind"] == "prepared_input" {
            replay["preparation_id"] == frame["command_id"]
        } else {
            replay["preparation_id"] != frame["command_id"]
        },
        "invalid replay command identity",
    )?;
    require(
        string(&replay["rendered_prompt"])?.len() <= 131072,
        "rendered prompt exceeds byte limit",
    )?;
    digest(&replay["processor_revision"])?;
    digest(&replay["processing_digest"])?;
    integer(&replay["host_array_bytes"], 0, 67108864)?;
    media(&replay["media"])?;
    let sources = array(&replay["media"], 0, 4)?;
    let bindings = array(&replay["image_bindings"], sources.len(), sources.len())?;
    let ids = array(&frame["token_ids"], 1, 16384)?;
    let mut previous = 0;
    for (index, binding) in bindings.iter().enumerate() {
        fields(
            binding,
            &[],
            &[
                "source_index",
                "token_start",
                "token_end",
                "pixel_shape",
                "target_size",
            ],
            &[],
        )?;
        integer(&binding["source_index"], index as u64, index as u64)?;
        let start = integer(&binding["token_start"], previous as u64, ids.len() as u64)? as usize;
        let end = integer(&binding["token_end"], start as u64 + 1, ids.len() as u64)? as usize;
        require(
            end - start <= 1024
                && ids[previous..start].iter().all(|id| *id != 248056)
                && ids[start..end].iter().all(|id| *id == 248056),
            "image token correspondence mismatch",
        )?;
        previous = end;
        let target = array(&binding["target_size"], 2, 2)?;
        let height = integer(&target[0], 1, 1024)?;
        let width = integer(&target[1], 1, 1024)?;
        require(height * width <= 4096, "image target exceeds layout limit")?;
        let shape = array(&binding["pixel_shape"], 3, 3)?;
        integer(&shape[0], 3, 3)?;
        integer(&shape[1], 14, 14)?;
        integer(&shape[2], 14 * height * width, 14 * height * width)?;
    }
    require(
        ids[previous..].iter().all(|id| *id != 248056),
        "unbound image token",
    )
}

use super::*;
use crate::decision::{validate_encoding, validate_probabilities, validate_record, DecisionResult};
use serde_json::json;

pub const PROTOCOL: &str = "apxinf-worker/2.6";
pub const LOADER_SHA256: &str = "852223c944819a32fad5cf798d9d1dff30419820eaf5ad1f10cb9698afec97d5";
pub const SOURCE_NAMES: [&str; 6] = [
    "clef_worker.py",
    "clef_runtime.py",
    "clef_identity.py",
    "clef_contracts.py",
    "contracts.py",
    "clef_artifacts.json",
];

pub fn approved_artifacts() -> Value {
    serde_json::from_str(include_str!(
        "../../../../python/apxinf/apxinf/serving/clef_artifacts.json"
    ))
    .expect("The committed Clef artifact pins must be valid JSON.")
}

fn file_record(value: &Value) -> Result<()> {
    fields(value, &[], &["path", "sha256"], &[])?;
    require(
        std::path::Path::new(string(&value["path"])?).is_absolute(),
        "identity path must be absolute",
    )?;
    digest(&value["sha256"])
}

pub fn validate_ready(frame: &Value) -> Result<()> {
    fields(
        frame,
        BASE,
        &[
            "model_revision",
            "capability_revision",
            "model_manifest",
            "capabilities",
            "limits",
            "model_path",
            "runtime",
            "memory",
        ],
        &[],
    )?;
    let runtime = json!({"python":"3.12.10","mlx":"0.32.3","mlx_lm":"0.32.0","transformers":"5.19.0","numpy":"2.5.4","pillow":"12.3.0"});
    require(frame["runtime"] == runtime, "unsupported Clef runtime")?;
    require(
        std::path::Path::new(string(&frame["model_path"])?).is_absolute(),
        "model path must be absolute",
    )?;
    require(
        frame["limits"]
            == json!({"max_frame_bytes":MAX_FRAME_BYTES,"max_depth":MAX_DEPTH,"max_commands":MAX_COMMANDS}),
        "incompatible Clef transport limits",
    )?;
    counts(
        &frame["memory"],
        &["active_bytes", "peak_bytes", "cache_bytes"],
    )?;
    let manifest = &frame["model_manifest"];
    fields(
        manifest,
        &[],
        &[
            "checkpoint",
            "checkpoint_revision",
            "adapter_revision",
            "platform",
            "artifacts",
            "loader",
            "runtime",
            "sources",
            "python",
            "runtime_files",
            "execution",
        ],
        &[],
    )?;
    require(
        manifest["checkpoint"] == "mlx-community/clef-flash-4bit"
            && manifest["checkpoint_revision"] == "b5adbe2c3a95200d4bf04a0727c8ec457157d5e1"
            && manifest["adapter_revision"] == "apxinf-python-clef-text-v1"
            && manifest["platform"] == "macos-metal"
            && manifest["runtime"] == runtime,
        "unsupported Clef identity",
    )?;
    file_record(&manifest["loader"])?;
    require(
        manifest["loader"]["sha256"] == LOADER_SHA256,
        "unsupported Clef loader digest",
    )?;
    file_record(&manifest["python"])?;
    let sources = &manifest["sources"];
    fields(sources, &[], &SOURCE_NAMES, &[])?;
    for (name, record) in object(sources)? {
        file_record(record)?;
        require(
            std::path::Path::new(string(&record["path"])?)
                .file_name()
                .is_some_and(|file| file == name.as_str()),
            "source filename mismatch",
        )?;
    }
    let runtime_files = &manifest["runtime_files"];
    fields(runtime_files, &[], &["modules", "libraries"], &[])?;
    fields(
        &runtime_files["modules"],
        &[],
        &[
            "mlx_core",
            "mlx_lm",
            "transformers",
            "numpy",
            "pillow",
            "tokenizers",
        ],
        &[],
    )?;
    for record in object(&runtime_files["modules"])?.values() {
        file_record(record)?;
    }
    for record in array(&runtime_files["libraries"], 1, 1)? {
        file_record(record)?;
    }
    let mut previous = "";
    require(
        manifest["artifacts"] == approved_artifacts(),
        "Clef checkpoint artifacts differ from approved pins",
    )?;
    for artifact in array(&manifest["artifacts"], 1, 1024)? {
        fields(artifact, &[], &["path", "size_bytes", "sha256"], &[])?;
        let path = string(&artifact["path"])?;
        require(
            path > previous
                && std::path::Path::new(path).components().count() == 1
                && !matches!(path, "." | "..")
                && !path.contains('/')
                && !path.contains('\\')
                && ["json", "safetensors", "jinja"]
                    .contains(&path.rsplit('.').next().unwrap_or("")),
            "invalid Clef artifact path",
        )?;
        previous = path;
        integer(&artifact["size_bytes"], 0, MAX_SAFE_INTEGER)?;
        digest(&artifact["sha256"])?;
    }
    let execution = &manifest["execution"];
    fields(
        execution,
        &[],
        &[
            "provider",
            "precision",
            "backend",
            "max_context",
            "max_output_tokens",
            "idle_cache_limit_bytes",
            "memory_limit_bytes",
            "truncate",
            "kv_reuse",
            "worker_lifetime",
        ],
        &[],
    )?;
    require(
        execution["provider"] == "python-clef"
            && execution["precision"] == "backbone-4bit-head-bf16"
            && execution["backend"] == "lm"
            && execution["truncate"] == false
            && execution["kv_reuse"] == false
            && execution["worker_lifetime"] == "one-attempt",
        "unsupported Clef execution",
    )?;
    integer(&execution["max_context"], 1, 4096)?;
    integer(&execution["max_output_tokens"], 0, 0)?;
    integer(&execution["idle_cache_limit_bytes"], 0, 0)?;
    integer(&execution["memory_limit_bytes"], 1, MAX_SAFE_INTEGER)?;
    let caps = &frame["capabilities"];
    fields(
        caps,
        &[],
        &[
            "workload",
            "text",
            "noul",
            "choice",
            "score",
            "image",
            "streaming",
            "token_events",
            "kv_reuse",
            "state_schema",
            "max_active_requests",
            "vocab_size",
            "max_context",
            "max_output_tokens",
            "cancel_granularity",
        ],
        &[],
    )?;
    for key in ["text", "noul", "choice", "score"] {
        require(caps[key] == true, "missing Clef capability")?;
    }
    for key in ["image", "streaming", "token_events", "kv_reuse"] {
        require(caps[key] == false, "unsupported Clef capability")?;
    }
    require(
        caps["workload"] == "decision"
            && caps["state_schema"] == "none"
            && caps["cancel_granularity"] == "process_exit"
            && caps["max_context"] == execution["max_context"],
        "Clef capability mismatch",
    )?;
    integer(&caps["max_active_requests"], 1, 1)?;
    integer(&caps["max_output_tokens"], 0, 0)?;
    integer(&caps["vocab_size"], 1, 2_147_483_648)?;
    require(
        frame["model_revision"] == canonical_identity_digest("apxinf-model-identity-v1", manifest)?
            && frame["capability_revision"]
                == canonical_identity_digest("apxinf-capability-v1", caps)?,
        "Clef identity digest mismatch",
    )
}

pub fn validate_preparation(value: &Value) -> Result<()> {
    fields(
        value,
        &[],
        &[
            "worker_epoch",
            "request_id",
            "attempt",
            "model_revision",
            "capability_revision",
            "preparation_id",
            "record",
            "encoding",
            "effective_input_tokens",
            "record_digest",
            "encoding_digest",
            "preparation_digest",
        ],
        &[],
    )?;
    for key in ["worker_epoch", "request_id", "preparation_id"] {
        uuid(&value[key])?;
    }
    integer(&value["attempt"], 1, MAX_SAFE_INTEGER)?;
    for key in [
        "model_revision",
        "capability_revision",
        "record_digest",
        "encoding_digest",
        "preparation_digest",
    ] {
        digest(&value[key])?;
    }
    let record = validate_record(&value["record"])?;
    let encoding = validate_encoding(&value["encoding"], &record, 4096, 2_147_483_648)?;
    require(
        value["effective_input_tokens"].as_u64() == Some(encoding.input_ids.len() as u64),
        "Clef preparation count mismatch",
    )?;
    require(
        value["record_digest"]
            == canonical_identity_digest("apxinf-clef-record-v1", &value["record"])?
            && value["encoding_digest"]
                == canonical_identity_digest("apxinf-clef-encoding-v1", &value["encoding"])?,
        "Clef preparation content digest mismatch",
    )?;
    let identity: Map<String, Value> = [
        "worker_epoch",
        "request_id",
        "attempt",
        "model_revision",
        "capability_revision",
        "preparation_id",
        "record_digest",
        "encoding_digest",
    ]
    .iter()
    .map(|key| ((*key).into(), value[*key].clone()))
    .collect();
    require(
        value["preparation_digest"]
            == canonical_identity_digest("apxinf-clef-preparation-v1", &Value::Object(identity))?,
        "Clef preparation digest mismatch",
    )
}

pub fn validate_decision_frame(frame: &Value) -> Result<()> {
    tree(frame, 1, false)?;
    require(
        frame["protocol"] == PROTOCOL,
        "unsupported decision protocol",
    )?;
    uuid(&frame["worker_epoch"])?;
    let kind = string(&frame["kind"])?;
    if kind == "ready" {
        return validate_ready(frame);
    }
    uuid(&frame["command_id"])?;
    if [
        "prepare_decision",
        "prepared_decision",
        "submit_decision",
        "accepted",
        "decision_result",
        "failure",
    ]
    .contains(&kind)
    {
        uuid(&frame["request_id"])?;
        integer(&frame["attempt"], 1, MAX_SAFE_INTEGER)?;
    }
    if ["accepted", "decision_result", "failure"].contains(&kind) {
        integer(&frame["event_seq"], 0, 1)?;
    }
    match kind {
        "prepare_decision" => {
            fields(
                frame,
                REQUEST,
                &["model_revision", "capability_revision", "record"],
                &[],
            )?;
            validate_record(&frame["record"])?;
        }
        "prepared_decision" | "submit_decision" => {
            fields(
                frame,
                REQUEST,
                if kind == "prepared_decision" {
                    &["model_revision", "capability_revision", "preparation"]
                } else {
                    &[
                        "model_revision",
                        "capability_revision",
                        "preparation",
                        "capacity_lease_ids",
                    ]
                },
                &[],
            )?;
            validate_preparation(&frame["preparation"])?;
            for key in [
                "worker_epoch",
                "request_id",
                "attempt",
                "model_revision",
                "capability_revision",
            ] {
                require(
                    frame[key] == frame["preparation"][key],
                    "Clef preparation envelope mismatch",
                )?;
            }
            require(
                (frame["command_id"] == frame["preparation"]["preparation_id"])
                    == (kind == "prepared_decision"),
                "Clef preparation command mismatch",
            )?;
            if kind == "submit_decision" {
                leases(&frame["capacity_lease_ids"])?;
            }
        }
        "accepted" => {
            fields(
                frame,
                EVENT,
                &["command_id", "model_revision", "capability_revision"],
                &[],
            )?;
            integer(&frame["event_seq"], 0, 0)?;
        }
        "decision_result" => {
            fields(
                frame,
                EVENT,
                &[
                    "command_id",
                    "preparation_digest",
                    "probabilities",
                    "usage",
                    "metrics",
                ],
                &[],
            )?;
            integer(&frame["event_seq"], 1, 1)?;
            digest(&frame["preparation_digest"])?;
            for vector in array(&frame["probabilities"], 1, 16)? {
                fields(vector, &[], &["question_id", "option_ids", "values"], &[])?;
                string(&vector["question_id"])?;
                for id in array(&vector["option_ids"], 2, 16)? {
                    string(id)?;
                }
                let mut sum = 0.0;
                for value in array(
                    &vector["values"],
                    vector["option_ids"].as_array().unwrap().len(),
                    vector["option_ids"].as_array().unwrap().len(),
                )? {
                    let p = value
                        .as_f64()
                        .ok_or_else(|| ContractError("expected decision probability".into()))?;
                    require(
                        p.is_finite() && (0.0..=1.0).contains(&p),
                        "invalid decision probability",
                    )?;
                    sum += p;
                }
                require(
                    (sum - 1.0).abs() <= 0.00001,
                    "unnormalized decision probabilities",
                )?;
            }
            counts(&frame["usage"], &["input_tokens", "output_tokens"])?;
            integer(&frame["usage"]["input_tokens"], 1, 4096)?;
            integer(&frame["usage"]["output_tokens"], 0, 0)?;
            counts(&frame["metrics"], &["elapsed_ms", "peak_device_bytes"])?;
        }
        "failure" => {
            fields(frame, EVENT, &["command_id", "code", "message"], &[])?;
            choice(
                &frame["code"],
                &[
                    "invalid_request",
                    "context_limit",
                    "model_unavailable",
                    "worker_lost",
                    "protocol_fault",
                    "internal_error",
                ],
            )?;
            require(
                (1..=4096).contains(&string(&frame["message"])?.len()),
                "invalid Clef failure message",
            )?;
        }
        "shutdown" => fields(frame, COMMAND, &[], &[])?,
        "drained" => {
            fields(frame, COMMAND, &["reason"], &[])?;
            choice(&frame["reason"], &["attempt_complete", "shutdown"])?;
        }
        _ => return Err(ContractError("unknown decision frame kind".into())),
    }
    for key in ["model_revision", "capability_revision"] {
        if let Some(value) = frame.get(key) {
            digest(value)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
enum State {
    New,
    Prepared,
    Submitted,
    Accepted,
    Result,
    Failed,
    Succeeded,
    ReapedFailure,
}

#[derive(Debug, Clone)]
pub struct DecisionTracker {
    epoch: String,
    request_id: String,
    attempt: u64,
    record: Value,
    model_revision: String,
    capability_revision: String,
    preparation: Option<Value>,
    preparation_command: String,
    submission_command: Option<String>,
    state: State,
    drained: bool,
    result: Option<DecisionResult>,
    resources_released: bool,
}

impl DecisionTracker {
    pub fn new(prepare: &Value) -> Result<Self> {
        validate_decision_frame(prepare)?;
        require(
            prepare["kind"] == "prepare_decision",
            "tracker needs preparation command",
        )?;
        Ok(Self {
            epoch: string(&prepare["worker_epoch"])?.into(),
            request_id: string(&prepare["request_id"])?.into(),
            attempt: integer(&prepare["attempt"], 1, MAX_SAFE_INTEGER)?,
            record: prepare["record"].clone(),
            model_revision: string(&prepare["model_revision"])?.into(),
            capability_revision: string(&prepare["capability_revision"])?.into(),
            preparation: None,
            preparation_command: string(&prepare["command_id"])?.into(),
            submission_command: None,
            state: State::New,
            drained: false,
            result: None,
            resources_released: false,
        })
    }
    pub fn observe(&mut self, frame: &Value) -> Result<()> {
        let mut next = self.clone();
        next.apply(frame)?;
        *self = next;
        Ok(())
    }
    fn apply(&mut self, frame: &Value) -> Result<()> {
        validate_decision_frame(frame)?;
        require(
            frame["worker_epoch"] == self.epoch && !self.drained,
            "stale or drained decision event",
        )?;
        let kind = string(&frame["kind"])?;
        if kind == "drained" {
            require(
                matches!(self.state, State::Result | State::Failed)
                    && frame["reason"] == "attempt_complete"
                    && frame["command_id"]
                        == self
                            .submission_command
                            .as_deref()
                            .unwrap_or(&self.preparation_command),
                "unexpected Clef drain",
            )?;
            self.drained = true;
            return Ok(());
        }
        require(
            frame["request_id"] == self.request_id && frame["attempt"] == self.attempt,
            "stale Clef attempt",
        )?;
        match kind {
            "prepared_decision" => {
                require(
                    self.state == State::New
                        && frame["command_id"] == self.preparation_command
                        && frame["model_revision"] == self.model_revision
                        && frame["capability_revision"] == self.capability_revision
                        && frame["preparation"]["record"] == self.record,
                    "unexpected Clef preparation",
                )?;
                self.preparation = Some(frame["preparation"].clone());
                self.state = State::Prepared;
            }
            "submit_decision" => {
                require(
                    self.state == State::Prepared
                        && self.preparation.as_ref() == Some(&frame["preparation"]),
                    "unexpected Clef submission",
                )?;
                self.submission_command = Some(string(&frame["command_id"])?.into());
                self.state = State::Submitted;
            }
            "accepted" => {
                require(
                    self.state == State::Submitted
                        && frame["command_id"] == self.submission_command.as_deref().unwrap()
                        && frame["model_revision"] == self.model_revision
                        && frame["capability_revision"] == self.capability_revision,
                    "unexpected Clef acceptance",
                )?;
                self.state = State::Accepted;
            }
            "decision_result" => {
                require(
                    self.state == State::Accepted
                        && frame["command_id"] == self.submission_command.as_deref().unwrap(),
                    "unexpected Clef result",
                )?;
                let preparation = self.preparation.as_ref().unwrap();
                require(
                    frame["preparation_digest"] == preparation["preparation_digest"]
                        && frame["usage"]["input_tokens"] == preparation["effective_input_tokens"],
                    "Clef result preparation mismatch",
                )?;
                let record = validate_record(&preparation["record"])?;
                self.result = Some(DecisionResult {
                    probabilities: validate_probabilities(&frame["probabilities"], &record)?,
                    input_tokens: frame["usage"]["input_tokens"].as_u64().unwrap(),
                });
                self.state = State::Result;
            }
            "failure" => {
                require(
                    matches!(
                        self.state,
                        State::New | State::Prepared | State::Submitted | State::Accepted
                    ),
                    "unexpected Clef failure",
                )?;
                let accepted = self.state == State::Accepted;
                require(
                    frame["event_seq"] == u64::from(accepted)
                        && frame["command_id"]
                            == self
                                .submission_command
                                .as_deref()
                                .unwrap_or(&self.preparation_command),
                    "Clef failure identity mismatch",
                )?;
                self.state = State::Failed;
            }
            _ => return Err(ContractError("unexpected decision tracker frame".into())),
        }
        Ok(())
    }
    pub fn reaped(&mut self, exit_code: i32, eof: bool) -> Result<()> {
        self.resources_released = true;
        if self.state == State::Failed && self.drained && eof {
            self.state = State::ReapedFailure;
            return Ok(());
        }
        if self.state == State::Result && self.drained && eof && exit_code == 0 {
            self.state = State::Succeeded;
            return Ok(());
        }
        self.result = None;
        self.state = State::ReapedFailure;
        Err(ContractError(
            "decision attempt did not settle successfully".into(),
        ))
    }
    pub fn result(&self) -> Option<&DecisionResult> {
        if self.state == State::Succeeded {
            self.result.as_ref()
        } else {
            None
        }
    }
    pub fn resources_released(&self) -> bool {
        self.resources_released
    }
}

#[cfg(test)]
mod tests;

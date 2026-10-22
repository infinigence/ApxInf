//! One native model owner with bounded command and control queues.
use apxinf_core::{DType, Device, Error};
use apxinf_model::{
    AutoModel, GenerationBoundary, GenerationOptions, GenerationRequest, LlmInput, LoadOptions,
};
use apxinf_serving::{contracts, native_identity};
use apxinf_tokenizer::Tokenizer;
use clap::Parser;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    io::{BufRead, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{self, Receiver},
        Arc,
    },
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    protocol: String,
    #[arg(long)]
    worker_epoch: String,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    metal_library: PathBuf,
    #[arg(long)]
    max_context: usize,
    #[arg(long)]
    max_output_tokens: usize,
    #[arg(long)]
    memory_limit_bytes: u64,
    #[arg(long)]
    prefill_step_size: usize,
    #[arg(long)]
    output_batch_tokens: usize,
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("Native worker failed: {error}");
        std::process::exit(1);
    }
}

fn memory() -> Result<Value> {
    let m = apxinf_mlx::memory_stats()?;
    Ok(json!({"active_bytes":m.active_bytes,"cache_bytes":m.cache_bytes,"peak_bytes":m.peak_bytes}))
}

struct Attempt {
    command: Value,
    sequence: u64,
    output_count: usize,
    text_bytes: usize,
    consumed: usize,
    started: Instant,
    first_token_ns: u64,
    stop: Option<(String, String)>,
}

struct Outbox {
    sender: Option<mpsc::SyncSender<Vec<u8>>>,
    pending: Arc<AtomicUsize>,
    fault: Arc<AtomicBool>,
    done: Receiver<()>,
}

impl Outbox {
    fn start(mut sink: impl Write + Send + 'static, fault: Arc<AtomicBool>) -> Self {
        let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(64);
        let (finished, done) = mpsc::sync_channel(1);
        let pending = Arc::new(AtomicUsize::new(0));
        let queued = pending.clone();
        let failed = fault.clone();
        std::thread::spawn(move || {
            while let Ok(bytes) = receiver.recv() {
                if sink.write_all(&bytes).and_then(|()| sink.flush()).is_err() {
                    failed.store(true, Ordering::Release);
                    return;
                }
                queued.fetch_sub(bytes.len(), Ordering::AcqRel);
            }
            let _ = finished.send(());
        });
        Self {
            sender: Some(sender),
            pending,
            fault,
            done,
        }
    }

    fn send(&self, bytes: Vec<u8>, critical: bool) -> Result<()> {
        if self.fault.load(Ordering::Acquire) {
            return Err("Worker transport failed".into());
        }
        let limit = contracts::MAX_FRAME_BYTES * if critical { 4 } else { 3 };
        let count = bytes.len();
        if self
            .pending
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                n.checked_add(count).filter(|total| *total <= limit)
            })
            .is_err()
        {
            self.fault.store(true, Ordering::Release);
            return Err("Worker output byte limit exceeded".into());
        }
        if self
            .sender
            .as_ref()
            .ok_or("Output closed")?
            .try_send(bytes)
            .is_err()
        {
            self.pending.fetch_sub(count, Ordering::AcqRel);
            self.fault.store(true, Ordering::Release);
            return Err("Worker output queue exhausted".into());
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.sender.take();
        self.done.recv_timeout(Duration::from_secs(2))?;
        if self.fault.load(Ordering::Acquire) {
            return Err("Worker output failed".into());
        }
        Ok(())
    }
}

struct State {
    output: Outbox,
    epoch: String,
    controls: Receiver<Value>,
    fault: Arc<AtomicBool>,
    records: HashMap<String, ([u8; 32], Option<Value>)>,
    attempts: HashMap<(String, u64), [u8; 32]>,
    active: Option<Attempt>,
    draining: bool,
    shutdown: bool,
    completed: usize,
    cancelled: usize,
}

impl State {
    fn frame(&self, kind: &str) -> Value {
        json!({"protocol":contracts::NATIVE_PROTOCOL,"kind":kind,"worker_epoch":self.epoch})
    }
    fn send(&self, value: Value) -> Result<()> {
        let bytes = contracts::encode_frame(&value)?;
        let critical = !matches!(
            value["kind"].as_str(),
            Some("tokens" | "prefill_progress" | "prepared_input")
        );
        self.output.send(bytes, critical)
    }
    fn reply(&mut self, command: &Value, status: &str, error: Option<Value>) -> Result<()> {
        let mut value = self.frame("command_result");
        value["command_id"] = command["command_id"].clone();
        value["status"] = status.into();
        if let Some(error) = error {
            value["error"] = error;
        }
        self.send(value.clone())?;
        if let Some(record) = self
            .records
            .get_mut(command["command_id"].as_str().unwrap())
        {
            record.1 = Some(value);
        }
        Ok(())
    }
    fn register(&mut self, command: &Value) -> Result<bool> {
        let id = command["command_id"]
            .as_str()
            .ok_or("Missing command ID")?
            .to_owned();
        let digest: [u8; 32] = Sha256::digest(serde_json::to_vec(command)?).into();
        if let Some((previous, reply)) = self.records.get(&id) {
            if *previous != digest {
                let mut rejected = self.frame("command_result");
                rejected["command_id"] = command["command_id"].clone();
                rejected["status"] = "error".into();
                rejected["error"] = error("invalid_request", "Conflicting command ID.");
                self.send(rejected)?;
            } else if matches!(
                command["kind"].as_str(),
                Some("cancel_request" | "stop_generation")
            ) && self.attempts.contains_key(&key(command))
                && !self
                    .active
                    .as_ref()
                    .is_some_and(|a| key(&a.command) == key(command))
            {
                self.reply(command, "already_terminal", None)?;
            } else if let Some(reply) = reply.clone() {
                self.send(reply)?;
            } else {
                self.reply(command, "already_registered", None)?;
            }
            return Ok(false);
        }
        if self.records.len() >= contracts::MAX_COMMANDS as usize {
            return Err("Command history exhausted".into());
        }
        self.records.insert(id, (digest, None));
        Ok(true)
    }

    fn register_attempt(&mut self, command: &Value) -> Result<bool> {
        let mut payload = command.clone();
        payload.as_object_mut().unwrap().remove("command_id");
        let fingerprint: [u8; 32] = Sha256::digest(serde_json::to_vec(&payload)?).into();
        if let Some(previous) = self.attempts.get(&key(command)) {
            if *previous == fingerprint {
                self.reply(command, "already_registered", None)?;
            } else {
                self.reply(
                    command,
                    "error",
                    Some(error("invalid_request", "Conflicting attempt.")),
                )?;
            }
            return Ok(false);
        }
        self.attempts.insert(key(command), fingerprint);
        Ok(true)
    }
    fn event(&mut self, kind: &str, fields: Value) -> Result<()> {
        let mut value = self.frame(kind);
        let a = self.active.as_mut().ok_or("No active attempt")?;
        value["request_id"] = a.command["request_id"].clone();
        value["attempt"] = a.command["attempt"].clone();
        value["event_seq"] = a.sequence.into();
        value
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        a.sequence += 1;
        self.send(value)
    }
    fn control(&mut self, command: Value) -> Result<()> {
        if !self.register(&command)? {
            return Ok(());
        }
        match command["kind"].as_str().unwrap() {
            "drain" | "shutdown" => {
                self.draining = true;
                self.shutdown |= command["kind"] == "shutdown";
                self.reply(&command, "accepted", None)
            }
            "query_stats" => {
                let mut frame = self.frame("stats");
                frame["command_id"] = command["command_id"].clone();
                frame["active_requests"] = usize::from(self.active.is_some()).into();
                frame["completed_requests"] = self.completed.into();
                frame["cancelled_requests"] = self.cancelled.into();
                self.send(frame.clone())?;
                self.records
                    .get_mut(command["command_id"].as_str().unwrap())
                    .unwrap()
                    .1 = Some(frame);
                Ok(())
            }
            "cancel_request" | "stop_generation" => {
                let attempt_key = key(&command);
                if self
                    .active
                    .as_ref()
                    .is_some_and(|a| key(&a.command) == attempt_key)
                {
                    let a = self.active.as_mut().unwrap();
                    if command["kind"] == "stop_generation"
                        && (command["output_token_count"].as_u64().unwrap() > a.output_count as u64
                            || command["text_byte_cutoff"].as_u64().unwrap() > a.text_bytes as u64)
                    {
                        return self.reply(
                            &command,
                            "error",
                            Some(error("invalid_request", "Unavailable output boundary.")),
                        );
                    }
                    if a.stop.is_none() {
                        a.stop = Some(if command["kind"] == "stop_generation" {
                            (
                                "completed".into(),
                                command["cause"].as_str().unwrap().into(),
                            )
                        } else if command["reason"] == "deadline_exceeded" {
                            ("expired".into(), "deadline_exceeded".into())
                        } else {
                            (
                                "cancelled".into(),
                                command["reason"].as_str().unwrap().into(),
                            )
                        });
                    }
                    self.reply(&command, "accepted", None)
                } else if self.attempts.contains_key(&attempt_key) {
                    self.reply(&command, "already_terminal", None)
                } else {
                    self.reply(
                        &command,
                        "error",
                        Some(error("invalid_request", "Unknown attempt.")),
                    )
                }
            }
            _ => Err("Unexpected control command".into()),
        }
    }
    fn check_controls(&mut self) -> Result<bool> {
        if self.fault.load(Ordering::Acquire) {
            return Err("Input transport failed".into());
        }
        while let Ok(command) = self.controls.try_recv() {
            self.control(command)?;
        }
        if let Some(a) = &mut self.active {
            if a.stop.is_none()
                && a.started.elapsed()
                    >= Duration::from_millis(a.command["remaining_timeout_ms"].as_u64().unwrap())
            {
                a.stop = Some(("expired".into(), "deadline_exceeded".into()));
            }
            return Ok(a.stop.is_none());
        }
        Ok(true)
    }
}

fn key(command: &Value) -> (String, u64) {
    (
        command["request_id"].as_str().unwrap().into(),
        command["attempt"].as_u64().unwrap(),
    )
}
fn error(code: &str, message: &str) -> Value {
    json!({"code":code,"message":message,"scope":"request","state_validity":"none"})
}

struct Incoming {
    value: Value,
    valid: bool,
}

fn read_commands(
    epoch: String,
    fault: Arc<AtomicBool>,
    commands: mpsc::SyncSender<Incoming>,
    controls: mpsc::SyncSender<Value>,
) {
    let result = (|| -> Result<()> {
        let mut source = std::io::stdin().lock();
        loop {
            let mut bytes = Vec::new();
            loop {
                let available = source.fill_buf()?;
                if available.is_empty() {
                    return Err("Input closed".into());
                }
                let end = available.iter().position(|b| *b == b'\n').map(|p| p + 1);
                let n = end.unwrap_or(available.len());
                if bytes.len() + n > contracts::MAX_FRAME_BYTES {
                    return Err("Input frame too large".into());
                }
                bytes.extend_from_slice(&available[..n]);
                source.consume(n);
                if end.is_some() {
                    break;
                }
            }
            let command = contracts::parse_document(&bytes)?;
            contracts::validate_command_envelope(&command)?;
            if command["protocol"] != contracts::NATIVE_PROTOCOL || command["worker_epoch"] != epoch
            {
                return Err("Input identity mismatch".into());
            }
            let valid = contracts::validate_frame(&command).is_ok();
            let shutdown = valid && command["kind"] == "shutdown";
            if !valid || matches!(command["kind"].as_str(), Some("submit" | "prepare_input")) {
                commands.try_send(Incoming {
                    value: command,
                    valid,
                })?;
            } else {
                controls.try_send(command)?;
            }
            if shutdown {
                return Ok(());
            }
        }
    })();
    if let Err(error) = result {
        eprintln!("Native input stopped: {error}");
        fault.store(true, Ordering::Release);
    }
}

fn run(args: Args) -> Result<()> {
    if args.protocol != contracts::NATIVE_PROTOCOL
        || !(1..=16384).contains(&args.max_context)
        || args.prefill_step_size != 256
        || args.output_batch_tokens != 1
        || args.max_output_tokens > 65536
    {
        return Err("Unsupported native launch profile".into());
    }
    let loaded_library = std::fs::canonicalize(apxinf_mlx::loaded_library_path()?)?;
    if loaded_library != std::fs::canonicalize(&args.metal_library)? {
        return Err("Loaded MLX library differs from the selected runtime.".into());
    }
    let runtime = native_identity::runtime(&std::env::current_exe()?, &loaded_library)?;
    let manifest = native_identity::manifest(
        &args.model,
        runtime.clone(),
        native_identity::execution(
            args.max_context,
            args.max_output_tokens,
            args.memory_limit_bytes,
        ),
    )?;
    let model_revision =
        contracts::canonical_identity_digest("apxinf-model-identity-v1", &manifest)?;
    let capabilities = json!({"text":true,"image":false,"greedy":true,"sampling":false,"token_events":true,
        "cancel_granularity":"token_or_prefill_chunk","max_active_sequences":1,"max_context":args.max_context,
        "max_output_tokens":args.max_output_tokens,"exact_append":false,"state_schema":"none"});
    let capability_revision =
        contracts::canonical_identity_digest("apxinf-capability-v1", &capabilities)?;
    apxinf_mlx::set_memory_limit(usize::try_from(args.memory_limit_bytes)?)?;
    let mut model = AutoModel::load_model(
        Device::Metal(0),
        &args.model,
        &LoadOptions {
            model_name: Some("metal_minicpm5".into()),
            model_variant: Some("bf16-compiled".into()),
            text_weight_dtype: Some(DType::BF16),
            ..Default::default()
        },
    )?;
    let tokenizer = Tokenizer::from_file(args.model.join("tokenizer.json"))?;
    let eos = model
        .generation_defaults()?
        .eos_token_ids
        .clone()
        .ok_or("Checkpoint has no EOS IDs")?;
    if !eos.contains(&1) || !eos.contains(&130073) {
        return Err("Unexpected MiniCPM EOS identity".into());
    }
    model.text_mut()?.backend().synchronize()?;
    apxinf_mlx::clear_cache()?;
    let resident = apxinf_mlx::memory_stats()?;
    if resident.active_bytes as u64 > args.memory_limit_bytes {
        return Err("Resident model exceeds memory limit".into());
    }
    let (command_tx, commands) = mpsc::sync_channel(8);
    let (control_tx, controls) = mpsc::sync_channel(32);
    let fault = Arc::new(AtomicBool::new(false));
    let mut state = State {
        output: Outbox::start(std::io::stdout(), fault.clone()),
        epoch: args.worker_epoch.clone(),
        controls,
        fault: fault.clone(),
        records: HashMap::new(),
        attempts: HashMap::new(),
        active: None,
        draining: false,
        shutdown: false,
        completed: 0,
        cancelled: 0,
    };
    let mut ready = state.frame("ready");
    ready.as_object_mut().unwrap().extend(json!({"model_revision":model_revision,"capability_revision":capability_revision,"model_manifest":manifest,
        "capabilities":capabilities,"limits":{"max_frame_bytes":contracts::MAX_FRAME_BYTES,"max_depth":contracts::MAX_DEPTH,"max_commands":contracts::MAX_COMMANDS},
        "model_path":args.model,"eos_token_ids":eos,"vocab_size":tokenizer.vocab_size(),"runtime":runtime,"memory":memory()?}).as_object().unwrap().clone());
    state.send(ready)?;
    let epoch = args.worker_epoch.clone();
    std::thread::spawn(move || read_commands(epoch, fault, command_tx, control_tx));
    let mut drained = false;
    let mut pending = None;
    loop {
        match pending
            .take()
            .map_or_else(|| commands.recv_timeout(Duration::from_millis(10)), Ok)
        {
            Ok(Incoming {
                value: command,
                valid,
            }) => {
                if !state.register(&command)? {
                    continue;
                }
                if !valid {
                    reject_invalid(&mut state, command)?;
                    continue;
                }
                if command["kind"] == "prepare_input" {
                    let prepared = prepare(&tokenizer, &command, &model_revision, args.max_context);
                    match prepared {
                        Ok(tokens) => {
                            let mut frame = state.frame("prepared_input");
                            frame.as_object_mut().unwrap().extend(json!({"command_id":command["command_id"],"model_revision":model_revision,
                                "effective_input_tokens":tokens.len(),"prompt_digest":contracts::token_prefix_digest(&tokens)?,"token_ids":tokens}).as_object().unwrap().clone());
                            state.send(frame)?;
                        }
                        Err(failure) => state.reply(&command, "error", Some(failure))?,
                    }
                } else {
                    submit(
                        &mut state,
                        &mut model,
                        &tokenizer,
                        command,
                        &model_revision,
                        &capability_revision,
                        &eos,
                        &args,
                    )?;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                state.check_controls()?;
                if !state.shutdown {
                    return Err("Input reader closed".into());
                }
            }
        }
        state.check_controls()?;
        if state.draining && !drained {
            pending = commands.try_recv().ok();
            if pending.is_none() {
                state.send(state.frame("drained"))?;
                drained = true;
            }
        }
        if state.shutdown && drained {
            state.output.finish()?;
            return Ok(());
        }
    }
}

fn reject_invalid(state: &mut State, command: Value) -> Result<()> {
    let failure = error(
        "invalid_request",
        "Command content does not match the native profile.",
    );
    if command["kind"] != "submit" {
        return state.reply(&command, "error", Some(failure));
    }
    if state.attempts.contains_key(&key(&command)) {
        return state.reply(&command, "error", Some(failure));
    }
    let mut payload = command.clone();
    payload.as_object_mut().unwrap().remove("command_id");
    state.attempts.insert(
        key(&command),
        Sha256::digest(serde_json::to_vec(&payload)?).into(),
    );
    let count = command["token_ids"].as_array().map_or(0, Vec::len);
    let leases = command["capacity_lease_ids"].clone();
    state.active = Some(Attempt {
        command,
        sequence: 0,
        output_count: 0,
        text_bytes: 0,
        consumed: 0,
        started: Instant::now(),
        first_token_ns: 0,
        stop: None,
    });
    state.event("rejected", json!({"error":failure}))?;
    state.event("terminal",json!({"status":"failed","cause":"invalid_request","error":failure,
        "usage":{"input_tokens":count,"output_tokens":0},"state_result":{"validity":"none","consumed_position":0,"history_count":count},
        "text_delta":"","metrics":{"elapsed_ns":0,"ttft_ns":0,"peak_memory_bytes":0}}))?;
    state.event(
        "resources_released",
        json!({"released_lease_ids":leases,"retained_lease_ids":[]}),
    )?;
    state.active = None;
    Ok(())
}

fn prepare(
    tokenizer: &Tokenizer,
    command: &Value,
    revision: &str,
    limit: usize,
) -> std::result::Result<Vec<u32>, Value> {
    if command["model_revision"] != revision
        || command["template_options"]["enable_thinking"] != false
    {
        return Err(error("invalid_request", "Preparation identity differs."));
    }
    let mut messages = command["messages"].as_array().unwrap().clone();
    for message in &mut messages {
        if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
            for call in calls {
                if let Some(text) = call["function"]["arguments"].as_str() {
                    call["function"]["arguments"] = contracts::parse_document(text.as_bytes())
                        .map_err(|e| error("invalid_request", &e.to_string()))?;
                }
                if !call["function"]["arguments"].is_object() {
                    return Err(error(
                        "invalid_request",
                        "Tool arguments must be an object.",
                    ));
                }
            }
        }
    }
    let options = json!({"enable_thinking":false,"tools":command["tools"]});
    let prompt = tokenizer
        .apply_rich_chat_template(&messages, options.as_object().unwrap())
        .map_err(|e| error("invalid_request", &e.to_string()))?;
    let tokens = tokenizer
        .encode(&prompt)
        .map_err(|e| error("invalid_request", &e.to_string()))?;
    if tokens.is_empty() || tokens.len() > limit {
        return Err(error(
            "context_limit",
            "Prepared input exceeds the context limit.",
        ));
    }
    Ok(tokens)
}

#[allow(clippy::too_many_arguments)]
fn submit(
    state: &mut State,
    model: &mut apxinf_model::LoadedModel,
    tokenizer: &Tokenizer,
    command: Value,
    model_revision: &str,
    capability_revision: &str,
    eos: &[u32],
    args: &Args,
) -> Result<()> {
    if !state.register_attempt(&command)? {
        return Ok(());
    }
    let tokens: Vec<u32> = command["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let maximum = command["max_tokens"].as_u64().unwrap() as usize;
    let invalid = command["model_revision"] != model_revision
        || command["capability_revision"] != capability_revision
        || command["eos_token_ids"] != json!(eos)
        || maximum > args.max_output_tokens
        || tokens
            .len()
            .checked_add(maximum)
            .is_none_or(|n| n > args.max_context)
        || tokens.iter().any(|t| *t as usize >= tokenizer.vocab_size())
        || state.draining;
    state.active = Some(Attempt {
        command,
        sequence: 0,
        output_count: 0,
        text_bytes: 0,
        consumed: 0,
        started: Instant::now(),
        first_token_ns: 0,
        stop: None,
    });
    let mut failure = None;
    let mut tail = String::new();
    let mut mutated = false;
    if invalid {
        failure = Some(error(
            "invalid_request",
            "Submission differs from native capabilities.",
        ));
        state.event("rejected", json!({"error":failure}))?;
    } else {
        state.event("accepted", json!({}))?;
        if state.check_controls()? && maximum > 0 {
            mutated = true;
            let mut output_ids = Vec::new();
            let mut emitted = String::new();
            let mut options = GenerationOptions::greedy(maximum, None);
            options.eos_token_ids = Some(eos.to_vec());
            let generated = model.text_mut()?.generate_streaming_controlled_dyn(GenerationRequest {input:LlmInput::text(&tokens), options:&options}, &mut |boundary| {
                let result = (|| -> Result<bool> {
                    match boundary {
                        GenerationBoundary::PrefillStart => {}
                        GenerationBoundary::PrefillProgress {consumed_tokens} => {
                            state.active.as_mut().unwrap().consumed = consumed_tokens;
                            state.event("prefill_progress",json!({"consumed_tokens":consumed_tokens}))?;
                        }
                        GenerationBoundary::Token(token) => {
                            let terminal = eos.contains(&token.token_id);
                            if !terminal { output_ids.push(token.token_id); }
                            let decoded = tokenizer.decode_with_options(&output_ids,false)?;
                            let text = if decoded.ends_with('\u{fffd}') { String::new() } else {
                                let suffix = decoded.strip_prefix(&emitted).ok_or("Tokenizer changed emitted text")?.to_owned();
                                emitted = decoded;
                                suffix
                            };
                            let a = state.active.as_mut().unwrap();
                            if a.output_count == 0 { a.first_token_ns = a.started.elapsed().as_nanos() as u64; }
                            let index = a.output_count;
                            state.event("tokens",json!({"output_index":index,"token_ids":[token.token_id],"text_delta":text}))?;
                            let a = state.active.as_mut().unwrap();
                            a.output_count += 1;
                            a.text_bytes += text.len();
                            a.consumed = tokens.len()+a.output_count-1;
                            if terminal { a.stop = Some(("completed".into(),"eos".into())); }
                        }
                    }
                    let m = apxinf_mlx::memory_stats()?;
                    if (m.active_bytes+m.cache_bytes) as u64 > args.memory_limit_bytes { return Err("Native memory limit exceeded".into()); }
                    state.check_controls()
                })();
                result.map_err(|e|Error::Other(e.to_string()))
            });
            if let Err(error_value) = generated {
                eprintln!("Native generation failed: {error_value}");
                failure = Some(error("internal_error", "Native model execution failed."));
            } else {
                let decoded = tokenizer.decode_with_options(&output_ids, false)?;
                tail = decoded
                    .strip_prefix(&emitted)
                    .ok_or("Tokenizer changed final text")?
                    .to_owned();
            }
        }
    }
    let a = state.active.as_ref().unwrap();
    let (status, cause) = if failure.is_some() {
        ("failed".into(), "invalid_request".into())
    } else {
        a.stop
            .clone()
            .unwrap_or(("completed".into(), "length".into()))
    };
    let cause = failure
        .as_ref()
        .map_or(cause, |e| e["code"].as_str().unwrap().to_owned());
    let mut terminal = json!({"status":status,"cause":cause,"usage":{"input_tokens":tokens.len(),"output_tokens":a.output_count},
        "state_result":{"validity":"none","consumed_position":a.consumed,"history_count":tokens.len()+a.output_count},"text_delta":tail,
        "metrics":{"elapsed_ns":a.started.elapsed().as_nanos() as u64,"ttft_ns":a.first_token_ns,"peak_memory_bytes":apxinf_mlx::memory_stats()?.peak_bytes}});
    if let Some(failure) = failure {
        terminal["error"] = failure;
    }
    state.event("terminal", terminal)?;
    if mutated {
        model.text_mut()?.release_request_state()?;
        apxinf_mlx::clear_cache()?;
    }
    eprintln!("native_request_settled {}", memory()?);
    let leases = state.active.as_ref().unwrap().command["capacity_lease_ids"].clone();
    state.event(
        "resources_released",
        json!({"released_lease_ids":leases,"retained_lease_ids":[]}),
    )?;
    if status == "completed" {
        state.completed += 1;
    }
    if status == "cancelled" {
        state.cancelled += 1;
    }
    state.active = None;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<u8>>>);
    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    fn state() -> (State, mpsc::SyncSender<Value>, Capture) {
        let (sender, controls) = mpsc::sync_channel(32);
        let capture = Capture::default();
        let fault = Arc::new(AtomicBool::new(false));
        (
            State {
                output: Outbox::start(capture.clone(), fault.clone()),
                epoch: "00000000-0000-4000-8000-000000000001".into(),
                controls,
                fault,
                records: HashMap::new(),
                attempts: HashMap::new(),
                active: None,
                draining: false,
                shutdown: false,
                completed: 0,
                cancelled: 0,
            },
            sender,
            capture,
        )
    }
    fn fixture(name: &str) -> Value {
        let corpus: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/serving/serial-v0.1.json"
        ))
        .unwrap();
        let mut value = corpus["templates"][name].clone();
        value["protocol"] = contracts::NATIVE_PROTOCOL.into();
        value
    }
    fn frames(capture: Capture) -> Vec<Value> {
        capture
            .0
            .lock()
            .unwrap()
            .split_inclusive(|b| *b == b'\n')
            .map(|b| contracts::decode_frame(b).unwrap())
            .collect()
    }
    #[test]
    fn invalid_submit_rejects_then_settles_without_execution() {
        let (mut state, _controls, capture) = state();
        let mut submit = fixture("submit");
        state.epoch = submit["worker_epoch"].as_str().unwrap().into();
        submit["max_tokens"] = true.into();
        reject_invalid(&mut state, submit.clone()).unwrap();
        state.output.finish().unwrap();
        let events = frames(capture);
        assert_eq!(
            events
                .iter()
                .map(|e| e["kind"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["rejected", "terminal", "resources_released"]
        );
        assert_eq!(
            events[2]["released_lease_ids"],
            submit["capacity_lease_ids"]
        );
        assert_eq!(events[1]["state_result"]["consumed_position"], 0);
    }
    #[test]
    fn cancellation_is_applied_at_a_boundary_and_fenced_by_attempt() {
        let (mut state, controls, capture) = state();
        let submit = fixture("submit");
        state.epoch = submit["worker_epoch"].as_str().unwrap().into();
        state.register_attempt(&submit).unwrap();
        state.active = Some(Attempt {
            command: submit.clone(),
            sequence: 0,
            output_count: 0,
            text_bytes: 0,
            consumed: 0,
            started: Instant::now(),
            first_token_ns: 0,
            stop: None,
        });
        let mut cancel = state.frame("cancel_request");
        cancel["command_id"] = uuid::Uuid::new_v4().to_string().into();
        cancel["request_id"] = submit["request_id"].clone();
        cancel["attempt"] = submit["attempt"].clone();
        cancel["reason"] = "user_cancel".into();
        controls.send(cancel.clone()).unwrap();
        assert!(!state.check_controls().unwrap());
        assert_eq!(
            state.active.as_ref().unwrap().stop,
            Some(("cancelled".into(), "user_cancel".into()))
        );
        state.active = None;
        state.control(cancel).unwrap();
        state.output.finish().unwrap();
        let events = frames(capture);
        assert_eq!(events[0]["status"], "accepted");
        assert_eq!(events[1]["status"], "already_terminal");
    }

    #[test]
    fn duplicate_attempt_ignores_a_new_command_id_but_rejects_new_payload() {
        let (mut state, _controls, capture) = state();
        let mut command = fixture("submit");
        assert!(state.register_attempt(&command).unwrap());
        command["command_id"] = uuid::Uuid::new_v4().to_string().into();
        assert!(!state.register_attempt(&command).unwrap());
        command["max_tokens"] = 99.into();
        assert!(!state.register_attempt(&command).unwrap());
        state.output.finish().unwrap();
        let events = frames(capture);
        assert_eq!(events[0]["status"], "already_registered");
        assert_eq!(events[1]["status"], "error");
        assert!(state.active.is_none());
    }
    #[test]
    fn preparation_preserves_tool_arguments_and_reports_context_limit() {
        let tokenizer = Tokenizer::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/serving/native-tokenizer/tokenizer.json"
        ))
        .unwrap();
        let command = json!({"model_revision":"revision","template_options":{"enable_thinking":false},"tools":[],"messages":[
            {"role":"assistant","content":"hello","tool_calls":[{"id":"a","type":"function","function":{"name":"Read","arguments":{"path":"brief"}}}]},
            {"role":"tool","tool_call_id":"a","content":"hello"}]});
        assert_eq!(
            prepare(&tokenizer, &command, "revision", 3).unwrap(),
            vec![1, 2, 1]
        );
        assert_eq!(
            prepare(&tokenizer, &command, "revision", 2).unwrap_err()["code"],
            "context_limit"
        );
        assert_eq!(
            prepare(&tokenizer, &command, "other", 3).unwrap_err()["code"],
            "invalid_request"
        );
    }
    #[test]
    fn output_backpressure_returns_without_waiting_for_writer() {
        struct Blocked(Receiver<()>);
        impl Write for Blocked {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                let _ = self.0.recv();
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let (release, wait) = mpsc::channel();
        let fault = Arc::new(AtomicBool::new(false));
        let mut output = Outbox::start(Blocked(wait), fault.clone());
        output
            .send(vec![0; 3 * contracts::MAX_FRAME_BYTES], false)
            .unwrap();
        assert!(output.send(vec![0], false).is_err());
        assert!(fault.load(Ordering::Acquire));
        drop(release);
        output.sender.take();
        output.done.recv_timeout(Duration::from_secs(1)).unwrap();
    }
}

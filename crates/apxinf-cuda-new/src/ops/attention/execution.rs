#[cfg(test)]
use std::ffi::CStr;
use std::ffi::CString;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Error, Result};

use super::contracts::Normalized;
use crate::ffi::abi::{attention as abi, status};
use crate::{CudaBuffer, CudaContext};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExecutionKey {
    spec: abi::Spec,
    device: usize,
    query: usize,
    key: usize,
    value: usize,
    offsets: usize,
    output: usize,
    stream: usize,
    scale: u32,
    output_scale: u32,
    workspace_limit: u64,
    graph_safe: bool,
    deterministic: bool,
}

impl ExecutionKey {
    fn new(ctx: &CudaContext, normalized: &Normalized) -> Self {
        Self {
            spec: normalized.spec,
            device: ctx.device_id(),
            query: normalized.bindings.query as usize,
            key: normalized.bindings.key as usize,
            value: normalized.bindings.value as usize,
            offsets: normalized.bindings.offsets as usize,
            output: normalized.bindings.output as usize,
            stream: normalized.bindings.stream as usize,
            scale: normalized.bindings.scale.to_bits(),
            output_scale: normalized.bindings.output_scale.to_bits(),
            workspace_limit: normalized.policy.workspace_limit as u64,
            graph_safe: normalized.policy.graph_safe,
            deterministic: normalized.policy.deterministic,
        }
    }
}

pub(crate) struct Execution {
    raw: abi::Execution,
    stream: Arc<crate::CudaStream>,
    _storage: Vec<CudaBuffer>,
    #[cfg(test)]
    summary: String,
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for Execution {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        unsafe { abi::apxinf_attention_destroy(self.raw) }
    }
}

pub(crate) fn prepare(ctx: &CudaContext, normalized: Normalized) -> Result<Rc<Execution>> {
    let key = ExecutionKey::new(ctx, &normalized);
    if let Some(execution) = crate::workspace::lookup_execution(&key) {
        crate::workspace::use_execution(&execution)?;
        return Ok(execution);
    }
    if !crate::workspace::may_prepare_native_resources() {
        return Err(Error::Other(
            "Attention execution cache miss during capture; prepare the same bindings first".into(),
        ));
    }

    let cache = normalized
        .policy
        .cache_dir
        .as_ref()
        .map(|path| CString::new(path.as_str()))
        .transpose()
        .map_err(|_| Error::Other("cache path contains NUL".into()))?;
    let policy = abi::Policy {
        workspace_limit: normalized.policy.workspace_limit as u64,
        online_tune: normalized.policy.online_tune as u32,
        allow_fallback: normalized.policy.allow_fallback as u32,
        graph_safe: normalized.policy.graph_safe as u32,
        deterministic: normalized.policy.deterministic as u32,
        cache_dir: cache
            .as_ref()
            .map_or(std::ptr::null(), |path| path.as_ptr()),
    };
    let mut raw = std::ptr::null_mut();
    unsafe {
        status::check(abi::apxinf_attention_prepare(
            ctx.runtime(),
            &normalized.spec,
            &policy,
            &normalized.bindings,
            &mut raw,
        ))?;
    }
    if raw.is_null() {
        return Err(Error::Other(
            "native Attention prepare returned a null execution".into(),
        ));
    }
    #[cfg(test)]
    let summary = unsafe { CStr::from_ptr(abi::apxinf_attention_summary(raw)) }
        .to_string_lossy()
        .into_owned();
    let execution = Rc::new(Execution {
        raw,
        stream: ctx.shared_stream(),
        _storage: normalized.storage,
        #[cfg(test)]
        summary,
        _not_send: PhantomData,
    });
    crate::workspace::store_execution(key, Rc::clone(&execution));
    crate::workspace::use_execution(&execution)?;
    Ok(execution)
}

pub(crate) fn execute(ctx: &CudaContext, normalized: Normalized) -> Result<()> {
    let execution = prepare(ctx, normalized)?;
    crate::workspace::validate_capture_target(
        execution.stream.device(),
        execution.stream.handle() as usize,
    )?;
    execution.stream.set_current_device().map_err(Error::Cuda)?;
    unsafe { status::check(abi::apxinf_attention_enqueue(execution.raw))? };
    if crate::workspace::is_capturing()
        || (crate::workspace::has_active_session() && !crate::workspace::is_preparing_session())
    {
        Ok(())
    } else {
        ctx.synchronize().map_err(Error::Cuda)
    }
}

#[cfg(test)]
impl Execution {
    pub(crate) fn summary(&self) -> &str {
        &self.summary
    }

    pub(crate) fn enqueue_for_test(&self) -> Result<()> {
        self.stream.set_current_device().map_err(Error::Cuda)?;
        unsafe { status::check(abi::apxinf_attention_enqueue(self.raw)) }
    }
}

#[cfg(test)]
pub(crate) fn validate_candidates(
    ctx: &CudaContext,
    normalized: &Normalized,
    expected: &[f32],
) -> Result<()> {
    let expected_len = (normalized.spec.batch as usize)
        .checked_mul(normalized.spec.query_tokens as usize)
        .and_then(|value| value.checked_mul(normalized.spec.query_heads as usize))
        .and_then(|value| value.checked_mul(normalized.spec.head_dim as usize))
        .ok_or_else(|| Error::Other("Attention validation output size overflow".into()))?;
    if expected.len() != expected_len || expected.iter().any(|value| !value.is_finite()) {
        return Err(Error::Other(
            "Attention validation output does not match the L3 semantic".into(),
        ));
    }
    let cache = normalized
        .policy
        .cache_dir
        .as_ref()
        .map(|path| CString::new(path.as_str()))
        .transpose()
        .map_err(|_| Error::Other("cache path contains NUL".into()))?;
    let policy = abi::Policy {
        workspace_limit: normalized.policy.workspace_limit as u64,
        online_tune: normalized.policy.online_tune as u32,
        allow_fallback: normalized.policy.allow_fallback as u32,
        graph_safe: normalized.policy.graph_safe as u32,
        deterministic: normalized.policy.deterministic as u32,
        cache_dir: cache
            .as_ref()
            .map_or(std::ptr::null(), |path| path.as_ptr()),
    };
    unsafe {
        status::check(abi::apxinf_attention_test_validate_candidates(
            ctx.runtime(),
            &normalized.spec,
            &policy,
            &normalized.bindings,
            expected.as_ptr(),
            expected.len() as u64,
        ))
    }
}

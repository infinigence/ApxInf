#[cfg(test)]
use std::ffi::CStr;
use std::ffi::CString;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Error, Result};

use super::contracts::{invalid, Normalized};
use crate::ffi::abi::{gemm as abi, status};
use crate::{CudaBuffer, CudaContext};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExecutionKey {
    spec: abi::Spec,
    device: usize,
    a: usize,
    b: usize,
    b_version: u64,
    b_is_immutable: u32,
    bias: usize,
    a_scales: usize,
    b_scales: usize,
    output: usize,
    stream: usize,
    alpha: u32,
    output_scale: u32,
    workspace_limit: u64,
    graph_safe: bool,
    deterministic: bool,
}

impl ExecutionKey {
    fn new(
        ctx: &CudaContext,
        spec: abi::Spec,
        bindings: abi::Bindings,
        policy: &super::contracts::GemmPolicy,
    ) -> Self {
        Self {
            spec,
            device: ctx.device_id(),
            a: bindings.a as usize,
            b: bindings.b as usize,
            b_version: bindings.b_version,
            b_is_immutable: bindings.b_is_immutable,
            bias: bindings.bias as usize,
            a_scales: bindings.a_scales as usize,
            b_scales: bindings.b_scales as usize,
            output: bindings.output as usize,
            stream: bindings.stream as usize,
            alpha: bindings.alpha.to_bits(),
            output_scale: bindings.output_scale.to_bits(),
            workspace_limit: policy.workspace_limit as u64,
            graph_safe: policy.graph_safe,
            deterministic: policy.deterministic,
        }
    }
}

/// Rust ownership boundary for one native execution bound to fixed addresses.
/// It is deliberately thread-confined and is not a second planning object.
pub(crate) struct Execution {
    raw: abi::Execution,
    stream: Arc<crate::CudaStream>,
    _storage: Vec<CudaBuffer>,
    #[cfg(test)]
    summary: String,
    _not_send: PhantomData<Rc<()>>,
}

impl Execution {
    #[cfg(test)]
    pub(crate) fn summary(&self) -> &str {
        &self.summary
    }

    #[cfg(test)]
    pub(crate) fn enqueue(self: &Rc<Self>) -> Result<()> {
        crate::workspace::validate_capture_target(
            self.stream.device(),
            self.stream.handle() as usize,
        )?;
        self.stream.set_current_device().map_err(Error::Cuda)?;
        crate::workspace::retain_resource(self);
        unsafe { status::check(abi::apxinf_gemm_enqueue(self.raw)) }
    }

    #[cfg(test)]
    pub(crate) fn weight_prepack_count(&self) -> u64 {
        unsafe { abi::apxinf_gemm_execution_weight_prepack_count(self.raw) }
    }
}

impl Drop for Execution {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        unsafe { abi::apxinf_gemm_destroy(self.raw) }
    }
}

pub(crate) fn prepare(ctx: &CudaContext, normalized: Normalized<'_>) -> Result<Rc<Execution>> {
    let Normalized {
        spec,
        policy: options,
        bindings,
        storage,
        validation_reference,
    } = normalized;
    let key = ExecutionKey::new(ctx, spec, bindings, &options);
    if let Some(execution) = crate::workspace::lookup_execution(&key) {
        crate::workspace::use_execution(&execution)?;
        return Ok(execution);
    }
    if !crate::workspace::may_prepare_native_resources() {
        return Err(invalid(
            "GEMM execution cache miss during capture; prepare the same bindings first",
        ));
    }

    let cache = options
        .cache_dir
        .as_ref()
        .map(|path| CString::new(path.as_str()))
        .transpose()
        .map_err(|_| invalid("cache path contains NUL"))?;
    let policy = abi::Policy {
        workspace_limit: options.workspace_limit as u64,
        online_tune: options.online_tune as u32,
        allow_fallback: options.allow_fallback as u32,
        graph_safe: options.graph_safe as u32,
        deterministic: options.deterministic as u32,
        cache_dir: cache
            .as_ref()
            .map_or(std::ptr::null(), |path| path.as_ptr()),
    };
    let tuning_bindings = if let Some(reference) = validation_reference {
        abi::TuningBindings {
            execution: bindings,
            expected_output: reference.expected.as_ptr(),
            expected_output_len: reference.expected.len() as u64,
            reference_kind: 1,
        }
    } else {
        abi::TuningBindings {
            execution: bindings,
            expected_output: std::ptr::null(),
            expected_output_len: 0,
            reference_kind: 0,
        }
    };
    let mut raw = std::ptr::null_mut();
    unsafe {
        status::check(abi::apxinf_gemm_prepare(
            ctx.runtime(),
            &spec,
            &policy,
            &tuning_bindings,
            &mut raw,
        ))?;
    }
    if raw.is_null() {
        return Err(invalid("native GEMM prepare returned a null execution"));
    }
    #[cfg(test)]
    let summary = unsafe { CStr::from_ptr(abi::apxinf_gemm_summary(raw)) }
        .to_string_lossy()
        .into_owned();
    #[cfg(test)]
    EXECUTION_CREATE_COUNT.with(|count| count.set(count.get() + 1));
    let execution = Rc::new(Execution {
        raw,
        stream: ctx.shared_stream(),
        _storage: storage,
        #[cfg(test)]
        summary,
        _not_send: PhantomData,
    });
    crate::workspace::store_execution(key, Rc::clone(&execution));
    crate::workspace::use_execution(&execution)?;
    Ok(execution)
}

pub(crate) fn execute(ctx: &CudaContext, normalized: Normalized<'_>) -> Result<()> {
    let execution = prepare(ctx, normalized)?;
    crate::workspace::validate_capture_target(
        execution.stream.device(),
        execution.stream.handle() as usize,
    )?;
    execution.stream.set_current_device().map_err(Error::Cuda)?;
    unsafe { status::check(abi::apxinf_gemm_enqueue(execution.raw))? };
    if crate::workspace::is_capturing()
        || (crate::workspace::has_active_session() && !crate::workspace::is_preparing_session())
    {
        Ok(())
    } else {
        #[cfg(test)]
        EXECUTION_SYNC_COUNT.with(|count| count.set(count.get() + 1));
        ctx.synchronize().map_err(Error::Cuda)
    }
}

#[cfg(test)]
thread_local! {
    static EXECUTION_CREATE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static EXECUTION_SYNC_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_execution_create_count() {
    EXECUTION_CREATE_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn execution_create_count() -> usize {
    EXECUTION_CREATE_COUNT.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn reset_execution_sync_count() {
    EXECUTION_SYNC_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn execution_sync_count() -> usize {
    EXECUTION_SYNC_COUNT.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn seed_recipe(
    ctx: &CudaContext,
    normalized: &Normalized<'_>,
    provider_id: u32,
    implementation_id: u32,
    implementation_version: u32,
    configuration: i32,
) -> Result<()> {
    let cache = normalized
        .policy
        .cache_dir
        .as_ref()
        .ok_or_else(|| invalid("test recipe seeding requires a cache directory"))?;
    let cache = CString::new(cache.as_str()).map_err(|_| invalid("cache path contains NUL"))?;
    let policy = abi::Policy {
        workspace_limit: normalized.policy.workspace_limit as u64,
        online_tune: normalized.policy.online_tune as u32,
        allow_fallback: normalized.policy.allow_fallback as u32,
        graph_safe: normalized.policy.graph_safe as u32,
        deterministic: normalized.policy.deterministic as u32,
        cache_dir: cache.as_ptr(),
    };
    unsafe {
        status::check(abi::apxinf_gemm_test_seed_recipe(
            &normalized.spec,
            &policy,
            ctx.device_id() as i32,
            provider_id,
            implementation_id,
            implementation_version,
            configuration,
        ))
    }
}

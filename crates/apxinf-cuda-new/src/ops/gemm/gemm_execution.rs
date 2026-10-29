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
    residual: usize,
    a_scales: usize,
    b_scales: usize,
    a_block_scales: usize,
    b_block_scales: usize,
    output: usize,
    stream: usize,
    alpha: u32,
    output_scale: u32,
    workspace_limit: u64,
    graph_safe: bool,
    deterministic: bool,
    online_tune: bool,
    allow_fallback: bool,
    cache_dir: Option<String>,
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
            residual: bindings.residual as usize,
            a_scales: bindings.a_scales as usize,
            b_scales: bindings.b_scales as usize,
            a_block_scales: bindings.a_block_scales as usize,
            b_block_scales: bindings.b_block_scales as usize,
            output: bindings.output as usize,
            stream: bindings.stream as usize,
            alpha: bindings.alpha.to_bits(),
            output_scale: bindings.output_scale.to_bits(),
            workspace_limit: policy.workspace_limit as u64,
            graph_safe: policy.graph_safe,
            deterministic: policy.deterministic,
            online_tune: policy.online_tune,
            allow_fallback: policy.allow_fallback,
            cache_dir: policy.cache_dir.clone(),
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
}

impl Drop for Execution {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        unsafe { abi::apxinf_gemm_destroy(self.raw) }
    }
}

// Own native trial resources across all failure paths. No trial execution is
// enqueued after preparation; the selected execution retains its actual B below.
struct NativeExecution(abi::Execution);
impl Drop for NativeExecution {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { abi::apxinf_gemm_destroy(self.0) }
        }
    }
}
impl NativeExecution {
    fn into_raw(mut self) -> abi::Execution {
        let raw = self.0;
        self.0 = std::ptr::null_mut();
        raw
    }
}
fn prepare_native(
    ctx: &CudaContext,
    spec: &abi::Spec,
    policy: &abi::Policy,
    bindings: &abi::Bindings,
    layout: Option<&mut u32>,
) -> Result<NativeExecution> {
    let mut raw = std::ptr::null_mut();
    unsafe {
        status::check(abi::apxinf_gemm_prepare(
            ctx.runtime(),
            spec,
            policy,
            bindings,
            &mut raw,
            layout.map_or(std::ptr::null_mut(), |value| value),
        ))?;
    }
    if raw.is_null() {
        return Err(invalid("native GEMM prepare returned a null execution"));
    }
    Ok(NativeExecution(raw))
}
fn prepare_weight(
    ctx: &CudaContext,
    weight: &mut super::weight::WeightState,
    choice: super::weight::LayoutChoiceKey,
    spec: &mut abi::Spec,
    policy: &abi::Policy,
    bindings: &mut abi::Bindings,
    storage: &mut Vec<CudaBuffer>,
) -> Result<NativeExecution> {
    let independent_trial = weight.layout != 0
        && weight.cache.as_ref().is_some_and(|cache| {
            !cache.choices.contains_key(&choice) && cache.budget >= weight.buffer.len()
        });
    if independent_trial {
        let mut layouts = 0;
        unsafe {
            status::check(abi::apxinf_gemm_candidate_layouts(
                ctx.runtime(),
                spec,
                policy,
                &mut layouts,
            ))?;
        }
        if layouts == (1u32 << 4) {
            let existing = if weight.layout == 4 {
                Some(weight.buffer.clone())
            } else {
                weight
                    .cache
                    .as_ref()
                    .and_then(|cache| cache.buffers.get(&4))
                    .cloned()
            };
            if let Some(buffer) = existing {
                // All applicable candidates read canonical bytes, so selecting
                // with layout 0 cannot repack this allocation. Keep the same
                // unrestricted recipe key (including restored recipes) without
                // allocating/copying a canonical trial unnecessarily.
                bindings.b = buffer.ptr();
                storage.push(buffer);
                spec.b_layout = 0;
                let mut selected_layout = 0;
                let execution =
                    prepare_native(ctx, spec, policy, bindings, Some(&mut selected_layout))?;
                if selected_layout != 4 {
                    return Err(invalid("canonical-only candidate query changed layout"));
                }
                spec.b_layout = 4;
                weight
                    .cache
                    .as_mut()
                    .expect("shared layout cache")
                    .choices
                    .insert(choice, 4);
                return Ok(execution);
            }
        }
    }

    let execution = if independent_trial {
        // Recover canonical bytes into a private trial even when the original
        // allocation is packed. Existing graphs keep reading immutable storage.
        let trial = CudaBuffer::alloc(weight.buffer.len(), ctx.device_id()).map_err(Error::Cuda)?;
        let (source, source_layout) = weight
            .cache
            .as_ref()
            .and_then(|cache| cache.buffers.get(&4))
            .map(|buffer| (buffer, 4))
            .unwrap_or((&weight.buffer, weight.layout));
        unsafe {
            status::check(abi::apxinf_gemm_relayout_weight(
                source.ptr(),
                trial.ptr(),
                spec.k,
                spec.n,
                spec.b_dtype,
                source_layout,
                4,
                bindings.stream,
            ))?;
        }
        let mut trial_spec = *spec;
        trial_spec.b_layout = 0;
        let mut trial_bindings = *bindings;
        trial_bindings.b = trial.ptr();
        let mut layout = 0;
        let selected =
            prepare_native(ctx, &trial_spec, policy, &trial_bindings, Some(&mut layout))?;
        let cache = weight.cache.as_mut().expect("shared layout cache");
        let existing = if layout == weight.layout {
            Some(weight.buffer.clone())
        } else {
            cache.buffers.get(&layout).cloned()
        };
        if let Some(buffer) = existing {
            let mut rebound = std::ptr::null_mut();
            unsafe {
                status::check(abi::apxinf_gemm_rebind_weight(
                    ctx.runtime(),
                    selected.0,
                    buffer.ptr(),
                    policy,
                    &mut rebound,
                ))?;
            }
            if rebound.is_null() {
                return Err(invalid("shared weight binding returned null"));
            }
            spec.b_layout = layout;
            bindings.b = buffer.ptr();
            storage.push(buffer);
            NativeExecution(rebound)
        } else if cache
            .buffers
            .values()
            .map(CudaBuffer::len)
            .sum::<usize>()
            .checked_add(trial.len())
            .is_some_and(|bytes| bytes <= cache.budget)
        {
            spec.b_layout = layout;
            bindings.b = trial.ptr();
            cache.buffers.insert(layout, trial.clone());
            if std::env::var_os("APXINF_GEMM_REPORT").is_some() {
                eprintln!(
                    "[apxinf-gemm-weight] k={} n={} layout={} extra_bytes={} cached_layouts={}",
                    spec.k,
                    spec.n,
                    layout,
                    cache.buffers.values().map(CudaBuffer::len).sum::<usize>(),
                    cache.buffers.len()
                );
            }
            storage.push(trial);
            selected
        } else {
            // A caller-supplied budget is a hard constraint. Keep the base
            // representation and tune compatible implementations without ever
            // changing an allocation referenced by an older execution.
            drop(selected);
            spec.b_layout = weight.layout;
            bindings.b = weight.buffer.ptr();
            prepare_native(ctx, spec, policy, bindings, None)?
        }
    } else {
        let unprepared = spec.b_layout == 0;
        let execution = prepare_native(
            ctx,
            spec,
            policy,
            bindings,
            unprepared.then_some(&mut weight.layout),
        )?;
        if unprepared {
            spec.b_layout = weight.layout;
        }
        execution
    };
    if let Some(cache) = weight.cache.as_mut() {
        cache.choices.insert(choice, spec.b_layout);
    }
    Ok(execution)
}

pub(crate) fn prepare(ctx: &CudaContext, normalized: Normalized) -> Result<Rc<Execution>> {
    let Normalized {
        mut spec,
        policy: options,
        mut bindings,
        mut storage,
        repacked_weight,
    } = normalized;
    if repacked_weight.is_none() {
        let buffer = storage
            .iter()
            .find(|buffer| buffer.ptr().cast_const() == bindings.b)
            .ok_or_else(|| invalid("GEMM B has no retained storage"))?;
        super::weight::reject_canonical_alias(buffer)?;
    }
    let choice_key = super::weight::LayoutChoiceKey::new(spec, &options);
    let mut weight_guard = repacked_weight
        .as_ref()
        .map(|weight| {
            weight
                .state
                .lock()
                .map_err(|_| invalid("weight state lock poisoned"))
        })
        .transpose()?;
    if let Some(weight) = weight_guard.as_ref() {
        if bindings.b_is_immutable == 0 {
            return Err(invalid(
                "repacked weights must be immutable after preparation",
            ));
        }
        if weight.layout == abi::WEIGHT_LAYOUT_INVALID {
            return Err(invalid("weight layout commit failed; reload the model"));
        }
        if weight.buffer.ptr().cast_const() != bindings.b
            || weight.buffer.device() != ctx.device_id()
        {
            return Err(invalid(
                "weight layout state belongs to a different allocation",
            ));
        }
        if weight.shape != [spec.k as usize, spec.n as usize]
            || super::contracts::dtype(weight.dtype)? != spec.b_dtype
        {
            return Err(invalid(
                "weight layout state has different geometry or dtype",
            ));
        }
        spec.b_layout = weight
            .cache
            .as_ref()
            .and_then(|cache| cache.choices.get(&choice_key).copied())
            .unwrap_or(weight.layout);
        if spec.b_layout != weight.layout {
            let buffer = weight
                .cache
                .as_ref()
                .and_then(|cache| cache.buffers.get(&spec.b_layout))
                .ok_or_else(|| invalid("selected shared weight layout is missing"))?
                .clone();
            bindings.b = buffer.ptr().cast_const();
            storage.push(buffer);
        }
    }
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
    let native = if let Some(weight) = weight_guard.as_mut() {
        prepare_weight(
            ctx,
            weight,
            choice_key,
            &mut spec,
            &policy,
            &mut bindings,
            &mut storage,
        )?
    } else {
        prepare_native(ctx, &spec, &policy, &bindings, None)?
    };
    let raw = native.into_raw();
    let key = ExecutionKey::new(ctx, spec, bindings, &options);
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

pub(crate) fn execute(ctx: &CudaContext, normalized: Normalized) -> Result<()> {
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
    normalized: &Normalized,
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

#[cfg(test)]
pub(crate) fn validate_candidates(
    ctx: &CudaContext,
    normalized: &Normalized,
    expected: &[f32],
) -> Result<()> {
    let output_width = if normalized.spec.semantic == super::contracts::Semantic::GemmGeglu as u32 {
        normalized.spec.n as usize / 2
    } else {
        normalized.spec.n as usize
    };
    let expected_output = (normalized.spec.m as usize)
        .checked_mul(output_width)
        .ok_or_else(|| invalid("GEMM validation output size overflow"))?;
    if expected.len() != expected_output {
        return Err(invalid(
            "Torch validation output does not match the L3 semantic",
        ));
    }
    if expected.iter().any(|value| !value.is_finite()) {
        return Err(invalid("Torch validation output must be finite"));
    }

    let cache = normalized
        .policy
        .cache_dir
        .as_ref()
        .map(|path| CString::new(path.as_str()))
        .transpose()
        .map_err(|_| invalid("cache path contains NUL"))?;
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
        status::check(abi::apxinf_gemm_test_validate_candidates(
            ctx.runtime(),
            &normalized.spec,
            &policy,
            &normalized.bindings,
            expected.as_ptr(),
            expected.len() as u64,
        ))
    }
}

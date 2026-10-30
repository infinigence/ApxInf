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
    _weight_lease: Option<Arc<super::weight::WeightStorage>>,
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
fn execution_choice(execution: &NativeExecution) -> Result<abi::Choice> {
    let mut choice = abi::Choice::default();
    unsafe {
        status::check(abi::apxinf_gemm_get_choice(execution.0, &mut choice))?;
    }
    Ok(choice)
}

fn prepare_choice(
    ctx: &CudaContext,
    spec: &abi::Spec,
    policy: &abi::Policy,
    bindings: &abi::Bindings,
    choice: &abi::Choice,
) -> Result<NativeExecution> {
    let mut raw = std::ptr::null_mut();
    unsafe {
        status::check(abi::apxinf_gemm_prepare_choice(
            ctx.runtime(),
            spec,
            policy,
            bindings,
            choice,
            &mut raw,
        ))?;
    }
    if raw.is_null() {
        return Err(invalid("GEMM choice binding returned null"));
    }
    Ok(NativeExecution(raw))
}

fn relayout(
    source: &CudaBuffer,
    destination: &CudaBuffer,
    spec: &abi::Spec,
    from: u32,
    to: u32,
    stream: crate::ffi::abi::types::CudaStream,
) -> Result<()> {
    unsafe {
        status::check(abi::apxinf_gemm_relayout_weight(
            source.ptr(),
            destination.ptr(),
            spec.k,
            spec.n,
            spec.b_dtype,
            from,
            to,
            stream,
        ))
    }
}

/// No execution may observe this allocation while its bytes change. A weak
/// lease is authoritative; Tensor/allocation aliases are not reader counts.
fn rewrite_base(
    weight: &mut super::weight::WeightState,
    spec: &abi::Spec,
    layout: u32,
    stream: crate::ffi::abi::types::CudaStream,
    prepared: Option<&CudaBuffer>,
) -> Result<()> {
    if weight.layout == layout {
        return Ok(());
    }
    if weight.base_lease.strong_count() != 0 {
        return Err(invalid(
            "cannot change a weight layout with live executions",
        ));
    }
    let from = weight.layout;
    let converted = if let Some(prepared) = prepared {
        prepared.clone()
    } else if let Some(existing) = weight.extra(layout) {
        existing.buffer.clone()
    } else {
        let temporary =
            CudaBuffer::alloc(weight.buffer.len(), weight.buffer.device()).map_err(Error::Cuda)?;
        relayout(&weight.buffer, &temporary, spec, from, layout, stream)?;
        temporary
    };
    // A failed write may leave partial bytes. Never make that weight usable.
    weight.layout = abi::WEIGHT_LAYOUT_INVALID;
    relayout(&converted, &weight.buffer, spec, layout, layout, stream)?;
    weight.layout = layout;
    if std::env::var_os("APXINF_GEMM_REPORT").is_some() {
        eprintln!(
            "[apxinf-gemm-weight-transition] k={} n={} from={} to={}",
            spec.k, spec.n, from, layout
        );
    }
    Ok(())
}

fn can_add_layout(weight: &super::weight::WeightState) -> bool {
    weight.cache.as_ref().is_some_and(|cache| {
        cache
            .extra_bytes()
            .checked_add(weight.buffer.len())
            .is_some_and(|bytes| bytes <= cache.budget)
    })
}

fn extra_lease(
    weight: &mut super::weight::WeightState,
    buffer: CudaBuffer,
    layout: u32,
) -> Arc<super::weight::WeightStorage> {
    let lease = Arc::new(super::weight::WeightStorage { buffer, layout });
    let cache = weight.cache.as_mut().expect("shared layout state");
    cache.buffers.insert(layout, Arc::downgrade(&lease));
    if std::env::var_os("APXINF_GEMM_REPORT").is_some() {
        eprintln!(
            "[apxinf-gemm-weight] k={} n={} layout={} extra_bytes={} cached_layouts={}",
            weight.shape[0],
            weight.shape[1],
            layout,
            cache.extra_bytes(),
            cache
                .buffers
                .values()
                .filter(|value| value.strong_count() != 0)
                .count()
        );
    }
    lease
}

fn bind_storage(
    lease: &Arc<super::weight::WeightStorage>,
    spec: &mut abi::Spec,
    bindings: &mut abi::Bindings,
    storage: &mut Vec<CudaBuffer>,
) {
    spec.b_layout = lease.layout;
    bindings.b = lease.buffer.ptr();
    storage.push(lease.buffer.clone());
}

fn prepare_weight(
    ctx: &CudaContext,
    weight: &mut super::weight::WeightState,
    key: super::weight::LayoutChoiceKey,
    spec: &mut abi::Spec,
    policy: &abi::Policy,
    bindings: &mut abi::Bindings,
    storage: &mut Vec<CudaBuffer>,
) -> Result<(NativeExecution, Arc<super::weight::WeightStorage>)> {
    if let Some(choice) = weight.choices.get(&key).copied() {
        let layout = choice.weight_layout;
        let lease = if layout == weight.layout {
            Some(weight.lease_base())
        } else if let Some(existing) = weight.extra(layout) {
            Some(existing)
        } else if weight.base_lease.strong_count() == 0
            && weight.buffer.len() as u64 <= policy.workspace_limit
        {
            rewrite_base(weight, spec, layout, bindings.stream, None)?;
            Some(weight.lease_base())
        } else if can_add_layout(weight) {
            let buffer =
                CudaBuffer::alloc(weight.buffer.len(), ctx.device_id()).map_err(Error::Cuda)?;
            relayout(
                &weight.buffer,
                &buffer,
                spec,
                weight.layout,
                layout,
                bindings.stream,
            )?;
            Some(extra_lease(weight, buffer, layout))
        } else {
            None
        };
        if let Some(lease) = lease {
            bind_storage(&lease, spec, bindings, storage);
            return Ok((prepare_choice(ctx, spec, policy, bindings, &choice)?, lease));
        }
        // A live reader and exhausted budget can constrain this particular
        // binding. Keep the unconstrained metadata for a later retired plan.
        spec.b_layout = weight.layout;
        bindings.b = weight.buffer.ptr();
        let execution = prepare_native(ctx, spec, policy, bindings, None)?;
        return Ok((execution, weight.lease_base()));
    }

    let independent = weight
        .cache
        .as_ref()
        .is_some_and(|cache| cache.budget >= weight.buffer.len());
    let base_available = weight.base_lease.strong_count() == 0;
    let can_restore_base =
        matches!(weight.layout, 0 | 4) || weight.buffer.len() as u64 <= policy.workspace_limit;
    let execution;
    let lease;
    let no_live_extras = weight
        .cache
        .as_ref()
        .is_none_or(|cache| cache.extra_bytes() == 0);
    if independent && base_available && can_restore_base && no_live_extras {
        if !matches!(weight.layout, 0 | 4) {
            rewrite_base(weight, spec, 4, bindings.stream, None)?;
        }
        spec.b_layout = 0;
        bindings.b = weight.buffer.ptr();
        execution = prepare_native(ctx, spec, policy, bindings, Some(&mut weight.layout))?;
        spec.b_layout = weight.layout;
        lease = weight.lease_base();
    } else if independent {
        // A live base is immutable. Select independently on private storage;
        // retain an extra representation only when concurrent readers need it.
        let mut layouts = 0;
        unsafe {
            status::check(abi::apxinf_gemm_candidate_layouts(
                ctx.runtime(),
                spec,
                policy,
                &mut layouts,
            ))?;
        }
        let canonical = if weight.layout == 4 {
            weight.base_lease.upgrade()
        } else {
            weight.extra(4)
        };
        if layouts == (1u32 << 4) && canonical.is_some() {
            lease = canonical.expect("live canonical layout");
            bind_storage(&lease, spec, bindings, storage);
            spec.b_layout = 0;
            let mut layout = 0;
            execution = prepare_native(ctx, spec, policy, bindings, Some(&mut layout))?;
            if layout != 4 {
                return Err(invalid("canonical-only selection changed layout"));
            }
            spec.b_layout = 4;
        } else {
            let trial =
                CudaBuffer::alloc(weight.buffer.len(), ctx.device_id()).map_err(Error::Cuda)?;
            if let Some(canonical) = weight.extra(4) {
                relayout(&canonical.buffer, &trial, spec, 4, 4, bindings.stream)?;
            } else {
                relayout(
                    &weight.buffer,
                    &trial,
                    spec,
                    weight.layout,
                    4,
                    bindings.stream,
                )?;
            }
            let mut trial_spec = *spec;
            trial_spec.b_layout = 0;
            let mut trial_bindings = *bindings;
            trial_bindings.b = trial.ptr();
            let mut layout = 0;
            let selected =
                prepare_native(ctx, &trial_spec, policy, &trial_bindings, Some(&mut layout))?;
            let choice = execution_choice(&selected)?;
            weight.choices.insert(key.clone(), choice);
            if layout == weight.layout {
                lease = weight.lease_base();
                bind_storage(&lease, spec, bindings, storage);
                execution = prepare_choice(ctx, spec, policy, bindings, &choice)?;
            } else if let Some(existing) = weight.extra(layout) {
                lease = existing;
                bind_storage(&lease, spec, bindings, storage);
                execution = prepare_choice(ctx, spec, policy, bindings, &choice)?;
            } else if base_available && can_restore_base {
                // Other graphs may lease extra layouts while the original
                // allocation is unused. Reuse that address for a new choice.
                rewrite_base(weight, spec, layout, bindings.stream, Some(&trial))?;
                lease = weight.lease_base();
                bind_storage(&lease, spec, bindings, storage);
                execution = prepare_choice(ctx, spec, policy, bindings, &choice)?;
            } else if can_add_layout(weight) {
                lease = extra_lease(weight, trial, layout);
                bind_storage(&lease, spec, bindings, storage);
                execution = selected;
            } else {
                spec.b_layout = weight.layout;
                bindings.b = weight.buffer.ptr();
                execution = prepare_native(ctx, spec, policy, bindings, None)?;
                // Do not replace the independently selected choice with this
                // budget-constrained fallback.
                return Ok((execution, weight.lease_base()));
            }
        }
    } else {
        spec.b_layout = weight.layout;
        bindings.b = weight.buffer.ptr();
        let unprepared = weight.layout == 0;
        execution = prepare_native(
            ctx,
            spec,
            policy,
            bindings,
            unprepared.then_some(&mut weight.layout),
        )?;
        spec.b_layout = weight.layout;
        lease = weight.lease_base();
    }
    let choice = execution_choice(&execution)?;
    weight.choices.insert(key, choice);
    Ok((execution, lease))
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
    // Keep a discovered live extra leased until native preparation binds it.
    let mut selected_storage = None;
    if let Some(weight) = weight_guard.as_mut() {
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
        weight.prune_layouts();
        let layout = weight
            .choices
            .get(&choice_key)
            .map(|choice| choice.weight_layout)
            .unwrap_or(weight.layout);
        spec.b_layout = weight.layout;
        if layout != weight.layout {
            if let Some(lease) = weight.extra(layout) {
                bind_storage(&lease, &mut spec, &mut bindings, &mut storage);
                selected_storage = Some(lease);
            }
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
    let (native, weight_lease) = if let Some(weight) = weight_guard.as_mut() {
        let (native, lease) = prepare_weight(
            ctx,
            weight,
            choice_key,
            &mut spec,
            &policy,
            &mut bindings,
            &mut storage,
        )?;
        (native, Some(lease))
    } else {
        (prepare_native(ctx, &spec, &policy, &bindings, None)?, None)
    };
    drop(selected_storage);
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
        _weight_lease: weight_lease,
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

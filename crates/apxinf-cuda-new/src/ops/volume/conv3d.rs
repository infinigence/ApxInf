use super::Unfold3dGeometry;
use crate::ffi::abi::{status, volume as abi};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::{CudaBuffer, CudaContext, CudaStream};
use apxinf_core::{DType, Result, Tensor};
use std::{
    ffi::{CStr, CString},
    ptr::NonNull,
    rc::Rc,
    sync::Arc,
};

/// BF16 cross-correlation with FP32 accumulation and BF16 output rounding.
/// Input/history/output are contiguous THWC. Weights are immutable for the
/// execution lifetime, stored as [kernel_t * kernel_h * kernel_w * Cin, Cout].
/// Bias is deliberately a separate operation after the BF16 rounding.
pub struct Conv3dArgs<'a> {
    pub input: &'a Tensor,
    pub history: Option<&'a Tensor>,
    pub weight: &'a Tensor,
    pub geometry: Unfold3dGeometry,
    pub out: &'a mut Tensor,
    /// Maximum bytes for packed weights, transformed input and device scratch.
    pub workspace_limit: usize,
}

/// Selection policy for deterministic, graph-safe BF16 convolution.
/// Preparation may tune; enqueue never selects or allocates resources.
#[derive(Clone, Debug)]
pub struct Conv3dPolicy {
    pub online_tune: bool,
    pub allow_fallback: bool,
    /// None uses the context cache; Some("") disables disk persistence.
    pub cache_dir: Option<String>,
}
impl Default for Conv3dPolicy {
    fn default() -> Self {
        Self {
            online_tune: false,
            allow_fallback: true,
            cache_dir: Some(String::new()),
        }
    }
}

/// Bound, thread-confined convolution; preparation must occur before capture.
/// Enqueue allocates nothing and graphs retain the execution and all storage.
pub struct PreparedConv3d(Rc<Execution>);
struct Execution {
    raw: NonNull<std::ffi::c_void>,
    _storage: Vec<CudaBuffer>,
    stream: Arc<CudaStream>,
}
/// Session-cache identity: the conv geometry plus the bound storage addresses
/// and selection policy. Identical bindings reuse one native execution across
/// prepare and capture.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExecutionKey {
    spec: abi::ConvSpec,
    device: usize,
    input: usize,
    history: usize,
    weight: usize,
    output: usize,
    stream: usize,
    workspace_limit: u64,
    online_tune: bool,
    allow_fallback: bool,
    cache_dir: Option<String>,
}
fn product(dims: &[usize]) -> Result<usize> {
    dims.iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .filter(|&v| v <= i64::MAX as usize / 2)
        .ok_or_else(|| invalid("Conv3d element count overflow"))
}
impl PreparedConv3d {
    pub fn new(ctx: &CudaContext, args: Conv3dArgs<'_>) -> Result<Self> {
        Self::with_policy(ctx, args, &Conv3dPolicy::default())
    }
    pub fn with_policy(
        ctx: &CudaContext,
        args: Conv3dArgs<'_>,
        selection: &Conv3dPolicy,
    ) -> Result<Self> {
        let d = args.input.shape().dims();
        let w = args.weight.shape().dims();
        if d.len() != 4 || d.contains(&0) || w.len() != 2 || w.contains(&0) {
            return Err(invalid("Conv3d requires THWC input and [K,Cout] weight"));
        }
        product(d)?;
        product(w)?;
        let input = tensor_storage(ctx, args.input, DType::BF16, d)?;
        let weight = tensor_storage(ctx, args.weight, DType::BF16, w)?;
        let h = args
            .history
            .map_or(0, |h| h.shape().dims().first().copied().unwrap_or(0));
        let history = args
            .history
            .map(|t| {
                if h == 0 {
                    return Err(invalid("Conv3d history must be nonempty"));
                }
                product(&[h, d[1], d[2], d[3]])?;
                tensor_storage(ctx, t, DType::BF16, &[h, d[1], d[2], d[3]])
            })
            .transpose()?;
        let g = args.geometry;
        let k = product(&[g.kernel[0], g.kernel[1], g.kernel[2], d[3]])?;
        if w[0] != k {
            return Err(invalid("Conv3d weight K disagrees with kernel/input"));
        }
        let mut output = [0usize; 4];
        output[3] = w[1];
        for i in 0..3 {
            if g.kernel[i] == 0 || g.stride[i] == 0 {
                return Err(invalid("Conv3d kernel/stride must be positive"));
            }
            let padded = d[i]
                .checked_add(if i == 0 { h } else { 0 })
                .and_then(|x| x.checked_add(g.padding_before[i]))
                .and_then(|x| x.checked_add(g.padding_after[i]))
                .filter(|&x| x <= i64::MAX as usize)
                .ok_or_else(|| invalid("Conv3d padded extent overflow"))?;
            if padded < g.kernel[i] {
                return Err(invalid("Conv3d kernel exceeds padded input"));
            }
            output[i] = (padded - g.kernel[i]) / g.stride[i] + 1;
        }
        product(&output)?;
        let out = tensor_storage(ctx, args.out, DType::BF16, &output)?;
        let overlaps = |a: &CudaBuffer| {
            let a0 = a.ptr() as usize;
            let b0 = out.ptr() as usize;
            a0 < b0.saturating_add(out.len()) && b0 < a0.saturating_add(a.len())
        };
        if overlaps(&input) || overlaps(&weight) || history.as_ref().is_some_and(overlaps) {
            return Err(invalid("Conv3d output must not alias input/history/weight"));
        }
        let dim =
            |x: usize| i64::try_from(x).map_err(|_| invalid("Conv3d extent exceeds ABI range"));
        let triple =
            |x: [usize; 3]| -> Result<[i64; 3]> { Ok([dim(x[0])?, dim(x[1])?, dim(x[2])?]) };
        let spec = abi::ConvSpec {
            geometry: abi::Spec {
                version: 1,
                reserved: 0,
                input: [dim(d[0])?, dim(d[1])?, dim(d[2])?, dim(d[3])?],
                history_frames: dim(h)?,
                kernel: triple(g.kernel)?,
                stride: triple(g.stride)?,
                padding_before: triple(g.padding_before)?,
                padding_after: triple(g.padding_after)?,
                output: triple([output[0], output[1], output[2]])?,
                row_start: 0,
                rows: dim(product(&output[..3])?)?,
            },
            output_channels: dim(w[1])?,
        };
        let bindings = abi::ConvBindings {
            volume: abi::Bindings {
                input: input.ptr().cast_const(),
                history: history
                    .as_ref()
                    .map_or(std::ptr::null(), |h| h.ptr().cast_const()),
                output: out.ptr(),
                stream: ctx.stream().handle(),
            },
            weight: weight.ptr().cast_const(),
        };
        let cache_dir = selection
            .cache_dir
            .as_ref()
            .map(|s| CString::new(s.as_str()))
            .transpose()
            .map_err(|_| invalid("Conv3d cache directory contains NUL"))?;
        let policy = abi::ConvPolicy {
            workspace_limit: args.workspace_limit as u64,
            version: 1,
            online_tune: selection.online_tune as u32,
            allow_fallback: selection.allow_fallback as u32,
            reserved: 0,
            cache_dir: cache_dir.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
        };
        let key = ExecutionKey {
            spec,
            device: ctx.device_id(),
            input: input.ptr() as usize,
            history: history.as_ref().map_or(0, |h| h.ptr() as usize),
            weight: weight.ptr() as usize,
            output: out.ptr() as usize,
            stream: ctx.stream().handle() as usize,
            workspace_limit: args.workspace_limit as u64,
            online_tune: selection.online_tune,
            allow_fallback: selection.allow_fallback,
            cache_dir: selection.cache_dir.clone(),
        };
        if let Some(execution) = crate::workspace::lookup_execution(&key) {
            crate::workspace::use_execution(&execution)?;
            return Ok(Self(execution));
        }
        if !crate::workspace::may_prepare_native_resources() {
            return Err(invalid(
                "Conv3d cache miss during capture; prepare the same bindings first",
            ));
        }
        let mut raw = std::ptr::null_mut();
        unsafe {
            status::check(abi::apxinf_conv3d_prepare(
                ctx.runtime(),
                &spec,
                &policy,
                &bindings,
                &mut raw,
            ))?;
        }
        let mut storage = vec![input, weight, out];
        storage.extend(history);
        let execution = Rc::new(Execution {
            raw: NonNull::new(raw).ok_or_else(|| invalid("null Conv3d execution"))?,
            _storage: storage,
            stream: ctx.shared_stream(),
        });
        crate::workspace::store_execution(key, Rc::clone(&execution));
        crate::workspace::use_execution(&execution)?;
        Ok(Self(execution))
    }
    pub fn enqueue(&self) -> Result<()> {
        crate::workspace::validate_capture_target(
            self.0.stream.device(),
            self.0.stream.handle() as usize,
        )?;
        crate::workspace::use_execution(&self.0)?;
        unsafe { status::check(abi::apxinf_conv3d_enqueue(self.0.raw.as_ptr())) }
    }
    /// Diagnostic implementation summary; it is not a provider-selection API.
    pub fn summary(&self) -> String {
        unsafe {
            CStr::from_ptr(abi::apxinf_conv3d_summary(self.0.raw.as_ptr()))
                .to_string_lossy()
                .into_owned()
        }
    }
}
impl Drop for Execution {
    fn drop(&mut self) {
        unsafe {
            abi::apxinf_conv3d_destroy(self.raw.as_ptr());
        }
    }
}

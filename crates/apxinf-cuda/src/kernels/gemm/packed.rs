//! Bit-exact packed-BF16 weights for the batch-1 decode GEMV.
//!
//! Decode reads every weight of the model once per generated token -- 8.41 GB
//! for this checkpoint -- and the cuBLAS GEMVs that do it already run at 96%
//! of the device's measured 259.7 GB/s. The only thing left to trade is the
//! number of bytes, and the accuracy budget does not allow changing any of
//! them, so the bytes have to come down losslessly.
//!
//! They can. A BF16 weight's exponent field carries 2.58 bits of entropy
//! across this checkpoint against 7.97 for sign and mantissa together, and in
//! every tensor sampled 100.0000% of 128-weight blocks have an exponent range
//! that fits in five bits. Storing the sign and mantissa as one byte, the
//! exponent as a five-bit offset from a per-block base, and the base once per
//! 128 weights is 13.0625 bits against 16 -- and reconstruction returns the
//! original sixteen bits, so the GEMV's output is bit-identical.
//!
//! Measured on the lm_head shape (248320 x 2560, 1212 MB, far past the 32 MB
//! L2, one variant per process, three rounds): BF16 4987-6675 us, packed
//! 4163-4174 us. 1.196x, at 248.7 GB/s -- still bandwidth-bound, so the
//! reconstruction arithmetic is free.
//!
//! The packed copies are built on first use and kept for the process. They sit
//! beside the BF16 originals rather than replacing them, because prefill wants
//! real BF16 for the tensor cores; the pair costs about 16 GB for this model.

use std::collections::HashMap;
use std::sync::Mutex;

use apxinf_core::{DType, Error, Result, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;

const PACK_BLOCK: usize = 128;

pub(crate) struct PackedWeight {
    lo: CudaBuffer,
    off_lo: CudaBuffer,
    off_hi: CudaBuffer,
    base: CudaBuffer,
    k: usize,
    n: usize,
}

/// Keyed by the weight's device address and shape. Model weights live for the
/// process, so an entry is valid for as long as it can be looked up.
static CACHE: Mutex<Option<HashMap<(usize, usize, usize), PackedWeight>>> = Mutex::new(None);

/// Whether the packed decode path is on. Off unless asked for.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("APXINF_PACKED_DECODE").as_deref(),
            Ok("1") | Ok("on") | Ok("true")
        )
    })
}

/// Whether the hand-written BF16 GEMV replaces cuBLAS for decode. A
/// diagnostic: it answers what a fused decode kernel would have to start from.
pub(crate) fn plain_gemv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("APXINF_PLAIN_GEMV").is_some())
}

/// Narrowest output width worth packing.
///
/// The GEMV over the packed form wins above this and loses badly below it.
/// Measured per process on real weights at k=2560, packed against BF16:
///
///   n      9216   0.371x      n   36864   1.277x
///   n     18432   0.368x      n   73728   1.610x
///                             n  248320   1.578x
///
/// The cause is load instructions, not bytes: the byte stream alone reads
/// 22.5 MB in 137.7 us where BF16 reads 45 MB in 184.8, and a ladder that
/// adds one piece at a time (probes/steps.cu) puts the whole difference on the
/// three exponent planes -- 282 us with one stream, 608 with four. A layout
/// that reconstructs from a single stream would not have the crossover at all.
/// `APXINF_PACKED_DECODE_MIN_N` moves it.
fn min_packed_width() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("APXINF_PACKED_DECODE_MIN_N")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(36864)
    })
}

fn build(ctx: &CudaContext, weight: &Tensor, k: usize, n: usize) -> Result<Option<PackedWeight>> {
    if k % PACK_BLOCK != 0 {
        return Ok(None);
    }
    let count = k * n;
    let device = ctx.device_id();
    let lo = CudaBuffer::alloc(count, device).map_err(Error::Cuda)?;
    let off_lo = CudaBuffer::alloc(count / 2, device).map_err(Error::Cuda)?;
    let off_hi = CudaBuffer::alloc(count / 8, device).map_err(Error::Cuda)?;
    let base = CudaBuffer::alloc(count / PACK_BLOCK, device).map_err(Error::Cuda)?;
    let reject = CudaBuffer::alloc_zeros(4, device).map_err(Error::Cuda)?;
    let src = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_pack_bf16_transposed(
            src.ptr(),
            lo.ptr(),
            off_lo.ptr(),
            off_hi.ptr(),
            base.ptr(),
            reject.ptr(),
            k as i32,
            n as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
        ffi::check_cuda(ffi::cudaStreamSynchronize(ctx.stream().handle())).map_err(Error::Cuda)?;
    }
    let mut flag = [0u8; 4];
    reject.copy_to_host(&mut flag).map_err(Error::Cuda)?;
    if i32::from_ne_bytes(flag) != 0 {
        // Some block's exponents do not fit five bits, so this packing would
        // not be a faithful copy. Say so and leave the weight on BF16.
        eprintln!("[apxinf] packed decode: {k}x{n} weight has a block wider than five exponent bits; keeping BF16");
        return Ok(None);
    }
    Ok(Some(PackedWeight {
        lo,
        off_lo,
        off_hi,
        base,
        k,
        n,
    }))
}

/// Run the batch-1 GEMV from the packed copy of `weight`, building that copy
/// on first use. `Ok(false)` means the caller should use its own path: the
/// shape is unsupported, or the weight does not pack faithfully.
pub(crate) fn gemv(
    ctx: &CudaContext,
    weight: &Tensor,
    activation: &CudaBuffer,
    output: &CudaBuffer,
    k: usize,
    n: usize,
) -> Result<bool> {
    if n < min_packed_width() {
        return Ok(false);
    }
    let ptr = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?.ptr() as usize;
    let mut guard = CACHE.lock().map_err(|_| Error::Other("packed weight cache poisoned".into()))?;
    let map = guard.get_or_insert_with(HashMap::new);
    let key = (ptr, k, n);
    if !map.contains_key(&key) {
        match build(ctx, weight, k, n)? {
            Some(packed) => {
                map.insert(key, packed);
            }
            None => return Ok(false),
        }
    }
    let packed = &map[&key];
    if packed.k != k || packed.n != n {
        return Ok(false);
    }
    unsafe {
        ffi::check_cuda(ffi::apxinf_packed_gemv_bf16(
            packed.lo.ptr(),
            packed.off_lo.ptr(),
            packed.off_hi.ptr(),
            packed.base.ptr(),
            activation.ptr(),
            output.ptr(),
            n as i32,
            k as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    let _ = DType::BF16;
    Ok(true)
}

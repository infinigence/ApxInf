//! BF16 GEMM operator contracts.

use apxinf_core::{Error, Result, Tensor};

use super::super::contracts::{
    bf16_output, matrix_tensor, require_matrix_bf16, require_output_matrix,
};
use super::observe_bf16;
use crate::context::MusaContext;
#[cfg(apxinf_musa_available)]
use crate::error::musa_error;
#[cfg(apxinf_musa_available)]
use crate::ffi;
#[cfg(apxinf_musa_available)]
use crate::kernels::contracts::{checked_i32, tensor_ptr};
#[cfg(not(apxinf_musa_available))]
use crate::kernels::native_unavailable;
use crate::kernels::{not_ported, TacticDispatch};
#[cfg(apxinf_musa_available)]
use crate::tuning::{choose_bf16, Bf16Choice, TuningMode};

/// Physical BF16 GEMM contract: `[M, K] @ [K, N] -> [M, N]`.
///
/// Native: aligned MP22 GEMMs inside a decode scope and measured encode shapes
/// with registered NN weights call native MP22 assembly; other calls use the
/// muBLAS/muDNN provider path. See [`launch`] for
/// the layout and numerical contract the adapter mirrors.
pub fn gemm_bf16(ctx: &MusaContext, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (rows, _, cols) = require_contraction(ctx, activation, weight, "gemm")?;
    let mut output = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    gemm_bf16_into(ctx, activation, weight, &mut output)?;
    Ok(output)
}

/// Write `[M, K] @ [K, N] -> [M, N]` into a caller-provided output.
pub fn gemm_bf16_into(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
) -> Result<()> {
    gemm_bf16_into_with(
        ctx,
        activation,
        weight,
        output,
        TacticDispatch::AllowFallback,
    )
}

/// [`gemm_bf16_into`] with the caller's fallback constraint.
///
/// `TacticDispatch::RequireExact` refuses a bucket extrapolation and the vendor
/// default when no record covers this exact shape (autotuning may measure one,
/// which is not a fallback). The assembly route is a platform optimisation, not
/// a tactic fallback: when a registered weight and a supported shape select it,
/// it runs under either dispatch.
pub fn gemm_bf16_into_with(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
    tactic: TacticDispatch,
) -> Result<()> {
    let (rows, inner, cols) = require_contraction(ctx, activation, weight, "gemm")?;
    // Calibration observes the operands the GEMM actually consumes, so the
    // notification happens before the computation, as on CUDA.
    observe_bf16(activation, weight)?;
    require_output_matrix(ctx, output, rows, cols, "gemm", "out")?;
    crate::workspace::note_shape(crate::workspace::OperationShape::Gemm {
        m: rows,
        n: cols,
        k: inner,
    })?;
    launch(ctx, activation, weight, output, rows, inner, cols, tactic)
}

fn require_contraction(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    operation: &str,
) -> Result<(usize, usize, usize)> {
    let (rows, inner) = require_matrix_bf16(ctx, activation, operation, "activation")?;
    let (weight_inner, cols) = require_matrix_bf16(ctx, weight, operation, "weight")?;
    if inner != weight_inner {
        return Err(Error::Other(format!(
            "MUSA {operation} contraction mismatch: [{rows}x{inner}] @ [{weight_inner}x{cols}]"
        )));
    }
    Ok((rows, inner, cols))
}

/// `[M, K] @ [K, N] -> [M, N]` through native MP22 assembly or a vendor provider.
///
/// # Layout
/// Row-major `D = A @ B` is computed through the column-major identity
/// `D^T = B^T @ A^T`, so nothing is transposed in memory. The adapter owns the
/// convention; it is the same one `crates/apxinf-cuda/adapters/cublaslt_adapter.cu`
/// uses and the one the port's muBLAS probe verified
/// against a double-precision reference on device.
///
/// # Numerical contract
/// The recorded contract is `_f32(activation) @ _f32(weight)`, rounded once on
/// the way out — a torch eager bridge computes it that way. The
/// adapter's `MUBLAS_COMPUTE_32F` + `MUSA_R_32F` is the same contract — f32
/// accumulators, one rounding at the output — because that is what the recorded
/// golden tensors were produced under. An algorithm with a narrower accumulator
/// would round per contraction step and move the numbers.
///
/// # Stream
/// The launch goes on `ctx.stream()`. That is ordered against the torch
/// operators that produce the operands and consume the result, because
/// `MusaStream` is a blocking stream and blocking streams synchronise with the
/// legacy default stream in both directions; the port's stream-ordering
/// probe measures it. No event or synchronisation is needed here, and adding one per
/// GEMM would cost two device syncs on the hottest operator in the model.
#[cfg(apxinf_musa_available)]
fn launch(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    rows: usize,
    inner: usize,
    cols: usize,
    tactic: TacticDispatch,
) -> Result<()> {
    let assembly = if super::in_decode_scope() {
        super::assembly::launch(ctx, activation, weight, output, rows, cols, inner)?
    } else {
        super::assembly::launch_encode(ctx, activation, weight, output, rows, cols, inner)?
    };
    if assembly {
        return Ok(());
    }
    let tuning = ctx.tuning();
    let decision = choose_bf16(
        tuning.bf16_provider(rows, cols, inner),
        tuning.autotune_enabled(),
        tactic.requires_exact(),
        rows,
        cols,
        inner,
    )?;
    let provider = match decision {
        // A record measured at this very shape is the answer in every mode.
        Bf16Choice::Exact(provider) => provider,
        // Autotuning upgrades everything else — a bucket-derived guess as much
        // as a miss — into a measurement of this shape, which is CUDA's rule
        // too (`needs_exact` is true for a bucket match, see
        // `crates/apxinf-cuda/src/kernels/gemm/plan.rs:99`). It is also what
        // lets a tuning run densify the records at the shapes a deployment
        // actually reaches.
        Bf16Choice::Measure => tune(ctx, activation, weight, output, rows, cols, inner)?,
        // Inference, and a recorded M within a factor of two: an extrapolation,
        // taken and said once per bucket.
        Bf16Choice::Bucket {
            provider,
            recorded_m,
        } => {
            crate::tuning::note_bf16_bucket(rows, cols, inner, recorded_m, provider);
            provider
        }
        // Inference, and nothing within a factor of two: muBLAS. Said once per
        // shape — but not while autotuning merely *suppresses* tuning (warm-up
        // and capture run under `without_autotune`), where the fallback is
        // transient and the generation bump re-captures with a tuned provider.
        Bf16Choice::Miss => {
            if tuning.mode() == TuningMode::Inference {
                crate::tuning::note_bf16_miss(rows, cols, inner);
            }
            crate::tuning::Bf16Provider::Mublas
        }
    };
    dispatch(ctx, activation, weight, output, rows, cols, inner, provider)
}

#[cfg(apxinf_musa_available)]
fn dispatch(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    rows: usize,
    cols: usize,
    inner: usize,
    provider: crate::tuning::Bf16Provider,
) -> Result<()> {
    let m = checked_i32("gemm", "rows", rows)?;
    let n = checked_i32("gemm", "cols", cols)?;
    let k = checked_i32("gemm", "inner", inner)?;
    if provider == crate::tuning::Bf16Provider::Mudnn {
        let status = unsafe {
            ffi::custom::apxinf_musa_mudnn_bf16_gemm(
                tensor_ptr(activation)?,
                tensor_ptr(weight)?,
                tensor_ptr(output)?,
                m,
                n,
                k,
                ctx.stream().handle(),
            )
        };
        return ffi::custom::check_mudnn(status, || {
            format!("muDNN BF16 GEMM {rows}x{inner}x{cols}")
        })
        .map_err(musa_error);
    }
    // SAFETY: `require_matrix_bf16` has established that both operands are
    // contiguous 2-D BF16 tensors on this context's device, and `matrix_tensor`
    // that the output is a `[rows, cols]` BF16 allocation on the same device.
    // `tensor_ptr` borrows storage the caller's `Tensor`s keep alive for the
    // whole call. The launch is asynchronous on `ctx.stream()`, and every
    // buffer it touches outlives the call: the operands and the output by those
    // `Tensor`s, the mublas handle by the adapter's per-stream map.
    unsafe {
        ffi::mublas::check_mublas(ffi::mublas::apxinf_musa_bf16_gemm(
            tensor_ptr(activation)?,
            tensor_ptr(weight)?,
            tensor_ptr(output)?,
            m,
            n,
            k,
            1.0,
            ctx.stream().handle(),
        ))
        .map_err(musa_error)?;
    }
    Ok(())
}

#[cfg(apxinf_musa_available)]
fn values(ctx: &MusaContext, tensor: &Tensor, count: usize) -> Result<Vec<f64>> {
    ctx.synchronize()?;
    let mut bytes = vec![
        0u8;
        count
            .checked_mul(2)
            .ok_or_else(|| musa_error("BF16 benchmark size overflow"))?
    ];
    unsafe {
        ffi::check_runtime(ffi::musa::musaMemcpy(
            bytes.as_mut_ptr().cast(),
            tensor_ptr(tensor)?,
            bytes.len(),
            ffi::musa::MUSA_MEMCPY_DEVICE_TO_HOST,
        ))
    }
    .map_err(musa_error)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16) as f64)
        .collect())
}

#[cfg(apxinf_musa_available)]
fn tune(
    ctx: &MusaContext,
    a: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<crate::tuning::Bf16Provider> {
    use crate::tuning::Bf16Provider;
    use std::time::Instant;
    dispatch(ctx, a, weight, output, m, n, k, Bf16Provider::Mublas)?;
    let reference = values(ctx, output, m * n)?;
    let measure = |provider| -> Result<f64> {
        dispatch(ctx, a, weight, output, m, n, k, provider)?;
        ctx.synchronize()?;
        let mut times = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            for _ in 0..10 {
                dispatch(ctx, a, weight, output, m, n, k, provider)?;
            }
            ctx.synchronize()?;
            times.push(start.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        Ok(times[1])
    };
    let baseline = measure(Bf16Provider::Mublas)?;
    let mut winner = Bf16Provider::Mublas;
    let candidate = (|| -> Result<Option<f64>> {
        dispatch(ctx, a, weight, output, m, n, k, Bf16Provider::Mudnn)?;
        let actual = values(ctx, output, m * n)?;
        let (mut dot, mut n1, mut n2, mut error) = (0f64, 0f64, 0f64, 0f64);
        for (x, y) in reference.iter().zip(&actual) {
            if !x.is_finite() || !y.is_finite() {
                return Ok(None);
            }
            dot += x * y;
            n1 += x * x;
            n2 += y * y;
            error += (x - y) * (x - y);
        }
        // A degenerate reference cannot separate "computed correctly" from
        // "computed nothing": every kernel that writes zeros passes, and a
        // negative control fails too (wrong weights times zero is still zero).
        // The replay record rules exactly this way on the two recorded
        // GEMM shapes whose activations are all zero, and the verdict it reaches
        // is *undecided*, not passed. Accepting here would write a muDNN winner
        // bound to the device fingerprint on the strength of a timing alone, so
        // a zero reference stays on muBLAS.
        if n1 == 0. {
            return Ok(None);
        }
        let aligned = dot / (n1 * n2).sqrt() >= 0.99999 && (error / n1).sqrt() <= 0.005;
        if !aligned {
            return Ok(None);
        }
        Ok(Some(measure(Bf16Provider::Mudnn)?))
    })();
    match candidate {
        Ok(Some(time)) if time < baseline * 0.95 => winner = Bf16Provider::Mudnn,
        Err(error) => {
            ctx.synchronize()?;
            unsafe {
                let _ = ffi::musa::musaGetLastError();
            }
            eprintln!("MUSA BF16 candidate rejected {m}x{k}x{n}: {error}");
        }
        _ => {}
    }
    ctx.tuning().record_bf16(ctx, m, n, k, winner)?;
    dispatch(ctx, a, weight, output, m, n, k, winner)?;
    Ok(winner)
}

/// No toolkit: report the gap, do not fall back.
///
/// Deliberately not the eager bridge. The bridge is a separate, explicit
/// scaffold whose membership is tracked operator by operator; reaching it from
/// a native kernel would make GEMM's coverage invisible and hide a toolkit-less
/// build behind a working one. It would not run either: the fallback is
/// registered by `apxinf_py` and executes through torch_musa, which needs the
/// same toolkit.
#[cfg(not(apxinf_musa_available))]
fn launch(
    _ctx: &MusaContext,
    _activation: &Tensor,
    _weight: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _inner: usize,
    _cols: usize,
    _tactic: TacticDispatch,
) -> Result<()> {
    Err(native_unavailable("gemm::bf16"))
}

/// BF16 gate/up projection followed by GeGLU: `[M, K] @ [K, 2 * H] -> [M, H]`.
///
/// Native: a BF16 GEMM into a workspace gate buffer, then
/// [`crate::kernels::activation::geglu_bf16`] over it. mublasLt has no GeGLU
/// epilogue, so the fusion is in the composition rather than in one launch --
/// which is what the CUDA backend's `TacticBackend::GemmThenGeGlu` does too.
///
/// # Numerics: one extra rounding, deliberately
/// A torch bridge multiplies in **f32** and gates in f32, rounding once at the
/// output. The split rounds the GEMM product to bf16 in between,
/// because that is what a bf16 GEMM produces and what the operator's own
/// contract says the intermediate is. Measured ~0.2% relative on the product,
/// which lands well inside the 1%-of-peak gate but is the thinnest margin in
/// this batch -- the number comes from the replay, and the end-to-end
/// alignment is what confirms it downstream.
///
/// # Layouts
/// * `bf16_dual_geglu_interleaved` reports that the *primary* weight is packed
///   as gate/up tiles (`pi05/weights/bf16.rs`'s `interleave_gate_up_host`).
///   The bridge unpacked that on the GEMM output. **This port has no native
///   path for it**: the packing would have to be undone either by copying the
///   weight or by reading the gate buffer in a different order, and neither has
///   a recorded reference to be checked against. It is refused by name rather
///   than mis-multiplied.
/// * The other two name alternative packings, consumed by CUDA kernels this
///   port does not have. They are accepted and not read — hence the `_`, and
///   hence the names, which are the model layer's names for CUDA layouts. Not
///   reading them is the correct answer rather than a skipped one: the model
///   layer sets one layout mode at a time and leaves the primary tensor plain
///   whenever the dual flag is false, so the plain GEMM above is what those
///   modes want. The dual flag is what carries a request for a packing this
///   port cannot honor.
pub fn gemm_bf16_geglu_fused(
    ctx: &MusaContext,
    activation: &Tensor,
    packed_weight: &Tensor,
    bf16_dual_geglu_interleaved: bool,
    _bf16_dual_geglu_auto_interleaved: Option<&Tensor>,
    _bf16_sm89_geglu_interleaved: Option<&Tensor>,
) -> Result<Tensor> {
    let (rows, inner) = require_matrix_bf16(ctx, activation, "geglu gemm", "activation")?;
    let (weight_inner, packed_cols) =
        require_matrix_bf16(ctx, packed_weight, "geglu gemm", "weight")?;
    if inner != weight_inner {
        return Err(Error::Other(format!(
            "MUSA geglu gemm contraction mismatch: [{rows}x{inner}] @ [{weight_inner}x{packed_cols}]"
        )));
    }
    if packed_cols % 2 != 0 {
        return Err(Error::Other(format!(
            "MUSA geglu gemm needs an even packed width, got {packed_cols}"
        )));
    }
    if bf16_dual_geglu_interleaved {
        return Err(not_ported(
            "gemm::bf16_geglu_fused with bf16_dual_geglu_interleaved",
        ));
    }
    // No `observe_bf16` here: `gemm_bf16` below is the call that consumes the
    // operands, and it observes them itself. Observing twice would double-count
    // one logical GEMM in the calibration statistics.
    let gate = gemm_bf16(ctx, activation, packed_weight)?;
    crate::kernels::activation::geglu_bf16(ctx, &gate)
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    //! Device tests. They need a MUSA device, and `apxinf_musa_available` only
    //! records that the *toolkit* was found when this crate was built — so they
    //! are `#[ignore]`d, which also keeps `cargo test --workspace` passing on
    //! machines with neither.
    //!
    //! ```text
    //! cargo test -p apxinf-musa --lib -- --ignored --nocapture
    //! ```
    //!
    //! There are no device-free cases here, although it would be tidier: every
    //! validation this operator does (`dtype`, `device`, `shape`) runs against a
    //! `MusaContext`, and constructing one requires a device. A validation test
    //! that cannot be written is better than one that silently skips.

    use apxinf_core::{DType, Shape};

    use crate::context::MusaContext;
    use crate::kernels::gemm::bf16::{gemm_bf16, gemm_bf16_into_with};
    use crate::kernels::test_support::{
        from_bf16_bits, gate_for, read_back, to_bf16_bits, upload, values,
    };
    use crate::kernels::TacticDispatch;

    /// One shape, sampled the way the port's muBLAS probe samples: a
    /// few rows and columns, accumulated over the full contraction in f64.
    ///
    /// The gate is that probe's — the largest absolute error must stay under 1%
    /// of the output's own largest magnitude. It is an absolute gate because
    /// these outputs are sums of thousands of terms that cancel: a relative
    /// one reports noise on the elements that land near zero, which the probe's
    /// `vision qkv` case demonstrates (0.06 next to 59).
    fn check_shape(ctx: &MusaContext, m: usize, k: usize, n: usize) {
        let activation_values = values(m * k, 0x5eed_1234);
        let weight_values = values(k * n, 0x0bad_c0de);
        let activation = upload(ctx, &activation_values, Shape::new(vec![m, k]));
        let weight = upload(ctx, &weight_values, Shape::new(vec![k, n]));

        let out = gemm_bf16(ctx, &activation, &weight).expect("native gemm");
        ctx.stream().synchronize().expect("synchronise");
        assert_eq!(out.shape().dims(), &[m, n]);
        let got = read_back(&out);

        let tolerance = gate_for(&got);
        let rows = m.min(8);
        let cols = n.min(8);
        let mut worst = 0.0f32;
        for row in 0..rows {
            for col in 0..cols {
                let mut expected = 0.0f64;
                for inner in 0..k {
                    let a = f64::from(from_bf16_bits(to_bf16_bits(
                        activation_values[row * k + inner],
                    )));
                    let w = f64::from(from_bf16_bits(to_bf16_bits(weight_values[inner * n + col])));
                    expected += a * w;
                }
                let observed = f64::from(got[row * n + col]);
                let error = (observed - expected).abs() as f32;
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "[{m}x{k}]@[{k}x{n}] out[{row}][{col}]: {observed} vs {expected}, \
                     abs {error:.3e} > {tolerance:.3e}"
                );
            }
        }
        println!(
            "  [{m:>4} x {k:>5}] @ [{k:>5} x {n:>5}]  worst abs {worst:.3e} (gate {tolerance:.3e})"
        );
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn bf16_gemm_matches_a_double_precision_reference() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        // A shape small enough to reason about, then the four shapes the
        // original probe never covered: they carry 780 of the 878 calls that
        // fell outside it, and M=1 is further below one tile than the M=10
        // case that motivated probing at all. `(10, 1024, 32)` is the opposite
        // corner -- a 32-column output.
        for (m, k, n) in [
            (3, 5, 2),
            (1, 1024, 1024),
            (10, 32, 1024),
            (10, 1024, 32),
            (512, 588, 1152),
        ] {
            check_shape(&ctx, m, k, n);
        }
    }

    /// `allow_fallback = false` must not return success for a shape no record
    /// covers. `(3, 5, 2)` is outside the assembly's `k % 32` rule, so the
    /// provider path is the one under test; the context starts with an empty
    /// inference session.
    #[test]
    #[ignore = "needs a MUSA device"]
    fn a_required_exact_tactic_is_refused_without_a_record() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let (m, k, n) = (3, 5, 2);
        let activation = upload(&ctx, &values(m * k, 0x5eed_1234), Shape::new(vec![m, k]));
        let weight = upload(&ctx, &values(k * n, 0x0bad_c0de), Shape::new(vec![k, n]));
        let mut output = ctx
            .allocate_output(Shape::new(vec![m, n]), DType::BF16)
            .expect("an output");

        let error = gemm_bf16_into_with(
            &ctx,
            &activation,
            &weight,
            &mut output,
            TacticDispatch::RequireExact,
        )
        .expect_err("no tactic record covers this shape");
        assert!(
            error.to_string().contains("allow_fallback=false"),
            "{error}"
        );

        gemm_bf16_into_with(
            &ctx,
            &activation,
            &weight,
            &mut output,
            TacticDispatch::AllowFallback,
        )
        .expect("the muBLAS default is allowed");
    }
}

/// The fallback rule, without a device.
///
/// `choose_bf16` takes the resolved record and the autotune flag as values, so
/// every branch is reachable here; the device tests above cover what happens
/// after a provider is chosen.
#[cfg(test)]
mod policy_tests {
    use crate::tuning::{choose_bf16, Bf16Choice, Bf16Provider, ResolvedBf16, TacticMatch};

    fn resolved(provider: Bf16Provider, source: TacticMatch, recorded_m: usize) -> ResolvedBf16 {
        ResolvedBf16 {
            provider,
            source,
            recorded_m,
        }
    }

    #[test]
    fn an_exact_tactic_is_honoured_when_fallback_is_forbidden() {
        let decision = choose_bf16(
            Some(resolved(Bf16Provider::Mudnn, TacticMatch::Exact, 10)),
            false,
            true,
            10,
            1024,
            1024,
        )
        .expect("an exact record satisfies the constraint");
        assert!(matches!(decision, Bf16Choice::Exact(Bf16Provider::Mudnn)));
    }

    #[test]
    fn a_bucket_is_refused_when_fallback_is_forbidden() {
        let error = choose_bf16(
            Some(resolved(Bf16Provider::Mudnn, TacticMatch::Bucket, 10)),
            false,
            true,
            11,
            1024,
            1024,
        )
        .expect_err("a neighbouring record is an extrapolation, not an exact tactic");
        assert!(error.to_string().contains("neighbouring record"), "{error}");
    }

    #[test]
    fn a_miss_is_refused_when_fallback_is_forbidden() {
        let error = choose_bf16(None, false, true, 11, 1024, 1024)
            .expect_err("the muBLAS default is a fallback");
        assert!(error.to_string().contains("no tactic record"), "{error}");
    }

    #[test]
    fn autotuning_is_not_a_fallback() {
        let decision = choose_bf16(
            Some(resolved(Bf16Provider::Mudnn, TacticMatch::Bucket, 10)),
            true,
            true,
            11,
            1024,
            1024,
        )
        .expect("a measurement writes the exact record the caller demanded");
        assert_eq!(decision, Bf16Choice::Measure);
    }

    #[test]
    fn an_exact_record_is_not_re_measured() {
        let decision = choose_bf16(
            Some(resolved(Bf16Provider::Mudnn, TacticMatch::Exact, 11)),
            true,
            true,
            11,
            1024,
            1024,
        )
        .expect("an exact record");
        assert!(matches!(decision, Bf16Choice::Exact(Bf16Provider::Mudnn)));
    }

    #[test]
    fn allowed_fallback_keeps_the_bucket_and_the_default() {
        assert_eq!(
            choose_bf16(
                Some(resolved(Bf16Provider::Mudnn, TacticMatch::Bucket, 10)),
                false,
                false,
                11,
                1024,
                1024,
            )
            .expect("a bucket is served when it is allowed"),
            Bf16Choice::Bucket {
                provider: Bf16Provider::Mudnn,
                recorded_m: 10,
            }
        );
        assert_eq!(
            choose_bf16(None, false, false, 11, 1024, 1024).expect("the default is allowed"),
            Bf16Choice::Miss
        );
    }
}

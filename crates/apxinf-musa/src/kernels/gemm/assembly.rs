//! Native MP22 assembly GEMM with explicitly registered immutable NN or NT weights.
#[cfg(apxinf_musa_available)]
use crate::MusaContext;
#[cfg(apxinf_musa_available)]
use apxinf_core::{Result, Tensor};

/// Retain source allocations to prevent address reuse while their immutable
/// representations remain registered. NN retains the source without copying;
/// NT also retains a transposed buffer. Buffers live with the context.
#[derive(Default)]
pub(crate) struct PreparedWeights {
    entries: std::collections::HashMap<
        (usize, usize, usize),
        (apxinf_core::Tensor, Option<apxinf_core::Tensor>),
    >,
}

/// Prepare a fixed KN BF16 weight as contiguous NK once, outside capture.
/// The caller must not mutate the source after registration. Other targets
/// and unsupported K dimensions keep the vendor path and need no preparation.
pub fn prepare_bf16_nt_weight(
    ctx: &crate::MusaContext,
    weight: &apxinf_core::Tensor,
) -> apxinf_core::Result<()> {
    prepare(ctx, weight, true)
}

/// Register a fixed KN BF16 weight for NN without transposing or copying it.
/// The caller must not mutate the source after registration.
pub fn prepare_bf16_nn_weight(
    ctx: &crate::MusaContext,
    weight: &apxinf_core::Tensor,
) -> apxinf_core::Result<()> {
    prepare(ctx, weight, false)
}

#[cfg(not(apxinf_musa_available))]
fn prepare(
    _ctx: &crate::MusaContext,
    _weight: &apxinf_core::Tensor,
    _transpose: bool,
) -> apxinf_core::Result<()> {
    Err(super::super::native_unavailable(
        "assembly::prepare_bf16_weight",
    ))
}

#[cfg(apxinf_musa_available)]
fn prepare(ctx: &MusaContext, weight: &Tensor, transpose: bool) -> Result<()> {
    use super::super::contracts::{require_matrix_bf16, tensor_ptr};
    use crate::{buffer::MusaBuffer, error::musa_error, ffi::mubin};
    use apxinf_core::{DType, Shape};
    let (k, n) = require_matrix_bf16(ctx, weight, "ASM prepare", "weight")?;
    if !supports(1, n, k) {
        return Ok(());
    }
    if crate::workspace::is_capturing() {
        return Err(musa_error(
            "ASM weights must be registered before graph capture",
        ));
    }
    let key = (tensor_ptr(weight)? as usize, k, n);
    let mut prepared = ctx
        .asm_weights
        .lock()
        .map_err(|_| musa_error("ASM weight lock poisoned"))?;
    if let Some((_, nt)) = prepared.entries.get(&key) {
        if nt.is_some() != transpose {
            return Err(musa_error(
                "ASM weight already registered with another layout",
            ));
        }
        return Ok(());
    }
    let transposed = if transpose {
        let transposed = MusaBuffer::alloc_on(ctx, k * n * 2)
            .map_err(musa_error)?
            .into_tensor(Shape::new(vec![n, k]), DType::BF16);
        // Both tensors have validated BF16 storage; preparation owns the output
        // and orders this transpose before any decode launch on the same stream.
        let status = unsafe {
            mubin::apxinf_musa_asm_transpose(
                tensor_ptr(weight)?,
                tensor_ptr(&transposed)?,
                k as i32,
                n as i32,
                ctx.stream().handle(),
            )
        };
        mubin::check_asm_gemm(status, "asm transpose").map_err(musa_error)?;
        Some(transposed)
    } else {
        None
    };
    prepared.entries.insert(key, (weight.clone(), transposed));
    Ok(())
}

/// ASM images require a contraction axis divisible by 32 and int32 byte sizes.
/// Unsupported shapes keep the existing vendor path, without padding or copies.
fn supports(m: usize, n: usize, k: usize) -> bool {
    crate::MUSA_ARCH == "mp_22"
        && m > 0
        && n > 0
        && k > 0
        && k % 32 == 0
        && [(m, k), (k, n), (m, n)].iter().all(|&(a, b)| {
            a.checked_mul(b)
                .and_then(|v| v.checked_mul(2))
                .is_some_and(|bytes| bytes <= i32::MAX as usize)
        })
}

#[cfg(apxinf_musa_available)]
pub(super) fn launch(
    ctx: &MusaContext,
    a: &Tensor,
    w: &Tensor,
    out: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<bool> {
    launch_impl(ctx, a, w, out, m, n, k, true)
}

/// Only measured encode shapes with explicitly registered NN weights opt in.
/// Other shapes and unregistered weights retain the vendor path.
#[cfg(apxinf_musa_available)]
pub(super) fn launch_encode(
    ctx: &MusaContext,
    a: &Tensor,
    w: &Tensor,
    out: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<bool> {
    if !supports(m, n, k)
        || unsafe {
            crate::ffi::mubin::apxinf_musa_asm_encode_variant(m as i32, n as i32, k as i32)
        } < 0
    {
        return Ok(false);
    }
    launch_impl(ctx, a, w, out, m, n, k, false)
}

#[cfg(apxinf_musa_available)]
fn launch_impl(
    ctx: &MusaContext,
    a: &Tensor,
    w: &Tensor,
    out: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    decode: bool,
) -> Result<bool> {
    use super::super::contracts::tensor_ptr;
    use crate::{error::musa_error, ffi::mubin};
    if !supports(m, n, k) {
        return Ok(false);
    }
    let prepared = ctx
        .asm_weights
        .lock()
        .map_err(|_| apxinf_core::Error::Other("ASM weight lock poisoned".into()))?
        .entries
        .get(&(tensor_ptr(w)? as usize, k, n))
        .cloned();
    let Some((source, transposed)) = prepared else {
        if decode {
            return Err(apxinf_core::Error::Other(
                "ASM decode weight was not registered before inference".into(),
            ));
        }
        return Ok(false);
    };
    if !decode && transposed.is_some() {
        return Ok(false);
    }
    let weight = transposed.as_ref().unwrap_or(&source);
    // Shape checks above cover the int32 ABI and allocation bounds; the
    // context retains both source and packed weights for graph replay.
    let status = unsafe {
        mubin::apxinf_musa_asm_gemm(
            tensor_ptr(a)?,
            tensor_ptr(weight)?,
            tensor_ptr(out)?,
            m as i32,
            n as i32,
            k as i32,
            i32::from(transposed.is_some()),
            ctx.stream().handle(),
        )
    };
    mubin::check_asm_gemm(status, "asm bf16 gemm").map_err(musa_error)?;
    Ok(true)
}

#[cfg(test)]
mod support_tests {
    use super::*;
    #[test]
    fn unsupported_shapes_do_not_enter_assembly() {
        for shape in [
            (512, 1152, 588),
            (512, 1152, 4304),
            (0, 128, 32),
            (1, 0, 32),
            (1, 128, 0),
            (usize::MAX, 128, 32),
        ] {
            assert!(!supports(shape.0, shape.1, shape.2));
        }
        assert_eq!(supports(10, 1024, 1024), crate::MUSA_ARCH == "mp_22");
    }
}

#[cfg(all(test, apxinf_musa_available))]
mod device_tests {
    use super::*;
    use crate::kernels::test_support::{
        bf16_bytes, from_bf16_bits, read_back, to_bf16_bits, upload, values,
    };
    use apxinf_core::{Graph, Shape};
    use std::sync::Arc;

    #[test]
    #[ignore = "requires an MP22 MUSA device"]
    fn encode_nn_reference_and_graph_replay() {
        let ctx = Arc::new(MusaContext::new(0).unwrap());
        if crate::MUSA_ARCH != "mp_22" {
            return;
        }
        for (n, k) in [
            (2560, 2048),
            (2048, 2048),
            (16384, 2048),
            (32768, 2048),
            (2048, 16384),
        ] {
            let m = 522;
            let av = values(m * k, 1234);
            let wv = values(k * n, 5678);
            let a = upload(&ctx, &av, Shape::new(vec![m, k]));
            let w = upload(&ctx, &wv, Shape::new(vec![k, n]));
            // Before registration the public operator must still use the vendor.
            let vendor = super::super::bf16(&ctx, &a, &w).unwrap();
            ctx.synchronize().unwrap();
            let reference = read_back(&vendor);
            assert!(!launch_encode(&ctx, &a, &w, &vendor, m, n, k).unwrap());
            prepare_bf16_nn_weight(&ctx, &w).unwrap();
            let out = super::super::bf16(&ctx, &a, &w).unwrap();
            ctx.synchronize().unwrap();
            let expected = read_back(&out);
            assert!(expected.iter().all(|v| v.is_finite()));
            let error: f64 = expected
                .iter()
                .zip(&reference)
                .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
                .sum();
            let norm: f64 = reference.iter().map(|x| f64::from(*x).powi(2)).sum();
            assert!(
                (error / norm).sqrt() < 0.005,
                "encode reference {m}x{n}x{k}"
            );
            assert!(!launch_encode(&ctx, &a, &w, &out, 521, n, k).unwrap());
            let graph = crate::graph::capture(Arc::clone(&ctx), || {
                assert!(launch_encode(&ctx, &a, &w, &out, m, n, k)?);
                Ok(())
            })
            .unwrap();
            for scale in [-1f32, 0., 1.] {
                let changed: Vec<_> = av.iter().map(|v| v * scale).collect();
                crate::buffer::MusaBuffer::from_tensor(&a)
                    .unwrap()
                    .copy_from_host(&bf16_bytes(&changed))
                    .unwrap();
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                let got = read_back(&out);
                assert!(
                    got.iter().zip(&expected).all(|(x, y)| *x == *y * scale),
                    "encode graph {m}x{n}x{k} scale={scale}"
                );
                assert!(launch_encode(&ctx, &a, &w, &out, m, n, k).unwrap());
                ctx.synchronize().unwrap();
                assert_eq!(got, read_back(&out));
            }
        }
    }

    #[test]
    #[ignore = "requires an MP22 MUSA device"]
    fn decode_nn_nt_reference_and_graph_replay() {
        let ctx = Arc::new(MusaContext::new(0).unwrap());
        if crate::MUSA_ARCH != "mp_22" {
            return;
        }
        // All PI0.5 decode projection geometries, plus tile boundaries/tails.
        for (m, k, n) in [
            (10, 32, 1024),
            (10, 1024, 32),
            (10, 1024, 3072),
            (10, 2048, 1024),
            (10, 1024, 8192),
            (10, 4096, 1024),
            (1, 32, 7),
            (33, 64, 129),
            (129, 64, 256),
            (257, 64, 129),
            (513, 288, 1024),
        ] {
            for nt in [true, false] {
                let av = values(m * k, 1234);
                let wv = values(k * n, 5678);
                let a = upload(&ctx, &av, Shape::new(vec![m, k]));
                let w = upload(&ctx, &wv, Shape::new(vec![k, n]));
                prepare(&ctx, &w, nt).unwrap();
                assert!(prepare(&ctx, &w, !nt).is_err());
                let out = {
                    let _scope = super::super::decode_scope();
                    super::super::bf16(&ctx, &a, &w).unwrap()
                };
                ctx.synchronize().unwrap();
                let expected = read_back(&out);
                assert!(expected.iter().all(|v| v.is_finite()));
                // Independent f64 reference on sampled rows/columns, including tails.
                let gate = expected.iter().fold(0f64, |x, y| x.max(f64::from(y.abs()))) * 0.01;
                for row in [0, m / 2, m - 1] {
                    for col in [0, n / 2, n - 1] {
                        let mut sum = 0f64;
                        for i in 0..k {
                            sum += f64::from(from_bf16_bits(to_bf16_bits(av[row * k + i])))
                                * f64::from(from_bf16_bits(to_bf16_bits(wv[i * n + col])));
                        }
                        assert!(
                            (f64::from(expected[row * n + col]) - sum).abs() <= gate,
                            "reference {m}x{k}x{n} nt={nt} [{row},{col}]"
                        );
                    }
                }
                let graph = crate::graph::capture(Arc::clone(&ctx), || {
                    assert!(launch(&ctx, &a, &w, &out, m, n, k)?);
                    Ok(())
                })
                .unwrap();
                for scale in [1f32, -1., 0., 1.] {
                    let changed: Vec<_> = av.iter().map(|v| v * scale).collect();
                    crate::buffer::MusaBuffer::from_tensor(&a)
                        .unwrap()
                        .copy_from_host(&bf16_bytes(&changed))
                        .unwrap();
                    graph.replay().unwrap();
                    ctx.synchronize().unwrap();
                    let got = read_back(&out);
                    assert!(
                        got.iter().zip(&expected).all(|(x, y)| *x == *y * scale),
                        "graph output/repeat mismatch {m}x{k}x{n} nt={nt} scale={scale}"
                    );
                    assert!(launch(&ctx, &a, &w, &out, m, n, k).unwrap());
                    ctx.synchronize().unwrap();
                    assert_eq!(got, read_back(&out), "eager/graph mismatch");
                }
            }
        }
        let a = upload(&ctx, &values(32, 1), Shape::new(vec![1, 32]));
        let w = upload(&ctx, &values(32 * 8, 2), Shape::new(vec![32, 8]));
        let _scope = super::super::decode_scope();
        assert!(
            super::super::bf16(&ctx, &a, &w).is_err(),
            "unregistered weight must fail"
        );
    }
}

//! Rotary position embedding contracts.

use apxinf_core::{DType, Error, Result, Tensor};

pub use super::attention::QkvTensors;
#[cfg(apxinf_musa_available)]
use super::contracts::{checked_i32, optional_tensor_ptr, tensor_ptr};
use super::contracts::{bf16_tensor, require_optional_vector_bf16, require_output_tensor};
use crate::context::MusaContext;
#[cfg(apxinf_musa_available)]
use crate::ffi;
#[cfg(apxinf_musa_available)]
use crate::kernels::check_native;
#[cfg(not(apxinf_musa_available))]
use crate::kernels::native_unavailable;
use crate::kernels::not_ported;

/// Validate the packed QKV projection both rope entry points consume.
fn require_packed_qkv(
    ctx: &MusaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<usize> {
    let width = (q_heads + 2 * kv_heads)
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("MUSA rope projection width overflow".into()))?;
    let dims = qkv.shape().dims();
    if qkv.dtype() != DType::BF16
        || qkv.device() != apxinf_core::Device::Musa(ctx.device_id())
        || dims.len() != 2
        || dims[1] != width
    {
        return Err(Error::Other(format!(
            "MUSA rope expects a BF16 [tokens, {width}] projection, got {} {:?}",
            qkv.dtype(),
            dims
        )));
    }
    if head_dim == 0 || head_dim % 2 != 0 || head_dim > 256 {
        return Err(Error::Other(format!(
            "MUSA rope needs an even head dimension up to 256, got {head_dim}"
        )));
    }
    require_optional_vector_bf16(ctx, bias, width, "rope", "bias")?;
    Ok(dims[0])
}

/// Split a packed QKV projection, apply rotary position embedding to Q and K,
/// and return the three parts.
///
/// Native: `kernels/custom/rope.cuh`'s `qkv_rope_bf16_kernel`, through
/// `adapters/custom_kernels.mu`.
#[allow(clippy::too_many_arguments)]
pub fn split_qkv_apply_bf16(
    ctx: &MusaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<QkvTensors> {
    let tokens = require_packed_qkv(ctx, qkv, bias, q_heads, kv_heads, head_dim)?;
    let q = bf16_tensor(ctx, &[tokens, q_heads, head_dim])?;
    let k = bf16_tensor(ctx, &[tokens, kv_heads, head_dim])?;
    let v = bf16_tensor(ctx, &[tokens, kv_heads, head_dim])?;
    let (mut q, mut k, mut v) = (q, k, v);
    split_qkv_apply_bf16_into(
        ctx,
        qkv,
        bias,
        &mut q,
        &mut k,
        &mut v,
        q_heads,
        kv_heads,
        head_dim,
        theta,
        position_offset,
    )?;
    Ok(QkvTensors { q, k, v })
}

/// Write the split-and-RoPE result into caller-provided q/k/v outputs.
#[allow(clippy::too_many_arguments)]
pub fn split_qkv_apply_bf16_into(
    ctx: &MusaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q: &mut Tensor,
    k: &mut Tensor,
    v: &mut Tensor,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<()> {
    let tokens = require_packed_qkv(ctx, qkv, bias, q_heads, kv_heads, head_dim)?;
    require_output_tensor(ctx, q, &[tokens, q_heads, head_dim], "rope", "q")?;
    require_output_tensor(ctx, k, &[tokens, kv_heads, head_dim], "rope", "k")?;
    require_output_tensor(ctx, v, &[tokens, kv_heads, head_dim], "rope", "v")?;
    launch_rope(
        ctx,
        qkv,
        bias,
        q,
        k,
        v,
        tokens,
        q_heads,
        kv_heads,
        head_dim,
        theta,
        position_offset,
        0,
    )
}

/// Apply rotary position embedding to Q and **write K/V into the cache in
/// place**, returning only Q.
///
/// The caches are 2-D BF16 `[rows, head_dim]` and this operator is their
/// writer; the matching reader is `attention::mqa_bf16`, which is handed the
/// same two tensors. `kv_heads` must be 1 — the writer has no grouping to
/// scatter across.
///
/// `position_offset` (the RoPE phase) and `output_offset` (the first cache row
/// to write) are separate parameters, matching the CUDA contract. PI0.5 passes
/// the same value for both at every call site, because its cache rows and its
/// positions advance together; a caller that needed them to differ would have
/// a cache layout that disagrees with its positions.
#[allow(clippy::too_many_arguments)]
pub fn apply_q_write_kv_bf16(
    ctx: &MusaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    k_cache: &Tensor,
    v_cache: &Tensor,
    output_offset: usize,
) -> Result<Tensor> {
    let tokens = require_packed_qkv(ctx, qkv, bias, q_heads, kv_heads, head_dim)?;
    let mut q = bf16_tensor(ctx, &[tokens, q_heads, head_dim])?;
    apply_q_write_kv_bf16_into(
        ctx,
        qkv,
        bias,
        &mut q,
        q_heads,
        kv_heads,
        head_dim,
        theta,
        position_offset,
        k_cache,
        v_cache,
        output_offset,
    )?;
    Ok(q)
}

/// Write Q and the KV-cache rows, leaving the cache in place and the output
/// with the caller.
#[allow(clippy::too_many_arguments)]
pub fn apply_q_write_kv_bf16_into(
    ctx: &MusaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q: &mut Tensor,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    k_cache: &Tensor,
    v_cache: &Tensor,
    output_offset: usize,
) -> Result<()> {
    let tokens = require_packed_qkv(ctx, qkv, bias, q_heads, kv_heads, head_dim)?;
    if kv_heads != 1 {
        return Err(Error::Other(format!(
            "MUSA rope cache write requires kv_heads == 1, got {kv_heads}"
        )));
    }
    for (name, cache) in [("k_cache", k_cache), ("v_cache", v_cache)] {
        let dims = cache.shape().dims();
        // The device is checked here rather than by `require_matrix_bf16`: these
        // are written through a raw pointer by the kernel, so a same-shaped
        // tensor on another device would be a device fault rather than an error.
        if cache.dtype() != DType::BF16
            || cache.device() != apxinf_core::Device::Musa(ctx.device_id())
            || dims.len() != 2
            || dims[1] != head_dim
        {
            return Err(Error::Other(format!(
                "MUSA rope cache write expects a BF16 [rows, {head_dim}] {name} on MUSA{}, \
                 got {} {:?} on {}",
                ctx.device_id(),
                cache.dtype(),
                dims,
                cache.device()
            )));
        }
        let rows = dims[0];
        let end = output_offset.checked_add(tokens);
        if end.is_none_or(|end| end > rows) {
            return Err(Error::Other(format!(
                "MUSA rope cache write needs {tokens} rows at offset {output_offset} in {name}, which holds {rows}"
            )));
        }
    }
    require_output_tensor(ctx, q, &[tokens, q_heads, head_dim], "rope", "q")?;
    launch_rope(
        ctx,
        qkv,
        bias,
        q,
        k_cache,
        v_cache,
        tokens,
        q_heads,
        kv_heads,
        head_dim,
        theta,
        position_offset,
        output_offset,
    )
}

/// The launch, split by whether this build has a MUSA toolkit.
///
/// One launcher for both entry points, as on CUDA: the only difference between
/// them is where `k` and `v` point and what `kv_output_offset` is. `k` and `v`
/// are `&Tensor` here rather than the owned buffers `q` is, because in the cache
/// case they are the caller's tensors and not this operator's to allocate.
///
/// # Numerics
/// The kernel reproduces the bridge's rounding chain -- bf16 after the bias add,
/// bf16 for `cos`/`sin`, and a round for each multiply and each add -- rather
/// than the CUDA kernel's single final rounding, because the bridge is what
/// recorded the reference. See `kernels/custom/rope.cuh`; the two disagree on
/// 35.6% of elements, so this is not a matter of taste.
///
/// What is left to the device is `sinf`/`cosf` and `powf`: the bridge used
/// torch's, which run on the same device, so a difference can only come from a
/// different implementation rather than from the algorithm.
/// The port's device-math probe measures the gap over pi05's own angles -- 1 ulp
/// at worst, and zero bf16 rounding flips across all 67,691 of them.
#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_rope(
    ctx: &MusaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    kv_output_offset: usize,
) -> Result<()> {
    let tokens = checked_i32("rope", "tokens", tokens)?;
    let q_heads = checked_i32("rope", "q_heads", q_heads)?;
    let kv_heads = checked_i32("rope", "kv_heads", kv_heads)?;
    let head_dim = checked_i32("rope", "head_dim", head_dim)?;
    let position_offset = checked_i32("rope", "position_offset", position_offset)?;
    let kv_output_offset = checked_i32("rope", "kv_output_offset", kv_output_offset)?;
    let bias_pointer = optional_tensor_ptr(bias)?;
    // SAFETY: `require_packed_qkv` established the projection and the optional
    // bias; `q` is a fresh `[tokens, q_heads, head_dim]` allocation and `k`/`v`
    // are either the same shape or the `[rows, head_dim]` caches whose
    // `[kv_output_offset, kv_output_offset + tokens)` rows were checked to
    // exist. Every buffer is owned by a caller `Tensor` that outlives the
    // asynchronous launch on `ctx.stream()`.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_qkv_rope_bf16(
                tensor_ptr(qkv)?,
                bias_pointer,
                tensor_ptr(q)?,
                tensor_ptr(k)?,
                tensor_ptr(v)?,
                tokens,
                q_heads,
                kv_heads,
                head_dim,
                theta,
                position_offset,
                kv_output_offset,
                ctx.stream().handle(),
            ),
            "rope",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch_rope(
    _ctx: &MusaContext,
    _qkv: &Tensor,
    _bias: Option<&Tensor>,
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _tokens: usize,
    _q_heads: usize,
    _kv_heads: usize,
    _head_dim: usize,
    _theta: f32,
    _position_offset: usize,
    _kv_output_offset: usize,
) -> Result<()> {
    Err(native_unavailable("rope::split_qkv_apply_bf16"))
}

// ── FP16 rotary embedding (not ported) ───────────────────────────────────

/// Split QKV and apply rotary embedding in FP16.
#[allow(clippy::too_many_arguments)]
pub fn split_qkv_apply_f16(
    ctx: &MusaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<QkvTensors> {
    let _ = (ctx, qkv, bias, q_heads, kv_heads, head_dim, theta, position_offset);
    Err(not_ported("rope::split_qkv_apply_f16"))
}

/// Apply rotary embedding to Q and write K/V into the cache, in FP16.
///
/// The CUDA signature names the last parameter `kv_output_offset`; the cache
/// row and the RoPE phase are separate there, as in the BF16 sibling.
#[allow(clippy::too_many_arguments)]
pub fn apply_q_write_kv_f16(
    ctx: &MusaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    k_cache: &Tensor,
    v_cache: &Tensor,
    kv_output_offset: usize,
) -> Result<Tensor> {
    let _ = (
        ctx,
        qkv,
        bias,
        q_heads,
        kv_heads,
        head_dim,
        theta,
        position_offset,
        k_cache,
        v_cache,
        kv_output_offset,
    );
    Err(not_ported("rope::apply_q_write_kv_f16"))
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    //! Device test for the rotary position embedding.
    //!
    //! The replay harness covers both entry points and both come out
    //! **bit-identical** to the recorded tensors, which is the strongest
    //! evidence this port has for any operator. Two things it cannot reach:
    //!
    //! * the **bias** branch -- neither recorded rope call carried one, because
    //!   PI0.5 folds the q/k/v bias into the projection;
    //! * `kv_heads > 1` -- the recorded call has one, which makes the
    //!   `(row * kv_heads + head) * head_dim` destination address degenerate to
    //!   `row * head_dim`. A wrong stride there would still have matched.
    //!
    //! Plus the one refusal a single-device host can still construct: a cache
    //! tensor that is not on this device. The kernel writes the caches through
    //! a raw pointer, so that has to be an error rather than a fault.
    //!
    //! The assertion is the usual 1%-of-peak gate rather than bit equality.
    //! Not because the kernel is approximate, but because the reference below
    //! computes `powf`/`cos`/`sin` with the **host** libm while the kernel uses
    //! the device's, and the two are different implementations of the same
    //! function: the port's device-math probe measures 1 ulp between them, which
    //! is far below this gate and not provably zero.
    //!
    //! Needs a MUSA device, so it is `#[ignore]`d:
    //! `cargo test -p apxinf-musa --lib -- --ignored --nocapture`.

    use apxinf_core::{DType, Shape, Tensor};

    use crate::context::MusaContext;
    use crate::kernels::rope::{apply_q_write_kv_bf16, split_qkv_apply_bf16};
    use crate::kernels::test_support::{
        from_bf16_bits, gate_for, read_back, to_bf16_bits, upload, values,
    };

    /// The kernel's arithmetic, spelled out: bias in f32, bf16 for the operands
    /// and for `cos`/`sin`, and a round for each multiply and each combine.
    fn reference(
        qkv: &[f32],
        bias: Option<&[f32]>,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        theta: f32,
        position_offset: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let round = |value: f32| from_bf16_bits(to_bf16_bits(value));
        let half = head_dim / 2;
        let q_width = q_heads * head_dim;
        let kv_width = kv_heads * head_dim;
        let fused = q_width + 2 * kv_width;
        let (mut q, mut k, mut v) = (
            vec![0.0f32; tokens * q_width],
            vec![0.0f32; tokens * kv_width],
            vec![0.0f32; tokens * kv_width],
        );
        for token in 0..tokens {
            let position = (position_offset + token) as f32;
            for pair in 0..half {
                let frequency = theta.powf(-(pair as f32) / (half as f32));
                let angle = position * frequency;
                let (cosine, sine) = (round(angle.cos()), round(angle.sin()));
                let rotate = |first: f32, second: f32| {
                    let (first, second) = (round(first), round(second));
                    (
                        round(round(first * cosine) - round(second * sine)),
                        round(round(second * cosine) + round(first * sine)),
                    )
                };
                let segment = |base: usize| {
                    let (first, second) = (
                        qkv[token * fused + base + pair],
                        qkv[token * fused + base + half + pair],
                    );
                    match bias {
                        Some(bias) => (first + bias[base + pair], second + bias[base + half + pair]),
                        None => (first, second),
                    }
                };
                for head in 0..q_heads {
                    let base = head * head_dim;
                    let (first, second) = segment(base);
                    let (a, b) = rotate(first, second);
                    q[token * q_width + base + pair] = a;
                    q[token * q_width + base + half + pair] = b;
                }
                for head in 0..kv_heads {
                    let base = q_width + head * head_dim;
                    let (first, second) = segment(base);
                    let (a, b) = rotate(first, second);
                    k[token * kv_width + base - q_width + pair] = a;
                    k[token * kv_width + base - q_width + half + pair] = b;
                }
                for head in 0..kv_heads {
                    let base = q_width + kv_width + head * head_dim;
                    let (first, second) = segment(base);
                    let destination = token * kv_width + head * head_dim;
                    v[destination + pair] = round(first);
                    v[destination + half + pair] = round(second);
                }
            }
        }
        (q, k, v)
    }

    fn check(
        ctx: &MusaContext,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        bias: bool,
    ) {
        let (theta, position_offset) = (10000.0f32, 5usize);
        let width = (q_heads + 2 * kv_heads) * head_dim;
        let qkv_values = values(tokens * width, 0x5eed_1234);
        let bias_values = values(width, 0x0bad_c0de);
        let qkv = upload(ctx, &qkv_values, Shape::new(vec![tokens, width]));
        let bias_tensor = bias.then(|| upload(ctx, &bias_values, Shape::new(vec![width])));

        let split = split_qkv_apply_bf16(
            ctx,
            &qkv,
            bias_tensor.as_ref(),
            q_heads,
            kv_heads,
            head_dim,
            theta,
            position_offset,
        )
        .expect("native rope split");
        ctx.stream().synchronize().expect("synchronise");

        // The kernel consumes bf16 operands, so the reference has to see the
        // same rounded inputs rather than the f32 values that were uploaded.
        let round = |value: f32| from_bf16_bits(to_bf16_bits(value));
        let rounded: Vec<f32> = qkv_values.iter().map(|v| round(*v)).collect();
        let rounded_bias: Vec<f32> = bias_values.iter().map(|v| round(*v)).collect();
        let (q, k, v) = reference(
            &rounded,
            bias.then_some(rounded_bias.as_slice()),
            tokens,
            q_heads,
            kv_heads,
            head_dim,
            theta,
            position_offset,
        );

        for (name, tensor, expected) in [("q", &split.q, &q), ("k", &split.k, &k), ("v", &split.v, &v)]
        {
            let got = read_back(tensor);
            let tolerance = gate_for(&got);
            let worst = got
                .iter()
                .zip(expected.iter())
                .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
            assert!(
                worst <= tolerance,
                "{name} [{tokens},{q_heads},{kv_heads},{head_dim}]: worst abs {worst:.3e} \
                 exceeds gate {tolerance:.3e}"
            );
            println!("  rope {name} q_heads {q_heads} kv_heads {kv_heads} bias {bias}  worst abs {worst:.3e}");
        }
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn rope_matches_a_reference_with_a_bias_and_several_kv_heads() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        check(&ctx, 3, 2, 2, 8, true);
        check(&ctx, 2, 4, 1, 6, false);
        // The shape PI0.5 actually uses, minus the token count.
        check(&ctx, 2, 8, 1, 256, false);
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn rope_refuses_a_cache_on_another_device() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let (tokens, q_heads, kv_heads, head_dim) = (2usize, 2usize, 1usize, 8usize);
        let width = (q_heads + 2 * kv_heads) * head_dim;
        let qkv = upload(&ctx, &values(tokens * width, 0x5eed_1234), Shape::new(vec![tokens, width]));

        // The caches are written through a raw pointer, so a same-shaped tensor
        // on the wrong device is a fault rather than a wrong answer.
        let k_cache = Tensor::zeros(Shape::new(vec![8, head_dim]), DType::BF16);
        let v_cache = upload(&ctx, &values(8 * head_dim, 0x0bad_c0de), Shape::new(vec![8, head_dim]));
        let error = apply_q_write_kv_bf16(
            &ctx, &qkv, None, q_heads, kv_heads, head_dim, 10000.0, 0, &k_cache, &v_cache, 0,
        )
        .expect_err("a host cache must be refused");
        assert!(
            error.to_string().contains("k_cache on MUSA0"),
            "unhelpful refusal: {error}"
        );
        println!("  rope cache write refused: {error}");
    }
}

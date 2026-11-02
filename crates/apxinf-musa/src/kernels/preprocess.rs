//! Input preprocessing operator contracts.

use apxinf_core::{DType, Error, Result, Tensor};

#[cfg(apxinf_musa_available)]
use super::contracts::{checked_i32, tensor_ptr};
use super::contracts::{
    require_buffers, require_dtype_on, require_finite, require_output_tensor,
};
use crate::buffer::MusaBuffer;
use crate::context::MusaContext;
#[cfg(apxinf_musa_available)]
use crate::ffi;
#[cfg(apxinf_musa_available)]
use crate::kernels::check_native;
#[cfg(not(apxinf_musa_available))]
use crate::kernels::native_unavailable;
use crate::kernels::not_ported;

/// Memory layout of a fixed-shape batch of RGB `uint8` images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLayout {
    /// `[views, image_size, image_size, 3]`.
    Nhwc,
    /// `[views, 3, image_size, image_size]`.
    Nchw,
}

impl ImageLayout {
    fn name(self) -> &'static str {
        match self {
            Self::Nhwc => "nhwc",
            Self::Nchw => "nchw",
        }
    }

    /// The code the C ABI takes: 0 for NHWC, 1 for NCHW.
    ///
    /// Private, like the CUDA backend's: the number is an agreement between
    /// this facade and `adapters/custom_kernels.mu`, not something a caller
    /// may depend on. `name()` is the caller-facing spelling, and the values
    /// here have to match CUDA's so that a build of either backend compiles
    /// against the same model source.
    #[cfg(apxinf_musa_available)]
    fn kernel_value(self) -> i32 {
        match self {
            Self::Nhwc => 0,
            Self::Nchw => 1,
        }
    }
}

impl std::fmt::Display for ImageLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Fused image preprocessing into a BF16 patch-major layout.
///
/// Writes `patches` in place: `views * (image_size / patch_size)^2` rows of
/// `3 * patch_size^2`, row-major over the patch grid and channel-major within a
/// row. `layout` describes the *input* only.
///
/// Native: one gather plus the affine `(u8 / 255) * 2 - 1`, rounded once, in
/// `kernels/custom/preprocess.cuh` through `adapters/custom_kernels.mu`. There
/// is no resize and no per-channel statistic here — the frames are already
/// resized when they arrive, and PaliGemma's SigLIP patch embedding expects
/// `[-1, 1]`. The kernel's device test asserts bit equality against a host
/// reference, which is the only reason the arithmetic is spelled out with
/// explicit rounding intrinsics.
#[allow(clippy::too_many_arguments)]
pub fn rgb_u8_to_patches_bf16(
    ctx: &MusaContext,
    images: &MusaBuffer,
    patches: &Tensor,
    views: usize,
    image_size: usize,
    patch_size: usize,
    layout: ImageLayout,
) -> Result<()> {
    if views == 0 || image_size == 0 || patch_size == 0 || image_size % patch_size != 0 {
        return Err(Error::Other(
            "invalid static inference BF16 image preprocessing shape".into(),
        ));
    }
    let expected_bytes = views
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_mul(image_size))
        .and_then(|bytes| bytes.checked_mul(image_size))
        .ok_or_else(|| Error::Other("preprocess image byte count overflow".into()))?;
    let side = image_size / patch_size;
    let expected_shape = [views * side * side, 3 * patch_size * patch_size];
    require_buffers(
        ctx,
        "preprocess::rgb_u8_to_patches_bf16",
        &[("images", images, expected_bytes)],
    )?;
    require_output_tensor(
        ctx,
        patches,
        &expected_shape,
        "preprocess::rgb_u8_to_patches_bf16",
        "patches",
    )?;
    launch_rgb_u8_to_patches_bf16(ctx, images, patches, views, image_size, patch_size, layout)
}

#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_rgb_u8_to_patches_bf16(
    ctx: &MusaContext,
    images: &MusaBuffer,
    patches: &Tensor,
    views: usize,
    image_size: usize,
    patch_size: usize,
    layout: ImageLayout,
) -> Result<()> {
    let views = checked_i32("image preprocessing", "views", views)?;
    let image_size = checked_i32("image preprocessing", "image_size", image_size)?;
    let patch_size = checked_i32("image preprocessing", "patch_size", patch_size)?;
    // SAFETY: the tensor contract above established a `views * 3 * image_size^2`
    // byte device buffer and a BF16 `patches` of exactly the matching element
    // count on this context's device, and both outlive the asynchronous launch
    // on `ctx.stream()`. The adapter re-checks the shape relation and the layout
    // code before selecting an instantiation.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_rgb_u8_to_patches_bf16(
                images.ptr(),
                tensor_ptr(patches)?,
                views,
                image_size,
                patch_size,
                layout.kernel_value(),
                ctx.stream().handle(),
            ),
            "preprocess::rgb_u8_to_patches_bf16",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch_rgb_u8_to_patches_bf16(
    _ctx: &MusaContext,
    _images: &MusaBuffer,
    _patches: &Tensor,
    _views: usize,
    _image_size: usize,
    _patch_size: usize,
    _layout: ImageLayout,
) -> Result<()> {
    Err(native_unavailable("preprocess::rgb_u8_to_patches_bf16"))
}

/// Fused image preprocessing into a static FP8 (E4M3) patch-major layout.
///
/// Normalization, channel-first patch reordering, FP16 rounding and the static
/// scale all happen in one stream-ordered step.
///
/// Not ported, for a second reason on top of the one above: `scale` belongs to
/// the FP8 precision route, which is not ported either. Every other FP8 entry in
/// the facade answers the same way.
#[allow(clippy::too_many_arguments)]
pub fn rgb_u8_to_patches_e4m3(
    ctx: &MusaContext,
    images: &MusaBuffer,
    patches: &Tensor,
    views: usize,
    image_size: usize,
    patch_size: usize,
    _layout: ImageLayout,
    scale: f32,
) -> Result<()> {
    if views == 0 || image_size == 0 || patch_size == 0 || image_size % patch_size != 0 {
        return Err(Error::Other(
            "invalid static inference FP8 image preprocessing shape".into(),
        ));
    }
    require_finite("static inference FP8 image preprocessing", &[scale])?;
    if scale <= 0.0 {
        return Err(Error::Other(
            "static inference FP8 image preprocessing scale must be positive".into(),
        ));
    }
    let expected_bytes = views
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_mul(image_size))
        .and_then(|bytes| bytes.checked_mul(image_size))
        .ok_or_else(|| Error::Other("preprocess image byte count overflow".into()))?;
    let side = image_size / patch_size;
    let expected_shape = [views * side * side, 3 * patch_size * patch_size];
    require_buffers(
        ctx,
        "preprocess::rgb_u8_to_patches_e4m3",
        &[("images", images, expected_bytes)],
    )?;
    require_dtype_on(
        ctx,
        patches,
        DType::F8E4M3,
        "preprocess::rgb_u8_to_patches_e4m3",
        "patches",
    )?;
    if patches.shape().dims() != expected_shape {
        return Err(Error::Other(format!(
            "MUSA preprocess::rgb_u8_to_patches_e4m3 patches expects {expected_shape:?}, got {:?}",
            patches.shape().dims()
        )));
    }
    Err(not_ported("preprocess::rgb_u8_to_patches_e4m3"))
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    //! Device tests for raw RGB to BF16 patches.
    //!
    //! These assert **bit equality** against a host reference, which is
    //! stronger than the 1%-of-peak gate the arithmetic-heavy operators use,
    //! and it is available for a reason: this kernel computes one division, one
    //! multiply by two and one decrement over an integer input, so it touches
    //! no libm and there is no host/device implementation difference to absorb.
    //! The bits are therefore reproducible, and a mismatch is a defect rather
    //! than a rounding.
    //!
    //! The reference is written in Rust here and asserts bit equality against
    //! the kernel; the arithmetic is one division, one multiply by two and one
    //! decrement over an integer input, so a mismatch is a defect rather than a
    //! rounding.
    //!
    //! Needs a MUSA device, so every case is `#[ignore]`d:
    //! `cargo test -p apxinf-musa --lib -- --ignored --nocapture`.

    use apxinf_core::Tensor;

    use crate::buffer::MusaBuffer;
    use crate::context::MusaContext;
    use crate::kernels::contracts::{bf16_tensor, f32_tensor};
    use crate::kernels::preprocess::{rgb_u8_to_patches_bf16, ImageLayout};
    use crate::kernels::test_support::to_bf16_bits;

    /// Bytes that are cheap to recompute on either side of the comparison.
    /// `131` is odd, so `i * 131` alone walks every residue class mod 256; the
    /// second term breaks the pattern the first one would otherwise impose on
    /// neighbouring elements.
    fn image_bytes(count: usize) -> Vec<u8> {
        (0..count)
            .map(|i| ((i * 131 + (i >> 6) * 17) % 256) as u8)
            .collect()
    }

    fn upload_rgb(ctx: &MusaContext, bytes: &[u8]) -> MusaBuffer {
        let buffer = MusaBuffer::alloc_on(ctx, bytes.len()).expect("allocate images");
        buffer.copy_from_host(bytes).expect("upload images");
        buffer
    }

    fn output_bits(tensor: &Tensor) -> Vec<u16> {
        let buffer = MusaBuffer::from_tensor(tensor).expect("device tensor");
        let mut bytes = vec![0u8; tensor.size_in_bytes()];
        buffer.copy_to_host(&mut bytes).expect("read back");
        bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect()
    }

    /// The host reference: the same index derivation and the same arithmetic
    /// the kernel's comments promise, in f32 with one bf16 rounding at the end.
    /// Rust does not contract floating point by default, so the plain
    /// expression here is the same sequence the kernel spells with
    /// `__fdiv_rn`/`__fmul_rn`/`__fsub_rn`.
    fn reference(bytes: &[u8], views: usize, image_size: usize, patch_size: usize,
                 nhwc: bool) -> Vec<u16> {
        let side = image_size / patch_size;
        let patch_area = patch_size * patch_size;
        let patch_width = 3 * patch_area;
        let mut out = Vec::with_capacity(views * side * side * patch_width);
        for patch_index in 0..views * side * side {
            let view = patch_index / (side * side);
            let patch_in_view = patch_index % (side * side);
            let (patch_y, patch_x) = (patch_in_view / side, patch_in_view % side);
            for patch_element in 0..patch_width {
                let channel = patch_element / patch_area;
                let pixel = patch_element % patch_area;
                let (dy, dx) = (pixel / patch_size, pixel % patch_size);
                let (y, x) = (patch_y * patch_size + dy, patch_x * patch_size + dx);
                let index = if nhwc {
                    ((view * image_size + y) * image_size + x) * 3 + channel
                } else {
                    ((view * 3 + channel) * image_size + y) * image_size + x
                };
                let value = bytes[index] as f32;
                out.push(to_bf16_bits(((value / 255.0) * 2.0) - 1.0));
            }
        }
        out
    }

    /// FNV-1a over the output bits, so an offline numpy cross-check can compare
    /// a digest instead of 300k values through the console.
    fn digest(bits: &[u16]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for value in bits {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    /// Upload `bytes`, run the operator over them, and hand back the patch bits.
    ///
    /// Every positive case here does the same five steps — size the output,
    /// allocate it, upload the frames, launch, synchronise, read back — so they
    /// live in one place and a case is only what it asserts.
    fn patch_bits(
        ctx: &MusaContext,
        bytes: &[u8],
        views: usize,
        image_size: usize,
        patch_size: usize,
        layout: ImageLayout,
    ) -> Vec<u16> {
        let side = image_size / patch_size;
        let patches = bf16_tensor(ctx, &[views * side * side, 3 * patch_size * patch_size])
            .expect("allocate patches");
        let images = upload_rgb(ctx, bytes);
        rgb_u8_to_patches_bf16(ctx, &images, &patches, views, image_size, patch_size, layout)
            .expect("native preprocess");
        ctx.stream().synchronize().expect("synchronise");
        output_bits(&patches)
    }

    fn match_rate(got: &[u16], expected: &[u16]) -> f64 {
        let hits = got
            .iter()
            .zip(expected.iter())
            .filter(|(a, b)| a == b)
            .count();
        hits as f64 / got.len() as f64
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn rgb_u8_to_patches_bf16_normalizes_and_orders_exactly() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        // The model's own shape first, then a patch size that is not 14 and
        // does not divide the image into a power-of-two grid.
        for (views, image_size, patch_size) in [(2usize, 224usize, 14usize), (1, 30, 5)] {
            let bytes = image_bytes(views * 3 * image_size * image_size);
            let got = patch_bits(&ctx, &bytes, views, image_size, patch_size, ImageLayout::Nhwc);
            let expected = reference(&bytes, views, image_size, patch_size, true);

            // Non-vacuous before equal: a kernel that wrote one constant would
            // still match on a degenerate input, and a reference that collapsed
            // to one value could not separate anything.
            let distinct_input: std::collections::HashSet<u8> = bytes.iter().copied().collect();
            assert_eq!(distinct_input.len(), 256, "input must cover every uint8 value");
            let distinct_expected: std::collections::HashSet<u16> =
                expected.iter().copied().collect();
            assert!(
                distinct_expected.len() >= 200,
                "reference collapsed to {} distinct bit patterns",
                distinct_expected.len()
            );

            assert_eq!(
                got, expected,
                "views={views} image_size={image_size} patch_size={patch_size}"
            );
            println!(
                "preprocess replay: views={views} image_size={image_size} patch_size={patch_size} \
                 elements={} digest=0x{:016x} distinct={}",
                got.len(),
                digest(&got),
                distinct_expected.len()
            );
        }
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn rgb_u8_to_patches_bf16_covers_the_whole_value_domain() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        // One view, one patch covering the whole image: 16 * 16 * 3 = 768
        // outputs, and with `byte[i] = i % 256` each uint8 value lands exactly
        // three times. The operator's output domain is 256 values wide, so this
        // case exercises all of them.
        let (image_size, patch_size) = (16usize, 16usize);
        let bytes: Vec<u8> = (0..3 * image_size * image_size)
            .map(|i| (i % 256) as u8)
            .collect();
        let got = patch_bits(&ctx, &bytes, 1, image_size, patch_size, ImageLayout::Nhwc);

        let expected: std::collections::BTreeSet<u16> =
            (0..=255u32).map(|v| to_bf16_bits(((v as f32) / 255.0) * 2.0 - 1.0)).collect();
        let produced: std::collections::BTreeSet<u16> = got.iter().copied().collect();
        assert_eq!(
            produced, expected,
            "the produced 256-value domain must be exactly the reference's"
        );
        let mut counts = std::collections::BTreeMap::new();
        for bit in &got {
            *counts.entry(*bit).or_insert(0usize) += 1;
        }
        assert!(
            counts.values().all(|count| *count == 3),
            "each value should appear once per channel, i.e. three times"
        );
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn rgb_u8_to_patches_bf16_reads_both_layouts() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let (views, image_size, patch_size) = (2usize, 8usize, 2usize);
        let planar = image_bytes(views * 3 * image_size * image_size);
        // NHWC(i, y, x, c) == NCHW(i, c, y, x).
        let mut interleaved = vec![0u8; planar.len()];
        for view in 0..views {
            for channel in 0..3 {
                for y in 0..image_size {
                    for x in 0..image_size {
                        interleaved[((view * image_size + y) * image_size + x) * 3 + channel] =
                            planar[((view * 3 + channel) * image_size + y) * image_size + x];
                    }
                }
            }
        }
        let mut outputs = Vec::new();
        for (bytes, layout) in [
            (&interleaved, ImageLayout::Nhwc),
            (&planar, ImageLayout::Nchw),
        ] {
            outputs.push(patch_bits(&ctx, bytes, views, image_size, patch_size, layout));
        }
        assert_eq!(
            outputs[0], outputs[1],
            "the same image in either layout must produce the same patches"
        );
        assert_eq!(outputs[0], reference(&interleaved, views, image_size, patch_size, true));
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn rgb_u8_to_patches_bf16_refuses_a_shape_it_cannot_lay_out() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let (views, image_size, patch_size) = (1usize, 8usize, 2usize);
        let side = image_size / patch_size;
        let bytes = image_bytes(views * 3 * image_size * image_size);

        // image_size not divisible by patch_size.
        let images = upload_rgb(&ctx, &bytes);
        let patches = bf16_tensor(&ctx, &[views * side * side, 3 * patch_size * patch_size])
            .expect("allocate");
        assert!(rgb_u8_to_patches_bf16(&ctx, &images, &patches, views, 7, patch_size, ImageLayout::Nhwc)
            .is_err());

        // A patch tensor of the wrong dtype.
        let wrong_dtype =
            f32_tensor(&ctx, &[views * side * side, 3 * patch_size * patch_size]).expect("allocate");
        let error = rgb_u8_to_patches_bf16(
            &ctx, &images, &wrong_dtype, views, image_size, patch_size, ImageLayout::Nhwc,
        )
        .expect_err("an f32 patch tensor must be refused");
        assert!(error.to_string().contains("mismatch"), "{error}");

        // A patch tensor of the wrong shape.
        let wrong_shape = bf16_tensor(&ctx, &[views * side * side + 1, 3 * patch_size * patch_size])
            .expect("allocate");
        assert!(
            rgb_u8_to_patches_bf16(&ctx, &images, &wrong_shape, views, image_size, patch_size, ImageLayout::Nhwc)
                .is_err()
        );

        // Fewer image bytes than the shape needs.
        let short = upload_rgb(&ctx, &bytes[..bytes.len() - 1]);
        assert!(
            rgb_u8_to_patches_bf16(&ctx, &short, &patches, views, image_size, patch_size, ImageLayout::Nhwc)
                .is_err()
        );

        // Zero-sized dimensions, which the adapter would otherwise divide by.
        assert!(rgb_u8_to_patches_bf16(&ctx, &images, &patches, 0, image_size, patch_size, ImageLayout::Nhwc)
            .is_err());
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn rgb_u8_to_patches_bf16_negative_controls_separate() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let (views, image_size, patch_size) = (2usize, 224usize, 14usize);
        let bytes = image_bytes(views * 3 * image_size * image_size);
        let got = patch_bits(&ctx, &bytes, views, image_size, patch_size, ImageLayout::Nhwc);

        let expected = reference(&bytes, views, image_size, patch_size, true);
        assert_eq!(match_rate(&got, &expected), 1.0, "the positive control");

        // Three wrong answers, none of which may look like the right one. A
        // control that does not separate is replaced, not dropped: it is not
        // evidence that the test can tell a wrong kernel from a right one.
        let mut shifted = expected.clone();
        shifted.rotate_left(1);
        let offset = match_rate(&got, &shifted);

        let layout_swap = match_rate(&got, &reference(&bytes, views, image_size, patch_size, false));

        let no_scale: Vec<u16> = bytes
            .iter()
            .map(|v| to_bf16_bits((*v as f32) * 2.0 - 1.0))
            .collect();
        let unscaled = match_rate(&got, &no_scale);

        println!(
            "negative controls: offset={offset:.6} layout_swap={layout_swap:.6} no_255={unscaled:.6}"
        );
        for (name, rate) in [("offset", offset), ("layout_swap", layout_swap), ("no_255", unscaled)]
        {
            assert!(rate < 0.05, "negative control {name} matched {rate}");
        }
    }
}

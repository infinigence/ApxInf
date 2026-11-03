//! Gather operations — EmbeddingLookup, BiasPosition, RgbToPatches.
//!
//! Uses `aclnnEmbedding` for token embeddings, `aclnnAdd` for bias/position.

use apxinf_core::{DType, Device, Error, Result, Shape, Storage, Tensor};

use crate::l3::buffer::AscendBuffer;
use crate::l3::context::AscendNewContext;

// ── Args structs ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatherSemantic {
    EmbeddingLookup,
    BiasPosition,
    RgbToPatches,
}

#[derive(Clone, Copy, Debug)]
pub struct GatherPatchGeometry {
    pub views: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub nhwc: bool,
}

pub struct GatherArgs<'a> {
    pub semantic: GatherSemantic,
    pub input: Option<&'a Tensor>,
    pub ids: Option<&'a AscendBuffer>,
    pub images: Option<&'a AscendBuffer>,
    pub bias: Option<&'a Tensor>,
    pub position: Option<&'a Tensor>,
    pub out: &'a mut Tensor,
    pub vocab_size: usize,
    pub tokens_per_view: usize,
    pub patches: Option<GatherPatchGeometry>,
}

impl<'a> GatherArgs<'a> {
    pub fn new(semantic: GatherSemantic, input: &'a Tensor, out: &'a mut Tensor) -> Self {
        Self {
            semantic,
            input: Some(input),
            ids: None,
            images: None,
            bias: None,
            position: None,
            out,
            vocab_size: 0,
            tokens_per_view: 0,
            patches: None,
        }
    }

    pub fn rgb_to_patches(
        images: &'a AscendBuffer,
        out: &'a mut Tensor,
        patches: GatherPatchGeometry,
    ) -> Self {
        Self {
            semantic: GatherSemantic::RgbToPatches,
            input: None,
            ids: None,
            images: Some(images),
            bias: None,
            position: None,
            out,
            vocab_size: 0,
            tokens_per_view: 0,
            patches: Some(patches),
        }
    }
}

// ── Implementation ────────────────────────────────────────────────────

pub fn gather(ctx: &AscendNewContext, args: GatherArgs<'_>) -> Result<()> {
    match args.semantic {
        GatherSemantic::EmbeddingLookup => gather_embedding(ctx, args),
        GatherSemantic::BiasPosition => gather_bias_position(ctx, args),
        GatherSemantic::RgbToPatches => gather_rgb_to_patches(ctx, args),
    }
}

/// Token embedding lookup via `aclnnEmbedding`.
fn gather_embedding(ctx: &AscendNewContext, args: GatherArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.input.expect("EmbeddingLookup requires input").dtype();

    let table = args.input.unwrap();
    let ids_buf = args.ids.ok_or_else(|| Error::Other("EmbeddingLookup requires ids".into()))?;

    // ids_buf is a DeviceBuffer of u32 token IDs — create an aclTensor
    let id_count = ids_buf.size() / 4; // u32 = 4 bytes
    let ids_shape = Shape::new(vec![id_count]);
    let acl_table = crate::acl_tensor::AclTensor::from_device(
        device_ptr(table)?, table.shape(), dtype,
    );
    let acl_ids = crate::acl_tensor::AclTensor::from_device(
        ids_buf.raw_ptr(), &ids_shape, DType::I32,
    );
    let acl_out = crate::acl_tensor::AclTensor::from_device(
        device_ptr(args.out)?, args.out.shape(), dtype,
    );

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnEmbeddingGetWorkspaceSize(
                acl_table.raw(),
                acl_ids.raw(),
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnEmbedding(ws, ws_size, exec, s)
        },
        stream,
        "aclnnEmbedding",
    )
    .map_err(|e| Error::Other(format!("gather embedding: {e}")))?;

    // Scale by sqrt(embed_dim) as in the CUDA kernel
    let embed_dim = table.shape().dim(-1)? as f32;
    let scale = embed_dim.sqrt();
    let acl_out = crate::acl_tensor::AclTensor::from_device(
        device_ptr(args.out)?, args.out.shape(), dtype,
    );
    let acl_scaled = crate::acl_tensor::AclTensor::from_device(
        device_ptr(args.out)?, args.out.shape(), dtype,
    );
    let acl_scale = crate::acl_tensor::AclScalar::f32(scale);

    // In-place scaling: out = out * sqrt(embed_dim) via aclnnMuls
    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnMulsGetWorkspaceSize(
                acl_out.raw(),
                acl_scale.raw(),
                acl_scaled.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnMuls(ws, ws_size, exec, s)
        },
        stream,
        "aclnnMuls (embedding scale)",
    )
    .map_err(|e| Error::Other(format!("gather embedding scale: {e}")))?;

    Ok(())
}

/// Bias + position embedding: out = input + bias + position
fn gather_bias_position(ctx: &AscendNewContext, args: GatherArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.input.expect("BiasPosition requires input").dtype();
    let shape = args.input.unwrap().shape().clone();

    let input_ptr = device_ptr(args.input.unwrap())?;
    let out_ptr = device_ptr(args.out)?;
    let acl_alpha = crate::acl_tensor::AclScalar::f32(1.0);

    // out = input + bias
    if let Some(bias) = args.bias {
        let acl_input = crate::acl_tensor::AclTensor::from_device(input_ptr, &shape, dtype);
        let acl_bias = crate::acl_tensor::AclTensor::from_device(device_ptr(bias)?, bias.shape(), dtype);
        let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, &shape, dtype);

        crate::acl_tensor::run_op(
            |ws, exec| unsafe {
                crate::ffi::aclnn::aclnnAddGetWorkspaceSize(
                    acl_input.raw(),
                    acl_bias.raw(),
                    acl_alpha.raw(),
                    acl_out.raw(),
                    ws,
                    exec,
                )
            },
            |ws, ws_size, exec, s| unsafe {
                crate::ffi::aclnn::aclnnAdd(ws, ws_size, exec, s)
            },
            stream,
            "aclnnAdd (bias_position bias)",
        )
        .map_err(|e| Error::Other(format!("gather bias_position bias: {e}")))?;
    } else {
        // Just copy input → out via aclnnInplaceCopy (RI-capture compatible)
        super::aclnn_d2d_copy(out_ptr, input_ptr, shape.numel(), dtype, stream)?;
    }

    // out = out + position
    // NOTE: aclnnAdd does not support in-place operation (self == out returns
    // 161002 ACLNN_ERR_PARAM_INVALID). We allocate a temporary buffer, compute
    // temp = out + pos, then D2D-copy temp → out.
    // NOTE: position_embedding may have shape [patches_per_view, width] while
    // out has [num_views * patches_per_view, width]. In that case, tile
    // position to match out's shape before adding.
    if let Some(position) = args.position {
        let pos_shape = position.shape().clone();
        let out_rows = shape.dims()[0];
        let pos_rows = pos_shape.dims()[0];
        let cols = shape.dims()[shape.dims().len() - 1];

        // Determine the effective position tensor to use (tile if needed)
        let (pos_ptr, pos_eff_shape) = if pos_rows < out_rows && out_rows % pos_rows == 0 {
            // Tile position: copy pos_rows repeatedly into a new buffer
            let num_repeats = out_rows / pos_rows;
            let pos_bytes = pos_rows * cols * dtype.size_in_bytes();
            let total_bytes = out_rows * cols * dtype.size_in_bytes();
            let tiled_buf = crate::l3::buffer::AscendBuffer::alloc_zeros_session_aware(total_bytes, ctx.device_id())?;
            let pos_numel = pos_rows * cols;
            let pos_ptr = device_ptr(position)?;
            for i in 0..num_repeats {
                super::aclnn_d2d_copy(
                    tiled_buf.raw_ptr() + i * pos_bytes,
                    pos_ptr,
                    pos_numel,
                    dtype,
                    stream,
                )?;
            }
            (tiled_buf.raw_ptr(), Shape::new(vec![out_rows, cols]))
        } else {
            (device_ptr(position)?, pos_shape)
        };

        // out += pos via aclnnInplaceAdd (avoids temp buffer + copy-back)
        super::aclnn_inplace_add(
            out_ptr,
            pos_ptr,
            &shape,
            &pos_eff_shape,
            dtype,
            stream,
        )
        .map_err(|e| Error::Other(format!("gather bias_position pos inplace add: {e}")))?;
    }

    Ok(())
}

/// RGB to patches: CPU preprocessing → H2D
/// Converts uint8 RGB images to normalized BF16 patches.
/// Mirrors the CUDA kernel `rgb_u8_to_patches`: each pixel is normalized
/// to [-1, 1] via `(pixel / 255.0) * 2.0 - 1.0`, then patches are flattened
/// in channel-major order (channel → dy → dx).
fn gather_rgb_to_patches(ctx: &AscendNewContext, args: GatherArgs<'_>) -> Result<()> {
    let images = args.images.expect("RgbToPatches requires images");
    let patches = args.patches.expect("RgbToPatches requires patches geometry");

    let image_bytes = images.size();
    let mut cpu_images = vec![0u8; image_bytes];
    crate::transfers::d2h(
        cpu_images.as_mut_ptr() as *mut std::ffi::c_void,
        images.raw_ptr() as *const std::ffi::c_void,
        image_bytes,
    )
    .map_err(|e| Error::Other(format!("rgb_to_patches d2h: {e}")))?;

    let views = patches.views;
    let img_size = patches.image_size;
    let patch_size = patches.patch_size;
    let nhwc = patches.nhwc;
    let patches_per_side = img_size / patch_size;
    let patches_per_view = patches_per_side * patches_per_side;
    let patch_area = patch_size * patch_size;
    let patch_width = 3 * patch_area; // = embed_dim per patch
    let total_patches = views * patches_per_view;

    let out_dims = args.out.shape().dims();
    let embed_dim = out_dims[out_dims.len() - 1];
    // Output is BF16
    let mut cpu_patches = vec![0u8; total_patches * embed_dim * 2]; // BF16 = 2 bytes

    for view in 0..views {
        for patch_in_view in 0..patches_per_view {
            let patch_y = patch_in_view / patches_per_side;
            let patch_x = patch_in_view - patch_y * patches_per_side;
            for channel in 0..3usize {
                for pixel_in_patch in 0..patch_area {
                    let dy = pixel_in_patch / patch_size;
                    let dx = pixel_in_patch - dy * patch_size;
                    let y = patch_y * patch_size + dy;
                    let x = patch_x * patch_size + dx;

                    let input_index = if nhwc {
                        ((view * img_size + y) * img_size + x) * 3 + channel
                    } else {
                        ((view * 3 + channel) * img_size + y) * img_size + x
                    };

                    let normalized =
                        (cpu_images[input_index] as f32 / 255.0) * 2.0 - 1.0;

                    let patch_index = view * patches_per_view + patch_in_view;
                    let patch_element = channel * patch_area + pixel_in_patch;
                    let output_index = patch_index * patch_width + patch_element;

                    // Write BF16 bytes (little-endian)
                    let bf16_bits = bf16_from_f32(normalized);
                    cpu_patches[output_index * 2] = (bf16_bits & 0xFF) as u8;
                    cpu_patches[output_index * 2 + 1] = (bf16_bits >> 8) as u8;
                }
            }
        }
    }

    let patches_bytes = total_patches * embed_dim * 2; // BF16
    let out_ptr = device_ptr(args.out)?;
    crate::transfers::h2d(
        out_ptr as *mut std::ffi::c_void,
        cpu_patches.as_ptr() as *const std::ffi::c_void,
        patches_bytes,
    )
    .map_err(|e| Error::Other(format!("rgb_to_patches h2d: {e}")))?;

    Ok(())
}

/// Convert f32 to BF16 bits (round to nearest even).
fn bf16_from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    // Round to nearest even on the BF16 truncation boundary
    let rounded = bits + 0x7FFF + ((bits >> 16) & 1);
    (rounded >> 16) as u16
}

// ── Helpers ───────────────────────────────────────────────────────────

fn device_ptr(tensor: &Tensor) -> Result<usize> {
    match tensor.storage() {
        Storage::Gpu { handle, .. } => Ok(handle.ptr()),
        _ => Err(Error::Other(format!(
            "expected NPU tensor, got {:?}",
            tensor.device()
        ))),
    }
}

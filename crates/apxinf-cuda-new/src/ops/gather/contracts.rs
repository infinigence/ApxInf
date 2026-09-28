use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::gather as abi;
use crate::{CudaBuffer, CudaContext, CudaDeviceAddress};

/// Which gather / layout operation the bindings describe. See
/// `gather_types.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GatherSemantic {
    /// Token-embedding gather, scaled by `sqrt(cols)`.
    EmbeddingLookup,
    /// Projection + learned position embedding, plus an optional bias.
    BiasPosition,
    /// RGB u8 images to flattened patches in `[-1, 1]`.
    RgbToPatches,
}

impl GatherSemantic {
    fn code(self) -> u32 {
        match self {
            Self::EmbeddingLookup => 0,
            Self::BiasPosition => 1,
            Self::RgbToPatches => 2,
        }
    }

    fn may_have_bias(self) -> bool {
        matches!(self, Self::BiasPosition)
    }
}

/// Geometry for [`GatherSemantic::RgbToPatches`].
#[derive(Clone, Copy, Debug)]
pub struct PatchGeometry {
    pub views: usize,
    pub image_size: usize,
    pub patch_size: usize,
    /// True when the source images are NHWC rather than NCHW.
    pub nhwc: bool,
}

pub struct GatherArgs<'a> {
    pub semantic: GatherSemantic,
    /// Embedding table or activation, unused for raw-image preprocessing.
    pub input: Option<&'a Tensor>,
    /// Token ids for [`GatherSemantic::EmbeddingLookup`].
    pub ids: Option<&'a CudaBuffer>,
    /// Stable device address for graph-replayed token ids. This is mutually
    /// exclusive with [`Self::ids`]. It exists for host-mapped control
    /// buffers whose contents change between graph replays while their device
    /// address remains fixed.
    pub ids_address: Option<CudaDeviceAddress>,
    /// Packed u8 images for [`GatherSemantic::RgbToPatches`].
    pub images: Option<&'a CudaBuffer>,
    pub bias: Option<&'a Tensor>,
    /// Learned position embedding for [`GatherSemantic::BiasPosition`].
    pub position: Option<&'a Tensor>,
    pub out: &'a mut Tensor,
    pub vocab_size: usize,
    pub tokens_per_view: usize,
    pub patches: Option<PatchGeometry>,
}

impl<'a> GatherArgs<'a> {
    pub fn new(semantic: GatherSemantic, input: &'a Tensor, out: &'a mut Tensor) -> Self {
        Self {
            semantic,
            input: Some(input),
            ids: None,
            ids_address: None,
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
        images: &'a CudaBuffer,
        out: &'a mut Tensor,
        geometry: PatchGeometry,
    ) -> Self {
        Self {
            semantic: GatherSemantic::RgbToPatches,
            input: None,
            ids: None,
            ids_address: None,
            images: Some(images),
            bias: None,
            position: None,
            out,
            vocab_size: 0,
            tokens_per_view: 0,
            patches: Some(geometry),
        }
    }

    /// Bind token ids through a stable device address. The pointed-to values
    /// are read by the kernel at launch/replay time and are deliberately not
    /// part of the normalized structural spec.
    pub fn with_ids_address(mut self, ids: CudaDeviceAddress) -> Self {
        self.ids = None;
        self.ids_address = Some(ids);
        self
    }
}

pub(crate) struct Normalized {
    pub spec: abi::Spec,
    pub bindings: abi::Bindings,
    pub storage: Vec<CudaBuffer>,
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        _ => Err(invalid("Gather currently supports F16 and BF16")),
    }
}

fn alignment_of(pointer: usize) -> u32 {
    if pointer == 0 {
        return 256;
    }
    (1usize << pointer.trailing_zeros().min(8)) as u32
}

fn buffer_of(ctx: &CudaContext, tensor: &Tensor, minimum_bytes: usize) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id()) {
        return Err(invalid("Gather tensor is not on this device"));
    }
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    if buffer.len() < minimum_bytes {
        return Err(invalid("Gather tensor storage is too small"));
    }
    Ok(buffer)
}

fn typed_buffer(
    ctx: &CudaContext,
    tensor: &Tensor,
    dtype: DType,
    elements: usize,
) -> Result<CudaBuffer> {
    if tensor.dtype() != dtype {
        return Err(invalid("Gather tensor dtype mismatch"));
    }
    let bytes = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| invalid("Gather size overflow"))?;
    buffer_of(ctx, tensor, bytes)
}

pub(crate) fn normalize(ctx: &CudaContext, args: GatherArgs<'_>) -> Result<Normalized> {
    let semantic = args.semantic;
    let dims = args.out.shape().dims();
    if dims.len() != 2 {
        return Err(invalid("Gather output must be rank 2 [rows, cols]"));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if rows == 0 || cols == 0 {
        return Err(invalid("Gather output must be non-empty"));
    }
    let rows_abi = i32::try_from(rows)
        .map_err(|_| invalid("Gather row count exceeds the CUDA kernel range"))?;
    let cols_abi = i32::try_from(cols)
        .map_err(|_| invalid("Gather column count exceeds the CUDA kernel range"))?;
    let dtype = args.out.dtype();
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("Gather size overflow"))?;

    if args.bias.is_some() && !semantic.may_have_bias() {
        return Err(invalid("Gather semantic does not take a bias"));
    }
    match semantic {
        GatherSemantic::EmbeddingLookup => {
            if args.images.is_some() || args.position.is_some() || args.patches.is_some() {
                return Err(invalid("embedding lookup received unrelated bindings"));
            }
        }
        GatherSemantic::BiasPosition => {
            if args.ids.is_some()
                || args.ids_address.is_some()
                || args.images.is_some()
                || args.patches.is_some()
            {
                return Err(invalid("bias-position received unrelated bindings"));
            }
        }
        GatherSemantic::RgbToPatches => {
            if args.input.is_some()
                || args.ids.is_some()
                || args.ids_address.is_some()
                || args.bias.is_some()
                || args.position.is_some()
            {
                return Err(invalid("RGB-to-patches received unrelated bindings"));
            }
        }
    }

    let mut storage = Vec::new();

    // The input is a u8 image buffer for RgbToPatches and a typed tensor
    // otherwise, so its size check differs per semantic.
    let input_buffer = match semantic {
        GatherSemantic::EmbeddingLookup => {
            let vocab = args.vocab_size;
            if vocab == 0 {
                return Err(invalid("embedding lookup requires a vocabulary size"));
            }
            let elements = vocab
                .checked_mul(cols)
                .ok_or_else(|| invalid("Gather size overflow"))?;
            typed_buffer(
                ctx,
                args.input
                    .ok_or_else(|| invalid("embedding lookup requires a table"))?,
                dtype,
                elements,
            )?
        }
        GatherSemantic::BiasPosition => typed_buffer(
            ctx,
            args.input
                .ok_or_else(|| invalid("bias-position requires an input"))?,
            dtype,
            count,
        )?,
        GatherSemantic::RgbToPatches => {
            let geometry = args
                .patches
                .ok_or_else(|| invalid("RGB-to-patches requires a geometry"))?;
            if geometry.views == 0
                || geometry.image_size == 0
                || geometry.patch_size == 0
                || geometry.image_size % geometry.patch_size != 0
            {
                return Err(invalid("invalid RGB-to-patches geometry"));
            }
            let patches_per_side = geometry.image_size / geometry.patch_size;
            let expected_rows = geometry
                .views
                .checked_mul(patches_per_side)
                .and_then(|value| value.checked_mul(patches_per_side))
                .ok_or_else(|| invalid("Gather patch row count overflow"))?;
            let expected_cols = geometry
                .patch_size
                .checked_mul(geometry.patch_size)
                .and_then(|value| value.checked_mul(3))
                .ok_or_else(|| invalid("Gather patch width overflow"))?;
            if rows != expected_rows || cols != expected_cols {
                return Err(invalid("RGB output shape disagrees with the geometry"));
            }
            let pixels = geometry
                .views
                .checked_mul(geometry.image_size)
                .and_then(|value| value.checked_mul(geometry.image_size))
                .and_then(|value| value.checked_mul(3))
                .ok_or_else(|| invalid("Gather size overflow"))?;
            let images = args
                .images
                .ok_or_else(|| invalid("RGB-to-patches requires an image buffer"))?;
            if images.device() != ctx.device_id() || images.len() != pixels {
                return Err(invalid("RGB image buffer device/size mismatch"));
            }
            images.clone()
        }
    };
    let input = input_buffer.ptr() as *const std::ffi::c_void;
    let input_alignment = alignment_of(input_buffer.ptr() as usize);
    storage.push(input_buffer);

    if args.ids.is_some() && args.ids_address.is_some() {
        return Err(invalid(
            "embedding lookup accepts either an owned id buffer or a stable id address, not both",
        ));
    }
    let ids = match (args.ids, args.ids_address) {
        (Some(buffer), None) => {
            let required = rows
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or_else(|| invalid("Gather token-id size overflow"))?;
            if buffer.device() != ctx.device_id() || buffer.len() < required {
                return Err(invalid("Gather token-id buffer device/size mismatch"));
            }
            let buffer = buffer.clone();
            let pointer = buffer.ptr() as *const u32;
            storage.push(buffer);
            pointer
        }
        (None, Some(address)) => {
            let required = rows
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or_else(|| invalid("Gather token-id size overflow"))?;
            if address.device() != ctx.device_id() || address.len() < required {
                return Err(invalid("Gather token-id address device/size mismatch"));
            }
            address.ptr().cast_const().cast::<u32>()
        }
        (None, None) if matches!(semantic, GatherSemantic::EmbeddingLookup) => {
            return Err(invalid("embedding lookup requires token ids"))
        }
        (None, None) => std::ptr::null(),
        (Some(_), Some(_)) => unreachable!("exclusive token-id bindings validated above"),
    };

    let (bias, bias_alignment) = match args.bias {
        Some(tensor) => {
            let buffer = typed_buffer(ctx, tensor, dtype, cols)?;
            let pointer = buffer.ptr() as *const std::ffi::c_void;
            let alignment = alignment_of(buffer.ptr() as usize);
            storage.push(buffer);
            (pointer, alignment)
        }
        None => (std::ptr::null(), 256),
    };

    let tokens_per_view = match semantic {
        GatherSemantic::BiasPosition => {
            if args.tokens_per_view == 0 || rows % args.tokens_per_view != 0 {
                return Err(invalid(
                    "bias-position rows must be a whole number of views",
                ));
            }
            args.tokens_per_view
        }
        _ => 0,
    };

    let position = match args.position {
        Some(tensor) => {
            let elements = tokens_per_view
                .checked_mul(cols)
                .ok_or_else(|| invalid("Gather size overflow"))?;
            let buffer = typed_buffer(ctx, tensor, dtype, elements)?;
            let pointer = buffer.ptr() as *const std::ffi::c_void;
            storage.push(buffer);
            pointer
        }
        None if matches!(semantic, GatherSemantic::BiasPosition) => {
            return Err(invalid("bias-position requires a position embedding"))
        }
        None => std::ptr::null(),
    };

    let output_buffer = typed_buffer(ctx, args.out, dtype, count)?;
    let output = output_buffer.ptr() as *mut std::ffi::c_void;
    let output_alignment = alignment_of(output_buffer.ptr() as usize);
    storage.push(output_buffer);

    let geometry = args.patches.unwrap_or(PatchGeometry {
        views: 0,
        image_size: 0,
        patch_size: 0,
        nhwc: false,
    });

    let spec = abi::Spec {
        version: abi::SPEC_VERSION,
        semantic: semantic.code(),
        dtype: dtype_code(dtype)?,
        has_bias: u32::from(!bias.is_null()),
        vocab_size: if matches!(semantic, GatherSemantic::EmbeddingLookup) {
            u32::try_from(args.vocab_size)
                .map_err(|_| invalid("Gather vocabulary size exceeds the ABI range"))?
        } else {
            0
        },
        tokens_per_view: u32::try_from(tokens_per_view)
            .map_err(|_| invalid("Gather tokens-per-view exceeds the ABI range"))?,
        views: u32::try_from(geometry.views)
            .map_err(|_| invalid("Gather view count exceeds the ABI range"))?,
        image_size: u32::try_from(geometry.image_size)
            .map_err(|_| invalid("Gather image size exceeds the ABI range"))?,
        patch_size: u32::try_from(geometry.patch_size)
            .map_err(|_| invalid("Gather patch size exceeds the ABI range"))?,
        nhwc: u32::from(geometry.nhwc),
        input_alignment,
        bias_alignment,
        output_alignment,
        rows: i64::from(rows_abi),
        cols: i64::from(cols_abi),
    };

    let bindings = abi::Bindings {
        input,
        ids,
        bias,
        position,
        output,
        stream: ctx.stream().handle() as abi::CudaStream,
    };

    Ok(Normalized {
        spec,
        bindings,
        storage,
    })
}

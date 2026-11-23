//! Shared checkpoint-owned RGB canonicalization for Qwen-style vision towers.
//! Image container decoding belongs to the application; layout, dtype and
//! image-token expansion belong to the loaded model.
use apxinf_core::{DType, Error, Result, Tensor};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct DecodedImage {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Debug)]
pub struct VisionContract {
    pub model: String,
    pub max_images: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub image_token_id: u32,
    pub pixel_dtype: DType,
    pub image_mean: [f64; 3],
    pub image_std: [f64; 3],
    pub rescale_factor: f64,
}

pub struct PreparedImage {
    pub pixel_values: Vec<f32>,
    pub grid_thw: [u32; 3],
    pub resized: (usize, usize),
    pub image_tokens: usize,
}

pub struct PreparedVision {
    pub tokens: Vec<u32>,
    pub pixels: Tensor,
    pub grid: Vec<[u32; 3]>,
}

impl VisionContract {
    /// Parse fixed transforms from local checkpoint files before loading weights.
    /// Omitted flags/rescale_factor use the documented Qwen2VL fast-processor
    /// defaults; geometry and normalization arrays are never inferred.
    pub fn from_dir(
        path: &Path,
        model: &str,
        max_images: usize,
        pixel_dtype: DType,
    ) -> Result<Self> {
        let read = |name: &str| -> Result<Value> {
            let file = path.join(name);
            let bytes = std::fs::read(&file)
                .map_err(|e| Error::Other(format!("{}: {e}", file.display())))?;
            serde_json::from_slice(&bytes)
                .map_err(|e| Error::Other(format!("{}: {e}", file.display())))
        };
        let config = read("config.json")?;
        let p = read("preprocessor_config.json")?;
        let fail = |field: &str| {
            Error::Other(format!(
                "{}: unsupported or missing field {field}",
                path.join("preprocessor_config.json").display()
            ))
        };
        let number = |value: &Value, field: &str| {
            value
                .as_u64()
                .and_then(|x| usize::try_from(x).ok())
                .filter(|x| *x > 0)
                .ok_or_else(|| fail(field))
        };
        let geometry = |key: &str, other: &str| -> Result<usize> {
            let n = number(&p[key], key)?;
            let v = &config["vision_config"][other];
            if v.as_u64() != Some(n as u64) {
                return Err(fail(&format!(
                    "{key} disagrees with config.json:vision_config.{other}"
                )));
            }
            Ok(n)
        };
        for flag in ["do_resize", "do_rescale", "do_normalize", "do_convert_rgb"] {
            if p.get(flag).is_some_and(|v| v != true) {
                return Err(fail(flag));
            }
        }
        if p.get("resample").is_some_and(|v| v.as_u64() != Some(3)) {
            return Err(fail("resample"));
        }
        if p.get("image_processor_type")
            .and_then(Value::as_str)
            .is_some_and(|v| {
                !matches!(
                    v,
                    "Qwen2VLImageProcessorFast"
                        | "Qwen2VLImageProcessor"
                        | "Qwen3VLImageProcessorFast"
                        | "Qwen3VLImageProcessor"
                )
            })
        {
            return Err(fail("image_processor_type"));
        }
        let array = |field: &str| -> Result<[f64; 3]> {
            let values = p[field]
                .as_array()
                .filter(|x| x.len() == 3)
                .ok_or_else(|| fail(field))?;
            let mut result = [0.0; 3];
            for (i, x) in values.iter().enumerate() {
                result[i] = x
                    .as_f64()
                    .filter(|x| x.is_finite())
                    .ok_or_else(|| fail(field))?;
            }
            Ok(result)
        };
        let min_pixels = number(
            p.get("min_pixels").unwrap_or(&p["size"]["shortest_edge"]),
            "size.shortest_edge/min_pixels",
        )?;
        let max_pixels = number(
            p.get("max_pixels").unwrap_or(&p["size"]["longest_edge"]),
            "size.longest_edge/max_pixels",
        )?;
        let rescale_factor = match p.get("rescale_factor") {
            Some(v) => v
                .as_f64()
                .filter(|x| x.is_finite() && *x > 0.0)
                .ok_or_else(|| fail("rescale_factor"))?,
            None => 1.0 / f64::from(u8::MAX),
        };
        let result = Self {
            model: model.into(),
            max_images,
            min_pixels,
            max_pixels,
            patch_size: geometry("patch_size", "patch_size")?,
            temporal_patch_size: geometry("temporal_patch_size", "temporal_patch_size")?,
            spatial_merge_size: geometry("merge_size", "spatial_merge_size")?,
            image_token_id: config["image_token_id"]
                .as_u64()
                .and_then(|x| u32::try_from(x).ok())
                .ok_or_else(|| {
                    Error::Other(format!(
                        "{}: unsupported or missing field image_token_id",
                        path.join("config.json").display()
                    ))
                })?,
            pixel_dtype,
            image_mean: array("image_mean")?,
            image_std: array("image_std")?,
            rescale_factor,
        };
        let factor = result
            .patch_size
            .checked_mul(result.spatial_merge_size)
            .ok_or_else(|| fail("patch_size * merge_size"))?;
        if min_pixels > max_pixels || factor.checked_mul(factor).is_none_or(|x| x > max_pixels) {
            return Err(fail("size"));
        }
        if result.image_std.iter().any(|x| *x <= 0.0) {
            return Err(fail("image_std"));
        }
        if result
            .image_std
            .iter()
            .chain(&result.image_mean)
            .any(|x| !((*x / rescale_factor) as f32).is_finite())
        {
            return Err(fail("rescale_factor"));
        }
        if result
            .patch_size
            .checked_mul(result.patch_size)
            .and_then(|n| n.checked_mul(result.temporal_patch_size))
            .and_then(|n| n.checked_mul(3))
            .is_none()
        {
            return Err(fail("temporal_patch_size"));
        }
        Ok(result)
    }

    pub fn prepare(
        &self,
        tokens: &[u32],
        images: &[DecodedImage],
        max_prompt_tokens: usize,
    ) -> Result<PreparedVision> {
        if images.is_empty() || images.len() > self.max_images {
            return Err(Error::Other(format!(
                "request has {} images; model '{}' declares max_images={}",
                images.len(),
                self.model,
                self.max_images
            )));
        }
        let mut counts = Vec::new();
        for image in images {
            let (h, w) = smart_resize(
                image.height,
                image.width,
                self.patch_size * self.spatial_merge_size,
                self.min_pixels,
                self.max_pixels,
            )?;
            counts.push(
                (h / self.patch_size) * (w / self.patch_size)
                    / (self.spatial_merge_size * self.spatial_merge_size),
            );
        }
        let markers = tokens
            .iter()
            .filter(|&&id| id == self.image_token_id)
            .count();
        if markers != images.len() {
            return Err(Error::Other(format!(
                "image placeholder count {markers} does not match {} decoded images",
                images.len()
            )));
        }
        let expanded = counts
            .iter()
            .try_fold(tokens.len() - markers, |n, c| n.checked_add(*c))
            .ok_or_else(|| Error::Other("image token count overflow".into()))?;
        if expanded > max_prompt_tokens {
            return Err(Error::Other(format!("image-expanded prompt has {expanded} tokens, exceeding the admitted prompt limit {max_prompt_tokens}")));
        }
        let mut output = Vec::with_capacity(expanded);
        let mut next = 0;
        for &id in tokens {
            if id == self.image_token_id {
                output.extend(std::iter::repeat_n(id, counts[next]));
                next += 1;
            } else {
                output.push(id);
            }
        }
        let mut values = Vec::new();
        let mut grids = Vec::new();
        for image in images {
            let prepared = preprocess(self, &image.rgb, image.width, image.height)?;
            grids.push(prepared.grid_thw);
            values.extend(prepared.pixel_values);
        }
        let features = 3 * self.temporal_patch_size * self.patch_size * self.patch_size;
        let shape = vec![values.len() / features, features];
        let pixels = match self.pixel_dtype {
            DType::F32 => Tensor::from_f32(shape, &values)?,
            DType::BF16 => Tensor::from_bf16(
                shape,
                &values
                    .into_iter()
                    .map(half::bf16::from_f32)
                    .collect::<Vec<_>>(),
            )?,
            _ => return Err(Error::Other("unsupported vision pixel dtype".into())),
        };
        Ok(PreparedVision {
            tokens: output,
            pixels,
            grid: grids,
        })
    }
}

/// Python's `round` (ties to even), which `smart_resize` uses and Rust's
/// `f32::round` does not implement.
fn round_half_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction > 0.5 {
        floor + 1.0
    } else if fraction < 0.5 {
        floor
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

/// `Qwen2VLImageProcessor.smart_resize`: both sides divisible by
/// `patch * merge`, pixel count inside `[min_pixels, max_pixels]`, aspect
/// ratio preserved as closely as possible (and rejected above 200:1).
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 || factor == 0 {
        return Err(Error::Other(
            "vision image resize needs positive dimensions and factor".into(),
        ));
    }
    if height.max(width) as f64 / height.min(width) as f64 > 200.0 {
        return Err(Error::Other(format!(
            "vision image aspect ratio {} exceeds the processor limit 200",
            height.max(width) as f64 / height.min(width) as f64
        )));
    }
    let area = height
        .checked_mul(width)
        .ok_or_else(|| Error::Other("vision dimensions overflow".into()))?;
    if min_pixels == 0
        || min_pixels > max_pixels
        || factor.checked_mul(factor).is_none_or(|n| n > max_pixels)
    {
        return Err(Error::Other("invalid vision resize pixel budget".into()));
    }
    let mut resized_height = round_half_even(height as f64 / factor as f64) as usize * factor;
    let mut resized_width = round_half_even(width as f64 / factor as f64) as usize * factor;
    if resized_height * resized_width > max_pixels {
        let beta = (area as f64 / max_pixels as f64).sqrt();
        resized_height = ((height as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
        resized_width = ((width as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
    } else if resized_height * resized_width < min_pixels {
        let beta = (min_pixels as f64 / area as f64).sqrt();
        resized_height = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        resized_width = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    Ok((resized_height, resized_width))
}

/// Keys' cubic convolution with `a = -0.5`, the filter PyTorch uses for
/// bicubic with antialiasing enabled (PIL-compatible).
fn aa_filter(value: f64) -> f64 {
    const A: f64 = -0.5;
    let x = value.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// One axis's antialiased taps: `(input offset, int16 weights)` per output
/// index plus the fixed-point precision the accumulation shifts by.
pub(crate) fn axis_taps(input_size: usize, output_size: usize) -> Result<(Vec<(usize, Vec<i16>)>, u32)> {
    let scale = input_size as f64 / output_size as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let max_interp = (support.ceil() as usize) * 2 + 1;
    let mut rows = Vec::with_capacity(output_size);
    let mut weight_max = 0.0f64;
    for index in 0..output_size {
        let center = scale * (index as f64 + 0.5);
        let inverse_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        let minimum = ((center - support + 0.5) as i64).max(0) as usize;
        let size = ((center + support + 0.5) as i64)
            .min(input_size as i64)
            .saturating_sub(minimum as i64)
            .clamp(0, max_interp as i64) as usize;
        let mut weights = vec![0.0f64; size];
        let mut total = 0.0f64;
        for (tap, weight) in weights.iter_mut().enumerate() {
            *weight = aa_filter((tap as f64 + minimum as f64 - center + 0.5) * inverse_scale);
            total += *weight;
        }
        if total != 0.0 {
            for weight in &mut weights {
                *weight /= total;
                weight_max = weight_max.max(*weight);
            }
        }
        rows.push((minimum, weights));
    }
    // Same precision search as PyTorch: the largest weight must fit in int16.
    let mut precision = 22u32;
    for candidate in 0..22u32 {
        let next = (0.5 + weight_max * (1i64 << (candidate + 1)) as f64) as i64;
        if next >= (1 << 15) {
            precision = candidate;
            break;
        }
    }
    // The accumulation rounds with `1 << (precision - 1)`, so precision must
    // stay positive exactly as PyTorch's own assert requires.
    let precision = precision.max(1);
    let scale_by = (1i64 << precision) as f64;
    let quantized = rows
        .into_iter()
        .map(|(minimum, weights)| {
            let weights = weights
                .into_iter()
                .map(|weight| {
                    let value = weight * scale_by;
                    (if value < 0.0 {
                        (-0.5 + value) as i64
                    } else {
                        (0.5 + value) as i64
                    }) as i16
                })
                .collect();
            (minimum, weights)
        })
        .collect();
    Ok((quantized, precision))
}

/// PyTorch's native uint8 bicubic antialiased resize, separable with the
/// horizontal pass first and `u8` intermediates (each pass rounds through the
/// fixed-point accumulation).
pub fn resize_bicubic(
    chw: &[u8],
    height: usize,
    width: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Vec<u8>> {
    if chw.len() != 3 * height * width {
        return Err(Error::Other(format!(
            "vision image RGB buffer holds {} bytes for 3 x {height} x {width}",
            chw.len()
        )));
    }
    if (out_h, out_w) == (height, width) {
        return Ok(chw.to_vec());
    }
    let mut intermediate = chw.to_vec();
    let mut current_w = width;
    if out_w != width {
        let (taps, precision) = axis_taps(width, out_w)?;
        let mut output = vec![0u8; 3 * height * out_w];
        for channel in 0..3 {
            for row in 0..height {
                let source = (channel * height + row) * width;
                let target = (channel * height + row) * out_w;
                for (column, (minimum, weights)) in taps.iter().enumerate() {
                    let mut accumulator = 1i64 << (precision - 1);
                    for (tap, &weight) in weights.iter().enumerate() {
                        accumulator += intermediate[source + minimum + tap] as i64 * weight as i64;
                    }
                    output[target + column] = (accumulator >> precision).clamp(0, 255) as u8;
                }
            }
        }
        intermediate = output;
        current_w = out_w;
    }
    if out_h != height {
        let (taps, precision) = axis_taps(height, out_h)?;
        let mut output = vec![0u8; 3 * out_h * current_w];
        for channel in 0..3 {
            for row in 0..out_h {
                let (minimum, weights) = &taps[row];
                for column in 0..current_w {
                    let mut accumulator = 1i64 << (precision - 1);
                    for (tap, &weight) in weights.iter().enumerate() {
                        let source = (channel * height + minimum + tap) * current_w + column;
                        accumulator += intermediate[source] as i64 * weight as i64;
                    }
                    output[(channel * out_h + row) * current_w + column] =
                        (accumulator >> precision).clamp(0, 255) as u8;
                }
            }
        }
        intermediate = output;
    }
    Ok(intermediate)
}

/// Decoded RGB `u8` → the vision tower's input tensors.
///
/// `rgb` is **interleaved HWC** — the layout every image decoder produces —
/// and is converted to the channel-first form the patchify layout and the
/// separable resize need.
pub fn preprocess(
    contract: &VisionContract,
    rgb: &[u8],
    width: usize,
    height: usize,
) -> Result<PreparedImage> {
    if width == 0 || height == 0 {
        return Err(Error::Other("vision image must be nonempty".into()));
    }
    if width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(rgb.len()) {
        return Err(Error::Other(format!(
            "vision RGB buffer holds {} bytes for 3 x {width} x {height}",
            rgb.len()
        )));
    }
    let (resized_height, resized_width) = smart_resize(
        height,
        width,
        contract.patch_size * contract.spatial_merge_size,
        contract.min_pixels,
        contract.max_pixels,
    )?;
    let grid_h = resized_height / contract.patch_size;
    let grid_w = resized_width / contract.patch_size;
    if grid_h == 0 || grid_w == 0 {
        return Err(Error::Other(format!(
            "vision resize produced {resized_height} x {resized_width}, below one patch"
        )));
    }
    // HWC (decoder layout) → CHW for the separable resize and patchify.
    let mut source = vec![0u8; 3 * height * width];
    for row in 0..height {
        for column in 0..width {
            for channel in 0..3 {
                source[channel * height * width + row * width + column] =
                    rgb[(row * width + column) * 3 + channel];
            }
        }
    }
    let resized = resize_bicubic(&source, height, width, resized_height, resized_width)?;
    // The checkpoint fast processor fuses rescaling into normalization before
    // converting the mean/std tensors to F32. Preserve that rounding order.
    let means = contract
        .image_mean
        .map(|x| (x / contract.rescale_factor) as f32);
    let stds = contract
        .image_std
        .map(|x| (x / contract.rescale_factor) as f32);
    let patch_dim = 3 * contract.temporal_patch_size * contract.patch_size * contract.patch_size;
    let merge = contract.spatial_merge_size;
    let pixel_count = grid_h * grid_w;
    let mut pixel_values = vec![0.0f32; pixel_count * patch_dim];
    for block_row in 0..grid_h / merge {
        for block_column in 0..grid_w / merge {
            for inner_row in 0..merge {
                for inner_column in 0..merge {
                    let row = block_row * merge + inner_row;
                    let column = block_column * merge + inner_column;
                    let patch = ((block_row * (grid_w / merge) + block_column) * merge + inner_row)
                        * merge
                        + inner_column;
                    let target = &mut pixel_values[patch * patch_dim..(patch + 1) * patch_dim];
                    let mut feature = 0usize;
                    for channel in 0..3 {
                        for _temporal in 0..contract.temporal_patch_size {
                            for patch_row in 0..contract.patch_size {
                                let source_row = row * contract.patch_size + patch_row;
                                for patch_column in 0..contract.patch_size {
                                    let source_column = column * contract.patch_size + patch_column;
                                    let value = resized[(channel * resized_height + source_row)
                                        * resized_width
                                        + source_column]
                                        as f32;
                                    target[feature] = (value - means[channel]) / stds[channel];
                                    feature += 1;
                                }
                            }
                        }
                    }
                    debug_assert_eq!(feature, patch_dim);
                }
            }
        }
    }
    let image_tokens = pixel_count / (merge * merge);
    Ok(PreparedImage {
        pixel_values,
        grid_thw: [1, grid_h as u32, grid_w as u32],
        resized: (resized_height, resized_width),
        image_tokens,
    })
}

//! Pillow 12.3.0 compatible RGB8 bicubic resize with a materialized uint8
//! result after each horizontal and vertical pass.
//!
//! Geometry, coefficients and stable device buffers are prepared before graph
//! capture. Replaying the plan only reads the current raw input bytes.
use std::collections::{BTreeMap, HashMap};

use apxinf_core::{Error, Result};

use crate::{buffer::CudaBuffer, context::CudaContext, ffi};

/// Dimensions selected by the caller. `stage` is the first uint8 image; the
/// optional second PIL resize goes from `stage` to `final_size`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RgbResizeFrame {
    pub source_width: u32,
    pub source_height: u32,
    pub stage_width: u32,
    pub stage_height: u32,
    pub final_width: u32,
    pub final_height: u32,
}

impl RgbResizeFrame {
    fn source_bytes(self) -> Option<usize> {
        checked_rgb_bytes(self.source_width, self.source_height)
    }
    fn final_bytes(self) -> Option<usize> {
        checked_rgb_bytes(self.final_width, self.final_height)
    }
    fn key(self) -> [u32; 6] {
        [
            self.source_width,
            self.source_height,
            self.stage_width,
            self.stage_height,
            self.final_width,
            self.final_height,
        ]
    }
}

fn checked_rgb_bytes(w: u32, h: u32) -> Option<usize> {
    if w == 0
        || h == 0
        || w > i32::MAX as u32
        || h > i32::MAX as u32
        || u64::from(w) * u64::from(h) > i32::MAX as u64
    {
        return None;
    }
    usize::try_from(w)
        .ok()?
        .checked_mul(usize::try_from(h).ok()?)?
        .checked_mul(3)
}

fn bytes_i32(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}
fn bytes_i64(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}
fn upload_i32(ctx: &CudaContext, values: &[i32]) -> Result<CudaBuffer> {
    let host = bytes_i32(values);
    let device = CudaBuffer::alloc(host.len(), ctx.device_id()).map_err(Error::Cuda)?;
    device.copy_from_host(&host).map_err(Error::Cuda)?;
    Ok(device)
}
fn upload_i64(ctx: &CudaContext, values: &[i64]) -> Result<CudaBuffer> {
    let host = bytes_i64(values);
    let device = CudaBuffer::alloc(host.len(), ctx.device_id()).map_err(Error::Cuda)?;
    device.copy_from_host(&host).map_err(Error::Cuda)?;
    Ok(device)
}

fn cubic(mut x: f64) -> f64 {
    const A: f64 = -0.5;
    if x < 0.0 {
        x = -x;
    }
    if x < 1.0 {
        return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    }
    if x < 2.0 {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * A;
    }
    0.0
}

struct Coeff {
    bounds: CudaBuffer,
    weights: CudaBuffer,
    ksize: i32,
}

fn make_coeff(ctx: &CudaContext, in_size: u32, out_size: u32) -> Result<Coeff> {
    // Pillow passes the box endpoint through float before computing double scale.
    let scale = f64::from(in_size as f32) / f64::from(out_size);
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ksize = (support.ceil() as usize)
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or_else(|| Error::Other("Pillow coefficient support overflow".into()))?;
    let coeff_count = (out_size as usize)
        .checked_mul(ksize)
        .filter(|&n| n <= 64 * 1024 * 1024)
        .ok_or_else(|| Error::Other("Pillow coefficient table too large".into()))?;
    let mut bounds = vec![0i32; out_size as usize * 2];
    let mut weights = vec![0i32; coeff_count];
    let inv_filterscale = 1.0 / filterscale;
    for xx in 0..out_size as usize {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5) as i32).max(0);
        let xmax = ((center + support + 0.5) as i32).min(in_size as i32);
        let taps = (xmax - xmin).max(0) as usize;
        if taps > ksize {
            return Err(Error::Other("Pillow coefficient bounds overflow".into()));
        }
        bounds[xx * 2] = xmin;
        bounds[xx * 2 + 1] = taps as i32;
        let mut pre = vec![0f64; ksize];
        let mut sum = 0.0;
        for (x, value) in pre.iter_mut().enumerate().take(taps) {
            let w = cubic((x as f64 + f64::from(xmin) - center + 0.5) * inv_filterscale);
            *value = w;
            sum += w;
        }
        if sum != 0.0 {
            for value in pre.iter_mut().take(taps) {
                *value /= sum;
            }
        }
        for (x, &value) in pre.iter().enumerate() {
            weights[xx * ksize + x] =
                ((if value < 0.0 { -0.5 } else { 0.5 }) + value * f64::from(1 << 22)) as i32;
        }
    }
    Ok(Coeff {
        bounds: upload_i32(ctx, &bounds)?,
        weights: upload_i32(ctx, &weights)?,
        ksize: ksize as i32,
    })
}

struct Group {
    key: [u32; 6],
    batch: i32,
    source_offsets: CudaBuffer,
    h1_offsets: CudaBuffer,
    stage_offsets: CudaBuffer,
    h2_offsets: Option<CudaBuffer>,
    final_offsets: CudaBuffer,
    h1: CudaBuffer,
    stage: Option<CudaBuffer>,
    h2: Option<CudaBuffer>,
    coeff: [usize; 4],
}

/// Model-neutral prepared RGB8 resize. The caller owns the policy geometry,
/// but all physical buffers and coefficient tables belong to this plan.
pub struct PillowBicubicRgbPlan {
    raw: CudaBuffer,
    final_rgb: CudaBuffer,
    groups: Vec<Group>,
    coeffs: Vec<Coeff>,
    frames: Vec<RgbResizeFrame>,
    raw_bytes: usize,
    final_bytes: usize,
}

impl PillowBicubicRgbPlan {
    pub fn new(ctx: &CudaContext, frames: &[RgbResizeFrame]) -> Result<Self> {
        if frames.is_empty() || frames.len() > 65535 {
            return Err(Error::Other("invalid Pillow RGB frame count".into()));
        }
        let mut raw_offsets = Vec::with_capacity(frames.len());
        let mut final_offsets = Vec::with_capacity(frames.len());
        let (mut raw_bytes, mut final_bytes) = (0usize, 0usize);
        for frame in frames {
            checked_rgb_bytes(frame.stage_width, frame.stage_height)
                .ok_or_else(|| Error::Other("invalid Pillow first-stage geometry".into()))?;
            raw_offsets.push(
                i64::try_from(raw_bytes).map_err(|_| Error::Other("raw offset overflow".into()))?,
            );
            final_offsets.push(
                i64::try_from(final_bytes)
                    .map_err(|_| Error::Other("final offset overflow".into()))?,
            );
            raw_bytes = raw_bytes
                .checked_add(
                    frame
                        .source_bytes()
                        .ok_or_else(|| Error::Other("invalid Pillow source geometry".into()))?,
                )
                .ok_or_else(|| Error::Other("Pillow raw byte count overflow".into()))?;
            final_bytes = final_bytes
                .checked_add(
                    frame
                        .final_bytes()
                        .ok_or_else(|| Error::Other("invalid Pillow final geometry".into()))?,
                )
                .ok_or_else(|| Error::Other("Pillow final byte count overflow".into()))?;
        }
        i64::try_from(raw_bytes).map_err(|_| Error::Other("raw extent overflow".into()))?;
        i64::try_from(final_bytes).map_err(|_| Error::Other("final extent overflow".into()))?;
        let raw = CudaBuffer::alloc(raw_bytes, ctx.device_id()).map_err(Error::Cuda)?;
        let final_rgb = CudaBuffer::alloc(final_bytes, ctx.device_id()).map_err(Error::Cuda)?;
        let mut classes: BTreeMap<[u32; 6], Vec<usize>> = BTreeMap::new();
        for (index, frame) in frames.iter().enumerate() {
            classes.entry(frame.key()).or_default().push(index);
        }
        let mut coeffs = Vec::new();
        let mut coeff_map = HashMap::<(u32, u32), usize>::new();
        let mut coeff_index = |in_size: u32, out_size: u32| -> Result<usize> {
            if let Some(&index) = coeff_map.get(&(in_size, out_size)) {
                return Ok(index);
            }
            let index = coeffs.len();
            coeffs.push(make_coeff(ctx, in_size, out_size)?);
            coeff_map.insert((in_size, out_size), index);
            Ok(index)
        };
        let mut groups = Vec::with_capacity(classes.len());
        for (key, ids) in classes {
            let [sw, sh, tw, th, fw, fh] = key;
            let batch = i32::try_from(ids.len())
                .map_err(|_| Error::Other("Pillow batch overflow".into()))?;
            let h1_stride = checked_rgb_bytes(tw, sh)
                .ok_or_else(|| Error::Other("Pillow H1 extent overflow".into()))?;
            let stage_stride = checked_rgb_bytes(tw, th)
                .ok_or_else(|| Error::Other("Pillow stage extent overflow".into()))?;
            let second = (tw, th) != (fw, fh);
            let h2_stride = if second {
                checked_rgb_bytes(fw, th)
                    .ok_or_else(|| Error::Other("Pillow H2 extent overflow".into()))?
            } else {
                0
            };
            let alloc = |stride: usize| -> Result<CudaBuffer> {
                let size = stride
                    .checked_mul(ids.len())
                    .ok_or_else(|| Error::Other("Pillow grouped extent overflow".into()))?;
                CudaBuffer::alloc(size, ctx.device_id()).map_err(Error::Cuda)
            };
            let packed_offsets = |stride: usize| -> Result<Vec<i64>> {
                ids.iter()
                    .enumerate()
                    .map(|(i, _)| {
                        i.checked_mul(stride)
                            .and_then(|v| i64::try_from(v).ok())
                            .ok_or_else(|| Error::Other("Pillow grouped offset overflow".into()))
                    })
                    .collect()
            };
            let source = ids.iter().map(|&i| raw_offsets[i]).collect::<Vec<_>>();
            let final_out = ids.iter().map(|&i| final_offsets[i]).collect::<Vec<_>>();
            let coeff = [
                coeff_index(sw, tw)?,
                coeff_index(sh, th)?,
                coeff_index(tw, fw)?,
                coeff_index(th, fh)?,
            ];
            groups.push(Group {
                key,
                batch,
                source_offsets: upload_i64(ctx, &source)?,
                h1_offsets: upload_i64(ctx, &packed_offsets(h1_stride)?)?,
                stage_offsets: upload_i64(ctx, &packed_offsets(stage_stride)?)?,
                h2_offsets: second
                    .then(|| packed_offsets(h2_stride))
                    .transpose()?
                    .map(|v| upload_i64(ctx, &v))
                    .transpose()?,
                final_offsets: upload_i64(ctx, &final_out)?,
                h1: alloc(h1_stride)?,
                stage: if second {
                    Some(alloc(stage_stride)?)
                } else {
                    None
                },
                h2: if second {
                    Some(alloc(h2_stride)?)
                } else {
                    None
                },
                coeff,
            });
        }
        Ok(Self {
            raw,
            final_rgb,
            groups,
            coeffs,
            frames: frames.to_vec(),
            raw_bytes,
            final_bytes,
        })
    }

    pub fn raw(&self) -> &CudaBuffer {
        &self.raw
    }
    pub fn final_rgb(&self) -> &CudaBuffer {
        &self.final_rgb
    }
    pub fn raw_bytes(&self) -> usize {
        self.raw_bytes
    }
    pub fn final_bytes(&self) -> usize {
        self.final_bytes
    }
    pub fn frames(&self) -> &[RgbResizeFrame] {
        &self.frames
    }

    /// Only launches kernels. No allocation, host coefficient work or env read.
    pub fn run(&self, ctx: &CudaContext) -> Result<()> {
        if self.raw.device() != ctx.device_id() || self.final_rgb.device() != ctx.device_id() {
            return Err(Error::Other("Pillow resize plan device mismatch".into()));
        }
        for group in &self.groups {
            let [sw, sh, tw, th, fw, fh] = group.key;
            let axis = |input: &CudaBuffer,
                        output: &CudaBuffer,
                        input_offsets: &CudaBuffer,
                        output_offsets: &CudaBuffer,
                        coeff: &Coeff,
                        in_w: u32,
                        in_h: u32,
                        out_w: u32,
                        out_h: u32,
                        horizontal: bool|
             -> Result<()> {
                unsafe {
                    ffi::check_cuda(ffi::apxinf_pillow_bicubic_u8_axis(
                        input.ptr().cast(),
                        output.ptr(),
                        input_offsets.ptr().cast(),
                        output_offsets.ptr().cast(),
                        coeff.bounds.ptr().cast(),
                        coeff.weights.ptr().cast(),
                        coeff.ksize,
                        in_w as i32,
                        in_h as i32,
                        out_w as i32,
                        out_h as i32,
                        group.batch,
                        horizontal,
                        ctx.stream().handle(),
                    ))
                }
                .map_err(Error::Cuda)
            };
            axis(
                &self.raw,
                &group.h1,
                &group.source_offsets,
                &group.h1_offsets,
                &self.coeffs[group.coeff[0]],
                sw,
                sh,
                tw,
                sh,
                true,
            )?;
            let stage_out = group.stage.as_ref().unwrap_or(&self.final_rgb);
            let stage_offsets = if group.stage.is_some() {
                &group.stage_offsets
            } else {
                &group.final_offsets
            };
            axis(
                &group.h1,
                stage_out,
                &group.h1_offsets,
                stage_offsets,
                &self.coeffs[group.coeff[1]],
                tw,
                sh,
                tw,
                th,
                false,
            )?;
            if let (Some(stage), Some(h2), Some(h2_offsets)) =
                (&group.stage, &group.h2, &group.h2_offsets)
            {
                axis(
                    stage,
                    h2,
                    &group.stage_offsets,
                    h2_offsets,
                    &self.coeffs[group.coeff[2]],
                    tw,
                    th,
                    fw,
                    th,
                    true,
                )?;
                axis(
                    h2,
                    &self.final_rgb,
                    h2_offsets,
                    &group.final_offsets,
                    &self.coeffs[group.coeff[3]],
                    fw,
                    th,
                    fw,
                    fh,
                    false,
                )?;
            }
        }
        Ok(())
    }
}

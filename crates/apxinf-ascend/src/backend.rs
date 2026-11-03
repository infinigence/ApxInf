//! `Backend` trait implementation for Ascend NPU.
//!
//! Phase 2: aclnn operators are implemented for the BF16 inference path.
//! Each operator follows the two-phase aclnn API pattern:
//!   GetWorkspaceSize → alloc workspace → Execute → sync → cleanup

use std::any::Any;

use apxinf_core::{
    Backend, Device, DType, Error, Graph, KvCache, Result,
    Shape, Storage, Tensor,
};

use crate::acl_tensor::{run_op, AclIntArray, AclScalar, AclTensor, AclTensorList};
use crate::buffer::AscendBuffer;
use crate::context::AscendContext;
use crate::ffi::acl;
use crate::ffi::aclnn;
use crate::kv_cache::AscendKVCache;
use crate::transfers;

/// Report a not-yet-implemented operator by name.
fn pending<T>(op: &'static str) -> Result<T> {
    Err(Error::Other(format!(
        "apxinf-ascend: `{op}` is not implemented yet"
    )))
}

/// An Ascend NPU backend.
pub struct AscendBackend {
    device_id: usize,
    ctx: AscendContext,
}

impl AscendBackend {
    pub fn new(device_id: usize) -> Result<Self> {
        let ctx = AscendContext::new(device_id)
            .map_err(|e| Error::Other(format!("apxinf-ascend: init failed: {e}")))?;
        Ok(Self { device_id, ctx })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn context(&self) -> &AscendContext {
        &self.ctx
    }

    /// Allocate a device buffer and return it as a `Tensor` with NPU storage.
    fn alloc_device_tensor(&self, shape: Shape, dtype: apxinf_core::DType) -> Result<Tensor> {
        let num_bytes = shape.numel() * dtype.size_in_bytes();
        let buf = AscendBuffer::malloc(num_bytes, self.device_id)
            .map_err(|e| Error::Other(format!("alloc failed: {e}")))?;
        let handle = buf.to_storage_handle();
        Ok(unsafe {
            Tensor::from_raw_parts_unchecked(
                shape,
                dtype,
                Device::Npu(self.device_id),
                Storage::Gpu {
                    device: Device::Npu(self.device_id),
                    handle,
                },
            )
        })
    }

    /// Extract the device pointer from a tensor's GPU storage.
    fn device_ptr(tensor: &Tensor) -> Result<usize> {
        match tensor.storage() {
            Storage::Gpu { handle, .. } => Ok(handle.ptr()),
            _ => Err(Error::Other(format!(
                "expected NPU tensor, got {:?}",
                tensor.device()
            ))),
        }
    }
}

impl Backend for AscendBackend {
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let stream = self.ctx.raw_stream();
        let shape = input.shape().clone();
        let dtype = input.dtype();

        let out = self.alloc_device_tensor(shape.clone(), dtype)?;
        let rstd = self.alloc_device_tensor(shape.clone(), dtype)?;

        let input_ptr = Self::device_ptr(input)?;
        let weight_ptr = Self::device_ptr(weight)?;
        let out_ptr = Self::device_ptr(&out)?;
        let rstd_ptr = Self::device_ptr(&rstd)?;

        let acl_input = AclTensor::from_device(input_ptr, input.shape(), dtype);
        let acl_weight = AclTensor::from_device(weight_ptr, weight.shape(), dtype);
        let acl_out = AclTensor::from_device(out_ptr, &shape, dtype);
        let acl_rstd = AclTensor::from_device(rstd_ptr, &shape, dtype);

        let eps_val = eps as f64;
        run_op(
            |ws, exec| unsafe {
                aclnn::aclnnRmsNormGetWorkspaceSize(
                    acl_input.raw(),
                    acl_weight.raw(),
                    eps_val,
                    acl_out.raw(),
                    acl_rstd.raw(),
                    ws,
                    exec,
                )
            },
            |ws, ws_size, exec, s| unsafe { aclnn::aclnnRmsNorm(ws, ws_size, exec, s) },
            stream,
            "aclnnRmsNorm",
        )
        .map_err(|e| Error::Other(format!("rms_norm: {e}")))?;

        Ok(out)
    }

    fn silu(&self, input: &Tensor) -> Result<Tensor> {
        let stream = self.ctx.raw_stream();
        let shape = input.shape().clone();
        let dtype = input.dtype();

        let out = self.alloc_device_tensor(shape.clone(), dtype)?;
        let input_ptr = Self::device_ptr(input)?;
        let out_ptr = Self::device_ptr(&out)?;

        let acl_input = AclTensor::from_device(input_ptr, input.shape(), dtype);
        let acl_out = AclTensor::from_device(out_ptr, &shape, dtype);

        run_op(
            |ws, exec| unsafe {
                aclnn::aclnnSiluGetWorkspaceSize(acl_input.raw(), acl_out.raw(), ws, exec)
            },
            |ws, ws_size, exec, s| unsafe { aclnn::aclnnSilu(ws, ws_size, exec, s) },
            stream,
            "aclnnSilu",
        )
        .map_err(|e| Error::Other(format!("silu: {e}")))?;

        Ok(out)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let stream = self.ctx.raw_stream();
        let shape = a.shape().clone();
        let dtype = a.dtype();

        let out = self.alloc_device_tensor(shape.clone(), dtype)?;
        let a_ptr = Self::device_ptr(a)?;
        let b_ptr = Self::device_ptr(b)?;
        let out_ptr = Self::device_ptr(&out)?;

        let acl_a = AclTensor::from_device(a_ptr, a.shape(), dtype);
        let acl_b = AclTensor::from_device(b_ptr, b.shape(), dtype);
        let acl_out = AclTensor::from_device(out_ptr, &shape, dtype);
        let acl_alpha = AclScalar::f32(1.0);

        run_op(
            |ws, exec| unsafe {
                aclnn::aclnnAddGetWorkspaceSize(
                    acl_a.raw(),
                    acl_b.raw(),
                    acl_alpha.raw(),
                    acl_out.raw(),
                    ws,
                    exec,
                )
            },
            |ws, ws_size, exec, s| unsafe { aclnn::aclnnAdd(ws, ws_size, exec, s) },
            stream,
            "aclnnAdd",
        )
        .map_err(|e| Error::Other(format!("add: {e}")))?;

        Ok(out)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let stream = self.ctx.raw_stream();
        let shape = a.shape().clone();
        let dtype = a.dtype();

        let out = self.alloc_device_tensor(shape.clone(), dtype)?;
        let a_ptr = Self::device_ptr(a)?;
        let b_ptr = Self::device_ptr(b)?;
        let out_ptr = Self::device_ptr(&out)?;

        let acl_a = AclTensor::from_device(a_ptr, a.shape(), dtype);
        let acl_b = AclTensor::from_device(b_ptr, b.shape(), dtype);
        let acl_out = AclTensor::from_device(out_ptr, &shape, dtype);

        run_op(
            |ws, exec| unsafe {
                aclnn::aclnnMulGetWorkspaceSize(acl_a.raw(), acl_b.raw(), acl_out.raw(), ws, exec)
            },
            |ws, ws_size, exec, s| unsafe { aclnn::aclnnMul(ws, ws_size, exec, s) },
            stream,
            "aclnnMul",
        )
        .map_err(|e| Error::Other(format!("mul: {e}")))?;

        Ok(out)
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        let stream = self.ctx.raw_stream();
        let shape = input.shape().clone();
        let dtype = input.dtype();

        let out = self.alloc_device_tensor(shape.clone(), dtype)?;
        let input_ptr = Self::device_ptr(input)?;
        let out_ptr = Self::device_ptr(&out)?;

        let acl_input = AclTensor::from_device(input_ptr, input.shape(), dtype);
        let acl_out = AclTensor::from_device(out_ptr, &shape, dtype);
        let acl_factor = AclScalar::f32(factor);

        run_op(
            |ws, exec| unsafe {
                aclnn::aclnnMulsGetWorkspaceSize(
                    acl_input.raw(),
                    acl_factor.raw(),
                    acl_out.raw(),
                    ws,
                    exec,
                )
            },
            |ws, ws_size, exec, s| unsafe { aclnn::aclnnMuls(ws, ws_size, exec, s) },
            stream,
            "aclnnMuls",
        )
        .map_err(|e| Error::Other(format!("scale: {e}")))?;

        Ok(out)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let stream = self.ctx.raw_stream();
        let out_shape = a.shape().matmul_shape(b.shape())?;
        let dtype = a.dtype();

        let out = self.alloc_device_tensor(out_shape.clone(), dtype)?;
        let a_ptr = Self::device_ptr(a)?;
        let b_ptr = Self::device_ptr(b)?;
        let out_ptr = Self::device_ptr(&out)?;

        let acl_a = AclTensor::from_device(a_ptr, a.shape(), dtype);
        let acl_b = AclTensor::from_device(b_ptr, b.shape(), dtype);
        let acl_out = AclTensor::from_device(out_ptr, &out_shape, dtype);

        run_op(
            |ws, exec| unsafe {
                aclnn::aclnnMatmulGetWorkspaceSize(
                    acl_a.raw(),
                    acl_b.raw(),
                    acl_out.raw(),
                    0,
                    ws,
                    exec,
                )
            },
            |ws, ws_size, exec, s| unsafe { aclnn::aclnnMatmul(ws, ws_size, exec, s) },
            stream,
            "aclnnMatmul",
        )
        .map_err(|e| Error::Other(format!("matmul: {e}")))?;

        Ok(out)
    }

    fn rope(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_offset: u32,
    ) -> Result<Tensor> {
        let stream = self.ctx.raw_stream();
        let shape = input.shape().clone();
        let dtype = input.dtype();
        let seq_len = if shape.dims().len() == 2 { 1 } else { shape.dims()[0] };

        // Precompute cos/sin tables on CPU then H2D.
        // cos/sin shape: [seq_len, head_dim / 2]
        let half_dim = head_dim / 2;
        let mut cos_vec = Vec::with_capacity(seq_len * half_dim);
        let mut sin_vec = Vec::with_capacity(seq_len * half_dim);
        for s in 0..seq_len {
            let pos = (pos_offset as f64) + (s as f64);
            for i in 0..half_dim {
                let freq = (theta as f64).powf(-2.0 * (i as f64) / (head_dim as f64));
                let angle = pos * freq;
                cos_vec.push(angle.cos() as f32);
                sin_vec.push(angle.sin() as f32);
            }
        }

        let cos_bytes = bytemuck::cast_slice(&cos_vec).to_vec();
        let sin_bytes = bytemuck::cast_slice(&sin_vec).to_vec();

        let cos_shape = Shape::new(vec![seq_len, half_dim]);
        let sin_shape = Shape::new(vec![seq_len, half_dim]);

        let cos_tensor = self.to_device(&Tensor::from_raw(
            cos_shape.clone(),
            DType::F32,
            Device::Cpu,
            cos_bytes,
        )?)?;
        let sin_tensor = self.to_device(&Tensor::from_raw(
            sin_shape.clone(),
            DType::F32,
            Device::Cpu,
            sin_bytes,
        )?)?;

        // Copy input to output (RoPE is in-place on Q)
        let out = self.alloc_device_tensor(shape.clone(), dtype)?;
        let input_ptr = Self::device_ptr(input)?;
        let out_ptr = Self::device_ptr(&out)?;
        let cos_ptr = Self::device_ptr(&cos_tensor)?;
        let sin_ptr = Self::device_ptr(&sin_tensor)?;
        let num_bytes = input.size_in_bytes();

        // D2D copy input → output
        let ret = unsafe {
            acl::aclrtMemcpy(
                out_ptr as *mut std::ffi::c_void,
                num_bytes,
                input_ptr as *const std::ffi::c_void,
                num_bytes,
                acl::aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE,
            )
        };
        acl::check(ret, "rope D2D copy").map_err(|e| Error::Other(e))?;

        let acl_out = AclTensor::from_device(out_ptr, &shape, dtype);
        let acl_cos = AclTensor::from_device(cos_ptr, &cos_shape, DType::F32);
        let acl_sin = AclTensor::from_device(sin_ptr, &sin_shape, DType::F32);

        // aclnnApplyRotaryPosEmb: in-place on Q (queryRef), K can be null.
        // layout: 1 = BNSD (batch, heads, seq, dim) — but APXInf is [seq, heads, dim]
        // For NSD layout we use layout=2 (BSND) treating seq as the batch dim.
        run_op(
            |ws, exec| unsafe {
                aclnn::aclnnApplyRotaryPosEmbGetWorkspaceSize(
                    acl_out.raw(),
                    std::ptr::null_mut(),
                    acl_cos.raw(),
                    acl_sin.raw(),
                    2, // layout: BSND (B=seq, S=1, N=heads, D=dim)
                    ws,
                    exec,
                )
            },
            |ws, ws_size, exec, s| unsafe {
                aclnn::aclnnApplyRotaryPosEmb(ws, ws_size, exec, s)
            },
            stream,
            "aclnnApplyRotaryPosEmb",
        )
        .map_err(|e| Error::Other(format!("rope: {e}")))?;

        Ok(out)
    }

    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor> {
        let stream = self.ctx.raw_stream();
        let dtype = table.dtype();

        let ids_shape = Shape::new(vec![ids.len()]);
        let ids_i32: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        let ids_bytes: Vec<u8> = bytemuck::cast_slice(&ids_i32).to_vec();

        let ids_tensor = self.to_device(&Tensor::from_raw(
            ids_shape.clone(),
            apxinf_core::DType::I32,
            Device::Cpu,
            ids_bytes,
        )?)?;

        let embed_dim = table.shape().dim(-1)?;
        let out_shape = Shape::new(vec![ids.len(), embed_dim]);
        let out = self.alloc_device_tensor(out_shape.clone(), dtype)?;

        let table_ptr = Self::device_ptr(table)?;
        let ids_ptr = Self::device_ptr(&ids_tensor)?;
        let out_ptr = Self::device_ptr(&out)?;

        let acl_table = AclTensor::from_device(table_ptr, table.shape(), dtype);
        let acl_ids = AclTensor::from_device(ids_ptr, &ids_shape, apxinf_core::DType::I32);
        let acl_out = AclTensor::from_device(out_ptr, &out_shape, dtype);

        run_op(
            |ws, exec| unsafe {
                aclnn::aclnnEmbeddingGetWorkspaceSize(
                    acl_table.raw(),
                    acl_ids.raw(),
                    acl_out.raw(),
                    ws,
                    exec,
                )
            },
            |ws, ws_size, exec, s| unsafe { aclnn::aclnnEmbedding(ws, ws_size, exec, s) },
            stream,
            "aclnnEmbedding",
        )
        .map_err(|e| Error::Other(format!("embedding: {e}")))?;

        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn sdpa_decode(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        _max_seq_len: usize,
    ) -> Result<Tensor> {
        let stream = self.ctx.raw_stream();
        let dtype = q.dtype();

        let cache = kv
            .as_any()
            .downcast_ref::<AscendKVCache>()
            .ok_or_else(|| Error::Other("expected AscendKVCache".into()))?;

        // Q: [1, n_heads, head_dim] → IFA expects BNSD with B=1
        // Output: [1, n_heads, head_dim]
        let out_shape = Shape::new(vec![1, n_heads, head_dim]);
        let out = self.alloc_device_tensor(out_shape.clone(), dtype)?;

        let q_ptr = Self::device_ptr(q)?;
        let out_ptr = Self::device_ptr(&out)?;
        let k_ptr = cache.k_ptr(layer_idx);
        let v_ptr = cache.v_ptr(layer_idx);

        // KV cache layout: [n_kv_heads, max_seq_len, head_dim]
        // IFA wants key/value as TensorList, each element [kv_len, n_kv_heads, head_dim]
        // But our cache is [n_kv_heads, max_seq_len, head_dim] — we create an aclTensor
        // with the full cache shape and let IFA use actualSeqLengths to limit the range.
        let kv_cache_shape = Shape::new(vec![
            1,          // batch
            n_kv_heads,
            cache.max_seq_len(),
            head_dim,
        ]);

        let acl_q = AclTensor::from_device(q_ptr, q.shape(), dtype);
        let acl_k = AclTensor::from_device(k_ptr, &kv_cache_shape, dtype);
        let acl_v = AclTensor::from_device(v_ptr, &kv_cache_shape, dtype);
        let acl_out = AclTensor::from_device(out_ptr, &out_shape, dtype);

        // Wrap K/V in TensorList (single element each)
        let k_list = AclTensorList::new(&[acl_k.raw()]);
        let v_list = AclTensorList::new(&[acl_v.raw()]);

        // actualSeqLengths: [kv_len]
        let seq_lengths = AclIntArray::new(&[kv_len as i64]);

        let scale_value = 1.0 / (head_dim as f64).sqrt();
        let layout = std::ffi::CString::new("BNSD").unwrap();

        run_op(
            |ws, exec| unsafe {
                aclnn::aclnnIncreFlashAttentionV4GetWorkspaceSize(
                    acl_q.raw(),
                    k_list.raw(),
                    v_list.raw(),
                    std::ptr::null(),
                    std::ptr::null(),
                    seq_lengths.raw(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    n_heads as i64,
                    scale_value,
                    layout.as_ptr(),
                    n_kv_heads as i64,
                    0, // blockSize
                    0, // innerPrecise
                    acl_out.raw(),
                    ws,
                    exec,
                )
            },
            |ws, ws_size, exec, s| unsafe {
                aclnn::aclnnIncreFlashAttentionV4(ws, ws_size, exec, s)
            },
            stream,
            "aclnnIncreFlashAttentionV4",
        )
        .map_err(|e| Error::Other(format!("sdpa_decode: {e}")))?;

        // Reshape output to [1, n_heads * head_dim] for APXInf convention
        let flat_shape = Shape::new(vec![1, n_heads * head_dim]);
        Ok(unsafe {
            Tensor::from_raw_parts_unchecked(
                flat_shape,
                dtype,
                Device::Npu(self.device_id),
                out.storage().clone(),
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn sdpa_prefill(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        _max_seq_len: usize,
    ) -> Result<Tensor> {
        let stream = self.ctx.raw_stream();
        let dtype = q.dtype();
        let seq_len = q.shape().dims()[0];

        let cache = kv
            .as_any()
            .downcast_ref::<AscendKVCache>()
            .ok_or_else(|| Error::Other("expected AscendKVCache".into()))?;

        // Q: [seq_len, n_heads, head_dim]
        // Output: [seq_len, n_heads * head_dim]
        let out_shape = Shape::new(vec![seq_len, n_heads, head_dim]);
        let out = self.alloc_device_tensor(out_shape.clone(), dtype)?;

        let q_ptr = Self::device_ptr(q)?;
        let out_ptr = Self::device_ptr(&out)?;
        let k_ptr = cache.k_ptr(layer_idx);
        let v_ptr = cache.v_ptr(layer_idx);

        // KV cache layout: [n_kv_heads, max_seq_len, head_dim]
        // PFA takes key/value as single aclTensor: [1, n_kv_heads, kv_len, head_dim]
        let kv_shape = Shape::new(vec![
            1,
            n_kv_heads,
            cache.max_seq_len(),
            head_dim,
        ]);

        let acl_q = AclTensor::from_device(q_ptr, q.shape(), dtype);
        let acl_k = AclTensor::from_device(k_ptr, &kv_shape, dtype);
        let acl_v = AclTensor::from_device(v_ptr, &kv_shape, dtype);
        let acl_out = AclTensor::from_device(out_ptr, &out_shape, dtype);

        // actualSeqLengths: [seq_len], actualSeqLengthsKv: [kv_len]
        let actual_seq = AclIntArray::new(&[seq_len as i64]);
        let actual_kv = AclIntArray::new(&[kv_len as i64]);

        let scale_value = 1.0 / (head_dim as f64).sqrt();
        let layout = std::ffi::CString::new("BNSD").unwrap();

        // Causal mask: preTokens=seq_len, nextTokens=0, sparseMode=2 (causal)
        run_op(
            |ws, exec| unsafe {
                aclnn::aclnnPromptFlashAttentionV3GetWorkspaceSize(
                    acl_q.raw(),
                    acl_k.raw(),
                    acl_v.raw(),
                    std::ptr::null(), // pseShift
                    std::ptr::null(), // attenMask
                    actual_seq.raw(),
                    actual_kv.raw(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    n_heads as i64,
                    scale_value,
                    seq_len as i64, // preTokens
                    0,              // nextTokens
                    layout.as_ptr(),
                    n_kv_heads as i64,
                    2, // sparseMode: causal
                    0, // innerPrecise
                    acl_out.raw(),
                    ws,
                    exec,
                )
            },
            |ws, ws_size, exec, s| unsafe {
                aclnn::aclnnPromptFlashAttentionV3(ws, ws_size, exec, s)
            },
            stream,
            "aclnnPromptFlashAttentionV3",
        )
        .map_err(|e| Error::Other(format!("sdpa_prefill: {e}")))?;

        // Reshape output to [seq_len, n_heads * head_dim]
        let flat_shape = Shape::new(vec![seq_len, n_heads * head_dim]);
        Ok(unsafe {
            Tensor::from_raw_parts_unchecked(
                flat_shape,
                dtype,
                Device::Npu(self.device_id),
                out.storage().clone(),
            )
        })
    }

    fn create_kv_cache(
        &self,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Box<dyn KvCache> {
        AscendKVCache::new(self.device_id, n_layers, n_kv_heads, head_dim, max_seq_len)
            .map(|c| Box::new(c) as Box<dyn KvCache>)
            .unwrap_or_else(|e| {
                // If allocation fails, there's no safe fallback — log and
                // return a zero-capacity cache that will error on use.
                eprintln!("apxinf-ascend: KV cache alloc failed: {e}");
                Box::new(AscendKVCache::new(self.device_id, 0, 0, 0, 1).unwrap())
            })
    }

    fn kv_append(
        &self,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        append_len: usize,
    ) -> Result<()> {
        let cache = kv
            .as_any_mut()
            .downcast_mut::<AscendKVCache>()
            .ok_or_else(|| Error::Other("expected AscendKVCache".into()))?;
        let k_ptr = Self::device_ptr(k)?;
        let v_ptr = Self::device_ptr(v)?;
        cache.append_d2d(layer_idx, k_ptr, v_ptr, append_len)?;
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        self.ctx
            .synchronize_device()
            .map_err(|e| Error::Other(format!("synchronize: {e}")))
    }

    fn begin_capture(&self) -> Result<()> {
        let stream = self.ctx.raw_stream();
        let ret = unsafe { acl::aclmdlRICaptureBegin(stream, 0) };
        acl::check(ret, "aclmdlRICaptureBegin").map_err(|e| Error::Other(e))
    }

    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        let stream = self.ctx.raw_stream();
        let mut model_ri: acl::aclmdlRI = std::ptr::null_mut();
        let ret = unsafe { acl::aclmdlRICaptureEnd(stream, &mut model_ri) };
        acl::check(ret, "aclmdlRICaptureEnd").map_err(|e| Error::Other(e))?;
        Ok(Box::new(AscendGraph { model_ri }))
    }

    fn device(&self) -> Device {
        Device::Npu(self.device_id)
    }

    fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.device() == Device::Npu(self.device_id) {
            return Ok(tensor.clone());
        }
        let cpu_bytes = match tensor.storage() {
            Storage::Cpu(bytes) => bytes.as_slice(),
            _ => {
                return Err(Error::Other(format!(
                    "to_device: source must be CPU, got {:?}",
                    tensor.device()
                )))
            }
        };

        let shape = tensor.shape().clone();
        let dtype = tensor.dtype();
        let num_bytes = shape.numel() * dtype.size_in_bytes();

        let buf = AscendBuffer::malloc(num_bytes, self.device_id)
            .map_err(|e| Error::Other(format!("to_device alloc: {e}")))?;

        transfers::h2d(
            buf.raw_ptr(),
            cpu_bytes.as_ptr() as *const std::ffi::c_void,
            num_bytes,
        )
        .map_err(|e| Error::Other(format!("to_device copy: {e}")))?;

        let handle = buf.to_storage_handle();
        Ok(unsafe {
            Tensor::from_raw_parts_unchecked(
                shape,
                dtype,
                Device::Npu(self.device_id),
                Storage::Gpu {
                    device: Device::Npu(self.device_id),
                    handle,
                },
            )
        })
    }

    fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.device() == Device::Cpu {
            return Ok(tensor.clone());
        }
        let dev_ptr = Self::device_ptr(tensor)?;
        let num_bytes = tensor.size_in_bytes();
        let mut cpu_buf = vec![0u8; num_bytes];

        transfers::d2h(
            cpu_buf.as_mut_ptr() as *mut std::ffi::c_void,
            dev_ptr as *const std::ffi::c_void,
            num_bytes,
        )
        .map_err(|e| Error::Other(format!("to_cpu copy: {e}")))?;

        Ok(unsafe {
            Tensor::from_raw_parts_unchecked(
                tensor.shape().clone(),
                tensor.dtype(),
                Device::Cpu,
                Storage::cpu_from_bytes(cpu_buf),
            )
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A captured Ascend execution graph.
struct AscendGraph {
    model_ri: acl::aclmdlRI,
}

impl Graph for AscendGraph {
    fn replay(&self) -> Result<()> {
        let stream = std::ptr::null_mut(); // default stream
        let ret = unsafe { acl::aclmdlRIExecuteAsync(self.model_ri, stream) };
        acl::check(ret, "aclmdlRIExecuteAsync").map_err(|e| Error::Other(e))
    }
}

impl Drop for AscendGraph {
    fn drop(&mut self) {
        if !self.model_ri.is_null() {
            unsafe { let _ = acl::aclmdlRIDestroy(self.model_ri); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> AscendBackend {
        AscendBackend::new(0).expect("AscendBackend::new failed")
    }

    #[test]
    fn reports_its_own_npu_device_through_the_trait() {
        let backend: Box<dyn Backend> = Box::new(backend());
        assert_eq!(backend.device(), Device::Npu(0));
        assert!(backend.device().is_gpu());
    }

    #[test]
    fn kv_cache_create_and_clear() {
        let backend = backend();
        let mut cache = backend.create_kv_cache(2, 4, 64, 128);
        assert_eq!(cache.n_layers(), 2);
        assert_eq!(cache.seq_len(), 0);
        cache.advance(10);
        assert_eq!(cache.seq_len(), 10);
        cache.clear().unwrap();
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn to_device_and_to_cpu_roundtrip() {
        let backend = backend();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let cpu_tensor = Tensor::from_f32(vec![2, 2], &data).unwrap();
        let dev_tensor = backend.to_device(&cpu_tensor).unwrap();
        assert_eq!(dev_tensor.device(), Device::Npu(0));
        let back = backend.to_cpu(&dev_tensor).unwrap();
        assert_eq!(back.device(), Device::Cpu);
        let result = back.as_f32().unwrap();
        assert_eq!(result, &data);
    }

    #[test]
    fn silu_basic() {
        let backend = backend();
        let data = vec![-1.0f32, 0.0, 1.0, 2.0];
        let cpu_tensor = Tensor::from_f32(vec![4], &data).unwrap();
        let dev_tensor = backend.to_device(&cpu_tensor).unwrap();
        let result_dev = backend.silu(&dev_tensor).unwrap();
        let result = backend.to_cpu(&result_dev).unwrap();
        let vals = result.as_f32().unwrap();
        for (i, &x) in data.iter().enumerate() {
            let expected = x / (1.0 + (-x).exp());
            assert!(
                (vals[i] - expected).abs() < 1e-5,
                "silu({x}) = {expected}, got {}",
                vals[i]
            );
        }
    }

    #[test]
    fn add_basic() {
        let backend = backend();
        let a = Tensor::from_f32(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_f32(vec![2, 2], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        let a_dev = backend.to_device(&a).unwrap();
        let b_dev = backend.to_device(&b).unwrap();
        let result_dev = backend.add(&a_dev, &b_dev).unwrap();
        let result = backend.to_cpu(&result_dev).unwrap();
        let vals = result.as_f32().unwrap();
        assert_eq!(vals, &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn scale_basic() {
        let backend = backend();
        let a = Tensor::from_f32(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let a_dev = backend.to_device(&a).unwrap();
        let result_dev = backend.scale(&a_dev, 2.5).unwrap();
        let result = backend.to_cpu(&result_dev).unwrap();
        let vals = result.as_f32().unwrap();
        assert_eq!(vals, &[2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn mul_basic() {
        let backend = backend();
        let a = Tensor::from_f32(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_f32(vec![2, 2], &[2.0, 3.0, 4.0, 5.0]).unwrap();
        let a_dev = backend.to_device(&a).unwrap();
        let b_dev = backend.to_device(&b).unwrap();
        let result_dev = backend.mul(&a_dev, &b_dev).unwrap();
        let result = backend.to_cpu(&result_dev).unwrap();
        let vals = result.as_f32().unwrap();
        assert_eq!(vals, &[2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn matmul_basic() {
        let backend = backend();
        let a = Tensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::from_f32(vec![3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let a_dev = backend.to_device(&a).unwrap();
        let b_dev = backend.to_device(&b).unwrap();
        let result_dev = backend.matmul(&a_dev, &b_dev).unwrap();
        let result = backend.to_cpu(&result_dev).unwrap();
        let vals = result.as_f32().unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);
        assert!((vals[0] - 58.0).abs() < 1e-4, "got {}", vals[0]);
        assert!((vals[1] - 64.0).abs() < 1e-4, "got {}", vals[1]);
        assert!((vals[2] - 139.0).abs() < 1e-4, "got {}", vals[2]);
        assert!((vals[3] - 154.0).abs() < 1e-4, "got {}", vals[3]);
    }

    #[test]
    fn rms_norm_basic() {
        let backend = backend();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight_data = vec![1.0f32; 4];
        let input = Tensor::from_f32(vec![1, 4], &data).unwrap();
        let weight = Tensor::from_f32(vec![4], &weight_data).unwrap();
        let input_dev = backend.to_device(&input).unwrap();
        let weight_dev = backend.to_device(&weight).unwrap();
        let result_dev = backend.rms_norm(&input_dev, &weight_dev, 1e-5).unwrap();
        let result = backend.to_cpu(&result_dev).unwrap();
        let vals = result.as_f32().unwrap();
        let rms = (7.5f32 + 1e-5).sqrt();
        for (i, &x) in data.iter().enumerate() {
            let expected = x / rms;
            assert!(
                (vals[i] - expected).abs() < 1e-3,
                "rms_norm[{i}] = {expected}, got {}",
                vals[i]
            );
        }
    }
}

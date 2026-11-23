//! aclnn operator FFI declarations.
//!
//! Declares the aclnn types (aclTensor, aclScalar, aclOpExecutor) and the
//! two-phase operator functions needed by PI0.5. Each operator has:
//! - `aclnnXxxGetWorkspaceSize` —planning phase, returns workspace + executor
//! - `aclnnXxx` —execution phase, runs the executor on a stream

use std::ffi::c_void;
use std::os::raw::{c_char, c_int};

use super::acl::aclrtStream;

// ---- aclnn status ----

/// aclnn error code. 0 = OK.
pub type aclnnStatus = c_int;

// ---- aclnn init / finalize ----

extern "C" {
    pub fn aclnnInit(config_path: *const c_char) -> aclnnStatus;
    pub fn aclnnFinalize() -> aclnnStatus;
}

// ---- Opaque types (from acl_meta.h) ----

#[repr(C)]
pub struct aclOpExecutor {
    _private: [u8; 0],
}

#[repr(C)]
pub struct aclTensor {
    _private: [u8; 0],
}

#[repr(C)]
pub struct aclScalar {
    _private: [u8; 0],
}

#[repr(C)]
pub struct aclIntArray {
    _private: [u8; 0],
}

#[repr(C)]
pub struct aclTensorList {
    _private: [u8; 0],
}

// ---- ACL data types and formats (from acl_base_rt.h) ----

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum aclDataType {
    ACL_FLOAT = 0,
    ACL_FLOAT16 = 1,
    ACL_INT8 = 2,
    ACL_INT32 = 3,
    ACL_UINT8 = 4,
    ACL_INT64 = 9,
    ACL_BOOL = 12,
    ACL_BF16 = 27,
    ACL_FLOAT8_E4M3FN = 36,
    ACL_FLOAT4_E2M1 = 40,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum aclFormat {
    ACL_FORMAT_UNDEFINED = -1,
    ACL_FORMAT_NCHW = 0,
    ACL_FORMAT_NHWC = 1,
    ACL_FORMAT_ND = 2,
}

// ---- aclTensor / aclScalar lifecycle (from acl_meta.h) ----

extern "C" {
    pub fn aclCreateTensor(
        view_dims: *const i64,
        view_dims_num: u64,
        data_type: aclDataType,
        stride: *const i64,
        offset: i64,
        format: aclFormat,
        storage_dims: *const i64,
        storage_dims_num: u64,
        tensor_data: *mut c_void,
    ) -> *mut aclTensor;

    pub fn aclDestroyTensor(tensor: *const aclTensor) -> aclnnStatus;

    pub fn aclCreateScalar(value: *mut c_void, data_type: aclDataType) -> *mut aclScalar;

    pub fn aclDestroyScalar(scalar: *const aclScalar) -> aclnnStatus;

    pub fn aclCreateIntArray(value: *const i64, size: u64) -> *mut aclIntArray;

    pub fn aclDestroyIntArray(array: *const aclIntArray) -> aclnnStatus;

    pub fn aclDestroyAclOpExecutor(executor: *mut aclOpExecutor) -> aclnnStatus;

    pub fn aclCreateTensorList(value: *const *const aclTensor, size: u64) -> *mut aclTensorList;

    pub fn aclDestroyTensorList(array: *const aclTensorList) -> aclnnStatus;
}

// ---- Operator: aclnnSilu ----

extern "C" {
    pub fn aclnnSiluGetWorkspaceSize(
        self_: *const aclTensor,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnSilu(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnAdd ----

extern "C" {
    pub fn aclnnAddGetWorkspaceSize(
        self_: *const aclTensor,
        other: *const aclTensor,
        alpha: *const aclScalar,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnAdd(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;

    // In-place add: selfRef += alpha * other (no output tensor, modifies selfRef)
    pub fn aclnnInplaceAddGetWorkspaceSize(
        self_ref: *const aclTensor,
        other: *const aclTensor,
        alpha: *const aclScalar,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnInplaceAdd(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnMul ----

extern "C" {
    pub fn aclnnMulGetWorkspaceSize(
        self_: *const aclTensor,
        other: *const aclTensor,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnMul(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnMuls (scalar multiplication) ----

extern "C" {
    pub fn aclnnMulsGetWorkspaceSize(
        self_: *const aclTensor,
        other: *const aclScalar,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnMuls(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnRmsNorm ----

extern "C" {
    pub fn aclnnRmsNormGetWorkspaceSize(
        x: *const aclTensor,
        gamma: *const aclTensor,
        epsilon: f64,
        y_out: *const aclTensor,
        rstd_out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnRmsNorm(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnMatmul ----

extern "C" {
    pub fn aclnnMatmulGetWorkspaceSize(
        self_: *const aclTensor,
        mat2: *const aclTensor,
        out: *mut aclTensor,
        cube_math_type: i8,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnMatmul(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnEmbedding ----

extern "C" {
    pub fn aclnnEmbeddingGetWorkspaceSize(
        weight: *const aclTensor,
        indices: *const aclTensor,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnEmbedding(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnApplyRotaryPosEmb (RoPE) ----

extern "C" {
    pub fn aclnnApplyRotaryPosEmbGetWorkspaceSize(
        query_ref: *mut aclTensor,
        key_ref: *mut aclTensor,
        cos: *const aclTensor,
        sin: *const aclTensor,
        layout: i64,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnApplyRotaryPosEmb(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnRotaryPositionEmbedding (RoPE, simpler op) ----
// Note: requires cos/sin same shape as x (no broadcasting in use here).
// mode=0 (half): out[i] = x[i]*cos[i] - x[i+d/2]*sin[i]; out[i+d/2] = x[i+d/2]*cos[i+d/2] + x[i]*sin[i+d/2]
// cos/sin must be full head_dim (half_dim values repeated).

extern "C" {
    pub fn aclnnRotaryPositionEmbeddingGetWorkspaceSize(
        x: *const aclTensor,
        cos: *const aclTensor,
        sin: *const aclTensor,
        mode: i64,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnRotaryPositionEmbedding(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnIncreFlashAttentionV4 (decode attention) ----

extern "C" {
    pub fn aclnnIncreFlashAttentionV4GetWorkspaceSize(
        query: *const aclTensor,
        key: *const aclTensorList,
        value: *const aclTensorList,
        pse_shift: *const aclTensor,
        atten_mask: *const aclTensor,
        actual_seq_lengths: *const aclIntArray,
        dequant_scale1: *const aclTensor,
        quant_scale1: *const aclTensor,
        dequant_scale2: *const aclTensor,
        quant_scale2: *const aclTensor,
        quant_offset2: *const aclTensor,
        antiquant_scale: *const aclTensor,
        antiquant_offset: *const aclTensor,
        blocktable: *const aclTensor,
        kv_padding_size: *const aclTensor,
        num_heads: i64,
        scale_value: f64,
        input_layout: *const c_char,
        num_key_value_heads: i64,
        block_size: i64,
        inner_precise: i64,
        attention_out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnIncreFlashAttentionV4(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnPromptFlashAttentionV3 (prefill attention) ----

extern "C" {
    pub fn aclnnPromptFlashAttentionV3GetWorkspaceSize(
        query: *const aclTensor,
        key: *const aclTensor,
        value: *const aclTensor,
        pse_shift: *const aclTensor,
        atten_mask: *const aclTensor,
        actual_seq_lengths: *const aclIntArray,
        actual_seq_lengths_kv: *const aclIntArray,
        deq_scale1: *const aclTensor,
        quant_scale1: *const aclTensor,
        deq_scale2: *const aclTensor,
        quant_scale2: *const aclTensor,
        quant_offset2: *const aclTensor,
        num_heads: i64,
        scale_value: f64,
        pre_tokens: i64,
        next_tokens: i64,
        input_layout: *const c_char,
        num_key_value_heads: i64,
        sparse_mode: i64,
        inner_precise: i64,
        attention_out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnPromptFlashAttentionV3(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnFlashAttentionScore (training-domain flash attention) ----
// PFA may lack compiled kernels on some CANN builds; FAS is preferred.
// softmaxMaxOut/softmaxSumOut must be [batch, heads, seq, 8] FP32.

extern "C" {
    pub fn aclnnFlashAttentionScoreGetWorkspaceSize(
        query: *const aclTensor,
        key: *const aclTensor,
        value: *const aclTensor,
        real_shift_optional: *const aclTensor,
        drop_mask_optional: *const aclTensor,
        padding_mask_optional: *const aclTensor,
        atten_mask_optional: *const aclTensor,
        prefix_optional: *const aclIntArray,
        scale_value: f64,
        keep_prob: f64,
        pre_tokens: i64,
        next_tokens: i64,
        head_num: i64,
        input_layout: *const c_char,
        inner_precise: i64,
        sparse_mode: i64,
        softmax_max_out: *const aclTensor,
        softmax_sum_out: *const aclTensor,
        softmax_out_out: *const aclTensor,
        attention_out_out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnFlashAttentionScore(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnLayerNorm ----

extern "C" {
    pub fn aclnnLayerNormGetWorkspaceSize(
        input: *const aclTensor,
        norm_shape: *const aclIntArray,
        weight: *const aclTensor,
        bias: *const aclTensor,
        epsilon: f64,
        out: *const aclTensor,
        mean_out: *const aclTensor,
        rstd_out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnLayerNorm(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnGelu ----

extern "C" {
    pub fn aclnnGeluGetWorkspaceSize(
        self_: *const aclTensor,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnGelu(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnGeGlu ----
// GeGLU: split input along dim, out = A * Gelu(B)
// approximate: 0 = tanh, 1 = none

extern "C" {
    pub fn aclnnGeGluGetWorkspaceSize(
        self_: *const aclTensor,
        dim: i64,
        approximate: i64,
        out: *mut aclTensor,
        out_gelu: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnGeGlu(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnAddRmsNorm ----
// Fused: x = x1 + x2; y = (x / RMS(x)) * gamma; rstd = 1 / RMS(x)
// Note: rstd_out must be 2D [rows, 1], not 1D [rows]

extern "C" {
    pub fn aclnnAddRmsNormGetWorkspaceSize(
        x1: *const aclTensor,
        x2: *const aclTensor,
        gamma: *const aclTensor,
        epsilon: f64,
        y_out: *const aclTensor,
        rstd_out: *const aclTensor,
        x_out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnAddRmsNorm(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnSplitWithSize ----
// Split tensor into variable-size chunks along a dimension

extern "C" {
    pub fn aclnnSplitWithSizeGetWorkspaceSize(
        self_: *const aclTensor,
        split_sizes: *const aclIntArray,
        dim: i64,
        out_list: *const aclTensorList,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnSplitWithSize(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnPermute (Transpose) ----
// Permute dimensions. P0: VLM 180M + AE 720M, attention weight reshape.

extern "C" {
    pub fn aclnnPermuteGetWorkspaceSize(
        self_: *const aclTensor,
        dims: *const aclIntArray,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnPermute(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnCast ----
// Type conversion. P0: VLM 171M + AE 1110M, BF16->FP32 cast.

extern "C" {
    pub fn aclnnCastGetWorkspaceSize(
        self_: *const aclTensor,
        dtype: aclDataType,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnCast(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnCat (Concat) ----
// Concatenate tensors along a dimension. P0: VLM 3M + AE 364M, KV cache concat.

extern "C" {
    pub fn aclnnCatGetWorkspaceSize(
        tensor_list: *const aclTensorList,
        dim: i64,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnCat(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnExpand (BroadcastTo) ----
// Broadcast tensor to expanded shape. P0: VLM 35M + AE 361M, attention mask

extern "C" {
    pub fn aclnnExpandGetWorkspaceSize(
        self_: *const aclTensor,
        shape: *const aclIntArray,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnExpand(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnSub ----
// self - alpha * other. P1: VLM 4M + AE 1M

extern "C" {
    pub fn aclnnSubGetWorkspaceSize(
        self_: *const aclTensor,
        other: *const aclTensor,
        alpha: *const aclScalar,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnSub(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnDiv ----
// Element-wise division. P1: VLM 1M + AE 1M

extern "C" {
    pub fn aclnnDivGetWorkspaceSize(
        self_: *const aclTensor,
        other: *const aclTensor,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnDiv(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnClamp (ClipByValue) ----
// Clamp tensor to [min, max]. P1: VLM 2M attention mask clip.

extern "C" {
    pub fn aclnnClampGetWorkspaceSize(
        self_: *const aclTensor,
        min: *const aclScalar,
        max: *const aclScalar,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnClamp(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnConvolution ----
// N-d convolution. P1: VLM 1M ViT patch embedding (STUB).

extern "C" {
    pub fn aclnnConvolutionGetWorkspaceSize(
        input: *const aclTensor,
        weight: *const aclTensor,
        bias: *const aclTensor,
        stride: *const aclIntArray,
        padding: *const aclIntArray,
        dilation: *const aclIntArray,
        transposed: bool,
        output_padding: *const aclIntArray,
        groups: i64,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnConvolution(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnSoftmax ----
// Softmax along a dimension. P3: independent softmax ( ?FAS ?.

extern "C" {
    pub fn aclnnSoftmaxGetWorkspaceSize(
        self_: *const aclTensor,
        dim: i64,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnSoftmax(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnBatchMatMul ----
// Explicit batched matrix multiply. P3: autotune candidate for batched GEMM.

extern "C" {
    pub fn aclnnBatchMatMulGetWorkspaceSize(
        self_: *const aclTensor,
        mat2: *const aclTensor,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnBatchMatMul(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnFusedMatmul ----
// Fused GEMM (bias + fusedOp). P3: autotune candidate for GEMM+bias+activation.

extern "C" {
    pub fn aclnnFusedMatmulGetWorkspaceSize(
        x1: *const aclTensor,
        x2: *const aclTensor,
        bias: *const aclTensor,
        x3: *const aclTensor,
        fused_op_type: *const c_char,
        cube_math_type: i8,
        y: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnFusedMatmul(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnGemmaRmsNorm ----
// Gemma-specific RMSNorm. P3: PI0.5 uses Gemma2B + Gemma300M.

extern "C" {
    pub fn aclnnGemmaRmsNormGetWorkspaceSize(
        x: *const aclTensor,
        gamma: *const aclTensor,
        epsilon: f64,
        y_out: *const aclTensor,
        rstd_out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnGemmaRmsNorm(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnRange ----
// Generate range [start, end) with step. P1: VLM 1M + AE 1M sequence generation.

extern "C" {
    pub fn aclnnRangeGetWorkspaceSize(
        start: *const aclScalar,
        end: *const aclScalar,
        step: *const aclScalar,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnRange(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnCumsum ----
// Cumulative sum along a dimension. P1: VLM 1M + AE 1M

extern "C" {
    pub fn aclnnCumsumGetWorkspaceSize(
        self_: *const aclTensor,
        dim: i64,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnCumsum(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnStack ----
// Stack tensors along a new dimension. P2: VLM 2M

extern "C" {
    pub fn aclnnStackGetWorkspaceSize(
        tensor_list: *const aclTensorList,
        dim: i64,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnStack(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnInplaceCopy ----
// In-place copy. P2: KV cache updates.

extern "C" {
    pub fn aclnnInplaceCopyGetWorkspaceSize(
        self_: *mut aclTensor,
        src: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnInplaceCopy(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnSlice ----
// Extract a sub-tensor along a dimension. Used for QKV split (3 calls
// replace 3*seq_len per-token D2D copies).

extern "C" {
    pub fn aclnnSliceGetWorkspaceSize(
        self_: *const aclTensor,
        dim: i64,
        start: i64,
        end: i64,
        step: i64,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnSlice(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Operator: aclnnFillScalar ----
// Fill tensor with a scalar value. P2: AE 2M

extern "C" {
    pub fn aclnnFillScalarGetWorkspaceSize(
        self_: *const aclTensor,
        value: *const aclScalar,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnFillScalar(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ---- Helper: check aclnn status ----

pub fn check(ret: aclnnStatus, ctx: &str) -> Result<(), String> {
    if ret == 0 {
        Ok(())
    } else {
        Err(format!("aclnn error {ret} in {ctx}"))
    }
}

// ━ Operator: aclnnInplaceAddRmsNorm (fused add+rmsnorm) ━
extern "C" {
    pub fn aclnnInplaceAddRmsNormGetWorkspaceSize(
        x1_ref: *const aclTensor,
        x2_ref: *const aclTensor,
        gamma: *const aclTensor,
        epsilon: f64,
        rstd_out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnInplaceAddRmsNorm(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ━ Operator: aclnnAddLayerNorm (fused add+layernorm) ━
extern "C" {
    pub fn aclnnAddLayerNormGetWorkspaceSize(
        x1: *const aclTensor,
        x2: *const aclTensor,
        gamma: *const aclTensor,
        beta: *const aclTensor,
        bias: *const aclTensor,
        epsilon: f64,
        additional_output: bool,
        y_out: *const aclTensor,
        mean_out: *const aclTensor,
        rstd_out: *const aclTensor,
        x_out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnAddLayerNorm(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ── Operator: aclnnAdds (scalar add) ───────────────────────────────────
extern "C" {
    pub fn aclnnAddsGetWorkspaceSize(
        x: *const aclTensor,
        y: *const aclScalar,
        alpha: *const aclScalar,
        out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnAdds(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

// ── Operator: aclnnGeluV2 (gelu with approximate) ──────────────────────
extern "C" {
    pub fn aclnnGeluV2GetWorkspaceSize(
        x: *const aclTensor,
        approximate: i64,
        y: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;

    pub fn aclnnGeluV2(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> aclnnStatus;
}

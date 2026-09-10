use apxinf_core::{Backend, DType, Result, Shape, Tensor};
use half::{bf16, f16};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::kernels::activation::*;
use crate::kernels::attention::*;
use crate::kernels::elementwise::*;
use crate::kernels::embedding::*;
use crate::kernels::fused::*;
use crate::kernels::gemm::{
    gemm_fp8_dynamic_bf16, prepare_cublaslt_fp8_gemm, DynamicFp8WeightView,
};
use crate::kernels::norm::*;
use crate::kernels::preprocess::*;
use crate::kernels::quantization::*;
use crate::workspace::{prepare_with_workspace, with_workspace, GraphWorkspace};
use crate::{CudaArchFamily, CudaBackend};

#[test]
fn thor_database_resolves_custom_bias_residual_plan() {
    let backend = CudaBackend::new(0).unwrap();
    if backend.context().caps().sm != 110 {
        return;
    }
    let key = crate::kernels::fused::fp8_fused_tuning_key(
        backend.context(),
        522,
        2048,
        16384,
        crate::tuning::TuningDType::F16,
        crate::tuning::Epilogue::BiasResidual,
    );
    let tactic = crate::tuning::TacticId {
        backend: crate::tuning::TacticBackend::CublasLtCustomBias,
        value: 18_926_998,
    };
    let store = crate::tuning::TacticStore::from_gemm_records([crate::tuning::GemmTuningRecord {
        key: key.clone(),
        tactic,
        implementation_version: Some(tactic.backend.implementation_version()),
        milliseconds: Some(0.144243),
    }])
    .unwrap();
    backend
        .context()
        .install_tuning(crate::tuning::TuningSession::inference(store))
        .unwrap();
    let plan = backend
        .context()
        .gemm_plans()
        .resolve(
            backend.context(),
            &key,
            crate::tuning::TacticId {
                backend: crate::tuning::TacticBackend::Vendor,
                value: 0,
            },
        )
        .unwrap();
    assert_eq!(plan.source, crate::kernels::gemm::PlanSource::Exact);
    assert_eq!(
        plan.tactic.backend,
        crate::tuning::TacticBackend::CublasLtCustomBias
    );
    assert_eq!(plan.tactic, tactic);
}

fn gpu_ptr(tensor: &Tensor) -> Result<*mut std::ffi::c_void> {
    Ok(CudaBuffer::from_tensor(tensor)
        .map_err(apxinf_core::Error::Cuda)?
        .ptr())
}

#[test]
fn fused_gqa_qkv_mrope_cache_matches_composed_kernels() {
    const TOKENS: usize = 3;
    const CACHE_TOKENS: usize = 5;
    const Q_HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const WIDTH: usize = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM;
    let backend = CudaBackend::new(0).unwrap();
    let values = (0..TOKENS * WIDTH)
        .map(|index| bf16::from_f32((index as f32 - 71.0) / 53.0))
        .collect::<Vec<_>>();
    let bias_values = (0..WIDTH)
        .map(|index| bf16::from_f32((index as f32 - 19.0) / 97.0))
        .collect::<Vec<_>>();
    let qkv = backend
        .to_device(&Tensor::from_bf16(vec![TOKENS, WIDTH], &values).unwrap())
        .unwrap();
    let bias = backend
        .to_device(&Tensor::from_bf16(vec![WIDTH], &bias_values).unwrap())
        .unwrap();
    let positions = [2u32, 3, 5, 7, 11, 13, 17, 19, 23];
    let position_bytes = positions
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let position_ids = CudaBuffer::alloc(position_bytes.len(), backend.device_id()).unwrap();
    position_ids.copy_from_host(&position_bytes).unwrap();

    let q_width = Q_HEADS * HEAD_DIM;
    let kv_width = KV_HEADS * HEAD_DIM;
    let split_values = |offset: usize, width: usize| {
        (0..TOKENS * width)
            .map(|index| {
                let token = index / width;
                let col = index % width + offset;
                bf16::from_f32(
                    values[token * WIDTH + col].to_f32() + bias_values[col].to_f32(),
                )
            })
            .collect::<Vec<_>>()
    };
    let q = backend
        .to_device(
            &Tensor::from_bf16(
                vec![TOKENS, Q_HEADS, HEAD_DIM],
                &split_values(0, q_width),
            )
            .unwrap(),
        )
        .unwrap();
    let k = backend
        .to_device(
            &Tensor::from_bf16(
                vec![TOKENS, KV_HEADS, HEAD_DIM],
                &split_values(q_width, kv_width),
            )
            .unwrap(),
        )
        .unwrap();
    let reference_v = split_values(q_width + kv_width, kv_width)
        .into_iter()
        .map(bf16::to_f32)
        .collect::<Vec<_>>();
    let reference_q = crate::kernels::rope::apply_mrope(
        backend.context(),
        &q,
        Q_HEADS,
        HEAD_DIM,
        10_000.0,
        [2, 1, 1],
        &position_ids,
    )
    .unwrap();
    let reference_k = crate::kernels::rope::apply_mrope(
        backend.context(),
        &k,
        KV_HEADS,
        HEAD_DIM,
        10_000.0,
        [2, 1, 1],
        &position_ids,
    )
    .unwrap();
    let fused = split_gqa_qkv_mrope_cache_bf16(
        backend.context(),
        &qkv,
        Some(&bias),
        &position_ids,
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        10_000.0,
        [2, 1, 1],
        CACHE_TOKENS,
        None,
    )
    .unwrap();
    backend.synchronize().unwrap();

    let reference_q = backend.to_cpu(&reference_q).unwrap().to_f32_vec().unwrap();
    let reference_k = backend.to_cpu(&reference_k).unwrap().to_f32_vec().unwrap();
    let fused_q = backend.to_cpu(&fused.q).unwrap().to_f32_vec().unwrap();
    let fused_k = backend.to_cpu(&fused.k).unwrap().to_f32_vec().unwrap();
    let fused_v = backend.to_cpu(&fused.v).unwrap().to_f32_vec().unwrap();
    let kv_elements = TOKENS * KV_HEADS * HEAD_DIM;
    let q_max_error = fused_q
        .iter()
        .zip(&reference_q)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let k_max_error = fused_k[..kv_elements]
        .iter()
        .zip(&reference_k)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    eprintln!("fused QKV mRoPE max error: q={q_max_error}, k={k_max_error}");
    assert!(q_max_error <= 0.03 && k_max_error <= 0.03);
    assert_eq!(&fused_v[..kv_elements], reference_v.as_slice());
}

#[test]
fn fused_vision_qkv_rope_matches_composed_kernels() {
    const TOKENS: usize = 3;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const WIDTH: usize = 3 * HEADS * HEAD_DIM;
    let backend = CudaBackend::new(0).unwrap();
    let values = (0..TOKENS * WIDTH)
        .map(|index| bf16::from_f32((index as f32 - 43.0) / 61.0))
        .collect::<Vec<_>>();
    let bias = (0..WIDTH)
        .map(|index| bf16::from_f32((index as f32 - 17.0) / 101.0))
        .collect::<Vec<_>>();
    let qkv = backend
        .to_device(&Tensor::from_bf16(vec![TOKENS, WIDTH], &values).unwrap())
        .unwrap();
    let bias = backend
        .to_device(&Tensor::from_bf16(vec![WIDTH], &bias).unwrap())
        .unwrap();
    let positions = [2u32, 3, 5, 7, 11, 13];
    let position_bytes = positions
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let position_ids = CudaBuffer::alloc(position_bytes.len(), backend.device_id()).unwrap();
    position_ids.copy_from_host(&position_bytes).unwrap();

    let split = split_qkv_bias_bf16(backend.context(), &qkv, Some(&bias), HEADS, HEAD_DIM).unwrap();
    let reference_q = crate::kernels::rope::apply_vision_2d(
        backend.context(),
        &split.q,
        HEADS,
        HEAD_DIM,
        10_000.0,
        &position_ids,
    )
    .unwrap();
    let reference_k = crate::kernels::rope::apply_vision_2d(
        backend.context(),
        &split.k,
        HEADS,
        HEAD_DIM,
        10_000.0,
        &position_ids,
    )
    .unwrap();
    let fused = split_vision_qkv_rope_bf16(
        backend.context(),
        &qkv,
        Some(&bias),
        &position_ids,
        HEADS,
        HEAD_DIM,
        10_000.0,
    )
    .unwrap();
    backend.synchronize().unwrap();

    let reference_q = backend.to_cpu(&reference_q).unwrap().to_f32_vec().unwrap();
    let reference_k = backend.to_cpu(&reference_k).unwrap().to_f32_vec().unwrap();
    let reference_v = backend.to_cpu(&split.v).unwrap().to_f32_vec().unwrap();
    let fused_q = backend.to_cpu(&fused.q).unwrap().to_f32_vec().unwrap();
    let fused_k = backend.to_cpu(&fused.k).unwrap().to_f32_vec().unwrap();
    let fused_v = backend.to_cpu(&fused.v).unwrap().to_f32_vec().unwrap();
    assert_eq!(fused_q, reference_q);
    assert_eq!(fused_k, reference_k);
    assert_eq!(fused_v, reference_v);
}

#[test]
fn f16_qkv_fusions_match_explicit_bf16_cast() {
    const TOKENS: usize = 3;
    const Q_HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const GQA_WIDTH: usize = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM;
    const VISION_HEADS: usize = 2;
    const VISION_WIDTH: usize = 3 * VISION_HEADS * HEAD_DIM;
    let backend = CudaBackend::new(0).unwrap();
    let upload_f16 = |width: usize, offset: usize| {
        let values = (0..TOKENS * width)
            .map(|index| f16::from_f32(((index * 17 + offset) % 251) as f32 / 73.0 - 1.6))
            .collect::<Vec<_>>();
        backend
            .to_device(&Tensor::from_f16(vec![TOKENS, width], &values).unwrap())
            .unwrap()
    };
    let upload_bias = |width: usize| {
        let values = (0..width)
            .map(|index| bf16::from_f32((index as f32 - width as f32 / 2.0) / 97.0))
            .collect::<Vec<_>>();
        backend
            .to_device(&Tensor::from_bf16(vec![width], &values).unwrap())
            .unwrap()
    };

    let gqa = upload_f16(GQA_WIDTH, 11);
    let gqa_rounded = cast_f16_bf16(backend.context(), &gqa).unwrap();
    let gqa_bias = upload_bias(GQA_WIDTH);
    let gqa_positions = [2u32, 3, 5, 7, 11, 13, 17, 19, 23];
    let gqa_position_bytes = gqa_positions
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let gqa_position_ids =
        CudaBuffer::alloc(gqa_position_bytes.len(), backend.device_id()).unwrap();
    gqa_position_ids
        .copy_from_host(&gqa_position_bytes)
        .unwrap();
    let gqa_reference = split_gqa_qkv_mrope_cache_bf16(
        backend.context(),
        &gqa_rounded,
        Some(&gqa_bias),
        &gqa_position_ids,
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        10_000.0,
        [2, 1, 1],
        TOKENS,
        None,
    )
    .unwrap();
    let gqa_fused = split_gqa_qkv_mrope_cache_bf16(
        backend.context(),
        &gqa,
        Some(&gqa_bias),
        &gqa_position_ids,
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        10_000.0,
        [2, 1, 1],
        TOKENS,
        None,
    )
    .unwrap();

    let vision = upload_f16(VISION_WIDTH, 29);
    let vision_rounded = cast_f16_bf16(backend.context(), &vision).unwrap();
    let vision_bias = upload_bias(VISION_WIDTH);
    let vision_positions = [2u32, 3, 5, 7, 11, 13];
    let vision_position_bytes = vision_positions
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let vision_position_ids =
        CudaBuffer::alloc(vision_position_bytes.len(), backend.device_id()).unwrap();
    vision_position_ids
        .copy_from_host(&vision_position_bytes)
        .unwrap();
    let vision_reference = split_vision_qkv_rope_bf16(
        backend.context(),
        &vision_rounded,
        Some(&vision_bias),
        &vision_position_ids,
        VISION_HEADS,
        HEAD_DIM,
        10_000.0,
    )
    .unwrap();
    let vision_fused = split_vision_qkv_rope_bf16(
        backend.context(),
        &vision,
        Some(&vision_bias),
        &vision_position_ids,
        VISION_HEADS,
        HEAD_DIM,
        10_000.0,
    )
    .unwrap();
    backend.synchronize().unwrap();

    let exact = |reference: &Tensor, actual: &Tensor| {
        let reference = backend.to_cpu(reference).unwrap();
        let actual = backend.to_cpu(actual).unwrap();
        assert_eq!(actual.as_bf16().unwrap(), reference.as_bf16().unwrap());
    };
    exact(&gqa_reference.q, &gqa_fused.q);
    exact(&gqa_reference.k, &gqa_fused.k);
    exact(&gqa_reference.v, &gqa_fused.v);
    exact(&vision_reference.q, &vision_fused.q);
    exact(&vision_reference.k, &vision_fused.k);
    exact(&vision_reference.v, &vision_fused.v);
}

fn make_gpu_tensor(shape: Shape, dtype: DType, _device: usize, buffer: CudaBuffer) -> Tensor {
    buffer.into_tensor(shape, dtype)
}

fn fp8_gemm_f16(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<Tensor> {
    crate::kernels::gemm::fp8(
        ctx,
        activation,
        activation_scale,
        crate::kernels::gemm::Fp8WeightView {
            values_e4m3: weight,
            scale: weight_scale,
            dual_geglu_interleaved: false,
            dual_geglu_auto_interleaved: None,
        },
    )
}

fn fp8_gemm_bf16(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<Tensor> {
    crate::kernels::gemm::fp8_bf16(
        ctx,
        activation,
        activation_scale,
        crate::kernels::gemm::Fp8WeightView {
            values_e4m3: weight,
            scale: weight_scale,
            dual_geglu_interleaved: false,
            dual_geglu_auto_interleaved: None,
        },
    )
}

#[cfg(apxinf_fa2_f16_sm100)]
#[test]
fn fa2_language_mqa_f16_matches_cublas() {
    // FA2 requests >=48 KiB dynamic shared memory; serialize against
    // other such tests. See tests::gpu_smem_guard.
    let _gpu = crate::tests::gpu_smem_guard();
    const TOKENS: usize = 778;
    const HEADS: usize = 8;
    const HEAD_DIM: usize = 256;
    let backend = CudaBackend::new(0).unwrap();
    let q_values = (0..TOKENS * HEADS * HEAD_DIM)
        .map(|index| f16::from_f32(((index * 17 % 257) as f32 - 128.0) / 256.0))
        .collect::<Vec<_>>();
    let k_values = (0..TOKENS * HEAD_DIM)
        .map(|index| f16::from_f32(((index * 29 % 251) as f32 - 125.0) / 256.0))
        .collect::<Vec<_>>();
    let v_values = (0..TOKENS * HEAD_DIM)
        .map(|index| f16::from_f32(((index * 31 % 241) as f32 - 120.0) / 256.0))
        .collect::<Vec<_>>();
    let q = backend
        .to_device(&Tensor::from_f16(vec![TOKENS, HEADS, HEAD_DIM], &q_values).unwrap())
        .unwrap();
    let k = backend
        .to_device(&Tensor::from_f16(vec![TOKENS, 1, HEAD_DIM], &k_values).unwrap())
        .unwrap();
    let v = backend
        .to_device(&Tensor::from_f16(vec![TOKENS, 1, HEAD_DIM], &v_values).unwrap())
        .unwrap();

    let reference = cublas_mqa_f16(backend.context(), &q, &k, &v, TOKENS).unwrap();
    backend.synchronize().unwrap();
    let reference = backend.to_cpu(&reference).unwrap().to_f32_vec().unwrap();
    let workspace = GraphWorkspace::new(16 << 20, backend.device_id()).unwrap();
    let actual = prepare_with_workspace(&workspace, || {
        fa2_mqa_f16(backend.context(), &q, &k, &v, TOKENS)
    })
    .unwrap();
    backend.synchronize().unwrap();
    let actual = backend.to_cpu(&actual).unwrap().to_f32_vec().unwrap();

    let mut dot = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut error_norm = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&actual, &reference) in actual.iter().zip(&reference) {
        let actual = f64::from(actual);
        let reference = f64::from(reference);
        let error = actual - reference;
        dot += actual * reference;
        actual_norm += actual * actual;
        reference_norm += reference * reference;
        error_norm += error * error;
        max_abs = max_abs.max(error.abs());
    }
    let cosine = dot / (actual_norm * reference_norm).sqrt();
    let relative_l2 = (error_norm / reference_norm).sqrt();
    eprintln!(
        "FA2 language MQA: cosine={cosine:.9}, relative_l2={relative_l2:.9}, max_abs={max_abs:.6}"
    );
    assert!(
        cosine >= 0.999 && relative_l2 <= 0.02 && max_abs <= 0.05,
        "FA2 language MQA mismatch: cosine={cosine}, relative_l2={relative_l2}, max_abs={max_abs}"
    );
}

#[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
#[test]
fn action_flash_decode_matches_regular_causal_gqa() {
    let _gpu = crate::tests::gpu_smem_guard();
    const QUERY_TOKENS: usize = 10;
    const KEY_TOKENS: usize = 810;
    const QUERY_HEADS: usize = 16;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 128;

    let backend = CudaBackend::new(0).unwrap();
    let q_values = (0..QUERY_TOKENS * QUERY_HEADS * HEAD_DIM)
        .map(|index| bf16::from_f32(((index * 17 % 127) as f32 - 63.0) / 128.0))
        .collect::<Vec<_>>();
    let k_values = (0..KEY_TOKENS * KV_HEADS * HEAD_DIM)
        .map(|index| bf16::from_f32(((index * 29 % 131) as f32 - 65.0) / 128.0))
        .collect::<Vec<_>>();
    let v_values = (0..KEY_TOKENS * KV_HEADS * HEAD_DIM)
        .map(|index| bf16::from_f32(((index * 31 % 137) as f32 - 68.0) / 128.0))
        .collect::<Vec<_>>();
    let q = backend
        .to_device(
            &Tensor::from_bf16(vec![QUERY_TOKENS, QUERY_HEADS, HEAD_DIM], &q_values).unwrap(),
        )
        .unwrap();
    let k = backend
        .to_device(&Tensor::from_bf16(vec![KEY_TOKENS, KV_HEADS, HEAD_DIM], &k_values).unwrap())
        .unwrap();
    let v = backend
        .to_device(&Tensor::from_bf16(vec![KEY_TOKENS, KV_HEADS, HEAD_DIM], &v_values).unwrap())
        .unwrap();

    let workspace = GraphWorkspace::new(32 << 20, backend.device_id()).unwrap();
    let (reference, actual) = prepare_with_workspace(&workspace, || {
        let reference = crate::kernels::attention::fa2_attention_causal(
            backend.context(),
            &q,
            &k,
            &v,
            QUERY_TOKENS,
            KEY_TOKENS,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_DIM,
        )?;
        let actual = crate::kernels::attention::fa2_attention_splitkv(
            backend.context(),
            &q,
            &k,
            &v,
            1,
            QUERY_TOKENS,
            KEY_TOKENS,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_DIM,
            true,
        )?;
        Ok((reference, actual))
    })
    .unwrap();
    backend.synchronize().unwrap();
    let reference = backend.to_cpu(&reference).unwrap().to_f32_vec().unwrap();
    let actual = backend.to_cpu(&actual).unwrap().to_f32_vec().unwrap();

    let mut dot = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut error_norm = 0.0f64;
    for (&actual, &reference) in actual.iter().zip(&reference) {
        let actual = f64::from(actual);
        let reference = f64::from(reference);
        let error = actual - reference;
        dot += actual * reference;
        actual_norm += actual * actual;
        reference_norm += reference * reference;
        error_norm += error * error;
    }
    let cosine = dot / (actual_norm * reference_norm).sqrt();
    let relative_l2 = (error_norm / reference_norm).sqrt();
    eprintln!("Action flash decode: cosine={cosine:.9}, relative_l2={relative_l2:.9}");
    assert!(
        cosine >= 0.999 && relative_l2 <= 0.02,
        "Action flash decode mismatch: cosine={cosine}, relative_l2={relative_l2}"
    );
}

#[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
#[test]
fn fa2_causal_gqa_hdim128_matches_fp32_reference() {
    let _gpu = crate::tests::gpu_smem_guard();
    const TOKENS: usize = 37;
    const QUERY_HEADS: usize = 16;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 128;

    let backend = CudaBackend::new(0).unwrap();
    let q_values = (0..TOKENS * QUERY_HEADS * HEAD_DIM)
        .map(|index| bf16::from_f32(((index * 17 % 251) as f32 - 125.0) / 128.0))
        .collect::<Vec<_>>();
    let k_values = (0..TOKENS * KV_HEADS * HEAD_DIM)
        .map(|index| bf16::from_f32(((index * 29 % 241) as f32 - 120.0) / 128.0))
        .collect::<Vec<_>>();
    let v_values = (0..TOKENS * KV_HEADS * HEAD_DIM)
        .map(|index| bf16::from_f32(((index * 31 % 239) as f32 - 119.0) / 128.0))
        .collect::<Vec<_>>();
    let q = backend
        .to_device(&Tensor::from_bf16(vec![TOKENS, QUERY_HEADS, HEAD_DIM], &q_values).unwrap())
        .unwrap();
    let k = backend
        .to_device(&Tensor::from_bf16(vec![TOKENS, KV_HEADS, HEAD_DIM], &k_values).unwrap())
        .unwrap();
    let v = backend
        .to_device(&Tensor::from_bf16(vec![TOKENS, KV_HEADS, HEAD_DIM], &v_values).unwrap())
        .unwrap();

    let workspace = GraphWorkspace::new(16 << 20, backend.device_id()).unwrap();
    let actual = prepare_with_workspace(&workspace, || {
        crate::kernels::attention::fa2_attention_causal(
            backend.context(),
            &q,
            &k,
            &v,
            TOKENS,
            TOKENS,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_DIM,
        )
    })
    .unwrap();
    backend.synchronize().unwrap();
    let actual = backend.to_cpu(&actual).unwrap().to_f32_vec().unwrap();

    let q_values = q_values
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let k_values = k_values
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let v_values = v_values
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let mut reference = vec![0.0f32; actual.len()];
    let scale = (HEAD_DIM as f32).sqrt().recip();
    let heads_per_kv = QUERY_HEADS / KV_HEADS;
    let mut scores = vec![0.0f32; TOKENS];
    for query_token in 0..TOKENS {
        for query_head in 0..QUERY_HEADS {
            let kv_head = query_head / heads_per_kv;
            let q_base = (query_token * QUERY_HEADS + query_head) * HEAD_DIM;
            let visible_tokens = query_token + 1;
            let mut maximum = f32::NEG_INFINITY;
            for key_token in 0..visible_tokens {
                let k_base = (key_token * KV_HEADS + kv_head) * HEAD_DIM;
                let mut score = 0.0f32;
                for dim in 0..HEAD_DIM {
                    score += q_values[q_base + dim] * k_values[k_base + dim];
                }
                scores[key_token] = score * scale;
                maximum = maximum.max(scores[key_token]);
            }
            let mut denominator = 0.0f32;
            for score in &mut scores[..visible_tokens] {
                *score = (*score - maximum).exp();
                denominator += *score;
            }
            let out_base = (query_token * QUERY_HEADS + query_head) * HEAD_DIM;
            for key_token in 0..visible_tokens {
                let probability = scores[key_token] / denominator;
                let v_base = (key_token * KV_HEADS + kv_head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    reference[out_base + dim] += probability * v_values[v_base + dim];
                }
            }
        }
    }

    let mut dot = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut error_norm = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&actual, &reference) in actual.iter().zip(&reference) {
        let actual = f64::from(actual);
        let reference = f64::from(reference);
        let error = actual - reference;
        dot += actual * reference;
        actual_norm += actual * actual;
        reference_norm += reference * reference;
        error_norm += error * error;
        max_abs = max_abs.max(error.abs());
    }
    let cosine = dot / (actual_norm * reference_norm).sqrt();
    let relative_l2 = (error_norm / reference_norm).sqrt();
    eprintln!(
        "FA2 causal GQA hdim128: cosine={cosine:.9}, relative_l2={relative_l2:.9}, max_abs={max_abs:.6}"
    );
    assert!(
        cosine >= 0.999 && relative_l2 <= 0.02 && max_abs <= 0.05,
        "FA2 causal GQA hdim128 mismatch: cosine={cosine}, relative_l2={relative_l2}, max_abs={max_abs}"
    );
}

#[cfg(apxinf_fa2_direct_e4m3_sm100)]
#[test]
fn fa2_language_direct_e4m3_matches_packed4_bytes() {
    // FA2 requests >=48 KiB dynamic shared memory; serialize against
    // other such tests. See tests::gpu_smem_guard.
    let _gpu = crate::tests::gpu_smem_guard();
    const TOKENS: usize = 522;
    const HEADS: usize = 8;
    const HEAD_DIM: usize = 256;
    const SCALE: f32 = 0.05901227678571429;
    let backend = CudaBackend::new(0).unwrap();
    let q_values = (0..TOKENS * HEADS * HEAD_DIM)
        .map(|index| f16::from_f32(((index * 17 % 37) as f32 - 18.0) / 128.0))
        .collect::<Vec<_>>();
    let k_values = (0..TOKENS * HEAD_DIM)
        .map(|index| f16::from_f32(((index * 17 + 7) % 37) as f32 / 128.0 - 0.140625))
        .collect::<Vec<_>>();
    let v_values = (0..TOKENS * HEAD_DIM)
        .map(|index| f16::from_f32(((index * 17 + 11) % 37) as f32 / 64.0 - 0.28125))
        .collect::<Vec<_>>();
    let q = backend
        .to_device(&Tensor::from_f16(vec![TOKENS, HEADS, HEAD_DIM], &q_values).unwrap())
        .unwrap();
    let k = backend
        .to_device(&Tensor::from_f16(vec![TOKENS, 1, HEAD_DIM], &k_values).unwrap())
        .unwrap();
    let v = backend
        .to_device(&Tensor::from_f16(vec![TOKENS, 1, HEAD_DIM], &v_values).unwrap())
        .unwrap();
    let workspace = GraphWorkspace::new(32 << 20, backend.device_id()).unwrap();
    let (expected, actual) = prepare_with_workspace(&workspace, || {
        let fp16 = fa2_mqa_f16(backend.context(), &q, &k, &v, TOKENS)?;
        let expected = quantize_f16_e4m3(backend.context(), &fp16, SCALE)?;
        let actual = mqa_f16_e4m3_522(backend.context(), &q, &k, &v, SCALE)?;
        Ok((expected, actual))
    })
    .unwrap();
    backend.synchronize().unwrap();
    let expected = backend.to_cpu(&expected).unwrap();
    let actual = backend.to_cpu(&actual).unwrap();
    assert_eq!(actual.as_f8_e4m3().unwrap(), expected.as_f8_e4m3().unwrap());

    let wrong_q = q.reshape(vec![TOKENS, 4, HEAD_DIM * 2]).unwrap();
    assert!(mqa_f16_e4m3_522(backend.context(), &wrong_q, &k, &v, SCALE).is_err());
    assert!(mqa_f16_e4m3_522(backend.context(), &q, &k, &v, 0.0).is_err());
    assert!(mqa_f16_e4m3_522(backend.context(), &q, &k, &v, f32::NAN).is_err());
    assert!(mqa_f16_e4m3_522(backend.context(), &q, &k, &v, f32::INFINITY).is_err());
}

#[cfg(apxinf_cutlass_fmha)]
#[test]
fn packed_vision_qkv_matches_split_layout() {
    const TOKENS: usize = 768;
    const TOKENS_PER_VIEW: usize = 256;
    const HEADS: usize = 16;
    const HEAD_DIM: usize = 72;
    const WIDTH: usize = 3 * HEADS * HEAD_DIM;
    let backend = CudaBackend::new(0).unwrap();
    let qkv_values = (0..TOKENS * WIDTH)
        .map(|index| f16::from_f32(((index * 17 % 257) as f32 - 128.0) / 256.0))
        .collect::<Vec<_>>();
    let bias_values = (0..WIDTH)
        .map(|index| f16::from_f32(((index * 29 % 251) as f32 - 125.0) / 1024.0))
        .collect::<Vec<_>>();
    let qkv = backend
        .to_device(&Tensor::from_f16(vec![TOKENS, WIDTH], &qkv_values).unwrap())
        .unwrap();
    let bias = backend
        .to_device(&Tensor::from_f16(vec![WIDTH], &bias_values).unwrap())
        .unwrap();

    let split = split_qkv_bias_f16(backend.context(), &qkv, Some(&bias), HEADS, HEAD_DIM).unwrap();
    let reference = mha_f16(
        backend.context(),
        &split.q,
        &split.k,
        &split.v,
        TOKENS_PER_VIEW,
    )
    .unwrap();
    backend.synchronize().unwrap();
    let reference = backend.to_cpu(&reference).unwrap().to_f32_vec().unwrap();

    let workspace = GraphWorkspace::new(16 << 20, backend.device_id()).unwrap();
    let actual = prepare_with_workspace(&workspace, || {
        mha_packed_qkv_bias_f16(
            backend.context(),
            &qkv,
            Some(&bias),
            TOKENS_PER_VIEW,
            HEADS,
            HEAD_DIM,
        )
    })
    .unwrap();
    backend.synchronize().unwrap();
    let actual = backend.to_cpu(&actual).unwrap().to_f32_vec().unwrap();

    let mut dot = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut error_norm = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&actual, &reference) in actual.iter().zip(&reference) {
        let actual = f64::from(actual);
        let reference = f64::from(reference);
        let error = actual - reference;
        dot += actual * reference;
        actual_norm += actual * actual;
        reference_norm += reference * reference;
        error_norm += error * error;
        max_abs = max_abs.max(error.abs());
    }
    let cosine = dot / (actual_norm * reference_norm).sqrt();
    let relative_l2 = (error_norm / reference_norm).sqrt();
    eprintln!(
        "packed vision QKV: cosine={cosine:.9}, relative_l2={relative_l2:.9}, max_abs={max_abs:.6}"
    );
    assert!(
        cosine >= 0.999_999 && relative_l2 <= 0.001 && max_abs <= 0.005,
        "packed vision QKV mismatch: cosine={cosine}, relative_l2={relative_l2}, max_abs={max_abs}"
    );
}

#[test]
fn raw_rgb_nhwc_and_nchw_match_fp16_patch_pipeline() {
    const VIEWS: usize = 2;
    const IMAGE_SIZE: usize = 28;
    const PATCH_SIZE: usize = 14;
    const SCALE: f32 = 1.0 / 448.0;

    let backend = CudaBackend::new(0).unwrap();
    let image_bytes = VIEWS * IMAGE_SIZE * IMAGE_SIZE * 3;
    let nhwc = (0..image_bytes)
        .map(|index| ((index * 73 + index / 11 + 19) & 0xff) as u8)
        .collect::<Vec<_>>();
    let mut nchw = vec![0u8; image_bytes];
    for view in 0..VIEWS {
        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                for channel in 0..3 {
                    let source = ((view * IMAGE_SIZE + y) * IMAGE_SIZE + x) * 3 + channel;
                    let destination = ((view * 3 + channel) * IMAGE_SIZE + y) * IMAGE_SIZE + x;
                    nchw[destination] = nhwc[source];
                }
            }
        }
    }

    let patches_per_side = IMAGE_SIZE / PATCH_SIZE;
    let patch_rows = VIEWS * patches_per_side * patches_per_side;
    let patch_width = 3 * PATCH_SIZE * PATCH_SIZE;
    let mut reference = vec![f16::ZERO; patch_rows * patch_width];
    for view in 0..VIEWS {
        for patch_y in 0..patches_per_side {
            for patch_x in 0..patches_per_side {
                let row = view * patches_per_side * patches_per_side
                    + patch_y * patches_per_side
                    + patch_x;
                for channel in 0..3 {
                    for dy in 0..PATCH_SIZE {
                        for dx in 0..PATCH_SIZE {
                            let y = patch_y * PATCH_SIZE + dy;
                            let x = patch_x * PATCH_SIZE + dx;
                            let source = ((view * IMAGE_SIZE + y) * IMAGE_SIZE + x) * 3 + channel;
                            let column = channel * PATCH_SIZE * PATCH_SIZE + dy * PATCH_SIZE + dx;
                            let normalized = (f32::from(nhwc[source]) / 255.0) * 2.0 - 1.0;
                            reference[row * patch_width + column] = f16::from_f32(normalized);
                        }
                    }
                }
            }
        }
    }
    let reference = backend
        .to_device(&Tensor::from_f16(vec![patch_rows, patch_width], &reference).unwrap())
        .unwrap();
    let expected = quantize_f16_e4m3(backend.context(), &reference, SCALE).unwrap();

    let run_layout = |images: &[u8], layout| {
        let input = CudaBuffer::alloc(image_bytes, backend.device_id()).unwrap();
        input.copy_from_host(images).unwrap();
        let output = backend
            .to_device(&Tensor::zeros(vec![patch_rows, patch_width], DType::F8E4M3))
            .unwrap();
        rgb_u8_to_patches_e4m3(
            backend.context(),
            &input,
            &output,
            VIEWS,
            IMAGE_SIZE,
            PATCH_SIZE,
            layout,
            SCALE,
        )
        .unwrap();
        backend.to_cpu(&output).unwrap()
    };
    let actual_nhwc = run_layout(&nhwc, ImageLayout::Nhwc);
    let actual_nchw = run_layout(&nchw, ImageLayout::Nchw);
    let expected = backend.to_cpu(&expected).unwrap();
    assert_eq!(
        actual_nhwc.as_f8_e4m3().unwrap(),
        expected.as_f8_e4m3().unwrap()
    );
    assert_eq!(
        actual_nchw.as_f8_e4m3().unwrap(),
        expected.as_f8_e4m3().unwrap()
    );
}

#[test]
fn fp8_identity_gemm_runs_on_device() {
    const SIZE: usize = 16;
    let backend = CudaBackend::new(0).unwrap();
    let activation = (0..SIZE * SIZE)
        .map(|i| (i as f32 % 13.0 - 6.0) / 8.0)
        .collect::<Vec<_>>();
    let mut identity = vec![0.0f32; SIZE * SIZE];
    for i in 0..SIZE {
        identity[i * SIZE + i] = 1.0;
    }
    let activation_scale = activation.iter().fold(0.0f32, |m, x| m.max(x.abs())) / 448.0;
    let weight_scale = 1.0 / 448.0;
    let activation_f16 = activation
        .iter()
        .map(|x| f16::from_f32(*x))
        .collect::<Vec<_>>();
    let identity_f16 = identity
        .iter()
        .map(|x| f16::from_f32(*x))
        .collect::<Vec<_>>();
    let activation_gpu = backend
        .to_device(&Tensor::from_f16(vec![SIZE, SIZE], &activation_f16).unwrap())
        .unwrap();
    let weight_gpu = backend
        .to_device(&Tensor::from_f16(vec![SIZE, SIZE], &identity_f16).unwrap())
        .unwrap();
    let activation_fp8 =
        quantize_f16_e4m3(backend.context(), &activation_gpu, activation_scale).unwrap();
    let weight_fp8 = quantize_f16_e4m3(backend.context(), &weight_gpu, weight_scale).unwrap();
    let output = fp8_gemm_f16(
        backend.context(),
        &activation_fp8,
        &weight_fp8,
        activation_scale,
        weight_scale,
    )
    .unwrap();
    let output = backend.to_cpu(&output).unwrap().to_f32_vec().unwrap();
    for (actual, expected) in output.iter().zip(&activation) {
        assert!((actual - expected).abs() < 0.04, "{actual} != {expected}");
    }
}

#[test]
fn dynamic_fp8_row_channel_scales_match_bf16_reference() {
    // Use an actual Action projection shape because cuBLASLt only publishes
    // the fast NNT FP8 heuristics for production-sized matrices on Thor.
    const M: usize = 10;
    const N: usize = 1024;
    const N_PADDED: usize = 1024;
    const K: usize = 2048;
    const K_PADDED: usize = 2048;
    let backend = CudaBackend::new(0).unwrap();
    if backend.context().caps().arch_family != CudaArchFamily::Sm100 {
        return;
    }
    let activation = (0..M * K)
        .map(|index| {
            let row = index / K;
            let col = index % K;
            ((col * 17 % 31) as f32 - 15.0) / 19.0 * (row as f32 + 1.0) / 5.0
        })
        .collect::<Vec<_>>();
    let weight = (0..K * N)
        .map(|index| {
            let row = index / N;
            let col = index % N;
            ((row * 13 % 29) as f32 - 14.0) / 23.0 * (col as f32 + 2.0) / 7.0
        })
        .collect::<Vec<_>>();
    let bias = (0..N)
        .map(|col| (col as f32 - 7.0) / 11.0)
        .collect::<Vec<_>>();
    let activation_bf16 = activation
        .iter()
        .copied()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    let mut weight_bf16 = vec![bf16::ZERO; N_PADDED * K_PADDED];
    for row in 0..K {
        for col in 0..N {
            weight_bf16[col * K_PADDED + row] = bf16::from_f32(weight[row * N + col]);
        }
    }
    let mut bias_values_bf16 = bias.iter().copied().map(bf16::from_f32).collect::<Vec<_>>();
    bias_values_bf16.resize(N_PADDED, bf16::ZERO);
    let activation_gpu = backend
        .to_device(&Tensor::from_bf16(vec![M, K], &activation_bf16).unwrap())
        .unwrap();
    let weight_gpu = backend
        .to_device(&Tensor::from_bf16(vec![N_PADDED, K_PADDED], &weight_bf16).unwrap())
        .unwrap();
    let bias_gpu = backend
        .to_device(&Tensor::from_bf16(vec![N_PADDED], &bias_values_bf16).unwrap())
        .unwrap();
    let activation_fp8 =
        quantize_rows_bf16_e4m3_padded(backend.context(), &activation_gpu, K_PADDED).unwrap();
    let weight_fp8 = quantize_rows_bf16_e4m3(backend.context(), &weight_gpu).unwrap();
    let output_padded = gemm_fp8_dynamic_bf16(
        backend.context(),
        &activation_fp8.values,
        &activation_fp8.scales,
        DynamicFp8WeightView {
            values_e4m3: &weight_fp8.values,
            channel_scales: &weight_fp8.scales,
        },
        Some(&bias_gpu),
    )
    .unwrap();
    let output = slice_columns_bf16(backend.context(), &output_padded, N).unwrap();
    backend.synchronize().unwrap();

    let row_scales = backend
        .to_cpu(&activation_fp8.scales)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let column_scales = backend
        .to_cpu(&weight_fp8.scales)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    assert!(row_scales.windows(2).all(|pair| pair[1] > pair[0]));
    assert!(column_scales[..N].windows(2).all(|pair| pair[1] >= pair[0]));
    assert!(column_scales[N - 1] > column_scales[0]);
    assert!(column_scales[N..].iter().all(|scale| *scale == 1.0e-12));

    let output = backend.to_cpu(&output).unwrap().to_f32_vec().unwrap();
    let mut maximum_error = 0.0f32;
    let mut maximum_reference = 0.0f32;
    for row in 0..M {
        for col in 0..N {
            let mut expected = bias[col];
            for inner in 0..K {
                expected += activation[row * K + inner] * weight[inner * N + col];
            }
            maximum_error = maximum_error.max((output[row * N + col] - expected).abs());
            maximum_reference = maximum_reference.max(expected.abs());
        }
    }
    assert!(
        maximum_error / maximum_reference < 0.04,
        "dynamic FP8 relative max error is {}",
        maximum_error / maximum_reference
    );
}

#[test]
fn dynamic_fp8_vla_fusions_match_composed_operators() {
    const ROWS: usize = 10;
    const COLS: usize = 64;
    const PADDED: usize = 72;
    let backend = CudaBackend::new(0).unwrap();
    let values = (0..ROWS * COLS)
        .map(|index| bf16::from_f32(((index * 17 % 101) as f32 - 50.0) / 29.0))
        .collect::<Vec<_>>();
    let residual_values = (0..ROWS * COLS)
        .map(|index| bf16::from_f32(((index * 11 % 67) as f32 - 33.0) / 41.0))
        .collect::<Vec<_>>();
    let norm_values = (0..COLS)
        .map(|index| bf16::from_f32(0.75 + index as f32 / 244.0))
        .collect::<Vec<_>>();
    let bias_values = (0..COLS)
        .map(|index| bf16::from_f32((index as f32 - 30.0) / 211.0))
        .collect::<Vec<_>>();
    let input = backend
        .to_device(&Tensor::from_bf16(vec![ROWS, COLS], &values).unwrap())
        .unwrap();
    let residual = backend
        .to_device(&Tensor::from_bf16(vec![ROWS, COLS], &residual_values).unwrap())
        .unwrap();
    let norm_weight = backend
        .to_device(&Tensor::from_bf16(vec![COLS], &norm_values).unwrap())
        .unwrap();
    let bias = backend
        .to_device(&Tensor::from_bf16(vec![COLS], &bias_values).unwrap())
        .unwrap();

    let decode = |quantized: &DynamicFp8Tensor| {
        let shape = quantized.values.shape().dims();
        let width = shape[1];
        let values = backend
            .to_cpu(&quantized.values)
            .unwrap()
            .as_f8_e4m3()
            .unwrap()
            .iter()
            .map(|byte| {
                let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
                let exponent = (byte >> 3) & 0x0f;
                let mantissa = byte & 0x07;
                if exponent == 0 {
                    sign * mantissa as f32 * 2f32.powi(-9)
                } else if exponent == 0x0f && mantissa == 0x07 {
                    f32::NAN
                } else {
                    sign * (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
                }
            })
            .collect::<Vec<_>>();
        let scales = backend
            .to_cpu(&quantized.scales)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        values
            .into_iter()
            .enumerate()
            .map(|(index, value)| value * scales[index / width])
            .collect::<Vec<_>>()
    };
    let compare = |name: &str, actual: &[f32], expected: &[f32]| {
        let error = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm = expected
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(error / norm < 0.02, "{name} relative L2={}", error / norm);
    };

    let norm_reference = rms_bf16(backend.context(), &input, &norm_weight, 1.0e-6).unwrap();
    let norm_reference =
        quantize_rows_bf16_e4m3_padded(backend.context(), &norm_reference, PADDED).unwrap();
    let norm_fused =
        rms_quantize_rows_bf16_e4m3(backend.context(), &input, &norm_weight, 1.0e-6, PADDED)
            .unwrap();
    compare(
        "RMSNorm quantization",
        &decode(&norm_fused),
        &decode(&norm_reference),
    );

    let residual_reference = bias_residual_rms_bf16(
        backend.context(),
        &input,
        Some(&bias),
        &residual,
        &norm_weight,
        1.0e-6,
    )
    .unwrap();
    let residual_reference =
        quantize_rows_bf16_e4m3_padded(backend.context(), &residual_reference.normalized, PADDED)
            .unwrap();
    let residual_fused = bias_residual_rms_quantize_rows_bf16_e4m3(
        backend.context(),
        &input,
        Some(&bias),
        &residual,
        &norm_weight,
        1.0e-6,
        PADDED,
    )
    .unwrap();
    compare(
        "residual RMSNorm quantization",
        &decode(&residual_fused.normalized),
        &decode(&residual_reference),
    );

    let mut gate_up_values = (0..ROWS * (2 * COLS))
        .map(|index| bf16::from_f32(((index * 19 % 83) as f32 - 41.0) / 37.0))
        .collect::<Vec<_>>();
    let mut physical_gate_up = vec![bf16::from_f32(100.0); ROWS * 2 * PADDED];
    for row in 0..ROWS {
        physical_gate_up[row * 2 * PADDED..row * 2 * PADDED + 2 * COLS]
            .copy_from_slice(&gate_up_values[row * 2 * COLS..(row + 1) * 2 * COLS]);
    }
    gate_up_values.clear();
    let gate_up = backend
        .to_device(&Tensor::from_bf16(vec![ROWS, 2 * PADDED], &physical_gate_up).unwrap())
        .unwrap();
    let gate_up_logical = slice_columns_bf16(backend.context(), &gate_up, 2 * COLS).unwrap();
    let swiglu_reference = swiglu_bf16(backend.context(), &gate_up_logical).unwrap();
    let swiglu_reference =
        quantize_rows_bf16_e4m3_padded(backend.context(), &swiglu_reference, PADDED).unwrap();
    let swiglu_fused =
        swiglu_quantize_rows_bf16_e4m3(backend.context(), &gate_up, None, COLS, PADDED).unwrap();
    backend.synchronize().unwrap();
    compare(
        "SwiGLU quantization",
        &decode(&swiglu_fused),
        &decode(&swiglu_reference),
    );
}

#[test]
fn dynamic_fp8_vec4_and_large_vec8_fusions_match_composed_operators() {
    let backend = CudaBackend::new(0).unwrap();
    let decode = |quantized: &DynamicFp8Tensor| {
        let width = quantized.values.shape().dims()[1];
        let values = backend
            .to_cpu(&quantized.values)
            .unwrap()
            .as_f8_e4m3()
            .unwrap()
            .iter()
            .map(|byte| {
                let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
                let exponent = (byte >> 3) & 0x0f;
                let mantissa = byte & 0x07;
                if exponent == 0 {
                    sign * mantissa as f32 * 2f32.powi(-9)
                } else if exponent == 0x0f && mantissa == 0x07 {
                    f32::NAN
                } else {
                    sign * (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
                }
            })
            .collect::<Vec<_>>();
        let scales = backend
            .to_cpu(&quantized.scales)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        values
            .into_iter()
            .enumerate()
            .map(|(index, value)| value * scales[index / width])
            .collect::<Vec<_>>()
    };
    let compare = |name: &str, actual: &[f32], expected: &[f32]| {
        let error = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm = expected
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(error / norm < 0.02, "{name} relative L2={}", error / norm);
    };

    const SWIGLU_ROWS: usize = 3;
    const INNER: usize = 60;
    const SWIGLU_PADDED: usize = 64;
    let logical = (0..SWIGLU_ROWS * 2 * INNER)
        .map(|index| bf16::from_f32(((index * 13 % 89) as f32 - 44.0) / 31.0))
        .collect::<Vec<_>>();
    let mut physical = vec![bf16::from_f32(100.0); SWIGLU_ROWS * 2 * SWIGLU_PADDED];
    for row in 0..SWIGLU_ROWS {
        physical[row * 2 * SWIGLU_PADDED..row * 2 * SWIGLU_PADDED + 2 * INNER]
            .copy_from_slice(&logical[row * 2 * INNER..(row + 1) * 2 * INNER]);
    }
    let gate_up = backend
        .to_device(&Tensor::from_bf16(vec![SWIGLU_ROWS, 2 * SWIGLU_PADDED], &physical).unwrap())
        .unwrap();
    let logical_gate_up = slice_columns_bf16(backend.context(), &gate_up, 2 * INNER).unwrap();
    let swiglu_reference = swiglu_bf16(backend.context(), &logical_gate_up).unwrap();
    let swiglu_reference =
        quantize_rows_bf16_e4m3_padded(backend.context(), &swiglu_reference, SWIGLU_PADDED)
            .unwrap();
    let swiglu_fused =
        swiglu_quantize_rows_bf16_e4m3(backend.context(), &gate_up, None, INNER, SWIGLU_PADDED)
            .unwrap();
    compare(
        "vec4 SwiGLU quantization",
        &decode(&swiglu_fused),
        &decode(&swiglu_reference),
    );

    const RESIDUAL_ROWS: usize = 72;
    const COLS: usize = 64;
    const RESIDUAL_PADDED: usize = 72;
    let projection_values = (0..RESIDUAL_ROWS * COLS)
        .map(|index| bf16::from_f32(((index * 17 % 101) as f32 - 50.0) / 29.0))
        .collect::<Vec<_>>();
    let residual_values = (0..RESIDUAL_ROWS * COLS)
        .map(|index| bf16::from_f32(((index * 11 % 67) as f32 - 33.0) / 41.0))
        .collect::<Vec<_>>();
    let weight_values = (0..COLS)
        .map(|index| bf16::from_f32(0.75 + index as f32 / 256.0))
        .collect::<Vec<_>>();
    let bias_values = (0..COLS)
        .map(|index| bf16::from_f32((index as f32 - 31.0) / 211.0))
        .collect::<Vec<_>>();
    let projection = backend
        .to_device(&Tensor::from_bf16(vec![RESIDUAL_ROWS, COLS], &projection_values).unwrap())
        .unwrap();
    let residual = backend
        .to_device(&Tensor::from_bf16(vec![RESIDUAL_ROWS, COLS], &residual_values).unwrap())
        .unwrap();
    let weight = backend
        .to_device(&Tensor::from_bf16(vec![COLS], &weight_values).unwrap())
        .unwrap();
    let bias = backend
        .to_device(&Tensor::from_bf16(vec![COLS], &bias_values).unwrap())
        .unwrap();
    let residual_reference = bias_residual_rms_bf16(
        backend.context(),
        &projection,
        Some(&bias),
        &residual,
        &weight,
        1.0e-6,
    )
    .unwrap();
    let residual_reference = quantize_rows_bf16_e4m3_padded(
        backend.context(),
        &residual_reference.normalized,
        RESIDUAL_PADDED,
    )
    .unwrap();
    let residual_fused = bias_residual_rms_quantize_rows_bf16_e4m3(
        backend.context(),
        &projection,
        Some(&bias),
        &residual,
        &weight,
        1.0e-6,
        RESIDUAL_PADDED,
    )
    .unwrap();
    backend.synchronize().unwrap();
    compare(
        "large vec8 residual RMSNorm quantization",
        &decode(&residual_fused.normalized),
        &decode(&residual_reference),
    );
}

#[test]
fn fp8_large_k_gemm_matches_quantized_cpu_reference() {
    // Keep M/N on the 16-element alignment required by native FP8
    // cuBLASLt so the same accumulation check covers both native and
    // emulated execution.
    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 2048;
    let backend = CudaBackend::new(0).unwrap();
    let activation = (0..M * K)
        .map(|index| ((index * 17 % 127) as f32 - 63.0) / 31.0)
        .collect::<Vec<_>>();
    let weight = (0..K * N)
        .map(|index| ((index * 31 % 113) as f32 - 56.0) / 37.0)
        .collect::<Vec<_>>();
    let activation_scale = activation
        .iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()))
        / 448.0;
    let weight_scale = weight
        .iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()))
        / 448.0;
    let upload = |shape, values: &[f32]| {
        let values = values
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        backend
            .to_device(&Tensor::from_f16(shape, &values).unwrap())
            .unwrap()
    };
    let activation_fp8 = quantize_f16_e4m3(
        backend.context(),
        &upload(vec![M, K], &activation),
        activation_scale,
    )
    .unwrap();
    let weight_fp8 = quantize_f16_e4m3(
        backend.context(),
        &upload(vec![K, N], &weight),
        weight_scale,
    )
    .unwrap();
    let decode_e4m3 = |byte: u8| {
        let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
        let exponent = (byte >> 3) & 0x0f;
        let mantissa = byte & 0x07;
        if exponent == 0 {
            sign * mantissa as f32 * 2f32.powi(-9)
        } else if exponent == 0x0f && mantissa == 0x07 {
            f32::NAN
        } else {
            sign * (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
        }
    };
    let activation_quantized = backend
        .to_cpu(&activation_fp8)
        .unwrap()
        .as_f8_e4m3()
        .unwrap()
        .iter()
        .copied()
        .map(decode_e4m3)
        .collect::<Vec<_>>();
    let weight_quantized = backend
        .to_cpu(&weight_fp8)
        .unwrap()
        .as_f8_e4m3()
        .unwrap()
        .iter()
        .copied()
        .map(decode_e4m3)
        .collect::<Vec<_>>();
    assert!(activation_quantized.iter().all(|value| value.is_finite()));
    assert!(weight_quantized.iter().all(|value| value.is_finite()));
    let output = fp8_gemm_f16(
        backend.context(),
        &activation_fp8,
        &weight_fp8,
        activation_scale,
        weight_scale,
    )
    .unwrap();
    let output = backend.to_cpu(&output).unwrap().to_f32_vec().unwrap();

    for row in 0..M {
        for column in 0..N {
            let expected = (0..K)
                .map(|inner| {
                    activation_quantized[row * K + inner] * weight_quantized[inner * N + column]
                })
                .sum::<f32>()
                * activation_scale
                * weight_scale;
            let actual = output[row * N + column];
            assert!(
                (actual - expected).abs() < 0.02,
                "output[{row},{column}] {actual} != {expected}"
            );
        }
    }
}

#[test]
fn fused_fp8_bias_matches_decomposed_path() {
    const M: usize = 64;
    const N: usize = 192;
    const K: usize = 128;
    let backend = CudaBackend::new(0).unwrap();
    if backend.context().caps().arch_family != CudaArchFamily::Sm100 {
        return;
    }
    let values = [0xb8, 0xb0, 0x00, 0x30, 0x38];
    let activation = (0..M * K)
        .map(|index| values[index % values.len()])
        .collect::<Vec<_>>();
    let mut weight = vec![0u8; K * N];
    for col in 0..N {
        weight[(col % K) * N + col] = 0x38;
    }
    let bias = (0..N)
        .map(|col| f16::from_f32((col as f32 % 7.0 - 3.0) * 0.01))
        .collect::<Vec<_>>();
    let activation = backend
        .to_device(&Tensor::from_f8_e4m3(vec![M, K], &activation).unwrap())
        .unwrap();
    let weight = backend
        .to_device(&Tensor::from_f8_e4m3(vec![K, N], &weight).unwrap())
        .unwrap();
    let bias = backend
        .to_device(&Tensor::from_f16(vec![N], &bias).unwrap())
        .unwrap();

    let fused = try_fp8_gemm_bias_f16(backend.context(), &activation, &weight, &bias, 1.0, 1.0)
        .unwrap()
        .expect("Thor must expose the fused FP8 bias epilogue");
    let projection = fp8_gemm_f16(backend.context(), &activation, &weight, 1.0, 1.0).unwrap();
    let reference = bias_f16(backend.context(), &projection, Some(&bias)).unwrap();
    let fused = backend.to_cpu(&fused).unwrap().to_f32_vec().unwrap();
    let reference = backend.to_cpu(&reference).unwrap().to_f32_vec().unwrap();
    let max_abs = fused
        .iter()
        .zip(&reference)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 0.01,
        "fused bias GEMM max abs error is {max_abs}"
    );
}

#[test]
fn fused_fp8_bias_autotune_publishes_one_exact_key() {
    const M: usize = 64;
    const N: usize = 192;
    const K: usize = 128;
    let backend = CudaBackend::new(0).unwrap();
    if backend.context().caps().arch_family != CudaArchFamily::Sm100 {
        return;
    }
    crate::kernels::gemm::configure_tuning(
        backend.context(),
        crate::tuning::TuningMode::AutoTune,
        &[],
        None,
    )
    .unwrap();
    let activation = backend
        .to_device(&Tensor::from_f8_e4m3(vec![M, K], &vec![0x38; M * K]).unwrap())
        .unwrap();
    let weight = backend
        .to_device(&Tensor::from_f8_e4m3(vec![K, N], &vec![0x30; K * N]).unwrap())
        .unwrap();
    let bias = backend
        .to_device(&Tensor::from_f16(vec![N], &vec![f16::from_f32(0.25); N]).unwrap())
        .unwrap();

    try_fp8_gemm_bias_f16(backend.context(), &activation, &weight, &bias, 1.0, 1.0)
        .unwrap()
        .expect("Thor must expose the fused FP8 bias epilogue");
    let key = crate::kernels::fused::fp8_fused_tuning_key(
        backend.context(),
        M,
        N,
        K,
        crate::tuning::TuningDType::F16,
        crate::tuning::Epilogue::Bias,
    );
    let first_generation = backend.context().tuning().generation();
    assert_eq!(first_generation, 1);
    assert!(backend.context().tuning().lookup_gemm_exact(&key).is_some());

    try_fp8_gemm_bias_f16(backend.context(), &activation, &weight, &bias, 1.0, 1.0)
        .unwrap()
        .expect("cached fused FP8 bias plan must remain runnable");
    assert_eq!(backend.context().tuning().generation(), first_generation);
}

#[test]
fn fused_vision_fc1_gelu_matches_decomposed_path() {
    const M: usize = 512;
    const N: usize = 4304;
    const K: usize = 1152;
    const OUTPUT_SCALE: f32 = 0.01;
    let backend = CudaBackend::new(0).unwrap();
    if backend.context().caps().arch_family != CudaArchFamily::Sm100 {
        return;
    }
    let values = [0xb8, 0xb0, 0x00, 0x30, 0x38]; // -1, -0.5, 0, 0.5, 1
    let activation = (0..M * K)
        .map(|index| values[index % values.len()])
        .collect::<Vec<_>>();
    let mut weight = vec![0u8; K * N];
    for col in 0..N {
        weight[(col % K) * N + col] = 0x38; // 1.0
    }
    let bias = (0..N)
        .map(|col| f16::from_f32((col as f32 % 5.0 - 2.0) * 0.05))
        .collect::<Vec<_>>();
    let activation = backend
        .to_device(&Tensor::from_f8_e4m3(vec![M, K], &activation).unwrap())
        .unwrap();
    let weight = backend
        .to_device(&Tensor::from_f8_e4m3(vec![K, N], &weight).unwrap())
        .unwrap();
    let bias = backend
        .to_device(&Tensor::from_f16(vec![N], &bias).unwrap())
        .unwrap();

    let fused = try_fp8_gemm_bias_gelu_e4m3(
        backend.context(),
        &activation,
        &weight,
        &bias,
        1.0,
        1.0,
        OUTPUT_SCALE,
    )
    .unwrap()
    .expect("Thor must expose the fused FP8 GELU epilogue");
    let projection = fp8_gemm_f16(backend.context(), &activation, &weight, 1.0, 1.0).unwrap();
    let reference =
        bias_gelu_quant_f16_e4m3(backend.context(), &projection, &bias, OUTPUT_SCALE).unwrap();
    let fused = backend.to_cpu(&fused).unwrap();
    let reference = backend.to_cpu(&reference).unwrap();
    let fused = fused.as_f8_e4m3().unwrap();
    let reference = reference.as_f8_e4m3().unwrap();
    let mismatches = fused
        .iter()
        .zip(reference)
        .filter(|(actual, expected)| actual != expected)
        .count();
    assert!(
        mismatches * 1000 <= fused.len(),
        "fused FC1 GELU differs in {mismatches}/{} E4M3 values",
        fused.len()
    );
}

#[test]
fn fused_vision_fc2_residual_matches_decomposed_path() {
    const M: usize = 512;
    const N: usize = 1152;
    const K: usize = 4304;
    let backend = CudaBackend::new(0).unwrap();
    if backend.context().caps().arch_family != CudaArchFamily::Sm100 {
        return;
    }
    let values = [0xb0, 0x00, 0x30, 0x38]; // -0.5, 0, 0.5, 1
    let activation = (0..M * K)
        .map(|index| values[index % values.len()])
        .collect::<Vec<_>>();
    let mut weight = vec![0u8; K * N];
    for col in 0..N {
        weight[(col % K) * N + col] = 0x38;
    }
    let bias = (0..N)
        .map(|col| f16::from_f32((col as f32 % 7.0 - 3.0) * 0.01))
        .collect::<Vec<_>>();
    let residual = (0..M * N)
        .map(|index| f16::from_f32((index as f32 % 9.0 - 4.0) * 0.02))
        .collect::<Vec<_>>();
    let activation = backend
        .to_device(&Tensor::from_f8_e4m3(vec![M, K], &activation).unwrap())
        .unwrap();
    let weight = backend
        .to_device(&Tensor::from_f8_e4m3(vec![K, N], &weight).unwrap())
        .unwrap();
    let bias = backend
        .to_device(&Tensor::from_f16(vec![N], &bias).unwrap())
        .unwrap();
    let residual = backend
        .to_device(&Tensor::from_f16(vec![M, N], &residual).unwrap())
        .unwrap();

    let fused = try_fp8_gemm_bias_residual_f16(
        backend.context(),
        &activation,
        &weight,
        Some(&bias),
        &residual,
        1.0,
        1.0,
    )
    .unwrap()
    .expect("Thor must expose the fused FP8 residual epilogue");
    let projection = fp8_gemm_f16(backend.context(), &activation, &weight, 1.0, 1.0).unwrap();
    let reference =
        bias_residual_f16(backend.context(), &projection, Some(&bias), &residual).unwrap();
    let fused = backend.to_cpu(&fused).unwrap().to_f32_vec().unwrap();
    let reference = backend.to_cpu(&reference).unwrap().to_f32_vec().unwrap();
    let max_abs = fused
        .iter()
        .zip(&reference)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 1.0e-3,
        "fused FC2 residual max abs error is {max_abs}"
    );
}

#[test]
fn geglu_matches_gelu_tanh_reference() {
    const ROWS: usize = 2;
    const INNER: usize = 8;
    const SCALE: f32 = 0.01;
    let backend = CudaBackend::new(0).unwrap();
    let mut source = Vec::with_capacity(ROWS * 2 * INNER);
    let mut expected = Vec::with_capacity(ROWS * INNER);
    for row in 0..ROWS {
        let gates = (0..INNER)
            .map(|col| (row * INNER + col) as f32 * 0.4 - 3.0)
            .collect::<Vec<_>>();
        let ups = (0..INNER)
            .map(|col| (col as f32 - 3.5) * 0.25)
            .collect::<Vec<_>>();
        source.extend(gates.iter().chain(&ups).map(|value| f16::from_f32(*value)));
        expected.extend(gates.iter().zip(&ups).map(|(gate, up)| {
            let cdf =
                0.5 * (1.0 + (0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)).tanh());
            gate * cdf * up
        }));
    }
    let input = backend
        .to_device(&Tensor::from_f16(vec![ROWS, 2 * INNER], &source).unwrap())
        .unwrap();
    let output = geglu_quant_f16_e4m3(backend.context(), &input, SCALE).unwrap();
    let output = backend.to_cpu(&output).unwrap();
    let decode_e4m3 = |bits: u8| {
        let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
        let exponent = ((bits >> 3) & 0x0f) as i32;
        let mantissa = (bits & 0x07) as f32;
        if exponent == 0 {
            sign * (mantissa / 8.0) * 2.0f32.powi(-6)
        } else {
            sign * (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
        }
    };
    for (bits, reference) in output.as_f8_e4m3().unwrap().iter().zip(expected) {
        let actual = decode_e4m3(*bits) * SCALE;
        assert!(
            (actual - reference).abs() < 0.08,
            "GeGLU output {actual} differs from reference {reference}"
        );
    }
}

#[test]
fn packed_swiglu_quant_matches_silu_reference_with_bias() {
    const ROWS: usize = 3;
    const INNER: usize = 12;
    const SCALE: f32 = 0.02;
    let backend = CudaBackend::new(0).unwrap();
    let mut source = Vec::with_capacity(ROWS * 2 * INNER);
    for row in 0..ROWS {
        source.extend(
            (0..INNER).map(|col| f16::from_f32(((row * 13 + col * 7) % 29) as f32 / 8.0 - 1.75)),
        );
        source.extend(
            (0..INNER).map(|col| f16::from_f32(((row * 11 + col * 5) % 23) as f32 / 9.0 - 1.1)),
        );
    }
    let bias = (0..2 * INNER)
        .map(|index| bf16::from_f32((index as f32 - INNER as f32) / 64.0))
        .collect::<Vec<_>>();
    let source_gpu = backend
        .to_device(&Tensor::from_f16(vec![ROWS, 2 * INNER], &source).unwrap())
        .unwrap();
    let bias_gpu = backend
        .to_device(&Tensor::from_bf16(vec![2 * INNER], &bias).unwrap())
        .unwrap();
    let output =
        swiglu_quant_f16_e4m3(backend.context(), &source_gpu, Some(&bias_gpu), SCALE).unwrap();
    let output = backend.to_cpu(&output).unwrap();
    let decode_e4m3 = |bits: u8| {
        let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
        let exponent = ((bits >> 3) & 0x0f) as i32;
        let mantissa = (bits & 0x07) as f32;
        if exponent == 0 {
            sign * (mantissa / 8.0) * 2.0f32.powi(-6)
        } else {
            sign * (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
        }
    };
    for (index, bits) in output.as_f8_e4m3().unwrap().iter().enumerate() {
        let row = index / INNER;
        let col = index % INNER;
        let row_base = row * 2 * INNER;
        let gate = source[row_base + col].to_f32() + bias[col].to_f32();
        let up = source[row_base + INNER + col].to_f32() + bias[INNER + col].to_f32();
        let expected = gate / (1.0 + (-gate).exp()) * up;
        let actual = decode_e4m3(*bits) * SCALE;
        let tolerance = expected.abs() * 0.065 + SCALE;
        assert!(
            (actual - expected).abs() <= tolerance,
            "packed SwiGLU output {actual} differs from {expected} at {index} (tolerance {tolerance})"
        );
    }
}

#[test]
fn bias_gelu_quant_matches_tanh_reference() {
    const ROWS: usize = 2;
    const COLS: usize = 8;
    const SCALE: f32 = 0.01;
    let backend = CudaBackend::new(0).unwrap();
    let source = (0..ROWS * COLS)
        .map(|index| f16::from_f32(index as f32 * 0.35 - 2.5))
        .collect::<Vec<_>>();
    let bias = (0..COLS)
        .map(|col| f16::from_f32((col as f32 - 3.5) * 0.1))
        .collect::<Vec<_>>();
    let source = backend
        .to_device(&Tensor::from_f16(vec![ROWS, COLS], &source).unwrap())
        .unwrap();
    let bias = backend
        .to_device(&Tensor::from_f16(vec![COLS], &bias).unwrap())
        .unwrap();
    let output = bias_gelu_quant_f16_e4m3(backend.context(), &source, &bias, SCALE).unwrap();
    let output = backend.to_cpu(&output).unwrap();
    let decode_e4m3 = |bits: u8| {
        let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
        let exponent = ((bits >> 3) & 0x0f) as i32;
        let mantissa = (bits & 0x07) as f32;
        if exponent == 0 {
            sign * (mantissa / 8.0) * 2.0f32.powi(-6)
        } else {
            sign * (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
        }
    };
    for (index, bits) in output.as_f8_e4m3().unwrap().iter().enumerate() {
        let x = index as f32 * 0.35 - 2.5 + (index % COLS) as f32 * 0.1 - 0.35;
        let expected = 0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh());
        let actual = decode_e4m3(*bits) * SCALE;
        let tolerance = expected.abs() * 0.065 + SCALE;
        assert!(
                // E4M3 rounding can contribute up to roughly 6.25% relative error.
                (actual - expected).abs() < tolerance,
                "bias GELU output {actual} differs from reference {expected} at {index} (tolerance {tolerance})"
            );
    }
}

#[test]
fn mqa_flash_uniform_scores_average_values() {
    const PREFIX: usize = 16;
    const SUFFIX: usize = 4;
    const HEADS: usize = 8;
    const DIM: usize = 256;
    let backend = CudaBackend::new(0).unwrap();
    let zeros_q = vec![f16::ZERO; SUFFIX * HEADS * DIM];
    let zeros_prefix_k = vec![f16::ZERO; PREFIX * DIM];
    let zeros_suffix_k = vec![f16::ZERO; SUFFIX * DIM];
    let prefix_v = (0..PREFIX)
        .flat_map(|token| std::iter::repeat_n(f16::from_f32(token as f32), DIM))
        .collect::<Vec<_>>();
    let suffix_v = (0..SUFFIX)
        .flat_map(|token| std::iter::repeat_n(f16::from_f32((PREFIX + token) as f32), DIM))
        .collect::<Vec<_>>();
    let upload = |shape, values: &[f16]| {
        backend
            .to_device(&Tensor::from_f16(shape, values).unwrap())
            .unwrap()
    };
    let q = upload(vec![SUFFIX, HEADS, DIM], &zeros_q);
    let prefix_k = upload(vec![PREFIX, DIM], &zeros_prefix_k);
    let prefix_v = upload(vec![PREFIX, DIM], &prefix_v);
    let suffix_k = upload(vec![SUFFIX, DIM], &zeros_suffix_k);
    let suffix_v = upload(vec![SUFFIX, DIM], &suffix_v);
    let output = mqa_prefix_suffix_f16(
        backend.context(),
        &q,
        &prefix_k,
        &prefix_v,
        &suffix_k,
        &suffix_v,
    )
    .unwrap();
    let output = backend.to_cpu(&output).unwrap().to_f32_vec().unwrap();
    assert!(
        output.iter().all(|value| (*value - 9.5).abs() < 0.01),
        "first MQA outputs: {:?}",
        &output[..output.len().min(16)]
    );
}

#[test]
fn mqa_odd_key_count_uses_aligned_scalar_softmax() {
    const PREFIX: usize = 4;
    const SUFFIX: usize = 3;
    const HEADS: usize = 2;
    const DIM: usize = 8;
    let backend = CudaBackend::new(0).unwrap();
    let upload = |shape, values: &[f16]| {
        backend
            .to_device(&Tensor::from_f16(shape, values).unwrap())
            .unwrap()
    };
    let q = upload(
        vec![SUFFIX, HEADS, DIM],
        &vec![f16::ZERO; SUFFIX * HEADS * DIM],
    );
    let prefix_k = upload(vec![PREFIX, DIM], &vec![f16::ZERO; PREFIX * DIM]);
    let suffix_k = upload(vec![SUFFIX, DIM], &vec![f16::ZERO; SUFFIX * DIM]);
    let values = (0..PREFIX + SUFFIX)
        .flat_map(|token| std::iter::repeat_n(f16::from_f32(token as f32), DIM))
        .collect::<Vec<_>>();
    let prefix_v = upload(vec![PREFIX, DIM], &values[..PREFIX * DIM]);
    let suffix_v = upload(vec![SUFFIX, DIM], &values[PREFIX * DIM..]);
    let output = mqa_prefix_suffix_f16(
        backend.context(),
        &q,
        &prefix_k,
        &prefix_v,
        &suffix_k,
        &suffix_v,
    )
    .unwrap();
    let output = backend.to_cpu(&output).unwrap().to_f32_vec().unwrap();
    assert!(output.iter().all(|value| (*value - 3.0).abs() < 0.01));
}

#[test]
fn cublas_language_mqa_exact_shape_averages_values() {
    const TOKENS: usize = 712;
    const HEADS: usize = 8;
    const DIM: usize = 256;
    let backend = CudaBackend::new(0).unwrap();
    let upload = |shape, values: &[f16]| {
        backend
            .to_device(&Tensor::from_f16(shape, values).unwrap())
            .unwrap()
    };
    let q = upload(
        vec![TOKENS, HEADS, DIM],
        &vec![f16::ZERO; TOKENS * HEADS * DIM],
    );
    let k = upload(vec![TOKENS, 1, DIM], &vec![f16::ZERO; TOKENS * DIM]);
    let values = (0..TOKENS)
        .flat_map(|token| std::iter::repeat_n(f16::from_f32(token as f32), DIM))
        .collect::<Vec<_>>();
    let v = upload(vec![TOKENS, 1, DIM], &values);
    let output = mqa_f16(backend.context(), &q, &k, &v).unwrap();
    let output = backend.to_cpu(&output).unwrap().to_f32_vec().unwrap();
    let max_error = output
        .iter()
        .map(|value| (*value - 355.5).abs())
        .fold(0.0f32, f32::max);
    assert!(
        output.iter().all(|value| value.is_finite()) && max_error < 1.0,
        "language MQA max error {max_error}, first outputs: {:?}",
        &output[..16]
    );
}

#[test]
fn cublas_action_mqa_exact_shape_averages_values() {
    const PREFIX: usize = 712;
    const SUFFIX: usize = 10;
    const HEADS: usize = 8;
    const DIM: usize = 256;
    let backend = CudaBackend::new(0).unwrap();
    let upload = |shape, values: &[f16]| {
        backend
            .to_device(&Tensor::from_f16(shape, values).unwrap())
            .unwrap()
    };
    let q = upload(
        vec![SUFFIX, HEADS, DIM],
        &vec![f16::ZERO; SUFFIX * HEADS * DIM],
    );
    let prefix_k = upload(vec![PREFIX, DIM], &vec![f16::ZERO; PREFIX * DIM]);
    let suffix_k = upload(vec![SUFFIX, DIM], &vec![f16::ZERO; SUFFIX * DIM]);
    let prefix_values = (0..PREFIX)
        .flat_map(|token| std::iter::repeat_n(f16::from_f32(token as f32), DIM))
        .collect::<Vec<_>>();
    let suffix_values = (PREFIX..PREFIX + SUFFIX)
        .flat_map(|token| std::iter::repeat_n(f16::from_f32(token as f32), DIM))
        .collect::<Vec<_>>();
    let prefix_v = upload(vec![PREFIX, DIM], &prefix_values);
    let suffix_v = upload(vec![SUFFIX, DIM], &suffix_values);
    let output = mqa_prefix_suffix_f16(
        backend.context(),
        &q,
        &prefix_k,
        &prefix_v,
        &suffix_k,
        &suffix_v,
    )
    .unwrap();
    let output = backend.to_cpu(&output).unwrap().to_f32_vec().unwrap();
    let max_error = output
        .iter()
        .map(|value| (*value - 360.5).abs())
        .fold(0.0f32, f32::max);
    assert!(
        output.iter().all(|value| value.is_finite()) && max_error < 1.0,
        "action MQA max error {max_error}, first outputs: {:?}",
        &output[..16]
    );
}

#[test]
fn embedding_concat_and_euler_run_on_device() {
    let backend = CudaBackend::new(0).unwrap();
    let table = backend
        .to_device(
            &Tensor::from_f16(
                vec![3, 2],
                &[
                    f16::from_f32(1.0),
                    f16::from_f32(2.0),
                    f16::from_f32(3.0),
                    f16::from_f32(4.0),
                    f16::from_f32(5.0),
                    f16::from_f32(6.0),
                ],
            )
            .unwrap(),
        )
        .unwrap();
    let ids = CudaBuffer::alloc(8, 0).unwrap();
    let mut id_bytes = Vec::with_capacity(8);
    id_bytes.extend_from_slice(&2u32.to_ne_bytes());
    id_bytes.extend_from_slice(&0u32.to_ne_bytes());
    ids.copy_from_host(&id_bytes).unwrap();
    let embedded = lookup_f16(backend.context(), &table, &ids, 2).unwrap();
    let first = backend
        .to_device(
            &Tensor::from_f16(vec![1, 2], &[f16::from_f32(7.0), f16::from_f32(8.0)]).unwrap(),
        )
        .unwrap();
    let joined = concat_rows_f16(backend.context(), &first, &embedded).unwrap();
    let velocity = backend
        .to_device(&Tensor::from_f16(vec![3, 2], &vec![f16::from_f32(2.0); 6]).unwrap())
        .unwrap();
    let updated = euler_update_f16(backend.context(), &joined, &velocity, -0.5).unwrap();
    let updated = backend.to_cpu(&updated).unwrap().to_f32_vec().unwrap();
    let normalizer = 2.0f32.sqrt();
    let expected = [
        6.0,
        7.0,
        5.0 * normalizer - 1.0,
        6.0 * normalizer - 1.0,
        normalizer - 1.0,
        2.0 * normalizer - 1.0,
    ];
    assert!(
        updated
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (*actual - expected).abs() < 0.01),
        "updated embedding/Euler values: {updated:?}"
    );
}

#[test]
fn workspace_reuses_stable_output_addresses() {
    let backend = CudaBackend::new(0).unwrap();
    let host = Tensor::from_f16(vec![4, 4], &vec![f16::ONE; 16]).unwrap();
    let input = backend.to_device(&host).unwrap();
    let workspace = GraphWorkspace::new(4096, 0).unwrap();

    let first = with_workspace(&workspace, || {
        quantize_f16_e4m3(backend.context(), &input, 1.0)
    })
    .unwrap();
    let first_ptr = gpu_ptr(&first).unwrap();
    assert_eq!(workspace.used(), 16);
    let first_values = backend
        .to_cpu(&first)
        .unwrap()
        .as_f8_e4m3()
        .unwrap()
        .to_vec();
    drop(first);

    let replacement = Tensor::from_f16(vec![4, 4], &vec![f16::from_f32(2.0); 16]).unwrap();
    crate::transfers::copy_cpu_to_cuda(&replacement, &input).unwrap();
    let second = with_workspace(&workspace, || {
        quantize_f16_e4m3(backend.context(), &input, 1.0)
    })
    .unwrap();
    assert_eq!(gpu_ptr(&second).unwrap(), first_ptr);
    backend.synchronize().unwrap();
    let second_values = backend
        .to_cpu(&second)
        .unwrap()
        .as_f8_e4m3()
        .unwrap()
        .to_vec();
    assert_ne!(first_values, second_values);
}

#[test]
fn vision_mha_exact_siglip_shape_keeps_views_independent() {
    let backend = CudaBackend::new(0).unwrap();
    const VIEWS: usize = 2;
    const TOKENS: usize = 256;
    const HEADS: usize = 16;
    const DIM: usize = 72;
    let shape = vec![VIEWS * TOKENS, HEADS, DIM];
    let zeros = vec![f16::ZERO; shape.iter().product()];
    let values = (0..VIEWS)
        .flat_map(|view| {
            (0..TOKENS).flat_map(move |token| {
                (0..HEADS).flat_map(move |head| {
                    (0..DIM).map(move |dim| {
                        f16::from_f32(
                            view as f32 * 20.0
                                + token as f32 / 256.0
                                + head as f32 / 32.0
                                + dim as f32 / 1024.0,
                        )
                    })
                })
            })
        })
        .collect::<Vec<_>>();
    let upload = |values: &[f16]| {
        backend
            .to_device(&Tensor::from_f16(shape.clone(), values).unwrap())
            .unwrap()
    };
    let q = upload(&zeros);
    let k = upload(&zeros);
    let v = upload(&values);
    let output = mha_f16(backend.context(), &q, &k, &v, 256).unwrap();
    let output = backend.to_cpu(&output).unwrap().to_f32_vec().unwrap();
    for view in 0..VIEWS {
        for token in 0..TOKENS {
            for head in 0..HEADS {
                for dim in 0..DIM {
                    let index = ((view * TOKENS + token) * HEADS + head) * DIM + dim;
                    let expected = view as f32 * 20.0
                        + 127.5 / 256.0
                        + head as f32 / 32.0
                        + dim as f32 / 1024.0;
                    assert!(
                        (output[index] - expected).abs() < 0.015,
                        "view {view}, token {token}, head {head}, dim {dim}: {} != {expected}",
                        output[index]
                    );
                }
            }
        }
    }
}

#[test]
fn vision_mha_exact_siglip_shape_matches_sampled_cpu_reference() {
    let backend = CudaBackend::new(0).unwrap();
    const VIEWS: usize = 2;
    const TOKENS: usize = 256;
    const HEADS: usize = 16;
    const DIM: usize = 72;
    let shape = vec![VIEWS * TOKENS, HEADS, DIM];
    let elements = shape.iter().product::<usize>();
    let make_values = |multiplier: usize, offset: usize, amplitude: f32| {
        (0..elements)
            .map(|index| {
                let centered =
                    ((index * multiplier + index / DIM * 7 + offset) % 97) as f32 / 48.0 - 1.0;
                f16::from_f32(centered * amplitude)
            })
            .collect::<Vec<_>>()
    };
    let q_host = make_values(17, 3, 0.25);
    let k_host = make_values(29, 11, 0.25);
    let v_host = make_values(43, 23, 0.75);
    let upload = |values: &[f16]| {
        backend
            .to_device(&Tensor::from_f16(shape.clone(), values).unwrap())
            .unwrap()
    };
    let q = upload(&q_host);
    let k = upload(&k_host);
    let v = upload(&v_host);
    let output = mha_f16(backend.context(), &q, &k, &v, TOKENS).unwrap();
    let output = backend.to_cpu(&output).unwrap().to_f32_vec().unwrap();
    let q_host = q_host.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
    let k_host = k_host.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
    let v_host = v_host.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
    let scale = 1.0 / (DIM as f32).sqrt();

    for view in 0..VIEWS {
        for query in [0, 37, TOKENS - 1] {
            for head in [0, 5, HEADS - 1] {
                let q_base = ((view * TOKENS + query) * HEADS + head) * DIM;
                let mut scores = Vec::with_capacity(TOKENS);
                for key in 0..TOKENS {
                    let k_base = ((view * TOKENS + key) * HEADS + head) * DIM;
                    let dot = (0..DIM)
                        .map(|dim| q_host[q_base + dim] * k_host[k_base + dim])
                        .sum::<f32>();
                    scores.push(dot * scale);
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator = scores
                    .iter_mut()
                    .map(|score| {
                        *score = (*score - maximum).exp();
                        *score
                    })
                    .sum::<f32>();
                for dim in [0, 31, DIM - 1] {
                    let expected = (0..TOKENS)
                        .map(|key| {
                            let v_index = ((view * TOKENS + key) * HEADS + head) * DIM + dim;
                            scores[key] / denominator * v_host[v_index]
                        })
                        .sum::<f32>();
                    let actual = output[q_base + dim];
                    assert!(
                            (actual - expected).abs() < 0.003,
                            "view {view}, query {query}, head {head}, dim {dim}: {actual} != {expected}"
                        );
                }
            }
        }
    }
}

#[test]
fn vision_mha_matches_nonuniform_cpu_reference() {
    const BATCHES: usize = 2;
    const TOKENS: usize = 5;
    const HEADS: usize = 3;
    const DIM: usize = 8;
    let backend = CudaBackend::new(0).unwrap();
    let elements = BATCHES * TOKENS * HEADS * DIM;
    let make_values = |phase: f32| {
        (0..elements)
            .map(|index| {
                let value = ((index * 17 + 3) % 29) as f32 / 14.0 - 1.0 + phase;
                f16::from_f32(value)
            })
            .collect::<Vec<_>>()
    };
    let q_host = make_values(0.0);
    let k_host = make_values(0.125);
    let v_host = make_values(-0.25);
    let upload = |values: &[f16]| {
        backend
            .to_device(&Tensor::from_f16(vec![BATCHES * TOKENS, HEADS, DIM], values).unwrap())
            .unwrap()
    };
    let q = upload(&q_host);
    let k = upload(&k_host);
    let v = upload(&v_host);
    let output = mha_f16(backend.context(), &q, &k, &v, TOKENS).unwrap();
    let output = backend.to_cpu(&output).unwrap().to_f32_vec().unwrap();
    let q_host = q_host.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
    let k_host = k_host.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
    let v_host = v_host.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
    let scale = 1.0 / (DIM as f32).sqrt();
    for batch in 0..BATCHES {
        for query in 0..TOKENS {
            for head in 0..HEADS {
                let q_base = ((batch * TOKENS + query) * HEADS + head) * DIM;
                let mut scores = Vec::with_capacity(TOKENS);
                for key in 0..TOKENS {
                    let k_base = ((batch * TOKENS + key) * HEADS + head) * DIM;
                    let dot = (0..DIM)
                        .map(|dim| q_host[q_base + dim] * k_host[k_base + dim])
                        .sum::<f32>();
                    scores.push(dot * scale);
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator = scores
                    .iter_mut()
                    .map(|score| {
                        *score = (*score - maximum).exp();
                        *score
                    })
                    .sum::<f32>();
                for dim in 0..DIM {
                    let expected = (0..TOKENS)
                        .map(|key| {
                            let v_index = ((batch * TOKENS + key) * HEADS + head) * DIM + dim;
                            scores[key] / denominator * v_host[v_index]
                        })
                        .sum::<f32>();
                    let output_index = q_base + dim;
                    assert!(
                        (output[output_index] - expected).abs() < 0.002,
                        "batch {batch}, query {query}, head {head}, dim {dim}: {} != {expected}",
                        output[output_index]
                    );
                }
            }
        }
    }
}

#[cfg(apxinf_cutlass_gemm)]
#[test]
fn cutlass_vision_qkv_tactics_match_cublaslt() {
    const M: usize = 512;
    const N: usize = 3456;
    const K: usize = 1152;
    let backend = CudaBackend::new(0).unwrap();
    let activation_host = (0..M * K)
        .map(|index| {
            let value = ((index * 17 + index / K * 3) % 31) as f32 / 15.0 - 1.0;
            f16::from_f32(value)
        })
        .collect::<Vec<_>>();
    let weight_host = (0..K * N)
        .map(|index| {
            let value = (((index * 7 + index / N * 5) % 23) as f32 / 11.0 - 1.0) * 0.125;
            f16::from_f32(value)
        })
        .collect::<Vec<_>>();
    let activation_scale = 1.0 / 448.0;
    let weight_scale = 0.125 / 448.0;
    let activation = backend
        .to_device(&Tensor::from_f16(vec![M, K], &activation_host).unwrap())
        .unwrap();
    let weight = backend
        .to_device(&Tensor::from_f16(vec![K, N], &weight_host).unwrap())
        .unwrap();
    let activation = quantize_f16_e4m3(backend.context(), &activation, activation_scale).unwrap();
    let weight = quantize_f16_e4m3(backend.context(), &weight, weight_scale).unwrap();
    prepare_cublaslt_fp8_gemm(M, N, K).unwrap();

    let run = |tactic: Option<i32>| {
        let output = CudaBuffer::alloc_zeros(M * N * 2, backend.device_id()).unwrap();
        let status = unsafe {
            match tactic {
                Some(tactic) => ffi::apxinf_static_cutlass_fp8_gemm_f16(
                    gpu_ptr(&activation).unwrap(),
                    gpu_ptr(&weight).unwrap(),
                    output.ptr(),
                    M as i32,
                    N as i32,
                    K as i32,
                    activation_scale * weight_scale,
                    tactic,
                    backend.context().stream().handle(),
                ),
                None => ffi::apxinf_static_fp8_gemm_f16(
                    gpu_ptr(&activation).unwrap(),
                    gpu_ptr(&weight).unwrap(),
                    output.ptr(),
                    M as i32,
                    N as i32,
                    K as i32,
                    activation_scale * weight_scale,
                    backend.context().stream().handle(),
                ),
            }
        };
        assert_eq!(status, 0, "FP8 GEMM launch failed for tactic {tactic:?}");
        backend.synchronize().unwrap();
        let output = make_gpu_tensor(
            Shape::new(vec![M, N]),
            DType::F16,
            backend.device_id(),
            output,
        );
        backend.to_cpu(&output).unwrap().to_f32_vec().unwrap()
    };
    let reference = run(None);
    let reference_l2 = reference
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    let mut failures = Vec::new();
    for tactic in 0..=7 {
        let output = run(Some(tactic));
        let dot = reference
            .iter()
            .zip(&output)
            .map(|(left, right)| f64::from(*left) * f64::from(*right))
            .sum::<f64>();
        let output_l2 = output
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        let relative_l2 = reference
            .iter()
            .zip(&output)
            .map(|(left, right)| f64::from(*left - *right).powi(2))
            .sum::<f64>()
            .sqrt()
            / reference_l2;
        let max_abs = reference
            .iter()
            .zip(&output)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        let cosine = dot / (reference_l2 * output_l2);
        println!(
                "CUTLASS tactic {tactic}: cosine={cosine:.9}, relative_l2={relative_l2:.9}, max_abs={max_abs:.6}"
            );
        if cosine < 0.99999 || relative_l2 > 0.001 || max_abs > 0.02 {
            failures.push((tactic, cosine, relative_l2, max_abs));
        }
    }
    assert!(
        failures.is_empty(),
        "CUTLASS tactics disagree with cuBLASLt: {failures:?}"
    );
}

#[cfg(apxinf_cutlass_gemm)]
#[test]
fn cutlass_action_qkv_shape_runs_on_device() {
    let backend = CudaBackend::new(0).unwrap();
    let activation = backend
        .to_device(&Tensor::from_f8_e4m3(vec![10, 1024], &vec![0; 10 * 1024]).unwrap())
        .unwrap();
    let weight = backend
        .to_device(&Tensor::from_f8_e4m3(vec![1024, 2560], &vec![0; 1024 * 2560]).unwrap())
        .unwrap();
    let output = fp8_gemm_f16(backend.context(), &activation, &weight, 1.0, 1.0).unwrap();
    let output = backend.to_cpu(&output).unwrap();
    assert!(output.as_f16().unwrap().iter().all(|x| *x == f16::ZERO));
}

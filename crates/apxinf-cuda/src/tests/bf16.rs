use std::path::Path;

use apxinf_core::{Backend, Tensor};
use half::bf16;

use crate::tuning::{
    lookup_gemm_exact, DeviceFingerprint, Epilogue, GemmLayout, GemmOp, GemmTuningKey, ScaleMode,
    TacticBackend, TuningDType,
};
use crate::CudaBackend;

#[test]
fn persisted_bf16_cublaslt_tactic_matches_vendor() {
    const M: usize = 10;
    const N: usize = 32;
    const K: usize = 1024;

    let Some(tactics_path) = std::env::var_os("APXINF_TEST_BF16_TACTICS") else {
        eprintln!("set APXINF_TEST_BF16_TACTICS to run persisted BF16 tactic validation");
        return;
    };
    let backend = CudaBackend::new(0).unwrap();
    let activation_values = (0..M * K)
        .map(|index| bf16::from_f32(((index * 17 % 31) as f32 - 15.0) / 128.0))
        .collect::<Vec<_>>();
    let weight_values = (0..K * N)
        .map(|index| bf16::from_f32(((index * 13 % 29) as f32 - 14.0) / 128.0))
        .collect::<Vec<_>>();
    let activation = backend
        .to_device(&Tensor::from_bf16(vec![M, K], &activation_values).unwrap())
        .unwrap();
    let weight = backend
        .to_device(&Tensor::from_bf16(vec![K, N], &weight_values).unwrap())
        .unwrap();

    let reference = crate::kernels::gemm::matmul(backend.context(), &activation, &weight).unwrap();
    let database = crate::tuning::TuningDb::from_json_file(Path::new(&tactics_path)).unwrap();
    crate::kernels::gemm::install_tuning_db(backend.context(), &database).unwrap();
    let key = GemmTuningKey {
        op: GemmOp::Bf16,
        device: DeviceFingerprint::from(backend.context().caps()),
        m: M,
        n: N,
        k: K,
        activation_dtype: TuningDType::Bf16,
        weight_dtype: TuningDType::Bf16,
        output_dtype: TuningDType::Bf16,
        layout: GemmLayout::RowMajor,
        scale_mode: ScaleMode::None,
        epilogue: Epilogue::None,
        workspace_limit: usize::MAX,
    };
    let tactic = lookup_gemm_exact(&key).expect("missing exact BF16 test tactic");
    assert_eq!(tactic.backend, TacticBackend::CublasLt);
    let actual = crate::kernels::gemm::bf16(backend.context(), &activation, &weight).unwrap();

    let reference = backend.to_cpu(&reference).unwrap().to_f32_vec().unwrap();
    let actual = backend.to_cpu(&actual).unwrap().to_f32_vec().unwrap();
    let mut max_abs = 0.0f32;
    let mut square_error = 0.0f64;
    for (&expected, &observed) in reference.iter().zip(&actual) {
        let error = (expected - observed).abs();
        max_abs = max_abs.max(error);
        square_error += f64::from(error * error);
    }
    let rmse = (square_error / reference.len() as f64).sqrt();
    eprintln!(
        "persisted BF16 {:?}:{} vs vendor: max_abs={max_abs}, rmse={rmse}",
        tactic.backend, tactic.value
    );
    assert!(
        max_abs <= 0.125 && rmse <= 0.02,
        "persisted BF16 tactic diverged from vendor: max_abs={max_abs}, rmse={rmse}"
    );
}

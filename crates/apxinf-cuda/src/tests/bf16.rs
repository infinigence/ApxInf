use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;

use apxinf_core::{Backend, DType, Tensor};
use half::bf16;

use crate::tuning::{
    DeviceFingerprint, Epilogue, GemmLayout, GemmOp, GemmTuningKey, ScaleMode, TacticBackend,
    TacticMatch, TuningDType,
};
use crate::{CudaBackend, CudaBuffer};

struct CountingObserver(Cell<usize>);

impl crate::kernels::gemm::Bf16ActivationObserver for CountingObserver {
    fn observe(&self, _activation: &Tensor, _weight: &Tensor) -> apxinf_core::Result<()> {
        self.0.set(self.0.get() + 1);
        Ok(())
    }
}

#[test]
fn normalized_temporal_merged_rgb_preprocessing_matches_reference_order() {
    const VIEWS: usize = 2;
    const IMAGE_SIZE: usize = 8;
    const PATCH_SIZE: usize = 2;
    const TEMPORAL_PATCH_SIZE: usize = 2;
    const MERGE_SIZE: usize = 2;
    const GRID_SIZE: usize = IMAGE_SIZE / PATCH_SIZE;
    const GROUPS_PER_SIDE: usize = GRID_SIZE / MERGE_SIZE;
    const PATCH_ROWS: usize = VIEWS * GRID_SIZE * GRID_SIZE;
    const PATCH_WIDTH: usize = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;
    const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_72];
    const STD: [f32; 3] = [0.268_629_55, 0.261_302_6, 0.275_777_1];

    let backend = CudaBackend::new(0).unwrap();
    let nhwc = (0..VIEWS * IMAGE_SIZE * IMAGE_SIZE * 3)
        .map(|index| (index * 17 % 256) as u8)
        .collect::<Vec<_>>();
    let mut expected = vec![bf16::ZERO; PATCH_ROWS * PATCH_WIDTH];
    for view in 0..VIEWS {
        for group_y in 0..GROUPS_PER_SIDE {
            for group_x in 0..GROUPS_PER_SIDE {
                for merge_y in 0..MERGE_SIZE {
                    for merge_x in 0..MERGE_SIZE {
                        let row = ((((view * GROUPS_PER_SIDE + group_y) * GROUPS_PER_SIDE
                            + group_x)
                            * MERGE_SIZE
                            + merge_y)
                            * MERGE_SIZE)
                            + merge_x;
                        for channel in 0..3 {
                            for temporal in 0..TEMPORAL_PATCH_SIZE {
                                for dy in 0..PATCH_SIZE {
                                    for dx in 0..PATCH_SIZE {
                                        let y = (group_y * MERGE_SIZE + merge_y) * PATCH_SIZE + dy;
                                        let x = (group_x * MERGE_SIZE + merge_x) * PATCH_SIZE + dx;
                                        let source = ((view * IMAGE_SIZE + y) * IMAGE_SIZE + x) * 3
                                            + channel;
                                        let column = (((channel * TEMPORAL_PATCH_SIZE + temporal)
                                            * PATCH_SIZE
                                            + dy)
                                            * PATCH_SIZE)
                                            + dx;
                                        let scaled =
                                            (f64::from(nhwc[source]) * (1.0 / 255.0)) as f32;
                                        expected[row * PATCH_WIDTH + column] =
                                            bf16::from_f32((scaled - MEAN[channel]) / STD[channel]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut nchw = vec![0u8; nhwc.len()];
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

    for (bytes, layout) in [
        (&nhwc, crate::kernels::preprocess::ImageLayout::Nhwc),
        (&nchw, crate::kernels::preprocess::ImageLayout::Nchw),
    ] {
        let input = CudaBuffer::alloc(bytes.len(), backend.device_id()).unwrap();
        input.copy_from_host(bytes).unwrap();
        let output = backend
            .to_device(&Tensor::zeros((PATCH_ROWS, PATCH_WIDTH), DType::BF16))
            .unwrap();
        crate::kernels::preprocess::rgb_u8_to_normalized_temporal_merged_patches_bf16(
            backend.context(),
            &input,
            &output,
            VIEWS,
            IMAGE_SIZE,
            PATCH_SIZE,
            TEMPORAL_PATCH_SIZE,
            MERGE_SIZE,
            layout,
            1.0 / 255.0,
            MEAN,
            STD,
        )
        .unwrap();

        let actual = backend.to_cpu(&output).unwrap();
        assert_eq!(actual.as_bf16().unwrap(), expected);
    }
}

#[test]
fn bf16_geglu_cold_autotune_resolves_dependencies_without_reentering_tune_lock() {
    const M: usize = 3;
    const K: usize = 5;
    const FULL_N: usize = 14;

    let backend = CudaBackend::new(0).unwrap();
    crate::kernels::gemm::configure_tuning(
        backend.context(),
        crate::tuning::TuningMode::AutoTune,
        &[],
        None,
    )
    .unwrap();
    let activation = backend
        .to_device(&Tensor::from_bf16(vec![M, K], &vec![bf16::ZERO; M * K]).unwrap())
        .unwrap();
    let weight = backend
        .to_device(&Tensor::from_bf16(vec![K, FULL_N], &vec![bf16::ZERO; K * FULL_N]).unwrap())
        .unwrap();
    let observer = Rc::new(CountingObserver(Cell::new(0)));
    let _guard = crate::kernels::gemm::install_bf16_observer(observer.clone()).unwrap();

    let fused = crate::kernels::gemm::bf16_geglu_fused(
        backend.context(),
        &activation,
        &weight,
        false,
        None,
        None,
    )
    .unwrap();
    assert_eq!(fused.shape().dims(), [M, FULL_N / 2]);
    assert_eq!(
        observer.0.get(),
        1,
        "the decomposed GeGLU path must observe exactly once"
    );
    for epilogue in [Epilogue::None, Epilogue::GeGlu] {
        let key = GemmTuningKey {
            op: GemmOp::Bf16,
            device: DeviceFingerprint::from(backend.context().caps()),
            m: M,
            n: FULL_N,
            k: K,
            activation_dtype: TuningDType::Bf16,
            weight_dtype: TuningDType::Bf16,
            output_dtype: TuningDType::Bf16,
            layout: GemmLayout::RowMajor,
            scale_mode: ScaleMode::None,
            epilogue,
            workspace_limit: usize::MAX,
        };
        assert!(backend.context().tuning().lookup_gemm_exact(&key).is_some());
    }
}

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
    let resolved = backend
        .context()
        .tuning()
        .lookup_gemm(&key)
        .expect("missing exact BF16 test tactic");
    assert_eq!(resolved.source, TacticMatch::Exact);
    let tactic = resolved.tactic;
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

#[test]
fn temporal_merged_rescale_matches_float64_reference_at_bf16_boundary() {
    let backend = CudaBackend::new(0).unwrap();
    let bytes: Vec<u8> = (0..256).flat_map(|v| [v as u8; 3]).collect();
    let input = CudaBuffer::alloc(bytes.len(), backend.device_id()).unwrap();
    input.copy_from_host(&bytes).unwrap();
    let output = backend
        .to_device(&Tensor::zeros((256, 6), DType::BF16))
        .unwrap();
    let scale = 1.0f64 / 255.0;
    let expected: Vec<bf16> = (0..256)
        .flat_map(|v| [bf16::from_f32((f64::from(v) * scale) as f32 - 0.5); 6])
        .collect();
    assert_eq!(expected[127 * 6].to_bits(), 0xbb01);
    crate::kernels::preprocess::rgb_u8_to_normalized_temporal_merged_patches_bf16(
        backend.context(),
        &input,
        &output,
        1,
        16,
        1,
        2,
        1,
        crate::kernels::preprocess::ImageLayout::Nhwc,
        scale,
        [0.5; 3],
        [1.0; 3],
    )
    .unwrap();
    assert_eq!(
        backend.to_cpu(&output).unwrap().as_bf16().unwrap(),
        expected
    );
}

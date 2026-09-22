//! Regression for BF16 projection bias being applied after a BF16 dot rounding.
use apxinf_core::Tensor;
use apxinf_cuda::{kernels::gemm, transfers, CudaContext};
use half::bf16;

#[test]
#[ignore = "requires a CUDA device"]
fn bias_is_added_before_bf16_output_rounding() {
    let ctx = CudaContext::new(0).unwrap();
    for (n, k) in [(1, 1), (7, 9), (1024, 256), (1024, 1024), (6144, 1024)] {
        let mut weights = vec![bf16::from_f32(1.0 / 4096.0); n * k];
        let vector = vec![bf16::ONE; k];
        for row in 0..n {
            weights[row * k] = vector[0];
        }
        let upload = |shape, data: &[bf16]| {
            transfers::to_cuda(&Tensor::from_bf16(shape, data).unwrap(), 0).unwrap()
        };
        let weight = upload(vec![n, k], &weights);
        let input = upload(vec![k], &vector);
        let bias = upload(vec![n], &vec![bf16::from_f32(-1.0); n]);
        let output = gemm::bf16_addmv(&ctx, &weight, &input, &bias).unwrap();
        ctx.synchronize().unwrap();
        let values = transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap();
        let expected = bf16::from_f32((k - 1) as f32 / 4096.0).to_f32();
        assert!(
            values.iter().all(|&v| v == expected),
            "shape [{n}, {k}]: got {}, expected {expected}",
            values[0]
        );
    }
}

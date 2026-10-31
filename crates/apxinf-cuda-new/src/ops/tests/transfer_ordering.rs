//! A host upload must be complete before a non-blocking compute stream reads
//! or overwrites the destination. Pageable cudaMemcpy can return after staging.
use super::*;
use crate::{transfers, CudaContext};
use apxinf_core::Tensor;

#[test]
fn host_upload_is_complete_before_nonblocking_compute() {
    let ctx = CudaContext::new(0).unwrap();
    let n = 5120;
    let input = Tensor::from_f32(vec![1, n], &vec![2.; n]).unwrap();
    let zeros = Tensor::from_f32(vec![1, n], &vec![0.; n]).unwrap();
    let x = transfers::to_cuda(&input, 0).unwrap();
    let mut out = transfers::to_cuda(&zeros, 0).unwrap();
    for iteration in 0..2048 {
        transfers::copy_cpu_to_cuda(&zeros, &out).unwrap();
        let mut args = PointwiseArgs::new(PointwiseSemantic::Scale, &x, &mut out);
        args.dt = 3.;
        pointwise(&ctx, args).unwrap();
        ctx.synchronize().unwrap();
        let actual = transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap();
        if let Some((index, value)) = actual.iter().enumerate().find(|(_, v)| **v != 6.) {
            panic!("upload/compute ordering at iteration {iteration}, element {index}: expected 6, got {value}");
        }
    }
}

#[test]
fn captured_device_copy_retains_inputs_and_rejects_overlap() {
    use apxinf_core::{DType, Shape};
    let ctx = CudaContext::new(0).unwrap();
    let host = Tensor::from_f32(vec![1, 4], &[1., 2., 3., 4.]).unwrap();
    let source = transfers::to_cuda(&host, 0).unwrap();
    let destination = ctx
        .allocate_output(Shape::new(vec![1, 4]), DType::F32)
        .unwrap();
    let graph = crate::capture(&ctx, || {
        transfers::copy_cuda_to_cuda(&ctx, &source, &destination)
    })
    .unwrap();
    drop(source);
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(
        transfers::to_cpu(&destination)
            .unwrap()
            .to_f32_vec()
            .unwrap(),
        vec![1., 2., 3., 4.]
    );
    let buffer = crate::CudaBuffer::from_tensor(&destination).unwrap();
    let a = buffer
        .view(0, 12)
        .unwrap()
        .as_tensor(Shape::new(vec![1, 3]), DType::F32)
        .unwrap();
    let b = buffer
        .view(4, 12)
        .unwrap()
        .as_tensor(Shape::new(vec![1, 3]), DType::F32)
        .unwrap();
    assert!(transfers::copy_cuda_to_cuda(&ctx, &a, &b)
        .unwrap_err()
        .to_string()
        .contains("overlap"));
}

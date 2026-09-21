use apxinf_core::{Backend, DType, KvCache, Shape};
use apxinf_cuda::{kernels, CudaBackend, CudaBuffer, CudaKVCache};

fn address(buffer: &CudaBuffer) -> usize {
    buffer
        .as_tensor(Shape::new(vec![buffer.len() / 4]), DType::F32)
        .unwrap()
        .storage()
        .as_gpu()
        .unwrap()
        .ptr
}

#[test]
fn clear_preserves_buffers_used_by_captured_graph() {
    let backend = CudaBackend::new(0).expect("CUDA device required");
    let mut cache = CudaKVCache::new(0, 2, 2, 8, 4).unwrap();
    // Retain aliases so allocator address reuse cannot hide the regression.
    let buffers: Vec<_> = (0..2)
        .flat_map(|layer| [cache.k_buffer(layer).clone(), cache.v_buffer(layer).clone()])
        .collect();
    for buffer in &buffers {
        let bytes: Vec<_> = (0..buffer.len() / 4)
            .flat_map(|_| 3.0f32.to_le_bytes())
            .collect();
        buffer.copy_from_host(&bytes).unwrap();
    }
    cache.advance(3);
    let output = CudaBuffer::alloc_zeros(buffers[0].len(), 0).unwrap();
    backend.synchronize().unwrap();
    backend.begin_capture().unwrap();
    kernels::elementwise::add_into(
        backend.context(),
        DType::F32,
        &buffers[0],
        &buffers[1],
        &output,
        output.len() / 4,
    )
    .unwrap();
    let graph = backend.end_capture().unwrap();
    graph.replay().unwrap();
    backend.synchronize().unwrap();
    let mut actual = vec![0; output.len()];
    output.copy_to_host(&mut actual).unwrap();
    assert!(actual
        .chunks_exact(4)
        .all(|x| f32::from_le_bytes(x.try_into().unwrap()) == 6.0));

    cache.clear().unwrap();
    assert_eq!(cache.seq_len(), 0);
    for (index, original) in buffers.iter().enumerate() {
        let current = if index % 2 == 0 {
            cache.k_buffer(index / 2)
        } else {
            cache.v_buffer(index / 2)
        };
        assert_eq!(
            address(current),
            address(original),
            "clear must preserve captured addresses"
        );
        let mut bytes = vec![7; current.len()];
        current.copy_to_host(&mut bytes).unwrap();
        assert!(bytes.iter().all(|&x| x == 0));
    }
    graph.replay().unwrap();
    backend.synchronize().unwrap();
    output.copy_to_host(&mut actual).unwrap();
    assert_eq!(actual, vec![0; output.len()]);
}

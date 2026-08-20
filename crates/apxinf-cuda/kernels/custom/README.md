# Custom CUDA operators

This directory contains device-side CUDA operator implementations only. Files
are classified by physical operation, not by executor, precision, quantization
scheme, or inference stage. Precision is expressed by a function suffix or a
template specialization (for example, `rms_norm_bf16_kernel` and
`rms_norm_quant_f16_e4m3_kernel`).

- `activation.cuh`: GELU, SiLU, GeGLU, and activation-plus-quantization.
- `attention.cuh`: masks, softmax, MHA/MQA, vision attention, and decode attention.
- `cache.cuh`: KV-cache writes.
- `elementwise.cuh`: add, multiply, scale, bias, concatenate, and Euler update.
- `embedding.cuh`: embedding lookups.
- `fused.cuh`: operators spanning multiple physical stages, such as
  residual-plus-normalization and QKV-split-plus-RoPE.
- `normalization.cuh`: RMSNorm, LayerNorm, and adaptive normalization.
- `preprocess.cuh`: image-to-patch conversion.
- `quantization.cuh`: FP8/INT8 quantization and dequantization.
- `rope.cuh`: 1-D, batched, multimodal, vision, and decode RoPE.
- `selection.cuh`: argmax and token selection.
- `math.cuh` and `reduction.cuh`: shared device helpers, not public operators.

The files intentionally contain no stable C ABI, vendor-library planning, or
host launch policy. The translation units under `../../adapters/` include these
headers and own `extern "C"` entry points plus CUDA `<<<...>>>` launch
configuration.

Historical aggregate headers such as `core_operators.cuh`, `static_bf16.cuh`,
`pointwise.cuh`, and precision-named top-level headers are forbidden.

## Third-party provenance

The physical reclassification itself only moves existing ApxInf code. Imported
implementations retain source attribution and are accepted only after a
correctness comparison and a representative-device benchmark.

`quantization.cuh` contains a packed-four FP16-to-E4M3 path adapted from an
upstream implementation at commit `a260835599e9406067196968cfe7a67bf8d13b4d`,
Copyright (c) 2025 Xingkai Yu,
under the MIT license. ApxInf adds an
alignment guard and scalar tail so the existing arbitrary-length C ABI remains
unchanged.

On NVIDIA Thor SM110, an isolated CUDA-event A/B benchmark (100 warmups, 1,000
iterations except 300 for 8M values) produced the following averages on
2026-08-12. The packed path was bit-for-bit equal to the scalar path for counts
`1,2,3,4,5,7,31,32,33,1023,1024,1025,591872`.

| FP16 values | Scalar | Packed-four | Speedup |
| ---: | ---: | ---: | ---: |
| 65,536 | 14.563 us | 6.987 us | 2.084x |
| 591,872 | 12.475 us | 8.642 us | 1.444x |
| 8,388,608 | 63.696 us | 35.057 us | 1.817x |

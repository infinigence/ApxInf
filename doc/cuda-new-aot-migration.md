# AOT-to-native operator migration

This note records where the seven translated CUDA operators belong in
`apxinf-cuda-new`. It follows the operator classification and L3-to-L0 layering
in `cuda-new-design.md`: public APIs describe mathematical semantics, while
model names and fixed production shapes stay inside candidate support checks.

| Translated operator | cuda-new family | Shape | Status in this change |
|---|---|---|---|
| NVFP4 GEMM + SwiGLU + requantization | GEMM | Fixed | L3/L2/ABI/L1/L0; used by Qwen3.8-27B |
| BF16 GEMM + SwiGLU | GEMM | GEMM family candidate | L0 specialization compiled on SM110 |
| BF16 gated-delta-net prefill | Linear attention | Family provider | L1 provider and L0 kernel compiled on SM110 |
| D256 causal attention | Attention | Attention family candidate | L0 specialization compiled on SM110 |
| D256 split-batch attention | Attention | Attention family candidate | L0 specialization compiled on SM110 |
| Strided-group attention, 624 tokens | Attention | One shape-gated candidate | Shared L0 specialization compiled on SM110 |
| Strided-group attention, 2200 tokens | Attention | Same candidate as 624 | Shared L0 specialization compiled on SM110 |

The NVFP4 operator is exposed as a fixed GEMM-family API because it has a
distinct binding contract: packed NVFP4 operands, two scale outputs, and three
calibration scalars. The implementation accepts only its validated SM110 shape;
the model keeps the ordinary GEMM plus SwiGLU fallback for other row counts.

The other six translations are deliberately not exposed through model-shaped
public APIs. Their native code lives in the appropriate GEMM, attention, or
linear-attention family and is compiled so source and shared CUTLASS changes
cannot silently rot. Promoting them to active candidates requires first adding
the missing family-level semantics (SwiGLU activation, split-batch bindings,
and strided groups) and their fallback contracts. Until that work is complete,
Qwen Drive continues to use its existing legacy AOT path unchanged.

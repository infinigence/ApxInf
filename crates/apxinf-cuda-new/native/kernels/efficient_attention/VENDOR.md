# Vendored memory-efficient attention forward kernel

Source: [PyTorch](https://github.com/pytorch/pytorch/tree/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/transformers/cuda/mem_eff_attention),
revision `70d99e998b4955e0049d13a98d77ae1b14db1f45`.
The headers are byte-for-byte upstream copies of the forward dependency closure.
The accompanying BSD-style LICENSE is retained; embedded CUTLASS-derived files
retain their own notices. They reuse the existing vendored CUTLASS headers.

ApxInf's wrapper lives in `adapters/attention/providers/efficient.cu`.
It instantiates aligned SM80 32×128 forward tiles for F16/BF16 head width 384,
with dropout disabled and no additive bias binding. The upstream bias-capable
instantiation is retained because it rounds score scaling before softmax,
matching the pinned reference even when its bias pointer is null. There is no PyTorch/libtorch dependency.
The sibling `efficient_compat` supplies only host argument-check exceptions and
the Philox carrier required to parse the unused upstream dropout branch. It is
not a framework runtime or an RNG implementation for this provider.

Compile with precise NVCC math defaults. Provider selection, allocation,
preparation and graph lifetime use the existing Attention execution framework;
no host configuration is performed by the captured launch.

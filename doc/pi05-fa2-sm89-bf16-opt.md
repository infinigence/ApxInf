# Pi0.5 RTX 4090 SM89 BF16 Split-KV Attention

## Summary

ApxInf now supports a BF16 FlashAttention-2 split-KV path for the Pi0.5
action decoder on RTX 4090 / SM89. The optimization keeps the same attention
math and only changes the FA2 kernel variant used for small-query, long-KV MQA
shapes.

The optimization targets the action-stage attention shape:

```text
Sq <= 64
Sk > Sq
num_q_heads > num_kv_heads
head_dim == 256
```

For the measured Pi0.5 H10 profile this corresponds to:

```text
Sq = 10
num_q_heads = 8
num_kv_heads = 1
head_dim = 256
```

Regular FA2 leaves SM occupancy low on this shape. Split-KV partitions the KV
axis, runs more parallel FA2 work, then combines the partial outputs. This adds
a combine kernel but reduces total attention time.

## Implementation

The implementation reuses the vendored FlashAttention-2 source already present
in ApxInf. It does not introduce a new attention algorithm.

Main changes:

- Added a BF16 split-KV C++ wrapper around
  `run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, false>`.
- Added the SM80-family hdim256 BF16 split-KV FA2 instantiation.
- Exposed the new entry through the stable C ABI and Rust FFI.
- Routed `attention::mqa_bf16` to split-KV only for the small-query GQA/MQA
  shape above.
- Added split-KV graph scratch only at BF16 CUDA runtime capture time on
  SM80-family devices. This keeps non-SM80-family workspace sizing unchanged.
- Updated the RTX4090 SM89 BF16 tactics file kernel build id.

The default path remains regular FA2 for larger attention shapes, including the
language-stage prefix self-attention. The split-KV path can be disabled for
debugging with:

```bash
APXINF_DISABLE_FA2_SPLITKV=1
```

## Correctness

End-to-end BF16 CUDA graph integrity still passes with bitwise equality between
eager execution and graph replay:

```text
eager_vs_graph.bitwise_equal = true
max_abs = 0
relative_l2 = 0
```

Regular FA2 and split-KV FA2 are not expected to be bitwise identical because
split-KV changes the softmax/reduction partitioning order. Operator-level
comparison against regular FA2 showed BF16-scale numerical differences:

```text
2-view-like shape, Sq=10, Sk=522:
max_abs     = 6.1035e-5
mean_abs    = 3.287e-6
cosine      = 0.99999748
relative_l2 = 0.002245

3-view-like shape, Sq=10, Sk=788:
max_abs     = 3.0518e-5
mean_abs    = 2.855e-6
cosine      = 0.99999729
relative_l2 = 0.002327
```

These differences are expected for an equivalent BF16 FA2 split-KV execution
path.

## Performance

Measurements were taken on RTX 4090 / SM89 with Pi0.5 BF16, H=10, token=10,
10 flow steps, graph replay timing, and NHWC RGB input inside the captured
graph.

Using the retuned RTX4090 BF16 tactics file:

```text
2-view H10 token=10:
regular FA2   P50 = 35.35 ms
split-KV FA2  P50 = 31.38 ms
improvement       = 3.97 ms

3-view H10 token=10:
regular FA2   P50 = 47.02 ms
split-KV FA2  P50 = 40.93 ms
improvement       = 6.10 ms
```

The same runs passed the built-in BF16 eager-vs-graph integrity check:

```text
2-view split-KV: eager_vs_graph.bitwise_equal = true, max_abs = 0
3-view split-KV: eager_vs_graph.bitwise_equal = true, max_abs = 0
```

Workspace usage with split-KV enabled:

```text
2-view H10 token=10: capacity = 3,615,987,712 bytes, used = 1,511,330,688 bytes
3-view H10 token=10: capacity = 5,311,271,936 bytes, used = 3,089,184,640 bytes
```

Without the tactics file, the same trend was observed:

```text
2-view H10 token=10:
regular FA2   P50 = 37.28 ms
split-KV FA2  P50 = 33.35 ms

3-view H10 token=10:
regular FA2   P50 = 50.45 ms
split-KV FA2  P50 = 44.12 ms
```

## Platform Notes

This optimization is currently wired for the SM80-family FA2 build path, which
includes SM80, SM86, SM87, and SM89. RTX 4090 uses SM89.

Thor / SM100-family behavior is isolated:

- the Rust FFI declaration and `attention::mqa_bf16` dispatch are compiled only
  under `apxinf_fa2_sm80`;
- the new BF16 split-KV FA2 instantiation is added to the CUDA build only when
  the selected NVCC architecture is SM80-family;
- the C adapter export and C++ split-KV wrapper are guarded with
  `APXINF_FA2_SM80`, so SM100 builds do not require the split-KV BF16 symbol;
- the extra split-KV graph scratch is added only when runtime device caps report
  `CudaArchFamily::Sm80`;
- only `configs/pi05/rtx4090_sm89_bf16_v2_v3_h10_tactics.json` was retuned.
  Thor tactics files were not changed.

The same optimization principle applies to other platforms: whenever the Pi0.5
action decoder has a small query length and long KV cache, a split-KV attention
backend can improve occupancy without changing model semantics.

Platform-specific enablement requires:

- a compiled FA2 split-KV instantiation for the platform and head dimension;
- a wrapper that exposes split-KV scratch pointers;
- runtime workspace allocation for `softmax_lse_accum` and `o_accum`;
- a dispatch guard that only selects split-KV for shapes where it wins;
- numerical comparison against the platform's regular attention path.

For future non-SM89 targets, use the same validation pattern:

1. Compare regular attention vs split-KV attention on the exact action-stage
   shapes.
2. Check operator-level numerical error.
3. Check end-to-end eager-vs-graph integrity.
4. Benchmark 2-view and 3-view graph replay latency with the target platform's
   tuned GEMM tactics.

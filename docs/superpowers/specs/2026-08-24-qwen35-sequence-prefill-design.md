# Qwen3.5 Sequence Prefill Design

Date: 2026-08-24
Scope: midterm base correctness only
Model revision: `63768c10df38c0395e12ef49edac1bd539eaeeea`
Target GPU: `GPU-343bc895-b011-22fa-4449-97207aa2bdec`

## Goal

Replace prompt processing that repeatedly invokes the single-token decode path
with a true CUDA sequence-prefill path matching the Transformers Qwen3.5
semantics. Prefill is bounded to contiguous blocks of at most 64 prompt tokens
so full-attention score workspace does not scale as the square of the complete
prompt. Single-token generation remains on the recurrent decode path. The
implementation must preserve request-local state isolation and transactional
failure handling, and it must be checked against the approved oracle at
intermediate states rather than only final token IDs.

## Data flow

`Qwen35CudaSession::open` validates the non-empty prompt and positive generation
budget, checks `prompt_len + max_new_tokens <= max_model_len`, and allocates each
full-attention KV cache with the request-local capacity
`prompt_len + max_new_tokens`. It then partitions the prompt into contiguous
`[start, end)` ranges of at most 64 tokens. For each range, the session embeds
only that block into a `[B, hidden]` BF16 tensor and passes the block through all
64 layers in model order before embedding or executing the next block.

Each GDN layer performs batched input projections, causal depthwise convolution
starting from the ring state left by preceding blocks, BF16 q/k normalization,
sequence/chunk gated-delta recurrence starting from the retained FP32 state,
gated RMSNorm, output projection, and MLP/residuals. Each full-attention layer
performs batched q/k/v projections, q/k norm, partial RoPE using the block
start's absolute `position`, appends the block K/V rows to the retained
request-local cache, and applies causal SDPA over all preceding K/V plus the
causally visible prefix of the current block. GDN convolution/recurrent state
and full-attention KV state therefore carry across block boundaries.

Intermediate block hidden tensors are not retained for logits. After the final
block finishes, the session selects only that block's last row, advances the
session position to the complete prompt length `T`, and computes the first
pending token from final norm and the LM head. Subsequent `next_token` calls
execute exactly one decode row.

## State and failure semantics

Every request starts with zeroed GDN convolution/recurrent state and empty
full-attention KV caches. At the beginning of block `start..end`, each GDN
position and full-attention logical KV length must equal `start`. A successful
layer advances its request-local state by `B = end - start`, and a successful
complete prefill leaves every GDN position and full-attention KV length equal to
`T`.

The GDN convolution and recurrent primitives write through scratch state,
check device finite-status, and keep rollback handles for the most recent block
commit. If a later projection, norm, MLP, residual, or finite check in the same
GDN layer fails, that layer rolls back both current-block commits. Full
attention rejects position/capacity mismatches before launch and advances its
logical KV length only after attention, gating, projections, MLP/residuals, and
the final finite check succeed. If any layer or any later block fails,
`Qwen35CudaSession::open` returns an error and discards the entire partially
mutated session; no incomplete block state is exposed for decode or reuse.
Rollback failure is reported together with the original error. A new request
allocates fresh zero state, and no state or output is shared between requests.

## Interfaces

- `PREFILL_CHUNK_TOKENS` is 64, and `prefill_ranges(prompt_len)` yields ordered,
  gap-free blocks whose length is in `1..=64`.
- Request capacity is checked as `prompt_len + max_new_tokens`; that exact value,
  rather than `max_model_len`, is passed to each request-local full-attention
  state allocation.
- `Qwen35GdnState::causal_conv_silu_prefill` consumes
  `[B, conv_channels]` and returns `[B, conv_channels]`, using the prior ring and
  committing `B` ring updates after finite-status succeeds.
- `Qwen35GdnState::gated_delta_prefill` consumes the current block's q/k/v plus
  per-row gates, uses the prior FP32 recurrent matrix as its initial state, and
  commits one final FP32 state while returning `[B, value_width]` BF16 output.
- `Qwen35CudaGdnLayer::prefill` and
  `Qwen35CudaFullAttentionLayer::prefill` expose block-level operations over
  `[B, hidden]`. Full-attention prefill additionally receives the absolute
  block-start `position`, requires it to equal logical KV length, and attends
  through `position + B`.
- `Qwen35CudaSession::prefill` owns block planning, per-block embedding, the
  complete 64-layer order, cross-block state continuity, and final-block
  last-row selection.
- `request_state_bytes(config, max_model_len)` is a conservative admission
  estimate. It includes all layers' KV storage at `max_model_len`, three copies
  (current, scratch, and backup) of every GDN convolution ring and FP32
  recurrent matrix, peak GDN prefill workspace, and the two score-sized BF16
  buffers that coexist during attention scaling/softmax. The score workspace is
  bounded as
  `2 * full_attention_heads * min(64, max_model_len) * max_model_len` bytes.

## Verification

Tests are written before each production change. CUDA tests cover convolution
zero-left padding at the first block, ring and recurrent continuity across the
64-token boundary, BF16 SiLU, per-block recurrence outputs, final FP32 state,
q/k normalization and beta boundaries, same-layer rollback, and failed-session
isolation. Full-attention tests cover absolute RoPE positions at non-zero block
starts, causal visibility of prior KV and the current block prefix, logical KV
length, request-local capacity, and failure before logical cache advancement.

Runtime tests cover exact ranges for prompts longer than 64 tokens, completion
of all 64 layers before the next block, final-row selection from only the last
block, final GDN position/KV length `T`, single-row decode, and
`prompt_len + max_new_tokens` capacity. Memory-accounting tests assert three
GDN state copies and a score workspace bounded by 64 query rows rather than a
`T x T` full-prompt allocation. The approved oracle bundle is used for
embedding, selected block-boundary hidden states, GDN state, full-attention KV,
logits, and output tokens. A long-prompt GPU run must demonstrate that attention
score workspace remains bounded and does not reproduce the full-prompt
workspace OOM. Protocol and evaluator gates are rerun only against the actual
candidate build; unchecked plan steps are not evidence of a pass.

## Non-goals

No CPU, Transformers, vLLM, alternate model/GPU, multimodal path, long-context
bonus, batching, MTP, KV quantization, graph capture, or performance tuning is
introduced.

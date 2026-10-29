# Native Qwen3 on MLX

This Metal-specific module owns checkpoint interpretation, forward computation and
request KV state. It calls safe `apxinf-mlx` array operations, without Python or
an external model provider. Initial target: Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`, original BF16 weights, batch one,
Apple Silicon macOS.

Checkpoint loading admits only the 0.6B geometry above. The request limit is
40960 total tokens, including the prompt and reserved output. Other geometry,
RoPE scaling and contexts beyond the checkpoint limit are rejected.

`config.rs` reads the explicit `head_dim` (128 for this checkpoint), independently
of `hidden_size / num_attention_heads` (64). Sliding-window attention, RoPE
scaling, projection biases and alternative activations are rejected. `weights.rs`
checks every key, dtype and shape; maps HF `[out,in]` matrices to canonical
`[in,out]` private views; preserves tied embedding storage; and prepares norm
weights once. `bf16-compiled` retains checkpoint BF16 norm weights.
`bf16-public` and `mixed-w8` materialize FP32 norm weights.
Unrecognized checkpoint tensors fail loading.

The BF16 selections are `bf16-public` and `bf16-compiled` (default).
`bf16-public` computes norm statistics and weight multiplication in FP32 before
the BF16 cast. `bf16-compiled` uses native weighted RMSNorm with BF16 activations
and prepared BF16 weights for input, post-attention, Q, K, and final norms,
including prefill. This changes normalization rounding. The text-fixed check
differs at 222 of 1024 tokens from b2d58ea; external-teacher agreement is 221/222,
with neither language below its reference. Five clean 1K blocks improve total
time against the ordered implementation. The full MLX matrix remains pending.
Both selections retain BF16 cos/sin tables and the two separate BF16 RoPE
multiplication results. The compiled selection prepares local norm/RoPE functions
and the complete B1/L1 decoder block with explicit input/output KV arrays.
After prefix state exists, that block uses `Bf16QkTableRope` to rotate Q and K
in one Metal call. The backend owns shape validation and separately rounded
BF16 products and sums. The block captures the immutable kernel, while Q, K,
and table rows remain explicit array operands. Its 18-input/3-output ABI is
unchanged. Prefill and the fresh first-token path retain local compiled RoPE.
Prefill uses the same layer composition with compiled local seams. Whole
decode-step compilation is not implemented or claimed by this selection.

The private norm weight dtype selects the arithmetic. BF16 weights require BF16
activations and select native weighted RMSNorm. FP32 weights select the ordered
FP32 expression. The loader prepares one representation for every norm site,
including the final norm, with no inference-time weight cast or alternate copy.
The existing tiny public/compiled trajectory assertions remain unchanged.
Historical qualification below predates this arithmetic change.

`mixed-w8` uses affine 8-bit quantization with group size 64: W8 Q/K/V/O,
fused gate/up and down projections only
at one input row, with BF16 projections for larger prompts; a single W8 tied
table used for both embedding gather/dequantization and output projection;
compiled local norm/RoPE functions; and the fixed TG256 Q/K norm-to-RoPE Metal
fusion for B1/L1, 16 query heads, 8 key heads and width 128. The BF16 tied table
and separate gate/up halves are released after packing. Norm weights stay FP32.
BF16 prefill projections remain resident alongside their W8 decode copies.
This is a decode-performance selection, not a guaranteed memory-saving mode;
its resident weight memory can exceed BF16.
This variant does not select the BF16 decoder-block callable: its inherited
compilation boundary and valid-length attention remain explicit.

`Qwen3Model::prepare(prompt_len, max_new_tokens)` allocates a bounded KV profile
and warms its sequence shapes before inference. Capacity rounds up to 256 tokens
within the smaller checkpoint/40960-token limit. A new request can grow or
shrink this allocation after reset. RoPE tables use the same capacity.
Compiled decode slices attention to the next 256-token boundary of the active
prefix and masks unwritten positions. Requests with capacity at most 2048 keep
their original attention shape. Preparation warms every admitted one-row
attention bucket, including bucket crossings in direct `forward` calls.
`LlmTrait::prewarm_decode` invokes this preparation for the shared generation
driver; failures invalidate the next fallible operation. Direct callers must
prepare every sequence length they use. `reset` returns to
the retained immutable zero state, keeping compiled resources reusable.

Forward validates token range, position and capacity before computation. It
evaluates the full `[seq_len,vocab_size]` output and every new KV array before
publishing the state and advancing once. A computation failure invalidates the
session until reset. Outputs own immutable array handles; a later forward/reset
does not overwrite earlier results.

Generation `prefill` follows the source schedule: consume all but the last token
in chunks of at most 2048 for `bf16-compiled`, evaluating only updated KV;
then consume the last
token and return `[1,vocab_size]`. The chunk head is never evaluated. The shared
generation driver uses this hook, while `forward` continues
to return every row. All prompt chunk shapes are warmed by `prepare`.
The experimental `APXINF_QWEN3_PREFILL_CHUNK` override accepts 512, 1024 or
2048 and is captured at construction. Public and MixedW8 default to 512;
compiled BF16 defaults to 2048. A family-local
plan supplies the same prefix ranges to preparation and execution. Existing
prepared-row admission for public `forward` remains unchanged.

Long generation prepares the selected chunks, tail and final row. It does not prepare
a full prompt-sized `forward` above 2048 rows; that call fails explicitly.
This avoids allocating a large `[prompt,vocabulary]` result during generation.

On a 16 GiB host, the BF16 KV budget is
`2 * 28 layers * 8 KV heads * 128 channels * 2 bytes = 112 KiB/token`.
At 40960 tokens the live cache is 4.375 GiB, both RoPE tables total 20 MiB,
and one shared immutable reset buffer is 80 MiB. These are tensor sizes,
excluding weights, allocator cache and temporary arrays. Request execution
releases old state handles before evaluation so native buffer reuse is possible.
The qualification runs enforce a 12 GiB physical process limit, leaving room
for macOS and the other jobs; the context limit is not a reservation of that RAM.

Load through `AutoModel` with `model_name=metal_qwen3` (or checkpoint detection),
`Device::Metal(0)`, BF16, and the desired `model_variant`. The native CLI uses:

```sh
apxinf generate --model /path/to/Qwen3-0.6B --device metal --dtype bf16 \
  --model-variant bf16-compiled --greedy \
  --chat-options '{"enable_thinking":false}' --prompt 'Explain gravity briefly.'
```

Sampling capability is supplied by the MLX backend. Currently only greedy
selection without penalties/logprob is implemented; Qwen's random-sampling
generation defaults must be overridden explicitly. Unsupported options fail.
The compiled BF16 variant uses device greedy IDs with one-step async lookahead.
Set `APXINF_MLX_SYNC_DECODE=1` to use the synchronous generation path. Public
`forward` remains synchronous. EOS and cancellation drain the extra step and
restore the position of the published prefix; async failures invalidate state.
First preparation/compilation must be reported separately from warmed latency.
Preparation uses `Compiled::prepare` to evaluate each supported callable twice
and require no tracing on the second invocation. The presence of
`MLX_DISABLE_COMPILE`, including value `0`, rejects callable construction.
This proves reuse of the Rust tracing path, not absence of all native shader JIT.

Source attribution is retained in the repository [NOTICE](../../../../NOTICE)
and [LICENSE.engine-tailor](LICENSE.engine-tailor).

Metal lifecycle/parity tests are explicit opt-in tests:

```sh
cargo test -p apxinf-model --features mlx metal_qwen3::model::tests -- --ignored --test-threads=1
```

These small synthetic tests verify API/state behavior; they do not replace the
frozen real-input, reference-trajectory and complete-answer qualification.

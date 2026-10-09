# Native Qwen3 on MLX

This Metal-specific module owns checkpoint interpretation, ordered forward computation and
request KV state. It calls safe `apxinf-mlx` array operations, without Python or
an external model provider. Initial target: Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`, original BF16 weights, batch one,
Apple Silicon macOS.

Checkpoint loading admits only the 0.6B geometry above and caps the request
profile at 2048 tokens, even though the checkpoint advertises 40960. Other
geometry and larger contexts require a separately verified migration profile.

`config.rs` reads the explicit `head_dim` (128 for this checkpoint), independently
of `hidden_size / num_attention_heads` (64). Sliding-window attention, RoPE
scaling, projection biases and alternative activations are rejected. `weights.rs`
checks every key, dtype and shape; maps HF `[out,in]` matrices to canonical
`[in,out]` private views; preserves tied embedding storage; and caches FP32 norm
weights. Unrecognized checkpoint tensors fail loading.

The BF16 selections are `bf16-public` and `bf16-compiled` (default).
Both preserve FP32 norm statistics and weight multiplication before the BF16
cast, BF16 cos/sin tables, and the two separate BF16 RoPE multiplication results.
The compiled selection additionally prepares local norm/RoPE functions and the
complete B1/L1 decoder block with explicit input/output KV arrays. Prefill uses
the same ordered layer composition with compiled local seams. Whole decode-step
compilation is not implemented or claimed by this selection.

`mixed-w8` uses affine 8-bit quantization with group size 64: W8 Q/K/V/O,
fused gate/up and down projections only
at one input row, with BF16 projections for larger prompts; a single W8 tied
table used for both embedding gather/dequantization and output projection;
compiled local norm/RoPE functions; and the fixed TG256 Q/K norm-to-RoPE Metal
fusion for B1/L1, 16 query heads, 8 key heads and width 128. The BF16 tied table
and separate gate/up halves are released after packing. Norm weights stay FP32.
This variant does not select the BF16 decoder-block callable: its inherited
compilation boundary and valid-length attention remain explicit.

`Qwen3Model::prepare(prompt_len, max_new_tokens)` allocates a bounded KV profile
and warms its sequence shapes before inference. Capacity rounds up to 256 tokens
within the smaller checkpoint/2048-token profile limit. Growing capacity
requires a fresh request.
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
in chunks of at most 512, evaluating only updated KV; then consume the last
token and return `[1,vocab_size]`. The chunk head is never evaluated. The shared
generation driver uses this hook, while `forward` continues
to return every row. All prompt chunk shapes are warmed by `prepare`.

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

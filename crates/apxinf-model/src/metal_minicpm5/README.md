# MiniCPM5 on native MLX

This Metal-specific module owns the MiniCPM5-2B architecture and request KV state. It loads
through `AutoModel` with `model_name = "metal_minicpm5"` and `Device::Metal(0)`.
The official checkpoint identifies itself as `llama`; explicit family selection
prevents another Llama checkpoint from being admitted accidentally.

The supported checkpoint is `openbmb/MiniCPM5-2B` revision
`f97400052a43d642bbc6e9975e2397e3ae6a6b52`. The DSpark selection requires the
official BF16 drafter revision `114a20fdbf53220712c7fbdd7dccddbf1dedebb4`;
its configuration and weights are checked before loading.

`config.rs` checks the official 42-layer, width-2048, 16/2-head, head-width-128
geometry. `weights.rs` consumes every BF16 tensor explicitly. Linear operands
have canonical logical `[in, out]` dimensions while private MLX views retain
the checkpoint's physical backing and the reference's projection selection.
The embedding and output head are independent.

`math.rs` preserves the checkpoint's normalization rounding: the FP32-normalized
value is cast to BF16 before multiplication by the BF16 affine weight. NeoX RoPE
retains the separate BF16 products and sum. Public SwiGLU remains compiled as in
the source runtime, including the `bf16-public` selection.

Compiled prefill uses native weighted RMSNorm with the existing BF16 weights.
The pinned [MLX 0.32.3 Metal kernel](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/rms_norm.metal)
rounds the normalized value to BF16 before multiplying the BF16 weight.
Public and DSpark keep the ordered expression. Compiled decode keeps its
packed residual/norm. Short-token, external-teacher and eight long logits/state
checks match the unweighted path. Five clean 1K blocks give a median total-time
ratio of 0.9961 against that path; the complete MLX matrix remains pending.

`model.rs` owns the forward order, per-request position/KV, precomputed position
tables, and pure compiled decode function. `bf16-public` is the ordered reference
composition. `bf16-compiled` selects local norm/RoPE compilation and whole-step
decode with the fixed packed residual/norm kernel. Every new KV array and output
is evaluated before the request state is published; failed preparation cannot
be reported ready. Changing prepared capacity requires a reset.

Initial allocation gives all 84 K/V slots one shared immutable BF16 zero array.
This applies to Public, Compiled and DSpark. Each layer's functional update
returns its own K/V values; compiled inputs and outputs keep separate slots.
Reset retains the current KV and restores only the logical position. It does
not restore the initial zero array.

`LlmTrait::forward` always returns `[input_tokens, vocabulary]` logits.
Generation `prefill` consumes prompts up to 4096 tokens in the original shape.
Longer prompts use 1024-token chunks in compiled mode and 512 in public mode,
with a causal mask that includes the prior cache and the current chunk. Compiled non-final chunks evaluate all KV outputs
after dropping unused logits. The final chunk evaluates its final-row logits.
Public mode evaluates final-row logits on every chunk; both modes return only
the final chunk's result. Long preparation advertises these
chunk shapes, and rejects a full prompt-sized `forward`. Every admitted
`forward` still returns all of its input rows. Ordinary generation uses
ApxInf's shared sampler/EOS loop.

The experimental `APXINF_MINICPM5_PREFILL_CHUNK` override accepts 512, 1024 or
2048 and is captured at construction. It applies only
above 4096 prompt tokens. Short prompts and DSpark retain their original shape.
The prepared request owns one prefill plan whose ranges determine warm shapes,
position/row admission and execution. Chunk-size changes can alter BF16 rounding
and require separate numerical qualification.

Ordinary BF16 generation supports one sequence with at most 65792 total tokens,
including prompt and reserved output, within the checkpoint's 131072 positions.
This admits a 65536-token prompt with up to 256 reserved output tokens.
KV and RoPE capacity round the request up to 256 tokens. A reset request can
grow or shrink that allocation. Compiled decode attends to the active prefix
rounded to 256 tokens, with unwritten positions masked. Capacity at most 4096
retains the original attention shape. The actual decode buckets are prepared
before inference; crossing a bucket does not retrace the compiled step.

Ordinary compiled requests use bucketed attention by default. The experimental
construction-time override `APXINF_MINICPM5_EXACT_ATTN=1` selects exact KV views
without a mask for one-row decode. All other values retain bucketed attention.
Public and DSpark always keep their existing path.
Preparation owns the prompt, output budget, resolved attention mode and width
profiles together; changing exact geometry or attention mode replaces the
compiled graph. The first and last widths are evaluated during preparation;
intermediate widths receive graph-only preparation. This verifies Rust trace
reuse, not every Metal pipeline's warm state. Cold-request measurements include
remaining device compilation and allocation. Direct forward calls beyond the
prepared exact prompt/output profile are rejected. Exact attention changes BF16
rounding. The recorded short-input teacher suite does not exercise the long
exact-attention path. Long real-input numerical and semantic qualification is
incomplete, so exact attention remains opt-in. Historical timing and trajectory
observations do not establish default-on acceptance.

On a 16 GiB host, BF16 KV uses
`2 * 42 layers * 2 KV heads * 128 channels * 2 bytes = 42 KiB/token`.
At 65536 tokens the live cache is 2.625 GiB. The extra 256-token capacity bucket
adds 10.5 MiB, allowing the 64K-input context-sweep case to reserve output.
At the 65792-token limit, the two RoPE tables total 32.125 MiB.
Weights, allocator cache and temporary arrays are additional.
Execution releases prior state handles before evaluation; qualification runs
enforce a 12 GiB physical process limit. The full checkpoint context is not
admitted by this memory budget.

For 64K on the 16 GiB M4, the process must cap the MLX allocator cache at
2 GiB (2147483648 bytes). With the default allocator cache policy, the initial
65408-input/128-output diagnostic run exceeded the 12 GiB physical-process guard.
The explicit 2 GiB cache budget completed that request in the diagnostic worker.
Applications embedding the family must set their own process-wide MLX cache
policy; the model does not silently change it for other resident models.

DSpark retains its separate 4096-total-token limit and 1–256 output-token
budget. Requests beyond these limits fail with an explicit DSpark error.
Only greedy sampling without penalties/log-probability is currently implemented
by the native backend. Unsupported sampling requests fail explicitly. DSpark is
an explicit selection with a separately pinned drafter; it is never the default.

Ordinary compiled BF16 generation uses device greedy IDs and one-step async
lookahead. `APXINF_MLX_SYNC_DECODE=1` selects the synchronous fallback; public
`forward` remains synchronous. EOS/cancellation drain and trim unused work.
`LoadedModel::generate_streaming_cancellable` returns the published prefix;
the new cancellation entry is unsupported for DSpark. Async failures invalidate
state and require reset/preparation.

Source adaptations and MIT notices are recorded in the repository `NOTICE`.
Select the target and drafter explicitly through the public CLI:

```sh
apxinf generate --model /path/to/MiniCPM5-2B --model-name metal_minicpm5 \
  --device metal --dtype bf16 --model-variant dspark \
  --asset draft=/path/to/official-DSpark --greedy --max-tokens 256 \
  --chat-options '{"enable_thinking":false}' --prompt 'Explain binary search.'
```

Use `--model-variant bf16-compiled` without a draft asset for ordinary decoding.
Tool definitions can be supplied in `--chat-options`; the CLI preserves the
special `<function>` and `<param>` tags in generated tool calls.

Request admission is read-only: unsupported bounds or sampling settings return
an error without poisoning later generation or `forward` calls. A device/KV
execution error invalidates the instance until an explicit `reset`.

The ignored `rejected_requests_do_not_poison_generation_or_forward` regression
loads the official checkpoints once, rejects several invalid requests, and
checks subsequent valid generation on the same instance without caller resets.
Set `APXINF_MINICPM_CHECKPOINT`, `APXINF_DSPARK_CHECKPOINT` and
`APXINF_DSPARK_REPLAY_INPUT` (the existing real-prompt replay JSON schema), then
run under the host's shared Metal lock:

```sh
cargo test -p apxinf-model --features mlx --release \
  rejected_requests_do_not_poison_generation_or_forward -- --ignored --test-threads=1
```

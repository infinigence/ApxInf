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

`model.rs` owns the forward order, per-request position/KV, precomputed position
tables, and pure compiled decode function. `bf16-public` is the ordered reference
composition. `bf16-compiled` selects local norm/RoPE compilation and whole-step
decode with the fixed packed residual/norm kernel. Every new KV array and output
is evaluated before the request state is published; failed preparation cannot
be reported ready. Changing prepared capacity requires a reset.

`LlmTrait::forward` always returns `[input_tokens, vocabulary]` logits.
Generation `prefill` consumes the complete prompt while applying the untied head
only to its final normalized hidden row. This distinction retains the accepted
prefill optimization without weakening the full forward contract. Ordinary
generation uses ApxInf's shared sampler/EOS loop.

The supported workload is one text sequence with at most 4096 consumed positions.
Only greedy sampling without penalties/log-probability is currently implemented
by the native backend. Unsupported sampling requests fail explicitly. DSpark is
an explicit selection with a separately pinned drafter; it is never the default.

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

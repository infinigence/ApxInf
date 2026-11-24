# metal_lfm3 — LFM3-1.7B on the native MLX path

`metal_lfm3` is the Metal (MLX) port of **LiquidAI/LFM3-1.7B**. It is registered
under `metal_lfm3`; a checkpoint whose `config.json` says `model_type: lfm3`
resolves to it automatically through `AutoModel` (`metal_registry_name`), and
`--model-name metal_lfm3` selects it explicitly.

This family has no external-engine or Python execution path: the forward pass is
native MLX composition in `model.rs`, and it is not a copy of any other family's
code.

## Checkpoint and supported contract

| Item | Value |
| --- | --- |
| Checkpoint | `LiquidAI/LFM3-1.7B` BF16 safetensors (286 tensors, 3.45 GB) |
| Precision / variant | BF16 only; `auto` and `bf16-eager` (the only implemented variant) |
| Config pins | hidden 2048, intermediate 7680, 24 layers (13 `conv` / 6 `full_attention` / 5 `sliding_attention`), 16 query / 4 KV heads of 128, vocab 128000, norm eps 1e-5, conv width 7, sliding window 256, tied embeddings, sandwich norms |
| Rotary | two schemes: full theta 1e6 with partial factor 0.5 (64 rotated dims), sliding theta 1e4 (128 rotated dims); zero tail, FP32 tables |
| Context | **8192 positions** total (`MAX_CONTEXT`). The checkpoint declares 262144, but full-attention layers keep an unbounded dense KV row (12 KiB/token over six layers), so this port admits a smaller envelope instead of promising the checkpoint bound. |
| Prompt protocol | lane ChatML — see below |
| Serving | OpenAI-compatible gateway through `--provider native` |

Anything outside that envelope is rejected at load with an explicit error, not
silently degraded: non-BF16 weights, `conv_bias: true`, `use_sandwich_norm: false`,
untied embeddings, a non-`default` rope type, an unknown layer type, unknown
variants, CUDA calibration/tuning options, FP8 and synthetic weights.

## Execution

Per layer, with `use_sandwich_norm`:

```
n = operator_norm(x)
op = attention(n) | short_conv(n)          # by config.layer_types[i]
op = post_operator_norm(op); x = x + op
f  = post_ffn_norm(feed_forward(ffn_norm(x))); x = x + f
```

- Attention: per-head RMSNorm over 128, FP32 zero-tail RoPE, causal SDPA (full)
  or 256-token window (sliding, decomposed into query blocks so no dense `T x T`
  mask is materialised), a per-head `sigmoid(gate_proj(n))` output gate, then
  `o_proj`.
- Short conv: `in_proj -> chunk(B, C, x) -> u = B * x -> width-7 causal depthwise
  conv in FP32 -> y = C * conv -> out_proj`. The six retained input columns are
  request state, not output state.
- MLP: `silu(w1) * w3 -> w2`.
- Head: `embedding_norm` then the tied embedding used as `lm_head`.

Prefill evaluates every layer boundary; decode builds one graph per token and
evaluates once. Request state is dense KV rows (`[1, 4, capacity, 128]` per
attention layer; sliding layers read only their last 256 keys) plus
`[1, 2048, 6]` conv history per conv layer. `preparation_status` reports
`bf16-eager` with **no** compiled or captured scope: this family composes eager
device primitives only, and claiming otherwise would be false.

## Prompt protocol: lane ChatML, and why the checkpoint template is not used

The verified protocol (the one the frozen corpus and the registered gate were run
with) is a single-turn ChatML prompt whose assistant prefix ends immediately after
`<|im_start|>assistant\n`; the model then emits `<think>` itself as its first
generated token. `metal_lfm3::lane_chat_prompt` renders it in the family, for
serving and for any caller that needs the same bytes.

The checkpoint's `chat_template.jinja` is deliberately **not** used. Even with
`enable_thinking=false` it appends `<think>` plus the end-of-thinking token to the
assistant prefix, i.e. the Hugging Face thinking-prefill protocol. That is a
different protocol: it was never frozen against our corpus, and it would need its
own reference capture, frozen answers and gate. Do not enable it by routing LFM3
through the checkpoint chat template or by adding the thinking prefix to the
renderer; it is a separate experiment.

## Verification

Reference: the already-verified Python/MLX lane (`enginetailor/models/lfm3`), used
read-only. Private evidence lives under `devlocal/lfm3-port/` (not committed).

- **Config/inventory**: all 286 tensors mapped, unconsumed tensors are an error,
  tied-head bytes verified, unit tests cover the rejected configurations.
- **Rotary tables**: match the lane's captured tables (the lane's numpy float32
  `sin`/`cos` path) to within 1 ULP; the port generates them with the platform
  `sinf`/`cosf` in FP32, which is what the reference's float32 table path calls.
- **Prefill parity**: with the lane's exact rotary tables the whole 24-layer
  forward pass is **bit-exact** against the captured reference on 16-, 154- and
  1024-token prompts (`max |Δ| = 0` per layer and on the full last-row logits;
  argmax identical at every position up to exact-tie index ties in the capture).
  With the port's own tables the residual is ≤1 ULP of table difference:
  last-row `max |Δ|` 0.27/0.17/0.19 on those three prompts, argmax 16/16,
  153/154 and 1018/1024 (the misses are exact ties and one 1-ULP flip).
- **Decode parity**: teacher-forced top-1 on the frozen real-question corpus
  (q01..q08, 731 comparisons of the lane's frozen oracle trajectory) = **720/731
  = 98.50 %** against a 98 % bar; driving the same corpus with the lane's exact
  rotary tables gives the same 720/731, and the lane's own bf16 engine scores
  98.76 % on the same protocol (the trajectory is the float32 oracle's, so ~98.8 %
  is the ceiling for any faithful BF16 engine). A deliberately broken conv-state
  path drops it to 60.19 %, so the check convicts a broken state path.
- **Serving**: the OpenAI-compatible gateway answers `/v1/chat/completions` with
  `prompt_tokens: 16` for the frozen question q01 — byte-identical to the frozen
  corpus prompt ids — and returns the frozen answer's final text.

## Serving

```bash
MLX_ROOT=<mlx-0.32.3 sdk>            # include/ and lib/
DYLD_LIBRARY_PATH="$MLX_ROOT/lib"    # libmlx.dylib must be resolvable

apxinf-serve \
  --model <LFM3 snapshot directory> --model-id LFM3-1.7B \
  --provider native --model-name metal_lfm3 \
  --native-worker target/release/apxinf-native-worker \
  --metal-library "$MLX_ROOT/lib/libmlx.dylib" \
  --max-context 4096 --max-output-tokens 256

curl -sS -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"LFM3-1.7B","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":120,"temperature":0}'
# -> {"choices":[{"message":{"content":"The capital of France is **Paris**."},
#     "finish_reason":"stop"}],"usage":{"prompt_tokens":16,...}}
```

`--provider native` is the family-parameterized native text provider
(`apxinf-worker/2.8`, adapter `apxinf-native-text-v1`). It reuses the same worker
binary, identity manifest and verification path as the other native providers;
its execution identity binds the registry name, variant and prompt protocol, and
the model snapshot must be a directory of regular files (the identity manifest
hashes every artifact).

## Known limits and follow-ups

- Context is capped at 8192 positions (see above); long-context work would need
  a bounded or paged full-attention KV before raising it.
- No compiled or captured decode scope yet; `preparation_status` says so.
- The port uses the ordinary lazy MLX dispatch path for every operation; the
  Python lane's ANE-composed prefill and 4-bit packed decode are not ported.
- Rotary tables follow the platform C library, which matches the reference
  lane's float32 numpy path to within 1 ULP rather than bit-for-bit. A
  bit-exact table would require reproducing numpy's float32 SIMD trig.
- The Hugging Face thinking-prefill protocol is not implemented (see above).

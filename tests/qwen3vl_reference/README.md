# qwen3vl reference dumps

HuggingFace-transformers reference activations + greedy token IDs used by
ApxInf unit tests as the bit-exact/bf16-tolerance correctness gate.

Regenerate with:
```
python scripts/hf_reference_dump.py
```
Requires `transformers`, `torch`, `accelerate`, `pillow`, and a CUDA GPU.

## Files

| File | Prompt | Notes |
|------|--------|-------|
| `tinyllama_text.npz` | "The capital of Canada is" | Sanity baseline: fp32 model in bf16 dtype. |
| `qwen3vl_text.npz`   | "The capital of Canada is" (chat-templated) | Text-only path through Qwen3-VL. |
| `qwen3vl_image.npz`  | Synthetic 336×336 image + "Describe this image in one short sentence." | End-to-end vision + text. |

## Keys per file

Every file:
- `tokens[int64, seq]` — input token IDs (already chat-templated where applicable).
- `post_embedding_last[f32, hidden]` — post-`embed_tokens` at the final position.
- `hidden_L{i}_last[f32, hidden]` — per-layer hidden state (last position) at `i` = 0, mid, last.
- `layer_indexes[int64, 3]` — the three layer indexes captured, sorted.
- `post_final_norm_last[f32, hidden]` — post-final-RMSNorm at the final position.
- `logits_last[f32, vocab]` — full logits at the final position.
- `greedy_tokens[int64, 10]` — first 10 greedy tokens after the prompt.
- `prompt[U*]`, `decoded[U*]` — human-readable strings for debugging.

`qwen3vl_image.npz` additionally has:
- `image_pixel_values[f32, tokens*t, 3*p*p]` — preprocessor output.
- `image_grid_thw[int64, 1, 3]` — the `[T, H, W]` grid.
- `vision_primary[f32, tokens_after_merge, out_hidden]` — primary vision embedding sequence.
- `vision_deepstack_{k}[f32, tokens_after_merge, out_hidden]` — deepstack embeddings for k=0,1,2.

## Tolerances (from `doc/20260619-qwen3vl/plan.md`)

- Post-embedding: exact match.
- Per-layer hidden state: max_abs < 5e-2, mean_abs < 5e-3 in bf16.
- Final logits: argmax exact; top-5 overlap.
- Greedy tokens: exact match for the first 10.

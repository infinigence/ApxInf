#!/usr/bin/env python3
"""Dump Qwen3.5 hidden states + logits for comparison with dump_qwen35.

Usage: qwen38_dump_reference.py <input.json> <out_dir>
Reads input_ids from <input.json>, runs the patched torch CPU model with
output_hidden_states, writes layer_NN.f32 (raw LE f32) and logits.f32 + shape.json.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen38_torch_reference import load_patched

def main():
    input_file, out_dir = sys.argv[1], sys.argv[2]
    ids = json.load(open(input_file))
    model = load_patched()
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        inp = torch.tensor([ids], dtype=torch.long)
        # Warm-up triggers compressed-tensors onload so the direct model call
        # below can access the materialized weights.
        model.generate(inp, max_new_tokens=1, do_sample=False)
        out = model.model(inp, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # hs[0] = embeddings, hs[i] = after layer i-1
        for i in range(1, len(hs)):
            np.asarray(hs[i][0].float()).tofile(os.path.join(out_dir, f"layer_{i-1:02}.f32"))
        logits = model.lm_head(model.model.norm(hs[-1]))
        np.asarray(logits[0].float()).tofile(os.path.join(out_dir, "logits.f32"))
        json.dump({"seq": len(ids), "vocab": logits.shape[-1],
                   "layers": len(hs) - 1, "start_pos": 0},
                  open(os.path.join(out_dir, "shape.json"), "w"))
    print(f"dumped {len(hs)-1} layers to {out_dir}", file=sys.stderr)

if __name__ == "__main__":
    main()

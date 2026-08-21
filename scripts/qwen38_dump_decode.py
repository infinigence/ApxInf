#!/usr/bin/env python3
"""Per-step decode logits dump with KV cache, mirroring dump_qwen35 --gen-tokens.

Usage: qwen38_dump_decode.py <input.json> <out_dir> <forced.json>
Feeds the forced tokens (one per decode step) so both sides see identical inputs.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen38_torch_reference import load_patched

def main():
    input_file, out_dir, forced_file = sys.argv[1], sys.argv[2], sys.argv[3]
    ids = json.load(open(input_file))
    forced = json.load(open(forced_file))
    model = load_patched()
    os.makedirs(out_dir, exist_ok=True)
    gen = open(os.path.join(out_dir, "gen.jsonl"), "w")

    def dump(step, logits, tok, fed, margin):
        np.asarray(logits[0].float()).tofile(
            os.path.join(out_dir, "logits_prefill.f32" if step == 0
                         else f"logits_step_{step-1:02}.f32"))
        gen.write(json.dumps({"step": step, "token": tok, "fed": fed,
                              "margin": margin}) + "\n")

    with torch.no_grad():
        inp = torch.tensor([ids], dtype=torch.long)
        model.generate(inp, max_new_tokens=1, do_sample=False)  # onload warm-up
        out = model(inp, use_cache=True)
        cache = out.past_key_values
        logits = out.logits
        row = logits[0, -1]
        vals, idx = torch.topk(row.float(), 2)
        tok = int(idx[0])
        margin = float(vals[0] - vals[1])
        fed = tok
        dump(0, logits, tok, fed, margin)
        current = forced[0] if forced else tok
        pos = len(ids)
        for s in range(1, min(len(forced), 8) + 1):
            out = model(torch.tensor([[current]], dtype=torch.long),
                        past_key_values=cache, use_cache=True,
                        position_ids=torch.tensor([[pos]], dtype=torch.long))
            cache = out.past_key_values
            logits = out.logits
            row = logits[0, -1]
            vals, idx = torch.topk(row.float(), 2)
            tok = int(idx[0])
            margin = float(vals[0] - vals[1])
            dump(s, logits, tok, current, margin)
            current = forced[s] if s < len(forced) else tok
            pos += 1
    gen.close()
    print(f"dumped {len(forced)+1} logits to {out_dir}", file=sys.stderr)

if __name__ == "__main__":
    main()

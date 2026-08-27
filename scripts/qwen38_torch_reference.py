#!/usr/bin/env python3
"""CPU torch reference for Qwen3.5-27B INT4 correctness checking.

Loads the model with transformers+compressed_tensors, patches the modules the
installed transformers version mis-loads (in_proj_a/in_proj_b dense, layer-0
out_proj dense), then runs greedy decode on prompts given as input_ids.
"""
import json, sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

MODEL_DIR = "/mnt/chuangxin/team3/work/model/qwen"

def load_patched():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map="cpu")
    model.eval()

    # Patch mis-loaded dense modules: in_proj_a, in_proj_b (all layers),
    # out_proj (layer 0 only). Checkpoint stores these dense.
    from safetensors import safe_open
    files = {}
    idx = json.load(open(f"{MODEL_DIR}/model.safetensors.index.json"))
    for name, fname in idx["weight_map"].items():
        files.setdefault(fname, []).append(name)

    txt = model.model if hasattr(model, "language_model") is False else model
    # locate text model layers
    def get_layers(m):
        for attr in ("model", "language_model", "text_model"):
            cand = getattr(m, attr, None)
            if cand is not None and hasattr(cand, "layers"):
                return cand
        raise RuntimeError("no layers found")

    container = get_layers(model)
    layers = container.layers
    n = len(layers)

    cache = {}
    def load_tensor(name):
        if name in cache:
            return cache[name]
        fname = idx["weight_map"][name]
        with safe_open(f"{MODEL_DIR}/{fname}", framework="pt", device="cpu") as f:
            t = f.get_tensor(name)
        cache[name] = t
        return t

    patched = 0
    for i in range(n):
        la = getattr(layers[i], "linear_attn", None)
        if la is None:
            continue
        h = la.in_proj_a.in_features
        va = la.in_proj_a.out_features
        vb = la.in_proj_b.out_features
        pre = f"model.language_model.layers.{i}.linear_attn"
        dt = model.dtype
        la.in_proj_a = nn.Linear(h, va, bias=False, dtype=dt)
        la.in_proj_a.weight.data.copy_(load_tensor(f"{pre}.in_proj_a.weight").to(dt))
        la.in_proj_b = nn.Linear(h, vb, bias=False, dtype=dt)
        la.in_proj_b.weight.data.copy_(load_tensor(f"{pre}.in_proj_b.weight").to(dt))
        patched += 2
        if i == 0:
            out_h = la.out_proj.out_features
            la.out_proj = nn.Linear(la.out_proj.in_features, out_h, bias=False, dtype=dt)
            la.out_proj.weight.data.copy_(load_tensor(f"{pre}.out_proj.weight").to(dt))
            patched += 1

    print(f"patched {patched} dense modules", file=sys.stderr)
    return model

def greedy_tokens(model, input_ids, max_new_tokens=16):
    ids = torch.tensor([input_ids], dtype=torch.long)
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
        top_p=None, pad_token_id=None, eos_token_id=None)
    return out[0].tolist()[len(input_ids):]

if __name__ == "__main__":
    model = load_patched()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        ids = json.loads(line)
        with torch.no_grad():
            new = greedy_tokens(model, ids)
        print(json.dumps(new), flush=True)

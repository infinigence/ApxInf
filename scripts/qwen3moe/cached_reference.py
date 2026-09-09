#!/usr/bin/env python3
"""Independent FP32 torch-CPU Qwen3-MoE AWQ reference.

Dequantizes the original checkpoint directly. KV caching avoids repeated prompt
work, and 128-query attention blocks bound score workspace at long context.
Use --case-json and --check-npz to validate against the original full-forward
reference before creating --isl 128/1024/4096/8192 fixtures.
"""
import argparse
import json
import math
import os
import struct
import time

import numpy as np
import torch

torch.set_grad_enabled(False)

AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


# ----------------------------------------------------------------------------
# safetensors (mmap, zero copy) --------------------------------------------
class Shards:
    def __init__(self, model_dir):
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        self.where = idx
        self.files = {}
        self.headers = {}
        for shard in sorted(set(idx.values())):
            path = os.path.join(model_dir, shard)
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.headers[shard] = (json.loads(fh.read(n)), 8 + n)
            self.files[shard] = np.memmap(path, dtype=np.uint8, mode="r")

    def get(self, name):
        shard = self.where[name]
        hdr, base = self.headers[shard]
        info = hdr[name]
        a, b = info["data_offsets"]
        raw = self.files[shard][base + a: base + b]
        dt = {"F16": np.float16, "BF16": None, "F32": np.float32, "I32": np.int32}[info["dtype"]]
        if dt is None:
            arr = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            t = torch.from_numpy(arr.view(np.float32).copy())
        else:
            t = torch.from_numpy(np.frombuffer(raw, dtype=dt).copy())
        return t.view(*info["shape"])


def dequant_awq(qweight, qzeros, scales, group_size=128):
    """AutoAWQ GEMM layout -> fp32 W of shape [K(in), N(out)]."""
    K, packed = qweight.shape
    N = packed * 8
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    iw = (qweight.unsqueeze(-1) >> shifts).view(K, N) & 0xF
    iz = (qzeros.unsqueeze(-1) >> shifts).view(qzeros.shape[0], N) & 0xF
    order = torch.arange(N).view(-1, 8)[:, AWQ_REVERSE_ORDER].reshape(-1)
    iw = iw[:, order]
    iz = iz[:, order]
    groups = K // group_size
    assert groups == scales.shape[0] == iz.shape[0], (K, scales.shape, iz.shape)
    w = (iw.view(groups, group_size, N).float() - iz.view(groups, 1, N).float()) * scales.view(groups, 1, N).float()
    return w.view(K, N)


class Linear4:
    def __init__(self, shards, prefix):
        self.qw = shards.get(prefix + ".qweight")
        self.qz = shards.get(prefix + ".qzeros")
        self.sc = shards.get(prefix + ".scales")
        self._w = None

    def weight(self):
        if self._w is None:
            self._w = dequant_awq(self.qw, self.qz, self.sc)
        return self._w

    def __call__(self, x):
        return x @ self.weight()

    def drop(self):
        self._w = None


def rms_norm(x, w, eps):
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


def rope_tables(S, head_dim, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(S, dtype=torch.float32)
    freqs = torch.outer(pos, inv)  # [S, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):  # x [S, H, D]
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


def signature(v):
    v = v.float().flatten()
    return np.array([v.sum().item(), v.abs().sum().item(), v.norm().item(), v.abs().max().item(),
                     v.mean().item(), v.std().item()], dtype=np.float32)


class Qwen3MoeRef:
    def __init__(self, model_dir):
        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        c = self.cfg
        self.L = c["num_hidden_layers"]
        self.H = c["num_attention_heads"]
        self.KVH = c["num_key_value_heads"]
        self.D = c["head_dim"]
        self.E = c["num_experts"]
        self.topk = c["num_experts_per_tok"]
        self.eps = c["rms_norm_eps"]
        self.theta = c["rope_theta"]
        self.norm_topk = c.get("norm_topk_prob", True)
        self.cache = [None] * self.L
        self.length = 0
        self.sh = Shards(model_dir)
        self.embed = self.sh.get("model.embed_tokens.weight").float()
        self.final_norm = self.sh.get("model.norm.weight").float()
        self.lm_head = self.sh.get("lm_head.weight").float()  # [V, hidden]

    def layer_forward(self, i, x, cos, sin, offset, sig_out=None):
        p = f"model.layers.{i}."
        g = self.sh.get
        S = x.shape[0]
        h = rms_norm(x, g(p + "input_layernorm.weight").float(), self.eps)
        q = Linear4(self.sh, p + "self_attn.q_proj")(h).view(S, self.H, self.D)
        k = Linear4(self.sh, p + "self_attn.k_proj")(h).view(S, self.KVH, self.D)
        v = Linear4(self.sh, p + "self_attn.v_proj")(h).view(S, self.KVH, self.D)
        q = rms_norm(q, g(p + "self_attn.q_norm.weight").float(), self.eps)
        k = rms_norm(k, g(p + "self_attn.k_norm.weight").float(), self.eps)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if offset:
            old_k, old_v = self.cache[i]
            k = torch.cat([old_k, k], dim=0)
            v = torch.cat([old_v, v], dim=0)
        self.cache[i] = (k, v)
        rep = self.H // self.KVH
        kh = k.repeat_interleave(rep, dim=1).transpose(0, 1)
        vh = v.repeat_interleave(rep, dim=1).transpose(0, 1)
        qh = q.transpose(0, 1)
        # Query blocking bounds the score/mask workspace at long context.
        # These are ordinary FP32 torch operations, independent of ApxInf.
        pieces = []
        keys = torch.arange(offset + S)
        for start in range(0, S, 128):
            stop = min(S, start + 128)
            scores = qh[:, start:stop] @ kh.transpose(-1, -2) / math.sqrt(self.D)
            causal = keys[None, :] > torch.arange(offset + start, offset + stop)[:, None]
            scores.masked_fill_(causal, float('-inf'))
            pieces.append(torch.softmax(scores, dim=-1) @ vh)
        attn = torch.cat(pieces, dim=1).transpose(0, 1).reshape(S, self.H * self.D)
        o = Linear4(self.sh, p + "self_attn.o_proj")(attn)
        x = x + o
        h = rms_norm(x, g(p + "post_attention_layernorm.weight").float(), self.eps)
        # router: HF computes logits in model dtype then softmax in fp32
        logits = h @ g(p + "mlp.gate.weight").float().t()  # [S, E]
        probs = torch.softmax(logits, dim=-1)
        w, idx = torch.topk(probs, self.topk, dim=-1)
        if self.norm_topk:
            w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e in idx.unique().tolist():
            rows, slots = (idx == e).nonzero(as_tuple=True)
            hin = h[rows]
            gate = Linear4(self.sh, p + f"mlp.experts.{e}.gate_proj")(hin)
            up = Linear4(self.sh, p + f"mlp.experts.{e}.up_proj")(hin)
            y = Linear4(self.sh, p + f"mlp.experts.{e}.down_proj")(torch.nn.functional.silu(gate) * up)
            out.index_add_(0, rows, y * w[rows, slots].unsqueeze(-1))
        x = x + out
        if sig_out is not None:
            sig_out["topk"].append(idx[-1].numpy().astype(np.int32))
            sig_out["topw"].append(w[-1].numpy().astype(np.float32))
        return x

    def forward(self, ids, collect=False):
        S = len(ids)
        x = self.embed[torch.tensor(ids)]
        offset = self.length
        cos, sin = rope_tables(offset + S, self.D, self.theta)
        cos, sin = cos[offset:], sin[offset:]
        sigs = [signature(x[-1])]
        extra = {"topk": [], "topw": []} if collect else None
        for i in range(self.L):
            t0 = time.time()
            x = self.layer_forward(i, x, cos, sin, offset, extra)
            sigs.append(signature(x[-1]))
            if collect and i % 8 == 0:
                print(f"  layer {i} done in {time.time() - t0:.1f}s", flush=True)
        h = rms_norm(x, self.final_norm, self.eps)
        logits = h[-1] @ self.lm_head.t()
        self.length += S
        return logits, np.stack(sigs), extra


def main():
    ap = argparse.ArgumentParser(description="FP32 CPU reference with cached KV and bounded attention scores")
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--case-json', help='Reuse exact prompt/forced tokens from an existing case')
    ap.add_argument('--check-npz', help='Require max difference <= 0.05 to an existing independent full-forward reference')
    ap.add_argument('--isl', type=int)
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--threads', type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    forced = None
    if args.case_json:
        source = json.load(open(args.case_json))
        ids = source['token_ids']
        forced = source.get('greedy_tokens')
        prompt = source.get('prompt', '')
    else:
        if args.isl is None or args.isl < 64:
            ap.error('--isl >= 64 or --case-json is required')
        filler = tok('The research archive contains notes about rivers, mountains, libraries, and railway stations. Every entry is reviewed for clarity and factual consistency. ', add_special_tokens=False)['input_ids']
        suffix = tok('\nAnswer with the city name only. What is the capital of France?', add_special_tokens=False)['input_ids']
        ids = (filler * ((args.isl // len(filler)) + 1))[:args.isl-len(suffix)] + suffix
        prompt = tok.decode(ids)
    model = Qwen3MoeRef(args.model)
    started = time.monotonic()
    logits, sigs, extra = model.forward(ids, collect=True)
    print(f'prefill length={len(ids)} seconds={time.monotonic()-started:.3f}', flush=True)
    rows, greedy = [], []
    for step in range(args.steps):
        rows.append(logits.numpy().astype(np.float32))
        greedy.append(int(logits.argmax()))
        if step + 1 < args.steps:
            fed = forced[step] if forced else greedy[-1]
            logits, _, _ = model.forward([fed])
        print(f'step={step} predicted={greedy[-1]} {tok.decode([greedy[-1]])!r}', flush=True)
    values = np.stack(rows)
    if args.check_npz:
        reference = np.load(args.check_npz)['step_logits']
        assert reference.shape == values.shape, (reference.shape, values.shape)
        differences = np.max(np.abs(reference-values), axis=1)
        print('cached versus full-forward max differences:', differences.tolist(), flush=True)
        if not np.isfinite(values).all() or np.max(differences) > 0.05:
            raise SystemExit('cached reference equivalence check failed')
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out,args.case+'.npz'), token_ids=np.asarray(ids,dtype=np.uint32),
             prefill_logits=values[0], greedy_tokens=np.asarray(greedy,dtype=np.uint32),
             step_logits=values, layer_sig=sigs, router_topk=np.stack(extra['topk']), router_weights=np.stack(extra['topw']))
    meta = dict(token_ids=ids, greedy_tokens=forced or greedy, predicted_tokens=greedy,
                prompt=prompt, model=args.model, torch=torch.__version__,
                reference='torch CPU FP32, cached KV, attention query block 128',
                seconds=time.monotonic()-started)
    with open(os.path.join(args.out,args.case+'.json'),'w') as f: json.dump(meta,f,indent=2)
    print('saved', args.case, 'seconds', meta['seconds'], flush=True)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# Independent PyTorch (CPU f32) reference for Qwen3.5 text forward, used to
# cross-check the Rust CPU implementation on the same 8-token prompt.
import json, math, time, os
import numpy as np
import torch
from safetensors import safe_open

M = "/mnt/chuangxin/team1/Qwen3.8-27B-AWQ-INT4"
cfg = json.load(open(f"{M}/config.json"))["text_config"]

hidden = cfg["hidden_size"]
inter = cfg["intermediate_size"]
layers = cfg["num_hidden_layers"]
heads = cfg["num_attention_heads"]
kvh = cfg["num_key_value_heads"]
hd = cfg["head_dim"]
eps = cfg["rms_norm_eps"]
nk = cfg["linear_num_key_heads"]
nv = cfg["linear_num_value_heads"]
kd = cfg["linear_key_head_dim"]
vd = cfg["linear_value_head_dim"]
ltypes = cfg["layer_types"]
vocab = cfg["vocab_size"]
rp = cfg["rope_parameters"]
theta = rp["rope_theta"]
prf = rp.get("partial_rotary_factor", 1.0)
rotary_dim = int(hd * prf)

L = 8
IDS = [248045, 8678, 198, 24342, 286, 4879, 369, 716]

t_load = time.time()
idx = json.load(open(f"{M}/model.safetensors.index.json"))
shards = sorted(set(idx["weight_map"].values()))
data = {}
for sh in shards:
    with safe_open(f"{M}/{sh}", framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.language_model.") or k.startswith("lm_head."):
                data[k] = f.get_tensor(k)
print(f"[ref] loaded {len(data)} tensors in {time.time()-t_load:.1f}s", flush=True)

def b2f(t):
    return t.to(torch.float32)

SHIFTS = torch.tensor([0,4,8,12,16,20,24,28], dtype=torch.int32)

def dequant(base):
    packed = data[f"{base}.weight_packed"].to(torch.int32)      # [out, c8]
    scale = b2f(data[f"{base}.weight_scale"])                  # [out, g]
    zp = data[f"{base}.weight_zero_point"].to(torch.int32)     # [oz8, g]
    out, c8 = packed.shape
    g = scale.shape[1]
    in_ = g * 32
    nib = ((packed.unsqueeze(-1) >> SHIFTS) & 0xF).to(torch.float32)  # [out,c8,8]
    w4 = (nib - 8.0).reshape(out, c8 * 8)[:, :in_]
    del nib, packed
    sc = scale.repeat_interleave(32, dim=1)
    zsh = ((zp.unsqueeze(-1) >> SHIFTS) & 0xF).to(torch.float32)      # [oz8,g,8]
    zv = (zsh - 8.0)
    H, GG = zv.shape[0], zv.shape[1]
    zv = zv.permute(0, 2, 1).reshape(H * 8, GG)[:out, :].repeat_interleave(32, dim=1)
    del zsh, zp
    W = sc * (w4 - zv)
    del sc, w4, zv
    return W  # [out, in] f32

def rms(x, w, e):
    m = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(m + e) * (1.0 + w)

def silu(x):
    return x * torch.sigmoid(x)

def softplus(x):
    return torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))

# RoPE tables (rotary_dim=64, interleaved split-half)
inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))
freq = torch.arange(L, dtype=torch.float64)[:, None] * inv[None, :]  # [L,32] f64
cos32 = torch.cos(freq).float()  # [L,32]
sin32 = torch.sin(freq).float()
cos = torch.cat([cos32, cos32], -1)
sin = torch.cat([sin32, sin32], -1)

def rope(x):
    # x: [L, A, hd]; rotate the first rotary_dim dims (interleaved split-half)
    xr = x[..., :rotary_dim]
    d = rotary_dim // 2
    r = torch.cat([-xr[..., d:], xr[..., :d]], -1)
    c = cos.view(L, 1, rotary_dim)
    s = sin.view(L, 1, rotary_dim)
    y = xr * c + r * s
    return torch.cat([y, x[..., rotary_dim:]], -1)

def full_layer(prefix, x):
    w_in = b2f(data[f"{prefix}.input_layernorm.weight"])
    xn = rms(x, w_in, eps)
    Wq = dequant(f"{prefix}.self_attn.q_proj")
    Wk = dequant(f"{prefix}.self_attn.k_proj")
    Wv = dequant(f"{prefix}.self_attn.v_proj")
    qg = xn @ Wq.t()                      # [L, heads*2*hd]
    k = xn @ Wk.t()                       # [L, kvh*hd]
    v = xn @ Wv.t()
    qg = qg.view(L, heads, 2, hd)         # [L,h,2,hd]
    q = qg[:, :, 0, :]
    gate = qg[:, :, 1, :]
    k = k.view(L, kvh, hd)
    v = v.view(L, kvh, hd)
    wq = b2f(data[f"{prefix}.self_attn.q_norm.weight"])
    wk = b2f(data[f"{prefix}.self_attn.k_norm.weight"])
    q = rms(q, wq, eps)
    k = rms(k, wk, eps)
    q = rope(q)
    k = rope(k)
    scale = 1.0 / (hd ** 0.5)
    reps = heads // kvh
    k_exp = k.repeat_interleave(reps, dim=1)              # [L, heads, hd]
    v_exp = v.repeat_interleave(reps, dim=1)              # [L, heads, hd]
    scores = torch.einsum('qhd,lhd->qhl', q, k_exp) * scale  # [q, heads, key]
    msk = torch.tril(torch.ones(L, L)).bool().view(L, 1, L)
    scores = scores.masked_fill(msk == False, float('-inf'))
    attn = torch.softmax(scores, dim=-1)                  # [q, heads, key]
    o = torch.einsum('qhl,lhd->qhd', attn, v_exp)         # [L, heads, hd]
    o = o.reshape(L, heads * hd)
    o = o * torch.sigmoid(gate.reshape(L, heads * hd))
    Wo = dequant(f"{prefix}.self_attn.o_proj")
    o = o @ Wo.t()
    x = x + o
    w_post = b2f(data[f"{prefix}.post_attention_layernorm.weight"])
    xn = rms(x, w_post, eps)
    Wg = dequant(f"{prefix}.mlp.gate_proj")
    Wu = dequant(f"{prefix}.mlp.up_proj")
    Wd = dequant(f"{prefix}.mlp.down_proj")
    h = silu(xn @ Wg.t()) * (xn @ Wu.t())
    x = x + h @ Wd.t()
    return x

def gdelta_layer(prefix, x):
    w_in = b2f(data[f"{prefix}.input_layernorm.weight"])
    xn = rms(x, w_in, eps)
    Wqkv = dequant(f"{prefix}.linear_attn.in_proj_qkv")
    qkv = xn @ Wqkv.t()                    # [L, conv_dim]
    # causal depthwise conv kernel=4 + silu
    Wc = b2f(data[f"{prefix}.linear_attn.conv1d.weight"]).squeeze(1)  # [conv_dim,4]
    conv_dim = qkv.shape[1]
    conv = torch.zeros_like(qkv)
    s = torch.zeros_like(qkv)
    for j in range(4):
        if j == 0:
            s = qkv
        else:
            s = torch.zeros_like(qkv)
            s[j:] = qkv[:-j]
        conv += Wc[:, j] * s
    conv = silu(conv)
    q = conv[:, :nk*kd].view(L, nk, kd)
    k = conv[:, nk*kd:2*nk*kd].view(L, nk, kd)
    v = conv[:, 2*nk*kd:].view(L, nv, vd)
    Wz = dequant(f"{prefix}.linear_attn.in_proj_z")
    z = (xn @ Wz.t()).view(L, nv, vd)
    Wa = b2f(data[f"{prefix}.linear_attn.in_proj_a.weight"])
    Wb = b2f(data[f"{prefix}.linear_attn.in_proj_b.weight"])
    a = xn @ Wa.t()
    b = xn @ Wb.t()
    A = b2f(data[f"{prefix}.linear_attn.A_log"])
    dt = b2f(data[f"{prefix}.linear_attn.dt_bias"])
    beta = torch.sigmoid(b)                     # [L,nv]
    g = -torch.exp(A) * softplus(a + dt)        # [L,nv]
    reps = nv // nk
    q = q.repeat_interleave(reps, dim=1).view(L, nv, kd)
    k = k.repeat_interleave(reps, dim=1).view(L, nv, kd)
    scale = 1.0 / (kd ** 0.5)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    S = torch.zeros(nv, kd, vd)
    O = torch.zeros(L, nv, vd)
    for t in range(L):
        S = S * torch.exp(g[t]).view(nv, 1, 1)
        kv = torch.einsum('hkd,hk->hd', S, k[t])          # S^T @ k
        delta = (v[t] - kv) * beta[t].view(nv, 1)
        S = S + torch.einsum('hk,hd->hkd', k[t], delta)
        O[t] = torch.einsum('hkd,hk->hd', S, q[t])        # S^T @ q
    wnorm = b2f(data[f"{prefix}.linear_attn.norm.weight"])
    o = O / torch.sqrt((O * O).mean(-1, keepdim=True) + eps) * (1.0 + wnorm)
    o = o * silu(z)
    o = o.reshape(L, nv * vd)
    if f"{prefix}.linear_attn.out_proj.weight_packed" in data:
        Wo = dequant(f"{prefix}.linear_attn.out_proj")
    else:
        Wo = b2f(data[f"{prefix}.linear_attn.out_proj.weight"])
    x = x + o @ Wo.t()
    w_post = b2f(data[f"{prefix}.post_attention_layernorm.weight"])
    xn = rms(x, w_post, eps)
    Wg = dequant(f"{prefix}.mlp.gate_proj")
    Wu = dequant(f"{prefix}.mlp.up_proj")
    Wd = dequant(f"{prefix}.mlp.down_proj")
    h = silu(xn @ Wg.t()) * (xn @ Wu.t())
    x = x + h @ Wd.t()
    return x

t0 = time.time()
x = b2f(data["model.language_model.embed_tokens.weight"])[IDS]   # [L,hidden]
for li in range(layers):
    prefix = f"model.language_model.layers.{li}"
    if str(ltypes[li]) == "full_attention":
        x = full_layer(prefix, x)
    else:
        x = gdelta_layer(prefix, x)
    if li % 8 == 7:
        print(f"[ref] layer {li+1}/{layers} done {time.time()-t0:.1f}s", flush=True)

wfinal = b2f(data["model.language_model.norm.weight"])
hall = rms(x, wfinal, eps)
Wlm = b2f(data["lm_head.weight"])
logits_all = (Wlm @ hall.T).T                   # [L, V]
print(f"[ref] forward done {time.time()-t0:.1f}s", flush=True)
logits_all.numpy().astype(np.float32).tofile("/tmp/ref_logits_all.bin")
logits = logits_all[-1]
t5 = logits.topk(5)
print("top5:", [(int(i), round(float(v), 4)) for i, v in zip(t5.indices.tolist(), t5.values.tolist())])
print("argmax:", int(logits.argmax()))

if os.path.exists("/tmp/rust_logits.bin"):
    rust_all = np.fromfile("/tmp/rust_logits.bin", dtype=np.float32).reshape(-1, logits_all.shape[1])
    ref_all_pos = logits_all
    for pos in range(min(rust_all.shape[0], ref_all_pos.shape[0])):
        ra = int(np.argmax(rust_all[pos])); rb = int(np.argmax(ref_all_pos[pos]))
        if ra != rb:
            print(f"[pos {pos}] rust argmax {ra} != ref argmax {rb}")
    cos_all = float(np.dot(rust_all.ravel(), ref_all_pos.ravel())/(np.linalg.norm(rust_all.ravel())*np.linalg.norm(ref_all_pos.ravel())))
    print("full cos:", cos_all)
    rust = rust_all[-1]
    ref = logits.numpy().astype(np.float32)
    print("len rust/ref:", rust.shape[0], ref.shape[0])
    diff = np.abs(rust - ref)
    print("max abs diff:", float(diff.max()))
    print("mean abs diff:", float(diff.mean()))
    cos = float(np.dot(rust, ref) / (np.linalg.norm(rust) * np.linalg.norm(ref)))
    print("cosine:", cos)
    print("rust argmax:", int(np.argmax(rust)), " ref argmax:", int(np.argmax(ref)))
    rt = set(np.argsort(rust)[::-1][:5].tolist()); rft = set(np.argsort(ref)[::-1][:5].tolist())
    print("top5 intersection:", rt & rft)
else:
    print("[ref] /tmp/rust_logits.bin not found; skipped comparison")

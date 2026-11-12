"""Independent PyTorch functional PI0.5 oracle, raw checkpoint layout.
Norm and RoPE functions are executed unchanged from pinned OpenPI source.
Tiny geometry is a diagnostic fixture, not a trained PI0.5 checkpoint.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import Tensor, nn

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--openpi", type=Path, required=True, help="Pinned OpenPI source checkout"
)
parser.add_argument(
    "--output", type=Path, required=True, help="Output fixture directory"
)
args = parser.parse_args()
source = args.openpi / "src/openpi/models_pytorch"
sources = {
    "modeling_gemma.py": (
        source / "transformers_replace/models/gemma/modeling_gemma.py",
        "17eeb54e277939e58cd6f8a92719a275e36baf0c9f463b2754227d42f87aedde",
    ),
    "pi0_pytorch.py": (
        source / "pi0_pytorch.py",
        "dcd7b6f508808ce4b7170a1720a4c62c6da04edae3652d16c94a469dca924e16",
    ),
}
for path, expected_hash in sources.values():
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"Reference source hash mismatch: {path}")
ns = {"torch": torch, "nn": nn, "Optional": Optional, "math": math, "Tensor": Tensor}
for name, selected in [
    ("modeling_gemma.py", {"GemmaRMSNorm", "rotate_half", "apply_rotary_pos_emb"}),
    ("pi0_pytorch.py", {"get_safe_dtype", "create_sinusoidal_pos_embedding"}),
]:
    tree = ast.parse(sources[name][0].read_text())
    tree.body = [
        x
        for x in tree.body
        if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name in selected
    ]
    # Execute only selected definitions from the hash-verified upstream files.
    exec(compile(tree, name, "exec"), ns)  # noqa: S102
torch.manual_seed(195)
torch.set_num_threads(1)
w = {}


def rand(shape):
    return torch.randn(shape) * 0.18


def lin(name, dim, out, bias=False):
    w[name + ".weight"] = rand((out, dim))
    if bias:
        w[name + ".bias"] = rand((out,))


def ln(name, dim):
    w[name + ".weight"] = 1 + rand((dim,))
    w[name + ".bias"] = rand((dim,))


def rms(name, dim):
    w[name + ".weight"] = rand((dim,))


def ada(name, dim):
    lin(name + ".dense", dim, 3 * dim, True)


def attn(name, width):
    for key, out in [("q_proj", 8), ("k_proj", 4), ("v_proj", 4), ("o_proj", width)]:
        lin(name + "." + key, 8 if key == "o_proj" else width, out)


def mlp_weights(name, width):
    lin(name + ".gate_proj", width, 12)
    lin(name + ".up_proj", width, 12)
    lin(name + ".down_proj", 12, width)


vp = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
lp = "paligemma_with_expert.paligemma.model.language_model"
ap = "paligemma_with_expert.gemma_expert.model"
w[vp + ".embeddings.patch_embedding.weight"] = rand((8, 3, 2, 2))
w[vp + ".embeddings.patch_embedding.bias"] = rand((8,))
w[vp + ".embeddings.position_embedding.weight"] = rand((4, 8))
for i in range(2):
    p = vp + f".encoder.layers.{i}"
    ln(p + ".layer_norm1", 8)
    ln(p + ".layer_norm2", 8)
    for key in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        lin(p + ".self_attn." + key, 8, 8, True)
    lin(p + ".mlp.fc1", 8, 12, True)
    lin(p + ".mlp.fc2", 12, 8, True)
ln(vp + ".post_layernorm", 8)
lin("paligemma_with_expert.paligemma.model.multi_modal_projector.linear", 8, 8, True)
w[lp + ".embed_tokens.weight"] = rand((16, 8))
for i in range(2):
    p = lp + f".layers.{i}"
    rms(p + ".input_layernorm", 8)
    rms(p + ".post_attention_layernorm", 8)
    attn(p + ".self_attn", 8)
    mlp_weights(p + ".mlp", 8)
    p = ap + f".layers.{i}"
    ada(p + ".input_layernorm", 6)
    ada(p + ".post_attention_layernorm", 6)
    attn(p + ".self_attn", 6)
    mlp_weights(p + ".mlp", 6)
rms(lp + ".norm", 8)
ada(ap + ".norm", 6)
lin("action_in_proj", 3, 6, True)
lin("action_out_proj", 6, 3, True)
lin("time_mlp_in", 6, 6, True)
lin("time_mlp_out", 6, 6, True)


def linear(x, p):
    return F.linear(x, w[p + ".weight"], w.get(p + ".bias"))


def norm(x, p):
    return F.layer_norm(x, (x.shape[-1],), w[p + ".weight"], w[p + ".bias"], 1e-6)


def rmsnorm(x, p, cond=None):
    module = ns["GemmaRMSNorm"](
        x.shape[-1], cond_dim=6 if cond is not None else None
    ).float()
    prefix = p + "."
    module.load_state_dict(
        {k[len(prefix) :]: v for k, v in w.items() if k.startswith(prefix)}
    )
    return module(x, cond)


def rope(q, k, offset):
    pos = torch.arange(offset, offset + q.shape[0], dtype=torch.float32)
    freqs = 1 / (10000 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4))
    angles = (pos[:, None] * freqs[None, :]).repeat(1, 2)[None, :, :]
    a, b = ns["apply_rotary_pos_emb"](
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], angles.cos(), angles.sin()
    )
    return a[0].transpose(0, 1), b[0].transpose(0, 1)


def qkv(x, p, offset):
    q = linear(x, p + ".q_proj").reshape(-1, 2, 4)
    k = linear(x, p + ".k_proj").reshape(-1, 1, 4)
    v = linear(x, p + ".v_proj").reshape(-1, 1, 4)
    q, k = rope(q, k, offset)
    return q, k, v


def attend(q, k, v):
    if k.shape[1] != q.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], 1)
    return (
        F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        .transpose(0, 1)
        .reshape(q.shape[0], -1)
    )


def mlp(x, p):
    return linear(
        F.gelu(linear(x, p + ".gate_proj"), approximate="tanh")
        * linear(x, p + ".up_proj"),
        p + ".down_proj",
    )


@torch.no_grad()
def run(rgb, tokens, noise):
    img = rgb.float().permute(0, 3, 1, 2) * (2 / 255) - 1
    x = (
        F.conv2d(
            img,
            w[vp + ".embeddings.patch_embedding.weight"],
            w[vp + ".embeddings.patch_embedding.bias"],
            stride=2,
        )
        .flatten(2)
        .transpose(1, 2)
    )
    x = x + w[vp + ".embeddings.position_embedding.weight"]
    for i in range(2):
        p = vp + f".encoder.layers.{i}"
        h = norm(x, p + ".layer_norm1")
        q, k, v = [
            linear(h, p + ".self_attn." + key).reshape(2, 4, 2, 4).transpose(1, 2)
            for key in ["q_proj", "k_proj", "v_proj"]
        ]
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 4, 8)
        x = x + linear(out, p + ".self_attn.out_proj")
        x = x + linear(
            F.gelu(
                linear(norm(x, p + ".layer_norm2"), p + ".mlp.fc1"), approximate="tanh"
            ),
            p + ".mlp.fc2",
        )
    vision = linear(
        norm(x, vp + ".post_layernorm"),
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear",
    ).reshape(-1, 8)
    prefix = torch.cat(
        [vision, F.embedding(tokens, w[lp + ".embed_tokens.weight"]) * math.sqrt(8)]
    )
    x = prefix
    cache = []
    for i in range(2):
        p = lp + f".layers.{i}"
        h, _ = rmsnorm(x, p + ".input_layernorm")
        q, k, v = qkv(h, p + ".self_attn", 0)
        cache.append((k, v))
        x = x + linear(attend(q, k, v), p + ".self_attn.o_proj")
        h, _ = rmsnorm(x, p + ".post_attention_layernorm")
        x = x + mlp(h, p + ".mlp")
    state = noise.clone()
    steps = []
    for step in range(3):
        time = (
            torch.tensor(1.0, dtype=torch.float32)
            - torch.tensor(step, dtype=torch.float32) / 3
        ).reshape(1)
        # Rust config stores periods as f32, promoted to f64 by the original helper.
        cond = ns["create_sinusoidal_pos_embedding"](
            time, 6, float(torch.tensor(0.004)), 4.0, torch.device("cpu")
        ).float()
        cond = F.silu(linear(F.silu(linear(cond, "time_mlp_in")), "time_mlp_out"))
        x = linear(state, "action_in_proj")
        for i in range(2):
            p = ap + f".layers.{i}"
            h, gate = rmsnorm(x, p + ".input_layernorm", cond)
            q, k, v = qkv(h, p + ".self_attn", len(prefix))
            k = torch.cat([cache[i][0], k])
            v = torch.cat([cache[i][1], v])
            x = x + linear(attend(q, k, v), p + ".self_attn.o_proj") * gate
            h, gate = rmsnorm(x, p + ".post_attention_layernorm", cond)
            x = x + mlp(h, p + ".mlp") * gate
        h, _ = rmsnorm(x, ap + ".norm", cond)
        state = state - linear(h, "action_out_proj") / 3
        steps.append(state.flatten().tolist())
    return state, {
        "vision": vision.flatten().tolist(),
        "prefix": prefix.flatten().tolist(),
        "keys": [k.flatten().tolist() for k, v in cache],
        "values": [v.flatten().tolist() for k, v in cache],
        "steps": steps,
    }


fixture = args.output
fixture.mkdir(parents=True, exist_ok=True)
save_file(w, str(fixture / "model.safetensors"))
(fixture / "config.json").write_text(json.dumps({"model_type": "pi05"}))
cases = []
for seed, count in [(90, 3), (91, 2)]:
    torch.manual_seed(seed)
    rgb = torch.randint(0, 256, (2, 4, 4, 3), dtype=torch.uint8)
    tokens = torch.tensor([1, 3, 5][:count])
    noise = torch.randn(3, 3)
    out, stages = run(rgb, tokens, noise)
    cases.append(
        dict(
            rgb=rgb.flatten().tolist(),
            tokens=tokens.tolist(),
            noise=noise.flatten().tolist(),
            expected=out.flatten().tolist(),
            **stages,
        )
    )
(fixture / "cases.json").write_text(json.dumps(cases))
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "source": "OpenPI 215abfb217dbac7d5f1273282331b9b1866c0479",
            "cases": len(cases),
            "fixture": str(fixture),
        }
    )
)

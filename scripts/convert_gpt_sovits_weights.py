#!/usr/bin/env python3
"""Convert the official GPT-SoVITS V5 checkpoints into ApxInf safetensors.

The converter consumes the upstream PyTorch checkpoints and produces
component-level safetensors plus a small ``config.json``.  It never downloads
anything and never writes into the ApxInf repository.  Weights are emitted in
FP16, matching the reference inference precision.

Usage:
    python scripts/convert_gpt_sovits_weights.py \
        --gpt  .../pretrained_models/s1v3.ckpt \
        --sovits .../pretrained_models/gsv-v5-pretrained/s2Gv5turbo.pth \
        --vocoder .../pretrained_models/gsv-v5-pretrained/vocoder.pth \
        --hubert .../pretrained_models/chinese-hubert-base/pytorch_model.bin \
        --roberta .../pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin \
        --out /path/to/gpt-sovits-v5-apxinf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "weight" in payload:
        return payload["weight"]
    return payload


def take(state: dict, key: str, output: dict, name: str | None = None) -> torch.Tensor:
    if key not in state:
        raise KeyError(f"missing checkpoint key {key}")
    tensor = state.pop(key)
    output[name or key] = tensor
    return tensor


def linear(state: dict, prefix: str, output: dict, name: str) -> None:
    weight = take(state, prefix + ".weight", output, name + ".w")
    output[name + ".w"] = weight.t().contiguous()
    if prefix + ".bias" in state:
        take(state, prefix + ".bias", output, name + ".b")


def conv(state: dict, prefix: str, output: dict, name: str) -> None:
    take(state, prefix + ".weight", output, name + ".w")
    if prefix + ".bias" in state:
        take(state, prefix + ".bias", output, name + ".b")


def weight_norm_weight(state: dict, prefix: str, dim: int) -> torch.Tensor:
    g = state.pop(prefix + ".weight_g")
    v = state.pop(prefix + ".weight_v")
    if dim == 0:
        norm = v.norm(dim=tuple(range(1, v.dim())), keepdim=True).clamp_min(1e-12)
    elif dim == 2:
        norm = v.norm(dim=tuple(range(0, v.dim() - 1)), keepdim=True).clamp_min(1e-12)
    else:
        raise ValueError(f"unsupported weight_norm dim {dim}")
    return (v.float() * (g.float() / norm.float())).to(v.dtype)


def convert_gpt(state: dict) -> dict:
    output: dict = {}
    linear(state, "model.bert_proj", output, "gpt.bert_proj")
    take(state, "model.ar_text_embedding.word_embeddings.weight", output, "gpt.text_emb")
    take(state, "model.ar_text_position.alpha", output, "gpt.text_alpha")
    take(state, "model.ar_audio_embedding.word_embeddings.weight", output, "gpt.audio_emb")
    take(state, "model.ar_audio_position.alpha", output, "gpt.audio_alpha")
    layers = sorted(
        int(key.split(".")[3])
        for key in state
        if key.startswith("model.h.layers.") and key.endswith(".norm1.weight")
    )
    for index in layers:
        prefix = f"model.h.layers.{index}"
        qkv = take(state, prefix + ".self_attn.in_proj_weight", output, f"_tmp.qkv.{index}")
        bias = take(state, prefix + ".self_attn.in_proj_bias", output, f"_tmp.qkvb.{index}")
        q, k, v = qkv.chunk(3, dim=0)
        qb, kb, vb = bias.chunk(3, dim=0)
        for tag, tensor, tensor_bias in (("q", q, qb), ("k", k, kb), ("v", v, vb)):
            output[f"gpt.layers.{index}.{tag}.w"] = tensor.t().contiguous()
            output[f"gpt.layers.{index}.{tag}.b"] = tensor_bias.contiguous()
        linear(state, prefix + ".self_attn.out_proj", output, f"gpt.layers.{index}.o")
        linear(state, prefix + ".linear1", output, f"gpt.layers.{index}.ff1")
        linear(state, prefix + ".linear2", output, f"gpt.layers.{index}.ff2")
        take(state, prefix + ".norm1.weight", output, f"gpt.layers.{index}.ln1.w")
        take(state, prefix + ".norm1.bias", output, f"gpt.layers.{index}.ln1.b")
        take(state, prefix + ".norm2.weight", output, f"gpt.layers.{index}.ln2.w")
        take(state, prefix + ".norm2.bias", output, f"gpt.layers.{index}.ln2.b")
    output["gpt.out.w"] = state.pop("model.ar_predict_layer.weight").t().contiguous()
    for key in list(output):
        if key.startswith("_tmp."):
            del output[key]
    return output


def cfm_text_positions(dim: int, length: int = 4096) -> torch.Tensor:
    """Preserve the upstream nonpersistent CPU FP32 positional buffer."""
    if dim <= 0 or dim % 2:
        raise ValueError("CFM text dimension must be positive and even")
    frequency = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
    angles = torch.outer(torch.arange(length).float(), frequency)
    return torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()


def convert_sovits(state: dict) -> dict:
    output: dict = {}
    # TextEncoder
    conv(state, "enc_p.ssl_proj", output, "sovits.enc.ssl_proj")
    take(state, "enc_p.text_embedding.weight", output, "sovits.enc.text_emb")
    for group, layers in (("ssl", 3), ("text", 6), ("tail", 3)):
        prefix = "enc_p.encoder_ssl" if group == "ssl" else (
            "enc_p.encoder_text" if group == "text" else "enc_p.encoder2"
        )
        for index in range(layers):
            base = f"{prefix}.attn_layers.{index}"
            for tag in ("q", "k", "v", "o"):
                weight = take(state, f"{base}.conv_{tag}.weight", output, f"_tmp.{group}.{index}.{tag}.w")
                bias = take(state, f"{base}.conv_{tag}.bias", output, f"_tmp.{group}.{index}.{tag}.b")
                output[f"sovits.enc.{group}.{index}.{tag}.w"] = weight.squeeze(-1).t().contiguous()
                output[f"sovits.enc.{group}.{index}.{tag}.b"] = bias.contiguous()
            take(state, f"{base}.emb_rel_k", output, f"sovits.enc.{group}.{index}.emb_k")
            take(state, f"{base}.emb_rel_v", output, f"sovits.enc.{group}.{index}.emb_v")
            take(state, f"{prefix}.norm_layers_1.{index}.gamma", output, f"sovits.enc.{group}.{index}.ln1.w")
            take(state, f"{prefix}.norm_layers_1.{index}.beta", output, f"sovits.enc.{group}.{index}.ln1.b")
            take(state, f"{prefix}.norm_layers_2.{index}.gamma", output, f"sovits.enc.{group}.{index}.ln2.w")
            take(state, f"{prefix}.norm_layers_2.{index}.beta", output, f"sovits.enc.{group}.{index}.ln2.b")
            conv(state, f"{prefix}.ffn_layers.{index}.conv_1", output, f"sovits.enc.{group}.{index}.ff1")
            conv(state, f"{prefix}.ffn_layers.{index}.conv_2", output, f"sovits.enc.{group}.{index}.ff2")

    # MRTE
    for tag in ("q", "k", "v", "o"):
        weight = take(state, f"enc_p.mrte.cross_attention.conv_{tag}.weight", output, f"_tmp.mrte.{tag}.w")
        bias = take(state, f"enc_p.mrte.cross_attention.conv_{tag}.bias", output, f"_tmp.mrte.{tag}.b")
        output[f"sovits.mrte.{tag}.w"] = weight.squeeze(-1).t().contiguous()
        output[f"sovits.mrte.{tag}.b"] = bias.contiguous()
    for tag in ("c_pre", "text_pre", "c_post"):
        conv(state, f"enc_p.mrte.{tag}", output, f"sovits.mrte.{tag}")

    # MelStyleEncoder
    linear(state, "ref_enc.spectral.0.fc", output, "sovits.refenc.spec0")
    linear(state, "ref_enc.spectral.3.fc", output, "sovits.refenc.spec3")
    for index in range(2):
        weight = take(state, f"ref_enc.temporal.{index}.conv1.conv.weight", output, f"_tmp.temp.{index}")
        bias = take(state, f"ref_enc.temporal.{index}.conv1.conv.bias", output, f"_tmp.tempb.{index}")
        first, second = weight.chunk(2, dim=0)
        first_b, second_b = bias.chunk(2, dim=0)
        output[f"sovits.refenc.temp{index}.x1.w"] = first.contiguous()
        output[f"sovits.refenc.temp{index}.x1.b"] = first_b.contiguous()
        output[f"sovits.refenc.temp{index}.x2.w"] = second.contiguous()
        output[f"sovits.refenc.temp{index}.x2.b"] = second_b.contiguous()
    for tag in ("q", "k", "v"):
        linear(state, f"ref_enc.slf_attn.w_{tag}s", output, f"sovits.refenc.attn.{tag}")
    linear(state, "ref_enc.slf_attn.fc", output, "sovits.refenc.attn.o")
    linear(state, "ref_enc.fc.fc", output, "sovits.refenc.out")

    # Quantizer
    conv(state, "ssl_proj", output, "hubert.ssl_proj")
    take(state, "quantizer.vq.layers.0._codebook.embed", output, "sovits.quant.codebook")
    conv(state, "bridge.0", output, "sovits.bridge")

    # WN encoder
    conv(state, "wns1.pre", output, "sovits.wns.pre")
    for index in range(8):
        weight = weight_norm_weight(state, f"wns1.enc.in_layers.{index}", 0)
        take(state, f"wns1.enc.in_layers.{index}.bias", output, f"sovits.wns.in.{index}.b")
        output[f"sovits.wns.in.{index}.w"] = weight.contiguous()
    for index in range(8):
        weight = weight_norm_weight(state, f"wns1.enc.res_skip_layers.{index}", 0)
        bias = take(state, f"wns1.enc.res_skip_layers.{index}.bias", output, f"_tmp.rs.{index}.b")
        if index < 7:
            res, skip = weight.chunk(2, dim=0)
            res_b, skip_b = bias.chunk(2, dim=0)
            output[f"sovits.wns.rs.{index}.res.w"] = res.contiguous()
            output[f"sovits.wns.rs.{index}.res.b"] = res_b.contiguous()
            output[f"sovits.wns.rs.{index}.skip.w"] = skip.contiguous()
            output[f"sovits.wns.rs.{index}.skip.b"] = skip_b.contiguous()
        else:
            output[f"sovits.wns.rs.{index}.skip.w"] = weight.contiguous()
            output[f"sovits.wns.rs.{index}.skip.b"] = bias.contiguous()
    condition = weight_norm_weight(state, "wns1.enc.cond_layer", 0)
    condition_bias = take(state, "wns1.enc.cond_layer.bias", output, "_tmp.cond.b")
    for index in range(8):
        output[f"sovits.wns.cond.{index}.w"] = condition[index * 1024 : (index + 1) * 1024].contiguous()
        output[f"sovits.wns.cond.{index}.b"] = condition_bias[index * 1024 : (index + 1) * 1024].contiguous()
    conv(state, "wns1.proj", output, "sovits.wns.proj")

    # V5 CFM / DiT: the upstream positional buffer is not in state_dict.
    text_dim = state["cfm.estimator.text_embed.text_blocks.0.dwconv.weight"].shape[0]
    output["cfm.text_pe"] = cfm_text_positions(text_dim)
    linear(state, "cfm.estimator.time_embed.time_mlp.0", output, "cfm.time0")
    linear(state, "cfm.estimator.time_embed.time_mlp.2", output, "cfm.time2")
    for index in range(4):
        base = f"cfm.estimator.text_embed.text_blocks.{index}"
        take(state, f"{base}.dwconv.weight", output, f"cfm.text{index}.dw.w")
        take(state, f"{base}.dwconv.bias", output, f"cfm.text{index}.dw.b")
        take(state, f"{base}.norm.weight", output, f"cfm.text{index}.ln.w")
        take(state, f"{base}.norm.bias", output, f"cfm.text{index}.ln.b")
        linear(state, f"{base}.pwconv1", output, f"cfm.text{index}.pw1")
        take(state, f"{base}.grn.gamma", output, f"cfm.text{index}.grn.g")
        take(state, f"{base}.grn.beta", output, f"cfm.text{index}.grn.b")
        linear(state, f"{base}.pwconv2", output, f"cfm.text{index}.pw2")
    linear(state, "cfm.estimator.input_embed.proj", output, "cfm.input_proj")
    for index in (0, 2):
        take(
            state,
            f"cfm.estimator.input_embed.conv_pos_embed.conv1d.{index}.weight",
            output,
            f"cfm.convpos{index}.w",
        )
        take(
            state,
            f"cfm.estimator.input_embed.conv_pos_embed.conv1d.{index}.bias",
            output,
            f"cfm.convpos{index}.b",
        )
    take(state, "cfm.estimator.rotary_embed.inv_freq", output, "cfm.rotary.inv_freq")
    ada_names = ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
    for index in range(22):
        base = f"cfm.estimator.transformer_blocks.{index}"
        weight = take(state, f"{base}.attn_norm.linear.weight", output, f"_tmp.ada.{index}.w")
        bias = take(state, f"{base}.attn_norm.linear.bias", output, f"_tmp.ada.{index}.b")
        for offset, tag in enumerate(ada_names):
            output[f"cfm.blocks.{index}.ada.{tag}.w"] = weight[
                offset * 1024 : (offset + 1) * 1024
            ].t().contiguous()
            output[f"cfm.blocks.{index}.ada.{tag}.b"] = bias[
                offset * 1024 : (offset + 1) * 1024
            ].contiguous()
        for tag in ("q", "k", "v"):
            linear(state, f"{base}.attn.to_{tag}", output, f"cfm.blocks.{index}.{tag}")
        linear(state, f"{base}.attn.to_out.0", output, f"cfm.blocks.{index}.o")
        linear(state, f"{base}.ff.ff.0.0", output, f"cfm.blocks.{index}.ff1")
        linear(state, f"{base}.ff.ff.2", output, f"cfm.blocks.{index}.ff2")
    final_weight = take(state, "cfm.estimator.norm_out.linear.weight", output, "_tmp.final.w")
    final_bias = take(state, "cfm.estimator.norm_out.linear.bias", output, "_tmp.final.b")
    output["cfm.final.scale.w"] = final_weight[:1024].t().contiguous()
    output["cfm.final.scale.b"] = final_bias[:1024].contiguous()
    output["cfm.final.shift.w"] = final_weight[1024:].t().contiguous()
    output["cfm.final.shift.b"] = final_bias[1024:].contiguous()
    linear(state, "cfm.estimator.proj_out", output, "cfm.proj_out")

    for key in list(output):
        if key.startswith("_tmp."):
            del output[key]
    return output


def convert_vocoder(state: dict) -> dict:
    output: dict = {}
    conv(state, "conv_pre", output, "vocoder.conv_pre")
    for index in range(5):
        conv(state, f"ups.{index}", output, f"vocoder.ups.{index}")
    for index in range(15):
        for branch in ("convs1", "convs2"):
            for kernel in range(3):
                conv(
                    state,
                    f"resblocks.{index}.{branch}.{kernel}",
                    output,
                    f"vocoder.res.{index}.{branch}.{kernel}",
                )
    conv(state, "conv_post", output, "vocoder.conv_post")
    return output


def convert_hubert(state: dict) -> dict:
    output: dict = {}
    for index in range(7):
        conv(state, f"feature_extractor.conv_layers.{index}.conv", output, f"hubert.feat.conv{index}")
    take(state, "feature_extractor.conv_layers.0.layer_norm.weight", output, "hubert.feat.norm0.w")
    take(state, "feature_extractor.conv_layers.0.layer_norm.bias", output, "hubert.feat.norm0.b")
    take(state, "feature_projection.layer_norm.weight", output, "hubert.feat_proj.ln.w")
    take(state, "feature_projection.layer_norm.bias", output, "hubert.feat_proj.ln.b")
    linear(state, "feature_projection.projection", output, "hubert.feat_proj.proj")
    weight = weight_norm_weight(state, "encoder.pos_conv_embed.conv", 2)
    take(state, "encoder.pos_conv_embed.conv.bias", output, "hubert.posconv.b")
    output["hubert.posconv.w"] = weight.contiguous()
    take(state, "encoder.layer_norm.weight", output, "hubert.enc_ln.w")
    take(state, "encoder.layer_norm.bias", output, "hubert.enc_ln.b")
    for index in range(12):
        base = f"encoder.layers.{index}"
        for tag in ("q", "k", "v"):
            linear(state, f"{base}.attention.{tag}_proj", output, f"hubert.layers.{index}.{tag}")
        linear(state, f"{base}.attention.out_proj", output, f"hubert.layers.{index}.o")
        take(state, f"{base}.layer_norm.weight", output, f"hubert.layers.{index}.ln.w")
        take(state, f"{base}.layer_norm.bias", output, f"hubert.layers.{index}.ln.b")
        linear(state, f"{base}.feed_forward.intermediate_dense", output, f"hubert.layers.{index}.ff1")
        linear(state, f"{base}.feed_forward.output_dense", output, f"hubert.layers.{index}.ff2")
        take(state, f"{base}.final_layer_norm.weight", output, f"hubert.layers.{index}.out_ln.w")
        take(state, f"{base}.final_layer_norm.bias", output, f"hubert.layers.{index}.out_ln.b")
    return output


def convert_roberta(state: dict) -> dict:
    output: dict = {}
    take(state, "bert.embeddings.word_embeddings.weight", output, "roberta.word_emb")
    take(state, "bert.embeddings.position_embeddings.weight", output, "roberta.pos_emb")
    take(state, "bert.embeddings.token_type_embeddings.weight", output, "roberta.type_emb")
    take(state, "bert.embeddings.LayerNorm.weight", output, "roberta.emb_ln.w")
    take(state, "bert.embeddings.LayerNorm.bias", output, "roberta.emb_ln.b")
    for index in range(24):
        base = f"bert.encoder.layer.{index}"
        for tag, source in (("q", "query"), ("k", "key"), ("v", "value")):
            linear(state, f"{base}.attention.self.{source}", output, f"roberta.layers.{index}.{tag}")
        linear(state, f"{base}.attention.output.dense", output, f"roberta.layers.{index}.o")
        take(state, f"{base}.attention.output.LayerNorm.weight", output, f"roberta.layers.{index}.ln.w")
        take(state, f"{base}.attention.output.LayerNorm.bias", output, f"roberta.layers.{index}.ln.b")
        linear(state, f"{base}.intermediate.dense", output, f"roberta.layers.{index}.ff1")
        linear(state, f"{base}.output.dense", output, f"roberta.layers.{index}.ff2")
        take(state, f"{base}.output.LayerNorm.weight", output, f"roberta.layers.{index}.out_ln.w")
        take(state, f"{base}.output.LayerNorm.bias", output, f"roberta.layers.{index}.out_ln.b")
    return output


def save(output: dict, path: Path) -> None:
    fp16 = {}
    for key, tensor in output.items():
        if tensor.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            fp16[key] = tensor.to(torch.float16).contiguous()
        else:
            fp16[key] = tensor.contiguous()
    save_file(fp16, str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt", required=True, type=Path)
    parser.add_argument("--sovits", required=True, type=Path)
    parser.add_argument("--vocoder", required=True, type=Path)
    parser.add_argument("--hubert", required=True, type=Path)
    parser.add_argument("--roberta", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    inputs = {
        "gpt": args.gpt,
        "sovits": args.sovits,
        "vocoder": args.vocoder,
        "hubert": args.hubert,
        "roberta": args.roberta,
    }
    manifest = {
        "format": "apxinf-gpt-sovits-v5",
        "source_sha256": {name: sha256(path) for name, path in inputs.items()},
        "dtype": "float16",
        "version": "v5turbo",
    }

    components = {
        "gpt": convert_gpt(load_state(args.gpt)),
        "sovits": convert_sovits(load_state(args.sovits)),
        "vocoder": convert_vocoder(load_state(args.vocoder)),
        "hubert": convert_hubert(load_state(args.hubert)),
        "roberta": convert_roberta(load_state(args.roberta)),
    }
    for name, tensors in components.items():
        save(tensors, args.out / f"{name}.safetensors")
        print(f"{name}: {len(tensors)} tensors")

    config = {
        "model_type": "gpt_sovits",
        "architecture": "gpt-sovits-v5",
        "variant": "v5turbo",
        "sample_rate": 48000,
        "mel_frames_per_second": 100,
        "semantic_frames_per_second": 25,
        "gpt": {
            "hidden": 512,
            "heads": 16,
            "layers": 24,
            "ff": 2048,
            "phone_vocab": 732,
            "audio_vocab": 1025,
            "eos": 1024,
            "bert_dim": 1024,
        },
        "hubert": {"layers": 12, "hidden": 768, "heads": 12, "ff": 3072,
                   "conv_dim": 512, "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
                   "conv_stride": [5, 2, 2, 2, 2, 2, 2], "pos_kernel": 128,
                   "pos_groups": 16},
        "roberta": {"layers": 24, "hidden": 1024, "heads": 16, "ff": 4096,
                    "vocab": 21128, "output_hidden": -3},
        "sovits": {"inter": 192, "hidden": 192, "heads": 2, "layers": 6,
                   "ff": 768, "window": 4, "wn_layers": 8, "wn_kernel": 5},
        "cfm": {"dim": 1024, "depth": 22, "heads": 16, "head_dim": 64,
                "ff": 2048, "mel": 100, "text": 512, "text_conv_layers": 4,
                "steps": 4, "cfg": 0.0},
        "vocoder": {"initial_channel": 100, "upsample_rates": [10, 6, 2, 2, 2],
                    "upsample_kernels": [20, 12, 4, 4, 4],
                    "resblock_kernels": [3, 7, 11],
                    "resblock_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]]},
        "assets": {f"{name}.safetensors": f"{name}.safetensors" for name in components},
    }
    with open(args.out / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with open(args.out / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

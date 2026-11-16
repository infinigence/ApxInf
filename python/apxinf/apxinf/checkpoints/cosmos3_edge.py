"""Prepare NVIDIA Cosmos3-Edge-Policy-DROID assets for native loading.

Run ``python -m apxinf.checkpoints.cosmos3_edge --assets DIR --output DIR``.
Conversion is offline; inference does not import torch or the reference model.

Source layout (HuggingFace diffusers bundle):
  transformer/diffusion_pytorch_model-*-of-*.safetensors  (Mixture-of-Transformers:
      UND tower ``layers.N.{self_attn.to_*, mlp, input_layernorm}`` and GEN tower
      ``layers.N.{self_attn.add_*, to_add_out, mlp_moe_gen, *_moe_gen}``)
  vae/diffusion_pytorch_model.safetensors  (Wan2.2; policy needs ``encoder.*`` +
      ``quant_conv.*`` only)
  text_tokenizer/  (BPE tokenizer + chat template)

Prepared outputs:
  model.safetensors        transformer tensors, original names, BF16
  vae_encoder.safetensors  encoder + quant_conv, BF16
  text.safetensors         ``embeddings`` [num_prompts, S, 2048] gathered from
                           ``embed_tokens`` for the cached prompts
  prompts.json             the cached prompt list (task id order)
  config.json              geometry (written last = completion marker)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SOURCE_REVISION = "a16b2d162bb12245f98ebef4227a1d7fe9c67757"
CHECKPOINT_REVISION = "0110e64ca982589f1a3d856cbdd7d465dccb79bc"
PROFILE = "droid-policy-v1"

DEFAULT_PROMPTS = ["Pick up the banana and place it in the bowl."]


def _check_transformer_geometry(tensors: dict) -> None:
    """Refuse unsupported weights before labeling them as the fixed profile."""
    layers = {int(match.group(1)) for name in tensors
              if (match := re.match(r"layers\.(\d+)\.", name))}
    if layers != set(range(28)):
        raise ValueError("DROID direct requires exactly 28 transformer layers")
    shapes = {
        "embed_tokens.weight": (131072, 2048),
        "proj_in.weight": (2048, 192), "proj_out.weight": (192, 2048),
        "norm.weight": (2048,), "norm_moe_gen.weight": (2048,),
    }
    for i in range(28):
        for suffix, shape in {
            "self_attn.to_q.weight": (2048, 2048),
            "self_attn.to_k.weight": (1024, 2048),
            "self_attn.to_v.weight": (1024, 2048),
            "self_attn.to_out.weight": (2048, 2048),
            "self_attn.add_q_proj.weight": (2048, 2048),
            "self_attn.add_k_proj.weight": (1024, 2048),
            "self_attn.add_v_proj.weight": (1024, 2048),
            "self_attn.to_add_out.weight": (2048, 2048),
            "mlp.up_proj.weight": (9216, 2048),
            "mlp.down_proj.weight": (2048, 9216),
            "mlp_moe_gen.up_proj.weight": (9216, 2048),
            "mlp_moe_gen.down_proj.weight": (2048, 9216),
        }.items():
            shapes[f"layers.{i}.{suffix}"] = shape
    for name, shape in shapes.items():
        if name not in tensors or tuple(tensors[name].shape) != shape:
            raise ValueError(f"DROID direct transformer shape mismatch: {name}, expected {shape}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safetensors_files(directory: Path) -> list[Path]:
    # Prefer the real weight shards; skip metadata bundles such as
    # `cosmos_framework_model.safetensors` that carry prefixed duplicate keys.
    files = sorted(directory.glob("diffusion_pytorch_model*.safetensors"))
    if not files:
        files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors shards under {directory}")
    return files


def _load_shards(directory: Path) -> dict:
    from safetensors.torch import load_file

    merged: dict = {}
    for shard in _safetensors_files(directory):
        tensors = load_file(str(shard), device="cpu")
        duplicate = merged.keys() & tensors.keys()
        if duplicate:
            raise ValueError(f"duplicate tensor keys in {shard}: {sorted(duplicate)}")
        merged.update(tensors)
    return merged


def _check_bf16(tensors: dict, scope: str, allow_f32: tuple[str, ...] = ()) -> None:
    import torch

    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{scope}: non-tensor entry {name}")
        if tensor.dtype == torch.float32 and name.startswith(allow_f32):
            pass
        elif tensor.dtype != torch.bfloat16:
            raise ValueError(f"{scope}: expected BF16, got {tensor.dtype} for {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{scope}: non-finite tensor {name}")


def _build_policy_prompt(instruction, viewpoint="ego_view", action_chunk_size=16,
                         fps=5.0, height=544, width=736):
    """Reproduce the Edge-LLM buildPolicyPrompt JSON framing exactly."""
    import json
    sentence = instruction.rstrip()
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    end_seconds = action_chunk_size / fps
    time_end = int(round(end_seconds))
    duration = int(end_seconds)
    fps_text = f"{fps:.1f}"
    framing = {
        "ego_view": "This video is captured from a first-person perspective looking at the scene.",
        "third_person_view": "This video is captured from a third-person perspective looking towards the agent from the front.",
        "wrist_view": "This video is captured from a wrist-mounted camera.",
        "concat_view": "This video contains concatenated views from multiple camera perspectives.",
    }[viewpoint]
    time_str = f"{time_end // 60}:{time_end % 60:02d}"
    aspect = {(736, 544): "4,3", (832, 480): "16,9"}.get((width, height), f"{width},{height}")
    return ('{"cinematography": {"framing": ' + json.dumps(framing)
            + '}, "actions": [{"time": ' + json.dumps("0:00-" + time_str)
            + ', "description": ' + json.dumps(sentence) + '}], "duration": "' + f"{duration}s"
            + '", "fps": ' + fps_text
            + ', "resolution": {"H": ' + str(height) + ', "W": ' + str(width)
            + '}, "aspect_ratio": ' + json.dumps(aspect) + '}')

def _token_embeddings(assets: Path, transformer: dict, prompts: list[str]) -> dict:
    """Gather ``embed_tokens`` rows for each prompt's chat-templated token ids."""
    import torch
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(assets / "text_tokenizer" / "tokenizer.json"))
    if tokenizer.chat_template is None:
        template = (assets / "text_tokenizer" / "chat_template.jinja")
        if template.exists():
            tokenizer.chat_template = template.read_text()

    embed = transformer["embed_tokens.weight"]
    rows = []
    lengths = []
    for prompt in prompts:
        content = _build_policy_prompt(prompt)
        messages = [{"role": "user", "content": content}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=False, enable_thinking=True)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        ids += [eos_id]
        ids += tokenizer.encode("<|vision_start|>", add_special_tokens=False)
        rows.append(embed[torch.tensor(ids, dtype=torch.long)].contiguous())
        lengths.append(len(ids))
    if lengths != [126]:
        raise ValueError(f"DROID direct requires one 126-token cached prompt, got {lengths}")
    return {"embeddings": torch.stack(rows, dim=0)}


def convert(assets: Path, output: Path, prompts: list[str] | None = None) -> dict:
    import torch
    from safetensors.torch import save_file

    prompts = DEFAULT_PROMPTS if prompts is None else prompts
    if (len(prompts) != 1 or not isinstance(prompts[0], str)
            or not prompts[0].strip()):
        raise ValueError("DROID direct profile requires one nonempty cached prompt")
    output.mkdir(parents=True, exist_ok=False)

    transformer = _load_shards(assets / "transformer")
    _check_transformer_geometry(transformer)
    _check_bf16(transformer, "transformer", allow_f32=("time_embedder.",))
    if "embed_tokens.weight" not in transformer or "layers.0.self_attn.to_q.weight" not in transformer:
        raise ValueError("transformer shards missing expected UND keys")
    if "layers.0.self_attn.add_q_proj.weight" not in transformer:
        raise ValueError("transformer shards missing expected GEN keys")
    # Select the DROID embodiment row (domain_id=8) from the DomainAwareLinear
    # action tables and expose plain [out,in] matrices; convert the F32 timestep
    # embedder to BF16. Drop reasoner-only lm_head and the raw token table (raw
    # text embeddings are pre-gathered from embed_tokens below).
    domain = 8
    prepared = {
        name: tensor.contiguous()
        for name, tensor in transformer.items()
        if name not in {"lm_head.weight", "embed_tokens.weight",
                        "action_proj_in.fc.weight", "action_proj_in.bias.weight",
                        "action_proj_out.fc.weight", "action_proj_out.bias.weight",
                        "cosmos_framework_model"}
    }
    prepared["action_in.weight"] = transformer["action_proj_in.fc.weight"][domain].reshape(2048, 64).contiguous()
    prepared["action_in.bias"] = transformer["action_proj_in.bias.weight"][domain].reshape(2048).contiguous()
    prepared["action_out.weight"] = transformer["action_proj_out.fc.weight"][domain].reshape(64, 2048).contiguous()
    prepared["action_out.bias"] = transformer["action_proj_out.bias.weight"][domain].reshape(64).contiguous()
    for name in ("time_embedder.linear_1.weight", "time_embedder.linear_1.bias",
                 "time_embedder.linear_2.weight", "time_embedder.linear_2.bias"):
        prepared[name] = transformer[name].to(torch.bfloat16).contiguous()
    _check_bf16(prepared, "prepared")
    save_file(prepared, str(output / "model.safetensors"))

    vae = _load_shards(assets / "vae")
    encoder = {}
    for name, tensor in vae.items():
        if name.startswith(("encoder.", "quant_conv.")):
            encoder[name] = tensor.to(torch.bfloat16).contiguous()
        elif not name.startswith(("decoder.", "post_quant_conv.")):
            raise ValueError(f"unexpected VAE entry: {name}")
    _check_bf16(encoder, "vae_encoder")
    save_file(encoder, str(output / "vae_encoder.safetensors"))

    text = _token_embeddings(assets, transformer, prompts)
    _check_bf16(text, "text")
    save_file(text, str(output / "text.safetensors"))
    (output / "prompts.json").write_text(json.dumps({"prompts": prompts}, indent=2) + "\n")

    manifest = {
        "source_revision": SOURCE_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_sha256": {str(p.relative_to(assets)): sha256(p)
                          for p in sorted(assets.rglob("*.safetensors"))},
        "vae_scope": "encoder + quant_conv; decoder omitted (direct policy)",
        "text": "chat-templated token ids gathered from embed_tokens",
        "output_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")

    config = {
        "model_type": "cosmos3-edge", "profile": PROFILE, "format_version": 1,
        "source_revision": SOURCE_REVISION, "checkpoint_revision": CHECKPOINT_REVISION,
        "precision": "bf16", "num_tasks": len(prompts),
        "hidden_size": 2048, "num_layers": 28, "num_heads": 16, "num_kv_heads": 8,
        "head_dim": 128, "intermediate_size": 9216, "rms_norm_eps": 1e-5,
        "rope_theta": 1e8, "mrope_section": [24, 20, 20],
        "latent_channel": 48, "latent_patch_size": 2, "latent_shape": [48, 5, 34, 46],
        "action_chunk": 16, "raw_action_dim": 10, "max_action_dim": 64,
        "num_inference_steps": 4, "flow_shift": 5.0, "num_train_timesteps": 1000,
        "num_embodiment_domains": 32, "domain_id": 8,
        "vocab_size": 131072, "max_text_tokens": int(text["embeddings"].shape[1]),
        "image_size": [544, 736], "pixel_frames": 1,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=None,
                        help="optional JSON file with one 126-token cached prompt: {'prompts': [...]}")
    args = parser.parse_args()
    prompts = None
    if args.prompts is not None:
        prompts = json.loads(args.prompts.read_text())["prompts"]
    convert(args.assets, args.output, prompts)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Dump HuggingFace reference activations + greedy token IDs for the apxinf
Qwen3-VL test suite.

Produces .npz files under tests/qwen3vl_reference/ that the apxinf unit tests
diff against their own outputs. Every dump is deterministic and uses bf16 on
CUDA (matches the apxinf runtime dtype).

For each prompt we capture:
    - tokens              : input token IDs (int64)
    - post_embedding_last : hidden state after embed_tokens, last position
    - hidden_L{i}_last    : per-layer hidden state (last position) at
                            layers 0 / mid / last
    - post_final_norm_last: hidden state after final RMSNorm, last position
    - logits_last         : full logits at the last position (fp32)
    - greedy_tokens       : first 10 greedy token IDs generated after the
                            prompt

Vision prompts additionally capture:
    - image_pixel_values  : the preprocessed pixel tensor fed to the vision
                            tower (matches HF processor output)
    - image_grid_thw      : the [T, H, W] grid describing image_pixel_values
    - vision_primary      : the primary visual embedding sequence (2048-d)
                            that gets injected at the image_pad positions
    - vision_deepstack_{k} : k=0,1,2 - deepstack embeddings for injection at
                             the 3 LLM layer depths

Usage:
    pip install transformers torch accelerate pillow
    python scripts/hf_reference_dump.py               # all prompts
    python scripts/hf_reference_dump.py --only tinyllama_text
    python scripts/hf_reference_dump.py --only qwen3vl_text
    python scripts/hf_reference_dump.py --only qwen3vl_image

Not part of the Rust build. Run locally to produce reference files, commit
them (or ignore, see comment in tests/qwen3vl_reference/README.md), then
Rust tests consume them via numpy npz format.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "tests" / "qwen3vl_reference"

TINYLLAMA_PATH = "/mnt/user_dir/hanjinchen/models/TinyLlama-1.1B-Chat-v1.0"
QWEN3VL_PATH = "/hanjinchen/models/Qwen3-VL-2B-Instruct"

TEXT_PROMPT = "The capital of Canada is"
IMAGE_PROMPT = "Describe this image in one short sentence."


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def deterministic_test_image(size: int = 336):
    """A small fixed 336x336 test image.

    Fully deterministic - a simple gradient with a red square in the middle
    so the model has *something* to describe but the pixel values are
    reproducible without needing to ship a PNG in the repo.
    """
    from PIL import Image

    arr = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    arr[..., 0] = (xx * 255 // (size - 1)).astype(np.uint8)          # R gradient
    arr[..., 1] = (yy * 255 // (size - 1)).astype(np.uint8)          # G gradient
    arr[..., 2] = ((xx + yy) * 255 // (2 * (size - 1))).astype(np.uint8)
    q = size // 4
    arr[q:3 * q, q:3 * q, 0] = 220
    arr[q:3 * q, q:3 * q, 1] = 20
    arr[q:3 * q, q:3 * q, 2] = 60
    return Image.fromarray(arr, mode="RGB")


class HiddenCatcher:
    """Register hooks on decoder layers + final norm + embedding.

    Records only the last-position slice (keeps files small: ~4-8 KB / layer
    for hidden=2048). We snapshot on the first (prompt) forward pass only.
    """

    def __init__(self, model, layer_indexes, embed_module, final_norm_module,
                 decoder_layers):
        self.model = model
        self.layer_indexes = set(layer_indexes)
        self.hooks = []
        self.captured = {}
        self._snapshot_taken = False

        def snap_embed(_mod, _in, out):
            if self._snapshot_taken:
                return
            self.captured["post_embedding_last"] = out[0, -1, :].detach().float().cpu().numpy()

        def snap_layer(idx):
            def hook(_mod, _in, out):
                if self._snapshot_taken:
                    return
                # decoder layer output is a tuple (hidden_states, ...)
                hs = out[0] if isinstance(out, tuple) else out
                self.captured[f"hidden_L{idx}_last"] = hs[0, -1, :].detach().float().cpu().numpy()
            return hook

        def snap_final(_mod, _in, out):
            if self._snapshot_taken:
                return
            hs = out[0] if isinstance(out, tuple) else out
            self.captured["post_final_norm_last"] = hs[0, -1, :].detach().float().cpu().numpy()

        self.hooks.append(embed_module.register_forward_hook(snap_embed))
        for i, layer in enumerate(decoder_layers):
            if i in self.layer_indexes:
                self.hooks.append(layer.register_forward_hook(snap_layer(i)))
        self.hooks.append(final_norm_module.register_forward_hook(snap_final))

    def freeze(self):
        self._snapshot_taken = True

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def greedy_generate(model, input_ids, n_new, extra_forward_kwargs=None):
    """Greedy decode using the standard next-token loop.

    We roll our own instead of model.generate() so the exact same forward
    that apxinf does (single-batch, greedy, no sampling temperature) is used.
    """
    extra_forward_kwargs = extra_forward_kwargs or {}
    tokens = input_ids.clone()
    generated = []
    past_key_values = None

    for step in range(n_new):
        if step == 0:
            out = model(tokens, use_cache=True, **extra_forward_kwargs)
        else:
            out = model(
                tokens[:, -1:],
                use_cache=True,
                past_key_values=past_key_values,
            )
        past_key_values = out.past_key_values
        next_id = int(out.logits[0, -1, :].argmax().item())
        generated.append(next_id)
        tokens = torch.cat([tokens, torch.tensor([[next_id]], device=tokens.device)], dim=1)
    return generated


def choose_layer_indexes(n_layers):
    return sorted({0, n_layers // 2, n_layers - 1})


# ---------- TinyLlama text ----------

def dump_tinyllama_text(out_path: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[tinyllama_text] loading {TINYLLAMA_PATH} (bf16, cuda)")
    tok = AutoTokenizer.from_pretrained(TINYLLAMA_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        TINYLLAMA_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda").eval()

    n_layers = model.config.num_hidden_layers
    layer_indexes = choose_layer_indexes(n_layers)
    print(f"[tinyllama_text] layers={n_layers}, capturing indexes {layer_indexes}")

    input_ids = tok(TEXT_PROMPT, return_tensors="pt", add_special_tokens=True).input_ids.to("cuda")

    catcher = HiddenCatcher(
        model,
        layer_indexes,
        embed_module=model.model.embed_tokens,
        final_norm_module=model.model.norm,
        decoder_layers=model.model.layers,
    )
    with torch.no_grad():
        prompt_out = model(input_ids, use_cache=True)
    catcher.freeze()
    logits_last = prompt_out.logits[0, -1, :].detach().float().cpu().numpy()
    catcher.close()

    with torch.no_grad():
        greedy = greedy_generate(model, input_ids, n_new=10)

    payload = dict(catcher.captured)
    payload["tokens"] = input_ids[0].detach().cpu().numpy().astype(np.int64)
    payload["logits_last"] = logits_last
    payload["greedy_tokens"] = np.array(greedy, dtype=np.int64)
    payload["layer_indexes"] = np.array(sorted(layer_indexes), dtype=np.int64)
    payload["prompt"] = np.array(TEXT_PROMPT)
    payload["decoded"] = np.array(tok.decode(greedy))

    np.savez(out_path, **payload)
    print(f"[tinyllama_text] wrote {out_path}")
    print(f"[tinyllama_text] greedy tokens: {greedy}")
    print(f"[tinyllama_text] decoded: {tok.decode(greedy)!r}")

    del model
    torch.cuda.empty_cache()


# ---------- Qwen3-VL text-only ----------

def dump_qwen3vl_text(out_path: Path):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[qwen3vl_text] loading {QWEN3VL_PATH} (bf16, cuda)")
    processor = AutoProcessor.from_pretrained(QWEN3VL_PATH, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        QWEN3VL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda").eval()

    text_cfg = model.config.text_config
    n_layers = text_cfg.num_hidden_layers
    layer_indexes = choose_layer_indexes(n_layers)
    print(f"[qwen3vl_text] layers={n_layers}, capturing indexes {layer_indexes}")

    # Text-only chat message (no image) - tests the text stack of Qwen3-VL
    # against HF before any vision code exists.
    messages = [{"role": "user", "content": [{"type": "text", "text": TEXT_PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    input_ids = inputs["input_ids"]

    lm = model.model.language_model  # Qwen3VLTextModel

    catcher = HiddenCatcher(
        model,
        layer_indexes,
        embed_module=lm.embed_tokens,
        final_norm_module=lm.norm,
        decoder_layers=lm.layers,
    )
    with torch.no_grad():
        prompt_out = model(**inputs, use_cache=True)
    catcher.freeze()
    logits_last = prompt_out.logits[0, -1, :].detach().float().cpu().numpy()
    catcher.close()

    with torch.no_grad():
        greedy = greedy_generate(model, input_ids, n_new=10)

    payload = dict(catcher.captured)
    payload["tokens"] = input_ids[0].detach().cpu().numpy().astype(np.int64)
    payload["logits_last"] = logits_last
    payload["greedy_tokens"] = np.array(greedy, dtype=np.int64)
    payload["layer_indexes"] = np.array(sorted(layer_indexes), dtype=np.int64)
    payload["prompt"] = np.array(TEXT_PROMPT)
    payload["decoded"] = np.array(processor.decode(greedy))

    np.savez(out_path, **payload)
    print(f"[qwen3vl_text] wrote {out_path}")
    print(f"[qwen3vl_text] greedy tokens: {greedy}")
    print(f"[qwen3vl_text] decoded: {processor.decode(greedy)!r}")

    del model
    torch.cuda.empty_cache()


# ---------- Qwen3-VL text+image ----------

def dump_qwen3vl_image(out_path: Path):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[qwen3vl_image] loading {QWEN3VL_PATH} (bf16, cuda)")
    processor = AutoProcessor.from_pretrained(QWEN3VL_PATH, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        QWEN3VL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda").eval()

    text_cfg = model.config.text_config
    n_layers = text_cfg.num_hidden_layers
    layer_indexes = choose_layer_indexes(n_layers)
    print(f"[qwen3vl_image] layers={n_layers}, capturing indexes {layer_indexes}")

    image = deterministic_test_image(336)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": IMAGE_PROMPT},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    # Capture the pixel tensor + grid the processor produced so apxinf can
    # reproduce them.
    pixel_values = inputs["pixel_values"].detach().float().cpu().numpy()
    image_grid_thw = inputs["image_grid_thw"].detach().cpu().numpy().astype(np.int64)
    input_ids = inputs["input_ids"]

    # Vision tower snapshots: hook the visual module to grab (a) the primary
    # per-image embedding sequence going into the LLM at the image_pad
    # positions, and (b) the 3 deepstack embeddings.
    visual = model.model.visual
    vision_captured = {}

    def snap_visual(_mod, args, out):
        # HF Qwen3VLVisionModel.forward returns (primary_embeds, deepstack_list)
        # in transformers 4.57 series. Accept both tuple layouts.
        if isinstance(out, tuple):
            primary = out[0]
            if len(out) >= 2 and out[1] is not None:
                for k, e in enumerate(out[1]):
                    vision_captured[f"vision_deepstack_{k}"] = e.detach().float().cpu().numpy()
        else:
            primary = out
        vision_captured["vision_primary"] = primary.detach().float().cpu().numpy()

    v_hook = visual.register_forward_hook(snap_visual)

    lm = model.model.language_model
    catcher = HiddenCatcher(
        model,
        layer_indexes,
        embed_module=lm.embed_tokens,
        final_norm_module=lm.norm,
        decoder_layers=lm.layers,
    )
    with torch.no_grad():
        prompt_out = model(**inputs, use_cache=True)
    catcher.freeze()
    logits_last = prompt_out.logits[0, -1, :].detach().float().cpu().numpy()
    catcher.close()
    v_hook.remove()

    with torch.no_grad():
        greedy = greedy_generate(model, input_ids, n_new=10, extra_forward_kwargs={
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        })

    payload = dict(catcher.captured)
    payload.update(vision_captured)
    payload["tokens"] = input_ids[0].detach().cpu().numpy().astype(np.int64)
    payload["logits_last"] = logits_last
    payload["greedy_tokens"] = np.array(greedy, dtype=np.int64)
    payload["layer_indexes"] = np.array(sorted(layer_indexes), dtype=np.int64)
    payload["image_pixel_values"] = pixel_values
    payload["image_grid_thw"] = image_grid_thw
    payload["prompt"] = np.array(IMAGE_PROMPT)
    payload["decoded"] = np.array(processor.decode(greedy))

    np.savez(out_path, **payload)
    print(f"[qwen3vl_image] wrote {out_path}")
    print(f"[qwen3vl_image] pixel_values shape: {pixel_values.shape}")
    print(f"[qwen3vl_image] image_grid_thw: {image_grid_thw.tolist()}")
    print(f"[qwen3vl_image] greedy tokens: {greedy}")
    print(f"[qwen3vl_image] decoded: {processor.decode(greedy)!r}")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["tinyllama_text", "qwen3vl_text", "qwen3vl_image"],
        default=None,
        help="Run only one dump (default: all)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        ("tinyllama_text", OUTPUT_DIR / "tinyllama_text.npz", dump_tinyllama_text),
        ("qwen3vl_text", OUTPUT_DIR / "qwen3vl_text.npz", dump_qwen3vl_text),
        ("qwen3vl_image", OUTPUT_DIR / "qwen3vl_image.npz", dump_qwen3vl_image),
    ]

    for name, path, fn in targets:
        if args.only and args.only != name:
            continue
        print(f"\n=== {name} ===")
        fn(path)


if __name__ == "__main__":
    main()

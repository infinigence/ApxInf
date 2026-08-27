#!/usr/bin/env python3
"""
Dump HuggingFace reference activations + greedy token IDs for ApxInf model
ports. Qwen2.5-Omni targets are offline-only and bind the pinned metadata
files before loading any model payload.

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
    export APXINF_TINYLLAMA_MODEL_DIR=/path/to/TinyLlama-1.1B-Chat-v1.0
    export APXINF_QWEN3VL_MODEL_DIR=/path/to/Qwen3-VL-2B-Instruct
    python scripts/hf_reference_dump.py               # all prompts
    python scripts/hf_reference_dump.py --only tinyllama_text
    python scripts/hf_reference_dump.py --only qwen3vl_text
    python scripts/hf_reference_dump.py --only qwen3vl_image
    python scripts/hf_reference_dump.py --only qwen25_omni_text \
        --qwen25-omni-model-dir "$APXINF_QWEN25_OMNI_MODEL_DIR"

Not part of the Rust build. Run locally to produce reference files, commit
them (or ignore, see comment in tests/qwen3vl_reference/README.md), then
Rust tests consume them via numpy npz format.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "tests" / "qwen3vl_reference"
QWEN25_OMNI_OUTPUT_DIR = REPO_ROOT / "tests" / "qwen25_omni_reference"

TINYLLAMA_MODEL_ENV = "APXINF_TINYLLAMA_MODEL_DIR"
QWEN3VL_MODEL_ENV = "APXINF_QWEN3VL_MODEL_DIR"

QWEN25_OMNI_MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
QWEN25_OMNI_REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
QWEN25_OMNI_METADATA_SHA256 = {
    "config.json": "20790f362c37a1718a3e764f597ae33dfc20177399762018dffaabf8321d4dd1",
    "generation_config.json": "b9acf52072249e0e1850541d51ca4c85a30b9d7e23c2f2093ec0fb4139059ac7",
    "model.safetensors.index.json": "5b7629198e2ef80e37612a491d9bfd71639d2f212632d36d8ab086922e74e129",
    "preprocessor_config.json": "b47055ce61463ce143e9aab741d55c0aa520801a0a5d63be73c5b17cecb6bc69",
    "tokenizer_config.json": "569aa7a9171e36dfff80f0a7550ab0c9c09e46ac840fea5030641417394fb0d2",
}

TEXT_PROMPT = "The capital of Canada is"
IMAGE_PROMPT = "Describe this image in one short sentence."
AUDIO_PROMPT = "Describe the sound in one short sentence."


def require_model_path(variable: str) -> str:
    path = os.environ.get(variable)
    if not path:
        raise RuntimeError(
            f"{variable} is required; set it to the corresponding local model directory"
        )
    return path


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


def deterministic_test_audio(sample_rate: int = 16000, seconds: float = 1.0):
    """A finite, deterministic two-tone mono clip for processor references."""
    sample_count = int(sample_rate * seconds)
    time = np.arange(sample_count, dtype=np.float32) / np.float32(sample_rate)
    audio = 0.20 * np.sin(2.0 * np.pi * 440.0 * time)
    audio += 0.05 * np.sin(2.0 * np.pi * 880.0 * time)
    return audio.astype(np.float32)


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

    model_path = require_model_path(TINYLLAMA_MODEL_ENV)
    print(f"[tinyllama_text] loading {model_path} (bf16, cuda)")
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
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

    model_path = require_model_path(QWEN3VL_MODEL_ENV)
    print(f"[qwen3vl_text] loading {model_path} (bf16, cuda)")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
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

    model_path = require_model_path(QWEN3VL_MODEL_ENV)
    print(f"[qwen3vl_image] loading {model_path} (bf16, cuda)")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
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


# ---------- Qwen2.5-Omni Thinker text/image/audio ----------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_qwen25_omni_snapshot(model_dir: Path) -> None:
    """Fail closed unless the local metadata is the pinned Host snapshot."""
    if not model_dir.is_dir():
        raise ValueError(f"Qwen2.5-Omni model directory does not exist: {model_dir}")
    for name, expected in QWEN25_OMNI_METADATA_SHA256.items():
        path = model_dir / name
        if not path.is_file():
            raise ValueError(f"pinned Qwen2.5-Omni metadata is missing {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"pinned Qwen2.5-Omni metadata hash mismatch for {name}: "
                f"expected {expected}, got {actual}"
            )
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("architectures") != ["Qwen2_5OmniModel"]:
        raise ValueError("Qwen2.5-Omni architecture identity mismatch")
    if config.get("model_type") != "qwen2_5_omni":
        raise ValueError("Qwen2.5-Omni model_type identity mismatch")
    thinker = config.get("thinker_config", {})
    if thinker.get("torch_dtype") != "bfloat16":
        raise ValueError("Qwen2.5-Omni Thinker must be bfloat16")


def first_tensor(value):
    if torch.is_tensor(value):
        return value
    if hasattr(value, "last_hidden_state") and torch.is_tensor(value.last_hidden_state):
        return value.last_hidden_state
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class TowerCatcher:
    """Capture the output plus block 0/mid/last of one media tower."""

    def __init__(self, tower, blocks, prefix):
        if len(blocks) != 32:
            raise ValueError(f"{prefix} tower must have 32 blocks, got {len(blocks)}")
        self.captured = {}
        self.hooks = []

        def snapshot(name):
            def hook(_module, _inputs, output):
                if name in self.captured:
                    return
                tensor = first_tensor(output)
                if tensor is None:
                    raise RuntimeError(f"{name} did not return a tensor")
                self.captured[name] = tensor.detach().float().cpu().numpy()
            return hook

        self.hooks.append(tower.register_forward_hook(snapshot(f"{prefix}_tower_output")))
        for index in choose_layer_indexes(len(blocks)):
            self.hooks.append(
                blocks[index].register_forward_hook(snapshot(f"{prefix}_block_L{index}"))
            )

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def qwen25_omni_messages(case):
    if case == "text":
        return [{"role": "user", "content": [{"type": "text", "text": TEXT_PROMPT}]}]
    if case == "image":
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": deterministic_test_image(336)},
                {"type": "text", "text": IMAGE_PROMPT},
            ],
        }]
    if case == "audio":
        return [{
            "role": "user",
            "content": [
                {"type": "audio", "audio": deterministic_test_audio()},
                {"type": "text", "text": AUDIO_PROMPT},
            ],
        }]
    raise ValueError(f"unsupported Qwen2.5-Omni reference case: {case}")


def processor_tensor_payload(inputs):
    payload = {}
    for name, value in inputs.items():
        if not torch.is_tensor(value):
            continue
        if value.dtype.is_floating_point:
            array = value.detach().float().cpu().numpy()
        else:
            array = value.detach().cpu().numpy()
        payload[f"processor_{name}"] = array
    return payload


def dump_qwen25_omni(case: str, model_dir: Path, out_path: Path):
    from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

    model_dir = model_dir.resolve()
    validate_qwen25_omni_snapshot(model_dir)
    print(
        f"[qwen25_omni_{case}] loading {model_dir} "
        f"({QWEN25_OMNI_MODEL_ID}@{QWEN25_OMNI_REVISION}, bf16, cuda, offline)"
    )
    processor = AutoProcessor.from_pretrained(
        str(model_dir), local_files_only=True, use_fast=False
    )
    wrapper = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    thinker = wrapper.thinker
    if thinker.config.model_type != "qwen2_5_omni_thinker":
        raise ValueError(f"unexpected Thinker model_type: {thinker.config.model_type}")
    if len(thinker.model.layers) != 36:
        raise ValueError(f"Thinker must have 36 text layers, got {len(thinker.model.layers)}")

    messages = qwen25_omni_messages(case)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        sampling_rate=16000,
    ).to("cuda")
    input_ids = inputs["input_ids"]
    layer_indexes = choose_layer_indexes(len(thinker.model.layers))

    catcher = HiddenCatcher(
        thinker,
        layer_indexes,
        embed_module=thinker.model.embed_tokens,
        final_norm_module=thinker.model.norm,
        decoder_layers=thinker.model.layers,
    )
    injected = {}

    def snap_injected(_module, args):
        if "post_media_injection" in injected:
            return
        tensor = first_tensor(args)
        if tensor is None:
            raise RuntimeError("first Thinker layer did not receive hidden states")
        injected["post_media_injection"] = tensor.detach().float().cpu().numpy()

    injection_hook = thinker.model.layers[0].register_forward_pre_hook(snap_injected)
    tower_catcher = None
    if case == "image":
        tower_catcher = TowerCatcher(thinker.visual, thinker.visual.blocks, "vision")
    elif case == "audio":
        tower_catcher = TowerCatcher(
            thinker.audio_tower, thinker.audio_tower.layers, "audio"
        )

    with torch.no_grad():
        prompt_out = thinker(**inputs, use_cache=True, return_dict=True)
    catcher.freeze()
    logits_last = prompt_out.logits[0, -1, :].detach().float().cpu().numpy()
    catcher.close()
    injection_hook.remove()
    tower_payload = {}
    if tower_catcher is not None:
        tower_payload.update(tower_catcher.captured)
        tower_catcher.close()

    # Use the Thinker model's standard greedy generation machinery so its
    # cache_position and multimodal rope_delta propagation remain canonical.
    with torch.no_grad():
        sequences = thinker.generate(
            **inputs,
            do_sample=False,
            min_new_tokens=10,
            max_new_tokens=10,
            use_cache=True,
        )
    expected_length = input_ids.shape[1] + 10
    if sequences.ndim != 2 or sequences.shape != (1, expected_length):
        raise RuntimeError(
            f"Thinker greedy sequence shape {tuple(sequences.shape)} != (1, {expected_length})"
        )
    greedy = sequences[0, input_ids.shape[1]:].detach().cpu().numpy().astype(np.int64)

    payload = dict(catcher.captured)
    payload.update(injected)
    payload.update(tower_payload)
    payload.update(processor_tensor_payload(inputs))
    tokens = input_ids[0].detach().cpu().numpy().astype(np.int64)
    payload["tokens"] = tokens
    payload["logits_last"] = logits_last
    payload["greedy_tokens"] = greedy
    payload["layer_indexes"] = np.array(layer_indexes, dtype=np.int64)
    payload["model_id"] = np.array(QWEN25_OMNI_MODEL_ID)
    payload["model_revision"] = np.array(QWEN25_OMNI_REVISION)
    payload["case"] = np.array(case)
    payload["decoded"] = np.array(processor.decode(greedy.tolist()))

    if case == "image":
        payload["image_pixel_values"] = inputs["pixel_values"].detach().float().cpu().numpy()
        payload["image_grid_thw"] = (
            inputs["image_grid_thw"].detach().cpu().numpy().astype(np.int64)
        )
    elif case == "audio":
        features = inputs["input_features"][0].detach().float().cpu().numpy()
        if features.shape[0] == 128:
            features = features.T
        feature_mask = inputs.get("feature_attention_mask")
        if feature_mask is None:
            valid = features.shape[0]
        else:
            mask_values = feature_mask[0].detach().float().cpu().numpy().reshape(-1)
            valid = int(mask_values[:features.shape[0]].sum())
        if valid <= 0 or valid > features.shape[0]:
            raise RuntimeError(f"invalid processor audio feature length: {valid}")
        token_count = int((tokens == 151646).sum())
        if token_count <= 0:
            raise RuntimeError("processor produced no Qwen2.5-Omni audio placeholders")
        payload["audio_input_features"] = features[:valid].astype(np.float32)
        payload["audio_attention_mask"] = np.ones((valid,), dtype=np.float32)
        payload["audio_feature_lengths"] = np.array([valid], dtype=np.int64)
        payload["audio_token_counts"] = np.array([token_count], dtype=np.int64)

    np.savez(out_path, **payload)
    print(f"[qwen25_omni_{case}] wrote {out_path}")
    print(f"[qwen25_omni_{case}] prompt tokens: {len(tokens)}")
    print(f"[qwen25_omni_{case}] greedy tokens: {greedy.tolist()}")
    print(f"[qwen25_omni_{case}] decoded: {processor.decode(greedy.tolist())!r}")

    del wrapper
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=[
            "tinyllama_text",
            "qwen3vl_text",
            "qwen3vl_image",
            "qwen25_omni_text",
            "qwen25_omni_image",
            "qwen25_omni_audio",
        ],
        default=None,
        help="Run only one dump (default: all)",
    )
    parser.add_argument(
        "--qwen25-omni-model-dir",
        type=Path,
        default=None,
        help="Local pinned Qwen2.5-Omni snapshot; never resolved over the network",
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

    qwen25_requested = args.only is not None and args.only.startswith("qwen25_omni_")
    if qwen25_requested and args.qwen25_omni_model_dir is None:
        parser.error("Qwen2.5-Omni targets require --qwen25-omni-model-dir")
    if args.qwen25_omni_model_dir is not None:
        QWEN25_OMNI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for case in ("text", "image", "audio"):
            name = f"qwen25_omni_{case}"
            path = QWEN25_OMNI_OUTPUT_DIR / f"{name}.npz"
            targets.append(
                (
                    name,
                    path,
                    lambda output, selected=case: dump_qwen25_omni(
                        selected, args.qwen25_omni_model_dir, output
                    ),
                )
            )

    for name, path, fn in targets:
        if args.only and args.only != name:
            continue
        print(f"\n=== {name} ===")
        fn(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PyTorch reference script to capture TinyLlama activations for comparison with apxinf.

This script:
1. Loads TinyLlama with transformers library
2. Runs inference with temperature=0 (greedy, same as apxinf)
3. Hooks the forward pass to capture activations at each layer
4. Saves activations to NPZ for comparison

Usage:
    python pytorch_reference.py --prompt "Hello" --output pytorch_debug.npz --position 0
"""

import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


def capture_activations(model, position_to_capture):
    """
    Register forward hooks to capture activations.

    Returns a dict of activation_name -> (data, shape)
    """
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            # Handle tuple outputs (some layers return tuples)
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output

            # Detach and convert to numpy
            if hasattr(tensor, 'detach'):
                data = tensor.detach().cpu().numpy().flatten().tolist()
                shape = list(tensor.shape)
                activations[name] = (data, shape)
        return hook

    hooks = []

    # Hook embedding
    hooks.append(model.model.embed_tokens.register_forward_hook(
        make_hook("embed.token")
    ))

    # Hook each layer
    for i, layer in enumerate(model.model.layers):
        prefix = f"layer.{i}"

        # Input layernorm (before attention)
        hooks.append(layer.input_layernorm.register_forward_hook(
            make_hook(f"{prefix}.norm_attn.output")
        ))

        # Self attention projections
        hooks.append(layer.self_attn.q_proj.register_forward_hook(
            make_hook(f"{prefix}.attn.q_proj_output")
        ))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(
            make_hook(f"{prefix}.attn.k_proj_output")
        ))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(
            make_hook(f"{prefix}.attn.v_proj_output")
        ))

        # Attention output projection
        hooks.append(layer.self_attn.o_proj.register_forward_hook(
            make_hook(f"{prefix}.attn.proj_output")
        ))

        # Post attention layernorm (before FFN)
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            make_hook(f"{prefix}.norm_ffn.output")
        ))

        # MLP
        hooks.append(layer.mlp.gate_proj.register_forward_hook(
            make_hook(f"{prefix}.ffn.gate_proj_output")
        ))
        hooks.append(layer.mlp.up_proj.register_forward_hook(
            make_hook(f"{prefix}.ffn.up_proj_output")
        ))
        hooks.append(layer.mlp.down_proj.register_forward_hook(
            make_hook(f"{prefix}.ffn.output")
        ))

    # Final norm
    hooks.append(model.model.norm.register_forward_hook(
        make_hook("final.norm.output")
    ))

    # LM head (output logits)
    hooks.append(model.lm_head.register_forward_hook(
        make_hook("final.logits")
    ))

    return activations, hooks


def run_inference(model, tokenizer, prompt, position_to_capture, max_new_tokens=1, token_ids=None):
    """
    Run inference and capture activations at specified position.

    If token_ids is provided, use those directly instead of tokenizing the prompt.
    """
    activations, hooks = capture_activations(model, position_to_capture)

    if token_ids is not None:
        # Use provided token IDs directly
        input_ids = torch.tensor([token_ids])
        print(f"Using provided token IDs: {token_ids}")
    else:
        # Apply chat template (same as apxinf)
        messages = [{"role": "user", "content": prompt}]

        # Use the tokenizer's chat template if available
        if tokenizer.chat_template is not None:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        print(f"Formatted prompt:\n{formatted_prompt}")

        # Tokenize - add_special_tokens=False to match apxinf behavior
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"]
        print(f"Input tokens: {input_ids[0].tolist()}")

    print(f"Input length: {input_ids.shape[1]}")

    # Run a single forward pass to capture activations
    # We only capture at the specified position (token index)
    with torch.no_grad():
        # For position 0, we run forward on just the first token
        # For position N, we run forward on tokens 0..N+1

        tokens_to_process = input_ids[:, :position_to_capture + 1]
        print(f"Processing tokens up to position {position_to_capture}: {tokens_to_process[0].tolist()}")

        # Clear activations from any previous runs
        activations.clear()

        # Run forward pass
        outputs = model(tokens_to_process)
        logits = outputs.logits

        # Get predicted next token (greedy)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()
        print(f"Predicted next token: {next_token}")
        print(f"Predicted text: '{tokenizer.decode([next_token])}'")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations


def save_npz(activations, output_path):
    """
    Save activations to NPZ file (compatible with apxinf format).
    """
    # Convert to numpy arrays
    arrays = {}
    for name, (data, shape) in activations.items():
        arr = np.array(data, dtype=np.float32).reshape(shape)
        arrays[name] = arr

    np.savez(output_path, **arrays)
    print(f"Saved {len(activations)} activations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch TinyLlama activation capture")
    parser.add_argument("--model", default="models/tinyllama",
                        help="Model name or path")
    parser.add_argument("--prompt", help="Input prompt (ignored if --tokens is provided)")
    parser.add_argument("--tokens", help="Comma-separated list of token IDs to use directly")
    parser.add_argument("--output", default=None, help="Output NPZ file (if not given, no file is saved)")
    parser.add_argument("--position", type=int, default=0,
                        help="Position to capture activations for")
    parser.add_argument("--max-tokens", type=int, default=1,
                        help="Maximum new tokens to generate")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        local_files_only=True,
    )

    print(f"Model config: hidden={model.config.hidden_size}, "
          f"layers={model.config.num_hidden_layers}, "
          f"heads={model.config.num_attention_heads}")

    # Parse token IDs if provided
    token_ids = None
    if args.tokens:
        token_ids = [int(t.strip()) for t in args.tokens.split(",")]
        print(f"Using token IDs: {token_ids}")

    print(f"\nRunning inference for position {args.position}")
    activations = run_inference(
        model, tokenizer, args.prompt or "",
        position_to_capture=args.position,
        max_new_tokens=args.max_tokens,
        token_ids=token_ids
    )

    if args.output:
        save_npz(activations, args.output)

    # Print captured activation names
    print("\nCaptured activations:")
    for name, (data, shape) in sorted(activations.items()):
        print(f"  {name}: shape={shape}")


if __name__ == "__main__":
    main()
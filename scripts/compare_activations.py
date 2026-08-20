#!/usr/bin/env python3
"""
Compare activations between apxinf and PyTorch reference.

Usage:
    python compare_activations.py --apxinf debug.npz --pytorch pytorch_debug.npz
"""

import argparse
import numpy as np
from pathlib import Path


# Mapping from apxinf activation names to PyTorch names
NAME_MAPPING = {
    "embed.token": "embed.token",
    "final.norm.output": "final.norm.output",
    "final.logits": "final.logits",
}


def get_pytorch_name_for_layer(layer_idx, apxinf_name):
    """Map apxinf layer activation name to PyTorch equivalent."""
    # RMSNorm outputs
    if apxinf_name.endswith(".norm_attn.output"):
        return f"layer.{layer_idx}.norm_attn.output"
    if apxinf_name.endswith(".norm_ffn.output"):
        return f"layer.{layer_idx}.norm_ffn.output"

    # Attention projections (after matmul)
    if apxinf_name.endswith(".attn.q"):
        return f"layer.{layer_idx}.attn.q_proj_output"
    if apxinf_name.endswith(".attn.k"):
        return f"layer.{layer_idx}.attn.k_proj_output"
    if apxinf_name.endswith(".attn.v"):
        return f"layer.{layer_idx}.attn.v_proj_output"
    if apxinf_name.endswith(".attn.proj_output"):
        return f"layer.{layer_idx}.attn.proj_output"

    # MLP
    if apxinf_name.endswith(".ffn.gate"):
        return f"layer.{layer_idx}.ffn.gate_proj_output"
    if apxinf_name.endswith(".ffn.up"):
        return f"layer.{layer_idx}.ffn.up_proj_output"
    if apxinf_name.endswith(".ffn.output"):
        return f"layer.{layer_idx}.ffn.output"

    return None


def compare_activations(apxinf_path, pytorch_path, tolerance=1e-4):
    """
    Load and compare two NPZ files of activations.
    """
    apxinf_data = np.load(apxinf_path)
    pytorch_data = np.load(pytorch_path)

    apxinf_keys = sorted(apxinf_data.files)
    pytorch_keys = sorted(pytorch_data.files)

    print(f"ApxInf activations: {len(apxinf_keys)}")
    print(f"PyTorch activations: {len(pytorch_keys)}")

    # Build name mapping for layer activations
    apxinf_to_pytorch = {}
    for key in apxinf_keys:
        if key.startswith("layer."):
            parts = key.split(".")
            if len(parts) >= 3:
                layer_idx = int(parts[1])
                rest = ".".join(parts[2:])
                pytorch_name = get_pytorch_name_for_layer(layer_idx, f"layer.{rest}")
                if pytorch_name:
                    apxinf_to_pytorch[key] = pytorch_name
        elif key in NAME_MAPPING:
            apxinf_to_pytorch[key] = NAME_MAPPING[key]

    print(f"\nMapped {len(apxinf_to_pytorch)} apxinf activations to PyTorch equivalents")

    print("\n=== Comparison Results ===")
    print(f"{'ApxInf Name':<45} {'PyTorch Name':<45} {'Shape':<20} {'Max Error':<12} {'Match'}")
    print("-" * 140)

    errors = []
    matches = []
    skipped = []

    for apxinf_key in sorted(apxinf_to_pytorch.keys()):
        pytorch_key = apxinf_to_pytorch[apxinf_key]

        if pytorch_key not in pytorch_data:
            skipped.append((apxinf_key, pytorch_key, "not in PyTorch"))
            continue

        apxinf_arr = apxinf_data[apxinf_key]
        pytorch_arr = pytorch_data[pytorch_key]

        apxinf_shape = apxinf_arr.shape
        pytorch_shape = pytorch_arr.shape

        # PyTorch has extra batch dimension - squeeze it
        if len(pytorch_shape) == len(apxinf_shape) + 1 and pytorch_shape[0] == 1:
            pytorch_arr = pytorch_arr.squeeze(0)
            pytorch_shape = pytorch_arr.shape

        # Check shape match
        if apxinf_shape != pytorch_shape:
            print(f"{apxinf_key:<45} {pytorch_key:<45} {str(apxinf_shape):<20} SHAPE MISMATCH FAIL")
            errors.append((apxinf_key, pytorch_key, "shape mismatch", apxinf_shape, pytorch_shape))
            continue

        # Compute max absolute error
        max_error = np.max(np.abs(apxinf_arr - pytorch_arr))

        # Check if within tolerance
        if max_error < tolerance:
            status = "OK"
            matches.append((apxinf_key, pytorch_key))
        else:
            status = "FAIL"
            errors.append((apxinf_key, pytorch_key, max_error, apxinf_shape, pytorch_shape))

        print(f"{apxinf_key:<45} {pytorch_key:<45} {str(apxinf_shape):<20} {max_error:<12.6f} {status}")

    print("\n=== Summary ===")
    print(f"Matched: {len(matches)} / {len(apxinf_to_pytorch)}")
    print(f"Skipped: {len(skipped)}")

    if errors:
        print(f"\n=== First divergent activations ===")
        for apxinf_key, pytorch_key, error, apxinf_shape, pytorch_shape in errors[:5]:
            if error == "shape mismatch":
                print(f"  {apxinf_key}: shape mismatch (apxinf={apxinf_shape}, pytorch={pytorch_shape})")
            else:
                print(f"  {apxinf_key} <-> {pytorch_key}: max error = {error:.6f}")

        # Print first divergent activation details
        first_error = errors[0]
        apxinf_key, pytorch_key = first_error[0], first_error[1]
        if first_error[2] != "shape mismatch":
            print(f"\n=== Detailed analysis of '{apxinf_key}' ===")
            apxinf_arr = apxinf_data[apxinf_key]
            pytorch_arr = pytorch_data[pytorch_key]

            print(f"ApxInf shape: {apxinf_arr.shape}")
            print(f"PyTorch shape: {pytorch_arr.shape}")
            print(f"ApxInf min/max: {apxinf_arr.min():.6f} / {apxinf_arr.max():.6f}")
            print(f"PyTorch min/max: {pytorch_arr.min():.6f} / {pytorch_arr.max():.6f}")
            print(f"ApxInf mean: {apxinf_arr.mean():.6f}")
            print(f"PyTorch mean: {pytorch_arr.mean():.6f}")

            # Find first divergent element
            diff = np.abs(apxinf_arr - pytorch_arr)
            max_idx = np.argmax(diff)
            print(f"Max diff at index {max_idx}: apxinf={apxinf_arr.flat[max_idx]:.6f}, pytorch={pytorch_arr.flat[max_idx]:.6f}")

            # Print first few values
            print(f"\nFirst 10 values:")
            print(f"ApxInf:   {apxinf_arr.flat[:10]}")
            print(f"PyTorch: {pytorch_arr.flat[:10]}")

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Compare apxinf and PyTorch activations")
    parser.add_argument("--apxinf", required=True, help="ApxInf NPZ file")
    parser.add_argument("--pytorch", required=True, help="PyTorch NPZ file")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Tolerance for numerical comparison")
    args = parser.parse_args()

    success = compare_activations(args.apxinf, args.pytorch, args.tolerance)

    if success:
        print("\nAll activations match within tolerance!")
    else:
        print("\nMismatch detected - investigate divergent activation")


if __name__ == "__main__":
    main()
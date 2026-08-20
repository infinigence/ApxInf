#!/usr/bin/env python3
"""Generate an independent FlashRT BF16 oracle for ApxInf's π0.5 fixture.

The fixture deliberately starts at the model boundary instead of exercising
tokenisation or image preprocessing: normalized image patches, token IDs, and
diffusion noise are all zero.  This makes the inputs exactly reproducible in
ApxInf while still executing every vision, language, and action layer.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import pathlib
import re
import time

# This must be set before importing the FlashRT frontend.  It prevents an
# accidentally available low-precision backend from weakening the oracle.
os.environ["FVK_PI05_RTX_FORCE_BF16"] = "1"
# The Orin reference image ships the upstream ``flash_attn`` package rather
# than FlashRT's optional vendored RTX extension.
os.environ.setdefault("FVK_RTX_FA2", "0")

import torch

from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--token-count", type=int, default=10)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--calibration-output", type=pathlib.Path)
    return parser.parse_args()


class AmaxGemmProbe:
    """Record the BF16 activation entering every pipeline GEMM."""

    def __init__(self, runner, cudart, weight_names: dict[int, str]):
        self._runner = runner
        self._cudart = cudart
        self._weight_names = weight_names
        self._scratch = None
        self.records: dict[str, float] = {}

    def __getattr__(self, name):
        return getattr(self._runner, name)

    def bf16_nn(self, activation, weight, output, m, n, k, stream=0):
        count = int(m) * int(k)
        if self._scratch is None or self._scratch.numel() < count:
            self._scratch = torch.empty(count, dtype=torch.bfloat16, device="cuda")
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(self._scratch.data_ptr()),
            ctypes.c_void_p(int(activation)),
            count * 2,
            3,
            stream,
        )
        self._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
        amax = float(self._scratch[:count].abs().amax().float().item())
        name = self._weight_names.get(int(weight))
        if name is None:
            raise KeyError(f"unrecognized BF16 GEMM weight pointer {int(weight):#x}")
        self.records[name] = max(self.records.get(name, 0.0), amax)
        return self._runner.bf16_nn(
            activation, weight, output, m, n, k, stream=stream
        )


def download_bf16(cudart, pointer: int, count: int, stream: int) -> torch.Tensor:
    output = torch.empty(count, dtype=torch.bfloat16, device="cuda")
    cudart.cudaMemcpyAsync(
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(int(pointer)),
        count * 2,
        3,
        stream,
    )
    cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
    return output.float().cpu()


def tensor_signature(tensor: torch.Tensor) -> dict:
    values = tensor.reshape(-1).float()
    count = values.numel()
    sample_count = min(256, count)
    indices = torch.linspace(0, count - 1, sample_count, dtype=torch.float64)
    indices = indices.to(torch.long)
    return {
        "elements": count,
        "sum": float(values.sum().item()),
        "abs_checksum": float(values.abs().sum().item()),
        "l2": float(torch.linalg.vector_norm(values).item()),
        "max_abs": float(values.abs().amax().item()),
        "sample": values[indices].tolist(),
    }


class StageFvkProbe:
    """Capture the diffusion state after each Euler update."""

    def __init__(self, runner, cudart, pipeline):
        self._runner = runner
        self._cudart = cudart
        self._noise = pipeline.bufs["diffusion_noise"].ptr.value
        self._action = pipeline.bufs["decoder_action_buf"].ptr.value
        self._elements = pipeline.chunk_size * 32
        self.decoder_signatures: list[dict] = []

    def __getattr__(self, name):
        return getattr(self._runner, name)

    def residual_add(self, residual, update, count, stream=0):
        result = self._runner.residual_add(
            residual, update, count, stream=stream
        )
        if int(residual) == self._noise and int(update) == self._action:
            state = download_bf16(
                self._cudart, self._noise, self._elements, stream
            )
            self.decoder_signatures.append(tensor_signature(state))
        return result


def weight_pointer_names(weights: dict) -> dict[int, str]:
    result: dict[int, str] = {}
    for name, value in weights.items():
        if isinstance(value, int):
            result[value] = name
        elif isinstance(value, list) and all(isinstance(pointer, int) for pointer in value):
            result.update({pointer: f"{name}.{index}" for index, pointer in enumerate(value)})
    return result


def time_conditioning_amax(frontend, steps: int) -> tuple[float, float, float]:
    weights = frontend._ckpt_bf16
    schedule = weights["decoder_time_embeds"][:steps]
    time_input = float(schedule.abs().amax().float().item())
    time_hidden = 0.0
    conditioning = 0.0
    for step in range(steps):
        embedding = schedule[step : step + 1]
        hidden = embedding @ weights["decoder_time_mlp_in_w"]
        hidden = hidden + weights["decoder_time_mlp_in_b"][None, :]
        hidden = (hidden.float() * torch.sigmoid(hidden.float())).to(torch.bfloat16)
        time_hidden = max(time_hidden, float(hidden.abs().amax().float().item()))
        output = hidden @ weights["decoder_time_mlp_out_w"]
        output = output + weights["decoder_time_mlp_out_b"][None, :]
        output = (output.float() * torch.sigmoid(output.float())).to(torch.bfloat16)
        conditioning = max(conditioning, float(output.abs().amax().float().item()))
    return time_input, time_hidden, conditioning


def calibration_from_records(frontend, records: dict[str, float], steps: int) -> dict:
    scales: dict[str, float] = {}

    def record(name: str, amax: float) -> None:
        scales[name] = max(scales.get(name, 0.0), float(amax))

    direct = {
        "vision_patch_embedding_w": "vision.patch_input",
        "encoder_multi_modal_projector_w": "vision.post_norm",
        "decoder_action_in_proj_w": "action.input",
        "decoder_action_out_proj_w": "action.final_norm",
    }
    patterns = [
        (r"vision_attn_qkv_w\.(\d+)", "vision", "attention_norm"),
        (r"vision_attn_o_w\.(\d+)", "vision", "attention_output"),
        (r"vision_ffn_up_w\.(\d+)", "vision", "mlp_norm"),
        (r"vision_ffn_down_w\.(\d+)", "vision", "mlp_activation"),
        (r"encoder_attn_qkv_w\.(\d+)", "language", "attention_norm"),
        (r"encoder_attn_o_w\.(\d+)", "language", "attention_output"),
        (r"encoder_ffn_(?:gate|up)_w\.(\d+)", "language", "mlp_norm"),
        (r"encoder_ffn_down_w\.(\d+)", "language", "mlp_activation"),
        (r"decoder_attn_qkv_w\.(\d+)", "action", "attention_norm"),
        (r"decoder_attn_o_w\.(\d+)", "action", "attention_output"),
        (r"decoder_ffn_(?:gate|up)_w\.(\d+)", "action", "mlp_norm"),
        (r"decoder_ffn_down_w\.(\d+)", "action", "mlp_activation"),
    ]
    for weight_name, amax in records.items():
        if weight_name in direct:
            record(direct[weight_name], amax)
            continue
        for pattern, stage, site in patterns:
            match = re.fullmatch(pattern, weight_name)
            if match:
                record(f"{stage}.layers.{match.group(1)}.{site}", amax)
                break

    time_input, time_hidden, conditioning = time_conditioning_amax(frontend, steps)
    record("time.input", time_input)
    record("time.hidden", time_hidden)
    record("action.conditioning", conditioning)

    # The final language layer only materializes K/V; its tail is deliberately
    # skipped in both engines. The calibration schema still requires those
    # unused entries, so copy the corresponding preceding-layer values.
    for site in ["attention_output", "mlp_norm", "mlp_activation"]:
        scales[f"language.layers.17.{site}"] = scales[f"language.layers.16.{site}"]

    required = ["vision.patch_input", "vision.post_norm"]
    required += [
        f"vision.layers.{layer}.{site}"
        for layer in range(27)
        for site in ["attention_norm", "attention_output", "mlp_norm", "mlp_activation"]
    ]
    required += [
        f"language.layers.{layer}.{site}"
        for layer in range(18)
        for site in ["attention_norm", "attention_output", "mlp_norm", "mlp_activation"]
    ]
    required += ["action.input", "time.input", "time.hidden", "action.conditioning"]
    required += [
        f"action.layers.{layer}.{site}"
        for layer in range(18)
        for site in ["attention_norm", "attention_output", "mlp_norm", "mlp_activation"]
    ]
    required.append("action.final_norm")
    missing = [name for name in required if name not in scales]
    if missing:
        raise RuntimeError(f"calibration probe missed {len(missing)} sites: {missing[:8]}")

    return {
        "schema": "apxinf.pi05.fp8-calibration.v1",
        "source": "FlashRT BF16 activation amax",
        "fixture": "two-view zero normalized images, zero token IDs, zero noise",
        "token_count": frontend.current_prompt_len,
        "scales": {
            name: {
                "amax": scales[name],
                "scale": max(scales[name] / 448.0, 1.0e-8),
            }
            for name in sorted(scales)
        },
    }


def main() -> None:
    args = parse_args()
    if args.token_count <= 0 or args.token_count > 200:
        raise ValueError("--token-count must be in 1..=200")

    torch.manual_seed(0)
    started = time.perf_counter()
    frontend = Pi05TorchFrontendRtx(
        args.checkpoint,
        num_views=args.num_views,
        chunk_size=args.action_horizon,
        max_prompt_len=max(48, args.token_count),
        num_steps=args.steps,
        use_fp8=False,
        cache_frames=1,
    )

    # Build the exact-length pipeline without involving tokenizer versions.
    # Both implementations then receive token id 0 repeated token_count times.
    frontend._set_prompt_per_length(None, args.token_count)
    frontend.attn_backend.set_fixed_shape(False)
    token_ids = torch.zeros(args.token_count, dtype=torch.long, device="cuda")
    language = torch.nn.functional.embedding(token_ids, frontend.embedding_weight)
    language = language * math.sqrt(language.shape[-1])
    frontend.pipeline.set_language_embeds(
        language.contiguous().view(torch.uint16).cpu().numpy()
    )
    probe = None
    if args.calibration_output is not None:
        probe = AmaxGemmProbe(
            frontend.gemm,
            frontend._cudart,
            weight_pointer_names(frontend.pipeline.weights),
        )
        frontend.pipeline.gemm = probe
    stage_probe = StageFvkProbe(
        frontend.pipeline.fvk, frontend._cudart, frontend.pipeline
    )
    frontend.pipeline.fvk = stage_probe

    images = torch.zeros(
        args.num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda"
    )
    noise = torch.zeros(
        args.action_horizon, 32, dtype=torch.bfloat16, device="cuda"
    )
    output = torch.empty_like(noise)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream_id = stream.cuda_stream
        frontend._copy_tensor_to_pipeline_buf_stream(
            images, frontend.pipeline.input_images_buf, stream_id
        )
        frontend._copy_tensor_to_pipeline_buf_stream(
            noise, frontend.pipeline.input_noise_buf, stream_id
        )
        pipeline = frontend.pipeline
        pipeline._copy_lang_embeds_to_encoder_x(stream=stream_id)
        buffers = pipeline.bufs
        weights = pipeline.weights
        intermediate_signatures = {}

        # Spell out the vision driver so the oracle records the residual after
        # patch embedding and every SigLIP block.  The individual operations
        # are exactly those used by Pi05Pipeline.vision_encoder.
        vision_tokens_full = pipeline.vision_seq
        pipeline.fvk.patch_im2col(
            buffers["observation_images_normalized"].ptr.value,
            buffers["vision_patches"].ptr.value,
            pipeline.num_views,
            stream_id,
        )
        pipeline.gemm.bf16_nn(
            buffers["vision_patches"].ptr.value,
            weights["vision_patch_embedding_w"],
            buffers["vision_x"].ptr.value,
            vision_tokens_full,
            1152,
            3 * 14 * 14,
            stream=stream_id,
        )
        pipeline.fvk.bias_residual(
            buffers["vision_x"].ptr.value,
            buffers["vision_pos_embed_expanded"].ptr.value,
            weights["vision_patch_embedding_b"],
            vision_tokens_full,
            1152,
            stream=stream_id,
        )
        intermediate_signatures["vision_patch_embed"] = tensor_signature(
            download_bf16(
                frontend._cudart,
                buffers["vision_x"].ptr.value,
                vision_tokens_full * 1152,
                stream_id,
            )
        )
        pipeline.fvk.layer_norm(
            buffers["vision_x"].ptr.value,
            weights["vision_pre_attn_norm_w"][0],
            weights["vision_pre_attn_norm_b"][0],
            buffers["vision_x_norm"].ptr.value,
            vision_tokens_full,
            1152,
            1.0e-5,
            stream=stream_id,
        )
        use_fp8_vision = pipeline.use_fp8 and "vision_attn_qkv_w_0" in weights.get(
            "fp8", {}
        )
        last_vision_layer = pipeline.vision_num_layers - 1
        for layer in range(pipeline.vision_num_layers):
            pipeline._vision_layer(
                layer,
                vision_tokens_full,
                use_fp8_vision,
                stream_id,
                is_last=(layer == last_vision_layer),
            )
            intermediate_signatures[f"vision_layer_{layer}"] = tensor_signature(
                download_bf16(
                    frontend._cudart,
                    buffers["vision_x"].ptr.value,
                    vision_tokens_full * 1152,
                    stream_id,
                )
            )
        if pipeline.vision_pool_factor > 1:
            pipeline.fvk.avg_pool_vision_tokens(
                buffers["vision_x"].ptr.value,
                buffers["vision_x_pooled"].ptr.value,
                pipeline.num_views,
                16,
                16,
                1152,
                pipeline.vision_pool_factor,
                stream_id,
            )

        # Materialize the post-vision projection as an explicit stage oracle.
        vision_tokens = pipeline.vision_seq_enc
        pipeline.fvk.layer_norm(
            buffers["vision_x_pooled"].ptr.value,
            weights["vision_final_norm_w"],
            weights["vision_final_norm_b"],
            buffers["vision_x_norm"].ptr.value,
            vision_tokens,
            1152,
            1.0e-5,
            stream=stream_id,
        )
        pipeline.gemm.bf16_nn(
            buffers["vision_x_norm"].ptr.value,
            weights["encoder_multi_modal_projector_w"],
            buffers["encoder_x"].ptr.value,
            vision_tokens,
            2048,
            1152,
            stream=stream_id,
        )
        pipeline.fvk.add_bias_bf16(
            buffers["encoder_x"].ptr.value,
            weights["encoder_multi_modal_projector_b"],
            vision_tokens,
            2048,
            stream=stream_id,
        )
        intermediate_signatures["vision_projected"] = tensor_signature(
            download_bf16(
                frontend._cudart,
                buffers["encoder_x"].ptr.value,
                vision_tokens * 2048,
                stream_id,
            )
        )

        # transformer_encoder repeats the inexpensive projection above, then
        # writes all 18 prefix K/V cache layers.
        pipeline.transformer_encoder(stream=stream_id)
        for layer in [0, 17]:
            _, value_pointer = pipeline._enc_kv_layer_ptrs(layer, offset_tokens=0)
            intermediate_signatures[f"prefix_v_layer{layer}"] = tensor_signature(
                download_bf16(
                    frontend._cudart,
                    value_pointer,
                    pipeline.encoder_seq_len * 256,
                    stream_id,
                )
            )
        pipeline.transformer_decoder(stream=stream_id)
        frontend._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(frontend.pipeline.input_noise_buf.ptr.value),
            output.numel() * output.element_size(),
            3,
            stream_id,
        )
    frontend._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream.cuda_stream))

    actions = output.float().cpu().numpy()
    result = {
        "schema": "apxinf.pi05.integrity.v1",
        "reference": "FlashRT BF16",
        "checkpoint": str(args.checkpoint.resolve()),
        "fixture": {
            "normalized_images": "zeros",
            "token_ids": "zeros",
            "diffusion_noise": "zeros",
        },
        "num_views": args.num_views,
        "token_count": args.token_count,
        "action_horizon": args.action_horizon,
        "action_dim": 32,
        "flow_steps": args.steps,
        "dtype": "bfloat16",
        "elapsed_seconds": time.perf_counter() - started,
        "output_abs_checksum": float(abs(actions).sum()),
        "intermediate_signatures": {
            **intermediate_signatures,
            **{
                f"denoise_step_{index}": signature
                for index, signature in enumerate(stage_probe.decoder_signatures)
            },
        },
        "raw_actions": actions.reshape(-1).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.calibration_output is not None:
        calibration = calibration_from_records(frontend, probe.records, args.steps)
        args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_output.write_text(json.dumps(calibration, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "raw_actions"}, indent=2))


if __name__ == "__main__":
    main()

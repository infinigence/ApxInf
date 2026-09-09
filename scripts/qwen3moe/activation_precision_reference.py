#!/usr/bin/env python3
"""Evaluate activation-precision hypotheses using the independent CPU reference.

This is a diagnostic, not an acceptance reference or a runtime implementation.
The FP32 mode must reproduce an existing reference before interpreting FP16
results. Checkpoint head values remain unchanged. Do not run during Thor GPU
performance measurements, which share the CPU's memory bandwidth.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from cached_reference import Qwen3MoeRef, Linear4, rms_norm, apply_rope


class ActivationReference(Qwen3MoeRef):
    def __init__(self, model, precision, exact_prefill_weights=False,
                 fp32_attention=False, fp32_router_input=False, route_oracle=None,
                 fp32_residual_projections=False, fp32_projection_outputs=False):
        super().__init__(model)
        self.precision = precision
        self.decode = False
        self.exact_prefill_weights = exact_prefill_weights
        self.fp32_attention = fp32_attention
        self.fp32_router_input = fp32_router_input
        self.route_oracle = route_oracle
        self.route_memberships = {}
        self.oracle_differences = {}
        self.fp32_residual_projections = fp32_residual_projections
        self.fp32_projection_outputs = fp32_projection_outputs

    def activation(self, x):
        if self.precision == 'fp32':
            return x
        high = x.to(torch.float16).float()
        if self.precision in ('fp16_pair', 'fp16_pair_scaled'):
            # Representation-only hypothesis. CUDA would need separate tensor
            # products and has a different accumulation order; this does not
            # predict its numerical acceptance or throughput.
            scale = 4096.0 if self.precision == 'fp16_pair_scaled' else 1.0
            return high + ((x - high) * scale).to(torch.float16).float() / scale
        return high

    def attention_activation(self, x):
        return x if self.fp32_attention else self.activation(x)

    def linear(self, x, prefix, output_fp32=False):
        weight = Linear4(self.sh, prefix).weight()
        # Prefill tensor GEMM dequantizes to FP16. Decode's existing integer
        # dot plus FP32 scale/zero correction can retain the exact AWQ values.
        if not self.decode and not self.exact_prefill_weights:
            weight = self.activation(weight)
        result = x @ weight
        return result if output_fp32 or self.fp32_projection_outputs else self.activation(result)

    def layer_forward(self, i, x, cos, sin, offset, sig_out=None):
        self.decode = offset > 0
        p = f'model.layers.{i}.'
        get = self.sh.get
        size = x.shape[0]
        h = self.activation(rms_norm(x, get(p + 'input_layernorm.weight').float(), self.eps))
        q = self.linear(h, p + 'self_attn.q_proj', self.fp32_attention).view(size, self.H, self.D)
        k = self.linear(h, p + 'self_attn.k_proj', self.fp32_attention).view(size, self.KVH, self.D)
        v = self.linear(h, p + 'self_attn.v_proj', self.fp32_attention).view(size, self.KVH, self.D)
        q = self.attention_activation(rms_norm(q, get(p + 'self_attn.q_norm.weight').float(), self.eps))
        k = self.attention_activation(rms_norm(k, get(p + 'self_attn.k_norm.weight').float(), self.eps))
        q = self.attention_activation(apply_rope(q, cos, sin))
        k = self.attention_activation(apply_rope(k, cos, sin))
        if offset:
            old_k, old_v = self.cache[i]
            k = torch.cat([old_k, k], dim=0)
            v = torch.cat([old_v, v], dim=0)
        self.cache[i] = (k, v)
        kh = k.repeat_interleave(self.H // self.KVH, dim=1).transpose(0, 1)
        vh = v.repeat_interleave(self.H // self.KVH, dim=1).transpose(0, 1)
        qh = q.transpose(0, 1)
        pieces = []
        keys = torch.arange(offset + size)
        for start in range(0, size, 128):
            stop = min(size, start + 128)
            scores = qh[:, start:stop] @ kh.transpose(-1, -2) / math.sqrt(self.D)
            causal = keys[None, :] > torch.arange(offset + start, offset + stop)[:, None]
            scores.masked_fill_(causal, float('-inf'))
            # FP16 probabilities approximate the FA2 product precision. Actual
            # online-softmax reduction order still needs a CUDA model check.
            probabilities = self.attention_activation(torch.softmax(scores, dim=-1))
            pieces.append(probabilities @ vh)
        attn = self.attention_activation(torch.cat(pieces, dim=1).transpose(0, 1).reshape(size, self.H * self.D))
        x = x + self.linear(attn, p + 'self_attn.o_proj',
                            output_fp32=self.decode or self.fp32_residual_projections)
        normalized = rms_norm(x, get(p + 'post_attention_layernorm.weight').float(), self.eps)
        h = self.activation(normalized)
        # Keep router output, probability normalization, and residuals FP32.
        router_input = normalized if self.fp32_router_input else h
        logits = router_input @ get(p + 'mlp.gate.weight').float().t()
        probabilities = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(probabilities, self.topk, dim=-1)
        key = f'route_{offset}_{i}'
        self.route_memberships[key] = indices.numpy().astype(np.int32)
        if self.route_oracle is not None:
            # Diagnostic intervention only: isolate expert membership changes
            # from arithmetic error. Recompute weights from this run's logits.
            # These outputs must never serve as runtime acceptance evidence.
            expected = torch.from_numpy(self.route_oracle[key]).long()
            if (expected.shape != indices.shape or expected.min() < 0 or
                    expected.max() >= self.E or
                    (expected.sort(dim=-1).values.diff(dim=-1) == 0).any()):
                raise ValueError(f'invalid oracle memberships at {key}')
            changed = (indices.sort(dim=-1).values != expected.sort(dim=-1).values).any(dim=-1)
            self.oracle_differences[key] = int(changed.sum())
            weights, order = probabilities.gather(-1, expected).sort(dim=-1, descending=True)
            indices = expected.gather(-1, order)
        if self.norm_topk:
            weights = weights / weights.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for expert in indices.unique().tolist():
            rows, slots = (indices == expert).nonzero(as_tuple=True)
            hin = h[rows]
            gate = self.linear(hin, p + f'mlp.experts.{expert}.gate_proj')
            up = self.linear(hin, p + f'mlp.experts.{expert}.up_proj')
            intermediate = self.activation(torch.nn.functional.silu(gate) * up)
            y = self.linear(intermediate, p + f'mlp.experts.{expert}.down_proj',
                            output_fp32=self.decode or self.fp32_residual_projections)
            out.index_add_(0, rows, y * weights[rows, slots].unsqueeze(-1))
        if sig_out is not None:
            sig_out['topk'].append(indices[-1].numpy().astype(np.int32))
            sig_out['topw'].append(weights[-1].numpy().astype(np.float32))
        return x + out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--case-json', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'fp16_pair', 'fp16_pair_scaled'], required=True)
    parser.add_argument('--exact-prefill-weights', action='store_true',
                        help='disable FP16 rounding of dequantized prefill weights')
    parser.add_argument('--fp32-attention', action='store_true',
                        help='retain QKV projection outputs, QK norm, RoPE, KV, probabilities, and attention output in FP32')
    parser.add_argument('--fp32-router-input', action='store_true',
                        help='use the unrounded normalized residual for router input')
    parser.add_argument('--route-oracle', type=Path,
                        help='diagnostic only: force memberships from an exact FP32 control NPZ')
    parser.add_argument('--fp32-residual-projections', action='store_true',
                        help='retain prefill O/down projection outputs in FP32 before adding to the residual')
    parser.add_argument('--fp32-projection-outputs', action='store_true',
                        help='retain all linear projection outputs in FP32 before normalization, SiLU, or residual addition')
    args = parser.parse_args()
    case = json.loads(args.case_json.read_text())
    forced = case['greedy_tokens']
    if not forced:
        parser.error('case must contain teacher-forcing tokens')
    oracle = None
    if args.route_oracle is not None:
        with np.load(args.route_oracle) as source:
            if (source['activation_precision'].item() != 'fp32' or
                    source['forced_route_oracle'].item() or
                    not np.array_equal(source['token_ids'], case['token_ids']) or
                    not np.array_equal(source['teacher_forcing_tokens'], forced)):
                raise ValueError('route oracle must be an unforced FP32 control for this exact case')
            with np.load(args.reference) as reference:
                if (source['step_logits'].shape != reference['step_logits'].shape or
                        not np.isfinite(source['step_logits']).all() or
                        np.max(np.abs(source['step_logits'] - reference['step_logits'])) > 0.001):
                    raise ValueError('route oracle logits do not reproduce the independent reference')
            oracle = {key: source[key] for key in source.files if key.startswith('route_')}
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(12)
    started = time.monotonic()
    model = ActivationReference(args.model, args.precision, args.exact_prefill_weights,
                                args.fp32_attention, args.fp32_router_input, oracle,
                                args.fp32_residual_projections, args.fp32_projection_outputs)
    logits, sigs, extra = model.forward(case['token_ids'], collect=True)
    rows = []
    for step, token in enumerate(forced):
        rows.append(logits.numpy().copy())
        if step + 1 < len(forced):
            logits, _, _ = model.forward([token])
    rows = np.stack(rows)
    with np.load(args.reference) as ref:
        if not np.array_equal(ref['token_ids'], case['token_ids']):
            raise ValueError('reference prompt differs from teacher-forcing case')
        if rows.shape != ref['step_logits'].shape or not np.isfinite(rows).all():
            raise ValueError('incomplete or non-finite diagnostic output')
        errors = np.max(np.abs(rows - ref['step_logits']), axis=1)
        sorted_logits = np.sort(ref['step_logits'], axis=1)
        margins = sorted_logits[:, -1] - sorted_logits[:, -2]
        expected_experts = ref['router_topk']
        different_experts = [i for i, got in enumerate(extra['topk'])
                             if set(got.tolist()) != set(expected_experts[i].tolist())]
    if args.precision == 'fp32' and max(errors) > 0.001:
        raise ValueError(f'FP32 control did not reproduce the reference: {errors}')
    np.savez(args.out / 'activation_reference.npz', step_logits=rows, layer_sig=sigs,
             router_topk=np.stack(extra['topk']), router_weights=np.stack(extra['topw']),
             activation_precision=np.array(args.precision), forced_route_oracle=oracle is not None,
             token_ids=np.array(case['token_ids']), teacher_forcing_tokens=np.array(forced),
             **model.route_memberships)
    result = dict(scope='precision hypothesis only; not an acceptance reference',
                  activation_precision=args.precision, residual_precision='fp32',
                  low_component_scale=4096 if args.precision == 'fp16_pair_scaled' else 1,
                  exact_prefill_weights=args.exact_prefill_weights,
                  fp32_attention=args.fp32_attention, fp32_router_input=args.fp32_router_input,
                  forced_route_oracle=str(args.route_oracle) if oracle is not None else None,
                  fp32_projection_outputs=args.fp32_projection_outputs,
                  prefill_residual_projection_precision='fp32' if args.precision == 'fp32' or args.fp32_residual_projections or args.fp32_projection_outputs else args.precision,
                  rows_with_different_expert_memberships=model.oracle_differences,
                  decode_weight_precision='exact AWQ values', decode_residual_projection_precision='fp32',
                  router_output_precision='fp32', final_norm_precision='fp32',
                  head_weights='unchanged checkpoint values',
                  case=str(args.case_json), reference=str(args.reference),
                  teacher_forcing_tokens=forced, predicted_tokens=np.argmax(rows, axis=1).tolist(),
                  max_logit_difference=errors.tolist(), reference_margin=margins.tolist(),
                  layers_with_different_experts=different_experts,
                  seconds=time.monotonic()-started)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

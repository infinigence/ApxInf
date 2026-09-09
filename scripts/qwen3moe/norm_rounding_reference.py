#!/usr/bin/env python3
"""Isolate BF16 norm-weight rounding in the otherwise FP32 CPU reference.

This produces diagnostic evidence in a separate output directory. It must
never replace an acceptance reference. Do not run during GPU benchmarks on
Thor, since CPU memory traffic shares the GPU's memory bandwidth.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from cached_reference import Qwen3MoeRef


class RoundedNorms:
    def __init__(self, original):
        self.original = original

    def get(self, name):
        value = self.original.get(name)
        if name.endswith('norm.weight'):
            value = value.to(torch.bfloat16).float()
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--case-json', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(12)
    case = json.loads(args.case_json.read_text())
    forced = case['greedy_tokens']
    if not forced:
        parser.error('case must include reference teacher-forcing tokens')
    started = time.monotonic()
    model = Qwen3MoeRef(args.model)
    model.sh = RoundedNorms(model.sh)
    model.final_norm = model.final_norm.to(torch.bfloat16).float()
    logits, sigs, extra = model.forward(case['token_ids'], collect=True)
    rows, greedy = [], []
    for step, token in enumerate(forced):
        rows.append(logits.numpy().copy())
        greedy.append(int(logits.argmax()))
        if step + 1 < len(forced):
            logits, _, _ = model.forward([token])
    rows = np.stack(rows)
    with np.load(args.reference) as ref:
        if not np.array_equal(ref['token_ids'], case['token_ids']):
            raise ValueError('reference prompt differs from teacher-forcing case')
        if ref['step_logits'].shape != rows.shape or not np.isfinite(rows).all():
            raise ValueError('incomplete or non-finite diagnostic output')
        errors = np.max(np.abs(rows - ref['step_logits']), axis=1)
    np.savez(args.out / 'norm_bf16.npz', step_logits=rows, layer_sig=sigs,
             router_topk=np.stack(extra['topk']), router_weights=np.stack(extra['topw']))
    result = dict(scope='diagnostic only; not an acceptance reference',
                  changed_precision='norm weights rounded to BF16; all other arithmetic remains FP32',
                  case=str(args.case_json), reference=str(args.reference),
                  teacher_forcing_tokens=forced, predicted_tokens=greedy,
                  max_logit_difference=errors.tolist(), seconds=time.monotonic()-started)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

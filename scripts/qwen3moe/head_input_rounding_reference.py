#!/usr/bin/env python3
"""Isolate final activation rounding without changing checkpoint head weights.

The otherwise FP32 CPU reference is teacher-forced from an existing fixture.
This diagnostic does not replace acceptance references. Do not run it during
GPU performance measurements on Thor's shared-memory system.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import cached_reference as reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--case-json', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case_json.read_text())
    forced = case['greedy_tokens']
    if not forced:
        parser.error('case must include reference teacher-forcing tokens')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(12)
    started = time.monotonic()
    model = reference.Qwen3MoeRef(args.model)
    hidden = []
    rms_norm = reference.rms_norm

    def capture_final_norm(x, weight, epsilon):
        result = rms_norm(x, weight, epsilon)
        if weight is model.final_norm:
            hidden.append(result[-1].clone())
        return result

    reference.rms_norm = capture_final_norm
    rows = []
    try:
        logits, _, _ = model.forward(case['token_ids'])
        for step, token in enumerate(forced):
            rows.append(logits.numpy().copy())
            if step + 1 < len(forced):
                logits, _, _ = model.forward([token])
    finally:
        reference.rms_norm = rms_norm
    if len(hidden) != len(forced):
        raise ValueError('did not capture exactly one final activation per step')
    rows = np.stack(rows)
    with np.load(args.reference) as expected:
        if not np.array_equal(expected['token_ids'], case['token_ids']):
            raise ValueError('reference prompt differs from the teacher-forcing case')
        if expected['step_logits'].shape != rows.shape or not np.isfinite(rows).all():
            raise ValueError('incomplete or non-finite diagnostic output')
        reference_error = np.max(np.abs(rows - expected['step_logits']), axis=1)
        if np.max(reference_error) > 0.001:
            raise ValueError(f'CPU replay did not reproduce the reference: {reference_error}')
        sorted_logits = np.sort(expected['step_logits'], axis=1)
        margins = sorted_logits[:, -1] - sorted_logits[:, -2]
    hidden = torch.stack(hidden)
    outputs = dict(fp32=rows)
    # Only the input vector changes. The checkpoint head values and FP32
    # accumulation remain the same in both isolated rounding experiments.
    for name, dtype in [('bf16', torch.bfloat16), ('fp16', torch.float16)]:
        outputs[name] = torch.stack([
            h.to(dtype).float() @ model.lm_head.t() for h in hidden
        ]).numpy()
    stats = {}
    for name, values in outputs.items():
        if not np.isfinite(values).all():
            raise ValueError(f'non-finite {name} output')
        stats[name] = dict(max_error_vs_unrounded=np.max(np.abs(values - rows), axis=1).tolist(),
                           predicted_tokens=np.argmax(values, axis=1).tolist())
    np.savez(args.out / 'head_rounding.npz', hidden=hidden.numpy(), **outputs)
    result = dict(scope='isolated rounding diagnostic; not an acceptance reference',
                  case=str(args.case_json), reference=str(args.reference),
                  teacher_forcing_tokens=forced, reference_margin=margins.tolist(),
                  cpu_replay_max_error=reference_error.tolist(), head_weights='unchanged checkpoint values',
                  variants=stats, seconds=time.monotonic()-started)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

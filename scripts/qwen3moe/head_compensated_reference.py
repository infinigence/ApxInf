#!/usr/bin/env python3
"""Isolate two-component BF16 input error with unchanged BF16 head weights.

Reuse captured FP32 final activations; never regenerate or replace acceptance
references. This only diagnoses input representation, not whole-model accuracy.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from cached_reference import Shards

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--captured-head', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(12)
started = time.monotonic()
with np.load(args.captured_head) as source:
    hidden = torch.from_numpy(source['hidden'].copy())
    expected = source['fp32'].copy()
weight = Shards(args.model).get('lm_head.weight').float()
if not torch.equal(weight, weight.to(torch.bfloat16).float()):
    raise ValueError('checkpoint head is not exactly BF16 representable')
rows, ordinary, compensated = [], [], []
for h in hidden:
    rows.append((h @ weight.t()).numpy())
    hi = h.to(torch.bfloat16).float()
    lo = (h - hi).to(torch.bfloat16).float()
    first = hi @ weight.t()
    ordinary.append(first.numpy())
    compensated.append((first + lo @ weight.t()).numpy())
rows, ordinary, compensated = map(np.stack, (rows, ordinary, compensated))
if (rows.shape != expected.shape or not np.isfinite(rows).all() or
        np.max(np.abs(rows - expected)) > 0.001):
    raise ValueError('captured FP32 head replay did not match')
if not np.isfinite(compensated).all():
    raise ValueError('non-finite compensated output')
np.savez(args.out / 'head_compensated.npz', fp32=rows, bf16=ordinary, compensated=compensated)
result = dict(scope='isolated head-input diagnostic; not model acceptance',
              head_weights='unchanged checkpoint BF16 values',
              captured_head=str(args.captured_head),
              replay_max_error=np.max(np.abs(rows - expected), axis=1).tolist(),
              bf16_max_error=np.max(np.abs(ordinary - expected), axis=1).tolist(),
              compensated_max_error=np.max(np.abs(compensated - expected), axis=1).tolist(),
              predicted_tokens=np.argmax(compensated, axis=1).tolist(),
              seconds=time.monotonic()-started)
(args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))

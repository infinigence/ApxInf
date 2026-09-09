#!/usr/bin/env python3
"""Locate prefill router-set differences against cached_reference.py evidence.

Layer signatures are summaries, not a proof of elementwise equality. The
reference currently stores prefill router/signature data only; decode traces
are deliberately excluded from this comparison.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    with np.load(args.reference) as ref:
        position = len(ref['token_ids']) - 1
        entries = [json.loads(line) for line in args.trace.read_text().splitlines()]
        entries = [x for x in entries if x['phase'] == 'prefill' and x['position'] == position]
        layers = len(ref['router_topk'])
        if sorted(x['layer'] for x in entries) != list(range(layers)):
            raise ValueError(f'trace must contain exactly one prefill record per layer at position {position}')
        rows = []
        for x in sorted(entries, key=lambda x: x['layer']):
            layer = x['layer']
            values = np.asarray(x['residual'], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f'non-finite residual in layer {layer}')
            sig = np.array([values.sum(), np.abs(values).sum(), np.linalg.norm(values),
                            np.abs(values).max(), values.mean(), values.std(ddof=1)])
            expected = ref['router_topk'][layer].tolist()
            got = x['router_topk']
            missing = sorted(set(expected) - set(got))
            added = sorted(set(got) - set(expected))
            row = dict(layer=layer, reference_experts=expected, actual_experts=got,
                       missing_experts=missing, added_experts=added,
                       reference_signature=ref['layer_sig'][layer + 1].tolist(),
                       actual_signature=sig.tolist(),
                       signature_delta=(sig - ref['layer_sig'][layer + 1]).tolist())
            rows.append(row)
            if missing:
                print(f'layer {layer:2}: missing {missing}, added {added}; residual norm {sig[2]:.5f} vs {ref["layer_sig"][layer+1,2]:.5f}')
        result = dict(reference=str(args.reference), trace=str(args.trace), position=position,
                      layers=rows, layers_with_different_experts=sum(bool(x['missing_experts']) for x in rows))
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(f'{result["layers_with_different_experts"]}/{layers} layers select a different expert set; wrote {args.out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Isolate CUDA AWQ matrix-product error within the CPU precision diagnostic.

Normalization, attention, routing, residual additions, and the unchanged BF16
checkpoint head are evaluated by the independent CPU implementation. This
hybrid replay is neither a CUDA runtime nor an acceptance reference. It tests
whether CUDA component-product accumulation invalidates the passing CPU
representation hypothesis, before implementing a new runtime precision path.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from activation_precision_reference import ActivationReference
from cached_reference import Linear4


class CudaProducts:
    def __init__(self, library, mode):
        self.library = ctypes.CDLL(str(library.resolve()))
        self.library.qwen_product_error.argtypes = []
        self.library.qwen_product_error.restype = ctypes.c_char_p
        self.library.qwen_product_create.argtypes = []
        self.library.qwen_product_create.restype = ctypes.c_void_p
        self.library.qwen_product_destroy.argtypes = [ctypes.c_void_p]
        self.library.qwen_product_destroy.restype = None
        pointer = ctypes.POINTER(ctypes.c_float)
        self.library.qwen_product_run.argtypes = [ctypes.c_void_p, pointer, pointer, pointer,
                                                 ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int]
        self.library.qwen_product_run.restype = ctypes.c_int
        self.context = self.library.qwen_product_create()
        if not self.context:
            raise RuntimeError(self.library.qwen_product_error().decode())
        self.mode = mode
        self.calls = 0

    def close(self):
        if self.context:
            self.library.qwen_product_destroy(self.context)
            self.context = None

    def __call__(self, a, b):
        a, b = (np.ascontiguousarray(value, dtype=np.float32) for value in (a, b))
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError('expected compatible two-dimensional matrices')
        m, k = a.shape
        n = b.shape[1]
        if min(m, n, k) <= 0 or max(m * k, k * n, m * n) > (2**31 - 1) // 2:
            raise ValueError('matrix dimensions exceed diagnostic bounds')
        c = np.empty((m, n), dtype=np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        status = self.library.qwen_product_run(self.context, a.ctypes.data_as(pointer),
                                              b.ctypes.data_as(pointer), c.ctypes.data_as(pointer),
                                              m, n, k, self.mode)
        if status:
            raise RuntimeError(self.library.qwen_product_error().decode())
        if not np.isfinite(c).all():
            raise ValueError('CUDA product returned non-finite values')
        self.calls += 1
        return c


class CudaProductReference(ActivationReference):
    def __init__(self, model, products):
        precision = 'fp32' if products.mode == 0 else 'fp16_pair_scaled'
        super().__init__(model, precision, fp32_router_input=True)
        self.products = products

    def linear(self, x, prefix, output_fp32=False):
        weight = Linear4(self.sh, prefix).weight()
        # CUDA splits the exact checkpoint AWQ values itself. Inputs have
        # already crossed the same activation boundary as the CPU hypothesis.
        # Decode is deliberately included: a future paired tensor path must
        # validate its accumulation rather than borrow the CPU decode result.
        result = torch.from_numpy(self.products(x.numpy(), weight.numpy()))
        return result if output_fp32 else self.activation(result)


def verify_operator(products):
    """Exercise both layouts, tail sizes, cancellation, and low components."""
    rng = np.random.default_rng(20260908)
    records = []
    for m, n, k, magnitude in [(1, 19, 17, 1.), (7, 31, 129, 1.e-4),
                               (33, 64, 768, 1.), (128, 128, 2048, 1.)]:
        a = (rng.normal(size=(m, k)) * magnitude).astype(np.float32)
        b = rng.normal(size=(k, n)).astype(np.float32)
        got = products(a, b)
        expected = a.astype(np.float64) @ b.astype(np.float64)
        errors = np.abs(got - expected)
        # Absolute-sum conditioning gives a bound for cancellation-heavy dots.
        condition = np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64))
        if np.any(errors > 2.e-6 + 4.e-7 * condition):
            raise ValueError(f'CUDA product FP64 oracle failed at {(m, n, k)}')
        records.append(dict(shape=[m, n, k], magnitude=magnitude,
                            max_fp64_error=float(errors.max())))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--products', choices=['fp32', 'scaled3', 'scaled4'], required=True)
    parser.add_argument('--model')
    parser.add_argument('--case-json', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--operator-only', action='store_true')
    args = parser.parse_args()
    if not args.operator_only and not all((args.model, args.case_json, args.reference)):
        parser.error('model replay requires --model, --case-json, and --reference')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(12)
    started = time.monotonic()
    products = CudaProducts(args.library, {'fp32': 0, 'scaled3': 3, 'scaled4': 4}[args.products])
    result = dict(scope='hybrid CUDA-product/CPU-model diagnostic; not runtime acceptance',
                  products=args.products, head_weights='unchanged checkpoint BF16 values',
                  cpu_operations='normalization, attention, router, residuals, head',
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), Path(__file__).with_name('cuda_product_precision.cu'),
                                           Path(__file__).with_name('activation_precision_reference.py'),
                                           Path(__file__).with_name('cached_reference.py')]})
    try:
        result['operator_checks'] = verify_operator(products)
        if not args.operator_only:
            case = json.loads(args.case_json.read_text())
            forced = case['greedy_tokens']
            if not forced:
                raise ValueError('teacher-forcing tokens are empty')
            model = CudaProductReference(args.model, products)
            if not torch.equal(model.lm_head, model.lm_head.to(torch.bfloat16).float()):
                raise ValueError('checkpoint head is not exactly BF16 representable')
            logits, sigs, extra = model.forward(case['token_ids'], collect=True)
            rows = []
            for step, token in enumerate(forced):
                rows.append(logits.numpy().copy())
                print(f'step={step} predicted={int(logits.argmax())}', flush=True)
                if step + 1 < len(forced):
                    logits, _, _ = model.forward([token])
            rows = np.stack(rows)
            with np.load(args.reference) as reference:
                if not np.array_equal(reference['token_ids'], case['token_ids']):
                    raise ValueError('reference prompt does not match replay')
                expected = reference['step_logits']
                if rows.shape != expected.shape or not np.isfinite(rows).all():
                    raise ValueError('incomplete or non-finite model logits')
                errors = np.max(np.abs(rows - expected), axis=1)
                ordered = np.sort(expected, axis=1)
                margins = ordered[:, -1] - ordered[:, -2]
            # Report every margin without assuming filenames authorize ties.
            # Only the separate acceptance harness can apply its raw0 exemption.
            envelope = bool(np.all(errors <= 1.1) and np.all(errors < margins)
                            and errors[-1] <= 3 * errors[0])
            np.savez(args.out / 'product_reference.npz', step_logits=rows, layer_sig=sigs,
                     router_topk=np.stack(extra['topk']), router_weights=np.stack(extra['topw']),
                     activation_precision=model.precision, forced_route_oracle=False,
                     token_ids=np.array(case['token_ids']), teacher_forcing_tokens=np.array(forced),
                     **model.route_memberships)
            result.update(case=str(args.case_json), reference=str(args.reference),
                          max_logit_difference=errors.tolist(), reference_margin=margins.tolist(),
                          predicted_tokens=np.argmax(rows, axis=1).tolist(),
                          teacher_forcing_tokens=forced, diagnostic_strict_envelope=envelope)
        result['cuda_product_calls'] = products.calls
        result['seconds'] = time.monotonic() - started
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
    finally:
        products.close()


if __name__ == '__main__':
    main()

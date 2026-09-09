#!/usr/bin/env python3
"""Generate an isolated paired-FP32-FMA variant of the production GEMV.

PTX 8.6 introduced independently rounded fma.rn.f32x2 for SM100+:
https://docs.nvidia.com/cuda/archive/12.8.2/parallel-thread-execution/index.html#floating-point-instructions-fma
No production file is modified. Preserve the generated header/hash with results.
"""
import argparse
import hashlib
import json
from pathlib import Path

HELPER = r'''
__device__ __forceinline__ void gemv_f32x2_fma(
    float x, float2 q, float& first, float& second) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("{\n\t"
      ".reg .b64 xx, qq, cc, dd;\n\t"
      "mov.b64 xx, {%2, %2};\n\t"
      "mov.b64 qq, {%3, %4};\n\t"
      "mov.b64 cc, {%5, %6};\n\t"
      "fma.rn.f32x2 dd, xx, qq, cc;\n\t"
      "mov.b64 {%0, %1}, dd;\n\t}"
      : "=f"(first), "=f"(second)
      : "f"(x), "f"(q.x), "f"(q.y), "f"(first), "f"(second));
#else
  first = __fmaf_rn(x, q.x, first);
  second = __fmaf_rn(x, q.y, second);
#endif
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / 'crates/apxinf-cuda/kernels/custom/w4a16_pair.cuh'
    data = source.read_bytes()
    body = data.decode().replace('w4a16_gemv_pair_kernel', 'w4a16_gemv_f32x2_probe_kernel')
    for value in ('xv[u]', 'xv'):
        for name, offset in (('lo', '2*pair'), ('hi', '2*pair+1')):
            other = '2*pair+4' if name == 'lo' else '2*pair+5'
            old = f'acc[{offset}] += {value}*{name}.x;acc[{other}] += {value}*{name}.y;'
            assert body.count(old) == 1, (value, name)
            body = body.replace(old, f'gemv_f32x2_fma({value}, {name}, acc[{offset}], acc[{other}]);')
    header = '#pragma once\n' + HELPER + body.removeprefix('#pragma once\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header)
    info = dict(source=str(source.relative_to(root)), source_sha256=hashlib.sha256(data).hexdigest(),
                output_sha256=hashlib.sha256(header.encode()).hexdigest(),
                change='replace paired independent scalar FP32 FMAs with fma.rn.f32x2; retain all accumulation order')
    args.output.with_suffix('.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info))


if __name__ == '__main__':
    main()

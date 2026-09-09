#!/usr/bin/env python3
"""Generate private routing variants from the current Marlin kernel.

Mode bits: 1 counts valid rows with a warp reduction; 2 skips MMA work on
wholly padded 16-row subtiles. The stripe and K reduction schedules stay fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
source = Path(__file__).resolve().parents[2] / 'crates/apxinf-cuda/kernels/marlin/csrc/moe/marlin_moe_wna16/marlin_template.h'
s = source.read_text()
body = s[s.rindex('template <typename scalar_t,  // compute dtype'):s.rindex('}  // namespace MARLIN_NAMESPACE_NAME')]
assert body.count('__global__ void Marlin(') == 1
body = body.replace('__global__ void Marlin(', '__global__ void MarlinRouting(')
body = body.replace('          const bool fixed_awq\n', '          const bool fixed_awq,\n          const int routing_mode\n')
start = body.index('    block_num_valid_tokens = moe_block_size;')
end = body.index('    __syncthreads();', start)
original = body[start:end]
new = '''    if constexpr (routing_mode & 1) {
      int lane = threadIdx.x & 31;
      int first_invalid = moe_block_size;
#pragma unroll
      for (int base = 0; base < moe_block_size; base += 32) {
        int row = base + lane;
        if (row < moe_block_size &&
            sorted_token_ids_ptr[block_id * moe_block_size + row] >= prob_m * top_k)
          first_invalid = min(first_invalid, row);
      }
      block_num_valid_tokens = __reduce_min_sync(0xffffffff, first_invalid);
    } else {
''' + original + '    }\n\n'
body = body[:start] + new + body[end:]
marker = '''      for (int i = 0; i < thread_m_blocks; i++) {
        if constexpr (m_block_size_8) {'''
assert body.count(marker) == 1
body = body.replace(marker, '''      for (int i = 0; i < thread_m_blocks; i++) {
        if constexpr ((routing_mode & 2) && !m_block_size_8) {
          if (i * 16 >= block_num_valid_tokens) continue;
        }
        if constexpr (m_block_size_8) {''')
header = '// Generated diagnostic derivative; original license follows.\n'
header += s[:s.index('#ifndef MARLIN_NAMESPACE_NAME')]
header += '\nnamespace MARLIN_NAMESPACE_NAME {\n' + body + '}\n'
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(header)
print(json.dumps(dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      output=str(args.out), output_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest())))

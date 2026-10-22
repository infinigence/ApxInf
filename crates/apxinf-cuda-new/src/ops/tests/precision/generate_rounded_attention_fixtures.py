"""Generate a small independent PyTorch fixture for rounded additive attention.

Write to --output <worktree>/devlocal/<feature>/results/<file>.rs, inspect the
result, then curate it into maintained tests. Actual model-shape captures are
kept separately as private evidence.
"""
import argparse
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
b, q, k, h, d = 2, 3, 5, 2, 16
result = [f'// PyTorch {torch.__version__}, {torch.cuda.get_device_name()}.',
          '// Regenerate with generate_rounded_attention_fixtures.py.']
for dtype, name in [(torch.float16, 'F16'), (torch.bfloat16, 'BF16')]:
    def values(shape, multiplier, modulus):
        a = torch.arange(torch.tensor(shape).prod().item(), device='cuda')
        return (((a*multiplier)%modulus-modulus//2)/16.).reshape(shape).to(dtype)
    query = values((b,q,h,d), 3, 17)
    key = values((b,k,h,d), 5, 19)
    value = values((b,k,h,d), 7, 23)
    bias = values((b,h,q,k), 11, 13)
    bias[:,:,:,-1] = torch.finfo(dtype).min
    logits = torch.einsum('bqhd,bkhd->bhqk',query,key) + bias
    probability = logits.float().softmax(-1).to(dtype)
    output = torch.einsum('bhqk,bkhd->bqhd',probability,value)
    bits = output.contiguous().view(torch.uint16).flatten().cpu().tolist()
    result.append(f'pub(super) const {name}: &[u16] = &[')
    result += ['    '+', '.join(f'0x{x:04x}' for x in bits[i:i+16])+',' for i in range(0,len(bits),16)]
    result.append('];')
args.output.write_text('\n'.join(result)+'\n')

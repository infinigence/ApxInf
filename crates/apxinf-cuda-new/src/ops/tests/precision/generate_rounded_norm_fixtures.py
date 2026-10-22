"""Independent CUDA tensor-expression RMSNorm golden; task output is explicit."""
import argparse,json,torch
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
# Exactly representable rational inputs avoid host RNG/library differences.
x=torch.tensor([((i*37)%4093-2046)/128 for i in range(16*4096)],device='cuda').bfloat16().reshape(16,4096)
w=torch.tensor([((i*13)%127+1)/128 for i in range(4096)],device='cuda').bfloat16()
y=(x*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).bfloat16()*w
bits=y.contiguous().view(torch.uint16).cpu().flatten().tolist();h=14695981039346656037
for value in bits:
 for b in [value&255,value>>8]:h=((h^b)*1099511628211)&((1<<64)-1)
a.output.write_text(json.dumps({'torch':torch.__version__,'device':torch.cuda.get_device_name(),'shape':[16,4096],'output_bf16_le_fnv1a64':h},indent=2)+'\n')

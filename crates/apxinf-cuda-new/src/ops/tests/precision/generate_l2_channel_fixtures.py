"""Independent channel-first CUDA L2 golden; explicit task output only."""
import argparse,json,math,torch
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
rows=[]
for name,dtype in [('f32',torch.float32),('f16',torch.float16),('bf16',torch.bfloat16)]:
 for width in [96,192,384]:
  values=[0. if i<width else math.ldexp((-1 if i%3==0 else 1)*(1+(i*37%127)/128),(i*11%23)-11) for i in range(128*width)]
  x=torch.tensor(values,device='cuda',dtype=dtype).reshape(1,8,16,width).permute(0,3,1,2).contiguous()
  eps=1e-4 if dtype==torch.float16 else 1e-12
  y=torch.nn.functional.normalize(x,dim=1,eps=eps).permute(0,2,3,1).contiguous()
  data=y.view(torch.uint8).cpu().flatten().tolist();h=14695981039346656037
  for b in data:h=((h^b)*1099511628211)&((1<<64)-1)
  rows.append({'dtype':name,'width':width,'fnv1a64':h})
a.output.write_text(json.dumps({'torch':torch.__version__,'device':torch.cuda.get_device_name(),'rows':128,'layout':'reference NCHW, output channel rows','goldens':rows},indent=2)+'\n')
print(json.dumps(rows))

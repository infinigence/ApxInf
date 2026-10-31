"""CUDA Torch half-pixel, non-antialiased bicubic fixtures; explicit output."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F

parser=argparse.ArgumentParser();parser.add_argument("output",type=Path);args=parser.parse_args()
values=[((i*13)%37-18)/7 for i in range(2*3*4*2)]
result={"torch":torch.__version__,"device":torch.cuda.get_device_name(),"input":values,"outputs":{}}
for dtype in [torch.float32,torch.float16,torch.bfloat16]:
    x=torch.tensor(values,device="cuda",dtype=dtype).reshape(2,3,4,2).permute(0,3,1,2).contiguous()
    y=F.interpolate(x,size=(5,7),mode="bicubic",align_corners=False,antialias=False)
    result["outputs"][str(dtype)]=y.permute(0,2,3,1).float().cpu().flatten().tolist()
args.output.write_text(json.dumps(result,indent=2)+"\n")

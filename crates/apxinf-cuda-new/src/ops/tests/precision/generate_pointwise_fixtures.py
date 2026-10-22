"""Independent PyTorch golden values for dtype-materialized tanh GELU.

Use --output <worktree>/devlocal/<feature>/results/<file>.json, then curate
the small constants into the maintained test. No checkpoint is needed.
"""
import argparse
import json
import math
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
values = [-3.14, -2.5, -2., -1.5, -1., -0.5, -0.125, 0., 0.125, 0.5, 1., 1.5, 2., 2.5, 3.14, 4., -0.90625, -0.828125, 0.90625, 0.828125]
report = {'torch': torch.__version__, 'device': torch.cuda.get_device_name(), 'input': values, 'results': {}, 'swish': {}}
for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    x = torch.tensor(values, device='cuda', dtype=dtype)
    result = 0.5*x*(1.0+torch.tanh(math.sqrt(2.0/math.pi)*(x+0.044715*torch.pow(x,3.0))))
    report['results'][str(dtype)] = result.float().cpu().tolist()
    report['swish'][str(dtype)] = (x * torch.sigmoid(x)).float().cpu().tolist()
args.output.write_text(json.dumps(report, indent=2)+'\n')

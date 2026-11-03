"""Generate CUDA Torch FP64-complex rotation fixtures at an explicit path."""
import argparse
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
values = [((i * 11 % 67) - 33) / 16 for i in range(48)]
values[0:2] = [1.0, 0.0]
angles = torch.arange(12, dtype=torch.float64, device="cuda").reshape(3, 4) / 7
coeff = torch.view_as_real(torch.polar(torch.ones_like(angles), angles))
# Discriminates the reference cast through FP32 from a direct FP64->BF16 cast.
coeff[0, 0] = torch.tensor([1.00390625 + 1e-10, 0.0], device="cuda", dtype=torch.float64)
report = {"torch": torch.__version__, "device": torch.cuda.get_device_name(),
          "values": values, "coefficients": coeff.cpu().flatten().tolist(), "outputs": {}}
for dtype in [torch.float16, torch.bfloat16]:
    x = torch.tensor(values, device="cuda", dtype=dtype).reshape(3, 2, 8)
    z = torch.view_as_complex(x.double().reshape(3, 2, 4, 2))
    y = torch.view_as_real(z * torch.view_as_complex(coeff)[:, None]).flatten(2).to(dtype)
    report["outputs"][str(dtype)] = y.float().cpu().flatten().tolist()
args.output.write_text(json.dumps(report, indent=2) + "\n")

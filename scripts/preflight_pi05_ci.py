#!/usr/bin/env python3
"""Check an operator-approved Jetson environment; print the measurement receipt."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=15).strip()


def main() -> None:
    board = sys.argv[1]
    path = Path(os.environ["APXINF_CI_ENVIRONMENT"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != os.environ["APXINF_CI_ENVIRONMENT_SHA256"]:
        raise ValueError("unapproved environment profile")
    approved = json.loads(path.read_text())
    ceiling = float(approved["temperature_ceiling_c"])
    if not math.isfinite(ceiling) or ceiling <= 0:
        raise ValueError("invalid approved temperature ceiling")
    compatible = Path("/proc/device-tree/compatible").read_bytes().decode().replace("\x00", " ")
    chip = {"thor": "tegra264", "orin": "tegra234"}[board]
    if chip not in compatible or approved["hardware"] != board:
        raise ValueError("incorrect board")
    receipt = {"hardware": board, "compatible": compatible,
               "nvpmodel": command("nvpmodel", "-q"),
               "clocks": command("sudo", "-n", "/usr/bin/jetson_clocks", "--show")}
    for name in ("nvpmodel", "clocks"):
        if hashlib.sha256(receipt[name].encode()).hexdigest() != approved[f"{name}_sha256"]:
            raise ValueError(f"{name} differs from calibrated environment")
    temperatures = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        name = (zone / "type").read_text().strip()
        if "gpu" in name.lower() or "cpu" in name.lower() or "soc" in name.lower():
            temperatures[name] = int((zone / "temp").read_text()) / 1000
    if not temperatures or max(temperatures.values()) >= ceiling:
        raise ValueError("temperature outside approved performance envelope")
    # Thor's tegrastats may omit GR3D_FREQ entirely. Orin lacks universal SMI.
    if board == "thor":
        utilization = []
        for _ in range(3):
            utilization.append(command("nvidia-smi", "--query-gpu=utilization.gpu",
                                       "--format=csv,noheader,nounits"))
            time.sleep(1)
        output = "\n".join(utilization)
    else:
        process = subprocess.Popen(["stdbuf", "-oL", "tegrastats", "--interval", "1000"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            output, _ = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            output, _ = process.communicate(timeout=5)
        utilization = re.findall(r"GR3D_FREQ\s+(\d+)%", output)
    if not utilization or any(int(value) != 0 for value in utilization):
        raise ValueError("GPU busy or utilization unavailable")
    receipt.update(temperatures_c=temperatures, gpu_utilization=output)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

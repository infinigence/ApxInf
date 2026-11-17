# Building ApxInf on Windows with CUDA

Native Windows build of the CUDA runtime, the `apxinf` binary and the
`apxinf_py` Python package. Developed and checked on an RTX 5090 Laptop.

## Prerequisites

1. NVIDIA driver with CUDA 13 support .
2. CUDA Toolkit 13.x. On Windows the libraries are in `lib\x64` and the DLLs in `bin\x64`.
3. Visual Studio 2022 Build Tools: "Desktop development with C++" (MSVC v143, Windows SDK,
   C++ CMake tools).
4. Rust (`stable-x86_64-pc-windows-msvc`).
5. **Git on PATH.** `crates/apxinf-cuda-new/build.rs` applies the FlashAttention2 patch with
   `git apply` on Windows (`patch` on Linux).
6. Miniconda, only for the Python package.

## Build

From a normal PowerShell:

```powershell
# apxinf.exe only
powershell -NoProfile -File scripts\build-win.ps1

# plus the Python package (conda env, wheel, editable python/apxinf)
powershell -NoProfile -File scripts\build-win.ps1 -Wheel

# behind a network that cannot reach crates.io / PyPI / Anaconda: rsproxy + Tsinghua mirrors
powershell -NoProfile -File scripts\build-win.ps1 -Wheel -Mirror
```

Options: `-TargetDir` (default `C:\apx-target`, keep it short), `-CudaPath`, `-CudaArch`
(default from `nvidia-smi`), `-EnvName`, `-Conda`, `-Jobs` (nvcc/cargo parallelism, default 2),
`-DebugProfile`, `-Mirror`.

## LIBERO closed loop

Closed-loop simulation of `pi05_libero_base` on the same machine, in-process: the
policy is built by `AutoPolicy` inside `scripts/eval_libero.py`, which drives LIBERO/MuJoCo.

### Extra prerequisites

- Everything in [Prerequisites](#prerequisites), plus network access to GitHub and Hugging Face
  (LIBERO is cloned; the checkpoint is downloaded).
- An interactive Windows desktop session. MuJoCo renders through `glfw` (Windows has no EGL), which
  needs an OpenGL context; it does not work from a Windows service or a session without a display.
- About 5 GB free disk (LIBERO with assets 0.4 GB, torch and the checkpoint the rest) and 24 GB of GPU memory
  is comfortable for bf16.

### 1. Install the simulator

```powershell
# builds apxinf_py, then installs the pinned LIBERO dependencies and clones LIBERO at a fixed commit
powershell -NoProfile -File scripts\build-win.ps1 -Wheel -Libero [-Mirror]
```

`-Libero` installs [scripts/requirements-libero-win.txt](../scripts/requirements-libero-win.txt)
, clones
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) at `-LiberoRev` into `-LiberoDir`
(default `C:\apx-libero`), puts it on `sys.path` through a `.pth` file (its own `requirements.txt` pins `numpy==1.22.4`, so it is not pip-installed),
generates a non-interactive LIBERO config under `<LiberoDir>\_cfg`, and checks that the `libero_10`
suite imports. `-Mirror` only affects crates, conda and pip; the LIBERO clone always uses GitHub.

### 2. Get the checkpoint

```powershell
pip install -U "huggingface_hub[cli]"
huggingface-cli download lerobot/pi05_libero_base --local-dir D:\models\pi05_libero_base
curl.exe -fL https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json -o D:\models\pi05_libero_base\norm_stats.json
```

The `norm_stats.json` step is required: the Hugging Face repository lost its statistics, and without them
actions are not de-normalized and every episode fails.

### 3. Run

```powershell
# smoke check: libero_10, task 0, 10 trials, bf16
powershell -NoProfile -File scripts\run-libero-win.ps1 -ModelDir D:\models\pi05_libero_base
```

Options: `-Suite`, `-Tasks` (comma list or `all`), `-Trials`, `-Precision`, `-Seed` (default 7),
`-OutDir` (default `devlocal\libero-win\<suite>-<timestamp>`), `-LiberoDir`, `-EnvName`, `-CudaPath`,
`-MujocoGl`. The script wraps `scripts/eval_libero.py --backend in-process`
(see the README "LIBERO evaluation" section for every flag); the published protocol is
`-Tasks all -Trials 50`.

Output in `-OutDir`: `results.jsonl` (one row per trial, fsync'd, so a re-run with the same `-OutDir`
resumes), `summary.json` (success rate and per-segment latency) and `run.log`. The script exits
non-zero and prints `INCOMPLETE` if fewer episodes finished than requested.

A `[Warning]: datasets path ... does not exist!` line at start-up is harmless: rollouts use the
bundled init states, not the LIBERO demonstration datasets.

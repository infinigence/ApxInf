<#
.SYNOPSIS
  Closed-loop LIBERO evaluation of pi05_libero_base on native Windows + CUDA (in-process, no server).

.DESCRIPTION
  Wraps scripts\eval_libero.py --backend in-process. Prerequisite: scripts\build-win.ps1 -Wheel -Libero.
  Default is the smoke check: libero_10, task 0, 10 trials, bf16.

    powershell -NoProfile -File scripts\run-libero-win.ps1 -ModelDir D:\models\pi05_libero_base

  The model directory needs model.safetensors, config.json, paligemma_tokenizer.model and norm_stats.json
  (see doc/windows-cuda.md "LIBERO closed loop"). Runs are resumable: re-running with the same -OutDir
  skips finished trials.

  The file is ASCII on purpose: Windows PowerShell 5.1 misreads BOM-less UTF-8.
#>
param(
  [Parameter(Mandatory = $true)][string]$ModelDir,
  [string]$Suite     = 'libero_10',
  [string]$Tasks     = '0',               # comma list of task ids, or 'all'
  [int]$Trials       = 10,                # trials per task
  [string]$Precision = 'bf16',
  [int]$Seed         = 7,
  [string]$OutDir    = '',                # default: <repo>\devlocal\libero-win\<suite>-<timestamp>
  [string]$LiberoDir = 'C:\apx-libero',
  [string]$EnvName   = 'apxinf-win',
  [string]$CudaPath  = $(if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0' }),
  [string]$Conda     = '',
  [string]$MujocoGl  = 'glfw'             # Windows has no EGL; glfw needs a desktop session (not a service/RDP-less session)
)
$ErrorActionPreference = 'Continue'
$Repo = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) '..')).Path
function Need($ok, $msg) { if (-not $ok) { throw "PREREQUISITE MISSING: $msg" } }

if (-not $Conda) {
  $cands = @($env:CONDA_EXE, "$env:USERPROFILE\miniconda3\Scripts\conda.exe", "$env:USERPROFILE\anaconda3\Scripts\conda.exe", 'C:\ProgramData\miniconda3\Scripts\conda.exe', 'C:\ProgramData\anaconda3\Scripts\conda.exe')
  $Conda = ($cands | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1)
  if (-not $Conda) { $g = Get-Command conda -ErrorAction SilentlyContinue; if ($g) { $Conda = $g.Source } }
}
Need ($Conda -and (Test-Path $Conda)) 'conda: pass -Conda C:\path\to\conda.exe'
$envDir = (& $Conda info --base | Select-Object -First 1).Trim() + "\envs\$EnvName"
$py = "$envDir\python.exe"
Need (Test-Path $py) "env python not found: $py (run scripts\build-win.ps1 -Wheel -Libero first)"
foreach ($f in 'model.safetensors', 'config.json', 'paligemma_tokenizer.model', 'norm_stats.json') {
  Need (Test-Path (Join-Path $ModelDir $f)) "$f not found in $ModelDir (see doc/windows-cuda.md)"
}
$cfgDir = Join-Path $LiberoDir '_cfg'
Need (Test-Path "$cfgDir\config.yaml") "LIBERO not set up under $LiberoDir (run scripts\build-win.ps1 -Wheel -Libero)"
if (-not $OutDir) { $OutDir = Join-Path $Repo ("devlocal\libero-win\$Suite-" + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Force $OutDir | Out-Null

$env:CUDA_PATH = $CudaPath
$env:PATH = "$CudaPath\bin\x64;$CudaPath\bin;$envDir;$envDir\Scripts;$env:PATH"
$env:LIBERO_CONFIG_PATH = $cfgDir
$env:MUJOCO_GL = $MujocoGl
$env:PYTHONUNBUFFERED = '1'

Write-Host "env: $envDir`nmodel: $ModelDir`nsuite: $Suite tasks: $Tasks trials/task: $Trials precision: $Precision`nout: $OutDir"
& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# --norm-stats is REQUIRED for this checkpoint: without it actions are not de-normalized and every episode fails.
& $py "$Repo\scripts\eval_libero.py" --backend in-process --model-dir $ModelDir `
  --norm-stats "$ModelDir\norm_stats.json" --precision $Precision --action-horizon 10 `
  --suite $Suite --tasks $Tasks --trials-per-task $Trials --seed $Seed `
  --results-jsonl "$OutDir\results.jsonl" --summary-json "$OutDir\summary.json" 2>&1 | Tee-Object "$OutDir\run.log"
$code = $LASTEXITCODE
if (Test-Path "$OutDir\summary.json") {
  Write-Host "`n--- summary ($OutDir\summary.json) ---"; Get-Content "$OutDir\summary.json"
  # eval_libero.py can exit 0 with no episode finished (e.g. a missing dependency): do not report that as success.
  $s = Get-Content "$OutDir\summary.json" -Raw | ConvertFrom-Json
  if ($s.completed_runs -lt $s.expected_runs) {
    Write-Host "INCOMPLETE: $($s.completed_runs)/$($s.expected_runs) episodes finished, see $OutDir\run.log" -ForegroundColor Red
    if ($code -eq 0) { $code = 1 }
  } else { Write-Host "DONE: $($s.successes)/$($s.completed_runs) episodes succeeded" -ForegroundColor Green }
} elseif ($code -eq 0) { $code = 1 }
exit $code

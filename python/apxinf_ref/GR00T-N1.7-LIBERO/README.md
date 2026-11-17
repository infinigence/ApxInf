# NVIDIA GR00T-N1.7-LIBERO Torch eager reference

Independent model-core gold, following the `apxinf_ref/pi05` split between a
torch-free artifact module and a device-dependent capture module. Full model
identity is `nvidia/GR00T-N1.7-LIBERO`, task checkpoint `libero_10`, BF16.
No ApxInf runtime, policy adapter, kernel, or rewritten flow loop is used.

The reference calls the installed NVIDIA `Gr00tN1d7.get_action`. Supply an
Isaac-GR00T source tree containing N1.7 and a compatible environment with
PyTorch, transformers (Qwen3-VL support), numpy, and the NVIDIA source's
dependencies. No automatic installation or network download is performed.
The upstream model is not vendored; source file hashes are recorded instead.

## Generate on Orin

```bash
python generate_GR00T_N1_7_LIBERO_torch_gold.py \
  --source path_to/Isaac-GR00T \
  --checkpoint path_to/GR00T-N1.7-LIBERO/libero_10 \
  --backbone path_to/Cosmos-Reason2-2B \
  --device cuda:0 \
  --input path_to/libero-two-view \
  --output path_to/GR00T-N1.7-LIBERO-gold
```

Use the exact M1000 fixture, not independently generated random tensors or
reprocessed images. `--input` accepts the Rust preprocessed fixture directory
or an earlier artifact's `inputs.npz`. BF16 inputs retain uint16 bit patterns.
The source model receives the fixture noise at its original random draw; the
number and shape of draws are checked. The model's own flow schedule is used
and recorded. The local backbone constructor is populated by the full GR00T
checkpoint, with missing/unexpected weights rejected. The raw-observation
collator is bypassed because inputs are already preprocessed.

This is normalized model-core gold, not decoded robot actions or LIBERO success
evaluation. Capture defaults to three warmups and two exact-repeat checks; it
is not the W10/R50 performance benchmark. Existing outputs are never replaced.

The reference environment used for the current artifact is torch
2.9.1a0+gitd38164a, CUDA 12.6, transformers 4.57.3, numpy 1.26.4, SDPA
(no flash_attn), seed 0, with cudnn enabled/benchmark/deterministic and the
tf32 settings recorded in the manifest runtime block. Cross-process repeat
means a second independent capture with the same environment and seed;
compare the manifest hashes and debug.npz bit-for-bit.

## Artifact

- `inputs.npz`: original canonical fixture, including raw BF16 noise and embodiment.
- `debug.npz`: full normalized actions in float32 (exact widening of model output).
- `reference.json`: `output.shape/values`, directly consumable by
  `scripts/gr00t/compare_outputs.py --reference` and the benchmark parity wrapper.
- `manifest.json`: full model/config/source/weight identity, runtime, file and
  tensor SHA-256, capture checks, and encoding metadata.

Files are staged, read back, checked, then published as one new directory.
Before scoring on M1000, verify fixture bytes and checkpoint hashes against
the manifest. A shape-compatible result alone does not prove matching inputs.
The artifact module intentionally does not depend on torch; unit tests can run
with `python -m unittest discover -s tests -v`.

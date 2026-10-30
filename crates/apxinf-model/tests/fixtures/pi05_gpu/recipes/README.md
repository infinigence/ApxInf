# Qualified PI0.5 CUDA recipes

`thor.json` and `orin.json` map native `.recipe` filenames to their exact text.
The CI runner materializes a private copy under its run directory and disables
autotuning. They are public numerical regression assets, not universal performance
recommendations. Reports record the initial bank digest and preserve its files.

These banks were exercised on merged production source `9a1dbce7`, with the
`pi05_reference_gate` example committed in `8b7b73ea`. Recipe headers pin their
native build namespace, CUDA 13.2, driver 13.2, cuBLASLt 130400 and device geometry
(Thor SM110 / Orin SM87). Requalify after environment or kernel changes; an
incompatible namespace may cause native fallback. Passing a different runtime
requires actual measurements, not renaming recipe hashes.

Thor BF16/FP8 reproduce the original cosine baseline. Orin BF16 uses three
previously qualified plain-GEMM choices in the current native recipe format;
a controlled run with those choices restored the original outputs exactly.
Unrestricted fresh autotuning selected different tactics and failed the same
cosine floors. The floors were not reduced. INT8 is an advisory path.

For a deliberate new qualification, use `run_pi05_reference_gate.py --tuning DIR
--autotune` (an empty directory retunes all operations). Then replay without
`--autotune`, inspect all required case/mode results and preserve the environment
and reports before updating these banks. Improved results should raise the
per-case floors in `configs/pi05/reference-gate.json`; never lower existing floors
or change inputs to accommodate a faster but less accurate tactic.

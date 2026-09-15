# Where the Qwen-Drive divergence comes from

Measured on a Jetson AGX Orin (sm_87, BF16, CUDA 12.6) with the official
reference at the pinned revision `28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0`
running on the same machine, against the same fixed public inputs. Scene 0,
whose native run diverges at generated index 121.

## Finding

There is no first-diverging layer to find. The native stack already matches the
reference to within BF16 rounding at every layer, and what separates the two
runs is amplification through the GDN recurrence, not a defect in one operator.

Prefill, relative error `||native - ref|| / ||ref||` per row:

| layer | row 0 (first token) | row 3094 (last) |
|---|---|---|
| 0 | 0.0030 | 0.0077 |
| 15 | 0.0026 | 0.0229 |
| 31 | 0.0060 | 0.0315 |

BF16 carries a relative precision of `2^-8 = 0.0039`. Layer 0 row 0 sits at
**0.77 of one ULP** -- as close as the format allows, so nothing there is
wrong. Row 3094 starts at about two ULP because its recurrent state has already
absorbed 3095 tokens, and grows fourfold across the stack.

Decode compounds it further. At decode step 0 the layer-0 relative error is
already 0.41 and reaches 1.2 by layer 31, because every decode step starts from
the state prefill left behind.

The token sequence survives all of this until a near-tie finally tips. At
generated index 121 the native logits are 357 = 21.5 and 2532 = 21.5, an exact
tie that index order breaks toward 357; the reference has 2532 = 21.875 against
357 = 21.25, a 0.625 margin. The 4090 record shows the same shape at scene 2
index 87: native 2763 = 24.125 and 12682 = 24.125, reference 24.125 against
24.25.

## What this explains

- **Why the 4090 passes scenes 0 and 1 and fails scene 2 at 87.** The
  amplification happens everywhere; only the arrival of a near-tie differs.
- **Why Orin fails earlier, at scene 0 index 121.** sm_87 selects a different
  set of kernels, so the per-step rounding gap is larger and amplifies faster.
- **Why both repair attempts regressed scene 0.** Approximate exp2 in the
  recurrent decay and 128-dimensional recurrent tree reductions each changed
  rounding behaviour; under an amplifying recurrence any perturbation can tip
  some other near-tie the wrong way. Neither was a wrong fix for a right
  diagnosis -- the diagnosis was that a single operator was broken.

## Two measurement errors that hide this

**The decode trace is one step out of phase with the reference.** In the native
loop, the forward labelled `decode_step = k` runs at the end of iteration `k`
and produces the logits for `generated[k+1]`, while the reference's `k`-th
`vlm` forward produces `generated[k]`. Native `k` must be compared against
reference `k+1`. `layer_reference.py` uses `step[0] == 87` on both sides.

**A decode-step comparison cannot locate an origin.** The GDN state carries
whatever prefill produced, so every decode step begins from an already-diverged
state and every layer looks equally wrong. The origin is only visible in
prefill, and specifically in row 0, the token whose recurrent state has no
history.

## Reproducing

Reference and native on one machine. The reference is the package named in the
model card, imported through `PYTHONPATH` rather than installed:

```bash
export PYTHONPATH=<qwen-drive-source>/src
export APXINF_QWEN_ROOT=<root with models/ and outputs/>
export QWEN_DRIVE_REF_SRC=<qwen-drive-source>

# native traces: per-layer prefill rows, and decode rows at one step
APXINF_QWEN_TRACE_DIR=$PWD/trace APXINF_QWEN_TRACE_DECODE_STEP=0 \
  python3 native_trace.py 0 8

# origin: prefill, including the first token's row
python3 prefill_compare.py 0 $PWD/trace

# accumulated: one decode step, native k against reference k+1
python3 layer_compare.py 0 1 $PWD/trace 8
```

`APXINF_QWEN_TRACE_DECODE_STEP` now gates the per-layer decode trace as well as
the logits readback; without that gate every step overwrote the previous one
and only the last survived.

## Correction: the index is a sample, not a property

Everything above describes a real mechanism, but the specific index it names is
not reproducible and should not be used as a regression signal. The same binary
in the same configuration produced `first_different_index` 121, 121 and 255 on
three consecutive runs.

The cause is upstream of the recurrence. The FA2 head-256 causal prefill kernel
does not reproduce itself: given q, k and v identical bit for bit, about one
output element in ten thousand comes back one BF16 ULP different, and 982 of
1083 differing elements in a measured pair were exactly one ULP. Full-tensor
prefill traces place it exactly -- vision reproduces, text layers 0, 1 and 2
reproduce, and text layer 3, the first full-attention layer, does not; inside
layer 3 the traced input_norm, fused_qkv, cos, sin, q, k and v are identical
and only the attention output differs. Under `APXINF_CUDA_SKIP_OUTPUT_ZERO=poison`
that output contains no NaN, so nothing uninitialised is being read and every
element is written. It is FP32 accumulation order varying between runs in a
kernel with no atomics and a fixed grid.

`APXINF_ATTN_COMPOSED_PREFILL=1` routes the same call through the composed path,
which does reproduce: four runs identical, and the divergence index then holds
at 121 across repeated runs. It costs about 4.9% on the fixed VQA workload.

Two consequences for anyone working from this document. A single-run token hash
proves nothing, so any bit-exactness claim has to be made with the composed
prefill enabled. And a moved divergence index is not by itself evidence that a
change altered the arithmetic -- repeat it before drawing that conclusion.

## What would actually move this

Not a hunt for a broken kernel. The question is which operator contributes the
most rounding divergence per step, and whether it can be brought closer to the
reference's arithmetic -- the recurrent decay's `ex2.approx.f32`, the BF16
round trips through the chunk scan, and the reduction orders in the chunked
products are where to look. That is tuning numerical behaviour, and the gate to
beat is the fixed four-mode verifier, unchanged.

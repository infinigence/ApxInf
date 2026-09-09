# Marlin MoE W4A16 kernel

Source: https://github.com/vllm-project/vllm/tree/v0.10.2/csrc/moe/marlin_moe_wna16
and its gptq_marlin/core dependencies, pinned to v0.10.2. Apache-2.0;
original copyright notices and LICENSE are retained.

Local changes: removed PyTorch-only includes and host registration wrappers;
added explicit standard-library includes and a standalone ScalarType check;
made the AWQ repack namespace configurable. The group-128 BF16 mainloop also
accepts FP16 scales: integer differences are multiplied in FP32 and rounded once
to BF16, matching the checkpoint dequantization. This avoids a lossy BF16 scale
conversion that failed the full-model logit gate.
Only the AWQ U4, BF16, group-128, M32/N256/K64, four-stage variant is instantiated
by the ApxInf adapter. Model acceptance is required before enabling this path.

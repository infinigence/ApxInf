# Native MiniCPM acceptance

Native MiniCPM5 uses explicit BF16 greedy, sampling, thinking, and bounded-thinking profiles.
Historical retrieval checks passed at 2040, 8190, and 15998 input tokens with 128 output tokens.
A selected prefill cancellation check allowed a later request. These samples do not qualify every context and output combination.
The bounded profile counts its 512 hidden reasoning tokens against the output budget.
Claude tool-loop completion did not satisfy the pelican drawing criterion. The recorded visual acceptance checks failed.
Larger budgets, natural thinking closure, and repetition diagnostics did not resolve that task.
These failures do not establish a native arithmetic defect or general model incapacity.
This split repeats CPU checks only. Native MLX compilation and device execution require separate verification.

## Evidence scope

Historical observations come from the pinned fork tree `d82ae3b`. They are not new measurements from this split.
The local split report records the CPU commands and results for each branch.

Original records remain in the fork history:

- [minicpm-claude-priorities-20261022.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/minicpm-claude-priorities-20261022.md).
- [minicpm-closure-validation-20261023.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/minicpm-closure-validation-20261023.md).
- [minicpm-default-context-validation-20261023.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/minicpm-default-context-validation-20261023.md).
- [minicpm-native-validation-20261022.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/minicpm-native-validation-20261022.md).
- [minicpm-reference-validation-20261023.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/minicpm-reference-validation-20261023.md).
- [native-bounded-thinking-validation-20261023.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/native-bounded-thinking-validation-20261023.md).

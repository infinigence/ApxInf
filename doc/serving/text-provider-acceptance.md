# Text provider acceptance

Historical checks used Qwen3.5-2B and Claude Code 2.1.289 with a restricted service profile.
Selected Read, Edit, Bash, and cancellation tasks completed. These samples do not establish general Claude compatibility or a task success rate.
The first cancellation classification check failed. A later change corrected the classification.
The MLX provider requires the declared interpreter, package versions, model identity, and execution profile.
CPU tests use fake runtimes. They do not load a model or establish device performance.

## Evidence scope

Historical observations come from the pinned fork tree `d82ae3b`. They are not new measurements from this split.
Later checks must state their commit and configuration separately.

Original records remain in the fork history:

- [claude-code-validation-20261009.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/claude-code-validation-20261009.md).
- [serving-validation-20261009.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/serving-validation-20261009.md).
- [validation-20261011.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/validation-20261011.md).

# Vision acceptance

Vision uses a separate MiniCPM-V-4.7-1B Python provider. It does not add images to the native MiniCPM5 worker.
Historical checks covered serial text, two PNG inputs, cancellation, a later request, and selected HTTP requests.
Preparation replay and state restoration require checked identities. Unknown state requires worker fencing.
Retrieval checks at 4094, 8180, and 14999 input tokens used direct runtime diagnostics.
They do not establish HTTP or Claude acceptance at those lengths.
The real Claude image revision attempt failed on path handling, binary reads, and undeclared tools.
No successful image Read/Edit acceptance was recorded. The drawing criterion also remained unmet.
Broader media inputs, maximum output lengths, and the full memory range remain unverified.

## Evidence scope

Historical observations come from the pinned fork tree `d82ae3b`. They are not new measurements from this split.
The local split report records the CPU commands and results for each branch.

Original records remain in the fork history:

- [minicpm-vision-probe-validation-20261023.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/minicpm-vision-probe-validation-20261023.md).
- [minicpm-vision-transport-validation-20261023.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/minicpm-vision-transport-validation-20261023.md).
- [minicpm-vision-worker-validation-20261023.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/minicpm-vision-worker-validation-20261023.md).

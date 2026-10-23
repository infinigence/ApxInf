# Clef acceptance

The /v1/systemone endpoint provides text decisions with noul, choice, and score results.
It is not a chat or Anthropic provider. Each attempt uses one worker process.
Success requires clean EOF, exit status zero, and process reaping before a replacement worker loads.
The default macOS host-pressure gate did not pass.
All positive hardware HTTP and lifecycle results used an explicit diagnostic configuration with that gate disabled.
The longest recorded successful input contained 2828 tokens. A complete forward pass at the 4096-token boundary remains unverified.
Images, streaming, generation, thinking, and throughput acceptance are outside this profile.
This split runs CPU fixtures only and does not change those hardware acceptance limits.

## Evidence scope

Historical observations come from the pinned fork tree `d82ae3b`. They are not new measurements from this split.
The local split report records the CPU commands and results for each branch.

Original records remain in the fork history:

- [clef-serving-validation-20261023.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/clef-serving-validation-20261023.md).
- [clef-validation-20261023.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/clef-validation-20261023.md).

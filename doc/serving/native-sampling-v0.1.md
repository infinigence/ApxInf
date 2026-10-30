# Native MiniCPM sampling profile

Status: implementation contract. Runtime acceptance remains pending.
The [serving contract](contracts-v0.1.md) retains precedence.
Python worker protocol 2.0 and native greedy protocol 2.1 retain their existing behavior.
This profile introduces no dependency or new sampling algorithm.

## Selection and public requests

The operator explicitly selects the native sampled profile before startup.
Its worker protocol is `apxinf-worker/2.2`.
Its public profile name is `apxinf-native-sampled-v0.1`.
The model provider remains `metal_minicpm5`, with BF16 weights and variant `bf16-compiled`.
The profile supports one text sequence, 256-token prefill chunks, and at most 16384 positions.
It does not enable image input, state reuse, or model thinking.

Omitted sampling parameters select temperature 1.0 and top-p 0.95.
An explicit temperature of zero selects greedy generation.
Greedy generation permits an omitted top-p or an explicit top-p of 1.0.
A positive temperature selects random generation and must not exceed 2.0.
Random generation requires top-p greater than zero and at most 1.0.
Both values must remain finite and positive after conversion to binary32.
An omitted temperature with explicit top-p uses temperature 1.0.
Reject public top-k, seed, and penalty parameters.
The existing greedy profiles retain their previous defaults and parameter checks.

The internal request holds exactly one selection policy.
`Greedy` contains no random parameters.
`Random` contains temperature and top-p.
The coordinator retains this policy through queueing and submission.

## Worker submission

A 2.2 submit contains the existing 2.1 fields and one required `sampling` object.
A greedy object contains exactly `mode="greedy"`.
A random object contains exactly `mode="random"`, `temperature`, `top_p`, and `seed`.
Temperature and top-p use the numeric ranges above.
Seed is an integer from zero through 9007199254740991.
Reject Boolean values in numeric fields.
Reject unknown fields, missing fields, and mode-field conflicts.
Protocols 2.0 and 2.1 reject the sampling object.
Check the complete policy before model mutation.

The coordinator derives the seed from the request's lowercase UUID text.
Hash the bytes `apxinf-sampling-seed-v1`, one zero byte, then the ASCII UUID with SHA-256.
Interpret the first eight digest bytes as an unsigned big-endian integer.
Keep its least significant 53 bits.
All attempts for that request use the same seed.
Initialize the sampler with `RngKey(seed, 0, 0)` once per attempt.
The sampler owns its draw counter and history.
Do not share them between requests.

## Capability and execution identity

The 2.2 ready frame requires `sampling=true` and `greedy=true`.
Other capability fields retain the native 2.1 meanings and limits.
Its cancellation granularity must be `token_or_prefill_chunk`.
The runtime object retains the native 2.1 fields and loaded-library checks.

The model manifest declares `adapter_revision=apxinf-native-minicpm5-sampled-v1`.
Its execution object adds exactly this sampling description:

```json
{
  "algorithm": "apxinf-cpu-categorical-v1",
  "temperature": "1",
  "top_p": "0.95",
  "seed_policy": "request-id-sha256-v1"
}
```

Store this object in `execution.sampling`.
Identity defaults use strings because canonical identity objects exclude floating-point numbers.
Submit parameters remain JSON numbers.
The execution object otherwise retains the native 2.1 fields, including `enable_thinking=false`.
The launcher checks the selected execution identity before readiness.
Readiness exposes the sampled public profile and sampling capability.

## Generation and resources

The shared model loop accepts a caller-selected `TokenSampler` through one explicit entry.
It initializes the sampler and retains the existing EOS, output limit, and control behavior.
Check sampler vocabulary and capacity before model mutation.
Existing entries retain their backend sampler and use the same loop.

The native worker owns a private adapter with a cloned MLX backend and an existing CPU sampler.
The adapter transfers the selected logits row and delegates selection to that CPU sampler.
Random requests use this adapter. Greedy requests retain the Metal greedy path.
The worker uses the existing autoregressive loop and random algorithm.
CPU selection can delay the next token control boundary.
Checked device settlement and request cleanup remain required before capacity release.

## Verification

Rust and Python check the same original 2.2 fixtures.
Keep all existing 2.0 and 2.1 fixture meanings unchanged.
Test complete ready frames, both submit modes, seed bounds, invalid numbers, and mode-field conflicts.
Test identity digests and unchanged rejection behavior for earlier protocols.
Check zero output, sampler initialization, EOS, and cancellation in the shared loop.
Check actual HTTP parameter propagation and real Claude Code tool calls.
EOS completion does not establish drawing quality or visual task acceptance.

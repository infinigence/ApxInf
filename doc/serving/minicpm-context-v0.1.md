# Native MiniCPM5 context profile

Status: implementation profile 0.1. Historical observations have their original source and runtime scope.
Drawing quality and general long-context quality remain unverified.
The [serving contract](contracts-v0.1.md) controls admission, cancellation, and resource settlement.
This profile introduces no runtime dependency and changes no v1 protocol.
The model tests call the existing `apxinf-tokenizer` crate to encode original prompt fixtures.

## Controlled prefill interface

`LlmTrait::prefill_controlled` receives one complete `LlmInput` and a mutable callback.
The callback receives `GenerationBoundary` and returns `Result<bool>`.
A true result permits continued execution.
A false result stops execution and returns `Ok(None)`.
A callback error propagates to the caller.
Successful prefill returns `Ok(Some(logits))` for the shared sampler.

`PrefillStart` is a control boundary before prompt execution.
It claims no consumed positions and produces no progress frame.
`PrefillProgress { consumed_tokens }` reports evaluated prompt positions.
`Token` retains its existing meaning.
The default hook calls the existing whole-prompt `prefill` and reports one completed progress boundary.
The shared generation loop calls this hook without another prefill progress notification.

MiniCPM checks the complete prompt before the first control callback or KV mutation.
Validation includes image exclusion, token range, prepared prompt length, current offset, and capacity.
An invalid token in a later chunk rejects the complete prompt before execution.
The hook returns no logits after cancellation, including cancellation after the final chunk.
The caller retains responsibility for request cleanup with device synchronization after cancellation or failure.

## Model execution

A prefill chunk contains at most 256 consecutive prompt tokens.
Ordinary MiniCPM prefill executes these chunks in order.
Each chunk uses absolute RoPE positions and a causal mask over prior and current positions.
Each chunk evaluates its outputs before it publishes KV state, advances the offset, and reports progress.
The model checks control before the first chunk and after each completed chunk.
The final chunk returns one logit row for sampling.

The model admits each chunk at its exact prepared offset and row count.
A one-row tail uses the same eager prefill path as other chunks.
Preparation warms the complete prompt, each distinct chunk length, and the supported decode lengths.
`prepared_sequence_lengths` reports those admitted lengths in ascending order.
The full `forward` operation retains all input logit rows and its complete-prompt admission rule.
The shared sampler and EOS behavior remain unchanged.

DSpark retains whole-prompt prefill and all prompt taps.
Its context limit remains 4096 positions.
Chunked prefill does not establish DSpark support at larger lengths.

## Context limits

The initial equivalence gate passed with the existing 4096-position limit.
The ordinary BF16 model permits a 16384-position profile.
The limit applies to `bf16-public` and `bf16-compiled` only.
Variant limits control RoPE table size, preparation, and token admission together.
The [support overview](support-status.md) separates historical observations from current qualification.
These gates do not establish equivalent quality for every task or variant.
A 32768-position profile remains deferred.

Admission includes the complete prepared input and maximum output allowance.
The effective limit is the smaller model limit and deployment limit.
No path truncates a prompt, shifts context, or reuses a previous request prefix.
A chunk bounds query rows but does not bound total request memory.

## Verification gate

The real-checkpoint test loads the pinned BF16 weights once in one GPU process.
Its original prompt fixtures use the existing checkpoint tokenizer.
It compares chunked final logits with the final row from full `forward` at 255, 256, 257, and 513 tokens.
Both results must have finite values and the same greedy next token.
The declared comparison tolerance is absolute error 0.5 plus relative error 0.02 per logit.
This BF16 tolerance permits changes in reduction order without permitting a changed greedy choice.

The test also rejects an invalid token after the first chunk before any KV mutation.
Cancellation before prefill must consume zero tokens.
Cancellation after the first chunk must consume exactly 256 tokens and return no logits.
Release after device synchronization must permit a successful subsequent request on the same instance.
CPU checks cover shape admission, context boundaries, overflow, and shared control behavior.

Larger context acceptance additionally requires production HTTP quality, memory, cancellation, and recovery evidence.
Retain the exact candidate and complete result with each execution.

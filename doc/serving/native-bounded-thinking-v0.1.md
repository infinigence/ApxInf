# Native MiniCPM bounded initial reasoning profile

Status: implementation contract. Runtime acceptance remains pending.
The [serving contract](contracts-v0.1.md) retains precedence.
Protocols 2.0 through 2.3 retain their existing behavior and identities.
This profile adds no runtime dependency.
Its CPU generation tests use the existing workspace crate `apxinf-loader` as a test dependency.

## Deployment and identity

Select `--provider native-minicpm5-thinking-bounded` before startup.
The worker protocol is `apxinf-worker/2.4`.
The public profile is `apxinf-native-bounded-thinking-v0.1`.
The deployment retains the [thinking profile](native-thinking-v0.1.md) model, precision, context, and execution limits.
The initial reasoning budget is 512 generated tokens.
Prompt tokens do not consume this budget.
The budget constrains only the initial delimited prefix.
It does not classify later public text or guarantee useful final content.
The closing token consumes one additional position within the existing output allowance.
This profile adds no request setting for the initial reasoning budget.

The manifest declares `adapter_revision=apxinf-native-minicpm5-bounded-thinking-v1`.
Its execution object retains the 2.3 fields with two changes.
Set `output_policy="minicpm5-initial-thinking-fixed512-v1"`.
Add this exact object.

```json
{
  "initial_reasoning": {
    "budget_tokens": 512,
    "close_token_id": 9,
    "selection": "cpu-constrained-close-v1"
  }
}
```

The launcher checks the complete execution identity before readiness.
The worker retains the 2.3 capability shape.
The manifest changes the model revision.
Readiness reports `public_capabilities.initial_reasoning_budget_tokens=512`.
Readiness retains `model_thinking="initial_hidden"` and `thinking=false`.
Anthropic extended thinking remains unsupported.

## Tokenizer qualification

The worker checks the pinned tokenizer before readiness.
The tokenizer JSON must declare added token 9 with content `</think>` and `special=false`.
The tokenizer must encode `</think>` as exactly `[9]` without automatic special tokens.
The tokenizer must decode `[9]` as exactly `</think>` without special-token removal.
Token 9 must be inside the vocabulary and outside the EOS set.
A failed check prevents readiness.
The 2.4 ready frame also rejects an out-of-vocabulary or EOS closing token.
The manifest retains the tokenizer artifact hash.

## Selection state

One private worker sampler owns the initial reasoning state.
The state starts in `Reasoning` for each sampler initialization.
Initialization resets the CPU sampler history, capacity, and random stream.
Both greedy and random requests use this sampler.
The sampler retains at most 513 generated token IDs while reasoning remains open.
After each selection, it decodes the non-EOS prefix without special-token removal.
The observed prefix uses the same token IDs as the worker decoder.
The first complete `</think>` permanently changes the state to `Closed`.
This rule recognizes a natural delimiter across multiple token IDs.
Later literal delimiters cannot reactivate the policy.

Indices start at zero.
Indices 0 through 511 use actual model logits.
At index 512, an open state selects the closing token from a constrained CPU row.
That row uses CPU F32 storage directly.
It contains zero at token 9 and `-f32::MAX` elsewhere.
The row never passes through model precision or device conversion.
Every selection calls the existing CPU sampler exactly once.
The sampler checks that the constrained selection returns token 9 and completes the decoded delimiter.
A failed check returns the existing internal execution error.
After closure, all selections use actual model logits.
The model consumes the selected closing token at its ordinary sequence position when generation continues.

This policy requires neutral repetition, frequency, and presence penalties.
It does not return token log probabilities or accept top-k selection.
Random selection retains the existing serving temperature bound of two and its top-p bounds.
Temperature and top-p must remain positive and finite after conversion to F32.
The constrained row gives competing tokens zero probability under these parameters.
Initialization rejects unsupported parameters before selection.
The constrained selection records history, consumes capacity, and advances the CPU random counter once.
Greedy selections also advance this counter under the existing CPU sampler rules.
With an initial draw counter of zero, the forced selection uses draw 512.
The following selection uses draw 513.
The first 512 selections retain the existing random stream policy.

## Accounting and terminal boundaries

All selected tokens count toward output usage, including the closing token and EOS.
All selected tokens consume the existing output allowance and context reservation.
The profile creates no additional prompt or output allowance.
The sampler selects the close as token 513 at index 512.
An allowance of 512 or fewer cannot trigger the constrained selection.
An allowance of 513 can end immediately after the close.
A natural close before index 512 prevents the constrained selection.
EOS before the first close retains the incomplete-output error.
Length before the first close retains empty public content and full usage.

The worker retains ordinary token events, decoder text, event indices, and consumed positions.
The supervisor retains initial-prefix suppression, raw byte limits, stop offsets, and tool parsing.
The [2.3 stop and settlement rules](native-thinking-v0.1.md#stop-boundaries-and-accounting) remain unchanged.
Cancellation, expiry, failure, and callback refusal retain their existing cooperative boundaries.
Controls that arrive during a forward operation can take effect at the next token callback.
The worker never appends a closing token after generation ends.
Capacity remains held until resource release or worker recovery.

## Verification

Rust and Python check the same original 2.4 fixture corpus.
The corpus covers identity, exact budget fields, preparation, sampling, raw events, and stop controls.
It rejects changed budgets, closing IDs, selection policies, unknown fields, and identities under earlier protocols.
All earlier fixture corpora remain unchanged.

CPU tests cover both selection modes, reset, capacity, random continuation, and unsupported parameters.
Tests cover natural delimiters across multiple IDs and later literal delimiters.
Generation tests cover allowances 512 and 513, EOS, cancellation, and settlement.
Native execution checks the pinned tokenizer, constrained close, exact usage, and worker recovery.
Real Claude Code execution checks actual tools and the rendered SVG.
Structural SVG validity does not establish visual task quality.
The scratch prototype uses a different post-close random stream and cannot qualify this production policy.

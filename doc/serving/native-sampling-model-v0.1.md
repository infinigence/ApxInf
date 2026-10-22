# Caller-selected model sampling

Status: Model interface 0.1. Date: 2026-10-22.

`generate_streaming_controlled_with_sampler` accepts a model, a `GenerationRequest`, a mutable `TokenSampler`, and a boundary callback.
A caller-selected sampler is an existing sampler that the caller supplies for this request.
The caller owns the sampler and its backend resources.
The model owns its request state and produces logits.

The entry resolves `GenerationOptions` with the existing rules.
It rejects an empty prompt and unsupported image input.
It checks the sampler specification before model mutation.
The vocabulary size must equal the model vocabulary size.
The capacity must cover the prompt length plus the maximum output allowance.
An invalid specification or insufficient capacity returns `Error::Other`.
Sequence length overflow also returns `Error::Other`.

A zero-output request returns an empty output without sampler initialization, model mutation, or boundary callbacks.
The entry still checks the supplied sampler specification for that request.
For a positive output allowance, the entry calls `begin` before model mutation.
That call receives the exact prompt, resolved sampling parameters, and request RNG key.
The RNG key contains the resolved seed and zero sequence and draw counters.
A `begin` error returns before model mutation.

The existing backend-selected entries retain their behavior.
Both paths use one initialized generation loop.
That loop retains EOS output, prefill cancellation, token boundary stops, and timing behavior.
The caller must release partial request state after a stop or execution error.
The sampler must accept the model's logits and return valid token IDs.
Specification checks do not establish backend compatibility.

This interface introduces no dependency, worker adapter, or wire protocol change.
CPU model tests check sampler selection, initialization, validation, zero output, cancellation, and EOS.
Structural checks follow ASD-STE100 Issue 9 principles.
These checks do not include a complete dictionary review or certification.
